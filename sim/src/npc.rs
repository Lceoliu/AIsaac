//! NPC 每帧骨架（`Entity_NPC::Update`，FUN_006c4b30，RVA 0x002C4B30）与按类型分发的 AI。
//!
//! 原版一帧内的顺序（主路径）：
//! 1. `FUN_006b99e0` 计算速度倍率 → `speed_mult`（无状态效果时 1.0）；
//! 2. 第 302–965 行：`FLAG_APPEAR` 出现流程（状态 1）；
//! 3. 第 983–1009 行：`championRegenTimer == 0` 时调用 AI 分发器 `FUN_006c2990`；
//! 4. 第 2117–2123 行：`FUN_006d7010/6d7230`（流血特效、Boss 血条）与前置模式 AI（0xf18 位 0）；
//! 5. 第 2190–2194 行：`Entity::Update`；
//! 6. 第 2201–2205 行：后置模式 AI（0xf18 位 3）；
//! 7. 第 2528–2600 行：死亡处理（`FUN_006bbdc0`，掉落/尸体，这里只标记移除）。
//!
//! 网格碰撞不在这里：它在 `EntityList::Update` 的碰撞阶段（FUN_00419e70 尾部）对所有实体调用。

pub mod gaper;
pub mod gusher;

use crate::entity::{flags, Entity, EntityKind, TYPE_GAPER, TYPE_GUSHER};
use crate::physics;
use crate::rng::Rng;
use crate::room::Room;
use crate::world::Event;

/// NPC 的一帧（不含网格碰撞）。`player_pos` 是 `Entity_NPC::GetPlayerTarget` 的替身：单人局即玩家位置。
pub fn update_npc(
    e: &mut Entity,
    room: &mut Room,
    player_pos: Vec2,
    game_frame: u32,
    rng: &mut Rng,
    events: &mut Vec<Event>,
) {
    if !e.exists || e.kind != EntityKind::Npc {
        return;
    }
    // 1. 速度倍率（FUN_006b99e0）：本内核尚无减速/冰冻施加，保持 1.0
    e.speed_mult = 1.0;

    // 2. 出现流程
    if e.has_flag(flags::APPEAR) {
        let state = e.npc().state;
        if state == 1 {
            let counter = e.npc().appear_counter;
            match counter {
                0 => {
                    // 第 744–762 行：先让 AI 做一次初始化（状态 0），再转回出现状态并隐身
                    e.npc_mut().state = 0;
                    run_ai(e, room, player_pos, game_frame, rng, events);
                    if e.has_flag(flags::APPEAR) {
                        e.visible = false;
                        let s = e.npc().state;
                        e.npc_mut().saved_state = s;
                        e.npc_mut().state = 1;
                    }
                }
                1 => {
                    // 第 823–856 行：生成出现烟雾特效与音效（视觉/音效，略）
                }
                4 => {
                    // 第 857–866 行：显形；有 "Appear" 动画的类型开始播放
                    e.visible = true;
                }
                c if c >= 0x14 => {
                    // 第 869–893 行：没有 Appear 动画（Gaper）或动画已结束 → 恢复状态。
                    // FLAG_APPEAR 的清除点在伪 C 里被内联进精灵函数，无法定位，这里在同一时刻清除。
                    let saved = e.npc().saved_state;
                    e.npc_mut().state = saved;
                    e.frame_friction = e.friction;
                    e.flags &= !flags::APPEAR;
                }
                _ => {}
            }
            if e.npc().state == 1 {
                // 第 960–962 行：出现期间摩擦乘 0.01，实体基本冻在原地
                e.frame_friction *= 0.01;
            }
            e.npc_mut().appear_counter += 1;
        }
    } else if e.npc().state == 1 {
        // 第 304–306 行
        let saved = e.npc().saved_state;
        e.npc_mut().state = saved;
    }

    // 3. 常规 AI（第 966–1009 行）：死亡动画状态 0x11/0x12、或（被持有 / 出现状态 1）且未死亡时不调用 AI；
    //    后一种情况下若 animFreeze > 0（或 FUN_00417430(0x20,0) 为真，含义未定）本帧摩擦再乘 0.01（DAT_00baa06c）。
    let state = e.npc().state;
    let dying = state == 0x11 || state == 0x12;
    let held_or_appearing = (e.has_flag(flags::HELD) || state == 1) && !e.dead;
    if dying || held_or_appearing {
        if !dying && e.anim_freeze > 0 {
            e.frame_friction *= 0.01;
        }
    } else {
        if e.dead && state == 1 {
            // 第 998–1002 行：出现中死亡 → 直接恢复状态并显形
            e.frame_friction = e.friction;
            let saved = e.npc().saved_state;
            e.npc_mut().state = saved;
            e.visible = true;
        }
        if e.npc().champion_regen_timer == 0 {
            run_ai(e, room, player_pos, game_frame, rng, events);
        }
    }

    // 5. 基类更新
    physics::update(e);

    // 7. 死亡：AI 的死亡分支（Gaper 变 Gusher）已在 run_ai 中处理；仍然死亡则请求移除
    if e.dead {
        events.push(Event::NpcDied {
            id: e.id,
            etype: e.etype,
            variant: e.variant,
        });
        e.exists = false;
    }
}

use crate::math::Vec2;

/// AI 分发器（FUN_006c2990，RVA 0x002C2990）：`type-10` 查索引表跳转。这里只有已翻译的类型。
pub fn run_ai(
    e: &mut Entity,
    room: &mut Room,
    player_pos: Vec2,
    game_frame: u32,
    rng: &mut Rng,
    events: &mut Vec<Event>,
) {
    match e.etype {
        TYPE_GAPER => gaper::update(e, room, player_pos, game_frame, rng),
        TYPE_GUSHER => gusher::update(e, room, game_frame, rng, events),
        _ => {}
    }
}

/// `Entity_NPC::AnimWalkFrame(hori, vert, threshold)`（FUN_006c1330，RVA 0x002C1330）的可见结果：
/// 速度平方低于阈值平方时保持竖直行走动画；否则按 |vx| 与 |vy| 选横/竖动画，横向时按 vx 符号翻转。
pub fn anim_walk_frame(e: &mut Entity, threshold: f32) {
    let vx = e.vel.x;
    let vy = e.vel.y;
    let npc = e.npc_mut();
    if vx * vx + vy * vy < threshold * threshold {
        npc.anim = "WalkVert";
        return;
    }
    if vx.abs() <= vy.abs() {
        npc.anim = "WalkVert";
        npc.flip_x = false;
    } else {
        npc.anim = "WalkHori";
        npc.flip_x = vx < 0.0;
    }
    npc.anim_frame += 1;
}
