//! Gaper（类型 10）AI：FUN_00508f40，RVA 0x00108F40，`decompiled/01/00108F40_FUN_00508f40.c`。
//! 该函数同时服务类型 0xd0/0x101/0x11c/0x12e/0xfc（Nulls、Psychic Horf 等），这里只翻译类型 10。
//!
//! 变体：10.0 皱眉 Gaper，10.1 Gaper，10.2 燃烧 Gaper；10.3 腐烂 Gaper 走独立函数 `FUN_004da590`（未翻译）。

use crate::entity::{flags, Entity, TYPE_GUSHER};
use crate::math::Vec2;
use crate::npc::anim_walk_frame;
use crate::pathfinder;
use crate::rng::Rng;
use crate::room::Room;

/// 皱眉 Gaper 的 "Head" 覆盖动画帧数（`010.000_frowning gaper.anm2`）；Gaper 为 1 帧。
const FROWNING_HEAD_FRAMES: i32 = 20;

pub fn update(e: &mut Entity, room: &mut Room, player_pos: Vec2, game_frame: u32, rng: &mut Rng) {
    if e.variant == 3 {
        return; // 腐烂 Gaper：FUN_004da590，未翻译
    }

    // 第 87–130 行：状态 0 → 初始化为状态 4，并装上 "Head" 覆盖层
    if e.npc().state == 0 {
        let overlay_len = if e.variant == 0 {
            FROWNING_HEAD_FRAMES
        } else {
            1
        };
        let npc = e.npc_mut();
        npc.state = 4;
        npc.state_frame = 0;
        npc.overlay_frame = 0;
        npc.overlay_len = overlay_len;
        npc.overlay_loop = false;
    }
    if e.npc().state == 0 {
        return;
    }

    // 第 205–247 行：死亡分支——血量不低于 -25 时 50% 概率变成 Gusher/Pacer 并满血复活
    if e.dead {
        if e.hp <= -25.0 {
            return;
        }
        if rng.next_u32() & 1 != 0 {
            return;
        }
        let variant = if rng.next_u32().is_multiple_of(3) && e.variant != 2 {
            1
        } else {
            0
        };
        morph_to_gusher(e, variant);
        e.dead = false;
        e.hp = e.max_hp;
        e.npc_mut().projectile_cooldown = (rng.next_u32() % 20 + 20) as i32;
        return;
    }

    // 第 511–532 行：行走动画（阈值 0.1）
    anim_walk_frame(e, 0.1);

    // 第 533–563 行：追踪速度参数（+0xc34）
    let mut speed = 0.6f32; // 0x3f19999a
    match e.variant {
        1 => speed = 0.4980, // 0x3efef9db
        2 => speed = 0.5478, // 0x3f0c3c9f（燃烧 Gaper 同时每帧 PlayOverlay("Head")）
        _ => {
            if e.npc().overlay_frame > 0 {
                speed = 0.72; // 0x3f3851ec：皱眉 Gaper 头部覆盖动画播放中
            }
        }
    }
    e.npc_mut().move_speed = speed;

    // 第 564–588 行：未燃烧且未恐惧时寻路追击（PathMarker 900，UseDirectPath true）；
    // 否则 EvadeTarget 并把本帧摩擦乘 0.8（EvadeTarget 未翻译，这里只保留摩擦项）
    if !e.has_flag(flags::BURN | flags::FEAR) {
        let target = player_pos; // CalcTargetPosition：无混乱/恐惧时即目标玩家位置
        let mut pf = std::mem::take(&mut e.npc_mut().pathfinder);
        pathfinder::find_grid_path(&mut pf, e, room, target, speed, 900, true, game_frame, rng);
        e.npc_mut().pathfinder = pf;
    } else {
        e.frame_friction *= 0.8;
    }

    // 第 589–614 行：随机吼叫音效（略）；第 615–620 行：皱眉 Gaper 在 aggro 且覆盖层停止时重放 "Head"
    if e.variant == 0 && e.npc().aggro && e.npc().overlay_frame < 1 {
        e.npc_mut().overlay_frame = 0;
    }

    // 覆盖层动画推进（精灵系统的替身）
    let npc = e.npc_mut();
    if npc.overlay_frame >= 0 && npc.overlay_len > 0 {
        if npc.overlay_frame + 1 < npc.overlay_len {
            npc.overlay_frame += 1;
        } else if npc.overlay_loop {
            npc.overlay_frame = 0;
        } else {
            npc.overlay_frame = npc.overlay_len - 1;
        }
    }
}

/// `Entity_NPC::Morph(11, variant, 0, -1)`（FUN_006c0f30）的最小替身：换类型/变体并按 `entities2.xml`
/// 重设配置（Gusher 11.0 / Pacer 11.1：hp 10、半径 13、质量 3、接触伤害 1）。
fn morph_to_gusher(e: &mut Entity, variant: i32) {
    e.etype = TYPE_GUSHER;
    e.variant = variant;
    e.subtype = 0;
    e.max_hp = 10.0;
    e.mass = 3.0;
    e.collision_damage = 1.0;
    let npc = e.npc_mut();
    npc.state = 0; // Gusher AI 下一次调用时初始化
    npc.state_frame = 0;
    npc.pathfinder = pathfinder::Pathfinder::default();
    npc.overlay_frame = -1;
    npc.anim = "WalkHori";
}
