//! Gusher / Pacer（类型 11，变体 0/1）AI：FUN_0050af90，RVA 0x0010AF90，
//! `decompiled/01/0010AF90_FUN_0050af90.c`。该函数还服务类型 0xd2/0xee/0x118，这里只翻译类型 11。
//!
//! Gusher 随机游走（`MoveRandomlyBoss`），冷却归零时朝当前速度方向发射血弹，并周期性留下红色黏液；
//! Pacer 只随机游走。弹幕实体与黏液尚未翻译，发射以事件形式输出。

use crate::entity::{flags, Entity};
use crate::math::Vec2;
use crate::npc::anim_walk_frame;
use crate::pathfinder;
use crate::rng::Rng;
use crate::room::Room;
use crate::world::Event;

pub fn update(
    e: &mut Entity,
    room: &mut Room,
    game_frame: u32,
    rng: &mut Rng,
    events: &mut Vec<Event>,
) {
    // 第 70–102 行：初始化
    if e.npc().state == 0 {
        let cd = (rng.next_u32() % 20 + 20) as i32;
        let npc = e.npc_mut();
        npc.state = 4;
        npc.projectile_cooldown = cd;
        return;
    }

    // 第 130–160 行：行走动画
    anim_walk_frame(e, 0.1);

    let is_gusher = e.variant == 0;
    if is_gusher {
        // 第 268–338 行：冷却归零且无燃烧/恐惧/混乱时，沿速度方向射一发血弹；冷却 = rand%20 + 40
        let blocked = e.has_flag(flags::BURN | flags::FEAR | flags::CONFUSION);
        if e.npc().projectile_cooldown < 1 && !blocked {
            let speed = e.vel.length();
            if speed > 0.0 {
                let dir = Vec2::new(e.vel.x / speed, e.vel.y / speed);
                e.npc_mut().projectile_cooldown = (rng.next_u32() % 20 + 40) as i32;
                events.push(Event::FireProjectiles {
                    from: e.id,
                    pos: e.pos,
                    vel: Vec2::new(dir.x * 5.0, dir.y * 5.0), // DAT_00baa784 = 5
                });
            }
        } else {
            let dt = e.frame_delta();
            e.npc_mut().projectile_cooldown -= dt;
        }
    }

    // 第 339–351 行：摩擦乘 0.75，随机游走（燃烧/恐惧时 EvadeTarget，未翻译）
    e.frame_friction *= 0.75; // DAT_00baa380
    if !e.has_flag(flags::BURN | flags::FEAR) {
        let mut pf = std::mem::take(&mut e.npc_mut().pathfinder);
        pathfinder::move_randomly_boss(&mut pf, e, room, rng, false, game_frame);
        e.npc_mut().pathfinder = pf;
    }

    // 第 352–398 行：Gusher 周期性生成红色黏液（FUN_006b64d0，周期由寄存器参数决定，未定位）——略
}
