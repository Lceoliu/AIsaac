//! Normal Monstro 20.0, J460 FUN_0046c0a0. Not Monstro II / champions.
//! Animation event timing is from the Afterbirth+ override, not base animations.b.
use crate::{entity::Entity, math::Vec2, rng::Rng, room::Room, world::Event};

fn play(e: &mut Entity, state: i32, anim: &'static str) {
    let n = e.npc_mut();
    n.state = state;
    n.anim = anim;
    n.anim_frame = 0;
    n.anim_finished = false;
}

fn lock_target(e: &mut Entity, room: &Room, player: Vec2) {
    if e.npc().target_pos.is_zero() && room.grid_collision_at(player) == 0 {
        e.npc_mut().target_pos = room.cell_center(room.grid_index(player));
    }
}

pub fn update(e: &mut Entity, room: &Room, player: Vec2, rng: &mut Rng, events: &mut Vec<Event>) {
    let state = e.npc().state;
    let f = e.npc().anim_frame;
    e.frame_friction *= 0.9;
    match state {
        0 => e.npc_mut().state = 3,
        1 => {
            if e.npc().anim_finished {
                e.npc_mut().state = 3;
            }
        }
        3 => {
            e.entity_collision_class = 4;
            e.grid_collision_class = 5;
            match rng.next_u32() % 3 {
                0 => play(e, 4, "Walk"),
                1 => {
                    play(e, 8, "Taunt");
                    e.npc_mut().flip_x = player.x > e.pos.x;
                }
                _ => play(e, 6, "JumpUp"),
            }
        }
        4 => {
            lock_target(e, room, player);
            let airborne = (6..22).contains(&f);
            e.entity_collision_class = if airborne { 0 } else { 4 };
            e.grid_collision_class = if airborne { 3 } else { 5 };
            if f < 22 && !e.npc().target_pos.is_zero() {
                let delta = e.npc().target_pos - e.pos;
                // Native applies acceleration before base friction: .81*v + 1.08*dir.
                e.vel += delta.normalized() * (1.2 * e.speed_mult / e.frame_friction);
                e.frame_friction *= 0.9;
                e.npc_mut().flip_x = e.vel.x > 0.0;
            } else {
                e.frame_friction *= 0.8;
            }
            if e.npc().anim_finished {
                e.npc_mut().target_pos = Vec2::ZERO;
                if rng.next_u32() & 1 != 0 {
                    play(e, 4, "Walk");
                } else {
                    e.npc_mut().state = 3;
                }
            }
        }
        6 => {
            e.entity_collision_class = 4;
            e.grid_collision_class = 5;
            if f >= 10 {
                e.entity_collision_class = 0;
                e.grid_collision_class = 3;
                e.npc_mut().target_pos = Vec2::ZERO;
                play(e, 7, "JumpDown");
            }
        }
        7 => {
            // First JumpDown update locks the current player's grid centre ONCE.
            // I1==2 (Monstro's Tooth special spawn) is deliberately not normal boss AI.
            lock_target(e, room, player);
            if f < 32 && !e.npc().target_pos.is_zero() {
                e.entity_collision_class = 0;
                e.grid_collision_class = 3;
                let d = e.npc().target_pos - e.pos;
                e.vel += (d * 0.006 + d.normalized() * 0.3) * (e.speed_mult / e.frame_friction);
            } else {
                e.frame_friction *= 0.8;
                e.entity_collision_class = 4;
                e.grid_collision_class = 5;
            }
            if f == 34 {
                events.push(Event::BossVolley {
                    from: e.id,
                    pos: e.pos,
                    target: Vec2::ZERO,
                    count: 18,
                });
            }
            if e.npc().anim_finished {
                e.npc_mut().state = 3;
                e.npc_mut().target_pos = Vec2::ZERO;
            }
        }
        8 => {
            if f == 21 {
                events.push(Event::BossVolley {
                    from: e.id,
                    pos: e.pos,
                    target: player,
                    count: 13,
                });
            }
            if e.npc().anim_finished {
                e.npc_mut().state = 3;
            }
        }
        _ => panic!("unsupported normal Monstro state {state}"),
    }
    let duration = match e.npc().anim {
        "Appear" => 26,
        "Walk" => 45,
        "JumpUp" => 15,
        "JumpDown" => 64,
        "Taunt" => 66,
        _ => 1,
    };
    if e.npc().anim_frame + 1 >= duration {
        e.npc_mut().anim_finished = true;
    } else {
        e.npc_mut().anim_frame += 1;
    }
}

/// Derivable from the currently visible animation, not a hidden AI-state flag.
pub fn airborne(anim: &str, frame: i32) -> bool {
    match anim {
        "Walk" => (7..23).contains(&frame),
        "JumpUp" => frame >= 10,
        "JumpDown" => frame < 33,
        _ => false,
    }
}
pub fn body_visible(anim: &str, frame: i32) -> bool {
    !(anim == "JumpDown" && frame < 29 || anim == "JumpUp" && frame >= 11)
}
