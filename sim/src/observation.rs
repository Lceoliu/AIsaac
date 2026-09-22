//! Binary implementation of the shared monstro-transformer-v3 visible encoder.
//! Positions/age continuity, not engine velocities or NPC targets, define motion.
//! JSON snapshots remain diagnostic only. Python tests compare every feature.
use crate::{ffi::Slot, room::GridKind, EntityKind, Vec2, World};
use std::collections::HashMap;

pub const CAPACITY: usize = 256;
pub struct Previous {
    frame: u32,
    player: Vec2,
    entities: HashMap<u32, (f32, Vec2)>,
}
impl Previous {
    pub fn capture(w: &World, player: u32) -> Self {
        Self {
            frame: w.frame,
            player: w.entity(player).unwrap().pos,
            entities: w
                .entities
                .iter()
                .filter(|e| !e.dead)
                .map(|e| (e.id, (e.time_cur, e.pos)))
                .collect(),
        }
    }
}

#[repr(C)]
pub struct Frame {
    pub player: [f32; 23],
    pub player_anim: [i32; 32],
    pub active_kind: [i32; 1],
    pub entities: [[f32; 31]; CAPACITY],
    pub entity_kind: [[i32; 3]; CAPACITY],
    pub entity_anim: [[i32; 32]; CAPACITY],
    pub entity_mask: [f32; CAPACITY],
    pub terrain: [f32; 945],
    pub previous_action: [f32; 4],
    pub time: f32,
    pub history_mask: f32,
    pub reward: f32,
    pub done: i32,
    pub truncated: i32,
    pub outcome: i32,
    pub elapsed: u32,
    pub layout: u32,
    pub count: u32,
}

pub fn encode(s: &Slot, f: &mut Frame) -> usize {
    // Frame is a C record of plain numeric arrays (all-zero is valid).
    unsafe {
        std::ptr::write_bytes(f, 0, 1);
    }
    let w = &s.world;
    let p = w.entity(s.player).unwrap();
    let dt = s.previous.as_ref().map_or(0, |v| w.frame - v.frame);
    let (vx, vy) = s.previous.as_ref().map_or((0., 0.), |v| {
        (
            ((p.pos.x as f64 - v.player.x as f64) / dt as f64 / 520.) as f32,
            ((p.pos.y as f64 - v.player.y as f64) / dt as f64 / 280.) as f32,
        )
    });
    f.player = [
        ((p.pos.x as f64 - 60.) / 520.) as f32,
        ((p.pos.y as f64 - 140.) / 280.) as f32,
        vx,
        vy,
        (dt > 0) as u8 as f32,
        10. / 520.,
        p.hp / 6.,
        1.,
        0.,
        p.player().bombs as f32 / 10.,
        0.,
        0.,
        0.35,
        1.,
        1.,
        10. / 30.,
        260. / 520.,
        0.,
        0.,
        0.,
        0.,
        0.,
        dt as f32 / 30.,
    ];
    let mut i = 0;
    // Same stable grouping as rust_visible_observation, including dead filtering.
    for kind in [
        EntityKind::Npc,
        EntityKind::Tear,
        EntityKind::Projectile,
        EntityKind::Bomb,
    ] {
        for e in w
            .entities
            .iter()
            .filter(|e| e.kind == kind && !e.dead && (kind != EntityKind::Npc || e.etype == 20))
        {
            if i == CAPACITY {
                return i + 1;
            }
            let row = &mut f.entities[i];
            row[0] = ((e.pos.x as f64 - p.pos.x as f64) / 520.) as f32;
            row[1] = ((e.pos.y as f64 - p.pos.y as f64) / 280.) as f32;
            if let Some((age, pos)) = s.previous.as_ref().and_then(|v| v.entities.get(&e.id)) {
                if e.time_cur == *age + dt as f32 {
                    row[2] = ((e.pos.x as f64 - pos.x as f64) / dt as f64 / 520.) as f32;
                    row[3] = ((e.pos.y as f64 - pos.y as f64) / dt as f64 / 280.) as f32;
                    row[4] = 1.;
                }
            }
            row[5] = e.size / 520.;
            row[6] = 1.;
            row[7] = 1.;
            row[9] = 1.;
            row[10] = e.entity_collision_class as f32 / 5.;
            row[11] = e.grid_collision_class as f32 / 7.;
            row[12] = e.collision_damage / 10.;
            row[27] = 1.;
            let (etype, anim, frame) = match kind {
                EntityKind::Npc => {
                    let n = e.npc();
                    row[13] = 1.;
                    row[14] = 1.;
                    row[15] = e.hp.max(0.) / 250.;
                    row[16] = 1.;
                    row[18] = n.flip_x as u8 as f32;
                    row[26] = crate::npc::monstro::airborne(n.anim, n.anim_frame) as u8 as f32;
                    row[27] = crate::npc::monstro::body_visible(n.anim, n.anim_frame) as u8 as f32;
                    row[28] = row[0];
                    row[29] = row[1];
                    row[30] = 1.;
                    (20, n.anim, n.anim_frame as f32)
                }
                EntityKind::Tear => {
                    row[8] = e.tear().height / 280.;
                    row[9] = e.tear().scale;
                    (2, "", 0.)
                }
                EntityKind::Projectile => {
                    let q = e.projectile.as_ref().unwrap();
                    row[8] = q.height / 280.;
                    row[9] = q.scale;
                    (9, "", 0.)
                }
                EntityKind::Bomb => (4, "Pulse", 14. + e.time_cur),
                _ => unreachable!(),
            };
            row[17] = frame / 60.;
            f.entity_kind[i] = [etype, 0, 0];
            for (j, b) in anim.bytes().enumerate() {
                f.entity_anim[i][j] = b as i32;
            }
            f.entity_mask[i] = 1.;
            i += 1;
        }
    }
    for (i, c) in w.room.cells.iter().enumerate() {
        let inside = c.kind != GridKind::Wall;
        for (channel, value) in [
            inside,
            inside && c.collision_class == 0,
            c.collision_class >= 2,
            c.kind == GridKind::Pit,
            c.kind == GridKind::Rock,
            false,
            false,
        ]
        .iter()
        .enumerate()
        {
            f.terrain[channel * 135 + i] = *value as u8 as f32;
        }
    }
    // Native closed doors occupy the boundary grid cell nearest the door anchor.
    if w.entities.iter().any(|e| e.etype == 20) {
        for (slot, index) in [60, 7, 74, 127].iter().enumerate() {
            if w.room.doors[slot] {
                f.terrain[6 * 135 + index] = 1.;
            }
        }
    }
    f.previous_action = s.action;
    f.time = w.frame as f32 / 30.;
    f.history_mask = 1.;
    f.reward = s.reward;
    f.done = s.done as i32;
    f.truncated = s.truncated as i32;
    f.outcome = match s.outcome {
        "running" => 0,
        "death" => 1,
        "win" => 2,
        "time_limit" => 3,
        _ => unreachable!(),
    };
    f.elapsed = w.frame;
    f.layout = w.room.layout;
    f.count = i as u32;
    0
}
