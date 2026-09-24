//! In-process batch ABI. Ownership: create -> reset/step -> state -> destroy.
//! JSON pointer is valid until the next state() call on the same batch.
//! Only the checked Python wrapper calls this unsafe ABI; no engine subprocesses.
use crate::{arena, Event, PlayerInput, World};
use rayon::prelude::*;
use serde_json::json;
use std::ffi::{c_char, CString};

pub struct Slot {
    pub world: World,
    pub player: u32,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
    pub outcome: &'static str,
    pub previous: Option<crate::observation::Previous>,
    pub action: [f32; 4],
}
impl Slot {
    pub fn new(seed: u32) -> Self {
        let (world, player) = arena::monstro(seed);
        Self {
            world,
            player,
            reward: 0.,
            done: false,
            truncated: false,
            outcome: "running",
            previous: None,
            action: [0.; 4],
        }
    }
    /// Training-only start (OpenAI Five "Roshan health" randomisation): the same room, spawns and
    /// combat RNG as `new`, then player HP (half-hearts) and the Boss HP fraction are overridden.
    /// Max HP is unchanged, so visible HP and damage/max_hp rewards keep their meaning.
    pub fn with_start(seed: u32, player_hp: f32, boss_hp_fraction: f32) -> Self {
        let mut slot = Self::new(seed);
        let id = slot.player;
        let player = slot.world.entity_mut(id).unwrap();
        player.hp = player_hp.clamp(1.0, player.max_hp);
        for boss in slot.world.entities.iter_mut().filter(|e| e.etype == 20) {
            boss.hp = (boss.max_hp * boss_hp_fraction.clamp(0.01, 1.0)).max(1.0);
        }
        slot
    }
    pub fn step(&mut self, action: &[i32]) {
        assert!(!self.done, "reset required");
        self.previous = Some(crate::observation::Previous::capture(
            &self.world,
            self.player,
        ));
        self.action = [action[0] as f32, action[1] as f32, action[2] as f32, 1.];
        self.reward = 0.;
        let joint = action[0] as usize;
        const MOVES: [(f32, f32); 9] = [
            (0., 0.),
            (0., -1.),
            (1., -1.),
            (1., 0.),
            (1., 1.),
            (0., 1.),
            (-1., 1.),
            (-1., 0.),
            (-1., -1.),
        ];
        const SHOTS: [(f32, f32); 5] = [(0., 0.), (0., -1.), (1., 0.), (0., 1.), (-1., 0.)];
        let m = MOVES[joint / 5];
        let s = SHOTS[joint % 5];
        let mut input = PlayerInput::new(m.0, m.1, s.0, s.1);
        input.bomb = action[1] != 0;
        self.world.set_player_input(self.player, input);
        for _ in 0..2 {
            self.world.step();
            for event in &self.world.events {
                match event {
                    Event::PlayerDamaged { .. } => self.reward -= 1.,
                    Event::EnemyDamaged {
                        damage,
                        hits,
                        max_hp,
                        ..
                    } => self.reward += 0.05 * (*hits as f32) + damage / max_hp,
                    _ => {}
                }
            }
            let dead = self.world.entity(self.player).unwrap().dead;
            let won = !self.world.entities.iter().any(|e| e.etype == 20) && !dead;
            self.truncated = self.world.frame >= 3600 && !dead && !won;
            self.done = dead || won || self.truncated;
            self.outcome = if dead {
                "death"
            } else if won {
                "win"
            } else if self.truncated {
                "time_limit"
            } else {
                "running"
            };
            if won {
                self.reward += 1.;
            }
            if self.done {
                break;
            }
        }
    }
}
pub struct Batch {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    output: CString,
}
#[no_mangle]
pub extern "C" fn isaac_batch_new(count: usize, seed: u32, threads: usize) -> *mut Batch {
    Box::into_raw(Box::new(Batch {
        slots: (0..count)
            .map(|i| Slot::new(seed.wrapping_add(i as u32)))
            .collect(),
        pool: rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap(),
        output: CString::new("").unwrap(),
    }))
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_reset(batch: *mut Batch, index: usize, seed: u32) {
    (&mut *batch).slots[index] = Slot::new(seed);
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_reset_start(
    batch: *mut Batch,
    index: usize,
    seed: u32,
    player_hp: f32,
    boss_hp_fraction: f32,
) {
    (&mut *batch).slots[index] = Slot::with_start(seed, player_hp, boss_hp_fraction);
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_step(batch: *mut Batch, actions: *const i32) {
    let b = &mut *batch;
    let a = std::slice::from_raw_parts(actions, b.slots.len() * 3);
    b.pool.install(|| {
        b.slots
            .par_iter_mut()
            .zip(a.par_chunks_exact(3))
            .for_each(|(s, a)| s.step(a))
    });
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_step_one(batch: *mut Batch, index: usize, actions: *const i32) {
    (&mut *batch).slots[index].step(std::slice::from_raw_parts(actions, 3));
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_state(batch: *mut Batch) -> *const c_char {
    let b = &mut *batch;
    let values:Vec<_>=b.slots.iter().map(|s|json!({"state":crate::snapshot::state(&s.world,s.player),"reward":s.reward,"done":s.done,"truncated":s.truncated,"outcome":s.outcome})).collect();
    b.output = CString::new(serde_json::to_string(&values).unwrap()).unwrap();
    b.output.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_free(batch: *mut Batch) {
    drop(Box::from_raw(batch));
}

/// Caller owns `count` contiguous Frame records. No JSON, allocation, or hidden
/// AI state crosses this interface. Positive return value means capacity overflow.
#[no_mangle]
pub unsafe extern "C" fn isaac_batch_observe(
    batch: *mut Batch,
    out: *mut crate::observation::Frame,
) -> usize {
    let b = &*batch;
    let frames = std::slice::from_raw_parts_mut(out, b.slots.len());
    b.pool.install(|| {
        b.slots
            .par_iter()
            .zip(frames.par_iter_mut())
            .map(|(s, f)| crate::observation::encode(s, f))
            .max()
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "C" fn isaac_frame_size() -> usize {
    std::mem::size_of::<crate::observation::Frame>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_override_changes_only_hp() {
        for seed in [7u32, 1234, 2147483700] {
            let base = Slot::new(seed);
            let full = Slot::with_start(seed, 6.0, 1.0);
            assert_eq!(
                crate::snapshot::state(&full.world, full.player),
                crate::snapshot::state(&base.world, base.player)
            );
            let s = Slot::with_start(seed, 3.0, 0.4);
            assert_eq!(s.world.entity(s.player).unwrap().hp, 3.0);
            let boss = s.world.entities.iter().find(|e| e.etype == 20).unwrap();
            let base_boss = base.world.entities.iter().find(|e| e.etype == 20).unwrap();
            assert!((boss.hp - 100.0).abs() < 1e-4);
            assert_eq!(boss.max_hp, 250.0);
            assert_eq!(boss.pos, base_boss.pos);
            assert_eq!(s.world.room.layout, base.world.room.layout);
        }
    }

    #[test]
    fn low_hp_boss_is_cleared_and_rewarded() {
        let mut s = Slot::with_start(42, 6.0, 0.02);
        let mut total = 0.;
        for _ in 0..1800 {
            // Scripted probe: approach when far, shoot along the dominant axis toward the Boss.
            let p = s.world.entity(s.player).unwrap().pos;
            let b = s.world.entities.iter().find(|e| e.etype == 20).unwrap().pos;
            let (dx, dy) = (b.x - p.x, b.y - p.y);
            let shoot = if dx.abs() > dy.abs() { if dx > 0. { 2 } else { 4 } } else if dy > 0. { 3 } else { 1 };
            let far = dx.hypot(dy) > 160.;
            let step = |v: f32| if far && v.abs() > 20. { v.signum() as i32 } else { 0 };
            let (mx, my) = (step(dx), step(dy));
            let movement = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
                .iter()
                .position(|m| *m == (mx, my))
                .unwrap() as i32;
            s.step(&[movement * 5 + shoot, 0, 0]);
            total += s.reward;
            if s.done {
                break;
            }
        }
        assert_eq!(s.outcome, "win", "a 5-HP Monstro must be killable within the deadline");
        // Normalised damage uses max HP 250, so a 5-HP Boss yields at most 5/250 of damage reward.
        assert!(total < 1.0 + 0.05 * 3. + 5. / 250. + 1e-4);
    }
}
