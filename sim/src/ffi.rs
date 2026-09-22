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
        }
    }
    pub fn step(&mut self, action: &[i32]) {
        assert!(!self.done, "reset required");
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
