//! Seeded curriculum, not a reproduction of original stage-generation RNG.
use crate::{rng::Rng, room::GridKind, Room, Vec2, World};
include!("layouts.rs");
pub const ENTRANCES: [Vec2; 4] = [
    Vec2 { x: 80., y: 280. },
    Vec2 { x: 320., y: 160. },
    Vec2 { x: 560., y: 280. },
    Vec2 { x: 320., y: 400. },
];

pub fn clear_circle(room: &Room, p: Vec2, r: f32) -> bool {
    if p.x - r < room.top_left.x
        || p.x + r > room.bottom_right.x
        || p.y - r < room.top_left.y
        || p.y + r > room.bottom_right.y
    {
        return false;
    }
    for (i, c) in room.cells.iter().enumerate() {
        if c.kind == GridKind::Empty {
            continue;
        }
        let d = room.cell_center(i as i32) - p;
        if (d.x.abs() - 20.).max(0.).powi(2) + (d.y.abs() - 20.).max(0.).powi(2) < r * r {
            return false;
        }
    }
    true
}

pub fn monstro(seed: u32) -> (World, u32) {
    // Independent setup RNG: extra layout choices do not alter combat RNG calls.
    let mut rng = Rng::new(seed ^ 0xa511e9b3);
    let (variant, rocks, doors) = LAYOUTS[rng.next_u32() as usize % LAYOUTS.len()];
    let mut room = Room::new_rectangular(15, 9);
    room.room_type = 5;
    room.layout = variant;
    room.doors = doors;
    for &i in rocks {
        room.add_rock(i);
    }
    let entries: Vec<_> = ENTRANCES
        .iter()
        .enumerate()
        .filter(|(i, p)| doors[*i] && clear_circle(&room, **p, 12.))
        .map(|(_, p)| *p)
        .collect();
    let p = entries[rng.next_u32() as usize % entries.len()];
    // XML flags permit door slots; they do not mean every door is instantiated.
    // A single-room Boss curriculum has only the chosen entrance door.
    room.doors = ENTRANCES.map(|entry| entry == p);
    let candidates: Vec<_> = (0..135)
        .map(|i| room.cell_center(i))
        .filter(|b| clear_circle(&room, *b, 42.) && (*b - p).length() >= 200.)
        .collect();
    assert!(!candidates.is_empty(), "layout has no safe Boss spawn");
    let b = candidates[rng.next_u32() as usize % candidates.len()];
    let mut w = World::new(room, seed);
    let id = w.spawn_player(p);
    w.spawn_monstro(b);
    (w, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ten_thousand_safe_reproducible_spawns() {
        let mut layouts = std::collections::BTreeSet::new();
        let mut positions = std::collections::BTreeSet::new();
        for seed in 0..10000 {
            let (mut w, p) = monstro(seed);
            let player = w.entity(p).unwrap();
            let boss = w.entities.iter().find(|e| e.etype == 20).unwrap();
            assert!(ENTRANCES.contains(&player.pos));
            assert!(clear_circle(&w.room, player.pos, 12.));
            assert!(clear_circle(&w.room, boss.pos, 42.));
            assert!((boss.pos - player.pos).length() >= 200.);
            let (again, _) = monstro(seed);
            assert_eq!(again.entities[0].pos, player.pos);
            assert_eq!(again.entities[1].pos, boss.pos);
            layouts.insert(w.room.layout);
            positions.insert((boss.pos.x as i32, boss.pos.y as i32));
            assert_eq!(w.room.doors.iter().filter(|x| **x).count(), 1);
            for _ in 0..15 {
                w.step();
            }
            assert_eq!(w.entity(p).unwrap().hp, 6., "entry damage for seed {seed}");
        }
        assert_eq!(layouts.len(), 4);
        assert!(positions.len() > 20);
    }
}
