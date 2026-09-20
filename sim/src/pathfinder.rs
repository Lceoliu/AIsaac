//! `NPCAI_Pathfinder`：`FindGridPath`（FUN_007e33c0，RVA 0x003E33C0）与 `MoveRandomlyBoss`
//! （FUN_007e6300，RVA 0x003E6300）。结构体字段偏移来自 REPENTOGON `NPCAI_Pathfinder.zhl`。
//!
//! 常量（读自 J460）：0.1 / 0.2 / 0.5 / 0.7 / 0.85 / 2 / 19.96 / 20 / 40 / 70 / 120 / 750。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::entity::{flags, grid_collision_class as gcc, Entity};
use crate::math::Vec2;
use crate::rng::Rng;
use crate::room::{Room, MAX_CELLS, TILE};

#[derive(Clone, Debug)]
pub struct Pathfinder {
    /// +0x20：登记的自身格（有滞后，见 `interp_countdown`）。
    pub grid_index: i32,
    /// +0x24：换格滞后倒计时。
    pub interp_countdown: i32,
    /// +0x28：重算路径倒计时。
    pub recompute_countdown: i32,
    /// +0x2c：448 格代价图，重算后存 `-(cost)`（负数 = 已访问）。
    pub cost_map: Vec<i32>,
    /// +0x72c：与目标格之间视线畅通。
    pub has_direct_path: bool,
    /// +0x72d：上次搜索到达了自身格。
    pub path_found: bool,
    /// +0x730：目标位置备份（目标落在实心格里时使用）。
    pub target_backup: Vec2,
    /// +0x738。
    pub can_crush_rocks: bool,
    /// +0x4：`MoveRandomlyBoss` 的随机目标点。
    pub random_target: Vec2,
}

impl Default for Pathfinder {
    fn default() -> Self {
        Pathfinder {
            grid_index: -1,
            interp_countdown: 0,
            recompute_countdown: 0,
            cost_map: vec![0; MAX_CELLS],
            has_direct_path: false,
            path_found: false,
            target_backup: Vec2::ZERO,
            can_crush_rocks: false,
            random_target: Vec2::ZERO,
        }
    }
}

/// `FUN_006ac710(n, off)`：`(gameFrame + off) % n == 0`（非插值、非暂停时）。
pub fn every_n_frames(game_frame: u32, n: u32, off: u32) -> bool {
    game_frame.wrapping_add(off).is_multiple_of(n)
}

/// `FUN_007e3340`：实体是否按"飞行/自由"规则寻路。Gaper（类型 10，无魅惑，非玩家控制）为 false。
pub fn is_free_mover(e: &Entity) -> bool {
    if !e.has_flag(flags::CHARM) && e.etype != 0x322 {
        if !(10..=999).contains(&e.etype) {
            return false;
        }
        if let Some(npc) = &e.npc {
            if npc.controller_id == -1 {
                return false;
            }
        }
    }
    true
}

/// `NPCAI_Pathfinder::FindGridPath(Pos, Speed, PathMarker, UseDirectPath)`。
///
/// 省略：玩家控制的 NPC（`SimulatePlayerMovement`）、大体型实体的多点代价图与目标格修正、
/// 类型 0x10 房间的门特例、恐惧/燃烧时的 `EvadeTarget`（由调用方处理）。
#[allow(clippy::too_many_arguments)]
pub fn find_grid_path(
    pf: &mut Pathfinder,
    e: &mut Entity,
    room: &mut Room,
    target: Vec2,
    speed: f32,
    path_marker: i32,
    use_direct_path: bool,
    game_frame: u32,
    rng: &mut Rng,
) {
    // 第 96–120 行：混乱且未被魅惑：随机游走方向，速度按 speed 施加
    if e.has_flag(flags::CONFUSION) && !e.has_flag(flags::CHARM) {
        move_randomly_boss(pf, e, room, rng, true, game_frame);
        let dir = e.vel.normalized();
        e.vel = Vec2::ZERO;
        let f = e.frame_friction;
        if f != 0.0 {
            e.vel.y += (e.speed_mult * dir.y * speed) / f;
            e.vel.x += (e.speed_mult * dir.x * speed) / f;
        }
    }

    let dt = e.frame_delta();
    let w = room.width;
    let h = room.height;
    let ncells = (w * h).min(MAX_CELLS as i32);

    // 第 128–147 行：大体型偏移（直径 > 40 时把参考点移 19.96）
    let two_x = e.size * e.size_mult.x * 2.0;
    let two_y = e.size * e.size_mult.y * 2.0;
    let big = two_x > TILE || two_y > TILE;
    let mut off = Vec2::ZERO;
    if two_y > TILE {
        off.x += 19.96;
    }
    if two_x > TILE {
        off.y += 19.96;
    }
    let free_mover = is_free_mover(e) || path_marker < 0; // local_2f
    let familiar_rule = e.etype == 3 && path_marker <= 0; // local_2d
    let ref_pos = e.pos - off;

    // 第 150–190 行：目标格（目标在实心格里时用备份位置）
    let mut tcell = room.grid_index(target);
    let target_solid = (0..ncells).contains(&tcell) && room.path[tcell as usize] > 999;
    if !pf.can_crush_rocks {
        if tcell < 0 || target_solid {
            if !pf.target_backup.is_zero() {
                tcell = room.grid_index(pf.target_backup);
            }
        } else {
            pf.target_backup = target;
        }
    } else if tcell >= 0 && !target_solid {
        pf.target_backup = target;
    } else if !pf.target_backup.is_zero() {
        tcell = room.grid_index(pf.target_backup);
    }

    // 第 191–214 行：自身格登记（带滞后）
    let own_idx = room.grid_index(ref_pos);
    let own_center = room.cell_center(if own_idx >= 0 { own_idx } else { 0 });
    if own_idx != pf.grid_index {
        pf.interp_countdown += dt * -10;
        if pf.interp_countdown < 0 {
            pf.grid_index = own_idx;
            let s = if speed <= 0.2 { 0.2 } else { speed };
            pf.interp_countdown = (20.0 / s) as i32;
        }
    }

    // 第 216–247 行：前方格有标记则降速 0.7；给自身格打标记
    let mut speed_scale = 1.0f32;
    if !familiar_rule {
        if every_n_frames(game_frame, 3, 1) {
            let dir = e.vel.normalized();
            let ahead = Vec2::new(dir.x * TILE + ref_pos.x, dir.y * TILE + ref_pos.y);
            let aidx = room.grid_index(ahead);
            let v = if (0..ncells).contains(&aidx) {
                room.path[aidx as usize]
            } else {
                0
            };
            let v = Room::path_value_for_class(v, e.grid_collision_class);
            if (v as u32).wrapping_sub(1) < 0x3b5 {
                speed_scale = 0.7;
            }
        }
        if path_marker > -1 && (0..ncells).contains(&pf.grid_index) {
            let v = room.path[pf.grid_index as usize];
            if v < 0x3b6 {
                room.set_marker(pf.grid_index, path_marker);
            }
        }
    }

    // 第 249–274 行：每 10 帧刷新一次视线
    if every_n_frames(game_frame, 10, e.index as u32) {
        let tidx = room.grid_index(target);
        let tcenter = room.cell_center(if tidx >= 0 { tidx } else { 0 });
        let threshold = if free_mover || familiar_rule { 900 } else { 0 };
        if e.grid_collision_class == gcc::WALLS {
            pf.has_direct_path = true;
        } else {
            pf.has_direct_path = tidx >= 0 && room.line_check(own_center, tcenter, threshold);
        }
    }

    // 第 276–288 行：直线追击距离阈值
    let mut direct_range = if use_direct_path { 750.0 } else { 70.0 };
    if big && !use_direct_path {
        direct_range += (e.size * e.size_mult.x).max(e.size * e.size_mult.y);
    }

    let mut move_target = Vec2::ZERO;
    let mut direct = false;
    if pf.has_direct_path {
        let dist = (target - e.pos).length();
        if dist < direct_range {
            move_target = target - off;
            pf.recompute_countdown += -dt * 10;
            pf.interp_countdown += -dt * 50;
            direct = true;
        }
    }

    if !direct {
        // 第 290–306 行：需要时重算代价图
        if pf.grid_index != tcell {
            let own_solid =
                (0..ncells).contains(&pf.grid_index) && room.path[pf.grid_index as usize] == 1000;
            if own_solid || pf.recompute_countdown > 0 {
                pf.recompute_countdown -= dt;
            } else {
                pf.recompute_countdown = (rng.next_u32() % 10 + 10) as i32;
                recompute(pf, room, tcell, free_mover, ncells, w, h);
            }
        }
        if pf.path_found && pf.grid_index >= 0 {
            // 第 536–592 行：在自身格四邻里挑代价最小（负值最大）的格子
            let idx = pf.grid_index;
            let col = idx % w;
            let row = idx / w;
            let cands = [
                if col >= 1 { idx - 1 } else { -1 },
                if row >= 1 { idx - w } else { -1 },
                if col + 1 < w { idx + 1 } else { -1 },
                if row + 1 < h { idx + w } else { -1 },
            ];
            let mut best = idx;
            let mut best_val = -1000i32; // 0xfffffc18
            for c in cands.iter() {
                if (0..ncells).contains(c) {
                    let v = pf.cost_map[*c as usize];
                    if v < 0 && v > best_val {
                        best_val = v;
                        best = *c;
                    }
                }
            }
            move_target = room.cell_center(best);
        }
    }

    // 第 596–622 行：施加速度
    if move_target.is_zero() {
        e.frame_friction *= 0.85;
    } else {
        let d = Vec2::new(move_target.x - ref_pos.x, move_target.y - ref_pos.y);
        let len = d.length();
        if len <= 1.0 {
            e.frame_friction *= 0.5;
        } else {
            let mut acc = len * 0.1;
            if acc >= 2.0 {
                acc = 2.0;
            }
            acc = acc * speed_scale * speed;
            e.frame_friction *= 0.7;
            let f = e.frame_friction;
            if f != 0.0 {
                e.vel.x += ((d.x / len) * acc * e.speed_mult) / f;
                e.vel.y += ((d.y / len) * acc * e.speed_mult) / f;
            }
        }
    }
}

/// 第 307–535 行：从目标格出发的 Dijkstra（原版用 push_heap/pop_heap 维护的优先队列）。
/// 可通行：代价图值在 `[0, 950)`；空地代价 +1，带标记的格 +7（自由移动者一律 +1）；
/// 代价达到 `w*h/2` 停止；访问过的格写成 `-(cost)`。
fn recompute(
    pf: &mut Pathfinder,
    room: &Room,
    tcell: i32,
    free_mover: bool,
    ncells: i32,
    w: i32,
    h: i32,
) {
    for i in 0..MAX_CELLS {
        pf.cost_map[i] = if (i as i32) < ncells { room.path[i] } else { 0 };
    }
    pf.path_found = false;
    if !(0..ncells).contains(&tcell) {
        return;
    }
    let limit = (h * w) / 2;
    let mut heap: BinaryHeap<Reverse<(i32, i32)>> = BinaryHeap::new();
    heap.push(Reverse((0, tcell)));
    while let Some(Reverse((cost, idx))) = heap.pop() {
        if cost >= limit {
            break;
        }
        let col = idx % w;
        let row = idx / w;
        let neighbors = [
            if col >= 1 { idx - 1 } else { -1 },
            if row >= 1 { idx - w } else { -1 },
            if col + 1 < w { idx + 1 } else { -1 },
            if row + 1 < h { idx + w } else { -1 },
        ];
        for n in neighbors.iter() {
            if !(0..ncells).contains(n) {
                continue;
            }
            let v = pf.cost_map[*n as usize];
            if !(0..0x3b6).contains(&v) {
                continue;
            }
            let ncost = if v < 1 || free_mover {
                cost + 1
            } else {
                cost + 7
            };
            pf.cost_map[*n as usize] = -ncost;
            if *n == pf.grid_index {
                pf.path_found = true;
            }
            heap.push(Reverse((ncost, *n)));
        }
        if pf.path_found {
            break;
        }
    }
}

/// `NPCAI_Pathfinder::MoveRandomlyBoss(IgnoreStatusEffects)`（FUN_007e6300）。
/// 随机取 40 单位外的一个无碰撞点为目标，以加速度 0.5 靠近；到达（<3）、撞到网格或几乎停下时换目标。
/// 返回 true 表示本帧由其他逻辑接管（自由移动者转 FindGridPath；恐惧/燃烧的 EvadeTarget 这里未翻译）。
#[allow(clippy::approx_constant)] // 原版就是 3.14（DAT_00baa704），不是 π
pub fn move_randomly_boss(
    pf: &mut Pathfinder,
    e: &mut Entity,
    room: &mut Room,
    rng: &mut Rng,
    ignore_status: bool,
    game_frame: u32,
) -> bool {
    if !ignore_status {
        if is_free_mover(e) {
            let target = e.pos; // 原版：CalcTargetPosition → FindGridPath(target, 0.5, 0, true)
            find_grid_path(pf, e, room, target, 0.5, 0, true, game_frame, rng);
            return true;
        }
        if e.has_flag(flags::BURN | flags::FEAR) {
            return true;
        }
    }
    if pf.random_target.is_zero() {
        let ang = rng.next_f32() * 3.14 * 2.0; // DAT_00baa704 = 3.14
        let cand = Vec2::new(ang.cos() * TILE + e.pos.x, ang.sin() * TILE + e.pos.y);
        if room.grid_collision_at(cand) == 0 {
            pf.random_target = cand;
        }
    }
    if !pf.random_target.is_zero() {
        let d = Vec2::new(pf.random_target.x - e.pos.x, pf.random_target.y - e.pos.y);
        let len = d.length();
        if len < 3.0
            || e.collides_with_grid
            || (e.vel.length_sq() < 0.1 && rng.next_u32().is_multiple_of(10))
        {
            pf.random_target = Vec2::ZERO;
            return true;
        }
        if len > 0.0 {
            let f = e.frame_friction;
            if f != 0.0 {
                e.vel.x += ((d.x / len) * 0.5 * e.speed_mult) / f;
                e.vel.y += ((d.y / len) * 0.5 * e.speed_mult) / f;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics;

    #[test]
    fn dijkstra_routes_around_a_wall_of_rocks() {
        let mut room = Room::new_rectangular(15, 9);
        for row in 1..=6 {
            room.add_rock(room.index_of(7, row));
        }
        let mut pf = Pathfinder {
            grid_index: room.index_of(2, 2),
            ..Default::default()
        };
        let tcell = room.index_of(12, 2);
        recompute(
            &mut pf,
            &room,
            tcell,
            false,
            room.cell_count(),
            room.width,
            room.height,
        );
        assert!(pf.path_found);
        // 从自身格出发，沿代价递减走到目标格必须绕过岩石列
        // 目标格本身会被再次展开写成 -2（原版行为），所以沿代价递减只能走到与目标相邻（代价 1）的格子
        assert_eq!(pf.cost_map[tcell as usize], -2);
        let mut idx = pf.grid_index;
        let mut steps = 0;
        while pf.cost_map[idx as usize] != -1 && steps < 100 {
            let col = idx % room.width;
            let row = idx / room.width;
            let cands = [idx - 1, idx - room.width, idx + 1, idx + room.width];
            let mut best = idx;
            let mut best_val = -1000;
            for (k, c) in cands.iter().enumerate() {
                let ok = match k {
                    0 => col > 0,
                    1 => row > 0,
                    2 => col + 1 < room.width,
                    _ => row + 1 < room.height,
                };
                if ok {
                    let v = pf.cost_map[*c as usize];
                    if v < 0 && v > best_val {
                        best_val = v;
                        best = *c;
                    }
                }
            }
            assert_ne!(best, idx, "stuck at {}", idx);
            assert_ne!(room.grid_collision(best), 3);
            idx = best;
            steps += 1;
        }
        assert_eq!(pf.cost_map[idx as usize], -1);
        assert!((idx - tcell).abs() == 1 || (idx - tcell).abs() == room.width);
        assert!(
            steps >= 13,
            "path must go around the rock column: {} steps",
            steps
        );
    }

    #[test]
    fn direct_chase_reaches_target_speed() {
        let mut room = Room::new_rectangular(15, 9);
        let mut e = Entity::new_gaper(1, 1, Vec2::new(100.0, 280.0));
        e.flags &= !flags::APPEAR;
        let target = Vec2::new(500.0, 280.0);
        let mut pf = Pathfinder::default();
        let mut rng = Rng::new(1);
        let mut last_x = e.pos.x;
        let mut per_frame = 0.0;
        for frame in 0..40u32 {
            physics::pre_update(&mut e);
            find_grid_path(
                &mut pf, &mut e, &mut room, target, 0.498, 900, true, frame, &mut rng,
            );
            physics::update(&mut e);
            per_frame = e.pos.x - last_x;
            last_x = e.pos.x;
        }
        // 稳态：vel = 0.7*vel + 2*0.498 → 0.996/0.3 ≈ 3.32 单位/帧
        assert!(
            (per_frame - 3.32).abs() < 0.05,
            "per-frame displacement {}",
            per_frame
        );
    }
}
