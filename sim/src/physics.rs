//! `Entity::PreUpdate` / `Entity::Update`、实体-网格碰撞（NPC 版与玩家版）、碰撞采样点、圆形推挤。
//!
//! 出处：`decompiled/02/002AE7D0_FUN_006ae7d0.c`（PreUpdate）、`002AE820_FUN_006ae820.c`
//! （Update，RVA 0x002AE820）、`002A9C90_FUN_006a9c90.c`（网格碰撞，RVA 0x002A9C90）、
//! `002AB440_FUN_006ab440.c`（玩家网格碰撞 `PlayerCollideWithGrid`，RVA 0x002AB440）、
//! `002AC2A0_FUN_006ac2a0.c`（SetSize，RVA 0x002AC2A0）、`002B2940_FUN_006b2940.c`（圆碰撞推挤）。

use crate::entity::{flags, grid_collision_class as gcc, Entity, EntityKind, TYPE_PLAYER};
use crate::math::Vec2;
use crate::room::{grid_collision, Room};

/// `Entity::PreUpdate`（RVA 0x002AE7D0）：记录上一帧位置与速度。
pub fn pre_update(e: &mut Entity) {
    e.last_pos = e.pos;
    e.last_vel = e.vel;
}

/// `Entity::Update`（RVA 0x002AE820）中影响运动、计时与血量的部分。
///
/// 省略：精灵动画推进（第 107–118 行）、颜色参数、房间级物品效果对敌人施加的魅惑/混乱/恐惧
/// （`FUN_00665c60` 系列）、流血轨迹、磁化、水面特效、受伤闪白。
pub fn update(e: &mut Entity) {
    let interp = e.has_flag(flags::INTERPOLATION_UPDATE);

    // 第 105–107 行：时间累加器（0x3a0 <- 0x3a4，0x3a4 += speedMult）
    if !interp {
        e.time_prev = e.time_cur;
        e.time_cur += e.speed_mult;
    }

    // 第 122–192 行：击退
    if e.has_flag(flags::KNOCKED_BACK) {
        let collided = e.collides_with_grid;
        let mut keep = true;
        if !collided {
            // 速度已经比击退向量大且方向相反时结束击退
            let vlen = e.vel.length();
            let klen = e.knockback_dir.length();
            if !(vlen < klen || e.knockback_dir.dot(e.vel) >= 0.0) {
                keep = false;
            }
        }
        let mut applied = false;
        if keep {
            let klen = e.knockback_dir.length();
            if klen >= 2.0 {
                e.frame_friction = e.friction * 0.95; // DAT_00baa418
                let hit = if !e.has_flag(flags::ICE_FROZEN) {
                    e.knockback_dir = e.knockback_dir * 0.95;
                    e.collides_with_grid
                } else {
                    collided
                };
                if hit {
                    if e.has_flag(flags::APPLY_IMPACT_DAMAGE) {
                        // 第 158–169 行：撞墙反弹成随机方向 * 0.8 并受伤；这里只保留反弹缩放，
                        // 伤害与随机方向留待 TakeDamage 翻译后补上。
                        e.knockback_dir = e.knockback_dir * 0.8;
                        e.flags &= !flags::APPLY_IMPACT_DAMAGE;
                    }
                    e.knockback_countdown /= 2;
                }
                e.vel = e.knockback_dir;
                applied = true;
            }
        }
        if !applied {
            e.flags &= !(flags::KNOCKED_BACK | flags::APPLY_IMPACT_DAMAGE);
            e.knockback_countdown = 0;
        }
    }

    // 第 194–205 行：摩擦与积分
    let s = e.speed_mult;
    let mut f = e.frame_friction;
    if s != 1.0 {
        f /= (f + s) - f * s;
    }
    let old_vel = e.vel;
    e.vel = Vec2::new(f * old_vel.x, f * old_vel.y);
    e.pos.y += s * f * old_vel.y;
    e.pos.x += s * f * old_vel.x;
    e.frame_friction = e.friction;

    if interp {
        return;
    }

    // 第 264–398 行：状态倒计时（只保留会改变运动/伤害的项）
    dec_clear(&mut e.freeze_countdown, &mut e.flags, flags::FREEZE);
    if e.poison_countdown > 0 {
        e.poison_countdown -= 1;
        if e.poison_countdown < 1 {
            e.flags &= !flags::POISON;
            e.poison_damage = 2.0; // param_1[200] = 0x40000000
        }
    }
    dec_clear(&mut e.slowing_countdown, &mut e.flags, flags::SLOW);
    dec_clear(&mut e.charmed_countdown, &mut e.flags, flags::CHARM);
    dec_clear(&mut e.confusion_countdown, &mut e.flags, flags::CONFUSION);
    dec_clear(&mut e.fear_countdown, &mut e.flags, flags::FEAR);
    if e.burn_countdown > 0 {
        e.burn_countdown -= 1;
        if e.burn_countdown < 1 {
            e.flags &= !flags::BURN;
            e.burn_damage = 2.0;
        }
    }
    dec_clear(&mut e.midas_countdown, &mut e.flags, flags::MIDAS_FREEZE);
    if e.knockback_countdown > 0 {
        e.knockback_countdown -= 1;
        if e.knockback_countdown < 1 {
            e.flags &= !(flags::KNOCKED_BACK | flags::APPLY_IMPACT_DAMAGE);
        }
    }
    if e.anim_freeze > 0 {
        e.anim_freeze -= 1;
    }

    // 第 490–522 行：毒/燃烧每 20 帧结算一次（仅敌人与玩家）
    if e.is_enemy_type() || e.etype == TYPE_PLAYER {
        if e.has_flag(flags::POISON) {
            let c = e.poison_tick;
            e.poison_tick += 1;
            if c % 20 == 1 {
                e.pending_damage += e.poison_damage; // 原版走 TakeDamage(dmg, DAMAGE_POISON_BURN, ..., 30)
            }
        }
        if e.has_flag(flags::BURN) {
            let c = e.burn_tick;
            e.burn_tick += 1;
            if c % 20 == 1 {
                e.pending_damage += e.burn_damage;
            }
        }
    }

    // 第 640–727 行：把 `TakeDamage` 累计到 +0x14 的伤害记录逐条扣血，血量 ≤ 0 时 `Kill()`；
    // 之后清空记录（第 727 行 `param_1[5] = 0`）。玩家的血量由 Entity_Player::TakeDamage 直接处理。
    if e.pending_damage > 0.0 && e.kind != EntityKind::Player {
        e.hp -= e.pending_damage;
        if e.hp <= 0.0 {
            if e.hp < 0.0 && !e.is_enemy_type() {
                e.hp = 0.0;
            }
            e.dead = true;
        }
    }
    e.pending_damage = 0.0;
}

fn dec_clear(counter: &mut i32, flags: &mut u64, flag: u64) {
    if *counter > 0 {
        *counter -= 1;
        if *counter < 1 {
            *flags &= !flag;
        }
    }
}

/// `Entity::SetSize`（RVA 0x002AC2A0）：`size`、`sizeMulti` 与 `numGridCollisionPoints` 个圆周采样点，
/// 角度 `2πk/N`（原版先推偶数 k 再推奇数 k，顺序不影响使用），半径 `size*sizeMulti`。
pub fn set_size(e: &mut Entity, size: f32, size_mult: Vec2, num_points: i32) {
    e.size = size;
    e.size_mult = size_mult;
    e.collision_points.clear();
    let pi = std::f32::consts::PI; // DAT_00baa708 = 3.14159
    let mut push = |k: i32| {
        let ang = ((k + k) as f32 * pi) / num_points as f32;
        e.collision_points.push(Vec2::new(
            ang.cos() * size * size_mult.x,
            ang.sin() * size * size_mult.y,
        ));
    };
    let mut k = 0;
    while k < num_points {
        push(k);
        k += 2;
    }
    let mut k = 1;
    while k < num_points {
        push(k);
        k += 2;
    }
}

/// 实体-网格碰撞。`interp=false` 是逻辑帧调用（EntityList 碰撞阶段，FUN_00419e70 尾部），
/// `interp=true` 是插值帧调用（Entity::Interpolate）。
///
/// - 非玩家：FUN_006a9c90（RVA 0x002A9C90）。反弹系数 1.8（DAT_00baa5b4），推出后按撞墙分量衰减速度。
/// - 玩家：FUN_006ab440（RVA 0x002AB440，`PlayerCollideWithGrid`）。只在自身格越界/是墙时夹到房间矩形，
///   GROUND 类把 `WALL_EXCEPT_PLAYER(5)` 视为可通过，反弹系数 0.9（DAT_00baa3e0），推出后不再衰减速度。
///
/// 省略：矿车、子弹类(4)的特殊房、Bum 家族与眼泪的特殊分支、按类型的反弹倍率（99/0xd9/0xdc/0x393/0x37d）、
/// 敌人卡在实心格里时的 `FindFreeTilePosition` 脱困、玩家撞门/撞可破坏物的处理。
pub fn resolve_grid_collision(e: &mut Entity, room: &Room, interp: bool) {
    let player = e.kind == EntityKind::Player;
    if !interp {
        e.collides_with_grid = false;
    }
    let class = e.grid_collision_class;
    if class == gcc::NONE || e.collision_points.is_empty() && class != gcc::BULLET {
        return;
    }
    let clamped = room.clamp_position(e.pos, 0.0);
    let mut bound_normal = Vec2::ZERO;

    if player {
        // FUN_006ab440 第 116–135 行
        let idx = room.grid_index(e.pos);
        if idx < 0 || room.grid_collision(idx) == grid_collision::WALL {
            if !interp {
                e.collides_with_grid = e.pos != clamped;
            }
            e.pos = clamped;
        }
    } else {
        match class {
            gcc::WALLS_X => {
                if e.pos.x != clamped.x && !interp {
                    e.collides_with_grid = true;
                    e.grid_hit_velocity = e.vel;
                    e.grid_normal = Vec2::new(if clamped.x < e.pos.x { -1.0 } else { 1.0 }, 0.0);
                }
                e.pos.x = clamped.x;
                return;
            }
            gcc::WALLS_Y => {
                if e.pos.y != clamped.y && !interp {
                    e.collides_with_grid = true;
                    e.grid_hit_velocity = e.vel;
                    e.grid_normal = Vec2::new(0.0, if clamped.y < e.pos.y { -1.0 } else { 1.0 });
                }
                e.pos.y = clamped.y;
                return;
            }
            gcc::WALLS | gcc::GROUND | gcc::NOPITS | gcc::PITSONLY => {
                if !interp {
                    if e.pos == clamped {
                        e.collides_with_grid = false;
                    } else {
                        e.collides_with_grid = true;
                        e.grid_hit_velocity = e.vel;
                        if e.pos.x != clamped.x {
                            bound_normal.x += if clamped.x < e.pos.x { -1.0 } else { 1.0 };
                        }
                        if e.pos.y != clamped.y {
                            bound_normal.y += if clamped.y < e.pos.y { -1.0 } else { 1.0 };
                        }
                    }
                }
                e.pos = clamped;
            }
            gcc::BULLET => {
                // 第 145–187 行：只报告碰撞，不修正位置（眼泪、弹幕）
                if interp {
                    return;
                }
                let c = room.grid_collision_at(e.pos);
                if c == grid_collision::WALL
                    || c == grid_collision::WALL_EXCEPT_PLAYER
                    || c == grid_collision::SOLID
                {
                    e.collides_with_grid = true;
                    e.grid_hit_velocity = e.vel;
                }
                return;
            }
            _ => {}
        }
    }

    // 累加所有落在障碍里的采样点偏移（NPC 第 194–330 行；玩家第 158–354 行）
    let offending = |c: i32| -> bool {
        match class {
            gcc::GROUND => {
                if player {
                    c != grid_collision::NONE && c != grid_collision::WALL_EXCEPT_PLAYER
                } else {
                    c != grid_collision::NONE
                }
            }
            gcc::PITSONLY => c != grid_collision::PIT,
            gcc::NOPITS => c != grid_collision::NONE && c != grid_collision::PIT,
            _ => {
                if player {
                    c == grid_collision::WALL
                } else {
                    (c == grid_collision::WALL || c == grid_collision::WALL_EXCEPT_PLAYER)
                        && room.room_type != 0x10
                }
            }
        }
    };
    let mut sum = Vec2::ZERO;
    for off in e.collision_points.iter() {
        let idx = room.grid_index(e.pos + *off);
        if offending(room.grid_collision(idx)) {
            sum += *off;
        }
    }
    if sum.is_zero() || sum.length_sq() < 0.001 {
        if !interp {
            e.grid_normal = bound_normal;
        }
        return;
    }

    let n = sum.normalized(); // FUN_00a0ffc0
    if !interp {
        e.grid_normal = n;
    }
    // into = -(n·vel)：为负表示正在往墙里走
    let into = (-n.x) * e.vel.x - e.vel.y * n.y;
    if into >= 0.0 {
        return;
    }
    if !interp {
        e.grid_hit_velocity = e.vel;
        e.collides_with_grid = true;
        // NPC：into·1.8 − 0.1（DAT_00baa5b4/DAT_00baa120）；玩家：into·0.9 − 0.1（DAT_00baa3e0）
        let k = if player {
            into * 0.9 - 0.1
        } else {
            into * 1.8 - 0.1
        };
        e.vel = Vec2::new(e.vel.x + k * n.x, e.vel.y + k * n.y);
    }

    // 沿法线二分，把实体推到刚好不与障碍相交的位置（NPC 第 512–630 行；玩家第 491–561 行）
    let speed = e.vel.length();
    if speed > 0.0 {
        let radius = Vec2::new(n.x * e.size_mult.x, n.y * e.size_mult.y).length() * e.size;
        let mut lo = radius - speed.max(2.0);
        let mut hi = radius;
        let iterations = if class == gcc::GROUND { 12 } else { 6 };
        for _ in 0..iterations {
            let mid = (hi - lo) * 0.5 + lo;
            let p = Vec2::new(mid * n.x + e.pos.x, mid * n.y + e.pos.y);
            let c = room.grid_collision(room.grid_index(p));
            let blocked = match class {
                gcc::GROUND => {
                    if player {
                        c != grid_collision::NONE && c != grid_collision::WALL_EXCEPT_PLAYER
                    } else {
                        c != grid_collision::NONE
                    }
                }
                gcc::PITSONLY => c != grid_collision::PIT,
                gcc::NOPITS => c != grid_collision::NONE && c != grid_collision::PIT,
                _ => {
                    if player {
                        c == grid_collision::WALL
                    } else {
                        c == grid_collision::WALL || c == grid_collision::WALL_EXCEPT_PLAYER
                    }
                }
            };
            if blocked {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        e.pos.x += n.x * (lo - radius);
        e.pos.y += n.y * (lo - radius);
    }

    // NPC 第 632–635 行：按撞墙分量比例衰减速度（最多 50%）；玩家版本没有这一步
    if !player && !interp && speed > 0.0 {
        let factor = 1.0 - ((-into) / speed) * 0.5;
        e.vel = Vec2::new(factor * e.vel.x, factor * e.vel.y);
    }
}

/// 两个同缩放圆形实体的重叠推挤（FUN_006b2940，RVA 0x002B2940）。
/// 返回是否发生碰撞。省略：碰撞配对规则中的物品/类型特例、火焰弹的音效、`FLAG_NO_PHYSICS_KNOCKBACK`
/// 之外的玩家特殊处理（玩家侧 `FUN_008279a0` 分支按普通实体处理）。
pub fn circle_push(a: &mut Entity, b: &mut Entity) -> bool {
    // 原版先把两者按类型排序（类型小者为 fVar6），推挤方向从 a 指向 b
    let (lo, hi) = if b.etype < a.etype { (b, a) } else { (a, b) };
    let d = Vec2::new(hi.pos.x - lo.pos.x, hi.pos.y - lo.pos.y);
    let rsum = lo.size + hi.size;
    let dist_sq = d.length_sq();
    if dist_sq >= rsum * rsum {
        return false;
    }
    let dist = dist_sq.sqrt();
    if dist < 0.0001 {
        return true; // DAT_00baa010：重合时不推
    }
    let mut n = Vec2::new(d.x / dist, d.y / dist);
    let mut ndist = dist;
    // 第 44–60 行：相对速度较大且相向时，把方向向相对速度方向偏折
    let rel = Vec2::new(lo.vel.x - hi.vel.x, lo.vel.y - hi.vel.y);
    if rel.length() > rsum * 0.5 && d.dot(rel) < 0.0 {
        let a_dir = d.normalized();
        let r_dir = rel.normalized();
        let k = a_dir.dot(r_dir);
        let adj = Vec2::new(d.x + 2.0 * k * d.x, d.y + 2.0 * k * d.y);
        ndist = adj.length();
        if ndist > 0.0 {
            n = Vec2::new(adj.x / ndist, adj.y / ndist);
        }
    }
    let ratio_hi = lo.mass / (lo.mass + hi.mass); // hi 被推的份额 = lo 的质量占比
    let overlap = (rsum - ndist) / ndist;
    let mut share_lo = 1.0 - ratio_hi;
    if lo.etype == TYPE_PLAYER {
        share_lo *= 0.5; // 第 82–84 行
    }
    let rel_len = rel.length();
    let push_vel = Vec2::new(n.x * rel_len, n.y * rel_len);
    let moved = Vec2::new(d.x * overlap, d.y * overlap);
    if !lo.has_flag(flags::NO_PHYSICS_KNOCKBACK) {
        lo.pos.x -= moved.x * share_lo;
        lo.pos.y -= moved.y * share_lo;
        lo.vel.x -= push_vel.x * share_lo;
        lo.vel.y -= push_vel.y * share_lo;
    }
    if !hi.has_flag(flags::NO_PHYSICS_KNOCKBACK) {
        hi.pos.x += moved.x * ratio_hi;
        hi.pos.y += moved.y * ratio_hi;
        hi.vel.x += push_vel.x * ratio_hi;
        hi.vel.y += push_vel.y * ratio_hi;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friction_integration_matches_formula() {
        let mut e = Entity::new_player_stub(1, Vec2::new(100.0, 200.0));
        e.vel = Vec2::new(3.0, 0.0);
        e.frame_friction = 0.7;
        pre_update(&mut e);
        update(&mut e);
        assert!((e.vel.x - 2.1).abs() < 1e-5);
        assert!((e.pos.x - 102.1).abs() < 1e-5);
        assert_eq!(e.frame_friction, 1.0);
        assert_eq!(e.frame_delta(), 1);
    }

    #[test]
    fn set_size_generates_points_on_circle() {
        let mut e = Entity::new_player_stub(1, Vec2::ZERO);
        set_size(&mut e, 13.0, Vec2::new(1.0, 1.0), 12);
        assert_eq!(e.collision_points.len(), 12);
        for p in &e.collision_points {
            assert!((p.length() - 13.0).abs() < 1e-4);
        }
    }

    #[test]
    fn wall_stops_ground_entity() {
        let room = Room::new_rectangular(15, 9);
        // 半径 13，边缘在 x=62；本帧向左移动 6 后边缘穿入墙面 (x=60) 4 个单位
        let mut e = Entity::new_gaper(1, 1, Vec2::new(75.0, 200.0));
        e.vel = Vec2::new(-6.0, 0.0);
        pre_update(&mut e);
        update(&mut e);
        assert!((e.pos.x - 69.0).abs() < 1e-4);
        resolve_grid_collision(&mut e, &room, false);
        assert!(e.collides_with_grid);
        // 二分推出后边缘正好贴在墙面上：pos.x ≈ 73
        assert!((e.pos.x - 73.0).abs() < 0.05, "pos.x={}", e.pos.x);
        // 速度被反弹：-6 + (-6*1.8-0.1)*(-1) = 4.9，再按撞墙分量衰减
        assert!(e.vel.x > 0.0 && e.vel.x < 4.9, "vel.x={}", e.vel.x);
    }

    #[test]
    fn wall_stops_player_without_bounce_amplification() {
        let room = Room::new_rectangular(15, 9);
        // 半径 10，边缘 x=61；向左 4.41 后边缘 56.6，穿入 3.4
        let mut e = Entity::new_player(1, Vec2::new(71.0, 200.0));
        e.vel = Vec2::new(-4.41, 0.0);
        pre_update(&mut e);
        update(&mut e);
        resolve_grid_collision(&mut e, &room, false);
        assert!(e.collides_with_grid);
        // -4.41 + (-4.41*0.9 - 0.1)*(-1) = -0.341（只剩一成向墙速度，无反向弹开）
        assert!((e.vel.x + 0.341).abs() < 1e-3, "vel.x={}", e.vel.x);
        // 二分区间按反弹后的速度取 [radius-2, radius]，一帧最多推出 2 单位：66.59 → 68.59
        assert!((e.pos.x - 68.59).abs() < 0.05, "pos.x={}", e.pos.x);
        // 第二帧继续推出并贴到墙面（边缘 x=60）
        pre_update(&mut e);
        update(&mut e);
        resolve_grid_collision(&mut e, &room, false);
        assert!((e.pos.x - 70.0).abs() < 0.05, "pos.x={}", e.pos.x);
        assert!(e.vel.x.abs() < 0.1, "vel.x={}", e.vel.x);
    }

    #[test]
    fn pending_damage_is_applied_in_update() {
        let mut g = Entity::new_gaper(1, 1, Vec2::new(200.0, 200.0));
        g.pending_damage = 4.0;
        update(&mut g);
        assert!((g.hp - 6.0).abs() < 1e-6);
        assert_eq!(g.pending_damage, 0.0);
        g.pending_damage = 7.0;
        update(&mut g);
        assert!(g.dead && g.hp < 0.0);
    }
}
