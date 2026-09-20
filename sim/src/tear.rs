//! 眼泪：`Entity_Tear::Init`（FUN_006660a0，RVA 0x002660A0）、`Entity_Player::FireTear`（0x00390AF0）、
//! `GetTearHitParams`（0x003BF6C0）、`Entity_Tear::Update`（0x002670F0，Ghidra 重反编译于
//! `decompiled-retry/006670f0.c` 与汇编 `failed_disassembly/002670F0_FUN_006670f0.asm`）、
//! `Entity_Tear::HandleCollision`（0x00273660，`decompiled-retry/00673660.c`）。
//!
//! 出处与行号见 `analysis/docs/J460_PLAYER_TEAR_MODEL.md` §5–§7。

use crate::entity::{grid_collision_class, Entity, EntityKind, TYPE_TEAR};
use crate::math::Vec2;
use crate::rng::Rng;
use crate::world::Event;

/// `TearFlags`（enums.lua，1<<n）子集，加上 J460 用到的内部位。
pub mod tear_flags {
    pub const SPECTRAL: u128 = 1 << 0;
    pub const PIERCING: u128 = 1 << 1;
    pub const HOMING: u128 = 1 << 2;
    pub const PERSISTENT: u128 = 1 << 9;
    pub const WAIT: u128 = 1 << 17;
    pub const KNOCKBACK: u128 = 1 << 24;
    pub const TRACTOR_BEAM: u128 = 1 << 42;
    pub const POP: u128 = 1 << 58;
    pub const HYDROBOUNCE: u128 = 1 << 61;
    pub const PUNCH: u128 = 1 << 64;
    pub const DECELERATE: u128 = 1 << 82;
    /// 内部位 114（`.data` 常量 DAT_00c34780 = 1<<114）。置位时 `Entity_Tear::Update` 走"平飞到
    /// 年龄 > TearRange×0.25 后再下落"的模型，命中时对敌人做质量推挤；不置位时走"每帧
    /// fallingSpeed = 0.9·v + 0.1 + accel"的持续下落模型，且 TearRange 对飞行无影响。
    /// 静态分析未找到给普通眼泪置位的代码（默认标志 TEAR_NORMAL = 0），但只有置位模型能解释
    /// 射程道具生效与眼泪推动敌人，故默认置位；**待实机验证**（见文档 §9）。
    pub const INTERNAL_RANGE_FALL: u128 = 1 << 114;
}

/// `Entity_Tear` 专有字段（偏移见 EntityTear.zhl，J460 相同）。
#[derive(Clone, Debug, PartialEq)]
pub struct TearState {
    /// +0x410 Height（负值为离地高度，≥ −5 视为落地）。
    pub height: f32,
    /// +0x414 FallingSpeed，+0x418 FallingAcceleration。
    pub falling_speed: f32,
    pub falling_accel: f32,
    /// +0x420 Scale（+0x424 BaseScale 同值）。
    pub scale: f32,
    /// +0x454 BaseDamage（碰撞伤害另存在 Entity+0x388）。
    pub base_damage: f32,
    /// +0x820 TearRange。
    pub range: f32,
    /// +0x460 KnockbackMultiplier（默认 1.7）。
    pub knockback_mult: f32,
    /// +0x428 128 位 TearFlags。
    pub flags: u128,
    /// +0x438 WaitFrames。
    pub wait_frames: i32,
    /// +0x450 TearIndex。
    pub tear_idx: i32,
    /// +0x444 已命中列表（穿透用）。
    pub hit_list: Vec<u32>,
    /// 死因（观测/调试）：0 存活，1 落地，2 撞墙，3 命中。
    pub death_cause: u8,
}

impl TearState {
    pub fn has(&self, f: u128) -> bool {
        self.flags & f != 0
    }
}

/// `GetTearHitParams` 的普通眼泪结果。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TearParams {
    pub damage: f32,
    pub scale: f32,
    pub height: f32,
    pub mass_mult: f32,
    pub knockback_mult: f32,
    pub speed_mult: f32,
    pub flags: u128,
}

/// `Entity_Player::GetTearHitParams(out, WEAPON_TEARS, damageScale=1, displacement, source=player)`
/// （FUN_007bf6c0）无道具路径：伤害 = Damage×scale；尺寸第 1869 行：
/// `(sqrt(damage)·0.23 + S·0.55 + damage·0.01)·1.0`，S = 1（S>1 时取 ln(S)+1）；高度 = TearHeight；
/// 质量倍率 1；击退倍率 1.7；速度倍率 1（TearParams 构造函数 FUN_005cd690）。
pub fn tear_hit_params(
    damage_stat: f32,
    tear_height: f32,
    flags: u128,
    damage_scale: f32,
) -> TearParams {
    let damage = damage_stat * damage_scale;
    let s = 1.0f32;
    let s_eff = if s > 1.0 { s.ln() + 1.0 } else { s };
    let scale = (damage.sqrt() * 0.23 + s_eff * 0.55 + damage * 0.01) * 1.0;
    TearParams {
        damage,
        scale,
        height: tear_height,
        mass_mult: 1.0,
        knockback_mult: 1.7,
        speed_mult: 1.0,
        flags,
    }
}

/// 眼泪碰撞半径：`Entity_Tear::ResetSpriteScale`（FUN_00677780）第 321 行
/// `SetSize(config.collisionRadius(7) × Scale, sizeMulti, 8 点)`。
pub fn apply_scale(e: &mut Entity, scale: f32) {
    crate::physics::set_size(e, 7.0 * scale, Vec2::new(1.0, 1.0), 8);
}

/// `Entity_Player::FireTear(pos, vel, ...)`（FUN_00790af0）：在 `pos + vel` 生成 2.0 眼泪，
/// 速度 × speedMult，高度 = TearHeight，初始下落速度 = rand·0.2 − TearFallingSpeed（非 POP），
/// 加速度 = TearFallingAcceleration，射程 = TearRange，然后立刻 `Update()` 一次并 `pos -= vel`。
pub fn fire_tear(player: &mut Entity, pos: Vec2, vel: Vec2, rng: &mut Rng) -> Entity {
    let stats = player.player().stats;
    let params = tear_hit_params(stats.damage, stats.tear_height, stats.tear_flags, 1.0);

    let spawn_pos = pos + vel;
    let mut t = Entity::base(0, EntityKind::Tear, TYPE_TEAR, 0, 0, spawn_pos);
    // Entity_Tear::Init：gridCollision BULLET(4)，collisionDamage 3.5，mass 8（entities2 collisionMass），
    // homingFriction 0.85，knockback 1.7，range 260
    t.grid_collision_class = grid_collision_class::BULLET;
    t.mass = 8.0;
    t.mass2 = 8.0 * params.mass_mult; // FireTear：+0x37c *= massMultiplier
    t.max_hp = 0.0;
    t.hp = 0.0;
    t.friction = 1.0;
    t.frame_friction = 1.0;
    t.vel = Vec2::new(vel.x * params.speed_mult, vel.y * params.speed_mult);
    t.spawner = Some(player.id);
    t.collision_damage = params.damage;

    let falling_speed = if params.flags & tear_flags::POP == 0 {
        rng.next_f32() * 0.2 - stats.tear_falling_speed
    } else {
        -stats.tear_falling_speed
    };
    let tear_idx = {
        let p = player.player_mut();
        let idx = p.tear_counter;
        p.tear_counter += 1;
        idx
    };
    t.tear = Some(TearState {
        height: params.height,
        falling_speed,
        falling_accel: stats.tear_falling_accel,
        scale: params.scale.max(0.01),
        base_damage: params.damage,
        range: stats.tear_range,
        knockback_mult: params.knockback_mult,
        flags: params.flags,
        wait_frames: 0,
        tear_idx,
        hit_list: Vec::new(),
        death_cause: 0,
    });
    apply_scale(&mut t, params.scale.max(0.01));

    // FireTear 第 214–217 行：立即 Update 一次，再退回一步
    update(&mut t);
    t.pos -= t.vel;
    t
}

/// `Entity_Tear::Update` 的普通眼泪路径，每逻辑帧一次：
/// 1. 第 2806 行：`height > −5` → 落地消失（HYDROBOUNCE 另有预测式判定）；
/// 2. 第 2891 行：上一碰撞阶段撞到网格且 `height > −60` → 撞墙消失（SPECTRAL 例外）；
/// 3. 第 3517 行：`Entity::Update`（位置积分、年龄 +1）；
/// 4. 第 3834–3875 行（汇编 0x00272E06–0x0027308C）：下落更新，见 `tear_flags::INTERNAL_RANGE_FALL`。
pub fn update(e: &mut Entity) {
    if !e.exists || e.kind != EntityKind::Tear {
        return;
    }
    // 速度倍率：Room::GetTearTimeScale（FUN_007ea3e0）无道具、单人时为 1.0
    e.speed_mult = 1.0;

    let (height, flags) = {
        let t = e.tear();
        (t.height, t.flags)
    };
    if e.dead {
        return;
    }
    // 1. 落地
    if height > -5.0 {
        e.dead = true;
        e.tear_mut().death_cause = 1;
        return;
    }
    // 2. 撞墙（BULLET 类只报告不推出；上一帧碰撞阶段置位）
    if e.collides_with_grid && height > -60.0 && flags & tear_flags::SPECTRAL == 0 {
        e.dead = true;
        e.tear_mut().death_cause = 2;
        return;
    }

    // 3. Entity::Update
    crate::physics::update(e);

    // 4. 下落
    let sm = e.speed_mult;
    let age = e.time_cur;
    let t = e.tear_mut();
    if t.wait_frames < 1 {
        let v = t.falling_speed;
        let range_based = t.flags & tear_flags::INTERNAL_RANGE_FALL != 0;
        let new_v = if range_based {
            if age > t.range * 0.25 {
                // 汇编 0x00272F0D–0x00272F8A：v' = 0.9·v + sm·max(0, accel + 0.2)
                0.9 * v + sm * (t.falling_accel + 0.2).max(0.0)
            } else {
                0.0
            }
        } else {
            // 汇编 0x00272FA6–0x00272FE5：v' = sm·0.1 + 0.9·v + sm·accel
            sm * 0.1 + 0.9 * v + sm * t.falling_accel
        };
        t.falling_speed = new_v;
        t.height += sm * new_v; // FUN_00675fc0(height + sm·v')
        let accel = t.falling_accel;
        let h = t.height;
        if !range_based {
            // 汇编 0x00273048–0x00273084：无内部位时的屏幕方向下坠
            if accel >= 0.001 {
                e.vel.y += accel;
            } else if h > -10.0 {
                e.vel.y += 0.5;
            }
        }
    }
}

/// 眼泪命中 NPC（`Entity_Tear::HandleCollision` 普通路径）：
/// - 第 832–834 行：`npc->TakeDamage(collisionDamage, 0, source=tear, 30)`（进入 NPC 的待结算伤害）；
/// - 第 996–1042 行：带 PERSISTENT 或内部位 114 时按质量份额推挤 NPC：
///   `npc.vel += d̂ · |npc.vel − tear.vel| · m_tear/(m_tear + m_npc)`（d 从眼泪指向 NPC，相向高速时偏折）；
/// - 第 1165–1200 行：非穿透眼泪消失；PERSISTENT 眼泪伤害超过 NPC 血量时扣除后继续飞行。
///
/// 返回眼泪是否消失。
pub fn on_hit_npc(tear: &mut Entity, npc: &mut Entity, events: &mut Vec<Event>) -> bool {
    if tear.dead || npc.dead || !npc.exists {
        return false;
    }
    if tear.tear().hit_list.contains(&npc.id) {
        return false;
    }
    let damage = tear.collision_damage;
    npc.pending_damage += damage;
    events.push(Event::TearHit {
        tear: tear.id,
        npc: npc.id,
        damage,
    });

    let flags = tear.tear().flags;
    if flags & (tear_flags::PERSISTENT | tear_flags::INTERNAL_RANGE_FALL) != 0 {
        let d = npc.pos - tear.pos;
        let dist = d.length();
        if dist >= 0.0001 {
            let mut dir = Vec2::new(d.x / dist, d.y / dist);
            let rel = tear.vel - npc.vel;
            if rel.length() > (npc.size + tear.size) * 0.5 && rel.dot(d) < 0.0 {
                let a = d.normalized();
                let r = rel.normalized();
                let k = a.dot(r);
                let adj = Vec2::new(d.x + 2.0 * k * d.x, d.y + 2.0 * k * d.y);
                dir = adj.normalized();
            }
            let share = tear.mass2 / (tear.mass2 + npc.mass2);
            let rel_speed = (npc.vel - tear.vel).length();
            npc.vel = Vec2::new(
                npc.vel.x + dir.x * rel_speed * share,
                npc.vel.y + dir.y * rel_speed * share,
            );
        }
    }

    if flags & tear_flags::PIERCING != 0 {
        tear.tear_mut().hit_list.push(npc.id);
        return false;
    }
    if flags & tear_flags::PERSISTENT != 0 && damage > npc.hp {
        // 第 1186–1190 行：溢出伤害继续飞行，按比例缩小
        let remaining = damage - npc.hp;
        tear.collision_damage = remaining;
        let scale = tear.tear().scale * remaining / damage;
        tear.tear_mut().scale = scale;
        apply_scale(tear, scale);
        tear.tear_mut().hit_list.push(npc.id);
        return false;
    }
    tear.dead = true;
    tear.tear_mut().death_cause = 3;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics;

    fn fire_from(player_pos: Vec2, dir: Vec2, flags: u128) -> Entity {
        let mut p = Entity::new_player(1, player_pos);
        p.player_mut().stats.tear_flags = flags;
        let mut rng = Rng::new(5);
        fire_tear(
            &mut p,
            player_pos,
            Vec2::new(dir.x * 10.0, dir.y * 10.0),
            &mut rng,
        )
    }

    #[test]
    fn base_tear_params() {
        let params = tear_hit_params(3.5, -23.75, 0, 1.0);
        assert!((params.damage - 3.5).abs() < 1e-6);
        assert!(
            (params.scale - 1.0153).abs() < 0.001,
            "scale {}",
            params.scale
        );
        assert_eq!(params.knockback_mult, 1.7);
    }

    #[test]
    fn range_fall_model_lands_after_range_quarter_plus_drop() {
        let mut t = fire_from(
            Vec2::new(100.0, 280.0),
            Vec2::new(1.0, 0.0),
            tear_flags::INTERNAL_RANGE_FALL,
        );
        assert!(
            (t.pos.x - 110.0).abs() < 1e-3,
            "spawn one step ahead: {}",
            t.pos.x
        );
        assert_eq!(t.time_cur, 1.0);
        let mut frames = 1;
        while !t.dead && frames < 400 {
            physics::pre_update(&mut t);
            update(&mut t);
            frames += 1;
        }
        assert!(
            t.dead && t.tear().death_cause == 1,
            "cause {}",
            t.tear().death_cause
        );
        // 平飞 65 帧（260×0.25）后以 v' = 0.9v + 0.2 下落，约 17 帧落地
        assert!((81..=84).contains(&frames), "landed at frame {}", frames);
        assert!((t.pos.x - (100.0 + 10.0 * (frames as f32 - 1.0))).abs() < 1.0);
    }

    #[test]
    fn continuous_fall_model_lands_around_28_frames() {
        let mut t = fire_from(Vec2::new(100.0, 280.0), Vec2::new(1.0, 0.0), 0);
        let mut frames = 1;
        while !t.dead && frames < 400 {
            physics::pre_update(&mut t);
            update(&mut t);
            frames += 1;
        }
        assert!(t.dead && t.tear().death_cause == 1);
        assert!((25..=31).contains(&frames), "landed at frame {}", frames);
        // 无内部位时接近地面会向屏幕下方漂移
        assert!(t.pos.y > 280.0);
    }

    #[test]
    fn hit_kills_tear_and_queues_damage_with_push() {
        let mut t = fire_from(
            Vec2::new(100.0, 280.0),
            Vec2::new(1.0, 0.0),
            tear_flags::INTERNAL_RANGE_FALL,
        );
        let mut g = Entity::new_gaper(2, 1, Vec2::new(120.0, 280.0));
        let mut events = Vec::new();
        assert!(on_hit_npc(&mut t, &mut g, &mut events));
        assert!(t.dead);
        assert!((g.pending_damage - 3.5).abs() < 1e-6);
        // 推挤：份额 8/(8+5)，相对速度 10 → 约 6.15
        assert!(
            (g.vel.x - 10.0 * 8.0 / 13.0).abs() < 1e-3,
            "push {}",
            g.vel.x
        );
        physics::update(&mut g);
        assert!((g.hp - 6.5).abs() < 1e-6);
    }
}
