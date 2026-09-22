//! 玩家：移动（`Entity_Player` 的移动函数 FUN_00779b40，RVA 0x00379B40）、射击输入
//! （`GetShootingInput` FUN_00779620，RVA 0x00379620）、开火节奏（`Weapon_Tears::Update` 0x0060D870 与
//! `Weapon_Tears::Fire` 0x0060A720）、眼泪参数（`GetTearHitParams` 0x003BF6C0）与发射
//! （`Entity_Player::FireTear` 0x00390AF0），以及受伤无敌（`Entity_Player::TakeDamage` 0x003729D0）。
//!
//! 规则出处见 `analysis/docs/J460_PLAYER_TEAR_MODEL.md`；移动/眼泪已通过原版留出轨迹验证。

use crate::entity::{flags, Entity, TYPE_PLAYER};
use crate::math::Vec2;
use crate::rng::Rng;
use crate::tear;
use crate::world::Event;

/// 一帧的玩家输入。`move_dir`/`shoot_dir` 的分量取 [-1, 1]：键盘为 0/±1（左右相减、上下相减），
/// 手柄可给模拟量。对应 `Input::GetActionValue` 的动作 0..3（移动）与 4..7（射击）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerInput {
    pub move_dir: Vec2,
    pub shoot_dir: Vec2,
    pub bomb: bool,
}

impl PlayerInput {
    pub fn new(move_x: f32, move_y: f32, shoot_x: f32, shoot_y: f32) -> PlayerInput {
        PlayerInput {
            move_dir: Vec2::new(move_x, move_y),
            shoot_dir: Vec2::new(shoot_x, shoot_y),
            bomb: false,
        }
    }
}

/// 玩家属性（`Entity_Player::EvaluateItems` FUN_00763570 无道具时的结果）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerStats {
    /// +0x1568 MoveSpeed。
    pub move_speed: f32,
    /// +0x1460 MaxFireDelay（Isaac 10 → 每 11 帧一发）。
    pub max_fire_delay: f32,
    /// +0x1464 ShotSpeed。
    pub shot_speed: f32,
    /// +0x1470 Damage。
    pub damage: f32,
    /// +0x1474 TearHeight。
    pub tear_height: f32,
    /// +0x1478 TearFallingSpeed (range conversion at EvaluateItems tail).
    pub tear_falling_speed: f32,
    /// +0x147c TearFallingAcceleration。
    pub tear_falling_accel: f32,
    /// +0x1480 TearRange（单位；显示值 = /40）。
    pub tear_range: f32,
    /// +0x156c Luck。
    pub luck: f32,
    /// +0x1570 CanFly。
    pub can_fly: bool,
    /// +0x1488 128 位 TearFlags。默认 `TEAR_NORMAL`（全局 DAT_00c79100，位于 .bss、未见写入 → 0），
    /// Native unmodified Isaac reports zero, not internal bit 114.
    pub tear_flags: u128,
}

impl PlayerStats {
    /// Isaac 无道具：range 260、height −23.75、accel 0；fall 在 EvaluateItems 末尾由射程重算。
    /// 伤害 3.5、射速 1.0、移速 1.0、开火延迟 10（Init 0x1460 = 10.0）。
    pub fn isaac() -> PlayerStats {
        PlayerStats {
            move_speed: 1.0,
            max_fire_delay: 10.0,
            shot_speed: 1.0,
            damage: 3.5,
            tear_height: -23.75,
            tear_falling_speed: falling_speed_from_range(-23.75, 260.0, 1.0, 0.0),
            tear_falling_accel: 0.0,
            tear_range: 260.0,
            luck: 0.0,
            can_fly: false,
            tear_flags: 0,
        }
    }
}

/// EvaluateItems tail (RVA 0x00370564) calls 0x0027DA40 with XMM0..3;
/// Ghidra's no-argument pseudocode drops this calculation completely.
pub fn falling_speed_from_range(height: f32, range: f32, shot_speed: f32, accel: f32) -> f32 {
    let frames = range / (shot_speed * 10.0);
    let k = ((0.9_f32.powf(frames) - 1.0) * -9.491221_f32).max(0.01);
    -((-5.0 - height - (accel * 10.0 + 1.0) * (frames - k)) / k)
}

/// 玩家移动方向枚举（+0x1624）：0 左、1 上、2 右、3 下、-1 无。
pub const DIR_NONE: i32 = -1;
pub const DIR_LEFT: i32 = 0;
pub const DIR_UP: i32 = 1;
pub const DIR_RIGHT: i32 = 2;
pub const DIR_DOWN: i32 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct PlayerState {
    pub stats: PlayerStats,
    pub input: PlayerInput,
    /// +0x410 控制是否启用，+0x414 控制冷却。
    pub controls_enabled: bool,
    pub controls_cooldown: i32,
    /// +0x1660：移动更新开始时的速度快照（眼泪继承用）。
    pub recent_movement: Vec2,
    /// +0x1644：本帧移动输入；+0x1654：最近一次非零移动输入（瞄准回退方向）。
    pub movement_input: Vec2,
    pub last_direction: Vec2,
    /// +0x1624。
    pub move_direction: i32,
    /// +0x165c：滑行计数（房间/道具效果置 60 后每帧递减；普通房间为 0）。
    pub slip_counter: i32,
    /// 武器（`Weapon_Tears`）：+0xc FireDelay、+0x20 Direction、+0x38 NumFired。
    pub fire_delay: f32,
    pub weapon_direction: Vec2,
    pub num_fired: i32,
    /// +0x13bc 受伤后的无敌倒计时（帧）。
    pub damage_cooldown: i32,
    pub bombs: i32,
    pub bomb_held: bool,
    pub bomb_cooldown: i32,
    /// +0x1578：眼泪左右眼位移符号（每发交替 ±1；基础眼泪不用它改位置）。
    pub tear_displacement: i8,
    /// +0x146c：已发射眼泪计数（写入眼泪的 TearIndex）。
    pub tear_counter: i32,
    /// 本帧发射的眼泪数（观测/调试用）。
    pub fired_this_frame: i32,
}

impl Default for PlayerState {
    fn default() -> Self {
        PlayerState {
            stats: PlayerStats::isaac(),
            input: PlayerInput::default(),
            controls_enabled: true,
            controls_cooldown: 0,
            recent_movement: Vec2::ZERO,
            movement_input: Vec2::ZERO,
            last_direction: Vec2::new(0.0, 1.0),
            move_direction: DIR_NONE,
            slip_counter: 0,
            fire_delay: 0.0,
            weapon_direction: Vec2::ZERO,
            num_fired: 0,
            damage_cooldown: 0,
            bombs: 1,
            bomb_held: false,
            bomb_cooldown: 0,
            tear_displacement: 1,
            tear_counter: 0,
            fired_this_frame: 0,
        }
    }
}

/// 玩家一帧（`Entity_Player::Update` 0x00382AF0 中与运动/射击有关的顺序）：
/// 1. 第 726 行 `Entity::Update`（用上一帧算出的速度积分位置）；
/// 2. 无敌倒计时递减；
/// 3. 第 1556 行移动函数（读输入、改速度）；
/// 4. 第 1556 行之后的 `FUN_0077b020`：先对每把武器调 `Weapon::Update`（开火延迟递减，第 928–948 行），
///    再按射击输入调 `Weapon::Fire`（第 1584 行经 `FUN_007826d0`）。
///
/// 网格碰撞不在这里：它在 `EntityList::Update` 的碰撞阶段调用玩家版本 `FUN_006ab440`。
pub fn update_player(
    e: &mut Entity,
    rng: &mut Rng,
    events: &mut Vec<Event>,
    spawned: &mut Vec<Entity>,
) {
    if !e.exists || e.dead {
        return;
    }
    e.speed_mult = 1.0;

    // ACTION_BOMB is triggered on a rising edge, not repeatedly while held.
    let pressed = e.player().input.bomb;
    e.player_mut().bomb_cooldown = (e.player().bomb_cooldown - 1).max(0);
    if pressed
        && !e.player().bomb_held
        && e.player().bombs > 0
        && e.player().bomb_cooldown == 0
        && e.player().controls_enabled
    {
        e.player_mut().bombs -= 1;
        e.player_mut().bomb_cooldown = 30;
        let mut b = crate::bomb::new(e.pos, e.vel * 0.1, e.id);
        // Native player-created bombs receive their first update immediately.
        crate::bomb::update(&mut b);
        b.time_cur = 0.;
        crate::bomb::update(&mut b);
        spawned.push(b);
    }
    e.player_mut().bomb_held = pressed;

    // 1. Entity::Update：积分、状态倒计时、待结算伤害
    crate::physics::update(e);

    // 2. 无敌帧（+0x13bc）递减（原版递减点未在伪 C 中定位，这里每帧减一）
    {
        let p = e.player_mut();
        p.fired_this_frame = 0;
        if p.damage_cooldown > 0 {
            p.damage_cooldown -= 1;
        }
        if p.controls_cooldown > 0 {
            p.controls_cooldown -= 1;
        }
    }

    // 3. 移动
    update_movement(e);

    // 4. 武器：Weapon_Tears::Update → Weapon_Tears::Fire
    weapon_update(e);
    weapon_fire(e, rng, events, spawned);
}

/// Manager's odd 60 Hz tick calls Player::Update with INTERPOLATION_UPDATE.
/// Movement is real, not visual interpolation; timers and tears remain 30 Hz.
/// J460 Game::UpdateInterpolation RVA 0x002FD3F0, ordinary gameplay branch.
pub fn interpolate_player(e: &mut Entity) {
    if !e.exists || e.dead {
        return;
    }
    e.flags |= crate::entity::flags::INTERPOLATION_UPDATE;
    crate::physics::update(e);
    update_movement(e);
    e.flags &= !crate::entity::flags::INTERPOLATION_UPDATE;
}

/// `GetMovementInput`（FUN_00779360）：x = 右 − 左，y = 下 − 上（动作 0..3 的模拟量）。
fn movement_input(e: &Entity) -> Vec2 {
    let p = e.player();
    if !p.controls_enabled || p.controls_cooldown > 0 {
        return Vec2::ZERO;
    }
    p.input.move_dir
}

/// 移动函数 FUN_00779b40（RVA 0x00379B40）的普通角色路径（无道具、无饰品、非特殊形态）。
///
/// 常量（`.rdata`）：0.25（DAT_00baa1d4）、0.75（DAT_00baa380）、60（DAT_00baa950）、0.2/0.8（DAT_00baa194/3a4）、
/// 0.7/0.2（DAT_00baa354/198）、3.0（DAT_00baa6fc）、0.4（DAT_00baa280）、0.185/0.775（DAT_00baa180/388）、
/// 4.41（DAT_00baa76c）、0.5（DAT_00baa2d0）、0.225（DAT_00baa1c4 重力）。
pub fn update_movement(e: &mut Entity) {
    // 第 53–65 行：被目标实体牵引时直接指向目标（未翻译），高度为负（跳跃中）时返回
    let input_raw = movement_input(e);
    let can_fly = e.player().stats.can_fly;
    let move_speed = e.player().stats.move_speed;
    let apply_gravity = e.has_flag(flags::APPLY_GRAVITY);

    // 第 67–68 行：速度快照 → +0x1660（眼泪继承用）
    let vel_snapshot = e.vel;
    {
        let p = e.player_mut();
        p.recent_movement = vel_snapshot;
    }

    // 第 74–91 行：输入长度超过 1 时归一化
    let mut input = input_raw;
    if input.length() > 1.0 {
        input = input.normalized();
    }

    // 第 145–150 行：记录输入与最近非零输入
    {
        let p = e.player_mut();
        p.movement_input = input;
        if !input.is_zero() {
            p.last_direction = input;
        }
    }

    // 第 151–176 行：速度系数与移动方向枚举
    let speed_factor = move_speed * 0.25 + 0.75;
    let scaled = input * speed_factor;
    let dir = if scaled.x.abs() <= scaled.y.abs() {
        if scaled.y > 0.0 {
            DIR_DOWN
        } else if scaled.y < 0.0 {
            DIR_UP
        } else {
            DIR_NONE
        }
    } else if scaled.x >= 0.0 {
        DIR_RIGHT
    } else {
        DIR_LEFT
    };
    e.player_mut().move_direction = dir;

    // 第 190–192 行：滑行比例 = 计数 / 60；减速状态下为 0
    let mut ratio = e.player().slip_counter as f32 / 60.0;
    if e.has_flag(flags::SLOW) {
        ratio = 0.0;
    }
    let old_vel = e.vel;

    // 第 216–230 行：有输入时把与输入垂直的速度分量乘 sqrt(0.2·ratio + 0.8)
    if !scaled.is_zero() && (!apply_gravity || can_fly) {
        let dot = e.vel.dot(scaled);
        let along = scaled * dot;
        let k = (ratio * 0.2 + 0.8).sqrt();
        e.vel = Vec2::new(
            (e.vel.x - along.x) * k + along.x,
            (e.vel.y - along.y) * k + along.y,
        );
    }

    // 第 231–258 行：加速度
    let dot2 = e.vel.dot(scaled);
    let mut t = (ratio * 0.7 - 0.2) * dot2;
    if t <= 0.0 {
        t = 0.0;
    }
    let accel_mag = (t / move_speed + 3.0) * 0.4;
    let post = (ratio * 0.185 + 0.775).sqrt();
    let max_speed = move_speed * 4.41;
    let a = (accel_mag * 0.25 - accel_mag) * ratio + accel_mag;
    e.vel = Vec2::new(scaled.x * 0.5 * a + e.vel.x, scaled.y * 0.5 * a + e.vel.y);

    // 第 259–274 行：滑行中超过最大速度时按旧速度大小缩回
    if ratio > 0.0 && e.vel.length_sq() > max_speed * max_speed {
        let old_len = old_vel.length();
        let new_len = e.vel.length();
        if new_len > 0.0 {
            e.vel = e.vel * (old_len / new_len);
        }
    }

    // 第 276–283 行：整体乘 sqrt(0.775)（≈0.88）；受重力且不会飞时 y 方向改为加 0.225
    e.vel.x *= post;
    if !apply_gravity || can_fly {
        e.vel.y *= post;
    } else {
        e.vel.y += 0.225;
    }
}

/// `Weapon_Tears::Update`（FUN_00a0d870，RVA 0x0060D870）：`fireDelay > -1` 时每帧减去拥有者的速度倍率。
pub fn weapon_update(e: &mut Entity) {
    let sm = e.speed_mult;
    let p = e.player_mut();
    if p.fire_delay > -1.0 {
        p.fire_delay -= sm;
    }
}

/// `GetShootingInput`（FUN_00779620）：x = 右(5) − 左(4)，y = 下(7) − 上(6)。
fn shooting_input(e: &Entity) -> Vec2 {
    let p = e.player();
    if !p.controls_enabled || p.controls_cooldown > 0 {
        return Vec2::ZERO;
    }
    p.input.shoot_dir
}

/// `Weapon_Tears::Fire(dir, isShooting, isInterpolated=false)`（FUN_00a0a720）的普通眼泪路径：
/// 方向轴对齐（`Weapon::IsAxisAligned` 无 Marked/模拟摇杆时为真），`fireDelay < 0` 时开火，
/// 每发 `fireDelay += MaxFireDelay + 1`（最多补 8 发）。
pub fn weapon_fire(
    e: &mut Entity,
    rng: &mut Rng,
    events: &mut Vec<Event>,
    spawned: &mut Vec<Entity>,
) {
    let shoot = shooting_input(e);
    let is_shooting = !shoot.is_zero();

    // 第 613–650 行：方向来源与轴对齐
    let raw_dir = if is_shooting {
        shoot
    } else {
        e.player().weapon_direction
    };
    let dir = if raw_dir.x.abs() <= raw_dir.y.abs() {
        Vec2::new(0.0, if raw_dir.y < 0.0 { -1.0 } else { 1.0 })
    } else {
        Vec2::new(if raw_dir.x < 0.0 { -1.0 } else { 1.0 }, 0.0)
    };
    if is_shooting {
        e.player_mut().weapon_direction = shoot;
    }

    if !is_shooting {
        return;
    }

    // 第 434–447 行：开火次数
    let max_fire_delay = e.player().stats.max_fire_delay;
    let mut count = 0;
    {
        let p = e.player_mut();
        if p.fire_delay < 0.0 {
            while count < 8 {
                count += 1;
                p.fire_delay += max_fire_delay + 1.0;
                if p.fire_delay >= 0.0 {
                    break;
                }
            }
        }
    }

    for _ in 0..count {
        fire_one(e, dir, rng, events, spawned);
    }
}

/// 一发普通眼泪：`Weapon_Tears::Fire` 第 654–700 行 + 多发参数（单发：无偏移、无旋转）+
/// `FUN_009f5ad0` → `Entity_Player::FireTear`。
fn fire_one(
    e: &mut Entity,
    dir: Vec2,
    rng: &mut Rng,
    events: &mut Vec<Event>,
    spawned: &mut Vec<Entity>,
) {
    let stats = e.player().stats;
    // 第 654–655 行：方向 × 10（DAT_00baa81c）
    let mut base = Vec2::new(dir.x * 10.0, dir.y * 10.0);
    // 第 658–662 行：加上移动继承（GetTearMovementInheritance）
    let inherit = tear_movement_inheritance(e, dir);
    base += inherit;
    // GetMultiShotPositionVelocity：单发时 vel = 方向向量 × ShotSpeed，位置偏移 0
    let vel = base * stats.shot_speed;
    // Alternating eyes, Weapon_Tears::Fire RVA 0x0060B9xx: a random
    // perpendicular offset of 0.3..0.5 times the inherited shot vector.
    // It is not a change to velocity, and the first shot uses the negative eye.
    let eye = -(e.player().tear_displacement as f32) * (rng.next_f32() * 0.2 + 0.3);
    let pos = Vec2::new(e.pos.x - base.y * eye, e.pos.y + base.x * eye);

    // Integer-delay Isaac: intra-frame phase -1-fireDelay is zero at emission.
    let tear_entity = tear::fire_tear(e, pos, vel, rng);
    {
        let p = e.player_mut();
        p.num_fired += 1;
        p.fired_this_frame += 1;
        // Weapon_Tears::Fire 第 700 行前后：左右眼交替
        p.tear_displacement = -p.tear_displacement;
    }
    events.push(Event::TearFired {
        player: e.id,
        tear: tear_entity.id,
        pos: tear_entity.pos,
        vel: tear_entity.vel,
    });
    spawned.push(tear_entity);
}

/// `Entity_Player::GetTearMovementInheritance(out, direction, ignore)`（FUN_007aaea0，RVA 0x003AAEA0）：
/// 继承 = 移动速度快照 × 0.6 × 2；若继承在射击方向上的分量为负（向后射），去掉该分量。
pub fn tear_movement_inheritance(e: &Entity, direction: Vec2) -> Vec2 {
    let recent = e.player().recent_movement;
    let mut inherit = Vec2::new(recent.x * 0.6 * 2.0, recent.y * 0.6 * 2.0);
    if !direction.is_zero() {
        let d = direction.normalized();
        let proj = inherit.dot(d);
        let comp = Vec2::new(d.x * proj, d.y * proj);
        if comp.dot(direction) < 0.0 {
            inherit -= comp;
        }
    }
    inherit
}

/// `Entity_Player::TakeDamage` 的最小子集：无敌中（+0x13bc > 0）忽略；否则按半心扣血，
/// 无敌帧 = 60 × min(2, round(damage))（第 596–606 行与 2745–2748 行；多人时经 FUN_007db330 缩短）。
/// 返回是否真的受伤。
pub fn take_damage(
    e: &mut Entity,
    damage: f32,
    source: Option<u32>,
    events: &mut Vec<Event>,
) -> bool {
    if e.kind != crate::entity::EntityKind::Player || e.dead || damage <= 0.0 {
        return false;
    }
    if e.player().damage_cooldown > 0 {
        return false;
    }
    let rounded = (damage + 0.5) as i32;
    let units = rounded.max(1);
    e.hp -= units as f32;
    let cooldown = 60 * units.min(2);
    e.player_mut().damage_cooldown = cooldown;
    if e.hp <= 0.0 {
        e.hp = 0.0;
        e.dead = true;
    }
    events.push(Event::PlayerDamaged {
        player: e.id,
        damage: units as f32,
        hp_after: e.hp,
        source,
    });
    true
}

/// 玩家类型判定辅助。
pub fn is_player(e: &Entity) -> bool {
    e.etype == TYPE_PLAYER
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics;

    fn run_frames(e: &mut Entity, n: usize) -> Vec<f32> {
        let mut speeds = Vec::new();
        for _ in 0..n {
            physics::pre_update(e);
            physics::update(e);
            update_movement(e);
            speeds.push(e.vel.length());
        }
        speeds
    }

    #[test]
    fn steady_state_speed_is_4_41_at_speed_stat_1() {
        let mut e = Entity::new_player(1, Vec2::new(300.0, 280.0));
        e.player_mut().input = PlayerInput::new(1.0, 0.0, 0.0, 0.0);
        let speeds = run_frames(&mut e, 60);
        let last = *speeds.last().unwrap();
        assert!((last - 4.41).abs() < 0.01, "steady speed {}", last);
        // 加速：约 8 帧后到达 60%
        assert!(
            speeds[7] > 2.6 && speeds[7] < 4.41,
            "speed after 8 frames {}",
            speeds[7]
        );
    }

    #[test]
    fn diagonal_input_is_normalized() {
        let mut e = Entity::new_player(1, Vec2::new(300.0, 280.0));
        e.player_mut().input = PlayerInput::new(1.0, 1.0, 0.0, 0.0);
        let speeds = run_frames(&mut e, 60);
        assert!((speeds.last().unwrap() - 4.41).abs() < 0.01);
        assert!((e.vel.x - e.vel.y).abs() < 1e-4);
    }

    #[test]
    fn release_stops_within_a_few_frames() {
        let mut e = Entity::new_player(1, Vec2::new(300.0, 280.0));
        e.player_mut().input = PlayerInput::new(0.0, -1.0, 0.0, 0.0);
        run_frames(&mut e, 60);
        e.player_mut().input = PlayerInput::default();
        let speeds = run_frames(&mut e, 20);
        // 每帧乘 sqrt(0.775) ≈ 0.88：20 帧后剩约 7.8%
        assert!(speeds[19] < 0.4, "speed after release {}", speeds[19]);
        assert!(speeds[19] > 0.2);
    }

    #[test]
    fn fire_delay_gives_one_tear_every_eleven_frames() {
        let mut e = Entity::new_player(1, Vec2::new(300.0, 280.0));
        e.player_mut().input = PlayerInput::new(0.0, 0.0, 1.0, 0.0);
        let mut rng = Rng::new(3);
        let mut events = Vec::new();
        let mut spawned = Vec::new();
        let mut fire_frames = Vec::new();
        for frame in 0..110 {
            physics::pre_update(&mut e);
            update_player(&mut e, &mut rng, &mut events, &mut spawned);
            if e.player().fired_this_frame > 0 {
                fire_frames.push(frame);
            }
        }
        assert_eq!(spawned.len(), fire_frames.len());
        assert!(
            spawned.len() == 10,
            "fired {} tears in 110 frames: {:?}",
            spawned.len(),
            fire_frames
        );
        for w in fire_frames.windows(2) {
            assert_eq!(w[1] - w[0], 11, "intervals {:?}", fire_frames);
        }
        let t = &spawned[0];
        assert!(
            (t.vel.length() - 10.0).abs() < 1e-4,
            "tear speed {}",
            t.vel.length()
        );
        assert!(t.vel.x > 0.0 && t.vel.y == 0.0);
    }

    #[test]
    fn moving_forward_adds_inheritance_but_backward_does_not() {
        let mut e = Entity::new_player(1, Vec2::new(300.0, 280.0));
        e.player_mut().input = PlayerInput::new(1.0, 0.0, 0.0, 0.0);
        run_frames(&mut e, 60);
        let forward = tear_movement_inheritance(&e, Vec2::new(1.0, 0.0));
        assert!(
            (forward.x - 1.2 * 4.41).abs() < 0.02,
            "forward inheritance {:?}",
            forward
        );
        let backward = tear_movement_inheritance(&e, Vec2::new(-1.0, 0.0));
        assert!(
            backward.x.abs() < 1e-4,
            "backward inheritance {:?}",
            backward
        );
        let side = tear_movement_inheritance(&e, Vec2::new(0.0, 1.0));
        assert!((side.x - forward.x).abs() < 1e-4 && side.y.abs() < 1e-4);
    }

    #[test]
    fn contact_damage_and_invulnerability() {
        let mut e = Entity::new_player(1, Vec2::ZERO);
        let mut events = Vec::new();
        assert!(take_damage(&mut e, 1.0, None, &mut events));
        assert_eq!(e.hp, 5.0);
        assert_eq!(e.player().damage_cooldown, 60);
        assert!(!take_damage(&mut e, 1.0, None, &mut events));
        e.player_mut().damage_cooldown = 0;
        assert!(take_damage(&mut e, 2.0, None, &mut events));
        assert_eq!(e.hp, 3.0);
        assert_eq!(e.player().damage_cooldown, 120);
    }
}
