//! 实体结构。字段名后的偏移是 J460 `Entity`/`Entity_NPC`/`Entity_Player`/`Entity_Tear` 对象里的字节偏移
//! （NetFix `profiles/j460/game_api.json` 与 REPENTOGON `Entity.zhl`/`EntityNPC.zhl`/`EntityTear.zhl` 交叉核对；
//! 玩家字段的 J460 偏移见 `analysis/docs/J460_PLAYER_TEAR_MODEL.md` §2，与 J273 相比整体后移）。

use crate::math::Vec2;
use crate::pathfinder::Pathfinder;
use crate::player::PlayerState;
use crate::tear::TearState;

/// `EntityFlag`（enums.lua）子集，位号与原版一致，存放在 u64 里（0x168 低 32 位、0x16c 高 32 位）。
pub mod flags {
    pub const NO_STATUS_EFFECTS: u64 = 1 << 0;
    pub const NO_INTERPOLATE: u64 = 1 << 1;
    pub const APPEAR: u64 = 1 << 2;
    pub const NO_TARGET: u64 = 1 << 4;
    pub const FREEZE: u64 = 1 << 5;
    pub const POISON: u64 = 1 << 6;
    pub const SLOW: u64 = 1 << 7;
    pub const CHARM: u64 = 1 << 8;
    pub const CONFUSION: u64 = 1 << 9;
    pub const MIDAS_FREEZE: u64 = 1 << 10;
    pub const FEAR: u64 = 1 << 11;
    pub const BURN: u64 = 1 << 12;
    pub const INTERPOLATION_UPDATE: u64 = 1 << 14;
    pub const APPLY_GRAVITY: u64 = 1 << 15;
    pub const FRIENDLY: u64 = 1 << 29;
    pub const NO_PHYSICS_KNOCKBACK: u64 = 1 << 30;
    pub const KNOCKED_BACK: u64 = 1 << 47;
    pub const APPLY_IMPACT_DAMAGE: u64 = 1 << 48;
    pub const ICE_FROZEN: u64 = 1 << 49;
    pub const HELD: u64 = 1 << 57;
}

/// `EntityCollisionClass`（enums.lua）。
pub mod entity_collision {
    pub const NONE: i32 = 0;
    pub const PLAYERONLY: i32 = 1;
    pub const PLAYEROBJECTS: i32 = 2;
    pub const ENEMIES: i32 = 3;
    pub const ALL: i32 = 4;
}

/// `EntityGridCollisionClass`（enums.lua）。
pub mod grid_collision_class {
    pub const NONE: i32 = 0;
    pub const WALLS_X: i32 = 1;
    pub const WALLS_Y: i32 = 2;
    pub const WALLS: i32 = 3;
    pub const BULLET: i32 = 4;
    pub const GROUND: i32 = 5;
    pub const NOPITS: i32 = 6;
    pub const PITSONLY: i32 = 7;
}

pub const TYPE_PLAYER: i32 = 1;
pub const TYPE_TEAR: i32 = 2;
pub const TYPE_GAPER: i32 = 10;
pub const TYPE_GUSHER: i32 = 11;
pub const TYPE_MONSTRO: i32 = 20;
pub const TYPE_PROJECTILE: i32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    Player,
    Npc,
    Tear,
    Projectile,
}

/// `Entity_NPC` 专有字段。
#[derive(Clone, Debug)]
pub struct NpcState {
    /// AI-only locked destination. Never included in actor observations.
    pub target_pos: Vec2,
    pub anim_finished: bool,
    /// +0xb64 `State`（1 = 出现中，4 = 常规行动，0x11/0x12 = 死亡动画）。
    pub state: i32,
    /// +0x410 `StateFrame`。
    pub state_frame: i32,
    /// +0xb68：出现动画期间保存的状态。
    pub saved_state: i32,
    /// +0xb70：出现流程帧计数。
    pub appear_counter: i32,
    /// +0xba0 / +0xba4。
    pub projectile_cooldown: i32,
    pub projectile_delay: i32,
    /// +0xbc0 / +0xbc4 / +0xbb0 / +0xbb8。
    pub i1: i32,
    pub i2: i32,
    pub v1: Vec2,
    pub v2: Vec2,
    /// +0xc34：AI 每帧写入的追踪速度参数（Entity_NPC::Init 置 1.0）。
    pub move_speed: f32,
    /// +0xc74 `championRegenTimer`（u16）。
    pub champion_regen_timer: u16,
    /// +0xf1c：玩家控制的 NPC 用；-1 表示普通 NPC。
    pub controller_id: i32,
    /// +0xb74：AI 被调用时的模式（0 常规，1 前置，2 插值，4 后置）。
    pub ai_mode: i32,
    /// +0xb40：Gaper AI 读它决定音效与皱眉 Gaper 的头部覆盖动画；Gaper 自身从不写它，按 0 处理。
    pub aggro: bool,
    pub pathfinder: Pathfinder,
    /// 可见动画名与帧（观测用，近似 AnimWalkFrame 的结果）。
    pub anim: &'static str,
    pub anim_frame: i32,
    pub flip_x: bool,
    /// 覆盖层动画（Gaper 的 "Head"）当前帧；-1 表示没有覆盖层。
    pub overlay_frame: i32,
    pub overlay_len: i32,
    pub overlay_loop: bool,
}

impl Default for NpcState {
    fn default() -> Self {
        NpcState {
            target_pos: Vec2::ZERO,
            anim_finished: false,
            state: 1, // Entity_NPC::Init（RVA 0x002B8B20）第 77 行
            state_frame: 0,
            saved_state: 0,
            appear_counter: 0,
            projectile_cooldown: 0,
            projectile_delay: 0,
            i1: 0,
            i2: 0,
            v1: Vec2::ZERO,
            v2: Vec2::ZERO,
            move_speed: 1.0, // Init 第 89 行
            champion_regen_timer: 0,
            controller_id: -1, // Init 第 35 行
            ai_mode: 0,
            aggro: false,
            pathfinder: Pathfinder::default(),
            anim: "WalkHori",
            anim_frame: 0,
            flip_x: false,
            overlay_frame: -1,
            overlay_len: 0,
            overlay_loop: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    /// +0x28 / +0x2c / +0x30。
    pub etype: i32,
    pub variant: i32,
    pub subtype: i32,
    /// +0x172 / +0x173 / +0x171。
    pub exists: bool,
    pub dead: bool,
    pub visible: bool,
    /// +0x168/+0x16c。
    pub flags: u64,
    /// +0x33c 位置，+0x360 速度，+0x38c/+0x394 上一帧位置/速度（PreUpdate 写）。
    pub pos: Vec2,
    pub vel: Vec2,
    pub last_pos: Vec2,
    pub last_vel: Vec2,
    /// +0x368 `Friction`（配置值，Gaper=1），+0x36c 本帧摩擦（AI 每帧乘系数，Update 末尾重置为 Friction）。
    pub friction: f32,
    pub frame_friction: f32,
    /// +0x39c 速度倍率（减速/冰冻效果；无效果时 1.0）。
    pub speed_mult: f32,
    /// +0x3a0 / +0x3a4：更新时间累加器，`(int)cur - (int)prev` 即本帧逻辑帧数；`cur` 也是实体年龄（帧）。
    pub time_prev: f32,
    pub time_cur: f32,
    /// +0x370 `Size`（碰撞半径），+0x374/+0x378 `SizeMulti`。
    pub size: f32,
    pub size_mult: Vec2,
    /// +0x178：`numGridCollisionPoints` 个圆周采样点（Entity::SetSize 生成）。
    pub collision_points: Vec<Vec2>,
    /// +0x184 / +0x188 / +0x190 / +0x194 / +0x19c。
    pub grid_collision_class: i32,
    pub entity_collision_class: i32,
    pub collides_with_grid: bool,
    pub grid_hit_velocity: Vec2,
    pub grid_normal: Vec2,
    /// +0x2d4 `Mass`（圆形推挤用），+0x37c 击退质量（`Entity::AddKnockback` 与眼泪命中推挤用，
    /// `Entity_Player::FireTear` 里乘 TearParams.massMultiplier），+0x388 `CollisionDamage`。
    pub mass: f32,
    pub mass2: f32,
    pub collision_damage: f32,
    /// +0x380 / +0x384。
    pub hp: f32,
    pub max_hp: f32,
    /// +0x14：本帧累计的待结算伤害（`Entity::TakeDamage` 累加，`Entity::Update` 第 660–727 行扣血并清零）。
    pub pending_damage: f32,
    /// +0x308 击退方向（击退期间直接成为速度），+0x2f8 击退倒计时。
    pub knockback_dir: Vec2,
    pub knockback_countdown: i32,
    /// 状态倒计时子集：+0x218 冰冻、+0x21c 中毒、+0x220 减速、+0x224 魅惑、+0x228 混乱、+0x230 恐惧、+0x234 燃烧、+0x22c 点金冻结。
    pub freeze_countdown: i32,
    pub poison_countdown: i32,
    pub slowing_countdown: i32,
    pub charmed_countdown: i32,
    pub confusion_countdown: i32,
    pub fear_countdown: i32,
    pub burn_countdown: i32,
    pub midas_countdown: i32,
    /// +0x320 / +0x324：毒/燃烧每 20 帧一次的伤害值（Init 置 2.0），+0x2e0/+0x2e4 计数器。
    pub poison_damage: f32,
    pub burn_damage: f32,
    pub poison_tick: i32,
    pub burn_tick: i32,
    /// +0x3f0：暂停精灵动画的倒计时。
    pub anim_freeze: i32,
    /// +0x328：生成时的游戏帧。
    pub spawn_frame: u32,
    /// +0x20 `Index`：房间内创建序号。
    pub index: i32,
    /// +0x3c8 `SpawnerEntity`（眼泪记录发射者）。
    pub spawner: Option<u32>,
    pub npc: Option<NpcState>,
    pub player: Option<PlayerState>,
    pub tear: Option<TearState>,
    pub projectile: Option<crate::projectile::ProjectileState>,
}

impl Entity {
    /// 通用初始化（Entity::Init，RVA 0x002A93A0）里与运动相关的默认值。
    pub(crate) fn base(
        id: u32,
        kind: EntityKind,
        etype: i32,
        variant: i32,
        subtype: i32,
        pos: Vec2,
    ) -> Entity {
        Entity {
            id,
            kind,
            etype,
            variant,
            subtype,
            exists: true,
            dead: false,
            visible: true,
            flags: 0,
            pos,
            vel: Vec2::ZERO,
            last_pos: pos,
            last_vel: Vec2::ZERO,
            friction: 1.0,
            frame_friction: 1.0,
            speed_mult: 1.0, // Init：param_1[0xe7] = 1.0
            time_prev: 0.0,
            time_cur: 0.0,
            size: 10.0,
            size_mult: Vec2::new(1.0, 1.0),
            collision_points: Vec::new(),
            grid_collision_class: grid_collision_class::GROUND,
            entity_collision_class: entity_collision::ALL, // Init：param_1[0x62] = 4
            collides_with_grid: false,
            grid_hit_velocity: Vec2::ZERO,
            grid_normal: Vec2::ZERO,
            mass: 5.0,
            mass2: 5.0,
            collision_damage: 0.0,
            hp: 10.0,
            max_hp: 10.0,
            pending_damage: 0.0,
            knockback_dir: Vec2::ZERO,
            knockback_countdown: 0,
            freeze_countdown: 0,
            poison_countdown: 0,
            slowing_countdown: 0,
            charmed_countdown: 0,
            confusion_countdown: 0,
            fear_countdown: 0,
            burn_countdown: 0,
            midas_countdown: 0,
            poison_damage: 2.0,
            burn_damage: 2.0,
            poison_tick: 0,
            burn_tick: 0,
            anim_freeze: 0,
            spawn_frame: 0,
            index: 0,
            spawner: None,
            npc: None,
            player: None,
            tear: None,
            projectile: None,
        }
    }

    /// 按 `entities2.xml`（Repentance+）的 Gaper 配置建一个 Gaper。
    /// 10.0 皱眉 Gaper / 10.1 Gaper / 10.2 燃烧 Gaper：baseHP 10/10/9，collisionRadius 13，collisionMass 5，
    /// collisionDamage 1，friction 1，numGridCollisionPoints 12，gridCollision 默认（GROUND）。
    pub fn new_gaper(id: u32, variant: i32, pos: Vec2) -> Entity {
        let mut e = Entity::base(id, EntityKind::Npc, TYPE_GAPER, variant, 0, pos);
        e.max_hp = if variant == 2 { 9.0 } else { 10.0 };
        e.hp = e.max_hp;
        e.mass = 5.0;
        e.mass2 = 5.0;
        e.collision_damage = 1.0;
        e.friction = 1.0;
        e.frame_friction = 1.0;
        crate::physics::set_size(&mut e, 13.0, Vec2::new(1.0, 1.0), 12);
        // Entity_NPC::Init 第 75 行：flags |= FLAG_APPEAR
        e.flags |= flags::APPEAR;
        e.npc = Some(NpcState::default());
        e
    }

    /// 玩家（Isaac 基础属性）：`entities2.xml` 1.0 Player collisionRadius 10、collisionMass 5、
    /// numGridCollisionPoints 40、friction 1；属性值见 `player::PlayerStats::isaac()`。
    pub fn new_player(id: u32, pos: Vec2) -> Entity {
        let mut e = Entity::base(id, EntityKind::Player, TYPE_PLAYER, 0, 0, pos);
        e.mass = 5.0;
        e.mass2 = 5.0;
        e.max_hp = 6.0; // 3 颗红心 = 6 个半心
        e.hp = 6.0;
        crate::physics::set_size(&mut e, 10.0, Vec2::new(1.0, 1.0), 40);
        e.player = Some(PlayerState::default());
        e
    }

    /// Normal, non-champion Monstro (20.0): entities2.xml and native scenario HP 250.
    pub fn new_monstro(id: u32, pos: Vec2) -> Entity {
        let mut e = Entity::base(id, EntityKind::Npc, TYPE_MONSTRO, 0, 0, pos);
        e.hp = 250.0;
        e.max_hp = 250.0;
        e.mass = 50.0;
        e.mass2 = 50.0;
        e.collision_damage = 1.0;
        crate::physics::set_size(&mut e, 40.0, Vec2::new(1.0, 1.0), 12);
        e.npc = Some(NpcState {
            state: 1,
            anim: "Appear",
            ..NpcState::default()
        });
        e
    }

    /// 兼容旧名：不输入任何动作的玩家。
    pub fn new_player_stub(id: u32, pos: Vec2) -> Entity {
        Entity::new_player(id, pos)
    }

    pub fn has_flag(&self, f: u64) -> bool {
        self.flags & f != 0
    }

    /// 本帧逻辑帧数 `(int)time_cur - (int)time_prev`（NPC AI 里到处出现的写法）。
    pub fn frame_delta(&self) -> i32 {
        (self.time_cur as i32) - (self.time_prev as i32)
    }

    pub fn is_enemy_type(&self) -> bool {
        (10..1000).contains(&self.etype)
    }

    pub fn npc(&self) -> &NpcState {
        self.npc.as_ref().expect("entity is not an NPC")
    }

    pub fn npc_mut(&mut self) -> &mut NpcState {
        self.npc.as_mut().expect("entity is not an NPC")
    }

    pub fn player(&self) -> &PlayerState {
        self.player.as_ref().expect("entity is not a player")
    }

    pub fn player_mut(&mut self) -> &mut PlayerState {
        self.player.as_mut().expect("entity is not a player")
    }

    pub fn tear(&self) -> &TearState {
        self.tear.as_ref().expect("entity is not a tear")
    }

    pub fn tear_mut(&mut self) -> &mut TearState {
        self.tear.as_mut().expect("entity is not a tear")
    }
}
