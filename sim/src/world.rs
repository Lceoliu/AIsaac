//! 一帧的调度，按 `Game::Update → EntityList::Update`（FUN_004186c0，RVA 0x000186C0）与
//! `Room::Update`（FUN_00802980）还原：
//! 1. 所有实体 `PreUpdate`（第 135 行的虚表 +8）；
//! 2. 所有实体 `Update`（第 166 行虚表 +0xc；NPC → `Entity_NPC::Update`，玩家 → `Entity_Player::Update`
//!    （其中发射眼泪），眼泪 → `Entity_Tear::Update`）；
//! 3. `FUN_00419e70` 碰撞阶段：分区重建、实体-实体碰撞（`FUN_0041c070/FUN_0041bdf0` → 推挤 `FUN_006b2940`
//!    与双方 `HandleCollision`：NPC 撞玩家造成接触伤害，眼泪撞 NPC 造成伤害与推挤），
//!    最后对每个实体调用网格碰撞（非玩家 `FUN_006a9c90(0)`，玩家 `FUN_006ab440(0)`）；
//! 4. 房间：路径标记衰减（每 3 帧）。玩家不写路径图：`FUN_004cdd30` 只被 `FindGridPath` 调用，
//!    `Entity_Player::Update` 不触碰 `Room+0x76c`。
//!
//! 原版 Game::Update 内 Room::Update 与 EntityList::Update 的先后尚未核对，这里房间更新放在末尾。
//! 新眼泪出生时立即 Update 一次，加入本帧碰撞但不再重复 Update；已用原版出生/飞行轨迹验证。

use crate::entity::{Entity, EntityKind};
use crate::math::Vec2;
use crate::npc;
use crate::physics;
use crate::player::{self, PlayerInput};
use crate::rng::Rng;
use crate::room::Room;
use crate::tear;

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Internal spawn request; never an actor observation (contains current aim).
    BossVolley {
        from: u32,
        pos: Vec2,
        target: Vec2,
        count: usize,
    },
    /// Gusher 请求发射血弹（弹幕实体尚未实现）。
    FireProjectiles { from: u32, pos: Vec2, vel: Vec2 },
    /// NPC 死亡并被移除。
    NpcDied { id: u32, etype: i32, variant: i32 },
    /// NPC 与玩家发生接触（每个重叠帧都会有）。
    Contact {
        npc: u32,
        player: u32,
        collision_damage: f32,
    },
    /// 玩家发射了一颗眼泪。
    TearFired {
        player: u32,
        tear: u32,
        pos: Vec2,
        vel: Vec2,
    },
    /// 眼泪命中 NPC（伤害进入 NPC 的待结算伤害，下一帧扣血）。
    TearHit { tear: u32, npc: u32, damage: f32 },
    /// 眼泪消失：cause 1 落地、2 撞墙、3 命中。
    TearRemoved { tear: u32, cause: u8, pos: Vec2 },
    /// 玩家受伤（半心单位）。
    PlayerDamaged {
        player: u32,
        damage: f32,
        hp_after: f32,
        source: Option<u32>,
    },
}

#[derive(Clone, Debug)]
pub struct World {
    pub room: Room,
    pub entities: Vec<Entity>,
    /// `Game+0x264f8` 帧计数（每次 Game::Update 加一）。
    pub frame: u32,
    pub rng: Rng,
    pub events: Vec<Event>,
    next_id: u32,
    next_index: i32,
}

impl World {
    pub fn new(room: Room, seed: u32) -> World {
        World {
            room,
            entities: Vec::new(),
            frame: 0,
            rng: Rng::new(seed),
            events: Vec::new(),
            next_id: 1,
            next_index: 0,
        }
    }

    pub fn spawn(&mut self, mut e: Entity) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        e.id = id;
        e.index = self.next_index;
        self.next_index += 1;
        e.spawn_frame = self.frame;
        self.entities.push(e);
        id
    }

    pub fn spawn_gaper(&mut self, variant: i32, pos: Vec2) -> u32 {
        self.spawn(Entity::new_gaper(0, variant, pos))
    }

    /// 生成一名 Isaac（基础属性），初始无输入。
    pub fn spawn_player(&mut self, pos: Vec2) -> u32 {
        self.spawn(Entity::new_player(0, pos))
    }

    pub fn spawn_monstro(&mut self, pos: Vec2) -> u32 {
        self.spawn(Entity::new_monstro(0, pos))
    }

    /// 兼容旧名。
    pub fn spawn_player_stub(&mut self, pos: Vec2) -> u32 {
        self.spawn_player(pos)
    }

    pub fn entity(&self, id: u32) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn tears(&self) -> impl Iterator<Item = &Entity> {
        self.entities
            .iter()
            .filter(|e| e.kind == EntityKind::Tear && e.exists)
    }

    fn first_player_pos(&self) -> Vec2 {
        self.entities
            .iter()
            .find(|e| e.kind == EntityKind::Player && e.exists)
            .map(|e| e.pos)
            .unwrap_or(Vec2::ZERO)
    }

    /// 设置玩家本帧的输入（下一次 `step` 生效）。
    pub fn set_player_input(&mut self, id: u32, input: PlayerInput) {
        if let Some(p) = self.entity_mut(id) {
            if let Some(ps) = p.player.as_mut() {
                ps.input = input;
            }
        }
    }

    /// 推进一个逻辑帧（30 Hz）。
    pub fn step(&mut self) {
        self.interpolate_players();
        self.step_logic();
    }

    /// Odd manager tick (60 Hz); keyboard viewers can sample a new action here.
    pub fn interpolate_players(&mut self) {
        // From one MC_POST_UPDATE boundary to the next: odd manager tick first,
        // then the full 30 Hz game tick. Only players get this extra integration.
        for e in self.entities.iter_mut() {
            if e.kind == EntityKind::Player && e.exists {
                player::interpolate_player(e);
                physics::resolve_grid_collision(e, &self.room, false);
            }
        }
    }

    /// Even manager tick. Call after interpolate_players, or use step() for RL.
    pub fn step_logic(&mut self) {
        self.events.clear();
        let mut player_pos = self.first_player_pos();
        let frame = self.frame;

        // 1. PreUpdate
        for e in self.entities.iter_mut() {
            if e.exists {
                physics::pre_update(e);
            }
        }

        // 2. Update
        let mut events = std::mem::take(&mut self.events);
        let mut spawned: Vec<Entity> = Vec::new();
        for e in self.entities.iter_mut() {
            if !e.exists {
                continue;
            }
            match e.kind {
                EntityKind::Npc => npc::update_npc(
                    e,
                    &mut self.room,
                    player_pos,
                    frame,
                    &mut self.rng,
                    &mut events,
                ),
                EntityKind::Player => {
                    player::update_player(e, &mut self.rng, &mut events, &mut spawned);
                    player_pos = e.pos;
                }
                EntityKind::Tear => tear::update(e),
                EntityKind::Projectile => crate::projectile::update(e),
            }
        }
        for event in &events {
            if let Event::BossVolley {
                from,
                pos,
                target,
                count,
            } = event
            {
                spawned.extend(crate::projectile::boss_volley(
                    *pos,
                    *target,
                    *count,
                    *from,
                    &mut self.rng,
                ));
            }
        }
        // 本帧发射的眼泪加入实体表（TearFired 事件里的 id 在此分配前为 0，这里回填）
        for mut t in spawned.drain(..) {
            let id = self.next_id;
            self.next_id += 1;
            t.id = id;
            t.index = self.next_index;
            self.next_index += 1;
            t.spawn_frame = frame;
            for ev in events.iter_mut() {
                if let Event::TearFired { tear, pos, .. } = ev {
                    if *tear == 0 && *pos == t.pos {
                        *tear = id;
                    }
                }
            }
            self.entities.push(t);
        }

        // 3. 碰撞阶段：实体-实体，再网格碰撞
        let n = self.entities.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if !self.entities[i].exists || !self.entities[j].exists {
                    continue;
                }
                let (left, right) = self.entities.split_at_mut(j);
                let a = &mut left[i];
                let b = &mut right[0];
                match (a.kind, b.kind) {
                    (EntityKind::Npc, EntityKind::Npc) => {
                        physics::circle_push(a, b);
                    }
                    (EntityKind::Npc, EntityKind::Player) => npc_vs_player(a, b, &mut events),
                    (EntityKind::Player, EntityKind::Npc) => npc_vs_player(b, a, &mut events),
                    (EntityKind::Tear, EntityKind::Npc) => tear_vs_npc(a, b, &mut events),
                    (EntityKind::Npc, EntityKind::Tear) => tear_vs_npc(b, a, &mut events),
                    (EntityKind::Projectile, EntityKind::Player) => {
                        projectile_vs_player(a, b, &mut events)
                    }
                    (EntityKind::Player, EntityKind::Projectile) => {
                        projectile_vs_player(b, a, &mut events)
                    }
                    _ => {}
                }
            }
        }
        for e in self.entities.iter_mut() {
            if e.exists {
                physics::resolve_grid_collision(e, &self.room, false);
            }
        }

        // Keep the native-visible final integration until next tick's removal.
        for e in self.entities.iter_mut() {
            if e.exists && e.kind == EntityKind::Tear && e.dead {
                events.push(Event::TearRemoved {
                    tear: e.id,
                    cause: e.tear().death_cause,
                    pos: e.pos,
                });
            }
        }

        // 4. 房间
        self.room.update_path_markers(frame);

        self.entities.retain(|e| e.exists);
        self.events = events;
        self.frame += 1;
        self.room.frame = self.frame;
    }
}

/// NPC–玩家：圆形推挤（FUN_006b2940）+ `Entity_NPC::HandleCollision` 第 1094–1099 行的接触伤害
/// `player->TakeDamage(collisionDamage, 0, 0, source, 30)`。
fn npc_vs_player(npc_e: &mut Entity, player_e: &mut Entity, events: &mut Vec<Event>) {
    if npc_e.entity_collision_class == 0 {
        return;
    }
    if !physics::circle_push(npc_e, player_e) {
        return;
    }
    events.push(Event::Contact {
        npc: npc_e.id,
        player: player_e.id,
        collision_damage: npc_e.collision_damage,
    });
    if !npc_e.dead && npc_e.collision_damage > 0.0 {
        player::take_damage(player_e, npc_e.collision_damage, Some(npc_e.id), events);
    }
}

/// 眼泪–NPC：圆形重叠即命中（`Entity_Tear::HandleCollision`），眼泪不与发射者碰撞。
fn tear_vs_npc(tear_e: &mut Entity, npc_e: &mut Entity, events: &mut Vec<Event>) {
    if tear_e.dead || npc_e.dead || npc_e.entity_collision_class == 0 {
        return;
    }
    let d = npc_e.pos - tear_e.pos;
    let rsum = tear_e.size + npc_e.size;
    if d.length_sq() >= rsum * rsum {
        return;
    }
    tear::on_hit_npc(tear_e, npc_e, events);
}

fn projectile_vs_player(shot: &mut Entity, player: &mut Entity, events: &mut Vec<Event>) {
    if shot.dead || shot.entity_collision_class == 0 {
        return;
    }
    if (shot.pos - player.pos).length_sq() < (shot.size + player.size).powi(2) {
        shot.dead = true;
        crate::player::take_damage(player, 1.0, Some(shot.id), events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaper_appears_then_chases_player_in_empty_room() {
        let mut w = World::new(Room::new_rectangular(15, 9), 7);
        let player = w.spawn_player_stub(Vec2::new(500.0, 280.0));
        let gaper = w.spawn_gaper(1, Vec2::new(120.0, 280.0));
        let start = w.entity(gaper).unwrap().pos;
        // 出现流程：前 20 帧几乎不动
        for _ in 0..20 {
            w.step();
        }
        let after_appear = w.entity(gaper).unwrap().pos;
        assert!(
            (after_appear.x - start.x).abs() < 2.0,
            "moved {} during appear",
            after_appear.x - start.x
        );
        assert!(w.entity(gaper).unwrap().visible);
        // 之后向玩家移动并最终接触
        let mut contact = false;
        for _ in 0..240 {
            w.step();
            if w.events
                .iter()
                .any(|ev| matches!(ev, Event::Contact { .. }))
            {
                contact = true;
                break;
            }
        }
        let g = w.entity(gaper).unwrap();
        let p = w.entity(player).unwrap();
        assert!(
            contact,
            "gaper never reached the player; gaper at {:?}, player at {:?}",
            g.pos, p.pos
        );
        assert!(g.pos.x > 300.0);
        assert!(w
            .events
            .iter()
            .any(|ev| matches!(ev, Event::PlayerDamaged { .. })));
        assert_eq!(p.hp, 5.0);
    }

    #[test]
    fn killed_gaper_may_become_gusher() {
        let mut became_gusher = false;
        for seed in 1..40u32 {
            let mut w = World::new(Room::new_rectangular(15, 9), seed);
            w.spawn_player_stub(Vec2::new(500.0, 280.0));
            let gaper = w.spawn_gaper(1, Vec2::new(200.0, 280.0));
            for _ in 0..30 {
                w.step();
            }
            {
                let g = w.entity_mut(gaper).unwrap();
                g.hp = -1.0;
                g.dead = true;
            }
            w.step();
            if let Some(g) = w.entity(gaper) {
                assert_eq!(g.etype, crate::entity::TYPE_GUSHER);
                assert!(!g.dead);
                assert_eq!(g.hp, g.max_hp);
                became_gusher = true;
                break;
            }
        }
        assert!(became_gusher, "no seed produced a Gusher morph in 39 tries");
    }

    #[test]
    fn player_tears_kill_an_approaching_gaper() {
        let mut w = World::new(Room::new_rectangular(15, 9), 11);
        let player = w.spawn_player(Vec2::new(150.0, 280.0));
        let gaper = w.spawn_gaper(1, Vec2::new(450.0, 280.0));
        w.set_player_input(player, PlayerInput::new(0.0, 0.0, 1.0, 0.0));
        let mut hits = 0;
        let mut died = false;
        for _ in 0..150 {
            w.step();
            for ev in &w.events {
                match ev {
                    Event::TearHit { npc, .. } if *npc == gaper => hits += 1,
                    Event::NpcDied { id, .. } if *id == gaper => died = true,
                    _ => {}
                }
            }
            w.events.clear();
            if died {
                break;
            }
        }
        assert!(died, "gaper survived; hits={}", hits);
        assert!(hits >= 3, "hits={}", hits);
        assert_eq!(w.entity(player).unwrap().hp, 6.0);
    }

    #[test]
    fn tear_dies_on_the_wall_and_player_slides_along_it() {
        let mut w = World::new(Room::new_rectangular(15, 9), 2);
        let player = w.spawn_player(Vec2::new(100.0, 280.0));
        w.set_player_input(player, PlayerInput::new(-1.0, 0.0, -1.0, 0.0));
        let mut wall_deaths = 0;
        for _ in 0..40 {
            w.step();
            for ev in &w.events {
                if let Event::TearRemoved { cause: 2, .. } = ev {
                    wall_deaths += 1;
                }
            }
            w.events.clear();
        }
        assert!(wall_deaths >= 3, "wall deaths {}", wall_deaths);
        let p = w.entity(player).unwrap();
        assert!(p.pos.x >= 69.9 && p.pos.x < 74.0, "player x {}", p.pos.x);
        // 贴墙时按住斜向应能沿墙滑动
        w.set_player_input(player, PlayerInput::new(-1.0, 1.0, 0.0, 0.0));
        for _ in 0..30 {
            w.step();
        }
        let p = w.entity(player).unwrap();
        assert!(p.pos.y > 300.0, "player did not slide: y {}", p.pos.y);
        assert!(p.pos.x >= 69.9 && p.pos.x < 74.0);
    }
}
