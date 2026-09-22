//! isaac_sim — 以撒的结合（Repentance+ J460）单房间战斗内核。
//!
//! 每个模块都是对 J460 反编译伪 C 的翻译，注释里给出函数 RVA（VA = RVA + 0x00400000）与
//! `analysis/j460/exports/j460-baseline/decompiled/` 中对应文件的行号。规则笔记见
//! `analysis/docs/J460_NPC_MOVEMENT_MODEL.md`（实体/NPC）与 `analysis/docs/J460_PLAYER_TEAR_MODEL.md`（玩家/眼泪）。
//!
//! 已翻译：`Entity::PreUpdate/Update`（摩擦、积分、击退、状态倒计时、待结算伤害）、实体-网格碰撞
//! （NPC 版 FUN_006a9c90 与玩家版 FUN_006ab440）、碰撞采样点生成（Entity::SetSize）、房间网格/路径图/视线检测、
//! `NPCAI_Pathfinder::FindGridPath` 与 `MoveRandomlyBoss`、Gaper 与 Gusher/Pacer 的 AI、NPC 每帧骨架、
//! 玩家移动/射击输入/开火节奏/眼泪参数/无敌帧、眼泪飞行/下落/落地/撞墙/命中与推挤、NPC 对玩家的接触伤害、
//! EntityList 的更新顺序。
//!
//! 正常 Monstro 20.0 与其血弹已接入（原版条件运动/弹道及可见事件对照见 ENV_ARCHITECTURE §0.6）。
//! 尚未翻译（占位或事件）：其他敌人弹幕、状态效果的施加、其他精灵动画、
//! 道具/饰品效果、跨房。2026-09-21：基础 Isaac 的移动、射击与空房眼泪轨迹已通过
//! J460 原版校准及独立留出测试（sim/tests/fixtures）；NPC/伤害/击退仍未完成此级别验收。

pub mod entity;
pub mod bomb;
pub mod arena;
pub mod snapshot;
pub mod ffi;
pub mod observation;
pub mod math;
pub mod npc;
pub mod pathfinder;
pub mod physics;
pub mod player;
pub mod projectile;
pub mod rng;
pub mod room;
pub mod tear;
pub mod world;

pub use entity::{Entity, EntityKind, NpcState};
pub use math::Vec2;
pub use player::{PlayerInput, PlayerState, PlayerStats};
pub use room::Room;
pub use tear::TearState;
pub use world::{Event, World};
