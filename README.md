# Isaac RL 训练场

更新：2026-09-29。

## 目标

构建可迁移到原版以撒的完整流程 AI。先证明单房间战斗能力，再扩展探索、道具选择、资源管理和整局决策。策略推理只接收玩家可观察信息；视觉或内存只是采集手段，隐藏实体、内部 AI 状态、RNG 和未来事件不得进入策略输入。成绩只以原版引擎为准。

## 当前主线（2026-09-24 起）

- **战斗 RL：原版 AB+ 引擎**（[bridge/abplus/](bridge/abplus/README.md)）
  - 在《以撒的结合：胎衣†》v1.06 原生 Linux 版上，用加速后的原版引擎并行训练房间战斗策略（GPU PPO，Transformer 策略）。
  - 全部实验 A1–C40 记在 [EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md)，开头有总览表。
  - 进展：
    - 瞄准课程（C22–C30）之后，无敌时普通房稳定清房（C31）。
    - 阶段二从 C33 起去掉无敌，C37 以后主要在减少挨打。
    - C38–C40 试了对战场自对弈，没有带来提升。
    - C39 在第二台从头训练（combat-hp、四类房间、Room Buffer）。
  - 远端机器、看板和各台部署的桥接版本见该 README 的"现状"。
- **宏观规划**（[macro/](macro/README.md)，设计见 [MACRO_PLANNING.md](docs/MACRO_PLANNING.md)）
  - AB+ 楼层生成器的 Python 翻译，与引擎逐种子、逐房间一致；隐藏房推断、道具先验、规则规划器。
  - 忏悔+ 生成器的静态移植（没有和引擎核对），以及网页"找隐藏房"和以撒小测试。
- **可发布的 gym 仓库**（`../isaac-abplus-gym`，还没发布）：AB+ 环境加 PPO 基线，GPL-3.0，英文文档。
- **外部评审报告**：[COMBAT_RL_EXPERT_REVIEW.md](docs/COMBAT_RL_EXPERT_REVIEW.md)，截至 C37，含并行训练方案。

## 暂停的线

| 项 | 状态 | 位置 |
|---|---|---|
| L2 Rust 单房间模拟器 + 128 环境 GPU PPO | 2026-09-21–24 的主线。u425 迁移到 AB+：模拟器胜率 19.1%，AB+ 0.4%；原因是模拟器缺少 Monstro 出场阶段，策略学会了开局炸弹（A2）。修不修待定 | [sim/](sim/)、[ENV_ARCHITECTURE.md](docs/ENV_ARCHITECTURE.md) |
| 忏悔+（J460）原版桥接与 L1 加速 | 实机跑通；2/4 引擎共用模型训练，累计 512 局全部死亡；Windows 上三类原生崩溃的根因已查明。AB+ 路线接手后暂停 | [bridge/](bridge/README.md)、[turbo/](turbo/README.md)、[L1_FEASIBILITY_PLAN.md](docs/L1_FEASIBILITY_PLAN.md)、[NATIVE_CRASH_ANALYSIS.md](docs/NATIVE_CRASH_ANALYSIS.md) |
| 真人录制与模仿学习 | 六局录制和首轮 BC 完成；BC 后额外 64 局胜率 13/64 → 1/64，没有替换策略；暂缓 | `record_human.ps1`、[ENV_ARCHITECTURE.md §0.13](docs/ENV_ARCHITECTURE.md#013-原版真人录制2026-09-23) |
| 旧 v0 原型 `isaac_room/` | 已否定，见"历史原型" | [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md) |

## 文档

| 文档 | 内容 | 状态 |
|---|---|---|
| [bridge/abplus/README.md](bridge/abplus/README.md)、[EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md) | AB+ 桥接、训练管线和用法；全部实验 | 当前 |
| [macro/README.md](macro/README.md)、[MACRO_PLANNING.md](docs/MACRO_PLANNING.md) | 宏观层离线工具；路线规划与道具选择里规则、推断、搜索和学习的分工 | 当前 |
| [COMBAT_RL_EXPERT_REVIEW.md](docs/COMBAT_RL_EXPERT_REVIEW.md) | 给外部专家的进展汇总和并行训练方案 | 截至 C37（09-28） |
| [PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | 目标、观测边界、分阶段设计 | 目标和观测边界仍有效；"当前主线"停在 09-23 |
| [ENV_ARCHITECTURE.md](docs/ENV_ARCHITECTURE.md) | 三层环境栈（原版 / 加速 / 模拟器）与统一契约；模拟器训练、PPO 审查、真人录制 | 模拟器阶段（到 09-23） |
| [RELATED_WORK.md](docs/RELATED_WORK.md) | 游戏 RL 的相关工作：结构化观测、并行环境与迁移 | 09-18 的调研 |
| [TRAINING_ARENA_DESIGN.md](docs/TRAINING_ARENA_DESIGN.md)、[ARENA_RESEARCH_NOTES.md](docs/ARENA_RESEARCH_NOTES.md) | 把原版引擎做成训练场：参照项目和实施方案；原始调研笔记（每条附 URL） | 历史（J460 方案，09-20–22） |
| [L1_FEASIBILITY_PLAN.md](docs/L1_FEASIBILITY_PLAN.md) | J460 原版加速的可行性方案和控制层 | 历史（09-19–20） |
| [NATIVE_CRASH_ANALYSIS.md](docs/NATIVE_CRASH_ANALYSIS.md) | J460 worker 的原生崩溃根因与修复方案 | 历史（09-20） |
| [bridge/README.md](bridge/README.md)、[turbo/README.md](turbo/README.md) | J460 桥接 mod 和加速层 | 暂停 |
| [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md) | 旧 v0 原型的实现与自测 | 已否定 |

引擎逆向和游戏机制的共用材料在 [../analysis/](../analysis/README.md)，例如 AB+ 引擎改造层 [ABP_LINUX_REVERSE_ENGINEERING.md](../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md) 和游戏入门 [ISAAC_GAME_PRIMER.md](../analysis/docs/ISAAC_GAME_PRIMER.md)。

## 构建 L2 内核

`cargo` 不在 PATH 上；用 [sim/build.ps1](sim/build.ps1)（复用 `D:\Projects\fortune\.tooling` 的便携 GNU 工具链，与 `netfix/scripts/build_rust.ps1` 同一套环境）：

```powershell
.\rl\sim\build.ps1            # cargo test
.\rl\sim\build.ps1 run --example gaper_chase
.\rl\sim\build.ps1 run --example player_shoot
.\rl\sim\build.ps1 run --release --example bench
```

## 构建 L1 加速层

```powershell
pwsh -NoProfile -File .\rl\turbo\build.ps1     # DLL + 注入器 + 离线测试 + exe 字节核对
python .\rl\bridge\python\test_turbo_control.py  # Python 端共享内存双向测试
```

## 模拟器阶段的记录（2026-09-21–24）

AB+ 路线开始前的主线，细节都在 [ENV_ARCHITECTURE.md](docs/ENV_ARCHITECTURE.md)：
- **手感测试**：`pwsh -File sim/play_motion.ps1`，WASD 移动、方向键射击、R 复位、F1 显示碰撞圈；`-Monstro -Seed 42` 打 Monstro。展示用已有的解包原图，动力学走同一个 Rust 内核。普通 Monstro 的接近小跳、高跳锁定格子、落地环射、Taunt 前摇扇形喷弹和弹幕高度碰撞已实现，同 seed + 同输入可复现（[§0.6](docs/ENV_ARCHITECTURE.md#06-monstro-接入与可见观测2026-09-21)）。
- **Ubuntu 训练**：`/home/eolc/isaac-rl`，RTX 3080 Ti，128 环境 GPU PPO（部署验收时完整 PPO 更新约 287 条新 transition/s，峰值 CUDA 2.39 GiB），GPU 常驻原始帧和 rollout（[§0.9](docs/ENV_ARCHITECTURE.md#09-gpu-原始帧rollout-与冻结编码缓存2026-09-22已实现)），每 10 轮 checkpoint、固定留出种子评估（[§0.11](docs/ENV_ARCHITECTURE.md#011-checkpoint断点续训与固定留出评估2026-09-22)）。`python bridge/python/train_sim.py` 默认只展示配置，`--train` 才开训；`--pipeline legacy` 保留 CPU 对照。
- **PPO 审查**：旧实验 `monstro-128k-reward-v1-20260923`（batch 32 × 4 epochs）每轮更新过大（精确 KL 约 0.14），在 256 个未见种子上从第 425 轮的 19% 退到第 750 轮的 7%，不是 reward hacking（[§0.14](docs/ENV_ARCHITECTURE.md#014-当前阶段与-claude-只读审查交接2026-09-23)，`runs/ppo-audit/20260923-e1e2/`）。
- **最后一次模拟器训练**：`monstro-128k-seg1024-20260923`，09-23 21:29 从第 425 轮续训，batch 1024 × 2 epochs、帧去重分段、训练开局随机化、熵 0.01（[§0.15](docs/ENV_ARCHITECTURE.md#015-更新量帧去重与训练开局随机化2026-09-23已上线)）。09-24 它的 u8110 被用作 AB+ 第一次训练的热启动（EXPERIMENTS.md C2）。
- **迁移到原版**：同种子配对评估 u425，模拟器胜率 19.1%，AB+ 0.4%（A2）。修复办法是给模拟器的 Monstro 加出场阶段，待定。
- **真人录制**：`pwsh -File record_human.ps1` 启动独立的原版游戏副本和 http://127.0.0.1:8764 侧边面板，F6 倒数开始、F7 保存并重置、F8 结束；正式样本在 `runs/human/`，验收样本在 `runs/human-recorder-qa/`（[§0.13](docs/ENV_ARCHITECTURE.md#013-原版真人录制2026-09-23)）。
  - 数据处理：`python -m isaac_bridge.human_recording SESSION --prepare-to NEW-DIRECTORY --validation-episode episode-0006.jsonl.gz`。
  - BC：`python bridge/python/train_human_bc.py --checkpoint CHECKPOINT --dataset DATASET --out NEW-DIRECTORY --run`，须设 `PYTHONPATH=bridge/python`。

## 复用边界

- [分析区](../analysis/README.md)：帧循环、ZHL 命名表、资源索引等共用游戏事实。
- [NetFix](../netfix/README.md)：注入器、MinHook 接线、J460 地址档案与启动/日志经验，供 L1 复用；不以其未完成的任意帧恢复作为前置条件。
- [第三方区](../third_party/README.md)：REPENTOGON 只作 ABI/特征码参考，不能在 J460 上运行。

## 历史原型（已否定，仅记录）

`isaac_room/` 是 2026-09-19 之前写的通用射击原型：固定 60 Hz、自编敌人、几何绘图，没有使用任何原版资源或机制。它可以运行（`play.ps1`、`python -m isaac_room.benchmark`），37 项自测和 24/24 脚本清房只验证自编规则，不构成以撒仿真的验收。细节见 [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)。

## Git 管理

`Isaac/rl` 为独立本地Git仓库，与 `Isaac/netfix` 分开；没有配置远程，也没有推送。跟踪桥接/训练/原生控制层代码、测试、观测fixture、设计文档，以及已明确标为历史的模拟器代码。`runs/`、模型/转储/日志、依赖目录、Rust target及原生build产物不入库。

本仓库仍依赖工作区的 `../third_party/minhook`、`../../.tools` 工具链及 `../analysis` 机制材料；首次提交不表示已成为可单独克隆构建的发行包。先保留依赖关系，不复制第三方或游戏资源进本仓库。
