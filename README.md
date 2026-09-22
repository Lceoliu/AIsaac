# Isaac RL 训练场

更新：2026-09-22。**原版四引擎累计512局训练已完成，但零清房；Rust 基础手感/Monstro/炸弹已接入，批量采样与紧凑训练缓冲区已实测，新策略尚未开训。** 原版训练与可视化验收保留，不修改已有 Transformer + 地图CNN方案。Rust基础Isaac已通过原版644帧校准和6776帧留出对照；Monstro 条件运动、弹道及高度碰撞对照见下文，尚无新策略的sim2real迁移成绩。旧 `isaac_room/` 已否定，只作历史记录。详细结果与边界见 [ENV_ARCHITECTURE.md §0.5](docs/ENV_ARCHITECTURE.md#05-rust-基础手感原版对照已通过2026-09-21)。

**手感测试：** `pwsh -File sim/play_motion.ps1`，WASD移动、方向键射击、R复位、F1显示碰撞圈。展示使用已有解包原图，动力学走同一Rust内核。

**Monstro 已接入（2026-09-21）：** `pwsh -File sim/play_motion.ps1 -Monstro -Seed 42`。普通 Monstro 的接近小跳、高跳锁定格子、落地环射、Taunt 前摇扇形喷弹及弹幕高度碰撞已实现；同 seed + 同输入可复现。原版 2,700 帧动作轨迹和额外高度碰撞探针已采集。观测走现有 CNN + Transformer，详见 [§0.6](docs/ENV_ARCHITECTURE.md#06-monstro-接入与可见观测2026-09-21)。尚未开始 Linux 128 并行或新策略训练。

## 目标

**2026-09-22 GPU缓存已实现，未正式开训：** GPU常驻原始帧/rollout，分块pinned双缓冲与异步传输，冻结采样期间只编码新帧；训练从原始历史重算梯度，minibatch默认32。本机128路短测约4088决策/s、峰值CUDA2.02GiB，默认1块快于2/4块；物理仍在CPU。28项测试及隔离更新探针通过，详见 [最新实现与实测](docs/ENV_ARCHITECTURE.md#09-gpu-原始帧rollout-与冻结编码缓存2026-09-22已实现)。`python bridge/python/train_sim.py` 默认只展示配置，`--train`才开训；`--pipeline legacy`保留CPU对照。

构建可迁移到原版以撒的完整流程 AI。先证明单房间战斗能力，再扩展探索、道具选择、资源管理和整局决策。策略推理只接收玩家可观察信息；视觉或内存只是采集手段，隐藏实体、内部 AI 状态、RNG 和未来事件不得进入策略输入。成绩只以原版（L0）为准。

## 当前状态

| 项 | 状态 | 位置 |
|---|---|---|
| 原版桥接与单房间课程 | 实机通过；Isaac / 无道具 / Monstro / 固定空房。首次建入口快照，此后 rewind + 本地 Boss 模板，实测重置约0.27秒 | [bridge/](bridge/README.md) |
| Gymnasium / PPO策略 | 地图CNN、实体注意力、64帧时序Transformer；2/4原版引擎共用模型训练已跑通，累计512局全部死亡 | [并行训练入口](bridge/python/train_parallel.py)、[实测](docs/ENV_ARCHITECTURE.md#04-独立引擎并行采样2026-09-21) |
| 奖励与观测 | 受伤−1、扣血命中+0.05、实际伤害/敌人初始MaxHP、清房+1、死亡不重复扣分；当前模型地图7通道，可见实体含敌弹 | [契约与边界](docs/ENV_ARCHITECTURE.md#03-transformer-战斗策略-v12026-09-21) |
| 渲染控制与稳定性 | 保留原版无渲染训练和可视化模式，虚拟时钟关闭；本轮两个正式校准采样进程均正常退出0。历史崩溃修复及边界见专项文档 | [崩溃根因分析](docs/NATIVE_CRASH_ANALYSIS.md)、[L1 验收](docs/L1_FEASIBILITY_PLAN.md) |
| L2 独立模拟器 | 玩家/Monstro/普通炸弹对照通过，4种地形与10K种子安全出生，44项Rust测试；批量采样/推理跑到128环境，未开训，完整战斗迁移尚未验收 | [sim/](sim/)、[最新验收](docs/ENV_ARCHITECTURE.md#07-炸弹场景分布与训练准备2026-09-22) |
| 相关项目调研 | 已含 Isaac 专项项目的实测数据；2026-09-20 新增 SC2/Dota/SoulsGym/EnvPool/Isaac-RL 的引擎适配做法调研（逐条附 URL）与本项目的训练场差距清单和实施顺序 | [RELATED_WORK.md](docs/RELATED_WORK.md)、[训练场设计](docs/TRAINING_ARENA_DESIGN.md)、[调研笔记](docs/ARENA_RESEARCH_NOTES.md) |
| 旧 v0 原型 | 已否定；代码与自测记录保留 | [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)、`isaac_room/` |

已有真实训练接入结果，但**尚无原版单房间能力达标证据**。最新512步训练用时74.27秒，参数L2变化0.6133；4个结束回合全部死亡。另3局规则控制器获得24/23/25个正奖励步骤，不是学习成绩。原始日志/模型保留在被Git排除的 `runs/l1/20260920-rewind/`；可复现入口、奖励语义和结果摘要维护在代码与文档中。旧 Rust 约95万帧/秒只是简化子集基准，不能与完整原引擎训练吞吐等同。

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

## 下一步

1. 用户验收 Rust Monstro 战斗；保留原版作为对照量具。
2. 将当前 Rust 观测适配器扩展为无 JSON/子进程开销的批量训练接口，接入既定奖励和 120 秒单局上限，再测试 Linux 128 并行。
3. 用独立种子评估原版迁移；不把模拟器胜率当作原版通关成绩。

## 复用边界

- [分析区](../analysis/README.md)：帧循环、ZHL 命名表、资源索引等共用游戏事实。
- [NetFix](../netfix/README.md)：注入器、MinHook 接线、J460 地址档案与启动/日志经验，供 L1 复用；不以其未完成的任意帧恢复作为前置条件。
- [第三方区](../third_party/README.md)：REPENTOGON 只作 ABI/特征码参考，不能在 J460 上运行。

## 历史原型（已否定，仅记录）

`isaac_room/` 是 2026-09-19 之前写的通用射击原型：固定 60 Hz、自编敌人、几何绘图，没有使用任何原版资源或机制。它可以运行（`play.ps1`、`python -m isaac_room.benchmark`），37 项自测和 24/24 脚本清房只验证自编规则，不构成以撒仿真的验收。细节见 [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)。

## Git 管理

`Isaac/rl` 为独立本地Git仓库，与 `Isaac/netfix` 分开；本次未配置远程或推送。跟踪桥接/训练/原生控制层代码、测试、观测fixture、设计文档，以及已明确标为历史的模拟器代码。`runs/`、模型/转储/日志、依赖目录、Rust target及原生build产物不入库。

本仓库仍依赖工作区的 `../third_party/minhook`、`../../.tools` 工具链及 `../analysis` 机制材料；首次提交不表示已成为可单独克隆构建的发行包。先保留依赖关系，不复制第三方或游戏资源进本仓库。
