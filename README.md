# Isaac RL 训练场

更新：2026-09-20。**当前主线是原版 J460 训练 worker**：Lua 同步桥接 → Gymnasium → SB3 PPO，已在真实游戏中完成采样、参数更新和模型重载。训练默认跳过渲染但不改变逻辑时钟，另保留可视化验收。L2 独立模拟器保留为课程子集与校准研究，不再作为接入原引擎训练的前置条件。旧自编规则原型 `isaac_room/` 已否定，只作历史记录。正式命令与详细验收见 [bridge/README.md](bridge/README.md)，设计见 [ENV_ARCHITECTURE.md](docs/ENV_ARCHITECTURE.md)。

## 目标

构建可迁移到原版以撒的完整流程 AI。先证明单房间战斗能力，再扩展探索、道具选择、资源管理和整局决策。策略推理只接收玩家可观察信息；视觉或内存只是采集手段，隐藏实体、内部 AI 状态、RNG 和未来事件不得进入策略输入。成绩只以原版（L0）为准。

## 当前状态

| 项 | 状态 | 位置 |
|---|---|---|
| 原版桥接与单房间课程 | 实机通过；Isaac / 无道具 / Monstro / 固定空房。首次建入口快照，此后 rewind + 本地 Boss 模板，实测重置约0.27秒 | [bridge/](bridge/README.md) |
| Gymnasium / PPO v2 | 单 worker、CPU、4逻辑帧/动作、双层64单元 MLP；22项测试通过，512步实机 PPO 更新/存取档/重载动作通过，终局仍全部死亡 | [训练入口](bridge/python/train_monstro.py) |
| 奖励与观测 | 用户确认受伤−1、扣血命中+0.05、实际伤害/敌人初始MaxHP、清房+1、死亡不重复扣分；地形5×9×15已进策略，敌弹标记已修复 | [契约与边界](docs/ENV_ARCHITECTURE.md#02-当前训练实现与下一阶段2026-09-20覆盖前文的历史优先级) |
| 渲染控制与稳定性 | 无渲染20回合历史验收和新版3回合验收通过；虚拟时钟关闭；不是独立无窗口引擎。三类原生崩溃的机制、调用链与修复方案已成文：A 字体 shader 栈下溢（入栈失败、注册表完好）、B 启动期归档挂载逐条目 `fopen` 失败、C 退出时 NVIDIA 驱动内跳转；探针与缓解已有代码并通过离线构建；覆盖与实机有效性仍待验收 | [崩溃根因分析](docs/NATIVE_CRASH_ANALYSIS.md)、[L1 验收](docs/L1_FEASIBILITY_PLAN.md) |
| L2 独立模拟器（保留研究） | Rust Gaper/Gusher/Pacer、移动/眼泪/碰撞等子集，历史24项单测通过；未完成原版轨迹校准，不代表全游戏引擎 | [sim/](sim/)、[机制分析](../analysis/docs/J460_NPC_MOVEMENT_MODEL.md) |
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

1. 按 [崩溃根因分析 §5](docs/NATIVE_CRASH_ANALYSIS.md) 的顺序先核对已有探针覆盖/错误码保真，再做冷启动循环和可视化回归；不把跳过渲染称为修复。
2. 同进程50回合 rewind 耐久，检查每局状态、伤害奖励账目和精确逻辑帧，再扩大训练预算。
3. 独立可视化留出验收；后续再补运动历史、危险区域、伤害来源归因与多实例隔离。

## 复用边界

- [分析区](../analysis/README.md)：帧循环、ZHL 命名表、资源索引等共用游戏事实。
- [NetFix](../netfix/README.md)：注入器、MinHook 接线、J460 地址档案与启动/日志经验，供 L1 复用；不以其未完成的任意帧恢复作为前置条件。
- [第三方区](../third_party/README.md)：REPENTOGON 只作 ABI/特征码参考，不能在 J460 上运行。

## 历史原型（已否定，仅记录）

`isaac_room/` 是 2026-09-19 之前写的通用射击原型：固定 60 Hz、自编敌人、几何绘图，没有使用任何原版资源或机制。它可以运行（`play.ps1`、`python -m isaac_room.benchmark`），37 项自测和 24/24 脚本清房只验证自编规则，不构成以撒仿真的验收。细节见 [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)。

## Git 管理

`Isaac/rl` 为独立本地Git仓库，与 `Isaac/netfix` 分开；本次未配置远程或推送。跟踪桥接/训练/原生控制层代码、测试、观测fixture、设计文档，以及已明确标为历史的模拟器代码。`runs/`、模型/转储/日志、依赖目录、Rust target及原生build产物不入库。

本仓库仍依赖工作区的 `../third_party/minhook`、`../../.tools` 工具链及 `../analysis` 机制材料；首次提交不表示已成为可单独克隆构建的发行包。先保留依赖关系，不复制第三方或游戏资源进本仓库。
