# Isaac RL 训练场

更新：2026-09-23。**当前主线是Ubuntu上的Rust单房间模拟器、128环境并行与GPU PPO，combat-v1目标128K完整回合。** PPO审查确认旧配置每轮更新过大导致振荡退化；21:29起以batch 1024×2、帧去重、训练开局随机化和小熵奖励从第425轮续训，见§0.15。暂不继续真人模仿学习。原版2/4引擎训练已跑通，累计512局零清房；原版保留作校准与迁移验收，尚无新策略迁移达标成绩。最新过程、证据目录和待查问题见 [ENV_ARCHITECTURE.md §0.14](docs/ENV_ARCHITECTURE.md#014-当前阶段与-claude-只读审查交接2026-09-23)。旧`isaac_room/`已否定，不是当前Rust引擎。

**手感测试：** `pwsh -File sim/play_motion.ps1`，WASD移动、方向键射击、R复位、F1显示碰撞圈。展示使用已有解包原图，动力学走同一Rust内核。

**Monstro 已接入（2026-09-21）：** `pwsh -File sim/play_motion.ps1 -Monstro -Seed 42`。普通 Monstro 的接近小跳、高跳锁定格子、落地环射、Taunt 前摇扇形喷弹及弹幕高度碰撞已实现；同 seed + 同输入可复现。原版 2,700 帧动作轨迹和额外高度碰撞探针已采集。观测走现有 CNN + Transformer，详见 [§0.6](docs/ENV_ARCHITECTURE.md#06-monstro-接入与可见观测2026-09-21)。Linux 128环境训练已启动，当前结果见§0.14。

**Ubuntu已部署（2026-09-22）：** `/home/eolc/isaac-rl`，RTX3080Ti。完整PPO更新验收约287条新transition/s，峰值CUDA2.39GiB。128环境正式训练已启动，当前目标128K完整回合；GPU管线新增每10轮checkpoint、每25轮16个固定留出种子评估、房间现场续训和离线状态回放。详见 [存档与评估](docs/ENV_ARCHITECTURE.md#011-checkpoint断点续训与固定留出评估2026-09-22)。

## 目标

**真人录制入口（2026-09-23）：** `pwsh -File record_human.ps1` 启动独立原版游戏副本与 `http://127.0.0.1:8764` 侧边面板，不启动训练。准备场景后 F6 倒数开始，F7 保存并 rewind 重置，F8 结束本局；原版键位保持不变。每局保留连续原始 JSONL、校验后的 gzip 和可视化状态回放，死亡/胜利/120 秒自动结束。正式样本在 `runs/human/`，验收样本在 `runs/human-recorder-qa/`，不能混用。详见 [真人录制契约](docs/ENV_ARCHITECTURE.md#013-原版真人录制2026-09-23)。

**真人训练数据与首轮 BC（已完成，现暂缓后续）：** 六局已处理到 `runs/human-prepared/20260923-session01-v2/`，第 1–5 局训练、第 6 局验证。复用现有 Transformer 的 15 Hz / H=64 输入，按输出头屏蔽无法可靠对齐的动作，batch 默认 32。处理命令为 `python -m isaac_bridge.human_recording SESSION --prepare-to NEW-DIRECTORY --validation-episode episode-0006.jsonl.gz`；`python bridge/python/check_human_dataset.py DATASET` 仅检查接线。实际 BC 实验入口为 `python bridge/python/train_human_bc.py --checkpoint CHECKPOINT --dataset DATASET --out NEW-DIRECTORY --run`。首轮从第425轮PPO微调，验证选第11个epoch，但额外64局胜率 **13/64→1/64**，不替换现有策略；结果与128局状态回放在 `runs/human-bc/20260923-update425/experiment/index.html`。须设置 `PYTHONPATH=bridge/python` 并使用既有训练环境；完整结果见上述真人录制契约。

**GPU缓存已实现并用于远端训练：** GPU常驻原始帧/rollout，分块pinned双缓冲与异步传输，冻结采样期间只编码新帧；训练从原始历史重算梯度，minibatch默认32。本机128路短测约4088决策/s、峰值CUDA2.02GiB，默认1块快于2/4块；物理仍在CPU。28项测试及隔离更新探针通过，详见 [最新实现与实测](docs/ENV_ARCHITECTURE.md#09-gpu-原始帧rollout-与冻结编码缓存2026-09-22已实现)。`python bridge/python/train_sim.py` 默认只展示配置，`--train`才开训；`--pipeline legacy`保留CPU对照。

构建可迁移到原版以撒的完整流程 AI。先证明单房间战斗能力，再扩展探索、道具选择、资源管理和整局决策。策略推理只接收玩家可观察信息；视觉或内存只是采集手段，隐藏实体、内部 AI 状态、RNG 和未来事件不得进入策略输入。成绩只以原版（L0）为准。

## 当前状态

| 项 | 状态 | 位置 |
|---|---|---|
| 原版桥接与单房间课程 | 实机通过；Isaac / 无道具 / Monstro / 固定空房。首次建入口快照，此后 rewind + 本地 Boss 模板，实测重置约0.27秒 | [bridge/](bridge/README.md) |
| Gymnasium / PPO策略 | 地图CNN、实体注意力、64帧时序Transformer；2/4原版引擎共用模型训练已跑通，累计512局全部死亡 | [并行训练入口](bridge/python/train_parallel.py)、[实测](docs/ENV_ARCHITECTURE.md#04-独立引擎并行采样2026-09-21) |
| 奖励与观测 | combat-v1：受伤−1、命中+0.05、归一化伤害、清房+3及速度奖金0～1、超时−1；地图7通道、可见敌弹及剩余时间 | [契约与边界](docs/ENV_ARCHITECTURE.md#03-transformer-战斗策略-v12026-09-21) |
| 渲染控制与稳定性 | 保留原版无渲染训练和可视化模式，虚拟时钟关闭；本轮两个正式校准采样进程均正常退出0。历史崩溃修复及边界见专项文档 | [崩溃根因分析](docs/NATIVE_CRASH_ANALYSIS.md)、[L1 验收](docs/L1_FEASIBILITY_PLAN.md) |
| AB+ 原版引擎桥接与训练管线（2026-09-24） | 与模拟器同种子同局面、逐帧可复现，6 实例 286 决策/秒。u425 迁移评估：模拟器胜率 19.1%，AB+ 0.4%；差距来自模拟器缺少 Monstro 出场阶段，策略学会了开局炸弹（模拟器 49 胜中 48 胜靠它）。训练管线已就绪：独占 GPU 的学习器批量推理、桥接 v2（每步 1 ms）、房间级混合（竞技场 + 495 个普通房 + 77 个 Boss 房）、奖励 combat-v2（血量曲线、死亡 −5、按游戏计分校准的时间、炸弹和清房分），环境侧上限 2,349 决策/秒 = 157× 实时 | [bridge/abplus/](bridge/abplus/README.md) |
| L2 独立模拟器 | 玩家/Monstro/普通炸弹对照通过，4种地形与10K种子安全出生，44项Rust测试；128环境GPU PPO已正式训练，完整战斗迁移尚未验收 | [sim/](sim/)、[最新验收](docs/ENV_ARCHITECTURE.md#07-炸弹场景分布与训练准备2026-09-22) |
| 相关项目调研 | 已含 Isaac 专项项目的实测数据；2026-09-20 新增 SC2/Dota/SoulsGym/EnvPool/Isaac-RL 的引擎适配做法调研（逐条附 URL）与本项目的训练场差距清单和实施顺序 | [RELATED_WORK.md](docs/RELATED_WORK.md)、[训练场设计](docs/TRAINING_ARENA_DESIGN.md)、[调研笔记](docs/ARENA_RESEARCH_NOTES.md) |
| 旧 v0 原型 | 已否定；代码与自测记录保留 | [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)、`isaac_room/` |

已有真实训练接入结果，但**尚无原版单房间能力达标证据**。历史原版512步探针用时74.27秒，参数L2变化0.6133；4个结束回合全部死亡。另3局规则控制器获得24/23/25个正奖励步骤，不是学习成绩。原始日志/模型保留在被Git排除的 `runs/l1/20260920-rewind/`；可复现入口、奖励语义和结果摘要维护在代码与文档中。旧 Rust 约95万帧/秒只是简化子集基准，不能与完整原引擎训练吞吐等同。

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

## 当前训练更新（2026-09-23）

**当前实验：** `/home/eolc/isaac-rl/runs/monstro-128k-seg1024-20260923`，21:29从旧combat-v1实验的第425轮精确续训。
- 优化：batch 1024×2 epochs（每轮32次优化步），片段minibatch帧去重，微批256梯度累积，ent_coef 0.01。
- 训练开局随机化：50%训练局Boss以10%–100%血量开局，25%训练局玩家以3–5个半心开局；评估始终满血。
- 每50轮存档并做16种子评估；每轮约8秒，旧配置约61秒。
- 配置、代码改动与起点选择见[§0.15](docs/ENV_ARCHITECTURE.md#015-更新量帧去重与训练开局随机化2026-09-23已上线)。

**旧实验：** `monstro-128k-reward-v1-20260923`，batch32×4epochs，每轮2048次Adam步。审查确认其实现无误，但每轮更新过大（精确KL约0.14，约22%状态的确定性动作被改写）；在256个未见种子上从第425轮的19%退到第750轮的7%，回报同样下降，不是reward hacking。该实验在第1150轮停止，目录保留。证据见[§0.14](docs/ENV_ARCHITECTURE.md#014-当前阶段与-claude-只读审查交接2026-09-23)与`runs/ppo-audit/20260923-e1e2/`。

## 下一步

0. **（待决定）修模拟器的 Monstro 出场阶段。** `new_monstro` 缺 `FLAG_APPEAR`；J460 录像和 AB+ 里，Monstro 开局约 30 帧不行动。修复后用 256 个种子重评 u425 和新实验的检查点，确认胜率是否依赖开局炸弹。证据见 [bridge/abplus/EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md#a2-u425-迁移评估2026-09-24) 的 A2。
1. 观察新配置：价值拟合是否恢复（EV/价值损失）、满血开局训练局胜率，以及256种子验证集（2147483728–2147483983）上的满血确定性/随机胜率。训练曲线须按每局`start`字段分开统计。
2. 若满血表现停滞，再单独调整一项：开局随机化比例、熵系数或清房奖励权重。帧编码去重之后，时序Transformer成为更新的主要耗时。
3. 稳定单房间学习后，用原版留出场景做迁移验收；2147500000起的种子保留为最终测试。真人录制/BC数据保留，暂不继续模仿学习。

## 复用边界

- [分析区](../analysis/README.md)：帧循环、ZHL 命名表、资源索引等共用游戏事实。
- [NetFix](../netfix/README.md)：注入器、MinHook 接线、J460 地址档案与启动/日志经验，供 L1 复用；不以其未完成的任意帧恢复作为前置条件。
- [第三方区](../third_party/README.md)：REPENTOGON 只作 ABI/特征码参考，不能在 J460 上运行。

## 历史原型（已否定，仅记录）

`isaac_room/` 是 2026-09-19 之前写的通用射击原型：固定 60 Hz、自编敌人、几何绘图，没有使用任何原版资源或机制。它可以运行（`play.ps1`、`python -m isaac_room.benchmark`），37 项自测和 24/24 脚本清房只验证自编规则，不构成以撒仿真的验收。细节见 [SINGLE_ROOM_V0.md](docs/SINGLE_ROOM_V0.md)。

## Git 管理

`Isaac/rl` 为独立本地Git仓库，与 `Isaac/netfix` 分开；本次未配置远程或推送。跟踪桥接/训练/原生控制层代码、测试、观测fixture、设计文档，以及已明确标为历史的模拟器代码。`runs/`、模型/转储/日志、依赖目录、Rust target及原生build产物不入库。

本仓库仍依赖工作区的 `../third_party/minhook`、`../../.tools` 工具链及 `../analysis` 机制材料；首次提交不表示已成为可单独克隆构建的发行包。先保留依赖关系，不复制第三方或游戏资源进本仓库。
