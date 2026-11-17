# AIsaac

用强化学习玩《以撒的结合》的研究仓库。训练直接在原版引擎上进行：用的是 Afterbirth+ v1.06 的原生 Linux 版，经一个 `LD_PRELOAD` 加速层改成无渲染、虚拟时钟、可 `fork` 克隆的环境。一台机器并行跑几十个游戏进程，GPU 上的 PPO 训练一个只看可见信息、打整局的策略。目标是一个能打完整局、最终超过人类玩家的 AI；现在离这个目标还很远。

## 现状（2026-11-17）

- **当前主线**：单序列因果 transformer "seq-v1"，输入用兼容 Repentance 的新格式 visible-v3。设计已批准（[docs/SINGLE_TRANSFORMER_DESIGN.md](docs/SINGLE_TRANSFORMER_DESIGN.md)、[docs/VISIBLE_V3_DESIGN.md](docs/VISIBLE_V3_DESIGN.md)），代码还没写。
- **参考基线**："旧模型"（逐帧 token transformer + 时间 transformer）。已核对的最好成绩：
  - C99 从零训练 58,506 游戏小时，在 512 个留出种子的整局评估里平均过 **1.953** 层（±0.035），29% 的局到第四层以上；
  - C100 把模型放大约 1.5 倍，25,005 游戏小时时 512 种子平均过 1.006 层；C99 在同样训练量（25,379 小时）时是 1.19；
  - 以上见 [EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md) C99、C100。
- **搁置**：block-memory v1（四 latent 分块记忆）。在 400 游戏小时的对照里，以"清房 ≥ 1 的局所占比例"为指标：
  - 旧模型 0.483–0.506（C110）；
  - block-memory 最好的一臂 0.253，阈值是 0.35（C113 rt-hf）。

  11-17 起不再投入，代码保留。
- **暂停或已结束**：Rust 模拟器、忏悔+ J460 桥与加速层、宏观规划工具、目标条件选项、Go-Explore、对战场、memory-chain v2 / v3。见文末"历史"。

## 怎么工作

结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。要点：

- **引擎与加速层**（`bridge/engine/abp_turbo.c`）：
  - 虚拟时钟、跳过渲染、随机数重播；
  - 进程级 `fork`：克隆体与父进程逐帧相同（A19）；
  - 观测和步进的热路径在 C 里。
- **桥**（`bridge/abplus/abp_bridge.lua`）：游戏内的 Lua，在 `127.0.0.1` 上接收 JSON 命令（重置、step、fork 等），回二进制观测。
- **worker**：每个游戏实例一个进程，不导入 torch，每局是一个克隆体。
  - 每个决策（4 个逻辑帧）写一条记录到共享内存；
  - 动作晚一个决策生效。
- **训练进程**（`bridge/python/train_tok.py`）：
  - actor 批量推理（旧模型用 CUDA Graph，block 一套用 C++ driver）；
  - learner 做 PPO，可以放到另一块卡上、与采样重叠；
  - 多个分片同步梯度，组成一次训练。
- **教师**：用克隆体回到受伤前或选择点，试别的走法，把更好的分支作为训练目标。共有三种：受伤回溯（C56）、教师 v2（B19）、反事实分支（B34）。克隆体知道未来，所以只用来产生训练目标，评估时不用。
- **观测边界**：只有玩家看得到的信息进 actor；特权信息（例如真实楼层布局）只进 learner 的价值头。见 [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)。

## 仓库地图

| 目录 | 内容 | 状态 |
|---|---|---|
| `bridge/engine/` | 加速层 `abp_turbo.c`、实例启动脚本、C / C++ 原生组件 | 在用 |
| `bridge/abplus/` | AB+ 的桥 `abp_bridge.lua`、房间目录、实验记录 `EXPERIMENTS.md` | 在用 |
| `bridge/python/` | 训练 `train_tok.py`、评估 `eval_tok.py`、`isaac_bridge/` 包、探针和基准脚本、测试 | 在用；其中 `train_abplus.py` 一套是 9 月的旧管线 |
| `bridge/ops/bm1/` | 多卡主机上的运行环境：容器、锁、计时、测试门、正式训练守护 | 在用 |
| `bridge/ops/` 其余 | 过去实验的启动和测量脚本 | 历史记录 |
| `docs/` | 项目说明、系统结构、设计文档；`docs/archive/` 是归档 | 在用 |
| `macro/` | AB+ 楼层生成器（与引擎逐房间一致）、隐藏房推断、规划工具、网页小测试 | 已完成 |
| `bridge/README.md`、`bridge/mod/isaac_rl_bridge/`、`turbo/` | 忏悔+ J460（Windows）的桥和加速层 | 暂停 |
| `bridge/mod/isaac_human_recorder/` | 人类录制 mod 的旧副本；录制器已移到单独的仓库 isaac-run-recorder | 历史遗留 |
| `sim/` | Rust 单房间模拟器 | 9 月起搁置 |
| `isaac_room/`、`play.ps1`、`tests/` | 最早的自编规则原型和它的测试 | 已否定 |
| `code-tests/` | 一个采样器读写竞争问题的复现脚本 | 历史 |

## 上手

需要准备：
- 一台 Linux 机器，带 NVIDIA GPU（驱动支持 CUDA 12.8），以及 gcc / g++、xvfb、Python 3.10。
- 自己的一份《以撒的结合：胎衣†》v1.06 Linux 版游戏文件。**游戏文件不在本仓库里。**
  - 不要改动 Steam 安装目录，复制一份作为 runtime。
  - runtime 的 `resources/scripts/main.lua` 要能用 `dofile(ABP_LUA)` 加载桥。做法写在工作区的逆向文档里，那份文档不在本仓库。

步骤：
1. **编加速层**：`gcc -shared -fPIC -O2 -o $ABP_HOME/tools/libabp_turbo.so bridge/engine/abp_turbo.c -ldl`。
   - `ABP_HOME` 默认是 `~/isaac-abplus`；其中 `runtime/` 放游戏副本，`tools/` 还要放 `run_instance.sh` 和打桩表（见 [bridge/engine/README.md](bridge/engine/README.md)）。
   - 用 bm1 时，`deploy.sh abphome` 会做这一步。
2. **装依赖**：torch 2.7.1（cu128），再加上 `bridge/ops/bm1/env/requirements-bm1.txt`。
3. **测试与冒烟**：在 bm1 里先跑 `gate.sh <树>`（全量测试），再跑 `UPDATES=2 ./prof.sh <tag> <树>`（两个更新的训练冒烟）。见 [bridge/ops/bm1/README.md](bridge/ops/bm1/README.md)。
4. **训练与评估**：入口是 `bridge/python/train_tok.py` 和 `eval_tok.py`。
   - 模式、奖励和开关见 [bridge/abplus/README.md](bridge/abplus/README.md)；
   - 正式训练用 `train_guard.sh`，手册模板见 [bridge/ops/bm1/TRAINING.md](bridge/ops/bm1/TRAINING.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | 目标、观测边界、立场、当前阶段、不做的事 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统结构：每层的代码位置和对应实验 |
| [docs/SINGLE_TRANSFORMER_DESIGN.md](docs/SINGLE_TRANSFORMER_DESIGN.md)、[docs/VISIBLE_V3_DESIGN.md](docs/VISIBLE_V3_DESIGN.md) | 当前主线的模型设计和输入格式设计 |
| [bridge/abplus/EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md) | 全部实验 |
| [bridge/abplus/README.md](bridge/abplus/README.md)、[bridge/ops/bm1/README.md](bridge/ops/bm1/README.md) | 训练入口的用法；运行环境 |
| 其余 `docs/*.md` | 仍有效的专题设计：时间注意力、反事实教师、C++ actor、预测记忆、宏观规划、相关工作 |
| [docs/archive/README.md](docs/archive/README.md) | 归档文档的索引 |

## 实验记录

所有实验只记在 [bridge/abplus/EXPERIMENTS.md](bridge/abplus/EXPERIMENTS.md)：
- 每条分"实验目的 / 实验方法 / 实验结果"三段，开头有总览表；
- 编号：A 桥与环境，B 性能与稳定性，C 奖励与训练，各自按时间递增；
- 训练量用游戏小时（逻辑帧 ÷ 30 ÷ 3600）表示，速度用相对实时的倍率；
- 其他文档只引用编号，不抄数字。

## 历史

| 线 | 时间（2026） | 状态 | 文档 |
|---|---|---|---|
| 自编规则原型 `isaac_room/` | 09-19 前 | 已否定 | [SINGLE_ROOM_V0.md](docs/archive/sim-era/SINGLE_ROOM_V0.md) |
| 忏悔+ J460 原版桥与加速层 | 09-19 – 09-22 | 暂停 | [bridge/README.md](bridge/README.md)、[turbo/README.md](turbo/README.md)、[归档索引](docs/archive/README.md) |
| Rust 模拟器 + GPU PPO | 09-21 – 09-24 | 搁置：迁移到原版后胜率从 19.1% 掉到 0.4%（A2） | [ENV_ARCHITECTURE.md](docs/archive/sim-era/ENV_ARCHITECTURE.md) |
| AB+ 房间战斗（`train_abplus.py`） | 09-24 – 10-03 | 被 token 策略取代 | [abplus-README-2026-10.md](docs/archive/abplus-README-2026-10.md)、[COMBAT_RL_EXPERT_REVIEW.md](docs/archive/2026-09-combat/COMBAT_RL_EXPERT_REVIEW.md) |
| 目标条件选项、碰撞护盾、Go-Explore、对战场 | 09-28 – 10-03 | 结束 | [GOAL_CONDITIONED_DESIGN.md](docs/archive/goal-conditioned/GOAL_CONDITIONED_DESIGN.md)；A6–A8、A17、C38–C54 |
| 宏观规划与楼层生成器 | 09-25 起 | 已完成 | [macro/README.md](macro/README.md)、[MACRO_PLANNING.md](docs/MACRO_PLANNING.md) |
| token 策略 + fork 训练器（旧模型） | 10-03 – 10-19 | 参考基线 | [TEMPORAL_ATTENTION.md](docs/TEMPORAL_ATTENTION.md)、[COUNTERFACTUAL_TEACHER.md](docs/COUNTERFACTUAL_TEACHER.md)；C55–C104 |
| memory-chain v2 / v3 | 10 月下旬 – 11-06 | 被 block-memory 取代 | [PROJECT_SPEC.block-memory-v1.md](docs/archive/block-memory-v1/PROJECT_SPEC.block-memory-v1.md) §9.9–9.11 |
| block-memory v1 + C++ actor | 11-06 – 11-17 | 搁置 | [PROJECT_SPEC.block-memory-v1.md](docs/archive/block-memory-v1/PROJECT_SPEC.block-memory-v1.md)、[B50_CPP_ACTOR_DESIGN.md](docs/B50_CPP_ACTOR_DESIGN.md)；B41–B62、C105–C113 |
| seq-v1 on visible-v3 | 11-17 起 | 主线，设计阶段 | [SINGLE_TRANSFORMER_DESIGN.md](docs/SINGLE_TRANSFORMER_DESIGN.md) |

游戏的图像、音频和程序文件都不包含在本仓库里。
