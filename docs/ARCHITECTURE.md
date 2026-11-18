# AIsaac 系统结构

更新：2026-11-17。本文是一张地图：每一层是什么、代码在哪、在哪个实验里引入或测量（编号指 [bridge/abplus/EXPERIMENTS.md](../bridge/abplus/EXPERIMENTS.md)）。目标和原则见 [PROJECT_SPEC.md](PROJECT_SPEC.md)。路径都相对仓库根。

## 1. 总图

```text
游戏进程 × N（AB+ v1.06 Linux 原版，LD_PRELOAD libabp_turbo.so）
  ├─ abp_turbo.c：虚拟时钟、不渲染、fork 克隆、原生观测 / 原生 step / 行编码
  └─ abp_bridge.lua：TCP（127.0.0.1）上的命令协议，重置、step、精简观测
        │ 精简观测（lean）
        ▼
worker 进程 × N（不导入 torch；tok_sampler / tok_floor）
  root 实例只做重置 → 停着的 template 克隆 → 每局一个克隆体
  每个决策：解码 → 写一条记录（ROW；block 一套另写 visible-v2 帧）到共享内存 → 等动作
        │ 共享内存 + 管道
        ▼
训练进程（train_tok.py，持有 GPU）
  actor：旧模型 = Python 主线程 + GraphActor；block 一套 = C++ ActorDriver
  lane 服务：教师分支的推理请求（受伤回溯 / 教师 v2 / 反事实）
  learner：PPO + 教师损失 + 辅助损失（可放第二块卡、可多分片同步梯度）
        │ 检查点
        ▼
eval_tok.py：留出种子评估（128 / 512 个种子）
```

动作晚一个决策生效：worker 发出第 t 条记录后，用第 t−1 条的回答走下一步，游戏在推理期间照常前进（`isaac_bridge/tok_sampler.py` 文档串，C55）。每决策 4 个逻辑帧。

## 2. 引擎与加速层（`bridge/engine/`）

| 文件 | 作用 |
|---|---|
| `abp_turbo.c` | `LD_PRELOAD` 进游戏的控制层，不改磁盘上的文件。虚拟时钟（`ABP_TURBO`）、渲染跳过 / 空 GL（`ABP_RENDER_EVERY`、`ABP_NULLGL`）、重播随机数（`ABP_RESEED`、`ABP_STARTSEED`）、音源状态（`ABP_AL_STOPPED`）、进程克隆（`ABP_FORK`）、原生输入（`ABP_INPUT`）、原生精简观测和 ROW 编码（`abp_row_encode`） |
| `stub_render_h.txt`、`stub_render_g.txt`、`null_render_e.txt` | 打桩表：渲染路径里可以跳过的函数（H 是 worker 的默认，B14、B15） |
| `run_instance.sh` | 从 `$HOME/isaac-abplus/runtime`（自备的游戏文件副本）起一个隔离实例：独立存档目录、关 Steam 云、关 mod、静音 |
| `abp_visible.c` | visible-v2 帧的 C 实现，worker 用（B45） |
| `abp_actor.cpp`、`abp_core.h`、`abp_lanes.h`、`abp_lineage.h` | block 一套的 C++ actor：页存储、driver 线程、lane 服务、谱系（B52、B54、B58a–e） |
| `abp_learn.c` | block learner 的主机侧打包，在 GIL 之外（B60） |
| `abp_schema_v3.h` | visible-v3 常量，由 `bridge/python/tools/gen_schema_v3_h.py` 从 `schema_v3.py` 生成 |
| `core_tests/` | C++ 核心的单元测试 |

引擎层的实测：
- fork 克隆体与父进程逐帧一致（A19）；
- 精简模式不改变游戏（B9）；
- 原生观测（B11）、step 行和原生 step（B12）；
- 一帧的成本（B14）。

逆向分析材料在工作区的 `analysis/`，不在本仓库。

## 3. 桥与步协议（`bridge/abplus/abp_bridge.lua`）

- 版本 abp-0.2.19；Python 端 `isaac_bridge/abplus.py` 的 `BRIDGE_VERSION` 必须相同。
- 游戏以 `--luadebug` 启动，桥在 `127.0.0.1` 上监听，每行一个 JSON 命令：`step`、`reset`、`lua`、`obs`、`fork`、`lean`、`play`、`priv` 等。协议写在文件头的注释里。
- 致命伤在 `MC_ENTITY_TAKE_DMG` 里拦下，记为死亡（AB+ 没有 rewind）。
- 精简模式（`lean`，abp-0.2.14）：观测是定长二进制记录，由 `isaac_bridge/abplus_lean.py` 解码。热路径上，记录的固定部分和地形检查由 `abp_turbo.c` 在 C 里填（B11）。
- `fork`：把当前游戏进程克隆一份，克隆体连到新的端口。worker 的每局、教师的每个分支都是克隆体（A19、B10）。
- `play`：一串动作批量重放，用于从恢复点回到受伤前的状态（A8、A22）。
- `priv`（abp-0.2.19）：只给 learner 的特权帧，不改精简观测（`isaac_bridge/tok_priv.py` 文档串）。

重置：`reset_mode = 'floor'` 开真实楼层（跳过迷宫、迷失、迷路诅咒，A9、A20）。`train_tok.py --characters` 选开局角色（A25）。

## 4. 观测格式

| 格式 | 用在 | 定义 | 引入 |
|---|---|---|---|
| ROW（`tok_obs.ROW`，约 20 KB 一条） | 旧模型的输入；所有模型的奖励、事件位、辅助目标 | `isaac_bridge/tok_obs.py`；C 端 `abp_row_encode` 逐字节相同 | C55；道具 A21；激光 / 标志 A24；观测 v2（价格、心、房间类型）B35 |
| visible-v2 帧（变长，上限 131,072 字节，schema `isaac-visible-prlmens-g-v2`） | block-memory v1 的输入 | 常量 `isaac_bridge/schema_v2.py`；生产 `visible_producer.py` / `visible_native.py`（C：`abp_visible.c`）；解码 `structured_input.py` | B41、B45 |
| visible-v3（schema `isaac-visible-prlmens-g-v3`） | 计划中，seq-v1 的输入 | 冻结的接口文件 `isaac_bridge/schema_v3.py`、`bridge/engine/abp_schema_v3.h`；设计 [VISIBLE_V3_DESIGN.md](VISIBLE_V3_DESIGN.md) | 只有接口；生产端 11-17 开工，模型端还没写 |

坐标的注意事项（[SINGLE_TRANSFORMER_DESIGN.md](SINGLE_TRANSFORMER_DESIGN.md) 第 9 节，2026-11-17 的审核）：
- visible-v2 有已知的坐标问题，不要照它的做法写新代码：实体向量被缩放两次（生产端一次，`structured_input.py` 里再整体除以 64）；x 除以 520 而 y 除以 280；位置以 `GetTopLeftPos` 为原点，在窄房间里和地形画布差 2 行或 4 列；激光端点是世界坐标。
- visible-v3 按坐标契约做：原点是 0 号格的左上角，两轴同一缩放，缩放只在解码处做一次，数值列统一截断，相对玩家的位移由模型在解码时算。AB+ 引擎上 12 种房型的实测数据见同一节。

ROW 是每个决策一条定长记录。它包含：
- 玩家；
- 最多 64 个实体，取最近的；
- 门、地形、局部地形、小地图；
- 这一步的奖励项和结局，这部分不是策略输入。

visible 帧按对象组织：每个身体一个 P、实体 E（上限 128，超出的按优先级保留）、库存 N、槽位 S、房间画布、小地图。字段的取舍见 [archive/block-memory-v1/PROJECT_SPEC.block-memory-v1.md](archive/block-memory-v1/PROJECT_SPEC.block-memory-v1.md) §9.1–9.7。

## 5. worker 与采样器

- `isaac_bridge/tok_sampler.py`：单房模式和 worker 主循环。root 实例只重置，template 停在起始状态，每局从它克隆（fork 采样器，B10）。worker 不导入 torch，记录写进共享内存。
- `isaac_bridge/tok_floor.py`：整层模式和整局模式。
  - 整层（`--mode floor`，C57）：一局是一整层 Basement I。
  - 整局（`--mode run`，A21、C67）：一局一直打到死亡或时限。
  - 起点库：楼层中途存下的克隆体，用作开局起点。
  - 无进展规则：`--floor-stall-seconds`、`--stall-progress`。
- `isaac_bridge/fork_sampler.py`、`isaac_bridge/abplus_lean.py`：克隆和精简观测的底层。
- 实例默认用 H 打桩表、`ABP_FAST=3` 等设置（`tok_sampler.INSTANCE_ENV_DEFAULTS`，B15）。内存增长和 root 卡死的处理见 B16。

## 6. actor

**旧模型**：
- `train_tok.py` 的主线程就是 actor：对已就绪的记录批量采样动作。
- 推理用 `isaac_bridge/tok_policy.py` 的 `GraphActor`：固定形状的 CUDA Graph，采样用 Gumbel-max（C55、B29）。
- 时间 transformer 的状态由 actor 按 worker 保存（`TemporalState`，B30、B31）。
- `--actor-processes 2` 可以加第二个 actor 进程（B38 第五步）。

**block-memory v1**：
- C++ `ActorDriver` 在自己的线程里轮询 worker、回放 step graph、写页存储。Python 只消费它产出的记录。
- 代码：`bridge/engine/abp_actor.cpp`；Python 胶水 `isaac_bridge/actor_native.py`、`block_driver.py`。
- 实验：B52、B54、B58a–e。
- 编不出扩展时，退回 Python 的 `BlockActor`（`block_training.py`）。

**lane 服务**：教师分支的推理请求分批、异步地处理。
- 旧模型：`isaac_bridge/tok_lanes.py`（B36）。
- block 一套：`block_lanes.py` 和 C++ 的 `abp_lanes.h`（B54、B58d）。

## 7. learner

- **算法**：PPO，按 worker 序列算 GAE，`--epochs` 轮、裁剪目标（`train_tok.py` 的 `learn()`）。教师和辅助损失并进同一个小批。
- **与采样重叠**：`--overlap` 让 learner 在 actor 继续采样时更新，策略最多滞后一次更新；`--decoupled 1` 把裁剪中心放在更新开始时的权重上（B11、B13）。
- **快路径**：`--learner-fast 1|2`（B18、B38 第四步）。
- **learner 位置**：
  - 放到第二块卡：`--learner-device`（B38 第一步）；
  - 放进独立进程：`--learner-process 1`（B38 第二步）。
- **block 一套的 learner**：
  - 按 part 打包执行：`isaac_bridge/block_learner.py`（B44）；
  - 主机侧打包移到 C、在 GIL 之外：`learn_native.py` + `abp_learn.c`（B60）；
  - 一个 learner 用两块卡：`learner_p2.py`，`--learner-devices`（BM1 P2：受主机发射限制，没有更快，不进生产）。
- **多分片**：`isaac_bridge/shard_sync.py`。设 `ISAAC_SHARD_WORLD = K` 时，K 个训练进程各自采样、各自算梯度，每个优化步按小批大小加权平均梯度，参数保持逐位相同。实验见 EXPERIMENTS 的 "BM1 C：跨分片梯度同步"，C105 起用于正式训练。
- **辅助损失与特权 critic**（[PREDICTIVE_MEMORY_DESIGN.md](PREDICTIVE_MEMORY_DESIGN.md)）：
  - `--aux-predict 1`：`isaac_bridge/tok_aux.py`；
  - `--priv-critic 1`：`isaac_bridge/tok_priv.py`，特权数据只进价值头；
  - C109 的 noaux 臂把两者都关掉做过对照。

## 8. 教师

所有教师都用克隆体和共用的随机数，因此知道未来。它们只产生训练目标，评估时不用。

| 教师 | 开关 | 代码 | 实验 |
|---|---|---|---|
| 受伤回溯（事后搜索）：受伤前分出 9 个移动，标出哪些安全 | `--teacher`；死亡加深 `--teacher-death-depths` | `tok_sampler.hindsight` | C56 |
| 教师 v2：整条分支蒸馏 | `--teacher-v2 1` | `isaac_bridge/tok_teacher2.py` | B19、C83、C84 |
| 反事实分支教师：在选择点分 K 支，用胜出分支监督 | `--cf 1` | `isaac_bridge/tok_cf.py`；设计 [COUNTERFACTUAL_TEACHER.md](COUNTERFACTUAL_TEACHER.md) | B34、B36、B37、C99 |

## 9. 模型

| 模型 | 代码 | 状态 |
|---|---|---|
| 旧模型：每帧一个空间 transformer（玩家、房间、小地图、门、实体各是 token），再接时间 transformer（窗口 256 个决策，加房间 token） | `isaac_bridge/tok_policy.py` 的 `TokPolicy`；时间侧见 [TEMPORAL_ATTENTION.md](TEMPORAL_ATTENTION.md) | 参考基线。从零长训见 C99；放大 1.5 倍见 C100；400 游戏小时能学会真实任务见 C107、C110 |
| block-memory v1：四个 latent 读出 + 32 步一块的独立记忆块 + 时间 transformer | `isaac_bridge/block_memory.py`、`block_training.py`、`block_*.py`；`train_tok.py --block-memory 1` | 2026-11-17 起不再投入。C107–C113 的 400 小时对照里学不会真实任务。代码保留 |
| seq-v1：每个决策一个 token 的单序列因果 transformer，输入用 visible-v3 | 还没有代码；设计 [SINGLE_TRANSFORMER_DESIGN.md](SINGLE_TRANSFORMER_DESIGN.md) | 当前主线，设计阶段 |

动作头：移动 9、射击 5、炸弹 2，开道具时再加主动道具 2、药丸 / 卡牌 2（`tok_policy.HEADS`、`ITEM_HEADS`）。

奖励：
- 默认值写在 `train_tok.py` 的 `ROOM_REWARD`、`FLOOR_REWARD` 里；
- `--reward name=weight,...` 可以覆盖，例如 `heal`、`resource`（只用于整层 / 整局模式）。

## 10. 评估

- `bridge/python/eval_tok.py`：每个种子一局，满血，默认在 CPU 上推理。
  - `--mode room|floor|run`；
  - 默认种子 `2147490000:128`，训练时不会抽到；
  - `--greedy` 取概率最大的动作，默认按策略采样。
- 128 个种子用于训练中的曲线；定论用 512 个种子（A27；C99、C100 的定论数字都是 512 个种子）。单点噪声的量级见 C61、C99 条目。
- 只用于评估的消融开关：`--ablate`、`--no-pedestals` 等（A28）。
- block 检查点也由 `eval_tok.py` 评估（`build_eval_model` 检查 `block_memory` 配置）。容器里的评估作业脚本例：`bridge/ops/bm1/ev_ab2.sh`（C107）。

## 11. 运维（`bridge/ops/bm1/`）

- 入口：
  - `run.sh <用途> <GPU 列表> <命令>`：在本道的 CPU / 内存节点 / GPU 上运行一条命令，Docker 或直接运行；
  - `train_guard.sh`：K 分片的正式训练，失败后自动续训；
  - `shardk.sh`：分片计时；
  - `prof.sh`：计时运行；
  - `gate.sh`：全量测试门；
  - `deploy.sh`：部署 `git archive` 树、编译 `libabp_turbo.so`。
- 主机相关的设置只放在 `bm1.env` 里，从 `env/*.env` 复制；新机器先跑 `probe_host.sh`。锁、前置检查和删进回收站在 `common.sh`。
- 用法见 [bridge/ops/bm1/README.md](../bridge/ops/bm1/README.md)。正式训练的操作手册模板是 [bridge/ops/bm1/TRAINING.md](../bridge/ops/bm1/TRAINING.md)（C105 main1）。
- `bridge/ops/` 这一层只留下仍被 bm1、测试或代码引用的 26 个旧脚本（清单见 [bridge/ops/README.md](../bridge/ops/README.md)）。其余 126 个过去实验的一次性启动器和测量脚本在 [bridge/ops/archive/](../bridge/ops/archive/README.md)。

## 12. 不在主线上的代码

| 位置 | 是什么 |
|---|---|
| `bridge/python/train_abplus.py`、`abplus_vec.py`、`abplus_worker.py`、`gpu_ppo.py`、`abplus_eval.py` | 9 月的房间战斗管线（C1–C54），已被 `train_tok.py` 代替；说明在 [archive/abplus-README-2026-10.md](archive/abplus-README-2026-10.md) |
| `bridge/python/goexplore_*.py`、`isaac_bridge/abplus_duel.py`、`abplus_options.py`、`abplus_nav.py`、`abplus_shield.py` | Go-Explore（A7、A8）、对战场（A6、C38、C40）、目标条件选项（C42–C45）、碰撞护盾（A17） |
| `bridge/README.md`、`bridge/mod/isaac_rl_bridge/`、`turbo/` | 忏悔+ J460 的桥和加速层，暂停 |
| `archive/human-recorder-2026-09/` | 9 月的人类录制 mod 和行为克隆一线，已归档（[archive/README.md](../archive/README.md)）；后来重写的录制器 11-10 起是单独的仓库 isaac-run-recorder |
| `sim/`、`isaac_room/` | Rust 模拟器（9 月起搁置）、已否定的原型 |
| `macro/` | AB+ 楼层生成器、隐藏房推断、规划工具（离线，[MACRO_PLANNING.md](MACRO_PLANNING.md)） |
