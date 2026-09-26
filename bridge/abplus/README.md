# AB+ 桥接与训练管线（2026-09-24）

目的：让模拟器训练出的策略直接在 Afterbirth+ v1.06 原生 Linux 版（原版引擎，经 `abp_turbo` 改造加速）上评估，将来也能在上面训练。协议和观测格式沿用 J460 的 IsaacRLBridge 0.2.0（combat_schema=3），所以 `VisibleHistory` 和策略网络不用改。

引擎层（实例隔离、虚拟时钟、严格等价的 render-lite 模式）见 [ABP_LINUX_REVERSE_ENGINEERING.md](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)。本桥接查明的引擎行为记在该文档 §15。

本文只写管线的设计、奖励定义和用法。所有实验（目的、方法、结果）单独记在 [EXPERIMENTS.md](EXPERIMENTS.md)，下文用编号（如 A2、C12）引用。

## 组成

| 文件 | 作用 |
|---|---|
| [`abp_bridge.lua`](abp_bridge.lua) | 游戏侧（现为 abp-0.2.3，各版本的改动见下文）。由 runtime 副本 `main.lua` 的钩子在 `--luadebug` 下 `dofile(ABP_LUA)` 加载，不走 mods 目录。协议比 0.2.0 多 `{"cmd":"lua"}`、`format`（二进制观测 v2）和 `profile`（观测各部分耗时）；致命伤在 `MC_ENTITY_TAKE_DMG` 里拦下并记为死亡（AB+ 没有 rewind，死了的玩家救不回来） |
| [`isaac_bridge/abplus.py`](../python/isaac_bridge/abplus.py) | `AbplusTrainingEnv`（竞技场重置）、`AbplusTransformerEnv`（Gymnasium 环境，deadline 观测）、`launch_abplus`/`stop_abplus`（起停实例）、`sim_arena`（逐位复刻 `rl/sim/src/arena.rs`） |
| [`abplus_eval.py`](../python/abplus_eval.py) | 多实例评估：每个 worker 进程带一个 AB+ 实例和一份策略，按 `FrameSampler` 的方式增量推理（每帧只编码一次，时序 Transformer 在缓存窗口上重算）；指标与模拟器审计 `e2_reeval.py` 相同 |
| [`abplus_transfer_report.py`](../python/abplus_transfer_report.py) | 同种子配对比较模拟器与 AB+ 的结局、行为指标和 Monstro 动画时长 |
| [`abplus_smoke.py`](../python/abplus_smoke.py) | 单实例随机动作冒烟测试和耗时拆分 |
| [`abp_turbo.c`](../../../analysis/scripts/abplus/abp_turbo.c) | 引擎改造层；本轮新增 `getenv("ABP_RESEED:<n>")` 钩子，重播全局 MT19937 |
| [`train_abplus.py`](../python/train_abplus.py)、[`isaac_bridge/abplus_vec.py`](../python/isaac_bridge/abplus_vec.py)、[`isaac_bridge/abplus_worker.py`](../python/isaac_bridge/abplus_worker.py)、[`train_abplus.sh`](train_abplus.sh) | 训练管线：独占 GPU 的学习器、每个环境一个 worker 进程（两个 AB+ 实例轮换）、共享内存帧、启动脚本，见"训练管线" |
| [`isaac_bridge/abplus_obs.py`](../python/isaac_bridge/abplus_obs.py) | 桥接 v2 二进制观测的解码器；验证脚本 `abplus_probe_obs_v2.py` |
| [`isaac_bridge/abplus_tasks.py`](../python/isaac_bridge/abplus_tasks.py)、[`catalog/`](catalog/) | 房间级混合：从运行中的引擎探测的房间目录、Basement I Boss 池、混合规格；探测脚本 `abplus_room_catalog.py`、`abplus_probe_boss_pool.py`、`abplus_probe_rooms.py` |
| [`isaac_bridge/abplus_reward.py`](../python/isaac_bridge/abplus_reward.py)、[`isaac_bridge/abplus_geometry.py`](../python/isaac_bridge/abplus_geometry.py) | 奖励 combat-v2 到 combat-v5；combat-v5 的射击几何（d_fire、射击方向标签、趋近标签） |
| [`isaac_bridge/plr.py`](../python/isaac_bridge/plr.py) | 按房间类别的 PLR 选房 |
| [`abplus_replay_view.py`](../python/abplus_replay_view.py)、[`isaac_bridge/abplus_replay.html`](../python/isaac_bridge/abplus_replay.html) | 把 `abplus_eval.py --replays` 的回放做成一个本地可看的网页 |
| `abplus_bench_workers.py`、`abplus_bench_train_cpu.py`、`abplus_probe_obs_cost.py` | 环境侧吞吐上限、CPU 上的 PPO 训练开销、Lua 观测各部分耗时 |
| `abplus_probe_*.py`、`abplus_bench_policy.py` | 诊断脚本，各自起一个实例：`door`（LeaveDoor 与门）、`goto`（goto 后逐帧生成了什么）、`champion`（冠军在何时出现）、`roomseed`（房间种子与 MT19937 的关系）、`determinism`（固定动作逐帧比对）、`reset`（重开后的楼层、诅咒、耗时）；`abplus_bench_policy.py` 测 CPU/GPU 推理延迟 |
| `test_abplus_*.py`、`test_plr.py`、`test_gpu_infra.py`、`test_graph_sampler.py`、`test_steam_watch.py` | 单元测试。远端 venv 没有 pytest，用 `python -m unittest` 运行 |

## 竞技场重置（每局约 0.1 秒）

与模拟器 `arena.rs` 同一个种子得到同样的布局、入口门、玩家出生点和 Monstro 出生点。步骤：

1. 用由种子导出的值重播全局 MT19937，执行 `restart 0`，重建玩家对象。
2. 设 `Level.LeaveDoor = (入口 + 2) % 4`，再重播一次，执行 `goto s.boss.<布局>`，等 8 帧。
3. 如果房间自带的 Monstro 是冠军（subtype ≠ 0），换下一个房间种子，回到第 2 步。
4. 执行一段 Lua：
   - 清掉所有实体，去掉道具、钥匙和硬币，回满血，炸弹设为 1；
   - 放好玩家位置；
   - 清掉诅咒，把特殊石头改回普通石头；
   - `FireDelay = 0`，清除受伤无敌帧；
   - 用种子重播，生成 Monstro，关门。
5. 逐帧推进，直到满足全部条件后才开局：
   - 唯一的可见敌人是 subtype 0 的 Monstro，位置与模拟器一致；
   - 只有入口门；
   - 石头集合与模拟器布局逐格一致；
   - 诅咒为 0。

每一步都对应一个实测问题，现象和静态依据见 EXPERIMENTS.md 的 A1。修好以后，同一种子、同一动作序列的所有观测字段在不同实例、不同时间都逐帧一致（实体 id 和帧计数除外）。

## 与模拟器的已知差异

- Monstro 的随机数流不同（AB+ 用 MT19937，模拟器用 xorshift），所以配对的是起始局面，不是轨迹。
- **出场阶段是模拟器的缺陷**。AB+ 的 Monstro 生成后隐身 4 帧，再播约 25 帧 `Appear`，期间不行动；J460 人类录像（`runs/human/20260923-session01`，6 局）开局也有约 31 帧不行动。模拟器的 `new_monstro` 把 state 设为 1，却没有设 `FLAG_APPEAR`（Gaper 设了），所以 AI 从第 1 帧就开始。策略利用了这一点，见 EXPERIMENTS.md 的 A2。
- **炸弹伤害是版本差异**。AB+ 的 `Entity_Bomb::Init` 把普通炸弹伤害设为 60.0，引信 45 帧；J460 录像里炸中 Monstro 一次是 100；模拟器用 100，与忏悔+ 一致。
- **`JumpDown` 时长也是版本差异**。AB+ 约 74 帧；J460 录像和模拟器都是 65 帧。其他动画时长基本一致。
- `range` 按 `260 × TearHeight / -23.75` 换算，对以撒是精确值。
- 致命伤由桥接拦截（虚拟死亡），结局判定与模拟器相同。

## 训练管线（2026-09-24）

用户决定：
- RL 只负责战斗，导航和道具选择先用规则；
- 采用房间级混合学习；
- PPO 在 GPU 上由独占进程完成，所有 worker 每步拼成一批，尽量减少通信损耗（依据见 B1）；
- 做桥接 v2；不开 TF32。

### 结构

```text
学习器进程（独占 CUDA）: GpuMaskablePPO + AbplusFrameVecEnv（abplus_vec.py）
  每步 1 次批量推理（--chunks 1 = 全部环境一批）→ 每个 worker 13 字节 → 等全部回应
  → 从共享内存拷回每个环境最新一帧 FRAME_DTYPE（72,664 B）→ pinned 槽 → 异步上传 GPU
worker 进程 × N（abplus_worker.py，不导入 torch）: 一个环境槽 = 两个 AB+ 实例
  活动实例打当前局；备用实例在后台线程里按下一局的种子重置 → 局间切换 0.02–0.04 ms
AB+ 实例 × 2N: 严格等价模式 + 桥接 v2 二进制观测
```

- 学习器端完全复用模拟器的 GPU PPO（帧级 GPU 缓存、帧去重分段 minibatch、checkpoint），`AbplusFrameVecEnv` 提供与 `GpuFrameVecEnv` 相同的 chunk 接口。
- worker 用 `VisibleHistory.encode` 只编码最新一帧，与原 `VisibleHistory.append` 的窗口末帧逐字段相同（核对见 A3）。
- 种子序列与 `FrameChunk` 相同：环境 i 的第 k 局用 `base_seed + i + N·k`。训练种子 < 2³¹；held-out 种子 ≥ 2³¹，其中 E2 验证块和 2147500000 起的最终测试块都不参与训练。
- 开局随机化与模拟器训练一致：50% 的局 Boss 血量为 10–100%，25% 的局玩家为 3–5 个半心。评估一律满血。
- 日志额外记录：
  - `game/hours_*`：游戏时，等于决策数 × 2 帧 ÷ 30 帧/秒；
  - `game/speed_x_realtime`：相对实时的倍率，等于决策/秒 ÷ 15；
  - worker 单步耗时、局间切换等待、错误数。

### 桥接 v2：二进制观测（abp-0.2.0）

实测读字段本身几乎不花时间，贵的是每步重发静态地形和纯 Lua 的 JSON 编码，两者合计约 75%（各部分耗时见 A3）。因此 v2 的做法是：
- 动态数据用 Lua 5.3 的 `string.pack` 打成二进制（每条记录 0.26 µs）；
- 地形和网格只在可能变化时发送：reset/query 事件、lua/exec/reset 命令之后、换房间、任一格的碰撞类型变化；
- 地形危险标志里动态的部分（带接触伤害的特效）由客户端按 Lua 的规则重算；
- 整数一律用 int64（重开 run 后玩家的 `Index` 是 0xFFFFFFFF）；
- 解码器（`abplus_obs.py`）输出与 JSON 路径完全相同的字典，下游代码不用改。

原生读内存推迟，因为实体字段读取只占 0.01–0.15 ms/步，收益不值得冒结构偏移的风险。

与 JSON 路径的一致性和提速见 A3（每步从 3.14 ms 降到 1.03 ms）。之后的桥接版本只在 combat 块和实体行加字段：abp-0.2.1 加 `blocking_hp`、`blocking_points`，abp-0.2.2 加 `blocking_count`，abp-0.2.3 加谱系相关的计数和两个实体标志（见下文各奖励一节）。客户端要求桥接版本完全一致。

### 房间级混合（`abplus_tasks.py`、`catalog/`）

AB+ 的房间数据直接从运行中的引擎取得，不用仓库里的 Repentance+ 房间文件（探测过程见 A4）：
- `abplus_room_catalog.py` 对 `goto d.<id>` 和 `goto s.boss.<id>` 逐个探测：Basement I 有 **1,022 个普通房**，其中 637 个是 1×1；特殊房表里有 **431 个 Boss 房**，覆盖全部楼层。
- `abplus_probe_boss_pool.py` 让关卡生成器重开 300 次，统计 Basement I 实际会出现的 Boss：共 13 种、82 个房间。
- `catalog/mixture_basement1.json` 的内容：
  - 普通房 495 个：1×1、开局未清房、至少有一个火堆以外的敌人；
  - Boss 房 77 个：上述 Boss 池里的 1×1 房间；
  - 默认权重：arena 0.2 / normal 0.45 / boss 0.35。
- `catalog/mixture_round1_monstro_horf.json`：combat-v5 第一、二轮用的子集，竞技场 0.25、13 个 Monstro Boss 房 0.25、42 个含 Horf 的普通房 0.5。
- 选房由种子决定：先按权重选任务，再选房间，再选入口。训练时可以改用 PLR 选房（见下文）。

普通房和 Boss 房的重置流程：
1. 与竞技场相同：重播 + `restart 0`，再重播 + `goto`。
2. 保留房间布局自然生成的敌人；Boss 的冠军掷骰按原样进行，不排除。
3. 执行玩家模板（同竞技场）、Boss 血量随机化、清诅咒、闩门，然后重播本局种子。
4. 入口槽不被房间接受时，换下一个槽。
5. 布局里的敌人刷新点带随机性，某些种子下开局就已清房。这时按种子确定性地换下一个候选房间（`RoomUnusable`），不重启实例。

重置的实测（耗时、可复现性，以及一处未查明的例外）见 A4。

### 奖励 combat-v2（`isaac_bridge/abplus_reward.py`）

**为什么换**：combat-v1 的伤害项按敌人个数计分，每个 `IsEnemy` 实体打死记 1.0，刷出来的怪、回血和火堆都算。在混合的 573 个房间上探测，Duke of Flies 的房间没打赢也能拿到约 47（C1）。

**校准依据：游戏自带的每日挑战计分。** 在 AB+ 1.06 反编译里逐项核对过，与 huijiwiki 给的公式一致：
- 击杀分 ⌈5·MaxHP^0.2⌉。只在 `Room::IsFirstVisit()` 且 `GetSpawnGridIndex() >= 0` 时计入（`Entity_Player::TriggerEnemyDeath`），敌人刷出来的怪不算。
- 伤害惩罚 = 探索分 × 0.8 × (1 − 0.8^(受伤半心数/12))。受伤数读 `GetTotalDamageTaken()`（玩家 +0x28b4）：`TakeDamage` 按伤害量累加，排除标志 0x20（扣红心的自伤）和 0x10000（诅咒房门）。
- 时间惩罚 = (楼层分 + 终点分) × 0.8 × (1 − 0.8^(秒数/期望时长))。
- 房间分：普通房清理 40，Boss 房 100，过地下室 I 500。

按一次完整通关折算（探索分约 15000、楼层分 20000、期望总时长 46 分钟）：
- 1 个半心 ≈ 200 分 ≈ 130–220 秒；
- 1 个炸弹 ≈ 0.08 个半心；
- 在地下室 I 死亡 ≈ 170 个半心。

**用户决定（2026-09-24）**：
- 受伤按曲线计：第一个半心 −1，最后一个约 −4；自伤不算；
- 死亡额外 −5；
- 时间和炸弹按分数口径；
- Boss 清房奖励高于普通房。

| 项 | 定义 |
|---|---|
| 受伤 | 血量价值 V(h) = 11.94·f(h/6)，f(x) = (x + 1 − (1−x)⁴)/2，即 OpenAI Five 的血量曲线。3 颗心满血时依次掉半心的代价为 1.00、1.06、1.29、1.80、2.70、4.09；超过 6 个半心的部分每个 1。只计游戏自己计入的伤害（桥接 `player_damage` 即 `GetTotalDamageTaken` 的增量）；被桥接拦下的致死一击按伤害量计入。自己的炸弹炸到自己也算，与游戏一致。 |
| 死亡 | 额外 −5 |
| 时间 | 每游戏秒 −1/170，全程计；120 秒约 −0.71 |
| 炸弹 | 每用一个 −0.08 |
| 清房 | (房间分 + 开局挡门敌人的击杀分 + 过层分) ÷ 200。普通房的房间分 40；Boss 房（含竞技场）100，加过层分 500。实测普通房 0.24–0.59，Boss 房 3.08–3.45 |
| 超时 | 120 秒时 −1，作为终止处理（同 combat-v1） |
| 进度 | 势函数：挡门 NPC（`CanShutDoors`）的总血量每减少 70 记 +1，基础泪弹每发 0.05；刷怪和回血计负。死亡或超时时把剩余势能一次付清，所以同一房间每局的进度总和固定：它只帮助学习，不改变最优打法 |

桥接 abp-0.2.1 在 combat 块里加了两个字段，二进制和 JSON 路径都有：
- `blocking_hp`：挡门 NPC 当前的总血量；
- `blocking_points`：这些 NPC 的击杀分之和。

验证（单元测试、JSON 与二进制对比、30 局引擎检查、worker 压测）见 C1。

**训练日志**：
- `episodes.jsonl` 每局带 `reward_components`；
- 每轮记录 `task/<类型>/episodes|win_rate|return` 和 `reward/<分项>` 的均值。

`--reward-profile combat-v1` 可切回模拟器的奖励。换奖励只能用 `--warm-start`（只取权重），`--resume` 会拒绝奖励不同的 checkpoint。

### 奖励 combat-v3、开局炸弹随机与 PLR（2026-09-25）

**为什么换**：
- 用 combat-v2 训到第 3900 轮后看回放，普通房超时是躲在角落挂机，通关几乎都靠开局炸弹和一种固定的绕房间走法（C2、C3）。
- 在 combat-v2 下，不受伤地躲满 120 秒只扣 1.71，而掉 2 个半心清一个普通房大约扣 1.4–2.0，所以躲才是最优策略。
- 进度奖励按设计是"打不打总额都一样"，推不动它去打怪。
- 于是用户决定：从头训练、换奖励、开局炸弹随机，房间配比改用 PLR。

**combat-v3**（`isaac_bridge/abplus_reward.py`，单位仍是半颗心）：

| 项 | 数值 |
|---|---|
| 掉血 | 与 combat-v2 相同的曲线：1.00、1.06、1.29、1.80、2.70、4.09 |
| 超时 | −20。清房时哪怕只剩半颗心，掉血最多 7.85，所以任何清房结局都远好于超时 |
| 死亡 | −25（超时的 20 加原来的 5），并补扣"从现在起不再打中"时剩余时间和停滞本该扣的分 |
| 时间 | 每秒 −(1/120 + 存活挡门怪数/60) × (1 + (t/60)²)：60 秒时翻倍，120 秒时 5 倍 |
| 停滞 | 任何时候连续 20 秒没打掉挡门怪的血（从开局或上次打中算起）：到 20 秒时 −10，之后每秒再 −1/6，直到下次打中 |
| 击中 | 挡门怪每掉 20 点血 +1，一发普通眼泪约 +0.175，当场给，不再补发 |
| 杀怪 | 挡门怪每少一只 +1 |
| 清房 | 普通房 +3，Boss 房和竞技场 +6 |
| 炸弹 | 每个 −0.1 |

- **击中和杀怪刷不了分**：新刷出来的怪和回的血在出现时先扣掉，所以一局下来总额只取决于开局那批怪清掉了多少。
- **排序**：清房 > 超时 > 死亡，既不会站桩，也不会"打不过就自杀"（单元测试验证）。
- **训练缩放**：训练时整体乘 0.25，数值范围与 combat-v2 相当；日志里的分项不缩放。
- **例子**（3 只 Horf，共 30 点血）：
  - 躲满 120 秒不受伤、一次没打中：约 −63（时间 16.3、停滞 26.7、超时 20）；
  - 开局掉 5 个半心，59 秒才第一次打中，118 秒清房：约 −43；
  - 30 秒被打死：约 −80。
- **停滞规则的由来**：第一版只看"开局一分钟内有没有打中"，abp-mix-02 学会了开局约 2 秒先打一下再躲（C4）。于是改成任何时候连续 20 秒没打中都罚（用户 2026-09-25 定的 20 秒），并在死亡时补扣这部分（C5）。

**配套改动**：
- **桥接 abp-0.2.2**：`combat.blocking_count` 报告挡门怪的数量（CanShutDoors，与 blocking_hp 同一判定）。远端旧版备份在 `bridge/abp_bridge.lua.abp-0.2.1`。
- **观测**：combat-v3 的帧多一项 `combat`，内容是存活挡门怪数/10、是否已打中过、挡门血量占开局的比例、距上次打中的秒数/60。
  - 这是时间曲线和停滞惩罚依赖的状态，挂在玩家 token 上（`transformer_obs.COMBAT_FIELDS`）。
  - AB+ 的帧格式 `FRAME_DTYPE_COMBAT` 在模拟器格式后面追加这一项，所以模拟器的 ABI 不变；`decode_frame` 按环境的 `frame_dtype` 解码。
- **开局炸弹**：`--start-bombs 0.5:3`，一半的局开局 0 个，另一半 1–3 个，由种子决定（`abplus_worker.sample_bombs`）。评估时按同一抽法、每个种子固定。
- **PLR**（`isaac_bridge/plr.py`，Jiang et al. 2021，`--room-sampling plr`）：
  - 每个房间是一个 level：竞技场、495 个普通房、77 个 Boss 房。
  - 每轮用 GAE 的正优势均值（positive value loss）给打完的局所在的房间打分；跨轮的局会累计。
  - 三类房间保持混合配比（竞技场 0.2、普通 0.45、Boss 0.35，来自 `--tasks-file` / `--task-weights`），PLR 只决定同一类里抽哪个房间：
    - P(房间) = 该类的权重 × 类内概率；
    - 类内概率 = 0.9 × [已玩过的房间：0.7 × 排名分 (1/rank) + 0.3 × 久未玩过的程度；没玩过的房间按比例均匀先试] + 0.1 × 类内均匀保底；排名、久未玩过的程度都只在同类房间之间比较。
  - 排名温度 β = 1 而不是论文的 0.1–0.3，避免 16 个环境挤进一两个房间。
  - 学习器每轮把概率写进共享内存，worker 每局开始时按它抽房间；PLR 状态随 checkpoint 存为 `plr.json`（含各类权重）。学习器写入第一版分布之前，worker 先按混合配比抽。
  - 评估仍按种子固定房间，与以前可比。
  - **修正（2026-09-25）**：最初的实现把 573 个房间放在同一个分布里排序，配比没有生效，竞技场和 Boss 房几乎抽不到（abp-mix-02、03、04 都受影响）。现在按上面的方式分类计算，各类份额精确等于配比（C8）。
- **日志**：
  - `behavior/no_hit_by_60s`（60 秒还没打中的局）、`behavior/camp_20s`（在一处停留 ≥20 秒的局）、`behavior/first_hit_s`、`behavior/longest_no_hit_s`、`behavior/cells`、`behavior/bomb_use`；
  - `task/<类型>/death_rate|timeout_rate|win_rate_0bombs|win_rate_bombs`；
  - `plr/*`：各类房间的份额和类内有效房间数（`effective_<类型>`）、已玩过的房间数、总有效房间数、最大概率。
  - 每局在 `episodes.jsonl` 里还带 `level`、开局炸弹数和 `stats`。
- **评估**：
  - 先跑最优动作一遍（和以前的评估可比），再跑按概率采样一遍（`--eval-sampled`），各录 `--eval-replays` 局；
  - 生成 `evaluations/<checkpoint>/replays.html`（`abplus_replay_view.py`）；
  - 结果按开局炸弹 0 个 / 1 个以上分开统计胜率。

### 两阶段训练，阶段一：combat-v4（2026-09-25）

**为什么又换**：
- combat-v3 为了防自杀，让死亡始终比超时更亏。当时根据 abp-mix-03 推断：只要清房几率不高，躲到超时就比冒险划算（C5）。
- 于是用户定了两阶段方案，并先做约 200 游戏小时的测试：
  - 阶段一：以奖励为主，学会进攻清房；
  - 阶段二：以保命为主，学会躲避。
- 后来的同游戏时对比（C7）显示，去掉惩罚后躲的比例没变，上面的推断不成立。

**阶段一奖励 combat-v4**（`isaac_bridge/abplus_reward.py`）：

| 项 | 数值 |
|---|---|
| 击中 | 挡门怪每掉 20 点血 +1（差分，新刷的怪和回血先扣后还） |
| 杀怪 | 挡门怪每少一只 +0.25（差分），奖励收尾，防止"都打残、没打死" |
| 受伤 | 每次受伤 −0.1，平坦，只用于信用分配，没有递增曲线 |
| 死亡 | −0.5，不做任何结算 |
| 清房 | +30 − 0.5 × 掉的半心数 − 0.5 × 用掉的炸弹数，只在清房时结算 |
| 超时 | 截断（学习器用价值函数自举），不惩罚 |

- **30 这个数的依据**：大于任何房间"击中 + 杀怪"能拿到的总和。开局挡门血量最多的是 Boss 房 1047，共 415 点，折合 20.75。它也远大于掉 6 个半心加用 3 个炸弹的 4.5，所以任何清房都比任何不清房好。
- **轨迹检验**：纯躲为 0；进攻后死掉是正的伤害奖励；清房约 +30。
  - 几乎没有负项，所以既不存在"躲着少扣"，也不存在"早死少扣"，不需要任何结算。
  - 随机策略的回报差异主要来自偶尔打中，梯度从一开始就指向瞄准。
- **删掉的**：时间代价、停滞惩罚，以及只为它们服务的观测（是否打中过、距上次打中的时间）。
- **留下的房间状态**：挡门怪数量和血量比例（`COMBAT_FIELDS_V4`）。
- **截断**：超时是截断，所以观测里去掉了剩余时间（`remaining_time`）。否则第 120 秒处的价值只能靠自举，永远校准不到真实结局。时间编码本来就只用窗口内的相对时间。
- **γ**：从 0.999 改为 0.9995。120 秒末端的权重从 0.165 升到 0.41，速度偏好不再过强。
- **训练缩放**：0.1。
- **保留的**：开局炸弹随机、PLR 选房间、行为日志、两种评估；每局记录是否被截断。

测试和 200 游戏小时测试运行 abp-mix-04 的结果见 C6；之后的三项检查见 C10。

### 阶段一修订版：combat-v5 与分解动作头（2026-09-26，现在的默认）

由来见 C10：刷怪的差分让打 Nest、Mulligan 的那一步拿到负奖励。

**原则不变**：躲着不打为 0；同一个房间里，任何清房都胜过任何不清房；没有按步扣分；超时按截断处理。

**奖励**（原始值，训练时 × 0.1；`isaac_bridge/abplus_reward.py` 的 `CombatV5`）：

| 项 | 数值 | 怎么量 |
|---|---|---|
| 击中 | 每 5 血 +1（一发眼泪 +0.7）；第一轮是每 10 血（`--hit-hp`） | 只算 roster 谱系怪掉的血（桥接 `combat.lineage_damage`），任何来源都算；每只最多算它加入谱系时的血量，回血不能重复刷分 |
| 杀怪 | 每只 +0.25 | 谱系怪死亡（`combat.lineage_kills`） |
| 对齐势 | γΦ(s′) − Φ(s)，Φ = −0.2 × d_fire / 40 | d_fire：到最近有效射击位置的像素距离（上限 600，`abplus_geometry.py`，见下）；清房和死亡步 Φ(s′) = 0，截断步照常 |
| 受伤、死亡、清房、清房扣血、清房扣炸弹、超时 | 同 combat-v4 | |
| 刷怪、回血、变身 | 0 | 不再有负项 |

- **roster 谱系**（桥接 abp-0.2.3）：开局设置完成时，房里所有挡门怪构成 roster；之后按下面的规则继承。
  - 同一实体的 Morph 自然保留。
  - 用户的规格是"死亡同帧、SpawnerEntity 指向它的新实体继承谱系"。但 AB+ 里死亡时生成的实体 SpawnerEntity 为空，这条规则区分不了（C11）。
  - 现在的做法：死亡那一帧、在死亡位置 60 像素内生成的挡门怪算继任者。
  - `lineage_mode`：
    - 0：不继承；
    - 1（默认，用户 2026-09-26 定）：谱系怪死亡留下的全部继承，包括 Nest 变成的 Big Spider / Trite、Big Spider 分出的 Spider、Mulligan 放出的苍蝇；
    - 2：只有死亡恰好留下一只时才继承（变身），Mulligan 的苍蝇、Big Spider 的 Spider 不继承。
    - 第一轮两个臂用的是 2，两种模式的差别见 C11。
  - 活着时刷出的怪（Nest 靠近时刷的蜘蛛）从不继承。
  - 与引擎的核对见 C11。
- **d_fire**：
  - 目标是活着的谱系怪，没有时取全部挡门怪。每个目标的容差 τ = 碰撞半径 + 眼泪半径 8.16。
  - 水平射击带：|y − e_y| ≤ τ，x 在目标左右各 R（玩家射程 260）以内，并截断在目标所在行上两侧第一格永久挡眼泪的格子处（未炸碎的石头、铁块、锁块、雕像）。便便、TNT、坑不截断，因为眼泪能打掉前两者、能飞过坑。垂直带对称，沿目标所在的列。
  - d_fire 是到两条带的距离取小，再对所有目标取小。
  - 不看目标所在行列以外的遮挡，也不看无敌阶段。画面里一个目标都没有时，d_fire 保持上一帧的值。
  - 第一轮用的是不截断的版本，从第二轮起截断（C13、C14）。
- **观测新增**：
  - 每个实体行加两位：是否谱系、是否挡门；
  - 标量 fire_distance = d_fire / 40（上限 15），只进价值分支；
  - 辅助头的标签 aim_label（能打中已对齐目标的射击方向，0 = 无）和 approach（9 个移动方向中哪些会让 d_fire 变小）。这两个只当标签，策略不读。
  - 上一步动作改成 9 + 5 + 2 + 2 的 one-hot。

**模型**（`transformer_policy.py` 的 `GeometryPolicy`）：
- 动作头拆成移动 9 类、射击 5 类，加上原有的炸弹、主动道具两个二元头。对数概率相加、熵相加，掩码不变。worker 协议仍是 joint，学习器发送前换算。
- fire_distance 不进主干：特征最后多一维，`SplitMlpExtractor` 只给价值分支。这样 d_fire 回归对辅助头来说不是抄输入。
- 几何辅助头：从主干特征接一个小 MLP，输出 5 类射击方向和 d_fire / 40 的回归。损失权重 0.2（交叉熵 + 平方误差），与 PPO 一起更新主干。
- 熵系数默认 0.003，可用 `--ent-coef-heads move=0.005,shoot=0.002` 分头设置。γ 默认 0.9995。

**新日志**：
- `train/entropy_{move,shoot,bomb,item}`；
- `aux/aim_accuracy`、`aux/aim_accuracy_aligned`、`aux/aligned_share`、`aux/distance_mse`；
- `behavior/mode_aim_rate`：存在对齐目标时，射击头的最大值等于标签的比例；
- `behavior/mode_approach_rate`：d_fire 超过 1 格时，移动头的最大值会让 d_fire 变小的比例；
- `train/hit_advantage_gap`：命中步的平均优势减去其他步，以标准差为单位。

实现核对、第一轮（熵 0.003 对 0.01）、第二轮（障碍几何 + 击中翻倍）和两轮的超时回放见 C11–C15。

**阶段二**（未做，等阶段一在大多数房间清房率超过 90%）：
- 清房奖励换成实测的 V_next(结束时血量, 炸弹, 层数)，失败为 0。
- 可以把 (λ_h, λ_b) 作为策略输入，训练时随机采样，让外层规划器下发风险偏好。

### 启动

```bash
# 等模拟器训练结束后自动开始，以它的最新 checkpoint 热启动；输出在 ~/isaac-abplus/train/<run>
setsid nohup bash ~/isaac-abplus/bridge/train_abplus.sh abp-mix-01 --wait-for-sim --game-hours 2500   --checkpoint-every 50 --eval-every 50 > ~/isaac-abplus/train/abp-mix-01.log 2>&1 < /dev/null &
```

开训后日志里的 `game/speed_x_realtime` 就是实测速度。各次训练的实际速度见 EXPERIMENTS.md（B2、C2 等）。

### AB+ 进程的内存泄漏与实例回收

严格等价模式原先短路了 `ImageManager::apply_frame_images`。这个函数除了画，还负责回收透明渲染批次，所以短路后每局泄漏约 0.27 MiB，重置也越来越慢（排查过程见 B3，根因详见 [ABP_LINUX_REVERSE_ENGINEERING.md §15.11](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)）。

**修复**：`tools/stub_render_f.txt` 去掉 `apply_frame_images` 一行（2026-09-24 22:18 远端生效，旧版备份为 `.bak-20260924`）。同种子同动作的轨迹哈希与原配置逐位相同，训练数据不变。

**兜底：实例回收**（worker 端）：
- 每个 AB+ 实例打满 `--recycle-episodes`（默认 200）局后，在它作为备用实例准备下一局时，先在后台线程里重启进程再重置；
- 停进程时等它真正退出（超时则 SIGKILL），端口释放后再用同一端口重开；
- 日志里 `abplus/instance_recycles` 记录重启次数。

**Steam 依赖**：每个 AB+ 进程启动时，可执行文件的 DRM 外壳都要找到正在运行、已登录的 Steam 客户端；找不到就执行 `steam.sh steam://run/250900` 拉起 Steam，然后自己退出。
- 已在运行的训练实例不受影响，但下一次回收重启会失败：重试期间整批卡住，最后训练崩溃。2026-09-25 00:11 遇到过一次（B4）。
- 以后看到评估长时间 0 局，或实例日志里有 `SteamAPI_IsSteamRunning() did not locate a running instance of Steam`，先检查远端的 Steam 是否在运行且已登录。

**Steam 预警**（2026-09-25 加入，从 abp-mix-01c 起生效）：
- **检测**：学习器每轮检查 Steam 客户端进程（`isaac_bridge/steam_watch.py`，进程名 `steam`、可执行文件 `…/ubuntu12_32/steam`）。`~/.steam/registry.vdf` 里的 `SteamPID` 与实际客户端对不上，不采用。
- **报警渠道**：
  - 训练日志里的 `steam_down` / `steam_up` 事件，以及 `abplus/steam_ok`；
  - 运行目录下的 `STEAM_DOWN` 文件；
  - 远端桌面通知；
  - `$ABP_ALERT_CMD`（启动训练前设置）。例如推送到手机：`export ABP_ALERT_CMD='curl -s -d "$ABP_ALERT_MESSAGE" https://ntfy.sh/<自己的主题>'`。
- **报警节奏**：Steam 一退出就报警，未恢复时每 10 分钟重复一次，恢复时再通知一次。
- **Steam 不在时**：跳过评估（`abplus_eval.py` 启动时也会检查并直接退出），实例回收推迟 50 局再试。
- **回收改为先启后停**：先用实例的第二套身份（名字加 `x`，端口加 2 × 环境数）启动新进程，确认它能服务桥接后再停旧进程。新进程起不来（例如 Steam 在运行但被别处登录挤下线）时，保留旧进程继续训练，记一次 `instance_start_failures` 并报警 `instance_start_failed`。

**采样提速：CUDA Graph**（2026-09-25，`isaac_bridge/graph_sampler.py`、`abplus_bench_sampler.py`）：
- **问题**：学习器每步推理约 9–11 ms，但 GPU 真正在算的只有约 2 ms。`encode_frames` 用布尔索引压缩实体，还调用 `.item()` 和 `nonzero()`；SB3 的带掩码分布会做参数校验，每步大量主机与设备同步和内核启动（测量见 B5）。
- **改法**：新增 `CombatTransformer.encode_frame_static`。它对 256 个实体槽全部计算，用注意力的 padding 掩码代替压缩，形状固定、不需要同步。采样改为 Gumbel-max，分布与原来相同。每个 chunk 捕获"编码新帧"和"采样动作"两张 CUDA Graph，每步直接重放。单个环境的重置、价值估计、确定性动作仍走原来的 eager 路径。
- **自检**：训练中每 1024 步与 eager 路径比一次（`sampler/graph_*`），差别只来自浮点求和顺序。
- 默认 `--sampler graph`。单步耗时、一致性和训练中的实测提速见 B5。

**训练与采样重叠：`--async-train`**（2026-09-25，`isaac_bridge/gpu_ppo.py`）：
- **做法**：
  - 策略复制一份作为采样用的 actor；学习器在后台线程、独立的低优先级 CUDA 流上训练上一轮数据的快照（第二份 rollout buffer，约 1.26 GB）；
  - 每轮采样结束时：等训练结束，再存 checkpoint / 评估，然后 actor 原地拷贝学习器权重（CUDA Graph 不用重新捕获），把刚采完的一轮拷进快照，开始下一次训练；
  - 采样的 compute 流设为高优先级；训练期间 Python 的 GIL 切换间隔从 5 ms 降到 0.5 ms，采样线程等 GIL 最多 0.5 ms。
- **代价**：每轮数据由比训练起点早一次更新的权重采集，即一次更新的策略滞后。
- **目标函数：解耦的 PPO**（IMPACT，Luo et al. 2019；Hilton et al. 2021 的 decoupled PPO）。第一版直接用采样时记录的对数概率做比值（Sample Factory APPO 的做法），训练发散（B6）。现在的做法：
  - 比值和裁剪相对训练起点的权重（proximal policy）：每次训练前先无梯度地算一遍这批动作在起点权重下的对数概率，多花约 1.5 s；
  - 每个样本的策略损失乘以 π_起点/π_采样，截断在 2，用来校正滞后；
  - 没有滞后时（同步训练、第一次更新）与标准 PPO 完全相同，有单元测试。
- **checkpoint 语义**：每轮边界存的 checkpoint 含第 k 次更新后的权重；此时第 k+1 轮已采完但还没训练，从这个 checkpoint 续训会丢掉这一轮。
- **日志**：`async/collect_s`、`async/update_s`、`async/join_wait_s`（采样等训练的时间）、`train/lag_kl`、`train/lag_weight_truncated`。
- 与同步训练的稳定性和速度对比见 B6。

停训练要用 SIGTERM，不能用 Ctrl-C。原因是后台（nohup/&）启动的进程继承了"忽略 SIGINT"，Python 就不装 KeyboardInterrupt。SIGTERM 之后，worker 发现管道关闭，会各自关掉自己的 AB+ 实例。

- 默认配置：16 个环境（32 个实例）、`--chunks 1`、每轮 1024 步 × 16 个环境 = 16,384 样本、batch 1024、2 epoch、奖励 combat-v5、熵 0.003、gamma 0.9995、严格 FP32。
- 每 10 轮存一次 checkpoint；每 20 轮用 2 个 CPU 实例在 64 个 held-out 混合种子上评估。
- 常用参数：`--task-weights arena=…,normal=…,boss=…`、`--tasks-file none`（只训练竞技场）、`--json-obs`（回到桥接 v1）、`--game-hours N`（按游戏时停止）。

已知的待定项：
- 从模拟器热启动时，价值头是按 combat-v1 训练的，换奖励后前几轮价值误差会偏大。
- 同步 PPO 在采样和训练之间串行；`--async-train` 让两者重叠（见上）。
- 周期评估要占用约 3 个 CPU 核。

## 用法（远端 Ubuntu，`~/isaac-abplus`）

```bash
cd ~/isaac-abplus/bridge/python
PYTHONDONTWRITEBYTECODE=1 nice -n 19 /home/eolc/isaac-rl/.venv/bin/python -u abplus_eval.py \
  --checkpoint ~/isaac-abplus/checkpoints/u0425 --seeds ~/isaac-abplus/eval/seeds-e2.json \
  --out ~/isaac-abplus/eval/u0425-det-e2 --instances 6 --device cpu --verify-steps 3 --replays 16
python abplus_transfer_report.py --sim <e2/results/u0425/det> --abplus <results.jsonl> \
  --sim-replays <e2/replays/u0425/det> --abplus-replays <replays> [<more replay dirs>] --out summary.json
```

- `--device cpu`：训练进程占着 GPU 时，多进程共用 GPU 要排时间片，单步推理平均约 30 ms；CPU 单线程轻载时 7–9 ms，6 个 worker 满载时约 12 ms。
- `--repeat N` 把每个种子跑 N 次，用轨迹哈希检查可复现性。
- `--verify-steps N` 在每局前 N 步把增量推理和整窗前向逐项比对（迁移评估里最大差都是 0.0）。
- 结果按种子追加写入 `results.jsonl`，中断后重跑会跳过已完成的种子。
