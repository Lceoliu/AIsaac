# AB+ 桥接与训练管线（2026-09-24）

目的：让模拟器训练出的策略直接在 Afterbirth+ v1.06 原生 Linux 版（原版引擎，经 `abp_turbo` 改造加速）上评估，将来也能在上面训练。协议和观测格式沿用 J460 的 IsaacRLBridge 0.2.0（combat_schema=3），所以 `VisibleHistory` 和策略网络不用改。

引擎层（实例隔离、虚拟时钟、严格等价的 render-lite 模式）见 [ABP_LINUX_REVERSE_ENGINEERING.md](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)。本桥接查明的引擎行为记在该文档 §15。

## 组成

| 文件 | 作用 |
|---|---|
| [`abp_bridge.lua`](abp_bridge.lua) | 游戏侧（abp-0.2.0）。由 runtime 副本 `main.lua` 的钩子在 `--luadebug` 下 `dofile(ABP_LUA)` 加载，不走 mods 目录。协议比 0.2.0 多 `{"cmd":"lua"}`、`format`（二进制观测 v2）和 `profile`（观测各部分耗时）；致命伤在 `MC_ENTITY_TAKE_DMG` 里拦下并记为死亡（AB+ 没有 rewind，死了的玩家救不回来） |
| [`isaac_bridge/abplus.py`](../python/isaac_bridge/abplus.py) | `AbplusTrainingEnv`（竞技场重置）、`AbplusTransformerEnv`（Gymnasium 环境，deadline 观测）、`launch_abplus`/`stop_abplus`（起停实例）、`sim_arena`（逐位复刻 `rl/sim/src/arena.rs`） |
| [`abplus_eval.py`](../python/abplus_eval.py) | 多实例评估：每个 worker 进程带一个 AB+ 实例和一份策略，按 `FrameSampler` 的方式增量推理（每帧只编码一次，时序 Transformer 在缓存窗口上重算）；指标与模拟器审计 `e2_reeval.py` 相同 |
| [`abplus_transfer_report.py`](../python/abplus_transfer_report.py) | 同种子配对比较模拟器与 AB+ 的结局、行为指标和 Monstro 动画时长 |
| [`abplus_smoke.py`](../python/abplus_smoke.py) | 单实例随机动作冒烟测试和耗时拆分 |
| [`abp_turbo.c`](../../../analysis/scripts/abplus/abp_turbo.c) | 引擎改造层；本轮新增 `getenv("ABP_RESEED:<n>")` 钩子，重播全局 MT19937 |
| [`train_abplus.py`](../python/train_abplus.py)、[`isaac_bridge/abplus_vec.py`](../python/isaac_bridge/abplus_vec.py)、[`isaac_bridge/abplus_worker.py`](../python/isaac_bridge/abplus_worker.py)、[`train_abplus.sh`](train_abplus.sh) | 训练管线：独占 GPU 的学习器、每个环境一个 worker 进程（两个 AB+ 实例轮换）、共享内存帧、启动脚本，见"训练管线" |
| [`isaac_bridge/abplus_obs.py`](../python/isaac_bridge/abplus_obs.py) | 桥接 v2 二进制观测的解码器；验证脚本 `abplus_probe_obs_v2.py` |
| [`isaac_bridge/abplus_tasks.py`](../python/isaac_bridge/abplus_tasks.py)、[`catalog/`](catalog/) | 房间级混合：从运行中的引擎探测的房间目录、Basement I Boss 池、混合规格；探测脚本 `abplus_room_catalog.py`、`abplus_probe_boss_pool.py`、`abplus_probe_rooms.py` |
| `abplus_bench_workers.py`、`abplus_bench_train_cpu.py`、`abplus_probe_obs_cost.py` | 环境侧吞吐上限、CPU 上的 PPO 训练开销、Lua 观测各部分耗时 |
| `abplus_probe_*.py`、`abplus_bench_policy.py` | 下表各项结论的诊断脚本，各自起一个实例：`door`（LeaveDoor 与门）、`goto`（goto 后逐帧生成了什么）、`champion`（冠军在何时出现）、`roomseed`（房间种子与 MT19937 的关系）、`determinism`（固定动作逐帧比对）、`reset`（重开后的楼层、诅咒、耗时）；`abplus_bench_policy.py` 测 CPU/GPU 推理延迟 |

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

每一步都对应一个实测问题：

| 步骤 | 不这样做时的实测现象 | 原因（静态依据） |
|---|---|---|
| 重播 MT19937 | 同一种子、同一确定性策略，两次结局不同（第 754 帧死亡 / 打满 3600 帧） | `ai_monstro` 全程用全局 `Random`/`RandomU32`/`RandomUnitVector`；引擎只在启动时 `InitRandomNumbers` 播种一次 |
| `LeaveDoor` | 门随机出现在右/上/下，策略从第一个动作起就不同 | `goto` 从 `LeaveDoor` 的对面进入，只生成那一扇门；实测设 L 得到门 (L+2)%4，玩家落点正是模拟器的入口坐标 |
| 等 8 帧再清场 | 原先只等 2 帧，房间自带的 Boss 在清场之后才出现 | 房间自带的 Boss 在 goto 后第 3–4 帧才生成 |
| 冠军检查 | 竞技场的 Monstro 有时一开局就变成冠军：subtype 1 会再分出一只（固定动作的 6 局里有 3 局，Boss 血量读数超过 100%，伤害比例为负）；subtype 2 行为不同 | `Entity_NPC::load_entity_config` 用 `RNG(房间描述符种子, 5)` 掷冠军，概率 0.3（本存档的解锁状态下）；同一房间里所有 Boss 结果相同；房间描述符种子取自全局 MT19937，先重播就能确定 |
| `restart 0` | 同一种子重放，眼泪左右眼交替和下落速度不同，第 1 步就分叉 | 玩家对象跨 goto 保留状态；AB+ 没有 rewind |
| 清诅咒、改石头 | 重开后的 run 会掷出黑暗、未知、迷宫等诅咒；布局里的石头有时会变成染色石、炸弹石或罐子 | 楼层生成和房间装饰随机化 |

修好以后，同一种子、同一动作序列的所有观测字段在不同实例、不同时间都逐帧一致（实体 id 和帧计数除外）。

## 与模拟器的已知差异

- Monstro 的随机数流不同（AB+ 用 MT19937，模拟器用 xorshift），所以配对的是起始局面，不是轨迹。
- **出场阶段是模拟器的缺陷**。AB+ 的 Monstro 生成后隐身 4 帧，再播约 25 帧 `Appear`，期间不行动；J460 人类录像（`runs/human/20260923-session01`，6 局）开局也有约 31 帧不行动。模拟器的 `new_monstro` 把 state 设为 1，却没有设 `FLAG_APPEAR`（Gaper 设了），所以 AI 从第 1 帧就开始。策略利用了这一点，见下文"迁移评估"。
- **炸弹伤害是版本差异**。AB+ 的 `Entity_Bomb::Init` 把普通炸弹伤害设为 60.0，引信 45 帧；J460 录像里炸中 Monstro 一次是 100；模拟器用 100，与忏悔+ 一致。
- **`JumpDown` 时长也是版本差异**。AB+ 约 74 帧；J460 录像和模拟器都是 65 帧。其他动画时长基本一致。
- `range` 按 `260 × TearHeight / -23.75` 换算，对以撒是精确值。
- 致命伤由桥接拦截（虚拟死亡），结局判定与模拟器相同。

## 迁移评估：u425 在 AB+ 上（2026-09-24）

- **检查点**：`monstro-128k-reward-v1-20260923` 的 update 425。model.zip 的 sha256 为 `04ddc76d…`，与模拟器审计用的是同一份。
- **种子**：模拟器审计 E2 的 256 个 held-out 种子（每种布局 64 个）。
- **设置**：确定性策略，严格等价模式，6 个实例与训练同时运行（nice 19）。
- **结果文件**：[`runs/abplus-transfer-20260924/u0425-det-e2/`](../../runs/abplus-transfer-20260924/u0425-det-e2/)。其中 `report.txt` 和 `summary.json` 由 `abplus_transfer_report.py` 生成。

| 指标（256 个配对种子） | 模拟器 | AB+ | 差（AB+ − 模拟器） |
|---|---|---|---|
| 胜率 [95% CI] | 0.191 [0.148, 0.244] | **0.004** [0.001, 0.022] | −0.188 |
| 死亡率 | 0.715 | 0.910 | +0.195 |
| 超时率 | 0.094 | 0.086 | −0.008 |
| Boss 伤害比例 | 0.677 | 0.384 | −0.293（约 73 HP） |
| 每分钟命中次数 | 28.1 | 26.4 | −1.7 |
| 受伤次数 / 首次受伤（秒） | 5.13 / 10.9 | 4.82 / 13.1 | −0.31 / +2.2 |
| 与 Monstro 平均距离 | 190.7 | 191.7 | +1.0 |
| 场上弹幕数 | 3.08 | 3.56 | +0.49 |

- **配对胜负**：只在模拟器赢 49 局，只在 AB+ 赢 1 局，两边都赢 0 局。McNemar p≈0。
- **分布局**：模拟器胜率 1010/1012/1037/1038 依次为 31/23/17/5%，AB+ 为 0/0/1.6/0%。

### 差距来自开局炸弹

- 策略在模拟器里 256 局**全部在第 1 步放炸弹**。
- 其中 **136 局（53%）** 在前 2 秒内被跳过来的 Monstro 踩中，一次掉 100 HP。
- 模拟器胜率：炸中时 **48/136 = 35%**，没炸中时 **1/120 = 0.8%**；49 场胜利中有 48 场靠开局炸弹。
- AB+ 64 局回放里开局炸弹命中 **0 次**，因为 Monstro 还在出场阶段。AB+ 胜率 0.4%，与模拟器"没炸中"的 0.8% 相当。

结论：
- u425 在模拟器里的胜率几乎全部来自"模拟器缺少出场阶段"这个缺陷。忏悔+ 里炸弹伤害 100，所以这个利用在模拟器里格外值钱。
- 除去这一点，两个引擎给出的战斗能力基本一致：命中率只差 6%，受伤次数 5.1 与 4.8 相近，不靠开局炸弹时两边胜率都在 1% 以下。策略本身还不会打 Monstro。
- 建议修正模拟器的 Monstro 出场阶段（以 J460 录像的约 31 帧为准），重新评估现有检查点，并留意正在进行的训练是否也依赖这个利用。这会改动训练环境，需要先决定再做，本轮没有动。

### Monstro 动画时长

64 个种子的回放，逻辑帧，中位数 [p10, p90]。采样间隔 2 帧，±2 帧以内视为一致。

| 动画 | 模拟器 | AB+ | J460 录像（6 局） |
|---|---|---|---|
| 开局 `Appear` | 2 [2, 2] | **26** [26, 26] | 约 26（开局约 31 帧不行动） |
| `JumpDown` | 66 [64, 66] | **74** [74, 76] | 65 |
| `JumpUp` | 10 [10, 10] | 12 [10, 12] | 10 |
| `Taunt` | 68 [66, 68] | 70 [68, 70] | 67 |
| `Walk` | 46 [44, 46] | 46 | 45–46 |

### 可复现性与吞吐

- **可复现性**：换一批实例、换一个时间重跑其中 48 个种子，结局、帧数和轨迹哈希 **48/48 完全一致**。
- **吞吐**：256 局 240,875 个决策用时 843 秒，即 **286 决策/秒**（6 个实例，含启动和每局重置）。
  - 这个负载下单个 worker 的中位数是 48.8 决策/秒，CPU 推理中位数 12.0 ms/步；轻载时约 70 决策/秒、推理 7–9 ms。
  - 256 局中 0 局出错。唯一一条异常记录出现在那局胜利上：清房前最后 10 帧 Monstro 已经不可见（死亡过程），属于正常情况。
- **推理校验**：增量推理与整窗前向在全部 256 局前 3 步的 logits 最大差为 0.0。

## 训练管线（2026-09-24）

用户决定：
- RL 只负责战斗，导航和道具选择先用规则；
- 采用房间级混合学习；
- PPO 在 GPU 上由独占进程完成，所有 worker 每步拼成一批，尽量减少通信损耗；
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
- worker 用 `VisibleHistory.encode` 只编码最新一帧。在 64 局回放、12,434 帧上，与原 `VisibleHistory.append` 的窗口末帧逐字段完全相同。
- 种子序列与 `FrameChunk` 相同：环境 i 的第 k 局用 `base_seed + i + N·k`。训练种子 < 2³¹；held-out 种子 ≥ 2³¹，其中 E2 验证块和 2147500000 起的最终测试块都不参与训练。
- 开局随机化与模拟器训练一致：50% 的局 Boss 血量为 10–100%，25% 的局玩家为 3–5 个半心。评估一律满血。
- 日志额外记录：
  - `game/hours_*`：游戏时，等于决策数 × 2 帧 ÷ 30 帧/秒；
  - `game/speed_x_realtime`：相对实时的倍率，等于决策/秒 ÷ 15；
  - worker 单步耗时、局间切换等待、错误数。

### 桥接 v2：二进制观测（abp-0.2.0）

先实测 Lua 观测每一部分的耗时（`{"cmd":"profile"}`，单位 ms/步），再决定改哪里：

| 部分 | 构建 | JSON 编码 |
|---|---:|---:|
| 地形（135 格，房间内基本不变） | 0.28–0.36 | **0.76–0.86** |
| 网格与门 | 0.09–0.20 | 含在总数里 |
| 实体（经 luabridge 读字段） | **0.01–0.15** | 0.03–0.38 |

读字段本身几乎不花时间，贵的是每步重发静态地形和纯 Lua 的 JSON 编码，两者合计约 75%。因此 v2 的做法是：
- 动态数据用 Lua 5.3 的 `string.pack` 打成二进制（每条记录 0.26 µs）；
- 地形和网格只在可能变化时发送：reset/query 事件、lua/exec/reset 命令之后、换房间、任一格的碰撞类型变化；
- 地形危险标志里动态的部分（带接触伤害的特效）由客户端按 Lua 的规则重算；
- 整数一律用 int64（重开 run 后玩家的 `Index` 是 0xFFFFFFFF）；
- 解码器（`abplus_obs.py`）输出与 JSON 路径完全相同的字典，下游代码不用改。

原生读内存推迟，因为实体字段读取只占 0.01–0.15 ms/步，收益不值得冒结构偏移的风险。

验证（`abplus_probe_obs_v2.py`，同一状态同时发 JSON 和二进制）：
- 1,139 个观测的字典全部一致（相对容差 1e-12）；
- 策略帧里地形、动画、实体种类、掩码、时间、动作都逐位相同。唯一的差别是 JSON 用 `%.14g` 打印位置，近乎静止的实体速度相减后带出舍入噪声，最大 4.5e-13；二进制更精确。

速度：step 往返加编码从 3.14 ms 降到 **1.03 ms**；训练 worker 单步从 4.4 ms 降到 **1.57 ms**。abp-0.2.1 只在 combat 块多了两个字段（见下文奖励一节）。

### 房间级混合（`abplus_tasks.py`、`catalog/`）

AB+ 的房间数据直接从运行中的引擎取得，不用仓库里的 Repentance+ 房间文件：
- `abplus_room_catalog.py` 对 `goto d.<id>` 和 `goto s.boss.<id>` 逐个探测（84 秒）：Basement I 有 **1,022 个普通房**，其中 637 个是 1×1；特殊房表里有 **431 个 Boss 房**，覆盖全部楼层。
- `abplus_probe_boss_pool.py` 让关卡生成器重开 300 次，统计 Basement I 实际会出现的 Boss：共 13 种、82 个房间。依次是 Monstro 42、Larry Jr. 40、Little Horn 38、Duke of Flies 38、Famine 35、Ragman 34、Steven 19、Dangle 15、Gemini 14、Gurglings 9、Dingle 8、Headless Horseman 6、Turdling 6。
- `catalog/mixture_basement1.json` 的内容：
  - 普通房 495 个：1×1、开局未清房、至少有一个火堆以外的敌人；
  - Boss 房 77 个：上述 Boss 池里的 1×1 房间；
  - 默认权重：arena 0.2 / normal 0.45 / boss 0.35。
- 选房完全由种子决定：先按权重选任务，再选房间，再选入口。

普通房和 Boss 房的重置流程：
1. 与竞技场相同：重播 + `restart 0`，再重播 + `goto`。
2. 保留房间布局自然生成的敌人；Boss 的冠军掷骰按原样进行，不排除。
3. 执行玩家模板（同竞技场）、Boss 血量随机化、清诅咒、闩门，然后重播本局种子。
4. 入口槽不被房间接受时，换下一个槽。
5. 布局里的敌人刷新点带随机性，某些种子下开局就已清房。这时按种子确定性地换下一个候选房间（`RoomUnusable`），不重启实例。

实测：普通房和 Boss 房各 20 局，全部重置成功，中位耗时分别为 70 ms 和 87 ms。同一种子连续重放结果一致（Famine 房连续 8 次相同）。已知的一处例外：普通房之后第一次进入 Boss 房时，出现过 1 次与后续重放不同，原因未查明。竞技场的可复现性不受影响（48/48 一致）。

### 奖励 combat-v2（`isaac_bridge/abplus_reward.py`）

**为什么换**：combat-v1 的伤害项按敌人个数计分，每个 `IsEnemy` 实体打死记 1.0，刷出来的怪、回血和火堆都算。`abplus_probe_reward_scale.py` 用一个朴素的瞄准射击脚本把混合里 573 个房间各打一局：
- v1 的伤害+命中项：普通房中位 4.65（最大 28.8），Boss 房中位 11.3（最大 55.7），竞技场 4.6；
- Duke of Flies 的 9 间房一间都没打赢，中位却拿到 47.3；
- 没打赢的 Boss 房中位 12.3，比打赢的（6.0）还高。

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

桥接 abp-0.2.1 在 combat 块里加了两个字段，二进制和 JSON 路径都有；客户端要求桥接版本完全一致：
- `blocking_hp`：挡门 NPC 当前的总血量；
- `blocking_points`：这些 NPC 的击杀分之和。

**验证**：
- `test_abplus_reward.py`：9 个单元测试，覆盖曲线、自伤、致死一击、时间、炸弹、清房分和进度守恒。
- `abplus_probe_obs_v2.py`：658 个观测里，JSON 与二进制的字典 0 处不一致（含新字段）。
- `abplus_probe_reward_v2.py`：30 局（竞技场、普通房、Boss 房各 10 局，半数从 3 个半心开局），三项检查全部 0 失败：
  - 开局计数与独立的 Lua 枚举一致；
  - 进度总和等于开局势能；
  - 未死亡时，受伤总和等于血量价值之差。
- `abplus_bench_workers.py`（2 个 worker、随机动作、90 秒）：293 局、0 错误，逐步奖励之和等于分项总和。

**训练日志**：
- `episodes.jsonl` 每局带 `reward_components`；
- 每轮记录 `task/<类型>/episodes|win_rate|return` 和 `reward/<分项>` 的均值。

`--reward-profile combat-v1` 可切回模拟器的奖励。换奖励只能用 `--warm-start`（只取权重），`--resume` 会拒绝奖励不同的 checkpoint。

### 实测吞吐（模拟器训练同时在跑，nice 19）

| 项 | 结果 |
|---|---|
| 环境侧上限（`abplus_bench_workers.py`：12 个 worker、24 个实例、混合任务、随机动作） | **2,349 决策/秒 = 156.6× 实时**，即每小时产出 156.6 游戏小时；90 秒内 710 局，最大切换等待 0.04 ms，0 错误 |
| 端到端训练冒烟（4 个环境、混合任务、u425 热启动） | 约 100 决策/秒（6.5× 实时）。每步约 36 ms 花在与模拟器训练争用 GPU 时间片的推理上，环境只占约 2.5 ms。不代表 GPU 独占时的速度 |

GPU 独占时的端到端速度要等模拟器训练结束后实测。按 [ENGINE_REUSE_ASSESSMENT.md §6.5](../../../analysis/docs/ENGINE_REUSE_ASSESSMENT.md) 的推算，同步 PPO 约 1,100 决策/秒（约 70× 实时，每天约 1,700 游戏小时），采样和训练重叠时约 2,000 决策/秒。

### 启动

```bash
# 等模拟器训练结束后自动开始，以它的最新 checkpoint 热启动；输出在 ~/isaac-abplus/train/<run>
setsid nohup bash ~/isaac-abplus/bridge/train_abplus.sh abp-mix-01 --wait-for-sim --game-hours 2500   --checkpoint-every 50 --eval-every 50 > ~/isaac-abplus/train/abp-mix-01.log 2>&1 < /dev/null &
```

2026-09-24 14:46（远端时间）按上面的命令挂上：
- 预计 16:45 左右开训。
- 预算 2,500 游戏小时，与模拟器这次训练同量级（约 2,580 游戏小时），约 8,240 轮更新。
- 每 50 轮存一次 checkpoint 并评估，总共约 165 个、约 10 GB。
- 预计耗时 26–42 小时，中间值约 32 小时。

估算方法：每轮 16,384 个样本。
- 采样：1,024 步 ×（16 个环境批量推理约 3–5 ms + 16 个 worker 同步单步约 3–5 ms）≈ 6–10 秒；
- 训练：约 5.5–7 秒，与模拟器同一批量；模拟器实测约 7 秒/轮，其中采样约 1.6 秒；
- 合计每轮约 12–17 秒，即 60–95× 实时。

开训后日志里的 `game/speed_x_realtime` 就是实测值。

**实际运行（abp-mix-01）**：
- 模拟器 15:47 正常结束。它的局数计数包含续训前约 6,500 局，所以比按局数推算的时间早。
- AB+ 训练 15:48 自动开始，从模拟器最终 checkpoint（第 8110 轮）热启动。
- 实测 **49–53× 实时**（732–795 决策/秒），低于估计。学习器每步的推理加 Python 开销约 9 ms，而不是估计的 3–5 ms。

### AB+ 进程的内存泄漏与实例回收

跑了 5.7 小时（275 游戏小时、25,689 局）后发现：
- 每个训练实例的内存从约 270 MiB 涨到 840–920 MiB，32 个合计每小时约 +3.5 GB，约 14 小时后会耗尽内存；
- 备用实例单次重置从 144 ms 线性涨到 540 ms，单步也在变慢，速度从 53× 降到 49×。

用新开的实例排查（`abplus_probe_leak.py`）：

| 操作 | 每次内存增长 | 耗时变化 |
|---|---:|---|
| 竞技场重置（restart + goto + Lua 设置）×150 | 0.03 MiB | 无 |
| 单独的 restart / goto / Lua 设置 ×150 | 0 | 无 |
| 混合房间重置，不打 ×120 | 0 | 无 |
| 不操作只走帧（20 步）×150 | 0.002 MiB | — |
| 竞技场里随机打（移动、射击、炸弹，最多 300 步）×120 | **0.20 MiB/局** | — |
| 混合房间里随机打 ×120 | **0.30 MiB/局** | — |

Python worker 的内存稳定（前后都是 8,856 MiB），Lua 堆稳定在约 950 KB。

**根因**（详见 [ABP_LINUX_REVERSE_ENGINEERING.md §15.11](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)）：严格等价模式短路了 `ImageManager::apply_frame_images`，而这个函数除了画，还负责回收透明渲染批次。
- 实体阴影和文字这类渲染路径每帧都会开批次。因为不回收，每帧都新分配，待处理表也只增不减；
- 释放图像时 `remove_image` 要遍历这张表，所以重置越来越慢。

各种组合在预热后每局的增长：

| 组合 | 每局增长 |
|---|---:|
| 原配置（两处都短路） | 0.27 MiB |
| 只短路 `apply_data` | 0 |
| 只短路 `apply_frame_images` | 38 MiB |

**修复**：`tools/stub_render_f.txt` 去掉 `apply_frame_images` 一行（2026-09-24 22:18 远端生效，旧版备份为 `.bak-20260924`）。
- 同种子同动作的轨迹哈希与原配置逐位相同，训练数据不变；
- 每局耗时略短；
- 训练中的实例在下一次回收重启时自动换成新配置，不用停训练。

处理办法（worker 端）：
- 每个 AB+ 实例打满 `--recycle-episodes`（默认 200）局后，在它作为备用实例准备下一局时，先在后台线程里重启进程再重置；
- 停进程时等它真正退出（超时则 SIGKILL），端口释放后再用同一端口重开；
- 日志里 `abplus/instance_recycles` 记录重启次数。

压力测试：每 3 局就重启，90 秒内重启 24 次，0 错误。训练中每实例 200 局重启一次，预计开销在 1% 以内。根因修复后仍保留回收，作为兜底。

**续训（abp-mix-01b）**：
- 21:41 停掉 abp-mix-01（停在第 936 轮），21:49 从第 900 轮（273.1 游戏小时）用 `--resume` 续训。
- 总量保持 2,500 游戏小时，即续训 `--game-hours 2226.9`，参数同上，加 `--recycle-episodes 200`。
- 刚开始时 54.3× 实时。按约 54× 算还需约 41 小时，预计 9 月 26 日下午 3 点左右（远端时间）完成。

**Steam 依赖**：每个 AB+ 进程启动时，可执行文件的 DRM 外壳都要找到正在运行、已登录的 Steam 客户端；找不到就执行 `steam.sh steam://run/250900` 拉起 Steam，然后自己退出。
- 2026-09-25 00:11 遇到过一次：Steam 在 23:19 到 00:11 之间退出，当时的定期评估 0 局，拉起的 Steam 停在登录窗口。
- 已在运行的训练实例不受影响，但下一次回收重启会失败：重试期间整批卡住，最后训练崩溃。
- 当时停掉了卡住的评估进程及其 worker，用户在远端桌面重新登录 Steam 后，回收和评估都恢复正常。
- 以后看到评估长时间 0 局，或实例日志里有 `SteamAPI_IsSteamRunning() did not locate a running instance of Steam`，先检查远端的 Steam 是否在运行且已登录。

停训练要用 SIGTERM，不能用 Ctrl-C。原因是后台（nohup/&）启动的进程继承了"忽略 SIGINT"，Python 就不装 KeyboardInterrupt。SIGTERM 之后，worker 发现管道关闭，会各自关掉自己的 AB+ 实例。

- 默认配置：16 个环境（32 个实例）、`--chunks 1`、每轮 1024 步 × 16 个环境 = 16,384 样本、batch 1024、2 epoch、ent 0.01、gamma 0.999、严格 FP32。
- 每 10 轮存一次 checkpoint；每 20 轮用 2 个 CPU 实例在 64 个 held-out 混合种子上评估。
- 常用参数：`--task-weights arena=…,normal=…,boss=…`、`--tasks-file none`（只训练竞技场）、`--json-obs`（回到桥接 v1）、`--game-hours N`（按游戏时停止）。

已知的待定项：
- 从模拟器热启动时，价值头是按 combat-v1 训练的，换到 combat-v2 后前几轮价值误差会偏大。
- 同步 PPO 在采样和训练之间串行，重叠两者需要改学习器。
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
- `--verify-steps N` 在每局前 N 步把增量推理和整窗前向逐项比对（本轮最大差都是 0.0）。
- 结果按种子追加写入 `results.jsonl`，中断后重跑会跳过已完成的种子。
