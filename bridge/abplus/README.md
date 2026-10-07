# AB+ 桥接与训练管线

目的：在 Afterbirth+ v1.06 原生 Linux 版（原版引擎，经 `abp_turbo` 改造加速）上训练和评估战斗策略；最初用来评估模拟器训练出的策略。协议和观测格式沿用 J460 的 IsaacRLBridge 0.2.0（combat_schema=3），`VisibleHistory` 和策略网络不用改。

引擎层（实例隔离、虚拟时钟、严格等价的 render-lite 模式）见 [ABP_LINUX_REVERSE_ENGINEERING.md](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)，本桥接查明的引擎行为在其 §15。本文只写设计、奖励定义和用法；实验都在 [EXPERIMENTS.md](EXPERIMENTS.md)，下文用编号（如 A2、C12）引用。

**最新同步状态（2026-10-07 22:35，Asia/Shanghai）**：第一台 3080 的 `fk2/runs/c67-run` 已完成 9885 更新 / 3000.1 游戏小时；最新实际评估为 u9600，32 局平均过 1.125 层，最终 `last.pt` 尚未评估。第二台 3090 的 `brn2` 已完成 B2 默认流程与噪声/快照对照，最新 `m3_nocrn` 选择头留出准确率 46.3%（41 例），尚未证明泛化有效。两机没有训练器或评估器；第一台 C67 的等待脚本仍在，本次未停止它。源码、模型、原始结果分别保存到本地 `runs/sync-20261007/host3080/` 与 `host3090/`；未用第一台较旧的实验源码覆盖工作区。详细结果见 EXPERIMENTS.md C67 和 B17 的 B2 补充。

**历史部署状态**（2026-09-30；后续按实验使用独立沙箱，不代表当前活动版本）：
- 桥接：主工作区是 abp-0.2.12（批量重放 `play`，Go-Explore 用，A8）。第一台部署的是 abp-0.2.9（C38、C40）；Go-Explore 另用一份主工作区的独立副本 `~/isaac-abplus/gx`（09-30 02:06，abp-0.2.12），见"Go-Explore"。第二台是 abp-0.2.11（C41），即 09-29 23:46 的主工作区，不含之后加的 `ABP_STARTSEED`、`ABP_SOUND_RESET`（A7）；C39 用的 abp-0.2.8-hp 留在第二台的 `~/isaac-abplus/bridge-abp-0.2.8-hp`。见"二进制观测与版本"。
- abp_turbo：第一台 09-30 已换成含 A7 钩子的新库，旧库留作 `.bak-20260930`；第二台还是旧库。主工作区的代码需要新库。源码又加了 `ABP_AL_STOPPED`（A8，默认关），这一版只编进了第一台的 `~/isaac-abplus/gx/tools`，给 Go-Explore 用；训练环境没有打开它。
- 远端：第一台 `ssh -p 2222 eolc@100.76.185.120`，第二台 `ssh rhythmo@10.19.131.132`。代码在 `~/isaac-abplus/bridge`，每次训练的输出在 `~/isaac-abplus/train/<运行名>`。两台各有一个训练看板。
- 实验进度和结论：EXPERIMENTS.md 的总览表。
- **2026-10-03 起的新方向**（用户要求：能随算力扩展的训练架构）：主工作区是 abp-0.2.14，加了进程克隆 `fork`、精简模式 `lean` 和 fork 采样器（见"克隆与精简模式""fork 采样器"，实验 A19、B9、B10）。这一版只放在第二台的沙箱 `~/isaac-abplus/fk`（代码、`abp_bridge.lua`、`tools/libabp_turbo.so`、`tools/stub_render_g.txt`、结果 `runs/`），两台已部署的 `bridge/`、`flr/`、`tools/` 都没动。旧的训练管线（`train_abplus.py`）不受影响：新功能都要显式打开。

**本文结构**：组成 → 桥接 → 房间与任务 → 奖励 → 训练 → Go-Explore → 运维 → 用法。

## 组成

| 文件 | 作用 |
|---|---|
| [`abp_bridge.lua`](abp_bridge.lua) | 游戏侧，现为 abp-0.2.15（`fork`、`lean`，见"克隆与精简模式"；0.2.15 给精简观测加了小地图，见"整层环境"）。runtime 副本 `main.lua` 的钩子在 `--luadebug` 下 `dofile(ABP_LUA)` 加载，不走 mods 目录；协议比 0.2.0 多 `{"cmd":"lua"}`、`format`（二进制观测 v2）、`profile`（观测分项耗时）、`play`（批量重放，见"Go-Explore"）；致命伤在 `MC_ENTITY_TAKE_DMG` 里拦下记为死亡（AB+ 没有 rewind）。实例加载的脚本：`launch_abplus(bridge_lua=...)`，否则环境变量 `ABP_BRIDGE_LUA`（worker 进程会继承，可以让测试副本和安装的那份同时跑），否则 `$ABP_HOME/bridge/abp_bridge.lua`（安装的那份）；Python 端 `BRIDGE_VERSION` 必须和它一致 |
| [`isaac_bridge/abplus.py`](../python/isaac_bridge/abplus.py) | `AbplusTrainingEnv`（竞技场、房间、练习场和对战场的重置）、`AbplusTransformerEnv`（Gymnasium 环境，deadline 观测）、`launch_abplus`/`stop_abplus`（起停实例）、`sim_arena`（逐位复刻 `rl/sim/src/arena.rs`） |
| [`abplus_eval.py`](../python/abplus_eval.py) | 多实例评估：每个 worker 一个 AB+ 实例加一份策略，按 `FrameSampler` 方式增量推理；指标同模拟器审计 `e2_reeval.py`；对战评估见"对战场" |
| [`abplus_transfer_report.py`](../python/abplus_transfer_report.py) | 同种子配对比较模拟器与 AB+ 的结局、行为指标和 Monstro 动画时长 |
| [`abplus_smoke.py`](../python/abplus_smoke.py) | 单实例随机动作冒烟测试和耗时拆分 |
| [`abp_turbo.c`](../../../analysis/scripts/abplus/abp_turbo.c) | 引擎改造层。2026-10-03 加了 `ABP_FORK`（克隆进程）和 `ABP_INPUT`（原生输入），见"克隆与精简模式"。getenv 钩子：`ABP_RESEED:<n>` 重播全局 MT19937；`ABP_STARTSEED:<n>` 让新一局的起始种子确定；`ABP_SOUND_RESET` 清掉音效的重放冷却（A7）。环境变量 `ABP_AL_STOPPED=1`：游戏读到的 OpenAL 音源一律"已停止"（A8，默认关） |
| [`train_abplus.py`](../python/train_abplus.py)、[`isaac_bridge/abplus_vec.py`](../python/isaac_bridge/abplus_vec.py)、[`isaac_bridge/abplus_worker.py`](../python/isaac_bridge/abplus_worker.py) | 训练管线：学习器、worker、共享内存帧 |
| [`isaac_bridge/gpu_ppo.py`](../python/isaac_bridge/gpu_ppo.py)、[`isaac_bridge/graph_sampler.py`](../python/isaac_bridge/graph_sampler.py)、[`isaac_bridge/transformer_policy.py`](../python/isaac_bridge/transformer_policy.py)、[`isaac_bridge/transformer_obs.py`](../python/isaac_bridge/transformer_obs.py) | 和模拟器共用的 GPU PPO、CUDA Graph 采样、策略网络和观测编码 |
| [`isaac_bridge/abplus_obs.py`](../python/isaac_bridge/abplus_obs.py) | 桥接 v2 二进制观测解码器；验证脚本 `abplus_probe_obs_v2.py` |
| [`isaac_bridge/abplus_tasks.py`](../python/isaac_bridge/abplus_tasks.py)、[`catalog/`](catalog/) | 房间级混合（房间目录、Basement I Boss 池、混合规格）；探测脚本 `abplus_room_catalog.py`、`abplus_probe_boss_pool.py`、`abplus_probe_rooms.py`；各任务文件见"任务文件" |
| [`isaac_bridge/abplus_reward.py`](../python/isaac_bridge/abplus_reward.py)、[`isaac_bridge/abplus_geometry.py`](../python/isaac_bridge/abplus_geometry.py) | 奖励（见"奖励"一节的表）；射击几何（d_fire、d_walk、射击方向标签、趋近标签、死角深度、被挡住的方向） |
| [`isaac_bridge/hit_rate.py`](../python/isaac_bridge/hit_rate.py) | combat-hitrate 的滑动窗口命中率，奖励和观测共用 |
| [`isaac_bridge/plr.py`](../python/isaac_bridge/plr.py) | 按房间类别的 PLR 选房 |
| [`isaac_bridge/abplus_groups.py`](../python/isaac_bridge/abplus_groups.py) | 并行任务组：分组文件、按步数分预算，见"并行任务组" |
| [`isaac_bridge/room_buffer.py`](../python/isaac_bridge/room_buffer.py) | Room Buffer 选种子，见"从头训练" |
| [`isaac_bridge/abplus_duel.py`](../python/isaac_bridge/abplus_duel.py)、[`catalog/duel_rooms.json`](catalog/duel_rooms.json) | 对战场：双方的第一人称观测、每局结局、对局 worker、脚本对手、地形；见"对战场" |
| [`isaac_bridge/abplus_goexplore.py`](../python/isaac_bridge/abplus_goexplore.py)、[`goexplore_abplus.py`](../python/goexplore_abplus.py)、[`goexplore_summary.py`](../python/goexplore_summary.py)、[`goexplore_replays.py`](../python/goexplore_replays.py)、[`abplus_probe_play.py`](../python/abplus_probe_play.py) | Go-Explore 第一阶段：cell、存档、选择、探索策略、返回核对、worker（库）；多进程驱动；运行汇总；最好轨迹的回放；`play` 与 `step` 的实机核对。诊断脚本 `abplus_probe_gxdiverge.py`（返回失败的 cell 在两种前史下逐步比较）、`abplus_probe_playstep.py`（同一轨迹在新实例里反复跑、比较终点，A8 用它查出音源状态）。见"Go-Explore" |
| [`isaac_bridge/steam_watch.py`](../python/isaac_bridge/steam_watch.py) | Steam 客户端检查和报警，见"Steam 依赖" |
| [`abplus_dashboard.py`](../python/abplus_dashboard.py)、[`isaac_bridge/abplus_dashboard.html`](../python/isaac_bridge/abplus_dashboard.html) | 实时训练看板，见"训练看板" |
| [`abplus_replay_view.py`](../python/abplus_replay_view.py)、[`isaac_bridge/abplus_replay.html`](../python/isaac_bridge/abplus_replay.html) | 把 `abplus_eval.py --replays` 的回放做成本地网页 |
| `abplus_bench_workers.py`、`abplus_bench_train_cpu.py`、`abplus_probe_obs_cost.py` | 环境侧吞吐上限、CPU 上的 PPO 训练开销、Lua 观测分项耗时 |
| `abplus_probe_*.py`、`abplus_bench_policy.py` | 诊断脚本，各起一个实例：`door`（LeaveDoor 与门）、`goto`（goto 后逐帧生成什么）、`champion`（冠军何时出现）、`roomseed`（房间种子与 MT19937）、`determinism`（固定动作逐帧比对）、`reset`（重开后的楼层、诅咒、耗时）；`abplus_bench_policy.py` 测 CPU/GPU 推理延迟 |
| [`isaac_bridge/fork_sampler.py`](../python/isaac_bridge/fork_sampler.py)、[`isaac_bridge/abplus_lean.py`](../python/isaac_bridge/abplus_lean.py)、[`abplus_bench_fork_sampler.py`](../python/abplus_bench_fork_sampler.py)、[`abplus_bench_standin_infer.py`](../python/abplus_bench_standin_infer.py)、[`abplus_bench_standin_train.py`](../python/abplus_bench_standin_train.py) | fork 采样器（每局是一个停好的起始状态的克隆体）、精简观测的解码、它的吞吐基准、替身网络的推理和训练速度；见"fork 采样器" |
| `abplus_probe_fork.py`、`abplus_probe_fork_stress.py`、`abplus_probe_fork_fan.py`、`abplus_probe_fork_rooms.py`、`abplus_probe_lean.py`、`abplus_bench_step.py`、`abplus_prof_step.py` | A19、B9 的探针：克隆体是否精确、连续 fork、并发克隆、克隆体里换房、精简模式和打桩是否改变游戏、单实例每决策耗时、游戏进程剖析（配 `analysis/scripts/abplus/prof_report2.py`） |
| [`isaac_bridge/tok_obs.py`](../python/isaac_bridge/tok_obs.py)、[`isaac_bridge/tok_policy.py`](../python/isaac_bridge/tok_policy.py)、[`isaac_bridge/tok_sampler.py`](../python/isaac_bridge/tok_sampler.py)、[`isaac_bridge/tok_floor.py`](../python/isaac_bridge/tok_floor.py)、[`train_tok.py`](../python/train_tok.py)、[`eval_tok.py`](../python/eval_tok.py) | token 策略这条线（C55 起）：每个决策的定长记录、策略网络和 CUDA Graph 采样、单房采样器和事后搜索教师、整层采样器和起点库、PPO 训练器、留出种子评估；见"token 策略与 fork 训练器"、"整层环境"。探针 `abplus_probe_hindsight.py`（受伤能不能靠换一个移动避开）、`abplus_probe_floor_lean.py`（整层的精简观测、克隆体跨房间是否精确） |
| `test_*.py` | 单元测试：`test_abplus_*`、`test_plr`、`test_hit_rate`、`test_groups`、`test_room_buffer`、`test_c39`、`test_duel`、`test_goexplore`、`test_gpu_infra`、`test_graph_sampler`、`test_steam_watch`；远端 venv 没有 pytest，用 `python -m unittest` 运行 |

## 桥接

### 竞技场重置（每局约 0.1 秒）

与模拟器 `arena.rs` 同种子得到同样的布局、入口门、玩家和 Monstro 出生点：

1. 用种子导出的值重播全局 MT19937，`restart 0`，重建玩家对象。
2. 设 `Level.LeaveDoor = (入口 + 2) % 4`，再重播，`goto s.boss.<布局>`，等 8 帧。
3. Monstro 是冠军（subtype ≠ 0）时换下一个房间种子，回到第 2 步。
4. 一段 Lua：清掉所有实体，去掉道具、钥匙、硬币，回满血，炸弹设为 1；放好玩家；清诅咒，特殊石头改回普通石头；`FireDelay = 0`，清受伤无敌帧；用种子重播，生成 Monstro，关门。
5. 逐帧推进到全部满足才开局：唯一可见敌人是 subtype 0 的 Monstro 且位置与模拟器一致；只有入口门；石头集合与模拟器布局逐格一致；诅咒为 0。

每一步对应一个实测问题（A1）。同种子、同动作序列的观测字段跨实例、跨时间逐帧一致（实体 id 和帧计数除外）。

普通房原先有例外（A7），约 15% 的局隐藏状态随实例的前史不同：
- `restart 0` 开的新局要到下一帧才从全局 MT19937 抽起始种子，这个种子跟上一局在哪个房间有关；
- 音效的重放冷却跨局保留，同一个动作可能多一次或少一次随机取数。

2026-09-30 起，重置在 `restart 0` 前依次调用 abp_turbo 的 `ABP_SOUND_RESET`、`ABP_RESEED`、`ABP_STARTSEED`，要求三个都返回 "1"。
- 需要新编的 `tools/libabp_turbo.so`（源码 analysis/scripts/abplus/abp_turbo.c），旧库会在第一次重置时报错；第一台已换，第二台还没换。
- 之后普通房 127/128 局在不同前史下逐步相同，同一实例再返回 99.4%；剩下的出在 Mulligan 躲人时，原因未找到。
- 先清空房间再摆放的场景（躲子弹场）完全确定。
- 剩下的一处在 A8 查明：游戏只在 16 个 OpenAL 音源里有空闲的时候才播音效，并按音源状态判断音效是否还在播；音频线程按真实时间播放，加速后这些答案随机器的时序和负载变化。个别房间（Embryo + Ulcer、Portal、Pooter + Stoney 等）里同一种子、同一动作会走到不同状态。abp_turbo 的 `ABP_AL_STOPPED=1` 让音源一律读作"已停止"，之后除了有 Mulligan 或 Portal 的房间（还有一处跨局问题，原因未找到）都逐次一致。训练还没有打开它（改变环境，由用户决定）；Go-Explore 默认打开。

### 普通房、Boss 房和练习场的重置

普通房和 Boss 房的重置与竞技场相同（重播 + `restart 0`，重播 + `goto`），但保留布局自然生成的敌人，Boss 冠军掷骰照常；然后执行玩家模板、Boss 血量随机化、清诅咒、闩门，再重播本局种子。入口槽不被接受就换下一个；刷新点有随机性，开局已清房时按种子确定性地换下一个候选房间（`RoomUnusable`），不重启实例。耗时、可复现性和一处未查明的例外见 A4。

瞄准场和对战场在此基础上清掉房里的实体，再放目标或对战 NPC，见"单个敌人的瞄准场"和"对战场"。

### 二进制观测与版本

每步重发静态地形和纯 Lua 的 JSON 编码约占 75%，读字段本身几乎不花时间（A3）。桥接 v2（abp-0.2.0）：
- 动态数据用 Lua 5.3 的 `string.pack` 打包（每条记录 0.26 µs）；整数一律 int64（重开 run 后玩家 `Index` 是 0xFFFFFFFF）。
- 地形和网格只在 reset/query 事件、lua/exec/reset 命令之后、换房间、任一格碰撞类型变化时发送；地形危险标志的动态部分（带接触伤害的特效）由客户端按 Lua 规则重算。
- `abplus_obs.py` 解码出与 JSON 路径相同的字典。一致性和提速见 A3。
- 原生读内存推迟：实体字段读取只占 0.01–0.15 ms/步，不值得冒结构偏移的风险。

之后的版本只在 combat 块和实体行加字段，客户端要求桥接版本完全一致：

| 版本 | 新增 |
|---|---|
| abp-0.2.1 | `blocking_hp`（挡门 NPC 总血量）、`blocking_points`（它们的击杀分之和） |
| abp-0.2.2 | `combat.blocking_count`（挡门怪数量）；远端旧版备份 `bridge/abp_bridge.lua.abp-0.2.1` |
| abp-0.2.3 | 谱系相关的计数和两个实体标志 |
| abp-0.2.4 | 无敌开关、`tear_hits`、`blocked_hits` |
| abp-0.2.5 | 谱系模式 3、`tear_misses`、`miss_units`、`miss_streak` |
| abp-0.2.6 | 空发上限 `AbpSetMissCap` |
| abp-0.2.7 | combat 块后的 `credits`（按发射帧记的伤害、击杀、空发） |
| abp-0.2.8 | 谱系成员死后以同一实体继续存在（Gaper 掉头变成 Gusher / Pacer）时，下一帧重新加入谱系，预算为它当时的血量；格式不变（C37） |
| abp-0.2.9 | 对战 NPC（`AbpDuelAttach`、`AbpDuelPlace`，step 命令的 `duel_move`、`duel_shoot`）；观测末尾加对战块（标志字节，没有对战时为 0；JSON 为 `obs.duel`）；远端旧版备份 `bridge/abp_bridge.lua.abp-0.2.8`（A6）。第一台部署的是这一版（C38、C40） |
| abp-0.2.8-hp | 在 abp-0.2.8 上只加 `combat.monster_damage`（所有怪物的真实掉血，引擎 `IsActiveEnemy` 口径），combat 块 18 个 double；和决斗分开做，只部署在第二台跑 C39；第二台部署前的代码备份在 `~/isaac-abplus/bridge-abp-0.2.8` |
| abp-0.2.10 | abp-0.2.9（决斗）加 abp-0.2.8-hp 的 `monster_damage`：combat 块 18 个 double，决斗块仍在最后。部署前先在测试副本上跑 A6 的 JSON 与二进制逐帧核对（`duel_check.py` 的 V 测试） |
| abp-0.2.11 | abp-0.2.10 加 `AbpSetStats`（C41）：房间设置时给玩家的移速、攻击力、弹速、射速、射程加修正，在 `MC_EVALUATE_CACHE` 里生效。弹速最低 0.6（引擎），射速取整到整数的 MaxFireDelay，射程改 TearHeight。修正全为 0 时行为和 0.2.10 相同。09-29 部署到第二台（C41），部署前跑过上一行的 V 测试 |
| abp-0.2.12 | 批量重放 `{"cmd":"play","actions":[code,…],"repeat":k,"repeats":[…],"stop_clear":bool}`（A8）：一串动作连着播，每个保持 repeat 帧，下一个动作在本该发 step 观测的地方施加，最后先回 `{"type":"ok","cmd":"play","played":n,"stop":"done"/"dead"/"clear"}` 再发一次 step 观测（credits 覆盖整批）；玩家死亡（含被拦下的致命伤）或 stop_clear 时清房就提前停；code = 移动 + 9 × 射击 + 45 × 炸弹 + 90 × 道具；不能和对战 NPC 一起用。其余路径不变 |

| abp-0.2.14 | `fork`（克隆进程）、`lean`（精简模式）；见下一节。格式 2 的观测不变 |
| abp-0.2.15 | 只改精简观测（整层环境用，C57）：房间号改为房间的 `SafeGridIndex`（大房间从哪个门进都一样）；门的第 4 个字节是"门后房间去过 / 已清"；出口活板门或楼梯出现时重发地形；新增小地图块（标志位 8）。其余同 abp-0.2.14 |

### 克隆与精简模式（abp-0.2.14，2026-10-03）

**克隆**：`{"cmd":"fork","host":h,"port":p,"tag":t,"alarm":秒,"reseed":n,"lean":bool}`。游戏进程 `fork()`（abp_turbo 的 `ABP_FORK`，机制见 [ABP_LINUX_REVERSE_ENGINEERING.md](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md) §7）：
- 父进程回 `{"type":"ok","cmd":"fork","pid":…}`，继续等命令，状态不变。
- 克隆体处在父进程读到这条命令时的状态。它丢掉继承来的 socket，连到 `host:port`，发 `hello`（`clone = true`），之后像普通实例一样接受命令，也可以再 `fork`。连接一断它就结束。
- `alarm`：看门狗的真实秒数（不给时用 `ABP_FORK_ALARM`，默认 120；0 为不限）。`reseed`：克隆体先重设全局 MT，同一状态的克隆体之后的随机数就不同。`lean`：克隆体进入精简模式。
- Python：`clone = bridge.fork(tag=…, alarm=…, reseed=…, lean=…)` 返回连着克隆体的新客户端（共用父客户端最后一帧观测和地形缓存）；`clone.close()` 结束克隆体。
- 实例必须用 `abplus.FORK_ENV` 启动（Mesa 软件 OpenGL、`LP_NUM_THREADS=0`、不要着色器磁盘缓存、OpenAL 空后端、`ABP_AL_STOPPED=1`），并用含 `ABP_FORK` 的 `libabp_turbo.so`。
- 克隆体带着父进程的全部随机数状态：不 `reseed` 时，同样的动作得到同样的未来。
- 精确性、代价和限制见 A19。

**精简模式**：`{"cmd":"lean","enabled":bool}`，或 `fork` 带 `lean = true`。给 fork 采样器的每局克隆体用：
- 不做每帧的 `update_combat` 和谱系、眼泪归因的登记；`combat` 块里的计数不再更新。
- 输入由 abp_turbo 直接回答引擎（`ABP_INPUT`），不进 `MC_INPUT_ACTION` 回调。
- `step`、`obs` 的观测是定长记录 `"L <字节数> <事件> <序号>\n" + 负载`，布局写在 `abp_bridge.lua` 的 `pack_lean` 上方，`abplus_lean.LeanDecoder` 直接读成 numpy 结构数组（玩家 36 个 double、门 16 字节、实体 108 字节）。带引擎的速度；奖励用的量只有玩家累计受伤、怪物总血量、挡门怪的血量和数量。
- 地形每步只查上次有东西的格子，每 30 个逻辑帧全查一次：打掉的石头、便便立刻更新，空格上新出现的东西最迟 1 秒后更新。
- 精简模式不改变游戏，观测与格式 2 逐字段一致（B9）。重置流程仍用原来的桥：采样器只让克隆体进精简模式。
- 原生观测（B11，含 `ABP_OBS_INIT` 的 `libabp_turbo.so`）：记录的固定部分和地形检查由 abp_turbo 在 C 里填，字节与 Lua 打包的相同；小地图块、地形 JSON、激光房仍由 Lua 打包。环境变量 `ISAAC_RL_NATIVE_OBS`：0 Lua，1 原生（默认），2 两者都算并比较。

### 与模拟器的已知差异

- Monstro 的随机数流不同（AB+ 用 MT19937，模拟器用 xorshift），配对的是起始局面，不是轨迹。
- **出场阶段是模拟器的缺陷**：AB+ 的 Monstro 生成后隐身 4 帧，再播约 25 帧 `Appear`，期间不行动（J460 人类录像 `runs/human/20260923-session01` 的 6 局开局也约 31 帧不行动）；模拟器 `new_monstro` 设了 state 1 却没设 `FLAG_APPEAR`（Gaper 设了），AI 从第 1 帧开始，被策略利用（A2）。
- **炸弹伤害是版本差异**：AB+ `Entity_Bomb::Init` 的普通炸弹伤害 60.0、引信 45 帧；J460 录像炸中 Monstro 一次是 100；模拟器用 100，与忏悔+ 一致。
- **`JumpDown` 时长也是版本差异**：AB+ 约 74 帧，J460 录像和模拟器 65 帧；其他动画时长基本一致。
- `range` 按 `260 × TearHeight / -23.75` 换算，对以撒是精确值；致命伤由桥接拦截（虚拟死亡），结局判定与模拟器相同。

## 房间与任务

### 房间目录

房间数据从运行中的引擎探测，不用仓库里的 Repentance+ 房间文件（A4）：
- `abplus_room_catalog.py` 逐个探测 `goto d.<id>`、`goto s.boss.<id>`：Basement I 有 **1,022 个普通房**（637 个 1×1），特殊房表有 **431 个 Boss 房**，覆盖全部楼层（`catalog/abplus_basement1_rooms.json`）。
- `abplus_probe_boss_pool.py` 让关卡生成器重开 300 次：Basement I 实际出现 13 种 Boss、82 个房间（`catalog/abplus_basement1_boss_pool.json`）。
- 选房由种子决定：任务 → 房间 → 入口；训练时可改用 PLR 或 Room Buffer（见"选房"）。

### 任务文件

`--tasks-file` 和 `--groups-file` 用的都在 `catalog/`：

| 文件 | 内容 | 用在 |
|---|---|---|
| `mixture_basement1.json` | 495 个普通房（1×1、开局未清、至少一个火堆以外的敌人）、77 个 Boss 房（上述 Boss 池里的 1×1 房间）和竞技场；默认权重 arena 0.2 / normal 0.45 / boss 0.35 | abp-mix-01 到 04（C2 起） |
| `mixture_round1_monstro_horf.json` | 竞技场 0.25、13 个 Monstro Boss 房 0.25、42 个含 Horf 的普通房 0.5 | combat-v5 第一、二轮（C12、C14） |
| `mixture_normal_rooms.json` | 只有 495 个普通房 | combat-hitrate-walk（C17） |
| `room_classes.json` | 495 个普通房的脚本分类 | C18 |
| `mixture_positioning_rooms.json` | 202 个要走位或怪不追击的普通房 | combat-hitrate-miss（C19 起），C37、C40 的训练和普通房评估 |
| `mixture_target_*.json`、`eval_detour_seeds_tier4.json` | 瞄准场各档、躲子弹场、绕路和死角测试集 | 见"单个敌人的瞄准场" |
| `parallel_groups.json` | 并行任务组的默认三组 | 见"并行任务组" |
| `scaling_groups.json`、`scaling_normal_rooms.json`、`mixture_target_horf_rocks.json`、`mixture_target_spawners.json` | 四类房间 | C39 |
| `duel_rooms.json` | 对战场的地形 | A6、C38 |

### 单个敌人的瞄准场（2026-09-27）

用户定的课程：先在只有一个敌人的场景里学瞄准，敌人分三档，达标再加下一档：不动的 Horf → 慢慢游走的 Clotty、Mulligan、Gusher → 追人的 Gaper、Attack Fly。达标线：采样策略在已对齐时朝对的方向射 ≥ 0.7。之后用这里的权重回到普通房。

- **场景**（`abplus.TARGET_LUA`、`target_cells`）：房间表有 `target` 字段（`{type, variant, min_cells}`）时，普通房任务改为照常进房、清掉玩家以外的所有实体，玩家和一个目标 NPC 各放在按种子选的内部格子上，间隔至少 `min_cells` 格。
- 目标在重置随机数之后刷出，刷出精英就重刷（最多 8 次）；NPC 前 4 帧不可见，设置完空跑 6 帧（`TARGET_SETTLE_FRAMES`）再开局。门封上、房间设为未清，目标死了就清房。
- `target` 由训练脚本写进 `config.json`，worker 和评估从那里读；评估加 `--target-from-tasks` 时改用 `--tasks-file` 里的 `target`。奖励用 combat-hitrate-miss，每局 30 秒（`--episode-seconds 30`）；辅助损失的设置见"训练选项"。

**房间表**（场景核对和训练见"实验"列）：

| 档 | 房间表 | 内容 | 实验 |
|---|---|---|---|
| 1 | `catalog/mixture_target_horf.json` | 63 个没有任何障碍物的 1×1 普通房，Horf（类型 12），间隔 ≥ 3 格；`--room-sampling mixture` 按种子均匀选房 | C22 |
| 2 | `catalog/mixture_target_tier2.json` | `target.types`：Horf、Clotty（15.0）、Mulligan（16.0）、Gusher（11.0），每局按种子均匀选一种（`abplus.target_kind`） | C23 |
| 3 | `catalog/mixture_target_tier3.json` | 再加 Gaper（10.1；10.0 是 Frowning Gaper）和 Attack Fly（18.0），六种每局均匀选一种 | C24 |
| 4 | `catalog/mixture_target_tier4.json` | 428 个有障碍物的 1×1 普通房（不含压力板、活板门），布局保留；`target.obstacles = true`：玩家和目标放在可走格子上，且玩家要能走到打得到目标的位置（`abplus.target_cells_walkable`），放不下换房间 | C26 |
| 5 | `catalog/mixture_target_tier5.json` | 只练绕路：第四档里绕路比例 ≥ 3% 的 207 个房间，只用 Horf，带 `detour_min`；训练时走路势每格 1.0（`--align-coef 1.0`） | C27 |
| 6 | `catalog/mixture_target_tier6.json` | `target.arms` 三组：第四档 40%（六种怪，428 个房间）；绕路 30%（第五档的设置）；死角 30%（Horf，死角深度 ≥ 30 像素，111 个房间） | C28 |
| 7 | `catalog/mixture_target_tier7.json` | 第六档的三组，每局再加 1–2 只六种之一的怪 | C29 |
| 8 | `catalog/mixture_target_tier8.json` | 第七档，但额外的怪 0、1、2 只各三分之一 | C30 |
| 躲子弹 | `catalog/mixture_target_dodge.json` | 第一档的 63 个房间，1–4 只远程怪（目标加 0–3 只额外的怪），每只从 Horf、Round Worm（244.0）、Pooter（14.0）里均匀抽；配 combat-hitrate-hurt，会受伤 | C35 |

- **绕路**：绕路比例是随机抽的玩家、目标格子里，走路距离比直线距离至少长 60 像素的比例，由引擎扫描得到（C27）。`target` 里的 `detour_min`（像素）要求开局的走路距离比直线距离至少长这么多，`attempts` 是按种子最多试几次（默认 64）；`detour_min` 默认 0，前四档不受影响。
- **死角**（`abplus_geometry.trap_depth`，`target` 里的 `trap_min`，像素）：从开局一直朝目标走，每步走到离目标直线距离最近的相邻可走格，直到不再变近；停下的格子不在射击带上时，从那里走到射击带途中最少要比停下处离目标远多少，就是死角深度。`trap_min` 要求开局的死角深度至少这么多。
- **分组**（`target.arms`）：每组有自己的比例（`weight`）、房间（`rooms`）和开局设置；每局先按种子抽组（`abplus_tasks.target_arm`），再从这一组抽房间，重抽房间时组不变；有 `arms` 时只能用 `--room-sampling mixture`。
- **额外的怪**（`target.extras`，也可写在某一组里）：`{counts, types, min_cells, attempts}`；每局从 `counts` 均匀抽数量、从 `types` 均匀抽种类（`abplus.target_extras`）；每只放在可走的内部格子上，离玩家至少 `min_cells` 格、离其他怪至少 2 格，且玩家能走到打得到它的位置；`TARGET_LUA` 在目标之后生成，遇到精英重刷。
- **绕路测试集** `catalog/eval_detour_seeds_tier4.json`：第四档里 12 个必须绕路的 Horf 开局（C26）。别的档的 checkpoint 用 `abplus_eval.py --tasks-file catalog/mixture_target_tier4.json --target-from-tasks` 跑（`--target-from-tasks` 让评估用任务文件里的 `target`，而不是 checkpoint 里的）。
- **死角测试集** `catalog/mixture_target_trap.json`：只有死角组，种子取 `range:2147493000:24`，同样加 `--target-from-tasks`。

### 对战场：自对弈（abp-0.2.9 起，用户 2026-09-28 提出；结果见 C38、C40）

**对手**是桥接里的对战 NPC。设置脚本生成一只 Pacer（11.1：只会乱走，不攻击，死后不留东西），调用 `AbpDuelAttach`；之后每步命令里的 `duel_move`、`duel_shoot`（取值和玩家的移动头、射击头一样）逐帧控制它，引擎自带的 AI 被覆盖。它按玩家的实测数值校准（A6）：
- **移动**：玩家的移动规律，每逻辑帧两个物理子步。每个子步速度先衰减再加速：沿输入方向的分量乘 0.8803（和输入反向时乘 0.845），垂直分量乘 0.78，再加 0.528 × 输入。
  - NPC 的速度指令当帧生效，位移恰为指令的 0.75 倍，所以能逐帧复刻。
  - 撞墙时，被挡住的那个分量清零。
  - 每帧末把它的碰撞圆推出挡路的格子：引擎会让斜着走的 NPC 比玩家更深地钻进墙角。
- **射击**：
  - 11 帧一发，和玩家的射击间隔一样；
  - 弹速 10 像素/帧，加 1.2 倍自身速度；
  - 出生在前方 10 像素、左右交替偏 3–5 像素；
  - Height −23.75、FallingSpeed 0.13，射程和眼泪一样（27 帧、280 像素）。
- **命中半径**：NPC 的 Size 改为玩家的 10，子弹的 Size 改为眼泪的 8.165。不能改 Scale：它不改半径，还会把伤害变成一整颗心。
- **血量、受伤**：
  - 血量 21，挨 6 发眼泪死；玩家 3 颗心，挨 6 发子弹死。
  - 受伤后 29 帧不再掉血，这期间打来的子弹照样被吸收，和玩家的伤害冷却一样。
  - 致命一击被拦下、记为死亡，和玩家一样。
  - 眼泪打中它时推它 0.3 × 眼泪速度，和玩家被子弹推的量一样。
  - 质量改为玩家的 5，身体相撞时互推的量一样。
- **出场阶段**：新生成的 NPC 约 20 帧里引擎不更新它。重置时等到它连续两帧被更新才开局，所以双方从第一步起都能动、能射。
- **记账**：桥接对双方用同一套计数：发射数、造成伤害的发数（命中）、没造成伤害就消失的发数（空发，含连续空发和空发单位，空发上限对双方都适用）、造成的伤害、受到的伤害。

**观测**（`abplus_duel.duel_views`）：两边各一份第一人称观测，格式和其他任务完全一样，所以策略、动作头、奖励代码都不用改。
- 玩家这一侧：就是原观测，只是 combat 计数换成对战里玩家一方的。
- NPC 这一侧是角色互换：
  - players[0] 是 NPC：它的位置、Size、剩余命中次数折成半心，其余属性抄玩家的；
  - 玩家变成一个实体，用 NPC 在玩家视角里的那条实体记录，所以种类、标志、碰撞字段都一样；
  - NPC 的子弹变成自己的眼泪，玩家的眼泪变成对方的弹幕，各带那一种的碰撞字段；
  - combat 计数是 NPC 一方的。
- 两边的血都按 NPC 的血量单位算（玩家的半心 = 3.5），所以一次命中在两边都是 3.5。
- `test_duel.py` 检查：把同一局的角色互换后，一方看到的编码帧和另一方完全相同。实体行按集合比较，因为策略的实体注意力和实体顺序无关。

**奖励**：两边用同一个奖励类，各一个实例，各喂自己的观测。默认沿用 C37：
- combat-hitrate-miss：命中每 1.25 血 +1，时间价格随命中率变，连续空发有惩罚；
- `--hurt-ends-episode` 加 `--hurt-rest-cost`：任一方第一次受伤，这一局就结束：
  - 受伤的一方结局是 hurt，并扣掉剩余时限的时间价格；
  - 另一方结局是 win；
  - 同一步两边都受伤，两边都是 hurt。
- 截断时限默认 30 秒。

**地形**（`catalog/duel_rooms.json`，按种子抽一个分支）：
- open：C22 的 63 个空房；
- rocks：第四档房间里只有石头、铁块、坑、锁、雕像的 292 个。有便便、TNT、尖刺、蛛网的房间去掉了：眼泪能打碎便便、引爆 TNT，NPC 的子弹不能；尖刺和蛛网只影响玩家。
- 双方出生在内部相距至少 4 格的两个格子上，而且不在同一行或同一列（用户 09-28 的决定：谁都不动就打不到对方；`same_line: true` 可以放开）。rocks 的两格都要能走，而且第一方能走到打得到第二方的位置。两个出生格以 1/2 概率互换。

**训练**：`train_abplus.py --duel-file catalog/duel_rooms.json --room-sampling mixture`
- 每一局 AB+ 占两个环境槽位：2k 是玩家，2k + 1 是 NPC。一个 worker 进程管一局（主实例加备用实例）。学习器一次推理同时给两边出动作，用当前权重自对弈。`--envs` 数的是槽位，必须是偶数。
- 游戏时长按局算：每步 n 个槽位 = n / 2 局 × 2 逻辑帧；`--game-hours` 也按局算。
- 日志：
  - 分边：`duel/<player|npc>/{win,hurt,death,time_limit}_rate`、`return`、`hit_rate`、`shots_per_s`、`first_hit_s`；
  - 整局：`duel/episode_s`、`duel/decided_rate`、`duel/both_hurt_rate`；
  - `episodes.jsonl` 每局带 `side`。
- 每个 checkpoint 的对战评估写到 `evaluations-<方式>/`：
  - `selfplay`：自对弈，取最大；
  - `selfplay-sampled`：自对弈，采样；
  - `scripted`：对脚本对手 `abplus_duel.ScriptedDuellist`。它走到滑行终点正好对齐的位置，不横滑时才开火，不会躲；
  - `vs-start`：对训练起点（`--warm-start` 的策略），一个不变的参照。
- 迁移评估（`--duel-transfer`，默认 `rooms,dodge`，`none` 关掉）：同一个 checkpoint 回到 C37 自己的评估上，取最大和采样，设置和 C37 完全一样，数字可以直接比：
  - `evaluations-rooms/`：202 个走位房，180 秒；
  - `evaluations-dodge/`：躲子弹场，30 秒。

**评估**：`abplus_eval.py --duel-file ... --opponent self|scripted|<checkpoint 目录>`
- 按游戏规则打：一方死亡或到时限才结束。
- 记录先命中的一方（也就是训练时 hurt 结束的那一刻）、胜者、两边的命中、发数和受伤。
- 对脚本对手或另一个 checkpoint 时，每个种子打两局，被评估的 checkpoint 各当一次玩家和 NPC。
- 回放页（`abplus_replay_view.py`）另外画出 NPC 的输入箭头和血量。

**速度**（A6）：8 局 16 槽 21–29 倍实时，16 局 32 槽 31–32 倍。学习端限速：每局每步两个样本。

**已知差异**（A6）：
- 贴着石头角急转时，两边的滑动略有不同；
- 子弹的高度曲线比眼泪低约 0.2；
- 玩家被打的那一帧有碰撞分离造成的位移，NPC 没有；
- 身体相撞时的速度冲量只作用于玩家。

## 奖励

### 一览

`--reward-profile` 选奖励，默认 combat-v5。换奖励只能 `--warm-start`（只取权重），`--resume` 会拒绝奖励不同的 checkpoint。

| profile | 要点 | 定义和实验 |
|---|---|---|
| combat-v1 | 模拟器的奖励 | — |
| combat-v2 | 血量曲线扣血、死亡 −5、按游戏计分的时间和清房分、进度势 | 下文；C1 |
| combat-v3 | 挡门怪掉血当场给、时间价格随存活怪数和时长上涨、停滞惩罚 | 下文；C2–C5 |
| combat-v4 | 阶段一：清房 +30 才结算、超时截断 | 下文；C6、C10 |
| combat-v5（默认） | 谱系怪掉血和杀怪、对齐势；配分解动作头和几何辅助头 | 下文；C11–C15 |
| combat-hitrate | 无敌，命中率决定时间价格 | 下文；C16 |
| combat-hitrate-walk | 再加走路距离的对齐势 | 下文；C17 |
| combat-hitrate-miss | 召唤出来的小怪也计奖励（谱系模式 3）、连续空发惩罚 | 下文；C18、C19；之后的瞄准场（C22–C31）和 C37、C38、C40 都用它 |
| combat-hitrate-fire | 命中、击杀、空发记到发射那一步 | 下文；C20 |
| combat-hitrate-hurt | 阶段二：去掉无敌，受伤按血量曲线，死亡扣剩余时间 | 下文；C33–C36 |
| combat-hp | 所有怪物掉血、清房 +100、期限和死亡 −100 | "从头训练"一节；C39 |

受伤相关的开关（都在"阶段二"小节）：`--no-death-cost`（C36）、`--hurt-ends-episode`（C37）、`--hurt-rest-cost`（C38、C40）；无敌用 `--invincible` / `--no-invincible`。各奖励的参数（`--hit-hp`、`--miss-cost`、`--miss-cap`、`--align-coef`、`--lineage-mode`、`--time-cost-hit`、`--time-cost-miss`）见各小节。

### combat-v2

替换 combat-v1（伤害项按 `IsEnemy` 实体个数计分）的原因见 C1。击杀分和房间分取游戏自带的每日挑战计分（AB+ 1.06 反编译逐项核对，与 huijiwiki 一致）：击杀分 ⌈5·MaxHP^0.2⌉，只在 `Room::IsFirstVisit()` 且 `GetSpawnGridIndex() >= 0` 时计入（`Entity_Player::TriggerEnemyDeath`）。

- **受伤**：V(h) = 11.94·f(h/6)，f(x) = (x + 1 − (1−x)⁴)/2（OpenAI Five 的血量曲线）；3 颗心满血时依次掉半心的代价为 1.00、1.06、1.29、1.80、2.70、4.09，超过 6 个半心的部分每个 1。
- 受伤按桥接 `player_damage`（`GetTotalDamageTaken()` 的增量，玩家 +0x28b4）计：`TakeDamage` 按伤害量累加，排除标志 0x20（扣红心的自伤）和 0x10000（诅咒房门）；被桥接拦下的致死一击按伤害量计入，自己的炸弹也算。
- **死亡**额外 −5；**时间**每游戏秒 −1/170，全程计（120 秒约 −0.71）；**炸弹**每个 −0.08；**超时** 120 秒 −1，按终止处理（同 combat-v1）。
- **清房**：(房间分 + 开局挡门敌人的击杀分 + 过层分) ÷ 200；普通房的房间分 40；Boss 房（含竞技场）100，另加过层分 500。
- **进度**（势函数）：挡门 NPC（`CanShutDoors`）总血量每减少 70 记 +1，基础泪弹每发 0.05，刷怪、回血计负；死亡或超时时付清剩余势能，每局总和固定，不改变最优打法。

验证见 C1。日志：`episodes.jsonl` 每局带 `reward_components`；每轮记 `task/<类型>/episodes|win_rate|return` 和 `reward/<分项>` 的均值。

### combat-v3 与开局炸弹（2026-09-25）

换奖励的原因见 C2、C3。用户决定：从头训练、换奖励、开局炸弹随机，房间配比改用 PLR（见"选房"）。

**combat-v3**（单位仍是半颗心，训练时 × 0.25，日志分项不缩放）：
- 掉血同 combat-v2；超时 −20；死亡 −25，并补扣"从现在起不再打中"时剩余时间和停滞本该扣的分。排序：清房 > 超时 > 死亡。
- 时间每秒 −(1/120 + 存活挡门怪数/60) × (1 + (t/60)²)；停滞：连续 20 秒没打掉挡门怪的血（从开局或上次打中算起）时 −10，之后每秒 −1/6，直到下次打中（C4、C5）。
- 挡门怪每掉 20 点血 +1（一发普通眼泪约 +0.175）、每少一只 +1，当场给，新刷的怪和回血出现时先扣；清房普通房 +3、Boss 房和竞技场 +6；炸弹每个 −0.1。
- 观测：玩家 token 上加 `combat` 项（`transformer_obs.COMBAT_FIELDS`：存活挡门怪数/10、是否已打中过、挡门血量占开局的比例、距上次打中的秒数/60）；AB+ 帧格式 `FRAME_DTYPE_COMBAT` 在模拟器格式后追加（模拟器 ABI 不变），`decode_frame` 按环境的 `frame_dtype` 解码。

**开局炸弹** `--start-bombs 0.5:3`：一半的局 0 个，另一半 1–3 个，由种子决定（`abplus_worker.sample_bombs`）；评估按同一抽法、每个种子固定。

### 两阶段训练，阶段一：combat-v4（2026-09-25）

两阶段方案（由来见 C5、C7）：阶段一以奖励为主，学会进攻清房；阶段二以保命为主，学会躲避。

**combat-v4**（训练缩放 0.1，γ 从 0.999 改为 0.9995）：
- 挡门怪每掉 20 点血 +1、每少一只 +0.25（差分，新刷的怪和回血先扣后还）；受伤每次 −0.1（平坦，只用于信用分配）；死亡 −0.5，不结算。
- 清房 +30 − 0.5 × 掉的半心数 − 0.5 × 用掉的炸弹数，只在清房时结算；30 大于任何房间"击中 + 杀怪"的总和（开局挡门血量最多的 Boss 房 1047 共 415 点，折合 20.75）。
- 超时截断（价值函数自举），不惩罚。去掉时间代价、停滞惩罚及其观测；房间状态只留挡门怪数量和血量比例（`COMBAT_FIELDS_V4`），观测去掉 `remaining_time`；每局记录是否被截断。

测试和 200 游戏小时的 abp-mix-04 见 C6，之后的三项检查见 C10。

### 阶段一修订版：combat-v5（2026-09-26，`--reward-profile` 的默认值）

由来见 C10（刷怪的差分让打 Nest、Mulligan 的那一步拿到负奖励）。原则不变：躲着不打为 0；同一房间里任何清房都胜过任何不清房；没有按步扣分；超时截断。同时引入的几何观测、分解动作头和辅助头见"模型与观测"。

**奖励**（`isaac_bridge/abplus_reward.py` 的 `CombatV5`，训练时 × 0.1）：
- 击中：谱系怪每掉 5 血 +1（一发眼泪 +0.7；第一轮每 10 血，`--hit-hp`），按 `combat.lineage_damage`，任何来源都算，每只最多算它加入谱系时的血量。
- 杀怪：每只谱系怪 +0.25（`combat.lineage_kills`）。
- 对齐势：γΦ(s′) − Φ(s)，Φ = −0.2 × d_fire / 40；清房和死亡步 Φ(s′) = 0，截断步照常。
- 受伤、死亡、清房、清房扣血、清房扣炸弹、超时同 combat-v4；刷怪、回血、变身为 0。

**roster 谱系**（abp-0.2.3）：开局设置完成时房里所有挡门怪构成 roster，同一实体的 Morph 自然保留；继任者是死亡那一帧、在死亡位置 60 像素内生成的挡门怪（AB+ 里死亡时生成的实体 SpawnerEntity 为空，用户原规格的 SpawnerEntity 判定区分不了，C11）。`lineage_mode`：
- 0：不继承；
- 1（默认，用户 2026-09-26 定）：死亡留下的全部继承，包括 Nest 变成的 Big Spider / Trite、Big Spider 分出的 Spider、Mulligan 放出的苍蝇；
- 2：只有死亡恰好留下一只时才继承（变身）；第一轮两个臂用它；
- 3（abp-0.2.5，combat-hitrate-miss 的默认）：不看继承，所有挡门怪第一次出现就加入，召唤出来的也算。

模式 0–2 下，活着时刷出的怪（Nest 靠近时刷的蜘蛛）从不继承。模式 1、2 的差别和引擎核对见 C11。

**死后变身**（abp-0.2.8）：Gaper 掉头变成 Gusher / Pacer 时，游戏先触发死亡回调，再让同一个实体（Index、InitSeed 不变）继续活着。0.2.7 及以前这个键在死亡那一帧已经记为离开，永远回不到谱系：打它没有命中奖励，打中它的眼泪还算空发，C37 因此学会了去撞它（C37）。0.2.8 起模式 1–3 下它从死亡后的下一帧重新加入，预算为当时的血量；它之后的死亡再算一次击杀。

**d_fire**（`abplus_geometry.py`，上限 600 像素）：到最近有效射击位置的距离。
- 目标：活着的谱系怪，没有时取全部挡门怪；容差 τ = 碰撞半径 + 眼泪半径 8.16。
- 水平射击带：|y − e_y| ≤ τ，x 在目标左右各 R（玩家射程 260）内，截断在目标所在行两侧第一格永久挡眼泪的格子（未炸碎的石头、铁块、锁块、雕像；便便、TNT、坑不截断）；垂直带对称。
- 对两条带、所有目标取最小；不看目标行列以外的遮挡和无敌阶段；画面里没有目标时保持上一帧。第一轮不截断，第二轮起截断（C13、C14）。

实现核对、第一轮（熵 0.003 对 0.01）、第二轮（障碍几何 + 击中翻倍）和超时回放见 C11–C15。

### combat-hitrate：命中率决定时间价格（2026-09-26，临时测试）

玩家无敌、不给炸弹、180 秒截断（自举），训练时 × 0.1，各项在发生那一步给：
- 击中、杀怪同 combat-v5（谱系怪每 5 血 +1，每只 +0.25）；时间每游戏秒 −(2 − 1.5 × 命中率)；没有清房、对齐势、受伤、死亡、炸弹项，清房只是让时间惩罚停下。
- 命中率（`isaac_bridge/hit_rate.py`）：最近 3 秒内有效命中的眼泪数 / 发射的眼泪数，上限 1，没射为 0。发射数取桥接 `events.tears`（`MC_POST_FIRE_TEAR` 计数）；有效命中取 `combat.tear_hits`（眼泪对活着、可受伤的谱系怪造成伤害，每颗最多算一次）。命中按落地时刻、发射按发射时刻计，开始或停止射击时滞后一个眼泪飞行时间（射程 260 时最多约 0.9 秒）。命中率也是观测（`combat` 第三项 `hit_rate`，`COMBAT_FIELDS_HITRATE`），与奖励同一个类、按 `logic_frames` 计时。
- 无敌：设置脚本每局调用 `AbpSetInvincible(true)`，`MC_ENTITY_TAKE_DMG` 对玩家返回 false，`combat.blocked_hits` 记被取消的次数；reset 时关掉。
- `--reward-profile combat-hitrate` 默认 `--invincible`、`--episode-seconds 180`、`--start-bombs 0`（炸弹动作被掩码）；`--time-cost-hit`、`--time-cost-miss` 改两端价格。观测、分解动作头和几何辅助头沿用 combat-v5；评估同样无敌、0 炸弹、180 秒（按 checkpoint 的 config）。
- 日志 `behavior/hit_rate`（本轮结束的局的总命中数 / 总发射数）、`behavior/shots_per_s`；每局 `stats` 多 `shots`、`tear_hits`。

引擎核对和训练结果见 C16。

### combat-hitrate-walk：走路距离的对齐势（2026-09-26）

combat-hitrate 加走位势，只用普通房（`catalog/mixture_normal_rooms.json`）：
- Φ(s) = −0.2 × d_walk / 40（`--align-coef`，每格 0.2），每步 γΦ(s′) − Φ(s)（γ 同学习器）；清房步 Φ(s′) = 0，截断照常。
- d_walk（`abplus_geometry.fire_geometry(walk=True)`，上限 600 像素）：玩家走到最近的能打中位置（沿用射击带）要走的距离。直线距离会把玩家引到石头前卡住（C15）。
  - 在可走格子上做 8 方向多源最短路（不能斜穿被挡住的拐角）。起点是射击带经过的格子，初始代价是格子中心到射击带的直线距离。
  - 玩家的距离取自己所在格子和相邻格子的最小值（到格子中心的直线距离加上该格的值）；射击带经过玩家所在的格子时，直接用到射击带的直线距离。
  - 都走不到时取上限，势函数是平的。
- critic 的 `fire_distance` 改为 d_walk / 40，趋近标签改为"哪些能走上可走格子的移动会让 d_walk 变小"；每步约 0.3 ms（最多约 0.7 ms），观测和奖励共用一次计算（`_WALK_MEMO`）。

引擎核对和训练结果见 C17。

### combat-hitrate-miss：全部小怪计奖励与连续空发惩罚（2026-09-26）

在 combat-hitrate-walk 上改两处，房间只用需要走位或怪不追击的：
- 谱系模式 3（`--lineage-mode 3`，本奖励默认）：召唤出来的小怪也有奖励。
- 空发：第 k 次连续空发 −0.01 × min(k, 20)（`--miss-cost`、`--miss-cap`，`--miss-cap 0` 不设上限），下一次命中后归零。眼泪消失时没打中过谱系怪算一次空发，在消失那一步扣；桥接字段 `tear_misses`、`miss_streak`、`miss_units`，上限由设置脚本每局调用 `AbpSetMissCap(k)`，reset 时关掉。不封顶时 n 次连续空发共扣 0.01 × n(n+1)/2（C18）。
- 观测：`combat` 在 hit_rate 后加 `miss_streak / 20`（上限 100，`COMBAT_FIELDS_MISS`）。
- 房间：`catalog/room_classes.json` 是 C18 的脚本扫描（站桩、走位脚本能否清及用时，开局怪数和追击数，15 秒内召唤的怪数和血量，召唤怪奖励上限 `spawn_income`；`solve` 取 turret / needs_positioning / neither，`non_chasing` 要求开局至少一只怪且都不追击）。`catalog/mixture_positioning_rooms.json` 取 needs_positioning 和 non_chasing 的并集 205 个，去掉能刷怪的 878、318、881（命中率分别高于 64%、76%、88% 时刷召唤怪比清房合算），剩 202 个，原因在 `excluded`。
- 日志 `behavior/miss_share`（空发数 / 发射数）；每局 `stats` 多 `misses`。

引擎核对、房间扫描和冒烟见 C18，上限的核对和训练见 C19。

### combat-hitrate-fire：命中记到发射那一步（2026-09-27）

各项同 combat-hitrate-miss，但每颗眼泪的命中、击杀、空发记到发射那一步（眼泪要飞 0–14 步，C19、C20）：
- 桥接 abp-0.2.7 按眼泪记账：发射帧 = `MC_POST_FIRE_TEAR` 时的 `logic_frames + 1`（`MC_POST_UPDATE` 在实体更新之后才计数）；掉血按命中先后、每颗最多扣它的伤害，记到打中它的眼泪；击杀记到 6 帧内最后打中它的眼泪；空发记到那颗眼泪；找不到眼泪的记到当前帧。每步观测带 `combat.credits`（`[发射帧, 伤害, 击杀, 空发单位]`，按帧排序；二进制在 combat 块后，u16 条数加每条 `<qddd`）。
- `abplus_reward.CombatHitRateFire`：credit 换算为 伤害 / 5 + 0.25 × 击杀 − 0.01 × 空发单位，按发射帧归到 j 步之前（j = 1–32），存入帧记录的 `credit[j − 1]`（已乘 0.1）。
- 学习器（`gpu_ppo`、`GpuHistoryRolloutBuffer`）在算 GAE 前用 `apply_credits` 统一挪到发射那一步（本轮之前发射的留在原处），总回报不变；日志 `train/credit_moved`、`train/credit_kept`。

引擎核对、速度对照和训练见 C20。

### 阶段二：combat-hitrate-hurt 与受伤选项（2026-09-27 起）

用户定的阶段二第一版（C33）：同样 202 个房间，只去掉无敌。

- **奖励** = combat-hitrate-miss，再加：
  - **受伤**：combat-v2 的血量曲线。从 h 个半心掉 n 个扣 V(h) − V(h − n)：满血时第一个半心 1.00，越残越贵，最后一个 4.09。按游戏的 GetTotalDamageTaken 计数（`combat.player_damage`）。
  - **死亡**：终局，扣掉剩到截断时间的时间成本（按不命中时的价格，每秒 2），再扣 5，所以死亡永远不比拖到超时便宜。
  - **死亡时走路势不结清**：保持当前值；清房时照常结清。
- **玩家**：不无敌，0 炸弹。`--player-hp-prob 0.5 --player-hp-min 3` 让一半的局开局 3–5 个半心；评估一律满血。
- **评估**：`abplus_eval.py --mortal` 让在无敌下练出来的 checkpoint 也按会受伤来评估。
- **注意**：桥接在致命那一下就结束一局，那一帧的心数还没更新，所以按受伤计数器算，不要按最后一帧的心数算。
- **去掉死亡项**（`--no-death-cost`，C36）：致命那一下仍结束这一局，掉的半心照血量曲线扣，但不再扣剩余时间成本和 5；这样死亡也会让时间价格停下。
- **受伤即结束**（`--hurt-ends-episode`，C37）：训练时第一次受伤就结束这一局，结局记为 `hurt`，作为终止（不自举），其他什么也不扣。玩家必须不无敌。一般配 combat-hitrate-miss 加 `--no-invincible`，这个 profile 没有受伤项和死亡项。评估不受影响，仍按游戏规则算。
- **受伤扣剩余时间**（`--hurt-rest-cost`，配 `--hurt-ends-episode`，combat-hitrate-miss / -fire，用户 2026-09-28 定）：结束这一局的那次受伤再扣剩到截断时间的时间成本，按不命中时的价格（`--time-cost-miss`，默认每秒 2），记在奖励分项 `rest`。它等于 combat-hitrate-hurt 的死亡项去掉常数 5，所以受伤结束永远不比站到截断便宜。量级：180 秒截断下，第 4 秒受伤扣约 352（训练里乘 0.1），而 C37 一局的平均回报约 17。C38、C40 和 C40 的对照用了它。C40 的对照里，这套设置几个小时就让 C37 变慢、变稳；推测（未验证）主要来自这一项。
- **原计划里还没做的两项**：清房奖励换成实测的 V_next(结束时血量, 炸弹, 层数)，失败为 0；把 (λ_h, λ_b) 作为策略输入、训练时随机采样，让外层规划器下发风险偏好。

引擎核对和训练见 C33。

## 训练

### 结构（2026-09-24）

用户决定：RL 只负责战斗，导航和道具选择先用规则；房间级混合学习；PPO 在 GPU 上由独占进程完成，所有 worker 每步拼成一批，减少通信损耗（B1）；做桥接 v2；不开 TF32。

```text
学习器进程（独占 CUDA）: GpuMaskablePPO + AbplusFrameVecEnv（abplus_vec.py）
  每步 1 次批量推理（--chunks 1 = 全部环境一批）→ 每个 worker 13 字节 → 等全部回应
  → 从共享内存拷回每个环境最新一帧 FRAME_DTYPE（72,664 B）→ pinned 槽 → 异步上传 GPU
worker 进程 × N（abplus_worker.py，不导入 torch）: 一个环境槽 = 两个 AB+ 实例
  活动实例打当前局；备用实例在后台线程里按下一局的种子重置 → 局间切换 0.02–0.04 ms
AB+ 实例 × 2N: 严格等价模式 + 桥接 v2 二进制观测
```

- 学习器完全复用模拟器的 GPU PPO（帧级 GPU 缓存、帧去重分段 minibatch、checkpoint）；`AbplusFrameVecEnv` 与 `GpuFrameVecEnv` 的 chunk 接口相同。worker 用 `VisibleHistory.encode` 只编码最新一帧，与 `VisibleHistory.append` 的窗口末帧逐字段相同（A3）。
- 种子同 `FrameChunk`：环境 i 的第 k 局用 `base_seed + i + N·k`。训练种子 < 2³¹；held-out 种子 ≥ 2³¹，其中 E2 验证块和 2147500000 起的最终测试块不参与训练。
- 开局随机化同模拟器训练（默认）：50% 的局 Boss 血量 10–100%，25% 的局玩家 3–5 个半心；评估一律满血。
- 额外日志：`game/hours_*`（决策数 × 每决策帧数 ÷ 30 帧/秒）、`game/speed_x_realtime`（相对实时的倍率）、worker 单步耗时、局间切换等待、错误数。

### 模型与观测（combat-v5 起）

**观测**：实体行加两位（是否谱系、是否挡门）；fire_distance = d_fire / 40（上限 15），只进价值分支；辅助头标签 aim_label（能打中已对齐目标的射击方向，0 = 无）和 approach（9 个移动方向中哪些让 d_fire 变小），策略不读；上一步动作改为 9 + 5 + 2 + 2 的 one-hot。房间状态（`combat` 项）随奖励而定，见各奖励小节。

**模型**（`transformer_policy.py` 的 `GeometryPolicy`）：
- 移动 9 类、射击 5 类，加炸弹、主动道具两个二元头；对数概率和熵相加，掩码不变；worker 协议仍是 joint，学习器发送前换算。
- fire_distance 不进主干，`SplitMlpExtractor` 只把它给价值分支。
- 几何辅助头：主干特征接小 MLP，输出 5 类射击方向和 d_fire / 40 回归，损失权重 0.2（交叉熵 + 平方误差），与 PPO 一起更新主干。
- 熵系数默认 0.003（`--ent-coef-heads move=0.005,shoot=0.002` 分头设置）；γ 默认 0.9995。
- 模型大小可调，见"并行任务组、模型规模与在线蒸馏"。

### 选房

- **按种子**（`--room-sampling mixture`）：任务 → 房间 → 入口都由种子决定；瞄准场有 `arms` 时只能用它。
- **PLR**（`isaac_bridge/plr.py`，Jiang et al. 2021，`--room-sampling plr`，默认）：
  - 每个房间是一个 level（竞技场、495 个普通房、77 个 Boss 房）；每轮用 GAE 的正优势均值（positive value loss）给打完的局所在房间打分，跨轮累计。
  - 三类保持混合配比（竞技场 0.2、普通 0.45、Boss 0.35，来自 `--tasks-file` / `--task-weights`）：P(房间) = 该类权重 × 类内概率。
  - 类内概率 = 0.9 × [已玩过的：0.7 × 排名分 (1/rank) + 0.3 × 久未玩过的程度；没玩过的按比例均匀先试] + 0.1 × 类内均匀保底；排名和久未玩过的程度只在同类间比较；排名温度 β = 1（论文 0.1–0.3），避免 16 个环境挤进一两个房间。
  - 学习器每轮把概率写进共享内存，worker 每局开始时按它抽（第一版写入前按混合配比）；状态随 checkpoint 存为 `plr.json`（含各类权重）；评估仍按种子固定房间。
  - abp-mix-02、03、04 用的最初实现把 573 个房间放在一个分布里排序，配比没有生效；按类计算后份额精确等于配比（C8）。
  - 分组时按步数折算的 PLR（`--plr-by-steps`）见"从头训练"。
- **并行任务组**（`--groups-file`）和 **Room Buffer**（`--room-sampling buffer`）：见下面两节。

### 训练选项

- **低温 actor**（`train_abplus.py --greedy-actors N --greedy-temperature T`，C29）：前 N 个环境按 softmax(logits / T) 采样，行为接近取最大，训练数据里就有取最大会走到的局面；记下的行为 log-prob 是加温度后的，由异步训练的解耦权重纠偏，所以必须同时用 `--sampler graph` 和 `--async-train`。
- **屏蔽被挡住的方向**（`train_abplus.py --block-moves`，C30）：
  - `abplus_geometry.blocked_moves` 从观测判断哪些移动会被地形挡死：正交方向看碰撞圆外 1 像素处前沿的三个点，斜方向要两个正交分量都挡住；回放上精确率约 0.99。
  - worker 每帧把结果写进 `move_block` 字段（不是网络输入），采样器把它并进遮罩，遮罩随样本存进 buffer，学习端用同一份。
  - checkpoint 带 `block_moves` 时评估自动加同一遮罩，`abplus_eval.py --block-moves` 可对别的 checkpoint 强制打开；日志 `behavior/move_blocked_share`：至少一个方向被挡掉的步数占比。
- **辅助损失**（`gpu_ppo`）：`--aux-distance-coef` 是距离回归相对瞄准交叉熵的权重（默认 1，即原来的损失）；`--aux-balance` 让每个 micro-batch 里对齐帧和不对齐帧的交叉熵各占一半。瞄准场和之后的普通房训练用 `--aux-coef 1.0 --aux-distance-coef 0 --aux-balance`：普通房里距离回归会拖累瞄准（C21）；单个目标时对齐帧只占 7–9%，不平衡的话辅助头学成一律"不对齐"（C22）。
- **学习率调度**（C36）：`--lr-schedule linear|cosine --lr-final X` 让学习率在这次运行的 `--game-hours` 里从 `--learning-rate` 降到 X（默认是它的 0.1 倍），实现在 `gpu_ppo.LearningRateSchedule`，当前值记在 `train/learning_rate`。默认 `constant`。
- **决策间隔**（`--frames-per-decision`，默认 2）：见"从头训练"。

### CUDA Graph 采样与异步训练

**CUDA Graph 采样**（默认 `--sampler graph`；`isaac_bridge/graph_sampler.py`、`abplus_bench_sampler.py`）：
- 原路径每步大量主机与设备同步和内核启动：`encode_frames` 用布尔索引压缩实体，还调用 `.item()` 和 `nonzero()`；SB3 的带掩码分布做参数校验（B5）。
- `CombatTransformer.encode_frame_static` 对 256 个实体槽全部计算，用注意力 padding 掩码代替压缩，形状固定、无需同步；采样改为 Gumbel-max，分布不变。每个 chunk 捕获"编码新帧"和"采样动作"两张 CUDA Graph，每步重放；单环境重置、价值估计、确定性动作仍走 eager 路径。
- 每 1024 步与 eager 路径比一次（`sampler/graph_*`），差别只来自浮点求和顺序。耗时、一致性和提速见 B5。

**`--async-train`**（`isaac_bridge/gpu_ppo.py`）：
- 采样用策略副本（actor）；学习器在后台线程、独立的低优先级 CUDA 流上训练上一轮数据的快照（第二份 rollout buffer，约 1.26 GB）。每轮采样结束：等训练结束，存 checkpoint / 评估，actor 原地拷贝学习器权重（CUDA Graph 不用重新捕获），把刚采完的一轮拷进快照，开始下一次训练。采样 compute 流高优先级；训练期间 GIL 切换间隔从 5 ms 降到 0.5 ms。
- 策略滞后一次更新，所以用解耦的 PPO（IMPACT，Luo et al. 2019；Hilton et al. 2021 的 decoupled PPO）：比值和裁剪相对训练起点的权重（每次训练前无梯度地重算对数概率，约多 1.5 s）；每个样本的策略损失乘以 π_起点/π_采样，截断在 2；没有滞后时与标准 PPO 完全相同（有单元测试）。直接用采样时记录的对数概率做比值（Sample Factory APPO 的做法）会发散（B6）。
- checkpoint 含第 k 次更新后的权重，此时第 k+1 轮已采完未训练，从它续训会丢掉这一轮。
- 日志 `async/collect_s`、`async/update_s`、`async/join_wait_s`（采样等训练）、`train/lag_kl`、`train/lag_weight_truncated`；与同步训练的稳定性和速度对比见 B6。

### 默认值与日志

- **默认配置**：16 个环境（32 个实例）、`--chunks 1`、每轮 1024 步 × 16 个环境 = 16,384 样本、batch 1024、2 epoch、奖励 combat-v5、熵 0.003、gamma 0.9995、学习率 3e-4 不变、严格 FP32；每 10 轮存一次 checkpoint，每 20 轮用 2 个 CPU 实例在 64 个 held-out 混合种子上评估。最近用的普通房配方（C37、C40）见"启动与停止"。
- **常用参数**：`--task-weights arena=…,normal=…,boss=…`、`--tasks-file none`（只训练竞技场）、`--json-obs`（回到桥接 v1）、`--game-hours N`（按游戏时停止）。
- 同步 PPO 的采样和训练串行（`--async-train` 让两者重叠）；周期评估约占 3 个 CPU 核。
- **日志**（除上文各节提到的）：
  - `behavior/no_hit_by_60s`、`behavior/camp_20s`（一处停留 ≥20 秒）、`behavior/first_hit_s`、`behavior/longest_no_hit_s`、`behavior/cells`、`behavior/bomb_use`；
  - `task/<类型>/death_rate|timeout_rate|win_rate_0bombs|win_rate_bombs`；
  - `plr/*`：各类份额、类内有效房间数（`effective_<类型>`）、已玩过的房间数、总有效房间数、最大概率；`episodes.jsonl` 每局还带 `level`、开局炸弹数和 `stats`；
  - combat-v5 起：`train/entropy_{move,shoot,bomb,item}`；`aux/aim_accuracy`、`aux/aim_accuracy_aligned`、`aux/aligned_share`、`aux/distance_mse`；`behavior/mode_aim_rate`（有对齐目标时射击头最大值等于标签的比例）、`behavior/mode_approach_rate`（d_fire 超过 1 格时移动头最大值让 d_fire 变小的比例）；`train/hit_advantage_gap`：命中步平均优势减其他步，以标准差为单位。
- **KL 指标**（C32）：
  - `train/approx_kl` 只看被采到的那个动作，在多只怪时会被少数平局样本拉爆：几个方向都能打中，策略每轮极其确信地换一个方向。
  - 看 `train/kl_full`：用完整分布算的 KL(π_proximal ‖ π_new)，各动作头求和，由近端那一遍存下的 logits 算。另有 `train/ratio_outlier_share`：|log r| > 5 的样本占比。
  - 诊断开关 `train_abplus.py --kl-probe`：每轮记下概率比最大的前 20 个样本，写进运行目录的 `kl_probe.jsonl`。会带来主机同步，只在诊断时开。

### 并行任务组、模型规模与在线蒸馏（用户 2026-09-28 的方案；分组已用在 C39，模型规模、蒸馏和分阶段扩容还没用过）

- **并行任务组**（`train_abplus.py --groups-file`，`isaac_bridge/abplus_groups.py`）：一次训练同时练几类房间。
  - 分组文件列出各组：名字、任务文件（普通房的房间表，或带 `target` 的练习场）、预算份额和每局截断时间。默认的 `catalog/parallel_groups.json` 分三组：普通房（202 个走位房，180 秒，0.5）、障碍练习场（第八档，30 秒，0.25）、躲子弹场（30 秒，0.25）。份额是提案，要定。
  - 组内按种子从这一组的房间表抽房间（第八档的三组照常先抽组），不用 PLR，所以必须 `--room-sampling mixture`；`--tasks-file`、`--episode-seconds`、`--task-weights` 在分组模式下不用。
  - 每局的房间、练习场设置和截断时间都是这一局所属组的；`--hurt-rest-cost` 的剩余时间按这一组的截断时间算。
- **预算按环境步数分**（`--budget`）：换组只发生在一局结束时，不会打断正在进行的一局。worker 在一局开始时就为下一局准备备用实例，所以第 k+1 局的组在第 k 局开始时决定。
  - `steps`（默认）：每个环境槽位记下自己在各组里走过的决策数，下一局选实际份额比目标份额差得最多的组；正在进行的那一局按这一组的平均长度计入。
  - `slots`：每个槽位整个训练只练一组，按份额分配、交错排列（低温 actor 所在的前几个槽位覆盖所有组）；各槽位同步前进，所以步数份额就是槽位份额。
  - `episodes`：每局按种子以份额为概率抽组（按开局次数分）。两组各 50%、平均 3 秒和 30 秒时，长的那组占约 91% 的步数（`test_groups` 有这个例子）。
- **日志和评估**：`group/<组>/step_share`（这一轮里各组的步数占比）、`episodes`、`win_rate`、`hurt_rate`、`death_rate`、`time_limit_rate`、`return`、`episode_s`；`episodes.jsonl` 每局带 `group`。每个 checkpoint 按组评估：64 个 held-out 种子，用这一组的房间表、练习场（`--target-from-tasks`）和截断时间，两种模式，结果在 `evaluations-<组>/<checkpoint>/`，日志在 `eval-logs/`。
- **模型规模**（只对新模型）：`--model-width`（Transformer 和融合层宽度，默认 256）、`--model-layers`（4）、`--model-heads`（8）、`--entity-queries`（实体汇总的查询数，4）、`--entity-width`（实体 token 宽度，128）、`--head-width`（策略、价值 MLP 宽度，256）。默认值就是 C1–C37 的结构，旧 checkpoint 照常加载；续训沿用 checkpoint 里的结构，给了这些参数会报错。
- **在线蒸馏**（`--distill-from <checkpoint>`，`gpu_ppo.GpuMaskablePPO`）：老师是那个 checkpoint 的策略，冻结不训练。
  - 每个样本加一项 `--distill-coef` × T² × Σ(四个动作头) KL(老师 ‖ 学生)。两个分布都用这个样本存下的合法动作遮罩，T 是 `--distill-temperature`（默认 1）。
  - 瞄准辅助损失、价值损失照旧；PPO 的策略梯度项乘 `--distill-rl-coef`（默认 0，即纯蒸馏；大于 0 就是 kickstarting）。`--distill-coef-end` 让 KL 的权重在这次训练的预算里线性变到这个值。
  - 样本是学生自己新采的（在线）；`--distill-teacher-acts N` 让老师采样，直到模型做完 N 轮更新。老师和学生宽度不同时，采样器会自动重建特征缓存、重新捕获 CUDA graph。
  - 日志：`distill/kl`、`distill/coef`、`distill/agree_<头>`（两者取最大动作一致的比例）、`distill/teacher_collects`。
  - 运行目录的 `model_size.json` 记下参数量和结构。
- **分阶段增加环境数**：`--resume` 可以换 `--envs` 和 `--n-steps`（按新的数目重建 rollout buffer），所以 16 → 32 → 64 个环境是三次续训，每次加的是新鲜样本，不是重复训练同一批数据的次数。`--batch-size` 要整除 envs × n-steps；`--greedy-actors` 按比例加。
- **显存**（B7）：每个 AB+ 实例用显卡驱动的 OpenGL 时占约 80 MiB 显存，32 个环境（64 个实例）加学生模型就会超出 12 GB。`--software-gl`（训练和 `abplus_eval.py` 都有）让实例改用 Mesa 的软件 OpenGL（`__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1`，`abplus.SOFTWARE_GL_ENV`）：实例不占显存，轨迹逐位相同。评估在 CPU 上跑，现在带 `CUDA_VISIBLE_DEVICES=''`，每个评估 worker 不再白占约 270 MiB 显存。
- **核对**：单元测试 `test_groups.py`（分组文件、槽位分配、三种预算的长期份额、大模型能建能前向、带遮罩的四头 KL）；远端冒烟测试见 EXPERIMENTS.md B7。

### 从头训练：combat-hp、四类房间与 Room Buffer（C39，用户 2026-09-28/29 提出）

- **奖励 combat-hp**（`--reward-profile combat-hp`，`abplus_reward.CombatHp`）：
  - 房间里任何怪物每掉 3.5 血 +1，原有的和后生成的都算，不管挡不挡门。计数是桥的 `combat.monster_damage`（abp-0.2.8-hp 起）：引擎 `IsActiveEnemy` 的类型（10–999，去掉店主、火堆、便便、可移动 TNT），只算真实掉血。
  - 玩家每掉半颗心 −0.5；清房 +100；期限（默认 180 秒）−100；死亡 −100。期限和死亡都是终止，不 bootstrap。训练时 × 0.1。
  - 观测：combat-v5 的几何观测和四个动作头；房间状态只留挡门怪数和血量比例；加剩余时间 1 − t / 期限。剩余时间存进帧记录，因为每组的期限可以不同。
  - 默认 1 个炸弹（游戏的开局）、不无敌、谱系模式 3。
- **决策间隔**（`--frames-per-decision`，默认 2）：一个动作按住几个逻辑帧。
  - `AbplusTransformerEnv`、worker、评估都跟着这个值；游戏时、看板和回放页都按 `config.json` 里的值换算。
  - 换帧数时 γ 要跟着换，才能保持同样的时间尺度，比如 2 帧 0.9995 对应 4 帧 0.999。
  - 对战场按 2 帧标定，带 `--duel-file` 时只允许 2。
- **四类房间**（`catalog/scaling_groups.json`，各 25% 的步数，每局 180 秒）：
  - `scaling_normal_rooms.json`：491 个 Basement I 普通房；
  - `mixture_target_horf.json`：空房里一只 Horf；
  - `mixture_target_horf_rocks.json`：365 个有石头的房间，1、2、3 只 Horf 三个分支；
  - `mixture_target_spawners.json`：空房，Portal / Mulligan / Nest / Baby Long Legs 四个分支，另加 0–3 只常见的不生怪的怪。
- **组内按时长折算的 PLR**（`--room-sampling plr --plr-by-steps`，分组时也能用）：
  - 关卡是（组，房间，分支）；PLR 的份额按步数算，开局概率 = PLR 概率 ÷ 这个关卡实测的平均局长。
  - C39 最后没用它：两个空房组里 63 个房间清空后完全一样，按房间挑不出东西。
- **Room Buffer**（`--room-sampling buffer`，`isaac_bridge/room_buffer.py`；只和 `--groups-file`、combat-hp 一起用）：
  - 一个关卡就是一个种子，同一个种子重放时开局逐帧相同。
  - 每组保留 `--buffer-capacity`（2000）个种子，每个记清房率 p、清房局掉血 d 和局长的 EMA（`--buffer-alpha` 0.25）；新种子的初值取它所在房间（有分支的组是"房间 × 分支"）当前的统计。
  - 成功只算清房。重放优先分 4p(1−p) + λ·p·clip(d/d₀, 0, 1) + η·S + ε，对应 `--buffer-lambda` 0.5、`--buffer-d0` 2（半心）、`--buffer-eta` 0.1、`--buffer-eps` 0.02。S = min(1, 本组隔了多少局没玩它 ÷ 容量)。
  - 开局概率 ∝ 优先分 ÷ 局长，局长向所在房间的局长收缩（先验权重算 4 局）。
  - 新种子 `--buffer-fresh`（0.3）的步数：每个槽位在每组里按步数记账，和组间份额是同一个调度器；缓冲区还空时的新种子局不计入。
  - 新种子打完就进缓冲区，满了就替换 4p(1−p) + λ·p·clip(d/d₀) 最低的那个（前提是新种子更高）。所有局都用来训练。
  - 日志：`buffer/<组>/*`（大小、p 的分布、各项占优先分的比例、有效种子数、换入换出数）；`group/<组>/fresh_win_rate`（新种子的清房率，不受选房偏向影响）、`replay_win_rate`、`fresh_step_share`。checkpoint 里存 `room_buffer.json`，续训时读回。
- **评估设备**（`--eval-device cpu|cuda`，默认 cpu）：周期评估的策略放在哪里推理。cpu 时评估进程看不到 GPU，不占学习端显存；cuda 时每个评估 worker 约占 300 MB 显存、791 MB 内存。但训练把 GPU 占满时，GPU 评估反而更慢：C39 t2 每秒 17–18 步（CPU 约 90 步），还把训练拖慢了约 20%。所以训练期间的评估用 cpu。
- **后台启动时的 SIGINT**：`setsid nohup ... &` 启动的训练会继承"忽略 SIGINT"。`train_abplus.py` 开头会把它恢复成默认处理，所以 `kill -INT` 能正常收尾。停一次训练：先停评估链的 bash，再停评估进程，最后 `kill -INT` 训练进程。
- **内存保护**（C39 t3 在 122.6 小时被 OOM 杀掉之后加的）：
  - 实例回收门槛按槽位错开到 `--recycle-episodes` 的 1–2 倍（`abplus_worker.recycle_threshold`），不再十几个进程在同一轮里一起换；
  - 内存不够时杀进程的顺序：评估进程（oom_score_adj 1000）→ 游戏实例（500，由 worker 重启）→ 训练进程（0）。
  - 每个实例的常驻内存上限 `--recycle-rss-mib`（默认 0 = 不设；C39 t3r2 用 400）。C39 t3r 里有的实例一分钟涨 100–200 MiB，原因未明，按局数回收挡不住（EXPERIMENTS.md C39）。
    - 训练实例：每局结束时读打完这局的进程的 VmRSS，到上限就在后台准备线程里换进程，走和按局数回收相同的路径；换不成隔 10 局再试。`abplus/memory_recycles` 记次数，`abplus/instance_rss_max_mib` 是各槽位最近一局的最大值。
    - 评估实例：两局之间超过上限就重启，训练进程把自己的上限传给评估。
    - `episodes.jsonl` 每局记 `rss_mib`（打完时的常驻内存）、`instance`（0 = a，1 = b）、`instance_episodes`（这个进程打了几局），用来按房间组查增长。
    - 按这些记录，增长不是稳定泄漏：约一半的新进程在前 75 局里一次性涨 100–150 MiB，之后不再涨。所以有了内存上限以后，按局数回收只会不断制造新进程；C39 t3r3 用 `--recycle-episodes 0` 关掉了它（EXPERIMENTS.md C39）。
- **spawn 出的进程不 import torch**：`train_abplus.py` 在函数里才 import torch、SB3 和策略网络。spawn 出的 worker 会把主脚本当 `__mp_main__` 重新 import，顶层 import 时每个 worker 多约 300 MB（C39）。
- **核对**：单元测试 `test_c39.py`、`test_room_buffer.py`；引擎核对和冒烟见 EXPERIMENTS.md C39。

### 所有房型、Boss 房与开局随机：combat-hp2 / combat-hp2-camera（C41，用户 2026-09-29 提出）

- **奖励**：同 combat-hp，只是每掉半颗心 −1（`--hurt-cost` 可改；combat-hp 默认 0.5）。
- **两种大房间的观测**（位置、速度、大小和射程都按 1×1 房的内部尺寸 520 × 280 像素换算，`transformer_obs.ROOM_1X1`，1×1 房的数值和 combat-hp 相同，大房间最大到约 2）：
  - `--reward-profile combat-hp2-camera`（C41 续训用）：地形是玩家周围 15×9 的窗口，跟着玩家走，到房边停住，和游戏镜头一样（`room_scale='camera'`，`transformer_obs.camera_origin`）。1×1 房逐位同 combat-hp，观测空间和网络也相同，所以 combat-hp 的 checkpoint 能直接接着练。
  - `--reward-profile combat-hp2`（C41 先跑的从头版本）：地形放在 16×28 的画布上（Basement 最大的 2×2 网格），小房间在左上角，其余格子记为房外（`terrain_shape=BIG_TERRAIN, room_scale='fixed'`）。地形 CNN 的全连接层按画布定尺寸，所以它和 combat-hp 的 checkpoint 互不兼容。
- **从别的 checkpoint 接着练**：`--warm-start <checkpoint> --warm-start-optimizer` 载入权重和 Adam 状态，学习率按这次运行的日程，Room Buffer 和计数器从零开始。只有 `--warm-start` 时只载入权重。`--resume` 要求奖励配置相同，而且会接着原来的学习率日程。
- **分组**（`catalog/scaling2_groups.json`，1:1:1:1，每局 180 秒）：
  - `normal`：491 个 1×1 普通房；
  - `normal_big`：358 个其他形状的普通房（`scaling2_normal_big_rooms.json`）；
  - `horf_rocks`：同 C39；
  - `boss`：Basement I Boss 池的 82 个房间（`scaling2_boss_rooms.json`）。
  - 选房规则见 EXPERIMENTS.md C41。分组现在可以含 Boss 房，只排除模拟器的 Monstro 竞技场。
- **开局随机**（都由种子决定）：
  - 血量：`--player-hp-prob 0.8333333 --player-hp-min 1`，1–6 个半心等概率。
  - 炸弹：`--start-bombs 0.25:3`，0–3 个等概率。
  - 属性：`--stat-noise speed=0.5,damage=1,shot_speed=0.5,tears=1,range=1`，每项 U(−a, a)。
    - `tears` 是每秒发数的变化，`range` 是 Repentance 单位（每单位 40 像素）。
    - 由桥 abp-0.2.11 的 `AbpSetStats` 生效，客户端在重置后核对五项属性，每局记录的 `start.stats` 记下修正。
- **评估**：`--eval-bombs 1` 固定开局 1 个炸弹（不设时仍用训练的开局炸弹抽签）；满血；基础属性，训练的 `--stat-noise` 不用于评估。
- **启动**：第二台 `~/isaac-abplus/train/c41c.sh`（续训）；`c41.sh` 是停掉的从头版本。
- **核对**：单元测试 `test_c41.py`；引擎核对和冒烟见 EXPERIMENTS.md C41。

### fork 采样器（2026-10-03）

为"能随算力扩展"重做的采样结构（`isaac_bridge/fork_sampler.py`），和上面的训练管线并列，还没有接学习器。动机和测量见 B9、B10。

```text
worker 进程 × N（不导入 torch）
  root      一个 AB+ 实例（FORK_ENV）。它只做重置：重置完就停在一局的起始状态
  template  root 在起始状态上的克隆体，停着不动；它服务期间 root 已经在后台线程里建下一个起始状态
  episode   template 的克隆体（约 5 毫秒），重设全局 MT、进入精简模式，打到这一局结束后关掉
            一个 template 服务 episodes_per_state 局；下一个状态还没好时继续用
  每个决策：解码精简观测 → 往共享数组自己那一行写一条定长记录 FRAME → 通知服务端 → 等动作
服务端（持有 GPU 的进程，ForkSampler.serve）
  收集已经就绪的行（凑到 min_batch 行或等满 max_wait_ms）→ 一次批量前向 → 写动作 → 唤醒这些 worker
  不等还在走帧的 worker；worker 和游戏的优先级比服务端低（nice 10 / 19）
```

- 和旧管线的区别：每局不再用控制台命令重建房间（约 0.36 秒），也不需要每个环境一个备用实例；worker 每步不算奖励、几何和标签；推理不锁步。
- `FRAME` 是替身：玩家 24 维、最多 96 个实体各 20 维、7×16×28 的地形、这一步的掉血和怪物掉血、结局。正式的策略输入还没设计。
- 基准：`python abplus_bench_fork_sampler.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --workers 16 --policy random|uniform|model --episodes-per-state 16 --stub-list <tools>/stub_render_g.txt --out <目录>`。`random`：worker 自己抽动作；`uniform`：服务端回随机动作；`model`：服务端在 GPU 上跑一个替身网络并把记录存进 GPU 环形缓冲。`--full-bridge` 关掉精简模式作对照。

### token 策略与 fork 训练器（C55 起，2026-10-03）

用户 10-03 同意做"一个策略管整局"和"对着人类定的指标"，并定了三件事：推理路径和策略网络一起设计；克隆体预知随机数只能用来产生训练目标，评估时不能用；动作晚一个决策生效是环境的规定。这一节是建在 fork 采样器上的新训练线，和上面的 `train_abplus.py` 管线并列，互不依赖。结果见 C55、C56、C59、C60（整层 C57、C58、C61），速度和 `--overlap` 等选项见 B11、B12、B14、B15，GPU 共用时的 CUDA 崩溃和现在的运行规则见 B13。

```text
worker 进程 × N（tok_sampler.worker_main，不导入 torch）
  root / template / episode 同 fork 采样器
  每个决策：解码精简观测 → 写一条 tok_obs.ROW 到共享数组 → 通知训练器 → 用"上一条记录的回答"走这一步
训练器（train_tok.py，持有 GPU）
  收集：对就绪的记录批量采样动作（GraphActor：固定形状的 CUDA Graph，一次调用约 1.3 毫秒），存进各 worker 自己的序列
  更新：存够 --rollout 个决策后做 PPO（各 worker 的序列分别算 GAE，最后一个还没拿到奖励的决策留给下一轮）
```

- **环境的规定：动作晚一个决策生效**。worker 发出第 t 条记录后不等回答，直接用第 t−1 条记录的回答走下一步；游戏在推理期间照常前进。策略的反应时间是一个决策（每决策 4 帧时 133 毫秒，人类的视觉反应时间约 200–250 毫秒）。正在执行的动作（`pending`）是状态的一部分，输入给策略和价值。一局的第一步不动。
- **记录**（`tok_obs.ROW`，约 14 KB）：策略看到的就是玩家看得到的。
  - 玩家 31 维：位置、速度、各种心、炸弹钥匙硬币、属性、无敌时间、射击冷却、房间是否已清、时间用了多少等；
  - 实体最多 64 个（超出时取最近的），每个 33 维：相对玩家的位置和距离、引擎速度、大小、碰撞类别、接触伤害、动画帧、年龄、种类独热、敌人 / 可受伤 / Boss / 挡门标志、Boss 的血条、子弹的高度和下落速度，再加 type、variant、subtype 三个编号（查嵌入表）。小怪的血量、AI 状态和随机数不在里面；
  - 门最多 8 个，每个 7 维：相对位置、开着、锁着、门后房间类型、门后房间去过、已清；
  - 地形 7×16×28（`terrain_channels` 的 7 个通道）和以玩家为中心的 9×9 格局部地形；
  - 小地图 8×13×13（只在整层模式下有内容）；
  - 这一步的奖励项（掉了几个半心、打掉的怪物血量占比、事件位）和结局随记录一起传，不是策略输入；
  - 10-06/07 起在末尾追加道具（库存、台座编号、层数，`--items`）和蓄力（`pcharge`，`--charge`）的字段，14,288 → 14,720 字节，原有字段的偏移不变（EXPERIMENTS.md A21、A22；见下面"整局模式、道具、蓄力和分支"）。
- **网络**（`tok_policy.TokPolicy`，默认 192 宽、4 层、4 头；单房 277 万、整层 339 万参数）：玩家、房间、小地图、每个门、每个实体各是一个 token，位置都相对玩家，进 Transformer 之前不做池化；玩家 token 的输出接移动（9）、射击（5）、炸弹（2）三个头、价值头和"危险"头（9 个移动各自会不会受伤，给教师用）。没有炸弹时炸弹头只能选"不放"。
- **推理**（`GraphActor`）：最多 N 条记录、每条 32 个实体 token 的固定形状前向录成一个 CUDA Graph，采样用 Gumbel-max，图直接读模型参数，优化器更新后不用重录。实体超过 32 个的记录走即时执行。
- **奖励**（`--reward`，默认 `ROOM_REWARD`）：打掉的怪物血量占房间初始总血量的比例 × 1；每掉半心 −0.75；清房 +1；死亡 −2；超时 −1；每个决策 −0.002。没有别的塑形项。
- **事后搜索教师**（`--teacher`，C56）：
  - 一局结束后，对其中受伤的步（每局最多 2 个），worker 把这局恢复到受伤前 k 个决策（k 依次取 2、4、8）：template 的克隆体用同一个 reseed 重放前面的动作，状态与原局逐帧相同（A19）；
  - 从那个状态分出 9 个克隆体，各自先走完正在执行的动作，再一直按住 9 个移动之一（射击照原局），走到受伤那一步之后 4 个决策；没掉血的移动算"安全"；
  - 取还有安全移动的最晚的那个 k，把（当时的记录、正在执行的动作、9 个移动各自是否受伤）作为一条教师记录交给训练器；
  - 训练器用两项损失：危险头对 9 个标签做二元交叉熵；移动头的 −log（安全移动上的概率和）。每次更新抽的教师记录数是"每次更新新到的记录数 × `--teach-reuse`（8）"，摊到各个小批；小批里 b 条教师记录、m 个决策时，教师损失的权重是 `--teach-weight`（8）× b / m；
  - 搜索排在队列里，在训练器做更新、worker 本来空等的时候跑；`--teacher-share` 可以再让出一部分采样时间；
  - 克隆体和原局共用随机数，所以教师知道策略不知道的未来。按用户的决定，这只用来产生训练目标；评估时不用克隆体。
- **用法**（在桥的 `python/` 目录，`PYTHONPATH=.`）：
  - 训练：`python train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --groups normal:8,boss:5,normal_big:3 --workers 16 --game-hours 1500 --epochs 3 --out <目录> [--teacher] [--micro 1024] [--resume <checkpoint>] [--init <checkpoint>]`。`--groups` 是各组的 worker 比例；每个起始状态开局血量在 `--start-hp`（默认 2:6 个半心）里均匀抽；`--micro` 把一个小批分几份前向、累积梯度后走一步（同一步，显存小；一张 24 GB 的卡同时跑三个训练器时要用）。
  - 评估：`python eval_tok.py --checkpoint <…>/update-000300.pt --groups-file … --groups normal:6,boss:4,normal_big:2 --seeds 2147490000:128 --greedy --out <目录>`；每个种子一局、满血、克隆体用种子本身 reseed；`--random` 评估均匀随机动作。
  - 输出：`progress.csv`（每次更新一行，含各组的胜 / 死 / 超时、掉血、教师的统计）、`episodes.jsonl`、`checkpoints/`。
- **训练器的速度选项**（B11）：`--overlap`（学习器在线程里更新，actor 同时用上一份权重采下一轮）、`--decoupled 1`（解耦 PPO，与 `--overlap` 一起用）、`--sort-micro`（默认 1）；开 `--overlap` 时 `--teacher-share` 默认 0.15。SIGINT / SIGTERM 时先存 `last.pt`。`--resume` 时 `--game-hours` 是再练多少。`--amp bf16`（学习器前向用 bf16）、step 行 / 原生 step / 地形缓存 / FastRow / `fork_many` 的环境变量开关和 actor 槽的修正见 B12。默认设置（C60）：3 个 epoch、同步、float32、教师；bf16 不默认开。`--overlap` 现在不用：同卡还有别的 CUDA 进程时会崩溃（B13）。`train_tok.py`、`tok_policy.py` 在环境里没有时设 `CUDA_DEVICE_MAX_CONNECTIONS=1`；`eval_tok.py` 默认在 CPU 上推理（`TOK_EVAL_DEVICE` 可改）；一块 GPU 只跑一个 CUDA 训练器。
- **B14 的选项**：学习器 `--packed`、`--teach-merge`、`--teach-gpu`（默认都是 1，0 回到旧路径）。引擎帧改动（H 打桩表、`ABP_FAST=3`、`ISAAC_RL_PU_SKIP=1`）10-05 起是 worker 实例的默认（见下一条）；`ISAAC_RL_WATCH_PID=0` 让 worker 不设 `ABP_WATCH_PID`。精确性、速度和 watcher 的崩溃见 B14；两台的资源守护 `~/isaac-abplus/guard.sh` 见 EXPERIMENTS.md 的"已知问题"。
- **B15 的默认和选项**：worker 的 root 实例默认 H 打桩表（`default_stub_list()`，`--stub-list` 可换）和 `tok_sampler.INSTANCE_ENV_DEFAULTS`（`ABP_FAST=3`、`ISAAC_RL_PU_SKIP=1`、`ABP_FORK_LITE=1`；环境里已设的不覆盖，设 0 关掉）；`--teacher-pair`（默认 1）、`--teacher-skip-known`（默认 0，少测一个移动的标签）、`--teacher-slots`（默认 0 = 不限）；`--build-timeout`（默认 120 秒，起始状态建不好就杀掉重启）；测试用的 `ISAAC_RL_FAULT_HANG`。`eval_tok.py` 仍默认 G 表。成本、精确性和卡死见 B15。
- **没做的**：学习器的 CUDA Graph。（道具、药丸、卡牌的动作头 10-06 做了，见 A21。）

### 整层环境（C57 起，2026-10-03，桥 abp-0.2.15）

一局是一整层 Basement I：从新开一局的起点（以撒 3 颗红心、1 个炸弹）走门、清房、打 Boss，掉进活板门到下一层算通关。没有任何脚本参与（走门、选房间、捡东西都由策略的移动完成）。代码 `isaac_bridge/tok_floor.py`，`train_tok.py --mode floor`，`eval_tok.py --mode floor`。

- **重置**：root 实例用桥的 `reset_mode = 'floor'`（真实楼层；有迷宫、迷失、迷路诅咒的楼层跳过，同 A9），template 停在楼层起点。
- **一局的结束**：死亡；层数变了（通关）；超过 `--floor-seconds`（480 秒）；连续 `--floor-stall-seconds`（60 秒）没有进展。后两种都记超时。
  - 进展：进了新房间、清掉一个房间，或者（10-05 14:14 起，C61）这一步打掉了怪物的血。之前只有前两种，所以打了 60 秒还没清掉的房间会被判超时：C58 的诊断里同一策略时限 60 / 180 / 480 秒通关 12.5% / 38.5% / 49%（EXPERIMENTS.md C58）。两处代码：`tok_obs.encode_row`、`abp_turbo.c` 的 `abp_row_encode`。C57、C58 是旧规则下练的。
- **策略多看到的东西**：
  - 小地图（精简观测的地图块）：只含游戏自己在小地图上显示的房间（`RoomDescriptor.DisplayFlags ≠ 0`），每个房间给出位置、形状、是否去过、是否已清、是否当前房间，图标显示了或去过时才给类型。编码成 13×13 的 8 通道图：显示、去过、已清、当前、Boss、宝箱房、商店、其他特殊房间；
  - 每个门后面的房间是否去过、已清（小地图上看得到的信息）；
  - 出口（活板门、楼梯是网格实体）作为一个单独类型的实体 token；
  - 没进展的时间用了多少（停滞计时：没进展的决策数 ÷ 时限，封顶 1）。
- **奖励**（`FLOOR_REWARD`）：打掉怪物血量占比 × 1，每掉半心 −0.75，清掉一个有怪的房间 +1，第一次进一个房间 +0.5，清掉 Boss 房再 +2，到下一层 +3，死亡 −2，超时 −1，每个决策 −0.002。
- **起点库**（start-state archive）：
  - 一局第一次进某个房间时，以 `--snapshot-prob`（0.5）的概率在进房那一刻克隆一份停着，放进这个 worker 的起点库（`--archive-size` 个，满了就换掉用得最多的）；
  - 每局以 `--archive-prob`（0.7）的概率从起点库里随机取一个起点开局（它的克隆体，重设随机数），否则从楼层起点开局；
  - 作用：楼层深处的房间（Boss 房）和开头的房间练得一样多，不用每局都先打到那里。评估不用起点库，每个种子的楼层从起点打一次。
  - `--archive-boss p`（10-05，C62；默认 0 = 照旧）：Boss 房入口总是存进起点库，Boss 房入口不超过一半时最后被换掉，从起点库开局时以概率 p 从 Boss 房入口里取。C62（p = 0.5）3,000 游戏小时后 256 个楼层 59.6%，对照 C63 64.1%，Boss 房里的死亡没少：没有带来提升（EXPERIMENTS.md C62、C63）。
- **教师**：开着 `--teacher` 时每次换房都克隆一份作为恢复点，受伤从它所在房间的恢复点恢复（桥的 `play` 在换房处停下，所以恢复不跨房间）。停着的克隆体由起点库、正在打的这局和排队的搜索共用，用引用计数管理。
- **代价**：一个停着的克隆体在 root 换到别的楼层之后约占 40–50 MiB；每个 worker 起点库 10 个、搜索队列 12 个时约 1 GB。16 个 worker 时起点库合计约 140 个：B16 之前每个约 58 MiB（约 7.7 GiB），B16 起停下时 `malloc_trim`（`--trim-parked 1`，默认）后约 35 MiB（约 4.6 GiB）。内存和卡死的其他事项见"运维"。
- **验证**见 A20：克隆体在进房那一刻逐帧精确，`play` 恢复精确，小地图和事件位与引擎一致。训练见 C57（从零，失败）、C58（从 C56b 初始化，一个游戏年，旧规则）、C61（新规则，从 C58 最终接着练 3,000 游戏小时；256 个留出楼层通关 56.2%，C58 最终在新规则下 41.6%）、C62 / C63（从 C61 最终接着练 3,000 游戏小时：Boss 房起点加倍 59.6%，对照 64.1%）、C64（C63 最终续训一个游戏年：70.3%，中途 +2,827 游戏小时时 62.1%）、C65（加宽网络、补零初始化，失败的尝试）、C66（换种子续训 3,000 游戏小时：58.2%，运行之间差约 4 个百分点）。整局模式（C67 起）见下一节。比较检查点用 256 个楼层（`--seeds 2147492000:256 --dump-final`）：48 个楼层一个检查点的标准误约 0.07，看不出 10 个百分点的变化（C61）。
- **评估注意**：有的楼层不打 Boss 也能掉进活板门（种子 2147490016，见 EXPERIMENTS.md 的"已知问题"），所以 `eval_tok.py --mode floor` 另报 `boss_clear`。
- **用法**：`python train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode floor --workers 16 --game-hours 1500 --epochs 3 --teacher --teacher-queue 12 --archive-size 10 --out <目录>`；评估 `python eval_tok.py --checkpoint … --groups-file … --mode floor --workers 8 --seeds 2147490000:128 --out <目录>`（默认按策略概率采样；加 `--greedy` 取最大）。`--init <单房训练的 checkpoint>` 可以从单房训练的权重出发（输入多出来的列从 0 开始）。
- **没覆盖的**：整层模式本身仍只打 Basement I、不加 `--items` 时没有道具的观测和动作（整局和道具见下一节）；有迷宫、迷失、迷路诅咒的楼层。

### 整局模式、道具、蓄力和分支（2026-10-06/07；EXPERIMENTS.md A21、A22、B17、C67）

- **整局模式**（`--mode run --run-seconds 1800`）：一局从新开一局起，掉进活板门后接着打下一层，死亡或 30 游戏分钟结束（没有"通关"这一结局）。`eval_tok.py --mode run` 报到达的层、平均过的层数、死在第几层。C67（第一台，已完成）是第一次整局训练；最新已评估 u9600 平均过 1.125 层，不能称整局通关。
- **道具**（`--items`，A21）：库存块（持有的道具从新到旧带个数、主动道具和最大充能、饰品、药丸颜色和已知效果、卡牌、诅咒）进玩家 token，台座上的道具编号进实体 token（致盲诅咒下给 1023）；嵌入表道具 1024、饰品 256、药丸效果 256、卡牌 256、药丸颜色 16（按 Repentance(+) 的编号留了余量；只有道具 474 在 AB+ 和 Repentance+ 意义不同）。两个动作头：用主动道具、用药丸 / 卡牌（初始偏置 −4，用不了时屏蔽）。旧检查点 `--init` 时新输入补 0、新头新初始化。
- **蓄力**（`--charge 1`，默认开，A22）：玩家的蓄力计数（`Entity_Player+0x2634`）和武器类型进 ROW 的 `pcharge`；`--resume` 的检查点保持原来的设置。4 帧一决策不丢任何一档离散的蓄力效果。
- **`play` 与换房**（A22）：`play`、`fork_many` 遇到换房时把它当作一步的结束、接着走（与 step 行一致）；`stop_room: true` 回到以前"停下整批"的行为。地形块在网格实体的状态变化时也重发。
- **分支**（第二阶段，B17/B2，仅在第二台 `brn` / `brn2` 沙箱实验，没有部署到 C67）：`tok_branch.py`。在拿道具、留下道具离开、换房还有别的门的地方，从房间入口的克隆体分出几支（原动作 / `pickup_block` 不拿 / 脚本走到别的门或台座），共用就地 reseed 由 actor 各打一段，结果存成选择记录（`choices.jsonl`）；`train_tok.py --branch-share`、`--branch-noise`、`--branch-max`、`--branch-ppo` 等。B2 已完成共用策略噪声、快照、截断自举与选择头的实验；目前尚未证明 CRN/快照的优势或选择头的留出泛化收益。

### 对着人类定的指标（2026-10-03）

目的：以后的成绩都报在同一把尺子上，这把尺子人也能量。现在只定了第一层的版本，整局版本等策略能过第一层以后再定。

- **任务**：AB+ v1.06，以撒，普通难度，新开一局，打穿 Basement I（到达下一层）。不给任何辅助：没有脚本导航、没有护盾、没有评估时的搜索。
- **种子**：从 2147490000 起的留出种子；楼层由种子决定（`room_seed(seed, −2 − 第几次尝试)` 作起始种子，跳过有迷宫、迷失、迷路诅咒的楼层），训练不会抽到。
- **AI 一侧**（`eval_tok.py --mode floor`，每个种子一局）：
  - 通关率、死亡率、超时率（480 秒，或 60 秒没有进展；进展的定义见"整层环境"，10-05 改过）；
  - Boss 房清掉的比例、每局清掉和进入的房间数；
  - 每层掉的半心数、每清一个房间掉的半心数、无伤通关的比例；
  - 通关用的游戏时间。
- **人一侧**：同样的种子、同样的起点，用键盘打，记录同样的量。还没有数据：需要用户或别的玩家在 AB+ 上打一批（本地启动游戏会把远端训练挤下线，要用户安排）。在有数据之前，报告里只写 AI 的数和旧系统的数，不写"超过人类"。
- **此前系统在这把尺子上的位置**（A18，512 个留出种子，注意它用了脚本导航，且楼层种子的取法相同、时间限制不同）：带护盾 31.8%，不带护盾 16.8%。
- **现在的数**（10-07 02:50，256 个留出楼层，种子 2147492000 起，采样，新的进展规则，480 秒；EXPERIMENTS.md C64）：C64 最终通关 70.3% ± 2.9（每一局通关都打掉了 Boss），Boss 房里死 10.5%、别处死 4.3%，超时 14.8%（找路 9.8%、房间没清 5.1%）；每层掉 2.39 个半心，无伤通关 11.7%，通关平均用 107 秒游戏时间。同样的楼层上 C58 最终 41.6%、C61 最终 56.2%、C62 最终 59.6%、C63 最终 64.1%、C64 第 19200 次 62.1%、C66 最终 58.2%；同设置两次运行之间差约 4 个百分点（C64、C66）。人一侧还没有数。
- **整局版本**（以后）：从第一层打到妈妈或更后，报整局通关率和连胜；对照是高水平玩家公开的连胜记录和用户自己的成绩。

## Go-Explore（A8，用户 2026-09-30 提出）

目的：在原版引擎上给单个房间找出好轨迹（清房、少掉血、快），做后续训练的示范或起点。现在只有 Go-Explore 的第一阶段（探索）：每个房间一个 cell 存档，反复"选一个 cell → 返回 → 从那里随机探索 → 更新存档"。第二阶段（把轨迹变成稳健的策略，例如从示范的后段往前练）还没做。代码：`isaac_bridge/abplus_goexplore.py`（库）、`goexplore_abplus.py`（驱动）。首轮结果见 A8（64 个普通房各 1000 次探索：95% 找到通关、78% 无伤），输出在第一台 `~/isaac-abplus/gx/runs/a8-normal-0930-0246`，回放页 `…-replays.html`（本地副本 `rl/runs/abplus-replays/goexplore-a8/index.html`）。

### 返回：重置 + 批量重放 + 摘要核对

- **返回** = 按房间种子重置（A7 修复后的重置），再把 cell 的动作序列一次交给桥的 `play` 命令（abp-0.2.12，见"二进制观测与版本"）。桥在 Lua 里一个动作接一个动作地播，只在最后发一次观测，省掉每步的观测打包、传输和解码；帧与同样的 `step` 序列相同（A8 实测）。环境一层是 `AbplusTransformerEnv.play`：按 `step` 的规则算每个动作的帧数（每决策帧数，最后不超过截止时间），核对推进的帧数，给出同样的结局（胜、死、超时）。`play` 不经过中间的观测，所以历史窗口从最后一帧重新开始，和重置之后一样。
- **核对**：到达一个 cell 时记下隐藏状态摘要的哈希。摘要同 A7：玩家、其他所有实体（含 AI 状态、种子、下落参数）、格子、路径代价、全局 MT 的下标和状态哈希；不含随前史变化的诊断行（全局帧计数、实体列表顺序和 Index、abp_turbo 计数）。返回后要求：动作全部播完、帧数相同、摘要相同、cell 键相同。
- **对不上时**：摘要不同，先在同一实例里再返回一次（`--retry-mismatch 1`，事件 `digest_retry` 记下第二次是对上了、和第一次相同，还是又不同）；仍对不上则这次返回作废、不探索，cell 记一次失败，连续 `--max-fails 3` 次失败就退役（不再被选；之后如果有更快的轨迹到达它，重新计数）。
- **每个房间最后**：胜利轨迹按好坏依次在当时空闲的 worker 上再返回一次核对（`verify`），第一条通过的就是这个房间报告的最好轨迹；最多试 3 条，都不过就报告存档里最好的那条并标为未核对（`best_verified`）。
- **音源状态**：默认 `--al-stopped`，实例带 `ABP_AL_STOPPED=1`，用本副本 `tools/libabp_turbo.so`（`--preload` 可换）。不开时，返回约 4.7% 对不上，个别房间达三成；开了以后 1.1%，只剩有 Mulligan 或 Portal 的房间（A8）。第一次连上实例时检查库确实支持这个开关，不支持就报错。注意这样找到的轨迹是在"音效立刻播完"的环境里找到的，拿去训练时环境要一致。

### cell、选择与探索

以下属性和常数都是我们定的，只有计数形式来自 Go-Explore：
- **cell 键**：玩家位置按 `--cell-px 80` 分格（从房间左上角算），血量（半心，同桥的 `health_units`：红心 + 魂心 + 永恒心 + 2 × 骨心），炸弹数（封顶 2），挡门怪数量，挡门怪总血量按开局值分 `--hp-buckets 10` 桶（刷怪使总血量超过开局值时最多到 20 桶）。清房后每种血量一个 cell。
- **一个 cell 只留最好的轨迹**：先比受伤量（桥的 `combat.player_damage`，半心；不用血量，因为捡到的心会把受伤掩盖掉），再比逻辑帧，少的好；都相同时取怪物真实掉血（`monster_damage`）多的。房间的"最好结果"：先要清房，再比受伤少、血量多、进度大、帧数少。
- **选择权重**：(1/√(被选次数+1) + 1/√(上次带来新 cell 或更好轨迹之后被选的次数+1) + 0.5/√(探索中经过的次数+1)) × (1 + `--progress-bonus 2` × 进度) × `--hurt-factor 0.5`^(受伤半心数) × 0.5^(连续返回失败数)；进度 = 1 − 挡门怪血量 / 开局值，截在 0–1。胜利 cell 保留但不被选；死亡、超时（`--hurt-ends` 时还有受伤）结束这次探索，不存。
- **探索**：从返回点起最多 `--explore-steps 50` 个决策，粘滞随机：以 `--repeat-prob 0.9` 沿用上一次的移动和射击，否则重抽移动（9 选 1）和射击（以 `--aim-prob 0.5` 朝最近的挡门怪、取水平或竖直中偏移大的方向，否则 5 选 1）；有炸弹时每步以 `--bomb-prob 0.01` 放炸弹；不用主动道具。
- **何时算摘要**：探索中每一步先在 worker 本地查这个 cell 是不是新的或更好；是才取摘要。worker 本地的存档副本由驱动随任务增量下发（存档每次新增或更新都记一条日志，只发这个 worker 还没收到的部分）；副本只会比驱动的旧，所以不会漏掉真正的新 cell。
- **开局**：满血（`--start-hp 6`）、1 个炸弹、没有属性修正，每决策 `--frames-per-decision 4` 帧。

### 用法

驱动为每个 worker 起一个 AB+ 实例（默认严格等价模式），同时探索 `--rooms-at-once` 个房间，每个房间 `--iterations` 次探索。房间由种子按任务文件决定，和训练相同；`--groups-file` 加 `--group` 取分组文件里一组的房间和截止时间，或者用 `--tasks` 给一个任务文件。第一台上用独立副本，不动 `~/isaac-abplus/bridge`：

```bash
cd ~/isaac-abplus/gx/python
PYTHONPATH=. /home/eolc/isaac-rl/.venv/bin/python -u goexplore_abplus.py \
  --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147600000:64 \
  --iterations 1000 --workers 12 --rooms-at-once 8 --name gxn --port 27610 --out ../runs/<运行名>
PYTHONPATH=. /home/eolc/isaac-rl/.venv/bin/python goexplore_summary.py ../runs/<运行名> ../abplus/catalog/abplus_basement1_rooms.json
```

- 输出（`--out`）：`config.json`；`progress.jsonl`（每 `--progress-s` 秒：任务数、失败、重返结果、游戏时和倍率、重置耗时、返回和探索的帧速、摘要耗时、在跑房间的存档概况）；`events.jsonl`（返回失败、`digest_retry`、worker 故障）；`results.jsonl`（每房一行）；`rooms/<种子>.json`（汇总、所有 cell 不带轨迹、胜利 cell 和最好 cell 带轨迹）；`rooms/<种子>.cells.json.gz`（每个 cell 带轨迹，动作码十六进制、一个动作一字节：joint + 45 × 炸弹 + 90 × 道具）。`--resume` 跳过已写出的房间。
- 回放：`goexplore_replays.py --run <运行> --groups-file … --group … --out <目录>` 把每个房间的最好轨迹逐步重放一遍、录下原始观测（并再核对一次终点摘要），写成 `abplus_eval.py` 的目录格式，再用 `abplus_replay_view.py <页面>.html <目录>=Go-Explore` 生成回放页。
- `abplus_probe_play.py` 在一个实例上比较 `play` 和 `step`：同一串探索动作逐步跑、一次 `play`、一半 `play` 一半 `step`，比较最终摘要、观测和停止点，并测两种方式的帧速；还检查坏的 `play` 命令被拒、连接照常可用。
- 端口和实例名要和同时在跑的训练、评估错开（上例 27610 起，实例 `gxn0`…）。

## 运维

### AB+ 进程的内存泄漏与实例回收

严格等价模式原先短路了 `ImageManager::apply_frame_images`，而它还负责回收透明渲染批次：每局泄漏约 0.27 MiB，重置越来越慢（B3；根因见 [ABP_LINUX_REVERSE_ENGINEERING.md §15.11](../../../analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md)）。修复：`tools/stub_render_f.txt` 去掉 `apply_frame_images` 一行（2026-09-24 22:18 远端生效，旧版备份 `.bak-20260924`），轨迹哈希逐位相同，训练数据不变。

**实例回收**（worker 端兜底）：
- 实例打满 `--recycle-episodes`（默认 200）局后，在它作为备用实例准备下一局时，在后台线程里换掉进程再重置（`abplus/instance_recycles` 记次数）。设了 `--recycle-rss-mib` 时，常驻内存到上限也会这样换（C39 t3r2，见上文"内存保护"）。
- 先启后停：先用第二套身份（名字加 `x`，端口加 2 × 环境数）启动新进程，能服务桥接后再停旧进程（等它真正退出，超时则 SIGKILL）；起不来（例如 Steam 在运行但被别处登录挤下线）时保留旧进程继续训练，记 `instance_start_failures` 并报警 `instance_start_failed`。没有第二套身份时才先停旧进程，再用同一端口重开。
- 以上是旧训练器（`train_abplus.py`）的做法。fork 训练器（`train_tok.py`）的 root 不按局数回收，见下一节。

**fork 训练器的内存与 root 卡死**（2026-10-06，EXPERIMENTS.md B16；除 root 回收外默认都开，第一台的 C64 还没用上）：
- 内存按角色（16 个 worker 的整层训练，第二台）：起点库约 140 个停着的克隆体最大，但不涨；root 自己 2 小时只涨约 5 MiB；原来涨的是每个 root 一块 512 MiB 的 PNG 缓存区（2 小时 +1.2 GiB）和训练器里教师记录的主机副本（+0.8 GiB）；每次 48 个楼层的评估另占约 2.1 GiB。`abplus_probe_memory.py` 每 120 秒按角色记一行（worker 要带 `ISAAC_RL_MEM_LOG=<目录>`）。
- 一个 sampler 的所有实例共用 `/dev/shm` 里的一个 PNG 缓存文件（`ABP_PNG_CACHE_FILE`，关闭时删；`ISAAC_RL_PNG_SHARED=0` 回到每个 root 一块）；停着的克隆体停下时 `malloc_trim`（`--trim-parked 1`）；`--teach-gpu 1` 时教师记录不再留主机副本；`--root-anon-mib N`（默认 0 = 关）在 root 的 RssAnon 超过 N 时于起始状态之间换掉 root。改后同样设置 3.5 小时，评估以外可用内存每小时只降约 0.15 GiB。
- root 卡死的两个原因：桥的端口在游戏绑定时已被别的连接占用（训练的端口在临时端口范围 32768–60999 里时会发生；桥现在改绑空闲端口、写进 `ISAAC_RL_PORT_FILE`，记 `port_moves`）；游戏启动时 SIGSEGV（原因没找到；客户端看到进程退出就放弃，`instances/<名字>/stdout.log` 第一行应是 `ABP_TURBO start-up`）。
- 恢复：起始状态建立失败或超时就杀掉实例、重启，同一种子再建一次；时限第一次 `--build-timeout` 60 秒、之后 `--reset-timeout` 20 秒；杀掉之前的进程状态存在 `instances/<名字>/stalls/`。`progress.csv` 记 `stalls`、`build_retries`、`port_moves`、`root_recycles`、`parked_trims`。故障注入 `ISAAC_RL_FAULT_HANG=<worker>:<第几次建立>:<stop|spin|crash>`。

### Steam 依赖

- 可执行文件的 DRM 外壳每次启动都要找到正在运行、已登录的 Steam 客户端，否则执行 `steam.sh steam://run/250900` 拉起 Steam 后自己退出。运行中的实例不受影响，但下一次回收重启会失败，整批卡住，最后训练崩溃（B4）。
- 评估长时间 0 局，或实例日志有 `SteamAPI_IsSteamRunning() did not locate a running instance of Steam`：先查远端 Steam 是否在运行且已登录。
- 预警（abp-mix-01c 起）：学习器每轮检查 Steam 客户端进程（`isaac_bridge/steam_watch.py`，进程名 `steam`、可执行文件 `…/ubuntu12_32/steam`；不用 `~/.steam/registry.vdf` 的 `SteamPID`，它与实际客户端对不上）。
  - 报警渠道：训练日志的 `steam_down` / `steam_up` 事件和 `abplus/steam_ok`、运行目录的 `STEAM_DOWN` 文件、远端桌面通知、`$ABP_ALERT_CMD`（启动训练前设置），例如推送到手机：`export ABP_ALERT_CMD='curl -s -d "$ABP_ALERT_MESSAGE" https://ntfy.sh/<自己的主题>'`。
  - 一退出就报警，未恢复时每 10 分钟重复，恢复时再通知一次；Steam 不在时跳过评估（`abplus_eval.py` 启动时也检查并退出），实例回收推迟 50 局再试。
- 第一台和本机用同一个 Steam 账号：第一台训练期间在本机启动游戏，会把第一台挤下线。第二台用自己的账号。

### 启动与停止

每次训练写一个脚本，放在远端 `~/isaac-abplus/train/<名字>.sh`：训练日志写到 `<运行名>.log`，脚本自己的输出写到 `<名字>.out`。下面是 C40 对照（`rooms_control.sh`）的训练部分，也就是 C37、C40 用的普通房配方：

```bash
cd ~/isaac-abplus/bridge/python
PY=/home/eolc/isaac-rl/.venv/bin/python
CATALOG=~/isaac-abplus/bridge/abplus/catalog
env PYTHONDONTWRITEBYTECODE=1 $PY -u train_abplus.py --train --envs 16 --chunks 1 --n-steps 1024 --batch-size 1024 \
  --micro-batch 256 --segment-length 32 --n-epochs 2 --nice 0 --tasks-file $CATALOG/mixture_positioning_rooms.json \
  --episode-seconds 180 --game-hours 50 --checkpoint-every 30 --eval-every 30 --eval-seeds-count 64 --eval-instances 6 \
  --eval-replays 16 --sampler graph --async-train --greedy-actors 4 --greedy-temperature 0.1 --block-moves \
  --reward-profile combat-hitrate-miss --no-invincible --hurt-ends-episode --hurt-rest-cost --hit-hp 1.25 --align-coef 0 \
  --player-hp-prob 0 --miss-cost 0.01 --miss-cap 20 --aux-coef 1.0 --aux-distance-coef 0 --aux-balance \
  --learning-rate 1e-4 --warm-start <checkpoint 目录> --out ~/isaac-abplus/train/<运行名> > ~/isaac-abplus/train/<运行名>.log 2>&1
```

C37 本身没有 `--hurt-rest-cost`，学习率是 `--learning-rate 3e-4 --lr-schedule cosine --lr-final 3e-5`；这两处的影响见 C40。训练之后接着跑的评估（例如给每个 checkpoint 补跑躲子弹场）照 `rooms_control.sh` 的后半段写。

后台启动：

```bash
cd ~/isaac-abplus/train; nohup setsid bash <名字>.sh > <名字>.out 2>&1 < /dev/null & disown
```

不要用 `&&` 把别的命令接在 `nohup` 前面：整串命令会一起进后台，ssh 会一直挂着。

- 开训后日志里的 `game/speed_x_realtime` 就是实测速度；各次训练的实际速度见 EXPERIMENTS.md。
- 停训练：先停评估链的 bash，再停评估进程，最后 `kill -INT` 训练进程（`train_abplus.py` 开头把继承来的"忽略 SIGINT"恢复成默认处理，见"从头训练"）。SIGTERM 也可以：worker 发现管道关闭，各自关掉自己的 AB+ 实例。
- 以前的 `train_abplus.sh`（等模拟器训练结束后，从模拟器的 checkpoint 热启动）已删除，C2 里提到的就是它。

## 用法（远端 Ubuntu，`~/isaac-abplus`）

### 训练看板（2026-09-27）

`abplus_dashboard.py`（服务端，只用 Python 标准库）加 `isaac_bridge/abplus_dashboard.html`（页面，wandb 风格，图表用 jsdelivr 上的 uPlot）：
- 地址：第一台 http://100.76.185.120:8790/ ，第二台 http://10.19.131.132:8790/ 。第一台只绑在 Tailscale 地址上，同一个 tailnet 里的设备能打开。
- 只读。服务端读 `~/isaac-abplus/train/<运行>/` 下的 `progress.csv`、`config.json`、`episodes.jsonl`、`evaluations*/<检查点>/[sampled/]results.jsonl` 和 `replays.html`；运行名和检查点名必须是训练目录下的普通目录名。
- 页面每 10 / 20 / 60 秒增量刷新：
  - 左边是运行列表，勾选的运行叠加在同一张图里，颜色按运行名固定；
  - 顶部可以切横轴（游戏小时、训练轮、墙钟）和平滑系数（去偏的 EMA，淡线是原始值）；
  - 曲线按结局、瞄准、辅助头、奖励分项、PPO、系统分组，没归类的在"其他指标"里；各图光标同步，悬停时列出各运行的值；
  - 当前运行另有概况（进度、速度、预计剩余、配置）、评估表（每个检查点两种模式，带回放链接）、最近 200 局和机器状态（GPU、CPU、内存、磁盘、Steam）。
- 启动：
  ```bash
  cd ~/isaac-abplus/bridge/python
  setsid nohup python3 -u abplus_dashboard.py --host 100.76.185.120 --port 8790 > ~/isaac-abplus/dashboard.log 2>&1 < /dev/null &
  ```
  改页面不用重启，每次请求都会重新读页面文件。

### 评估与回放

**训练中的周期评估**：每 `--eval-every` 轮一次，先跑最优动作（与以前可比），再按概率采样（`--eval-sampled`，默认开），各录 `--eval-replays` 局，结果在 `evaluations/<checkpoint>/`（采样在其下的 `sampled/`），回放页 `evaluations/<checkpoint>/replays.html`（`abplus_replay_view.py`）；胜率按开局炸弹 0 个 / 1 个以上分开统计。分组训练和对战场另有 `evaluations-<组或方式>/`。

**单独评估**一个 checkpoint，例如在躲子弹场（`rooms_control.sh` 的写法）：

```bash
cd ~/isaac-abplus/bridge/python
A="--checkpoint <checkpoint 目录> --tasks-file ~/isaac-abplus/bridge/abplus/catalog/mixture_target_dodge.json --target-from-tasks
   --episode-seconds 30 --seeds range:2147490000:64 --instances 6 --device cpu --replays 16 --port 27340 --name dg"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /home/eolc/isaac-rl/.venv/bin/python -u abplus_eval.py $A --out <输出目录>
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /home/eolc/isaac-rl/.venv/bin/python -u abplus_eval.py $A --out <输出目录>/sampled \
  --stochastic --sample-seed 0
/home/eolc/isaac-rl/.venv/bin/python abplus_replay_view.py <输出目录>/replays.html "<输出目录>=最优" "<输出目录>/sampled=采样"
```

- 和正在跑的训练、评估同时用时，`--port` 和 `--name` 要和它们错开。
- `--device cpu`：训练进程占着 GPU 时，多进程共用 GPU 要排时间片，单步推理平均约 30 ms；CPU 单线程轻载时 7–9 ms，6 个 worker 满载时约 12 ms。
- `--repeat N` 把每个种子跑 N 次，用轨迹哈希检查可复现性。
- `--episode-seconds S` 用 S 秒代替 checkpoint 的每局时长，死亡扣分的截止时间也跟着换；在别的任务上评估时用。例如练习场的 checkpoint 回普通房评估，要配 `--target-from-tasks` 加普通房的任务文件，再加 `--episode-seconds 180`（C35）。
- `--mortal` 让在无敌下练出来的 checkpoint 按会受伤来评估；对战评估见"对战场"。
- `--verify-steps N` 在每局前 N 步把增量推理和整窗前向逐项比对（迁移评估里最大差都是 0.0）。
- 结果按种子追加写入 `results.jsonl`，中断后重跑会跳过已完成的种子。

**模拟器迁移评估**（A2 的做法）：

```bash
cd ~/isaac-abplus/bridge/python
PYTHONDONTWRITEBYTECODE=1 nice -n 19 /home/eolc/isaac-rl/.venv/bin/python -u abplus_eval.py \
  --checkpoint ~/isaac-abplus/checkpoints/u0425 --seeds ~/isaac-abplus/eval/seeds-e2.json \
  --out ~/isaac-abplus/eval/u0425-det-e2 --instances 6 --device cpu --verify-steps 3 --replays 16
python abplus_transfer_report.py --sim <e2/results/u0425/det> --abplus <results.jsonl> \
  --sim-replays <e2/replays/u0425/det> --abplus-replays <replays> [<more replay dirs>] --out summary.json
```
