# B40 草稿：反事实教师的"炸墙"触发点 + 门类目标等清房再走（`--cf-wall`、`--cf-defer-target`）

（草稿，供并入 `bridge/abplus/EXPERIMENTS.md`；分支 `b40-bomb-trigger`（基于 abplus-training 5d600da），提交 5ea44ca、3958686、897e58b 及本草稿，已推到 HPC 裸仓库；沙盒 HPC `ws16`（`mk_ws16.sh`，Xvfb `:96`，worktree 897e58b，库已重编）。另含协调者追加的两项：A30 的门类目标延迟（`--cf-defer-target`）和停滞规则选项（`--stall-progress pickups`）。）

## 实验目的

用户 10-18 同意"先准备 (b)"：不写任何奖励项，让策略自己发现"炸墙找隐藏房 / 炸开资源"。做法沿用 B34 反事实教师：教师只提出一个分支——"走到这面墙的中间放一颗炸弹"——由同一套随机数下的实测折扣回报决定它是否比策略原样更好，更好的分支走现有的蒸馏路径（V2 环、模仿损失）。

同一分支顺带修 A30 发现的 B34 问题：门类触发（宝箱 / 商店 / 锁门）在进房第二条记录就触发，69%（第 2 层起 81%）的时候房间还在打，target 分支 = "停火朝关着的门走 20 秒"，比策略差 5.6、死亡翻倍，这类点几乎不可能胜出，等于一直在教"别去"。A30 的 A2（等清房再走）在第 1 层给出 +1.40 ± 0.51（原样 target −1.10）。

## 实验方法

**触发点 `wall`**（`isaac_bridge/tok_cf.py`；`--cf-wall 1` 才检测，默认关）

- 条件：房间已清（player 第 23 列 `Room:IsClear`），玩家至少 1 颗炸弹（ROW `bombs`），本房间有"候选隐藏房墙"：房间在小地图上的某个格子（map 通道 3）的某一面墙，邻格在 13×13 网格内、小地图上没显示（通道 0）、不挨着 Boss 房（通道 4），这面墙上没有本房间的门，且这一面有门位（窄房间的长边没有门位）。Boss 房 / 隐藏房 / 超级隐藏房里不触发。
- 几何：房间内区 = 地形 `inside` 通道（ROW grid 通道 0）的外接框 = Room TopLeft..BottomRight，单位"格"；每个小地图格子占内区的 1/bw × 1/bh；墙中点按格子算（L 形房间的内角墙也算），炸弹点 = 墙中点往里一格（第一排格子的中心，正对门位）。每扇门按位置归到最近的墙中点。桥的门列表本来就不含隐藏门（`abp_bridge.lua` 的 `visible_door`：DOOR_HIDDEN 当墙），所以不泄露隐藏房的位置。
- 多个候选时取 p_vis 最高的，同分取离玩家近的。p_vis = AB+ 隐藏房选址规则在"可见小地图"上的近似：每个未显示、挨着已显示格子、不挨 Boss 的格子得分 U{10..14} + 偏移（已显示邻格 1 个 −6、2 个 −3、3 个以上 0），最高者胜、同分均分；`win_values` 是 `rl/macro/isaac_macro/secret.py` `win_probabilities` 的精确积分移植（单元测试逐值对照）。
- **没有直接调用 rl/macro 的隐藏房推断器**：`secret_posterior` 需要整层可见地图（房间形状、类型、布局门位），ROW 的小地图只有合并后的格子（无形状、无房间编号），而且一局中途只知道一部分楼层；rl/macro 也不在 bridge 包和沙盒的导入路径里。p_vis 保留了它的主导项（邻格数偏移）和同分规则。
- 一个房间每局最多一次：`detect` 不把 wall 记为已见，只有记录真正把它交给配额（`Quota.offer`，接受或跳过）时由调用方 `consume_wall` 记下（本房间的 seen + 本局的房间键 `(stage, 房间第一个小地图格)`），所以同一条记录被别的触发占用时下一条记录再试；已清但没有候选的房间立刻记为已见。与其他触发共用 `--cf-points` 配额和"本局未出现的类型优先"规则；`TRIGGERS` 里 wall 在 random 之前（random 仍是最后一个；索引只在运行时用，V2 环不进检查点）。
- 可观测性：`rock`（染色石头 / 挡着资源的可炸障碍）**没做**：地形通道只有 solid / destructible，分不出染色石头（`terrain_channels` 里没有 grid 变体），ROW 也不带网格实体列表。

**分支 `bomb_at`**（`tok_cf.BombIntent`）

- 朝炸弹点走（`tok_pickup.move_toward`），离炸弹点 ≤ 0.6 格就到达；卡住（8 个决策没有靠近 0.05 格）且在 2 格以内也算到达；到达那一步发 move 0 + 炸弹头 1（射击 / 道具 / 药仍用策略的），之后整个动作交还策略，分支照常跑满 `--cf-seconds`（180 秒）或出层，所以进隐藏房、捡东西都算进回报。手里没炸弹、换房间或 `--cf-target-seconds`（20 秒）内没到就不放。
- 每点的分支：taken（策略原样，不另跑）+ bomb_at + 2 个粘性高温分支（K=4 不变）。
- 蒸馏：bomb_at 分支的记录保留到放炸弹那一步（至少 `--cf-distill-steps` 8、至多 `--cf-wall-distill` 64），因为走过去 + 放炸弹才是要教的东西；其他触发不变。记录的 intent 码 `INTENT_BOMB = 11`。
- 计数（worker stats → progress.csv）：`cf_points/wall`、`cf_prior/wall`（已有的按触发类型统计自动带上），`cf_wall_done`（跑了 bomb_at 的点）、`cf_wall_bombed`、`cf_wall_opened`（放炸弹后邻格在小地图上出现，或这面墙上出现了门）、`cf_wall_gain`（bomb_at 分数 − taken 之和）、`cf_wall_best`（改进点的最好分支就是 bomb_at）、`cf_wall_p`；窗口比例 `cf_wall_opened_share`、`cf_wall_bombed_share`、`cf_wall_best_share`、`cf_wall_gain_mean`、`cf_wall_p_mean`。这些统计取自第一次运行（复跑定噪声底之前）。

**门类目标等清房（A30 的 A2，`--cf-defer-target 1`，默认关 = B34 原样）**

- 选了 (a)（target 等清房）而不是 (b)（把门触发挪到清房那条记录）：检测流程不动（"第一次看见"仍是触发事实，配额、每房一次、按类型统计都不变），只改分支：门类目标（locked / shop / treasure / cleared 的 door 目标）的 `TargetIntent` 和所有 `BombIntent` 在房间没清时整个动作用策略的，清房那条记录（`player[23]` 或 `EV_CLEARED`）起才开始走，20 秒上限从那时算；清房前换了房间就不再开始。拾取物目标不延迟。
- 蒸馏从分支自己的第一个决策开始（`play_branch` 的 `keep_from`：只保留 j0 之后的记录，`ret` 按原下标取），所以模仿记录教的是"清房后走过去"，不是策略自己的战斗。从没开始的最好分支不写记录。
- 计数：`cf_defer_points`、`cf_defer_started`、`cf_defer_wait`；progress.csv `cf_defer_started_share`、`cf_defer_wait_s`（开始前等了几秒）。
- wall 触发本来就要求已清房，延迟对它不起作用。

**停滞规则的"进展"可选含拾取（`--stall-progress pickups`，用户要求；默认 `events` = 原规则）**

- 原规则：整层局 `--floor-stall-seconds`（60 秒）内没有"进展"就结束，进展 = 进新房 / 清房（含 Boss、出层）或这一步打伤了怪；捡硬币 / 钥匙 / 炸弹 / 心、开箱子、炸开石头都不算，所以花时间收集或炸墙的策略会被截断。
- 新选项：`pickups` 时，玩家 0 的硬币、钥匙、炸弹、生命单位（红心 + 魂心 + 永恒心 + 2 × 骨心，即 ROW `player[7]` 的分子）任一比上一条记录多，也把进展时间重置为本条。只看增加，减少（花钱、用炸弹、受伤）不算。
- 两个编码器都改：`tok_obs.encode_row`（`EpisodeState.stall_pickups` / `prev_res`）和 `abp_turbo.c` 的 `abp_row_encode`（ctx 新增 `RC_STALL_MODE`、`RC_HAVE_RES`、`RC_PREV_RES..+3`，RC_COUNT 40 → 46；版本号仍是 7，旧的 .so 因 RC_COUNT 不同会被 `fast_row_function` 拒绝、退回 Python 路径而不会写错）；`FastRow` 在快慢路径之间同步这些值。接线：`TokSamplerConfig.stall_progress`，`train_tok` / `eval_tok --stall-progress`，tok_floor 的主局、分支局、lab 局和 `tok_teacher2.play_branch` 的 EpisodeState 都带上它。
- **会改变策略的输入**：停滞比例 `min((t − progress_t) / stall, 1)` 是观测（player 第 30 列）。开启后，捡到资源的那几秒这一列会回落，C99 权重在这些时刻看到的是训练里没见过的取值组合（以前只有进房 / 清房 / 打怪才回落）；也就是说 C103 同时改了环境（局更长）和输入分布，与 C99 对比时要记住。
- C103 默认开启；watcher 默认用 `events` 规则评估（与 C99 / C101 / C102 的曲线可比），`EVAL_STALL=pickups` 改用训练规则。

**C103 启动脚本**（`bridge/ops/hpc_c103.sh`，沙盒 `ws16`）

- COMMON / FAST 与 `hpc_c99.sh` 相同，加 `--cf-wall 1 --cf-wall-distill 64 --cf-defer-target ${DEFER:-1} --stall-progress ${STALL_PROGRESS:-pickups}`（`DEFER=0` 关延迟，`STALL_PROGRESS=events` 用原停滞规则）；B38 四卡布局（actor 0 / 2，learner 1，watcher 3）；`MB` 默认 4096；`--resume .../ws9/runs/c99-scratch/checkpoints/segment2-final.pt --game-hours 10000 --out $R/c103-wall`，name hc103，端口 49250（训练）/ 49480（watch）/ 49660（smoke）/ 49695（probe）。
- `smoke`：8 workers、约 60 次更新（18.3 游戏小时，rollout 8192）、两卡进程布局，打印 cf_wall_* / cf_defer_* 列；`probe`：`abplus_probe_wall.py`。

## 实验结果

**CPU 测试**（`tests/test_cf_wall_smoke.py`，68 项，本地与主机 B `ws16` 的 `/opt/aisaac-env` 均通过；同时重跑 `test_cf_smoke`、`test_lane_batch_smoke`、`test_no_active_smoke`、`test_eval_ablate_smoke`，均通过）：
- 检测：1×1 房间两面有门、两面邻格未知 → 只有无门的两面是候选，邻格已显示两个的那面（偏移 −3）排在只挨一个的（−6）前面，炸弹点 = 上墙中点往里一格 (6.5, 0.5)；已显示但无门的邻格不是候选；未清房 / 没炸弹 / 开关关闭不触发；Boss 房、隐藏房、挨着 Boss 的邻格排除；没有候选的已清房间立刻记为已见；清房那条记录 cleared 与 wall 同时出现时 cleared 在前。
- 每房一次：`detect` 不消耗，下一条记录还能找到；`consume_wall` 之后本房间不再触发，同一局再进同一房间也不触发，另一个房间、下一层的同一格照常触发。
- 几何：2×2 房间 8 面墙和炸弹点、门只归到它所在的那面墙；L 形房间的两面内角墙；横向 / 纵向窄房间长边无门位。
- p_vis：`win_values` 与 rl/macro `win_probabilities` 逐值相等（6 组随机偏移；HPC 上没有 rl/macro 时改用 2 万次蒙特卡洛对照），和为 1。
- `BombIntent`（模拟走路，含一步动作延迟）：先一路朝上走，到点那一步恰好一个 move 0 + bomb 1（其余头是策略的），之后完全是策略的动作；炸弹数 1 → 0；炸之前地图不变不算 opened，邻格出现或这面墙上出现门才算；卡在 1.3 格处 8 步后放炸弹，卡在 3 格外不放；没炸弹不放；换房间不放。
- 延迟（`--cf-defer-target`）：门目标在未清房时 5 条记录都原样返回策略动作，清房那条起朝门走，20 秒上限从清房起算；关闭时立即走（B34 原样）；等待中换房间则永不开始；延迟的 bomb_at：未清房 4 条原样，之后走、恰好一颗炸弹、再交还策略；`intents()` 只延迟门和墙目标，拾取物不延迟。
- `cf_search`（替身分支）：bomb_at 排第一、保留 `cf_wall_distill` 条记录，其摘要取自第一次运行（复跑之前），蒸馏条数 = 炸弹步 − 开始步 + 1（并受上限截断）；其他触发保持 `cf_distill_steps`、没有 wall 摘要。

**引擎探针**（`abplus_probe_wall.py`，主机 B `ws16`，AB+ 引擎，以撒 1 颗炸弹，地下室 I 开局房；房间真值用 Lua `GetRooms()` 只做判分）：
- 两轮共扫描 74 个种子，起始房旁边有隐藏房的 24 个（约 30%，全是普通隐藏房 type 7），对照 12 个。
- 有隐藏房的 24/24：隐藏房那面墙都在候选里，23/24 是 `detect` 选中的那面（p_vis 排第一）；按它的炸弹点跑 bomb_at 分支（替身策略：放完炸弹走回房间中央），**24/24 打开**：炸弹数 1 → 0，房间多出一扇门（门列表 +1，归到被炸的那面墙），邻格在小地图上出现；0 次受伤。
- 对照：有候选墙的 10 个种子炸了 10 次，0 次打开（炸弹照样用掉）；另有 2 个起始房四面都有门、没有候选。
- 结论：门位置 = 墙中点、往里一格放炸弹足以炸开隐藏房；隐藏门不在桥的门列表里，打开后作为普通门出现，所以 `opened` 判据（这面墙出现门 / 邻格出现）在引擎上成立。

**停滞选项的两条编码路径**（`tests/test_stall_progress_smoke.py`）：合成的整层 payload 序列（25 条，硬币 / 钥匙 / 炸弹 / 魂心 / 骨心 / 红心各增加一次，炸弹 / 红心 / 硬币各减少一次）：关闭时进展时间、停滞比例、done 与原规则相同（t=5 起停滞）；开启时每次增加都重置进展、减少不重置，只有 player 第 30 列和 done 不同。主机 B 上用本提交编译的 `libabp_turbo.so`（`ISAAC_RL_TEST_TURBO_SO`）：两种模式下 FastRow（24 条走 C 快速路径）与 encode_row 每条记录逐字节相同、进展时间相同；中间 4 条强制走 Python 慢路径再回到快速路径也逐字节相同。`test_obs_v2_smoke`（含 (e) FastRow == encode_row）在本地和主机 B 都通过。

**主机 B 冒烟：没有跑。** 主机 B 的 C100 512 种子评估到 10:55 UTC 还没结束，之后主机 B 被用户重建（GPU 被别的租户占用）。冒烟命令（等主机 B 回来或在主机 A 的 C101 结束到 C103 开始之间；沙盒 ws16 在 /101063 上，worktree 897e58b，`tools/libabp_turbo.so` 已按本提交编译，`hpc_c103.sh` 已复制）：

```
cd /101063/AIsaac/ws16 && CUDA_VISIBLE_DEVICES=0,1 bash hpc_c103.sh smoke
```

要看：Traceback 0；`cf_points/wall` > 0、`cf_wall_done`、`cf_wall_bombed_share`（应接近 1）、`cf_wall_opened_share`（随机房间约 0.1–0.3？没有先验）、`cf_wall_gain_mean`、`cf_prior/wall`；`cf_defer_started_share`、`cf_defer_wait_s`；`x_real_time` 与 C102 冒烟同配置对比（wall 点每点多走最多 64 条蒸馏记录、炸墙前后的分支本身与其他触发同价）。

## 未测试 / 风险

- 全配置训练路径（tok_floor 的 cf 线程 / fork 子进程里真实跑 `cf_search` + `play_branch` 的 `keep_from`、`ring_write`、learner 上的 cf_prior/wall 等列）只有单元测试和替身分支，没有在引擎 + actor 上跑过：冒烟未做。
- `play_branch` 的延迟保留窗口（只保留 j0 之后的记录，`ret = G[kidx]`）只在替身分支上间接测过；时间侧 / GRU 的存储输出按记录保存，理论上与位置无关。
- 探针只覆盖地下室 I 的 1×1 起始房（普通隐藏房 type 7；24/24 打开）；超级隐藏房、2×2 / L 形 / 窄房间的炸墙点和门归属只有合成测试；L 形房间的内角墙在真实引擎里的炸弹点位置没验证。
- p_vis 是近似（不知道没去过的房间、布局门位、Boss 以外特殊房间的黑名单），探针里 24 个里 23 个选对，但起始房候选少；中后期房间候选多时的命中率未知。
- bomb_at 放完炸弹交还策略：策略如果站在炸弹上会受伤（替身策略会走开，探针 0 受伤），实测回报自己决定——这正是设计，但早期 wall 点可能多数"不改进"。
- 延迟开启后，门类点在清房前的那段是策略原样 + 同一随机数，和 taken 很接近；只有清房后的那段决定差异，噪声底（复跑）照旧。
- `--stall-progress pickups` 改变观测分布（停滞比例列）和局长；对 C99 权重的影响没测。
- 运行中的其他沙盒不受影响（各自 worktree 和 .so）；以后任何沙盒拉到本分支都必须重编 `libabp_turbo.so`（RC_COUNT 变了，旧库会被拒绝、整条记录走 Python 慢路径——不会错，但慢）。`mk_ws16.sh` 会比较 abp_turbo.c 自动重编。
- `rock` 触发没做（染色石头在 ROW 里不可观测）。
