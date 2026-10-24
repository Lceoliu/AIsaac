# B34 反事实分支教师：设计（2026-10-16）

用户 10-16 决定：加进 C99（从零、无 GRU、无门、B35 观测），起始参数 4 支 × 180 游戏秒；每游戏小时的选择点数 c 按吞吐调
（起步 6），瓶颈是 actor 的 GPU 推理而不是 CPU（HPC160 配额 160 核，C96 时负载 28）。

## 1. 它是什么

在策略做选择的那一步，用 fork 克隆把引擎复制成 K 份，各走一条路（一条 = 策略原样 taken），同一套随机数（MT 重设钩子），
各跑到本层结束或 T 秒截断，比较折扣回报；"如果当时那样做"的实测差异成为学习信号。它是教师 v2（B19，`tok_teacher2.py`：
lane、意图、打分、复跑定噪声底、加权模仿记录）的推广：触发点从受伤/死亡扩到任意选择点，意图加入"朝目标走"，视野从 20 秒
扩到 180 秒，并补上 lane 的时间侧状态。

## 2. 选择点（触发都是观测里的事实，不含"该怎么做"）

- 视野里**第一次**出现：掉落物（type 5：硬币/钥匙/炸弹/心/药/卡/饰品）、道具底座、带钥匙时的锁门、商店门/宝箱房门（门 token 的
  目标类型）；
- 清房后站在 ≥2 扇门前的那一步（清房事件 EV_CLEARED 后的第一条记录）；
- 每游戏小时 `--cf-random`（默认 1）个随机记录。
- 配额：每个 worker 每游戏小时最多 `--cf-points` 个（默认 6），超过时优先保留本局还没出现过的触发类型；队列满时丢最旧的非死亡点。
- 死亡教师（`--teacher-death-depths`）照旧独立运行。

## 3. 分支（K = `--cf-branches`，默认 4）

1. taken：策略原样（本局自己的后续，不另跑：用 worker 记下的奖励 + 截断处 actor 的价值，同教师 v2）。
2. target：朝触发目标走（B23 `tok_pickup.move_toward` 的移动覆盖；射击/炸弹/道具/药每步仍由 actor 给），到达（拾取发生 / 进门 /
   目标消失）或 `--cf-target-seconds`（默认 20）后交回策略。
3–4. 粘性高温策略（教师 v2 的 sticky 意图，各一个种子）。
- 每支跑到本层结束或 `--cf-seconds`（默认 180）截断；分数 = 训练器同一套奖励按 γ 折扣的和 + 截断处 γ^T·V_global（全局价值头，
  γ=0.999；无全局 token 时用 V）。
- 最好的分支换一个 MT 种子复跑一次定噪声底：只有 best − taken > `--cf-margin`（0.3）且 > |best − 复跑| 才算改进。
- 分支在 worker 的 fork 子进程里跑（`--teacher-fork 1` 的机制），记录经 lane 送 actor 答，不进 rollout。

## 4. lane 的时间侧状态（B30 v1 拒绝教师 v2 的原因）

分支从一局中途 fork，策略需要父 worker 在 fork 那一步的时间侧状态：GraphActor 给每条 lane 一个槽位，fork 时把父槽位的每层 K/V
环、写指针、房间 token 表、prev_o、prev_map、steps/nroom **设备内复制**到 lane 槽位（`TemporalState` 加 `copy_slot_(src, dst)`）；
分支结束后 lane 槽位清零。复跑同理。不带时间侧的模型照旧。

## 5. 教

- 模仿：改进分支的前 `--cf-distill-steps`（默认 8）步做加权模仿记录（权重 w = min(best − taken, `--cf-cap` 2)），走教师 v2 的
  环与并入小批的路径（`--teacher-v2-buffer/batch/reuse` 复用）。
- 可选 `--cf-advantage 1`：选择点那条记录的 PPO 优势替换为 (best − taken)（stop-grad 常数）。默认关，先只用模仿。
- 自检（progress.csv `cf_*`）：选择点数按类型、分支数、改进份额、`cf_prior`（改进分支第一步动作在当前策略里的 log π，应随训练
  上升）、`cf_gain0_mean`、`cf_cpu`、`cf_lane_calls`、`cf_wait_s`。

## 6. 预算与验收

- c = 6 时分支决策量 ≈ 主采样的 1.2 倍：CPU +7 核，actor 推理行数 ×2.2，预计 C99 从 ~900× 降到 700–800×；稳定后按吞吐把
  `--cf-points` 提到 10–12。
- 验收：`cf_prior` 上升、`cf_improving_share` 下降；训练局里 gathered / 自主拾取 / 开锁门次数上升；128 种子过层不低于同小时数
  的 C77/C87 曲线。
