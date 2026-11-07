# 当前 block-memory visible-v1 输入（2026-11-07）

新 block 模型的 z/g 是 Temporal hidden 的 zr/zg 投影，不是旧 player/global 输入token。
原始记录保留legacy统计字段，但actor只读取 `visible_schema/visible_bytes/visible_frame`；完整新帧字节预算131072，P/E/N/S实际长度解码，无背包16项/槽4项截断。R/L无损离散格点编码；记录布局由 `tok_obs.ROW` 决定，不再是20800 B。
checkpoint、offline job、snapshot和共享内存均绑定 `isaac-visible-prlmens-g-v1`，旧schema拒绝，不隐式转换。下文20800 B及player token描述只适用于历史legacy日志。

当前format.json版本为2，记录input_schema、feature_semantics与ROW布局。Reader检查schema、dtype/record长度；合并world-model数据不得混用表示语义。原始ROW固定151880 B，其中visible_frame预算131072 B；有效帧长度由visible_bytes给出。z/g语义为temporal-zr-zg，只有单条ROW不足以重现时序表示，须配套对应历史/snapshot。

# 历史legacy日志格式（下文旧z/g与20,800 B数字不适用于visible-v1）

配套设计：`PREDICTIVE_MEMORY_DESIGN.md` §5.1。代码：`bridge/python/isaac_bridge/tok_stream.py`。读取用 `tok_stream.iter_records`，它能容忍崩溃截断的分片。依赖 `zstandard`（`requirements-training.txt`）。

## 目录与分片

运行目录下的 `stream/`：

| 文件 | 内容 |
|---|---|
| `format.json` | 版本、`width`、`heads`、抽样率、状态字段名、记录大小、`stream` 的字段描述（自描述） |
| `stream-w<worker>-p<part>.zst` | 决策流记录，每 worker 一条 zstd 流 |
| `row-w<worker>-p<part>.zst` | 完整 ROW 记录，每 worker 一条 zstd 流 |

每次更新刷新一个 zstd block，所以运行中的文件也能读到最近一次更新之前的内容。每个分片写满 1 GiB（未压缩）换下一个；续训时分片号接着往后排，不覆盖。

## 抽样

- 单位是**局**，不是决策：`h = episode_hash(worker, seed, episode)`，取值范围 `[0, 10000)`。
- `h < rate × 10000` 的局进决策流（默认 1/4）；`h < row_rate × 10000` 的局再存完整 ROW（默认 1/50）。**ROW 集合是决策流集合的子集**，所以这些局两种记录都有。
- 只写已闭合的决策（已知下一步结果）。rollout 末尾 open 的决策在下一个 rollout 里作为决策 0 写出，每个决策恰好写一次。
- 这样同一局的决策连续，不会有缺口；跨 rollout 也连续。例外：写入队列满时整批丢弃（见下）。
- 只记录主局。教师和反事实分支的记录不在这里。

## 决策流记录（packed，小端）

| 字段 | 类型 | 含义 |
|---|---|---|
| `z` | int16 × width | `z_t`，player token 归一化后的表示，**bfloat16 的原始位**。width 是 Spatial 宽度（C99：192） |
| `g` | int16 × width | `g_t`，global token 归一化后的表示，同上 |
| `state` | f4 × 13 | 显式状态，未标准化，字段见下 |
| `action` | i1 × heads | 这个决策采样的动作 |
| `pending` | i1 × heads | 进入这个决策时还在进行的动作（策略输入的一部分） |
| `reward` | f4 | `r_t`，下一条记录闭合这个决策时算出的奖励 |
| `hurt` | f4 | 走到这条记录的那一步损失的半心数（决策 t−1 → t） |
| `close_hurt` | f4 | 决策 t → t+1 这一步损失的半心数 |
| `events` | u1 | 走到这条记录的那一步的 `EV_*` 位 |
| `close_ev` | u1 | 决策 t → t+1 这一步的 `EV_*` 位 |
| `close_done` | u1 | 闭合这个决策的记录的 `done`：0 继续，1 通关，2 死亡，3 超时，4 分支截断，255 worker 错误 |
| `terminal` | u1 | 这一步之后局结束 |
| `first` | u1 | 局的第一条记录 |
| `stage` | u1 | 这条记录的层 |
| `room_type` | u1 | 这条记录的房间类型 |
| `room` | u1 | 小地图上玩家房间的第一个格子（0–168），地图不显示时 255。与 `stage` 合起来才是房间 ID |
| `worker` | u2 | worker 编号 |
| `t` | i4 | 局内决策号（ROW `t`） |
| `episode` | i8 | worker 内的局序号 |
| `seed` | i8 | 层种子 |
| `update` | u4 | 写入时已完成的更新数 |

局的唯一键是 `(worker, episode)`；同一 run 内不重复（续训后 episode 计数可能重新开始，这时再加 `update` 区分）。

`state` 的 13 列（`tok_aux.STATE`）：`x, y`（房间内归一化位置）、`hearts, soul`（红心、魂心 /12）、`black, eternal, bone`（`phearts`）、`coins, bombs, keys`（/50、/10、/10）、`enemies`（真实 token 内的 NPC 数）、`room_type`、`stage`。

事件位（`tok_obs.EV_*`）：1 清房、2 首次进入新房、4 Boss 房清完、8 换层、16 得到道具、32 用主动道具、64 用药丸或卡牌。

## 关于 `z_t`、`g_t` 的含义

- 来自**每次更新开始的 recompute 前向**（学习器当时的权重，inference mode），不是 actor 当时算的。权重最多差一次更新。
- 权重随训练变化，所以同一状态在不同更新里的 `z` 不可比。按 `update` 分段或看得出漂移时，离线评估要注意这一点。
- 需要换表示重算时，用完整 ROW 记录。

读 bfloat16：

```python
import numpy as np, torch
z = torch.from_numpy(rec['z'].astype(np.int16).copy()).view(torch.bfloat16).float()
```

## 完整 ROW 记录（packed，小端）

| 字段 | 类型 | 含义 |
|---|---|---|
| `row` | `tok_obs.ROW`（C99：20,800 B） | 决策的完整观测，和 rollout 逐字节相同 |
| `action` | i1 × heads | 采样的动作 |
| `pending` | i1 × heads | 进行中的动作 |
| `worker` | u2 | worker 编号 |
| `update` | u4 | 写入时已完成的更新数 |

ROW 的布局随 `tok_obs.py` 变化，`format.json` 的 `row_fields` 记录了写入时的版本。

## 写入与丢弃

- 学习器线程只负责收集：host 记录在 `fresh` 之前拷贝，GPU 行随本次更新的工作异步进 pinned 缓冲区，在更新末尾的同步之后读取。
- 写入线程负责压缩（zstd level 3）和落盘，队列最多 8 个更新。
- **训练绝不等日志**：队列满时整批丢弃，计入 `log_dropped`。所以 `log_dropped > 0` 时对应更新的决策有缺口；要用连续序列时按 `(worker, episode)` 检查 `t` 连续。
- 写失败（比如磁盘满）只停日志不停训练，错误写进 `log_error`。

## progress.csv 列

`log_rows`、`log_row_rows`（累计记录数）、`log_raw_gb`（累计未压缩 GiB）、`log_dropped`（累计丢弃行）、`log_queue`、`log_write_s`（写入线程累计秒数）、`log_error`、`u_log`（学习器线程在日志上的秒数）。

### visible-v1 / block-memory 并行诊断（2026-11-07）

新模型使用 `isaac-visible-prlmens-g-v1`，完整ROW长度以本次 `format.json` / `tok_obs.ROW` 为准。下面的C99体量估算仅是历史数据，不代表新输入布局，也不授权加载旧权重。

- `block_main_ready_ms`：主请求进入 `step_main` 到主结果就绪回调的平均墙时；不等旧版本教师组全部完成。不是纯CUDA kernel时间。
- `block_control_s`：本采集轮 `capture_main` 的累计墙时，含控制处理和选择点快照，不包含所有 `tick/serve_wait` 控制调用。
- `infer_ms`：整个 `step_main` 服务调用平均墙时，现含主回复回调和后续教师服务；不能与 `block_main_ready_ms` 混用。
- `block_teacher_forward_calls`：累计教师generation网络批次数，包含与主请求合并的批次；不同generation始终分开。
- `block_mixed_replay_calls`：上述教师子批中同时含正常搜索与replay的累计批次数。
- `block_ppo_selected_blocks / unique_objects / encoded_blocks / historical_rows / tail_rows`：本次PPO＋合并教师路径各forward的累计重算计数；分别是选中块、对象去重块、实际编码来源块、历史行、尾部行。不是跨optimizer step全局去重，也不覆盖独立教师优化路径。
- `idle_share`：worker等待答案的统计比例，不是GPU空闲率；`collect_s`与`update_s`存在重叠。在线速度使用相邻learner完成时间的 `wall_s` 与决策数差，按4帧/decision÷30fps换算。

## 体量估算

10,000 游戏小时 = 2.7 亿决策（4 帧/决策）：

- 决策流：1/4 × 2.7 亿 × 876 B ≈ 59 GB 未压缩。`z`、`g` 接近随机，压缩比低：真实数据约 1.4 倍，压缩后约 42 GB。
- 完整 ROW：1/50 × 2.7 亿 × 20.8 KB ≈ 112 GB 未压缩。大部分是零和重复的地形，压缩比高：真实数据约 84 倍，压缩后约 1.3 GB。

压缩比来自 3090 上一次 6 游戏小时的冒烟（约 5 万条决策流、1,400 条 ROW）。局长分布不同时比例会变；局层面抽样在短跑里波动也大。
