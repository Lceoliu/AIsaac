# memory-chain v3：有界梯度 horizon

日期 2026-10-24。候选代码在 `temp/memory-chain-v3/candidate`（本地 git，验收靠 `git diff 64ae451 bfd3656`；05e7cde 只额外加了默认关闭的内存计量）。没有改动 `temp/memory-chain-v2` 和 `temp/room-memory-write-v1` 里的任何文件。

> **暂停时的现状（2026-10-27，按用户要求暂停，由 GPT 另出方案）**
>
> - **正确性**：表 7 配置下梯度审计 `source_horizon_dropped_grad` 恰为 0（d345da5，加强口径复核，§18.16–18.17）。之后所有运行（表 8、§20 的短跑、§20.6 的解耦实验）也都是 0。H=0 与 v2 逐位等价一直保持（最后一次 v3r14）。生产版本没有替换，所有新开关默认关。
> - **各表数字**（x_real_time，加权稳态；旧架构 843.4×）：
>
> | 表 | 配置 | x_real_time | 学习器 update_s | 说明 |
> |---|---|---|---|---|
> | 表 6 | 5 卡、H=1024、四块 checkpoint、收尾即派发 | 334.0× | — | 审计非零，不作正确性结论 |
> | 表 7 | 表 6 + 保留范围修复 | 320.8×（208911e）/ 311.8×（d345da5 复核） | 11.58 s | 审计 0（d345da5） |
> | 表 8 t8 | 5 卡、均衡开、块 2、3 checkpoint | 326.8× | 10.17 s | 源卡峰值 20–23 GiB、容器 71.2 GiB，贴着上限，不作默认 |
> | 表 8 t8b | 5 卡、均衡开、四块 checkpoint | 306.0× | 12.02 s | 与表 7 可比 |
> | §20.6（短跑，4 卡，各 2 次） | 默认 / 每卡两个源进程 + merge 1 + fused / 再加批量 RPC | 304–314 / **341–382** / 334–337 | 10.9–11.6 / 9.3–10.8 / 10.2–10.6 | 单次噪声约 ±10% |
>
> - **最快组合**：每张源卡两个源进程 + `--source-vjp-merge 1` + `--source-fused-cast 1`，四块 checkpoint，均衡开，并发关，不加批量 RPC。只在 4 卡短跑上测过（约 361×，默认约 309×）。
> - **未完成项**：
>   1. **表 9 未跑**：上面这个组合的 5 卡 40 次稳态定格。等 GPU2、GPU3（被别人占用）期间收到暂停指示，等待脚本已停，未启动。
>   2. `--source-streams 2` 在生产设置下偶发与串行不一致（8 次中 2 次，1.4e-6 / 1.4e-7），来源未定位；开关保持关。
>   3. 批量 RPC 让记录批变慢（2.1 s 对 1.0–1.2 s），原因未分辨。
>   4. 块 2、3 checkpoint 的显存余量不足。
> - **瓶颈与路线**：学习器约 10 s，采集约 7.5 s；学习器等的是源卡；源卡的 VJP 与编码约 90% 是 host / launch 开销（§19.10、§20.2）。两条根治路线的估算见 **§21**：
>   - a）CUDA Graph + 函数式重算 VJP：VJP 7–9 → 1–1.5 s，约 15–25 天；
>   - b）多局合并编码块 + merge 1：VJP → 2–3 s，约 5–8 天，有记录钉住整块的显存风险。
> - 备份与提交号见 §22。

> **结论：horizon 按设计工作，但在这个测量窗口里不省资源；真正的瓶颈是主机内存。**
>
> - **horizon 本身**：前向逐位不变；屏蔽后的梯度与逐决策参考一致；被丢弃的梯度在所有运行中恰为 0；每局保留的图有上界。
> - **为什么不省资源**：从 checkpoint 恢复后每局都从头开始，前 24 次更新里单局最长约 4.9k 帧，被释放的图只占全部保留图的约 5%，源卡显存两组相近。
> - **主机内存才是瓶颈**：容器内存在 H=2048 和 H=0 下的曲线几乎重合，每次更新涨约 1 GiB，第 23–24 次更新撞上 72 GiB。
> - **这个增长不是旧问题**：旧架构（843×）的容器内存在第 2 次更新就到约 55 GiB，之后一直平在 55–57 GiB。
> - **增长跟着什么走**：匿名内存与 fork 出来的游戏进程数（相关系数 r=0.994，约 36 MB/个）和源侧在世的局数（r=0.989，主要是持续累积的教师/v2 记录对应的局）几乎同步上升；与在世局的帧数总和只是弱相关（r≈0.72）。
> - **短消融的结果**：把教师、CF 全关以后，前 10 次更新的斜率仍有 0.78 GiB/次更新（现配置 0.84，同配置另一次 1.17），所以这个窗口里的增长不是教师记录造成的。它跟着 floor 存档（parked 游戏进程）的填充走；v2/CF 记录要到第 10 次更新以后才出现，这次没测到（§10）。
> - **新旧架构逐项实测（§11）**：游戏 parked 进程两边都涨，只是新架构填得慢，所以在平台到来之前先撞上了上限。只有新架构才有的是 python 进程的增长：前 9 次更新约 +0.6 GiB/次更新，与记录无关，之后变平；第 9 次更新以后，教师/CF/v2 记录每次更新约再多 0.55 GiB python 内存和 0.2 GiB 共享内存，缓冲区填满前不会停。实测 Pss 能解释 cgroup 增长的 97–103%（后段）。§10 里"旧架构每个 parked 进程 3–5 MB"的推断是错的，已在 §11 更正。
> - **按批准规则修改保留策略后的验证（§12）**：规则已经生效（学习器上省掉 9.6 万帧的前缀拷贝；1415 条教师记录消费后释放图；丢弃梯度为 0），但第 9 次以后 python + shmem 的斜率没有降下来（+0.74 + 0.31，对比 A 组 +0.11 + 0.15）。在这个窗口里 C 和 A 的差别主要来自 CF 分支运行本身（actor 和源进程），而不是环形缓冲区里的记录。
> - **碎片还是保留（§13）**：换成 jemalloc 后 RSS 斜率没有下降，而每个进程的 RssAnon 和活着的 RawHistory 原始字节同步增长，所以是活对象保留，不是碎片（前 9 次也一样）。加上 donor 修正后，新保留策略把第 9–20 次的 python 斜率从 +0.71 降到 +0.31（A 组 +0.15），shmem 从 +0.32 降到 +0.17（A 组 +0.11）；但单次运行之间的差异约 ±0.3，需要重复运行确认。CF lane 的上下文在分支发布后会回落。
> - **规则 5（§14）**：已编码、已消费帧的原始字节现在会被释放，各进程的活原始字节不再随局长增长（学习器 RssAnon 平在 4.2 GiB），24 次更新正常跑完。但第 20–24 次的水位是 62–65 GiB，没达到 ≤58 的目标。剩下的是游戏存档的填充（与旧架构相同，会封顶）和新架构常驻 python 进程的固定开销（比旧架构多约 16 GiB）。
> - **吞吐（§15，初步：5 卡、H=2048、25 次稳态，第 35 次源卡 OOM）**：337×，旧架构 843×，v2 370×。每次更新墙钟 12.2 s 对 5.2 s；采集慢了 5.7 s，几乎全是 actor 推理（单次 17.1 ms 对 5.9 ms）；学习器多了 4.9 s（等源 3.4 s、记录批 1.2 s），在做 VJP 的更新里会超过采集。4 卡组和 H=0 组还在重跑。
> - **梯度审计已收口（§18.16–18.17，258f97a）**：表 7 配置重跑 40 次稳态、无 OOM、`source_horizon_dropped_grad` 恰好 0，窗口内有 7 次 v2 loss。修了三处：(1) 等待中的记录自己的 query 帧不在保留区间里（fc0ad4f / 1727f71）；(2) 在途分支的祖先只保留数值，改为保留 [ℓ − H, 共享帧数) 的图（c8f356f / 187ede3）；(3) 双 actor 进程时 `merge_jobs` 只保留第一个 actor 的在途报告（753a47a / 258f97a）。另补上审计盲区（新写的 Writer 块读已释放块时梯度静默断开，现计入审计；8ce8a58 / d345da5），在新口径下复跑（验收 4）仍为 0，窗口内 6 次 v2 loss，311.8×。速度 312.5×（旧架构 843.4×），源卡峰值显存比表 7 高约 0.5–1.9 GiB。
> - 更早的长测（§18.14–18.15，5 卡、H=1024、源每 rollout、收尾即派发、按层 checkpoint）：表 6（旧保留规则）与表 7（保留范围修复）都跑满 40 次稳态、无 OOM：334.0× 与 320.8×，但审计不为 0（表 6 第 39 次首次非零、累计 0.04076；表 7 第 4 次首次非零、累计 31.55；表 4 第 30 次、0.00963），所以它们不算正确性收口。表 7 的带图帧稳定在 12–13.5 万，表 6 一路涨到 15.8 万。
> - **决策摘要：Spatial 按层 checkpoint（§18.4，只做了隔离评估，默认关）**：每帧编码后留下的显存 340 → 57 KiB，源 VJP（最慢一张卡）6.06 → 6.16 s，只跑源的卡峰值 21 → 7 GiB，源梯度相对误差 9.3e-4（BF16 重算的浮点量级）。它在反传时重算 4 个 Spatial 块的前向，触及"不重算"的规则，是否在长跑中打开由用户在表 4 之后决定。
> - **VJP 派发晚（§18.5）**：学习器等 VJP 提交的约 3 s 来自 VJP 在源就绪后 2.5 s 才派发；`--source-dispatch-at-close 1`（纯调度，默认关，单测逐位一致）让 VJP 在区间关闭时就开始、提交仍按原规则在 minibatch 边界，表 5 用它对照。
> - **训练语义事实：这些测速窗口里 v2 教师几乎没起作用（§18.10）**：v2 记录要"有效 ≥ 256 条且当次 v2 步数 ≥ 8"才会被采样，而 `v2_step = min(teacher_v2_batch, teacher_v2_reuse × v2_rate / planned)` 在从 checkpoint 恢复后的这几十次更新里只有 2–8。5 卡各组（表 3、表 4、表 5、§15 的 H=2048 与 H=0）中只有表 4 有 1 次更新满足 ≥ 8、一共消费了 110 条，其余各组 v2 记录消费为 0；旧架构 843× 那组（50 次更新）`v2_step` 最大也只有 5，同样从未采样。所以这些吞吐数字都是在 v2 教师不训练的情况下测的，这与保留规则无关。
> - **原始观测字节不是主因**：horizon 之外的帧，原始观测字节只有每份 0.1 GB 量级。
> - 计量数据见 §9。

## 0. 状态摘要

| 项 | 状态 | 证据 |
|---|---|---|
| 实现 `--memory-horizon H`（默认 2048，0 = v2 无界） | 完成 | 提交 9388647、0c8be2e、bfd3656 |
| 验收 1：新单元测试（horizon 内帧梯度非零、horizon 外帧无图无梯度、前向与 H=0 逐位相同） | **通过**（CPU） | `tests/test_source_horizon.py`，退出码 0，§3.1 |
| 验收 2：保留图统计量进 progress.csv，有上界且不随局长增长 | **单测通过**（CPU，含源进程模式）；长测**未跑** | `tests/test_source_horizon_bound.py`，§3.2 |
| 验收 3：原有测试全部重跑并列表；H=0 复现 v2 | CPU 全量已跑，见 §4；H=0 与 v2 **逐位相同**（本地线程和源进程两种模式） | §4、§3.3 |
| CUDA 门测（主机 GPU 3–5） | horizon 三项和 actor、v2 CUDA 套件都通过；两项失败已查明是 v2 既有的 | §8.1 |
| 验收 4：8 卡主机完整管线长测（H=2048 + 对照 H） | **没有跑满**：GPU1 被别人占用，改做 4 卡组内 A/B。H=2048 跑了 18 次更新，被外部内存压力规则停掉；H=0 跑了 24 次更新，撞上自己容器 72 GiB 的 OOM | §8.2–8.4 |
| 验收 5：不达标时给出分项耗时和瓶颈 | 瓶颈是**主机内存**，不是 GPU 上的图：两组的容器内存曲线几乎重合（第 16 次更新都约 65 GiB）。在这个测量窗口里，horizon 对主机内存没有作用 | §8.4–8.5 |

**没有稳态吞吐结论**：两组都没跑满 40 次测量更新，而且 4 卡配置本身就不能和 843.4×、370× 直接比较。

**最重要的发现**：H=2048 和 H=0 的容器内存一模一样地往上涨，H=0 在第 24 次更新时撞到 72 GiB 并 OOM。也就是说，在这个窗口内 v2 的 OOM 来自主机内存，而不是"梯度无界、保留全部计算图"。此外，从 checkpoint 恢复后每局都从头开始，在 24 次更新内单局最长约 4.9k 帧（H=0），所以 horizon 能释放的图只占全部保留图的约 5%（释放 4–6.6k 帧，总共 10–16 万帧）。这和最初的诊断不一致，下一步应该先处理主机内存。

## 1. 语义

- **读取不变**：Reader 仍能读到本局全部已完成的记忆块，以及最近 W=256 帧。读到的值来自编码它们时的参数版本（被释放的帧和块保留它们的数值）。
- **梯度有界**：决策 t 的 loss 只传回年龄 ≤ H 的源。
  - 直接读的帧 s：年龄 = t − s。
  - Writer 块 b：年龄按块的首帧算，即 t − 64b。也就是说，只有一个块的 64 帧全部在 horizon 内时，它才收到梯度。
  - query 自身的帧（u 和 y）年龄为 0，梯度总是保留。
- **按决策–条目对屏蔽，而不是按条目**。同一局的多个决策共享 Reader 里的 bank，同一个记忆块可能对一个决策在 horizon 内、对另一个决策在 horizon 外，所以必须逐对处理。
- **只屏蔽源侧梯度**：Reader 自己的 K/V 投影权重、query 路径、头部仍然收到所有可见对的梯度。老源被当作 detached 常数，但 Reader 照样从它们学习。
- **释放时机**：每局的 frontier f(uid) 是将来任何 loss 还可能用到的最早决策。取值方式：本 rollout 的首个决策；teacher、v2/CF 记录的上下文长度（这些记录会在以后的更新里被重复采样）。
  - 末帧 < f − H 的编码 chunk、末块首帧 < f − H 的 Writer 作业，再也不会收到 cotangent。
  - 这时先把它们已累积的 credit 做一次 VJP 结算。用 `retain_graph=True`，因为同一局后面的图可能共享组件节点。结算结果进入 `direct_grads`，与同一个慢周期的 Adam 一起提交，和 v2 结算已关闭 episode 的路径相同。
  - 结算完成后丢掉图引用，只保留数值。
- **钉住规则**：如果一个 chunk 还被仍在 horizon 内的 Writer 作业引用，它的图实际上没有被释放，所以先不标记为 detached，等那个作业释放时再一起放。这样统计量反映的是真实保留，而不是名义上的标记。
- **保持不变的部分**：micro 8192、无前向重放（不重放源编码器）、不截断记忆读取、PPO/CF/教师路径、actor、评测语义。
- **不使用 pipeline 的路径**（不设 `--source-devices`）：horizon 不生效。这是 v2 留下的参考路径，生产配置不走它。

## 2. 改动（`git diff 64ae451 bfd3656`，7 个文件，+654/−22）

| 文件 | 改动 |
|---|---|
| `isaac_bridge/rebuild_memory.py` | 新增 `HorizonAttention`（autograd.Function）。前向就是原 grouped Reader 的同一串操作：`F.embedding` → reshape → SDPA，所以输出逐位相同。反向手工重算注意力权重 P，不保存分数矩阵，按 group 分片，每片不超过 2^26 个元素。dq 和 K/V 权重梯度按所有可见对计算；源表梯度只按 `grad_valid`（年龄 ≤ H）计算，经 `dsrc = dkv_masked @ W` 回到源表。`grouped_layout` 在 H>0 时生成 `grad_valid`。H>0 时 `SourceBatch`/`SourceView` 的 K/V 由 `layer.kv(src.detach())` 计算，数值与原来相同。H=0 走原生路径，代码与 v2 完全一样。 |
| `isaac_bridge/source_pipeline.py` | `horizon` 参数。chunk 记录 `rows`/`last`，write 记录 `last_start`/`chunks`。`retain_histories(..., frontiers=)` 调用 `_detach_horizon`：先选候选，处理钉住，再结算 VJP（`horizon_backward`），最后 FIFO 执行 detach。credit 遇到 detached 的结果会跳过，并把被跳过的梯度累加进审计量 `horizon_dropped`；如果 query 来自 detached 帧，直接 assert 报错。版本快照只为未 detached 的图保留。新增 `horizon_stats()`。 |
| `isaac_bridge/source_worker.py` | 新命令 `detach`：u 换成已导出的 IPC 值，丢掉 y 和 mem 的图边。内存回报里新增 `detached_chunks`、`detached_writes`、`graph_frames`（worker 自己实际持有的图帧数）。 |
| `train_tok.py` | 新参数 `--memory-horizon`（默认 2048）。三处 `retain_histories` 调用都传入 `horizon_frontiers()`（本 rollout 请求、`v2_contexts`、`teach_contexts`）。progress.csv 每次更新追加 `source_horizon`、`source_graph_frames`、`source_graph_frames_uid_max`、`source_graph_writes`、`source_detached_frames`、`source_detached_total`、`source_horizon_dropped_grad`、`source_alloc_gib_max`，进程模式下还有 `source_worker_graph_frames`。 |
| `tests/test_source_horizon.py` | 新增：验收 1，以及逐对参考、结算等价、源进程模式。 |
| `tests/test_source_horizon_bound.py` | 新增：验收 2 的单测。 |
| `tests/bench_horizon_attention.py` | 新增：在 GPU 上对比原生 SDPA autograd 与 `HorizonAttention` 的耗时和峰值显存（还没在 GPU 上跑过）。 |

另外，首个提交带了一个 `.gitignore`（`__pycache__/`、`*.pyc`），这是唯一与"v2 原样"不同的文件。

审计口径：`source_horizon_dropped_grad` 是被丢弃的 cotangent 的 L1 和，累计值，在学习器上用 float64 累加。按设计它必须恰好为 0；如果不为 0，说明有 horizon 内的梯度被误丢了。

## 3. 正确性验证（CPU，本机 torch 2.12.0+cpu）

### 3.1 `test_source_horizon.py`（验收 1）

命令：`PYTHONPATH=. python -X utf8 tests/test_source_horizon.py`，在 `candidate/bridge/python` 目录下运行，**退出码 0**。日志：`results/tests-v3-bfd3656/test_source_horizon.log`。

设定：2 局 × 48 个决策，block 4，W 4，micro 8，H=12，两个源 worker，update_rollouts=2。rollout 1 是决策 24–31，rollout 2 是决策 40–47。

| 检查 | 结果 |
|---|---|
| Reader 前向，H=12 与 H=0 比较（本地模式） | `torch.equal` 对所有输出成立 |
| 测试确实触发了屏蔽 | 可见但被屏蔽的对：136 个；被移除的记忆块梯度 L1 = 2.204 |
| 批量屏蔽梯度 vs 逐决策参考（原生 H=0 autograd，每次只算一个决策，把年龄 > H 的 leaf 行置零）：u、mem、y 三个 leaf | `assert_close`（rtol 1e-4，atol 1e-7）通过 |
| Reader 权重梯度：H=12 与 H=0 | 最大差 1.9e-6（手工反向与原生 kernel 的数值差） |
| 释放：detached 的 4 个 chunk、2 个 write | 原图张量的 weakref 已失效；值没有 `grad_fn`；y 为 None。另有 2 个 chunk 被在世 Writer 钉住 |
| 不变式 | 每个仍保留的图，要么还能被未来决策触达（末帧 ≥ f−H），要么被在世 Writer 引用 |
| 被 detached 的值的读取，与从不 detach 的参考流水线比较 | `torch.equal` |
| 结算后的参数：detach 流水线 vs 从不 detach 的流水线 | 最大差 6.0e-8；丢弃梯度恰为 0.0 |
| 源进程模式（2 个 CPU 源进程）vs 本地模式 | 参数最大差 2.5e-6；worker 回报的 detached 数（4 个 chunk、2 个 write）与学习器一致 |
| 单一决策 t=47 的 loss（H=12） | 梯度非零的帧：36–47；没有图的帧：0–31；有图但梯度恰为 0 的帧：32–35（所在 chunk 跨过 horizon 边界）。H=0 对照的最早非零帧是 0 |

关于"horizon 外的帧没有图"：释放的粒度是 chunk（一个 chunk 是一次预取增量里同一局、同一 worker 的连续帧，最多 `room_source_micro` 行）。所以跨在边界上的 chunk，其中超出 horizon 的帧仍有图，但梯度恰为 0。测试对这两类帧分别做了断言。

### 3.2 `test_source_horizon_bound.py`（验收 2 的单测）

命令：`tests/test_source_horizon_bound.py` 和 `tests/test_source_horizon_bound.py --process`，**退出码都是 0**。日志：`results/tests-v3-bfd3656/test_source_horizon_bound.log`、`results/test_source_horizon_bound-process.log`。

设定：2 局 × 320 个决策，每个 rollout 每局 32 个决策，共 10 个 rollout，跑完整的流水线循环（retain/detach → 常驻 bank → Reader → credit → 慢 VJP/Adam）。

| rollout | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| H=24：`source_graph_frames_uid_max` | 32 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| H=0：`source_graph_frames_uid_max` | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 288 | 320 |
| H=24：累计 detached 帧数 | 0 | 0 | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |

- H=24 稳定在 64 = H + 一个 rollout（32）+ 一个 chunk（8）。H=0 随局长线性增长，最后等于 T=320。
- 丢弃梯度 = 0.0。
- `--process` 模式下，每个 rollout 都断言 worker 自己回报的 `graph_frames` 之和等于学习器的统计量。这说明进程侧的图确实被释放了，而不只是学习器在记账。

上界的定义：每局保留图的帧数 ≤ H + 当前 rollout 及已预取未消费的帧 + 一个 chunk + 一个被钉住的 Writer 作业跨度。这比任务描述里设想的"≤ H + 一个块"宽，原因有两个：一是释放粒度是 chunk 和 Writer 作业；二是当前 rollout 和预取帧的图本来就必须存在，因为它们的决策还没被消费。这个界是常数，不随局长增长。生产上（rollout 每局约 512 帧，预取每局约 128 帧/次）的实际值，要在长测的 progress.csv 里读出来。

### 3.3 H=0 与 v2 的逐位等价

脚本：`scripts/h0_equivalence.py`。日志：`results/h0-equivalence/h0-equivalence.log`；数据：`results/h0-equivalence/*.pt`。

做法：在 v2 原样代码（`git worktree` 检出 64ae451 到 `temp/memory-chain-v3/baseline-64ae451`）和 v3 代码（bfd3656，`horizon=0`，同时传入 frontiers，让新代码路径也被走到）上，各自用独立进程跑同一个确定性场景：2 局，4 个 rollout，每次 2 个 Reader 步，update_rollouts=2，源版本走到 2。然后比较全部 Reader 输出和最终参数。

| 模式 | 运行退出码（v2 / v3） | 比较结果 | 比较退出码 |
|---|---|---|---|
| 本地线程源 worker | 0 / 0 | `BITWISE_EQUAL outputs=8 params=172 source_version=2 encoded_rows=80` | 0 |
| 源进程（spawn） | 0 / 0 | `BITWISE_EQUAL outputs=8 params=172 source_version=2 encoded_rows=80` | 0 |

主机上的 H=0 短跑（v2 行为在真实管线上的复现）也包含在 §6 的脚本里，还没跑。

## 4. 原有测试重跑（CPU）

命令：`python -X utf8 run_tests.py <checkout> <out> 900`。每个测试单独一个进程、不带参数运行；每个测试的完整输出在 `<out>/<test>.log`，汇总在 `<out>/summary.json`。

- v2 基线：`results/tests-baseline-64ae451/`
- v3：`results/tests-v3-bfd3656/`（另有一份更早的完整运行 `results/tests-v3-0c8be2e/`：当时还没有 bound 测试，共 54 个测试，45 个通过，非零的 9 项与本次相同；`tests-v3-bfd3656-interrupted-by-app-quit/` 是桌面应用退出时被中断的那次，退出码 0xC000013A，不计入）

共 55 个测试（v2 基线 53 个，新增 2 个）：v3 有 46 个退出码为 0，9 个非零。与基线相比退出码不同的测试：无。

| 测试 | v2 基线 64ae451 | v3 bfd3656 | 非零退出原因（v3 日志最后一行） |
|---|---|---|---|
| test_actor_d2h | 1 | 1 | AssertionError: Torch not compiled with CUDA enabled |
| test_actor_graph_pool | 1 | 1 | ModuleNotFoundError: No module named 'gymnasium' |
| test_actor_processes_smoke | 0 | 0 |  |
| test_actor_reader_version | 0 | 0 |  |
| test_actor_shutdown_smoke | 0 | 0 |  |
| test_actor_snapshot_storage | 0 | 0 |  |
| test_actor_streams | 1 | 1 | AssertionError: Torch not compiled with CUDA enabled |
| test_cf_smoke | 0 | 0 |  |
| test_cf_wall_smoke | 0 | 0 |  |
| test_crosscycle_cache | 0 | 0 |  |
| test_eval_ablate_smoke | 0 | 0 |  |
| test_fused_tokens_smoke | 0 | 0 |  |
| test_incremental_ipc | 0 | 0 |  |
| test_inventory_all_smoke | 0 | 0 |  |
| test_item_global_smoke | 0 | 0 |  |
| test_lane_batch_smoke | 0 | 0 |  |
| test_learner_device_smoke | 0 | 0 |  |
| test_learner_fast2_smoke | 0 | 0 |  |
| test_memory_smoke | 0 | 0 |  |
| test_no_active_smoke | 0 | 0 |  |
| test_no_gru_smoke | 0 | 0 |  |
| test_obs_v2_smoke | 0 | 0 |  |
| test_patch_cache | 0 | 0 |  |
| test_pickup_smoke | 0 | 0 |  |
| test_reader_indices | 0 | 0 |  |
| test_rebuild_actor_cuda | 1 | 1 | AssertionError: Torch not compiled with CUDA enabled |
| test_rebuild_memory_cuda | 1 | 1 | AssertionError: this gate requires a real CUDA device |
| test_rebuild_memory_smoke | 0 | 0 |  |
| test_rebuild_state_storage | 0 | 0 |  |
| test_recent_frame_union | 0 | 0 |  |
| test_room_memory_smoke | 0 | 0 |  |
| test_room_pipeline_smoke | 0 | 0 |  |
| test_saved_activation | 1 | 1 | AssertionError: Torch not compiled with CUDA enabled |
| test_saved_activation_compile | 1 | 1 | AssertionError: Torch not compiled with CUDA enabled |
| test_source_async | 0 | 0 |  |
| test_source_compile_versions | 0 | 0 |  |
| test_source_component_parity | 2 | 2 | test_source_component_parity.py: error: the following arguments are required: --job, --checkpoint |
| test_source_export_edge | 0 | 0 |  |
| test_source_gradient_buffers | 0 | 0 |  |
| test_source_horizon | （新增） | 0 |  |
| test_source_horizon_bound | （新增） | 0 |  |
| test_source_lifetime | 0 | 0 |  |
| test_source_pipeline | 0 | 0 |  |
| test_source_pointwise | 0 | 0 |  |
| test_source_precision | 0 | 0 |  |
| test_source_precision_checkpoint | 2 | 2 | test_source_precision_checkpoint.py: error: the following arguments are required: --job, --checkpoin |
| test_source_preempt | 0 | 0 |  |
| test_source_process | 0 | 0 |  |
| test_source_rollout_interval | 0 | 0 |  |
| test_source_shared_cast | 0 | 0 |  |
| test_source_streaming | 0 | 0 |  |
| test_stall_progress_smoke | 0 | 0 |  |
| test_stateless_teacher_batch | 0 | 0 |  |
| test_teacher_context_lifetime | 0 | 0 |  |
| test_temporal_smoke | 0 | 0 |  |

CPU 上的非零退出全部是环境原因，并且与 v2 基线逐项相同：没有 CUDA；缺 `gymnasium`；`test_source_component_parity` 和 `test_source_precision_checkpoint` 需要 `--job/--checkpoint` 参数（这两项用的是主机上的学习器转储）。这些需要 CUDA 的门测已经写进 §6 的 `gates` 步骤，还没跑。

## 5. 新主机长测：第一次尝试时的网络阻塞（已解决，结果见 §8）

- 现象：本机向 10.19.136.184 发起的 TCP 连接能建立，但对方立即关闭、不发 SSH banner（paramiko 报 `Error reading SSH protocol banner`）。80、443、2222 端口的表现完全一样。
- 据发起会话转述：本机有代理/隐私守卫进程在拦截出站连接，只有用户能处理。按要求停止了重试，没有在任何主机上启动过任何东西，所以也没有需要清理的容器或进程。
- 探测记录：`results/ssh-probe.log`。
- 因此：x_real_time、collect/update 中位数、峰值显存和主机内存、是否跑满，目前都没有数据。也没有基于短测的推断。

## 6. 网络恢复后直接执行

```bash
C:/Users/eolc/miniconda3/python.exe -X utf8 D:/Projects/fortune/Isaac/temp/memory-chain-v3/scripts/run_remote_all.py
```

默认按顺序执行以下步骤，任何一步失败就以非零退出码停下。也可以只跑其中几步，例如 `... run_remote_all.py status deploy gates`。

1. `status`：主机快照。物理 GPU 1–5 上有任何计算进程就中止。GPU 0、6、7 属于别人，永远不用。
2. `deploy`：用 `git archive` 打包 candidate 的 HEAD（拒绝有未提交改动），送到 `cpu-wide/memory-v3-code-v3-<短哈希>/python`，与 v2 的代码目录布局相同。
3. `gates`：在 GPU 3、4、5 上运行
   - `test_source_horizon.py --learner cuda:0 --sources cuda:1,cuda:2`（CUDA 上的逐位前向和逐对参考）；
   - `test_source_horizon_bound.py`（加 `--process`）；
   - `bench_horizon_attention.py`（96 组 × 128 个 query × 1600 个条目的耗时和峰值显存）；
   - v2 的 CUDA 套件（`test_rebuild_memory_cuda`、`test_rebuild_actor_cuda --compile`、CUDA 版 `test_source_pipeline`、`test_source_process --bf16 --defer --shared`、`test_source_precision`）。
4. `h0`：H=0 短跑，15 游戏小时，约 13 次更新，在主机上复现 v2 行为（预期 `source_graph_frames` 持续增长）。
5. `h2048`、`h1024`：长测，各 60 游戏小时，约 53 次更新，丢弃前 10 次。`h4096` 是可选的第三个对照，要显式写进步骤列表才会跑。
6. `final`：确认没有残留的 `memory-v3` 容器。

每次长测的配置：v2 最好那次（long-59，370×）的 docker 命令原样（`host/run_v3_case.py` 只做以下替换，替换前离线核对过 115 个参数）：
- 用例名、运行目录、代码挂载路径；
- 选中的空闲 GPU UUID（按 v2 的顺序取物理卡 3,2,1,4,5）；
- `--memory/--memory-swap 72g`；
- 容器内 timeout 从 1500 s 改为 2400 s（只是为了防止在 60 游戏小时之前被截断）；
- 追加 `--memory-horizon H`。

其余全部沿用：64 workers、rollout 32768、minibatch=micro 8192、epochs 3、源侧 BF16 + compile + flash、MPS、源进程、overlap rollout、每 2 个 rollout 更新一次、prefetch 8192、actor 预留 7 GB、latespill 18 GB/0.75 GB、`--cpus 40`、nice 10、cpuset 和 v2 一样，以及 v2 的 guard（只停自己的容器；主机内存、外部 CPU、无进展规则不变；因为源编码要编译，首次更新超时用 900 s）。

产出在 `results/<case>/`：`train.log`、`progress.csv`、`telemetry.jsonl`、`guard.jsonl`、`result.json`，以及汇总 `summary.json`（steady x_real_time 按 wall_s 加权、collect/update 中位数、所选卡的峰值显存、容器内存峰值、guard 停止原因、是否跑满 40 次稳态更新）。所有运行的汇总写在 `results/v3-<短哈希>-runs-summary.json`。

对照基线：
- 旧架构 843.4×：`temp/memory-chain-v2/baseline-result.json`，3 GPU，collect 中位数 5.26 s，update 4.56 s，wall 5.22 s；
- v2 最好 370.2×：`temp/memory-chain-v2/actor2-…-latespill-long-59-result.json`，mem88，28 次更新后退出码 137，collect 10.75 s，update 7.57 s，wall 11.01 s。

## 7. 剩余问题和风险（都还没有用长测验证）

1. **手工 attention 反向的开销**：H>0 时所有 Reader 反向都走 `HorizonAttention`，这是为了保持一致，即使某个批次里没有被屏蔽的对。它是未融合的 fp32 实现。CPU 小测里，前向+反向约为原生的 1.36 倍（16 组 × 64 × 400：24.7 ms 对 33.6 ms）。GPU 上的耗时和显存要看 `gates/attention-timing`。如果它成了 update 的瓶颈，可以只在批次里存在被屏蔽对时才走它，或者对它做 compile/融合。
2. **y 值仍然保留**：进程模式下，detached chunk 的 y（空间 token）数值仍然留在共享的 IPC 导出缓冲区里，因为它和 u 在同一块扁平内存里。按 v2 的保存统计估算，约 4.6 KB/帧，而图约 470 KB/帧，大约是 1%，并且随局长线性增长。本地模式下已经丢掉了 y。如果长测里值内存变得显著，需要在 detach 时重新导出一份紧凑的 u。
3. **组件图缓存的残留**：开启 `ISAAC_SOURCE_COMPONENT_GRAPH_CACHE` 后，后面帧的图会引用前面 chunk 里创建的组件（grid/map/patch CNN）节点，所以这部分子图的释放会晚于 chunk 本身。它受每局每类组件 2048 条 LRU 的限制，是有界的，但没有计入 `source_graph_frames`。真实显存以 worker 回报的 `allocated`（`source_alloc_gib_max`）为准。
4. **上界比"H + 一个块"宽**：见 §3.2。生产上大约是 H + 当前 rollout + 预取 + 一个 chunk 和 Writer 作业的跨度（几百帧量级）。
5. **frontier 只覆盖已知的消费者**（rollout 请求、`v2_contexts`、`teach_contexts`）。不在 frontiers 里的 uid（例如下一 rollout 才出现的新局）不会被 detach，这是保守的做法。如果以后新增别的复用旧决策的消费者，必须把它加进 `horizon_frontiers()`；否则它会在审计量 `source_horizon_dropped_grad` 里表现为非零。
6. 本机 Windows 上源进程模式的周期性 `malloc_trim` 依赖 glibc，测试里设了 `ISAAC_HOST_TRIM_INTERVAL=0`。这只影响本机测试，主机是 Linux。

## 8. 主机上的结果（2026-10-24，代码 bfd3656）

### 8.1 CUDA 门测（GPU 3–5）

日志：`results/gate-v3-bfd3656-*.log`，退出码在对应的 `*.exit` 文件里。

| 门测 | 退出码 | 结果 |
|---|---|---|
| horizon-cuda：`test_source_horizon.py --learner cuda:0 --sources cuda:1,cuda:2` | 0 | CUDA 上前向逐位相同，逐对参考通过 |
| horizon-bound：`test_source_horizon_bound.py`（含 `--process`） | 0 | 有界，丢弃梯度 0 |
| attention-timing：`bench_horizon_attention.py`（96 组 × 128 个 query × 1600 个条目） | 0 | 前向相等；horizon 反向 27.0 ms，原生 30.3 ms；峰值显存 1.94 GiB 对 1.57 GiB |
| actor-cuda：test_actor_d2h、test_actor_streams、test_actor_graph_pool | 0 | 通过 |
| v2-cuda-suite2：test_rebuild_actor_cuda --compile、test_source_pipeline（CUDA）、test_source_process --bf16 --defer --shared、test_source_precision | 0 | 通过 |

既有失败（v2 本来就失败，不是 v3 引入的）：
- `test_actor_graph_pool.py --separate`：这是 v2 自己设计成必然失败的反例对照。失败的字节数 266338304 与 v2 的 actorpool-gate-separate-77 一模一样。我第一次误把它放进了门测，已经移除。
- `test_rebuild_memory_cuda.py`：在 v2 原样代码 64ae451 上，用同一条命令也报同样的错（`rebuild_actor._core` 里 BF16 index_put 的 dtype 不匹配），日志 `results/gate-v2-64ae451-rebuild-memory-cuda.log`，退出码 1。v3 没有改 `rebuild_actor.py`。

CPU 上那三个 actor 测试失败的原因：`test_actor_d2h`、`test_actor_streams` 需要 CUDA；`test_actor_graph_pool` 在本机报的是缺 `gymnasium`（主机镜像里有这个包）。三项在主机上都通过。

### 8.2 设置

- GPU1 一直被别人占用（约 10 GB），所以没法用 v2 的 5 卡布局。改做 4 卡组内 A/B：物理卡 3、2、4、5 对应容器内 cuda:0–3；学习器 cuda:1；actor 在 cuda:0、cuda:2；源在 cuda:0、cuda:2、cuda:3（共享）。
- 其余配置与 v2 long-59 相同，容器内存上限 72 GiB。
- **这组结果只能在组内比较（H=2048 对 H=0），不能和 843.4×、370× 比。**
- guard 的规则按发起会话的决定改过（`host/guard_v3.py`）：
  - 外部 cgroup 的 pressure full avg10 连续 3 个样本 > 10% 才停（v2 是 1%）；
  - 主机 pswpin 连续 3 个样本 > 1000 页/秒；
  - 外部 swap 从启动起累计增长 > 1 GiB；
  - 外部 OOM 计数增加立即停；
  - 主机可用内存 < 26 GiB、自己 cgroup OOM 两条与 v2 相同。
- 启动前检查 sl_dev 的块 IO：读写合计约 7.5 MB/s（`io.stat` 把两个块设备加在一起，可能重复计数），低于 10 MB/s 的阈值，判为不活跃。

前几次尝试（都只停了自己的容器）：
- 第一次：旧的 1% 规则触发，因为 sl_dev 的页缓存被回收。
- 第二次：当时的"任何外部 swap 增长就立即停"规则触发，H=2048 停在第 7 次更新，H=0 停在第 9 次。
- 这些运行的日志保留在 `results/v3-bfd3656-ab4h*-1024090*`、`*-10240919`、`*-10240924`。

### 8.3 这一轮（r3）怎么结束的

| 组 | 用例 | 更新次数 | 结束原因 |
|---|---|---|---|
| H=2048 | v3-bfd3656-ab4h2048-10240933 | 18 | guard 的 host_memory_pressure：外部 pressure 连续 3 个样本 > 10%（18.8%、23.7%、22.3%）。当时我们容器 65–67 GiB，主机可用约 37 GiB |
| H=0 | v3-bfd3656-ab4h0-10240940 | 24 | 我们自己 cgroup 的 OOM（72 GiB），外部 pressure 当时 31% |

结束后用 `run_remote_all.py final` 确认主机上没有 memory-v3 容器。

### 8.4 每次更新的记录

`graph_frames_*`、`detached_frames`、`dropped_grad`、`src_alloc_gib_max` 来自 progress.csv。容器内存的拆分来自 guard 每次更新写一行的 `memory.jsonl`（cgroup memory.stat）。最后几次更新没有内存行，是因为采样每 2 秒才检查一次，进程退出前来不及写。

**按进程角色拆的 RSS 在这一轮是无效的**：角色识别把 `timeout` 包装进程当成了训练主进程。脚本已修复，并在 H=0 的运行容器上验证过一次：训练主进程 6.5 GiB、学习器 7.4 GiB、3 个源进程合计 5.5 GiB、actor 5.9 GiB、64 个游戏桥接进程合计 4.2 GiB、编译进程 6.4 GiB；游戏本体的 RSS 合计 86.8 GiB，但它们是 fork 出来、共享页面的，RSS 重复计数严重。正确的分角色数据要等下一次运行。

#### H=2048

| # | update | x_real_time | collect_s | update_s | graph_frames_uid_max | graph_frames | detached_frames | dropped_grad | src_alloc_gib_max | 容器 memory GiB | anon GiB | shmem GiB | file GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48185 | 52 | 52.8 | 20.9 | 1071 | 57458 | 0 | 0.0 | 6.3 | 43.4 | 35.4 | 7.3 | 7.5 |
| 2 | 48186 | 793 | 5.1 | 4.6 | 1071 | 60494 | 0 | 0.0 | 6.4 | 44.7 | 36.7 | 7.3 | 7.5 |
| 3 | 48187 | 308 | 11.7 | 6.9 | 2080 | 117915 | 0 | 0.0 | 12.8 | 47.0 | 38.9 | 7.3 | 7.6 |
| 4 | 48188 | 649 | 6.8 | 5.5 | 2091 | 82131 | 0 | 0.0 | 10.1 | 48.2 | 40.1 | 7.4 | 7.6 |
| 5 | 48189 | 300 | 12.0 | 8.3 | 3114 | 147752 | 0 | 0.0 | 16.9 | 49.3 | 41.1 | 7.4 | 7.6 |
| 6 | 48190 | 516 | 5.9 | 7.3 | 3327 | 164271 | 0 | 0.0 | 18.3 | 52.0 | 42.7 | 8.5 | 8.7 |
| 7 | 48191 | 433 | 6.3 | 8.7 | 3327 | 180792 | 0 | 0.0 | 20.2 | 54.3 | 42.7 | 10.6 | 10.9 |
| 8 | 48192 | 306 | 6.6 | 13.3 | 3321 | 113873 | 2119 | 0.0 | 13.0 | 55.4 | 43.7 | 10.7 | 10.9 |
| 9 | 48193 | 836 | 8.2 | 4.1 | 3321 | 113982 | 2119 | 0.0 | 13.0 | 55.4 | 43.7 | 10.7 | 10.9 |
| 10 | 48194 | 298 | 12.2 | 6.6 | 3058 | 98833 | 2695 | 0.0 | 10.4 | 58.3 | 46.5 | 10.7 | 10.9 |
| 11 | 48195 | 263 | 16.5 | 6.7 | 3827 | 139882 | 2695 | 0.0 | 16.6 | 60.6 | 48.8 | 10.7 | 10.9 |
| 12 | 48196 | 321 | 10.6 | 9.0 | 4036 | 108208 | 1614 | 0.0 | 16.0 | 61.8 | 49.6 | 11.1 | 11.3 |
| 13 | 48197 | 211 | 10.7 | 16.6 | 4036 | 123294 | 3605 | 0.0 | 14.5 | 64.1 | 51.9 | 11.1 | 11.3 |
| 14 | 48198 | 396 | 10.3 | 6.6 | 4036 | 118411 | 4105 | 0.0 | 13.3 | 64.6 | 52.3 | 11.1 | 11.3 |
| 15 | 48199 | 412 | 13.5 | 6.5 | 4036 | 160197 | 4105 | 0.0 | 18.0 | 65.0 | 52.6 | 11.2 | 11.4 |
| 16 | 48200 | 382 | 11.1 | 6.9 | 4036 | 107364 | 6617 | 0.0 | 11.8 | 65.7 | 53.2 | 11.2 | 11.4 |
| 17 | 48201 | 231 | 10.5 | 15.0 | 4036 | 130765 | 5027 | 0.0 | 15.7 | — | — | — | — |
| 18 | 48202 | 394 | 10.0 | 8.0 | 4036 | 121073 | 4442 | 0.0 | 13.5 | — | — | — | — |

| 采样时间(s) | 主机可用 GiB | 外部 pressure full avg10 最大 % | 外部 swap 累计增长 MiB | 主机 pswpin 页/s | 我们容器 GiB | 停止原因 |
|---|---|---|---|---|---|---|
| 0 | 84.9 | 0.0 | 0 | 0 | 19.4 |  |
| 22 | 74.8 | 0.0 | 0 | 0 | 29.6 |  |
| 43 | 72.9 | 0.0 | 0 | 0 | 31.6 |  |
| 65 | 63.7 | 0.0 | 0 | 0 | 40.6 |  |
| 90 | 58.1 | 0.0 | 0 | 0 | 46.1 |  |
| 115 | 53.9 | 0.0 | 0 | 0 | 50.4 |  |
| 142 | 48.6 | 0.85 | 15 | 0 | 55.5 |  |
| 167 | 46.2 | 0.32 | 34 | 0 | 57.9 |  |
| 193 | 42.8 | 7.99 | 120 | 0 | 61.6 |  |
| 217 | 38.8 | 18.83 | 133 | 0 | 65.4 |  |
| 242 | 38.6 | 23.7 | 133 | 0 | 65.8 |  |
| 267 | 37.1 | 22.27 | 138 | 0 | 67.1 | host_memory_pressure |

#### H=0

| # | update | x_real_time | collect_s | update_s | graph_frames_uid_max | graph_frames | detached_frames | dropped_grad | src_alloc_gib_max | 容器 memory GiB | anon GiB | shmem GiB | file GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48185 | 51 | 51.9 | 24.0 | 1057 | 65648 | 0 | 0.0 | 7.2 | 44.4 | 35.4 | 7.3 | 8.6 |
| 2 | 48186 | 963 | 5.4 | 3.7 | 1058 | 60501 | 0 | 0.0 | 6.6 | 45.3 | 36.2 | 7.3 | 8.6 |
| 3 | 48187 | 304 | 11.3 | 6.0 | 2027 | 117935 | 0 | 0.0 | 12.8 | 48.5 | 39.3 | 7.3 | 8.6 |
| 4 | 48188 | 500 | 7.5 | 7.3 | 2386 | 105406 | 0 | 0.0 | 14.4 | 49.9 | 40.6 | 7.4 | 8.6 |
| 5 | 48189 | 536 | 8.7 | 6.4 | 3200 | 146493 | 0 | 0.0 | 15.7 | 50.8 | 41.6 | 7.4 | 8.6 |
| 6 | 48190 | 392 | 6.7 | 9.9 | 3235 | 171176 | 0 | 0.0 | 18.9 | 53.0 | 42.2 | 8.8 | 10.1 |
| 7 | 48191 | 303 | 8.9 | 12.9 | 3235 | 94162 | 0 | 0.0 | 10.6 | 53.2 | 42.4 | 8.9 | 10.1 |
| 8 | 48192 | 482 | 6.5 | 7.6 | 3486 | 93773 | 0 | 0.0 | 11.5 | 55.1 | 44.2 | 8.9 | 10.2 |
| 9 | 48193 | 437 | 10.4 | 7.6 | 3989 | 126616 | 0 | 0.0 | 15.2 | 56.4 | 45.4 | 8.9 | 10.2 |
| 10 | 48194 | 371 | 12.5 | 6.6 | 4331 | 107968 | 0 | 0.0 | 15.5 | 58.9 | 47.7 | 9.0 | 10.3 |
| 11 | 48195 | 212 | 12.8 | 13.8 | 4366 | 122084 | 0 | 0.0 | 15.2 | 61.7 | 50.4 | 9.0 | 10.3 |
| 12 | 48196 | 403 | 11.0 | 8.5 | 4366 | 113272 | 0 | 0.0 | 13.5 | 62.4 | 51.1 | 9.1 | 10.4 |
| 13 | 48197 | 489 | 10.5 | 6.9 | 4366 | 146657 | 0 | 0.0 | 18.1 | 63.1 | 51.8 | 9.1 | 10.4 |
| 14 | 48198 | 342 | 11.8 | 8.4 | 3617 | 117072 | 0 | 0.0 | 13.1 | 65.0 | 51.8 | 11.0 | 12.2 |
| 15 | 48199 | 263 | 10.3 | 14.3 | 3639 | 136337 | 0 | 0.0 | 16.5 | 65.0 | 51.7 | 11.0 | 12.3 |
| 16 | 48200 | 673 | 9.9 | 5.1 | 3902 | 118579 | 0 | 0.0 | 16.8 | 65.6 | 52.3 | 11.0 | 12.3 |
| 17 | 48201 | 364 | 10.4 | 6.5 | 4163 | 152188 | 0 | 0.0 | 18.8 | 66.4 | 52.6 | 11.4 | 12.6 |
| 18 | 48202 | 297 | 10.9 | 10.5 | 4894 | 139477 | 0 | 0.0 | 17.8 | 67.0 | 53.2 | 11.4 | 12.7 |
| 19 | 48203 | 542 | 9.6 | 6.3 | 4894 | 149886 | 0 | 0.0 | 19.4 | 68.2 | 54.3 | 11.4 | 12.7 |
| 20 | 48204 | 279 | 9.6 | 12.1 | 4056 | 147922 | 0 | 0.0 | 18.0 | 70.6 | 56.6 | 11.4 | 12.7 |
| 21 | 48205 | 557 | 11.1 | 5.4 | 4056 | 149407 | 0 | 0.0 | 18.9 | 71.0 | 56.8 | 11.6 | 12.9 |
| 22 | 48206 | 244 | 10.5 | 10.4 | 4223 | 151192 | 0 | 0.0 | 19.3 | 69.6 | 56.7 | 11.8 | 11.8 |
| 23 | 48207 | 236 | 11.0 | 15.3 | 4126 | 156758 | 0 | 0.0 | 18.8 | — | — | — | — |
| 24 | 48208 | 370 | 10.6 | 6.9 | 4385 | 141294 | 0 | 0.0 | 17.9 | — | — | — | — |

| 采样时间(s) | 主机可用 GiB | 外部 pressure full avg10 最大 % | 外部 swap 累计增长 MiB | 主机 pswpin 页/s | 我们容器 GiB | 停止原因 |
|---|---|---|---|---|---|---|
| 0 | 85.7 | 0.0 | 0 | 0 | 20.0 |  |
| 22 | 74.9 | 0.0 | 0 | 0 | 30.8 |  |
| 43 | 73.2 | 0.0 | 0 | 0 | 32.7 |  |
| 65 | 63.8 | 0.0 | 0 | 0 | 41.8 |  |
| 90 | 59.0 | 0.0 | 0 | 0 | 46.7 |  |
| 115 | 54.0 | 0.0 | 0 | 0 | 51.6 |  |
| 140 | 51.0 | 0.0 | 0 | 0 | 54.7 |  |
| 168 | 48.0 | 0.0 | 0 | 0 | 57.5 |  |
| 191 | 42.5 | 0.0 | 0 | 0 | 62.9 |  |
| 218 | 41.7 | 0.09 | 0 | 0 | 63.8 |  |
| 242 | 38.5 | 0.0 | 0 | 0 | 66.9 |  |
| 269 | 38.2 | 0.01 | 0 | 0 | 67.3 |  |
| 295 | 35.4 | 0.0 | 54 | 0 | 70.3 |  |
| 321 | 32.6 | 17.0 | 351 | 0 | 72.0 |  |
| 344 | 35.5 | 31.34 | 458 | 0 | 69.4 | benchmark_cgroup_oom |


### 8.5 结论（只限于这个测量窗口）

1. **horizon 本身按设计工作**：从第 8 次更新起开始释放图；丢弃梯度一直恰为 0。H=2048 时每局最多保留的图帧数稳定在 4036 左右，H=0 涨到 4894。
2. **但在这个窗口里它能省的不多**：从 checkpoint 恢复后每局从头开始，24 次更新内单局最长约 4.9k 帧，所以被释放的帧只有 4–6.6k，而保留的图帧总共 10–16 万帧。源卡显存峰值两组相近：H=2048 是 10–20 GiB，H=0 是 16–19 GiB。
3. **真正的瓶颈是主机内存**：两组的容器内存曲线几乎重合，第 16 次更新都约 65 GiB；主要是匿名内存（约 53 GiB），每次更新约涨 1 GiB。H=0 在第 24 次更新时撞上 72 GiB 的 OOM，H=2048 先被外部压力规则停了。horizon 只管 GPU 上的图，管不到这部分。
4. **吞吐（4 卡、非稳态、只作参考）**：
   - 第 2–10 次更新：H=2048 为 419.6×（collect 中位数 6.78 s，update 6.89 s）；H=0 为 426.6×（8.65 s，7.30 s）。
   - 第 11 次以后：H=2048 为 306.7×（10.63 s，7.46 s），H=0 为 334.6×（10.56 s，8.43 s）。
   - 两组差别在噪声范围内；后段 H=2048 正好和外部压力时段重叠。

### 8.6 估算：horizon 之外帧的原始观测占多少主机内存

- 每帧原始观测（ROW）22,856 字节，不含很小的 pending 部分。
- 每份完整历史：在世的帧总数（保留图的帧 + 已释放的帧）约 10–16 万，原始字节约 2.5–3.7 GB。
- 其中 horizon 之外的帧（已释放图、只需要 u/M 数值供读取，原始字节只有 CF/教师恢复会用到）只有 4–6.6k 帧，每份约 0.09–0.15 GB。
- 主机上原始历史有好几份拷贝（训练主进程里 rollout 的 raw_contexts、学习器解包后的历史、两个 actor 的 ActorMemory、IPC 缓冲）。拷贝份数要等修好的分角色 RSS 数据确认。
- 结论：在这个窗口里，horizon 之外帧的原始字节最多只占约 53 GiB 匿名内存中的 1 GB 左右，**不是**主机内存增长的主因。每次更新约 1 GiB 的增长另有来源，要等分角色 RSS 定位。随着局长增长，这部分占比会变大，所以它仍然可以作为以后省主机内存的候选，但现在不改代码。

## 9. 主机内存定位（H=2048，4 卡，带进程内计量，代码 05e7cde）

运行：`v3-bfd3656` 之后只加了默认关闭的计量（`ISAAC_MEMORY_ACCOUNT=1`），horizon 代码不变。用例 `v3-05e7cde-ab4h2048acct-10240954`，跑了 23 次更新后，guard 按我们自己 cgroup 的 OOM（72 GiB）停止，退出码 137。结束后已确认主机上没有 memory-v3 容器。

日志：
- `results/v3-05e7cde-ab4h2048acct-10240954/train.log`：`MEMORY_ACCOUNT` 行。学习器每次更新一行；两个 actor 每次交接一行。
- `memory.jsonl`：每次更新一行，含 cgroup memory.stat 拆分，以及按角色的 RSS（角色识别已修好）。
- `guard.jsonl`：外部压力、swap、pswpin 的时间线。

计量口径：
- 对指定结构递归遍历，深度最多 6 层；不同结构之间共享的 numpy 底层数组只计一次。
- 原始历史 = 可达 RawHistory 的全部数组。帧数 = 每局最长前缀之和。
- RSS 来自 `/proc/self/status`。torch 的锁页 host 分配来自 `torch.cuda.host_memory_stats()`。
- 训练主进程就是 actor0（B38 设计：第一个 actor 跑在主进程里）。
- 游戏进程（isaac.x64）是 fork 出来的，RSS 会重复计算共享页，所以这里只列进程数。

### 9.1 每次更新

| # | update | 容器 anon | shmem | RSS 训练主进程(=actor0) | 学习器 | 3 个源进程 | actor1 | 游戏桥接 | 游戏进程数 | 学习器 RssAnon | 学习器原始历史 GiB / 帧数 / 局数 | 学习器教师+v2 记录原始 GiB | torch 锁页 host 分配 GiB | actor0 RssAnon / rollout 原始 GiB | actor1 RssAnon / rollout 原始 GiB | 源侧在世局 / 帧 | uid_max | 已释放帧 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48185 | 35.61 | 7.31 | 5.72 | 5.88 | 4.85 | 5.71 | 4.24 | 324 | 2.90 | 1.37 / 60172 / 105 | 0.00 | 1.03 | 1.69 / 0.67 | 1.68 / 0.68 | 178 / 65668 | 1068 | 0 |
| 2 | 48186 | 36.22 | 7.31 | 6.01 | 5.96 | 5.21 | 5.59 | 4.24 | 324 | 2.97 | 1.39 / 60203 / 105 | 0.00 | 1.05 | 2.40 / 0.96 | 2.39 / 0.97 | 260 / 60463 | 1068 | 0 |
| 3 | 48187 | 40.16 | 7.33 | 6.22 | 7.45 | 5.50 | 5.62 | 4.25 | 385 | 4.43 | 1.92 / 83145 / 108 | 0.00 | 1.08 | 3.13 / 1.14 | 2.73 / 1.17 | 426 / 126071 | 2116 | 0 |
| 4 | 48188 | 40.15 | 7.34 | 6.54 | 7.44 | 5.54 | 6.05 | 4.25 | 399 | 4.42 | 1.94 / 83175 / 108 | 0.00 | 1.10 | 3.42 / 1.22 | 2.93 / 1.21 | 493 / 83668 | 2117 | 0 |
| 5 | 48189 | 42.15 | 7.37 | 6.58 | 8.11 | 5.62 | 5.84 | 4.25 | 477 | 4.92 | 1.71 / 73240 / 89 | 0.00 | 1.13 | 3.55 / 1.14 | 2.97 / 1.09 | 676 / 141116 | 2880 | 0 |
| 6 | 48190 | 42.00 | 7.89 | 6.48 | 7.99 | 6.14 | 5.62 | 4.26 | 505 | 5.06 | 1.88 / 80349 / 95 | 0.00 | 1.16 | 3.56 / 1.25 | 2.86 / 1.17 | 704 / 98529 | 2703 | 1616 |
| 7 | 48191 | 44.17 | 7.91 | 7.07 | 8.18 | 6.20 | 6.19 | 4.26 | 531 | 5.16 | 1.91 / 81816 / 103 | 0.00 | 1.18 | 3.68 / 1.28 | 2.93 / 1.09 | 803 / 113910 | 3350 | 1353 |
| 8 | 48192 | 44.38 | 7.92 | 6.93 | 7.77 | 6.14 | 6.18 | 4.26 | 552 | 5.15 | 1.55 / 66644 / 82 | 0.00 | 1.21 | 3.73 / 1.22 | 2.86 / 1.19 | 904 / 99101 | 2970 | 1526 |
| 9 | 48193 | 44.65 | 7.93 | 7.14 | 7.82 | 6.19 | 5.95 | 4.61 | 601 | 5.12 | 1.68 / 71339 / 90 | 0.00 | 1.23 | 3.69 / 1.32 | 2.96 / 1.11 | 1005 / 123791 | 2987 | 1526 |
| 10 | 48194 | 47.73 | 8.60 | 6.76 | 8.07 | 6.89 | 6.83 | 5.07 | 679 | 5.17 | 1.50 / 63126 / 73 | 0.00 | 1.26 | 3.89 / 1.25 | 3.09 / 1.22 | 1081 / 94094 | 2947 | 832 |
| 11 | 48195 | 50.11 | 8.62 | 7.87 | 8.40 | 6.87 | 7.29 | 5.60 | 738 | 5.32 | 1.89 / 80071 / 85 | 0.05 | 1.28 | 4.04 / 1.18 | 3.49 / 1.17 | 1128 / 107713 | 3280 | 1024 |
| 12 | 48196 | 51.20 | 8.63 | 8.16 | 7.94 | 6.85 | 7.80 | 5.73 | 770 | 5.41 | 1.72 / 73269 / 80 | 0.07 | 1.31 | 4.53 / 1.28 | 3.95 / 1.27 | 1219 / 109272 | 3098 | 717 |
| 13 | 48197 | 52.93 | 8.63 | 8.61 | 8.24 | 6.92 | 8.08 | 5.79 | 795 | 5.40 | 1.77 / 74813 / 80 | 0.14 | 1.34 | 4.99 / 1.30 | 4.63 / 1.24 | 1313 / 145046 | 3098 | 717 |
| 14 | 48198 | 53.36 | 9.85 | 8.86 | 8.26 | 8.02 | 8.13 | 5.68 | 819 | 5.61 | 1.79 / 76176 / 74 | 0.15 | 1.36 | 5.43 / 1.33 | 4.88 / 1.28 | 1352 / 99625 | 2811 | 1990 |
| 15 | 48199 | 55.60 | 9.88 | 9.42 | 8.67 | 8.23 | 8.76 | 5.23 | 829 | 5.74 | 2.08 / 88162 / 85 | 0.25 | 1.39 | 5.67 / 1.28 | 4.91 / 1.37 | 1428 / 130236 | 3088 | 2150 |
| 16 | 48200 | 54.53 | 9.88 | 9.03 | 8.48 | 8.12 | 8.19 | 4.93 | 835 | 5.79 | 1.96 / 83582 / 71 | 0.25 | 1.41 | 6.07 / 1.33 | 5.42 / 1.40 | 1499 / 125925 | 3265 | 2527 |
| 17 | 48201 | 54.81 | 9.89 | 9.04 | 8.59 | 8.20 | 7.77 | 4.76 | 834 | 5.88 | 2.33 / 99080 / 83 | 0.28 | 1.44 | 5.94 / 1.37 | 5.03 / 1.43 | 1578 / 152064 | 3528 | 2527 |
| 18 | 48202 | 56.00 | 11.20 | 9.46 | 8.70 | 9.33 | 7.94 | 4.94 | 881 | 5.96 | 2.41 / 101519 / 77 | 0.30 | 1.47 | 5.91 / 1.53 | 4.76 / 1.52 | 1659 / 127052 | 2866 | 7211 |
| 19 | 48203 | 57.95 | 11.25 | 8.57 | 9.44 | 9.57 | 7.26 | 4.95 | 904 | 6.36 | 2.62 / 109813 / 82 | 0.33 | 1.49 | 6.11 / 1.60 | 4.63 / 1.59 | 1728 / 167789 | 3333 | 12341 |
| 20 | 48204 | 57.57 | 11.26 | 9.61 | 9.48 | 9.40 | 8.07 | 5.48 | 930 | 6.44 | 2.36 / 99191 / 82 | 0.38 | 1.52 | 6.34 / 1.46 | 5.14 / 1.56 | 1751 / 129578 | 3068 | 9730 |
| 21 | 48205 | 59.65 | 11.27 | 9.97 | 9.15 | 9.55 | 8.48 | 5.48 | 957 | 6.67 | 2.25 / 94857 / 79 | 0.43 | 1.55 | 6.26 / 1.49 | 4.80 / 1.65 | 1867 / 153786 | 3042 | 13327 |
| 22 | 48206 | 58.88 | 11.31 | 9.66 | 9.82 | 9.54 | 8.39 | 5.25 | 957 | 6.65 | 2.07 / 87299 / 80 | 0.58 | 1.57 | 6.62 / 1.48 | 5.13 / 1.56 | 1941 / 140337 | 2865 | 12669 |


### 9.2 增长从哪来（第 1→22 次更新，容器匿名内存 +23.3 GiB）

| 部分 | 增长 | 其中能对上的结构 |
|---|---|---|
| 训练主进程（= actor0） | RSS +3.9 GiB（RssAnon 1.7→6.6） | 两份 rollout 的原始历史 +0.8 GiB；其余未被这些结构覆盖 |
| 学习器 | RSS +3.9 GiB（RssAnon 2.9→6.7） | 接收到的原始历史 +0.7 GiB；教师/v2 记录的原始历史 +0.6 GiB；torch 锁页 host 分配 +0.5 GiB；固定的 0.88 GiB 锁页暂存不增长 |
| 3 个源进程 | RSS +4.7 GiB | 源进程内部没有加计量。它们持有 CPU 侧的保存激活（每个上限 0.75 GiB）、IPC 导出和每个 chunk 的簿记 |
| actor1 | RSS +2.7 GiB | 两份 rollout 的原始历史 +0.9 GiB |
| 游戏桥接进程 | RSS +1.0 GiB（进程数 64→84） | — |
| 游戏进程 | 进程数 324→957 | 这些是 fork 出来的分支和存档游戏（教师/CF/archive），私有脏页算进容器匿名内存。增长的剩余部分（约 7 GiB）大概率在这里 |

与哪个量同步（第 1–22 次更新，Pearson 相关系数，匿名内存对各量）：
- 游戏进程数 r=0.994，斜率约 36 MB/个；
- 源侧在世局数 r=0.989，斜率约 14 MB/局。在世局数从 178 涨到 1941，主要是不断写入缓冲区的教师/v2 记录各自对应的局；缓冲区容量是 60000/30000 条，这个窗口里远没到上限；
- 学习器在世局的帧数总和 r=0.72；源侧在世帧数 r=0.74；
- 更新序号 r=0.985，斜率 1.16 GiB/次更新。

解读：增长与"按局保留的主轨迹帧数"只是弱相关；它跟着随时间单调增加的两样东西走——fork 出来的游戏进程，和教师/v2 记录对应的局。所以它更像是"按记录/按分支累积"的保留，而不是泄漏，也不是按主局长度保留的东西。不过这两个量本身都随时间单调上升，相关性无法把它们分开。

### 9.3 旧架构同窗口对照

来源：`temp/memory-chain-v2/memory-v2-baseline-20261022.guard.jsonl`，843.4× 那一组（3 卡）。

- 按 progress 的累计 wall 换算成更新序号：容器内存在第 2 次更新约 54.8 GiB；第 7–42 次更新一直在 55.4–57.0 GiB。
- 进程/线程数（pids）在约 120 s 时到 1921，之后稳定在约 1900–1970。
- 新架构这一轮：pids 从 764 慢慢涨到 2024（约 340 s），内存从 19.8 涨到 71.6 GiB。

结论：
- 旧架构没有"每次更新涨约 1 GiB"的现象，它很快就平了。这个增长是新记忆设计（或它带来的更慢的更新节奏）引入的，不是旧问题。
- 在 pids 相近（约 2000）的时候，新架构的容器内存比旧架构多约 15 GiB。这和上表中学习器、源进程、actor 三部分的增长量级一致。
- 新架构的游戏 fork 数量是逐步涨上去的。它最终会不会和旧架构一样走平，这个窗口（23 次更新）还看不出来：72 GiB 的上限先到了。

### 9.4 horizon 之外帧的原始观测（下一步省主机内存的候选，现在不改代码）

- 每帧原始观测 22,856 字节。
- 这一轮已释放图的帧约 0.4–7k 帧，对应原始字节约 0.01–0.15 GiB × 拷贝份数。原始历史至少在这几处各有一份：学习器（接收到的历史）、训练主进程/actor0 和 actor1（两份 rollout）、教师/v2 记录。
- 在这个窗口里，这部分最多约 0.5 GiB，**不是**增长的主因。
- 局变长以后它会线性增长（每帧每份 22.9 KB）。到时候可以对 horizon 之外的帧只保留 u/M 数值，原始字节只留给 CF/教师恢复真正需要的那些局；但以这次的数据来看，应该优先处理的是 §9.2 里 fork 游戏进程和教师/v2 记录对应局的累积，以及源进程主机侧的增长。

## 10. 引用链与短消融（H=2048，4 卡，代码 05e7cde，各跑 10 次更新）

### 10.1 引用链（读代码得到）

**谁持有游戏进程（isaac.x64）**：持有者是每个游戏 worker 里的 sampler（`isaac_bridge/tok_sampler.py`），不是记录本身。每个 worker 持有：
- 一个 root 实例、一个 parked template、一个正在跑的 episode clone；
- floor 存档 `archive_size=12` 个 parked 的进房快照，`snapshot_prob=0.5`，满了以后按规则淘汰；
- 等待中的分支点：`teacher_v2_queue=8`、`cf_queue=8`、`branch_queue=4` 个，各自持有本房间的 parked entry；
- 正在搜索的 fork：`teacher_slots=6`。

所以游戏进程数有硬上限，约 64×(3+12+20)+搜索中的 fork。progress.csv 的 `archive` 列显示，存档在 23 次更新里从 51 填到 663（上限 768），和游戏进程数 324→957 同步。旧架构用的是同一套 sampler 配置（逐项对比过两边的命令行：只有记忆/源相关参数不同），它的存档在第 13 次更新就填满了 768。

**谁让源侧在世的局变多**：
- `train_tok.py` 每次 retain 时，在世 uid 集合 = 本 rollout 的局 ∪ `L['v2_contexts']`（v2/CF 环形缓冲，30000 条）∪ `L['teach_contexts']`（教师环形缓冲，60000 条）∪ 预取 ∪ 已 collect 的局。
- 每条教师记录都是 `RawHistory.empty()`，有自己的 uid；它在源流水线里占一帧，带图，作为 stateless chunk，一直保留到环形缓冲区里它的槽位被覆盖。
- v2/CF 记录按分支 lane 共用 uid，各自重编码完整前缀。
- 源侧在世局数 1941 ≈ 教师记录 1815 + 主局 64 + v2 分支。旧架构的教师记录只是一行数据，在新架构里每条记录额外带着源侧的一帧图和簿记。

### 10.2 消融结果

A、B、C 三组都正常跑完 10 次更新，退出码 0。原始数据：`results/v3-05e7cde-abl{A,B,C}-*/`；逐次更新的表：`results/ablation-summary.txt`。

| 组 | 改动 | 容器匿名内存斜率（第 2–10 次更新） | 游戏进程数（第 1→10 次） | archive | 教师记录 | v2 记录 |
|---|---|---|---|---|---|---|
| A | 教师、CF（以及它产生的 v2 记录）全关 | **0.78 GiB/次更新** | 276→543 | 49→351 | 0 | 0 |
| B | 只开教师（CF 关），`--teach-buffer 2000 --teacher-v2-buffer 1000` | 0.89 | 323→592 | 51→360 | 54→1005 | 0 |
| C | 现配置 | 0.84 | 322→585 | 51→316 | 54→906 | 0 |
| 参照：带计量的那一轮（同 C 配置） | — | 1.17（第 2–10 次）；1.14（第 2–23 次） | 324→957（23 次） | 51→663 | 53→1815 | 0→288 |

读法：
1. **在前 10 次更新里，斜率在三组都存在，差别在同配置两次运行的差异范围之内**（C 是 0.84，同配置的计量那一轮是 1.17）。关掉教师和 CF 没有让斜率消失。所以这个窗口里的增长**不是**教师记录造成的。
2. **B 组没能测到"缓冲区上限"**：教师记录 10 次更新只涨到约 1000 条，还没到 2000 的上限，所以 B 实际上等于"教师开、CF 关"。
3. **v2/CF 记录在 10 次更新的窗口里还没出现**（第 10 次更新前 v2_held 一直是 0），这次消融测不到它们。在 23 次更新那一轮里，它们从第 10 次更新后开始累积（到 288 条）。它们是否贡献了后段的斜率，需要跑满 20 次以上的 A/C 对照才能判断。
4. **A 组 +7.2 GiB 的分配**：
   - 训练主进程 +0.6 GiB、学习器 +1.4 GiB、源进程 +0.7 GiB，actor1 和游戏桥接基本不变；
   - 剩下约 4.5 GiB 没有落在 python 进程的 RSS 里。同期存档从 49 填到 351、游戏进程 +267 个，最可能是 parked 游戏进程的私有脏页（约 17 MB/个）。
   - 但游戏进程是 fork 出来的，RSS 无法给出私有页，这一项要用 smaps_rollup 的 Private_Dirty 才能确认。
5. **和旧架构对不上的地方**：旧架构在第 1–13 次更新里存档同样从 173 填到 768，容器内存却只从 54.8 涨到约 56 GiB，折算每个 parked 进程只有 3–5 MB。新架构里，同样的存档增长对应的内存增长明显更大。这个差异的原因还没定位。可能是 parked 游戏进程在新架构下私有页更多，也可能是其它没计量到的部分。

### 10.3 下一步建议（还没做，等一起定规则）

- 在 guard 里对游戏进程按 smaps_rollup 汇总 Private_Dirty 和 Pss，按存档 / 队列 / 运行中分组，直接量出每个 parked 进程的私有内存，并和旧架构同一窗口对比。
- 跑一组 20 次以上更新的 A/C 对照，看 v2/CF 记录出现以后斜率有没有分开。
- 在这两项数据出来之前，不改"记录被消费后释放"的保留策略。

## 11. 新旧架构逐项对照（smaps_rollup 实测，代码 05e7cde）

### 11.1 做法

- guard 每次更新做两件事：
  - 写一行 cgroup 的 memory.stat（全部字段）；
  - 在另一个线程里 `docker exec` 运行 `host/audit_inside.py`：读容器内每个进程的 `/proc/<pid>/smaps_rollup`（Pss、Private_Dirty 等），并读 `/dev/shm` 的占用。
- 游戏进程的角色来自 worker 自己写的角色记录（`ISAAC_RL_MEM_LOG`，即 tok_floor 的 mem_dump）：root、template、archive、hurt 搜索队列、其它 parked、正在跑的局、base。
- 限制：mem_dump 只在局边界写，所以只有一部分 worker 写过记录（旧架构 32/64）。没被映射到的游戏进程记为 "other"。
- 三组运行，结束后都确认主机上没有残留容器：
  - 旧架构：`v3-05e7cde-audOld-10241042`，现成 workspace 代码 + v2 的 baseline 命令，3 张卡（物理卡 3、2、4），72 GiB，12 游戏小时，10 次更新，退出码 0。第 9 次更新以后的审计行落在进程退出期间，已排除。
  - 新架构 C（现配置）：`v3-05e7cde-audC-10241047`，21 次更新，退出码 0，最高 67.6 GiB。
  - 新架构 A（教师、CF 全关）：`v3-05e7cde-audA-10241056`，21 次更新，退出码 0，最高 60.1 GiB。

### 11.2 每次更新各类内存的增量（Pss，GiB/次更新）

| GiB per update | 旧架构 (1->7) | 新C (1->7) | 新A (1->7) | 新C (9->20) | 新A (9->20) |
|---|---|---|---|---|---|
| games: archive (parked room entries) | +0.268 | +0.666 | +0.566 | +0.338 | +0.357 |
| games: other (unmapped, episodes, queues, root, template) | +0.839 | -0.269 | -0.090 | +0.384 | +0.272 |
| python: trainer (= actor0) | -0.004 | +0.097 | +0.104 | +0.207 | +0.053 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.003 | +0.457 | +0.403 | +0.168 | -0.016 |
| python: other spawned (3 source workers, ...) | -0.017 | +0.078 | +0.169 | +0.334 | +0.110 |
| compile + other | +0.000 | +0.000 | -0.000 | +0.000 | +0.000 |
| Pss sum | +1.089 | +1.030 | +1.152 | +1.430 | +0.776 |
| cgroup anon | +1.478 | +1.216 | +1.310 | +1.067 | +0.686 |
| cgroup shmem | +0.024 | +0.034 | +0.097 | +0.322 | +0.110 |
| coverage (Pss / anon+shmem) | 72% | 82% | 82% | 103% | 97% |
| archive procs | 215->278 | 24->136 | 38->136 | 166->319 | 172->331 |
| level at end (cgroup anon+shmem GiB) | 49.8 | 50.4 | 50.6 | 67.6 | 60.1 |


原始数据：每个运行目录下的 `memory.jsonl`；逐次更新的完整表：`results/audit-table-full.txt`；脚本：`scripts/audit_table.py`、`scripts/compare_table.py`。
覆盖率是 Pss 增量之和 ÷ cgroup（anon + shmem）增量：后段 97–103%，前段 72–82%。前段偏低，是因为前几次审计时一部分游戏进程还没有角色记录，而且 Pss 和 cgroup 计费的口径本身不完全相同。

### 11.3 结论（更正 §10 里的一处推断）

1. **游戏进程两边都有，旧架构也一样涨**：旧架构前 7 次更新匿名内存同样每次更新涨约 1.5 GiB，而且几乎全部是游戏进程（存档 parked 进程 + 其它 fork），python 进程基本不涨。每个 parked 存档进程的 Pss 在两边都约 25–30 MB。§10 里说旧架构每个 parked 进程只有 3–5 MB 是**错的**：那是用错位的时间–更新映射折算出来的，现以本节实测为准。
2. **差别在填满的速度**：按 progress.csv 的 `archive` 列（全部 64 个 worker），旧架构第 1 次更新 163 个、第 7 次 714 个、第 9 次 747 个，v2 baseline 第 13 次更新到 768 的上限；新架构 C 组第 7 次 260 个、第 20 次 613 个，A 组第 20 次 637 个，还在填。所以新架构在到达游戏平台之前，就先撞上了 72 GiB。
3. **python 进程的增长只有新架构有**：
   - **前 9 次更新，与记录无关**：C 和 A 都是每次更新约 +0.6–0.7 GiB，主要在 trainer 拉起的进程里（学习器、actor1、64 个游戏桥接）和 trainer 本身。A 组在第 9 次更新以后这部分变平（-0.016），说明是一次性的预热，不是持续增长。
   - **第 9 次更新以后，跟着记录走**：C 组 python 合计每次更新 +0.71 GiB（trainer +0.21、源进程 +0.33、学习器等 +0.17），A 组只有 +0.15；shmem 也是 C +0.32 对 A +0.11。v2/CF 记录正是从第 10 次更新左右开始累积的。也就是说，教师/CF/v2 记录在后段每次更新约多出 0.55 GiB 的 python 内存和 0.2 GiB 的共享内存。§10 的 10 次更新窗口正好错过了这一段。
4. **后段总斜率**：C 组 anon+shmem 每次更新 +1.39 GiB，A 组 +0.80 GiB。其中游戏部分两组接近（+0.63–0.72），会在存档填满 768 后停住；记录相关的部分（约 0.6 GiB/次更新）只有 C 组有，在缓冲区（60000/30000 条）填满之前不会停。
5. **还缺的一项**：CUDA 上下文和锁页内存在 host 侧的占用，这次没有单独列出来。smaps 已经按映射类型汇总了每个 python 进程（anon、heap、/dev/nvidia*、shm、file，在 `memory.jsonl` 的 `audit.python` 字段里），但还没做成表。学习器的 torch 锁页 host 分配（§9）是 1.0→1.6 GiB。

**下一步**：在这些数据的基础上，再一起定"记录被消费后释放其源历史"的具体规则。保留策略这次仍然没有改。

## 12. 记录的字节构成，以及按批准规则修改保留策略后的验证

### 12.1 （a）每条 v2/CF 记录占多少（旧保留策略，`v3-05e7cde-audC-10241047`，学习器的 MEMORY_ACCOUNT）

- **学习器侧原始帧**：第 21 次更新时有 256 条 v2 记录，分属 32 个分支局；学习器为它们持有原始帧 0.49 GiB。折合每条记录约 2.0 MB、每个分支局约 15.6 MB，每个分支局平均 720 帧（23065 帧 / 32），每帧 22,856 字节。
  - 这 720 帧几乎都是**从父局复制来的前缀**。原因：记录取在分支的前几步（`--cf-distill-steps 8`），而分支点就在当时主局的第几百帧。
  - 复制发生在 IPC 上：actor 端的分支历史本来只是引用父局数组（`fork()` 共享 blocks/tail），但 `job['v2']` 经过 pickle 送到学习器进程后，就变成了一份独立的完整拷贝。
- **源侧**：分支局有自己的 uid，旧代码会把它的整个前缀从第 0 帧起重新编码一遍，并保留图（源卡上约 470 KB/帧）。另外有两个事实：
  - v2 记录要等缓冲区里攒满 256 条才开始被采样（`L['v2_n'] >= 256`）。在 21 次更新里它们一次都没被消费，所以这期间一直带着图。
  - 教师记录每条在源侧占一帧并带图，直到 60000 条的环形缓冲区覆盖它的槽位。
- **共享内存里的 IPC 份额**：没有单独测出来。

### 12.2 （b）前 9 次更新 python 多出的内存按映射类型分（smaps，同一运行）

| 进程 | heap（glibc） | 匿名 mmap | /dev/nvidia*（CUDA 映射） | /dev/shm | 文件映射 |
|---|---|---|---|---|---|
| trainer（= actor0） | 1.88→3.23 GiB | 0.56→0.57 | 0.02→0.02 | 2.21→2.20 | 0.04→0.04 |
| 学习器 | 2.30→4.55 GiB | 0.64→0.56 | 0.10→0.10 | 0 | 1.12→1.11 |
| actor1 | 1.51→2.28 GiB | 0.55→0.56 | 0 | 2.21→2.20 | — |
| 3 个源进程合计 | 2.54→2.88 GiB | 1.25→1.36 | 0.07→0.07 | 0 | 0.11→0.21 |
| 编译进程 15 个 | 0.10→0.10 | 0.27→0.27 | — | — | — |

- 前 9 次的增长几乎全在 **glibc heap** 里：python 对象和小于 mmap 阈值的 numpy 数组，原始帧每行 22.9 KB 就落在这里。
- 锁页内存/CUDA 映射、编译缓存、/dev/shm 在这段时间都不变。
- 后段（第 9–20 次）源进程的增长出现在 `/dev/zero (deleted)` 这类共享匿名映射里，单个进程最多 1.33 GiB。这和 `--source-saved-cpu-gb 0.75` 的锁页 CPU spill 以及 CUDA 锁页缓存一致，cgroup 把这类映射记作 shmem。

### 12.3 按批准的规则实现（提交 83cf114、104176b）

1. **引用父局，不复制前缀**：
   - `RawHistory.origin` 记录分支的血统；源侧按"帧的所有者"设键，所以分支前缀用的就是父局那一份编码、那一张图。
   - 如果一个块里同时有父局帧和分支自己的帧，分支自己的帧交给这个块的 worker 编码，好让 Writer 在本地读到全部帧。
   - 在学习器上，用 `share_prefix` 把 pickle 带来的前缀拷贝换成对已有持有者数组的引用（`RecordLedger.share`）。
2. **消费后释放**：一次更新用过某条记录的损失以后，该记录自己的帧的 frontier 设为无穷，走 horizon 那套"先结算、再 detach"的路径释放图。以后再被采样时按冻结读取：没有源侧 cotangent，审计到的丢弃梯度恒为 0，Reader 权重照常学习。
3. **字节预算**：`--record-budget-gb 8`。超出时按最旧整条淘汰 v2 记录。progress.csv 新增 `record_bytes`、`record_evicted`、`record_shared_frames`、`record_v2_consumed`、`record_teach_consumed` 等列。
4. 哪些记录参与训练、分支内的读取、PPO 和教师的语义都不变。horizon 同样作用于分支历史：父局的 frontier 计入未消费记录的决策。

测试（CPU，均为退出码 0）：
- 新增 `tests/test_record_retention.py`：覆盖上面四条。
- `test_source_pipeline.py` 的分支检查改成"只编码分支自己的那 1 帧、前缀用父局的缓存值、自己那帧与当前权重的编码一致"。这是规则 1 带来的有意变化：以前分支前缀会用当前权重重新编码。
- `test_source_horizon`、`test_source_horizon_bound`、`test_source_lifetime`、`test_crosscycle_cache`、`test_source_process`、`test_stateless_teacher_batch`、`test_teacher_context_lifetime`、`test_source_streaming`、`test_source_async` 全部通过。
- H=0 与 v2 的逐位等价重新验证通过（本地和源进程两种模式，`results/h0-equivalence/v3r-*.pt`）。

### 12.4 验证运行（C 配置，代码 83cf114，`v3-83cf114-retC-10241124`）

- 跑了 18 次更新，被外部压力规则停掉：sl_dev 连续 3 个样本 34–36%，当时我们容器 66–67 GiB。退出码 137，只停了自己的容器，结束后已确认无残留。
- 规则在起作用：
  - 学习器上通过引用省掉的前缀帧累计 96,504 帧（约 2.1 GiB 的原始字节没有作为拷贝保留下来）；
  - 1,415 条教师记录在消费后释放了图；
  - 丢弃梯度一直为 0。
  - v2 记录到第 18 次更新只有 160 条，还没开始被采样，所以规则 2 对 v2 还没触发；字节预算也远没到（`record_bytes` 约 14 MB）。

| GiB per update | 新C旧保留 (9->17) | 新A (9->17) | 新C新保留 (9->17) |
|---|---|---|---|
| games: archive (parked room entries) | +0.383 | +0.353 | +0.405 |
| games: other (unmapped, episodes, queues, root, template) | +0.262 | +0.263 | +0.083 |
| python: trainer (= actor0) | +0.148 | -0.017 | +0.091 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.218 | -0.019 | +0.337 |
| python: other spawned (3 source workers, ...) | +0.284 | +0.147 | +0.312 |
| compile + other | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.296 | +0.727 | +1.228 |
| cgroup anon | +1.344 | +0.793 | +0.969 |
| cgroup shmem | +0.271 | +0.146 | +0.311 |
| coverage (Pss / anon+shmem) | 80% | 77% | 96% |
| archive procs | 166->294 | 172->289 | 159->292 |
| level at end (cgroup anon+shmem GiB) | 65.2 | 58.8 | 62.8 |


**结论：第 9 次以后的 python + shmem 斜率没有降到 A 组水平。**
- python 合计每次更新 +0.74 GiB（trainer +0.09、学习器/actor1/桥接 +0.34、源进程 +0.31）。旧保留策略是 +0.65，A 组是 +0.11。
- shmem +0.31，旧保留策略 +0.27，A 组 +0.15。
- 也就是说，在这个窗口里，C 和 A 的差别**主要不是**来自环形缓冲区里的记录：
  - 学习器 RssAnon 两次运行几乎一样（第 18 次更新 6.25 对 5.98 GiB）；
  - 增长出在 actor1/桥接、trainer 和源进程上。
  - 更可能的来源是 CF 分支运行本身：分支 lane 的帧（progress 的 `cf_frames` 到第 22 次更新累计约 98 万帧，每帧 22.9 KB）、分支搜索期间在 actor 端保留的上下文（`rebuild_contexts` 要等发布才释放），以及源进程为分支帧做的编码和锁页 spill。

**我自己的实现带来了一项额外开销，已修复**：ledger 把已经结束的祖先局的整段历史当作 donor 保留着，结果学习器原始字节在第 18 次更新是 1.23 GiB（旧保留策略同期 0.25 GiB）。104176b 改成只保留仍在世的历史作 donor（单测通过）。这次修复还没在主机上重跑。

**下一步需要定的**：要把 C 和 A 之间每次更新约 0.6 GiB 的差别定位到具体结构，需要在 actor 进程（sampler 的 `rebuild_contexts`、各 lane 的 ActorMemory）和源 worker 进程里也加计量；目前这些进程只有按映射类型分的 smaps，没有按结构分的明细。

## 13. 区分"活对象保留"和"分配器碎片"（代码 1fe66ab，含 104176b 的 donor 修正）

### 13.1 做法

两次 C 配置运行，代码相同，各跑 20 次更新，退出码都是 0，结束后已确认无残留容器：
- `v3-1fe66ab-acctC-10241140`：glibc malloc（默认）。
- `v3-1fe66ab-acctCje-10241149`：所有 python 进程 `LD_PRELOAD` 主机上的 jemalloc 5.3（`libjemalloc.so.2` 只需要 GLIBC ≤ 2.34，在容器里实测可以加载）。游戏进程通过 `ISAAC_GAME_ENV_STRIP=LD_PRELOAD` 不受影响。
  - 确实加载上了：jemalloc 那一组的 python 进程里**没有 [heap] 映射**，私有内存全部落在匿名 mmap（例如第 20 次更新 anon 19.9 GiB / heap 0；glibc 组是 anon 3.6 / heap 14.8 GiB）。

每次更新在学习器、actor0（= trainer）、actor1、3 个源 worker 里各写一行 `MEMORY_ACCOUNT`：
- 按结构的字节数；
- gc 活对象普查（RawHistory 等的个数、它们可达的去重原始字节）；
- actor 里另有 sampler `rebuild_contexts` 的条目数，以及 CF lane / bank ActorMemory 的原始字节；
- 用弱引用检查：上一次交接时 lane 槽里的历史，到这次还剩多少活着。

代价：gc 普查在 actor0 和学习器里每次约 4.5 s，会拉长这两次运行的墙钟。所以这两次运行**只用来看内存，不能当吞吐数据**。

### 13.2 各类内存的斜率（Pss，GiB/次更新，第 9→20 次）

| GiB per update | C旧保留 (9->20) | A (9->20) | C新保留_glibc (9->20) | C新保留_jemalloc (9->20) |
|---|---|---|---|---|
| games: archive (parked room entries) | +0.338 | +0.357 | +0.261 | +0.327 |
| games: other (unmapped, episodes, queues, root, template) | +0.384 | +0.272 | +0.362 | +0.415 |
| python: trainer (= actor0) | +0.207 | +0.053 | +0.096 | +0.135 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.168 | -0.016 | +0.044 | +0.151 |
| python: other spawned (3 source workers, ...) | +0.334 | +0.110 | +0.173 | +0.242 |
| compile + other | +0.000 | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.430 | +0.776 | +0.936 | +1.271 |
| cgroup anon | +1.067 | +0.686 | +0.795 | +0.904 |
| cgroup shmem | +0.322 | +0.110 | +0.166 | +0.225 |
| coverage (Pss / anon+shmem) | 103% | 97% | 97% | 112% |
| archive procs | 166->319 | 172->331 | 173->290 | 145->287 |
| level at end (cgroup anon+shmem GiB) | 67.6 | 60.1 | 64.9 | 65.8 |


### 13.3 活对象与 RSS（每个进程，第 9→20 次和第 1→9 次的斜率）

| per update, updates 9-20 | glibc | jemalloc |
|---|---|---|
| updates | 20 | 20 |
| anon | +0.807 | +0.861 |
| shmem | +0.210 | +0.157 |
| python Pss | +0.411 | +0.442 |
| games Pss | +0.530 | +0.599 |
| actor0 RssAnon | +0.230 | +0.154 |
| actor0 RawHistory raw | +0.163 | +0.113 |
| actor0 RawHistory objects | +1210 | +95 |
| actor0 lane refs alive | 17/32 | 24/32 |
| actor0 rebuild_contexts | 10379 | 8772 |
| actor1 RssAnon | +0.144 | +0.168 |
| actor1 RawHistory raw | +0.101 | +0.116 |
| actor1 RawHistory objects | +453 | -6 |
| actor1 lane refs alive | 20/30 | 21/32 |
| actor1 rebuild_contexts | 14580 | 12791 |
| learner RssAnon | +0.064 | +0.099 |
| learner RawHistory raw | +0.042 | +0.095 |
| learner RawHistory objects | +95 | +96 |
| source-cuda:0 RssAnon | -0.002 | +0.004 |
| source-cuda:0 RawHistory raw | +0.000 | +0.000 |
| source-cuda:0 RawHistory objects | +0 | +0 |

第 1→9 次：

| per update, updates 1-9 | glibc | jemalloc |
|---|---|---|
| updates | 20 | 20 |
| anon | +1.242 | +1.105 |
| shmem | +0.179 | +0.194 |
| python Pss | +0.714 | +0.694 |
| games Pss | +0.508 | +0.441 |
| actor0 RssAnon | +0.222 | +0.232 |
| actor0 RawHistory raw | +0.110 | +0.115 |
| actor0 RawHistory objects | +1096 | +1097 |
| actor0 lane refs alive | 0/0 | 0/0 |
| actor0 rebuild_contexts | 0 | 0 |
| actor1 RssAnon | +0.131 | +0.130 |
| actor1 RawHistory raw | +0.062 | +0.034 |
| actor1 RawHistory objects | +13 | +1 |
| actor1 lane refs alive | 0/0 | 0/0 |
| actor1 rebuild_contexts | 197 | 0 |
| learner RssAnon | +0.314 | +0.256 |
| learner RawHistory raw | +0.236 | +0.212 |
| learner RawHistory objects | +2290 | +2297 |
| source-cuda:0 RssAnon | +0.030 | +0.023 |
| source-cuda:0 RawHistory raw | +0.000 | +0.000 |
| source-cuda:0 RawHistory objects | +0 | +0 |

### 13.4 结论

1. **不是分配器碎片**：
   - jemalloc 没有让 RSS 斜率下降：python Pss 第 9–20 次是 +0.53，glibc 是 +0.31；第 1–9 次两边都约 +0.7。
   - 每个进程的 RssAnon 斜率和该进程**活着的** RawHistory 原始字节斜率是同一个量级：学习器第 1–9 次 RssAnon +0.31、活原始字节 +0.24；actor0 第 9–20 次 +0.23 对 +0.16。
   - 所以增长是活对象被保留，不是已释放的内存没还给系统。前 9 次的那部分也一样：学习器的活原始字节从 1.3 涨到 3.3 GiB，这是从 checkpoint 恢复后主局历史逐渐变长造成的，第 5 次以后基本走平。
2. **CF lane 的上下文会被释放，但在搜索期间占得不少**：
   - actor0 的 `rebuild_contexts` 在 CF 开始后（约第 12 次更新）涨到 1.3–3.2 万条，可达原始字节 0.4–1.1 GiB；lane ActorMemory 0.3–0.9 GiB。
   - 分支发布后会回落：jemalloc 那一组第 18 次交接时降到 120 条、0.01 GiB。
   - 弱引用检查显示，上一次交接时的 lane 历史有 17–28 / 29–32 个还活着。这符合分支跨多次交接运行（180 s）的情况，没有证据表明发布后仍被持有。
3. **新保留策略（加上 donor 修正）把后段的 python 斜率降了一半多，但还没到 A 组**：第 9–20 次 python 合计 +0.71（旧保留策略）→ +0.31（新，glibc）；A 组 +0.15。shmem +0.32 → +0.17；A 组 +0.11。但单次运行之间的差异大约有 ±0.3（§10），这个改善需要重复运行才能确认。
4. **剩下的差距**：主要在源进程（+0.17 对 A 组 +0.11）和 actor0 的 CF 预热（+0.10 对 +0.05）。从活对象数据看，它们是有上界的（会回落），不是无界增长。
5. **整体水位**：第 20 次更新时 anon+shmem 是 64.9 GiB（新保留策略，glibc），旧保留策略 67.6，A 组 60.1。游戏存档部分（两边都约 +0.3–0.4/次更新）要到 768 个才会停。

## 14. 规则 5：释放已编码、已消费帧的原始字节（代码 e5ce72e）

### 14.1 读者清单与决定

所有读 RawHistory 原始字节的位置，已在实现前列给发起会话：
- actor 侧：append 拼块、eager 路径的 `ActorMemory.refresh`、`pack_histories`、lane 上下文；
- 学习器侧：`unpack_histories`、`prefetch`、`SourceBatch`（只用长度）、`share_prefix`（只看身份）；
- 源进程：只在编码时用一次。

发起会话的决定：
- **D1**：eager / lane 槽的 refresh 改成和生产图路径一样，只刷新 Reader 投影，不再从原始字节整段重编码。这让 eager 路径和 v2 有意地不同，生产路径（TensorActorMemory）不受影响。
- **D2(a)**：有 lane 在途时，父局的源帧以 detached 值的形式留在流水线里；如果值和字节都没有，就抛异常并计数，绝不静默重编码。

### 14.2 实现

- 被释放的整块换成 `ReleasedRows`：只保留长度和 dtype，没有字节，任何读取都会抛异常。
- actor 在每次交接、增量包已发出后，释放学习器已经收到的帧。范围包括主槽、lane 槽、bank 槽、两份 rollout 的前缀和待发布的 lane 上下文。同时上报在途分支（以及本次交接所带记录）的祖先血统。
- 学习器在每次更新结束后，释放每局"最新决策之前"、已经编码并等待完成的帧（接收到的历史、v2 环形缓冲、donor）。
- 血统里的局和最近 16 次更新内结束的局（`--release-grace-updates`），都只保留 detached 值（frontier 设为无穷）。
- `prefetch` 只为流水线里没有的帧取字节。
- `unpack_histories` 只切分前缀结尾所在的那一块。

在主机上一共失败了三次，每次都是严格检查如实报错后再修：
- 8cae7ea：携带的 open 决策要切分一块已释放的块。改为只释放最新决策之前的帧。
- 37d8a6e：一条分支记录到达时，它的父局已经不在任何上报的持有者里。增加了 16 次更新的宽限期。
- aa3d84e：`share_prefix` 去读已释放 donor 块的行。改为整块共享占位、记录保留自己不到一块的行。另外 value 回退路径改成从缓存读。

测试：
- 新增 `test_raw_release`：各持有者的活原始字节有界（不随局长增长）；释放后的读取与从不释放的流水线逐位相同；携带决策的前缀能正常展开；血统能保住分支祖先的值；值和字节都没有时会抛异常并计数。
- `test_rebuild_memory_smoke` 里两处 eager 的对照按 D1 修改。
- 其余受影响的测试全部通过。
- H=0 与 v2 逐位相同（本地和源进程两种模式，`results/h0-equivalence/v3r5-*`）。
- CPU 全量（8cae7ea）：57 个测试、48 个通过，非零项与 v2 基线相同。

### 14.3 C 配置 24 次更新（`v3-e5ce72e-relC-10242237`，4 卡，H=2048，关掉 gc 普查）

- 正常结束，退出码 0，主机上无残留容器。累计释放 124 万帧，`source_released_misses` = 0。
- **活原始字节不再随局长增长**：
  - 学习器 0.4–1.1 GiB，RssAnon 平在 4.2 GiB（规则 5 之前同期是 5.5–6 GiB）；
  - 两个 actor 在 CF 开始前约 0.36 GiB，CF 运行时 0.8–1.9 GiB 来回波动，并有回落。
- **水位没有达标**：第 20–24 次更新 memory.current 是 62.3–64.8 GiB（anon+shmem 61.1–63.7），目标是 ≤ 58。

| GiB per update | 旧架构 (1->7) | C_规则1-4 (9->20) | C_规则5 (9->20) | A (9->20) |
|---|---|---|---|---|
| games: archive (parked room entries) | +0.268 | +0.261 | +0.363 | +0.357 |
| games: other (unmapped, episodes, queues, root, template) | +0.839 | +0.362 | +0.077 | +0.272 |
| python: trainer (= actor0) | -0.004 | +0.096 | +0.018 | +0.053 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.003 | +0.044 | +0.104 | -0.016 |
| python: other spawned (3 source workers, ...) | -0.017 | +0.173 | +0.195 | +0.110 |
| compile + other | +0.000 | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.089 | +0.936 | +0.757 | +0.776 |
| cgroup anon | +1.478 | +0.795 | +0.785 | +0.686 |
| cgroup shmem | +0.024 | +0.166 | +0.200 | +0.110 |
| coverage (Pss / anon+shmem) | 72% | 97% | 77% | 97% |
| archive procs | 215->278 | 173->290 | 173->318 | 172->331 |
| level at end (cgroup anon+shmem GiB) | 49.8 | 64.9 | 61.9 | 60.1 |


- **剩下的大头**：第 20 次更新时 python 进程合计 Pss 29.9 GiB，游戏进程 28.8 GiB。
  - 游戏存档还在填（映射到的 parked 进程 318 个），和旧架构的增长方式相同，会在 768 个时停住。
  - python 部分比旧架构的同类进程多约 16 GiB。主要是新架构才有的常驻进程：3 个源 worker（含 CUDA 上下文和锁页 spill）、更大的学习器和 actor、inductor 编译进程。原始历史已经不是主要原因。
- **吞吐只作参考**：这次运行开着轻量计量，第 11 次以后 x_real_time 在 187–273 之间，比之前没开计量的运行低。原因没有拆分，不能作为结论。

## 15. 吞吐组（进行中）与初步耗时拆分

### 15.1 两项语义中性的内存裁剪（代码 e5ce72e，4 卡，H=2048，计量同 relC，各 12 次更新）

| 第 12 次更新 | relC（无裁剪） | a）`--source-saved-cpu-gb 0` | a+b）再加 `TORCHINDUCTOR_COMPILE_THREADS=1` |
|---|---|---|---|
| memory.current | 59.4 GiB | 53.4 | 51.1 |
| python 合计 Pss | 31.0 | 26.1 | 23.8 |
| inductor 编译进程 | 15 个，1.99 GiB | 15 个，1.99 | 0 |
| 游戏 Pss | 26.6 | 27.0 | 27.0 |

两组都正常结束（退出码 0），12 次更新内没有 OOM（`results/v3-e5ce72e-trimA-10242256`、`…-trimAB-10242303`）。

但 a 不能用。带 a 的 4 组长跑（772addc）全部在源 worker 的源反向里 CUDA OOM，分别停在第 19、5、37、8 次更新（`results/v3-772addc-tp*-1024*`、`-10250227`）。所以吞吐组撤掉了 a，**只保留 b**（省 2.0 GiB）。

撤掉 a 后 4 组仍然 OOM（第 23、5、35、8 次），说明另有原因：
- **H=0 的原因是实现错误，已修复（d85fae3）**：H=0 时 `retain_histories` 从不 detach，规则 5 宽限期里已结束的局和在途分支的祖先保留的是整张图，而不是 D2a 规定的数值。修复后 H=0 与 v2 仍然逐位相同（`results/h0-equivalence/v3r6.log`）。
- **H=2048 的原因还在定位**：16 次更新的宽限期让源侧的活帧涨到 60–72 万（规则 5 前约 40 万），只放源 worker 的那张卡在 VJP 峰值时被撑满。

### 15.2 初步耗时拆分：5 卡、H=2048、25 次稳态（初步）

**这是初步结果**：单次运行 `results/v3-772addc-tp5h2048-10250252`，代码 772addc，只带裁剪 b，所有计量全关。运行在第 35 次更新因源卡 OOM 退出（退出码 1）。按发起会话的规定，取第 11–35 次（25 次稳态更新）作为结果。

布局：
- 旧架构 843.4× 是 3 卡（`temp/memory-chain-v2/baseline-*`，40 次稳态）；
- v2 370.2× 与本组同为 5 卡（v2 带模板里的学习器 cProfile，18 次稳态后退出码 137）；
- 本组：学习器 cuda:1，actor 在 cuda:0/2，源 worker 在 cuda:0/2/3/4。

所有数是中位数（括号内为四分位）。源侧的 VJP 每 2 个 rollout 做一次，所以更新耗时一次长一次短地交替，四分位差较大。

| 段（秒/次更新） | 旧架构 843× | v2 370× | v3 5 卡 H=2048（初步） | 列 |
|---|---|---|---|---|
| **墙钟** | 5.22 | 11.01 | **12.20**（11.42–14.89） | wall_s |
| x_real_time（按墙钟加权） | 843.4 | 370.2 | **337.0** | |
| **采集** | 5.26 | 10.75 | **10.95**（10.32–11.80） | collect_s |
| ├ actor 推理 | 2.28 | 8.92 | **9.19** | p_act |
| │ 单次推理（毫秒）/ 次数 | 5.92 / 2428 | 17.30 / 1879 | **17.12 / 2088** | infer_ms / calls_upd |
| └ 轮询、记账、回复、存储 | 3.34 | 2.74 | 2.79 | p_poll+p_book+p_reply+p_store |
| **等待学习器**（actor 侧） | 0.00 | 0.12 | **2.20**（0.50–4.88） | wait_learner_s |
| **学习器** | 4.56 | 7.57 | **9.46**（8.61–11.61） | update_s |
| ├ 准备 | — | — | 0.23 | u_rc_prep |
| ├ **等待源**（建 SourceBatch） | — | 含在 2.87 里 | **3.44** | u_rc_sources |
| │ ├ 等源编码 | — | 1.13 | 1.86 | host ready_encode_wait |
| │ ├ 等 Writer | — | 0.14 | 0.31 | host ready_writer_wait |
| │ └ 传输计划和数据（IPC） | — | 0.41 | 0.41 | host plan_transport |
| ├ 重算前向 | 0.90 | 2.87（含等源） | 0.71 | u_recompute |
| ├ PPO（含 Reader 前向与反传） | 2.93 | 2.59 | 3.04 | u_ppo |
| │ └ Reader 的 GPU 流时间 / host 提交 | — | 2.54 / 1.42 | 2.63 / 1.73 | reader_gpu / host_reader_submit |
| ├ 教师/v2 记录批 | 0.32 | 1.55 | 1.54 | u_teach_build+u_teach_new |
| └ gae、batch、opt、tail、temporal | 0.48 | 0.50 | 0.54 | |
| 源 worker 编码（最慢的一张卡，异步） | — | 1.92 | 2.40 | src_encode_max |
| 源 worker VJP（最慢的一张卡，异步） | — | 7.87 | 4.70（2.09–10.58） | src_source_backward_max |
| actor→学习器交接（IPC） | 0.01 | 0.76 | 0.65 | handoff_s |
| 显存峰值（MiB） | — | 23.0k | 24.1k（cuda:3 源卡，OOM） | telemetry |
| 第 35 次时的容器内存 | 55–57 平台 | — | 65.5 GiB（峰值 67.3） | memory.jsonl |

**差在哪一段（初步）**：每次更新的墙钟是 12.20 s，旧架构 5.22 s。843/337 = 2.5 倍。
1. **采集慢了 5.7 s，几乎全部是 actor 推理**。`p_act` 从 2.28 s 涨到 9.19 s，单次推理从 5.9 ms 涨到 17.1 ms（约 2.9 倍），而调用次数还少 14%。这一段 v2 和 v3 相同，是新架构 actor 侧（Reader 读记忆）的固定开销，与 horizon 和保留策略无关。
2. **学习器多了 4.9 s**：
   - 等源 3.4 s，主要是源 worker 先进先出，本 rollout 的编码排在上一轮的 VJP 后面；
   - 教师/v2 记录批多了 1.2 s；
   - PPO 和 Reader 与旧架构基本持平。
   学习器的中位数（9.46 s）比采集（10.95 s）短，但在做 VJP 的更新里会超过采集，于是 actor 中位数要等 2.2 s。
3. 所以要追上旧架构，**第一个瓶颈是 actor 推理**。只压学习器的话，墙钟最多降到采集时间附近（约 11 s，约 370×）。

完整列表：`results/prelim-tp5h2048-breakdown.md`（脚本 `scripts/throughput_table.py`）。

## 16. 源卡显存：每帧构成、各卡分布和四个选项（只加计量，代码 212388c）

数据：`v3-212388c-memC4-10250455`（H=2048，4 卡，25 次更新后源卡 OOM），`v3-212388c-memC4h1024-10250504`（H=1024，17 次更新后源卡 OOM），代码 212388c，只加计量。

### 16.1. 每帧构成（12 次编码，1544 帧，saved-tensor 统计）

| 部分 | KiB/帧 | 说明 |
|---|---|---|
| Spatial 4 层（整体编译，按形状拆） | **223** | 每层约 56 |
| ├ [T,192] bf16，每层 4 个 | 73.6 | LayerNorm 输出、残差、注意力输出等 |
| ├ [T,4,48] bf16，每层 2 个（与 qkv 共存储） | 73.6 | 注意力 q/k/v |
| └ [T,768] bf16，每层 1 个 | 73.6 | FFN 隐层（GELU 已按 pointwise 重算，只存一份） |
| entity MLP（fp32） | 4.6 | |
| patch / room / map CNN | 3.8 | |
| player、door、norm、gtok、reader.inp、物品 | 7.6 | |
| **保存张量合计** | **238** | |
| 编码调用实际留下的分配（含输出 y/u、autograd 元数据、分配取整） | **320–365** | 显存拆分实测，按这个估算 |

每帧约 18 个 token；宽度 192。

### 16.2. 三张源卡的带图帧数（学习器侧，每次更新结束时）

| | cuda:0（与 actor 共卡） | cuda:2（与 actor 共卡） | cuda:3（只跑源） | 合计 |
|---|---|---|---|---|
| H=2048 峰值 | 47.9k | 47.4k | **65.4k**（OOM 卡） | 160.7k |
| H=1024 峰值 | 50.9k | 52.3k | **67.5k**（OOM 卡） | 170.7k |
| 份额 | 约 29% | 约 29% | 约 42% | |
| 可放带图帧的显存（估） | 约 14.5 GiB（扣 actor 7 GiB 预留和数值、版本、缓存约 2 GiB） | 约 14.5 GiB | 约 21.5 GiB | 约 50 GiB |

调度已经大致按可用显存分配（cuda:3 约 42%，它的可用显存份额也约 42%）。OOM 发生在 cuda:3 的一次源反向里：已分配 20.2–21.2 GiB，OOM 前一次 retire 时 VJP 起点 20.1 GiB。

### 16.3. 四个选项（按 cuda:3 峰值 65–67k 帧、每帧 340 KiB 估算，现状约 22 GiB）

| 选项 | cuda:3 峰值估算 | 4 卡总需求估算 | 代价 | 语义 |
|---|---|---|---|---|
| 现状 | 约 22 GiB（OOM） | 52–55 GiB（容量约 50） | — | — |
| H=1024 | **实测没有下降**：峰值 67.5k 帧 / 20.8 GiB，第 17 次 OOM | 同现状 | 无额外计算 | 改训练语义（梯度只回传 1024 帧）。这个窗口里各局大多短于 1024 帧，带图帧主要来自待 VJP 的两个 rollout 加正在采的一个 rollout，而不是长局的历史 |
| 8 bit 保存激活 | 约 14 GiB（保存张量约 120 KiB/帧，另约 100 KiB 不变） | 约 35 GiB | 前向量化、反传反量化（访存型小 kernel，估计源卡 GPU 时间 +5–10%） | **近似**，不精确：反传用量化后的激活 |
| 按 Spatial 层 checkpoint，只存层输入 | 约 8.5 GiB（保存约 33 KiB/帧） | 约 22 GiB | 反传时重算 Spatial 4 层前向：每帧多一次 Spatial 前向，源卡 VJP 时间估计增加约一次编码的时间（目前每次更新编码约 2.4 s） | 精确，但**重算**，触及"不重算"的规则 |
| 按块在源卡间均衡 | 最多降到总需求的 42%，与现状相同 | 52–55 GiB 不变 | 无 | 精确，不重算。但 4 卡总容量约 50 GiB，小于需求，**单靠均衡解决不了**；5 卡布局（4 张源卡，约 72 GiB）在这个窗口里放得下，那组第 35 次 OOM 是因为后来需求继续增长 |

补充一项不在原清单里、但决定需求大小的因素：源侧每 2 个 rollout 才做一次 VJP（`--source-update-rollouts 2`），VJP 待提交期间不做 detach，所以显存里同时有 2–3 个 rollout 的图（每个约 3.3 万帧，约 11 GiB）。改成每个 rollout 一次 VJP，估计 4 卡总需求降约 11 GiB、cuda:3 降约 4.6 GiB。代价是源参数的 Adam 间隔变短，这改了训练语义。

## 17. actor 的 host 时间（A 线）与 CRN 修正

### 17.1 剖析结论（§15 之后的 actor 剖析，代码 61a2ea3 / d94f666，旧架构 oldprof-4/5/6）

- 剖析方法：torch.profiler 窗口（每进程 200 次调用）和不带 profiler 的纯计时窗口（每进程 1000 次调用，从进程第 9000 次主循环调用开始；`ISAAC_ACTOR_TORCH_PROFILE_MODE=timers`）。旧架构用主机工作区原代码，只加剖析补丁（`results/oldprof-code/`），整组跑 834–919×，与 843× 一致。
- **每次调用的 GPU 时间两边相同（约 2.8 ms）**，result 里等 GPU 约 0.01 ms：actor 是 host 受限，GPU 不是瓶颈。
- 纯计时口径：新架构单次延迟中位数 11.7–12.3 ms，旧架构 2.8 ms；submit 的 host 时间 3.4–3.7 ms 对 0.73 ms。线程 CPU 时间约等于 wall time（不是在等 GIL）；新架构 actor 进程只有 2 个线程。
- 两槽之间的空档（延迟 − submit）新 8 ms、旧 2 ms，主要是另一个槽的 submit；按调用的记账/存储/轮询/回复两边相差约 0.5 ms。粗算延迟 ≈ 2 × submit + 2.4 ms。
- Reader 在核心 CUDA 图内（torch.compile 后捕获），记忆库是 [零向量, 最近 W 帧, 按 capacity 补齐的记忆 token]，capacity 由该 actor 最长的一局决定，所有行都对整段带 mask 注意力；capacity 翻倍时所有核心图作废、下次调用按需重新捕获，记忆轴是动态维，不重编译。图外执行的有 Writer 收块（A2 之前）、换权重后的 K/V 重投影、tstate/tdest 写入（A1 之前）。窗口内 eager 回退 0 次，dynamo 无重编译。
- 图外 eager 重跑那一遍用来按模块拆 GPU 时间，钩子只覆盖 8–20%，且 eager kernel 时间是图内的 7–10 倍，这一拆分不可用；因 GPU 不是瓶颈，没有再追。

### 17.2 逐项（纯计时，每项单独部署，同一 harness；x_real_time 为 19 次更新的短测，波动大）

| 项 | 代码 | submit host（两个进程，ms） | 延迟中位数 | x_real_time | CUDA 门测 |
|---|---|---|---|---|---|
| 基线 | 805868e | 3.41 / 3.34 | 9.3 / 12.2 | 368.9 | — |
| A1 状态写入并进核心图、不再写全零 K/V | fe05db2 | 3.45 / 3.48 | 10.8 / 11.2 | 330.2 | 3/3 通过 |
| A2 Writer 收块：按收块数分桶的捕获图 | 3c100d5 | 3.00 / 9.84（该进程窗口内有图重捕获） | 11.1 / 8.2 | 365.3 | 3/3 通过 |
| A3 每次调用的行整体复制一次、历史行用视图（暂缓，已撤回：切片会留住整批数组） | 9cc883d | 2.85 / 2.95（RawHistory 段 0.62–0.64，未降） | 7.9 / 10.4 | （运行被中止，未收） | — |
| A4、A1b 单项 | 0c4b059 / 2601f32 | 未单独测：按发起会话的新顺序改为组合验证（§17.4） | | | |
| 旧架构 | oldprof-6 | 0.76 / 0.82 | 2.8 / 3.0 | 919.3 | — |

- A1：状态写入段 0.23 → 0.035 ms，但 tdest 段 0.69 → 0.83 ms，submit 不变。原因是 `torch.as_tensor(tdest['rows'], device=…)` 是一次同步的 pageable H2D 拷贝，host 在这里等核心回放跑完（CUDA 自旋等待，算作 CPU 时间）。A1b 修这一点。
- A2：Writer 收块段 0.54 → 0.014 ms。

### 17.3 CRN 修正（实验记录）

- **实验目的**：恢复旧架构的采样随机数行为。
- **实验方法**：v2 起（64ae451）RebuildGraphActor 与 eager 的 ActorMemory 对所有记录取 `crn_uniforms(ROW crn, t)`，crn=0 的普通记录也不例外：所有 worker 在同一决策步 t 上共用一套 Gumbel 噪声，64 个 worker 的探索相关。旧架构的 GraphActor 只对 crn≠0 的分支记录用 CRN，其余用新抽的随机数。修正提交 1b570e0（主线）/ 726c14a（计时链）：只有分支记录用 `crn_uniforms`，普通记录 `uniform_()` 新抽；lane（ukeys）不变。新测试 `tests/test_actor_crn_cuda.py`。
- **实验结果**：（门测与计时测量中）
- **影响范围**：这个提交改变采样结果，所以对 v2 的"H=0 逐位等价"从这个提交起只对确定性部分（logits、价值、记忆读取）成立，采样出的动作有意不同。**v2 的所有长测，以及 C99 之后用 v2 候选代码跑的任何训练，都带着这个相关采样**；C99 本身是旧架构，不受影响。

### 17.4 本轮测速基线与组合（纯计时 + 19 次更新短跑；每进程 1000 次调用）

**表 1：新测速基线**（805868e 加 CRN 修正 = afd843a；CUDA 门测 rebuild-actor、actorprof、actor-crn 3/3 通过）

| 进程 | 延迟中位数（ms） | submit host（ms） | tdest | RawHistory | Writer 收块 | 组件键 | 状态写入 | h2d/uniform | x_real_time 短跑 |
|---|---|---|---|---|---|---|---|---|---|
| 694 | 11.61 | 3.19 | 0.73 | 0.60 | 0.53 | 0.47 | 0.24 | 0.12 | 318.1 |
| 88 | 11.50 | 3.34 | 0.70 | 0.66 | 0.57 | 0.50 | 0.23 | 0.13 | |

CRN 修正后 h2d/uniform 段从 0.28–0.30 ms 降到 0.12–0.13 ms（普通记录不再在 host 上算 CRN）。

**表 2：组合 A1 + A2 + A4 + A1b + CRN，不含 A3**（14a7405；4/4 门测通过：rebuild-actor、actor-state、actorprof、actor-crn）

| 进程 | 延迟中位数（ms） | submit host（ms） | 核心回放 | RawHistory | 组件键 | tdest | Writer 收块 | 状态写入 | x_real_time 短跑 |
|---|---|---|---|---|---|---|---|---|---|
| 694 | **6.66** | 4.31 | 2.61 | 0.55 | 0.39 | 0.16 | 0.012 | 0.014 | 381.0 |
| 88 | **6.51** | 3.11 | 1.01 | 0.75 | 0.48 | 0.18 | 0.014 | 0.017 | |

- 延迟中位数 11.5–11.6 → 6.5–6.7 ms，接近 6.5 ms 的目标；短跑 318 → 381×。
- tdest（0.70–0.73 → 0.16–0.18）、Writer 收块（0.53–0.57 → 0.01）、状态写入（0.23 → 0.015）都降了。
- 但核心回放的 host 时间从 0.02 涨到 1.0–2.6 ms：之前藏在 tdest 那次同步拷贝里的等待，现在落到了 graph launch 上。也就是说 host 已经在等 GPU（两个槽、两路 actor 共用一张卡），actor 开始变成 GPU 受限，再省 host 时间对延迟的帮助会变小。
- 剩下的 host 大头：RawHistory 追加 0.55–0.75 ms（A3 的整批切片已撤回，它的开销在每行创建 Python 对象和拼 tuple，而不在拷贝字节）、组件键 0.39–0.48 ms。

## 18. 用户的 B 线决定：源 VJP 每个 rollout 一次（`--source-update-rollouts 1`）

### 18.1 第一次 5 卡长跑（代码 3b492fb = 组合 + 回收计数；`v3-3b492fb-tp5u1-10250702`）

- 第 31 次更新在 cuda:4 的源 VJP（start_finish）里 OOM，退出码 1。21 次稳态 315.7×。
- **回收被 pending 阻塞**：`retain_histories` 在有 pending VJP 时直接返回。延迟派发的 VJP（`defer_backward`：等下一个 rollout 的 Reader 输入就绪才派发）在每次 retain 时都还没提交，30 次更新里跳过了 23 次。真正释放只发生在第 86、92、98、02、06、09、12、15 次（间隔 4–6 次），两次释放之间带图帧数从 11.4 万涨到 24.5 万，然后 OOM。所以"VJP 完成"不等于"图被释放"，这里是回收没有机会运行，不是容量不够。

### 18.2 修复：VJP pending 时也释放（ee8a98c 主线 / 2b6bc9d 计时链）

- pending 时，挂在 pending 包里的块保持为图，这些包所属的局整局保留到下一次调用（它们的其他块可能和包里的块共享组件节点）；其余的照常退役或 detach。
- 新测试 `tests/test_source_pending_release.py`：同一序列下"pending 时释放"与"pending 时不释放"相比，结束且无 pending 梯度的局立即释放，有 pending 梯度的局晚一个 rollout 释放；参数在前 3 个区间逐位相同，第 4 个区间最大差 1.5e-8（一次 VJP 分组里的块集合变了，浮点求和顺序不同）。H=0 对 v2 的等价脚本仍逐位相同（`results/h0-equivalence/v3r7.log`）。
- 第二次长跑（da07983 = 2b6bc9d 加每卡带图帧数计量，`v3-da07983-tp5u1-10250721`）在第 2 次更新的源反向里断言失败（输出不是张量）：`_detach_horizon` 里原本就有一个局部变量 `pinned`（被活 Writer 引用的块下标），新参数同名被覆盖，pending VJP 的块没有被保护，horizon detach 把它们的输出放掉了。改名为 `vjp_pinned`（17ecfa5 主线 / 010a06b 计时链）；新测试 `tests/test_source_pending_horizon.py`（线程与进程两种模式）在去掉保护时失败、修复后通过。
- 第三次长跑结果见 §18.3。

### 18.3 表 3：5 卡、源每个 rollout 一次、H=2048（010a06b，`v3-010a06b-tp5u1-10250732`）

**没有跑满 40 次**：第 27 次更新时 cuda:3 源 OOM（guard 因 180 s 无进度停止，退出码 137）。按规定取第 11–27 次的 17 次稳态。

| 段（中位数） | 旧架构 843×（3 卡） | 5 卡、源每 2 个 rollout（§15，772addc） | **5 卡、源每 rollout（本组）** |
|---|---|---|---|
| 稳态更新次数 | 40 | 25 | 17 |
| x_real_time | 843.4 | 337.0 | **280.9** |
| 墙钟（s） | 5.22 | 12.20 | 15.75 |
| 采集（s） | 5.26 | 10.95 | **6.89** |
| 单次推理（ms） | 5.9 | 17.1 | **9.1** |
| 学习器（s） | 4.56 | 9.46 | **13.45** |
| actor 等学习器（s） | 0.00 | 2.20 | **8.34** |
| 学习器等 VJP 提交（s） | — | 0.00 | 3.46 |
| 等源（s） | — | 3.44 | 4.14 |
| 容器内存峰值 | 55–57 GiB 平台 | 67.3 GiB | 66.6 GiB |

- 采集侧显著变快（组合优化：采集 10.95 → 6.89 s，单次推理 17.1 → 9.1 ms），但学习器每个 rollout 都要等上一个 VJP 提交（3.5 s），学习器 13.45 s 成为瓶颈，actor 每次等 8.3 s，端到端反而从 337× 降到 281×。
- **回收已不再被阻塞**：每次更新都有释放（每次 3–7 万帧），`source_retain_pinned` 每次更新加 1（pending 时也在释放）。
- **带图帧的持有者**（学习器侧，每次更新结束时）：

| 更新 | 带图帧合计 | 单局最多 | 被活 Writer 钉住（超出 horizon） | 记录自有帧 | detached 值帧 |
|---|---|---|---|---|---|
| 第 3 次 | 96.9k | 1.9k | 0 | 0 | 18k |
| 第 9 次 | 146.7k | 4.7k | 24.0k | 0 | 174k |
| 第 18 次 | 194.2k | 5.0k | 22.0k | 0.2k | 425k |
| 第 27 次 | 174.9k | 4.1k | 0.3k | 0.35k | 462k |

  每张卡（第 27 次）：cuda:0 34.7k、cuda:2 35.7k、cuda:3 50.7k、cuda:4 53.8k。
- 结论：Writer 钉住的帧（≤2.4 万）和教师/v2 记录（<400 帧）都不是大头；剩下的是 64 个在途局 horizon 内的带图帧加正在采的一个 rollout，随局长逼近 2048 帧趋向 64 × 2048 + 3.3 万 ≈ 16.4 万帧以上，按每帧约 340 KiB 约 55 GiB，加上数值、版本、VJP 峰值，超过 4 张源卡的容量。**这是容量不足，不是回收被阻塞。** 显存没有进入平台（源卡分配峰值随更新上升到 20.7 GiB）。

### 18.4 Spatial 按层 checkpoint（隔离评估，`--source-layer-checkpoint 1`，默认关；代码 3a1c189 主线 / 7d9c227）

- 实现：4 个 Spatial 块各自编译，每块外套非重入的 `torch.utils.checkpoint`：只存块输入，VJP 时重算块内前向。整块编译的默认路径不变。
- CUDA 门测 `tests/test_source_layer_checkpoint_cuda.py`（编译、varlen flash、BF16，与长跑同设置，512 帧，一个区间）：源梯度相对误差 **9.3e-4**（单个张量最大 7.7e-3），一次 Adam 后参数最大差 6.4e-4（BF16 重算的浮点量级）。退出码 0。
- 隔离短跑（5 卡、源每 rollout、H=2048、21 次更新、11 次稳态；`v3-7d9c227-ckU1-10250804` 对 `v3-7d9c227-tlU1-10250755`）：

| | 完整图（tlU1） | 按层 checkpoint（ckU1） |
|---|---|---|
| 每帧编码后留下的显存（显存拆分实测） | 约 340 KiB | **56–58 KiB** |
| 每帧保存张量（saved-tensor 统计） | 238 KiB | **36 KiB**（其中 Spatial 20 KiB，是 4 个块的输入） |
| 源 VJP，最慢一张卡（中位数，四分位） | 6.06 s（5.34–6.99） | **6.16 s**（5.54–6.84） |
| 源编码，最慢一张卡 | 2.86 s | 2.99 s |
| 源卡峰值（只跑源的卡，nvidia-smi） | 20.9–21.3 GiB | **6.9–7.1 GiB** |
| cuda:3 带图帧 / 分配（第 18–21 次） | — | 45–53k 帧 / 4.4–4.8 GiB |
| x_real_time（11 次稳态，短跑） | 286.3 | 301.6 |

- 结论：在这个配置里，重算几乎不增加 VJP 时间（+0.1 s，在四分位内），每帧显存降到约 1/6，源卡峰值降到约 7 GiB。按发起会话的要求，checkpoint 不开进长跑。

### 18.5 学习器等 VJP 提交的 3.5 s 从哪来（时间线，`ISAAC_TIMELINE=1`；tlU1 11 次稳态，时间从该次更新学习器开始算）

| 时刻（中位数，四分位） | 完整图 tlU1 | checkpoint ckU1 |
|---|---|---|
| 本 rollout 的源就绪（rc_sources 结束） | 4.10（2.99–4.50） | 3.69 |
| 上一区间的 VJP 派发 | **6.56**（4.27–6.95） | 6.02 |
| VJP 最慢一张卡的时长 | 6.06（5.34–6.99） | 6.16 |
| VJP 结束（最后一张卡） | 11.67（10.81–13.79） | 12.49 |
| minibatch 全部结束 | 9.62（8.42–10.50） | 9.30 |
| 等提交（commit_wait） | **3.06**（2.30–4.05） | 2.89 |
| 学习器结束 | 13.52（11.56–14.33） | 12.61 |

- 原因：**VJP 启动晚**，不是 VJP 长、也不是提交串行。为了让新 rollout 的编码先于 VJP 跑（源 worker 先进先出），VJP 被推迟到第一个 minibatch 的 SourceBatch 建好之后才派发（`dispatch_finish`），而那时已经过了 rc_sources、重算、gae、教师/v2 记录批等，比源就绪晚 **2.47 s**（中位数）。VJP 本身 6 s，而派发后学习器只剩约 3 s 的 minibatch，所以要等约 3 s。源编码排在 VJP 后面的情况在这次时间线里没有出现：本 rollout 的编码在 VJP 派发前就已结束。
- **纯调度修法 1**：在常驻 SourceBatch 建好（rc_sources 结束）时就派发上一区间的 VJP，不等第一个 minibatch。记录批（教师/v2）随后的编码请求会排在 VJP 的各组之间（`ISAAC_SOURCE_PREEMPT_VJP=1` 下 worker 在每个 episode 组之间服务前向请求）。按本组数据估算：等提交 3.06 → **0.60 s**（四分位 0–1.27），学习器结束 13.5 → **10.5 s**；墙钟约为 max(采集约 8–9 s, 学习器 10.5 s)，x_real_time 约 286 × 15.2 / 11 ≈ **390–400×**（估算，未实测）。记录批的源可能会等一部分 VJP，u_teach_build 可能上升，需要实测。
- **纯调度修法 2**：在上一次更新结束时（`finish_rollout` 里）就立即派发，让 VJP 与下一轮采集重叠（采集约 6–9 s，足够容纳 6 s 的 VJP），学习器完全不等 VJP；代价是与采集期间的投机预取编码（SOURCE_PREFETCH）争用源卡，需用 preempt 服务保证编码延迟。收益上限与修法 1 相同（学习器约 10.5 s），但对 VJP 时长的波动更不敏感。
- 这两项都不改训练语义（同一区间、同一 Adam 时刻，只改派发时机）。

### 18.6 表 4：5 卡、源每 rollout 一次、H=1024（7d9c227，`v3-7d9c227-tp5u1h1024-10250814`，开时间线）

**没有跑满 40 次**：第 37 次更新 cuda:3 源 OOM（guard 180 s 无进度，退出码 137）。取第 11–37 次的 27 次稳态。

| 段（中位数） | 旧架构 843×（3 卡） | 表 3：H=2048（010a06b） | **表 4：H=1024** |
|---|---|---|---|
| 稳态更新次数 | 40 | 17 | 27 |
| x_real_time（按墙钟加权） | 843.4 | 280.9 | **304.7** |
| 墙钟（s） | 5.22 | 15.75 | 13.65 |
| 采集（s） | 5.26 | 6.89 | 7.47 |
| 学习器（s） | 4.56 | 13.45 | 12.46 |
| actor 等学习器（s） | 0.00 | 8.34 | 6.30 |
| 学习器等 VJP 提交（s） | — | 3.46 | 2.81 |
| 源 VJP，最慢一张卡（s） | — | 4.70 | 5.23 |
| 带图帧（稳态中位数 / 最大） | — | 15–19 万 | 18.3 万 / 20.4 万 |
| 被活 Writer 钉住（中位数 / 最大） | — | ≤2.4 万 | 2.1 万 / 4.3 万 |
| 单局带图帧最多（中位数 / 最大） | — | 4–5k | 3.0k / **6.9k** |
| 容器内存峰值 | 55–57 GiB 平台 | 66.6 GiB | 68.8 GiB（第 1/11/21/31/36 次：42.0/55.5/63.9/66.9/68.8） |

- H=1024 没有让带图帧下降：它们主要不是"在途局 horizon 内"的帧。单局带图帧最多到 6.9k，远超 1024 加一个 rollout。
- **原因：教师/v2 记录持图**。`RecordLedger.frontier` 给未消费记录的祖先局设 frontier = 记录长度，而 horizon 规则对每局只有一个下界（帧的末尾 < frontier − H 才 detach）。于是祖先局从"记录分叉点 − H"一直到当前 rollout 的所有帧都保持为图，包括记录最后一个决策之后、当前 rollout 的 horizon 之前那一段：这段帧没有任何待算的损失能回传到（记录只读到它自己的长度为止，rollout 的决策只读回 H 帧）。
- 时间线（表 4 同一组）：VJP 在源就绪后 2.36 s 才派发，等提交中位数 2.81 s；若源就绪即派发，估算等提交 0（四分位 0–0.41 s），学习器结束 12.46 → 9.72 s。

### 18.7 表 5：表 4 的配置加 `--source-dispatch-at-close 1`（419fce9；`v3-419fce9-tp5u1h1024d-10250833`）

门测：horizon-cuda、horizon-bound、attention-timing、actor-cuda、v2-cuda-suite2、rebuild-actor、actor-state、actor-crn 共 8 项全部通过。**没有跑满 40 次**：第 32 次更新 cuda:4 源 OOM（退出码 137），取第 11–32 次的 22 次稳态。

| 段（中位数） | 旧架构 843× | 表 4：H=1024 | **表 5：H=1024 + 收尾即派发** |
|---|---|---|---|
| 稳态更新次数 | 40 | 27 | 22 |
| x_real_time（按墙钟加权） | 843.4 | 304.7 | **345.0** |
| 墙钟（s） | 5.22 | 13.65 | 12.76 |
| 采集（s） | 5.26 | 7.47 | 6.99 |
| 学习器（s） | 4.56 | 12.46 | **10.81** |
| 学习器等 VJP 提交（s） | — | 2.81 | **0.00** |
| actor 等学习器（s） | 0.00 | 6.30 | 5.92 |
| 记录批（u_teach_build，s） | 0.28 | 1.19 | 2.27 |
| 等源编码（ready_encode_wait，s） | — | 1.91 | 1.75 |
| 源 VJP 最慢一张卡（s） | — | 5.23 | 5.96 |
| 容器内存峰值 | 55–57 平台 | 68.8 GiB | 68.4 GiB |

- 时间线：VJP 在本次更新学习器开始前 1.6 s（即上一次更新收尾时）派发，4.3 s 结束，早于本次的 minibatch 结束（10.5 s），等提交为 0。
- 代价：记录批（教师/v2 记录的源）从 1.19 升到 2.27 s，因为这些编码请求要插在仍在跑的 VJP 各组之间；采集侧的预取编码没有变慢（等源编码 1.91 → 1.75 s）。
- 净效果：学习器 −1.65 s，吞吐 304.7 → 345.0×（+13%）。显存行为不变（仍 OOM，原因见 §18.8）。

### 18.8 带图帧被谁持有（计量重跑，表 4 的配置，b657026；`v3-b657026-tp5u1h1024-10250847`）

这组在第 28 次更新 OOM（表 4 那组是第 37 次），所以给出第 10/15/20/25/28 次的数（每帧字节在各类之间相同，字节占比 = 帧数占比，按每帧约 340 KiB 折算）：

| 更新 | 带图帧合计 | 在途 rollout 的 horizon 内 | **未消费记录 / 在途分支提前的 frontier** | pending VJP 的包 | 活 Writer | 未消费 v2 记录（条） | 其中最老的已等（次） |
|---|---|---|---|---|---|---|---|
| 10 | 140k（46 GiB） | 84.1k（60%） | 0.5k（0%） | 17.9k（13%） | 28.8k（21%） | 0 | 0 |
| 15 | 148k | 84.7k（57%） | 24.0k（16%） | 12.9k（9%） | 25.9k（18%） | 152 | 5 |
| 20 | 170k | 84.6k（50%） | 34.9k（20%） | 11.4k（7%） | 14.8k（9%） | 256 | 10 |
| 25 | 181k | 82.7k（46%） | 52.3k（29%） | 13.6k（8%） | 15.4k（9%） | 336 | 15 |
| 28 | 196k（64 GiB） | 83.7k（43%） | **61.4k（31%，约 20 GiB）** | 14.3k（7%） | 20.2k（10%） | 408 | 18 |

- 在途 rollout 的部分稳定在约 8.4 万帧；**增长全部来自未消费记录**：v2 记录从第 12 次左右开始到达，到第 28 次一条都没有被消费（`record_v2_consumed_new` 一直为 0）。
- 为什么不被消费：采样要同时满足"有效 v2 记录 ≥ 256 条"（约第 20 次更新达到）和"当次 v2 步数 ≥ 8"（`train_tok.py` 3421 行：`v2_step >= 8`）；这几组里 v2 步数只有 2–7，第二个条件很少满足。表 4 那组到第 31 次才第一次消费（110 条），到第 37 次 608 条里仍只消费了 110 条。
- 教师记录消费及时：首次使用等待中位数 0–2 次更新，p90 2–6 次。

### 18.9 三个候选规则（按 §18.8 第 28 次更新的数估算；4 张源卡，最忙的卡约占 30%）

| 规则 | 第 28 次的带图帧估算 | 显存（合计 / 最忙的卡，带图部分） | 上界 | 语义影响 |
|---|---|---|---|---|
| 现状 | 196k | 约 64 GiB / 约 19 GiB（OOM） | 随未消费记录数无界增长 | — |
| a）记录对祖先帧只读 detached 值，只有分支自己的帧保留图 | 约 135k（记录部分 → 自有帧，<1k） | 约 44 GiB / 约 13 GiB | 在途 rollout + VJP 包 + Writer，与记录数无关 | 教师/v2 记录的损失不再训练祖先帧的编码和 Writer；仍训练 Reader 和分支自己的帧 |
| b）K 次更新内消费则祖先帧保留图，超过 K 次转为值（K = 4） | 约 150k（只剩最近 4 次更新进入的记录，约 100 条 × 约 150 帧 ≈ 1.5 万） | 约 49 GiB / 约 15 GiB | 由 K × 每次进入的记录数决定 | 按现在的消费等待（v2 记录 ≥18 次仍未消费，教师记录 p90 ≤ 6），**v2 记录实际上全部失去对祖先的梯度**（等同 a 对 v2）；教师记录大多保留 |
| c）降低 v2 攒批门槛（例如 256 → 64）和/或"v2 步数 ≥ 8"的条件，让 v2 记录更快被消费 | 若 2–4 次更新内消费：约 150k | 约 49 GiB / 约 15 GiB | 取决于新的消费等待分布 | 改 v2 教师的采样分布（更早、更小的候选池；可能更频繁地采样） |

- a 的上界最清楚，与记录积压无关；b 在 v2 当前的消费节奏下效果与 a 接近，但对消费及时的教师记录保留梯度；c 不改保留规则，改的是 v2 的采样。
- 三者都不涉及重算。b 和 c 的估算依赖于假设的消费等待，需要实测确认。

### 18.10 表 5 学习器拆分与纯工程候选（只出清单，不实现）

**学习器 10.81 s 的组成**（表 5，22 次稳态中位数；采集 6.99 s；目标：学习器 ≤ 采集）

| 段 | 秒 | 其中 |
|---|---|---|
| 准备（rc_prep） | 0.19 | |
| 等源（rc_sources） | 3.14 | 等源编码 1.75，等 Writer 0.46，传输计划 0.39，SourceBatch 组装约 0.5 |
| 重算前向（recompute） | 0.80 | |
| gae、batch、teach_new | 0.39 | |
| 记录批（teach_build） | 2.27 | 教师/v2 记录的源编码与组装（表 4 是 1.19） |
| PPO（含 Reader 前向与反传） | 3.41 | Reader host 提交 1.66，Reader GPU 流 2.69（与 host 部分重叠） |
| opt、tail、temporal | 0.39 | |

**候选（都不改训练语义）**

| # | 候选 | 估算节省 | 风险 |
|---|---|---|---|
| 1 | 记录批的编码请求与 VJP 分流：源 worker 里 VJP 放到独立线程和独立的低优先级 CUDA stream（autograd 在 C++ 里释放 GIL），前向请求在主线程、高优先级 stream 上并发执行，而不是只在 VJP 的 episode 组之间插队 | 记录批 2.27 → 约 1.2 s（回到表 4 的水平），等源编码也可能 −0.3 s：合计约 −1.1 到 −1.4 s | 中：同一张卡上前向与反传并发会增加显存峰值（两套工作区）；共享组件节点被并发读（只读，应安全）；CUDA IPC 事件的顺序需要重新核对；需门测（逐位等价）和显存测量 |
| 2 | 等源编码 1.75 s：把 actor 的源预取阈值从 8192 条记录降到 4096（或按时间触发），让更多本 rollout 的帧在采集期间就编码完 | 约 −0.8 s（未预取的尾部减半） | 低到中：actor 侧多一次 `pack_histories` 和 IPC（p_store 增加）；预取编码与收尾即派发的 VJP 在采集期间争用源卡；版本不变（提交被闸门拦到同一点） |
| 3 | Reader host 提交 1.66 s：每个 minibatch 在 host 上重建分组布局、query 索引和 credit 计划；改为每次更新只为整个 rollout 建一次（minibatch 只做切片），并把索引构建向量化 | 约 −0.6 到 −1.0 s | 低：纯索引代码，需逐位等价测试；CUDA Graph 不适用（每个 minibatch 形状不同） |
| 4 | recompute 0.80 s：`--decoupled 1` 需要用本次更新开始时的权重重算旧策略的 logp 和价值，这是 PPO 目标的一部分，**不能去掉**；旧架构也有（0.90 s） | 0（可与等源部分重叠的余地很小，它依赖源值） | — |
| 5 | actor 等学习器 5.9 s：用 B38 式双 actor 让采集继续进行下一轮 | — | **改语义**：actor 会用更旧的权重多采一轮（staleness +1），不属于纯工程，不列入 |
| 6 | SourceBatch 组装和传输计划（约 0.9 s）：沿用上一次更新的规划结果、减少 host 拷贝 | 约 −0.3 s | 低 |

合计（1+2+3+6）估算约 −2.8 到 −3.5 s，学习器约 7.3–8.0 s，接近采集的 7.0 s；按墙钟约为 max(采集, 学习器) 估算，x_real_time 约 345 × 12.76 / 8.5 ≈ 500×（估算，未实测）。

### 18.11 `v2_step >= 8` 的审查说明（只说明，不改代码）

- 位置：`train_tok.py` 3112 行与 3421 行。
- 含义：`v2_step` 是**每个 PPO minibatch 里分到的 v2 样本数**，不是训练步数：`v2_step = int(min(teacher_v2_batch, teacher_v2_reuse × v2_rate / planned_v))`，其中 `v2_rate` 是每次更新新到 v2 记录数的指数滑动平均（0.9/0.1），`planned_v = epochs × (rollout 决策数 / minibatch) = 3 × (32768 / 8192) = 12`，`teacher_v2_reuse = 4`，`teacher_v2_batch = 256`。
- 采样条件（3421 行）：`v2_step >= 8` 且有效 v2 记录 ≥ 256 条，才在该 minibatch 里取 `v2_step` 条 v2 记录算模仿损失。
- 推论：`v2_step >= 8` 等价于 `v2_rate >= 8 × 12 / 4 = 24` 条/次更新。这几组从 checkpoint 恢复后的窗口里，每次更新新到的 v2 记录约 8–40 条（表 4：第 13–37 次多为 8–40，滑动平均多在 16–24），`v2_step` 只有 2–8，所以条件几乎不满足；记录持续积压、几乎不被使用（表 4 一共消费 110 条，其余各组 0 条）。
- 这个门槛的作用看起来是"每个 minibatch 的 v2 样本太少时不算 v2 损失"；它与 `teacher_v2_reuse`（期望每条记录被用约 4 次）在低到达率时相互矛盾：记录到得慢时，一条都不会被用。是否调整属于 v2 教师的训练语义，本轮不改。

### 18.12 关于对照基线

旧架构 843× 那组同样从未算过 v2 损失（50 次更新里 `v2_step` 最大为 5）。因此表 3–5 与 843× 的对照**只代表"v2 教师实际停用"工况下的速度**（表 6 的 40 次稳态里有 1 次、表 7 有 9 次算了 v2 loss，见 §18.15）；v2 教师开始工作后（到达率更高、每个 minibatch 多出 v2 样本与其源读取），两种架构的学习器开销都会增加，不能用本轮数字外推。生产版本继续不替换。

### 18.13 保留范围修复（选项 d，32263d4 主线 / 208911e 计时链）

- `RecordLedger.frontier` 原来把一个局的所有消费者压成一个"最早待算决策"，只能表达下界：未消费记录的分叉点之后到当前 rollout horizon 之前的帧也被保留。现在按每个消费者真正可达的帧区间取并集：rollout 是 [F − H, ∞)；未消费记录（决策在 L、分叉点 f）的自有帧是 [max(f, L − H), L)，每个祖先只取它共享的那部分 [L − H, 共享帧数)（分叉点比 L 早超过 H 时为空）。并集之外的块先结算已有梯度、再 detach（保留数值）；pending VJP 的包、活 Writer、共享图依赖的保护不变；沿用现有 chunk 粒度（块带上首帧、Writer 带上首块起点）。
- 例（发起会话给出）：主局当前决策 6000、H=1024；记录决策 1000、分叉点 900：需要的祖先范围 [0, 900) ∪ [4976, ∞)，中间 4076 帧现在会被释放。
- 测试 `tests/test_source_reach_intervals.py`（线程与进程两种模式）：记录在进入后第 5 个 rollout 才被消费，参数与"所有图全保留"的参照**逐位相同**（最大差 0.0），说明延迟消费的记录对祖先 Encoder/Writer 的梯度与参照一致；等待期间两段之间的块被释放（持有 16 帧，参照 56 帧），消费后记录的范围也被释放；对照组（祖先范围去掉、只读数值）从消费那一步起参数不同，说明这份梯度确实存在并被保留。原有相关测试全过；H=0 对 v2 的确定性部分仍逐位相同（`results/h0-equivalence/v3r8.log`）。
- 同时修了一个被它暴露的旧边界：某个源 worker 的图全部被释放后可能不再持有任何权重版本，之后发给它的空 VJP 请求会因为找不到副本而报错；现在没有该 worker 的包时直接返回空梯度。
- 持有原因统计改为多标签、与总数同一时点计数（`source_hold_rollout/record/vjp/writer/total/none/multi`）。
- 没有改任何采样阈值。

### 18.14 表 6：表 5 的配置加 `--source-layer-checkpoint 1`（b657026；`v3-b657026-t6-10250950`）

**第一组跑满 40 次稳态的 5 卡运行**：52 次更新，退出码 0，guard 未触发。取第 11–50 次的 40 次稳态。

| 段（中位数） | 旧架构 843×（3 卡） | 表 5（22 次稳态，第 32 次 OOM） | **表 6（40 次稳态）** |
|---|---|---|---|
| x_real_time（按墙钟加权） | 843.4 | 345.0 | **334.0** |
| x_real_time 中位数（四分位） | 837.8（809–886） | 342.8（331–360） | 342.0（313–359） |
| 墙钟（s） | 5.22 | 12.76 | 12.78 |
| 采集（s） | 5.26 | 6.99 | 7.34 |
| 学习器（s） | 4.56 | 10.81 | 10.94 |
| 学习器等 VJP 提交（s） | — | 0.00 | 0.00 |
| 源 VJP，最慢一张卡（s，四分位） | — | 5.96（4.96–7.07） | **8.18**（7.52–9.56） |
| 源编码，最慢一张卡（s） | — | 2.81 | 2.94 |
| 源卡峰值（nvidia-smi，只跑源的两张） | 约 24 GiB（OOM） | — | 9.0 / 9.1 GiB（与 actor 共卡的 11.8 / 12.1 GiB） |
| 源侧分配（最大一张卡，第 10/20/30/40/52 次） | — | — | 2.5 / 3.6 / 4.4 / 5.1 / 6.3 GiB |
| 第 40 次的容器内存 / 峰值 | 55–57 GiB 平台 | 66.5 / 68.4 GiB | 66.3 / 66.7 GiB（第 36–51 次在 65.2–66.1 之间） |

- **checkpoint 的长测代价**：源 VJP 比表 5 长 2.2 s（短测里只多 0.1 s：那时局短、每次 VJP 涉及的帧少）。因为 VJP 已在收尾时派发、与下一轮采集重叠，这 2.2 s 没有进入学习器的关键路径（等提交仍为 0），端到端吞吐与表 5 持平（334 对 345，在表 5 只有 22 次稳态的波动范围内）。时间线：VJP 在本次更新开始前 1.7 s 派发，6.8 s 结束；若 VJP 再长约 4 s 就会开始让学习器等待。
- **显存**：源卡不再 OOM，但带图帧仍在上涨：98k（第 10 次）→ 158k（第 52 次），其中"被未消费记录的可达范围持有"从 0.5k 涨到 76.6k，rollout horizon 内的稳定在约 9.2 万。表 7（保留范围修复）用来看这部分是否回落、显存是否进入平台。

### 18.15 表 7：表 6 的配置加保留范围修复（208911e；`v3-208911e-t7-10252238`）

门测：horizon-cuda、horizon-bound、attention-timing、actor-cuda、v2-cuda-suite2、rebuild-actor、actor-state、actor-crn、layer-ckpt 共 9 项全部通过。**跑满 40 次稳态**：52 次更新，退出码 0，guard 未触发。

| 段（中位数） | 旧架构 843×（3 卡） | 表 6（40 次稳态） | **表 7（40 次稳态）** |
|---|---|---|---|
| x_real_time（按墙钟加权） | 843.4 | 334.0 | **320.8** |
| x_real_time 中位数（四分位） | 837.8（809–886） | 342.0（313–359） | 319.6（295–363） |
| 墙钟（s） | 5.22 | 12.78 | 13.67 |
| 采集（s） | 5.26 | 7.34 | 7.53 |
| 学习器（s） | 4.56 | 10.94 | 11.42 |
| 学习器等 VJP 提交（s） | — | 0.00 | 0.00 |
| 等源（rc_sources，s） | — | 3.41 | 3.79 |
| 源 VJP，最慢一张卡（s） | — | 8.18 | 8.89 |
| 源卡峰值（只跑源的两张 / 与 actor 共卡） | OOM | 9.0–9.1 / 11.8–12.1 GiB | 8.9 / 10.3–10.7 GiB |
| 第 40 次的容器内存 / 峰值 | 55–57 平台 | 66.3 / 66.7 GiB | 65.1 / 66.8 GiB |

**带图帧（多标签，与总数同一时点；第 20/30/37 次为发起会话要的点，另加第 40/52 次）**

| 更新 | 表 7 带图帧 | rollout horizon 内 | 未消费记录的可达范围 | pending VJP | 活 Writer | 无原因 | 多原因 | 源侧分配（最大卡） | 表 6 带图帧 / 分配 | 未消费 v2 记录（条）/ 最老已等（次） |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 112.2k | 89.2k | 11.9k | 0 | 102.0k | 0.1k | 99.1k | 3.7 GiB | 118.1k / 3.6 GiB | 240 / 11 |
| 30 | 134.5k | 102.4k | 22.3k | 0 | 130.7k | 0.2k | 121.1k | 5.0 GiB | 135.6k / 4.4 GiB | 432 / 21 |
| 37 | 127.3k | 95.5k | 27.5k | 0 | 123.4k | 0.1k | 118.6k | 5.0 GiB | 144.2k / 5.1 GiB | 430 / 28 |
| 40 | 125.5k | 92.1k | 30.9k | 0 | 121.0k | 0.1k | 118.5k | 5.3 GiB | 146.0k / 5.1 GiB | 334 / 31（本次第一次消费 69 条） |
| 52 | 120.6k | 79.6k | 39.1k | 0 | 116.6k | 0 | 114.7k | 5.5 GiB | 157.7k / 6.3 GiB | 355 / 43 |

（"活 Writer" 几乎覆盖所有带图帧，因为 rollout horizon 内的 Writer 块图包含其所有帧的编码图；各原因可重叠，总和大于带图帧数。统计时点与总数一致：第 20 次 112.2k 与总数之差 8.2k 是那一次 `source_graph_frames` 在本次更新后续又编码了记录帧。）

- **梯度审计未通过**：`source_horizon_dropped_grad` 表 7 第 4 次首次非零、累计 31.55（表 6 第 39 次首次非零、累计 0.04076）。被丢弃的梯度说明有消费者仍会回传的帧被提前释放了，因此下面的显存结论只是容量层面的，不是正确的保留。
- **容量**：表 6 的带图帧一路涨到 157.7k，表 7 在第 30 次以后回落并稳定在 12–13.5 万；记录持有的部分仍随积压的 v2 记录缓慢上升（11.9k → 39.1k），但只保留它们真正可达的区间，第 52 次比表 6 少约 3.7 万帧。由于 checkpoint 已让每帧只占约 57 KiB，源侧显存本来就小（5.5 GiB 对 6.3 GiB），显存的改善主要在"不随记录积压无界增长"。
- **吞吐**：320.8× 对表 6 的 334.0×，低约 4%（两组都是 40 次稳态，四分位重叠，但这不足以判为噪声：**未分辨**）。学习器多 0.5 s，等源多 0.4 s、源 VJP 多 0.7 s，可能来自释放更多块时多出的结算反传（horizon_backward），未单独确认。
- **v2 loss**：表 7 的 40 次稳态里有 9 次更新算了 v2 loss，表 6 只有 1 次；两组 v2 教师的参与程度不同，不能写成"都没起作用"。
- **容器内存**：两组都在 64–67 GiB，没有回到旧架构的 55–57 GiB 平台；这部分主要是游戏存档和常驻 python 进程（§14），与源侧图无关。

### 18.16 梯度审计（`source_horizon_dropped_grad`，每张表必报；验收 = 恰好 0）

| 运行 | 代码 | 首次非零的更新 | 累计 | 算了 v2 loss 的更新数（全部 / 40 次稳态内） |
|---|---|---|---|---|
| relC（§14） | e5ce72e | — | 0 | 0 / — |
| §15 5 卡 H=2048 | 772addc | — | 0 | 0 / 0 |
| 表 3 | 010a06b | — | 0 | 0 / 0 |
| 表 4 | 7d9c227 | 30 | 0.00963 | 1 / 1 |
| 表 5 | 419fce9 | — | 0 | 0 / 0 |
| 表 6 | b657026 | 39 | 0.04076 | 1 / 1 |
| 表 7 | 208911e | **4** | **31.55** | 9 / 9 |
| 验收 1（只修原因 1） | 1727f71 | — | 0 | 0 / 0 |
| 验收 2（+原因 2） | 187ede3 | 29 | 0.05467 | 10 / 10 |
| 测量 t7d | 019dda7 | — | 非零（首次在版本 38） | 见 §18.17 |
| 测量 t7d 2 | 9086793 | — | 非零（首次在版本 21） | 见 §18.17 |
| **验收 3（+原因 3）** | **258f97a** | — | **0** | **7 / 7** |
| **验收 4（加强审计）** | **d345da5** | — | **0** | **6 / 6** |

- 表 4、表 6 的首次非零都出现在第一次算 v2 loss 的那次更新：旧保留规则下，被消费的 v2 记录有一部分可达帧已被释放。
- 表 7 在第 4 次（任何 v2 loss 之前）就非零，且累计大三个数量级：新的区间规则自己漏了需要的帧（定位见下一节）。

### 18.17 第一个非零丢弃梯度的定位与修复

- **审计上下文**：`_drop()` 现在在第一次非零时打印一行 `SOURCE_DROP_FIRST`（字段 y/u/mem、被释放的块或 Writer 块的 uid 与帧范围、释放时的可达区间、编码版本与释放时版本、读取它的那个 batch 里相关的请求及其长度、分叉点、冻结数），只打印一次，不改任何行为。
- **原因 1（已复现、已修，fc0ad4f 主线 / 1727f71 计时链）**：发起会话给出的线索成立。未消费记录决策在 L，它读取自己的当前观测（帧 L，随记录一起编码）；选项 d 给它自有帧的区间是 [max(分叉点, L − H), L)，不含 L。当帧 L 单独成一个块、又没有活 Writer 覆盖时（同一分支上连续两条记录，后一条的 query 恰好开启一个新的记忆块），它会被释放，记录之后被消费时这帧的 u/y 余切就被丢弃。新测试 `tests/test_source_record_query.py`（线程与进程两种模式）复现了它：修复前丢弃 0.0177，修复后恰好 0，参数与全保留逐位相同。自有区间改为 [max(分叉点, L − H), L + 1)。
- **验收重跑 1（1727f71，只修原因 1；`v3-1727f71-t7-10260033`）**：9 个门全过；52 次更新、无 OOM、`source_horizon_dropped_grad` 全程恰好 0，`SOURCE_DROP_FIRST` 没有出现；稳态 324.3x。但这次 52 次更新里**没有一次算了 v2 loss**（`v2_loss_pi` 全为 0；表 7 有 9 次），所以原因 2 的路径在主机上没有被走到，这次的 0 不能当作原因 2 不存在的证据。
- **原因 2（已用失败用例复现、已修，c8f356f 主线 / 187ede3 计时链）**：在途分支（lane、选择点 bank、待发布的上下文）的祖先按规则 5 的 D2a 只保留数值。分支以后发布的记录决策在 L ≥ 分支当前长度 ℓ，要带梯度读祖先的帧 [L − H, 共享帧数)；若这段图在记录到达前已因"只保留数值"被释放，记录被消费时这段余切就被丢弃。新测试 `tests/test_source_inflight_branch.py`（线程与进程两种模式）：按旧规则（在途祖先无区间）丢弃 2.05e-06、参数与全保留不同；按新规则丢弃恰好 0，每次更新参数与全保留相同（max diff 0.0），A 持有的带图帧 [0, 8, 8, 16, 16, 16, 16, 8]，全保留 [0, 8, …, 56]。
- **修法**：actor 交接时报告每个在途分支对每个祖先的（祖先 uid、共享帧数、分支当前长度 ℓ）（`record_retention.lineage_intervals`）；learner 把 [ℓ − H, 共享帧数) 并入该祖先的可达区间（`in_flight_frontier`），H = 0 时为 [0, 共享帧数)。这改变了 D2a：在途分支的祖先从"只保留数值"变为"保留这段可达帧的图"，显存会多占这一段（H=1024 时每个祖先至多 1024 帧）。已结束局的宽限期仍只保留数值（不变）；若宽限期内有未报告的分支以后发布记录，审计会再次非零，`SOURCE_DROP_FIRST` 会给出来源。CPU 回归 67 个测试与上一轮相同（13 个需要 CUDA 的本机失败，与之前同一批），H=0 等价逐位相同（v3r7，本地与进程两种模式）。
- **验收重跑 2（187ede3，表 7 配置；`v3-187ede3-t7-10260108`）：未通过。** 9 个门全过；52 次更新、40 次稳态、无 OOM、退出码 0、稳态 304.1x（同配置 1727f71 为 322.9x，这次差约 6%，原因未分辨）。这次有 10 次更新算了 v2 loss（第一次在第 29 行，update 48214），`source_horizon_dropped_grad` 从这一次起非零，累计 0.05467，与表 4、表 6 同一量级。注：`run_remote_all.py deploy` 之前不传提交号，部署的是主线 HEAD c8f356f；它与 187ede3 的 `bridge/python` 树逐字节相同（`git diff --quiet` 为真），所以结果按 187ede3 记；脚本已改为部署标签所指的提交。
- **这次的 `SOURCE_DROP_FIRST`**：字段 u；被释放的是局 (694, 919) 的一个编码块，帧 1472–1684，在版本 15 编码、版本 20 释放，释放时它的可达区间只有本 rollout 的 [1693, ∞)；读它的是 v2 记录 (694, 1192)，决策在 L = 1816，自有帧从 1809 起，祖先链 (694, 1025)@1809 → (694, 919)@1809，读取发生在版本 29。也就是说在版本 20，learner 既没有这条记录（ledger 里没有），也没有任何在途报告覆盖 919 的 [792, 1809)。现有的在途来源（lane、选择点 bank、本次交接的记录、待发布上下文）在这段时间里有一处没报告到它。具体是哪一处还没定位：按代码看 bank 行与 lane 都是 `fork()` 得到的、origin 里带 919@1809，按理应该已经覆盖。
- **测量重跑（019dda7，`t7d` = 表 7 配置 + `ISAAC_LINEAGE_SOURCES=1`；`v3-019dda7-t7d-10260133`）**：第一次非零在版本 38，字段 u；被释放的是 (693, 702) 的编码块，帧 64–253，版本 8 编码、版本 11 释放，释放时它的区间为空（只保留数值：一个已结束、在宽限期内的局）；读它的是 v2 记录 (693, 830)，L = 184，祖先链 (693, 729)@182 → (693, 702)@182；这条分支的记录在版本 11 / update 48196 到达。learner 记下的 64 次在途报告里**没有任何一次**提到 702（连 actor 主状态的报告也没有）。64 次的环太短（释放与丢弃相隔 27 次更新），所以又换成每局每个来源的首末次报告（1c5c374 / 9086793），同时在释放的元数据里记下是哪一次 retain 调用。
- **原因 3（已用失败用例复现、已修）**：读代码找到了根因。`--actor-processes 2` 时 learner 用 `merge_jobs(halves)` 合并两个 actor 进程的半份交接，合并结果以 `dict(halves[0])` 开始，`lineage`（以及测量用的 `lineage_src`）只保留了第一个 actor 的。第二个 actor 的在途分支祖先从来没有报告给 learner，于是只保留数值（局还在跑时只剩本 rollout 的区间，结束后进入宽限期），它们以后发布的记录就读到已释放的图。这与上面两次都对得上：被丢弃的局与读它的记录同属一个 actor 进程（uid 第一项是进程号：(694, ·)、(693, ·)），而 learner 从没收到过关于它的报告。代码里"宽限期"注释推测的原因（"复用的 bank 行"）很可能就是这个。新测试 `tests/test_merge_jobs_lineage.py`：修复前合并结果只有第一个 actor 的 1 条，断言失败；修复后两个 actor 的 3 条都在，`in_flight_frontier` 给第二个 actor 的祖先 [ℓ − H, 共享帧数)。修法：`merge_jobs` 取两半 `lineage` 的并集（`lineage_src` 按来源取并集）。`test_actor_processes_smoke.py` 仍全过。宽限期规则没改。
- **测量重跑 2（9086793，`v3-9086793-t7d-10260212`）确认了归因**：第一次非零在版本 21；被释放的是 (694, 705) 的块，帧 0–240，版本 8 编码，版本 12 在 VJP 之后那次 retain（update 48196）释放，当时 705 还在本 rollout 里（区间 [499, ∞)）；读它的是 v2 记录 (694, 882)（L = 5，祖先链 (694, 706)@1 → (694, 705)@1），版本 14 到达。按局记下的首末次报告里 705、706、882 **一次都没有**，连"actor 主状态"这一项也没有；而 705 当时就在 learner 的 rollout 里。也就是说它属于一个其报告从未到达 learner 的 actor，与 `merge_jobs` 只取第一个 actor 的结论一致。（这两次测量运行用的是修复前的 `merge_jobs`，所以测量字段本身也只有第一个 actor 的。）
- **修复提交**：753a47a 主线 / 258f97a 计时链（两树 `bridge/python` 逐字节相同）。CPU 回归 68 个测试：除新增的 `test_merge_jobs_lineage` 通过外，与上一轮相同（13 个需要 CUDA 的本机失败）。
- **验收重跑 3（258f97a，表 7 配置，不开测量变量；`v3-258f97a-t7-10260236`）：通过。** 部署提交 258f97a91c2e（`code-v3-258f97a.commit.txt`）；9 个门全过（gate 日志 `gate-v3-258f97a-*`）；52 次更新、40 次稳态、无 OOM（train.log 无 out of memory）、退出码 0、`RUNNER_EXIT 0`；`source_horizon_dropped_grad` 52 行全部为 0.0，`SOURCE_DROP_FIRST` 未出现；**这次有 7 次更新算了 v2 loss，全在稳态窗口内**（update 48215、48216、48222–48226），所以原因 2、3 的路径这次被走到了。
  - 速度：稳态加权 312.5×（表 7 208911e 为 320.8×，低约 2.6%，原因未分辨；单次运行之间也有这个量级的差别，例如 1727f71 与 187ede3 同配置差约 6%）。中位 collect 7.57 s、update 11.81 s、wall 13.77 s。
  - 显存：在途祖先现在保留图，源卡峰值比表 7 高：卡 1/3/4/5 分别 11.2/11.4/10.7/10.8 GiB（表 7 10.7/10.3/8.9/8.9 GiB），学习器卡 2 为 23.4 GiB（表 7 23.6）。容器内存第 40 次 65.6 GiB、峰值 66.4 GiB（72 GiB 上限）。
  - 覆盖范围的限制：只有 7 次 v2 loss；teacher（teach）记录走的是无状态的空历史，不经过这条路径；宽限期规则（已结束局只保留数值）仍在，这次没有触发丢弃，但不能证明它永远不会触发。
- **宽限期路径的失败用例与审计盲区（8ce8a58 主线 / d345da5 计时链）**：
  - 新测试 `tests/test_source_grace_late.py`（线程与进程两种模式）：局 P 在 rollout 2 后结束（rollout 3 起换成局 Q），宽限期 2 次更新；分支 B 在 rollout 1 从 P 的第 10 帧分叉、在 actor 里在途，记录（L = 14）在 rollout 4 才到 learner（P 已结束），rollout 6 被消费。frontier 走生产代码：`train_tok` 里的 frontier 拼装与宽限期簿记原样移进 `record_retention.learner_frontier` / `grace_keep`（逻辑不变），测试直接调用它们。
  - A（生产规则：在途时 actor 报告 B 的 lineage）：dropped_grad = 0，每次更新参数与全保留逐位相同（max diff 0.0）。对照 V（只有宽限期、没有在途报告）：参数与全保留不同。
  - **对照 V 暴露了审计盲区**：修复前 V 的 dropped_grad 也是 0，参数却不同。原因：P 的编码块被释放（只剩数值）之后，记录到达时新写的 Writer 块直接读这些数值，梯度在这个 Writer 块处就断了，而审计只统计"回传到已释放块的余切"，这里没有余切可数。修法（只改审计，不改行为）：调度 Writer 块时标出读了已释放块的那些块（`_stale_rows`），它们以后收到的 horizon 内余切计入 `source_horizon_dropped_grad`（字段名 `mem_on_released_input`）。加上以后 V 的审计为 8.5e-04（线程、进程都是），A 仍为 0。
  - 结论：宽限期本身（只保留数值）保不住迟到记录的梯度，保住它的是在途报告。修好 `merge_jobs` 之后每个在途分支都有报告，所以宽限期只是数值上的保险。现有规则在有报告时做得到，按"保留可达区间的图"的原则不需要再改规则。
  - CPU 回归 69 个测试：除新增测试通过外与上一轮相同；H=0 等价逐位相同（v3r8，本地与进程两种模式）。
- **验收重跑 4（d345da5，表 7 配置，加强后的审计；`v3-d345da5-t7-10260320`）：通过。** 9 个门全过；52 次更新、40 次稳态、无 OOM、退出码 0、`RUNNER_EXIT 0`；`source_horizon_dropped_grad`（现在含 `mem_on_released_input`）52 行全为 0.0；稳态窗口内 6 次 v2 loss；稳态 311.8×；源卡峰值 11.5/13.4/10.5/11.0 GiB，学习器卡 23.1 GiB，容器第 40 次 66.4 GiB。验收重跑 3 的 0 是在旧审计口径下得到的，这次在新口径下复核。

## 19 表 8 加速（进行中）

### 19.1 备份更新（2026-10-26）

- rl 仓库：`exp/memory-chain-v3` 合并候选主线 753a47a（合并提交 05f91e7）后加快照提交 dd6f104（REPORT、三条原因、三个失败用例、验收重跑 1–3 与两次测量运行的 progress/result/summary、`drop_first.json.txt`、门测日志、提交映射）；`exp/memory-chain-v3-timing` 快进到 258f97a；两者已推送 origin。分支 README 首行已改为"容量长测与梯度审计通过（40 次稳态、无 OOM、dropped_grad = 0，v2 loss 7 次），速度未达标（312.5× 对 843.4×）"。
- 3090：`~/isaac-abplus/archive-memory-chain-v3/memory-chain-v3-20261026-0257.tar.gz`，145,558,082 字节，sha256 `606e972e7236960557db697be551cb5102f7786a2b740d78897cc384484b07cd`（远端核对一致）；旧的 `memory-chain-v3-20261026.tar.gz` 未动。
- 之后的提交（宽限期用例、审计盲区、表 8 各步）还不在备份里。

### 19.2 一次稳态更新的 host 剖析（`t7prof`，d345da5，cProfile 第 16 次更新）

cProfile 下学习器 11.25 s（含剖析开销，只看相对大小）。与表 8 四项相关的 host 部分：
- Reader 索引（SourceView 构造 0.20 s、`_credit_plan` 0.29 s、`_gather_queries` 0.25 s、`_query_tokens` 0.10 s）。驻留 batch 的 credit/传输计划**本来就是每 rollout 建一次**（`pipeline_plans`），每个 minibatch 只剩查询帧查找与分组布局：按生产规模（32768 中取 8192）在 CPU 上实测约 4 ms/minibatch。
- SourceBatch 组装（`__init__` 非等待部分约 0.35 s、`materialize` 非等待约 0.67 s、`schedule` 0.74 s、`key()` 29 万次 0.15 s）。
- **不在四项清单里、但更大的**：`retain_histories` 两次调用 1.67 s（扫描约 70 万个帧键的字典/集合推导 1.2 s）；在途 lineage 三元组的集合运算约 0.7 s（我在原因 2 修复里引入的，每个上下文一条）；审计每次 credit 的 `float()` 同步（339 次，我在 `SOURCE_DROP_FIRST` 诊断里引入的）。

### 19.3 已做的改动（各自 CPU 回归 + H=0 逐位等价）

| 提交（主线 / 计时链） | 内容 | 语义 |
|---|---|---|
| 0b7e51c / e8930b5 | 审计不再每次同步（第一次非零在每次更新的 `horizon_stats` 统一找，一次同步）；在途 lineage 每个（祖先, 共享帧数）只报最短分支（同一并集，`tests/test_lineage_compact.py` 随机分支树、H = 0/3/10/50 可达帧逐一相同） | 不变 |
| 5883d2d / 346bb83 | **表 8 第 1 步**：minibatch 的查询帧改为按索引取驻留 batch 的数组 | 不变（同一数组） |
| d83ae01 / a8cc4a0 | `retain_histories` 用按 uid 的键索引删除退出局的条目，活块/活写由块自身的 uid 推出（与全扫描同一集合；`ISAAC_RETAIN_CHECK=1` 时重算全扫描并断言相等，CPU 全套在此开关下通过） | 不变 |

### 19.4 短跑（表 7 配置，25 游戏小时，稳态取第 8–20 行，单次运行，噪声约 ±5%）

| | e8930b5（基线：审计不同步 + lineage 压缩） | a8cc4a0（+ 第 1 步 + retain 索引） |
|---|---|---|
| 门测 | 5/5 | 5/5 |
| dropped_grad | 全 0 | 全 0 |
| x_real_time（加权） | 320.1× | 353.0× |
| 学习器 update_s 中位 | 10.21 | 10.01 |
| u_ppo | 3.37 | **2.57** |
| u_teach_build | 2.13 | **1.46** |
| u_rc_sources | 3.25 | 3.97 |
| 其中 ready_encode_wait | 2.10 | 2.94 |
| 采集 collect_s 中位 | 7.49 | 9.08 |

- host 侧省下的约 1.5 s（PPO −0.8、记录批 −0.7）**大部分被"等源编码"吃回去了**（+0.7 s），学习器中位只降 0.2 s。x_real_time 的 +10% 主要来自这次的采集更慢、学习器等待被掩盖，单次运行之间不可分辨，不作为加速结论。
- 原因：**源卡接近饱和而且负载不均**。a8cc4a0 每次更新各源卡忙碌时间中位（encode + writer + source_backward）：cuda:0 7.4 s、cuda:2 6.8 s、cuda:3 10.6 s、cuda:4 9.9 s，墙钟 12.4 s。最忙的卡 85% 时间在算；学习器等的是它。共享 actor 的两张卡预算更小，分到的块更少，VJP 也更少。
- 推论：学习器要到 ≤ 7.5 s，最忙源卡每次更新的工作量（现在约 10.6 s）也必须降到 7.5 s 以下。只做 host 侧（第 1、2 步）不够；源卡上编码与 VJP 的并发（第 4 步）、按实际忙碌时间均衡四张源卡、或减少 VJP 本身的工作量才是主因。第 3 步（预取时机）只能把编码挪到采集期间，而源卡在采集期间也有约 85% 在忙。

### 19.5 源卡按实测工作量均衡（`--source-balance 1`，308ecb0 主线 / 8af8ff1 计时链；默认 0 = 原行为）

- 做法：每次更新开始时统计上一周期每张源卡的 encode + writer + source_backward 秒数（`observe_busy`）；权重按"每卡近 5 次中位 / 各卡均值"做平方根积分更新（几何均值 1，限制在 [1/4, 4]）；新块组分给"加权负载最小、且显存预算还装得下"的卡（都装不下时按原来的显存比例）。共享 actor 的卡仍按扣掉 actor 预留后的剩余预算判断能否装下。纯调度：值在哪张卡上算都一样（`tests/test_source_balance.py`：cpu 上 Reader 输出差 4e-9，只来自分块不同），CPU 回归、H=0 逐位等价均通过；progress.csv 新增 `source_busy_w{i}`、`source_balance_w{i}`。
- **GPU2 这段时间被别人占用（12 GB），5 卡布局起不来**，所以均衡和下面的测量都在 4 卡布局上做（物理卡 3、1、4、5：学习器 cuda:1，actor cuda:0/2，源 cuda:0/2/3，与 §16 的 4 卡布局角色相同），表 7 配置其余不变，25 游戏小时，第 8–20 行。5 卡上的均衡效果还没测。

| 4 卡，按层 checkpoint | 均衡关（efeb5bb） | 均衡开（efeb5bb） |
|---|---|---|
| 三张源卡忙碌秒（cuda:0 / 2 / 3） | 9.93 / 10.36 / **17.15** | 13.91 / 14.73 / 13.96 |
| 差距（max − min）/ 均值 | **57.8%** | **5.7%**（第 8–19 行 7.4%） |
| 最忙卡忙碌 | 17.15 s | 14.73 s |
| 墙钟中位 / x_real_time | 14.25 s / 306.3× | 13.19 s / 318.3× |
| dropped_grad | 全 0 | 全 0 |

- 目标"四卡忙碌之差 < 10%"在 4 卡布局上达到。但忙碌秒已大于墙钟（源卡任务重叠：VJP 期间插队的编码），**均衡之后源卡整体仍然饱和**，所以墙钟只降 1.1 s。

### 19.6 VJP 工作量的测量（只测量，不定方案；4 卡布局，均衡开，`ISAAC_SOURCE_MEMORY_BREAKDOWN`）

生产配置短跑（每帧字节来自源卡显存拆分，VJP 秒为三张源卡的中位，"每千带图帧"是 VJP 秒 / 该卡带图帧）：

| | 每帧显存（KiB，cuda:3） | VJP 秒 / 次更新（三卡中位） | 每千带图帧的 VJP 秒 | 源卡峰值（nvidia-smi） | 源梯度误差（基准见下） |
|---|---|---|---|---|---|
| 全部 4 块 checkpoint（现状，t8mck4） | 47–51 | 11.5 | **0.232** | 10.7 / 10.4 / 7.4 GiB | 相对完整图 7.8e-4 |
| a）只 checkpoint 块 2、3（t8mck24） | 177–188 | 7.4 | **0.163** | 17.5 / 17.9 / 14.1 GiB | 相对完整图 7.8e-4 |
| 完整图（t8mfull4） | 295–324 | 4.6 | **0.118** | **24.0 / 23.7** / 21.2 GiB：第 20 次更新在共享卡 cuda:2 上 OOM（源进程 17.3 GiB + actor 进程 3.5 + 1.8 GiB） | 0 |
| b）`--source-flash 0`（t8mnoflash4） | — | — | — | — | — |

- 4 个 Spatial 块保存的张量形状完全相同（§16.1：每块约 56 KiB/帧），所以没有"更重的两块"，a）取块 2、3。
- 生产里 checkpoint 的 VJP 代价远大于 §18.4 隔离评估（当时局短，+0.1 s）：完整图每千带图帧 0.118 s，checkpoint 两块 0.163 s（+38%），四块 0.232 s（+97%）。
- b）`--source-flash 0` 在生产配置下**跑不起来**：第一次 VJP 时报 `Trying to backward an extra time ... selective activation checkpoint`（非 flash 的分桶注意力路径用了选择性重算 `ISAAC_SOURCE_POINTWISE_RECOMPUTE`，不允许同一区域反传两次），1 次更新后被守护停掉（退出码 137）。只能给 CUDA 基准里的数（下一条）。
- CUDA 基准 `tests/bench_source_variants_cuda.py`（宽 192、4 块、1024 帧、编译、BF16，单区间；小模型的帧比生产帧小，只看相对值）：

| | 编码后每帧留存 KiB | 编码 s | VJP s | 误差：相对完整图 | 误差：相对 fp32 源 |
|---|---|---|---|---|---|
| 完整图（flash） | 556 | 0.265 | 0.111 | 0 | 1.15e-2 |
| 全部 checkpoint | 199 | 0.300 | 0.150 | 7.8e-4 | 1.16e-2 |
| 块 2、3 checkpoint | 378 | 0.238 | 0.120 | 7.8e-4 | 1.16e-2 |
| flash 关（分桶注意力） | 639 | 0.245 | 0.097 | 1.3e-3 | 1.23e-2 |
| fp32 源（flash 关） | 1175 | 0.219 | 0.090 | 1.16e-2 | 0 |

  误差都由 BF16 主导（相对 fp32 约 1.2e-2）；flash 与非 flash 之间 1.3e-3，checkpoint 重算 7.8e-4。flash 关：留存 +15%，VJP −12%。

### 19.7 c）完全不开 checkpoint 时最忙卡峰值的估算

- 每帧按实测 295–324 KiB（取 310），加上非带图部分约 2 GiB（数值、版本、缓存），VJP 瞬时峰值另加约 0.2 GiB（`peak-before`）。
- **按表 7 的带图帧平台**（208911e：合计中位 12.6 万、最大 16.5 万）：均衡后每张源卡约 1/4，即中位 3.1 万、最大 4.1 万帧 → 9.3 / 12.2 GiB，加 2.2 → **11.5–14.4 GiB**。专用源卡（约 21.5 GiB 可用）和共享卡（24 − actor 约 5.4 GiB ≈ 18 GiB）都放得下。
- **但现在的带图帧比表 7 多**（d345da5：合计中位 19.4 万、最大 27.4 万；D2a 改为保留在途祖先的图，以及 v2 记录更活跃）：每卡中位 4.8 万、最大 6.9 万帧 → 14.3 / 20.3 GiB，加 2.2 → **16.5–22.5 GiB**，超过共享卡的约 18 GiB，最大时也接近专用卡上限。4 卡布局的完整图短跑在第 20 次更新已在共享卡 OOM，与此一致。
- 所以完全不开 checkpoint（VJP 每帧相对现状 −49%）在 5 卡上只有带图帧回到表 7 水平时才稳；部分 checkpoint（块 2、3）每帧 180 KiB，按现在的最大值约 6.9 万帧 × 180 KiB + 2.2 ≈ 14 GiB，放得下，VJP 每帧相对现状 −30%。这些是估算，5 卡上没有实测。

### 19.8 备份（第二次）

- rl `exp/memory-chain-v3` = 12af01c（合并主线 0053e5d + 快照 4），`exp/memory-chain-v3-timing` = efeb5bb，均已推送。
- 3090：`memory-chain-v3-20261026-0707.tar.gz`，170,940,203 字节，sha256 `51830af257d4590de9d6e7c4e5d4deffa9e3b02d959cf38c0956c5d49de5a640`（第一次 scp 中途断开，同名的不完整文件已用完整上传覆盖，远端核对一致）；更早的两个 tar 未动。

### 19.9 源卡上 encode 与 VJP 的多 stream 并发（`--source-streams 2`，默认 0；用户已允许）

- 实现（主线 6b84d22 起，到 1cc515a；计时链对应到 f2fe8bb）：源进程里 VJP 在独立线程里跑，主线程继续处理 encode / Writer。原来主线程只在 VJP 的局组之间插队处理，现在不再插队。需要 `ISAAC_SOURCE_PREEMPT_VJP=1` 提供的独立 VJP 连接。
  - CUDA 上用 4 条 stream 轮换，每来一次 VJP，新工作换到下一条。autograd 让每个反传算子跑在其前向所在的 stream 上，所以 VJP 处理前几个周期的图时，不会排在本周期的编码后面。
  - 依赖都用事件显式表达：
    - encode / Writer 导出后 record 事件，默认 stream 先 wait 这个事件，再做 IPC 发送。
    - Writer 的 stream wait 它读的各块的事件，并对这些块的 u 做 `record_stream`，分配器在 Writer 的 stream 越过之前不会回收这块存储。
    - VJP 用到的各 stream 先 wait 默认 stream（导入的余切）；VJP 结束后默认 stream wait 全部 stream，再导出梯度。
    - VJP 释放的工作区只会被同一 stream 上之后的工作复用（缓存分配器按 stream 分池）。
  - IPC 导出串行化，因为分配器设置是进程级的。
- 第一次上卡就暴露了两个线程竞争（都已修，并补进门测的 `--production` 变体）：
  1. checkpoint 块的重算在 VJP 线程里调用编译函数，主线程这时正在 Dynamo 追踪 → 重算改为在同一把（可重入）锁下进行。
  2. VJP 线程给副本的 `named_parameters()` 设梯度时，并发的 encode 正把这些参数临时换成 BF16 cast（`_cast_context`）→ dtype 错。修法：副本创建时列出真实参数，VJP 用这张表；并发时 encode / Writer 整个调用持锁，重算也拿这把锁。
- 另外：CUDA 门测里源进程在正常 close 之后偶发 SIGSEGV（-11），串行调度下也有，faulthandler 没有任何 Python 帧，属于原生 teardown。改为 close 回复之后 `os._exit(0)`（1e264df）。测试主进程在打印 PASS 后也有一次 SIGSEGV，同样处理。
- 测试 `tests/test_source_streams.py`：串行（S）对比并发 + 故意打乱的调度（C1、C2：每次 encode / Writer 之前和 VJP 的每个局组之间，随机 host sleep，CUDA 上再加随机 `torch.cuda._sleep`）；horizon 6 让 detach / retire 在 VJP 进行中到达。
  - CPU：参数和每次提交的源梯度都**逐位相同**，审计为 0，并确认有编码落在 VJP 区间之内。
  - CUDA 门测（lr 0，比较每次提交的源梯度）：默认设置下 C 相对 S 是 6.8e-10 / 7.0e-10，S 重跑一遍与自身差 6.6e-10（index_add_ 原子操作的跑间噪声）；`--production`（编译、flash、BF16、checkpoint）下 C 相对 S 是 3.4e-10，S 重跑逐位相同。都在 1e-6 以内，门测通过。
  - CPU 回归 72 个测试除新增外与上一轮相同；H=0 逐位等价通过（v3r12）。
- **4 卡短跑（GPU2 仍被占用；均衡开、四块 checkpoint；f2fe8bb `t8st4` 对 efeb5bb `t8mck4`）：没有加速。**

| | 不开并发 | `--source-streams 2` |
|---|---|---|
| 墙钟中位 / x_real_time | 13.19 s / 318.3× | 13.71 s / 300.6× |
| 等源编码 ready_encode_wait | 4.16 s | **3.10 s** |
| 等 VJP 提交 commit_wait | 0.21 s | **1.43 s** |
| VJP 最慢一张卡 | 11.97 s | 13.13 s |
| 三卡忙碌秒（cuda:0/2/3） | 13.9 / 14.7 / 14.0 | 14.7 / 15.5 / 14.4 |
| 源卡峰值（nvidia-smi，物理卡 3/4/5） | 10.7 / 10.4 / 7.4 GiB | 13.8 / 12.4 / 8.7 GiB（**+1.3 到 +3.0 GiB**，两路工作区同时在） |
| dropped_grad | 全 0 | 全 0 |

  编码确实不用再排在 VJP 后面（等编码 −1.1 s），但 VJP 也随之变慢、更晚结束，学习器改为等提交（+1.2 s）。源卡在两者并发前就已经算满（忙碌秒大于墙钟），并发只能重新分配算力，不能凭空增加，所以墙钟没有下降（−4%，在单次运行的噪声范围内）。
- 表 8（表 7 配置 40 次稳态）还没跑：要等 GPU2 空出来，5 卡布局才能起；等待脚本每 10 分钟检查一次，最多 8 小时。由于 4 卡上并发没有收益，表 8 用"均衡开、并发关"（`t8`），并发开另跑一组（`t8x`）对照。

### 19.10 源 VJP 的 kernel 级诊断（`ISAAC_SOURCE_VJP_PROFILE=10`，d63180b，4 卡，均衡开；只出诊断，不改实现）

每张源卡第 10 次源 VJP 单独用 torch.profiler 采（这次 VJP 中间不插队处理编码）；全部四块 checkpoint（`t8pck4`）与完整图（`t8pfull4`）各一次。完整表和前 10 类 kernel 见 `results/vjp-profile-d63180b.md`。

| | ck cuda:3 | ck cuda:0 | ck cuda:2 | full cuda:0 | full cuda:3 | full cuda:2 |
|---|---|---|---|---|---|---|
| VJP 墙钟（剖析下）s | 12.12 | 16.73 | 17.68 | 12.83 | 12.74 | 16.31 |
| 本次 VJP 的帧（编码块行数） | 11,453 | 17,679 | 21,813 | 16,287 | 17,878 | 18,824 |
| 包：编码块 / Writer 块 | 94 / 115 | 142 / 166 | 148 / 165 | 120 / 168 | 136 / 158 | 149 / 198 |
| `autograd.backward` 调用（按局分组） | 56 | 62 | 84 | 57 | 66 | 83 |
| 每块帧数 中位 / 均值 / 最大 | 92 / 122 / 264 | 62 / 124 / 267 | 200 / 147 / 264 | 168 / 136 / 266 | 169 / 131 / 270 | 108 / 126 / 275 |
| 每次 backward 调用的帧 | 205 | 285 | 260 | 286 | 271 | 227 |
| kernel 数 | 91,755 | 128,302 | 137,711 | 112,236 | 112,080 | 134,368 |
| 每千帧 kernel 数 | 8,011 | 7,257 | 6,313 | 6,891 | 6,269 | 7,138 |
| kernel 时长 均值 / 中位 / p90（µs） | 5.0 / 2.3 / 9.1 | 5.3 / 2.4 / 10.3 | 5.2 / 2.3 / 10.2 | 5.1 / 2.3 / 9.2 | 4.9 / 2.2 / 8.4 | 4.9 / 2.2 / 8.7 |
| kernel 时间合计 s | 0.46 | 0.68 | 0.72 | 0.57 | 0.55 | 0.65 |
| GPU 活动窗口 s / 其中忙 s | 7.01 / 0.47 | 9.89 / 0.70 | 10.59 / 0.74 | 6.85 / 0.59 | 6.65 / 0.57 | 8.31 / 0.77 |
| **窗口内 GPU 空闲占比** | **93.3%** | **92.9%** | **93.0%** | **91.4%** | **91.5%** | **90.7%** |
| autograd 节点（每千帧） | 42,202（3,685） | 59,270（3,353） | 63,935（2,931） | 56,030（3,440） | 57,753（3,230） | 71,094（3,777） |

- 同一批运行里不剖析的 VJP：三卡中位 7.8–8.5 s（checkpoint）、6.1–7.6 s（完整图）；而一次 VJP 的 kernel 时间合计只有 0.5–0.8 s。**GPU 在 VJP 期间约 90% 以上的时间是空的，源 VJP 是 launch / host 开销瓶颈，不是算力瓶颈**。
- 每千帧约 6–8 千个 kernel、约 3.3 千个 autograd 节点（每帧约 7 个 kernel、3 个节点）；kernel 中位 2.3 µs，最耗时的一类是 float 的逐元素加（每次 VJP 2–3 万次，梯度累加）。节点里最多的是 `AccumulateGrad`（1.1–2.0 万次）和 `ImmutableSourceCastBackward`（0.7–1.2 万次）：每个按局分组的 `backward` 调用都对约 200 个参数各累加一次，每个编码块的 BF16 参数 cast 也各有一个反传节点。
- 每次 VJP 处理 200–350 个包（编码块 + Writer 块），每块中位 60–200 帧，分 56–84 次 `backward` 调用，每次约 200–290 帧。
- checkpoint 与完整图的 GPU 时间几乎一样（0.46–0.72 对 0.55–0.65 s），两者墙钟的差别主要在 host 上（重算多出来的那些小 kernel 和节点）。
- 结论（与发起会话的推测一致）：要降 VJP 时间，应该把每次反传的粒度做大，而不是继续在调度上找时间。可选方向：更大的块；同一张卡上多个块（多局）合成一次 `autograd.backward`；参数 cast 每个版本做一次而不是每块一次；在固定形状桶上用 CUDA Graph 捕获反传。这里只出诊断，没有改实现。
- 剖析器本身有开销，所以剖析下的墙钟（12–17 s）比平时长；kernel 数、节点数、GPU 忙时不受它影响。

### 19.9 补充：`--source-streams 2` 在生产设置下并非每次都与串行逐位一致

- 修完两个竞争之后（1cc515a / f2fe8bb 起），CUDA 门测的 `--production` 变体（编译、flash、BF16、checkpoint、组件图缓存）按同一随机种子重复跑了 4 组（31b1470 一组，bc138f5 三组：原样两次、关组件缓存一次、固定 cuBLAS 工作区一次），每组两次并发对一次串行：
  - 8 次并发里 6 次源梯度与串行**逐位相同**；1 次差 1.4e-6（超过 1e-6 的门限，31b1470 那组的门测因此失败；bc138f5 原样那次复现了同一个值），1 次差 1.4e-7（关组件缓存那组）。
  - 串行重跑与自身始终逐位相同。
- 也就是说，并发调度偶尔会改变数值，而且超过了浮点累加次序能解释的量级。来源尚未定位：时序相关，种子固定也时有时无；候选是并发时 retire / detach 与编码的相对次序（串行时它们排在 VJP 之后处理）。
- 并发本身在 4 卡上没有加速（§19.9），表 8 不开并发；在定位清楚之前，`--source-streams` 保持默认关，不建议使用。

### 19.11 表 8（发起会话定的配置：5 卡、均衡开、并发关；40 次稳态；31b1470）

- 第一次跑（f2fe8bb）：主组 t8（只 checkpoint 块 2、3）在第 36 次更新 OOM。一张源卡在一次更新内从 3.8 万涨到 9.4 万带图帧（这个 rollout 的新块全部给了权重最低的卡），VJP 时在 20 GiB 分配处溢出。均衡因此改为（31b1470）：均衡挪块时目标卡的预算还要留 20%，每次更新权重最多变 1.25 倍。同一代码上的对照组 t8b 跑完了（264.6×），记为旧均衡下的数据，不作表 8。
- 31b1470 上的门测：除并发门测外都通过；表 8 不开并发，并发门测见 §19.9 补充。

| | 表 7（d345da5，均衡关，四块 checkpoint） | **t8**（均衡开，块 2、3 checkpoint） | **t8b**（均衡开，四块 checkpoint） |
|---|---|---|---|
| 更新 / 稳态 | 52 / 40 | 52 / 40 | 52 / 40 |
| 退出码 / OOM | 0 / 无 | 0 / 无 | 0 / 无 |
| **dropped_grad** | 全 0 | **全 0** | **全 0** |
| 稳态内算了 v2 loss 的更新 | 6 | 6 | 5 |
| **x_real_time（加权）** | 311.8× | **326.8×** | 306.0× |
| 墙钟中位 | 13.58 s | 12.31 s | 14.16 s |
| 学习器 update_s 中位 | 11.58 s | 10.17 s | 12.02 s |
| 采集 collect_s 中位 | 7.72 s | 7.54 s | 7.43 s |
| 等源编码 | 2.42 s | 2.86 s | 4.29 s |
| VJP 最慢一张卡（每次更新） | 9.51 s | 8.71 s | 11.12 s |
| 四张源卡忙碌秒中位（cuda:0 / 2 / 3 / 4） | 7.66 / 8.89 / 11.42 / 11.52 | 8.02 / 8.10 / 9.81 / 8.78 | 9.67 / 9.55 / 9.47 / 9.03 |
| 卡间差距（各卡中位） | 39.2% | 20.6% | 6.8% |
| 每次更新最忙卡中位（q75） / 当次差距中位 | 12.40 s（14.32） / 52.6% | 11.54 s（13.10） / 50.6% | 14.05 s（16.17） / 69.2% |
| 源卡带图帧中位（cuda:0 / 2 / 3 / 4） | 38k / 39k / 58k / 56k | 44k / 44k / 48k / 46k | 45k / 47k / 40k / 44k |
| 源卡峰值（nvidia-smi，物理卡 3 / 1 / 4 / 5；GiB） | 13.1 / 11.2 / 10.2 / 10.7 | **22.8 / 20.3 / 23.0 / 22.4**（卡上限 23.6） | 18.3 / 13.9 / 9.6 / 9.0 |
| 源进程分配最大值（学习器侧统计） | 8.2 GiB | 19.7 GiB | 9.7 GiB |
| 容器内存第 40 次 / 峰值（上限 72） | 66.4 / 67.1 GiB | 69.4 / **71.2** GiB | 66.1 / 67.3 GiB |

- **t8 比表 7 快 4.8%**（326.8× 对 311.8×，学习器 −1.4 s），主要来自 VJP 变短（块 2、3 不重算）。但余量很小：四张源卡 nvidia-smi 峰值 20.3–23.0 GiB（卡上限 23.6 GiB，含分配器缓存），容器内存峰值 71.2 GiB，接近 72 GiB 上限。（注：§19.5–19.9 表里的"源卡峰值"是 nvidia-smi 的 MiB ÷ 1000，约为 GB，不是 GiB。）
- **t8b 比表 7 慢 1.9%**（单次运行，在噪声范围内）。均衡把各卡的长期中位拉平到 6.8%，但**每次更新内的差距中位仍是 69%**：每次更新最忙的是不同的卡，当次最忙卡的中位（14.05 s）反而高于不均衡的表 7（12.40 s）。所以"四卡忙碌之差 < 10%"按长期中位达到了，按每次更新没有达到，均衡也没有带来墙钟收益。原因是新块按 rollout 成批分配、VJP 按局成批进行，单次更新的负载天然在卡间跳动；而按长期中位调整的控制器管不到这种逐次波动。
- 均衡权重在 t8 中被推到上下限（0.25 / 4.0）：两张共享卡更闲，但它们的显存预算加 20% 余量已放不下更多块。
- 与 §19.10 合起来看：源卡在 VJP 期间 GPU 约 90% 空闲，瓶颈在 host / launch。所以表 8 的这些调度与 checkpoint 选项，只能在几个百分点内移动结果。

### 19.12 备份（第三次）

- rl `exp/memory-chain-v3` = 0fa9023，`exp/memory-chain-v3-timing` = bc138f5，均已推送。
- 3090：`memory-chain-v3-20261026-1047.tar.gz`，195,539,614 字节，sha256 `386426529a7d8fda2cafc25513906389d2150105d4fea283e7a4e72a26f1045c`（第一次 scp 断开，重传覆盖同名不完整文件，远端核对一致）。
- 运维注意：本会话里对后台 `bash 脚本` 的 TaskStop 没有杀掉脚本及其 python 子进程，残留的旧链在 GPU 空出后起了一个旧代码的容器（48775f7 的 t8bm）。已手动杀掉残留进程并停掉该容器，确认没有自己的容器残留。

## 20 反传粒度（发起会话 10-27 的 1–3 项；全部不改语义）

### 20.1 实现（6d9fd25 / 4cee31f 主线，ac21ba9 / 2120139 计时链；默认全关）

- `--source-vjp-merge`：`0` = 原来每个局组一次 `autograd.backward`；`1` = 每张卡每次 VJP 只调一次（engine 先把每个参数的入梯度加起来，再做一次 AccumulateGrad）；`N > 1` = 把整局组打包，每次调用至少 N 个编码帧，调用之间仍然处理等待中的编码。`retain_graph` 和 pinned / 已结算的判定都不变。
- `--source-fused-cast 1`：`ImmutableSourceCastFused`，每次编码只用一个 cast 节点覆盖全部 BF16 参数别名。每个参数的 BF16 梯度仍在累加前先转成 FP32（一次平铺转换），数值与原来逐位相同。
  - 说明：发起会话设想的"每个参数版本只有一份 cast、一个反传节点、各块共用"会让各块的梯度先在 BF16 里相加，再转 FP32。这是 `ImmutableSourceCast` 特意避免的精度损失，不只是浮点次序差。所以这里实现的是"每次编码一个节点"，不是"每个版本一个节点"。
- `ISAAC_SOURCE_ENCODE_PROFILE=first:count`：用一个剖析器覆盖编码 first 到 first+count−1，只做测量。
- 测试 `tests/test_source_vjp_merge.py`（CPU 线程模式按局分组、CPU 进程模式、CUDA 门测）：
  - 与默认相比，源梯度差：merge 4.6e-9，pack 2.1e-9，fused 逐位相同；CUDA 门测 merge 2.3e-8、fused 逐位相同。
  - 审计为 0；源版本、编码行数相同。
  - CPU 计数：engine 调用 44 → 16（merge）/ 22（pack 24），AccumulateGrad 4212 → 1092，cast 反传 1920 → 120。
- CPU 回归 73 个测试除新增外不变；H=0 逐位等价通过（v3r13）。新门测 `vjp-merge-cuda` 通过。并发门测（§19.9 补充里的偶发差异）已移出默认门测集合，因为并发保持关闭。

### 20.2 同口径剖析（ac21ba9，4 卡，四块 checkpoint，均衡开，第 10 次 VJP）

| | 默认 | merge 1 | fused | merge 1 + fused |
|---|---|---|---|---|
| autograd 节点 / 每次 VJP | 5.8–6.8 万 | 3.9–5.0 万 | 5.8–6.3 万 | **3.2–4.1 万** |
| 其中 AccumulateGrad | 1.8 万 | 基本消失 | 1.8 万 | 基本消失 |
| 其中 ImmutableSourceCastBackward | 1.1 万 | 1.1 万 | 消失 | 消失 |
| kernel 数 / 每次 VJP | 12.7–14.3 万 | 11.0–14.2 万 | 14.4–15.3 万 | 11.0–14.3 万 |
| kernel 中位时长 | 2.3 µs | 2.4 µs | 2.2 µs | 2.2 µs |
| GPU 空闲占比 | 92–93% | 92–93% | 92% | 92–93% |

- 节点数降了 40–50%，**kernel 数基本没变**：梯度的逐边加法、编码块里组件与 Spatial 的反传 kernel 仍然一个都不少。剩下的节点主要来自每个编码块的组件和拼接（Select、Embedding、View、ToCopy、Add、IndexSelect、Cat、Slice、CompiledFunctionBackward……）。按包数回归，每次 VJP 的 kernel 约为"每个编码块 600 个 + 每个 Writer 块 100 个 + 每次 backward 调用 330 个"。
- 编码（同次剖析，只取剖析窗口里没混进反传的 8 段，每段 40 次编码）：每块中位 115 帧、约 260 个 kernel，每千帧约 2.3 千个 kernel，GPU 每帧约 14 µs，kernel 中位 2.5 µs。按平时编码约 0.15–0.2 ms/帧算，**编码同样约 90% 是 host 开销**。

### 20.3 4 卡短跑（2120139，25 游戏小时，第 8–20 行；默认组跑了 3 次，用来看噪声）

| | 默认 ① | 默认 ② | 默认 ③ | pack 512 | pack 1024 | pack 2048 + fused | pack 4096 + fused | merge 1 + fused |
|---|---|---|---|---|---|---|---|---|
| x_real_time | 335.1 | 330.8 | 316.1 | 317.4 | 327.8 | 314.0 | 305.2 | 297.6 |
| 学习器 update_s | 10.06 | 11.31 | 10.97 | 11.18 | 11.56 | 12.10 | 11.93 | 12.07 |
| VJP 最慢一张卡 | 11.28 | 12.87 | 9.25 | 9.68 | 9.15 | 8.91 | 9.30 | **7.21** |
| 等源编码 | 3.52 | 3.51 | 3.16 | 4.68 | 3.68 | 5.47 | 5.95 | 5.48 |
| 等 VJP 提交 | 0.16 | 1.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dropped_grad | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **单次短跑的噪声很大**：同一配置三次，VJP 最慢卡 9.3–12.9 s、x 316–335。pack 512 到 4096 的 VJP（8.9–9.7 s）都落在这个范围里，分辨不出来。只有 merge 1 的 7.2 s 明显低于默认的三次（−22% 到 −44%）。
- **但学习器没有变快，反而变慢**：merge 越大，等编码越长（3.2–3.5 → 5.5–6.0 s）。原因在源进程的结构：VJP 与编码在同一个单线程源进程里，编码请求只能在两次 `backward` 调用之间被处理（`service_forward`）。调用次数从约 75 次降到 1 次，编码就要等完整个 VJP；而学习器每个 rollout 的编码正好落在 VJP 进行期间。VJP 省下的时间全部转成了编码等待，提交等待已经是 0，所以墙钟不降。
- fused cast（每次编码一个 cast 节点）去掉了约 1.1 万个节点，但在噪声范围内看不出时间变化。
- 结论：在"一张卡一个单线程源进程、编码与 VJP 轮流"的结构下，单靠把反传粒度做大不能提高吞吐。只有当编码不必等 VJP 时，VJP 省下的时间才会进到学习器上。按发起会话的条件（"1 和 2 做完后若最忙卡 VJP 明显下降，再跑表 9"）：VJP 只在 merge 1 下明显下降，而那组吞吐最低，所以**没有跑表 9**。

### 20.4 第 3 项：编码块为什么只有 60–200 帧（只评估，不改）

- 编码块由 `schedule()` 按（局, 卡）把一次调度里的新帧分组，每组再按源 micro（2048）切。所以块大小 = 一次调度里一局的新帧数。主要的调度来自 actor 的源预取（每 8192 条决策一次，64 局 → 每局约 128 帧）和学习器的记录批。实测中位 60–200、最大约 270 帧，与此一致。Writer 块按 64 帧对齐、属于块所在的卡，不限制编码块大小。
- 要做到 1024 帧量级，只能把多局放进同一次编码（像现在无状态教师记录那样的多 uid 块）。代价：
  - 块的寿命变成"其中最后一局退出或越过 horizon"。同一 rollout 的各局按同样的步伐前进，horizon 释放大体同步，但局一结束就能释放的图要等同块其他局。
  - 退出、detach 都按整块进行，峰值显存会升高。按每块 600 个 kernel 的回归估计，每次 VJP 的编码块从约 130 个降到约 20 个，可省约 6.5 万个 kernel（约一半）。编码那边每块约 260 个 kernel，也会按比例减少。
- 这一项会改变图的保留粒度（显存行为），而且不先解决 §20.3 的"编码等 VJP"，VJP 再短也进不了吞吐，所以没有实现。

### 20.5 建议的下一步（待定）

1. **先让编码不必等 VJP**：只要编码请求不再排在单线程源进程的 VJP 后面，merge 1（VJP −22% 到 −44%）就能直接转成学习器时间。候选做法：
   - 学习器把一个调度的全部编码合成一次 RPC，减少往返；
   - 或者让 VJP 在 C++ 里跑完整个 autograd、host 上不持 GIL，再配合已有的线程并发。`--source-streams 2` 现在没有收益，原因之一可能就是 GIL：VJP 线程里的 Python hook / 自定义 Function 和编码线程争锁。
2. 然后才是固定形状桶上的 CUDA Graph（编码与反传都适用）和多局编码块（§20.4）。
3. 4 卡单次短跑的噪声约 ±10%（默认三次：VJP 9.3–12.9 s），之后的对比需要重复跑或用更长的窗口。

## 21 "根治 launch 开销"两条路线的估算（只估算，不实现）

基准（§19.10、§20.2，四块 checkpoint，每张源卡每次更新）：
- VJP：不剖析时约 7–9 s，处理 1.5–2.2 万帧，11–15 万个 kernel，GPU 忙 0.5–0.8 s（空闲约 92%）。kernel 数 ≈ 600 × 编码块 + 100 × Writer 块 + 330 × backward 调用；编码块约 130 个、Writer 块约 170 个、调用约 75 次。
- 编码：约 2.2–2.6 s，每块约 115 帧、约 260 个 kernel，GPU 每帧约 14 µs（同样约 90% 是 host 开销）。

### 21.1 a）编码与反传都在固定形状桶上用 CUDA Graph 捕获（含 checkpoint 重算）

- **做法**：现有 autograd 图不能直接用 CUDA Graph 回放。`make_graphed_callables` 的保存张量放在图的私有显存池里，下次回放会被覆盖，而我们要同时保留几十到上百个块的图，到下一个更新才反传。所以要改成"函数式重算 VJP"：
  - 前向图（不建 autograd）只输出 u / y 和每块的 Spatial 输入，拷出来按块保存。这与现在 checkpoint 的保存内容相近，约 36–50 KiB/帧。
  - VJP 图在捕获区内"从保存的输入重算前向 + backward"。参数先拷进静态缓冲，梯度累加进静态梯度缓冲。
  - 每个（桶形状）一张图，不同参数版本共用；回放前把该版本参数拷进静态缓冲（每次几 MB）。
- **预计**：
  - kernel：每块从约 600（VJP）/ 260（编码）个 kernel 变为 1 次图回放，加上每次回放的参数拷贝和输入输出拷贝，约 5–10 个。
  - VJP 时间：GPU 上每帧约等于"一次前向 + 一次反向"，按编码 GPU 14 µs/帧估约 40–50 µs/帧，1.5–2.2 万帧约 0.7–1.1 s，加上桶填充浪费（约 +20–30%）约 1–1.5 s。现状 7–9 s，下降约 80–85%。
  - 编码时间：每帧 GPU 约 14 µs，加填充与拷贝，约 0.3–0.5 s。现状 2.2–2.6 s，下降约 80%。
  - 学习器上的意义：源卡这两项合计从约 10 s 降到约 2 s，编码等待和 VJP 都不会再是瓶颈。届时学习器剩下 PPO（约 2.3 s）、记录批（约 1.1 s）、重算（约 0.6 s）、host 计划等，估计约 5–6 s，低于采集的 7.5 s。这是估算，未实测。
- **显存峰值**：
  - 每块保存量与现在全部 checkpoint 相当或更少（不留 autograd 元数据），约 40–50 KiB/帧。
  - 每张卡多出静态池：最大桶约 1024 帧 × 完整图约 340 KiB/帧 ≈ 340 MiB，几个桶共享同一个池时约 0.5–1.5 GiB。
  - 共享 actor 的卡要算进这部分；总体与现在四块 checkpoint 相近。
- **要改的模块**：
  - `tok_policy` 的源编码路径：每帧的 token 规划、实体与组件的动态去重（`StepComponents` / 组件图缓存）、varlen flash 的 `max_seqlen`，都要改成桶内固定形状、无 host 同步。
  - `source_pipeline` 与 `source_worker`：块的保存从 autograd 图改成"保存的输入张量"；`_backward` 改成按桶回放；Writer 块同样处理；IPC 导出不变。
  - horizon 的 detach / retire 簿记：要释放的从图变成保存的输入，逻辑可沿用。
  - 门测：新增"函数式 VJP 对 autograd 参考"的逐块梯度比较，和现有审计 / H=0 用例。
- **工程量**：约 15–25 个工作日。编码改为可捕获约 5–8 天，函数式 VJP + Writer 约 5–8 天，与现有保留、审计、IPC 集成约 3–5 天，测试与长跑约 3–4 天。
- **主要风险**：
  1. 可捕获性：编码路径里任何 host 同步（`.item()`、`nonzero`、按内容决定形状的去重）都会打断捕获，组件去重和实体数可变是最大的改造点。
  2. 数值：全部重算后梯度与现在"四块重算、组件不重算"的差在 BF16 重算量级（§19.6 测到 checkpoint 对完整图 7.8e-4）。门限应沿用 checkpoint 门测的口径，而不是 1e-6；H=0 与 v2 的逐位等价只能在默认关时保持。
  3. 显存池：一次 VJP 内各桶顺序回放、共用一个池，不能并发。
  4. 参数版本：回放前的拷贝必须与图顺序严格对齐（同一 stream），否则会用错版本。

### 21.2 b）多局合并的编码块（约 1024 帧）

- **做法**：`schedule()` 把同一次调度里同一张卡上各局的新帧合进一个编码块（现在只对无状态教师记录这样做），块大小上限约 1024 帧。块的生命期按"其中任一局仍活着 / 仍可达"判断（`retain_histories` 和 `_detach_horizon` 已支持多 uid 块）。Writer 块也可同样按卡合并（可选）。
- **预计**（按 §20.2 回归）：
  - 每次 VJP 的编码块从约 130 个降到约 15–22 个，kernel 少 600 × 110 ≈ 6.6 万（约 −50%），节点少约 1.1 万。
  - 与 merge 1（调用约 75 → 1，再少约 2.5 万个 kernel）叠加后约 3 万个 kernel，约 −75%。按 host 开销与 kernel / 节点数近似成正比，VJP 约 8 s → 2–3 s。
  - 编码：每块 260 个 kernel，块数降到约 1/7，编码约 2.4 s → 0.6–0.8 s。
  - 若 Writer 块也合并，再少约 1.5 万个 kernel。
  - 但只要编码仍排在同进程的 VJP 后面（§20.3），这些节省要和解耦（双进程或批量 RPC）一起才能进到学习器上。
- **显存峰值（主要代价）**：
  - 同一 rollout 窗口内各局步伐一致，horizon 释放大体同步，这部分影响小。
  - 局结束后，本可立刻释放的帧要等同块其他局，约多留 1–2 个 rollout。
  - **最大风险是记录**：一条等待中的记录要保留它祖先 [L − H, 共享帧数) 的图。现在只钉住该局自己的块；合并后会钉住整块（可能是 8 局的帧），等待中的记录（表 7 里 1.2–3.9 万帧）可能被放大数倍。按每帧 50 KiB 估，最坏可多出 10 GiB 以上。需要配套规则：只合并没有记录引用的 rollout 帧，或者在有记录等待时拆块。
- **要改的模块**：`source_pipeline.schedule`（分组）、`_detach_horizon` 与 `retain_histories`（多 uid 块的判定已有，要补"记录引用的帧不合并"）、Writer 分组（可选）、`source_worker` 不变；新增保留 / 审计的失败用例（多局块下的迟到记录、局在块内先结束）。
- **工程量**：约 5–8 个工作日（不含 Writer 合并的 2–3 天）。
- **主要风险**：
  1. 保留粒度变粗导致的显存放大，尤其是记录，见上。
  2. 正确性方面风险较低：多留只会让审计更安全，不会丢梯度；但"只在全部可达区间都离开后才 detach"的判定要对多 uid 块重新验证。
  3. 编码批次形状变了，BF16 GEMM 的分块可能不同，输出不是逐位相同，门限要用 checkpoint 门测的口径；默认关时 H=0 逐位不受影响。

### 21.3 对比

| | a）CUDA Graph | b）多局编码块（+ merge 1） |
|---|---|---|
| 预计 VJP | 7–9 s → 1–1.5 s | 7–9 s → 2–3 s |
| 预计编码 | 2.2–2.6 s → 0.3–0.5 s | 2.2–2.6 s → 0.6–0.8 s |
| 显存 | 与现在相近，每卡另加 0.5–1.5 GiB 静态池 | 记录引用会被放大，需要配套规则 |
| 工程量 | 15–25 天 | 5–8 天（+ Writer 2–3 天） |
| 主要风险 | 编码路径能否捕获（组件去重、可变实体数）；数值门限要放宽到重算量级 | 记录钉住整块的显存放大 |
| 前提 | 不依赖解耦：编码和 VJP 都变短，同进程排队的影响随之变小 | 仍需先解耦（§20.3），否则省下的 VJP 时间会转成编码等待 |

### 20.6 两项解耦实验（4 卡，四块 checkpoint，均衡开，25 游戏小时，第 8–20 行；每组 2 次，报两次的中位与范围）

- 1）每张源卡两个源进程（`--source-devices cuda:0,cuda:0,cuda:2,cuda:2,cuda:3,cuda:3`），各自持有、各自反传自己编码的局，加 merge 1 + fused。管线早已支持同卡多进程（预算按每卡进程数平分），不需要改代码。
- 2）在 1 的基础上加 `--source-encode-batch 1`（66d9521 / 1c4fc72）：一次调度里发给某个源进程的全部编码合成一次 RPC，该进程按原顺序编码，返回列表；每块仍有自己的 future。编码相同、次序相同；进程模式的 CPU 用例与不合并的结果一致，编码请求数 24 → 8。CPU 回归不变，H=0 逐位等价通过（v3r14）。
- 门测 6/6 通过；6 次运行都是退出码 0、无 OOM、dropped_grad 全为 0。

| | 默认（3 个源进程） | 1）双进程 + merge 1 + fused | 2）+ 批量编码 RPC |
|---|---|---|---|
| x_real_time（两次） | 303.7 / 313.7 | 341.2 / **381.6** | 337.4 / 334.0 |
| 学习器 update_s（两次） | 10.92 / 11.57 | 10.77 / **9.28** | 10.20 / 10.63 |
| 墙钟（两次） | 13.65 / 13.98 | 13.32 / 11.35 | 13.02 / 12.69 |
| 等源编码（两次） | 3.84 / 4.80 | 3.60 / 1.66 | 2.27 / 2.37 |
| 等 VJP 提交（两次） | 1.08 / 0.00 | 0 / 0 | 0 / 0 |
| 记录批 teach_build（两次） | 0.98 / 1.20 | 1.44 / 1.80 | 2.15 / 2.12 |
| 源卡峰值（物理卡 3 / 4 / 5，nvidia-smi MiB ÷ 1000） | 12.4 / 12.0 / 7.3；11.9 / 10.6 / 7.5 | 13.0 / 11.7 / 8.8；11.1 / 10.6 / 8.1 | 11.6 / 11.7 / 8.3；12.0 / 11.2 / 8.8 |
| 容器内存峰值（GiB） | 60.9 / 59.8 | 63.5 / 63.8 | 63.2 / 64.9 |

- **吞吐**：双进程两次都比默认快（x 中位约 361 对 309，约 +17%），但两次之间差别很大（341 / 382）。等编码在其中一次降到 1.7 s，另一次 3.6 s，与默认相近。
- **学习器时间**：双进程 10.8 / 9.3 s，加批量 RPC 10.2 / 10.6 s，两组中位都在 10 s 上下（10.0 / 10.4）。批量 RPC 让等编码稳定在约 2.3 s，但记录批变慢（2.1 s 对 1.0–1.2 s，原因未分辨），学习器没有进一步变短。
- **代价**：多出的 3 个源进程每个带一份 CUDA 上下文、编译缓存和参数副本。每卡显存峰值的变化在 ±1 GB 内，与单次运行的噪声分辨不开（专用卡 5 两次都略高，+0.8–1.5 GB）；主机内存多约 3–4 GiB。
- **按发起会话定的停止条件**（两项之后学习器仍 ≥ 10 s 就不再在调度层面试）：两组的中位约 10.0 s 和 10.4 s，处在门限上，单次最好 9.3 s。所以调度层面就此停止；若要继续，就要走 §21 的根治路线。双进程 + merge 1 + fused 是目前测到的最快组合（x 约 +17%，单次噪声约 ±10%）；它没在 5 卡上测过，也没跑 40 次。

## 22 暂停时的备份

- rl：`exp/memory-chain-v3` = 83dd678（含本报告到 §21 和全部小结果文件），`exp/memory-chain-v3-timing` = 1c4fc72，`exp/memory-chain-v3-crnbase` = afd843a；均已推送 origin（Lceoliu/AIsaac）。
- 候选仓库：master = 66d9521，a-chain = 1c4fc72（两者 `bridge/python` 逐字节相同）；计时链工作树在 `temp/memory-chain-v3/wt-a`。
- 3090 上的 tar：`rhythmo@10.19.131.132:~/isaac-abplus/archive-memory-chain-v3/memory-chain-v3-20261027-0425.tar.gz`，245,977,559 字节，sha256 `1aec59df144c29bccec776fafbc243b882796ddfa8cede64a14058d9abceb56d`（远端核对一致；不含计时链工作树 `wt-a`，其提交都在 `candidate/.git` 里；第一次 scp 断开后重传覆盖同名不完整文件）。更早的四个 tar 未动。
- 主机：暂停时确认没有我们的容器，也没有我们的残留进程。表 9 的等待脚本已停，表 9 未启动。

## 23 GPT 的 replay-v1 验证（`--source-replay 1` 配合源卡；发起会话 10-27 的步骤 1–5）

### 23.1 提交（候选仓库新分支 `replay-v1`，工作树 `temp/memory-chain-v3/wt-replay`；主线 master 未动）

| 提交 | 内容 |
|---|---|
| 6e49896 | GPT 的补丁原样（`five-replay/DIFF_FILE.patch`，打在 66d9521 上，`core.autocrlf=false`）。文档那一块对 LF 检出打不上，改为从 `five-replay/candidate` 取 `docs/TEMPORAL_ATTENTION.md` 并转成 LF，内容与补丁一致。另外 6 个文件与 `five-replay/candidate` 逐字节相同。 |
| 6eacc63 | 三条保护：replay=1 时强制关闭 `--source-layer-checkpoint`、`--source-checkpoint-blocks`、`--source-saved-8bit`，忽略 `--source-vjp-merge`、`--source-streams`，并打印说明；`horizon_stats` 新增 `source_replay_raw_bytes`（源进程为 replay 保留的原始行 + pending 字节，实测每帧 22,896 字节）；新测试 `tests/test_source_replay_guards.py`（线程、进程）；`tests/test_source_replay.py` 加 `--device/--sources/--flash/--compile/--grad-threshold`，供 CUDA 门测用。 |
| 9e9cf04 | VJP 剖析行里记下这次 VJP 的 replay 批次（编码器 / Writer 批数、重算行数），只做测量。 |

- CPU 回归（76 个测试）：除新增的 `test_source_replay`、`test_source_replay_guards` 通过外与上一轮相同。`test_source_replay_lifetime` 是包装器，需要传测试名，单独运行时按设计报参数缺失。
- **H=0 与 v2 逐位等价（replay 关）**：本地与进程两种模式都逐位相同（replay1）。

### 23.2 门测（6eacc63，门测 GPU 用物理卡 1、4、5，因为 GPU3 当时被别人占用）

**replay 关时的现有门测**：horizon-cuda、horizon-bound、vjp-merge-cuda、attention-timing、actor-cuda、v2-cuda-suite2 全部通过。

**replay 对 stored 的源梯度一致性**（`tests/test_source_replay.py --mode compare`，6 局、4 次源更新、变长 token、晚到分支，micro 8192）：

| 门 | 设置 | 源梯度相对 L2 | logits 最大差 | 门限 | 结果 |
|---|---|---|---|---|---|
| replay-grad-fp32-notf32 | FP32，关 TF32，线程 | **4.2e-7** | 4.8e-7 | 1e-6 | 通过 |
| replay-grad-fp32-notf32-process | 同上，源进程 | **4.5e-7** | 2.4e-7 | 1e-6 | 通过 |
| replay-grad-fp32 | FP32，默认（cuDNN 卷积用 TF32） | 3.2e-5 | 2.2e-6 | 1e-6 | 未通过（TF32 不是 FP32 运算；关 TF32 后通过） |
| replay-grad-fp32-process | 同上，源进程 | 3.2e-5 | — | 1e-6 | 同上 |
| **replay-grad-bf16** | 生产设置：BF16、flash、编译 | **8.4e-3** | 1.6e-3 | 2e-2 | 通过 |

- BF16 门限按 §19.6 的 BF16 噪声底（BF16 对 FP32 的相对 L2 约 1.2e-2）定为 2e-2。**这是 BF16 级别的一致，不是 1e-6 等价**：replay 重新分批以后，BF16 的归约次序不同。
- 所有门测和用例里，审计 `source_horizon_dropped_grad` 都是 0。

**现有门测在 replay=1 下**（用 `test_source_replay_lifetime.py` 包装器强制 replay；严格版失败时再跑数值版 rtol 1e-6）：

| 门 | 严格 | 数值 | 说明 |
|---|---|---|---|
| source-process | 通过 | — | |
| horizon-cuda | 失败 | 失败 | 断言 `u.grad_fn is not None`：检查的是保留图本身，replay 设计上没有长寿命图，不适用 |
| horizon-bound | 失败 | 失败 | 断言保留的带图帧数等于 T；replay 恒为 0，不适用 |
| source-pipeline、source-precision | 失败 | 失败 | 对保留的 u 调 `retain_grad()`；replay 的 u 没有梯度，不适用 |
| vjp-merge | 失败 | 失败 | 梯度比较都通过；失败在"批量 RPC 请求数少于不批量"这一条，replay 下所有变体都强制批量，不适用 |

- CPU 上用同一包装器跑保留与迟到记录的用例：
  - grace（线程、进程）、pending_horizon、pending_release、reach_intervals、raw_release 严格通过；
  - inflight_branch、record_query 只有数值版通过（Reader 输出不再逐位相同，rtol 1e-6 内）；
  - record_retention 不适用（也是 `retain_grad`）。
  - 审计全为 0，与 GPT 的 CPU 结果一致。
- 判断：凡是检验梯度或审计的门都通过；失败的都是检查保留图内部结构的断言，replay 下不适用。等价性的结论是"FP32 在 1e-6 内（关 TF32），BF16 在噪声底内"。

### 23.3 replay 模式的 kernel 剖析（9e9cf04，`r10p`，4 卡，均衡开，micro 8192，第 10 次 VJP；编码 200–239 次）

| | replay cuda:3 | replay cuda:2 | replay cuda:0 | 保留图（§19.10 / §20.2 的范围） |
|---|---|---|---|---|
| VJP 墙钟（剖析下） | 5.25 s | 9.13 s | 12.19 s | 12–18 s |
| 本次 VJP 的编码块帧 / 重算行数 | 14,755 / 14,755 | 13,622 / 27,457 | 31,731 / 49,647 | 1.1–2.2 万（不重算） |
| 编码器重算批 / Writer 重算批 | 3 / 2 | 7 / 6 | 9 / 8 | — |
| kernel 数 | **3,754** | **8,414** | **11,565** | 9.2–15.3 万 |
| 每千帧 kernel 数 | **254** | **618** | **364** | 6,100–9,800 |
| kernel 时长 均值 / 中位 | 48.9 / 3.0 µs | 44.8 / 2.7 µs | 58.7 / 2.9 µs | 约 5 / 2.3 µs |
| GPU 忙 / 活动窗口 | 0.26 / 4.79 s | 0.53 / 8.50 s | 0.95 / 11.42 s | 0.5–0.9 / 7–11 s |
| **GPU 空闲占比** | **94.6%** | **93.7%** | **91.7%** | 91–93% |
| autograd 节点（每千帧） | 907（61） | 2,195（161） | 2,843（90） | 3.2–7.1 万（约 3,300） |

- **replay 把 kernel 和节点降了一个多数量级**：每千帧 kernel 约 1/15，节点约 1/30，平均 kernel 时长约 10 倍长。但 **GPU 空闲占比没有变**，仍是 92–95%。VJP 的时间从"launch 开销"移到了 host 上的别处：按 micro 8192 帧把原始行重新组批（`to_batch`、token 规划、H2D 拷贝，拷贝本身 0.1–0.3 s），再做一次带梯度的前向。剖析只记了 GPU 侧，host 上的具体分布还没进一步拆分。
- 重算行数可以超过本次 VJP 的编码块帧数（cuda:2、cuda:0）：Writer 的梯度会落到没有直接 Reader 读取的旧编码行上，这些行也要重算。
- 编码（无梯度合批）：干净的那段（cuda:3，7,121 帧、40 次编码）每千帧 205 个 kernel，保留图约 2,300 个。另外两段的剖析窗口里混进了 VJP 的反传 kernel，不计入。

### 23.4 4 卡短跑（9e9cf04；表 7 配置但不开 checkpoint，均衡开，并发关，每 rollout 一次源 Adam，H=1024；各两次）

| | stored ①（无 checkpoint） | stored ② | replay ① | replay ② | 参照：stored + 四块 checkpoint ①（1c4fc72 t9b） | 参照 ② |
|---|---|---|---|---|---|---|
| 结局 | **第 17 行 OOM**（退出码 1） | **第 19 行 OOM**（137） | 22 行正常 | 22 行正常 | 22 行正常 | 22 行正常 |
| dropped_grad | 0 | 0 | 0 | 0 | 0 | 0 |
| x_real_time（第 8–20 行；stored 只有 8–16 行） | 277.1 | 327.7 | 244.1 | 278.1 | 303.7 | 313.7 |
| 学习器 update_s | 13.40 | 11.91 | **15.25** | **13.00** | 10.92 | 11.57 |
| 墙钟 | 16.19 | 13.50 | 17.17 | 15.41 | 13.65 | 13.98 |
| 等源编码 | 5.69 | 4.07 | 4.88 | 4.31 | 3.84 | 4.80 |
| VJP 最慢一张卡 | 7.23 | 8.33 | 13.17 | 6.89 | 12.79 | 11.77 |
| 其中 replay 编码器重算（最慢卡） | — | — | 10.39 | 4.73 | — | — |
| 编码最慢卡 | 2.94 | 3.51 | 2.87 | 2.52 | 3.20 | 3.41 |
| 源卡峰值（物理卡 3 / 4 / 5，nvidia-smi MiB ÷ 1000） | **24.1** / 21.6 / 21.9 | 20.9 / 21.0 / **24.1** | 13.9 / 11.8 / 10.4 | 13.1 / 12.3 / 9.7 | 12.4 / 12.0 / 7.3 | 11.9 / 10.6 / 7.5 |
| 容器内存峰值（GiB） | 58.5 | 59.5 | 62.0 | 61.3 | 60.9 | 59.8 |
| 源进程原始行字节（replay，中位 / 最大） | — | — | 2.88 / 3.93 GiB | 3.00 / 3.68 GiB | — | — |

- 不开 checkpoint 的 stored 两次都在源卡上 OOM，所以这一档没有稳态数据。列出的是 OOM 前第 8–16 行的数，只作参考。
- replay 在显存上做到了它要做的：源卡峰值约 10–14 GB，与四块 checkpoint 相当，远低于完整图；源进程为 replay 多保留约 3–4 GiB 原始行，主机内存约 +1–2 GiB。
- **但 replay 组的学习器（13.0 / 15.3 s）比能跑的 stored 组（四块 checkpoint，10.9 / 11.6 s）慢**，x_real_time 也低（244 / 278 对 304 / 314）。两次 replay 之间差别很大：最慢卡的重算 10.4 s 对 4.7 s，均衡在 replay 下每次更新的卡间差距达 127–160%。

### 23.5 结论：不跑表 10

- 步骤 2（门测）：梯度和审计相关的都通过；FP32 关 TF32 时在 1e-6 内，BF16 在噪声底内。检查保留图内部结构的旧断言不适用。
- 步骤 3（剖析）：kernel 数和节点数降了一个多数量级，但 GPU 空闲占比仍是 92–95%。
- 步骤 4（短跑）：replay 组学习器中位没有低于 replay=0。能跑的 replay=0（四块 checkpoint）更快；不开 checkpoint 的 replay=0 会 OOM。
- 按发起会话的条件（"只有 2、3、4 都通过且 replay 组学习器中位明显低于 replay=0 时才跑表 10"），**表 10 不跑**。
- 原因：replay 解决了显存和 launch 次数，但 VJP 时间没有下降。时间转到了 host 上的按批重建：原始行组批、token 规划、H2D、带梯度前向，加上 Writer 梯度带来的额外重算行。均衡在 replay 下也不稳定。若继续这条线，下一步应先拆分 replay VJP 的 host 时间（例如对 `backward_replay` 做 cProfile），再决定是缓存组批计划还是把 `to_batch` 挪进编码 worker 预先做好。

### 23.6 备份

- rl：`exp/memory-chain-v3` = fa4c58f（报告至 §23.5 与全部小结果文件），`exp/memory-chain-v3-replay` = 9e9cf04（候选仓库分支 `replay-v1`），`exp/memory-chain-v3-timing` = 1c4fc72；均已推送。
- 3090：`~/isaac-abplus/archive-memory-chain-v3/memory-chain-v3-20261027-0700.tar.gz`，255,693,765 字节，sha256 `d03d5720f9d8a9d0b6809b952b1dfd898bd0c28c7644e4c9fa25607670db03f9`（远端核对一致；不含两个工作树 `wt-a`、`wt-replay`，它们的提交都在 `candidate/.git` 里）。
- 主机上没有我们的容器，本机没有残留的链脚本。

## 24 源进程 host 时间的 cProfile 拆分（2c918a6；只测量，不改实现，不开新实验）

### 24.1 做法

- 提交 2c918a6（`replay-v1`，脚本 `scripts/patches/patch_cprofile.py`）：源进程里加两个只读开关。`ISAAC_SOURCE_VJP_CPROFILE=10` 对第 10 次保留 VJP 单独做 cProfile（期间摘掉 service_forward，前后各 `cuda.synchronize`）。`ISAAC_SOURCE_ENCODE_CPROFILE=100:20` 对第 100–119 次编码请求做 cProfile（只在请求内开启，请求之间的等待不计）。每次打印一行 `SOURCE_CPROFILE`，内容是累计和自身时间各前 25 的函数，以及指定函数的累计秒数。
- 两次运行都在 4 卡（物理 3,1,4,5）、18 游戏小时、均衡开、micro 8192、H=1024 下完成，exit 0，结束后主机上没有我们的容器：
  - `r11p` = replay（`--source-replay 1`）
  - `r11s` = stored（replay 0，`--source-layer-checkpoint 1`）
- 原始行在 `results/v3-2c918a6-r11{p,s}-*/train.log`，渲染结果在 `results/cprofile-2c918a6.md`（`scripts/cprofile_table.py`）。
- 打开 cProfile 会让 Python 函数调用变慢，所以本节的 µs 只能用来比较占比，不是真实耗时。这两次的 ×实时（294 / 360）也不作计时结论。
- cProfile 只记录开启它的那个线程。autograd 在 CUDA 设备线程上执行，所以 `run_backward` 是黑盒，里面包括 C++ 引擎调度、checkpoint 重算和等待 GPU。

### 24.2 replay VJP（r11p 第 10 次 VJP；cuda:0 36,493 行 3.172 s，cuda:2 37,798 行 3.405 s）

各项按调用树互斥切分，加起来等于总数（µs/帧，cuda:0 / cuda:2）：

| 类别 | 内容 | µs/帧 |
|---|---|---|
| 行拼接 | `_encode` 自身时间，主要是 `np.concatenate(parts)` 拼原始行（numpy 的 C 函数没有单独计入，这一项是推断） | 14.4 / 17.0 |
| to_batch：原始行解码 | `np.ascontiguousarray(rows[field])`，即结构化数组按字段做跨步拷贝（168/192 次） | 10.9 / 11.3 |
| to_batch：H2D | `to_batch` 里的 `.to(device)`（pageable 内存） | ≈6.0 / ≈5.6 |
| token 规划：布局 | `plan_tokens` + `learner_plan` + `item_count` | 3.4 / 3.1 |
| token 规划：组件去重 | `rebuild_memory.encode` 自身 + `payload_keys`（含 `tolist`）。按原始键给 player/grid/map/items 去重，属于 StepComponents 的簿记，执行位置在带梯度前向里面 | 20.4 / 18.7 |
| 带梯度前向（除去去重） | `encode_rebuild` 其余部分：CNN 和 attention 的 torch/Dynamo 调度（`_cnn` 累计 0.50 / 0.56 s） | 11.4 / 12.6 |
| `_cast_context` | 参数 cast 上下文 | 1.2 / 1.5 |
| backward | `_engine_run_backward`（C++，黑盒） | 11.2 / 11.7 |
| Writer 重算 | `rebuild_memory.write` | 0.3 / 0.3 |
| 其它 Python 簿记 | `backward_replay` 里除上面各项之外的部分（`_packed_indices`、梯度装配、`as_tensor` 小拷贝等） | 6.8 / 7.7 |
| **合计** | | **86.9 / 90.1** |

- 其中"原始行 → 张量"（to_batch 整体）：17.4 / 17.4 µs/帧。纯解码约 11 µs，H2D 约 6 µs。如果把行拼接也算进来，"原始行到张量"一共约 31–34 µs/帧，占 VJP 的 36–38%。
- 两项最大的成本都不是 to_batch：
  - 组件去重的 Python 簿记：19–20 µs
  - 带梯度前向的模块调度：11–13 µs
  - 后者加 backward 一共 23–24 µs
- 对照：同一个 `to_batch` 在本机不开 cProfile、只用 CPU（合成 2,560 帧，没有 H2D）时是 4.9–6.7 µs/帧，这说明 cProfile 加上主机负载会放大 Python 部分的耗时。
- cuda:3 的第 10 次 VJP 是空的（0 行），不计入。

### 24.3 stored 编码与 VJP（r11s；第 100–119 次编码请求，第 10 次 VJP）

| | cuda:3 | cuda:2 | cuda:0 |
|---|---|---|---|
| 编码帧数（20 个请求） | 4,175 | 2,952 | 2,545 |
| 编码 host 合计 µs/帧 | 156.8 | 330.1 | 398.5 |
| 其中 to_batch | 14.0 | 18.2 | 14.5 |
| 其中 token 规划（plan/learner_plan/item_count） | 6.8 | 12.6 | 8.9 |
| 其中 `_cast_context` | 13.8 | 30.1 | 18.6 |
| 其中 `encode_rebuild`（带梯度前向，含 checkpoint 包装和组件去重） | 107.1 | 239.3 | 336.2 |
| 其中 `ipc_export` | 1.1 | 2.8 | 1.8 |
| VJP 行数 | 16,580 | 19,976 | 16,163 |
| VJP host µs/帧 | 365.1 | 316.5 | 338.0 |
| 其中 `run_backward`（黑盒） | 363.1 | 314.9 | 336.1 |

- stored 编码每个请求只有约 130–210 帧，所以每次调用的固定开销被摊到很少的帧上，µs/帧不能直接和 replay VJP 里每次约 5,200 帧的重建比较。cuda:3 每个请求约 33 ms。
- cuda:0 的 `encode_rebuild` 里有一次 Dynamo 重新编译（`convert_frame._compile` 0.47 s，占这 20 个请求的 47%）。这是一次性的，不属于稳态。cuda:3 没有重编译：模块调度、Dynamo/AOT 包装和 checkpoint 包装合计约 0.44 s / 4,175 帧，组件去重本身约 10 µs/帧，因为每个请求的帧少、键也少。
- stored VJP 的 host 时间几乎全部是 `run_backward`（C++ 引擎，加上 layer checkpoint 在设备线程上的重算），Python 侧可以忽略。
- replay 模式的无梯度编码（缓存原始行的那一步）没有采到：r11p 在 18 小时里批量编码请求不到 100 次，日志里只有 stored 的编码行。它的组成和 24.2 里重建的前向部分相同，但不带梯度。

### 24.4 to_batch 输出缓存的每帧字节

- 在本机用 `to_batch(entities=64)` 测输出：24 个张量，共 **22,818 B/帧**。主要是 `ent`（float32 [64,56]）14,336 B、`grid` 3,136 B、`inv_all` 2,048 B、`map` 1,352 B。
- 原始行是 22,856 B/帧（另有 pending 64 B）。所以缓存补零后的张量和缓存原始行占用相同。如果按 `n_ent` 截断实体，`ent` 这部分会随实际实体数缩小，但这次没有测实际的 `n_ent` 分布。
- r11p 每个源记录的 `source_replay_raw_bytes` 峰值是 0.59–1.37 GB。把同样大小的张量放到 GPU 上缓存，每个源大约多占这么多显存。省下的是"行拼接 + to_batch"，约 31–34 µs/帧（replay VJP 的 36–38%）。组件去重、模块调度和 backward 不变（合计约 50 µs/帧）。
- 如果改成向量化/native 的行解码，只能省"解码 + 拼接"这一段（约 25–28 µs/帧；H2D 约 6 µs 还在，除非改用 pinned 内存）。上限和缓存方案接近，但不需要额外显存。
- 以上两种方案都不会碰到最大的单项，也就是组件去重的 Python 簿记（约 19–20 µs/帧）。这一项应该另外评估。

### 24.5 备份

- 见本节提交后的 `exp/memory-chain-v3`（报告至 §24、`results/cprofile-2c918a6.md`、两次运行的小结果文件）和 `exp/memory-chain-v3-replay` = 2c918a6。3090：`~/isaac-abplus/archive-memory-chain-v3/memory-chain-v3-20261027-0741.tar.gz`，257,417,663 字节，sha256 `81be994724772c6c8eef81019882319cfe8fcb01f4e567df1456ac76a7716c4d`。远端 `sha256sum -c`、`gzip -t`、`tar -tzf` 都通过，共 8,417 个条目，包含 §23、§24、`candidate/.git` 里的 `replay-v1` 分支（2c918a6）以及 r11p/r11s 的小结果文件。和以前一样不含 `wt-a`、`wt-replay` 两个工作树，旧 tar 都保留。tar 里的 REPORT 还是写这一句之前的版本。

## 25 估算：改为 rollout 内截断梯度（用户 10-30 决定；只做估算，不改代码）

本节的依据是读候选主线 66d9521 的代码，以及 §15、§17、§24 的实测数。所有时间和工作日都是估计，没有新的测量。

### 25.1 设计落到代码上的样子

- **推理不变**：actor 的 `ActorMemory` / `TensorActorMemory` 照旧维护每个槽的 `recent`（最近 256 帧 u）和 `mem`（已完成块的 4 个 token），Reader 读整局记忆。
- **训练（新）**：学习器把每个 worker 的 512 个决策当作一个块，相当于把旧架构的 `memory_chunks` / `temporal_bounds` 的 T 从 256 改成 512。一个 minibatch 8192 行等于 16 个 worker 的完整 rollout，按 micro 8192 一次处理。块内：
  1. Spatial `encode_rebuild` 带梯度编码 512 帧，得到 u、y、pad。这和旧架构学习器的 Spatial 前向加反传是同一份工作量。
  2. Writer 对所有在本 rollout 内闭合的块带梯度重算。跨边界块的输入是 [rollout 前的 u 常数 | rollout 内带梯度的 u]，所以 rollout 内的那部分帧也能收到 Writer 梯度。
  3. 每个决策的 Reader bank = [零向量, 最近 256 帧（rollout 前为常数，rollout 内带梯度）, 已闭合块（rollout 前闭合的为常数，rollout 内闭合的带梯度）]。用 `grouped_layout` 做法，每个 worker 一个共享 bank，按决策加因果掩码。
  4. 接 PPO 损失。
- **常数从哪来**：actor 每个决策随 rollout 回传本帧 u（D=256，bf16 512 B），块闭合时再回传 4 个 mem token。学习器从 rollout 流里自己维护每个 worker 的携带状态（滚动 256 帧 u、本局全部 mem、本局帧号），遇到 `first` 就清零。
  - 旧架构有先例：它已经用 CUDA IPC 回传每个决策的 `t_kv` / `t_o`。
  - 回传量约 16 MB/rollout，现有原始行回传是 750 MB/rollout，可以忽略。
  - 这样 rollout 前的常数和 actor 当时读到的值逐位相同。
- **位置与对齐**：位置编码用本局帧号，现在由 actor 的 `h.length` 给出，学习器从携带的帧号加块内偏移得到。块按本局帧号的 64 对齐，和 rollout 边界无关，所以一般会有一个跨边界块。

### 25.2 代码范围与工作日

**退出主路径**（首个里程碑只把它们断开，不删文件；删除放到 M2，降低回归风险）：
- `source_pipeline.py`（1453 行）、`source_worker.py`（485 行）、`source_replay.py`（只在 replay-v1 上）
- `SourceBatch` 的 pipeline 分支与 `JoinedSources` / `SourceView`
- `HorizonAttention` 的 `grad_valid` 路径、`retain_histories` / `horizon_frontiers`、版本化图、detach / 宽限期
- `RecordLedger` 的可达区间规则、学习器侧的规则 5 原始字节释放
- `--source-*` 一族开关（均衡、merge、fused、streams、encode-batch、replay、MPS）
- `train_tok` 里等源和交接的分段（`u_rc_sources` 等）

**保留**：
- `TokPolicy.encode_rebuild`（Spatial 与组件去重）、`MemoryReader`（Writer/Reader 模块不变）
- actor 侧全部：`RebuildActor`、`ActorMemory` / `TensorActorMemory`、CRN 修正、A1/A2/A4/A1b 优化
- teacher v2 / CF 的车道与记录机制、PPO / CF / 教师损失、rollout 存储、`memory_chunks` / `temporal_bounds`

| # | 新写或改动 | 内容 | 工作日 |
|---|---|---|---|
| N1 | actor 回传 u/mem | 两个 actor 进程，含 `TensorActorMemory` 快路径；块闭合时附带 mem；车道（v2/CF）同样回传 | 1.0–1.5 |
| N2 | 学习器携带状态 | 每个 worker 的滚动 256 帧 u、本局 mem、帧号；`first` 清零；`--resume` 时各局从头开始，所以不进 checkpoint | 0.5 |
| N3 | rollout 内批量前向与反传（`TruncatedBatch`） | 512 帧块计划；Spatial 带梯度；Writer 含跨边界块；grouped bank 与因果掩码；块内开新局 | 2.0–2.5 |
| N4 | 学习器起点的 value 重算 | `--recompute-values` 改用 N3 的无梯度前向 | 0.5 |
| N5 | 记录批（教师 / v2 / CF） | 旧教师记录本来就是空历史（`teach_contexts` = `RawHistory.empty()`），不用改。v2/CF 记录：M1 先用学习器无梯度重编码它们的原始上下文得到常数（复用 `SourceBatch` 非 pipeline 路径加 `no_grad`，只在被采样时付费）；M2 换成每局的 value 日志（u 每帧 512 B + mem 每帧 32 B，按 uid 共享前缀，CF 分支在分叉点共享），由车道回传 | M1 0.5，M2 1.5–2 |
| N6 | `train_tok` 主路径切换 | 新开关（如 `--memory-train truncated`），不构造 SourcePipeline；progress.csv 换成新统计；退出码与守护不变 | 1.0 |
| N7 | 审计 | "窗口外梯度恒为 0"：常数在构造时就不带梯度；测试里用 hook 断言常数的 grad 是 None，窗口内的源 grad 非零 | 0.5 |
| N8 | 测试（见 25.3） | CPU 全套与 CUDA 门测 | 2.0 |
| N9 | 部署与长跑 | 短跑一次，40 次稳态长跑一次 | 1.0 |
| M2 | 清理 | 删除源路径代码与开关，旧测试归档 | 1.0 |

合计：M1（到首组 40 次长跑）约 **9–10 个工作日的工作量**。按下面 25.6 的顺序压缩并行，日历上约 **5 天**，前提是不出现大的意外。M2 再加 2.5–3 天。

### 25.3 语义与验收

**有效梯度窗口**：设 k 为决策距本 rollout 起点的步数；如果本局在 rollout 内开始，则取距本局起点的步数。
- 新做法：k ∈ [0, 511]。局长大于 rollout 时 k 均匀分布，均值约 256；局短于 512 时均值更小。
  - 直接读的 256 帧里，有 min(k, 256) 帧带梯度。
  - 经 Writer，rollout 内闭合的块（含跨边界块 rollout 内的部分）带梯度，最远约 k 帧。
- 旧架构（843×，`--temporal-bptt 256`，`memory_chunks`）：每个 worker 按 256 切块，块前的 K/V 和 o 是 actor 存的常数，所以 k ∈ [0, 255]，均值约 128。
- 结论：新做法平均梯度距离约为旧架构的 2 倍，最远 511 对 255。rollout 前的帧照样可读，只是没有梯度。

**必须通过的测试**：
1. **actor 与学习器一致**：同一参数、同一历史，跨至少 3 个 rollout，覆盖块内开新局、块恰好在 rollout 起点闭合、跨边界块、256 窗口跨 rollout 边界。学习器批量输出与 actor 逐步输出：FP32（关 TF32）≤1e-6，BF16 在噪声底内。这一项替代原来的"H=0 与 v2 逐位等价"。
2. **未来帧不泄漏**：改动 t' > t 的帧，t 的输出逐位不变。
3. **窗口外梯度恒为 0**：rollout 前的常数梯度为 None 或 0；窗口内的帧和跨边界块的 rollout 内部分，梯度非零。生产上的审计从 `source_horizon_dropped_grad` 改成"常数带梯度的个数 = 0"。
4. **携带状态正确**：学习器从 rollout 流重建的每个 worker 状态，与 actor 在 rollout 起点的 `ActorMemory` 逐位相同。
5. **记录**：v2/CF 记录的 bank 与产生它的车道在记录时刻的 bank 一致。M1 的重编码常数用当前参数算，只能做到 FP 容差，要单独说明。
6. **延迟线索测试的窗口内版本**：线索距离 d < 512 时能学会，并区分直接读和 Writer 两条路；d 超出 rollout 起点时，读得到但梯度为 0。
7. 原有测试中与源路径无关的部分全部重跑，并补 CUDA 门测。

**H=0 等价不再适用**：新做法的前向（rollout 前的常数来自 actor 当时的参数）和梯度（截断）都和 v2 不同，所以不再以 v2 为参照，改用第 1 项"actor 与学习器一致"。

### 25.4 吞吐估算（尚未测量）

每次更新约 4,100–4,400 游戏秒（843.4×5.22 s、337×12.2 s 两种算法）。

| 学习器分段（s/次更新） | 旧架构实测（§15） | 新做法估算 | 依据 |
|---|---|---|---|
| value 重算 | 0.90 | 1.1–1.3 | 加一遍 Reader 无梯度前向：约 Reader 2.6 s ÷ 3 epoch ÷ 前反约 3 倍 |
| PPO（Spatial 前反 + 时序 + 头） | 2.93 | 4.6–5.1 | 去掉旧时序 transformer（0.5–1.0，没有单独测过），加 Reader 前反 2.63（§15 的 v3 实测 GPU 流时间） |
| Writer 重算与反传 | — | 0.1–0.2 | 每个 minibatch 128 块，计算量很小，主要是 launch |
| 记录批 | 0.32 | 0.4–0.6 | 记录自己的帧带梯度，bank 是常数；v2 被采样时加重编码 |
| gae / batch / opt / tail | 0.48 | 0.5 | |
| **合计** | **4.56** | **约 6.7–7.7** | |

- 显存：Spatial 激活按 micro 8192 分批，与旧架构学习器相同（旧架构 3 卡运行的 gpu_reserved 15.5 GB）。Reader 的 bank 已经在 v3 学习器上跑过。估计单卡 24 GB 够用。
- 主机内存：学习器不再持有历史原始行（22.8 KB/帧），改为 value 日志（约 0.5 KB/帧），§9–§14 的主机内存问题大部分消失。
- **3 卡布局**（学习器 1 卡，actor 2 卡）：采集 7.3 s（现上限），学习器 6.7–7.7 s，并行重叠。墙钟约 7.3–7.9 s，即约 **520–600×**。
- **5 卡布局**：多出的 2 卡给 actor（每卡一个 actor 进程）。§17.4 显示 actor 已开始受 GPU 限制，采集可能降到约 6–7 s，但学习器还是单卡 6.7–7.7 s，墙钟约 6.7–7.7 s，即约 **530–660×**。要再往上，需要学习器多卡数据并行（另估约 2 天）或把记录批移到第二张卡，不在 M1 范围内。
- 以上都是估计。M1 的短跑会先给出实测拆分，再决定 5 卡怎么分配。

### 25.5 风险

1. **rollout 边界效应**：rollout 起点附近的决策梯度窗口接近 0。这和旧架构每 256 帧的块起点一样，是截断的固有性质。跨边界块：按 25.1 的混合输入重算，rollout 内部分有梯度。另一种做法是把它整体当常数，更简单但丢掉一部分梯度，默认不用。帧号和块号差 1 的错误由测试 1、4 覆盖。
2. **常数的版本**：rollout 前的 u/mem 是 actor 当时的参数算的，长局里可能落后很多次更新。这和 v3 里 horizon 外的语义、旧架构的 K/V 常数一样。Reader 的 K/V 仍按当前权重投影（与 actor 的 refresh 相同）。
3. **记录批**：去掉源路径后，v2/CF 记录需要历史常数。M1 用学习器无梯度重编码。按 §18.10，测速窗口里 v2 几乎不被采样，所以对测速影响小，但被采样时会有尖峰（长上下文的 Spatial 前向）。M2 改为 value 日志后消失。CF 分支的前缀共享在 value 日志里需要 fork 语义（RawHistory 已有先例）。
4. **显存**：rollout 内全部帧的 Spatial 激活按 micro 8192 分批，与旧架构相同；Reader 每个 worker 一个 bank，最多 256 + 512 帧加 mem token。估计风险低，M1 短跑会确认。
5. **训练效果未知**：梯度从"rollout 外 horizon 1024"缩到"rollout 内 0–511"，对学习的影响要由长跑和评测决定，这里不做结论。

### 25.6 实施顺序与里程碑（预计 3–5 天出首组 40 次长跑）

1. **第 1 天**：N1 + N2 + 测试 4（携带状态逐位一致），CPU。
2. **第 2–3 天**：N3 + N4 + 测试 1、2、3（CPU）。N7 审计同时完成。
3. **第 4 天**：N6 主路径切换，N5 的 M1 版本（v2 重编码），CPU 全套回归，CUDA 门测（只用空闲卡，遵守原有的 GPU 规则）。
4. **第 5 天**：3 卡 25 游戏小时短跑（拆分学习器各段，核对 25.4），然后 40 次稳态长跑，报 x_real_time 与游戏小时、审计、显存、容器内存。
5. **之后（M2，约 2.5–3 天）**：v2/CF 的 value 日志、延迟线索窗口内版本、5 卡布局、删除源路径代码。

每一步完成后推送备份，生产版本不替换，新开关默认关。

## 26 实施 M1：rollout 内截断梯度（候选仓库分支 `truncated-v1`，工作树 `temp/memory-chain-v3/wt-trunc`；主线 master 未动）

新开关 `--memory-train truncated`，默认 `source`，默认下行为不变。

### 26.1 N1 + N2（提交 dfc7a02）

- **rollout 带上 actor 的常数**：
  - `t_s` 改为 float32，逐位保存 actor 写进环的 u。
  - 新增 `t_m [rows, tokens, D]`：某个决策闭合一个块时，存下 actor 算出的 Writer token。
  - 携带的上下文行数 P = W = 256（原来是 W − 1，因为 Reader 读的是当前帧之前的 256 帧）。
  - 图 actor 通过 control 第 21 行写 `t_m`，eager actor 通过 `mtok` / `mclosed` 写。`Rollout.carry` 把未结束决策的 `t_m` 一起带过去。
- **交到学习器**：IPC 张量（`rollout_tdev`）、学习器镜像（`mirror_temporal`）、双 actor 合并（`merge_rollouts`）都带上 `t_s` / `t_m`。
- **`isaac_bridge/truncated_memory.py`**：
  - `LearnerCarry`：每个 worker 当前局在下一个 rollout 之前已闭合块的 token，由 rollout 自身累积；只延续 rollout 里的最后一局。
  - `worker_context`：携带行里的 rollout 前 u。
  - 两者都计不一致数（`mismatch`），必须为 0。
- **测试 4**（`tests/test_truncated_carry.py`，CPU，eager actor）：3 个 worker、40 步、含两次中途开新局、3 次交接。学习器重建的 rollout 前 u 和 Writer token，与 actor 在该决策时刻的状态**逐位相同**，9 个上下文全部一致，mismatch 0，退出码 0。
- **回归**（CPU）：`test_temporal_smoke`、`test_rebuild_memory_smoke`、`test_learner_device_smoke`、`test_no_gru_smoke` 全部退出码 0。
- **还没验证**：图 actor（CUDA）的 `t_m` 写入路径和双 actor 合并，放到第 4 天的 CUDA 门测。
- **备份**：rl 分支 `exp/memory-chain-v3-truncated` = dfc7a02，`exp/memory-chain-v3` = dc89a42，均已推送。3090 上新增 `~/isaac-abplus/archive-memory-chain-v3/memory-chain-v3-20261030-0859.tar.gz`，386,645,221 字节，sha256 `1e19a721d504dba5bd6e461f87412637c201dc6f88894ad954da6c8dbdb26a9f`，远端 `sha256sum -c` 和 `gzip -t` 都通过，旧 tar 未动。tar 不含 `wt-*` 工作树，它们的提交在 `candidate/.git` 里。tar 比上一份大，主要是目录里的 `five-replay`（553 MB 未压缩）。
