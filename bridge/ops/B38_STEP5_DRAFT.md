### B38 第五步：第二个 actor 进程（`--actor-processes 2`）（2026-10-17，代码完成；4 卡吞吐待测）

**实验目的**

第四步之后学习器一次 update 1.94 s，周期由 collect 决定（C99 全标志 collect 约 2.8 s；第三步测得 collect 是 actor 进程主线程的主机工作限速：调用、记账、写 rollout、轮询、回复、lane 服务合计约 2.75 s / 2.84 s，worker 一半时间在等）。第五步按 3b(ii)：第二个 actor 进程管一半 worker，各自的主机工作减半，周期约 max(collect / 2, 1.94 s)，64 worker 目标 2,000× 以上。

**实验方法**

- **分支** `b38-actors`（从 abplus-training `0f6f88d`；`c6d767c` 主体，之后几个修正提交）；opus 子代理实现。
- **代码**（`train_tok.py`）：
  - `--actor-processes 2`（需要 `--learner-process 1`，worker 数和 rollout 能被整除；默认 1 = 原来的路径不变）、`--actor-devices cuda:0,cuda:2`。训练器进程是第一个 actor（aid 0），它 spawn 学习器进程和第二个 actor 进程（`actor_child_entry` → `main(argv, actor=...)`）。每个 actor 进程：自己的采样器（`--workers / 2` 个 worker）、GraphActor、lane 服务、温度状态、局记账、搜索槽位（`--teacher-slots` 平分），在自己的卡上；每次交接 `--rollout / 2` 个决策。第二个进程的 worker 用自己的种子（`--seed + 1000003`）、实例名（`<name>a1`）和端口（`--port + 第一个 worker 的全局编号`），采样的随机数种子也偏移（否则两边的动作抽样完全相同）。
  - **rollout**：两个 actor 共用一套"全部 worker"的共享内存主机数组（第一个 actor 用 memfd 建好，`full_from_local`；第二个映射后取自己那一半行，`slice_into`），各自的 rollout 对象就是这些行的视图，所以学习器不用拼主机数组。温度的设备张量（K/V、o 的行池）各在各的卡上；学习器在配对两半时把两个池按顺序拷进一个镜像（`merge_rollouts`：K/V 行先 A 后 B，o 行先放两边的"带过来的房间源"、再放两池的行，保持"o 行 = K/V 行 + t_obase"），并把 B 那一半的 `t_row` / `rooms_src` 在主机副本上加上偏移（共享数组本身不动）。于是 `learn()` 完全不变，看到的是一个 64 worker 的 rollout。
  - **作业**：每个 actor 给自己的半份作业标上 aid 和交接序号，发到学习器的同一个队列；学习器按序号配对，两半都到才更新（`merge_jobs`：局列表、教师 / teacher-v2 / cf 记录按 actor 顺序拼接，worker 计数和调用统计相加，worker 统计行上下叠，collect / 周期 / 等待秒取较慢的那个，另记每个 actor 的 collect 秒 `collect_actors` 和显存 `gpu_reserved_gb_actors`）。两个 actor 做同样次数的交接（多 actor 时循环按交接次数结束，单 actor 时与原来一样按决策数），所以每一半都能配对。
  - **权重**：学习器把权重字节写进每个 actor 卡上各自的 IPC 缓冲（第二个 actor 的缓冲随它的启动消息到达），各自的消息和事件；actor 拿权重的逻辑（`take_weights`、`--overlap`、`busy`、`version`）不变，所以落后一份 rollout 的语义不变。
  - **结束与故障**：正常结束时第二个 actor 交完最后半份后置"已交完"事件并暂停 worker 等待；第一个 actor 等到这个事件才给学习器发停止消息，学习器做完最后一次更新、存 `last.pt`、回 `done`，然后第一个 actor 叫第二个结束。学习器出错 → 两个 actor 都收到错误、停下；第二个 actor 死掉 → 第一个 actor 报错（`the actor process 1 ended`）并正常收尾；第一个 actor 死掉 → 第二个发现父进程消失而退出、学习器存 `last.pt`；SIGINT 发给整个进程组，各自收尾。
  - 拒绝：分支、build lab、学习器 bench（与第二步相同）。
- **测试**：新 `tests/test_actor_processes_smoke.py`（CPU，3 项全过）：(a) 两个 actor 通过各自的 rollout 对象写入的共享全量数组 == 两个独立 rollout 的数组，逐位（两个缓冲、每个数组，含池增长）；(b) 合并后的 rollout 上学习器的分块前向 == 各 actor 自己 rollout 上的，每个决策（logits、value、value_global、GRU 状态；CPU 上差 0），拷贝字节数正确；(c) `merge_jobs` 的拼接与加总。`test_lane_batch_smoke` 的源码检查改了一处定位（循环条件变了）。其余冒烟测试（共 13 个文件）在主机 A GPU 2、3 上全过。
- **真机**（主机 A，GPU 2、3，不带搜索，负载守护 90；C99 在 GPU 0、1）：只有两张卡，两个 actor 只能共用 GPU 2（学习器在 GPU 3）——这是测试布局，两个 CUDA 进程分时共用一张卡；冒烟 2×4 worker（q）对 1×8（p）；故障三项（fail5 学习器注入错误、kill5 在第 5 次更新后 SIGKILL 第二个 actor、int5 运行 240 s 后 SIGINT）；正确性对照：2×16 worker（q）对 1×32 worker（p），从 `segment1-final.pt` 续 30 次更新。

**实验结果**

- **冒烟**（0 个 Traceback，5 次更新）：q 277×、kl 0.006、ev 0.69；p 286×、kl 0.006、ev 0.74。`collect_actors` 两边 3.0–4.7 s 相近。
- **故障**：fail5：两个 actor 都以学习器的错误结束，`finished 2 updates`，学习器存了 `last.pt`；kill5：第一个 actor 报 `the actor process 1 ended (exit code -9)` 后收尾，`finished 5 updates`，`last.pt` 已存，无遗留游戏进程；int5：SIGINT 后 `finished 59 updates`、`last.pt` 已存，Traceback 都是 KeyboardInterrupt（另有一条 PyTorch 的 "Producer process has been terminated before all shared CUDA tensors released" 警告，退出时的正常现象）。
- **正确性对照**（第 6–30 次更新中位数）：

  | 运行 | × 实时 | collect 秒 | update 秒 | 推理 ms | kl | clip_frac | ev | 熵 | 显存 GB（学习器 / actor） | 核数（actor + 学习器） |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
  | p：1 个 actor × 32 worker | 1,298 | 3.33 | 2.03 | 2.29 | 0.0235 | 0.117 | 0.955 | 2.32 | 8.2 / 0.94 | 1.0 + 1.6 |
  | q：2 个 actor × 16 worker，同一张卡 | 793 | 5.68（两边 4.3 / 5.0） | 2.03 | 3.86 | 0.0243 | 0.121 | 0.937 | 2.25 | 8.6 / 0.62 + 0.60 | 2.0 + 1.0 |

  - 学习数字一致（kl、clip_frac、ev、熵在噪声内，update 相同），两半的合并没有改变训练。
  - **同一张卡上两个 actor 进程更慢**：推理从 2.3 ms 涨到 3.9 ms，collect 从 3.3 s 涨到 5.7 s。两个 CUDA 上下文在一张卡上是分时的（没有 MPS 时不同进程的 kernel 不能并发），每次切换都有代价，这就是为什么设计要求每个 actor 一张卡；这个布局只用来验证正确性，不代表速度。
- **4 卡吞吐（待测，需要 C99 停机窗口）**：在 `ws14`（分支 `b38-actors` 已检出）：
  - 无搜索：`B38_C99_STOPPED=1 B38_GPUS=0,1,2,3 B38_SMI=0,1,2,3 bash src/bridge/ops/hpc_b38.sh tput6 p6 t6`（p6 = 一个 actor 在 GPU 0、学习器在 GPU 1；t6 = 两个 actor 在 GPU 0 和 GPU 2、学习器在 GPU 1；64 worker、60 次更新、`--learner-fast 2`、FAST actor 标志，从 `segment1-final.pt` 续）。
  - 全标志：先 `B38_ALLOW_SEARCHES=1 ... hpc_b38.sh smoke6-full t6`（8 worker），再 `B38_ALLOW_SEARCHES=1 B38_C99_STOPPED=1 ... hpc_b38.sh tput6-full t6`（64 worker、60 次更新）；结果用 `ops/b38_summary.py` 汇总（`collect_actors` 一列是两个 actor 各自的 collect）。
  - 预期：无搜索时单 actor 64 worker 的 collect 1.81 s、update 1.94 s，两个 actor 后 collect 约 1.0 s，周期 = update ≈ 1.95 s → 约 2,200×（学习器限速）；全标志时 collect 约 2.8 s → 两个 actor 后约 1.5 s，周期约 max(1.5, 2.0) ≈ 2.0 s → 约 2,100×。CPU：两个 actor 后 worker 不再空等一半，worker 的 CPU 约增加一倍，加上 C99 停机，应在 90 核以内。
- **风险 / 没测**：4 卡上的速度与 CPU 占用；全标志（搜索、lane、教师 v2 / cf 记录经队列拼接）下的两个 actor；GRU 记忆模型；两个 actor 的局种子与单 actor 运行不同（分布相同，序列不同）；`--teacher-slots` 奇数时向下取整；两个 actor 的交接次数相同，所以总决策数是 `ceil(目标 / --rollout)` 次更新，与单 actor 的按决策数结束相差至多一次更新。
