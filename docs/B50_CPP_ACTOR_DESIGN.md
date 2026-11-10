# B50 设计：actor 调度热路径迁到 C++（阶段 0 设计稿，2026-11-09，已回填 nsys 数字）

对应实验记录：`bridge/abplus/EXPERIMENTS.md` B50 节；字段：`docs/PROJECT_SPEC.block-memory-v1.md` 的 `actor_b50`。
目标：不改模型计算、batch 组成、CF/teacher 语义和任何预算，把完整短 profile 的 collect 从 27.9 s 压到 ≤ 22 s，
update 不变差，×实时 ≥ 200。

## 1. 现状（B47 代码，nsys prof-b50m5：5 个 collect、154.4 s 窗口；每 collect ≈ 一个 rollout 32768 决策）

actor 线程是一条 Python 串行链。每 collect 约 2,130 次主调用（平均 15.4 行）、1,480 次 lane-only 步、
450 个 lane 轮次并入主调用，合计约 3,600 次同步 step graph 调用（稳态 3,700–4,300）。各段（B50 节的表）：

| 段 | s/collect | 每次 | cProfile（B47） |
|---|---:|---|---:|
| g.sync（GPU 等待，含排队） | 12.5 | 3.47 ms/步（p50 2.03） | 3.33 ms/步 |
| 写页 step.pages | 3.2 | 0.89 ms/步（write_dec 0.45，逐行 Python 0.29，新页 44 ms × 19） | 0.85 |
| step 内其余 host（解包、H2D、launch、D2H、prep、setup、after） | 2.8 | 0.78 ms/步 | 1.21 |
| 应答 answer（store + reply） | 2.3 | 1.07 ms/调用 | 0.81 |
| book.arrive | 1.7 | 0.81 ms/调用 | 0.39 |
| capture_main / service_control | 1.5 | 0.70 ms/调用（不含一次 9.2 s 的 lane 预录停顿） | 0.65 |
| lane 轮次组装（eager/submit/finish 不含 step） | 3.8 | 2.0 ms/轮 | 1.2–1.4 |
| GIL 等待（actor 线程，分散在上面各段里） | 2.6 | 114k 次，p50 36 µs，p99 1.1 ms | 不可见 |

主调用从进入 lane 服务到应答 p50 8.7 ms、均值 11.6 ms（learner 忙 12.6 / 闲 8.2）。worker 每步游戏只要 2–3 ms，
其余都在等 actor，所以 collect ≈ 主调用次数 × 周期，周期 ≈ 13–14 ms。

learner 线程：每 update wall 25.4–27.8 s（含追踪开销），forward_packed 14–15 s，host_decode 2.4–2.8 s，
持有 GIL 3.5 s，**等待 GIL 4.4–5.2 s**（几乎全在 forward_packed 的算子分发重获 GIL 上）。

GPU 等待的构成：若每步都是空闲时的 2.0 ms，每 collect 7.2 s；实际 12.5 s，多出的 5.3 s 里 5.0–6.0 s 来自每
collect 约 125 次 > 10 ms 的等待（平均 44 ms），它们发生在 learner 的 PPO 前向/反向段：actor 流优先级再高也要等
正在执行的 learner 长 kernel 结束。**这部分不随 Python → C++ 变化。**

## 2. 迁移的边界

进 C++（同进程、一条独立原生线程 `ActorDriver`，全程不持 GIL）：
1. **收请求**：`poll()` 16 个 worker 管道 fd（`TokSampler._fds` / `_by_fd`），读 5 字节消息得 (worker, slot)。
2. **组批**：从共享内存 `rows[slot]`（ROW 记录，帧内联）取行；`go = done == 0 | done == DONE_BRANCH`，
   终止记录只应答不推理；`first` 行：arena 行重置、pending 清零。
3. **解包进 pinned 槽**：现有 `abp_visible_unpack`（C）直接写 graph 的 pinned 输入 slab；pending / ids / u /
   replay / complete / version 一并填入。
4. **uniform**：主行 `at::rand({n, heads})`（CPU 默认生成器，与 Python 路径同一个 mt19937 流，顺序一致则逐位相同）；
   lane 行 splitmix64 CRN（移植 `crn_uniforms`）。
5. **H2D → graph replay → D2H → 同步**：`at::cuda::CUDAGraph::replay()`（第 3 节），全部在 actor 的高优先级流上。
6. **应答**：pforce 覆盖、`sampler.actions[a_idx]`、`main_value` / `main_vg`、`aux[value, logp, used]` 写入共享内存，
   然后向 worker 管道写应答字节。**应答在写页之前**（现在是之后，相差写页 + answer ≈ 2 ms）。
7. **写页**：K/V、z 从 pinned D2H 缓冲 memcpy 进 CPU 页（页由 C++ 的 `PageStore` 持有的 CPU 张量，预分配、按页
   4096 行增长；`d_free` / `d_used` / `d_meta`、块页 `b_*` 与 raw 帧字节都在 C++ 侧）；`step_host += 1`。
8. **交账**：每步向 Python 推一条记录（worker 列表、slot、决策 id、K/V 槽 id、host 输出 [n, heads+3]、
   步前 step / pending、是否 first / complete、完成块的 bid）进无锁队列。

留在 Python（主线程改为“消费者”，不在 worker 可见的延迟路径上）：
- `Episodes.arrive` 的奖励 / 回合结束记账、`store()` 入 rollout、`EpisodeLog.push` / `push_block`（谱系）、
  `PagedSnapshot` 对象（供 learner 与 lane 的 restore）、`lanes.observe_rows / record_main`；
- CF 点捕获 `capture_main`：现在在推理前做，改为在消费者里做。依据：`PagedSnapshot` 只是 (log, step, pending,
  version) 引用，步后仍能构造出步前状态；worker 只在**之后的**记录才检查 `status == -1`（`tok_floor.py` 1916 行），
  分支由搜索进程按 `status == 2` 启动，晚几毫秒无语义差别。阶段 1 的等价测试要覆盖（点的 snapshot 内容相同）。
- 权重发布 `take_weights`（排空 driver 后做）、graph 新键的捕获（Python 捕获，driver 拿到 shared_ptr 后重放）、
  lane 服务（阶段 1 仍 Python，通过 driver 的同步 step 接口跑 lane 步；阶段 2 进 driver）。
- 两代 lane actor 的 graph 在启动时一起预录（现在第二代出现时在 actor 线程上录 9 张图，停 9.2 s）。

## 3. graph 的触发方式：候选 A / B 与选择

- **A. libtorch 里用 `at::cuda::CUDAGraph` 在 C++ 捕获 / 重放**。重放可行；**捕获不可行**：step 的 body 是 Python
  定义的模块（compiled Spatial、`model.act`、`memory.read`、Writer），在 C++ 重建 forward 等于重写模型，违反
  “不改模型计算”，也无法保证与 eager 等价。
- **B. graph 由 Python 捕获，C++ 线程重放时短暂取 GIL 回调 Python**。每步一次 GIL 获取：本次测到 actor 重获 GIL
  平均 324 µs（p90 600 µs），3,600 步/collect × 0.3 ms ≈ 1.1 s/collect 白等，且 learner 侧的 GIL 等待（每 update
  4.4–5.2 s）一点不少。
- **选择 B′（混合）**：graph 仍由 Python 捕获（`torch.cuda.CUDAGraph`，不改 `ActorStepGraph._capture`），
  Python 把 `torch._C._CUDAGraph` 对象交给扩展；它是 pybind11 的 `std::shared_ptr<at::cuda::CUDAGraph>` holder，
  扩展用 torch 自带的 pybind11 头编译后直接取到 shared_ptr，在 C++ 线程上 `CUDAStreamGuard` 切到 actor 流后
  `replay()`。不需要 GIL，不需要 cudaGraph 裸句柄，和 Python 的 `graph.replay()` 调的是同一个函数。
  开销：`replay()` = 一次 `cudaGraphLaunch` 加 RNG 偏移更新（两个小 fill_），数十 µs（本次 g.replay p50 30 µs）。

构建：不用 ninja（4090 上没有），像 `visible_native.py` 一样按源码哈希缓存，直接 g++：torch 的 include（含自带
pybind11）、`-D_GLIBCXX_USE_CXX11_ABI=1`（torch 2.7.1+cu118 为 True）、CUDA 头用 venv 里 pip 包
`nvidia/cuda_runtime/include` 等（11.8，与 torch 一致；**不用**系统 CUDA 12.4 的头），链接
`libtorch_python libtorch_cuda libc10_cuda libtorch_cpu libc10`。扩展不直接调 cudart：流、同步、拷贝全走 c10 / ATen。

## 4. 数据结构

- **worker 环**：现有共享内存 `rows`（ROW，2 slot/worker）、`actions`、`stats`、`control`、CF 的 `main_value /
  main_vg / cf_snap`、`aux`：Python 把这些 numpy 数组的指针、步长和 fd 列表一次交给 driver（进程生命周期的
  共享内存，不会重分配）。
- **pinned 输入槽**：沿用 `ActorStepGraph._layout` 的布局（每个 graph 键一个 slab + 对应的 pinned D2H 缓冲）；
  Python 捕获后把 (key → slab 地址、布局偏移、输出缓冲地址、graph 对象) 注册给 driver。
- **页数组**：`PageStore`（C++）：`kv_pages / z_pages / b_pages` 为 CPU `at::Tensor` 列表（Python 侧 `PagedStore.kv()`
  等读接口不变，learner 仍按页 gather）；`d_meta / b_meta / b_src / b_srcver` 为 C++ 数组并暴露 numpy 视图；
  raw 帧用按决策 id 索引的字节 arena（`source_arena` 直接拼）；`collect()` 的存活计算仍在 Python（需要谱系），
  只把要释放的 id 交给 C++。
- **lane 请求队列**（阶段 2）：lane fd 也进 driver 的 poll；lane 行按 generation 分组，与主版本相同的并入
  下一次主步，其余作为 lane-only 步排在主步之后同一条流上。

## 5. Python 回退开关与等价测试

- `ISAAC_BLOCK_CPP_ACTOR=0`（或扩展编译失败、键未捕获、`greedy` / 评测路径）→ 现有 Python 路径不变。
- 等价（阶段 1 门槛）：同一记录序列（录制的 worker 消息 + rows）喂两条路径：动作、logp、价值、vg 逐位相等；
  页内容（K/V bf16 按位、z float32 按位、块值）逐位相等；`d_meta`、谱系、snapshot 的 `snapshot_parts` 相等；
  CF 点捕获的 snapshot 相同。现有 graph ≡ eager 测试、A8 play digest 8/8、全量 pytest 不变。

## 6. 预估（按第 1 节的 nsys 分解）

| 项 | 现在 s/collect | 预计 | 依据 |
|---|---:|---:|---|
| 写页 | 3.2 | 0.3 | memcpy 16 行 × (K/V 4×2×4×4×48 × 2 B = 12 KB + z) ≈ 0.2 MB/步 < 0.1 ms；无逐行 Python，页预分配 |
| step 内其余 host | 2.8 | 0.6 | C 解包已有；省去 numpy 槽填充、`check_frames`、`torch.rand` 的 Python 层、torch 的 Python 分发 |
| GIL 等待（actor） | 2.6 | 0 | driver 不持 GIL |
| answer + book + control（每主调用 2.6 ms） | 5.5 | 延迟路径上 ≤ 0.5 | 应答由 C++ 提前发出；记账在消费者线程，不再挡下一次 poll |
| lane 轮次组装 | 3.8 | 阶段 1 不变；阶段 2 ≈ 0.5 | 阶段 2 进 driver |
| GPU 等待 | 12.5 | 12.5 | 3,600 × 2.0 ms = 7.2 s 固定；长尾 5.3 s 是 learner 长 kernel，不随迁移变 |

按周期估：阶段 2 的主调用周期 ≈ 3.5（主步，均值含长尾）+ 0.7 × 3.5（lane-only 步分摊）+ 0.5（driver host）+ 2.5
（游戏）≈ 8.9 ms → collect ≈ 2,130 × 8.9 ≈ **19 s**；阶段 1（lane 仍 Python，组装 1.4 ms/轮留在路径上）≈ 9.9 ms →
**≈ 21 s**。如果阶段 1 只做同步的 C++ step（记账仍在 actor 线程、应答仍在写页之后），周期只降到约 10.4 ms →
≈ 22 s，不够稳，所以阶段 1 就按 driver 线程 + 消费者做。
update 侧：learner 每 update 4.4–5.2 s 的 GIL 等待随 actor 离开 GIL 消失，22.9 → 约 18–19 s；这是 update ≤ 22 s 的来源。
风险：GPU 长尾若在 C++ 之后仍是每 collect 5–6 s，collect 的下限约 18–19 s；再往下要动 learner 的 kernel 粒度（不在本步范围）。

## 7. GPT 建议里不成立或不采纳的

1. “在 C++ 里捕获 CUDA graph”：模型是 Python 模块，捕获只能在 Python（第 3 节）。
2. “C++ 线程每步回调 Python 重放”：回到 GIL 争用，每步重获 GIL 约 0.3 ms，learner 的 GIL 等待也不会减少（第 3 节 B）。
3. “用 cudaGraph 裸句柄 / cuda-python 重放”：torch 2.7.1 的 `CUDAGraph` 不暴露 `cudaGraphExec_t`；
   另起一个 exec 实例也没必要，shared_ptr 直接重放即可。
4. “把 actor 工作挪进 learner 时间来重叠”：B47 已证伪（异步 lane / 流水化让 update 等量变差）；本方案的重叠是
   去掉 GIL 和缩短延迟路径，不是把 GPU 工作换个时间做。
5. “GPU 等待全是 Python 开销，C++ 后 GPU 等待会掉到 2 ms/步”：不成立，5.3 s/collect 的多余等待是 learner 长 kernel
   阻塞（本次按时长分桶 + learner 段归因）。
6. “加 worker 到 32”：内存不够（B47）。
7. “两代 lane 一次同步”：只有在 lane 组装进 C++ 后才可能有收益（阶段 2 评估），每轮的主要成本现在是组装，不是同步次数。

## 8. 阶段 1（B52）实现记录与条件对应

- **顺序不变量（条件 3）**：`ActorDriver::step_batch` 在 driver 线程内依次：同步 → 写共享数组 → `reply_rows` → `PageStore::write_decisions_raw` /
  `write_blocks_raw`（页 + 元数据 + raw 帧，持 store 互斥锁）→ `step_host` 递增 → `push_record`（队列互斥锁）。消费者只从队列拿记录，
  snapshot（`PagedSnapshot(store, log, step_before, pending_before, version)`）、`EpisodeLog.push`、CF 捕获（`capture_deferred`）、
  lane 的 `restore` 都由记录触发；learner 的 gather 只读 `d_meta` 里已写的 id。队列互斥锁的 release/acquire 给出写页 → 读页的
  happens-before。测试 `test_driver_pressure_pages_complete_before_records`：6 个假 worker 随机延迟，一条读线程持续 gather 记录过的页、
  用记录重建的 snapshot 做 lane restore，driver 同时在跑；校验 d_meta、raw、b_src/b_srcver 与消费者侧 EpisodeLog 一致。
- **随机数（条件 4）**：driver 用自己的 xoshiro256**（种子 `args.seed + 1000003*aid + 0xB52`，splitmix64 展开），uniform 取高 24 位，
  分辨率与 torch.rand 相同；lane 行继续 CRN（阶段 2 进 C++）。等价测试两层：(a) `set_fixed_uniform(0.5)` 下 driver 与 Python
  `step(uniform=0.5)` 的动作 / logp / 价值 / vg / 页 / 块 / 元数据 / arena 行逐位相等（`test_driver_single_worker_matches_python_step_with_fixed_uniforms`）；
  run_graph 中段在任意 uniform 下与 Python 中段逐位相等（`test_run_graph_middle_matches_python_middle`，fp32 与 bf16）；(b) 同种子可复现：`test_driver_same_seed_reproducible_sampling`（两个同种子 driver 的采样动作 / 输出 / K/V 页逐位相等，异种子的动作不同；Python 路径在 `torch.manual_seed` 下两次输出相等），整条训练路径的复现由 A8 8/8 覆盖。
- **消费者的 GIL 账（条件 2，预估，待 nsys 回填）**：留在 Python 主线程的每主调用工作 ≈ book 0.8 + snapshot/谱系/record_main/CF 0.5 + store 0.9
  ≈ 2.2 ms，×2,130 ≈ 4.7 s/collect；阶段 1 的 lane 轮次（组装 3.8 s + lane 步的 Python 外壳 1.3 s）≈ 5 s/collect。合计约 10 s 的 Python
  对比原来 actor 线程约 18 s 的 Python 墙钟（持有 GIL 11.4 s）——learner 的 GIL 等待预计从 4.4–5.2 s/update 降到约 2.5–3 s，不会归零；
  归零要靠阶段 2（lane 进 driver）和把 `Episodes.arrive` / `EpisodeLog.push` 的热部分下沉。
- **回退（条件 7）**：`ISAAC_BLOCK_CPP_ACTOR=0`、无 CUDA torch、无 g++、编译/加载失败 → `actor_native.load()` 返回 None，PagedStore 用 Python 数组，
  train_tok 走原循环；测试文件整体 skip。构建：g++ 直接编译 `abp_visible.c`（-x c）+ `abp_actor.cpp`（-x c++ -std=c++17），torch 自带 pybind11 头，
  venv 的 `nvidia/*/include`（CUDA 11.8）在前、系统 CUDA 的 include 在后（只为 `crt/host_defines.h`），按源码 + torch 版本哈希缓存 .so，原子替换。
  holder 假设由扩展本身验证：`register_graph` 收 `std::shared_ptr<at::cuda::CUDAGraph>`，driver 线程 `replay()` 与 Python 重放逐位一致。
- **4090 门里修掉的三处**：driver 线程要在 `c10::InferenceMode` 下工作（graph 的 pinned 缓冲是 inference 张量）；bf16 缓冲用无类型指针；
  输出行宽是 nh+3（动作数 + logp/value/vg），不是 sum(heads)+3。

- **阶段 1 实测（B52，2026-11-10，详见 EXPERIMENTS.md B52）**：门全过（cpp 47、入口门 2、全量 204、A8 8/8）。u2–u14 中位：collect 27.17 s
  （同树 Python 28.82，门槛 22：不过），update 17.10 s（同树 Python 18.72，门槛 19：过），×实时 160.8。主调用延迟 11.7 → 3.6 ms，
  learner GIL 等待 4.4–5.2 → 2.50 s/update（与本节预估 2.5–3 一致），主线程 GIL 等待 2.06 s/collect。
- **预估错在 lane 轮次**：第 6 节把阶段 1 的 lane 成本按 B50 记为 ≈ 5 s 且能与主步重叠；实际 lane 与主 actor 共用 arena 与流，driver 必须
  为每个 lane 轮次 `pause()`，lane 行也不再像 Python 路径那样并进主调用，actor.tick（driver 暂停）10.3 s/collect、p_tick 12.1 s，collect 基本
  不降。CF 关时 collect 11.5 s（其中 6.5 s 在等 learner）。阶段 2 的首要目标因此是“lane 请求进 driver 的批、与主步合并进同一次 graph 调用”，
  而不是 Episodes.arrive / EpisodeLog.push 下沉（消费者 GIL 等待已只有 2 s/collect）；按 B52 数字估 collect ≈ 18–19 s。
- **learner 长 kernel（条件 6）**：B48 dump 固定输入重放（主线程 torch profiler）没有 ≥ 10 ms 的计算 kernel（最长 5.4 ms），≥ 10 ms 的只有
  15 次 packed 输入的 HtoD 上传（10–16 ms，228 ms/update）。第 1 节“长尾来自 learner 长 kernel”的解释需要修正为“排队 / 拷贝引擎”，待 CUDA 追踪确认。

## 9. 阶段 3（B58 / B59）设计草案（2026-11-10，B57 数字回填前的版本）

本节先按代码写结构和等价条件，B57 的周期预算表出来后回填第 9.0 节的数字，再交父会话审。分工照 A1 的 B60：`block_learner.py` 与 train_tok 的 `learner_loop` 不动；采集循环、消费者、交接、`abp_actor.cpp` 在这条线上改。

### 9.0 起点（B57 周期预算，prof-b57m 5 个周期 + prof-b54-cpp 中位）

- **周期**：max(collect 13.4, update 21.4) s，瓶颈是 learner，actor 每周期等 8.1–8.2 s。
- **GIL（每周期）**：主线程持有 7.35 s、等待 3.85 s；learner 持有 2.36 s、等待 3.82 s；pt_autograd 持有 1.41 s、等待 2.57 s；其它约 60 个 Python 线程合计持有约 1.1 s；driver 为 0。
- **争用损失**：learner 各段在 actor 采集时比 actor 等待时慢——PPO 前向 / 反向 1.88 倍，forward_packed 1.90 倍，host_decode 5.8 倍，opt 不变；合计每周期约 7.4 s。host_decode 是纯主机工作，所以争用主要是 GIL，GPU 优先级（prio）对此无效。
- **消费者热点**（cProfile 一个 collect）：Episodes.arrive 1.33 s，store 1.10 s（其中 compact_rows 1.37 s 累计），run_driver_collect 循环 2.64 s，drain_driver 1.0 s，tok_obs append 0.71 s，_service_control 0.57 s（每次 tick 都跑）。
- **交接**：driver 每周期暂停约 0.27 s（pause / drain 8 ms、取权重 9 ms、job 47 ms、GC 214 ms）；wait_learner 期间 driver 只跑 lane，不算暂停。
- **T\***：约 13.4 s（learner 在等待窗口里的速度，约 326 倍），1.10 × T* ≈ 14.7 s。learner 在不持 GIL 的打包（A1 的 B60）之后还会更低。
- **B58 / B59 的目标**：主线程 GIL 7.35 → ≤ 1.0 s，learner 回到接近等待窗口的速度，交接暂停 0.27 s → ≤ 50 ms（GC 214 ms 要移出暂停：在 driver 继续采集时跑，或增量化）。

### 9.1 B58：消费者出 GIL

现在每条记录在 Python 里做的事，逐项给出去向：

| 现在（consume_record） | 每 collect | B58 去向 |
|---|---|---|
| `bdrv.record_rows` 视图、`lanes.observe_rows`（受伤源候选） | 小 | C++：每 worker 的 (episode, t) 环与 hurt 标记；Python 控制逻辑按需读 |
| `book.arrive`：结算上一决策的奖励、终止、close_*、episode 计数 | 约 2.2 s | C++：同样的向量公式逐行算，写 rollout 的原生数组（reward / terminal / close_* / open），episode 累计量放原生数组；episode 结束汇总按批交给 Python |
| 逐行 `EpisodeLog` 新建 / push / push_block | 约 1 s | C++：每 worker 一条整数谱系（dec / kv / blk / blk_end，按 episode 分段），Python 端用只读视图对象（同 EpisodeLog 接口：decisions / blocks_before / branch / offset / end） |
| 逐行 `PagedSnapshot`（含 `store.track` 的 WeakSet） | 约 1–1.5 s | 只在需要时物化：rollout 的块起点（每 worker 每 128 步或 episode 开头）、open 决策、CF 捕获点、受伤源被教师用到的那一刻。GC 改为从 C++ 谱系表和 Python 的少量物化快照取 live (log, step) |
| `lanes.record_main`（受伤源字典，逐决策） | 约 0.5 s | C++：受伤源只记整数 (worker, episode, t) → (谱系 id, step, pending)，教师要用时物化 |
| `store`：rollout 行（ROWC 压缩）、pending / actions / logp / value / version / vg / priv / boot / length / open | 约 2.8 s | C++：driver 写页之后直接写进 rollout 的原生数组（`compact_rows` 的 C 版，结构与 `ROWC` 逐字节一致） |
| `lanes.capture_deferred`（CF 捕获） | 次数少 | 仍在 Python：C++ 把带请求的记录放进“事件”批，Python 物化快照、`capture_point`、`capture_done` |
| `next_record(0.002)` 轮询 | 2 ms 粒度 | 条件变量：Python 只在“有事件”时醒（CF 捕获、episode 结束批、kind 1 / 5、错误、rollout 满） |

- **Python 留下的低频事件**：CF 捕获请求；episode 结束汇总（按批）；运行期图捕获（kind 1）；lane 作业需要 Python 计算（kind 5）；错误；rollout 满 / 交接。lane 的 kind 4 记录仍在 Python 回填应答，次数约 1,000–2,000/collect；B58 第二步再看要不要把应答写入也下沉到 C++。
- **未迁移的路径显式回退**：lab、priv、duel、temporal / GRU memory、多 actor 进程、learner 进程，任一开着时打印原因并走 B54 的 Python 消费者，不允许静默差异。
- **开关**：`ISAAC_BLOCK_CPP_ACTOR=0`（整条 Python）、`ISAAC_BLOCK_CPP_CONSUMER=0`（driver 照旧，消费者用 B54 的 Python 版），两者都进门测。
- **等价测试**（不依赖时序）：录下一段真实 driver 记录流（kind 0 / 4、带 CF 请求），分别喂给 C++ 消费者和 Python 消费者，比较 rollout 全字段（逐字节）、谱系（每 worker 的 dec / kv / blk / blk_end）、块起点快照的 (log 区间, step, pending, version)、CF 捕获点集合与状态、episode 结束汇总；同种子下两条路径跑同一组假 worker 也要一致。
- **门槛**：collect 期间主线程 GIL 持有 ≤ 1.0 s；取消数 0；运行期捕获 0；全量门、入口门、A8 8/8。
- **分步提交**：B58a rollout 原生写入 + `store` 下沉；B58b `Episodes.arrive` 下沉；B58c 谱系表 + 懒物化 + GC；B58d 事件批 + 条件变量。每步一个回退开关、一组等价测试、一次 16 update profile。

### 9.1a 按需物化：线程、GIL、每周期耗时（父会话要求补充）

交给 learner 的 job 结构和 BlockRollout payload 形式在 B58 / B60 期间冻结不变：job 里仍是同样的 PagedSnapshot（块起点、open 决策、教师 / v2 的上下文），只是不再每行生成。

| 物化什么 | 何时 | 哪个线程 | 持 GIL | 每周期（估） |
|---|---|---|---|---|
| BlockRollout 块起点 + open 决策（每 worker 每 128 步、每个 episode 开头，加 16 个 open） | rollout 写满、构建 job 时 | 主线程（消费者） | 是 | 约 350 个 × 15–30 µs ≈ 5–10 ms |
| CF 捕获点 | 捕获事件到达时（约 48 个/collect） | 主线程 | 是 | < 2 ms |
| 受伤源上下文（教师记录引用的那些） | teacher_records 取用时，即交接构建 job | 主线程 | 是 | 约 90 个 ≈ 2–3 ms |

- 这些物化都在主线程上做，driver 不暂停：B59 之后交接不再 pause / drain，构建 job 与 driver 继续采集并行。因此它们不计入“交接停顿 ≤ 50 ms/周期”，计入主线程每周期的 GIL 持有（门槛 ≤ 1.0 s，物化约 0.01–0.02 s）。
- B58 期间如果交接仍有 pause（B59 之前），物化落在 pause 内，就计入停顿，表里分开报。
- learner 线程不做物化。它拿到的仍是已物化的 PagedSnapshot，update 不增加这部分时间。
- B58 加 NVTX 段 m.materialize，按周期报实测秒数。

### 9.1b learner 读页期间页不可变、不被回收的保证（给 A1 B60 / N3 的不变式）

1. **写一次再发布**：一行决策的 K/V / z、raw 帧，一个块的值、src / srcver，都由 driver 在 `write_decisions_raw` / `write_blocks_raw` 里于 store 互斥锁内写完，然后才把记录入队。之后这一行不再被写，除非它的 id 先被回收再重新分配。
2. **回收只发生在 GC**：`free_rows` 只由 `PagedStore.collect()` 调用。collect 只在主线程上跑，即交接时的 h.gc，B58 之后仍只在主线程。它只释放从 live (log, step) 对不可达的 id；live 对来自所有存活的 PagedSnapshot（WeakSet）和 slot 的根，B58 之后再加上 C++ 谱系表的根。
3. **job 持有引用直到 update 结束**：job 里的 PagedSnapshot（块起点、open、教师 / v2 上下文）被 learner 线程引用到这次 update 结束。所以 GC 跑在 update 期间也不会释放这次 update 能读到的任何行。B60 直接读原生页时，必须在读页期间一直持有 job 里的这些快照，不能先放掉再读。
4. **页内存不搬家**：页张量一旦分配就不再移动，只追加新页。但保存页张量的 `std::vector` 在追加时可能重新分配，如果 learner 在别的线程不加锁地读 `kv_pages[p]`，就有数据竞争。B58a 会在构造时把三个页向量 reserve 到上限（max_pages_d / max_pages_b），保证永不重新分配，之后页指针可以无锁读取。在那之前，跨线程取页指针必须拿 store 的互斥锁，这正是现在 kv_page / z_page / b_page 的做法。
5. **元数据**：d_meta / b_meta / b_src / b_srcver 是预分配容量的数组，不搬家。被回收的 id 写 −1；只要满足第 3 条，learner 读到的 id 就不会是 −1。

### 9.2 B59：交接不停机与 GPU 平衡

**(a) 交接不停机**

- **双缓冲 rollout**：现在已是 roll / spare 交换，但交换时 driver 被 pause + drain。B58 之后 rollout 由 C++ 消费者写，交接改成：rollout k 写满（usable ≥ rollout_actor）的那条记录处，C++ 消费者原子地把写入目标切到 k+1（`spare.carry` 的 open 决策搬运在 C++ 做），并发一个“满”事件给 Python。Python 在不暂停 driver 的情况下构建 job（teacher_records、v2_records、pack_snapshots、GC），放进 learner 队列。
- **GC 移出暂停**：GC 每周期 214 ms，B59 让它在 driver 继续写页时跑。这需要一个分配代号：PageStore 每次分配记下当时的 GC 代号，collect 开始时递增它，只释放在本次扫描开始之前分配、且不可达的行。否则 driver 在扫描期间新分配的行会被当成不可达而误删。free_rows 本身已在互斥锁内。
- **一个 update 的滞后不变**：k+1 写满而 update k 未完，C++ 消费者进入 main_hold（主步记录扣住，lane 继续），与现在的等待语义相同。
- **权重切换**：learner 把新权重写进影子缓冲（take_weights 已经把参数聚成一块字节）。driver 在两次 replay 之间用 actor 流上的一次拷贝把影子缓冲写进 actor 参数，然后原子地递增 version。graph 原地读参数，所以拷贝发生在 replay 之间就安全。
- **lane 不再整体 quiesce**：只需保证同一次 lane 调用内版本一致。合并批在执行时比较版本（已是如此）；lane-only 用各代模型的副本，不受主权重切换影响。
- **等价条件与测试**：
  - 每行记录的 version 等于该行 replay 时的版本；
  - 在两次 replay 之间切换，输出与 Python 用对应版本权重的结果逐位相等；
  - 同一合并批里不出现两种版本。
- **门槛**：driver 因交接或取权重暂停 ≤ 50 ms/周期。

**(b) GPU 平衡**

- 先看 prof-b54-prio（learner 流高优先级）。若 update 明显下降、collect ≤ 20 s，就设为默认。
- 若不够，再按进度平衡，方案和数据先交父会话批：driver 读 learner 的进度计数（已完成的 PPO 微批 / 本次 update 的总微批），和自己的进度（usable / rollout）比；actor 领先超过阈值时，在两批之间让出 GPU 0.5–2 ms。
- 只在 actor 领先时让，不丢行、不减预算，collect 仍 ≤ update。actor 本来就要在末尾等约 8 s，这里只是把等待挪到中间，周期不变或缩短。

**(c)** main_hold 期间的 lane 调度纳入同一套让出规则。

### 9.3 预估（B57 回填后定稿）

- B58：主线程 GIL 持有 8 s → ≤ 1 s，learner 的 GIL 等待随之减少。
- B59(a)：交接停顿 → ≤ 50 ms。
- B59(b)：update 靠近隔离值。
- 周期下界 T* 取 max(learner 隔离 + 必要的争用, actor host 下界)，以 B57 的测量为准。
