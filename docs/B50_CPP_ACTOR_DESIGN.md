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
