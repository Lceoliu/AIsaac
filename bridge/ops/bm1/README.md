# bm1：训练、计时与测试的运行环境

更新：2026-11-17。bm1 是一组 shell 脚本，在一台多卡主机上用 Docker 容器（或直接在系统里）跑 `train_tok.py`：计时、测试门、冒烟、正式训练和评估。最早是从租用的 4090 上的计时口径移植过来的（见文末"历史"）。源码在 git 的 `bridge/ops/bm1/`，部署到主机上的一个 bm1 目录（容器里是 `/bench/bm1`）。

## 主机、分道与 GPU

- 一条"道"是一组 CPU（`BM1_CPUSET`）、一个内存节点（`BM1_MEMS`）和允许用的 GPU（`BM1_GPUS_OK`），加上锁、内存上限和会话名。同一台主机上可以有几条道，每条道一个 bm1 目录、一份 `bm1.env`。
- 哪台主机、哪些 CPU 和 GPU 都只写在 `bm1.env` 里。部署时从 `env/` 里复制对应主机的一份；新机器照 `env/new-host.env.template` 和 `probe_host.sh` 的输出来写。脚本里不写硬件数字。
- 有故障或属于其他租户的 GPU，不要放进 `BM1_GPUS_OK`。`preflight` 也会拒绝在出错的卡上启动（见下）。
- 镜像 `aisaac-bench:<日期>`：Ubuntu 22.04、Python 3.10.12、g++ 11.4、xvfb。venv：torch 2.7.1+cu128，其余版本按 `env/requirements-bm1.txt`。

## 目录（bm1/）

| 路径 | 内容 |
|---|---|
| `run.sh` | 唯一的启动入口：`run.sh <用途> <GPU 列表> <命令>`，在本道的 CPU / 内存节点 / GPU 上跑一条命令。`dock.sh` 是它的旧名字，留着兼容 |
| `runcpu.sh` | 只用 CPU、不给 GPU 的变体（只有 docker 模式） |
| `common.sh` | 读 `bm1.env`；锁、`preflight`、发 SIGINT、`trash`、日志 |
| `xnew2.sh` | block-memory v1 的训练配置（`train_tok.py` 的完整命令行），在容器里执行 |
| `xold_c99.sh` | 旧模型 C99 的命令行（C107 的 A 组） |
| `prof.sh` | 计时运行：拿锁，`ISAAC_BLOCK_TIMING=1`，到 N 个 update 后给 train_tok 发 SIGINT。同时采样本卡的 nvidia-smi（1 s）、全部 GPU（5 s）和 `mpstat -P ALL 5`。跑完把检查点移进回收站 |
| `shard2.sh`、`shardk.sh`、`shardk_in.sh` | 两个 / K 个分片同时计时；K 个分片作为一次训练（梯度同步，`isaac_bridge/shard_sync.py`） |
| `train_guard.sh` | 正式训练：K 个同步分片，失败后自动续训，写 `guard.log`；`STOP` 文件让它收尾 |
| `gate.sh`、`gate3.sh` | 一棵树的全量测试门（pytest）；`gate3.sh` 是 entry、全部测试和 A8 连着跑。日志带时间戳，从不覆盖，失败的拷进 `reports/failed/` |
| `job.sh`、`jobcpu.sh` | 不计时的容器作业，在锁下运行 |
| `ev_ab1.sh`、`ev_ab2.sh`、`ev_c113.sh` | 留出种子评估的作业（C106、C107；C113 起主机无关的 `ev_c113.sh`）。状态行由 `evtools/ev_complete.py` 给出：每个请求的种子都有一局、作业 rc 0 才算完成，否则写 `summary yes, INCOMPLETE`；`job.sh` 把容器的退出码写进 `<日志>.rc` |
| `evtools/` | `eval_ent.py`（评测外壳：eval_tok 原样 + entropy.json / config.json）、`ev_table.py`（汇总表）、`ev_complete.py`（评测是否完整） |
| `deploy.sh` | `tree <名> <tar>`：解开一棵 git-archive 树。`abphome <树>`：建游戏目录，从树的 `bridge/engine` 编 `libabp_turbo.so`（本分支仍以生成物为准，不保证传出编译器的退出码）。`ops <树>`：从树的 `bridge/ops/bm1` 部署 bm1 自己的脚本（见下"部署 bm1 脚本"） |
| `test_lock.sh`、`test_deploy_ops.sh` | 锁和 `deploy.sh ops` 的回归测试，在一个临时目录里跑 |
| `lock.sh` | 本道的锁，内容是 "<session> <时间> <用途>"，用 noclobber 原子地取 |
| `probe_host.sh` | 只读地列出主机的 CPU、NUMA、内存、GPU、驱动、Xid 计数，供写 `bm1.env` |
| `memguard.py` | direct 模式下写不了 cgroup 时，按进程树 RSS 守内存上限 |
| `m0_mem.py`、`summ.py`、`seg_table.py`、`train_summary.py`、`train_watch.py` | 采样和汇总：内存和线程、计时汇总、learner 分段、训练进度 |
| `diag_threads.py`、`ptc_trace.c`、`abrt_trace.c`、`nsys_gil.sh`、`nsys_update.sh`、`tsan_core.sh` | 诊断：线程出生栈、`pthread_create` 的原生栈、SIGABRT 时的原生栈、nsys（只开 nvtx 和 python-gil）、C++ 核心的 TSAN |
| `ab*_*`、`c108/`、`c110_*`–`c112_*` | C106–C113 的一次性脚本，留作记录 |
| `env/` | 各主机 / 各道的 env 文件、模板、`requirements-bm1.txt` |
| `trees/<名>` | 主机上部署的代码树：`git archive <commit> bridge/python bridge/ops bridge/abplus bridge/engine` |
| `home/` | 容器里的 HOME：各种缓存（isaac-actor、learn-native、visible、Inductor、Triton） |
| `home/isaac-abplus` | ABP_HOME。`runtime` 指向共享的只读游戏副本；`tools/` 从树的 `bridge/engine` 编出来；`instances/` 是本目录自己的 |
| `payloads/` | 全量门要用的载荷 |
| `reports/` | 所有结果和日志：总日志 `bm1.log`；`<tag>/progress.csv`、`<tag>.stdout`、`<tag>.gpu`、`<tag>.allgpu`、`<tag>.mpstat` |

`trees/`、`home/`、`payloads/`、`reports/` 只在主机上，不在 git 里。

## 部署 bm1 脚本（2026-11-18）

- `deploy.sh ops <树>` 在同一文件系统的暂存区备好整套文件，只规范化 CRLF。先用启动桩对 `run.sh`、`train_guard.sh` 做一允一拒的 GPU 行为测试，再切换；切换或复核失败恢复旧文件，`.ops_deployed` 最后更新。请在没有作业启动或其它运维编辑的窗口执行。首次升级旧部署脚本时，用新树中的 `deploy.sh`，并显式将 `BM` 指向主机上的现有 bm1 目录。
- 首次没有基线时用新树比对已有文件；以后和 `.ops_deployed` 比对。冲突退出 3，保留主机文件。确认后可用 `OPS_ACCEPT=文件1,文件2` 按名允许覆盖。旧文件和部署现场留在 `BM1_TRASH` 下按日期分的目录；该目录须与 BM 同一文件系统。如回滚本身失败，脚本会留下恢复目录和锁，不要跳过锁再部署。
- 需要永久禁卡的主机在生效的 `bm1.env` 设置 `BM1_GPU_FORBID_REQUIRED=1`，并配置 `BM1_GPU_FORBID_UUID=<GPU-UUID>[,<GPU-UUID>...]`。也接受 `BM1_GPU_FORBID=<索引>[,<索引>...]`；索引依赖当前设备排序，固定禁物理卡应优先用 UUID。同一张卡不必在两个列表都写。没有这项政策且两张名单均为空的主机继续传原来的显式索引。受保护主机按实际物理 UUID 启动，名单、映射有问题即拒绝；设备只由 GPU 参数选择，不得经 `RUNENV`/`DOCK_ENV` 再覆盖。
- 评测只在请求的 seed 块完整且本次作业退出码为 0 时可复用。新作业用 `<tag>.log.request` 领取标签，发布退出码后写 `<tag>.log.recorded`。退出状态未成功发布的作业不会被下次当成 done；历史无 rc 显示 `EXIT STATUS UNKNOWN`（退出 2），不计入完整。要重跑未完成的评测，请改用新 tag，或先归档旧目录及其收据。

## 用法（在主机的 bm1 目录里）

```sh
# 部署一棵树（本地：git archive --format=tar -o mX.tar <commit> bridge/python bridge/ops bridge/abplus bridge/engine，上传到 bm1/）
./deploy.sh tree m77657fc m77657fc.tar
./deploy.sh abphome m77657fc            # 第一次，或 engine/abp_turbo.c 变了的时候
# 计时 24 轮（上限 4500 s，30.0 游戏小时，24 个 update 后发 SIGINT）
GPU=1 UPDATES=24 HOURS=30.0 ./prof.sh base-m77657fc m77657fc
# 全量门连跑 3 次
GPU=1 RUNS=3 ./gate.sh m77657fc
```

额外的环境变量用 `ENVS="-e K=V ..."` 传进容器，例如 `ENVS="-e ISAAC_LEARN_CKPT_LEAN=0"`。正式训练见 [TRAINING.md](TRAINING.md)。

规则：
- 计时运行持有锁的时候，本道不许有编译、pytest 或别的 GPU 作业。`prof.sh`、`gate.sh`、`probe.sh` 都要先拿同一把锁。
- 删除只进回收站（`trash`：有 `gio` 用 `gio trash`，没有就移到 `bm1/trash/<时间>/`）。
- 测速 run 不留检查点。

## bm1.env 与两种启动方式

`BM1_MODE` 决定启动方式：
- `docker`：容器 `$BM1_NAME-<用途>`（默认 `aisaac-bench-bm1-a1-<用途>`）。参数：`--rm`，只读根文件系统，`/tmp` 是 tmpfs，用调用者的 uid，`--network none`，只给指定的 GPU（容器里从 cuda:0 起编号）。
- `direct`：直接在当前系统里跑。
  - CPU 亲和用 `taskset -c $BM1_CPUSET`，内存节点用 `numactl --membind=$BM1_MEMS`（有 numactl 时），GPU 用 `CUDA_VISIBLE_DEVICES`，进程组用 `setsid`。
  - 内存上限 `BM1_MEM`：能写 cgroup v2 时建子 cgroup，设 memory.max；写不了就由 `memguard.py` 每 2 s 加总进程树的 RSS，超限时写日志并杀掉进程树。
  - 这种方式没有网络隔离，也没有只读根文件系统，所以游戏 runtime 的文件权限要设成只读。

前置检查 `preflight`：每个作业启动前都做。
- `nvidia-smi -L` 不能有错误行。
- 设了 `BM1_GPU_EXPECT` 时，卡数要够。
- 内核日志里的 NVRM Xid 行数不能比基线多。
- 不通过就写 `STOP_QUEUE` 并记日志，之后所有脚本都拒绝启动，直到有人手动删掉这个文件。
- 新机器启动后先跑一次 `sh -c '. ./common.sh; preflight_baseline'`。

整机读写锁（设了 `BM1_LOCKDIR` 时）：整台机器一把锁，所有会话共用。
- 计时运行拿独占锁，写 `X`。写下之后，新的共享持有者进不来（写者优先），再等正在运行的共享持有者全部结束。
- 门、编译、冒烟、探针这类非计时作业拿共享锁，各写一个 `S.<会话>.<pid>`，彼此可以同时跑。
- pid 已经不在的锁文件算陈旧，下一个来拿锁的会删掉它并记日志。
- 没设 `BM1_LOCKDIR` 时，用每条道一个文件的旧锁。
- 计时运行可以用另一组 CPU（`BM1_TIMED_CPUSET`）。

同一台主机上的第二条道（cpp-actor 道）是另一个 bm1 目录，带自己的 `bm1.env`（模板 `env/cpp-lane.env.template`）。道的差别全在 env 里：
- `BM1_NAME`、`BM1_PYTHONPATH`、`BM1_RUNTIME_LINK`；
- CPU / 内存节点 / GPU / 锁 / 会话名。

## 从零装环境（新机器）

1. 系统包（Ubuntu 22.04 的名字）：python3.10、python3.10-venv、python3.10-dev、g++（11.4）、gcc、xvfb、util-linux、procps、numactl、sysstat（可选）、zstd、git。
2. NVIDIA 驱动要支持 CUDA 12.8（≥ 570）。CUDA toolkit 不需要，torch 的 pip 包里带了。
3. Python 3.10 的 venv：
   ```sh
   pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
   pip install -r env/requirements-bm1.txt
   ```
4. 从现有主机拷：
   - 游戏 runtime（约 610 MB）→ `BM1_GAME_RUNTIME`，设成只读；
   - `tools/` 里的打桩表（`libabp_turbo.so` 由 `deploy.sh abphome` 从代码树重新编译）；
   - `payloads/`（全量门要用）。
   - 不用拷：instances 目录（会自动重建）；venv（按第 3 步重装）。
5. 代码：`git archive <commit> bridge/python bridge/ops bridge/abplus bridge/engine` → `deploy.sh tree <名> <tar>` → `deploy.sh abphome <名>`。
6. 写 `bm1.env`（照 `probe_host.sh` 的输出），跑 `preflight_baseline`，先做一次 2 轮的冒烟测试（`UPDATES=2 ./prof.sh ...`）。

## cpp-actor 的默认开关（2026-11-11）

- **lane 合并 M1**：默认开，lane 作业并进 driver 的下一个主批。`ISAAC_BLOCK_LANE_MERGE=0` 关掉，回到 d1。验收见 EXPERIMENTS 的"cpp 线 M1 背靠背四次计时"。
- **批量记录 B58e**：默认开，`ISAAC_BLOCK_CPP_BATCH=0` 关掉。
- **原生 lane 服务**：默认开，跑在 driver 线程上。`ISAAC_BLOCK_CPP_LANES=0` 回到 B54 的 Python lane 服务。
- **独立 lane 线程（d2）**：默认关，`ISAAC_BLOCK_CPP_LANE_THREAD=1` 打开；开了以后 M1 自动关。

## 已知事项

- `tests/test_block_memory.py` 在 import 时调用 `torch.set_num_threads(4)`：只要它被收集，之后的所有测试都在 4 个 intra-op 线程下跑。现有的全部通过记录都在这个条件下，暂时不改。
- 同一台机器上跑几个 trainer 时，`xnew2.sh` 的 `NAME` 和 `PORT` 每个要不同（默认 b42new / 39200；`shard2.sh` 用 s0a / 39200 和 s1b / 39600）。
- Nsight Systems 2023.4.4 在 cu128 运行时 + 580 驱动上做 CUDA 跟踪（`-t cuda`）时：
  - learner 那张卡的 kernel 没有记下来；
  - trainer 在 capture 结束后段错误退出（rc 139）。
  
  原因未确认。只开 nvtx 和 python-gil 时可以用。learner 的分段时间改用 `ISAAC_BLOCK_TIMING=1` 下的 CUDA event 分段（`isaac_bridge/learn_segments.py`，写 `<run>/segments.jsonl`，用 `seg_table.py` 出表）。nsys 往每个被跟踪的进程注入线程，可能撞上容器的 pids 上限（`PIDS`）。

## 历史

bm1 是 2026-11-11 从租用的 4090 单卡机移植过来的（4090 当天到期）。当时为了让计时可比，配置和 4090 上逐个旗标相同（workers 16、rollout 32768、minibatch / micro 8192、overlap、decoupled、bf16），只改路径。和 4090 的差别：
- 硬件：4090 机器是 16 核 + RTX 4090；8x3090 主机的一条道是 24 个逻辑 CPU + RTX 3090。
- torch：都是 2.7.1，4090 是 cu118，这里是 cu128，所以跨机器的数值不要求逐位一致。
- 当时用 `pydeps/` 把共享 venv 缺的或版本不同的包（lz4、pytest、py-spy、numpy、zstandard）补成 4090 的版本；现在 `env/requirements-bm1.txt` 已是叠加后的精确版本。
- `libabp_turbo.so` 用同一份 `bridge/engine/abp_turbo.c` 编出，和 4090 的 .so 大小相同。
- 4090 只采本卡的 `<tag>.gpu`；bm1 另外采全部 GPU 和 mpstat，用来判断有没有被其他租户污染。

迁移和各轮计时的数字见 EXPERIMENTS.md 的"BM1 8x3090 环境"到"BM1 C"各节。
