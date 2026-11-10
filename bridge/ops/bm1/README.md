# bm1：8x3090（dgene）上 block-memory v1 的计时与测试环境

这是 4090 上 `xnew2.sh` / `prof2.sh` / `b62_step0.sh` 计时口径的 Docker 版。train_tok 的配置和 4090 一个旗标都不差（workers 16、rollout 32768、minibatch 8192、micro 8192，teacher / CF / aux / priv-critic 的预算、overlap、decoupled、AMP bf16、learner 在进程内），只改路径。源码在 git 的 `bridge/ops/bm1/`，部署到主机的 `~/aisaac-bench-20261019/cpu-wide/bm1/`（容器里是 `/bench/bm1`）。

## 机器与分道

- 主机 dgene：2x Xeon Gold 6146（24 核 48 线程），8 张 RTX 3090，驱动 580.178.04，内核 6.8.0-147。
- 本道是 node0：CPU `0-11,24-35`、内存节点 0、GPU 1/2/3。cpp-actor 用 node1 和 GPU 4/5。GPU 0、6、7 是其他租户的，永远不碰。
- 其他道的参数用环境变量覆盖：`CPUSET`、`MEMS`、`GPUS_OK`、`MEM`、`LOCK`、`SESSION`。
- 镜像 `aisaac-bench:20261019`：Ubuntu 22.04、Python 3.10.12、g++ 11.4、xvfb。
- 共享 venv `/bench/home/isaac-abplus/.venv`：torch 2.7.1+cu128、cuDNN 9.7.1。4090 是 cu118 / cuDNN 9.1.0，所以跨机器的数值不要求逐位一致。

## 目录（bm1/）

| 路径 | 内容 |
|---|---|
| `dock.sh` | 在镜像里跑一条命令。容器名 `aisaac-bench-bm1-a1-<用途>`，加 `--rm`，内存 64g，只读根文件系统，`/tmp` 是 tmpfs，用调用者的 uid，`--network none`，只给一张 GPU（容器里是 cuda:0） |
| `xnew2.sh` | 4090 `xnew2.sh` 的移植，在容器里执行 |
| `prof.sh` | 计时 profile，对应 4090 的 `prof2.sh` + `b62_step0.sh P24`：拿锁，`ISAAC_BLOCK_TIMING=1`，到 N 个 update 后给 train_tok 发 SIGINT。同时采样该卡的 nvidia-smi（1 s）、全部 GPU（5 s）和 `mpstat -P ALL 5`。跑完把 checkpoints 移进回收站 |
| `gate.sh` | 一棵树的全量门（pytest tests）。日志带时间戳，从不覆盖，失败的拷进 `reports/failed/` |
| `probe.sh` | `b62_capture_probe.py`（B62 的 capture 探针） |
| `deploy.sh` | `tree <名> <tar>`：解开一棵 git-archive 树。`abphome <树>`：建游戏目录 |
| `lock.sh` | 本道的锁 `LOCK_node0`，内容是 "<session> <时间> <用途>"，用 noclobber 原子地取 |
| `trees/<名>` | 部署的代码树：`git archive <commit> bridge/python bridge/ops bridge/abplus bridge/engine` |
| `home/` | 容器里的 HOME：各种缓存（isaac-actor、learn-native、visible、Inductor、Triton）都在这里 |
| `home/isaac-abplus` | ABP_HOME。`run_instance.sh` 从 `$HOME/isaac-abplus` 启动游戏：`runtime` 是指向共享只读 runtime 的符号链接，`tools/` 是从树的 `bridge/engine` 编出来的（libabp_turbo.so），`instances/` 是本目录自己的 |
| `pydeps/` | 共享 venv 缺的包，或版本和 4090 不同的包（见下），经 PYTHONPATH 加载 |
| `payloads/b45-payloads2` | 全量门要用的载荷（来自 4090 存档） |
| `tools/nsys-2023.4.4` | Nsight Systems 2023.4.4.54，和 4090 同版本。从 NVIDIA CUDA 官方源下载 deb，按 Packages 里的 sha256 校验后用 `dpkg -x` 解开，没有装进宿主系统。nsys 在 `opt/nvidia/nsight-systems/2023.4.4/bin/nsys`，QdstrmImporter 在 `.../host-linux-x64/` |
| `tools/cf/cf_rate.py` | 协调方的 CF 归一化列（4090 的 /tmp/cf_rate.py） |
| `reports/` | 所有结果和日志：`bm1.log` 是总日志，`<tag>/progress.csv`、`<tag>.stdout`、`<tag>.gpu`、`<tag>.allgpu`、`<tag>.mpstat` |

## 用法（在主机上）

```sh
cd ~/aisaac-bench-20261019/cpu-wide/bm1
# 部署一棵树（本地：git archive --format=tar -o mX.tar <commit> bridge/python bridge/ops bridge/abplus bridge/engine，上传到 bm1/）
./deploy.sh tree m77657fc m77657fc.tar
./deploy.sh abphome m77657fc            # 第一次，或 engine/abp_turbo.c 变了的时候
# 计时 24 轮（4090 的 P24 口径：上限 4500 s，30.0 游戏小时，24 个 update 后发 SIGINT）
GPU=1 UPDATES=24 HOURS=30.0 ./prof.sh base-m77657fc m77657fc
# 全量门连跑 3 次
GPU=1 RUNS=3 ./gate.sh m77657fc
# capture 探针
GPU=1 ./probe.sh m77657fc
```

额外的环境变量用 `ENVS="-e K=V ..."` 传进容器。例如 `ENVS="-e ISAAC_LEARN_CKPT_LEAN=0"`，或 `-e ISAAC_CUDA_BLOCKING_SYNC=blocking`。

规则：
- 计时运行持有锁的时候，本道不许有编译、pytest 或别的 GPU 作业。prof.sh、gate.sh、probe.sh 都要先拿同一把锁。
- 删除只用 `gio trash`。
- 测速 run 不留检查点。

## 和 4090 的差异

- 硬件：4090 是 Xeon Gold 6530（16 核）+ RTX 4090；这里是 Gold 6146 的 24 个逻辑 CPU（12 核 + 超线程，node0）+ RTX 3090。
- torch：4090 是 cu118，这里是 cu128（同为 2.7.1，triton 3.3.1）。
- venv 和 4090 的 pip freeze（存档 `env/pip-freeze.txt`）对比：
  - 共享 venv 缺、经 `pydeps/` 补上的（`--no-deps`，版本按 4090）：lz4 4.4.5（block_training 要用）、pytest 9.2.0（还有 iniconfig 2.3.1、pluggy 1.7.0、exceptiongroup 1.3.1、tomli 2.5.0、pygments 2.21.0）、py-spy 0.4.3。
  - 版本不同、经 `pydeps/` 改成 4090 版本的：numpy 1.26.4 → 2.2.6，zstandard 0.23.0 → 0.25.0。
  - 不同但没动的：filelock 3.32.3（4090 是 4.1.0）、fsspec 2026.7.0（4090 是 2026.9.0）、nvidia-*-cu12（4090 是 -cu11）。
  - venv 有而 4090 没有的：pandas、matplotlib、stable-baselines3、sb3-contrib、psutil 等，train_tok 不用。
- 游戏：runtime 用这台机器原有的只读 runtime。libabp_turbo.so 在容器里用 gcc 11.4 从同一份 `bridge/engine/abp_turbo.c` 编出来，和 4090 的源码字节相同；大小 111,584 字节，和 4090 的 .so 一样。bridge Lua 用树里自己的，和 4090 用的 candidate-a1 那份内容相同。
- 采样：4090 只有 `<tag>.gpu`。这里另外采全部 GPU 和 mpstat，用来判断有没有被其他租户污染。

## 两条道（lane.sh）

所有脚本先 source 同目录的 `lane.sh`。`LANE=a1`（默认）的行为和 5caa59a 相同。`LANE=cpp`：

- 容器名 `aisaac-bench-bm1-cpp-<用途>`。
- CPU `12-23,36-47`，内存节点 1，GPU 4/5。
- 锁 `bm1/LOCK_node1`，SESSION `bm1-cpp`。
- 脚本、HOME、游戏目录、reports、trees 都在 `bm1/cpp/` 下：
  - 脚本在 `bm1/cpp/ops/`，HOME 在 `bm1/cpp/home`；
  - 游戏目录 `bm1/cpp/home/isaac-abplus`：`runtime` 指向同一份共享只读 runtime，`tools/` 和 `instances/` 是本道自己的；
  - 结果和树在 `bm1/cpp/reports`、`bm1/cpp/trees`。
- PYTHONPATH 先放共享的 `bm1/pydeps`，再接本道的 `bm1/cpp/pydeps`。

两条道共享且只读的：镜像、共享 venv、游戏 runtime、`bm1/pydeps`、`bm1/tools`、`bm1/payloads`。在主机上用 cpp 道时，运行 `bm1/cpp/ops/` 下的那一份脚本，例如：

```sh
LANE=cpp ~/aisaac-bench-20261019/cpu-wide/bm1/cpp/ops/prof.sh <tag> <tree>
```
