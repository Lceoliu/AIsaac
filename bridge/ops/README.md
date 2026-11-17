# 运维脚本

更新：2026-11-17。

## 入口：bm1

现在所有的训练、计时、测试门和评估都通过 [bm1/](bm1/README.md) 运行。主机相关的设置只放在 `bm1.env` 里，从 `bm1/env/*.env` 复制，脚本里不写硬件数字。操作手册模板是 [bm1/TRAINING.md](bm1/TRAINING.md)。

## 这一层留下的脚本

这 26 个文件仍被 bm1 的脚本、测试或代码引用（直接引用，或被这些文件引用），所以留在原位。它们本身大多是过去实验的脚本，参数对应 [EXPERIMENTS.md](../abplus/EXPERIMENTS.md) 的同号条目。

| 脚本 | 谁在用 |
|---|---|
| `hpc_c99.sh`、`hpc_env.sh` | `bm1/xold_c99.sh`、`bm1/ev_ab2.sh`：旧模型 C99 的命令行和评估协议 |
| `hpc_b38.sh`、`hpc_guard.sh`、`mk_ws14.sh` | `tests/test_learner_device_smoke.py` |
| `b62_step0.sh`、`b60_gpu.sh`、`b62_gpu_idle.py`、`run_b50m.sh`、`run_b54p.sh` | `bm1/gate.sh`、`bm1/prof.sh` |
| `run_b54g.sh` | `bm1/gate3.sh` |
| `run_b52g.sh`、`run_b52p.sh`、`b52_kernels.py`、`b52_learner_kernels.py` | `tests/test_b52_entry.py` |
| `b57_nsys.py`、`b57_threads.py`、`b62_learner_sections.py`、`b62_capture_probe.py` | `bm1/nsys_gil.sh`、`bm1/nsys_update.sh`、`bm1/prof.sh`、`bm1/m0_mem.py`、`bm1/probe.sh` |
| `run_core_tsan.sh` | `bm1/tsan_core.sh` |
| `b60_prefetch_spread.py` | `tests/test_b48_prefetch.py` |
| `b42_kernels.py`、`b43_memclear_compare.py`、`b44_learner_bench.py`、`b62_capture_probe2.py`、`clone_stats.py` | `bridge/python`、`bridge/engine` 里的代码或注释指向它们 |

## 归档的脚本

其余 126 个一次性启动器和测量脚本在 2026-11-17 移到了 [archive/](archive/README.md)，内容没有改。

## git

- 远端 `origin` 是 GitHub 上的 `Lceoliu/AIsaac`，仓库根就是本目录的上两级。
- `main` 是主线。block-memory v1 合并进 `main` 之后，`exp/block-memory-v1` 与它指向同一个提交。
- 旧分支只以 `archive/*` 标签保留，例如：
  - `archive/snapshot/*`：当年各台主机正在跑的代码快照；
  - `archive/exp/memory-chain-v3*`。
- 部署不在主机上 `git pull`：用 `git archive <commit> bridge/python bridge/ops bridge/abplus bridge/engine` 打成 tar，再用 `bm1/deploy.sh tree` 解开。每棵树用提交号命名，正在跑的运行不受之后提交的影响。
