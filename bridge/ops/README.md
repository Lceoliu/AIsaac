# 运维脚本

更新：2026-11-17。

## 入口：bm1

现在所有的训练、计时、测试门和评估都通过 [bm1/](bm1/README.md) 运行。主机相关的设置只放在 `bm1.env` 里，从 `bm1/env/*.env` 复制，脚本里不写硬件数字。操作手册模板是 [bm1/TRAINING.md](bm1/TRAINING.md)。

## 其余脚本

这一层的大部分脚本是过去实验的一次性启动器和测量脚本，留作记录。它们的参数就是当时的实验记录，对应 [EXPERIMENTS.md](../abplus/EXPERIMENTS.md) 的同号条目。很多脚本写死了已经不可用的机器上的路径，不要直接重跑。

| 脚本 | 内容 |
|---|---|
| `c6x_start.sh`–`c8x_start.sh`、`hpc_*.sh`、`launch_*.sh` | C68–C104 的启动脚本；HPC 节点的环境、守护和十游戏年运行 |
| `mk_ws*.sh` | HPC 节点上的沙箱建立脚本 |
| `b38_*`–`b66_*`、`b53/`、`run_*.sh` | B38–B66 的测量和计时 |
| `host1_*.sh`、`host2_*.sh`、`sync_h1.sh`、`sync_to.sh`、`sync_hpc.sh` | 已不可用的主机 1、主机 2、HPC 节点的同步和迁移 |
| `run_evals.py`、`choice_stats.py`、`lab_trend.py`、`lab_compare.py`、`cf_rates.py`、`clone_stats.py` | 结果统计：整局评估表、分支记录、构筑实验室、反事实比率、克隆统计 |
| `pred_*` | 预测记忆实验（辅助损失、特权 critic，见 `docs/PREDICTIVE_MEMORY_DESIGN.md`）用的脚本和门槛文件 |

## git

- 远端 `origin` 是 GitHub 上的 `Lceoliu/AIsaac`，仓库根就是本目录的上两级。
- `main` 是主线。block-memory v1 合并进 `main` 之后，`exp/block-memory-v1` 与它指向同一个提交。
- 旧分支只以 `archive/*` 标签保留，例如：
  - `archive/snapshot/*`：当年各台主机正在跑的代码快照；
  - `archive/exp/memory-chain-v3*`。
- 部署不在主机上 `git pull`：用 `git archive <commit> bridge/python bridge/ops bridge/abplus bridge/engine` 打成 tar，再用 `bm1/deploy.sh tree` 解开。每棵树用提交号命名，正在跑的运行不受之后提交的影响。
