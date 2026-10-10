# 运维脚本（2026-10-10 起随仓库管理）

- `sync_h1.sh <沙箱>` / `sync_to.sh <沙箱>` / `sync_hpc.sh <沙箱>`：把 `bridge/python`、`bridge/abplus/abp_bridge.lua`、`bridge/abplus/catalog` 和 `bridge/engine/` 同步到第一台 / 第二台 / HPC 的 `~/isaac-abplus/<沙箱>`（HPC：`/101063/AIsaac/<沙箱>`）并重新编译 `libabp_turbo.so`。沙箱是某一时刻工作区的快照，正在跑的实验不随工作区变。
- `c6x–c8x_start.sh`：各次训练的启动脚本（`smoke|run`），参数即实验记录（EXPERIMENTS.md）。
- `hpc_env.sh` / `hpc_bench.sh` / `hpc_guard.sh` / `hpc_c77.sh`：HPC 节点的环境、吞吐基准、资源守护、十游戏年运行。
- `run_evals.py`（整局评估表）、`choice_stats.py`（分支记录统计）、`lab_trend.py` / `lab_compare.py`（构筑实验室）。
