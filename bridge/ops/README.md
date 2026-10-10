# 运维脚本（2026-10-10 起随仓库管理）

- `sync_h1.sh <沙箱>` / `sync_to.sh <沙箱>` / `sync_hpc.sh <沙箱>`：把 `bridge/python`、`bridge/abplus/abp_bridge.lua`、`bridge/abplus/catalog` 和 `bridge/engine/` 同步到第一台 / 第二台 / HPC 的 `~/isaac-abplus/<沙箱>`（HPC：`/101063/AIsaac/<沙箱>`）并重新编译 `libabp_turbo.so`。沙箱是某一时刻工作区的快照，正在跑的实验不随工作区变。
- `c6x–c8x_start.sh`：各次训练的启动脚本（`smoke|run`），参数即实验记录（EXPERIMENTS.md）。
- `hpc_env.sh` / `hpc_bench.sh` / `hpc_guard.sh` / `hpc_c77.sh`：HPC 节点的环境、吞吐基准、资源守护、十游戏年运行。
- `run_evals.py`（整局评估表）、`choice_stats.py`（分支记录统计）、`lab_trend.py` / `lab_compare.py`（构筑实验室）。

## git 管理（2026-10-10 起）

- 本地仓库 `rl/`（GitHub `Lceoliu/AIsaac`，分支 `abplus-training` = `main`）。每个里程碑提交并推送。
- 三处远端各有一个裸仓库 + 工作副本：第一台 `~/isaac-abplus/repo.git` / `repo`，第二台同，HPC `/101063/AIsaac/repo.git` / `repo`（第二台和 HPC 没有外网，从 PC `git push ssh://…/repo.git --all` 推过去，再在远端 `git pull`）。
- 正在跑的运行用的代码各记成一个快照分支（`snapshot/host1-tv` = C83、`snapshot/host2-lf` = C80/C81/C82、`snapshot/hpc-ws1` = C77），都在 GitHub 上；远端的 `repo` 检出的是各自的快照分支。
- 新沙箱从今往后用仓库建：`git -C <repo> worktree add <沙箱>/src <分支或提交>`，然后 `ln -s src/bridge/python python; ln -s src/bridge/abplus abplus`，`tools/` 由 `sync_*.sh` 的编译步骤或手动 `gcc -shared -fPIC -O2 -o tools/libabp_turbo.so src/bridge/engine/abp_turbo.c -ldl` 生成；`sync_*.sh` 的 tar 同步仍可用于未提交的工作区。
