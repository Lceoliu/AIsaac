# 归档的运维脚本

2026-11-17 用 `git mv` 从 `bridge/ops/` 移到这里，共 126 个文件，内容没有改。它们是过去实验的一次性启动器、测量脚本和结果统计脚本，现行的脚本、测试和代码都不引用它们。

注意：
- [EXPERIMENTS.md](../../abplus/EXPERIMENTS.md) 的条目里写的 `ops/<文件名>` 或 `bridge/ops/<文件名>`，如果在 `bridge/ops/` 下找不到，就在这里。条目正文保留了当时的路径，没有改。
- 这些脚本写死了已经不可用的机器（主机 1、主机 2、HPC 节点、租用的 4090）上的路径，不要直接重跑。它们的价值是记录当时用的参数。
- 脚本之间按相对路径互相调用。被现行脚本引用的 26 个留在了 `bridge/ops/`（见 [../README.md](../README.md)），所以这里的脚本里有些相对路径现在指不到文件。

| 文件 | 内容 |
|---|---|
| `c68_start.sh`–`c89_start.sh`（17 个） | C68–C89 的启动脚本（主机 1、主机 2） |
| `hpc_c77.sh`、`hpc_c84.sh`–`hpc_c104.sh`、`hpc_c92_resume.sh`、`hpc_c92_ws6.sh` | HPC 节点上 C77–C104 的启动脚本（`hpc_c99.sh` 留在上一层） |
| `launch_*.sh`（6 个） | HPC 节点上“等上一个运行结束再接下一个”的链 |
| `mk_ws5.sh`–`mk_ws16.sh`（`mk_ws14.sh` 留在上一层） | HPC 节点上的沙箱建立脚本 |
| `hpc_b24.sh`、`hpc_b28_throughput.sh`、`hpc_b29_knobs.sh`、`hpc_b38_step4.sh`、`hpc_bench.sh` | HPC 节点上 B24–B38 的测量 |
| `b38_summary.py`、`b42_*`、`b48_*`、`b49_grad_stats.py`、`b50_nsys.py`、`b51_scale_bench.py`、`b55_compile_probe.py`、`b56_verify*.sh`、`b58_*`、`b58e_gilq.py`、`b61a.sh`、`b65_learner_probe.py`、`b66_static_waste.py` | B38–B66 的测量、剖析和汇总 |
| `b53/` | B53 四臂对照的脚本和臂定义（四臂未运行） |
| `run_b52k.sh`、`run_b52t.sh`、`run_b54t.sh`、`run_b57m.sh`、`run_b58*.sh`、`run_d1*.sh`、`run_m1t.sh`、`run_merge_probe.sh`、`threads_snapshot.sh`、`p2p_probe.py` | 4090 和 8x3090 主机上 B52–B58、cpp 线的计时运行 |
| `host1_*.sh`、`host2_*.sh`、`sync_h1.sh`、`sync_to.sh`、`sync_hpc.sh`、`relay_eval.sh`、`ev128_mem.sh`、`ev512_3090.sh` | 已不可用主机的同步、迁移和评估中转 |
| `run_evals.py`、`choice_stats.py`、`lab_trend.py`、`lab_compare.py`、`cf_rates.py` | 结果统计：整局评估表、分支记录、构筑实验室、反事实比率 |
| `pred_evalq.py`、`pred_host.py`、`pred_*.json` | 预测记忆实验（辅助损失、特权 critic）的脚本和门槛文件 |
