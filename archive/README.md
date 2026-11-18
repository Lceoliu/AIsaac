# 归档的代码

这里放已经不属于任何现行管线、但保留作记录的代码。文档的归档在 [docs/archive/](../docs/archive/README.md)，一次性运维脚本的归档在 [bridge/ops/archive/](../bridge/ops/archive/README.md)。

## human-recorder-2026-09/

2026-11-17 用 `git mv` 移入，内容没有改，目录结构保持原来相对仓库根的路径。

这是 9 月的人类录制和行为克隆一线，共 13 个文件：

| 文件（相对本目录） | 内容 |
|---|---|
| `bridge/mod/isaac_human_recorder/` | 游戏内的录制 mod（忏悔+ J460，Windows） |
| `record_human.ps1`、`bridge/python/record_human.py` | 启动录制的脚本 |
| `bridge/python/isaac_bridge/human_recording.py`、`human_dataset.py` | 录制文件的读取、校验和数据集 |
| `bridge/python/isaac_bridge/bc_experiment.py`、`bridge/python/train_human_bc.py`、`check_human_dataset.py` | 用录制数据做行为克隆的实验 |
| `bridge/python/test_human_recording.py`、`test_human_dataset.py`、`test_bc_experiment.py` | 以上代码的测试 |

为什么归档：
- 这一组只在彼此之间引用，仓库里别的代码不导入它们。
- 录制器后来重写了（按帧记录、可精确回放），2026-11-10 起是单独的仓库 isaac-run-recorder。这里的是更早的那一版，不是它的副本。
- 归档后这些模块不在 `isaac_bridge` 包里，不能直接导入；要运行需要放回原路径。它们依赖的 `transformer_obs.py`、`transformer_policy.py` 仍在 `bridge/python/isaac_bridge/`。
