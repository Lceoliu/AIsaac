# 文档归档

这里放已经过时、但保留作记录的文档。2026-11-17 整理时用 `git mv` 移入，内容没有改，只把内网地址、ssh 登录串和本机路径换成了别名（`<主机 1>`、`<主机 2>`、`<项目根>`、`<仓库根>` 等）。

注意：
- 归档文档里的相对链接仍按它们原来的位置写，很多已经打不开。要找对应的文件，按文件名在仓库里搜。
- 文中的"当前""现状""第一台 / 第二台 / HPC"都是写作当时的情况。这些机器现在都已不可用。
- 实验结果以 [bridge/abplus/EXPERIMENTS.md](../../bridge/abplus/EXPERIMENTS.md) 为准。
- 这里只放文档。归档的代码在 [archive/](../../archive/README.md)，归档的一次性运维脚本在 [bridge/ops/archive/](../../bridge/ops/archive/README.md)。

| 文件 | 最后一次内容修改 | 所属的线 | 为什么归档，由什么代替 |
|---|---|---|---|
| [j460/ARENA_RESEARCH_NOTES.md](j460/ARENA_RESEARCH_NOTES.md) | 2026-09-20 | 忏悔+ J460 原版训练场 | 把原版引擎做成训练场的原始调研笔记。J460 线已暂停；训练改在 AB+ 引擎上做，见 [ARCHITECTURE.md](../ARCHITECTURE.md) |
| [j460/TRAINING_ARENA_DESIGN.md](j460/TRAINING_ARENA_DESIGN.md) | 2026-09-22 | 忏悔+ J460 原版训练场 | 同上，J460 训练场的实施方案 |
| [j460/L1_FEASIBILITY_PLAN.md](j460/L1_FEASIBILITY_PLAN.md) | 2026-09-20 | 忏悔+ J460 加速层（L1） | J460 加速的可行性方案。加速层 `turbo/` 已暂停；AB+ 上的加速层在 `bridge/engine/` |
| [j460/NATIVE_CRASH_ANALYSIS.md](j460/NATIVE_CRASH_ANALYSIS.md) | 2026-09-20 | 忏悔+ J460 worker | Windows 上原生崩溃的根因分析。只和暂停的 J460 线有关 |
| [sim-era/ENV_ARCHITECTURE.md](sim-era/ENV_ARCHITECTURE.md) | 2026-09-30 | Rust 模拟器阶段 | 三层环境栈（原版 / 加速 / 模拟器）和模拟器训练记录。模拟器 `sim/` 9 月起搁置；现在的系统见 [ARCHITECTURE.md](../ARCHITECTURE.md) |
| [sim-era/SINGLE_ROOM_V0.md](sim-era/SINGLE_ROOM_V0.md) | 2026-09-20 | 旧原型 `isaac_room/` | 已否定的自编规则原型 |
| [sim-era/PROJECT_SPEC-2026-09.md](sim-era/PROJECT_SPEC-2026-09.md) | 2026-09-30 | 项目目标（9 月版） | 主线描述停在 09-23 的模拟器阶段。目标和观测边界（§1、§3）已并入新的 [PROJECT_SPEC.md](../PROJECT_SPEC.md) |
| [sim-era/SCALING_THESIS.md](sim-era/SCALING_THESIS.md) | 2026-10-10 | 10 月的训练思路 | 10-08 的算力 / 数据 / 价值视野讨论。立场（§1）并入 [PROJECT_SPEC.md](../PROJECT_SPEC.md)；其余是当时的安排，已执行完或过时。代码注释里的 "SCALING_THESIS §5b" 指这里 |
| [2026-09-combat/COMBAT_RL_EXPERT_REVIEW.md](2026-09-combat/COMBAT_RL_EXPERT_REVIEW.md) | 2026-09-30 | 9 月的房间战斗 RL | 截至 C37 的进展汇总，给外部评审用 |
| [goal-conditioned/GOAL_CONDITIONED_DESIGN.md](goal-conditioned/GOAL_CONDITIONED_DESIGN.md) | 2026-10-07 | 目标条件 COMBAT / GOTO | 9 月底的 π(a\|s,g) 方案，C42–C45 做过，10-03 起换成 token 策略 |
| [goal-conditioned/goal_conditioned/](goal-conditioned/goal_conditioned/) | 2026-10-07 | 同上 | 方案用的原型脚本、留出房间划分脚本和 10-07 的远端结果快照 |
| [block-memory-v1/PROJECT_SPEC.block-memory-v1.md](block-memory-v1/PROJECT_SPEC.block-memory-v1.md) | 2026-11-10 | block-memory v1 | 四 latent 分块记忆模型的设计（§9.13）。C107–C113 里它学不会真实任务，2026-11-17 起不再投入；新方向见 [SINGLE_TRANSFORMER_DESIGN.md](../SINGLE_TRANSFORMER_DESIGN.md) |
| [block-memory-v1/STREAM_LOG_FORMAT.md](block-memory-v1/STREAM_LOG_FORMAT.md) | 2026-11-07 | 预测记忆 / 世界模型 | `--log-stream` 决策流日志的格式。它属于 block-memory 一线时期的文档集；写日志的代码 `tok_stream.py` 和读日志的 `abplus_world_model.py` 还在，注释仍指向旧路径 |
| [abplus-README-2026-10.md](abplus-README-2026-10.md) | 2026-10-16 | AB+ 桥接与训练管线 | 10 月中旬的完整 README（约 980 行）：旧 `train_abplus.py` 管线、奖励演变、瞄准场、对战场、Go-Explore、token 策略各开关的说明、运维。新的精简版是 [bridge/abplus/README.md](../../bridge/abplus/README.md) |
