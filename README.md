# AIsaac：原版引擎上的完整流程强化学习

更新：2026-10-21。

## 当前阶段

主线是在 **Afterbirth+ 原生 Linux 引擎**上训练一个覆盖战斗、探索、拾取和资源取舍的策略。Rust 模拟器、旧自编规则 v0、COMBAT/GOTO 组合与护盾专家是历史路线，不是当前主线。

- 冻结参照：C99 第二段最终权重 `segment2-final.pt`，u48184，累计 58,506.2 游戏小时；512 种子平均过层 **1.953**。这是平均过层数，不是整局通关率。
- 当前工作：房间记忆的信用分配、训练时旧记忆的使用，以及输入信息缺失。候选实现与验证在工作区 `temp/room-memory-write-v1/`，不能自动视为已合并到本仓库或已通过完整训练验收。
- 下一版输入结构已讨论并画入修改副本；**文字设计与 Draw.io 的统一另做下一步**。旧设计稿中道具查询池化等方案尚未同步，不能继续当最新定案。
- C103/C104 的旧“进行中”和 ETA 已撤下；本地总账没有可据以判定最终胜负的完整结果。本次文档整理没有连接远端，也没有启动训练。

## 文档导航

| 要查什么 | 唯一主入口 |
|---|---|
| 已实现模型、时序状态、教师与 loss 的关系 | [架构](docs/ENV_ARCHITECTURE.md) |
| 用户确认的可观察性、操作接口与待实现项 | [接口规范](docs/PROJECT_SPEC.md) |
| 训练、评估、回放与运行产物 | [训练管线](bridge/abplus/README.md) |
| 各实验的关键条件、结论与数据 | [实验总账](bridge/abplus/EXPERIMENTS.md) |
| Bitter Lesson、实验设计和验收原则 | [研究原则](docs/SCALING_THESIS.md) |
| 外部项目和接口经验 | [相关工作](docs/RELATED_WORK.md) |

设计入口：[原始结构图](docs/model_structure_versions/20261018/main.drawio)、[修改副本](docs/model_structure_versions/20261020-token-expansion-proposal/main.drawio)、[待同步的下一版文字设计](docs/model_structure_versions/20261018/resources/NEXT_MODEL_DESIGN.md)。

## 代码与工具

| 目录 | 用途 |
|---|---|
| `bridge/abplus/` | AB+ Lua 桥接与房间目录 |
| `bridge/python/` | `train_tok.py`、`eval_tok.py`、策略、采样器、教师和评估工具 |
| [bridge/engine](bridge/engine/README.md) | 原生加速、fork 克隆与观测编码 |
| [bridge/ops](bridge/ops/README.md) | 部署、资源隔离、完整实验备份 |
| [macro](macro/README.md) | 离线生成器、推断与对照工具，不是在线策略的手写控制器 |
| [bridge](bridge/README.md)、[turbo](turbo/README.md) | 历史 J460 桥接、加速与实机证据 |
| `isaac_room/`、`sim/` | 历史模拟器与原型；不再作为当前训练主线 |
| `runs/` | 历史运行、评估及证据；不是当前设计入口 |

公开 Gym 接口在独立的 [isaac-abplus-gym 仓库](../isaac-abplus-gym/README.md)，不要把它与本研究仓的所有实验选项混为一谈。

## 验收与状态

只向策略提供玩家可观察的信息；训练教师可用克隆分支产生监督，但正式评估不能读取未来或靠教师代打。按相同种子、检查点、开局分布和终止规则比较；128 种子用于跟踪，512 种子用于候选正式比较，必要时再做多训练种子复验。

远端进程、GPU 占用和可用路径必须现场检查；历史主机名、脚本与 ETA 不是当前运行状态。每个完整大实验结束后，将最终/可恢复权重、配置、训练和评估日志、评估结果同步到备用 3090，并确认可读。

## 历史路线与文档整理

模拟器 u425 在模拟器上胜率 19.1%、迁移 AB+ 只有 0.4%（A2），因此转向原版引擎训练。已否定的自编规则 v0 不计作以撒能力成果。

2026-10-21 将 23 份正文整理为 13 份：时序/反事实设计并入架构，旧故障与加速结论并入模块入口，宏观规划与目标条件路线的有效边界并入离线工具文档，重复研究稿并入相关工作；旧阶段计划和专家评审副本删除。原始内容仍可从 Git 历史查阅，不在当前目录再保存一套“完整旧版正文”。
