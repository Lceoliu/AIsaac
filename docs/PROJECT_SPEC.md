# AIsaac：目标、观测边界与当前阶段

更新：2026-11-17。本文只写长期有效的原则和当前阶段。系统的组成见 [ARCHITECTURE.md](ARCHITECTURE.md)，实验和数字只在 [bridge/abplus/EXPERIMENTS.md](../bridge/abplus/EXPERIMENTS.md)。9 月的旧版（含模拟器阶段的分阶段设计）在 [archive/sim-era/PROJECT_SPEC-2026-09.md](archive/sim-era/PROJECT_SPEC-2026-09.md)。

## 1. 目标

- 做一个能打完整局《以撒的结合》的 AI：战斗、探索、资源管理、道具组合和路线决策都由策略自己完成。
- 训练和评估都在原版引擎上做。现在用的是 Afterbirth+ v1.06 的原生 Linux 版，经加速层并行运行；成绩只以原版引擎为准。
- 输入格式按兼容 Repentance 设计（visible-v3，见 [VISIBLE_V3_DESIGN.md](VISIBLE_V3_DESIGN.md) §1）：AB+ 上练出的检查点以后能在 Repentance / Repentance+ 上加载运行，人类录像也能转成同一格式。这是设计目标，还没有实现。

## 2. 观测边界

策略推理时只能用玩家能得到的信息。视觉识别或读内存都只是采集手段；内存里读得到，不等于策略可以用。

区分四层：引擎的完整状态、当下允许的观测、由过去允许的观测形成的记忆、评估和训练用的完整真值。只有中间两层送进 actor。

| 信息 | 策略可以接收 | 不应直接接收 |
|---|---|---|
| 玩家 | 可见 HUD、已知持有物、位置、允许的角色属性 | 现在看不到、也不能由历史确定的内部值 |
| 敌人与弹幕 | 当下可见实体的位置、形状、类别、可见动作、公开属性 | 隐藏或钻地实体的实时位置、内部 AI 状态、下次攻击计时器、未来随机数 |
| 血量与效果 | 屏幕上显示的信息，或确认允许的公开属性 | 没有血条的敌人的精确实时血量 |
| 道具 | 可识别的外观、已知效果、已持有的道具 | 问号道具的真实 ID、还没揭示的随机结果 |
| 地图 | 当前可见的几何、已揭示的地图、以前看到的内容 | 没探索的房间里的怪物、奖励、完整楼层拓扑 |
| 历史 | 自己过去的动作、过去允许的观测，以及由此做的推断 | 用隐藏的实时状态刷新"记忆" |

规则：
- 可见不等于实体中心落在镜头里：还要考虑实际显隐、遮挡和当前的游戏效果。
- 实体数量、排序、padding、动作掩码都只能依赖允许的信息。不能用内存地址、随机种子或能推出隐藏信息的实体 ID。
- 特权信息（例如真实楼层布局）只能进 learner 的价值头（特权 critic，见 [PREDICTIVE_MEMORY_DESIGN.md](PREDICTIVE_MEMORY_DESIGN.md) §4），不能经过奖励、日志特征或记忆状态流回 actor。
- 教师（受伤回溯、反事实分支）用克隆的引擎和共用的随机数，因此知道策略不知道的未来。它只用来产生训练目标；评估时不用克隆体。
- 接口验收：构造允许观测的历史相同、隐藏状态不同的成对场景，actor 的输入、动作掩码和记忆初始化必须完全相同。

## 3. 动作与时钟

- 动作只能是原版接受的合法输入。现在的动作头是移动 9、射击 5、炸弹 2，开道具时再加主动道具 2、药丸 / 卡牌 2（`tok_policy.HEADS`、`ITEM_HEADS`）；visible-v3 另加 Drop。不能用直接设速度、设位置或调用生成函数来代替按键。
- 每个决策推进固定的逻辑帧数（现在是 4 帧）。动作晚一个决策生效：这是环境的规定，正在执行的动作是状态的一部分。
- 死亡、通关这类终止和超时截断要分开。按所有者 11-12 的决定，整局时限和无进展都按截断处理（`train_tok.py --cap-truncate 1 --stall-truncate 1`，见 [SINGLE_TRANSFORMER_DESIGN.md](SINGLE_TRANSFORMER_DESIGN.md) R4、R16）。

## 4. 立场：可扩展性优先

- 总思路是 "big blob of compute" 假说 / Sutton 的 Bitter Lesson：长期看，专用的聪明办法比不过算力、数据量和一个能扩展的目标。手写规则、护盾、专才网络都已放弃（2026-10-03 的决定）。
- 模型结构、规模和算法都是开放变量，由对照实验和留出种子评估决定。我们自己加的归纳偏置要做消融。
- 证据：同配方下把模型放大约 1.5 倍（C100 对 C99），前 12,000 游戏小时过层领先 1.5–2 倍，到 25,000 小时时差别消失（EXPERIMENTS.md C100）。训练量相对 OpenAI Five、AlphaStar 一类项目仍差几个数量级（[archive/sim-era/SCALING_THESIS.md](archive/sim-era/SCALING_THESIS.md) §1）。
- 吞吐和硬件利用率是一等目标：CPU（游戏进程）和 GPU（推理、学习）都要用满，等待和空转要能定位到具体原因。

## 5. 当前阶段（2026-11-17）

- **主线**：单序列因果 transformer "seq-v1"（[SINGLE_TRANSFORMER_DESIGN.md](SINGLE_TRANSFORMER_DESIGN.md)），每个决策一个 token，输入直接用 visible-v3。所有者 2026-11-17 批准。设计稿阶段，代码还没写；visible-v3 目前只有冻结的接口文件（`bridge/python/isaac_bridge/schema_v3.py`、`bridge/engine/abp_schema_v3.h`）。
- **参考基线**："旧模型"（逐帧 token transformer + 时间 transformer，[TEMPORAL_ATTENTION.md](TEMPORAL_ATTENTION.md)）。它能学会：从零的长训见 EXPERIMENTS.md C99、C100，400 游戏小时对照见 C107、C110。
- **搁置**：block-memory v1（设计见 [archive/block-memory-v1/](archive/block-memory-v1/PROJECT_SPEC.block-memory-v1.md)）。C107–C113 的 400 游戏小时对照里它学不会真实任务，旧模型能。11-17 起不再投入；代码（`block_*.py`、C++ actor driver、C 端打包）留在仓库里，seq-v1 会复用其中的系统部分。
- 每个新模型先过分级的诊断阶梯（例如 C112 的 L1 / L2 奖励阶梯），再上真实任务的 400 小时对照，最后才是正式长训。

## 6. 不做的事

- 不在模拟器上训练再迁移：Rust 模拟器 `sim/` 9 月起搁置。
- 不引入 RGB 视觉网络：输入是结构化的可见信息。
- 不把手写的宏观规划器接进训练（10-03 放弃手写规则的决定）：`macro/` 的工具只用于离线分析，教师里只移植了隐藏房位置这一条游戏规则。
- 不用人类数据作为训练的前提。人类录像格式另有仓库（isaac-run-recorder），能否用于训练是以后的问题。
- 不修改玩家自己的游戏安装；仓库不包含游戏文件。
