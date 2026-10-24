# 游戏 RL：结构化观测、并行环境与迁移经验

调研日期：2026-09-18。以下使用原论文、官方接口和项目作者源码；社区成绩仅按作者报告记录，本轮没有运行这些外部项目。项目设计建议见 [PROJECT_SPEC.md](PROJECT_SPEC.md)。

## 1. 总览

| 游戏/项目 | 数据从哪里来 | 并行训练依赖什么 | 对 Isaac 最值得借鉴的部分 |
|---|---|---|---|
| Dota 2 / OpenAI Five | Bot API 的结构化实体与特征，不是截图 | 多个真实游戏环境产生 rollout，推理与学习分离 | 结构化观测、战争迷雾限制、异步采样与版本管理 |
| StarCraft II / PySC2、AlphaStar | 官方 API 的 raw units / feature layers / RGB 等接口 | 官方引擎的多个实例；PySC2 支持 Linux 无头环境 | 可控环境接口、镜头/可见性约束、小任务向全局扩展 |
| Clash Royale / 社区项目 | 视觉识别成结构化棋盘，或独立模拟器状态 | 独立 headless 模拟批量跑；真实端接模拟器/设备 | 两端统一观测动作，测量视觉误差和仿真偏差 |
| Sekiro / alpha-sekiro | 截图与可选姿态估计，加内存状态 | 控制真实 Windows 游戏；未见规模化环境证据 | 原版 reset 自动化、战斗任务闭环；不能当纯视觉基准 |
| Dark Souls III / SoulsGym | 内存与控制接口，状态式 Gym 环境 | 依赖真实游戏进程 | 环境重置/控制工程；警惕精确隐藏状态混入观测 |

各行证据与限制如下。大规模 RL 通常不是“把训练代码开多线程”就成立，首先需要可重置、可计时、可采样的环境后端。

## 2. Dota 2：API 观测，而不是从视频中学会一切

OpenAI Five 使用游戏 Bot API 获取结构化观测并输出动作。战争迷雾约束仍然存在；不可见的敌方英雄可保留先前可见时的信息，但不是持续读取其隐藏实时状态。CPU rollout、GPU 推理和优化分工，大量环境生成训练轨迹；论文讨论了异步采样的策略陈旧度问题。[官方介绍](https://openai.com/index/openai-five/)、[原论文](https://arxiv.org/html/1912.06680v1)

对 Isaac 的启发：结构化输入与玩家信息约束并不矛盾。采集器可以知道更多，actor 不可以；已知历史与实时隐藏真值必须分开。大量模拟环境也应记录策略版本，避免只追求每秒步数而忽略数据有效性。这是方法迁移建议，不意味着 Isaac 已拥有 Dota 的接口或基础设施。

## 3. StarCraft：raw 数据、摄像机和可见性是不同约束

PySC2 暴露官方 StarCraft II API，包含特征层、RGB 和 raw unit 相关接口，提供环境封装与 mini-games；官方环境支持 Linux headless 使用。并行工作的基础是多个游戏实例，不是独立重写了 StarCraft 引擎。[PySC2](https://github.com/google-deepmind/pysc2)、[环境说明](https://github.com/google-deepmind/pysc2/blob/master/docs/environment.md)

raw units 不自动等于“当前屏幕能看到的一切且仅有这些”。接口配置包含镜头位置、隐形单位与额外图层等选项，必须检查具体配置和字段。[接口实现](https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/features.py)

不要混淆早期 AlphaStar 展示与后来的 Grandmaster 版本：后者明确收紧了摄像机与动作方面的约束。[早期介绍](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)、[Grandmaster 版本](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)

对 Isaac 的启发：先定义信息和动作契约，再定义网络。小任务课程可以先验证战斗控制，再进入有记忆的整局探索；“从内存读到实体”本身并不能通过公平观测验收。

## 4. 皇室战争：最接近独立模拟器加真实端桥接，但要看负面证据

**Hasty-CR** 的仓库描述了从游戏数据构建独立模拟、批量 headless 运行、视觉桥接以及共用观测/动作接口；真实端用目标检测等组件恢复棋盘信息，并提供录制轨迹与模拟行为比较的思路。仓库提供批量 worker 和多环境训练入口。其精度、测试数及胜率是作者陈述，本轮未复现。[项目仓库](https://github.com/hastylmao/Hasty-CR)

**ClashAI** 当前 README 记录过一个反例：手写模拟器加 RL 的路线未能超过原有模仿学习起点，作者将其退役。当前视觉管线将实体、卡牌、圣水和血量转成结构化状态；项目采用观测契约并讨论基于测得误差的退化处理。这不是“所有独立模拟都失败”的证据，但说明接口统一和模拟器能跑并不保证策略迁移。[项目与实验回顾](https://github.com/vegetableleaf/ClashAI)

对 Isaac 的启发：优先统一语义接口、录制可对齐的测试片段、识别会改变决策的动力学误差。不能以作者宣传的仿真精度或本地模拟胜率代替原版测试。这里只研究工程方法，不部署在线竞技自动化。

## 5. 只狼：真实客户端闭环，且不是纯视觉输入

`alpha-sekiro` 使用视觉网络/DQN，README 报告了特定 Boss 阶段的实验，不能据此推断全游戏能力。[作者仓库](https://github.com/Chen-Wendi/alpha-sekiro)

源码核查发现：`Observer.getObs()` 用截图构建图像输入，可选目标检测与姿态特征；但玩家 HP、架势和 Boss HP 直接来自 `memory.getStatus()`。因此它是**混合观测**，不能把“用了视觉模型”表述成纯视觉实验。[observation.py](https://github.com/Chen-Wendi/alpha-sekiro/blob/main/src/env/observation.py#L113-L190)

环境 reset 会移动角色、恢复角色与 Boss，并操作窗口；内存模块还包含伤害修改。这些控制有助于训练，但会改变实验条件，评估时必须分离恢复机制和战斗规则。当前核查没有证明其支持大规模并行实例。[环境代码](https://github.com/Chen-Wendi/alpha-sekiro/blob/main/src/env/sekiro_env.py#L106-L131)、[内存模块](https://github.com/Chen-Wendi/alpha-sekiro/blob/main/src/env/memory.py)

对 Isaac 的启发：内存可以承担场景管理和采样，不代表所有内存字段都应该送进策略。训练时便利的无敌、回血、减伤或锁定辅助，不能悄悄进入最终验收环境。

## 6. SoulsGym：把动作游戏变成 Gym，依然需要观测审计

SoulsGym 用真实游戏的内存/输入控制构建 Gym 环境，文档涉及重置、速度控制和游戏状态。它不是只狼项目；当前功能与后续路线图也不能混为一谈。仓库明确提醒初始环境中的精确位置、动画时间等真值并非全部是玩家直接可见的信息。[仓库](https://github.com/amacati/SoulsGym)、[API 文档](https://soulsgym.readthedocs.io/en/latest/soulsgym.html)

对 Isaac 的启发：可以借鉴 reset/step 和进程控制方法，但必须自行加玩家视角投影。真实引擎可靠不等于无头高速或大规模多开已经解决。

## 7. 对本项目的结论

**推荐混合路线：独立简化模拟器承担样本量，少量原版实例承担校准、迁移训练和最终验收。** 这是综合上述证据作出的工程建议，不是现有性能结论。

1. 先实现结构化可见状态的语义契约；视觉是可替换的感知前端，内存是可替换的采集后端。
2. 先有单房间原版闭环，再扩大独立模拟覆盖；不等完整反编译，也不跳过原版数据。
3. 独立环境并行数、真实端速度与 reset 稳定性都需测量，不能套用 Dota/StarCraft 的规模。
4. 将仿真误差与感知误差分开测试；单独报告结构化、视觉、零样本迁移和微调后的成绩。
5. 接口审计、留出场景和原版指标优先于模拟器内容数量。全局策略在战斗达标后加入，不把“全游戏 AI”缩减成单 Boss 演示。

已有 Isaac RL 探索也值得复用接口思路，但要核对动作是否真等价于原版输入；例如直接改速度或调用生成攻击接口不能自动当成合法按键策略。[The-RL-of-Isaac](https://github.com/Seladus/The-RL-of-Isaac)

## 8. 以撒专项项目（2026-09-19 补充）

| 项目 | 做法 | 结果与教训 |
|---|---|---|
| [iamyanbo/Isaac-RL](https://github.com/iamyanbo/Isaac-RL) | Repentance 1.7.9b；Lua mod + 游戏自带 LuaSocket 做 TCP（每实例一端口、换行 JSON、序号/请求 ID）；游戏在每个动作边界同步等待 Python；8 个逻辑帧一个动作；观测为 10×16×28 空间网格 + 392 个标量/实体/门特征（最近 16 个敌人、16 个弹幕、8 个拾取物）；动作 `MultiDiscrete([9,5,4])`；用控制台命令与种子重置 | 实测采样吞吐：单实例 3.6 动作/秒、3 实例 10.6、6 实例 20.3；六个 worker 累计 1,221,432 步时 19.29 步/秒；最终 12,381,564 次决策、2284 局 GRU 训练，一层 Boss 零胜，作者自述"尚未达到要求的成功标准"。这是"只靠实时原版桥接"的负面证据：吞吐不足两个数量级 |
| [2-X/binding-of-isaac-ai](https://github.com/2-X/binding-of-isaac-ai) | Afterbirth+ 时代的 Lua mod，需 Steam 启动参数开启 Lua debug；规则/脚本式代理，实时运行 | 无成绩报告；证明 mod 内直接驱动玩家的路径早已可行 |
| [Supermaxman/SerpentSuperAIsaacGameAgentPlugin](https://github.com/Supermaxman/SerpentSuperAIsaacGameAgentPlugin) | SerpentAI 截图 + 按键，Monstro 单 Boss PPO | 纯视觉、异步实时，不可扩展到多实例 |
| [LanbingIce/IsaacSocket-Mod](https://github.com/LanbingIce/IsaacSocket-Mod) | Lua mod + 外部连接工具，为其他 mod 提供 WebSocket/HTTP/剪贴板及未开放 API | 证明 Rep/Rep+ 上 Lua 与外部进程通信是成熟做法；本项目不依赖它 |

对本项目的结论：Lua mod 同步单步 + 控制台重置是经过验证的"半独立可操控环境"做法，直接采用；但实时原版只能出真值与评估数据，训练样本必须来自加速原版或独立模拟器，见 [ENV_ARCHITECTURE.md](ENV_ARCHITECTURE.md)。

吞吐参照：PufferLib 报告 Pokémon Red（PyBoy 模拟器，无头多进程）在单台桌面机上超过 6000 步/秒（[PufferLib 1.0 博客](https://pufferai.github.io/build/html/rst/blog.html)）。以撒没有模拟器，达到同量级只能靠 L1 多实例加速与 L2 重实现。
