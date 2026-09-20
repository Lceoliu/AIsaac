# Isaac RL 环境架构：三层环境栈与统一契约

更新：2026-09-21。当前主线为原版 J460 worker 的稳定训练、rewind 重置、密集奖励和玩家可见观测；§0.3 是当前模型实现，后文明确标记的 L2 方案/模块快照保留为历史设计。目标与信息边界沿用 [PROJECT_SPEC.md](PROJECT_SPEC.md)，外部经验见 [RELATED_WORK.md](RELATED_WORK.md)。

## 0. 结论

- **当前主线：复用原版引擎训练**。全局 Agent 不以逐个翻译数百种怪物/道具为前置条件；J460 worker 已完成实际 PPO。此前“原版永远不是训练引擎/样本只能来自 L2”的断言撤销。
- **原版桥接同时服务训练、校准和验收**：Lua 同步单步、Gymnasium 与 SB3 已实机接通；推理只读取玩家可见字段。当前是单 worker，不把目标中的多实例说成已实现。
- **L1 控制层已实机使用跳渲染**：默认 headless，虚拟时钟关闭，另保留 visible 验收；高速时钟等价性和多实例成本仍未完成。L2 保留为课程子集/校准研究。详见 [L1_FEASIBILITY_PLAN.md](L1_FEASIBILITY_PLAN.md)。
- 三层共享同一份观测/动作契约；策略成绩最终以原版可视化验收为准。社区实验提示原版采样成本高，但不能据此认定独立模拟器是唯一可行来源；当前先验证原引擎训练 worker。
- 今天为 L2 做实的输入：J460 里 11 个实体类的虚表、NPC 逐类型 AI 分发表与单房间敌人集合的 AI 函数地址、Repentance+ `entities2.xml`（1337 条）、Basement 1181 个房间（其中 197 个 1×1 普通房只含首批敌人）、基础包 790 个带名 anm2（含玩家/眼泪/首批敌人的动画时长）。见 §5。
- 阶段 0 的桥接 mod 与 Python 客户端骨架已写出（`rl/bridge/`），只用于采校准数据；一行都还没在游戏里跑过。

### 0.1 本次确认的做法（2026-09-19）

1. 目标是可迁移到原版的完整全局 AI，第一道关卡是单房间实战，不要求整台引擎逐位相同。是否足够像，以原版留出场景上的策略迁移成绩判断，而不是以演示画面或自测清房率判断。
2. 继续现有 Rust `rl/sim`，不重启旧 `isaac_room`；从真实配置、房间、动画事件和 J460 伪代码提炼会影响决策的规则，按模块补齐并用 L0 对照校准。
3. 先闭合玩家输入 → 移动/射击 → 眼泪/敌弹 → 碰撞/伤害/无敌帧 → 死亡/清房，再扩大敌人和房间课程；无渲染训练先行，视觉渲染不占首版优先级。
4. 策略只接收玩家可观察信息。视觉或内存读取是采集方式，不是放宽信息边界的理由；NPC 内部状态、隐藏实体、种子和未来事件不能进入 actor。当前 Monstro v2 白名单已实机使用，并有隐藏字段不改变策略张量的测试；其他实体/道具组合仍需逐类验收。
5. 进度分别记录为：**地址定位、有效伪 C、机制理解、Rust 移植、L0 验证**。这五项不能互相替代。最新逐模块清单见 §11。

### 0.2 v2 MLP 基线（2026-09-20，保留作对照）

当前执行顺序：先稳定原引擎 worker，再扩展战斗数据，最后扩大训练。L2 手写模拟器不作为当前阻塞项；前文“模拟器是唯一可行样本来源”不是已证实结论，不能由一个社区实验推广到所有算法与工程方案。

**现有 RL 是最小 PPO 接入，不是已经成熟的战斗 Agent：**

- 原版 J460 + Lua bridge + 本机 TCP + Gymnasium + Stable-Baselines3 PPO；单 worker、CPU。用户已确认默认无渲染训练，现已接入已有控制层的 skip-render 功能，虚拟时钟保持关闭；另用 `--render-mode visible` 验收。
- 每个决策保持 4 个逻辑帧，动作是 `MultiDiscrete([9,5,2,2])`：移动、射击、炸弹、主动道具。协议支持的药丸/卡牌、丢弃尚未接入此 Gym 动作空间。
- 策略输入为 12 维玩家属性、128×14 个实体字段、128 个有效槽位标记，以及 **5×9×15 地形张量**（房内、可通行、碰撞类别/5、坑、潜在尖刺）。地形/实体表展平后送 MLP；不是注意力/CNN，也没有循环记忆、帧堆叠或实际速度估计。动画仍未送入策略。
- Actor 与 critic 都使用上述同一份输入，均为两层 64 单元 MLP；并未实现特权 critic。PPO 参数：每次采样 128 步、batch 64、4 epochs、learning rate 3e-4。
- **用户确认的 v2 奖励**：实际受伤每次 −1，敌人实际扣血结算每次 +0.05，另加实际伤害/该敌人初始最大HP，清房 +1；死亡不重复扣 −1。时间上限为 truncated。没有示范学习、课程随机化或多进程并行采样。
- 固定种子的 512 步只证明采样、梯度更新、存取档和再次执行动作已连通，不能代表收敛。已结束的学习回合均死亡。

**数据准备清单（现状与计划分开）：**

| 数据 | 当前读取 | 下一阶段准备 |
|---|---|---|
| 玩家属性 | 心数、伤害、攻速相关参数、射程、移速、幸运、资源、主动道具等；策略只取其中 12 维 | 加被动道具/饰品清单、武器类别、可见状态效果；区分射击间隔与剩余内部冷却 |
| 房内实体 | 可见实体 type/variant/subtype、位置、Size/SizeMulti、实体/地形碰撞类别、接触伤害；效果实体仍有白名单 | 诊断通道列全实体并保留可见性；策略端单独执行可见性与字段白名单，不能把“全部实体”直接当玩家观测 |
| 碰撞几何 | 基本半径和倍率，不是完整碰撞场 | 普通实体圆/椭圆、激光路径/宽度/有效阶段、刀/爆炸/地面伤害各自建模；记录几何来源，并用真实命中事件校验 |
| 行动状态 | 原始 obs 有动画名/帧；info.npcs 有 State/StateFrame、ProjectileCooldown/Delay、I1/I2/V1/V2、真实速度 | 可见动作由动画/朝向/历史运动组成；内部状态保留为诊断标签，不直接泄露给 actor |
| 血量 | Boss 归一化血量在 obs；普通 NPC 精确 HP/MaxHP 在 info | 继续完整记录诊断血量。普通怪精确剩余 HP 不默认进入玩家等价输入；多 Boss 血条对应关系需额外验收 |
| 战斗事件 | `combat` 每逻辑帧结算玩家受伤次数、敌人扣血次数/伤害量/归一化伤害；旧 events 仍保留为诊断，不直接算奖励 | 当前同一敌人同帧多击合为一次结算；补来源归因及逐次命中，避免未来把环境自伤当玩家命中 |
| 地形与地图 | 原始 grid/doors/room + 全格 terrain；地形已进入 v2 actor，随当前飞行能力更新 | 当前只支持1×1房型/格心通行；继续加入火/毒液/伸缩尖刺相位、连续几何，禁止未探索房间泄露 |

2026-09-20 已修复 `entity_record` 敌弹分支的 `projectile=true` 缺失：v2 三个实机回合的1642条敌弹记录全部正确标记。观测版本改为 `monstro-combat-v2`，新增地形并重新训练；历史模型不要直接加载。

几何依据：[entities2.xml 的 collisionRadius / X-Y multiplier](https://wofsauge.github.io/IsaacDocs/rep/xml/entities2.html)、[EntityLaser 路径采样 API](https://wofsauge.github.io/IsaacDocs/rep/EntityLaser.html)。这些是数据入口，不等于已验证每一种攻击的实际命中判定。

崩溃历史证据：`../runs/l1/20260920-crash/`；本轮 rewind/密集奖励/地形/PPO 证据：`../runs/l1/20260920-rewind/`。20项测试通过；新版512步实机 PPO 更新、存取档和重载执行通过，仍未学会击杀。详见 [桥接当前验收](../bridge/README.md)。原生冷启动/可视化崩溃根因仍待解决。


### 0.3 Transformer 战斗策略 v1（2026-09-21）

用户确认的架构：实体集合注意力 + 地形CNN + 64步因果时序Transformer。首版不使用LSTM、世界模型、跨房间规划或特权critic。实现入口为 `bridge/python/isaac_bridge/transformer_obs.py`、`transformer_policy.py`，训练仍复用SB3，动作屏蔽复用sb3-contrib 2.7.1的MaskablePPO。

**数据路径**：当前可见实体→共享128维编码→4个玩家条件注意力查询；7通道地形→CNN→128维（保留空间位置，不使用全局平均池化）；23维玩家状态、动画、主动道具→64维；上一动作→32维。拼接后生成256维时刻token，进入4层、8头、FFN1024的Transformer，使用相对经过时间正弦编码、因果遮罩与历史padding遮罩。Actor/critic共享同一份可见历史，末端分别为128维MLP与动作/value输出。

| 输入 | shape（不含batch） | 语义 |
|---|---|---|
| player | 64×23 | 房间位置、可见位置差分速度/有效位、尺寸、血量、资源、角色属性、飞行、主动充能/可用性、动画帧、flip、dt |
| entity numeric | 64×N×26 | 相对位置、位置差分速度/有效位、尺寸、碰撞属性、可见Boss血条、动画帧、激光端点/方向/长度或环半径 |
| entity kind | 64×N×3 | 类型/变种/子类型，分别embedding；不将编号作为有序实数 |
| animation | 玩家64×32，实体64×N×32 | UTF-8名字字节的有序embedding，不做不稳定的Python hash；超32字节报错 |
| terrain | 64×7×9×15 | 房内、玩家可通行、实体障碍、坑、可破坏障碍、地面危险、关闭的门 |
| previous_action | 64×4 | 上一步joint/bomb/item及有效位；reset后全零 |
| masks/time | 历史mask、实体mask、经过时间 | 新回合历史清空；不暴露全局绝对时间、种子、NPC隐藏HP/状态/RNG |

`N`是Gym批处理分配容量，默认256，可用`--entity-capacity`改大；不是最近K个实体，也不是全游戏数量上界。超限立即报错，包含曲线激光拆出的线段。网络会紧凑收集有效实体，注意力不依赖槽位顺序；空房使用独立null token，不会对全masked序列做softmax。种类embedding当前支持type<1024、variant<8192、subtype<4096；本轮针对原版单房间，不声称任意Mod编号兼容。

速度仅来自连续可见位置差分。Index只用于匹配，不进入网络；实体年龄连续性用于拒绝复用Index。失去可见性或rewind后断开追踪，不读取隐藏运动。普通敌人HP、引擎速度、受伤无敌计时及奖励结算计数不进入Actor，也不进入本版critic。动画帧是公开表现，不是NPC AI状态。

**输出**：`MultiDiscrete([45,2,2])`；`joint=move*5+shoot`，移动9种、射击5种同时选择，另有炸弹/主动道具二分类。策略线性输出49个logit，拆成三个分布。MaskablePPO在采样、训练log-prob和推理时应用同一可用动作mask。每次动作推进2个逻辑帧，历史覆盖约4秒；炸弹/道具沿用bridge的按键边沿语义。主动道具mask目前为已有道具且不NeedsCharge，特殊充能道具仍需独立课程验证。

**训练一致性**：每条PPO样本携带完整原始历史窗口，不缓存旧网络编码。minibatch可以打乱样本顺序，但每个样本内部的64步顺序不变；重算窗口的梯度会训练CNN、实体编码及时间注意力。dropout=0，避免采样与PPO log-prob重算不一致。训练和推理使用相同有限窗口，没有跨worker KV缓存。奖励仍为受伤−1、有效命中+0.05、归一化实际伤害、清房+1，死亡不额外扣分。

**桥接schema=3**：原有terrain.cells前8列保留，追加solid/destructible；可破坏障碍识别岩石、粪便、TNT。危险通道是潜在尖刺及白名单中可见、有碰撞伤害特效的保守网格覆盖，不是完整全游戏伤害预测；关闭门映射到最近格。激光提供原生GetSamples路径/端点，曲线按线段入表、环形按圆入表；当前实机核对直线激光，环/曲线还需对应场景扩展。旧MLP忽略新增列，仍可用 `--model mlp` 对照。

**已发现并修复的适配错误**：uint8的`64×256×32`动画表被SB3当成RGB图片，自动转置为`32×64×256`；小容量单测没有暴露这个问题。新增完整尺寸失败回归后，类别表改为int32，避免图像预处理。失败训练进程PID28028的游戏正常退出0，这不是原生崩溃。另用相同轨迹、相差1000万逻辑帧的worker时钟验证输入一致性：原实现失败，修正为先用整数减去本回合起点，再转浮点时间，避免绝对运行时间泄露及精度损失。过程日志保留在 `runs/l1/20260921-transformer/`。

**本轮验收（证据在 `runs/l1/20260921-transformer/`）**：

- 38项Python测试通过，包括因果未来隔离、过去影响当前、实体顺序/padding不变性、空实体、梯度进入CNN/实体/时序编码、可见速度/遮挡断轨、隐藏字段隔离、动作mask、reset、checkpoint存取、完整64×256尺寸的SB3转置回归及worker运行时间平移不变性。
- PID8240实机跑100次2帧动作，窗口达到64并在回合截断后清空；直线激光采样点(176,200)→(600,200)，原生宽度32。原生LaserLength在此现场为0，因此模型长度由实际采样几何计算，不盲信该字段。生成岩石的格67实测walkable=0、solid=1、destructible=1；退出0。
- 最终候选Transformer参数 **3,974,866**；PID24580完成512步PPO / 16 epochs / 124.89秒，参数L2变化7.90082，保存/重载动作一致，再执行通过；3个已结束回合均死亡，余下未结束回合不计胜负。WM_CLOSE退出0。checkpoint为`runs/l1/20260921-transformer/transformer-final/ppo_monstro.zip`；此前PID37072的通过结果保留为历史记录，不覆盖。
- MLP兼容对照PID14116完成128步 / 4 epochs，存取档与再次执行通过，退出0。
- 加载最终Transformer checkpoint，PID21900完成3个可视化回合：1死亡、2时间截断，2193次真实渲染、28509次字体调用，无字体跳过、shader入栈失败或虚拟时钟，退出0。评估执行的是学习策略，不是track规则控制器。
- 密集弹幕探针PID30812：请求生成200发，采样时原生列表199、原生可见199，桥接/Actor均199发，逐位置核对无遗漏/误差，加载的Transformer能实际输出动作；退出0。初版探针错误地把200次生成请求当作采样时的存活数量，在199上断言失败；修正为同一冻结帧独立枚举引擎实体作为基准，保留两份报告。证明超过旧128槽的覆盖，不是高密度弹幕战斗达标。
- 无有效Monstro击杀。以上是架构、梯度、协议及运行验收；训练预算太小且评估上限为360逻辑帧，不能据此与旧模型比较胜率。下一步是冻结实现后进行等预算、多初始状态的训练/评估。

复用接口参考：[SB3自定义编码器](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)、[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)、[激光采样API](https://wofsauge.github.io/IsaacDocs/rep/EntityLaser.html)。

### 0.4 独立引擎并行采样（2026-09-21）

用户验收标准：先2引擎共同训练一个模型，再实测4引擎的吞吐与资源；正式训练至少累计512个完整回合，单局上限120秒（3600逻辑帧）。step不等于局，死亡、清房、超时分开统计。

实现：`train_parallel.py` + `isaac_bridge/parallel.py`。游戏进程独立、端口递增；同一Python进程的线程池并发发送/接收各自TCP动作，采样在SB3 VecEnv边界汇合，由单个MaskablePPO统一计算动作与梯度。模型/可见观测/奖励不变，总rollout固定256条transition（2实例各128步，4实例各64步），batch8、epochs4。CUDA运行库仅安装在本仓库忽略目录，使用本机4060，不改系统驱动。

隔离：J460路径初始化函数`FUN_009a9510`首先动态调用`GetUserProfileDirectoryA`，不是直接相信USERPROFILE。原生可选`ISAAC_RL_WORKER_PROFILE`钩子重定向该worker到独立profile；配置SteamCloud=0，存档/日志独立；runtime拥有独立mods/data，资源目录junction及EXE/DLL硬链接共用未修改原始字节。启动验收读取每个runtime的savedatapath.txt与Lua加载日志，确认仅桥接Mod执行；推进worker0两帧，其他worker的逻辑帧、位置、历史窗口必须不变。

已观察的失败与修正：
- 初版给注入器设置cwd没有改变其子游戏目录：注入器按EXE所在目录启动。改为每worker独立runtime EXE入口，保留失败报告，两个游戏均正常退出。
- SubprocVecEnv版2引擎通过，但4引擎分配512MiB动画历史数组时内存不足。改用线程并发TCP，避免额外四份Python/PyTorch，游戏仍是四个进程；没有缩短64步历史或减少实体容量。
- 新runtime触发Steam重新同步订阅Mod，初版误把其他Mod重新启用；四引擎并发冷启动还出现握手超时。预置各订阅Mod的disable.it、核验仅桥接Lua执行，并逐个完成启动握手后再启动下一实例；正式采样仍并行。此前短测不能作为最终隔离配置的性能基准。

43项离线测试通过，含线程并发屏障、独立写目录、累计局数而非步数、自动reset/terminal_observation/时间截断、动作mask和CUDA梯度更新。报告位于`runs/l1/20260921-parallel/`；正式512局完成情况以最终report与逐局episodes.jsonl为准，短测通过不替代该门槛。

## 1. 证据基线（2026-09-19 核实）

| 主题 | 事实 | 证据 |
|---|---|---|
| 逻辑时钟 | 外层循环（RVA `0x00531050`）每帧调 `Manager::Update`（NetFix 名 `J460_MainLoopUpdate`，RVA `0x00554CD0`），其内按 `Manager+0x4ABBC` 奇偶每 2 次调一次 `Game_Update`（RVA `0x002FADC0`）；渲染 `FUN_009555c0`（RVA `0x005555C0`）独立于逻辑帧；帧率限制是 `Sleep` + 自旋，时间源 `FUN_00a266a0`（RVA `0x006266A0`） | [J460_FRAME_LOOP.md](../../analysis/docs/J460_FRAME_LOOP.md) |
| Lua 观测入口 | `Isaac.GetRoomEntities()`；`Entity.Position/Velocity/Size/SizeMulti/Visible/EntityCollisionClass/GridCollisionClass/CollisionDamage/Index/FrameCount`；`EntityTear.Height/FallingSpeed/Scale`；`EntityProjectile.Height/FallingSpeed/Scale`；`EntityNPC.State/StateFrame/ProjectileCooldown/I1/I2/V1/V2`（内部真值）；`Room:GetGridEntity/GetGridCollision/GetGridPosition/GetDoor/IsClear/GetAliveEnemiesCount/GetRoomShape/GetTopLeftPos`；`EntityPlayer:GetHearts()` 等心数、`Damage/MaxFireDelay/ShotSpeed/TearRange/MoveSpeed/Luck`、`GetActiveCharge/GetDamageCooldown/GetHeadDirection` | 随包官方文档 `tools/LuaDocs/class_*.html` |
| Lua 控制入口 | `MC_INPUT_ACTION(entity, hook, action)`，hook 为 `IS_ACTION_PRESSED/IS_ACTION_TRIGGERED/GET_ACTION_VALUE`，返回非 nil 即覆盖；`Isaac.ExecuteCommand`；`MC_POST_UPDATE`（30 Hz）、`MC_POST_NEW_ROOM`、`MC_POST_GAME_STARTED` | `resources/scripts/enums.lua` 第 26–34 行；`namespace_isaac.html` |
| 控制台 | `spawn`、`goto s.<type>.<variant>` / `goto d.<variant>`、`stage`、`gridspawn`、`debug 3/4/7/8/9/10`、`giveitem`、`remove`、`restart`、`seed`、`reseed`、`curse`、`lua/luarun/luamod`、`repeat`、`combo` | `tools/LuaDocs/debug_console_primer.html` |
| luasocket | 游戏自带 `resources/scripts/socket/core.dll` 与 `socket.lua`；`main.lua` 只在非 `--luadebug` 时禁用 `debug/dofile/load/loadfile`；社区项目 Isaac-RL 就用"游戏自带 LuaSocket"做 TCP 桥接 | 安装目录；`main.lua` 第 1185 行；RELATED_WORK §8 |
| REPENTOGON | 只运行在 J273，Launcher 靠降级安装；本机 J460 不能直接用其 DLL。它的 1199 条 ZHL 特征码在 J460 唯一命中 686 条（Entity 39、Entity_NPC 26、Entity_Player 118、Room 31、Game 34、Level 20 …），可作读伪 C 的命名表；未命中的包括 `Entity::Update`、`Game::Spawn`、`InputManager::*` | [repentogon.com/install](https://repentogon.com/install.html)；[J460_ZHL_MATCHES.md](../../analysis/j460/zhl-scan/J460_ZHL_MATCHES.md) |
| 资源 | `afterbirthp.a` 用 Gibbed 带文件名完整解包（10228 文件，含 AB+ 各层 STB）；`repentance.a` 解出 4180 文件但无文件名（哈希算法未知，官方提取器同样解析失败并崩溃），已按内容识别：1 份 `entities2.xml`（1337 条实体）、27 份配置、53 个 STB1 房间文件（可转 XML）、833 个 anm2（含 spritesheet 路径、动画名、fps、事件） | `analysis/resources/repentance-a/index.json`；`analysis/scripts/index_repentance_archive.py` |
| 版本/存档 | `isaac-ng.exe` 为 v1.9.7.17.J460；存档路径固定拼接 `Documents/My Games/Binding of Isaac Repentance+/`；只导入 `SteamInternal_SteamAPI_Init`，无 `SteamAPI_RestartAppIfNecessary` | `strings.jsonl`、`imports.csv` |
| 社区结果 | Isaac-RL：Lua mod + LuaSocket 同步单步、8 帧/动作，单实例 3.6 动作/s、6 实例 20.3 动作/s；12,381,564 次决策、2284 局 GRU，一层 Boss 零胜 | RELATED_WORK §8 |

## 2. 历史三层架构（层次划分保留，当前优先级以 §0.2 为准）

```text
                统一契约：PolicyObservation（玩家可见） / Action（合法按键） / PrivilegedInfo（评估通道）
   ┌───────────────────────────────┬───────────────────────────────┬────────────────────────────────┐
   │ L0 原版桥接（量具/验收）        │ L1 原版加速（可选，暂缓）        │ L2 独立模拟器（主线，训练引擎）   │
   │ isaac-ng.exe J460 + Lua mod    │ L0 + turbo DLL：虚拟时钟/跳渲染 │ 真实资源 + 反编译逻辑重实现       │
   │ 30 Hz 同步单步，实时           │ 多实例、隔离存档                │ Rust 无头，批量接口待接           │
   └──────────────┬────────────────┴──────────────┬────────────────┴───────────────┬────────────────┘
                  │ 轨迹/事件记录（校准集、留出集）    │ 同接口                          │ sim-to-game 差异分析
                  ▼                                  ▼                                ▼
          校准数据 + 最终验收                 仅当需要原版微调                 全部训练样本
```

分层原则：观测、动作、重置的语义只在 Lua mod 里实现一次；L1 的 DLL 只改变时间和渲染，不重复实现实体读取；L2 输出与 L0 相同的结构化观测，训练器不区分来源。

## 3. L0：原版桥接环境（量具：校准与验收，不用于训练）

### 3.1 组成

- mod：[`rl/bridge/mod/isaac_rl_bridge/`](../bridge/mod/isaac_rl_bridge/main.lua)（`main.lua`、`metadata.xml`）。安装方式：复制或目录联接到 `<game>/mods/isaac_rl_bridge`，在游戏 Mods 菜单启用。这一步由用户操作，本项目不改安装目录。
- 启动：`isaac-ng.exe --luadebug`，Steam 已登录；环境变量 `ISAAC_RL_PORT`（每实例一个端口）、`ISAAC_RL_PRIVILEGED=1`（评估模式附带真值）、`ISAAC_RL_ENGINE_VEL=1`（观测附带引擎速度，默认关）。用户目录 `options.ini` 设 `PauseOnFocusLost=0`；`VSync=0` 为 L1 准备。
- 客户端：[`rl/bridge/python/isaac_bridge/env.py`](../bridge/python/isaac_bridge/env.py)（`IsaacBridgeEnv.reset/step`、`TrajectoryRecorder`）、`launch.py`、`smoke_test.py`。

### 3.2 同步语义

- 决策边界 = 第 k 次 `MC_POST_UPDATE`（k 为动作重复帧数，默认 4 帧约 133 ms；Isaac-RL 用 8）。到边界时 mod 发送观测并在 `receive` 上阻塞，主线程停在 `Game::Update` 之后、渲染之前；一步恰好推进 k 个逻辑帧，与墙钟无关。
- 动作在收到 `step` 的那一帧写入 `held/triggered` 表，下一逻辑帧的玩家更新经 `MC_INPUT_ACTION` 读取；`triggered` 只在首帧为真，供炸弹/主动道具的边沿判断。
- 重置 = 控制台命令序列 + 等待 `MC_POST_NEW_ROOM` + settle 帧。`restart` 在下一帧才生效，所以 Python 侧按"阶段"分次发送：`[["restart"], ["seed XXXX XXXX"], ["goto d.12"]]`。
- 终止由 Python 判定：全部玩家 `dead`，或 `room.clear`；超时由训练器截断。

### 3.3 观测白名单 v1（已写入 mod）

| 组 | 字段 | 来源 | 为什么算玩家可见 |
|---|---|---|---|
| `players[]` | `pos, size`；`hearts, max_hearts, soul, black, bone, eternal, golden, lives`；`coins, bombs, keys`；`damage, fire_delay_max, shot_speed, range, speed, luck, can_fly`；`active, active_charge`；`invulnerable`（`GetDamageCooldown()>0`）；`controls`；`head_dir, fire_dir, move_dir`；`anim, aframe, flip`；`dead, ptype` | `EntityPlayer` 字段与 getter | HUD、Found HUD（数值面板）、受伤闪烁与精灵朝向都在屏幕上 |
| `entities[]` | `id`（`Entity.Index`，房内创建序号）、`type, variant, subtype`；`pos, size, size_multi`（碰撞圆）；`coll, gcoll, cdmg`；`anim, aframe, flip, age`；眼泪/弹幕 `height, fall, scale`；NPC `enemy, vulnerable, boss, champion`，Boss 另有 `boss_hp` | `Isaac.GetRoomEntities()` + 子类字段 | 实体外观、影子高度、冠军颜色、Boss 血条可见；接触伤害与碰撞半径属于实体类型的公开配置（`entities2.xml`），不是实例秘密 |
| `grid[]` | `[idx, type, variant, state, collision, x, y]`（跳过空格与装饰） | `Room:GetGridEntity` | 岩石/坑/尖刺/门等地形可见 |
| `doors[]` | `slot, open, locked, pos, target_type` | `Room:GetDoor` | 门的开闭与目标房型图标可见 |
| `room` | `type, shape, gw, gh, top_left, bottom_right, clear, alive, frame, stage, stage_type, curses, room_idx, variant, name` | `Room`、`Level`、`RoomDescriptor.Data` | 房型/层数/诅咒有提示；`alive` 对应门开关状态；`variant/name` 是日志用的标签，不应进入策略输入 |

不进观测的（只在特权 `info`）：NPC `HitPoints/MaxHitPoints`、`State/StateFrame/ProjectileCooldown/ProjectileDelay/I1/I2/V1/V2`、玩家 `FireDelay`、引擎 `Velocity`（默认关）、任何 RNG/种子。`Visible=false` 的实体不进观测；钻地、隐身、传送等各类隐藏规则还要逐类核对，眼下这是最低限度的过滤。效果实体（type 1000）只放白名单：爆炸、地面毒液、尖刺、瞄准标记、火。

### 3.4 动作

`MultiDiscrete[9 移动, 5 射击, 2 炸弹, 2 主动道具]`（药丸/丢弃预留），映射到 `ButtonAction`，对角线由游戏归一化。合法性：`MC_INPUT_ACTION` 是游戏读取动作时的钩子，与手柄输入走同一路径；没有改速度、位置或直接生成眼泪。受控期间键盘对玩家无效，`control` 指令可切回手动。

### 3.5 阶段 0 探针清单（必须逐项回答后才冻结契约）

1. `--luadebug` 下 `require("socket")` 与 `socket.bind` 是否可用（预期可）。
2. `goto d.N` 是否把房间放进 `ROOM_DEBUG_IDX = -3` 并每次重新生成、再次 `goto` 是否重生敌人；`restart` 后紧接 `goto` 的时序是否需要分阶段。
3. `Sprite:GetAnimation()` 在 J460 是否存在（随包 LuaDocs 未列出；mod 用 `pcall` 兜底，缺失时 `anim=nil`）。
4. 玩家死亡后 `MC_POST_UPDATE` 是否继续触发，`restart` 能否从 Game Over 发起；备选是在 `MC_ENTITY_TAKE_DMG` 拦到致命伤时立即 `restart`。
5. 同机双实例：Steam 是否允许；`log.txt`、`options.ini`、存档并发写是否冲突；`PauseOnFocusLost=0` 是否足够让后台实例继续跑。
6. 每步 JSON 体积与 30 Hz 下 Lua 编码开销（估计 5–20 KB/步）；必要时改为按需发送 `grid`。
7. `MC_INPUT_ACTION` 返回 `false` 是否会干扰非玩家实体或菜单（mod 已限定 `entity:ToPlayer()` 且只覆盖 12 个动作）。
8. 实体 `Index` 在房间内是否稳定可作跟踪 ID，删除/重生后是否复用。

### 3.6 预期吞吐

单实例受 30 Hz 限制：k=4 时 7.5 决策/秒，k=8 时 3.75 决策/秒；N 实例线性叠加（Isaac-RL 6 实例 20.3 动作/秒）。这足够采校准/评估数据，不足以训练。

## 4. L1：原版加速（可行性实验中）

2026-09-19 晚更新：L1 不再暂缓，作为全局 Agent 的通用环境与 L2 并行推进；实现、证据与实验协议见 [L1_FEASIBILITY_PLAN.md](L1_FEASIBILITY_PLAN.md)。下表的"虚拟时钟"与"跳渲染"两行已实现为 `rl/turbo/isaac_turbo.dll`（时钟只在 `Manager::Update` 返回时推进，读时钟不推进；渲染只在游戏本会绘制的调用上放行），"逐帧同步"仍由 Lua 完成，"存档隔离"与"隐藏窗口"未做。

目标：同一 mod、同一协议，但每实例逻辑帧率不受墙钟限制，并可跳渲染。上限由 `Game_Update` 成本决定，需要实测；粗估单实例 300–1000 逻辑帧/秒，对应 k=4 时 75–250 决策/秒，8 实例约 1000 决策/秒。

| 目的 | 位置（静态证据） | 做法 |
|---|---|---|
| 去掉限帧 | `FUN_00a266a0`（RVA `0x006266A0`，QPC 秒） | Hook 后返回虚拟时钟：每次 `Manager::Update` 后推进 1/60 s，`Sleep` 与自旋立即满足；配合 `VSync=0` |
| 跳渲染 | `FUN_009555c0`（RVA `0x005555C0`） | 每 N 帧或按需（截图/调试）才调用原函数 |
| 逐帧同步 | 不需要原生接口 | 由 Lua `MC_POST_UPDATE` 阻塞完成 |
| 存档隔离 | `CreateFileA` 路径含 `My Games/Binding of Isaac Repentance+` | 每实例重定向到独立目录，或用独立 Windows 用户 |
| 隐藏窗口 | GLFW 3.4 窗口 | 最小化/离屏；OpenGL 上下文仍需窗口句柄 |

工程复用 NetFix：注入器（`netfix/src/tools/Injector.cpp`）、MinHook 接线与身份/签名门（`Hooks.cpp`）、`profiles/j460/game_api.json` 地址档案、遥测；新建一个独立小 DLL，不装 NetFix 的网络钩子。风险：跳过渲染是否影响 `Game_Update` 读取的相机/插值状态（`Room+0x11F8`）；虚拟时钟对音频、Steam 回调的副作用。验收：同种子同输入下 L1 与 L0 逐帧位置一致。

## 5. L2：独立模拟器（历史方案，现为保留研究）

只做"会改变动作选择"的机制，范围随课程扩大；每个常数与规则都要能指到资源文件或 J460 函数，不能凭印象写。

### 5.1 数据来源（已到位）

| 数据 | 位置 | 内容 |
|---|---|---|
| 实体配置（Rep+ 最终值） | `analysis/resources/repentance-a/config/entities2.xml`；首批敌人抽取表 `derived/basement_enemies_entities2.json` | 44 条记录：Gaper 10.1 hp 10 / 半径 13 / 质量 5 / 接触伤害 1；Fly 13 hp 3 / 半径 17 / 接触伤害 0 / 只碰墙；Pooter 14 hp 8；Clotty 15 hp 15 质量 10；Attack Fly 18 hp 5；Charger 23 hp 20；Hopper 29 hp 10；Horf 12 hp 10 质量 3 |
| 房间 | `derived/stb1_stage_identification.json`（53 个 STB1 按与 AB+ 有名房间集的重叠识别：Basement=`4F40A7CB1DA67D16` 共 1181 房，Cellar、Caves、Catacombs、Depths、Necropolis、Womb、Utero、Cathedral、Dark Room、Chest、特殊房均已对上）；`derived/basement_1x1_rooms_enemy_subset.json` | 197 个 Basement 1×1 普通房只含首批敌人：Gaper 35、Horf 20、Attack Fly 19、Hopper 17、Clotty 12、Mulligan 10、Fly 8、Pooter 8、Gusher 8 及混合房；难度分布 1/5/10 |
| 动画 | `analysis/resources/animations-a/anm2/`（基础包 790 个有名 anm2 中的玩家、眼泪、弹幕、首批敌人）；`derived/anm2_timing_enemy_set_base.json` | 30 fps；例如 Pooter `Attack` 14 帧、Horf `Attack` 20 帧、Clotty `Attack` 23 帧、Hopper `Hop` 26 帧、Charger 移动 12/24 帧、玩家 `Hit` 8 帧、眼泪 15 帧 |
| Rep+ 覆盖关系 | `derived/anm2_names_from_base.json` | Rep+ 的 833 个无名 anm2 只有 17 个与基础包同图集，首批敌人的动画未被 Rep+ 覆盖 |

### 5.2 逻辑来源（已定位，部分已翻译）

| 模块 | J460 入口 | 大小 | 说明 |
|---|---|---|---|
| 敌人 AI 分发 | `0x002C2990` | 6.2 KB | `type-10` 查索引表 → 跳转表 → 各类型 AI 函数，见 [J460_ENTITY_CLASSES.md](../../analysis/docs/J460_ENTITY_CLASSES.md) |
| Gaper AI | `0x00108F40` | 6.0 KB | 追踪玩家，含 Frowning/Flaming/Rotten 变体分支 |
| Fly / Pooter / Attack Fly AI | `0x00106BD0` | 9.1 KB | 8 个飞行类型共用，内部按类型/变体分支 |
| Horf / Clotty / Charger / Hopper AI | `0x00113080` / `0x0010D370` / `0x0012DB50` / `0x00126470` | 1.4 / 4.4 / 12.1 / 10.8 KB | |
| NPC 公共更新与碰撞 | `Entity_NPC::Update 0x002C4B30`、`HandleCollision 0x002CD410` | 24.0 / 12.0 KB | 状态帧、目标选择、击退、接触伤害 |
| 基类运动 | `Entity::Update 0x002AE820` | 9.8 KB | 速度积分、摩擦、网格碰撞 |
| 玩家 | `Entity_Player::Update 0x00382AF0`、`HandleCollision 0x0038E1F0`、`TakeDamage 0x003729D0` | 34.7 / 5.7 / 26.5 KB | 输入到速度、射击节奏、受伤与无敌 |
| 眼泪 | `Entity_Tear::Update 0x002670F0`、`Weapon::Fire 0x005F3540`、`GetTearMovementInheritance 0x003AAEA0`、`GetMultiShotPositionVelocity 0x00382160` | 50.2 KB / 225 行 / 已读 / — | 眼泪高度、下落、射程、继承速度（系数 `DAT_00baa304`≈0.6） |
| 敌人弹幕 | `Entity_NPC::FireProjectiles 0x002CB2E0`、`Entity_Projectile::Update 0x00253070` | 6.8 KB / Ghidra 需建立正确函数边界并补导 | FireProjectiles 已有伪 C；Projectile::Update 尚无独立有效导出 |
| 网格 | `Room::GetGridCollision 0x003F0800`、`Room::GetGridEntity 0x00036060`、`Room::CheckLine 0x003F0DF0` | | 40 单位一格；1×1 房含墙 15×9（可行走 13×7），可行走区 (60,140)–(580,420) |
| 寻路 | `NPCAI_Pathfinder::FindGridPath 0x003E33C0`、`MoveRandomlyBoss 0x003E6300` | | 有视线直线追，否则目标格反向 Dijkstra；NPC 自身格打 900 标记，每 3 帧衰减 100 |

补导状态：`Entity_Tear::Update` 已在 `decompiled-retry/006670f0.c` 成功补导；`Entity_Tear::HandleCollision`（RVA `0x00273660`）已有 `00673660.c`，但临时建函数的 `body_bytes=1` 与类型传播告警需核对，不能当作完整边界验证。详见 §11.1。

已翻译（2026-09-19，见 [J460_NPC_MOVEMENT_MODEL.md](../../analysis/docs/J460_NPC_MOVEMENT_MODEL.md)、[J460_PLAYER_TEAR_MODEL.md](../../analysis/docs/J460_PLAYER_TEAR_MODEL.md) 与 `rl/sim`）：`Entity::PreUpdate/Update`（含待结算伤害扣血）、实体-网格碰撞（NPC 版与玩家版）、`SetSize`、房间网格/路径图/视线、`FindGridPath`/`MoveRandomlyBoss`、`Entity_NPC::Update` 出现流程与 AI 门控、Gaper、Gusher/Pacer（弹幕以事件占位）、玩家移动/射击输入/`Weapon_Tears` 开火节奏/眼泪参数与发射/眼泪飞行、下落、落地、撞墙、命中推挤/NPC 接触伤害/玩家无敌帧。未翻译：敌弹、`EvadeTarget`、状态效果施加、道具效果、精灵动画、原版 RNG。眼泪下落模型有一处二选一（内部标志位 114），默认取射程生效的模型，靠 L0 判定。

翻译顺序（剩余）：飞行类 → 其余首批敌人 → NPC 弹幕。每翻译一个模块就用 L0 录制的同输入轨迹比对一次，不攒到最后。

### 5.3 实现形态与吞吐目标

- 当前内核已选 Rust（`rl/sim`，crate `isaac_sim`），复用已有工具链；后续暴露批量 `step(actions[N]) -> obs[N]` 的 Python 绑定。当前 `World` 使用 `Vec<Entity>`；固定槽位与容量上限仍是待评估设计，不是已实现事实。
- 目标：单核 ≥ 5 万逻辑帧/秒；16 核 ≥ 50 万逻辑帧/秒（4 帧一决策即 ≥ 12 万决策/秒）。按每局 300 决策算，100 万局 = 3 亿决策，约 40 分钟到 3 小时；瓶颈会落在神经网络而不是环境。
- 实测（2026-09-19，`rl/sim/examples/bench.rs`，release，单核）：1×1 空房、1 玩家占位 + 4 Gaper、无眼泪，约 95 万逻辑帧/秒。这只覆盖目前已翻译的逻辑，加入眼泪与更多实体后会下降，但离目标有约 19 倍余量。
- 输出与 L0 相同的结构化观测（§3.3 字段），训练器不区分来源。

### 5.4 校准与验收

用 L0 录制同房间、同种子、同输入的轨迹（`TrajectoryRecorder`，含特权真值），比较：玩家位移曲线（加速/摩擦/墙滑）、眼泪出膛速度与射程、命中帧、敌人首次攻击帧与弹幕速度、伤害事件、清房判定。差异超阈值先修模型再扩内容；策略层面看 L2 训练出的策略在 L0 留出房间上的清房率、受伤与耗时。

首版不做：完整道具系统、跨房、Boss、精灵渲染（视觉策略需要时再加）。

## 6. 契约与接口审计

字段以 mod 代码为准（§3.3、§3.4）。特权 `info`：`npcs[id,type,variant,hp,max_hp,state,state_frame,pcool,pdelay,i1,i2,v1,v2,vel,visible]`、`players[id,fire_delay,vel,damage_cooldown,total_damage_taken]`、`start_seed`、`room_spawn_seed`、`room_clear_count`。

审计方法：构造"可见历史相同、隐藏状态不同"的成对局面（同一房间、不同 spawn seed 下敌人冷却不同），actor 的输入张量与动作掩码必须逐位一致；`info` 不得进入 actor 的任何输入、奖励回流特征或循环状态。

## 7. 采样、标定与验收

- 轨迹格式：JSONL，每决策一行 `{t, obs, action, info}`；元数据：游戏版本 J460、mod 版本、seed、房间 `(stage, variant)`、k。
- 校准集与留出集按房间变体和敌人组合划分并固定种子；用于校准的房间不再当测试集。
- 单房间指标（L0 实测）：清房率、受伤心数、清房逻辑帧数、动作预算；对照基线：静止、随机、脚本追踪射击。
- 分开报告：结构化观测版与（未来）视觉版；零样本迁移（L2 → L0）与固定预算微调（L1）。

## 8. 历史 L2 并行训练方案（当前尚未实现多 worker）

- **训练采样只由 L2 承担**：先闭合单房间战斗与校准，再接 Rust 批量环境、Python 绑定和 PPO / PPO+GRU。批量接口、并行采样器和训练闭环尚未完成。
- L0 只采校准轨迹、检查观测/动作契约并做留出集验收，不安排 L0 多实例训练。
- L1 进入可行性实验：先证明可控、再证明加速正确、最后量成本与 1→2→4 实例扩展（[L1_FEASIBILITY_PLAN.md](L1_FEASIBILITY_PLAN.md) §4）；通过后才讨论用它做全局 Agent 的采样。
- 环境数量、线程数和学习器配置按完整战斗场景的实测再定；同时记录逻辑帧/秒、有效决策/秒、学习器等待时间及 L0 迁移成绩。旧的空房 Gaper 吞吐不能替代完整战斗吞吐。

## 9. 历史路线图（L2，非当前执行顺序）

| 阶段 | 产物 | 验收门 |
|---|---|---|
| 0 数据与逻辑落地（本周，已开始） | 首批敌人/房间/动画表（已出）；AI 与运动函数的伪 C 翻译笔记（`Entity::Update`、网格碰撞、寻路、Gaper/Gusher 已出）；L0 桥接装入并录 10 局校准轨迹（未做） | 每个模拟常数有资源或函数出处；校准轨迹可回放 |
| 1 L2 v1 | Rust 内核 `rl/sim`：房间网格、Gaper/Gusher、玩家运动与射击、眼泪、接触伤害（已有）；Fly（未做）；批量 Python 绑定；同 L0 的观测格式 | 同输入轨迹与 L0 的位置误差、命中帧误差在阈值内；单核 ≥ 5 万帧/秒（当前子集实测约 95 万） |
| 2 首批敌人集合与训练 | Pooter/Clotty/Horf/Attack Fly/Charger/Hopper；197 个 Basement 房课程；PPO(+GRU) 大规模训练 | L2 内清房率随训练上升；留出房间不退化 |
| 3 迁移验收 | L2 策略零样本跑 L0 留出集 | 清房率/受伤/耗时达到约定阈值；否则回到 1 修模型 |
| 4 扩展 | 更多敌人、房型、道具；必要时 L1 在原版微调 | 同上，按课程扩大 |

## 10. 未决问题

1. 阶段 1 敌人集合与达标阈值。建议从 Basement 常见类型起步：Gaper 10、Horf 12、Fly 13、Pooter 14、Clotty 15、Attack Fly 18、Charger 23、Hopper 29；阈值由用户定。
2. 视觉前端是否列入路线（当前只做结构化观测）。
3. `repentance.a` 文件名哈希：可从 `KAGE_Filesys` 归档加载代码逆向，或继续按内容匹配（anm2 用 spritesheet 路径与 `entities2.xml` 的 `anm2path` 对齐）。
4. 多实例与 Steam 的兼容性，只能实测。
5. 是否另装一份 J273 + REPENTOGON（提供 `GetCollisionCapsule`、Pathfinder 等更丰富 API）；当前判断不需要。

## 11. 反编译模块覆盖与缺口（2026-09-19 文件核对）

本节是当前模块进度快照，依据实际 `functions.jsonl`、单函数文件、补导日志和 `rl/sim/src`，不以旧 README 的待办代替最新产物。本次只更新文档，未重新运行 Ghidra、模拟器测试、性能测试或原版游戏。下列地址均为 **RVA**；补导文件名使用 **VA = RVA + 0x00400000**。

### 11.1 全局语料与真正尚缺的导出

- [原始 manifest](../../analysis/j460/exports/j460-baseline/manifest.json)：识别 20,994 个函数，20,990 个成功、4 个失败。实际 `decompiled/` 有 20,994 个 `.c`，其中失败文件只有错误注释，**不能按扩展名计为反编译成功**。
- [眼泪 Update 补导](../../analysis/j460/exports/j460-baseline/decompiled-retry/006670f0.c)：RVA `0x002670F0`，115,571 字节伪 C，头部 `body_bytes=50213`；[日志](../../analysis/j460/logs/redecompile-006670f0.log) 为 `completed=true elapsed=191s`。原始四个失败项中这一项已补齐，因此在原始识别集合内，目前有 20,991 项成功产物、3 项未补齐；原始 manifest 未回写，保留原批次统计。
- [眼泪 HandleCollision 补导](../../analysis/j460/exports/j460-baseline/decompiled-retry/00673660.c)：RVA `0x00273660`，46,016 字节；[日志](../../analysis/j460/logs/redecompile-00673660.log) 为 `completed=true created=true elapsed=3s`。原项目没有该函数起点，脚本临时创建后输出，read-only 工程未保存该分析变更。`body_bytes=1`、类型传播告警与覆盖范围仍需核对；状态是**已有候选伪 C，边界待核验**，不是“没有文件”，也不是“已完整理解”。此入口不在原始 20,994 个函数起点中，不能直接加进旧分母。
- `Entity_Projectile::Update` RVA `0x00253070`：虚表已定位，但 `functions.jsonl` 没有独立起点，`decompiled-retry/` 也无对应文件；邻近 `FUN_00652fa0`（RVA `0x00252FA0`）的 985 字节伪 C 不能替代整个 Update。需建立正确边界、补导并检查分支完整性。
- 原始失败项尚余：`0x001B39D0`（超时，语义未在本次核定）、`0x001D4380`（`Entity_Effect::Update`，超时）、`0x0022D300`（反编译器符号越界错误，语义未在本次核定）。错误见 [decompile_failures.jsonl](../../analysis/j460/exports/j460-baseline/decompile_failures.jsonl)，汇编仍在 `failed_disassembly/`。不把未知用途猜成已完成模块。
- 上述是**已识别函数集合**的导出覆盖，不是整台游戏引擎的完成率；漏切函数、内联逻辑、间接调用、错误类型和分支语义仍需专项审查。

### 11.2 单房间实战：逐模块状态

| 模块 | 有效伪 C / 定位证据 | 机制理解与当前 Rust 覆盖 | 还缺什么 |
|---|---|---|---|
| 帧循环与实体调度 | `Game_Update 0x002FADC0`、`EntityList::Update 0x000186C0`、`Room::Update 0x00402980` 均有成功导出；[帧循环](../../analysis/docs/J460_FRAME_LOOP.md)已整理 | 已理解 30 Hz 逻辑 / 60 Hz 插值，`world.rs` 已有帧骨架 | `Room::Update` 与 `EntityList::Update` 的精确先后尚未核定；Rust 暂放房间更新于帧末 |
| 基类运动、实体推挤、网格碰撞 | `0x002AE7D0 / 0x002AE820 / 0x002A9C90 / 0x002B2940` 均已导出；[运动笔记](../../analysis/docs/J460_NPC_MOVEMENT_MODEL.md) | `physics.rs` 已移植运动子集、SetSize、普通网格碰撞和圆形推挤 | 玩家专用碰撞 `0x002AB440` 已有伪 C 但未移植；脱困、部分状态/击退/特殊实体分支未全覆盖 |
| 房间网格、视线与寻路 | Room 查询、`FindGridPath 0x003E33C0`、`MoveRandomlyBoss 0x003E6300` 已有伪 C 与逐行笔记 | `room.rs / pathfinder.rs` 已有子集 | `EvadeTarget`、大体型与特殊状态分支、平局选格顺序；不能据此称房间全部机制已完成 |
| NPC 公共更新与 AI 分发 | `0x002C4B30 / 0x002C2990` 已导出；11 个实体类虚表、327 个 NPC 类型 case 已映射 | `npc.rs` 已有出现流程和 AI 门控 | APPEAR 标志清除点、完整 Morph/死亡掉落/状态效果；327 类型“定位”不等于 327 套 AI 已理解或移植 |
| Gaper、Gusher/Pacer | `0x00108F40 / 0x0010AF90` 已导出并有规则笔记 | `npc/gaper.rs / gusher.rs` 已移植部分行为 | Rotten Gaper、恐惧/燃烧逃离、完整 Morph；Gusher 只发射事件，尚无真实弹体与血地板伤害 |
| 玩家输入、移动与射击 | `Update 0x00382AF0`、`Weapon::Fire 0x005F3540`、`GetTearMovementInheritance 0x003AAEA0`、`GetMultiShotPositionVelocity 0x00382160` 均已导出 | 速度继承等少数 helper 有阅读记录；当前仍为 `spawn_player_stub / set_player_velocity` | 玩家更新的依赖链和字段语义、输入→移动、射击节奏、眼泪生成，尚未形成可玩的玩家控制 |
| 眼泪更新与命中 | Update 已成功补导；HandleCollision 有候选输出，见 §11.1 | Rust 尚无眼泪模块 | 核对碰撞函数边界；提炼基础眼泪位移、高度/下落、射程、命中与销毁链，再逐步覆盖 tear flags |
| NPC 弹幕 | `FireProjectiles 0x002CB2E0`、`Projectile::HandleCollision 0x0025B670` 已导出；Update 入口已定位但缺有效独立输出 | `Event::FireProjectiles` 占位 | 修复并补导 Update，厘清弹速/下落/碰撞/消失，再落实真实弹体 |
| 伤害、无敌帧与死亡/清房 | 玩家 `HandleCollision 0x0038E1F0 / TakeDamage 0x003729D0`、NPC `HandleCollision 0x002CD410 / TakeDamage 0x002D60A0` 均有伪 C | 接触主要输出 `Event::Contact`，NPC 仅有部分移除事件 | 接触/弹体/眼泪伤害、玩家无敌帧、心数、死亡与清房条件，需要完整闭环；不是缺少上述伪 C |
| 其他首批敌人 | Horf `0x00113080`；Fly/Pooter/Attack Fly `0x00106BD0`；Clotty `0x0010D370`；Mulligan `0x0010BBE0`；Charger `0x0012DB50`；Boom Fly `0x0010F280`；Hopper `0x00126470`；Boil `0x00117150`：上述 8 个入口均有成功伪 C | 已定位，尚无相应 Rust AI 模块 | 阅读状态机、攻击触发与共享 helper，再分批移植。首批资源集合共 12 个类型，扣除 Gaper/Gusher 后还含这 10 个类型，不应只写成 Fly 一类待办 |
| 动画事件与 RNG | 基础 ANM2、16 个相关动画与时长/事件表已取得；RNG 只有部分逆向线索 | 动画推进近似；`rng.rs` 是替代 xorshift32 | 动画事件与 AI 的同步、原版随机流/播种/调用顺序。若比较同种子逐帧轨迹，需先对齐随机流；不能把资源解包当引擎语义已还原 |
| 房间资源接入与观测契约 | 1337 条实体配置、44 条首批配置、197 房数据已取得；L0 Lua/Python 桥接已写 | Rust 有房间网格原语；资源表和桥接不是已经接好的训练环境 | 197 房加载、配置/动画接入、统一 reset/step/终止和可见观测、Python 批量绑定；L0 冒烟与校准轨迹仍未完成 |

本表中的 Rust 覆盖仅通过源码确认；**各模块的 L0 同输入轨迹校准均未完成**。文档中的 10 项单测和约 95 万帧/秒是历史有限场景结果，本次未复跑，不代表完整战斗或迁移达标。

### 11.3 其他已覆盖范围与后置范围

- **启动、菜单、联机与状态恢复**：[生命周期图谱](../../analysis/docs/J460_LIFECYCLE_ATLAS.md)与[逆向笔记](../../analysis/docs/J460_REVERSE_ENGINEERING.md)已整理 ApplicationMain/Mod/Cutscene、菜单/存档、OnlineLobby/建房/加入/Ready、原生输入与帧门、FrameState/GameState/RestoreState/desync、结束与退房的主要静态链路。仍有最小输入应用函数、room-transition helper 类型、KAGE message/lane 标注待细化；双机迟到输入收敛、跨房 replay、Steam/Quinn 切路和多人槽位变化属于实机验证，不是再批量反编译就能解决的缺口。
- **炸弹、拾取物、跟班、激光、刀**：各类 Update（`0x002A24D0 / 0x002E3940 / 0x0021DA10 / 0x00577CC0 / 0x00561B00`）已有成功伪 C，但未完成面向 L2 的机制整理和移植；首个基础眼泪单房间闭环不必等待它们全部完成。
- **Effect、完整道具/武器交互、Boss、跨房与整局决策**：属于后续课程；Effect::Update 当前还有真实导出失败。已定位 Boss 函数不代表 Boss 模块已完成。既不把这些列成首版全量前置条件，也不声称已经覆盖全局 AI。
- **资源缺口独立记录**：Rep+ 文件名哈希仍未知，53 份 STB1 中 3 份转换失败；基础与 DLC 同名动画覆盖关系仍需按内容/加载链确认。资源格式问题和玩法函数问题是两类工作。

### 11.4 接下来补齐的顺序（记录计划，未执行）

1. **反编译补洞**：核验 Tear::HandleCollision 的函数边界，建立 Projectile::Update 的正确函数边界并导出。其他 3 个原始失败项按当前战斗依赖决定先后，不以“全库零失败”为首版验收目标。
2. **最小实战闭环**：理解并移植基础玩家移动/射击、眼泪、双方碰撞伤害/无敌帧、死亡/清房；补 Gusher 的真实弹幕。玩家专用碰撞与公共更新顺序一起核对，不能只在 stub 上加发射按钮。
3. **校准同步推进**：先实机验收 L0 桥接，再录基础移动、射击、Gaper/Gusher 战斗轨迹，逐模块比较；冻结可见观测边界，迁移指标阈值仍待约定。
4. **扩课程与训练**：先 Fly/Pooter/Attack Fly，再其他首批敌人；接真实房间与配置、Python 批量环境和训练器，最后到 L0 留出房间验收。未支持的实体/网格不能静默忽略后称为“197 房已支持”。

本次状态变更：明确 Rust 主线，去掉 §8 旧 L0/L1 训练安排，加入最新补导结果及逐模块缺口；没有改变游戏、NetFix、Rust 或 Lua/Python 桥接代码。

## 附录 A：本次使用的扫描脚本

- `analysis/scripts/zhl_scan_j460.py`、`analysis/scripts/zhl_merge_j460.py`：REPENTOGON 特征码 → J460 地址 → Ghidra 函数表。
- `analysis/scripts/index_repentance_archive.py`：无名 `repentance.a` 解包目录 → 内容识别、配置/房间复制、`index.json`。
- 调用点扫描（找 Ghidra 漏掉的 `CALL`）：对 `.text` 中所有 `E8 rel32` 计算目标地址后按目标过滤，见 J460_FRAME_LOOP.md §1；同法可定位任何静态语料中"无调用者"的函数。
