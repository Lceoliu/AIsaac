# AIsaac：完整观测、操作接口与模型设计

更新：2026-11-07。9.9–9.11保留历史memory-chain v2/v3；当前新输入与四latent模型以9.13为准，未修改生产模型。本文维护接口设计、用户确认范围与验收状态；已删除旧 Rust 模拟器、旧训练阶段及 sim-to-game 主线叙述，不再作为当前状态引用。实时实验进度以对应日志/检查点为准。

- **9.1–9.4：用户已认可的设计方向**；本轮AB+实现范围与实际证据见9.13，Rep专属HUD和新增动作不由此宣称完成。
- **9.5：补充具体的结构化 HUD 表示**，不是图像识别方案。
- **9.6：待验证的最小动作扩展**，不预定改变四帧决策节拍。
- **9.7：结构化场景输入优先；当前不引入 RGB 视觉网络。**
- **9.8：版本、记忆与验收边界。**
- **9.9–9.12：记忆链路重构。9.9–9.11 描述候选代码 memory-chain v2/v3 的实际行为与 2026-10-24 的决定（有界 horizon、快慢双周期、延迟梯度）；未合入生产。**

保留 9.1–9.8 编号，9.9–9.12 扩展模型与训练设计。本轮只更新文档，不修改生产代码、训练配置、游戏或检查点。

审计依据：`D:/Projects/fortune/Isaac/analysis/j746/port-20260930/c99-replays-20261017/README.md` 及其中的 `segment2-20261018/logs/input-observability-audit.json`。

### 9.1 原则：补齐可观察性和可控制性，不编码玩法答案

- 策略可观察：当前画面/HUD、已合法见过的信息、自身输入历史。允许的公开物品知识须与当前实例的隐藏状态分开。
- 策略可控制：游戏本身接受的按钮/轴、保持与释放、推进的逻辑帧；不是瞬移、调用使用道具函数或代替游戏完成操作的宏。
- 学习负责：何时切换、等待、充能、丢弃、踩开关、使用主动、选择路线。编码器不提供推荐目标、用途评分、最佳配方或手写决策规则。
- `empty`、`unknown`、`unsupported` 和 `not currently visible` 不能共用数值 0；未知字段不得通过 ID、token 数量、排序、mask 或历史刷新泄漏。
- 结构化输入本身也有归纳偏置。按通用物体/槽/外观组织，比为每个角色和道具各建一套逻辑更少偏置；但它不等于完整图像。当前方案不引入 RGB 策略输入或视觉网络。RGB/HUD 截图仅用于少量隔离实例的离线核验，不要求训练 worker 逐帧渲染/截图。不宣称任意视觉信息都已覆盖；剩余缺口应逐项定位，不能默认用大视觉模型兜底。

### 9.2 角色身份、身体与控制器必须分离

**34 个可选角色不等于 34 个引擎 PlayerType。** 本地 Rep 资源含原始类型 0..40，另有 -1 特殊值；41 是枚举计数，不是角色。当前 `tok_obs.N_CHAR=32` 且 ptype clip 到 31，32..40 发生别名；C99 的 pchar 功能未启用，不能把这个缺口直接说成 C99 的失败原因。

| 菜单角色 | 表角色入口 ID | 里角色入口 ID |
|---|---:|---:|
| Isaac | 0 | 21 |
| Magdalene | 1 | 22 |
| Cain | 2 | 23 |
| Judas | 3 | 24 |
| ??? | 4 | 25 |
| Eve | 5 | 26 |
| Samson | 6 | 27 |
| Azazel | 7 | 28 |
| Lazarus | 8 | 29 |
| Eden | 9 | 30 |
| The Lost | 10 | 31 |
| Lilith | 13 | 32 |
| Keeper | 14 | 33 |
| Apollyon | 15 | 34 |
| The Forgotten | 16 | 35 |
| Bethany | 18 | 36 |
| Jacob & Esau | 19 | 37 |

剩余 7 个正 ID 是状态/身体：11 Lazarus2、12 BlackJudas、17 Soul、20 Esau、38 Lazarus2_B、39 Jacob2_B、40 Soul_B。身份词表覆盖这些类型并显式保留特殊/未知项，不再 clamp 为最后一个正常角色；以后扩展词表，而非重用已有身份。

`BodyToken`：局部稳定 `body_id`、`controller_id`、原始类型的无碰撞类别映射、已知菜单身份、main/twin/sub-body 关系、位置/速度、朝向、可见动画/外观、控制状态、HUD 归属。按当前可见性逐字段过滤；不可见替身的实时坐标不能透传。关系 ID 不是内存地址或随机种子。

同一控制器可能控制多个身体；Jacob/Esau 不能被赋予人类没有的两套完全独立移动命令。普通 Forgotten 的另一形态也不等于第二名本地玩家。稳定 ID 跟随可识别的身体，不因每帧排序变化而重建。现有 `o.players[0]` 必须改成身体集合消费；控制器与身体的关系由版本适配层采样核实。

### 9.3 统一物品槽，而非 player 上再堆几个单值字段

`SlotToken` 通用字段：`owner_body_id, family, slot_id, occupied, selected, display_order, item_kind, identity, identity_known, visible_variant, field_known_mask`。

| family | 范围 | 约束 |
|---|---|---|
| active | 主动槽 0 主手、1 Schoolbag、2 固定口袋主动、3 临时一次性口袋主动 | 四种槽号是 Rep API 契约；按版本支持，不能把空槽/不支持混为一谈 |
| consumable | 实际存在的卡牌/药丸槽及显示顺序 | 与 active 的 pocket 槽是不同命名空间；逐槽保存选中状态，不混成一个 pill/card 字段 |
| trinket | 实际装备槽 | 保留金色等可见修饰；吞服记录来自曾经观察/可查看菜单，不从隐藏全知清单刷新 |
| inventory | 当前菜单可见或从已见历史得到的持有物 | 身份、数量、顺序/选择按实际界面保存；不再只保留最近 16 件身份与 8 个独立 token |

槽身份稳定不代表物品稳定：切换、消耗、替换、丢弃、重排后的 ID/owner/selected 变化都由观测报告。未知药丸的颜色与已知效果分离；未识别效果始终未知。Blind 下所有身份旁路一并遮蔽，包括 subtype，不只是单独 ent_item。

Schoolbag 和第二药卡槽不是 Rep 才有；AB+ 已有相关接口。Rep 的 slot API 不得直接假设在 AB+ 或所有 Rep+ 构建都能工作。多人/双身体的槽集合归属于其 HUD/身体，不复制 player0 的值。

### 9.4 充能：分层表示，不能只给一个 ready

每个主动槽独立附带 `ChargeView`：

- `display_kind`：画面实际显示的分格条、连续进度、次数/资源计数、无显示；可附已知物品的公开 normal/timed/special 类别，不读未揭示身份推导它。
- `base_charge, capacity, battery_charge, partial_progress`：只在 HUD/允许公开属性能支持时给值，各有 known mask。基础条与额外电池条分开；不把 6/6 与 12/6 夹成同一个 1.0。
- `visible_cooldown_progress, visible_ready_cue, displayed_uses, visible_activation_phase`：保留画面上的冷却、提示、次数、举起/瞄准/持续/待释放状态。看不到的内部冷却帧数或 VarData 不直接输入。
- `game_tick, delta_logic_frames, previous_buttons, held_frames`：用游戏逻辑时间，不用墙钟时间；时间充能可由相邻可见进度及历史学习，不写“等 X 秒再使用”的策略。

**区分存在、充满、画面可用提示、按键已发出、可见使用结果。** `not NeedsCharge()` 不是通用可用性真值：API 文档明确 special 类型即使未满也可能返回 false。不得用它硬屏蔽策略动作。无可见提示时，ready 为未知，而不是由隐藏条件补成 true/false。

`ActiveItemDesc` 有 BatteryCharge、PartialCharge、TimedRechargeCooldown、VarData，但 API 可读不等于视觉允许，也不保证各构建的含义/单位一致。GetActiveSubCharge 与 GetPocketItem 的已知文档限制应在目标构建验证，不作为未经测试的统一读取捷径。

### 9.5 特殊角色 HUD：直接读取可见数值和序列，不做图像识别

**待讨论的具体表示，不是已经接入的字段。** 这里的 HUD 输入指：从引擎/Lua/原生接口读取“画面已经呈现的内容”，过滤后提供数值或类别；不截图、不 OCR、不引入检测/分割网络，也不建立一套通用 UI 控件树。

只扩充三类数据，尽量挂在已有 Body/Slot token 上，避免每个数字都单独增加 token：

| 数据 | 建议位置/表示 | 示例与边界 |
|---|---|---|
| 可见计数 | Body 的资源项 `(kind, value, known)` | 假设 Bethany 的 HUD 显示某资源 4，则输入对应类别与 4；不输入隐藏充能条件 |
| 可见进度 | 所属 Body/Slot 的 `(kind, progress, known)` | 只读已显示的充能/状态条。若画面只是粗分格，不能补成内部精确倒计时 |
| 可见有序图标 | 每个序列元素一个 `(owner, group, index, icon_id, selected, known)` token | 里 Cain 界面已显示的袋内材料、里 ??? 已显示的队列：保留顺序/选中项；不求最佳配方，也不揭示隐藏结果 |

有序心格属于 Body 的生命表示，保留黑心位置、金心、骨心、腐心等可见区别；Unknown 遮蔽当前心格，过去合法观测可以留在模型记忆，但不得从隐藏真值刷新。

这些类别可按资源、道具、图标的既有 ID 建立嵌入，值用数值编码，经小型投影接入现有 Transformer。不是让模型重新从像素识别“这是第几个格子/数字几”，也不是告诉模型这些值该如何影响决策。

每种特殊 HUD 仍需要少量采集适配：绑定实际 getter/可见性，核对单位、顺序和更新时机。这是观测工程，不能声称完全零人工偏置。当前未确认所有角色都有可用 getter；缺 API 时先标注缺口并检查 HUD 渲染所用状态，不直接补隐藏变量，也不自动升级为视觉网络。

双身体各自库存/生命复用前面身体与槽设计；Forgotten 的身体关系、Lilith/Eve 的可见随从归通用实体，不再塞进 HUD。这样避免同一个信息在多种 token 中重复表达。

### 9.6 操作：先验证固定四帧节拍加 Drop，不预定重写动作系统

**状态：实现假设，尚未通过目标游戏行为验收。** 先保留现有 move/shoot/bomb/item/pill 五头和四帧默认决策节拍，新增一个独立二值 `drop` 头，并贯通快速/批量/回放路径。Drop 与移动、射击、Item、PillCard 可以同时输出；不把它们做成互斥类别。

现有 Lua `apply_action` 已用前后 `held` 差异计算 `triggered`，JSON 路径也已有 drop，因此不是从零造按键系统。要确认的是 native 快路径、步进、触发清零、缓存恢复是否保持相同语义，以及组合键是否按目标版本预期工作。

假设每步实际推进四帧，下列是接口意图，**不是已经观察到的游戏效果**：

```text
Drop 0, 1, 0          -> 一次保持四帧的按下，然后释放
Drop 0, 1, 1, ..., 0  -> 连续保持若干步，不重复生成按下沿
Drop 0, 1, 0, 1, 0   -> 两次独立按下
Drop=1 + Move!=0     -> 同时持有两个输入，由游戏解释
```

不写死长按丢弃的秒数或自动替模型完成动作。四帧脉冲是否足以完成短按切换、双击及各类主动，要逐场景实测；它只能表达四帧粒度的序列，不能保证任意逐帧操作。如果确有被时间分辨率阻断的实例，再增加有限的短脉冲/子步控制。**暂不增加持续时间学习头，也不把策略推理频率提高四倍。**

先测 Schoolbag 与药卡切换、长按丢弃、Drop+Item/PillCard/移动、普通 Forgotten 切换、Jacob/Esau 控制和蓄力/释放；记录 requested/applied 动作、边沿、生效帧、实际推进帧数、pending 延迟与画面结果。接口测试通过不等于策略已学会。

身体多不等于控制器多；同一控制器下的身体共享真实按键语义，不给 Jacob/Esau 独立移动特权。协议按控制器保存 held，单人路径只使用一个控制器。模拟量方向、Map/菜单交互以及是否需要更高时间分辨率，作为待验证扩展，不与本轮 Drop 接通捆绑上线。

Pause/Restart/控制台由环境管理。动作 mask 不使用隐藏的“能否成功”条件；游戏允许按但未产生效果仍是合法动作。reset/回放/恢复时必须正确处理 held 与 pending，不能残留或凭空制造按下沿。

### 9.7 场景输入：修复结构化信息丢失，当前不引入视觉网络

**当前路线明确为：可见状态采集 → 结构化编码 → 现有策略/时间 Transformer。RGB 不进入 actor，也不增加 YOLO、ViT 或 SAM 类模型。**

| 已知缺口 | 优先修复位置 | 是否需要视觉网络 |
|---|---|---|
| 压力板/标记石等身份 | 原始 grid 已有 type/variant/state；编码阶段保留可见语义 | 不需要 |
| 冠军颜色 | 桥已采集 champion；将可见颜色类别送入模型 | 不需要 |
| 多数 NPC 动画身份 | 采集当前可见 sprite 的资源/动画/帧/叠加状态，经稳定无碰撞词表编码 | 不需要先做目标检测 |
| 特效白名单漏项 | 审核并采集可见 effect 的位置/外观/动画，取消按十种 variant 决定是否存在的限制 | 不需要先做像素分割 |
| 地图特殊图标合并 | 保留实际显示的类别与显示状态 | 不需要 |

“网格状态”和“动画元数据”仍需通过可见性审查：不能把隐藏 AI state、未来攻击倒计时或隐藏物品 ID 伪装成外观输入。读取 Sprite 等 API 是候选采集方法，不保证所有目标构建或关闭渲染时动画状态仍正确推进；需要真实采样确认。

**不是无成本扩容。** 尽量在已有实体/网格 token 上增补字段；新增实体/HUD 序列才增加 token 数。效果粒子数量、Lua/API 调用、IPC 字节、CPU 编码、Transformer 注意力和 rollout 存储都可能成为瓶颈。以实际实体/序列数量分布选择容量/分桶；不先承诺全部粒子无上限展开，也不静默丢弃后宣称信息完整。重复、装饰对象的压缩需证明保留可见任务区别，而非凭名字判无用。

RGB 的角色仅为离线审计：少量隔离录制实例，检查同一时刻的画面、原始采集和最终 tensor；短暂提示的验证可以只在这些实例逐帧采样。训练 worker 不因此开启渲染、截图或图像传输。记录截图不等于增加模型输入。

对“小视觉网络没意义”的判断需要拆开：整幅图缩小后丢掉的细节，大模型也不能可靠还原；但保留分辨率的固定局部区域可能是较简单的识别问题，不能仅凭参数量断言不可能。当前没有本项目的细节识别与吞吐实测，因此大小视觉网络都不作为解决方案承诺。只有出现确实无法通过允许的状态/渲染元数据获取的关键缺口，才单独评估局部识别；不把它放进当前默认架构。

升级前后要测端到端环境帧/秒、actor 延迟、采集/编码耗时、token 数、显存与训练更新耗时；不能只比较网络参数量。冻结一个视觉模型也只能免去其反向传播，不能免去渲染、图像搬运和视觉前向。没有测量结果，不声称慢几个百分点或一定可承受。

关于 bitter lesson：避免人为规定最佳动作、危险等级、用途和目标；并不意味着必须重复学习引擎已经可靠提供的可见颜色/网格身份。是否存在外观信息丢失，应由对照证据判断，而不是用“更少偏置”直接推导出必须 RGB 端到端。

### 9.8 版本、记忆与验收

版本适配声明字段支持情况，但只向 actor 提供公开的游戏版本/可见协议能力；原始引擎诊断单独存储。Rep/Rep+ 与 REPENTOGON 是不同构建/扩展，不能凭一个文档页面认定其 getter 都可互换。

新增身体、槽、HUD 和场景 token 先进入 Spatial；其输出再供时间记忆与决策读取。新版具体数据流见 9.9，不另造手工任务树。过去合法观测留在时间记忆中；菜单上可以立即读取的完整库存不应人为删去，反过来隐藏库存也不能借“记忆”名义从引擎刷新。训练和推理必须使用同一可见性过滤与缓存失效规则。

验收分开三层：

1. **表示测试（离线）**：所有 34 入口、7 内部状态/身体类型互不别名；第二身体变化确实影响输入；每个槽身份/充能/选择可区分；Unknown/Blind 的隐藏状态差异不能影响 actor 输入；Python 与 native 编码一致。
2. **操作测试（目标构建）**：短按、长按、释放、双击、组合、跨 step 保持和 reset 后清空；Schoolbag/药卡切换与丢弃、双身体/形态切换；基础、时间、特殊、电池和分段充能。对齐视频、逐帧 action、原始观测、最终 tensor 和可见结果。分别测 AB+、Rep、Rep+，无需枚举所有道具笛卡尔积，但所有协议分支及关键交互必须有实例。
3. **学习效果（后续实验）**：固定 held-out 种子、匹配游戏帧和训练预算，比较旧输入、新观测、新观测+完整动作；D6/无D6 分层报告，随后扩展角色和复杂槽。报告超时、往返、层数、交互成功、信息裁剪和吞吐；不根据某场景失败就往策略里塞规则。

旧 C99/C103/C104 语义冻结仅用于历史对照，新协议/动作空间另存版本；本轮禁止兼容载入及旧权重迁移。先完成结构化采集→编码→消费及最小 Drop 动作扩展的验证，再讨论大规模重训。

主要证据：本地 `rl/bridge/python/isaac_bridge/tok_obs.py:164,280-301,435-480`、`tok_policy.py:150-154`、`rl/bridge/abplus/abp_bridge.lua:911-939,1533-1547,1802-1814,2293-2307`；游戏 `resources/scripts_rep/enums.lua:1666-1713,4464-4468`。API 文档：[PlayerType](https://wofsauge.github.io/IsaacDocs/rep/enums/PlayerType.html)、[ActiveSlot](https://wofsauge.github.io/IsaacDocs/rep/enums/ActiveSlot.html)、[ActiveItemDesc](https://wofsauge.github.io/IsaacDocs/rep/PlayerTypes_ActiveItemDesc.html)、[NeedsCharge](https://wofsauge.github.io/IsaacDocs/rep/EntityPlayer.html#needscharge)、[AB+ EntityPlayer](https://wofsauge.github.io/IsaacDocs/abp/EntityPlayer.html)。文档契约和目标构建实测结论分开记录。

### 9.9 历史方案：可重建的时间记忆：取消 Temporal → H → Spatial

**状态：2026-10-24 按候选代码与当日决定改写。** 记忆链路已有候选实现 memory-chain v2（`temp/memory-chain-v2/candidate`，正确性门禁通过，吞吐验收未通过：最好 370.2×，旧架构同机 843.4×）；有界梯度 horizon 由 memory-chain v3 实现（`temp/memory-chain-v3/candidate`，CPU/CUDA 正确性门测通过，长测进行中）。两者都没有合入生产；本节描述的是候选代码的实际行为，不是生产模型。

- [新版 Draw.io](model_structure_versions/20261022-memory-rebuild-proposal/main.drawio)（10-24 重绘）：01 总览、02 记忆写入/读取/梯度、03 输入采集与按变化编码、04 旧 H/KV 结构对照页。
- [记忆路径预览](model_structure_versions/20261022-memory-rebuild-proposal/memory.png)；[输入与编码预览](model_structure_versions/20261022-memory-rebuild-proposal/encoding.png)。
- [上一版输入扩展图](model_structure_versions/20261020-token-expansion-proposal/main.drawio)保留不动。

#### 数据流与形状（候选代码 `isaac_bridge/rebuild_memory.py`、`tok_policy.py`）

1. `Y_t = Eθ(obs_t)`：输入编码器与 Spatial 只接收当前合法观测，不接 H。Spatial 输入/输出 `[B,T,192]`。历史候选当时 `T ≤ 88`（P1、R1、M1、D≤8、E≤64、I12、G1，去掉 H）；输入扩展（R 28+1、E≤128、全部 N+S 道具记录、每个身体一个 P）当时尚未实现；本轮已由9.13的新输入路径替代，不是当前规格。
2. `z_t = LN(Y_t[P])`、`g_t = LN_G(Y_t[G])`、`u_t = W_u[z_t; g_t]`，384 → 256。G 是学习查询，不是预先算好的历史摘要。所有历史帧可并行执行同一个 Eθ。
3. Writer：每 64 个决策一块，4 个学习查询对块内 u 序列做一层 cross-attention 加 LN（块内正弦位置编码），输出 `M_b: [4, 256]`。块之间独立，不读上一块、不读 Temporal 输出；全部帧参与，不做事件评分或道具重要性筛选。
4. 历史源 = 最近 256 帧的 u（`t−256 ≤ τ < t`，当前帧只作查询）加本局所有已完成块（`block_end ≤ ⌊t/64⌋·64`）。分块不随 rollout/minibatch 切割；换房不清空；不再有 128 次访问的上限。
5. Reader：四层 cross-attention，d=256、4 heads、FFN 1024，层内没有自注意力。`Q = u_t + PE(t)` 逐层更新；各层 K/V 由源 u 和 M 投影并带一个 null key，**不由过去决策的 Reader 输出生成**。mask 限制同 episode/分支、只读过去帧和已完成块。
6. `c_t = Attention(Q = W o_t（256→192）, K/V = Y_t)`，单层 4 heads，重新查看当前完整 token 序列。动作、局部价值、危险头读 `LN(Y_P + P(o_t) + c_t)`，全局价值读 `LN_G(Y_G + P_g(o_t) + W c_t)`。o/c 不回灌下一帧 Spatial，也不写入记忆源。

64/4/256 是起始预算，Writer 仍是有损学习压缩，块容量、宽度、层数可扩展；不能称其为无损或保证学会回头。读取范围随经历增长，单步读取成本也随之增长。

#### 参数版本、延迟梯度与有界 horizon（替代 10-22 的"参数新鲜度"条款）

**原则变更（用户 2026-10-22 确认）：不再要求历史表示由当前参数每步重建。** 原始可见观测、时间、episode/分支身份和源引用仍是历史的权威记录；PPO 的旧 log-prob、奖励与既定目标按原语义保存。

- **两套更新周期**（`source_pipeline.py`，参数集合见 `source_parameter`）。快周期：Reader、K/V 投影、`c_t`、各头，每个 minibatch 一次 Adam，候选实测每次 update 18 次。慢周期：输入编码器、Spatial、Writer、`W_u`、G，先累积所有消费者的梯度，再统一 clip 并做一次 Adam，之后源参数版本加一。频率由 `--source-update-steps` 控制（v2 默认 4，即每 4 次快更新一次；0 = 累积整个 rollout，配合 `--source-update-rollouts 1/2/4` 每 1/2/4 个 rollout 一次）。
- **版本化源图与延迟梯度。** 每个帧和块在编码它的参数版本下保留计算图和不可变权重快照（源侧精度 `--source-amp`，BF16 为单独标注的配置，Reader/master/Adam 保持 FP32）。后面房间的损失经 Reader 的 K/V 回到原来的图，梯度加到当前 master 参数上。这是延迟梯度优化，不等于当前参数下的梯度；候选文档和图都按此标注。默认不做前向重放，`--source-replay 1` 只是诊断用的省显存模式。
- **Actor 的参数刷新。** 一个采样段内固定 actor θ 做增量编码和缓存；新 θ 到达时只重投影 Reader 的 K/V（`rebuild_actor.py` 第 102–117 行），u 和 M 保留它们各自版本的值；组件缓存每 8 个版本清一次。不重建活动历史。
- **有界梯度 horizon（用户 2026-10-24 决定，v3 实现）。** `--memory-horizon H`，单位为该 episode 内的决策数，默认 2048（4 帧/决策时约 273 游戏秒），0 = 无界即 v2 行为。读取不变，仍是整局；决策 t 的损失只回到年龄不超过 H 的源：帧按 `t − s ≤ H`，Writer 块按起始帧 `t − b ≤ H`（整块在 horizon 内才算）。没有任何未来 in-horizon 决策能到达的图，先在同一个 Adam 间隔里结算已累积的梯度，再释放计算图，只保留值供读取；被仍在 horizon 内的 Writer 图引用的帧块随该 Writer 一起释放。于是每个 worker 保留图的帧数有上界：H 加一块再加被钉住的块。实现：`HorizonAttention`（前向与原生 SDPA 相同，反向按决策与源的年龄逐对屏蔽源侧梯度，Reader 自身 K/V 权重仍收到全部梯度）和 `retain_histories(frontiers)`。progress.csv 新增 `source_graph_frames`、`source_graph_frames_uid_max`、`source_detached_frames`、`source_horizon_dropped_grad`，最后一项必须恒为 0。
- **梯度路径。** 后续房间的 loss → 当前融合与 Reader → horizon 内的记忆块 → Writer → 对应帧的 Spatial attention → 输入编码器；最近帧分支同样回传。minibatch、房间、rollout 边界不截断，只有 horizon 截断。这是表示与写入的信用分配，不等于修复 PPO/GAE 对早期动作的长期奖励归因。
- **已放弃的方案及原因。** 每个 optimizer step 全历史重建：每帧平均 15.2 次 Spatial 前向，完整管线 34–57×。无限 horizon 加留图：保留激活随局长线性增长，24 GB 源卡和 72 GiB 容器内存上限下每次长测在 20–47 次更新后 OOM，最好 370×。CPU 激活转存：传输和主存压力使 learner 被容器终止。8 bit 保存激活：源显存减 23%，源梯度相对误差 1.3%，但 VJP 慢 24%，未启用。

### 9.10 历史方案：按变化编码 obs：复用独立源，不复用陈旧融合结果

**用户补充并认可：相邻决策中不变的道具、地图、属性等可统一编码，仅重算变化部分。** 候选实现为 `ComponentCache`（`rebuild_memory.py`）：模块/记录级复用，缓存边界在 Spatial attention 之前。输入采集仍要获得正确的当前可见状态；减少网络计算不自动意味着减少 API/IPC 成本。

| 源 | 可复用的编码 | 失效依据 |
|---|---|---|
| 道具/物品槽 | 每条独立记录的 embedding/MLP；相同 ID 的基础 embedding 可复用 | owner、ID、数量、槽位、充能、选择、顺序及 known mask 中任何被消费字段变化；不能只看道具 ID |
| 地图 | 完整 Map CNN 输出 | 任一输入通道变化，包括当前房间标记、探索/访问状态、可见图标和 mask |
| 房间地形与门 | 完整 Room CNN 的 28 个输出 | 地形、门位置/开闭/锁定、可见性等任一输入变化。**2026-10-24 决定：取消独立 D/X 门 token，门与出口字段作为 Room CNN 输入通道** |
| 身体与属性 | 完整 Body 编码；显式独立的静态身份 embedding | 生命、属性、动画、位置等消费字段变化。**2026-10-24 决定：诅咒标志和层数放进 P token；多个身体时每个身体一个 P token，排在主身体之后，动作头读主身体的 P** |
| 实体与 Local Patch | 输入完全相同的独立实体编码/局部 CNN | 实体位置/状态、顺序/位置编码、mask、玩家局部裁剪变化；预计复用率较低，须测量 |

缓存键 = 可见性过滤后的**原始载荷字节完全相等**加源参数版本，不用哈希或高成本指纹。源版本只在实际输入改变时更新，不能用隐藏引擎状态的变更泄漏信息。无可靠版本信号的字段比较完整编码输入，不能把"没收到事件"当成一定没变化。

复用后重组完整的当前 token 序列，应用当前 owner/顺序/位置/mask；不是只把变化 token 交给 Transformer。一个 token 变化可影响所有注意力输出，因此 Spatial 的融合结果 Y 不按单个道具/地图身份缓存，每次决策重算 Spatial 与当前融合。CNN 的局部输入变化可能通过感受野影响多个输出 patch；不把"某个地形格变了"实现成只更新同位置输出 token。

训练阶段：同一个源参数版本内编码唯一源，所有消费者的梯度先 scatter-add 到 FP32 缓冲，再对唯一源做一次累计 VJP；必须覆盖数量、owner、位置等独立组合分支。缓存数值不是永久 detach。**快周期的 Adam 不使源表示失效**；只有源参数版本推进后，新帧才用新版本编码，旧表示按其版本保留到 horizon 之外或 episode 结束。推理阶段缓存在固定 actor θ 下跨决策保留。

episode reset、存档恢复及反事实分支需要正确的源作用域；分支追加/修改不得覆盖父状态。可安全共享相同只读可见源，不能混用不同 episode 的 Reader 查询历史。

### 9.11 历史方案：训练提速与 PPO 更新边界

候选代码一次 update 的实际顺序（`train_tok.py` 与 `source_pipeline.py`）：

1. `retain_histories(frontiers)`：淘汰不在 rollout/教师工作集里的 episode；按 horizon 结算并释放不可再达的图（9.9）。
2. `prefetch`：只编码本 rollout 新增的帧、只写入新完成的块，旧表示直接复用；源编码在独立的源 GPU（`--source-devices`，可选独立进程 `--source-processes 1`）上与 Reader 重叠进行。
3. 建立 rollout 的常驻数值 bank；每个 minibatch（`--micro 8192`，固定）做 Reader 前向与 loss，累积 dL/du、dL/dM、dL/dY 到 FP32 缓冲，Reader/融合/头做一次 Adam。重复引用的梯度求和，按有效样本集合归一化。
4. 慢周期到点时，累计梯度回到各自版本的源图（VJP），统一 clip，源 Adam，源版本加一；保存检查点前提交未满一个周期的尾包。

规则与现状：

- micro 固定 8192，不为省显存缩小；显存确有需要时用 4096×2 或 2048×4 累积，保持等效 batch 与 Adam 步数不变。不做前向重放，不截断历史读取。
- 原"调度 B"（每 epoch 一次大更新，15→3 次）不再实施，由快慢双周期替代。慢周期频率是**待 A/B 的学习变量**：每 rollout 一次源 Adam 吞吐最好，但编码器和 Spatial 每 rollout 只更新 1 次，旧架构是 18 次；速度验收按每 rollout 一次跑，学习效果验收须同时跑"每 4 次快更新一次"的对照臂，必要时给慢周期单独配学习率。
- 吞吐现状（同一台 8×3090 主机，64 workers）：旧架构 843.4×（2 actor GPU + 1 learner GPU）；v2 最好 370.2×（5 卡，28 次更新后 OOM）；v3 有界 horizon 的 4 卡组内 A/B 与 5 卡可比组进行中。**验收门槛不变：相同条件下 40 次稳态更新的吞吐不低于旧架构，未达标前不替换生产版本。**
- 注意力优先使用 PyTorch SDPA；`HorizonAttention` 的自定义反向在 CUDA 门测中不慢于原生（27.0 ms 对 30.3 ms，96 组×128 查询×1600 条目），峰值显存多 23%，仍需在长测里确认。

### 9.12 实施与验收

新输入与块记忆共同构成本轮模型。所有权重（包括Spatial、输出头、Reader、Writer、aux与priv）及优化器从随机零步初始化；禁止C99及其任何派生initial.pt/checkpoint。只允许同一新schema训练产生的严格续训。保留PPO、反事实搜索与伤害教师。

- **正确性**：按9.13.3–9.13.4使用128步chunk、独立32步Writer块和默认4个窗外历史块重算（4/8可配置）；逐步actor与并行learner同历史输出对齐，未来不泄漏，选中源块有正确Spatial/Writer梯度，未选缓存detach；reset/分支恢复不串状态。
- **缓存等价性**：缓存开/关的前向和梯度一致；逐项变化 ID/count/owner/charge/map/door/known mask 等会触发相应失效；不可见隐藏字段变化不影响 actor；测试重复源多消费者的梯度求和。目标是被缓存的模块少算，不是省略当前信息。
- **记忆能力**：覆盖长于 256 步、跨 rollout、长房间与超过 128 次访问；沿用此前随机背景下失败的延迟线索任务，再测试真实游戏回头交互。非零梯度和简单线索成功不是正式游戏能力证明。
- **速度**：固定硬件/并发、输入分布与完整训练机制，按既有基准跑足稳态；计入采集、IPC、缓存查询/失效、actor 权重刷新、历史重算和教师成本。分别报告组件命中率×编码耗时、Spatial/Writer/Reader 时间、FPS、actor p50/p95、峰值显存及原始历史存储。命中率本身不是提速率。
- **学习收益**：旧主干仅作对照，新主干采用9.13的PPO调度和128步chunk；同游戏小时数与同墙钟成本同时比较。128 种子跟踪、512 种子正式比较，并报告无 D6/有 D6 的超时、往返、层数与交互行为。

与 [SCALING_THESIS.md](SCALING_THESIS.md) 一致：使用通用观察编码与注意力，不添加玩法答案；容量、压缩与时间预算显式可扩展；把梯度完整、状态新鲜、缓存等价、吞吐和学习收益分别验证。

### 9.13 下一代四 latent 分块记忆模型：最终设计（2026-11-06）

**状态纠正：新输入功能/稳定性测试已完成，但当前在线8.97×吞吐不合格；不能作为正式训练可用性验收。停止正式长训准备，先修实现与并行吞吐。** predictive-aux c45a583仅为代码基线来源，不是模型初始化来源。实现副本位于 D:/Projects/fortune/Isaac/temp/wt-block-memory-v1；原工作树、生产配置和检查点未改变。本节替代房间摘要、Temporal→Spatial反馈、长期持图和延迟源梯度，9.9–9.11仅为历史。

目标顺序：保留完整反事实教师；固定形状合批和并行计算；建立可学习的独立片段记忆；再扩参数。首版不同时改变奖励、动作空间、PPO调度、搜索预算或引入APPO/V-trace。下面是确定的V1规格，不是候选清单。

#### 9.13.1 结构与首版参数

    当前可见观察
      → Spatial 192维×4层，4头（全新visible-v1独立编码器）
      → 4个learned-query读出口 → 每步4×256 latent
          ├→ Temporal 256维×4层，4头，FFN=4d
          │    最近128决策步（含当前步），共512 token
          │    第2、4层插入长期cross-attention
          └→ 每32步组成独立块（128 latent）
               → Writer 256维×2层，4头
               → 每块8个memory tokens
               → FIFO bank：256块、2048 tokens
                            ↓
                    Temporal长期读取
                            ↓
                  当前4槽的learned-query聚合
                            ↓
                   同一策略的动作／价值头

- Spatial采用P每身体＋R28＋L每身体＋M＋E128＋完整N＋实际S＋G的新编码，不复用旧encoder参数。删除Q(o_prev)历史反馈；privileged输入仍不得进入actor。
- 4个独立learned queries直接cross-attend尚未池化的Spatial tokens；禁止把原单个z复制四份。槽不手工指定玩家／敌人等语义。
- Writer的两层均由8个query对128个输入的cross-attention、8槽self-attention和FFN构成，pre-LN残差，FFN=4d；块内步／槽位置显式编码。
- Temporal使用pre-LN局部self-attention；第2、4层在局部attention后增加长期cross-attention，然后FFN。memory从256维投影到Reader宽度。空bank时长期读取残差为零。
- 当前步4个最终输出通过一个learned query聚合。动作头仍为(9,5,2,2,2)，不是4套策略；局部／全局value保留各自目标，现有其他头按开关接入。
- predictive aux挂新的决策表示；privileged critic仍仅在learner。保持现有开关和系数，V1不加新的重建/world-model损失。

#### 9.13.2 独立块、因果性与生命周期

定义M_b = Writer(Z[32b:32b+32])。Writer不读Temporal输出、前一块摘要或旧bank，所有块可沿batch轴并行；跨块关系由Reader理解。

- 32步块从episode第0步对齐，不因房间／rollout切换重启。episode结束丢弃未完成块，不补未来；换房不清空记忆。
- 完整块从结束后的下一决策步可见。0–31步的摘要最早被第32步读取。训练按块结束时间mask，禁止未来泄漏。
- 同一步4槽双向可见；跨步严格因果。短期mask为0 <= query_step-key_step < 128，并隔离episode。
- 时间编码使用决策步、槽序号、块顺序／相对年龄；环形物理位置不能冒充时间。长期读取有块年龄和槽身份，不以房间编号替代时间。
- 允许完整摘要与短期窗口短暂重叠，V1不增加去重门控；这只是过去的不同粒度表示。
- 长期库按完整块FIFO。满256块后淘汰最旧块继续写入，不采用旧128房间槽满后停写的规则。
- 保存Writer latent作为长期记忆主体；Reader投影KV只是可失效的加速缓存。episode reset清空所有有效状态。
- 256块直接保存8192决策步；按4逻辑帧/决策、30帧/秒，约18.2游戏分钟。不是无限历史承诺；V1不引入稀疏检索或多级递归压缩。

#### 9.13.3 并行执行、算力和扩参

- 已知轨迹上：各帧Spatial并行；各块Writer并行；训练片段的Temporal用因果/局部mask并行。历史缓存是显式输入，不把截断上下文说成整局精确反传。
- 在线环境仍须等待动作后产生下一观察；并发来自多环境和搜索分支，不声称能并行生成未来环境状态。
- 复用PyTorch成熟attention实现，采用能表达本设计并跳过无效块的mask/kernel；仅给巨大dense矩阵填负无穷不等于稀疏提速。保留固定shape桶、pinned传输、GPU常驻与CUDA Graph。
- V1先固定Spatial192×4、Temporal256×4。扩参档M：Temporal384×6/6头、长期读取在3/6层；L：512×8/8头、读取在4/8层。FFN=4d，每头64维。latent和memory维度均保持256，读取时投影。
- 首轮只扩大Temporal宽深，不同时增大窗口、bank、latent数或搜索预算。Spatial/Writer扩容另行比较，接口支持但不与首轮混在一起。
- 参数规模约随L*d²，短期KV随4*W*L*d，长期原始latent随blocks*8*256增长。额外Reader投影KV、搜索状态、rollout、训练激活必须另计。
- 相同步数从1变4 latent，常规attention交互项可达16倍；512-token短窗不是免费加速。性能以含教师的端到端测量为准。

#### 9.13.4 PPO与预算化历史梯度

保持现有PPO目标、奖励、behavior log-prob、value/bootstrap、GAE／截断语义、优化步数和有界overlap。新训练chunk固定128决策步；逻辑batch、优化器步数和采样预算在对照中保持一致。

- 样本包括当前观察、历史块引用、短期上下文、episode／位置和版本。保留历史原始观察引用；只有旧latent不能训练Spatial来源。
- 128步chunk内部按当前参数重算；chunk前的短期逐层KV直接作为detached输入，不做无界burn-in。跨chunk边界的32步Writer块用原始观察补齐，归为本chunk完成的块，不占额外历史重算预算。每个chunk另从短窗外（end_step <= chunk_start - window）均匀无放回重算历史块；`--block-history-blocks` 默认4，保留4/8消融，预算不足时取全部。取消必选最新块，避免把预算花在短窗已可见的内容上。
- 被选块用当前Spatial→Writer重算、替换对应摘要并接入当前loss；其他历史摘要detach。每个query仍执行时间和FIFO可见mask。更新时重新采样，不用attention Top-K垄断梯度选择。
- 同一逻辑minibatch中重复源可去重编码并累积梯度；优化器步后不复用带图结果。不保留跨更新源图，不执行旧v3式延迟源梯度。
- 这是固定预算、截断历史的梯度近似，不是全历史完整或无偏梯度。独立Writer允许直接重算远期块，无需展开其到当前之间的所有时间状态。
- V1不新增压缩辅助loss。PPO、现有预测辅助及教师损失通过上述路径训练Writer；真实远期任务梯度只覆盖重算块。
- actor bank记录Spatial/Writer源版本，短期KV记录生成版本。发布权重沿用排空in-flight边界；Reader参数改变后旧投影KV失效，从保存latent按新参数重新投影。
- V1明确允许带版本的历史latent和短期KV陈旧，不在每次更新时重编码全部8192步。learner重算当前chunk及被选历史块，chunk前短期KV和其他历史缓存detach；不把训练重算结果回写线上历史。
- PPO分母保留实际执行时behavior log-prob，不用事后重算概率替换。必须记录policy lag、memory版本年龄、历史重算成本；部分历史重算和缓存陈旧都是学习评估的组成部分。
- 原始源观察按FIFO及训练／搜索引用计数释放，不永久保留整局数据图，不给每条样本复制完整历史。

#### 9.13.5 反事实教师与受伤回溯必须保留

当前CF搜索和独立受伤回溯教师都是必选迁移路径。保持每worker每游戏小时6点、4分支（含原轨迹、目标意图、sticky探索）、最长180游戏秒、目标干预20秒；保留换随机种子复跑，收益超过0.3且超过复跑差异才入库，胜出前8步、权重min(收益,2)。未启用的其他扩展不自动打开。

- 保持胜出动作加权监督和return-to-go价值回归，cf-advantage=0；不改为软动作分布，不替换PPO优势，不偷偷增加搜索预算。
- 使用相同网络架构的策略快照，不另训简化教师。每actor最多2个不可变teacher参数generation；同一选择点的候选与复跑绑定同一generation。无引用槽才可更新；满时搜索入口有界排队／回压，不无限复制模型，不暂停主actor等待搜索。
- 保留真实原轨迹已发生的奖励，末端bootstrap按统一比较版本计算。明确报告实际完成搜索点／分支数和teacher版本年龄。
- taken评分必须保留从选择点到评分终点的真实观察、已执行／在途动作和终止标记引用，或等价的可重放记录，直至评分结束。使用绑定generation从同一选择点记忆重放到终点，只重建记忆与bootstrap，不用新采样动作改写真实轨迹；该额外计算计入教师成本。
- 分叉快照在当前观察尚未消费前取得，包含环境恢复身份、短期KV、完整bank、未完成32步块的latent和源引用、episode/时间/块位置、mask、在途动作以及模型／缓存版本。
- 保留一拍动作延迟：分支第一步执行已在途动作，新选择从下一步生效。
- 已完成历史块只读共享；短期状态、未完成块和新增摘要为分支私有尾部。复跑必须从同一快照恢复，不能承接上个候选状态。假想未来不得进入主轨迹或兄弟分支记忆。
- 搜索目标可来自未来回报，但选择点的输入不得包含搜索未来观察或摘要。搜索过程无梯度；learner重算Reader并按可配置窗外历史预算重算，使教师loss可训练Writer和Spatial。
- 教师记录新增快照和历史源引用；不能仅存旧o_prev/o就认为完成接线。同一起点的胜出短轨迹合组处理，避免每条教师记录复制bank。
- 保留GPU lane动态合批、异步回复、请求序号验证与批次无关随机数。同lane有序，不同分支可混批；游戏RNG复跑与策略采样随机流分别管理。

#### 9.13.6 系统、初始化与完成标准

保留可扩展的CPU游戏worker／中央GPU推理／独立learner拓扑及overlap、decoupled、共享内存IPC。此次单4090采用一个actor进程、同进程learner线程与多CPU游戏/search worker；不冒充三卡验收。普通采样、CF和伤害lane使用相同完整新输入。

全部参数从随机零步初始化，旧Spatial也禁止迁移。`--init`在新模型入口明确拒绝；checkpoint/job/snapshot必须带新schema，旧格式拒绝。初始记忆、rollout、双教师buffer和Adam为空，不混入旧PPO数据。

以下为实施验收要求。**旧输入v1的r8/r2/aligned功能结果已完成并保留于下文；11-07审核修订的输入v2/相对位置/窗外历史预算不能沿用它们的验收结论。新一轮端到端验收、同条件旧主干性能比较、多GPU和第7项长期学习收益尚未完成，不自动替换生产**：
1. 固定权重与相同历史缓存下，逐步actor／并行learner／搜索分支对齐；未来观察改变不影响过去输出，空bank无非有限值。
2. 32步块边界、128步短窗、256块FIFO、episode reset、换房不reset、跨rollout块延续正确。
3. fork/restore与无fork重放在状态、动作延迟和随机流上一致；分支不互相污染，batch顺序不改变请求随机数。
4. PPO与CF loss均可通过选中历史块回传Writer/Spatial；未选缓存不产生伪梯度，教师输入无未来泄漏。
5. 完整CF和受伤回溯在真实游戏中跑通，不能以无教师PPO冒充完成。
6. 同硬件/并发/搜索预算/优化器步数，至少40次稳态更新测decisions/s、collect/update/wait、搜索完成率、actor延迟、GPU/host峰值内存、历史重算和版本年龄。吞吐未达同条件旧基线前不替换生产。
7. 同游戏步预算和同墙钟成本比较学习效果；沿用128种子跟踪、512种子正式评估，区分结构与参数扩容；兼容初始化仅历史对照，本轮禁止。非零梯度不等于学习收益。

最终路线：先交付包含真实采样、完整教师、有限历史梯度的新小模型，再按同一拓扑扩宽加深；不是无教师简化版，也不同时重写RL算法。

**2026-11-07 新输入实现与本轮验收（已完成）：**
- 新路径为 Lua visible extension → LeanDecoder → 独立完整帧 → typed batch → StructuredSpatial → 4 latent；不调用旧 `_embed_old`，也不保存旧encoder的闲置参数。
- P保留每身体/控制器/角色关系与可见状态；未知心量、未知动画遮蔽，内部 fire_delay/damage_cooldown/weapon charge 不输入。R保留客观碰撞、可见grid type/variant/state及门/出口通道，CNN为64×4×7→28 token；L按每个身体提取9×9独立patch。
- N逐身体扫描完整道具catalog并保留count；S按实际capacity枚举trinket/pocket，包含Schoolbag备用主动和电池充能。未知药效不伪造，未知充能不作为禁止使用按钮的通用硬门控。
- 传输为128KiB字节预算原子槽，内部是按P/E/N/S计数的完整帧。房间/局部离散通道无损打包，超过字节预算拒绝整帧，不截N/S。8身体×552种道具、E128的打包/读回纳入测试；这是资源预算，不是token语义上限。
- Spatial graph图外解码，键包含P/E/N/S形状，超1024空间token完整eager，最多缓存32种图。learner持capture admission锁，忙时新shape走eager，命中replay仍可overlap；不在另一线程提交CUDA时首次capture。
- 最终本地CPU：54 passed / 9 skipped（8 CUDA、1 Linux）；4090/Linux最终：63 passed / 0 failed / 0 skipped。另有stream回归、双教师数学检查及272项legacy CPU检查通过。真实Linux memfd测试携带有效新帧和block snapshot，子进程重算逐值相等，写回对父进程可见。
- r8的2次冒烟、50次离线、真实在线40次稳态要求全部达到。r7在线预检为纳入权重入口修复而主动停止，不计最终验收。命令、原始stdout/stderr、exit及失败修复历史保留在既有VERIFICATION.txt及本轮new-input-20261107目录。
- 历史旧输入的49项回归、CPU联调和旧4090离线50更新仅作历史证据；C99派生结果不计本次、旧三卡/双卡launcher不得恢复。


**4090最终测试结果与速度（r8；4原生workers、单GPU actor＋learner线程、BF16、overlap/decoupled）：**

| 验证 | 已观察结果 |
|---|---|
| 初始化 | Spatial、Reader、Writer、heads、aux、priv均从新随机权重起步；初始update=0、decisions=0、Adam state=0；所有验收init/resume为空，未加载C99及其派生权重 |
| 原生输入 | 连续3次观测均为77种道具、总数量78、6槽（双主动/双饰品/双口袋）；完整frame与独立游戏getter一致；原生该场景为单身体，多身体由合成测试覆盖 |
| 日志 | 两次独立冒烟各1026条完整raw rows逐有效frame字节读回一致；format=2，schema与temporal-zr-zg语义绑定 |
| 2次独立冒烟（每次2更新） | 每次1026、合计2052训练decisions，两次exit0；双教师记录进入buffer，但尚未预热到实际teacher优化阈值 |
| 50离线 | 25650训练decisions；伤害第18–50次优化、CF第32–50次优化，共同19次；全局梯度50/50有限且非零 |
| 真实在线 | 62次更新、127071训练decisions，exit0；第22–62次连续41次双教师实际优化（异步队列多完成1次），满足40次要求 |
| 在线错误 | 全程errors/stalls/cf_errors/v2_errors/log_dropped为0，log_error为空；所有非空loss/梯度无NaN/Inf |

- 离线50的总update时间158.802秒，161.522 decisions/s；含启动/保存的进程墙钟185.281秒，138.44 decisions/s。双教师共同激活19次的update中位4.608秒、111.68 decisions/s。**这是重复同一真实新策略job的离线重放，不是在线新采样。**
- 在线全程墙钟1803.058秒；最后40次（update23–62）新增81983训练decisions，逐更新wall_s合计1218.850秒，**67.263 decisions/s**。collect中位/P95为29.646/32.035秒；update为18.045/20.563秒；周期wall为30.825/33.967秒。wall_s是相邻完成间隔，overlap下不能把collect与update相加。
- 最后40次真实新增：CF完成270个搜索点、879条branch、58565分支decisions、48改善点；worker生成1536条CF记录，learner接收1568条（异步边界差一个32条批次）；伤害搜索344次、340次可避免结果、learner接收428条。两教师的pi/danger/CF-value损失40/40为正且有限，末40梯度范数1.164–8.141。
- **CF预算准入存在背压，不宣称零丢弃搜索**：末40次cf_nobank增加5000（选择点快照槽不可用/快照拒绝，不是长期记忆bank为空）、cf_dropped增加225（候选队列淘汰）。这与完整N/S观测不截断是不同边界。本轮采用CF10秒/目标3秒、加速触发的联调配置，不能冒充生产搜索预算。
- 单次H2D typed collate：batch1中位2.223→1.455ms（-34.6%），batch4为2.427→1.857ms（-23.5%）；**batch32为3.243→4.655ms，慢43.5%**，不宣称全batch提速。完整ROW仍151880字节，历史存储并非只保留有效frame。
- 资源采样83次、约20秒间隔：GPU used最高5313MiB、CUDA reserved最高4.334GiB，进程树RSS相加最高26.822GiB（共享页重复计数，不是PSS或cgroup用量；非严格瞬时峰值）。BlockActor未填旧infer_ms计数，日志0不代表0ms，不能用作actor延迟结果。
- 独立多GPU learner/actor拓扑保留，单4090不绕过同卡双进程保护。新payload真实memfd与snapshot IPC已验证，但不等同多GPU完整双教师训练验收。未做长期胜率/收益比较，不因梯度非零就宣称学习更强。

**本轮已修复且重测的实质问题：**新ROW握手字段被初始化清零；旧FastRow v7不携带新帧（新路径禁用）；动态P/E/N/S图键与capture/learner并发准入；stream schema/表示语义与匿名padding；自动resume和eval遗漏schema、兼容迁移旧权重入口。后者先新增3个失败用例，再修至3 passed；新block禁止load_compatible，exact schema用strict加载。旧m.player/旧graph-key假设属于迁移测试fixture修正，不当作模型学习成果。

完整机器可读统计：`D:/Projects/fortune/Isaac/analysis/model-design-20261106/new-input-20261107/RESULTS.json`。
原始验收与CSV/JUnit：`D:/Projects/fortune/Isaac/analysis/model-design-20261106/new-input-20261107/remote-results`。
远端结果：`/home/linux/isaac-block-memory-v1/reports/new-input-20261107`。

**2026-11-07 性能复核、原设计回溯与当前边界：**

原设计是解除 Temporal→Spatial 与递归记忆的串行依赖，使 Spatial 按帧、独立 Writer 按32步块、Reader 按已知轨迹并行；再以中央 GPU 服务汇集主环境和搜索分支。4 latent 不是自动加速器。当前训练时间维确实批量展开为 `[B,4T,D]`，不能误称 Reader 仍逐时间步运行；但中央服务外围仍有 Python 状态表拼装，完整异步 GPU slot 后端尚未实现。

历史口径已重新核实：旧三卡843.395×有原始记录；**另有单3090的220.905×**（12 workers，rollout32768，epochs3，minibatch/micro8192；两次25更新各去掉5次warmup）。后者有287点/993分支/608269分支decisions的实际CF搜索，也有伤害教师，但CF蒸馏尚未持续优化。证据在 `D:/Projects/fortune/Isaac/temp/memory-chain-v2/gpu-unlock/TRAINING_RESULTS.json` 和同目录 `prepared-evidence.tar.gz`。不能用“三卡对单卡”概括所有200×证据；也不能把旧输入、不同搜索/优化预算当成严格配对。仅查日志/配置，未加载C99或衍生权重。

已完成的r2历史验收：77项GPU回归、两次2更新冒烟、离线50、在线67更新（连续42次双教师）。最后40次82298 decisions / 513.493秒 = **21.369×**。旧宿主OOM已通过驻留LZ4历史、CPU快照独立storage修复，末轮零游戏/教师错误，最低MemAvailable23.976GiB。细节及失败记录保留于 performance-fix/FINAL_ACCEPTANCE-r2.json；这些完成记录不替代本次修改后的验收。

本轮按依赖关系实施，不减输入/模型/教师预算：

1. `forward_chunks(require_next_state=False)`：只训练损失时不构造下一状态；尾部与当前chunk总长不超过32时，尾部Spatial重算不影响当前loss，复用detached尾部。仍保留原Writer执行及每样本最近＋随机两历史块，避免改变None/zero梯度与AdamW语义。32个phase、跨块边界、CPU/CUDA FP32/BF16输出和所有参数梯度配对通过。
2. `RawBlock.encoding_key`：同一forward内，相同完整ROW＋pending的独立Writer来源共享一次带梯度重算，分支loss梯度照常相加；不跨optimizer step缓存输出。原始输入摘要替代昂贵重复编码，不是通用完整性脚手架。当前key保守覆盖整个ROW；进一步排除fork元数据必须先测native命中收益。
3. room位字段直接解到channel-first float32，覆盖全部19-bit值的精确测试；`move_snapshots`把kv_steps/versions/partial逐条阻塞复制改成一次字节slab，保留dtype/shape，CPU保存端拥有独立storage，包括同设备路径。
4. 控制面只在必要入口回收，向量解码引用，仅扫描injury ring的episode/t两列并保留双commit与jobs seqlock，先读kept再读ring的发布保护顺序不变。
5. 主动作就绪即答复，旧版本教师随后继续执行，让游戏推进与教师服务重叠；同一generation的replay和正常搜索合批，仅按权重版本隔离。replay严格输出pending、logp=0、不创建教师记录；CRN仍依原seq/lane，测试不消耗Torch RNG。无CRN模式保持抽样分布，不宣称逐seed轨迹一致。
6. 修复完整回归发现的旧GraphActor CUDA捕获错误：旧图显式固定legacy布局，在host拒绝visible-v1；新模型仍只走完整新输入路径。

配对离线profile：同一保存的真实新输入job、相同随机初始权重、逻辑minibatch256、micro512、相同教师预算，成熟u35–39更新中位 **4.818→1.785秒（2.70×）**。重复同一job有较高来源复用，不是原生在线速度。native pilot u25–36（12次共同教师更新）为30.446×；接通提前主回复后同窗口31.197×，主结果就绪中位18.930ms，而整个step_main服务39.384ms。两轮零游戏/教师错误；12次窗口不冒充40次稳态。

真实成熟actor cProfile再次定位：129轮主请求触发641次BlockActor.step，旧路径按generation/replay过度拆组；BlockMemory.forward累计2.822/7.436 profile秒，18,150次pad、12,243次stack。profile有测量开销，不参与速度验收。由此实施上述同generation合批，而不是盲目再调worker数。`infer_ms`是服务墙时，不是GPU kernel时间；`idle_share`是worker等答案比例，不是GPU空闲率；collect/update重叠，不能相加。

<!-- ALIGNMENT_FINAL_GATE -->
最新本地94通过/21硬件跳过，GPU115通过/零跳过；两次各2更新冒烟、离线50、在线63更新全部exit0。在线22–63连续42次双教师优化；末40次24–63共82303 decisions / 359.253秒 = **30.546×**（229.095 decisions/s），collect/update/cycle中位5.553/7.390/8.958秒。末40次CF315点/1028分支/70615分支decisions，新CF1760条、伤害390条，340次伤害搜索；全部错误计数为0，loss/gradient/checkpoint有限，init/resume为空。离线50的learner累计62.182秒，19次共同教师更新；重复输入不是新在线搜索。相同逻辑batch下Spatial micro512→2048成熟耗时1.393→1.403秒，未采纳。精确命令/结果见 architecture-alignment/SUMMARY.json。性能目标仍未达标，不启动正式长训。

**成熟learner剖析与输入通路修复（2026-11-07，已实现）：** 原生第31次更新同时有伤害与CF损失，torch profile实际重算370块/11840行；另一次原生cProfile为7.679秒，其中整ROW LZ4解压1.384秒、from_rows解码/打包1.517秒、forward_chunks自身0.872秒、Reader完整forward0.413秒。已改为PackedRows仅存新输入有效payload（first/schema/bytes/frame），不归档无语义wire尾或旧观测列，按原512行边界流式组装新输入；不展开整池151880字节ROW、不改抽样/历史梯度/优化batch。room/local/map的位域展开由GPU Inductor融合执行，固定无梯度/无autocast解码上下文，输出不使用可覆盖的CUDA graph缓存。旧pickle仍可解码；新历史源无损保留的是模型输入契约，不是旧ROW预留字节。

第一阶段原生第31次更新5.584秒（304块，非相同样本的严格配对）；第二阶段完整GPU回归120通过，CPU98通过/22硬件跳过。新增测试覆盖旧式整ROW展开禁用、所有参数梯度及None掩码、一次AdamW参数/状态一致、旧列/遗留尾扰动不改变有效输入与编码键、GPU解码与输出ownership。两次GPU失败（编译上下文变体超限、非对齐NumPy stride）已修复重测。BASELINE三项失败、MODIFIED三项通过、ROLLBACK三项失败，恢复hash与补丁重放均一致。

<!-- NATIVE_LEARNER_FINAL_GATE -->
第二阶段的压力在线门已主动中断（exit130），不计通过：learner31为4.520秒，但actor32的历史入库3.100秒，保留无语义尾导致重复展开/压缩，故弃用该中间实现。修正为仅存模型有效payload后，本地98通过/22硬件跳过；尚未重新跑GPU验收。此前120项GPU通过属于中间态，不晋升为最终结果。200×仍未达成；3600点/小时配置只作教师功能压力门，不再作正式速度数字。离线job8重复重放不能代表native来源复用；现有save_job也未保存完整teach_chunks/v2_chunks。

用户11-07审核已要求下一轮验收前贯通GPU常驻slot KV/bank/partial、索引快照与tensor rollout；配置默认4个窗外均匀历史块（4/8消融）、纯相对Reader、P补stage/curses及溢出计数、实体优先截断、取消全部use存在性mask、L保留9×9布局。正式速度对照固定同4090/同worker、rollout32768/micro8192、CF6点/180秒/share0.1，先跑旧c45a583再新模型；未完成这些修正前不重启下一轮验收。必须保留immutable fork快照、源/参数版本和Writer因果边界：新完成块下一决策才可读，FIFO满时先读再覆盖，长bank存canonical Writer value而非跨publish复用旧Reader投影。完整8192步长期运行、多GPU扩展和长期学习收益仍未验收。

本轮证据：`D:/Projects/fortune/Isaac/analysis/model-design-20261106/new-input-20261107/performance-fix/architecture-alignment`。四角色保持原路径，新增ALIGNMENT_CODE.patch及ROLLBACK.sh --alignment只扩展原事务，不替换历史记录。

成熟learner补充证据：`D:/Projects/fortune/Isaac/analysis/model-design-20261106/new-input-20261107/performance-fix/native-learner-optimization`；`ROLLBACK.sh --native-learner`恢复到本轮修改之前。

**11-07 Claude 审核落实（组件阶段，不是新一轮速度验收）：**

- B3/B4已接生产调用：默认4个短窗外均匀历史块；Reader不再使用绝对时间正弦，短窗采用可学习决策距离bias，长期memory_age保留；Writer只保留块内位置。任意共同时间平移不改变输出、KV、mask；CPU梯度精确不变，GPU Flex的memory_age原子累积采用独立舍入容差（同输入重复已观察到4.55e-13差异），不套用跨backend BF16宽容差。旧v1权重不做隐式迁移。
- C5–C8已接观测→协议→Spatial：schema为`isaac-visible-prlmens-g-v2`、版本2026110702，P=80维；stage、32位curse、5类溢出计数进入每个P。E优先互动物/出口→怪物→随从→弹幕→其他，类内按距最近身体排序，稳定保留128；R中的门/地形出口本来不受E上限约束。N/S仍完整。取消新输入全部use存在性/充能mask。L以9×9卷积、展平投影保留布局，仍每身体一个token。Unknown健康遮蔽通过非零健康的开/关对照验证。
- 最新本地`pytest tests`：105通过、27硬件/平台跳过；4090组件GPU单测覆盖slot核心、相对偏置/Flex/SDPA及梯度、graph/replay/publish、输入修正。GPU初次失败包含fixture的head维8低于Flex最小16，以及跨backend BF16误差0.00817超过初设0.004；修正fixture并分离backend舍入与同backend平移不变门，保留FP32严格门与原梯度阈值。所有失败及重测保留在review-unit事件与XML，不当作速度提升。
- A1仅完成`BlockSlotState`张量执行核心：GPU bank/KV/partial、索引reset/fork、先读后Writer提交、源版本保存；固定权重CPU逐步/FIFO/fork对照已通过。**尚未替换线上BlockActor/BlockLaneService、不可变分页快照、源租约和learner rollout取批；RawHistory.append与对象reference仍在生产路径。不能称A1完成，也不能用核心单测推断200×。**
- A2正式比较固定单4090、同worker、rollout32768、minibatch8192/micro8192、CF6点/180秒、teacher-share0.1，先旧c45a583后新模型，全随机初始化；双方其余teacher batch/reuse/weight、epochs、aux、critic、起始分布也须显式一致。旧smoke加载C99，旧pred_host还会把8192改成4096，均不能复用。官方X_old(4090)尚未测出。
- D9：原候选已提交并推送到`exp/block-memory-v1`，基底提交`f3264e0a5850892fb97bbed7ea997923d12a301b`；本段新增审核修改单独提交，不改变生产树。D10：`bridge/abplus/EXPERIMENTS.md` B41统一归档全部4090结果、游戏小时、失败及口径，压力×实时数不作为正式A2比较。
- 验收顺序：先完成A1生产集成与状态/梯度等价门，再同口径旧→新对照；组件GPU单测不冒充完整回归、两次冒烟、离线50或在线40稳态。所有未完成项目保留，禁止重启已退出130的旧native压力门。

#### 9.13.7 依据与机器可读定案

- [Perceiver IO](https://arxiv.org/html/2107.14795)：learned queries读取变长输入。四槽／八槽是本项目选型，不是论文最优结论。
- [Compressive Transformer](https://arxiv.org/html/1911.05507)：长期压缩与写入学习信号的区分；本规格的独立块与有限重算不是论文原算法。
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)：块mask与实际跳过无效计算。

<!-- BLOCK_MEMORY_V1_SPEC -->
```json
{
  "design_id": "block_memory_v1",
  "status": "functional_passed_performance_gate_failed",
  "candidate_worktree": "D:/Projects/fortune/Isaac/temp/wt-block-memory-v1",
  "base_provenance": {"worktree": "D:/Projects/fortune/Isaac/temp/wt-predictive-aux", "revision": "c45a583"},
  "spatial": {
    "width": 192,
    "layers": 4,
    "heads": 4,
    "history_feedback": false
  },
  "latents": {
    "per_step": 4,
    "width": 256,
    "readout": "learned_query_cross_attention"
  },
  "writer": {
    "independent_blocks": true,
    "block_steps": 32,
    "memory_tokens": 8,
    "width": 256,
    "layers": 2,
    "heads": 4
  },
  "memory": {
    "blocks": 256,
    "tokens": 2048,
    "eviction": "fifo_by_block",
    "visible": "step_after_block_end",
    "canonical_storage": "writer_latents"
  },
  "temporal": {
    "width": 256,
    "layers": 4,
    "heads": 4,
    "ffn_ratio": 4,
    "window_steps_including_current": 128,
    "window_tokens": 512,
    "cross_attention_layers": [
      2,
      4
    ],
    "within_step_bidirectional": true
  },
  "scaling": [
    {
      "width": 384,
      "layers": 6,
      "heads": 6,
      "cross_attention_layers": [
        3,
        6
      ]
    },
    {
      "width": 512,
      "layers": 8,
      "heads": 8,
      "cross_attention_layers": [
        4,
        8
      ]
    }
  ],
  "training": {
    "decision_chunk_steps": 128,
    "prefix_kv": "stored_detached",
    "cross_boundary_writer_source": "complete_from_raw_observations",
    "historical_recompute_blocks_per_chunk": 4,
    "history_sampling": "uniform_without_replacement_end_step_le_chunk_start_minus_window",
    "history_budget_cli": "--block-history-blocks; ablate 4/8",
    "full_episode_bptt": false,
    "persistent_source_graph": false,
    "new_compression_loss": false
  },
  "teacher": {
    "counterfactual": true,
    "injury_hindsight": true,
    "branches": 4,
    "points_per_worker_game_hour": 6,
    "max_game_seconds": 180,
    "intervention_seconds": 20,
    "winning_prefix_steps": 8,
    "cf_advantage": 0,
    "snapshot_pool_per_actor": 2,
    "taken_endpoint_replay": true,
    "branch_memory_isolation": true,
    "in_flight_action_delay_preserved": true
  },
  "algorithm": {
    "ppo_preserved": true,
    "existing_aux_flags_preserved": true,
    "action_head_sizes": [
      9,
      5,
      2,
      2,
      2
    ],
    "automatic_appo_or_vtrace_migration": false
  },
  "runtime": {
    "environment_workers": 64,
    "actor_gpus": 2,
    "learner_gpus": 1,
    "overlap": true,
    "bounded_policy_lag": true,
    "new_rpc_stack": false
  },
  "initialization": "all_random_zero_step_no_C99_or_derivatives",
  "observation": {
    "schema": "isaac-visible-prlmens-g-v2",
    "version": 2026110702,
    "P": "per body, 80 fields, stage + 32 curse bits + 5 entity-overflow counts",
    "R": 28,
    "L": "one per body, 9x9 convolution + flatten projection; no global average pooling",
    "M": "legal visible map",
    "E": 128,
    "N": "all distinct owner/item/count records",
    "S": "all actual slots including secondary active/battery",
    "G": "learned query",
    "transport_bytes": 131072,
    "overflow": "E priority+distance to128 with category counts; never truncate N/S; reject transport byte overflow",
    "use_action_masks": "none"
  }
}
```

## 文档边界

已实现系统见[架构](ENV_ARCHITECTURE.md)，实验结论见[总账](../bridge/abplus/EXPERIMENTS.md)。本文9.9–9.12保留截至2026-10-24的memory-chain v2/v3候选背景；9.13为2026-11-06新主干最终设计、接续实现与验收状态。新代码只在独立候选工作树中，尚未合入生产；旧图、生产模型、奖励、配置与检查点不变。
