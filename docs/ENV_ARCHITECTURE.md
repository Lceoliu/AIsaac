# 当前模型与训练架构

更新：2026-10-21。本文替代旧模拟器架构稿，并合并 B30 时间注意力与 B34 反事实教师说明。描述 C99 路线的已实现机制；具体宽度、开关和兼容行为以检查点配置与代码为准。新输入方案见[接口规范](PROJECT_SPEC.md)，不把设计图当作已实现能力。

## 1. 环境和数据流

```text
AB+ 原生引擎 / Lua 桥接
  → 可见状态与 pending action → tok_obs.ROW
  → Spatial Transformer → 局部 z_t、全局 g_t
  → Temporal Transformer（另一个 Transformer）→ o_t
  → 动作 / 局部价值 / 全局价值 / 危险辅助头
  → 引擎执行、奖励和终止事件 → rollout → PPO 更新
```

引擎加速不改变玩法逻辑；进程 fork 用来生成同状态的采样起点及训练分支。worker 不依赖 torch，actor 合批推理，learner 做更新；B38 允许 actor/learner 跨 GPU 或进程。部署与运行见[训练管线](../bridge/abplus/README.md)。

动作晚一个决策生效，正在执行的 pending action 是输入的一部分。奖励、动作、log-prob 与观察必须按实际执行时序对齐，不能把刚生成但尚未执行的动作当成已获得奖励。

## 2. 空间输入和读出

C99 典型配置为空间宽 192、4 层、4 heads、FFN 768。玩家、地形/地图、门、实体、道具、全局和历史输入进入注意力；动作和局部价值读玩家侧表征，全局价值读全局表征。

当前已实现输入仍有容量和语义限制：实体截断、库存记录与 token 数量限制、地形类型合并、身份/充能等字段缺失。地上 pickups 属于 Entity 编码，持有物属于库存/槽位编码；两者不能混称“道具 token 已经完整”。

新图中的 Player 拆分、Room 28+1、Entity 128、全部 N 个持有道具，以及 Door 的空间融合是后续设计，不在本次文档整理中修改模型。Blind/未知身份需要所有身份通道一致遮蔽，不能只遮专用 item ID 而留下 subtype 泄露。

## 3. 时间 Transformer 与记忆

典型配置：宽 256、4 层、4 heads、FFN 1024，256 个决策窗口和 128 个房间访问摘要槽。房间槽按访问/转换写入，**不是 128 个唯一房间**；反复进出会消耗访问槽。

- 当前步：`s_t = W_s [z_t; g_t]`；当前步查询历史房间摘要和不晚于自己的窗口 token。
- 房间摘要：上一房间离开时的时间输出经 `W_r` 投影，加访问序号/层号嵌入。房间摘要只读更早摘要，不读之后的决策窗口。
- 每层各有 K/V 缓存；actor 逐步写环形窗口，房间转换时写房间表；新一局重置状态。
- 读出：局部 `z_t + P(o_t)`，全局 `g_t + P_g(o_t)`；可选 GRU 依检查点配置保留。
- 反馈：`Q(o_{t-1}) + kind_history` 作为下一步空间序列中的普通 token；局首 mask。B33 删除兼容门，时间侧从第一步参与。

全局 token 和时间 Transformer 是不同层次：前者整合当前观察，后者访问历史。两者存在不代表已经学会有效规划或“提前记重要信息”。

## 4. 学习器的梯度与陈旧状态边界

学习按同一 worker 的连续块组织，块内使用因果注意力；块前上下文来自 rollout 缓存并停止梯度。反馈所用旧 `o_{t-1}` 同样有停止梯度边界。

房间 writer 不是完全没有梯度：块内 `room_token()` 的投影/嵌入可以训练；但来源 `o_exit` 若来自存储 rollout，就不能把未来损失继续传回产生它的旧编码过程。`room_table` 的块前刷新使用当前参数，但若输入仍是旧隐藏状态，不等于完全重编码。

因此要分别检验：

1. 未来损失是否能更新较早观察的编码与写入，而不只是更新 writer 的最后一层。
2. learner 使用的是旧缓存、用当前 writer 重投影的旧源，还是当前网络重编码的源。

候选实现和机制测试在工作区 `temp/room-memory-write-v1/`。CPU 梯度探针、actor/chunk 一致性、真实训练效果和 GPU 吞吐是不同验收，不互相替代；本次仅整理文档。

## 5. Reward、loss 与参数更新

奖励由真实事件与运行配置计算：伤害、受伤、时间、完成/失败事件，以及启用的清房、探索、Boss、换层、回血和资源项。旧 combat 奖励版本不再逐版抄在使用文档里；权重以训练配置和实现为准。

```text
δ_t = r_t + γ m_t V_{t+1} − V_t
A_t = δ_t + γλ m_t A_{t+1}
R_t = A_t + V_t
ρ_t = exp(log πθ(a_t|o_t) − old_logπ_t)
Lclip = −mean(min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t))
LV = 0.5 mean((V_t − R_t)^2)
L = Lclip + c_v LV − c_e H(π) + c_g LVglobal + enabled_auxiliary_losses
```

普通 PPO 路径如上；重叠/decoupled 路径另处理行为策略与更新起点的比值。真正终止不自举，截断按实现处理；未返回奖励的最后决策不能提前进入 GAE。局部优势驱动 PPO，全局 critic 使用自己的折扣回报目标，不直接替换局部优势。

`backward → gradient-norm clipping → Adam → 同步 actor → 新 rollout`。梯度更新实际连接的编码器、空间/时间网络与头；不会穿过游戏引擎或被 detach 的历史。

## 6. 教师与反事实分支

- 事后教师：受伤/死亡后的恢复点搜索；安全动作概率质量和危险方向头提供辅助信号。
- 教师 v2：闭环分支与多步蒸馏，而不是只教搜索起点的一个动作。
- 反事实教师：在可见选择点比较 taken、目标导向和策略采样分支。原 B34 配方通常 4 支、最长 180 游戏秒；按本次运行参数决定，不作为永久默认。
- 目标分支到目标或超时后交回策略；分数为折扣奖励加截断自举。用复跑估计噪声，超过 margin 与噪声底才记录改进。
- lane 从恢复点复制父 actor 的时间状态，与主 rollout 隔离；结束后回收槽位，不能共享写入主轨迹的记忆。
- 加权模仿与可选 CF advantage 是不同监督路径。分支 return 不自动等于当前策略的 `Vπ` 目标，不能把搜索结果未经区分地当普通 on-policy 回报。

搜索是否被策略内化，需要固定留出恢复点、自主行为和正常评估共同验证；只看搜索成功率或新采样点上的 `safe_mass_new` 不足以归因。

## 7. 实现入口

| 组件 | 代码 |
|---|---|
| 观测与输入 | [tok_obs.py](../bridge/python/isaac_bridge/tok_obs.py) |
| 空间/时间策略 | [tok_policy.py](../bridge/python/isaac_bridge/tok_policy.py) |
| PPO、奖励、优化器 | [train_tok.py](../bridge/python/train_tok.py) |
| 分支教师 | [tok_teacher2.py](../bridge/python/isaac_bridge/tok_teacher2.py) |
| 正式评估 | [eval_tok.py](../bridge/python/eval_tok.py) |

旧 `TEMPORAL_ATTENTION.md`、`COUNTERFACTUAL_TEACHER.md` 已并入本文；阶段曲线与速度放在[实验总账](../bridge/abplus/EXPERIMENTS.md)，不在架构文档追加日报。
