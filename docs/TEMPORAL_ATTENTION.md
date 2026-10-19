# B30 时间注意力：设计（2026-10-14）

图：`docs/figures/temporal_attention_arch.svg`。用户决定（10-14）：反馈路开、时间侧 256 宽 × 4 层、窗口 W = 256 步、GRU 先保留、
参数量与空间侧同量级；不做记忆消融；现有的每步空间 transformer 原样保留。训练速度基本不降（B29 的结论：actor 的开销是 pad 的
token 和小 kernel，时间侧每步只有一个查询，就地读缓存）。

**B33（2026-10-16，用户）：没有门。** B30 / B31 的门 g1 / g2 / g3（为了"在 C95 检查点上加上去的瞬间输出不变"）删除：时间侧是网络的
普通部分，P、P_g、Q 用默认初始化，从第一步起就参与；GRU 默认关（`--memory gru` 仍可用）。带门的检查点（C96、C98）用 commit
b0a89b9 的沙箱。下面的公式已按 B33 改写。

## 1. 模块

记号：空间侧宽 d=192；时间侧宽 D=256，L=4 层，H=4 头，FFN 1024，pre-LN；每步 t 一条记录。

- **压缩** `s_t = W_s · [z_t ; yg_t]`（2d → D）。z_t = 玩家 token 的 norm 输出（现在动作头的输入），yg_t = 全局 token 输出。
- **步 token 序列**：最近 W=256 步的 s（含当前步）。相对位置 = 步距，按头学一张 `[H, W+1]` 的偏置表（ALiBi 形式但可学）。
- **房间 token** `r_k = W_r · o_{t_exit(k)} + E_room[k] + E_stage[stage]`（k = 本局第 k 个离开的房间，≤128）：离开房间那一步的时间侧输出
  o，加房间序号和层号嵌入。房间 token 只对**更早的房间 token** 做注意力（不看步 token），因此它每层的 K/V 在创建时算一次就能缓存；
  步 token 看"所有已创建的房间 token + 窗口内 ≤ t 的步 token"。因果 mask：步 t 只看 ≤ t。
- **时间 transformer**：L 层，每层 = 注意力（查询 = 当前步 token；键/值 = 房间 token + 窗口步 token，各层各自的 K/V）+ FFN。输出 o_t（D 维）。
- **读出路**：动作头/价值头/危险头的输入 = `z_t (+ mem_out(h_t)) + P(o_t)`（P: D → d）。`value_global` 的输入 =
  `norm(yg_t) (+ mem_out(h_t)) + P_g(o_t)`。有 GRU 时它的输入仍是 z_t + mem_in(norm(yg_t))。
- **反馈路**：`u_t = Q(o_{t−1}) + kind_历史`（Q: D → d）作为一个额外的"历史 token"进入第 t 步的空间注意力
  （融合布局的普通 token；t 为局首时 mask 掉它的 key）。
- 参数量：4 层 × (4·D² + 2·D·1024) ≈ 3.1M，加投影/嵌入 ≈ 3.3M（空间侧 3.7M）。

## 2. actor（GraphActor，CUDA graph，静态形状）

每个 worker 槽位在 GPU 上的静态状态：
- 步 token 的每层 K/V 环形缓存 `[slots, L, W, 2, D]`（bf16 存储，计算 fp32/TF32），写指针；
- 房间 token 的每层 K/V 表 `[slots, L, 128, 2, D]` 与计数；
- 上一步的 o（反馈路输入）`[slots, D]`；
- 局首重置（和 GRU 状态表一样：reset 时清零/置计数 0）。
每次调用：空间前向（含历史 token）→ s_t → 逐层：写入本步 K/V 到环（就地）、1 个查询对 (≤128 + 256) 个键的注意力（mask 掉未写入的槽位与
未创建的房间）、FFN → o_t → 读出。离开房间（记录里的 EV_NEW_ROOM / EV_EXIT 事件，或房间序号变化）时，用上一房间最后一步的 o 创建房间
token 并算它的每层 K/V 写进表。**禁止**每次调用把窗口 K/V 拷出来拼接（B29：拷贝会占主导）；注意力直接对缓存张量做带 mask 的
SDPA。可与 B29 的 `--actor-compile/--actor-fold/--actor-row-buckets` 共存。

## 3. 学习器（分块，XL 式上下文不回传）

- rollout 每步多存：s_t（D）、o_t（D，反馈路与房间 token 用）、每层的 K/V（L×2×D，bf16）、房间 token 表的快照索引（每步"已创建到第几个
  房间"）、房间 token 本身（每局 ≤128 × D，按 worker 存一份，随步引用）。
- 块 = 同一 worker 连续 256 步（`--temporal-bptt 256`，与现在的记忆分块机制同构：minibatch 由整块组成）。块内：空间前向（并行）→ s →
  时间 transformer 以块内因果注意力（SDPA）+ **块前 256 步的上下文** 算 o；上下文 = rollout 里存下的 actor 时的每层 K/V（stop-grad，
  最多一次更新的陈旧度），v1 不做块与块之间的顺序依赖，任何 minibatch 组合都行。
- 反馈路的历史 token 在学习器里用 **rollout 存下的 o_{t−1}**（stop-grad）：否则空间前向要沿块串行 256 次。梯度通过 Q 和空间
  注意力流动（B33 前还经 g2），不通过 o_{t−1} 的生产者。这是 v1 的明确近似，记在文档里，之后再评估。
- 房间 token 在学习器里用存下的（stop-grad，除了 E_room/E_stage/W_r 以外没有梯度 → v1 中 W_r 也只通过"块内创建的房间 token"学：
  块内若发生离开房间，该房间 token 用块内算出的 o 重算，有梯度）。
- 分块 BPTT 对 GRU 照旧（32 步一块嵌在 256 步的块里）。
- **B32（10-15）不带 GRU 也能跑**（C98 消融）：学习器只按 `--temporal-bptt` 切段（不嵌 GRU 段、不补齐），时间侧不变；`--resume` 带 GRU 的检查点并显式 `--memory off` 时丢掉 `mem_gru/mem_out/gtok.mem_in`（`drop_memory`，其余逐位不变，输出 = 原模型去掉 mem_out(h)）；测试 `tests/test_no_gru_smoke.py`。
- **B31（10-15，用户批准）去掉三处多余的拷贝 / 重算，表达能力不变**：(1) rollout 的每步 K/V、s、o 和房间来源留在 GPU（`Rollout`
  的设备张量池，actor 的 step 输出在设备上 `index_copy_` 写进去，学习器在设备上 gather；每步只有计数过总线：写入行号 8 B 进、
  create/nroom/steps/stage_prev 16 B 出）；(2) 历史 token 是融合注意力里的**普通 token**（[…, 道具, 历史, 全局]，一次 softmax），
  无上一步时 mask 掉它的 key（B31 时 g2 乘在它的 value 上，B33 删除）；
  (3) 块前房间 token 每个 PPO epoch 开始时对整个 rollout 算一次（`room_table`，stop-grad，当前参数），各块按索引取。

## 4. 标志与检查点

`--temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256`。
关闭时与 HEAD 逐位相同（测试）。从没有时间侧的检查点 `--resume`：新参数按名字追加（输出从第一步起改变），Adam 状态 remap，
打印一行。eval_tok 支持（CPU 上每 worker 的状态）。summary.json 记 `temporal`。

## 5. 验证

- CPU 测试 `tests/test_temporal_smoke.py`：关闭 == HEAD；`temporal=None` == 关闭、新建模型第一步就与关闭不同；环形缓存逐步前向 == 整段一次性前向（同一序列）；房间 token
  创建/mask；块训练前向/反向（含 GRU 分块）；resume 转换加载；eval 路径。
- HPC 冒烟：8 worker 1.5 小时，看 `tracebacks`、KL（应与 C95 同量级）、`infer_ms`、`update_s`。
- C96：HPC160，C95 最终续训，64 worker，12,000 小时，每 1,000 次更新 128 种子；对照 = C95 的曲线与最终。看：速度（目标 ≥ C95 的
  1,100×）、过层、第四层局数、拿道具数（C96 时还看 g1/g2/g3 的走势；B33 起没有门）。

## 6. 2026-10-20：第 1 项，原始房间序列写入

**候选分支 `codex/room-memory-write-v1`，默认关闭，`--room-memory 1` 开启。**
现有 Spatial / Temporal 尺寸、1 token/visit、128 visit 上限、H 回灌、决策 KV、动作空间不变。
第 2 项（整个策略旧状态的一致性）仍未改动。

### 生产路径

- `RawRoomState` 是 `TemporalState` 的 CPU 部分：保存正在访问房间的只读 ROW/pending。
  房间变化先关闭旧访问，再加入新房间第一帧；reset 丢弃上一局开放访问。
  访问关闭后原始数组归 rollout 所有，跨 rollout carry 保留当前局，旧局在 rollout 释放时释放；
  达到原 128 次写入上限后停止继续积累不可能使用的源。没有抽帧或重要事件规则。
- `RoomSequenceWriter` 以一个学习查询、时间编码和 PyTorch MultiheadAttention 池化整次访问。
  `encode_room_sources` 使用当前 Spatial 权重及 z/g 投影，不读 H 或旧 o_exit；主策略 H 不变。
- GraphActor 在 graph 外编码刚关闭的访问，写入固定形状 room_source 输入；graph 内原 Temporal
  写入、sampling、fold/row buckets/compile 保留。eager/eval 使用 `TemporalState.step_inputs`。
- 完整原始访问在 `Rollout.raw_rooms` 中，随 `rollout_scalars` 经过现有 IPC；两 actor 合并时按
  worker 顺序合并，跨 rollout 与原 rooms_* 元数据同序。CF bank/lanes 的 slot copy/reset 同步复制
  或清除原始开放访问，分支 append 不污染父状态。teacher 已有 stored-output loss 语义不变。
- learner 在每个 Adam step 开始，用当前权重重算所需房间的因果前缀。`RoomSourceBatch`
  在该 step 内缓存源及其梯度，各 PPO micro-part 重建有梯度的房间表；最后重放源编码，应用累计
  VJP，再执行 gradient clipping/Adam。**缓存不跨权重更新，不把 source detach 当作最终梯度。**
  `--room-source-micro 2048` 限制 Spatial 编码批量，长房间不截断；Spatial 用 non-reentrant checkpoint。
- checkpoint 保存 room_memory；原 checkpoint resume 会按参数名保留 Adam 状态并新建 writer 状态；
  新 checkpoint resume 自动保持 writer。禁止把 writer 权重静默加载到 legacy 模型。

采用已有[Gradient Cache 思路](https://aclanthology.org/2021.repl4nlp-1.31/)和
[PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint)，不添加新的 RL objective。

### 正确性证据

`test_room_pipeline_smoke.py` 运行真实 Rollout/store/carry、temporal_inputs/part 和模型：
- 两个 rollout actor/chunk logits 最大差 2.39e-7；跨边界访问没有丢前半段。
- IPC pickle、双 actor merge、CF snapshot/reset/独立 append、LaneService、eval dispatch 通过。
- 完整计算图与分两段累计源梯度的参数梯度最大差 CPU 3.06e-5（同时启用完整库存后 6.11e-5）；writer 与 Spatial 均有梯度；
  Adam 后源重新计算发生变化。
- 新主机 GPU GraphActor fold=False/True 均通过固定 uniforms 动作和值对齐；fold=True 值误差 1.50e-7；同时启用完整库存后两种 fold 最大误差 2.09e-7。
- 旧 no-GRU、temporal、lane-batch、CF CPU smoke 全通过。

先前 `test_room_memory_smoke.py` 的因果/早帧梯度/严格权重往返继续通过。
最小中性背景延迟线索任务 100 步后 held-out 128 样本准确率 100%，移除早期线索 50%；
先前每批随机完整背景的困难版本 100 步仍为 50%，未通过。二者均不是游戏评估。

### 吞吐与验收

新训练主机 dgene / 10.19.136.184:22，3 张 3090、64 workers、双 actor + 独立 learner，
minibatch/micro 8192、rollout 32768、3 epochs，teacher/CF 等原 full-flags 不变。
所有测量为同机受限低优先级容器；外部任务不停止，外部负载/可用内存 guard 保留。

|版本|结果|实际吞吐 / learner耗时|
|---|---|---|
|原始对照|50 updates 正常退出，稳态取后40次|729.714×，median update 4.570s|
|dense raw，source micro256|2 updates 后因成本主动停掉自己的容器|43.73/39.71×；47.29/104.94s|
|packed + CPU grid去重，source2048|1 update 后停止，改动已撤销|30.42×；83.70s|
|dense raw，source2048|2 updates，正常 exit0|51.55/91.98×；20.94/44.23s|

**接入正确性成立，但速度不合格，保持 opt-in，不启动大训练。** 最后一组只有两次更新，
不能称为稳态测速；游戏/CF errors 均0，但 CF completed=0（短测期间搜索48，teacher held94），
CF 完成分支由 LaneService 测试覆盖，不能把这个短测说成所有辅助 loss 已进入稳态。
完整训练成绩/512 seed eval 未做。原始房间帧重复经过 Spatial、每 Adam step 重编码是当前成本来源；
不以固定抽帧/重要事件规则掩盖成本，后续需单独设计保持学习信号的高效写入。
证据与四角色文件沿用 `D:/Projects/fortune/Isaac/temp/room-memory-write-v1/`。


## 7. 2026-10-20：完整持有道具 ID / 数量输入

`--inventory-all 1`，对应 checkpoint config `obs_inventory_all=True`，默认关闭；旧 checkpoint
可以显式开启而不新增参数，之后 resume/eval 自动沿用。必须搭配 items。与 room-memory 独立开关。

- Lua 旧协议最多64种、ROW最多16种、旧池化忽略重复数量。新 inventory layout2 使用 uint16
  条目数，支持现有 vocab 的 ID1..1023；物品数量仍沿用 wire uint8 上限255。
- ROW 保留原20800字节及所有已有字段偏移，追加1024个int16数量和完整标志，共22856字节
  （增加2056字节/9.88%）。原尾部padding显式命名，防止 NumPy concatenate 在双 actor
  teacher合并时重排布局。旧序列化ROW需显式迁移，不能当成完整库存使用。
- Lua→LeanDecoder→EpisodeState→Python/native FastRow后处理→rollout/IPC→GPU decode均贯通。
  请求完整库存而加载旧Lua时明确报错，不静默使用被截断数据。native C前缀ABI不变。
- 原道具embedding复用，按全部ID及数量求和进入玩家分支；没有手写重要性排序、没有新增reward。
  独立道具token仍8个，原recent16顺序保留。**这不是全部道具信息修复**：其他主动/口袋槽、
  主动充能语义、饰品变体、多身体等未在本轮混改。
- `test_inventory_all_smoke.py` 覆盖400种ID（跨64/255边界）、数量/移除、v1兼容及拒绝降级、
  far ID399影响输出并收到梯度、旧分支不变、byte decoder、checkpoint、CUDA Graph。
- 两种特性同时开启的生产Rollout/IPC/CF/gradient-cache/GPU Graph测试通过。
- 实际游戏2 seeds、69条记录（63 native fast /6 slow）Python/native逐字节一致；
  GPU inventory graph最大值差2.24e-8。首次双actor实测因ROW padding重排失败（exit143由guard
  清理），对应失败测试已复现并修复；失败日志保留，不算成功测速。

**最终 r3 验收：50 updates、正常 exit0、guard未触发，日志无traceback/强制终止。**
丢弃前10次后40次稳态 **729.874×**，原始对照同样40次 **729.714×**，本次基本持平。
median learner update4.605s（原4.570s），GPU reserved15.469GiB（原15.238GiB）；
errors/stalls/CF errors均0，CF completed271、teacher held2118。单次短测、轨迹不同，不作统计显著性结论。
含启动/编译/清理总391.1s。先前 r2 的49 updates719.353×和退出异常保留在证据账本，不替代最终验收。
修复后native/Python重测69条仍逐字节一致。实际 room-memory 与两次inventory训练检查点已复制到本地，
严格读回模型及Adam状态；未做新检查点继续训练的独立实测，不将读回等同于恢复训练验收。

### 双 actor 退出竞态（由实测触发，非额外防御脚手架）

r2 的第二 actor 在 Queue.put 后立即设置 last-half event；但 put 只是放进本进程 feeder 缓冲，
父 actor 的结束 sentinel 可能先于另一 producer 的最后一份数据到达 learner。结果最后一份数据
未排空，进程退出卡在 Queue feeder join，最终 no-progress guard 清理；本次不能声称无异常退出。
在正常结束分支，先 Queue.close()/join_thread() 排空本 producer，再通知 last-half。没有取消
join/静默丢弃数据，也没有改训练算法。采用 [Python Queue 排空接口](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.join_thread)。
`test_actor_shutdown_smoke.py` 从实际 trainer AST 取出结束分支，用4MiB饱和队列复现：
旧代码 last 早于payload投递（exit1）；修复后先投递再通知，子进程干净退出（exit0）。
**已完成：** 同配置 r3 50 updates 干净退出，guard无停止原因，确认修复后的真实收尾。
