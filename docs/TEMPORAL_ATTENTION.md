# B30 时间注意力：设计（2026-10-14）

图：`docs/figures/temporal_attention_arch.svg`。用户决定（10-14）：反馈路开、时间侧 256 宽 × 4 层、窗口 W = 256 步、GRU 先保留、
参数量与空间侧同量级；不做记忆消融；现有的每步空间 transformer 原样保留。目标：在 C95 的检查点上续训时**加上去的瞬间逐位相同**
（所有门零起步），训练速度基本不降（B29 的结论：actor 的开销是 pad 的 token 和小 kernel，时间侧每步只有一个查询，就地读缓存）。

## 1. 模块

记号：空间侧宽 d=192；时间侧宽 D=256，L=4 层，H=4 头，FFN 1024，pre-LN；每步 t 一条记录。

- **压缩** `s_t = W_s · [z_t ; yg_t]`（2d → D）。z_t = 玩家 token 的 norm 输出（现在动作头的输入），yg_t = 全局 token 输出。
- **步 token 序列**：最近 W=256 步的 s（含当前步）。相对位置 = 步距，按头学一张 `[H, W+1]` 的偏置表（ALiBi 形式但可学）。
- **房间 token** `r_k = W_r · o_{t_exit(k)} + E_room[k] + E_stage[stage]`（k = 本局第 k 个离开的房间，≤128）：离开房间那一步的时间侧输出
  o，加房间序号和层号嵌入。房间 token 只对**更早的房间 token** 做注意力（不看步 token），因此它每层的 K/V 在创建时算一次就能缓存；
  步 token 看"所有已创建的房间 token + 窗口内 ≤ t 的步 token"。因果 mask：步 t 只看 ≤ t。
- **时间 transformer**：L 层，每层 = 注意力（查询 = 当前步 token；键/值 = 房间 token + 窗口步 token，各层各自的 K/V）+ FFN。输出 o_t（D 维）。
- **读出路**：动作头/价值头/危险头的输入 = `z_t + mem_out(h_t) + g1 · P(o_t)`（P: D → d，g1 标量零起步）。`value_global` 的输入 =
  `norm(yg_t) + g3 · P_g(o_t)`（g3 零起步）。GRU 不变（输入仍是 z_t + mem_in(norm(yg_t))）。
- **反馈路**：`u_t = g2 · Q(o_{t−1})`（Q: D → d，g2 零起步）作为一个额外的"历史 token"（kind 嵌入 = 历史）进入第 t 步的空间注意力
  （融合布局的普通 token；t 为局首时为零向量且 mask 掉）。
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
- 反馈路的历史 token 在学习器里用 **rollout 存下的 o_{t−1}**（stop-grad）：否则空间前向要沿块串行 256 次。梯度通过 g2、Q 和空间
  注意力流动，不通过 o_{t−1} 的生产者。这是 v1 的明确近似，记在文档里，之后再评估。
- 房间 token 在学习器里用存下的（stop-grad，除了 E_room/E_stage/W_r 以外没有梯度 → v1 中 W_r 也只通过"块内创建的房间 token"学：
  块内若发生离开房间，该房间 token 用块内算出的 o 重算，有梯度）。
- 分块 BPTT 对 GRU 照旧（32 步一块嵌在 256 步的块里）。
- **B32（10-15）不带 GRU 也能跑**（C98 消融）：学习器只按 `--temporal-bptt` 切段（不嵌 GRU 段、不补齐），时间侧不变；`--resume` 带 GRU 的检查点并显式 `--memory off` 时丢掉 `mem_gru/mem_out/gtok.mem_in`（`drop_memory`，其余逐位不变，输出 = 原模型去掉 mem_out(h)）；测试 `tests/test_no_gru_smoke.py`。
- **B31（10-15，用户批准）去掉三处多余的拷贝 / 重算，表达能力不变**：(1) rollout 的每步 K/V、s、o 和房间来源留在 GPU（`Rollout`
  的设备张量池，actor 的 step 输出在设备上 `index_copy_` 写进去，学习器在设备上 gather；每步只有计数过总线：写入行号 8 B 进、
  create/nroom/steps/stage_prev 16 B 出）；(2) 历史 token 是融合注意力里的**普通 token**（[…, 道具, 历史, 全局]，一次 softmax），
  g2 乘在它的 value 上（同 B24 的 vscale），无上一步时 mask 掉它的 key（从不按门值 mask：g2 = 0 时梯度非零）；代价是门为零时
  输出不再与关闭逐位相同（这把 key 占一份 softmax、value 为零，测试里 logit 偏差约 2–3%），Q 与历史 kind 在门为零时经 key 也有梯度；
  (3) 块前房间 token 每个 PPO epoch 开始时对整个 rollout 算一次（`room_table`，stop-grad，当前参数），各块按索引取。

## 4. 标志与检查点

`--temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256`。
关闭时与 HEAD 逐位相同；打开且门为零时与关闭逐位相同（测试）。从 C95 检查点 `--resume`：新参数按名字追加，Adam 状态 remap；
`convert` 打印一行。eval_tok 支持（CPU 上每 worker 的状态）。summary.json 记 `temporal`。

## 5. 验证

- CPU 测试 `tests/test_temporal_smoke.py`：关闭 == HEAD；门零 == 关闭；环形缓存逐步前向 == 整段一次性前向（同一序列）；房间 token
  创建/mask；块训练前向/反向（含 GRU 分块）；resume 转换加载；eval 路径。
- HPC 冒烟：8 worker 1.5 小时，看 `tracebacks`、KL（应与 C95 同量级）、`infer_ms`、`update_s`。
- C96：HPC160，C95 最终续训，64 worker，12,000 小时，每 1,000 次更新 128 种子；对照 = C95 的曲线与最终。看：速度（目标 ≥ C95 的
  1,100×）、过层、第四层局数、拿道具数，以及 g1/g2/g3 的走势（门打开多少 = 时间侧被用了多少）。
