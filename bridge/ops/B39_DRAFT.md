# B39 草稿：开局去掉主动道具（no-active starts）

（草稿，供并入 `bridge/abplus/EXPERIMENTS.md`；分支 `b39-no-active`，提交 4fa28c4 + f4df295，沙盒 HPC `ws15`。）

## 实验目的

A29 发现：AB+ 上以撒每局开局都拿着 The D6（主动道具 105，6/6 充能），C99 训练里从没见过“开局没有主动道具”的输入；评估时只在开局拿掉 D6、其余不变，C99 在 128 个种子上从 1.29 层掉到 0.25 层（第 1 层无进展超时 20 → 103 局，A-B-A 来回穿门）。B39 给训练加一个开局随机化：floor 开局的训练局以概率 p 拿掉玩家 0 的主动道具；评估加一个确定性的开关，每局开局都拿掉。用它训练 C102（C99 segment2-final + `--no-active-prob 0.5`），并在两条评估线上看：普通 128 种子、无 D6 128 种子。

## 实验方法

**机制**（`isaac_bridge/tok_pickup.py`、`tok_floor.py`）

- `NO_ACTIVE_LUA`：通过桥的 `lua` 命令在本局的复制体里执行 `p:RemoveCollectible(p:GetActiveItem())`（角色开局拿什么就拿掉什么；以撒 = D6；没有主动道具的角色什么也不做），返回 `拿掉的 id|之后的主动道具|该 id 之后的持有数`。`parse_no_active` 检查结果：之后仍有主动道具或持有数不为 0 就抛 `ValueError`，本局开局按失败处理（计入 errors，和其他开局失败一样）。
- 抽样位置：`floor_worker_main` 里本局 `src.fork(reseed=R)` 之后、开局道具（start build）之前、第一条观测之前。训练：只对 floor 开局（不对 archive 开局，archive 条目沿用其原局的状态，和 start build、stat aug 同一规则），用自己的随机数流 `default_rng([seed, worker, 41])`，不改变其他抽样；评估（`cfg.no_active`）：每局都拿掉。放在开局道具之前，所以开局道具若含主动道具则保留；B23 的 pedestal 开局道具照常生成。
- 观测：物品块的重扫描键里有 `GetActiveItem` 和 `GetCollectibleCount`，第一条观测（一个连接的第一条，非 step）总是完整扫描，所以第一条 ROW 就是 `player[21]` has_active 0、`player[22]` 0、`pitem[0]` 0、`pinv[0..1]` 0、没有持有道具 token。
- 计数：worker stats `no_actives`（拿掉过的开局数）、`no_active_removed`（其中确实持有主动道具的）。

**搜索的复原点（发现并修正的问题）**

- 起始房间的复原点原来是 `src.hold()` + 本局的 reseed，即“未改动的开局”。死亡教师、teacher v2、反事实教师（cf）在起始房间的点都从这里复原后重放动作——会把 D6 加回来；同一问题已存在于 C99：pedestal 开局道具（50% 的局）复原后没有 pedestal，stat aug 复原后属性回到原值。C99 的 cf 在起始房间的 pedestal 触发点恰好落在这种局里。
- 修正：开局被改动过（no-active、开局道具、stat aug）且开着搜索时，本局在第一条观测后 fork 一份停放（`Parked(episode.fork(tag='snap'))`，reseed None，和换房间时的停放相同），作为起始房间的复原点；未改动的开局照旧用 src。
- `ISAAC_RL_START_BASE_LEGACY=1`（`hpc_c102.sh` 的 `START_BASE_LEGACY=1`）：开局道具 / stat aug 的局保持 B39 之前的复原点（与 C99 完全一致），no-active 的局总是用本局自己的复原点。默认 0（修正）。

**训练 / 评估开关**（`train_tok.py`、`eval_tok.py`）

- `train_tok --no-active-prob p`（默认 0；只限 floor / run 模式，p 在 [0, 1]）。progress.csv：`no_actives`、`no_active_removed`（worker 累计），`noact/*` 和 `act/*`（本次更新结束的局按第一条记录的 has_active 分组：局数、exits、stage、timeout；archive 开局按其实际状态算）；episodes.jsonl 每局 `active0`。
- `eval_tok --no-active 1`（只限 floor / run 模式）：summary.json 带 `no_active: true`、`no_active_starts`、`no_active_removed`。
- 两个开关都关闭时行为不变（默认的 TokSamplerConfig、无改动的开局走原路径）。注意：开着 start build / stat aug 且开着搜索的配置（例如 C99 的配方）即使不开 no-active，起始房间的复原点也变了（上面的修正）；要逐字节复现旧行为用 `ISAAC_RL_START_BASE_LEGACY=1`。

**C102 启动脚本**（`bridge/ops/hpc_c102.sh`，沙盒 `ws15`：`mk_ws15.sh` 从 ws13 复制引擎工具，Xvfb `:95`）

- COMMON / FAST 与 `hpc_c99.sh` 相同；`--no-active-prob 0.5`；B38 四卡布局（`--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2`，CUDA_VISIBLE_DEVICES=0,1,2,3，watcher 用 3）；`MB`（默认 4096，C101 若支持 8192 则 `MB=8192`）；`--resume /101063/AIsaac/ws9/runs/c99-scratch/checkpoints/segment2-final.pt --game-hours 10000 --out $R/c102-noactive`，name hc102，端口 49200（训练）/ 49450、49470（watch 两条线）。
- `watch`：每个检查点同时跑两条线（各 8 workers）：普通 → `evals/<tag>` + `evals.jsonl`；`--no-active 1` → `evals-noactive/<tag>` + `evals-noactive.jsonl`。
- `smoke`：8 workers、1.5 h、从 segment2-final 续训、两卡进程布局（`SMOKE_LAYOUT`），打印 progress.csv 的 no-active 列和 episodes.jsonl 的 active0 分布。
- `probe`：`abplus_probe_no_active.py`（角色 0 / 3 / 5，两个种子）：有 / 无主动道具的第一条 ROW；新复原点（本局停放后 reseed None 再 fork）重放同一动作序列的逐条 digest 必须与本局相同；旧复原点（src + reseed）的第一条 ROW 与 digest；no-active + pedestal 开局道具同时存在。
- `baseline`：segment2-final.pt 在两条线上各 128 种子、16 workers 并行 → `$R/c102-baseline`。

## 实验结果

**CPU 测试**（`tests/test_no_active_smoke.py`，本地 torch 2.12 CPU 与 HPC `/opt/aisaac-env` 均通过；同时重跑 `test_eval_ablate_smoke`、`test_pickup_smoke`、`test_cf_smoke`、`test_obs_v2_smoke`，均通过）：Lua 片段和应答解析；`no_active_start` 在假复制体上计数正确（拿掉 1 次 + 无主动道具 1 次 → no_actives 2、no_active_removed 1，失败时不计数并抛错）；默认配置关闭、STATS 中 A28 两列仍在最后；tok_floor 中的顺序（fork → no-active → 开局道具 → stat aug → 第一条观测 → 起始复原点）；`train_tok.main` 到构建配置为止 `--no-active-prob 0.5` 传到 TokSamplerConfig（默认 0.0），room 模式和 [0, 1] 之外拒绝；`eval_tok --no-active 1` 传到 `no_active=True`，room 模式拒绝；`Episodes` 的 active0。

**引擎探针、8-worker smoke、segment2-final 两条线基线**：待 host B 的 C100 结束后运行（`hpc_c102.sh probe`、`smoke`、`baseline`），结果待补。参照：A29 有 D6 时第一条 ROW 为 has_active 1、最大充能 0.5、1 个持有 token；A29 无 D6 的 128 种子为 0.25 层（segment1-final）。
