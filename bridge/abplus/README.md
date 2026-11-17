# AB+ 桥与训练入口

更新：2026-11-17。这里是 Afterbirth+ v1.06 原生 Linux 版上的桥（游戏侧 Lua）和实验记录。训练代码在 [../python/](../python/)，加速层在 [../engine/](../engine/README.md)，主机上的运行环境在 [../ops/bm1/](../ops/bm1/README.md)。整体结构见 [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)。

10 月中旬以前的完整说明（约 980 行）归档在 [docs/archive/abplus-README-2026-10.md](../../docs/archive/abplus-README-2026-10.md)，包括：
- 旧的 `train_abplus.py` 房间战斗管线；
- 奖励的演变（combat-v2 到 combat-hp2）；
- 瞄准场、对战场、Go-Explore；
- 桥各版本的字段变化；
- token 策略各开关的长说明。

## 目录

| 文件 | 内容 |
|---|---|
| `abp_bridge.lua` | 游戏侧的桥，现为 abp-0.2.19。各版本的改动写在文件头的注释里 |
| `EXPERIMENTS.md` | 唯一的实验记录。编号：A 桥与环境，B 性能与稳定性，C 奖励与训练 |
| `catalog/` | 从运行中的引擎探测出的房间目录、Boss 池和任务 / 分组文件（见下） |
| `figures/` | 实验配图 |

## 桥怎么和引擎说话

- 加载方式：不走 mods 目录。游戏以 `--luadebug` 启动，游戏副本的 `resources/scripts/main.lua` 用 `dofile(ABP_LUA)` 加载桥。`ABP_LUA` 由 `isaac_bridge/abplus.py` 的 `launch_abplus` 设置，按以下顺序取：
  1. 参数 `bridge_lua=`；
  2. 环境变量 `ABP_BRIDGE_LUA`；
  3. `$ABP_HOME/bridge/abp_bridge.lua`。
- 实例由 `bridge/engine/run_instance.sh` 启动：`$ABP_HOME`（默认 `~/isaac-abplus`）下的 `runtime/` 是自备的游戏文件副本，`tools/` 放从 `bridge/engine/abp_turbo.c` 编出的 `libabp_turbo.so` 和打桩表。游戏文件不在本仓库。
- 协议：TCP，只监听 `127.0.0.1`，每行一个 JSON 命令，例如 `step`、`reset`、`lua`、`obs`、`fork`、`lean`、`play`、`priv`。完整列表在 `abp_bridge.lua` 的文件头。Python 端 `isaac_bridge/abplus.py` 的 `BRIDGE_VERSION` 必须和桥的版本相同，否则拒绝连接。
- 训练和评估走精简模式：
  - 观测是定长二进制记录（`isaac_bridge/abplus_lean.py` 解码），热路径上由 `abp_turbo.c` 原生填写（B9、B11、B12）；
  - 每局是一个 `fork` 出来的克隆体（A19、B10）；
  - 动作晚一个决策生效（C55）。
- 致命伤被桥拦下，记为死亡（AB+ 没有 rewind）。
- 特权帧：`priv` 命令（abp-0.2.19）只发给 learner 的特权 critic，不改精简观测。

## 训练入口 `train_tok.py` 的任务与奖励

在 `bridge/python/` 下运行，`PYTHONPATH=.`。下面的取值核对过 `train_tok.py` 的参数解析。

**模式**（`--mode`）：
- `room`（默认）：单房；
- `floor`：一整层 Basement I，从新开一局的起点打到活板门（C57）；
- `run`：整局，一层接一层，直到死亡或 `--run-seconds`（默认 1800 游戏秒；A21、C67）。

现在的训练都是 `--mode run --items`。

**任务分组**：
- `--groups-file`（必填）和 `--groups`（各组 worker 比例，默认 `normal:8,boss:5,normal_big:3`）；
- 整层 / 整局模式下只有一组。现在的命令行都用 `../abplus/catalog/scaling2_groups.json`。

**奖励**（`--reward name=weight,...` 覆盖默认值）：
- `room` 模式默认 `ROOM_REWARD = damage=1,hurt=0.75,clear=1,death=2,timeout=1,time=0.002`；
- `floor` / `run` 模式默认 `FLOOR_REWARD = damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002`；
- 另有 `heal`（回血，每半心）和 `resource`（每个硬币 / 炸弹 / 钥匙），只在整层 / 整局模式有效，默认 0：
  - C99 用 `heal=0.5,resource=0.25`；
  - main1（C105）用 `heal=0.5,resource=0.05`（[../ops/bm1/TRAINING.md](../ops/bm1/TRAINING.md)）。

**终止与截断**：
- `--floor-stall-seconds`：无进展多久结束一局；
- `--stall-progress events|pickups|items`：什么算进展；
- `--cap-truncate 1`、`--stall-truncate 1`：时限和无进展按截断处理（默认 0 = 终止）；
- `--characters`：开局角色，默认只有以撒（A25）。

**模型**：
- 旧模型（时间 transformer）：`--temporal 1 --memory off`；C99 的完整命令行见 `bridge/ops/bm1/xold_c99.sh`；
- block-memory v1：`--block-memory 1 --temporal 0`；命令行见 `bridge/ops/bm1/xnew2.sh`。这条线 11-17 起不再投入。

**教师**：
- `--teacher`：受伤回溯（C56）；`--teacher-death-depths`：死亡也回溯；
- `--teacher-v2 1`：教师 v2，整条分支蒸馏（B19）；
- `--cf 1`：反事实分支教师（B34，[docs/COUNTERFACTUAL_TEACHER.md](../../docs/COUNTERFACTUAL_TEACHER.md)）。

**辅助**：
- `--aux-predict 1`、`--priv-critic 1`（[docs/PREDICTIVE_MEMORY_DESIGN.md](../../docs/PREDICTIVE_MEMORY_DESIGN.md)）。

**学习器与速度**：
- `--overlap`、`--decoupled 1`（B11、B13）；
- `--learner-device`（B38）、`--learner-fast`（B18）；
- `--amp bf16`；
- 多分片由环境变量 `ISAAC_SHARD_WORLD` 打开（`isaac_bridge/shard_sync.py`）。

**学习率**：`--lr`、`--lr-final`、`--lr-schedule linear|cosine`、`--lr-warmup`。`--resume` 时 `--game-hours` 是再练多少。

参数很多，写在 `train_tok.py` 的 `--help` 里。真实运行用过的完整命令行，以 `bridge/ops/bm1/` 下的启动脚本和 EXPERIMENTS.md 各条的"实验方法"为准。

### catalog/ 里还在用的文件

| 文件 | 用途 |
|---|---|
| `scaling2_groups.json`（引用 `scaling_normal_rooms.json`、`scaling2_normal_big_rooms.json`、`scaling2_boss_rooms.json`） | `train_tok.py` / `eval_tok.py` 的分组文件 |
| `abplus_basement1_rooms.json`、`abplus_basement1_boss_pool.json` | 从引擎探测的 Basement I 房间目录和 Boss 池（A4） |
| `lab_items.json` | 起始构筑可用的道具（`--start-build-*`，A23、B23） |

其余的 `mixture_*`、`goal_*`、`duel_rooms.json`、`parallel_groups.json`、`room_classes.json`、`eval_detour_seeds_tier4.json`、`scaling_groups.json` 属于 9 月的房间战斗和目标条件实验，见归档 README。

## 冒烟测试与训练（bm1 容器）

主机准备（镜像或直接运行、`bm1.env`、游戏副本、venv）见 [../ops/bm1/README.md](../ops/bm1/README.md)。以下在主机的 bm1 目录里运行。

1. 部署一棵代码树：本地 `git archive --format=tar -o <名>.tar <commit> bridge/python bridge/ops bridge/abplus bridge/engine`，上传后：
   ```sh
   ./deploy.sh tree <名> <名>.tar
   ./deploy.sh abphome <名>     # 第一次，或 bridge/engine/abp_turbo.c 变了时
   ```
2. 测试门：`GPU=<卡> ./gate.sh <名>`。rc 0 才算过；rc 9 表示原生组件没编出来，或它的测试被跳过。
3. 冒烟：
   ```sh
   GPU=<卡> UPDATES=2 HOURS=30.0 ./prof.sh smoke-<名> <名>
   ```
   用的是 `xnew2.sh` 的配置，跑完 2 个更新发 SIGINT，检查点移进回收站。要改参数，用 `ENVS="-e EXTRA=--某参数=值"` 传给 `xnew2.sh`（`EXTRA` 到 `EXTRA3` 各放一个参数，见两个脚本的文件头）。
4. 正式训练：`train_guard.sh`。K 个同步分片，失败后自动续训；参数写在运行目录的 `train_args` 里。操作手册的模板是 [../ops/bm1/TRAINING.md](../ops/bm1/TRAINING.md)。

不在容器里也能直接跑：
```sh
cd bridge/python
PYTHONPATH=. python train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items \
  --workers 16 --game-hours <小时> --out <新目录> [其余参数]
```

前提：
- `ABP_HOME` 下有游戏副本和编好的 `libabp_turbo.so`；
- Python 依赖：`bridge/python/requirements-training.txt`，以及与 CUDA 匹配的 torch；
- 从零训练时 `--out` 不能是已存在的目录（EXPERIMENTS.md C99 的启动记录）。

## 评估

```sh
cd bridge/python
PYTHONPATH=. python eval_tok.py --checkpoint <检查点> --groups-file ../abplus/catalog/scaling2_groups.json \
  --mode run --items --workers 8 --seeds 2147490000:128 --out <新目录>
```

- 种子从 2147490000 起，训练时不会抽到。128 个种子用于看曲线，512 个用于定论（A27）。
- 默认按策略概率采样，`--greedy` 取最大。
- 只用于评估的消融开关：`--ablate`、`--no-pedestals`（A28）。
- 旧模型 C99 自己的评估协议见 `bridge/ops/bm1/ev_ab2.sh` 的 `old` 一支。

## 结果放在哪里

- 一次训练的输出目录：
  - `config.json`：全部参数；
  - `progress.csv`：每次更新一行；
  - `episodes.jsonl`：每局一行；
  - `checkpoints/`。
- bm1 下：
  - 计时运行在 `reports/<tag>/`，总日志 `reports/bm1.log`；
  - 正式训练在 train_guard 的运行目录，每个分片一个 `<名>-s<r>/`，外加 `guard.log`。
- 评估输出目录里有 `summary.json`。
- 数字和结论只写进 [EXPERIMENTS.md](EXPERIMENTS.md)：每条实验分"实验目的 / 实验方法 / 实验结果"三段，新实验按同样格式追加。别的文档只引用编号，不抄结果。
