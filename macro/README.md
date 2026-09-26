# 宏观层离线工具：AB+ 楼层生成器、隐藏房推断、道具先验、规则规划器（2026-09-25）

对应 [MACRO_PLANNING.md](../docs/MACRO_PLANNING.md) 的 P0。全部离线运行，不启动游戏。

**核心**：把 Afterbirth+ v1.06（Linux x64，带函数名）的楼层生成代码逐函数翻译成 Python。
- 给定种子，得到与引擎相同的楼层：网格、房间形状、房间类型和布局、Boss、诅咒，以及每个房间的三个种子。
- 与引擎逐种子、逐房间一致：9,774 个地下室种子的引擎日志，以及 6,000 个引擎整层导出（第 1–8、10、11 层）。范围和局限见下文"验证"。
- 有了它，就能离线生成任意多的楼层当训练数据，也能在已知生成规则下精确推断隐藏房。

**信息边界**（PROJECT_SPEC §3）
- 种子和生成器只用于离线造数据和评估。
- 送给策略的输入只能是玩家看得见的地图，例如 `Floor.visible()` 或规划器里的 `explored_map()`。

## 组成

| 文件 | 作用 | 对应的引擎函数（AB+ RVA） |
|---|---|---|
| [`archive.py`](isaac_macro/archive.py) | 读取 `.a` 资源档案（路径哈希、MiniZ 分块、校验和） | 引擎的档案读取代码 |
| [`rng.py`](isaac_macro/rng.py) | xorshift RNG（81 组移位参数）、`Seeds`（13 个楼层种子）、种子字符串 `XXXX XXXX` 的互转 | `RNG::*` 0x42A180 起，`Seeds::SetStartSeed` 0x463630，`Seed2String`/`String2Seed` 0x4633C0/0x4634E0 |
| [`roomconfig.py`](isaac_macro/roomconfig.py) | STB1 房间布局库和房间查询（按 float32 权重抽取） | `RoomConfig::LoadStageBinary`、`GetRandomRoom` 0x453260 等 |
| [`levelgen.py`](isaac_macro/levelgen.py) | 网格生成：房间扩张、死路、Boss/特殊房间改形、隐藏房选址 | `LevelGenerator::*`（33 个函数） |
| [`level.py`](isaac_macro/level.py) | 楼层：诅咒、房间数、重试循环、特殊房间放置顺序、Boss 选择、每个房间的三个种子 | `Level::Init` 0x33E570、`generate_dungeon`、`place_rooms` 0x337740、`choose_boss` 0x330DA0 |
| [`run.py`](isaac_macro/run.py) | 整局：从开局到后续各层（楼层类型、迷宫诅咒跳层、大教堂/阴间） | `Game::Start` 0x2E5800、`Game::StartDebug` 0x2DFDD0、`Level::SetNextStage` 0x332580 |
| [`floor.py`](isaac_macro/floor.py) | 抽象楼层：房间图、门、BFS 距离、玩家可见的地图、文字地图 | — |
| [`secret.py`](isaac_macro/secret.py) | 隐藏房的精确后验；超级隐藏房的候选格和权重 | 规则来自 `GetNewSecretRoom` 0x344D90、`build_secret_room_index_blacklist` 0x337210 |
| [`planner.py`](isaac_macro/planner.py) | 规则规划器 v0：探索、炸墙找隐藏房、去 Boss 房（不含战斗） | — |
| [`items.py`](isaac_macro/items.py) | 道具先验表：AB+ 的道具定义和 26 个道具池，品质和标签借用忏悔+ | — |
| [`dataset.py`](isaac_macro/dataset.py) | 按整局生成楼层，导出 JSONL 记录和 13×13 网格 | — |
| [`enginelog.py`](isaac_macro/enginelog.py) | 解析引擎 `log.txt` 里的楼层生成记录 | — |
| [`tools/`](tools/) | `export_floors.py`（多进程导出数据集）、`eval_secret.py`、`eval_planner.py`、`fit_super_secret.py`、`compare_engine_floors.py`（对照引擎导出） | — |
| [`tests/`](tests/) | pytest，33 项，约 10–25 秒；缺少档案时自动跳过；含 454 层引擎日志和 184 层引擎整层导出两份回归数据 | — |

**数据来源**
- 房间库只需要 AB+ 的 `afterbirthp.a`。
  - 这一个档案就包含 `stages.xml` 和第 0–24 号楼层的全部房间文件，并且在引擎里会覆盖 `rooms.a`、`afterbirth.a` 中的同名文件。已逐房间核对：只用它和三个档案叠加的结果相同。
  - 默认路径是 `analysis/abplus-linux/resources/packed/afterbirthp.a`，可以用环境变量 `ISAAC_ABPLUS_AFTERBIRTHP` 改。
- 道具品质来自 `analysis/resources/repentance-a/config/items.xml`（忏悔+）。

## 用法

```python
from isaac_macro.roomconfig import default_room_config
from isaac_macro.level import GameContext, generate_floor
from isaac_macro.run import generate_run
from isaac_macro.floor import Floor
from isaac_macro.secret import secret_posterior

rc = default_room_config()
level = generate_floor(rc, GameContext(), stage=1, stage_type=0, stage_seed=2334308359)
floor = Floor.from_level(level)
print(floor.ascii())                       # S 起点，B Boss，T 宝箱房，$ 商店，s 隐藏房，X 超级隐藏房
run = generate_run(rc, 'SXG6 R8XA', last_stage=8)   # 游戏里显示的种子字符串，或数字种子
post = secret_posterior(floor.visible())   # {格子: 概率}，只用玩家可见的信息
```

- `GameContext` 描述生成时要读的游戏状态：
  - 成就（默认全部解锁）、难度；
  - 玩家的心、钥匙、硬币、饰品、道具；
  - 本局的状态标志（跨层累积，例如已出现过的 Boss 和七宗罪）。
- `debug_start=True`：复现本项目引擎实例的开局方式。实例用 `--set-stage=1 --set-stage-type=0` 启动，走的是 `Game::StartDebug`，所以第一层固定是地下室 I（类型 0），并视为全部解锁。
  - 这也解释了之前 Boss 池探针的疑问："300 局的地下室类型全是 0"。
  - 正常开局走 `Game::Start` → `SetNextStage`，第一层可能是地窖或燃烧地下室。

**导出数据集**

```bash
python tools/export_floors.py --out ../runs/macro/datasets/abplus-floors-v1 --runs 20000 --workers 16 --seed 1 --last-stage 11
```

- 已导出一份：2 万局，共 198,666 层，包含第 1–8、10、11 层。
- 16 个进程用时 183 秒，共 68 MB，放在 `rl/runs/macro/datasets/abplus-floors-v1`（`runs/` 不进 git）。
- 字段见 [`dataset.py`](isaac_macro/dataset.py) 开头的说明：
  - 每层一条 JSON：种子、楼层、诅咒、玩家资源、所有房间（位置、形状、类型、布局编号、门、生成深度），以及 Boss 房、隐藏房、超级隐藏房的格子。
  - 另有 NPZ 格式：每层 5 张 13×13 网格（房间编号、类型、形状、深度、门）。
- 玩家资源会影响部分特殊房间是否出现（街机房要 5 个硬币，挑战房要满血，骰子房要 2 把钥匙，卧室要低血量）。导出时按 `random_player` 随机抽取，这个分布是粗略设定，没有用真实对局校准。

## 验证

**与引擎逐种子一致**〔事实〕
- 数据：本项目 AB+ 实例的 `log.txt`，共 11,048 条地下室 I 记录，去重后 9,774 个种子。其中 154 层有迷宫诅咒，472 层重试过生成。
- 比对内容全部一致：
  - 诅咒；
  - 每一次 `Generate` 的过程：放下的非 1×1 房间形状序列、房间数、循环数、补死路次数、判定结果；
  - "Map Generated in N Loops"；
  - 起始房间的 SpawnRNG 种子。这个种子来自楼层 RNG，它之前的所有抽取都要一致：Boss 选择、各特殊房间的布局与判定、跨层标志等。
- 引擎有 418 次"特殊房间放不下 → 整层重生成"（涉及 403 层），全部复现。这间接验证了 Boss 房和特殊房间在死路上的改形与选择。
- `tests/data/engine_levels_abplus.json` 收了其中 454 层作为回归测试：全部 154 个迷宫层、150 个有放置失败的层、150 个普通层。

**整层逐房间与引擎一致**〔事实，2026-09-26〕
- **怎么测的**：用 [`abplus_probe_floors.py`](../bridge/python/abplus_probe_floors.py) 在远端起一个额外实例（nice 19，约 5 分钟），跑了 600 局。
  - 每局先 `restart`，然后依次用控制台 `stage N`/`Na`/`Nb` 进入各层。
  - 每层用 `Level:GetRooms()` 读出所有房间。
  - `stage` 命令会同步调用 `Level::SetStage` 和 `Level::Init`（`submit_input` 0x12B2D1/0x12B2DD），所以读到的楼层种子就是生成时用的种子。
- **结果**：6,000 层全部一致（`tools/compare_engine_floors.py`）。
  - 113,897/113,897 个房间完全相同，比较的字段是格子、类型、布局编号、子类型、形状，以及三个房间种子；
  - 诅咒 6,000/6,000；
  - 生成时写入的跨层标志 6,000/6,000；
  - 迷宫层 44/44。
- **覆盖范围**：
  - 第 1–8 层的全部三种类型，第 10、11 层的两种类型，每种组合 200 层（地下室 I 普通类型 600 层）。
  - 核对了 6,000 个隐藏房和 6,000 个超级隐藏房的位置。
  - 受资源影响的房间：街机房 958、宝库 348、骰子房 139、献祭房 1,339、挑战房 2,045、卧室 57、图书馆 239、诅咒房 2,494、小头目房 1,196。
- **测试条件**：
  - 调试开局，全部成就解锁；
  - 以撒，带 D6，没有饰品和其他道具；
  - 硬币按局取 0/5/12，钥匙取 0/2，红心取满或少 1 颗。
- 原始数据在 `rl/runs/macro/engine-floors/`（`floors-600.jsonl`，实例的 `log.txt` 也在）。
- 其中 184 层（含全部 44 个迷宫层）收进了 `tests/data/engine_floors_abplus.json.gz` 作为回归测试。

**还没被引擎覆盖的分支**
- **正常开局**：`Game::Start` 从第 0 层走 `SetNextStage`。探针用 `stage` 命令跳层，没有经过这一步；楼层类型规则只核对了反编译。
- **依赖道具或饰品的分支**：
  - 第 7–8 层的银元、血冠；
  - 碎卡片带来的第二个隐藏房；
  - 黑蜡烛、妈妈的盒子；
  - 宝箱房子类型相关的饰品。
- **其他**：未全解锁的存档、困难模式、虚空（第 12 层）。

**未翻译**
- 贪婪模式、???（蓝子宫，第 9 层）、挑战、特殊种子。
- `AllowedDoors`（`precalc_allowed_doors`）。
- 房间内容：敌人和拾取物的生成、道具池抽取。
- 档案里经 ISAAC 加扰的存储分块。AB+ 的房间文件都是 MiniZ 分块，用不到。
- 第 11 层的墓室：它在隐藏房之后才改形，所以隐藏房推断在暗室层会有一点偏差。

**速度**
- 单核约 5–6 ms/层，含转换成抽象楼层。
- 16 进程约 1,100 层/秒（本机 20 线程）。

## 结果

除特别注明外，以下都在第 1–8 层上统计。
- **生成楼层**：楼层类型和诅咒按整局规则产生；隐藏房、超级隐藏房的真实位置来自生成器本身。
- **引擎楼层**：上面探针读出的 5,200 层（第 1–8 层）。生成器已被证明与引擎逐房间一致，两者本应给出相同的统计。

**隐藏房推断**（`tools/eval_secret.py`）

| 方法 | 引擎楼层（5,200）：第 1 颗命中 / 2 颗内 / 平均用弹 | 生成楼层（7,928）：同上 |
|---|---|---|
| 精确后验（已知超级隐藏房位置） | 43.3% / 70.0% / 2.14 | 41.6% / 68.8% / 2.19 |
| 精确后验，超级隐藏房位置按先验边缘化 | 43.0% / 69.7% / 2.16 | 41.5% / 68.9% / 2.20 |
| 同上，特殊房间的布局未知 | 41.0% / 67.5% / 2.24 | 39.4% / 66.7% / 2.28 |
| 基线：邻居最多的格子先炸 | 30.4% / 54.7% / 2.82 | 28.9% / 53.4% / 2.87 |

- **校准**：两类楼层上，精确后验的预测概率与实际命中频率在各档都吻合，例如引擎楼层上 0.325 对 0.327、0.755 对 0.755。
- **只有特殊房间的布局有影响**：黑名单只统计在隐藏房之前放下的房间，即 Boss 房、超级隐藏房和其他特殊房间。普通房间的布局在隐藏房之后才选，而且一定带着通向隐藏房的门。
- **超级隐藏房的规律**
  - 它总是除 Boss 房外最深的死路，只挨着 1 个普通房间（100%）。
  - 按"深度不小于所有可见死路"筛出的候选平均 5.4 格，真值 100% 在候选中。
  - 按深度差拟合权重（`tools/fit_super_secret.py`，11,896 层）。**结果取决于能否认出房间布局**：
    - 普通房间的门位从外观上看不出来，只有认出具体布局才知道哪面墙有门位；这要把房间里的石头、坑等和房间文件比对。
    - 认得出布局：第 1 颗命中 59%，平均 1.8 颗（引擎楼层 58.7%、1.85 颗）。
    - 认不出布局：第 1 颗命中 46%，平均 2.3 颗。

**规则规划器 v0**（`tools/eval_planner.py 200 4`，1,577 层，平均 19.1 个房间）

| 开局炸弹 | 换房次数 | 找到隐藏房 | 找到超级隐藏房 |
|---:|---:|---:|---:|
| 0 | 34.6 | 0% | 0% |
| 1 | 36.9 | 42% | 0% |
| 2 | 39.3 | 69% | 25% |
| 3 | 41.4 | 85% | 48% |
| 5 | 44.2 | 97% | 79% |

- 不含战斗：进房即清。
- 钥匙锁门规则尚未在反编译里核实，所以默认不设锁。

**道具先验**（`python -m isaac_macro.items --pools`）
- AB+ 有 549 个道具（ID 最大 552）、127 个饰品、26 个道具池；其中 547 个道具在忏悔+ 里有品质。
- 另有两份手工清单，逐个核对过 AB+ 名称：
  - 16 个改变攻击方式的道具；
  - 7 个对机器人有害的道具。
- 任务道具取自忏悔+ 的 `quest` 标签：钥匙碎片 1/2、宝丽来、底片、断铲两块、妈妈的铲子。
- **AB+ 与忏悔+ 的道具池差别很大**。例如 AB+ 的隐藏房池平均品质只有 1.65、品质 4 的概率为 0；忏悔+ 是 2.30 和 16%。

  | 池 | 道具数 | 平均品质 | P(品质≥3) | P(品质=4) |
  |---|---:|---:|---:|---:|
  | treasure | 339 | 1.81 | 26% | 5% |
  | shop | 78 | 1.92 | 26% | 1% |
  | boss | 56 | 1.80 | 25% | 2% |
  | devil | 61 | 2.24 | 44% | 9% |
  | angel | 42 | 2.52 | 56% | 14% |
  | secret | 31 | 1.65 | 19% | 0% |
  | curse | 23 | 2.25 | 54% | 0% |
  | library | 12 | 1.92 | 33% | 8% |

## 测试

```bash
python -m pytest -q tests
```
