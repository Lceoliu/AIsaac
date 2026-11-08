# Isaac Run Recorder（忏悔 J746 整局被动录制）

更新：2026-11-08（recorder 0.3.2：phase8-mouse 回放还原鼠标流，见第 8 节"鼠标还原"与第 9 节；格式不变 `isaac-run-v1`）。需求与决定见 `experiments/run-recorder/DESIGN_BRIEF.md` §6（D1–D12）与 §7（E1–E8），第一阶段实测见 `experiments/run-recorder/INVESTIGATION.md`，DLL 的依据见 `experiments/run-recorder/mt-probe/MT_PROBE.md`，验收结果见 `experiments/run-recorder/phase2/ACCEPTANCE.md`、`experiments/run-recorder/phase3/ACCEPTANCE.md`。

## 1. 它做什么、不做什么

- 目标构建：Repentance v1.7.8a.J746（本地离线 GOG/Galaxy x86），隔离副本 `experiments/run-recorder/game-rec-01`，游戏以 `--luadebug` 启动（mod 需要 `io`/`os`/`package.loadlib`）。
- **默认带最小 DLL**（E1，第 10 节）：经挂起式注入器加载 `rr_mt.dll`，存档重定向到副本内 `profile/`（E2），mod 每局开局用随机 salt reseed 引擎全局 MT 并逐帧记录 MT 校验值（E3/E4），于是"种子 + salt + 每 tick 输入 + 脚本命令"可以**精确**回放（E6）。`rl/record_run.ps1 -NoDll` 回退到第二阶段行为（不注入、真实 Documents 存档、回放只能近似）。
- **恢复依据是记录本身**（D1）：每个逻辑帧写完整可读状态（玩家、全部实体、房间头、网格/门的变化），每个 60 Hz tick 写引擎实际消费的输入，另记生成种子流、鼠标流、事件。回放只是分层一致性检查（第 8 节）。
- **被动**：不改规则、不冻结、不无敌、不禁控制、不执行控制台命令、不消耗游戏 RNG。唯一的例外是带 DLL 时每局开局 1–2 次 MT reseed（第 10 节）：它不改变游戏规则，只是让这一局的随机数流由 StartSeed xor salt 决定（原版由开机时间决定），salt 每局随机，所以同种子的两局照样不同。`MC_INPUT_ACTION` 只读并返回 `nil`；`MC_PRE_ENTITY_SPAWN` 只记录不改种子；读状态时用标志位把引擎内部再触发的输入查询忽略掉。画面上录制期间左上角常驻一个半大小的暗红 "REC"（每局前 2 秒更亮；环境变量 `IRR_HUD=0` 关闭）。
- 三种模式：
  - `record`（默认）：人类会话就是这个模式。
  - harness：仅当输出目录里有 `irr_control.json`（测试脚本放的）才加载 `harness.lua`：文件 IPC、`seed` 开局、测试机器人（`bot.lua`）、在指定帧执行脚本化控制台命令、帧数上限、开销基准、鼠标探针。harness 执行的一切命令都写进 run 文件（`hcmd` 行 / `header.harness`）。
  - `replay`：harness 之外再有 `irr_replay.lua`（`replay_run.py` 生成）时，harness 开出的局按记录注入输入、钉回生成种子，并把回放局本身完整录下。
- 文件：`main.lua`（回调、分局、写文件）、`state.lua`（状态快照与 digest）、`inputs.lua`（tick 输入）、`mt.lua`（DLL 导出函数、salt、reseed）、`harness.lua`、`replay.lua`、`bot.lua`（后三个只在测试时加载）。

## 2. 会话目录

`rl/record_run.ps1`（→ `rl/bridge/python/isaac_bridge/run_recording.py`）每次运行建一个会话目录 `<OutRoot>/<SessionName>/`：

| 路径 | 内容 |
|---|---|
| `session.json` | 会话元数据（第 3 节） |
| `summary.txt` | 会话摘要：局数、续局/截断局数、逻辑帧、游戏小时、MB、MB/小时、启动次数/失败次数、每局一行 |
| `runs/run-NNNN.jsonl.zst` | 每局一个文件，zstd-19 压缩，按开局顺序编号 |
| `lua/` | mod 的输出目录（环境变量 `IRR_DIR`）：原始 `run-<proc>-<NNNN>.jsonl`（`raw=keep` 时保留；默认校验后移到 `<OutRoot>/_raw_staging/<会话名>/`，不删除）、`signal-*.start/end.json`、`status.json`、`irr_boot.json`、`irr_session.json`、`errors.jsonl`（有错误才有） |
| `save/pre/`、`save/<run>/`、`save/post/` | 存档目录快照：会话前、每局开局后（mod 写开局信号 → Python 等存档目录静止 1 s 后复制）、会话后。每个快照有 `manifest.json`（文件名、字节、mtime、sha256）。排除 `*.dmp`、`log.txt` |
| `options/` | `options.before.ini`（会话前原样）、`options.effective.ini`（强制后）、`options.after-game.ini`（游戏退出时写回的） |
| `game/` | 每次启动的命令行与环境（`<tag>.command.json`，带 DLL 时含注入器的退出码与输出）、stdout/stderr（带 DLL 时是注入器的输出）、退出码（`<tag>.exit.json`）、游戏自己的 `log.txt` 副本、DLL 自己的日志 `rr_mt.<pid>.log`（身份门、存档重定向、每次 reseed） |

**存档目录**（E2）：带 DLL（默认）= 副本内 `game-rec-01\profile\Documents\My Games\Binding of Isaac Repentance (Galaxy)\`（DLL 把 `GetUserProfileDirectoryA` 指到 `game-rec-01\profile`，游戏写的 `game\savedatapath.txt` 会显示这个路径，原文记在 `session.json` 的 `savedatapath_txt`）；`-NoDll` = 真实用户的 `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance (Galaxy)\`（不带 DLL 时 `savedatapath.txt`、USERPROFILE 重定向都无效，INVESTIGATION §2）。快照、`-InitSaveFrom <dir>`（只把那份拷贝复制进本会话的存档目录，不写回来源）、`cmd_history.txt`、`options.ini` 的强制与还原都对本会话的存档目录做。会话期间另有一个只读线程每 2 秒检查真实 `Documents\My Games\` 下是否出现 `(Galaxy)` 目录（只 stat/listdir），结果写进 `session.json` 的 `real_documents_watch`。

## 3. session.json

| 字段 | 含义 |
|---|---|
| `schema` | `isaac-run-session-v1` |
| `name`、`created`、`finished` | 会话名、开始/结束时间（本地时区 ISO） |
| `host` | 主机名、平台、CPU、Python 版本 |
| `copy` | 副本路径；`binaries`：EXE、`Galaxy.dll`、`Lua5.3.3r.dll` 的字节数与 sha256；`resources` junction 目标 |
| `use_dll`、`dll` | 是否注入；`dll`：`rr_mt.dll` 与 `rr_launch.exe` 的路径与 sha256、`profile` 目录、构建记录（`<副本>/rec/build.json` 的时间与源码 sha256）；`-NoDll` 时 `{enabled: false}` |
| `save_dir`、`savedatapath_txt` | 本会话的存档目录；游戏启动后写的 `game\savedatapath.txt` 原文（带 DLL 时应指向副本 `profile`） |
| `real_documents_watch` | 真实 `Documents\My Games\...(Galaxy)` 的只读监视：`at_start`/`at_end`（是否存在、`My Games` 目录清单）、`changes`（存在性每次变化的时间与清单）、`polls`、`ever_existed`、`appeared_during_session` |
| `mod` | 安装到副本的 recorder 文件及 sha256 |
| `mods` | 副本 `mods/` 下每个 mod：目录、`metadata.xml` 的 name/id/version、是否有 `disable.it`、全部文件 sha256 |
| `game_version` | 游戏 log 里的版本行（`Binding of Isaac: Repentance v1.7.8a.J746`） |
| `options` | `before`（会话前全部键值）、`forced`（`EnableDebugConsole=0`）、`forced_before`、`restored`（会话后还原的键） |
| `save_pre`、`save_post`、`init_save_from` | 存档快照清单 |
| `launches` | 每次启动尝试：tag、pid、`use_dll`、`boot_s`（mod 加载用时）、`result`（ok/failed）、失败原因（注入器失败、进程退出码、log 洪水、mod 未加载）、版本行、`mod_mt`（mod 加载时的 MT 状态：available/reason/DLL 信息）、`dll_status`（DLL 日志里最后的状态行）、`dll_log` |
| `runs[]` | 每局：`file`、`lua_name`、`raw_bytes`/`zst_bytes`、`raw_sha256`/`zst_sha256`、`verified`、`validation`（帧数、首末 gf、header 摘要、end 摘要、事件计数、`truncated`、`errors`、`game_hours`）、`save_snapshot`（该局开局快照路径与信号）、`console_used`、`end_signal` |
| `console` | 会话期间 `cmd_history.txt` 的新增行（D8；有新增则本会话所有局 `console_used=true`） |
| `summary` | 与 summary.txt 相同的统计 |

## 4. run 文件（`isaac-run-v1`）

JSONL，每行一个 JSON 对象，第一个键总是 `k`（行种类）。文件以二进制模式写（LF 换行），`setvbuf("full", 256 KB)`，进房、暂停、每 300 帧 flush 一次，所以进程崩溃时最多丢最后约 10 秒且文件没有 `end` 行（归档时标 `truncated`）。

**通用规则**

- **帧序号 `n`**：本局第几个逻辑帧（从 1 起，每个 `MC_POST_UPDATE` +1）。`gf` = `Game():GetFrameCount()`，`t` = `Isaac.GetTime()`（毫秒，用于对齐外部视频），`ifc` = `Isaac.GetFrameCount()`（渲染帧计数）。
- **行顺序**：`room`/`level`/`items`/`ev` 行的 `n` 是它们所属的帧，写在该帧的 `f` 行**之前**（发生在第 n 帧的更新或第 n−1 帧之后的渲染里）；`hcmd`/`mark`/`perf` 行写在第 `n` 帧的 `f` 行**之后**。
- **定点**：带 ×100 标记的字段是 `floor(x*100+0.5)` 的整数（位置/速度单位 0.01 像素；HP、伤害、射速等同样 ×100）。哪些字段是定点由 `header.fields.fixed_x100` 给出。不用 `string.format("%a")`。
- `null`：API 不存在、读失败、NaN/Inf，或（输入）从未被游戏查询过。

### 4.1 header（第一行）

| 字段 | 含义 |
|---|---|
| `schema`、`recorder`、`format` | `isaac-run-v1`、mod 版本（第三阶段 `0.2.0`，v2 `0.3.0`）、格式版本（v2 = `2`，v1 文件没有此键；见 4.10） |
| `obs` | 观测模式：`full`（默认，每帧完整）/ `keyframe`（`IRR_OBS=keyframe`，非关键帧没有 `e`，见 4.4） |
| `mt` | 全局 MT reseed 记录（第 10 节）：`available`、`reason`（`ok` / `no_dll` / `loadlib failed: …` / `identity gate failed …`）、`dll`、`info`（DLL 的 `RRMT_Info`：ready、profile 调用次数、reseed 次数、PE 时间戳、SizeOfImage）、`continued`、`rule`、`salt`（本局 salt，32 位整数）、`start_seed`、`seeded_with`（= start_seed xor salt）、`reseeds[]`：每次 reseed 的 `at`（`POST_PLAYER_INIT` / `POST_GAME_STARTED`）、`gf`、`ifc`、`t`、`start_seed`、`salt`、`seeded_with`、`before`/`after`（reseed 前后的 [cursor, hash]）、`from_replay`（回放局用的是录制的 salt）、`stale`（PLAYER_INIT 时的种子与开局种子不同，异常情况）、`continuing`（0.3.1 起：这次 PLAYER_INIT reseed 发生在续局读档时）；`early_continue`（0.3.1 起：本进程是否启用续局早期 reseed，`IRR_MT_EARLY_CONTINUE=0` 关闭）、`affinity`（0.3.2 起：进程启动时的 CPU 亲和掩码 RR_MT_AFFINITY，十进制数；没设则无。§8：没限核的录制回放不保证精确） |
| `mode` | `record` / `replay` |
| `run`、`proc`、`run_index` | 文件名（不含扩展名）、进程标记（`IRR_PROC`）、本进程第几局 |
| `continued` | `MC_POST_GAME_STARTED(isContinued)`：续局为 true（续局 gf 不从 0 开始，状态来自 `gamestate1.dat`，见该局存档快照） |
| `gf`、`t`、`ifc`、`os_time`、`os_date` | 开局时的帧计数与时间（`os_date` 为 UTC） |
| `meta.globals` | `REPENTANCE`、`REPENTANCE_PLUS` 是否定义（字符串 `"true"` / `"nil"`）；J746 为 `"true"` / `"nil"` |
| `meta.seeds` | `start_seed`、`start_seed_string`、`player_init_seed`、`is_custom_run`、`count_seed_effects`、`seed_effects`（为 true 的 SeedEffect 名称列表，逐个 `HasSeedEffect`） |
| `meta.difficulty`、`challenge`、`victory_lap`、`greed_mode`、`time_counter`、`blue_womb_par_time`、`num_players` | 局参数 |
| `meta.players[]` | 每个玩家：`num`、`type`（PlayerType）、`name`、`controller_index`、`init_seed`、`drop_seed`、`index` |
| `meta.options` | Lua `Options` 表能读到的键（J746 实测 24 个，含 `MouseControl`、`PauseOnFocusLost`）；完整 `options.ini` 见会话目录 |
| `meta.lua_version` | `Lua 5.3` |
| `session` | Python 写的 `irr_session.json` 原样（单行 JSON）：会话名、EXE sha256、三个二进制的 sha256、游戏版本行、mods 清单（含 `disable.it`）、强制的 options、启动命令、主机名、存档目录 |
| `harness` | 测试局才有：开局命令、机器人、帧上限、脚本命令列表、是否回放；人类局为 `null` |
| `replay_of` | 回放局：被回放的录制文件路径 |
| `fields` | 各种数组行的列名（见下），以及 `fixed_x100`（定点字段）、`entity_extra_types`（附加字段类别对应的实体类型） |
| `prestart_spawns` | 本局 `MC_POST_GAME_STARTED` 之前（上一局结束之后）发生的生成 `[type, variant, subtype, seed, gf]`，最多 4096 条 |

### 4.2 level（开局与每次 `MC_POST_NEW_LEVEL`）

`stage`、`stage_type`、`curses`、`name`、`dungeon_seed`（`GetDungeonPlacementSeed`）、`start_room`、`is_alt`、`greed_wave`，以及 `rooms`：本层**全部** RoomDescriptor，每个一行，列 = `fields.room_desc`：
`list_index, grid_index, safe_grid_index, spawn_seed, decoration_seed, award_seed, type, variant, subtype, shape, name, difficulty, stage_id, flags, visited_count, clear, clear_count, display_flags`。

### 4.3 room（开局与每次 `MC_POST_NEW_ROOM`）

`room_index`、`desc`（当前房间的 RoomDescriptor，同上列）、`seeds`（[SpawnSeed, DecorationSeed, AwardSeed]）、`type`、`shape`、`gw`/`gh`（网格宽高）、`top_left`/`bottom_right`（×100）、`clear`、`room_frame`、`backdrop`、`stage`、`stage_type`、`curses`，
`grid`：房间里全部网格实体（含墙、门格），每格 `[index, type, variant, state, collision, var_data]`；`doors`：8 个门槽里存在的门，每个 `[slot, target_room_index, target_room_type, open, locked, state, variant]`。进房后第一帧强制为关键帧。

### 4.4 f（每逻辑帧一行，`MC_POST_UPDATE`）

| 键 | 含义 |
|---|---|
| `n`、`gf`、`t` | 见通用规则 |
| `kf` | `1` = 关键帧（本局第 1 帧、进房后第一帧、每 300 帧）：`p`/`e` 全部写完整行 |
| `tk` | 本帧的 60 Hz tick 输入（4.5） |
| `sp` | 本帧的生成种子流（`MC_PRE_ENTITY_SPAWN`，D4）：`[type, variant, subtype, seed, gf]`，`gf` 是生成时的帧计数（中间 tick 里发生的生成 gf = 上一帧）。回放局里是钉回后的实际种子 |
| `mt` | 全局 MT 校验（E4）：`[cursor, hash]`，`RRMT_State()` 在本帧 `MC_POST_UPDATE` 时只读（hash = 624 个状态字 + cursor 的 FNV-1a32，整数）；只在 `header.mt.available` 时有。DLL 把状态拷两份、两份一致才算数；若第一次两份不一致（读的同时内存被改），行里多一个第三数 = 读的次数（`[cursor, hash, reads]`），比较时只比前两个 |
| `ms` | 鼠标流（D6）：每个渲染帧读一次 `Input.GetMousePosition(false/true)` 与 `IsMouseBtnPressed(0/1/2)`，与上次不同才记 `[本帧窗口内第几次渲染, 屏幕x, 屏幕y, 游戏x, 游戏y, 按键位]`（坐标 ×100；按键位 左=1 右=2 中=4）。"第几次渲染" r = 上一次 `MC_POST_UPDATE` 之后的第 r 次 `MC_POST_RENDER`（每局第一条总会记，= 开局时的光标）。语义（phase8-mouse 静态分析）：屏幕坐标 = 引擎静态链接的 GLFW 主窗口的虚拟光标位置（double，只由 `WM_MOUSEMOVE` 更新，整数值；本机 150% 缩放下 = 客户区逻辑坐标 ×1.5 取整），引擎不用 `GetCursorPos` 取位置；游戏坐标是 Lua 按当前镜头与窗口尺寸换算的，窗口尺寸不同则不同。引擎每次 `Manager::Update` 末尾比较光标位置，变了就让游戏内光标显示约 300 次迭代（5 s） |
| `r` | 房间头：`[room_frame, clear, alive_enemies, room_index]` |
| `p` | 玩家行（列 = `fields.player`，共 69 列）：编号、PlayerType、位置/速度、各种心（红/上限/魂/黑/骨/永恒/金/碎/腐）、额外生命、钱/炸弹/巨型炸弹/钥匙/金钥匙/金炸弹/便便法力、伤害/当前射击冷却/射速上限/弹速/射程/移速/幸运/泪高/泪落速（×100）、飞行、4 个主动槽（道具、充能；槽 0 另有电池充能与子充能）、2 卡槽、2 药槽、2 饰品槽、道具数、受伤无敌计时、累计受伤、头/射击/移动方向、瞄准方向（×100）、死亡、可控、可见、碰撞半径（×100）、ControllerIndex、InitSeed、DropSeed、TearFlags 低/高 64 位、动画名/帧/叠加动画名 |
| `e` | （`obs=keyframe` 时只在关键帧有；其余帧缺省，实体状态由回放重建，`load_run.py` 把这些帧的 `entities` 设为 `None`）实体行（除玩家外 `Isaac.GetRoomEntities()` 的全部实体，含 1000.x 特效与不可见实体），列 = `fields.entity` 的 19 个公共字段：`idx`（Entity.Index）、`type`、`variant`、`subtype`、`x`、`y`、`vx`、`vy`、`hp`、`max_hp`（×100）、`frame`（FrameCount）、`init_seed`、`drop_seed`、`spawner_type`、`parent_idx`（无父为 −1）、`visible`、`coll`（EntityCollisionClass）、`anim`、`aframe`；其后平铺该类型的附加字段（`fields.entity_extra[类别]`：npc = State/StateFrame/I1/I2/冠军色/是否 Boss/V1/TargetPosition/ProjectileCooldown，tear、projectile、pickup、familiar、effect、bomb、laser、knife 各有一组） |
| `g` | 网格变化：`[index, type, variant, state, collision, var_data]` = 新状态，`[index]` = 该格网格实体消失。已知的非墙非门格每帧比较，全部格每 15 帧重扫一次（所以新出现的网格实体、墙变成的秘密门最多晚 0.5 s 记录） |
| `dr` | 门变化：同 `room.doors` 的行，`[slot]` = 门消失 |
| `dg` | digest 三层（4.6） |

**关键帧与差分行**（`p` 与 `e` 相同）：非关键帧里每个对象写成三种之一——
- 完整行（对象新出现，或实体的 `init_seed`/行长与上一帧不同）；
- 差分行 `[key, [[列, 值], ...]]`：只列出变化的列；**上一帧与本帧在该列都是整数时，值是差（本帧 − 上一帧），否则是新值**（字符串、null）；
  **位置列例外（位置预测）**：x、y 两列（`header.fields.position_prediction` 给出 x 的列号，0 起：玩家 2、实体 4；y 在其后，vx/vy 再后两列）在"上一帧 x、本帧 x、本帧 vx 都是整数"时写残差 `x − (上一帧 x + 本帧 vx)`，残差为 0 就不写（引擎每帧 位置 += 速度，残差多为 0 或 ±1）；解码时先还原其它列，再按 `x = 上一帧 x + 本帧 vx + 残差（缺省 0）` 还原；本帧 x 不是整数（null）时按普通规则写新值；
- 只有 `key`（整数）：与上一帧完全相同。
`key` 是第 0 列（玩家编号 / 实体 Index）。行的顺序就是本帧 `GetRoomEntities()` 的顺序；上一帧有、本帧没列出的对象已经消失。随机访问第 n 帧 = 从不晚于 n 的最近关键帧开始逐帧还原（`load_run.py` 已实现）。

### 4.5 tick 输入（D3，INVESTIGATION §3）

游戏按 60 Hz 迭代查询输入；有逻辑更新的那次迭代（帧计数先 +1 再查询）是 `"u"`，其余是中间 tick `"i<j>"`（j = 上次 `MC_POST_UPDATE` 之后的渲染次数）。正常房间一帧为 `i1`（完整一组玩家查询）、`i2`（只有全局查询）、`u`；换房过场与开局淡入会有更多中间 tick。中间 tick 的 triggered 会直接驱动逻辑（主动道具、卡、炸弹），所以必须按 tick 存。

- 每个玩家维护 36 个值（12 个动作 × value/pressed/triggered）的当前状态；某 tick 内被查询到的值覆盖进去，未被查询的沿用上一 tick，从未被查询过的是 `null`。同一 tick 内同一查询的多次结果恒等（实测），取第一次。
- `tk` 列表元素：
  - `"u"`、`"i3"`：该 tick 状态没变；`"i5-300"`：连续中间 tick 5…300 都没变（长时间暂停/过场时压缩用）；
  - `[kind, [[玩家编号, [[value, pressed, triggered] × 12]], ...], [[hook, action, who, value], ...]]`：第二项是状态变了的玩家的**完整** 12 个三元组（动作顺序 = ButtonAction 0–11：LEFT RIGHT UP DOWN SHOOTLEFT SHOOTRIGHT SHOOTUP SHOOTDOWN BOMB ITEM PILLCARD DROP；value 键盘为 0/1，手柄模拟量为浮点 `%.9g`，pressed/triggered 为 0/1）；第三项（可省）是本 tick 的其它查询（实体为 nil 的全局查询 PAUSE/MENU*/MAP/RESTART/CONSOLE…、非玩家实体、action ≥ 12）里的**非零**值，`hook` 0=PRESSED 1=TRIGGERED 2=VALUE，`who` −1=nil 实体、−3=非玩家实体、≥0=玩家编号。
- 鼠标射击（`MouseControl=1`）绕过 `MC_INPUT_ACTION`（INVESTIGATION §3），只能从 `ms` 流看到。

### 4.6 digest（`dg`，回放比对用）

三个 32 位整数，分层嵌套：`d1 = H(L1)`，`d2 = H(L2, 初值 d1)`，`d3 = H(L3, 初值 d2)`。`H` = 把输入串补零到 64 字节整数倍，按小端 32 位字 w 依次 `h = ((h XOR w) * 16777619) mod 2^32`，初值 2166136261。浮点用 `string.pack("<f", x)` 的精确位（引擎内部是 float32，所以是精确值；`0.0` 与 `-0.0` 不同），整数用 `"<j"`。
- L1「玩家与房间」：`[gf, room_frame, clear, room_index, stage, stage_type, SpawnSeed, DecorationSeed, AwardSeed]` + 每个玩家的全部定点字段（原始 float）与全部整数/布尔字段。
- L2「非外观实体」：每个非 1000.x 实体的 `[x, y, vx, vy, hp, max_hp]`（float）+ `[type, variant, subtype, FrameCount, NPC State, NPC StateFrame]`（非 NPC 为 −1）。不含 Index 与种子（特效生成次数不同会让后续 Index 整体错位）。
- L3「全量」：1000.x 特效同 L2 的内容、每个实体的 `[InitSeed, DropSeed, Index]` 与全部附加字段、当前网格与门状态的签名。
三层与第一阶段 `probes/divergence_views.py` 的 `player_room` / `gameplay_no_grid` / `full` 口径对应。

### 4.7 其它行

| 行 | 字段 |
|---|---|
| `items` | 道具清单：开局必写；道具数或主动槽 0 变化时、以及每个关键帧重扫，清单与上次不同才写。`p`（玩家编号）、`count`（`GetCollectibleCount`）、`items`：`[[道具 id, 数量], ...]`（对全部道具 id 调 `GetCollectibleNum(id, true)`） |
| `ev` | 事件：`ev` 名称 + 参数 `a`：`damage` [玩家, 伤害×100, DamageFlags, 来源 type, 来源 variant, 无敌帧]；`use_item` [道具 id]（`MC_PRE_USE_ITEM`）；`use_card` [卡 id]；`use_pill` [药效 id]；`cmd` [命令, 参数]（`MC_EXECUTE_CMD`，只对游戏不认识的自定义命令触发，内建命令见 D8）；`pause` / `unpause` [Isaac.GetFrameCount, Isaac.GetTime, 本逻辑帧窗口内第几次渲染]（第三个数 2026-11-07 起才有，= 翻转之后第一个中间 tick 的序号；`MC_POST_RENDER` 里 `IsPaused()` 翻转；暂停菜单、控制台、**以及每次换房过场约 18 个渲染帧**都会出现；Tab 地图不暂停，不记）；`game_end` [是否 game over]；`exit` [shouldSave] |
| `hcmd` | harness 执行的控制台命令：`tag`（script = 脚本命令，回放时同帧重放；exec = 测试驱动的其它命令）、`cmd` |
| `mark` | harness 的时间标记：`label`、`t`、`ifc` |
| `perf` | 每 150 帧一次：`dt_ms`、`renders`（这 150 帧内的渲染次数）、`cost`（每帧 Lua 耗时直方图，毫秒，`os.clock` 1 ms 分辨率：`hist_ms[i]` = 耗时 i ms 的帧数，最后一格是 ≥31 ms） |
| `end` | 局结束：`reason`（`death` = `MC_POST_GAME_END(true)`；`exit_save` / `exit` = `MC_PRE_GAME_EXIT(shouldSave)`；`restart` = 没有退出回调就开了新局；`frames` / `stop` = harness）、`frames`、`renders`、`errors`、`cost`（整局耗时直方图）、`missing_player_fields`（本构建上读不到的玩家字段）、`mt_unstable_reads`（DLL 进程内 MT 两份拷贝不一致的累计次数；无 DLL 为 null）、`fmt_repairs`/`fmt_examples`（本局写出前修掉的数字格式损坏次数与前 5 个片段，片段里的 0x 写成 "0 x"，见 4.9）、`replay`（回放局的钉种子计数） |

### 4.9 数字格式损坏的修复

这个 Lua 构建的数字转字符串偶发损坏：整数前多出 `0x`（十进制数字不变），会让整行 JSON 解析失败（第三阶段 88k 帧里见到 2 次：网格行 `0x1`、digest `0x4064…`；MT_PROBE §5.3 也见过）。mod 写出每一行前检查 `0x`，把 `0x` 后紧跟（可带负号的）数字处的 `0x` 去掉并计数（`end.fmt_repairs`）；我们写的行里数字从不是十六进制、字符串里也不会出现 `0x`+数字。`load_run.py`（`repair_line`）与会话归档校验（`validation.repaired_lines`）对更早的文件用同一规则容错读取。digest 由 `string.pack` 的字节计算，不受影响。修复不一定能还原原文：human2b 第 10522 帧的 `ea` 里一条差分行损坏成了裸整数 `924`（原差分已丢），`load_run.expand_sparse` 把这种格式外的裸整数当作"该对象外观未变"（phase8-mouse；外观不进 digest，不影响回放比对）。

### 4.8 分局规则

`MC_POST_GAME_STARTED` 开新文件（若上一局还开着先以 `restart` 关闭），写 header、level、room、items。`MC_POST_GAME_END(true)`（死亡）、`MC_PRE_GAME_EXIT`（回菜单、关窗口、`seed`/重开命令）关闭。赢了（`game_end` false）继续录到退出。续局照录并在 header 标 `continued=true`。按住 R 重开时，若新局的进房回调在旧局关闭前到达（gf 变小），不写进旧局。

### 4.10 v2：外观字段（recorder 0.3.0，`header.format = 2`，phase6-art）

v2 只**新增**键，旧字段（`p`/`e`/`g`/`dr`/`r`/`tk`/`mt`/`dg`、room/level/items 各行）的内容、顺序、编码与 v1 完全相同；`header.schema` 仍是 `isaac-run-v1`（会话归档校验 `run_recording.validate_lines`、路线 A 的工具都按这个字符串认文件），版本由 `header.format`（v1 文件没有这个键 = 1）与 `recorder` 区分。旧读取器忽略新键即可照常读 v2 文件；`load_run.py`、`replay_run.compare` 对 v1 文件（human1）行为不变。新字段**不进 digest**（三层 `dg` 的覆盖范围不变，没有新增 cosmetic 层），回放比对口径不变。

| 位置 | 键 | 内容 |
|---|---|---|
| header | `format` | `2` |
| header.fields | `appearance` | 外观行列名（26 列）：`key, file, ov_anim, ov_frame, flip, sx, sy, rot, r, g, b, a, ro, go, bo, sr, sg, sb, sa, sro, sgo, sbo, ox, oy, pox, poy` |
| | `player_appearance_extra` | 玩家外观行另有 `body_color, head_color`（`GetBodyColor/GetHeadColor`，皮肤色，-1 = 默认） |
| | `appearance_fixed_x100`、`appearance_string_columns` | ×100 定点列（sx…poy）；字典编码的字符串列（`file`、`ov_anim`） |
| | `grid_sprite`、`door_sprite` | `[index, file, anim, frame]`；`[slot, file, anim, frame, rot, ox, oy]`（rot/ox/oy ×100） |
| f、room | `sd` | 字符串字典的**新**条目 `[[id, "串"], ...]`：每局从 0 编号、全局唯一、不重定义；第一次用到某个串的那一行（帧行或进房行）里给出，写在使用它的键之前。随机访问时读取器先扫全部 `sd`（`load_run.Run.strings`） |
| f | `pa` / `ea` | 玩家 / 实体外观行（稀疏差分，见下） |
| f | `gs` / `ds` | 网格精灵 / 门精灵的变化（`[index]` / `[slot]` = 消失），规则同 `g`/`dr` |
| room | `grid_spr`、`door_spr` | 进房时全部非墙非门网格的精灵、全部门的精灵 |

外观列：`file` = `Sprite:GetFilename()`（anm2 路径）；`ov_anim`/`ov_frame` = 叠加动画名与帧号（没有叠加动画时 `null`/`-1`）；`flip` = FlipX + 2×FlipY；`sx sy rot` = `Sprite.Scale`、`Sprite.Rotation`（J746 实测与 `Entity.SpriteScale/SpriteRotation` 相同）；`r g b a ro go bo` = `Entity.Color`（乘色与加色偏移，×100 量化）；`sr…sbo` = `Sprite.Color`（只对 1000.x 特效与玩家每帧读，其余实体为 `null` = 未读，按白色处理）；`ox oy` = `Sprite.Offset`；`pox poy` = `Entity.PositionOffset`（泪弹/敌弹的离地高度、跳跃）。主动画名与帧号沿用 v1 的 `e` 行 `anim`/`aframe` 与玩家行 `anim`/`aframe`/`overlay`（不重复记录；v1 的 `anim` 列仍是字符串，只在变化时写，未改成字典）。

**稀疏差分**（`pa`/`ea`）：关键帧、对象新出现（或它的 `e` 行写成了完整行：种子/行长变化）写完整行；其余帧**只写外观有变化的对象**，差分行 `[key, [[列, 值], ...]]` 的规则同 `p`/`e`（两边是整数写差、否则写新值，无位置预测）；没列出的对象沿用上一帧；不在本帧 `p`/`e` 里的对象丢弃。`IRR_OBS=keyframe` 的非关键帧没有 `ea`。

**读取开销控制**（J746 上每次 Lua→引擎属性访问约 0.3–0.4 µs）：`GetFilename`、`FlipY`、`Sprite.Offset` 只在对象新出现、主动画名变化、关键帧时重读（其间按缓存值写）；叠加帧只在有叠加动画时读；门精灵每帧只读动画名与帧号；网格精灵在进房、该格网格行变化、每 15 帧全扫时读。

**Lua 读不到、因此没有记录的**：服装（costume，查看器按 `items` 行与 `costumes2.xml` 合成）、backdrop 墙面变体的挑法（只有 `room.backdrop` 类型）、ReplaceSpritesheet 换过的贴图（道具台的道具图、饰品等由 subtype 推回）、渲染层级 `DepthOffset`/`RenderZOffset`、anm2 的混合模式。J746 实测不存在：`Room:GetFloorColor/GetWaterAmount/GetFXParams/GetCamera`、`EntityPlayer:GetCostumeSpriteDescs/GetSmeltedTrinkets`（REPENTOGON 才有）、`Sprite.Zoom`。

**开销实测**（`experiments/run-recorder/phase6-art/REPORT.md`）：zstd-19 体积 +7–11%（同一局去掉 v2 键对照）；harness 同状态 A/B 基准每帧 Lua 增加 0.015–0.27 ms（1–35 个实体；关键帧多 0.5 ms）。

## 5. 压缩与校验

`run_recording.compress_and_verify`：`zstandard` level 19（带 zstd 内容校验和与原始长度）→ 解压回读，sha256 与原始 jsonl 逐字节一致才算 `verified`；再逐行解析，检查首行是 `isaac-run-v1` header、帧号连续、`end.frames` 与帧行数一致，没有 `end` 行标 `truncated`。结果写进 `session.json` 的 `runs[].validation`。

## 6. 读取：`rl/bridge/python/load_run.py`

```
python rl/bridge/python/load_run.py RUN.jsonl.zst --frame 1234        # 摘要
python rl/bridge/python/load_run.py RUN.jsonl.zst --frame 1234 --json # 全部字段
python rl/bridge/python/load_run.py RUN.jsonl.zst --roundtrip         # round-trip 测试
```
`Run(path).frame(n, with_inputs=True)` 返回 `FrameState`：该帧之前（含）的全部 tick 输入（`inputs_before`，完整状态）+ 该帧状态（玩家/实体已还原差分并把定点解码为 float，网格/门已应用变化，道具、楼层、房间、生成、鼠标、事件）。`Run.iter_frames()` 顺序重建全部帧。round-trip：每帧 解码→重新编码（含差分）与原始行逐字节相同、随机访问与顺序重建一致、tick 展开后压回与原 `tk` 一致。单元测试 `rl/bridge/python/test_run_recording.py`。

## 7. 开销控制

每帧工作都在 `MC_POST_UPDATE` 里，字符串拼接后一次写入；`MC_INPUT_ACTION` 每次查询只做一次原生读取并存进当前 tick。网格只比已知格、每 15 帧全扫；道具清单只在道具数变化时全扫。实测数字见 ACCEPTANCE.md 第 2、3 项。

## 8. 回放检查：`rl/bridge/python/replay_run.py`（回归测试）

```
python rl/bridge/python/replay_run.py RUN.jsonl.zst --name rep1 --report rep1.json
python rl/bridge/python/replay_run.py RUN.jsonl.zst --continue-save SNAPSHOT_DIR   # 续局回放（快照含 gamestate1.dat）
python rl/bridge/python/replay_run.py RUN_A --compare RUN_B          # 只比较两个文件
```
新进程（默认带 DLL，存档在副本 `profile/`；`--no-dll` 不带）→ 进菜单 → harness 执行 `seed <录制种子>`（续局：把快照拷进存档目录，菜单 Enter 续上的第一局就是回放局）→ mod 在录制 `header.mt.reseeds` 的同一时机用**录制的 salt** reseed 全局 MT（录制里没有的时机不 reseed）、在 `MC_INPUT_ACTION` 按 (帧, tick) 注入录制的 12 个动作（全局查询不注入；唯一例外是真人菜单暂停帧的开菜单，见 8.2）、在 `MC_PRE_ENTITY_SPAWN` 按「(gf, type.variant.subtype, 帧内第几次)」钉回生成种子、在同帧重放 harness 脚本命令 → 回放局完整录下 → 逐帧**按数值**比对 MT `[cursor, hash]` 与 `dg` 三层，输出每层首个分歧帧、分歧帧数、首个分歧帧上解码后哪些字段不同、两边 `header.mt.reseeds` 是否逐项相同（`mt.reseeds_match`：时机、seeded_with、reseed 前后的 MT），以及回放局记录的有效输入是否与录制一致（`first_tick_mismatch`）。`exact = true` 表示三层 + MT 全部 0 分歧、MT 每帧都有、同长度、结束方式相容（`end_compatible`：相同；或录制由退出/重开结束、回放在同一帧数被帧上限截止）、reseed 记录一致。

**续局回放的出生位置**（phase7-continue，recorder 0.3.1）：续局时引擎在进开局房之后、`MC_POST_GAME_STARTED` 之前用全局 MT 抽 1 次随机数给玩家选出生位置（实测四个门口 (320,160)/(80,280)/(560,280)/(320,400) 之一），0.3.0 及以前的续局录制只在 `POST_GAME_STARTED` reseed，这次抽取来不及控制。所以 `replay_run.py` 对 `header.continued` 的录制在回放表里写 `start_players = {[num] = {x100=…, y100=…}}`（录制第 1 帧各玩家位置的定点原值，`continue_start_players()`），`replay.lua` 的 `R.apply_start_pos()` 在续局的 `MC_POST_GAME_STARTED`（reseed 与开局行之后、第一次更新之前）把玩家放到该位置、速度清零，计入 `end.replay.start_pos_pinned`（钉了几个玩家）/ `start_pos_moved`（其中原位置不同的个数）。非续局的回放表没有这个字段，行为不变。0.3.1 起的续局录制另在读档后第一个 `MC_POST_PLAYER_INIT` reseed（第 10 节），出生位置本身就由 salt 决定，钉回只是不改变位置的保险。

**诊断开关**（phase7-continue）：`--inject-extras`（实验，默认关）把录制 tick 里非玩家 12 动作的非零查询（`extra`：地图键、菜单确认/左右等；例如真人按确认跳过 Boss VS 画面）也按 (帧, tick, hook, action, who) 注入，计入 `end.replay.extras_injected`。harness 下回放局结束时另写 `irr_pinlog-<run>.json`（生成种子按类型统计 钉回/相同/表里没有，前 400 次钉回），续局/开局各回调时的 gf、房间、StartSeed、玩家位置、MT 状态写 `irr_diag.jsonl`。`compare` 的 full 层首个分歧帧说明里新增 `effect_fields`（特效按 Index 逐字段对照；只差速度时以前是空的）。

**鼠标还原**（phase8-mouse，recorder 0.3.2，默认开；`--no-mouse` 关）：回放表另带录制的鼠标流（`mouse_init` = 第 1 帧第一条 ms，`mouse[n]` = 其余条目 `[r, 屏幕x×100, 屏幕y×100, 按键位]`），`replay_run.py` 给回放进程设 `RR_MT_MOUSE=1`，`rr_mt.dll` 于是导出可用的 `RRMT_SetMouse`（第 10 节）：直接写 GLFW 的虚拟光标位置与鼠标按键，并在第一次调用时子类化窗口过程、丢掉真实鼠标消息（有人把鼠标划过回放窗口也不影响结果；无焦点、加速、观战都能用）。对应规则：ms 条目 r 是"上一次 `MC_POST_UPDATE` 之后第 r 次渲染"读到的值，它在第 r 次迭代的消息泵里生效（每次迭代 = `Manager::Update`（泵消息 → 可能的逻辑更新 → 光标闲置计数）+ 1 次渲染），所以回放在第 r−1 次渲染的 `MC_POST_RENDER`（本次采样之后）设置，r = 1 在上一帧的 `MC_POST_UPDATE` 设置（同一次迭代的光标计数在逻辑更新之后）；有暂停对齐的帧，r−1 按 tick 的同一映射换算；本帧渲染不够（回放的暂停比录制短）而没设上的条目在本帧 `MC_POST_UPDATE` 补上（`folded`）。开局位置用 mode 2 无声设置（不触发"光标移动"，闲置计数清 0 = 假定开局时光标隐藏）。统计在 `end.replay.mouse`（available/reason/init/applied/folded/failed），`compare` 另报 `mouse`：回放局自己读到的 ms 流与录制逐帧比较（诊断，不进 `exact`）。DLL 没有 `RRMT_SetMouse`（旧构建，如 game-rec-01）或没设 `RR_MT_MOUSE` 时 `available=false`，回放照常、光标不动。
局限：录制里光标在开局前是否刚动过（闲置计数）不可知；回放的暂停比录制长/短时闲置计数的 5 s 窗口按迭代计，相对逻辑帧会错开；ms 的游戏坐标取决于实际窗口尺寸（human1 回放的游戏坐标与录制不同，屏幕坐标相同）。实测（phase8-mouse/FINDINGS.md）：human2b 回放的 ms 流到分歧之后的第 8464 帧都与录制逐字相同、human1 仍 exact；但鼠标**不是** human2b/human2 分歧的原因（第 9 节）。

**CPU 亲和**（phase8-mouse，默认 `--affinity auto`）：多核并行时引擎对 1000.x 环境特效偶发不确定（同条件重复回放，16 次里 4 次在随机帧出现特效分歧或单帧 full 层毛刺），把回放进程限制到 1 个逻辑 CPU 后 16 次 0 次（FINDINGS §H4）。所以 `replay_run.py` 默认给回放进程设 `RR_MT_AFFINITY=0x1`（逻辑 CPU 0；最后一个逻辑 CPU 在本机是 E 核，只有 6.4–6.7×），`rr_mt.dll` 在注入时（游戏主线程恢复之前）`SetProcessAffinityMask`，DLL 日志记 `affinity:` 行，报告 `info.affinity`。`--affinity 0` 不限制，`--affinity 0x…` 指定掩码。代价：加速回放约慢 5–10%（human2b 约 10.4× → 9.5–9.9×）。观战/导出不设（ffmpeg 子进程会继承亲和）。**录制会话也默认设**（recorder 0.3.2 起，`record_run.ps1 -Affinity auto`，`-Affinity 0` 不限制；run header `mt.affinity` 记下掩码（十进制数，如 1；行里不能出现 "0x"+数字），`session.json`/每次启动记录里有 `affinity`，DLL 日志没有成功的 `affinity:` 行算一次启动失败并重试）；0.3.1 及以前的录制没限核，所以其中发生过的这类事件（human2b 第 6911 帧、human2 第 10599 帧）回放无法复现。单核下真人游玩的流畅度未验证。

### 8.1 加速回放 `--turbo`（phase4-turbo）

```
python rl/bridge/python/replay_run.py RUN.jsonl.zst --turbo [gate|clock] --copy experiments/run-recorder/game-rec-03 \
    --init-save <录制会话>/save/run-p1-01-NNNN --report rep.json
```
- 默认关（不加 `--turbo` = 原速，行为与第三阶段相同；录制会话从不加速）。`--turbo` 让 Python 给游戏进程设 `RR_MT_TURBO=1`，`rr_mt.dll` 才额外安装 `rl/bridge/engine/j746_rec/rr_turbo.cpp` 的两个钩子（移植自 `analysis/j746/port-20260930/modified/turbo.cpp`）：`glfwGetTime`（RVA 0x5523B0）返回虚拟时钟、`Manager::Update`（RVA 0x499880）完成后推进 1/60 s，于是外层 60 Hz 调度器不再 Sleep/自旋等真实时间；主循环每次迭代仍是 1 次 Update + 1 次 Render，逻辑帧、tick 结构、渲染次数都不变。
- `gate`（`--turbo` 不带值时的默认）另设 `RR_MT_TURBO_GPU_GATE=1`：不提交 `DrawElements`/`DrawArrays`/`SwapBuffers`（窗口画面不再刷新），CPU 侧渲染、Lua `MC_POST_RENDER`、动画/粒子照常执行。`clock` 只换时钟、画面照常。没有整段跳过渲染（AB+ 上整段跳过会改变结果）。
- `RR_MT_TURBO=1` 但 EXE 不是 J746 或钩子装不上时，DLL 把控制块置为 HookFailed，注入器结束进程（计一次启动失败），不会悄悄按原速跑。DLL 日志里 `turbo:` 行记安装、时钟切换与每 36000 次 Update 的计数。
- 比较口径不变：三层 `dg` + MT。帧行的 `t`（`Isaac.GetTime`，墙钟毫秒）、header 的 `os_time`/`os_date`、`perf`/`cost` 等时间与开销字段在加速回放里不同，但不进 digest、也不参与比较。
- 加速时 stall 判断改为 10 s 无帧推进（原速 25 s）；报告 `info.progress_samples`（约每秒一个 [秒, 帧]）、`logic_fps`、`speedup_x_realtime`、`dll_turbo_log`。
- `--copy` 选隔离副本（默认 game-rec-01）；`--init-save` 把录制会话的开局存档快照（`save/run-...`：persistentgamedata、options.ini）拷进副本存档目录（不拷 `gamestate1.dat`），使解锁/道具池与录制时相同。
- 实测（`experiments/run-recorder/phase4-turbo/ACCEPTANCE.md`）：gate 模式真人局 39,501 帧 81–85 s（约 15–16× 实时，原速约 22 分钟），机器人 eB3L 8,400 帧 23–25 s（11.4–12.1×，换房极多的段落受房间初始化限制），普通短局 eD1 约 24×；clock 模式 eB3L 9.0×。加速回放与原速回放、与录制逐帧比较三层 digest + MT 全部一致（eB3L 两两 0 分歧；真人局前 10,874 帧加速与原速 0 分歧）。
- 真人开 Esc 菜单的帧：见 8.2（原速回放已验证；加速回放 + 菜单暂停注入未测）。

### 8.2 菜单暂停（phase5-pause）

- 现象（`experiments/run-recorder/phase5-pause/REPORT.md`）：真人按 Esc 开暂停菜单的那一帧，录制的 tick 序列是"开菜单前的中间 tick + 暂停期间每个渲染一个 tick（玩家输入冻结，暂停中松键要到解除暂停后的第一个 tick 才出现）+ 解除暂停后固定 2 个中间 tick + 更新 tick"。Esc 落在 `i2`（本应同时做逻辑更新的那次轮询）时，这一帧玩家比不暂停多走一个 60 Hz 子步；落在 `i1` 时与不暂停完全相同（但暂停中松开的键出现在解除暂停后的第一个 tick，旧回放把它折叠到更新 tick，同样会分歧）。结果是"开菜单落在哪个 tick"的确定性函数：同 tick 开菜单、暂停 82 / 622 / 2022 个渲染、期间改不改设置，三层 digest + MT 整局逐帧相同。换房过场、捡道具弹窗、活板门换层这些暂停由引擎自己触发，回放里同样发生，所以一直一致。
- 回放的处理（默认开）：`replay_run.py` 从录制里找"真人菜单暂停"（同一帧 pause→unpause，且 pause 前那个 tick 有实体为 nil 的 `TRIGGERED MENUBACK(15)` 或 `PAUSE(12)` = 1），写进回放表 `pauses`；mod 在同一 (帧, tick) 对 `TRIGGERED MENUBACK` 返回 true 打开菜单（`inputs.lua` 的全局查询覆盖，只在这一处），暂停 10 个渲染后写 `pausectl.txt` 请求，`replay_run.py` 发一次真实 Esc 关闭（菜单内的按键不经过 `MC_INPUT_ACTION`，Lua 关不掉；1.5 s 仍在暂停则重发）；解除暂停后的 tick 按"相对解除暂停"对齐（录制序号 j ≥ U_rec 的输入变化在回放序号 j − U_rec + U_rep 应用），所以回放的暂停长度（约 34 个渲染）与录制不同也不影响结果。回放局 `end.replay` / `status.replay` 记 `pause_opens`、`close_requests`、`pause_unpauses`、`remapped`；报告 `info.menu_pause_esc`、`info.table.menu_pauses`。`--no-menu-pause` 恢复旧行为。
- 引擎自己的暂停也做同样的"解除暂停后相对对齐"（只对齐、不注入；回放表 `pauses[n] = {engine=true, unpause_rs=U}`，`replay` 统计里 `engine_pauses`）：真人提前关掉的道具弹窗（录制 20 个渲染、回放 204 个），以及换层过场在录制里整体晚一个渲染开始的情况（human1 第 30982 帧，录制 i144/i245/i321 对回放 i143/i244/i320，暂停中按住的射击键在录制里于解除暂停后第一个 tick 松开）。`--no-pause-align` 关掉。
- 存档状态：回放默认用副本 `profile/` 里当时的存档，道具池随解锁/存档变化（phase5 实测 phase3 的 eC3 不带 `--init-save` 在第 3801 帧道具 subtype 250 对 139，带录制会话的开局快照则 0 分歧）；回归时建议总是 `--init-save <录制会话>/save/run-...`。
- `compare` 增加 `pause_report`：录制里的菜单暂停（帧、开菜单 tick、暂停渲染数）、回放是否在同一帧暂停；首个分歧帧若正好是回放没有复现的菜单暂停帧，`first_divergence_category = "menu_pause_not_replayed"`（已知类别；录制本身逐帧状态不受影响，只是回放从这一帧起不可比）。`exact` 的口径不变。
- 实测：机器人 15 个菜单暂停局的注入回放全部 `exact`（不注入时 i2 开菜单的 11 局都在暂停帧分歧）；真人局 human1（第 9125、21482 帧两次菜单暂停，`--init-save` 开局快照）原速回放 39,501 帧三层 + MT 0 分歧、`exact`、同帧死亡（以前在 9125 起分歧）。
- 未处理：没有 Esc 按键的暂停（`PauseOnFocusLost=1` 时失焦自动弹出的暂停菜单、控制台）当作引擎暂停不注入，若出现会在该帧分歧（`first_divergence_category = "unknown"`，需要人工看）；回放不重放菜单里改的设置（实测改音效音量不影响逻辑，其它设置未测）。

**含义与局限**：带 DLL 的录制期望 `exact = true`（第三阶段验收：`experiments/run-recorder/phase3/ACCEPTANCE.md`）。录制没有 MT（`-NoDll` 会话、DLL 不可用）时，J746 的全局 MT 不随开局重置（INVESTIGATION §4），泪弹抖动、跟班/蓝蜘蛛初速、环境特效等迟早分歧，回放只能近似，`replay_run.py` 会在 `issues` 里说明。非普通难度、挑战、非 Isaac 角色、`MouseControl=1` 的局列为不可完整回放（`MC_INPUT_ACTION` 无法注入鼠标射击）。录制时加载的其它 mod 若调用 `Random()`，回放时必须同样加载（它们共享同一个 MT）。

### 8.3 观战模式 `--watch`（phase6-watch）

```
python rl/bridge/python/replay_run.py RUN.jsonl.zst --watch --copy experiments/run-recorder/game-rec-03 \
    --init-save <录制会话>/save/run-p1-01-NNNN [--watch-speed 0.25..16] [--watch-paused] [--rewind restart|inproc] [--report R.json]
```
像 MOBA 回放那样看一局录制，画面就是真实游戏画面（窗口模式，不全屏）。需要带观战钩子的 `rr_mt.dll`（`python rl/bridge/engine/j746_rec/build.py --out <副本>/rec`；目前只构建进了 `game-rec-03`）。实测、截图与已知问题见 `experiments/run-recorder/phase6-watch/REPORT.md`。

- **组成**：Python 给游戏进程设 `RR_MT_WATCH=1`、`IRR_HUD=0`（不画 REC），强制 `PauseOnFocusLost=0`、`MouseControl=0`；放 `irr_watch.json`（模式、倍速、时间轴文件）与 `irr_watch_timeline.lua`（层、房间、受伤/Boss/死亡/菜单暂停事件，从录制生成），并把 mod 的 `resources/`（GUI 用的 1 像素贴图）拷进副本的 mod 目录。`watch.lua` 只在回放且有 `irr_watch.json` 时由 `main.lua` 加载；不带 `--watch` 时 mod 与 DLL 的行为不变。
- **速率 = DLL 的虚拟时钟定步**（`rr_turbo.cpp` 的 `watchPace`，Lua 调 `RRW_Set(rate, present_min_ms)`）：每次 `Manager::Update` 开头按墙钟定步（rate × 60 次迭代/秒）。rate 0（暂停）与 0<rate<1 的"空迭代"**整次跳过原 `Manager::Update`**：不轮询输入、不做逻辑/插值、不推进 Manager 计数；虚拟时钟照样推进，渲染与 Lua `MC_POST_RENDER` 照常 60 Hz。J746 的窗口消息泵在 `Manager::Update` 里面，所以空迭代由 DLL 自己泵消息（否则窗口"未响应"、热键收不到）。逻辑看到的迭代序列与不暂停时完全相同，只是两次迭代之间多了几次渲染；`main.lua` 用 `RRW_Info` 的"本迭代被跳过"把这些渲染排除在 tick 分类、渲染计数、鼠标采样之外。不用游戏自己的暂停菜单。16× = 不限速（turbo 上限）；高倍速/快进时每次迭代照常提交绘制，只按墙钟节流真正的 SwapBuffers（60 Hz；快进时 10 Hz）。
- **热键**（游戏窗口聚焦时）：空格 暂停/继续；`[` `]` 或 `-` `+`（含小键盘）变速 0.25/0.5/1/2/4/8/16；`.` 单步一个逻辑帧（自动暂停）；PgDn/PgUp 下/上一个房间（PgUp：进房超过 1.5 s 时回到本房间开头，否则到上一个房间）；Home 回到开局；G 输入帧号、Enter 跳转（Esc 取消）；方向键/WASD 平移镜头，C 回到跟随；H 收起/展开提示；F12 截图（BMP，含 GUI）。热键由 DLL 子类化窗口过程收集（`RRW_Keys`/`RRW_Held`，VK 码），暂停时也有效。
- **控制文件**：`<会话>/lua/watchctl.txt`，每行 `<id> <命令> [参数]`（`goto N`、`speed X`、`pause`/`resume`/`toggle`/`step`、`room next|prev`、`home`、`cam free|follow`、`campos X Y`、`hint on|off`、`shot PATH`、`shotat N PATH`、`quit`）。不带 `--no-console` 时 `replay_run.py` 从 stdin 读命令（纯数字 = 跳到该帧；`s 4` = 4 倍速；`p` = 暂停/继续；`q` = 退出）。状态在 `watchstate.json`，事件在 `watchlog.jsonl`。
- **往前跳** = 不限速跑到目标帧（屏幕显示"正在快进 至 N、剩余约 x 秒"），到了恢复原来的倍速/暂停状态。**往回跳**（默认 `--rewind restart`）：mod 停住、关局（end 行 `reason=rewind`）、写 `watchreq.txt`；Python 结束进程（不经 WM_CLOSE，不写存档），把存档目录里的 `gamestate1.dat` 挪进会话目录，重新拷入开局快照并强制观战选项，然后重开回放进程，从第 1 帧精确重跑，新进程开局即不限速跑到目标帧（屏幕显示"正在倒带 至 N"）。`--rewind inproc`：在同一进程里再发一次 harness 的 `start`（`seed` 命令）开新的回放局，省掉约 24 s 的进程启动；但上一局在本进程内存里改过的存档状态（解锁、统计）不会还原，所以不保证精确（human1 上实测是精确的，见 REPORT）。
- **输入隔离**：玩家 12 个动作由回放覆盖；观战时 `inputs.lua` 把其余所有动作查询（Esc 开菜单、R 重开、Tab 地图、M……，包括同一 tick 的重复查询）一律返回"没按"，与无人值守的回放进程相同。唯一的例外是录制里真人开菜单那一帧照常注入（§8.2）。鼠标：强制 `MouseControl=0`；光标按录制的 ms 流还原，观众的真实鼠标消息被 DLL 丢掉（phase8-mouse，第 8 节"鼠标还原"）。
- **自由镜头**：J746 的 Lua 没有镜头 API（`Room` 没有 `GetCamera`，没有 `Camera` 类，实测）。可写的只有 `Game.ScreenShakeOffset`（屏幕震动偏移），用它平移整个画面，所以游戏 HUD 也会一起平移（观战 GUI 已做补偿，不动）。换房时自动回到跟随；不能缩放。
- **GUI**：用游戏字体（`font/cjk/lanapixel.fnt` 中文、`font/pftempestasevencondensed.fnt` 数字、`font/luaminioutlined.fnt` 小字）和游戏贴图（`gfx/ui/ui_hearts.anm2`、`gfx/ui/minimap_icons.anm2`），加上 mod 自带的 1 像素贴图。底部一行显示层名、房间类型与房号、倍速、帧号与游戏时间；下面是整局进度条：按层分段着色并标 B1/B2/C1…，带房间刻度、受伤红点、Boss 黄点、死亡、菜单暂停标记。另有暂停标志、快进/倒带提示与进度条、跳帧输入框，右侧是热键提示。`lanapixel` 没有"帧"字（字形探针实测），GUI 文案避开了它。不改游戏 HUD。
- **验证**：观战结束后，报告里每个回放进程的回放局都与录制做 `compare`：往回跳之前的那些是前缀，跳帧重跑的是从第 1 帧起的完整重跑。

### 8.4 实录导出 `--capture`（phase6-watch）

```
python rl/bridge/python/replay_run.py RUN.jsonl.zst --capture OUT.mp4 --copy experiments/run-recorder/game-rec-03 \
    --init-save <录制会话>/save/run-p1-01-NNNN [--capture-vargs "<ffmpeg 输出参数>"] [--frames N]
```
- **规格**：视频第 k 帧（从 0 数）= 录制逻辑帧 n = k + 1，取该帧 `MC_POST_UPDATE` 之后、下一次逻辑更新之前的**最后一次渲染**；30 fps。尺寸 = 副本窗口客户区，即 options.ini 的 `WindowWidth`×`WindowHeight`，取自录制会话的开局快照（human1 为 1280×720）。不含观战 GUI 与 REC 标记。旁边写三个文件：`OUT.json`（`isaac-footage-v1`：run、fps、first_frame、frames、width、height、capture_mode、notes、encoder、dups）；DLL 的 `OUT.mp4.dll.json`（写出帧数、首末帧、重复帧数、ffmpeg 退出码）；ffmpeg 日志 `OUT.ffmpeg.log`。
- **做法**：`RR_MT_CAPTURE=1` 时，DLL 在每次 `SwapBuffers` 前用 PBO 环把默认帧缓冲的后台缓冲异步读回（BGRA）。Lua 在每个 `MC_POST_UPDATE` 调 `RRW_Mark(n)`，DLL 把"最近一次读回"作为第 n-1 帧写进 ffmpeg 的 stdin。ffmpeg 由 DLL 自己 `CreateProcess` 启动（`-f rawvideo -pix_fmt bgra -framerate 30 -vf vflip`），默认 `h264_nvenc`（可用时），否则 `libx264`。最后一帧（第 N 帧）在下一次逻辑更新时写出；若之后不再有逻辑更新（human1 以死亡后的结算画面结束，回放局关掉后游戏不再更新），就在它的第 2 次呈现时写出（正常每个逻辑帧 2 次渲染）并关 ffmpeg。两次 Mark 之间没有新渲染时计一次 `dups`。ffmpeg 来自 PATH，没有就用 `imageio-ffmpeg`。
- **为什么是 clock 不是 gate**：gate 不提交绘制，后台缓冲里没有画面。导出时虚拟时钟不限速、每次渲染都提交绘制，只把真正的 SwapBuffers 节流到每 250 ms 一次（窗口里能看到进度），`capture_mode = "clock"`。
- 实测（human1）与校验方法见 `experiments/run-recorder/phase6-watch/REPORT.md`。

## 9. 已知局限

- MT reseed 的时机（第一个 `POST_PLAYER_INIT` 先于开局房实体）只在单人以撒、普通难度、第 1 层开局上验证过（MT_PROBE §5.1、phase3 验收）；其他角色（Jacob&Esau、Lazarus、Tainted）、挑战、贪婪、困难模式：未测。续局早期 reseed 只在单人以撒、Basement II 续局存档上验证过（phase7-continue）。
- 续局真人录制 human2b 的加速回放（钉回出生位置后）从第 6911 帧起分歧：开局房起始位置、前 6910 帧的三层 digest、MT、全部玩家/实体行与外观行都一致，第 6911 帧一只 TINY_BUG 特效（1000.21）贴底墙时速度不同，第 6939 帧起 MT 随之分歧。5 次回放（加速 ×4，含注入 extra 输入；原速 ×1）结果完全相同，未查明录制进程里的哪个隐藏状态不同（phase7-continue/REPORT §3）。
  phase8-mouse（`experiments/run-recorder/phase8-mouse/FINDINGS.md`）逐一实测排除了鼠标（真实/PostMessage/DLL 还原录制的 ms 流，回放的 ms 与录制逐字相同仍在 6911 分歧）、窗口尺寸、焦点与 PauseOnFocusLost、Lua/引擎共用堆的布局、MXCSR/x87 控制字；并发现**引擎对 1000.x 环境特效（TINY_BUG、WISP、POOP_PARTICLE 等）本身偶发不确定**：同一条件的重复回放里，偶尔有一次在随机帧出现特效速度/位置不同（human2b 回放第 3400 帧 WISP、第 3686 帧 POOP_PARTICLE），另有只持续一帧、0.01 定点下看不出的 full 层 digest 毛刺（第 390、672、696、2220 帧等）。human2b 第 6911 帧与 human2 第 10599 帧的分歧是录制进程里发生的同类偶发事件，回放进程绝大多数时候不发生，所以看起来"稳定"。这类事件不由任何录制下来的输入决定，回放无法复现；它多数只影响外观特效（full 层），偶尔经特效的后续行为改变全局 MT 消耗而影响玩法（human2b 第 6939 帧）。
- 内建控制台命令（`giveitem`、`spawn`、`debug`…）抓不到（`MC_EXECUTE_CMD` 不触发），录制会话强制 `EnableDebugConsole=0`，会话结束把 `cmd_history.txt` 新增行写进 session（D8）。
- 解锁/收集无法从 Lua 读（`Isaac.GetPersistentGameData` 不存在），只能靠存档快照。
- 手柄模拟量、多人、Jacob&Esau 只做了格式支持，未实测。
- 网格新增/墙变门最多晚 15 帧记录；道具数与主动槽都不变的道具替换（例如 D4 重随）最多晚 300 帧（下一个关键帧）出现在 `items` 行。

## 10. 最小 DLL、salt 与 `-NoDll`（第三阶段，E1–E8）

**组成**（源码与构建脚本 `rl/bridge/engine/j746_rec/`，来自 `experiments/run-recorder/mt-probe/src/` 的最小版，去掉了 OpenAL 与 trace）：
- `rr_mt.dll`：① 挂 `userenv!GetUserProfileDirectoryA`，返回环境变量 `RR_MT_PROFILE`（= `<副本>\profile`），存档因此落在副本内；② 导出 Lua C 函数 `RRMT_Seed(seed)`（调用引擎自己的 `GlobalMT_Seed`，RVA 0x2BA260）、`RRMT_State()`（只读 MT 的 [cursor, FNV-1a32(624 字 + cursor)]，不推进；状态拷两份一致才返回，第三个返回值是读的次数，不一致时把不同的字记进 DLL 日志 `unstable MT read`）、`RRMT_Info()`、`RRMT_UnstableReads()`（进程内两份不一致的次数，写进 `end.mt_unstable_reads`）；③ 建 `IsaacTurbo.<pid>` 控制块供注入器判断。身份门：PE 时间戳 0x624393F6、SizeOfImage 8540160、两处 MT 指令的立即数；**身份门只管 MT 导出**：不通过时存档重定向照常安装、游戏照常运行，`RRMT_Seed/RRMT_State` 不返回值（E7）。没有任何其它钩子。测试开关 `RR_MT_TEST_IDENTITY_FAIL=1` 假装身份门不通过。
- `rr_mt.dll` 的鼠标导出（phase8-mouse，只给回放用）：环境变量 `RR_MT_MOUSE=1` 且 J746 身份门与 4 处代码签名（RVA 0x4014C6/0x4014F4 Lua `GetMousePosition` 读 GLFW 窗口 +0x48、0x499D6E/0x499DD1 `Manager::Update` 的光标闲置计数）通过时，`RRMT_SetMouse(x, y, buttons, mode)` 写 `[base+0x7A15D8]`（GLFW 主窗口）的 +0x48/+0x50（double 虚拟光标）与 +0x5C（按键 char[8]）；mode 1 同时改 Manager（`[base+0x7C768C]`）+0x1EA970/974 的"上次光标位置"使这次改动不触发闲置计数，mode 2 再把 +0x1EA978 的计数清 0；第一次调用时子类化窗口（+0x218 的 HWND）丢掉 `WM_MOUSEFIRST..WM_MOUSELAST`、`WM_MOUSELEAVE`。`RRMT_GetMouse()` 返回 x、y、按键、闲置计数、上次位置、丢掉的消息数、设置次数。录制会话不设 `RR_MT_MOUSE`，这两个函数不返回值，进程里没有子类化。环境变量 `RR_MT_AFFINITY=<掩码>`（十进制或 0x）时 DLL 初始化即把整个进程限制到这些 CPU（回放与带 DLL 的录制会话都默认设 0x1）。诊断导出 `RRMT_Fpu(mxcsr, x87cw)`（-1 = 不改；返回并记日志当前 MXCSR 与 x87 控制字），只在 phase8 的实验脚本里用。
- `rr_launch.exe --launch <dll> <exe> --luadebug`：以挂起方式建进程、注入、等控制块 Active 后才恢复主线程（存档钩子在游戏第一次查询用户目录前就绪），打印 `PASS launched pid=<pid>`；窗口不隐藏。
- 构建：`python rl/bridge/engine/j746_rec/build.py`（zig 0.16 x86，MinHook 从 `third_party/minhook` 源码编译），产物在 `<副本>/rec/`（`rr_mt.dll`、`rr_launch.exe`、`build.json`），不进 git（`rl/.gitignore` 排除 `*.dll *.exe *.pdb *.lib`）。

**会话接线**（`run_recording.py`）：默认 `Session(use_dll=True)`：环境变量 `RR_MT_PROFILE`、`RR_MT_LOG_DIR`（= 会话 `game/`，DLL 日志 `rr_mt.<pid>.log` 写在这里）、`IRR_MT_DLL`（mod 用 `package.loadlib` 取导出函数的路径）；注入器失败（退出码非 0 / 没有 PASS 行）与进程崩溃、log 洪水、mod 未加载一样计一次启动失败并自动重试。

**reseed 规则**（`mt.lua` + `main.lua`，E3）：
- 新局：第一个 `MC_POST_PLAYER_INIT` 且 `Game():GetFrameCount()==0`（开局房的墙虫/小飞虫等实体初始化之前）调用 `RRMT_Seed(StartSeed xor salt)`；`MC_POST_GAME_STARTED` 回调最开头用**同一个** salt 再调一次。只在 `POST_GAME_STARTED` reseed 不够（MT_PROBE §5.1）。
- 续局（`isContinued=true`，recorder 0.3.1 起，phase7-continue）：读档后第一个 `MC_POST_PLAYER_INIT`（此时还没有打开的局、`GetFrameCount()` = 存档帧数 > 0、StartSeed 已是存档的种子；在进开局房与选出生位置之前）用新 salt reseed，`MC_POST_GAME_STARTED` 用**同一个** salt 再 reseed 一次（续局状态来自 `gamestate1.dat`，MT 不在存档里）。实测（phase7-continue/REPORT）：续局时 PLAYER_INIT 在 MT cursor 48、进房（`POST_NEW_ROOM`）在 cursor 54、`POST_GAME_STARTED` 在 cursor 55，出生位置在后两者之间决定；只在 `POST_GAME_STARTED` reseed（0.3.0 及以前）时出生位置随进程随机，加上早期 reseed 后同 salt 每次相同、不同 salt 不同。`IRR_MT_EARLY_CONTINUE=0` 退回 0.3.0 规则。0.3.0 及以前的续局录制（如 human2b）回放时没有这个时机（不 reseed），出生位置靠回放表 `start_players` 钉回（第 8 节）。
- salt：每局随机 32 位非 0 整数 = `os.time`、`os.clock`、`Isaac.GetTime`、`Isaac.GetFrameCount`、计数器、一个新表地址的 `string.pack` 字节的 FNV 哈希；不碰游戏 RNG，不用 `math.random`。全部写进 `header.mt`（4.1）；整数照原样写，不用 `string.format`。
- 回放局：用录制 `header.mt.reseeds` 里同一时机的 salt；录制里没有的时机不 reseed。
- E7：没有 `IRR_MT_DLL`（`-NoDll`）、`package.loadlib` 失败、身份门不通过时 `header.mt.available=false`（`reason` 写原因），不 reseed、帧行不写 `mt`，其余照常录制。

**`-NoDll`**（`rl/record_run.ps1 -NoDll`，`run_recording --no-dll`）：完全回到第二阶段行为：直接启动 EXE，存档在真实 `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance (Galaxy)\`（不存在时由游戏新建），header `mt.available=false, reason=no_dll`，回放只能近似。

**观测模式**（E5）：默认 `full`；`rl/record_run.ps1 -Obs keyframe`（环境变量 `IRR_OBS=keyframe`）只在关键帧写实体行，其余帧照写 `mt`、玩家行、tick 输入、生成种子、房间头、网格/门变化与 digest。适合带 DLL、可精确回放的会话；实体状态需要回放重建。目前不默认。

## 11. 查看器：`rl/bridge/python/run_viewer/`（帧级回放，不启动游戏）

完全从录制文件重建每一帧的画面：房间俯视图（网格/门/实体/玩家）、可拖动的整局进度条与当前房间进度条、逐帧步进、播放与倍速、事件标记跳转、每层/每房间跳转列表、当帧数值面板（玩家、输入、道具、实体字段、MT、digest）。四种视图（底栏下拉框，`M` 切换，URL `#mode=geo|art|video|split`）：
- **几何**：原来的示意画法。
- **美术**（phase6-art，有解包素材时默认）：`static/viewer_art.js` 的 anm2 渲染器（Canvas 2D：spritesheet/层/根动画/关键帧插值、裁剪、轴心、缩放、旋转、乘色与加色偏移、可见性）按录制的 动画名+帧号 画每个实体（v2：FlipX/Y、Scale、Rotation、实体色×精灵色、叠加动画、PositionOffset 离地高度）；房间按 `backdrop` 类型拼墙与地板（1x1 用四象限镜像，其它形状按格铺墙带与地板、L 形缺角涂黑与内角；墙面变体 Lua 读不到，默认用无装饰的第 0 号象限）、开局房的操作说明地画、网格实体（v2 用录制的 anm2/动画/帧；石头/坑/门按 `backdrops.xml` 换关卡贴图）、门（v2 用录制的门精灵；普通门 anm2 按房间类型换贴图：商店/献祭/骰子/箱子房；未炸开的隐藏门不画）；玩家 = `001.000_player.anm2` 换 `players.xml` 皮肤 + 道具服装（`costumes2.xml`，按道具 id 自动映射）；HUD（`ui_hearts`、`hudpickups`、主动道具图标、充能槽、饰品、`minimap1`/`minimap_icons` 小地图、Boss 血条，`H` 开关）。v1 文件（如 human1）用 `entities2.xml` 的 anm2 与录制里已有的动画名/帧，翻转/颜色/叠加帧/网格变体为估计（画面底部提示）。
- **实录**：会话目录下有 `footage/<run>.mp4` + `<run>.json`（`{"schema":"isaac-footage-v1","fps":30,"first_frame":1,"frames":N,...}`，视频第 k 帧 = 逻辑帧 first_frame + k）时可用：`<video>` 按当前帧 seek 到 `(n - first_frame + 0.5)/fps`，拖动进度条实时跟随；播放时由视频时间驱动帧号（倍速 = `playbackRate`，>16× 退化为逐帧 seek）。**美术+实录** 并排。
```
cd rl/bridge/python
python -m run_viewer RUN.jsonl.zst                      # 本地 http 服务 127.0.0.1:8796 并打开浏览器（素材按需从磁盘只读）
python -m run_viewer RUN.jsonl.zst --html OUT.html      # 自包含 HTML（数据 + 本局用到的素材内嵌；实录视频用 file:// 引用）
python -m run_viewer RUN.jsonl.zst --footage X.mp4      # 指定实录视频（同名 .json 为元数据）
python -m run_viewer.verify RUN.jsonl.zst               # 查看器展开结果与 load_run 逐字段比对（需要 node；v2 含外观行与网格/门精灵）
```
选项：`--assets DIR`（解包素材的 resources 目录，默认 `experiments/run-recorder/assets/rep-extracted/resources`）、`--no-art`、`--port`（默认 8796）。
键位：空格 播放/暂停，←/→ 单帧（Shift = 30 帧），↑/↓ 上/下一个房间，[ / ] 上/下一个事件，+/- 倍速，E/L/V/G 切换特效/标签/速度箭头/格号，`M` 视图，`H` HUD，`F` 跟随玩家，`C` 游戏镜头（视野 = 游戏 480x270 画面、跟随玩家并夹在房间内，规则为估计），`R` 复位整房视角；画面上滚轮 = 以鼠标为中心缩放、拖拽 = 平移、Alt+滚轮 = 逐帧；进度条上滚轮 = 逐帧；URL `#f=<帧号>` 直接定位。
展开逻辑（`static/viewer_core.js`）是 `load_run.py` 的逐行移植，在浏览器里按关键帧段懒展开并缓存。坐标系、门槽位置表、验证与性能数字见 `experiments/run-recorder/viewer/REPORT.md`，美术模式见 `experiments/run-recorder/phase6-art/REPORT.md`。
