# L1 原版加速：可行性实验方案与控制层实现（2026-09-19）

用户 2026-09-19 提出的目标：训练器驱动 N 个"原版 worker"，每个 worker = 原版玩法 + 控制层，提供 `reset(房间, Boss, 道具, 种子)`、`step(合法动作, 逻辑帧数)`、`observe()`、`render()`；难点是"加速而不改变玩法"。本文把用户列出的五条"不能想当然"逐条对到 J460 的静态证据上，给出控制层的实现，并定义三阶段可行性实验及通过判据。

状态：控制层三件（桥接 mod、`isaac_turbo.dll`、Python 控制端与实验脚本）已写出并通过**离线**验证；**一行都还没在真实游戏进程里跑过**。实机需要用户完成 §5 的三步准备。地址均为 RVA（VA = RVA + 0x400000），证据文件见 [J460_FRAME_LOOP.md](../../analysis/docs/J460_FRAME_LOOP.md)。

## 0. 结论

1. 用户设想的 worker 架构可行，而且比"从零翻译几百种怪物"便宜得多：控制层只改**节拍来源**（时钟）和**是否绘制**，不碰 `Game::Update` 内的任何玩法状态。
2. 静态证据支持"加速不改玩法"：`Game::Update`（`0x002FADC0`）子树不调用 `glfwGetTime`；游戏逻辑里唯一的真实时间依赖是开局淡入的 ≥3 s 保持（§2.5），它用原始 QPC，加速不改变它的语义，只是每次重置都要多等 3 s。
3. "跳渲染 ≠ 跳动画更新"这一条用调用图核实了：动画推进函数从各实体的 `Update` 和 `Room::Update` 可达，渲染路径里只有 5 个 HUD/结算类函数推进动画（§1.2）。
4. 单步同步不需要原生接口：桥接 mod 在 `MC_POST_UPDATE` 里阻塞，主线程停在 `Game::Update` 之内，`step(a, N)` 恰好推进 N 个逻辑帧，等待期间连 `Manager::Update` 都不跑（可用 DLL 计数器证明）。
5. 预期成本（静态推算，待实机）：NetFix 2026-09-02 遥测里 `Game::Update` 均值 293–315 µs，纯逻辑上限约 3000 逻辑帧/秒（100× 实时）；加上桥接每步的 JSON 编码，预计 20–50×；重置 ≥3 s。

## 1. 五条"不能想当然"逐条对证

| 用户的判断 | 证据 | 本实现的做法 |
|---|---|---|
| 1 不能简单循环调用 `Game::Update()` | Isaac-RL README 原话："An experiment invoking `Game:Update()` between renders changed player displacement per update; that approach was removed."（随包 Lua 文档确有 `Game::Update`）。J460 里 `Manager::Update`（`0x00554CD0`）在调 `Game_Update` 之前做两件事：按 `Manager+0x4ABBC` 奇偶只在偶数迭代模拟；对每个带延迟提交标志（`+0x175`）的实体把 `+0x344/+0x348` 提交到 `+0x33C/+0x340`。从 Lua 直接调 `Game():Update()` 跳过这两步，位移自然不同 | 外层循环原样跑，只 Hook 时钟与渲染；不直接调用 `Game_Update` |
| 2 跳渲染不等于跳动画更新 | 动画推进 `AnimationState::AdvancePosition`（`0x00008D00`，经 `FUN_00409030` `0x00009030`）从 `Entity::Update 0x002AE820`、`Entity_NPC::Update 0x002C4B30`、`Entity_Player::Update 0x00382AF0`、`Entity_Tear::Update 0x002670F0`、`Room::Update 0x00402980` 可达（调用图 + 51 个实体类虚表根）；只从渲染根可达的推进者有 5 个：`0x00030750`、`0x0024B6C0`、`0x004EBA50`、`0x004EBDE0`（GameOver 结算画面附近）、`0x0044C2C0`（HUD 附近），未命名 | 只跳整帧渲染函数 `0x005555C0`；这 5 个函数列为实机核对项（阶段 2 的事件计数会暴露差异） |
| 3 隐藏窗口不等于无图形依赖 | 渲染函数内部含 `GetWindowLongA`、GL 目标切换；窗口由 GLFW 3.4 持有 | 保留窗口与 GL 上下文；渲染被跳过时 `SwapBuffers` 不再执行，VSync 不再限速 |
| 4 加速应是更快执行相同逻辑帧，不能每读一次时钟就推进 | 外层循环（`0x00531050`）每迭代读 3 次时钟（顶部 `0x931231`、限帧 `0x931361`、自旋 `0x931438`），平台层再读 1 次；限帧公式 `ms = (1/60 − elapsed) × (−1000)`，自旋 `while (1/60 > now − frame_start)` | 虚拟时钟只在 `Manager::Update` 返回时推进一步（1/60 s × (1 + 2⁻¹⁶)），读多少次都不动；`tests/clock_test.cpp` 复刻限帧算术验证 Sleep 恒为 0、自旋一次退出 |
| 5 重置不是恢复几个坐标 | NetFix `STATUS.md`：玩家运动中的选择性恢复已证伪；裸 `Game::RestoreState` 作为回滚原语已证伪 | 首版重置 = `restart` + `seed` + `goto`（新局 + 场景重建），不做任意帧快照 |

## 2. 静态证据（本轮新增）

### 2.1 时钟函数的身份、返回约定与调用者

`FUN_00a266a0`（`0x006266A0`）就是 GLFW 3.4 的 `glfwGetTime`：未初始化时报错 `0x10001` 返回 0.0，否则 `(QueryPerformanceCounter − _glfw.timer.offset) / frequency`，`offset` 在 `0x00874BA8`、`frequency` 在 `0x00874BB0`。**返回值在 XMM0**（LTCG 自定义约定，不是 x86 标准的 ST0）：6 处调用点（用 `E8 rel32` 全 `.text` 扫描得到）紧接 `movsd/cvtsd2ss xmm0`。Ghidra 的 `calls.csv` 只列出 3 个调用者（`FUN_00a616c0`、`FUN_00a714e0`、`FUN_00a8c430`），外层循环的 3 处因函数切分问题缺失。

调用者全部在外层循环、平台层（输入设备 delta）和 Steam 初始化；`Game::Update` 子树没有。因此 Hook 它只影响节拍。

### 2.2 帧循环的奇偶与绘制

`Manager::Update`：奇数迭代且 `Manager+0x2A3C0 == 0` 时只把计数器加一就返回；偶数迭代做延迟提交并调 `Game_Update`。渲染函数 `0x005555C0` 开头 `test byte [Manager+0x4ABBC],1; jne 绘制; cmp byte [Manager+0x2A3C0],0; je 直接返回`。`+0x2A3C0` 全程序只有 4 处比较（`0x00018DBD`、`0x00554D0F`、`0x00555238`、`0x00555610`）没有写入，按 0 处理。于是单机 J460 = 30 Hz 逻辑、30 fps 绘制、无插值；绘制紧跟在 `Game_Update` 之后的同一迭代。

渲染之后的 `FUN_008657c0`（`0x004657C0`）是 Lua GC 步进（`lua_gc` 按最近 10 帧内存增量调步），每迭代一次，与加速无关。

### 2.3 限帧常量

`DAT_00baa498 = 1/60`（double）、`DAT_00baad98 = −1000.0`、慢帧阈值 `1.1`（`0x00baa5e0`）、连续 30 次判定切换软件限帧；`DAT_00c798e4 & 0x200` 置位（VSync）时跳过软件限帧。虚拟时钟下每帧 `period/delta ≈ 0.99998 ≤ 1.1`，软件限帧不会被误开；即使开了，`elapsed ≥ period` 也使 Sleep 为 0、自旋立即退出。

### 2.4 逻辑路径里的墙钟、`rand`、`Sleep`（调用图 + 虚表根，含 `Game_Update` 与 51 个实体类的非渲染槽）

| 外部函数 | 逻辑可达的调用者 | 用途 | 影响 |
|---|---|---|---|
| `QueryPerformanceCounter` | `FUN_00a68490`（`0x00668490`，返回纳秒） | `Game_Update` 第 366–410 行的开局保持（§2.5）；KAGE 互斥体超时 | 见 §2.5 |
| `_time64` | `FUN_009292c0`（`0x005292C0`） | `save_backups/` 文件名里的日期 | 无 |
| `rand` | `FUN_006eef60`（`0x002EEF60`） | 全局 MT19937"未播种 RNG"，只在调试标志 `DAT_00c7ac68` 置位时才用 CRT `rand` | 同种子同输入的两局仍可能在未播种随机数上分歧（阶段 2 需 A/A 基线） |
| `Sleep` | `FUN_00a157f0`、`FUN_00a23ae0` | KAGE 互斥体争用时 `Sleep(1000)` | 单线程逻辑不触发 |

### 2.5 开局淡入的真实时间保持

`Game_Update` 第 366–410 行：当 `Game+0x26598 > 0`（淡入计数，`FUN_006f5210` 开局置 1.0）、单机且 `Game+0x269E8 == 0` 时，取 `FUN_00a68490()` 纳秒换算为毫秒，若距 `Manager+0x4AE20`（开局时间戳，`FUN_00951f60` 写）不足 `0xBB9 = 3001` ms 则提前返回，之后每帧按 `Game+0x265B8` 递减到 0 才开始模拟。这是**唯一进入玩法逻辑的真实时间**：每次 `restart` 至少等 3 s，加速不缩短它。要消除只能 Hook `FUN_00a68490` 或改写 `Manager+0x4AE20`，留到阶段 3 量出重置成本后再决定。

### 2.6 社区证据

Isaac-RL（Repentance 1.7.9b）：6 个游戏进程并行、每动作 8 帧、正常速度，单实例 3.6 动作/秒、六实例 20.3 动作/秒；作者尝试在渲染间额外调 `Game:Update()` 后发现位移改变并撤回。它证明了多实例与 Lua 同步单步可行，也证明了"额外调 Update"的路走不通。

## 3. 控制层实现

```text
训练器（Python）
  └─ worker i = isaac-ng.exe --luadebug（pid_i，端口 27015+i）
       ├─ IsaacRLBridge mod 0.2.0（rl/bridge/mod）  reset/step/observe：TCP，MC_POST_UPDATE 阻塞同步，MC_INPUT_ACTION 注入
       ├─ isaac_turbo.dll（rl/turbo）               时钟/渲染/计数：命名共享内存 IsaacTurbo.<pid>
       └─ Python：IsaacBridgeEnv + TurboControl（rl/bridge/python/isaac_bridge）
```

| 部件 | 位置 | 说明 |
|---|---|---|
| 目标地址表 | [rl/turbo/j460_targets.json](../turbo/j460_targets.json) → 生成 `src/generated/j460_targets.hpp` | 4 个函数的 RVA 与序言字节、GLFW 计时器全局、Manager 指针与计数器偏移；`scripts/verify_targets.py` 只读核对 exe |
| 虚拟时钟 | [rl/turbo/src/virtual_clock.hpp](../turbo/src/virtual_clock.hpp) | `read()` 不推进、`tick()` 推进一步；开/关切换单调续接（关闭时 `real + offset`，绝不回跳，回跳会让游戏 `Sleep` 十几分钟或自旋等真实时间追上） |
| DLL | [rl/turbo/src/turbo.cpp](../turbo/src/turbo.cpp) | 身份门（PE 时间戳 `0x69E6E3A7`、`SizeOfImage 0x0092C000`、4 段序言、GLFW 频率 == QPF）→ 一次性安装 4 个 MinHook：`glfwGetTime`（naked 桩，结果装入 XMM0）、`Manager::Update`（tick + 计数 + 计数器自检）、`Game::Update`（计时，<20 ms 的调用进 fast 桶）、渲染（按 flags / `render_every` / `render_request` 决定，且只在游戏本会绘制的调用上放行）。日志写 `%LOCALAPPDATA%\IsaacRL\turbo\turbo-<pid>.log`，不写游戏目录 |
| 控制块 | [rl/turbo/src/turbo_shared.hpp](../turbo/src/turbo_shared.hpp) ↔ [turbo.py](../bridge/python/isaac_bridge/turbo.py) | 4 KB：flags（虚拟时钟、跳渲染）、`render_every`、`render_request/done`、状态与计数器；DLL 每次 Hook 调用直接读 flags，可在线切换 |
| 注入 | [rl/turbo/src/injector.cpp](../turbo/src/injector.cpp) | x86 `CreateRemoteThread + LoadLibraryW`，与 NetFix 注入器同源 |
| Python | `isaac_bridge.turbo.launch_turbo / attach / TurboControl` | 默认直通模式启动，进入一局并连上桥接后再 `set_turbo(True)`；`request_render()` 即用户设想的 `render()` |
| 实验脚本 | [rl/bridge/python/feasibility/](../bridge/python/feasibility/) | `stage1_control.py`、`stage2_equivalence.py`、`stage3_cost.py`，共用 `common.py` |

不做的事：不改任何玩法字段；不从原生注入输入（仍走 `MC_INPUT_ACTION`）；不做快照；与 NetFix 的 DLL 不同时加载（两者都 Hook `Game::Update`/`Manager::Update`）。

## 4. 三阶段实验协议

| 阶段 | 检查 | 方法 | 通过判据 |
|---|---|---|---|
| 1 可控 | A 重置可靠 | 同 seed/房间连续 `reset` 两次 | 玩家位置、实体表逐项相同；事件计数清零 |
| | B 步进精确 | `step(noop, N)`，N ∈ {1, 2, 4, 8, 30} | `Game():GetFrameCount()`、`Room:GetFrameCount()`、mod 的 `logic_frames` 三者增量都恰为 N |
| | C 不偷跑 | 不发指令空等 3 s，再 `obs` 查询 | `game_frame` 不变；DLL 的 `manager_update_calls`、`game_update_calls` 不变 |
| | D 加速下重复 B、C | `set_turbo(True)` | 同上；并预览 300 帧吞吐 |
| 2 加速正确 | A1/A2/B 三条轨迹 | 同 seed、同房间、同脚本动作序列，A1、A2 正常，B 加速；逐步比较玩家位置、实体表（含眼泪高度）、心数、受伤/发射/敌死/清房计数 | A1 == A2 时要求 A1 == B 逐项一致；A1 ≠ A2（未播种 RNG）时 B 的首个分歧步不得早于、位置差不得大于 A/A 基线，否则 FAIL；只能给"不劣于基线"时记为 INCONCLUSIVE 并保留三条轨迹 |
| 3 真实成本 | 吞吐 | 加速开，repeat=1 与 repeat=8 各跑 3000 帧 | 逻辑帧/秒、决策/秒、`Game::Update` fast 桶均值 |
| | 重置 | 5 次 `restart+seed+goto` | 均值/最大（预期 ≥3 s，§2.5） |
| | 内存与扩展 | 1 → 2 → 4 实例并行（同一用户档案，端口递增） | 总帧/秒随实例数的比例、RSS 总量、报错数 |

数字全部落到 `rl/runs/l1/<时间戳>-stageN/*.json`，跑完后回填到本文 §6。

## 5. 运行步骤

准备（2026-09-19 17:33 已按用户要求完成前两步）：

1. 已在 `D:\Steam\steamapps\common\The Binding of Isaac Rebirth\mods\isaac_rl_bridge` 建立**目录联接**（Junction）指向 `rl/bridge/mod/isaac_rl_bridge`；游戏目录里没有复制任何文件，删除该联接即可还原。这是用户明确授权的唯一游戏目录改动。
2. `Documents\My Games\Binding of Isaac Repentance+\options.ini` 已改 `EnableMods=1`、`PauseOnFocusLost=0`（改前备份在 `rl/runs/l1/backup/options.ini.20260919-173336`，逐行比对只有这两处不同，CRLF 保持）。`EnableDebugConsole=1` 原本就是；`VSync=1` 未动。
3. Steam 已登录（用户手动启动游戏）。

构建（已通过）：

```powershell
pwsh -NoProfile -File D:\Projects\fortune\Isaac\rl\turbo\build.ps1
```

手动启动（用户操作）。必须带 `--luadebug`，否则 mod 里的 `require("socket")` 失败；端口不设环境变量时默认 27015：

```powershell
# 直接启动（Steam 已登录即可；等价于 python -m isaac_bridge.launch --port 27015 --privileged）
$env:ISAAC_RL_PORT = '27015'; $env:ISAAC_RL_PRIVILEGED = '1'
Start-Process -FilePath 'D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe' -ArgumentList '--luadebug' -WorkingDirectory 'D:\Steam\steamapps\common\The Binding of Isaac Rebirth'
```

或者在 Steam 里给游戏的启动选项填 `--luadebug` 后从 Steam 启动。启动后 `Documents\My Games\Binding of Isaac Repentance+\log.txt` 应出现 `[IsaacRLBridge] listening on 127.0.0.1:27015`；若出现 `luasocket unavailable`，说明没传 `--luadebug`。

第一次必须手动进入一局（菜单自动化未做）；之后所有重置走控制台命令。在 `rl/bridge/python` 目录下，先取 pid 再附着并注入：

```powershell
$pid_ = (Get-Process isaac-ng).Id
python -m feasibility.stage1_control --pid $pid_ --inject --port 27015 --room "goto d.12"
python -m feasibility.stage2_equivalence --pid $pid_ --port 27015 --room "goto d.12" --steps 150
python -m feasibility.stage3_cost --pid $pid_ --port 27015 --frames 3000 --resets 5      # 单实例
python -m feasibility.stage3_cost --launch --instances 1,2,4 --frames 3000 --resets 5    # 多实例由脚本启动
```

种子默认 `--seed-cmd auto`：第一次 `restart` 后读取游戏生成的起始种子串（带校验，`seed` 命令一定接受），之后每次重置固定用它；也可显式给 `--seed-cmd "seed XXXX XXXX"`。`goto d.N` 进入调试房。注入后 DLL 日志在 `%LOCALAPPDATA%\IsaacRL\turbo\turbo-<pid>.log`。

## 6. 已验证与未验证

离线已验证（2026-09-19）：

- `clock_test`：10 万迭代 × 3 组真实时间基准（0 s、1234 s、3×10⁷ s）Sleep 恒 0、自旋 0 次、读不推进、开关切换单调；限帧算术与反汇编一致（半帧提前 → Sleep 7 ms；回跳 1000 s → Sleep 1000015 ms，证明"回跳"是必须防的）。
- `probe_host`：非游戏宿主加载 DLL 被身份门拦住（`ERROR:identity pe timestamp=…`），不装 Hook，控制块与日志正常建立。
- `test_turbo_control.py`：Python 端附着到探针的命名共享内存，读到状态，写入 flags/render_every/render_request 被 DLL 侧回显 `flags=0x3 render_every=7 render_request=1`，布局一致。
- `verify_targets.py`：exe 上 4 段序言字节与 PE 身份全部匹配。
- DLL 里 `glfwGetTime` 桩的机器码确认为 `call; movsd xmm0,[g_turbo_now_value]; ret`。

实机更新（2026-09-19）：桥接握手、定帧步进、无指令不推进、死亡后重置已运行。原 Python `connect()` 未消费服务端初始观测，已通过正式入口 `isaac_bridge.training.IsaacTrainingEnv` 修正并实机验证；旧 feasibility 脚本仍使用旧入口，不能直接复用其结论。Turbo 曾短测约 508 逻辑帧/秒，但加速等价性未通过验收。`goto d.12` 即使固定整局种子也改变房间 SpawnSeed/出生点；自然房间在固定 SpawnSeed 下仍有 A/A 轨迹差异。

首个正式课程已按用户确认固定为 Isaac、零道具、Monstro、空地形；实现和命令见 [bridge/README.md](../bridge/README.md)。隔离另外 8 个 Mod 后，25 次建场与 12600 个逻辑帧通过，实机生成的 Monstro HP 为 312.5。**续跑纠正**：此前通过的 PID 30604 后来仍崩溃，新启动 PID 13508/11008 也在进入战斗前崩溃；三份转储均为 RVA 0x61C423 读取 0xC，且没有 Turbo/NetFix/REPENTOGON 模块。停用 Mod 不是已验证修复。Windows 锁屏阻挡了本轮菜单操作，尚不能认定锁屏是 shader 崩溃根因。

续跑已通过：11 项离线测试；三次自然死亡后重置；真实 4+4+1 帧超时截断；显式 NPC Kill 的清房正向协议测试；固定场景 512 次真实 PPO 动作、2048 战斗逻辑帧、16 个训练 epoch、checkpoint 保存/重载并再次驱动游戏。参数 L2 变化 0.3841785377871189，总耗时 74.8268 s，约 6.84 决策/s（含重置/训练/保存）。所有已结束的学习回合仍是死亡，尚无策略 Boss 击杀成绩。

原生 `--set-stage=1` 已实际跳过菜单开局，但其测试启动模式使旧 `seed` 控制台重置静默使用随机种子。正式入口现用 `Seeds:SetStartSeed` + `stage 1`，逐次读取实际 seed 验证。固定场景为 seed `9AM0 7PRP`、room seed `2979258163`、stage_type=0、Monstro HP312.5。解锁后的冷启动出现了归档流空指针（PID29048 RVA 0x668CB5；PID22060 RVA 0x668CD6）；成功进程 PID27476 跑完训练后于 22:04:37 再次出现 Shader stack empty，转储为 RVA 0x61C423、读取 0xC。崩溃尚未修复。替代进程 PID48584 完成实机种子回归：旧代码实际 8P1S BNCM，修复后 9AM0 7PRP，回退后 82AG Y2RE，再应用修复后恢复 9AM0 7PRP。首次回归因游戏已崩溃而连接拒绝的错误单独保留，不计为预期负向结果。证据：`rl/runs/l1/20260919-training-loop/resume-20260919-214224/`。下一优先级是 worker 崩溃诊断与生命周期稳定性，不是重复证明 PPO 已接通。

仍待验证：持续稳定性、Turbo 开关/跳渲染的安全性与玩法统计一致性、多实例存档/日志隔离、`render_every` 预览，以及重置计时与完整训练吞吐。

## 7. 风险与后续

- **存档隔离**：`savedatapath.txt` 只是信息文件；多实例共用 `Documents/My Games/Binding of Isaac Repentance+/`，Isaac-RL 也是这样跑的。要隔离需 Hook 路径拼接或用多个 Windows 用户。
- **3 s 开局保持**：占重置成本大头；阶段 3 量出后决定是否 Hook `FUN_00a68490`。
- **菜单自动化**：NetFix `MenuAutomation.cpp` 已有从标题进入一局的按键链，可移植；首版靠人工开一局。
- **观测成本**：JSON 每步 5–20 KB；若阶段 3 显示桥接是瓶颈，改为按需 `grid` 或二进制观测。
- **未播种 RNG**：阶段 2 若 A/A 不一致，说明只能做统计等价，训练不受影响，但校准比对要按分布做。
- **与 L2 的关系**：L2 保留为课程子集与校准工具；当前先稳定原引擎训练 worker，再扩展战斗数据。

## 2026-09-20：无渲染训练与可视化验收

用户已确认默认关闭训练 worker 的绘制，另保留可视化验收。正式入口新增 `--render-mode headless|visible`（默认 headless），复用已有 J460 控制层，只设置 skip-render，不开启虚拟时钟。已有 worker 用 `--pid`；新 worker 用 `--launch`；连接超时默认30秒，启动失败不自动重试或计为一场死亡。

实测无渲染20回合规则控制器压力测试完成，350.61秒，原渲染调用增量0、跳过20950次、虚拟 tick 增量0；随后512步真实 PPO 完成，74.96秒，参数改变且 checkpoint 重载执行通过，4个结束回合均死亡。相同种子/静止输入/9逻辑帧的可视化→无渲染→可视化回退→重新无渲染，渲染调用为18→0→18→0，四次均准确推进9逻辑帧。当前最后保留为无渲染模式。

边界：这是运行时绕开原渲染路径，不是无窗口/无显卡依赖的独立引擎，不是已验证的加速模式。原版冷启动归档错误仍未解决，可视化模式也仍可能触发原生渲染崩溃。20回合门禁最初因累计 manager counter 的21次 restart 归零而报错，分阶段对照已确认每次 reset 增1、实际战斗 step 增0，两种渲染模式一致；保留初次失败记录，不将其解释为战斗丢帧。

Python 崩溃传播已修复：坏连接不再进入安全房重置，两个 runner 如实记录 worker_error 与 broken_connection_closed，保留首个异常。16项离线测试及匹配输入的修改/回退检查通过。数据字段现状与扩展计划统一见 [ENV_ARCHITECTURE.md §0.2](ENV_ARCHITECTURE.md)。证据统一在 `../runs/l1/20260920-crash/`。

## 2026-09-20：单房间重置与奖励 v2 已接入

已完成：首次建无道具房间入口快照；每回合 `rewind` + 本地 Monstro 模板，原生恢复受伤/死亡/清房后的状态，三次重置约0.272–0.274秒；不再每局 restart/reseed/stage。初次也先 rewind，使自然 Boss HP 固定为250，旧312.5HP课程分开保留。固定种子不等于已证明逐帧确定性。

采用用户确认的实际受伤 −1、有效扣血结算 +0.05、实际伤害/敌人初始MaxHP、清房 +1、死亡不重复扣分。地形5×9×15已进入策略；物理走位验证空地/石头/坑及飞行通行，修复了飞行跨石头掩码。20项测试通过；新版512步实机 PPO（74.27秒）参数更新、存取档/重载动作通过，四个终局仍均死亡。另3局规则控制器各有24/23/25个正奖励步骤，不是学习成绩。证据在 `../runs/l1/20260920-rewind/`，详细语义维护于 [桥接 README](../bridge/README.md)。

下一门禁：同进程50回合 rewind 耐久，逐局种子/房间/HP/零计数校验，精确逻辑帧、渲染关闭、奖励账目与安全断开。通过后再扩大训练预算，并单独处理冷启动与可视化崩溃；不把渲染规避称为原生根因修复。

## 2026-09-20：原生崩溃故障链复核

后续更新：9 份转储的校验调用链、崩溃时 `.data` 全局状态（注册表完好、初始化位为 1）、B 类的逐条目 `fopen` 机制与三类修复方案见 [NATIVE_CRASH_ANALYSIS.md](NATIVE_CRASH_ANALYSIS.md)；本节保留当时的复核记录。

本次只读重解析5份已有转储并核对反编译代码；没有修改游戏EXE、重新启动游戏或调整驱动/覆盖层。新证据在 `../runs/l1/20260920-docs-native-crash/dump-recheck.json`，执行脚本 `dump_recheck.py` 退出0。扫描时已经没有存活的 isaac-ng 进程，不能仅据此判断上一轮 worker 如何退出。

### A. 字体绘制 shader 栈下溢：直接故障已确定，上游触发未确定

| 证据 | 观察 |
|---|---|
| PID27476 / 48584 / 32412 | 同一 RVA `0x0061C423`，`mov eax,[eax]`，EAX=`0xC`，ESI=`0xFFFFFFFF` |
| 故障时 shader 栈深度 | 三份转储均为0；这里是执行 pop 的瞬间，不是正常绘制外的“深度0” |
| shader registry 堆节点 | 三份转储都未捕获，不能读取失败时的字体 shader 对象/有效标记 |

字体函数 `FUN_00a1bd80`（RVA `0x0061BD80`）在第64行调用 `FUN_00a140c0("KAGE_ColorTextureShader")`，但不检查返回值；第158–167行即使发现栈深度0，也只记录 `Shader stack empty`，然后继续取栈顶并减一。取栈顶助手 RVA `0x00284FC0` 使用 `offset + size - 1`，空栈可形成下溢索引；实际转储确定最终读到了地址0xC。

入栈函数 RVA `0x006140C0` 第23–40行只有在“注册表找到对象、对象非空、对象flag bit0有效”时才压入旧 shader 并返回1，否则返回0。因此存在一条代码上明确的故障路径：**字体 shader 入栈失败 → 调用者仍然出栈 → 空栈访问**。但旧转储缺少 registry 堆，尚不能证明三次崩溃究竟是这次入栈失败，还是中途其他代码破坏了配对/状态。现有栈地址列表只是扫描出的候选值，不冒充完整展开的调用栈。

代码位置：`analysis/j460/exports/j460-baseline/decompiled/06/0061BD80_FUN_00a1bd80.c`、`006140C0_FUN_00a140c0.c`，以及 `decompiled/02/00284FC0_FUN_00684fc0.c`。反编译函数名中的地址按导出基址0x00400000表示；上表统一使用RVA，不能直接当运行时VA。

### B. 启动期 ArchivedFile 空流：与渲染崩溃分开处理

| 转储 | 故障点 | 确定状态 |
|---|---|---|
| PID29048 | RVA `0x00668CB5`，`mov edx,[ecx]` | ECX=0，ArchivedFile this+8 的流指针=0，archive kind=2 |
| PID22060 | RVA `0x00668CD6`，`mov edx,[ecx]` | 同样空流，archive kind=0 |

`ArchivedFile` 构造路径 RVA `0x00668A50` 第51–53行调用资源打开函数 RVA `0x006178D0`，把结果存入 this+8，**未检查空值就调用初始化/seek函数** RVA `0x00668C50`。后者在kind1、kind2和其他分支都会直接调用该流的虚函数。底层文件打开路径 RVA `0x00617EA0` → `0x00652540` 在路径无效或 `fopen(...,"rb")` 失败时可返回空对象。

所以已确认的是 **空的底层流继续进入原生虚调用**，且不是某一种解压算法独有。尚未确认的是为什么得到空流：失败文件名指针在两份转储中都指向未捕获的堆；没有当时的 errno / DOS错误码，也没有文件打开事件，因此不能断言某个.a损坏、权限错误、工作目录错误或文件被占用。当前磁盘文件存在不等于当时打开成功。

### 排除范围与下一次需要采到的数据

- 上述5次转储均没有 Turbo / NetFix / REPENTOGON 模块，故这些注入模块不是这5次崩溃的必要条件。Lua bridge 与 `--luadebug`、Steam/NVIDIA覆盖层仍可能在环境中；不能由“无Turbo”推出“完全纯净原版”，也不能把某个已加载模块直接当成根因。
- 关闭渲染能绕开A的执行入口，但不能修B。已解锁时仍有崩溃，不能把ToDesk锁屏当作统一根因。固定种子/rewind/PPO奖励与两处原生故障没有已证实的因果关系。
- A的下一探针：记录字体shader push返回值、push/pop前后深度、registry对象/flag、线程ID和渲染上下文；在**第一个配对异常**时保留完整堆，而不是等到最终空指针后只看残留地址。
- B的下一探针：在资源流返回空的现场记录完整路径、工作目录、线程ID、`errno`与`_doserrno`，立即采集堆；不在上层补一个空值返回就宣称修好了资源加载。
- 完整用户态转储可用 ProcDump 的 `-ma`；异常捕获应按目标异常收窄，而不是重复之前失败的全量启动期 first-chance 接管。进程可能自行捕获异常，单用 second-chance 监视可能漏掉。参数语义见 [Microsoft ProcDump](https://learn.microsoft.com/en-us/sysinternals/downloads/procdump)。已有转储全是非完整内存，缺少上述关键堆页，这是目前根因结论的实际边界。
- 后续A/B实验每次只改变一个条件：无Mod原版、仅bridge未连接、bridge同步步进；分别对照可视化/无渲染。已有结果不足以认定同步阻塞、覆盖层或驱动中的任何一个就是上游原因。
