# J460 worker 原生崩溃：根因分析与修复方案

更新：2026-09-20。本文只处理原版 `isaac-ng.exe`（v1.9.7.17.J460）在 RL worker 用法下的三类原生崩溃。全部结论按"已确定 / 未确定"分列；地址一律写 RVA（反汇编VA = RVA + 0x400000；运行时基址以每个进程/转储的模块表为准）。反编译文件在 `analysis/j460/exports/j460-baseline/decompiled/`，本轮新增证据在 [`../runs/l1/20260920-native-crash-rootcause/`](../runs/l1/20260920-native-crash-rootcause/)（`dump_walk.py` 只读解析 9 份 WER 转储，`dump-walk-all.json` 是逐份输出，`globals-at-crash.txt` 是崩溃时 `.data` 全局值）。此前的排查记录见 [bridge/README.md 崩溃排查状态](../bridge/README.md#崩溃排查状态) 与 [L1_FEASIBILITY_PLAN.md 故障链复核](L1_FEASIBILITY_PLAN.md#2026-09-20原生崩溃故障链复核)；本文与它们冲突处以本文为准。

## 0. 结论摘要

| 类别 | 直接故障 | 已确定 | 仍未确定 | 修复方向 |
|---|---|---|---|---|
| A 字体绘制 shader 栈下溢（RVA `0x61C423`） | 空栈出栈读到 `NULL+0xC`；入栈失败是重点候选 | CALL约束栈扫描支持正常渲染路径；注册表计数3、图形初始化位1，尚未验证树节点/键/对象 | 入栈返回值、目标对象状态及首次失配位置 | 记录push/pop、对象与堆；字体守卫只是待实机验收的缓解 |
| B 启动期归档流空指针（RVA `0x668CB5`/`0x668CD6`） | 逐条目构造拿到空流后未检查即虚调用 | 新探针记录到NULL路径；无Turbo的历史进程也会崩溃，发生于Mod/shader初始化前 | 相对路径搜索偶发失配的内部原因 | 已注册且确实存在于磁盘的容器走原生绝对路径分支；嵌套归档保留原生搜索，验收见§0.4 |
| C 退出时跳转到 `0xDEDEDEDE` | 执行无效地址，栈顶候选NVIDIA驱动+0x91F091 | 新反向对照复现相同现场；隔离NvCamera32/nvspcap的worker正常退出 | 两个捕获模块各自责任及驱动内部指针损坏来源 | worker启动链默认进程级隔离；仍保留真实异常退出分类 |

三类的故障现场均在原生代码，尚不能据此排除 Python 驱动的时序/生命周期是间接触发因素；A、B 在没有 Turbo/NetFix/REPENTOGON 模块的进程里也出现（9 份转储的模块表见 `dump-walk-all.json` 的 `nonsystem_modules`）。

## 0.1 2026-09-20 独立复核：提交时的证据边界

本次核对代码和已有报告，并重新执行本机构建/测试：DLL/注入器、clock_test、registry_test、身份门和7个地址字节核对通过，共享内存双向测试通过。Python首次回归发现test_rendering仍断言旧preconfigure参数；同步新探针开关并增加显式启用测试后，22项通过。没有启动游戏或改变杀软、驱动、存档。探针/缓解**已有实现**，不等于已经通过目标故障的实机验收。

- `cold-start-smoke/report.json`：PID28124在bridge listening之前崩溃，记录到NULL路径。`0x617EA0`对NULL路径可以直接返回失败、不调用CRT；故此处errno=2可能来自上游/之前调用，不能单凭它证明这次 `_access` 或 `fopen` 的错误。[CRT `_access` 的返回值/errno契约](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/access-waccess?view=msvc-170)
- `cold-start-12/report.json`当前仅保存6次运行，不是12次或20次成功验收；6次 `access_calls=0`、`access_retry_ok=0`、`font_calls=0`。先验证探针是否命中实际调用（含IAT目标、已加载DLL版本、路径过滤），再谈重试/字体守卫效果。`cold-start-control`为3次对照，全部 `reached_menu=false`、`exit_crash_after_shutdown`。
- `dump_walk.py`的“validated_chain”是带CALL约束的栈扫描，遇间接CALL未解析目标便继续接受；它比裸栈扫描强，但不是完整可靠展开。A类走渲染路径得到支持，不应声称穷尽了所有调用链。
- `writeProbeDump()`实际使用私有读写页、数据段和内存区域信息等flags，**没有 `MiniDumpWithFullMemory`**；必须检查新dump是否真的覆盖目标树节点/对象页。`FullMemoryInfo`不是全内存内容。[Microsoft MINIDUMP_TYPE](https://learn.microsoft.com/en-us/windows/win32/api/minidumpapiset/ne-minidumpapiset-minidump_type)
- 文件探针在失败后还会写日志、复查文件和转储；目前未保存/恢复所有返回时的errno、`_doserrno`、GetLastError，不能称作完全透明的只读探针。建议下一实现先分离诊断模式与重试模式，并保留调用者可见的错误状态。
- 当前策略：先验收探针覆盖，再取实际失败的路径/对象；不根据现有数据把火绒定为根因，不自动更改排除项，不把退出崩溃吞掉。

## 0.2 2026-09-20 17:07–17:12：探针覆盖实测，修正归档分支判断

证据目录：`../runs/l1/20260920-probe-coverage/`。本轮启动4个独立进程，均显式关闭虚拟时钟、字体守卫、重试和探针转储，开启skip-render；没有改变游戏EXE、杀软、驱动或其他Mod的启用状态。

- **Hook地址正确**：PID42268/16872的游戏IAT槽（RVA `0x7187C4`）与已加载ucrt `_access` 导出均指向 `0x774E91F0`，入口已是MinHook的E9跳转。新增过滤前日志确实捕获到options.ini、Mod标记和资源覆盖路径的调用。旧 `access_calls` 只统计绝对 `.a`，零值不是“Hook没执行”。
- **推翻“归档槽存绝对路径”**：PID3432的只读进程内存采样直接读到18个槽，含 `resources/packed/animations.a`、`config.a`、`afterbirth.a`、`afterbirthp.a` 和 `resources/secret.a`，均为相对路径。`archive_snapshots`保留槽号、指针、kind和采样时刻。结合 `0x617180` 分支，这些路径走 `0x616C60` 的资源搜索/索引查找，不是绝对路径 `_access` 分支。
- **B仍未修复**：PID46896和PID3432的新游戏转储均为RVA `0x668CB5`、读NULL；PID3432的CALL约束栈候选为 `0x6178D0 → 0x617F40 → 0x668A50 → 0x668C50`，说明还需检查挂载后的资源读取，不能把B限定为初次挂载。两份小转储都不含ArchivedFile对象的堆页，尚不能从崩溃对象确定具体容器；18槽采样不是崩溃瞬间的完整堆。PID42268也在listening前异常退出，但本轮未解析其转储，不强行分类。PID16872到达listening、执行64次GameUpdate，WM_CLOSE后仍以 `0xC0000005` 退出；这不构成修复成功或相同退出故障点的证据。
- **探针本身有一项已修正的问题**：`hookCrtAccess`新增过滤前限频采样，并保存/恢复返回时的errno、`_doserrno`和GetLastError；有重试时采用最后一次实际调用的状态。确定性原生回归用会污染这三个值的诊断桩：修改前绝对路径失败/重试成功两项失败，修改后三项全通过。此测试验证Hook逻辑，不代替真实CRT与游戏故障复现。`hookFileOpen`的错误状态透明性仍待处理，不能将整个探针称为完全透明。

下一步优先捕获 `0x617180` 的原始相对路径、解析返回值，以及 `0x616C60` 的搜索目录/索引命中；再区分缓存缺项、路径内容/生命周期或后续归档查找问题。当前没有证据支持先设置杀软排除，也不应扩大睡眠重试掩盖索引错误。

验证命令：`python turbo/tests/access_probe_test.py`（实际Hook函数编译到独立32位测试程序，3种输入）；`pwsh -NoProfile -File turbo/build.ps1`。初次控制器在进程已崩溃后枚举模块得到WinError299，已改为恢复主线程后立即读IAT；该工具错误与游戏的原生异常分别记录。

## 0.3 2026-09-20：worker启动隔离与实战重置修复

证据目录：`../runs/l1/20260920-resolver-fix/`。当前修复落在RL worker的启动链，而不是修改原游戏或NVIDIA驱动二进制。

### 已验证的退出故障处理

- 基线PID38200退出时 `EIP=EAX=0xDEDEDEDE`，栈顶返回候选 `nvwgf2um.dll+0x91F091`，与历史C类相同。单独设置 `SteamNoOverlayUI=1` 的对照仍异常退出。
- 挂起启动时，`hookLdrLoadDll`仅拒绝当前进程加载 `NvCamera32.dll` / `nvspcap.dll` 两个可选捕获叠加层，返回 `STATUS_DLL_NOT_FOUND`。不拒绝 `nvoglv32`、`nvwgf2um`、D3D11或Steam overlay，不改系统设置、不卸载已加载模块。实战worker模块表确认NVIDIA OpenGL驱动及Steam overlay仍加载，两项捕获模块未加载。
- 重新允许这两个模块后，PID48216到达120次逻辑更新，退出再次变成 `0xC0000005`；新WER转储同样为 `0xDEDEDEDE` / `nvwgf2um+0x91F091`。该转储已复制到本轮目录，避免WER轮转清除。这里证明的是两项捕获叠加层这一组对退出故障的影响，尚未拆分两者的个别责任。
- 隔离后的严格门禁：4个无渲染worker、5个可视化worker均达到至少120次GameUpdate并正常退出0；可视化每进程实际执行3107–3146次字体调用，未打开字体守卫、文件重试、虚拟时钟或resolver诊断。更早17次只等待listening后2–4秒，其中14次未进入逻辑更新，另3次只到58、59、117次更新，**均未达到严格120次门禁，不计入这9次实玩法/渲染门禁**。

`launch_suspended(..., disable_capture_overlays=True)`默认开启隔离，可传False做对照；`train_monstro.py --launch`和`monstro_rollout.py --launch`均在恢复主线程之前注入，修正了此前“启动完成之后才加控制层”的空档。`--pid`仍可附着，但已经加载的叠加层不能追溯移除，必须重启worker。

### 相对路径诊断保留的事实

PID27288和18872分别捕获 `afterbirthp.a`、`afterbirth.a` 解析返回NULL；PID18872的 `0x616C60` 搜索函数也返回NULL，但紧随其后的只读目录树遍历找到了对应hash、非空路径值，所经节点nil标记均为0。这把问题缩小到原生查找过程，而不是简单缺文件。没有依据直接修改文件索引或加睡眠重试。可选 `ISAAC_TURBO_RESOLVER_PROBE=1` 开启搜索/失败点取证Hook。后续PID10948在隔离捕获组件后仍复现B，因此不能把B归因为NVIDIA组件。当前正常worker只安装执行容器路径修复所需的ResolvePath Hook，关闭搜索树诊断。

### rewind后的玩家资源模板

首次2048步PPO在9个结束回合后被训练器的初始状态断言中止；再次诊断读到Isaac、无主动道具、最大红血6，但当前红血2、炸弹0。游戏仍存活，WM_CLOSE退出0，这不是原生崩溃。

`monstro_empty.lua`现在只在episode初始化边界补齐红血，并恢复既有fixture的1炸弹、0钥匙、0金币；仍使用原生rewind，不重启游戏、不重新生成楼层，不在战斗中回血或开无敌。玩家位置、道具数量、Boss类型/血量、房间锚点的原有校验保留。资源调整使用游戏原生[EntityPlayer API](https://docs.moddingofisaac.com/ab_p/beta/docs/entityplayer)，受伤/命中/实际伤害的奖励权重未改变。玩家校验失败现在打印实际字段，避免笼统归为连接故障。

### 第一轮长实战验收（历史结果，早于最终归档/房间边界修复）

- `ppo-2048-fixed/report.json`：PID39024，2048步PPO、17个结束回合、64个训练epoch，291.21秒；参数L2变化1.28119；模型保存/重载、重载后再次执行通过。渲染调用0、虚拟tick0，安全房断开后WM_CLOSE退出0。
- `visible-combat/report.json`：PID14188，5个完整实战回合，106.71秒；5821次真实渲染、75673次字体调用，push_failures=0、font_skipped=0、虚拟tick0；安全房断开后WM_CLOSE退出0。这里没有用字体守卫跳过文本来制造通过。
- 两个过程均未强制终止，完整退出记录在 `combat-acceptance-pre-bar.json`。初次失败在 `ppo-2048/report.json` / `combat-acceptance-initial.json`，诊断失败在 `reset-diagnosis/report.json`，保留而不覆盖。场景同输入的基线/修改/回滚结果见 `scenario-BASELINE`、`scenario-MODIFIED`、`scenario-ROLLBACK`。
- 该轮PPO记录16次死亡、1次“win”，但后来实测发现炸弹能炸开仅Close的门，旧逻辑会把邻接空房间视为胜利。该次win没有终局房间证据，**不计为有效Monstro击杀**。可视化规则策略5回合均死亡。

## 0.4 2026-09-20：物理容器路径修复与严格房间边界

### 修复路径与失败实验

- PID10948（正式PPO启动）及6924（首个NULL返回后重试候选）均复现RVA `0x668CB5`、ECX=0，转储解析保存在 `archive-failed-dumps.json`。退出故障隔离不能解决这个独立问题；“仅返回NULL才重试”没有通过门禁，已弃用。
- `hookResolvePath`只匹配32个挂载槽中容器路径的**原始指针身份**（包括计数尚未递增的正在挂载槽），且仅对确实存在的磁盘普通文件，将相对路径转为绝对路径后交给**原生解析函数**。原生函数仍负责存在性检查、字符串分配和后续释放；不伪造文件流、不吞读取错误、不添加睡眠重试、不修改资源索引、文件内容或RNG。
- 初次把所有槽都改为磁盘路径时，PID30780在 `resources/secret.a` 失败：它是嵌套容器，不是磁盘文件。该错误实验保留在 `absolute-1`。最终实现先检查物理文件属性，对嵌套容器及普通资源查找保持原样。不能把“所有.a文件”都当作OS文件。
- 默认 `launch_suspended(..., archive_path_fix=True)`；可传False做对照。单位测试直接编译实际Hook，覆盖第31槽、同名不同指针、嵌套容器、普通资源、已有绝对路径、真实失败传递、禁用开关及NULL输入，共8项；同时验证原生返回指针和errno/doserrno/LastError不被日志污染。
- `physical-1`至`physical-20`：关闭搜索诊断、字体守卫、虚拟时钟及文件重试，20次冷启动均进入游戏、至少120次逻辑更新，WM_CLOSE退出0，没有强制终止。日志记录实际 `archive_resolve_absolute ... success=1`；这不是仅编译通过的候选。
- 修复绕开已实测失配的“已知磁盘容器再次走相对资源索引”路径；**原生搜索为何偶发返回NULL/空路径仍未定位到内部写入点**。不要据此指控杀软、Steam或具体Mod，也不要把worker修复写成已修改原版引擎底层实现。

### 防止跨房假胜利

实机 `door-probe.log` 显示，Close后的门仍可炸开；执行原生 `Bar()` 后三扇门的 `CanBlowOpen=false`，`TryBlowOpen=true参数`也返回false。场景现在在初始化时Close+Bar，Gym在每个动作后检查room_idx不变，变化即报错而不是把邻接空房判为Boss胜利。对应单元回归在旧实现失败、修改实现通过、回滚实现再次失败。原生rewind、每次受伤−1、命中+0.05及按实际伤害归一化奖励均保留。

### 最终长实战验收（两项原生修复 + Bar + 房间边界断言）

- `ppo-2048-final/report.json`：PID21800，2048步PPO、64个训练epoch、17个结束回合，结局{'death': 17}；耗时293.31秒，参数L2变化1.318380。模型保存、重载、重载后再次执行通过。
- `visible-combat-final/report.json`：PID51328，5个结束回合，结局{'death': 5}；实际渲染5763次、字体调用74919次，字体跳过0、shader入栈失败0、虚拟tick0。
- 两者状态completed；安全房断开后WM_CLOSE实际退出0，均未强制终止。逐命令/退出记录在 `combat-acceptance.json`。20次冷启动和这2个长实战worker均未观察到崩溃；这是当前J460训练配置的通过结果，不是任意驱动/Mod组合下的无限稳定保证。
- 崩溃修复已通过本轮验收；**战斗学习尚未达标**，不能把训练执行成功当作Agent已经会躲弹/击败Boss。此前已撤回的假胜利不计入上述结果。

## 1. 崩溃清单

来源：Windows 应用程序日志 Event 1000/1001（`isaac-ng.exe`，时间戳 `0x69e6e3a7`），共 19 次；WER 转储在 `C:\Users\eolc\AppData\Local\CrashDumps\`，游戏自带转储在 `Documents\My Games\Binding of Isaac Repentance+\crash_dumps\`。

| 时间 | 类别 | 故障偏移 | 备注 |
|---|---|---|---|
| 09-19 17:37:05 | B | 0x668cb5 | 早期启动尝试，无保留日志 |
| 09-19 17:39:42 | A | 0x61c423 | |
| 09-19 17:43:25 | B | 0x668cb5 | |
| 09-19 17:50:38 | B | 0x668cb5 | 47 秒内三次连续启动全部 B |
| 09-19 17:51:00 | B | 0x668cd6 | |
| 09-19 17:51:25 | B | 0x668cb5 | |
| 09-19 18:06:55 | A | 0x61c423 | |
| 09-19 18:20:01 | A | 0x61c423 | |
| 09-19 18:21:41 | A | 0x61c423 | |
| 09-19 18:34:50 | A | 0x61c423 | |
| 09-19 19:24:48 | A | 0x61c423 | PID 30604：断开桥接后回主菜单，"Menu Online Awards Init" 后崩溃 |
| 09-19 20:26:35 | A | 0x61c423 | PID 13508：冷启动，首次主菜单初始化即崩溃 |
| 09-19 20:33:58 | A | 0x61c423 | PID 11008：同上 |
| 09-19 21:42:30 | B | 0x668cb5 | PID 29048：`--luadebug` 启动，版本横幅后即崩溃，kind 2 归档 |
| 09-19 21:45:14 | B | 0x668cd6 | PID 22060：kind 0 归档 |
| 09-19 22:04:30 | A | 0x61c423 | PID 27476：`stage 1` 后桥接断开，随后首帧 |
| 09-19 22:15:29 | A | 0x61c423 | PID 48584：同上 |
| 09-20 00:30:53 | A | 0x61c423 | PID 32412：`luarun` 建场后的设置帧 |
| 09-20 14:58:18 | C | EIP=0xdededede | PID 18788：正常退出流程之后，`nvwgf2um.dll` 内 |

两天里没有 GPU 复位（TDR，System 日志 4101）记录；09-19 20:20:14 有一次 Parsec 虚拟显示器 attach/detach 事件，之后 6 分钟出现 PID 13508 的 A 类崩溃，但 17:39–19:24 的 A 类崩溃前没有任何显示事件，**显示拓扑变化不是 A 的必要条件**。本机同时挂着 Parsec 与 ToDesk 两个虚拟显示适配器，杀软为火绒（Defender 实时保护已关闭）。

## 2. A 类：字体绘制 shader 栈下溢

### 2.1 故障机制（已确定，反编译 + 转储寄存器一致）

shader 栈是一个全局 `std::deque<Shader*>`，对象位于 RVA `0x8379CC`：`+4` 块表 `0x8379D0`、`+8` 块表容量 `0x8379D4`、`+0xC` 起始偏移 `0x8379D8`、`+0x10` 深度 `0x8379DC`；当前 shader 指针在 `0x8379B8`；注册表是 `std::map<uint32 hash, Shader*>`，头节点指针 `0x8379BC`、元素数 `0x8379C0`。哈希是对小写名字做的 djb2（`KAGE_ColorTextureShader` → `0xB3D14323`，与 09-19 活体读取的注册表键一致）。

- 入栈 `FUN_00a140c0`（RVA `0x6140C0`，stdcall，一个栈参数 = 名字；两个出口均为 `ret 4`）：查注册表；只有"节点存在 且 对象非空 且 对象 `+4` 的 bit0 为 1"才把当前 shader 压栈、把该对象设为当前、返回 1；否则**不压栈返回 0**。原文cdecl描述错误，现有C++ Hook的stdcall无需修改。
- 字体绘制 `FUN_00a1bd80`（RVA `0x61BD80`）第 64 行调用入栈但**不检查返回值**；第 158–167 行出栈：深度为 0 时只写日志 `Shader stack empty`，然后照样调用取栈顶助手 `FUN_00684fc0`（RVA `0x284FC0`）：索引 = 偏移 + 深度 − 1 = `0xFFFFFFFF`（转储 ESI），块号 `(−1>>2) & 7 = 7`，块指针为空，于是取 `0 + 3×4 = 0xC`（转储 EAX），`mov eax,[eax]` 触发读 `0xC` 的访问违例。
- 因此 A 类的充分条件是：**这一次入栈返回了 0**。字体绘制内部的四个助手（`0x61B580` 画字形、`0x61B8E0` 量宽、`0x61B510`、`0x62BFE0`）在静态调用图 5 层内不接触任何出栈/清栈代码（对 `0x8379DC` 的 20 处引用全部在入栈、`EndFrame`（`0x619180`）、`ResetState`（`0x613220`）和各个自带出栈的绘制例程里），降低了"由已检查的直接内部调用清空"的可能性；有限深度静态调用图不能排除间接调用、并发或内存破坏。

### 2.2 崩溃时的调用链（经 CALL 指令校验）

`dump_walk.py` 对栈上每个落在模块内的值检查其前一条指令是否为 CALL，且直接调用目标必须等于下一层帧所在函数；结果：

| 转储 | 校验通过的链（自内向外） |
|---|---|
| 48584（战斗中） | 字体 `0x61BD80` ← Found HUD 数值文本 `0x44C2C0+0x55C` ← `HUD::Render` `0x5A3EB0+0x17F8` ← `Game::Render` `0x2FBC10+0x794`（调用点 `LEA ECX,[Game+0x1DA04]`）← 整帧渲染 `0x5555C0+0x406` |
| 32412（建场后） | 字体 ← 字体包装 `0x61B140+0xAA` ← `HUD::Render` `0x5A3EB0+0xCA3` ← `Game::Render` ← 整帧渲染 |
| 11008（冷启动主菜单） | 字体 ← 字体包装 `0x61AEC0+0xA7` ← 菜单页 `0x4E6A80+0x48B` ← 菜单渲染 `0x58A040+0x26F` ← `0x618300+0x98`（间接）← 整帧渲染 `0x5555C0+0x222` |

三条链都在正常的整帧渲染里。此前 bridge/README 里"栈候选含 `Game::Update` 地址"的记录有误：`0x2FC3A4` 所在函数 `0x2FBC10` 是被整帧渲染 `0x5555C0+0x406` 调用的 `Game::Render`，不是 `Game::Update`（`0x2FADC0`）。`0x44C2C0` 按 `Manager+0x2A37C` 门控并按玩家类型 19/20（Jacob/Esau）分槽，对应 `options.ini` 的 `FoundHUD=1`。

### 2.3 崩溃时的全局状态（直接读转储 `.data`）

7 份 A 类转储全部一致（`globals-at-crash.txt`）：注册表元素数 `0x8379C0 = 3`，图形层初始化位 `0x8379B4 = 1`、`0x837984 = 1`，默认纹理指针与设备回调指针非空，栈深度 0、偏移 0、块表容量 8。三个内置 shader 由 `FUN_00a180b0`（`0x6180B0`）/`FUN_00a18e90`（`0x618E90`）经 `FUN_00a13fa0`（`0x613FA0`）注册，注册成功时 bit0 必为 1（`Shader::Create` `0x614620` 末尾 `flags |= 1`，失败则不插入）。整表销毁 `FUN_00a13060`（`0x613060`）会把元素数清零并清除初始化位，**转储否定了"崩溃时注册表已被销毁"**。

因此当前重点检查字体 shader 的键查找、对象指针与 bit0。元素数仍为3并不能证明该键存在或树结构完好，也不能将候选缩减为仅有空指针/bit0清零。这两种情况都要读堆才能区分，而现有 9 份转储都不含堆页（注册表节点、块表都读不到；此前文档里"块表全为 0"之类说法不能成立，读不到不等于为 0）。

### 2.4 已排除

- 不是同步阻塞把绘制拉进逻辑帧（§2.2）。
- 不是 Turbo/NetFix/REPENTOGON 注入：7 份 A 类转储里都没有这些模块。
- 不是其他 Mod：停用 8 个 Mod 后仍复现。
- 不是锁屏或显示切换的必要结果（§1）。
- 不是 GPU 复位：无 TDR 事件。
- 不是注册表被整表销毁（§2.3）。

### 2.5 修复方案

1. **根因探针（首选，必须先做）**：在 turbo DLL 里增加对入栈函数 `0x6140C0` 的 MinHook 钩子（序言 `53 8B DC 83 EC 08 83 E4 F8 83 C4 04 55 8B 6B 04`，与整帧渲染函数同一种 LTCG 序言，已在身份门里验证过同类字节）。钩子先调原函数；返回 0 时：读注册表节点（按哈希查 `0x8379BC` 红黑树）、对象指针及其 `+4` flag、深度/偏移、返回地址链，写入 turbo 日志，并用进程内已加载的 `dbghelp!MiniDumpWriteDump` 写带堆页的探针转储（当前实现不是 `MiniDumpWithFullMemory`）到 `%LOCALAPPDATA%\IsaacRL\turbo\`。这是文档此前要求的"在第一个配对异常现场保留完整堆"。
2. **运行期缓解（与探针并存，明确标为缓解）**：钩住字体绘制 `0x61BD80`（序言 `55 8B EC 83 E4 F8 83 EC 40 8B C1`，thiscall + 9 个栈参数 + XMM2/XMM3 浮点入参），进入时先做与入栈相同的注册表检查；若 `KAGE_ColorTextureShader` 不可用则直接 `ret 0x24` 跳过这一次文本绘制。效果是那一帧少画一段文字，不改任何玩法状态，训练默认无渲染时不会触发。
3. 已有的无渲染训练只是绕开渲染入口，不算修复；可视化验收仍需要 1 或 2。
4. 不修改 `isaac-ng.exe` 本体，不改显卡驱动。

探针拿到完整转储后，下一步是核对是谁在原地销毁/替换该对象：候选是设备丢失重建协议（`0x6180B0` 重建与 `0x613060` 销毁在同一张虚表 `.rdata 0x782434/0x782438`，静态没有直接调用者）和 `Shader` 自身虚表的销毁槽。

## 3. B 类：归档重开得到空流，解析路径已观察到NULL

### 3.1 机制（已确定）

- 归档挂载 `FUN_00a179c0`（`0x6179C0`）：解析路径（`0x617180` → `0x616C60` 在搜索目录表里定位）、分配 `0x837A14` 的槽位（上限 32）、`FUN_00a17ea0`（`0x617EA0`）打开容器文件、读 7 字节魔数 + 1 字节 kind（写入 `0x837A18[slot*8]`）+ 表偏移 + 条目数，然后**对每个条目**：分配 0x1C 的条目、`FUN_00a68a50`（`0x668A50`）构造 `ArchivedFile` 读全文算校验和、插入 0x8000 槽的全局哈希表、析构。
- `ArchivedFile` 构造再次按容器路径调用 `FUN_00a178d0`（`0x6178D0`）：先经 `0x617180` 解析，再查已挂载归档（`0x617F40`），未取得流则回退 `FUN_00a17ea0`。只有有效路径才可能继续 `KAGE::Filesys::File::Open`（`0x652540`）→ CRT `fopen(path,"rb")`；NULL路径会直接失败。最终空流未经构造函数检查就传给 `FUN_00a68c50`（`0x668C50`）虚调用 → 崩溃。kind 2（29048）走 `0x668CB5`，kind 0（22060）走 `0x668CD6`。
- 挂载函数自己对容器文件的首次打开是有检查的（失败会记 `Failed to open archive file`），但每条目的重复打开没有。一个 `afterbirthp.a` 有一万多个条目，等于启动阶段对同一文件做上万次 `fopen/fclose`。
- 调用链（29048/22060 一致）：`J460_ApplicationMain+0x95F` → 挂载 → 构造 → 初始化。两次都在 `--luadebug` 纯启动、"Binding of Isaac: Repentance+ v1.9.7.17.J460" 横幅之后、shader 初始化和 mod 加载之前。

### 3.2 实机复现后的精确机制（2026-09-20 16:13，PID 28124）

启动即注入的探针（§3.3-1）第一次冷启动就复现了 B：游戏在挂载第 12 个归档时崩溃（`0x668CB5`，校验链 `ApplicationMain+0x95F` → 挂载 → 构造 → 初始化，转储在 `../runs/l1/20260920-native-crash-rootcause/cold-start-smoke/`）。探针记录到文件打开函数 `0x617EA0` 收到的**路径是 NULL、errno = 2（ENOENT）**，栈上是 `ArchivedFile` 构造（`0x668B27`）与挂载函数（`0x617CCF`）。对照反汇编：

- **17:12实测修正**：槽表实际保存相对路径（见§0.2），不是这里先前推断的绝对路径。`0x617180`对绝对路径确有 `_access` 分支，但不能套用到已采样的归档路径。
- 相对路径转入 `0x616C60`：逆序遍历搜索目录，检查已挂载归档及每个目录的名字哈希索引；遍历完未命中时返回0。随后 `0x617180` 返回NULL，`0x6178D0`把它交给后续打开函数。**究竟哪个目录/键或路径内容导致首次失配，尚未捕获**。
- `errno=2`可能为遗留状态：新探针真实记录过 `_access`返回0且errno仍为2的调用。不能从下游NULL+errno2直接推断Windows文件访问失败，更不能据此确定过滤驱动/杀软根因。

下一故障门：原始相对路径 → 搜索目录与哈希索引 → 解析返回值。先取得这一链路的失败现场，再决定是索引修复、路径修复还是底层文件操作处理。

### 3.3 修复方案（探针与缓解已实现，2026-09-20）

1. **启动即注入的探针（已实现）**：注入器 `--launch` 模式以 `CREATE_SUSPENDED` 创建进程、注入、等控制块离开 loading 再 `ResumeThread`（`rl/turbo/src/injector.cpp`；Python `isaac_bridge.turbo.launch_suspended`）。钩子：`0x617EA0`（stdcall `(path, File**)`，`ret 8`）记录失败路径/errno/`_doserrno`/`GetLastError`/挂载槽数/栈上的返回地址候选；`ucrtbase!_access`（按导出名定位）对绝对路径 `.a` 的失败记录同样信息并立即用 `GetFileAttributesW`/`CreateFileW` 复查、记录进程句柄数；`kFlagProbeDump` 时写带私有读写页（堆）的 minidump，每进程最多 3 份。
2. **缓解层（已实现，`kFlagFileRetry`）**：`_access` 对绝对路径 `.a` 失败时按 10/20/30/40/50 ms 退避重试最多 5 次，成功即返回 0；`0x617EA0` 层实际为20/40/60/80/100 ms，共300 ms；与 `_access` 层的150 ms不是同一预算。当前策略不会重试相对路径或NULL路径；这不仅排除了可选文件，也漏掉了实测的 `resources/packed/*.a` 路径解析失败。是否恢复取决于后续调用能否成功；固定睡眠累计分别为150/300 ms，还不含文件API、日志、转储及嵌套调用耗时，不能承诺端到端最多150 ms或已经修好。每次重试均有记录。
3. 离线验证：`rl/turbo/build.ps1` 全过（DLL、注入器、`clock_test`、新增 `registry_test`、身份门探针、7 个目标字节核对），`test_turbo_control.py` 与原有 20 项单测通过；字体守卫桩的机器码经 capstone 反汇编核对（保存 ECX/EDX/XMM0-3 → 检查 → `jmp [g_orig_font_draw]` 或 `ret 0x24`）。
4. 冷启动循环 `rl/bridge/python/crash_probe/cold_start_loop.py`：挂起注入 → 等桥接 listening → 投递回车进入主菜单 → WM_CLOSE 正常退出 → 记录结局/探针计数/新转储；`--no-inject` 为不注入 DLL 的对照模式。
3. **训练器侧**：worker 启动串行化、前一个进程退出后等待固定冷却再拉起下一个；启动失败按 `worker_start_failed` 记录并自动重试，而不是算作一场死亡（现有 `train_monstro.py` 已不把启动失败当死亡，缺的是自动重试与冷却）。
4. 若探针显示错误码是共享冲突（`ERROR_SHARING_VIOLATION`）或访问拒绝，再决定是否需要用户把游戏目录加入火绒排除；这是用户的决定，本文不预设。

## 4. C 类：退出时 `nvwgf2um.dll` 内跳转到 `0xDEDEDEDE`

PID 18788 是 09-20 02:20 起的训练 worker（当时注入了 `isaac_turbo`，跳渲染、虚拟时钟关闭）。游戏日志末尾依次是两次 "OpenAL-SOFT device has changed, re-opening the device"、"Isaac is shutting down..."、"Isaac has shut down successfully"；之后 14:58:18 WER 记录 BEX，`P7: PCH_1B_FROM_ntdll+0x000787FC`，转储 EIP = EAX = `0xDEDEDEDE`，`[ESP]` 返回地址在 `nvwgf2um.dll+0x91F091`，栈上还有 `d3d11.dll+0x98705`。这是进程退出阶段 NVIDIA D3D11 用户态驱动（由叠加层拉进这个 OpenGL 进程）经函数指针跳到无效地址；0xDEDEDEDE提示释放/毒化模式，但尚未追到分配/释放链。已经完整接收并持久化的训练轨迹可保留，仍应检查尾部记录和存档完整性。

2026-09-20 冷启动循环的对照结果：**不注入任何 DLL** 的普通启动，经 WM_CLOSE 正常关闭（日志 "Isaac has shut down successfully"）后进程退出码同样是 `0xC0000005`（PID 37568）；注入探针 DLL 的三次（PID 37192/20860/6512）也一样。这几次都没有留下 WER 事件或转储，与 18788 那次（有 WER 记录）略有差异，仅说明 **异常退出可以在未注入 turbo 时发生**；相同退出码不能证明同一故障点，无法据此排除 turbo 的影响，也不能将其宣布为可忽略的固有行为（叠加层候选：`gameoverlayrenderer.dll`、`EOSOVH-Win32-Shipping.dll`、`NvCamera32.dll`/`nvspcap.dll`）。处理：保留 `exit_crash_after_shutdown` 分类与实际退出码，不折叠成 `clean_exit`；采至少一份无 turbo 的对应转储确认是否同一故障点。

## 5. 执行顺序

1. 两个探针、缓解与挂起注入已有代码；先按 §0.1 验证探针覆盖和错误状态保真，再继续实机门禁；`build.ps1` 离线测试与 `verify_targets.py` 字节核对通过后才能上机。
2. 冷启动循环：用挂起注入连续启动 20 次（每次到主菜单或桥接 "listening" 后正常退出，退出后冷却 5 秒），复现 B 并采到 errno；同时看 A 是否在主菜单首帧复现。
3. 战斗回归：可视化模式下跑 20 回合（建场 + rewind），复现 A 并拿到完整转储；无渲染训练不受影响，可并行继续。
4. 拿到 A 的对象状态后再定"根因修复"是什么；在此之前只允许把 §2.5-2 称为缓解。
