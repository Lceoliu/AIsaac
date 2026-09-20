# J460 worker 原生崩溃：根因分析与修复方案

更新：2026-09-20。本文只处理原版 `isaac-ng.exe`（v1.9.7.17.J460）在 RL worker 用法下的三类原生崩溃。全部结论按"已确定 / 未确定"分列；地址一律写 RVA（VA = RVA + 0x400000；转储里的运行时基址为 0x100000）。反编译文件在 `analysis/j460/exports/j460-baseline/decompiled/`，本轮新增证据在 [`../runs/l1/20260920-native-crash-rootcause/`](../runs/l1/20260920-native-crash-rootcause/)（`dump_walk.py` 只读解析 9 份 WER 转储，`dump-walk-all.json` 是逐份输出，`globals-at-crash.txt` 是崩溃时 `.data` 全局值）。此前的排查记录见 [bridge/README.md 崩溃排查状态](../bridge/README.md#崩溃排查状态) 与 [L1_FEASIBILITY_PLAN.md 故障链复核](L1_FEASIBILITY_PLAN.md#2026-09-20原生崩溃故障链复核)；本文与它们冲突处以本文为准。

## 0. 结论摘要

| 类别 | 直接故障 | 已确定 | 仍未确定 | 修复方向 |
|---|---|---|---|---|
| A 字体绘制 shader 栈下溢（RVA `0x61C423`） | 空栈出栈读到 `NULL+0xC`；入栈失败是重点候选 | CALL约束栈扫描支持正常渲染路径；注册表计数3、图形初始化位1，尚未验证树节点/键/对象 | 入栈返回值、目标对象状态及首次失配位置 | 记录push/pop、对象与堆；字体守卫只是待实机验收的缓解 |
| B 启动期归档流空指针（RVA `0x668CB5`/`0x668CD6`） | 逐条目构造拿到空流后未检查即虚调用 | 新探针记录到NULL路径；无Turbo的历史进程也会崩溃，发生于Mod/shader初始化前 | 路径何时变空；真实 `_access`/打开返回值及当时错误码 | 先验证探针命中，再记录原始路径/错误码；有限重试效果待验证 |
| C 退出时跳转到 `0xDEDEDEDE`（14:58，PID18788） | 执行无效地址，栈返回候选位于NVIDIA D3D11驱动 | 发生在shutdown日志之后；无Turbo对照也有异常退出，但没有相同现场转储 | 是否同一故障点、哪个组件损坏了指针 | 保留异常退出分类和退出码，对照取证，不当作clean_exit |

三类的故障现场均在原生代码，尚不能据此排除 Python 驱动的时序/生命周期是间接触发因素；A、B 在没有 Turbo/NetFix/REPENTOGON 模块的进程里也出现（9 份转储的模块表见 `dump-walk-all.json` 的 `nonsystem_modules`）。

## 0.1 2026-09-20 独立复核：提交时的证据边界

本次核对代码和已有报告，并重新执行本机构建/测试：DLL/注入器、clock_test、registry_test、身份门和7个地址字节核对通过，共享内存双向测试通过。Python首次回归发现test_rendering仍断言旧preconfigure参数；同步新探针开关并增加显式启用测试后，22项通过。没有启动游戏或改变杀软、驱动、存档。探针/缓解**已有实现**，不等于已经通过目标故障的实机验收。

- `cold-start-smoke/report.json`：PID28124在bridge listening之前崩溃，记录到NULL路径。`0x617EA0`对NULL路径可以直接返回失败、不调用CRT；故此处errno=2可能来自上游/之前调用，不能单凭它证明这次 `_access` 或 `fopen` 的错误。[CRT `_access` 的返回值/errno契约](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/access-waccess?view=msvc-170)
- `cold-start-12/report.json`当前仅保存6次运行，不是12次或20次成功验收；6次 `access_calls=0`、`access_retry_ok=0`、`font_calls=0`。先验证探针是否命中实际调用（含IAT目标、已加载DLL版本、路径过滤），再谈重试/字体守卫效果。`cold-start-control`为3次对照，全部 `reached_menu=false`、`exit_crash_after_shutdown`。
- `dump_walk.py`的“validated_chain”是带CALL约束的栈扫描，遇间接CALL未解析目标便继续接受；它比裸栈扫描强，但不是完整可靠展开。A类走渲染路径得到支持，不应声称穷尽了所有调用链。
- `writeProbeDump()`实际使用私有读写页、数据段和内存区域信息等flags，**没有 `MiniDumpWithFullMemory`**；必须检查新dump是否真的覆盖目标树节点/对象页。`FullMemoryInfo`不是全内存内容。[Microsoft MINIDUMP_TYPE](https://learn.microsoft.com/en-us/windows/win32/api/minidumpapiset/ne-minidumpapiset-minidump_type)
- 文件探针在失败后还会写日志、复查文件和转储；目前未保存/恢复所有返回时的errno、`_doserrno`、GetLastError，不能称作完全透明的只读探针。建议下一实现先分离诊断模式与重试模式，并保留调用者可见的错误状态。
- 当前策略：先验收探针覆盖，再取实际失败的路径/对象；不根据现有数据把火绒定为根因，不自动更改排除项，不把退出崩溃吞掉。

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

- 入栈 `FUN_00a140c0`（RVA `0x6140C0`，cdecl，一个栈参数 = 名字）：查注册表；只有"节点存在 且 对象非空 且 对象 `+4` 的 bit0 为 1"才把当前 shader 压栈、把该对象设为当前、返回 1；否则**不压栈返回 0**。
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

## 3. B 类：启动期归档挂载时 `fopen` 失败

### 3.1 机制（已确定）

- 归档挂载 `FUN_00a179c0`（`0x6179C0`）：解析路径（`0x617180` → `0x616C60` 在搜索目录表里定位）、分配 `0x837A14` 的槽位（上限 32）、`FUN_00a17ea0`（`0x617EA0`）打开容器文件、读 7 字节魔数 + 1 字节 kind（写入 `0x837A18[slot*8]`）+ 表偏移 + 条目数，然后**对每个条目**：分配 0x1C 的条目、`FUN_00a68a50`（`0x668A50`）构造 `ArchivedFile` 读全文算校验和、插入 0x8000 槽的全局哈希表、析构。
- `ArchivedFile` 构造再次按容器路径调用 `FUN_00a178d0`（`0x6178D0`）：先查"是否在已挂载归档内"（`0x617F40`，不会命中），再 `FUN_00a17ea0` → `KAGE::Filesys::File::Open`（`0x652540`）→ CRT `fopen(path,"rb")`。失败时返回空，构造函数**不检查**就把空流交给 `FUN_00a68c50`（`0x668C50`）做虚调用 → 崩溃。kind 2（29048）走 `0x668CB5`，kind 0（22060）走 `0x668CD6`。
- 挂载函数自己对容器文件的首次打开是有检查的（失败会记 `Failed to open archive file`），但每条目的重复打开没有。一个 `afterbirthp.a` 有一万多个条目，等于启动阶段对同一文件做上万次 `fopen/fclose`。
- 调用链（29048/22060 一致）：`J460_ApplicationMain+0x95F` → 挂载 → 构造 → 初始化。两次都在 `--luadebug` 纯启动、"Binding of Isaac: Repentance+ v1.9.7.17.J460" 横幅之后、shader 初始化和 mod 加载之前。

### 3.2 实机复现后的精确机制（2026-09-20 16:13，PID 28124）

启动即注入的探针（§3.3-1）第一次冷启动就复现了 B：游戏在挂载第 12 个归档时崩溃（`0x668CB5`，校验链 `ApplicationMain+0x95F` → 挂载 → 构造 → 初始化，转储在 `../runs/l1/20260920-native-crash-rootcause/cold-start-smoke/`）。探针记录到文件打开函数 `0x617EA0` 收到的**路径是 NULL、errno = 2（ENOENT）**，栈上是 `ArchivedFile` 构造（`0x668B27`）与挂载函数（`0x617CCF`）。对照反汇编：

- 逐条目重开用的是挂载时存进槽表的**已解析绝对路径**，它走 `FUN_00a17180`（`0x617180`）的绝对路径分支：`0x6171D8` 复制路径 → `0x6171E4` 调 `FUN_00a524b0`（`0x6524B0`）→ 其中 `0x6524FE` `CALL [0x00b187c4]` 即 ucrt `_access(path, 0)` → 返回 −1 时 `0x6171F1` 释放副本并返回 NULL。
- 所以 B 的重点候选起点是：**归档路径存在性检查失败，使解析结果为空**；当前 NULL 路径/ENOENT 下游记录尚不能直接确认 `_access` 失败及其当时 errno；解析结果为空后，`FUN_00a178d0` → `0x617EA0` 收到 NULL，构造函数拿到空流。此前把 `fopen` 当失败点的说法据此修正。
- `_access` 在 ucrt 里落到 `GetFileAttributesExW`。同一进程此前已对同一批文件做了 17662 次成功打开；这支持继续调查间歇性路径/访问问题，但没有失败瞬间的原始路径与底层调用结果，不能据此排除路径生命周期问题或认定过滤驱动是原因。本机运行火绒（HipsDaemon），Defender 实时保护关闭；`fltmc` 需要管理员权限，本轮没有枚举过滤驱动。

仍未确定：`_access` 失败瞬间 `GetLastError` 的具体值（下一轮探针直接钩 `ucrtbase!_access` 记录，并同时复查 `GetFileAttributesW` 与 `CreateFileW` 是否也失败），以及火绒是否为必要条件（需要用户决定是否临时把游戏目录加入排除做对照）。

### 3.3 修复方案（探针与缓解已实现，2026-09-20）

1. **启动即注入的探针（已实现）**：注入器 `--launch` 模式以 `CREATE_SUSPENDED` 创建进程、注入、等控制块离开 loading 再 `ResumeThread`（`rl/turbo/src/injector.cpp`；Python `isaac_bridge.turbo.launch_suspended`）。钩子：`0x617EA0`（stdcall `(path, File**)`，`ret 8`）记录失败路径/errno/`_doserrno`/`GetLastError`/挂载槽数/栈上的返回地址候选；`ucrtbase!_access`（按导出名定位）对绝对路径 `.a` 的失败记录同样信息并立即用 `GetFileAttributesW`/`CreateFileW` 复查、记录进程句柄数；`kFlagProbeDump` 时写带私有读写页（堆）的 minidump，每进程最多 3 份。
2. **缓解层（已实现，`kFlagFileRetry`）**：`_access` 对绝对路径 `.a` 失败时按 10/20/30/40/50 ms 退避重试最多 5 次，成功即返回 0；`0x617EA0` 层实际为20/40/60/80/100 ms，共300 ms；与 `_access` 层的150 ms不是同一预算。相对路径（如可选的 `resources/secret.a` 探测）失败是正常结果，不取证不重试。是否恢复取决于后续调用能否成功；固定睡眠累计分别为150/300 ms，还不含文件API、日志、转储及嵌套调用耗时，不能承诺端到端最多150 ms或已经修好。每次重试均有记录。
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
