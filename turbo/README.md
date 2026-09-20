# isaac_turbo：J460 原版加速层（L1）的原生部分

设计、证据与实验方案见 [rl/docs/L1_FEASIBILITY_PLAN.md](../docs/L1_FEASIBILITY_PLAN.md)。这里只放构建与文件说明。

## 做什么

| Hook（RVA） | 行为 |
|---|---|
| `glfwGetTime` `0x006266A0` | 返回虚拟时钟：每次 `Manager::Update` 返回推进 1/60 s（×(1+2⁻¹⁶)），读时钟不推进；关闭时 `real + offset` 单调续接 |
| `Manager::Update` `0x00554CD0` | 计数、推进虚拟时钟、自检 `Manager+0x4ABBC` 每次加一 |
| `Game::Update` `0x002FADC0` | 调用数与耗时（<20 ms 的进 fast 桶，排除被 Lua 桥接阻塞的帧） |
| 整帧渲染 `0x005555C0` | `flags` 含跳渲染时跳过；`render_every` / `render_request` 放行，且只在游戏本会绘制的调用上放行 |
| shader 入栈 `0x006140C0`（探针） | 返回 0 时记录注册表节点/对象 flags/栈状态/返回地址候选到日志与控制块；`kFlagProbeDump` 时写带堆的 minidump（每进程最多 3 份） |
| 字体绘制 `0x0061BD80`（缓解，`kFlagFontGuard`） | 先查 `KAGE_ColorTextureShader` 是否可用，不可用则跳过本次文本绘制（`ret 0x24`），避免空栈出栈崩溃；不改玩法状态 |
| KAGE 文件打开 `0x00617EA0`（探针 + 缓解，`kFlagFileRetry`） | 失败时记录路径 / errno / _doserrno / GetLastError；对 `*.a` 归档最多重试 5 次（20–100 ms 退避） |

探针的依据与判据见 [rl/docs/NATIVE_CRASH_ANALYSIS.md](../docs/NATIVE_CRASH_ANALYSIS.md)。`shader_registry.hpp` 是注册表/栈的纯逻辑解析，`tests/registry_test.cpp` 用伪造内存验证哈希、查找和"空栈出栈读 0xC"的算术。

不改任何玩法状态、不注入输入、不写游戏目录（日志与探针转储在 `%LOCALAPPDATA%\IsaacRL\turbo\`，可用 `ISAAC_TURBO_LOG_DIR` 改）。宿主不是 J460（PE 时间戳/大小/7 段序言/GLFW 频率任一不符）时不装任何 Hook。

启动期取证要用注入器的挂起启动模式：`isaac_turbo_inject.exe --launch <dll> <isaac-ng.exe> --luadebug`（Python：`isaac_bridge.turbo.launch_suspended`）。它以 `CREATE_SUSPENDED` 创建进程、注入、等控制块状态离开 loading 再恢复主线程，钩子因此先于第一个归档挂载生效；配置经 `ISAAC_TURBO_FONT_GUARD / ISAAC_TURBO_FILE_RETRY / ISAAC_TURBO_PROBE_DUMP / ISAAC_TURBO_SKIP_RENDER` 环境变量传入。

## 构建与测试

```powershell
pwsh -NoProfile -File D:\Projects\fortune\Isaac\rl\turbo\build.ps1
```

产物：`build/isaac_turbo.dll`、`build/isaac_turbo_inject.exe`（x86，zig cc + MinHook）。脚本依次运行 `clock_test`（虚拟时钟对限帧算术）、`probe_host`（非游戏宿主必须被拦住）、`scripts/verify_targets.py`（只读核对 exe 字节）。Python 端的共享内存测试：`python rl/bridge/python/test_turbo_control.py`。

## 文件

| 路径 | 内容 |
|---|---|
| `j460_targets.json` → `src/generated/j460_targets.hpp` | 地址与序言字节的单一来源（`scripts/gen_targets.py`） |
| `src/turbo_shared.hpp` | 4 KB 控制块布局（与 `isaac_bridge/turbo.py` 对应） |
| `src/virtual_clock.hpp` | 虚拟时钟纯逻辑 |
| `src/turbo.cpp` | DLL 本体 |
| `src/injector.cpp` | 注入器 |
| `tests/clock_test.cpp`、`tests/probe_host.cpp` | 离线测试 |

## 控制

预写配置：`ISAAC_TURBO=1`（时钟 + 跳渲染）、`ISAAC_TURBO_CLOCK=1`、`ISAAC_TURBO_SKIP_RENDER=1`、`ISAAC_TURBO_RENDER_EVERY=N`；若控制端已在注入前创建控制块（`TurboControl.preconfigure`），以控制块为准。运行中通过控制块随时切换（`TurboControl.set_turbo / set_render_every / request_render`）。
