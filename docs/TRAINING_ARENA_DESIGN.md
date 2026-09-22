# 把原版引擎做成标准 RL 训练场：现状、参照与实施方案

**2026-09-22 状态覆盖：** 下文是2026-09-20的原版引擎方案，单worker/共用存档等旧差距已不代表当前状态。L0已完成独立原引擎并行训练；L2现已接入普通炸弹、4种真实Monstro地形、10,000种子安全出生测试、共享v3观测和Rust/Rayon批量采样。1–128环境本机采样/推理基准已完成，未开训，完整数据见 [ENV_ARCHITECTURE §0.7](ENV_ARCHITECTURE.md#07-炸弹场景分布与训练准备2026-09-22)。紧凑历史rollout与CPU二进制观测已完成，最新实测见 [§0.8](ENV_ARCHITECTURE.md#08-infra-优化与远端桌面恢复2026-09-22)：128路197→501决策/s，128×128步观测0.975GiB。当前仍是CPU仿真+CUDA推理，不是GPU仿真；下一门槛为反向更新测速、增量GPU传输和Linux部署，不是直接开训。

更新：2026-09-20。本文回答"原版 J460 worker 离一个标准 RL 训练环境还差什么、别人怎么做的、本项目怎么做"。参照项目的原文与 URL 全部在 [ARENA_RESEARCH_NOTES.md](ARENA_RESEARCH_NOTES.md)（2026-09-20 调研，每条事实附来源，查不到的明确标注），本文只引用其中已核实的条目；本项目的数字来自 [bridge/README.md](../bridge/README.md) 与 [ENV_ARCHITECTURE.md](ENV_ARCHITECTURE.md)。目标与信息边界不变（[PROJECT_SPEC.md](PROJECT_SPEC.md)）；崩溃问题单独见 [NATIVE_CRASH_ANALYSIS.md](NATIVE_CRASH_ANALYSIS.md)。

## 0. 结论

一个"标准训练场"由八个能力组成。本项目已具备其中的**同步单步、游戏内 reset、跳渲染、固定种子**四项，缺**多实例并行、存档隔离、加速时钟、崩溃监督**四项，观测传输是 JSON 也需要压缩。这些缺项没有一项需要引擎源码或修改 `isaac-ng.exe`：多实例与存档隔离靠"每 worker 一个进程 + 环境变量/钩子"，加速靠已有的虚拟时钟但必须按 SoulsGym 的经验限制在 ≤3× 并先做 1× 对照，监督靠训练器自建。

| 能力 | 标准做法（参照） | 本项目现状 | 差距 |
|---|---|---|---|
| 同步单步 | SC2 非实时模式 `RequestStep.count`；Dota 2 gRPC↔Lua 互相阻塞；Isaac-RL 在 `MC_POST_UPDATE` 内 `receive` 阻塞 | Lua mod 在 `MC_POST_UPDATE` 阻塞，`step(a, k)` 恰好 k 个逻辑帧，已实机验证 | 无 |
| reset | 模拟器类用 save-state（ALE `cloneState`、PyBoy `load_state`）；原生游戏用游戏内命令（Isaac-RL `restart 0`/`seed`）或写内存（SoulsGym 传送+回血） | 首次建入口快照，此后 `rewind` + 本地 Boss 模板，约 0.27 s | 无（任意帧快照不可得，已接受） |
| 无头 | SC2 有引擎级 `-headlessNoRender`；Dota/Isaac-RL 隐藏窗口继续模拟 | DLL 跳过整帧渲染，窗口与 GL 上下文保留；虚拟时钟关闭 | 窗口仍可见；不是无显卡运行 |
| 加速 | SC2/Dota 引擎自带非实时；SoulsGym 用注入 DLL 改写计时器，3×，作者警告过高不稳定，建议 [1,3] | 虚拟时钟已实现，等价性未验收；Isaac-RL 曾试过多调 `Game:Update()` 导致位移变化并撤回 | 未验证；需 A/A 对照与倍率上限 |
| 多实例 | PySC2 每 `SC2Env` 一进程、`portpicker` 选端口；Isaac-RL 六个进程各一端口，18.3–20.0 steps/s；EnvPool `send/recv` 按到达顺序攒批 | 单 worker | 需要 worker 池、异步聚合与端口分配 |
| 实例隔离 | SC2 `-dataDir/-tempDir`；Dota Docker；Isaac-RL 私有 exe 副本改存档目录字符串 + `SteamCloud=0` | 所有实例共用 `Documents\My Games\...` | 需要不改 EXE 的存档目录重定向（§3.2） |
| 崩溃监督 | RLlib `restart_failed_sub_environments` + 健康探针；PySC2 `poll → terminate → kill`；Isaac-RL 明确不静默重启 | Python 已能如实记录 `worker_error`，无自动重启 | 需要带冷却与分类的监督器（§3.4） |
| 观测/动作契约 | SC2 protobuf feature layers；Dota 由 Valve 定制一次性状态采集 | 每步一行 JSON，约 23 KB（`fixtures/monstro_initial_obs.json`） | 需要二进制 actor 张量通道（§3.5） |

## 1. 参照项目各自的关键事实

- **StarCraft II**：Blizzard 提供 headless Linux 构建；非实时模式手动步进，`step_mul` 决定每次决策推进多少游戏循环；PySC2 每个环境一个进程、每进程独立端口与临时目录；单线程 200–700 游戏步/秒；进程崩溃直接抛异常、不自动重启。
- **Dota 2 / OpenAI Five**：Lua 机器人 API + 注入进程的 gRPC 服务，step 双向阻塞；引擎 30 步/秒、frameskip 4 → 7.5 决策/秒；Rollout 机器以约 1/2 实时运行以多开更多局；Rapid 规模 5 万–17 万 rollout CPU；崩溃只承认"需要重启"，没有描述机制。
- **SoulsGym（Dark Souls III）**：读写内存做观测与 reset，注入 DLL 改写 Windows 计时器实现 3× 加速，作者警告倍率过高不稳定；每步 0.1 s 游戏时间，步间暂停；单实例，多机才能并行。
- **EldenRL / Sekiro DQN**：屏幕抓取 + 按键，实时运行，约 2.4 fps；不是本项目要走的路。
- **向量化基础设施**：Gymnasium `AsyncVectorEnv`、SB3 `SubprocVecEnv` 用进程 + 管道；EnvPool 用 C++ 线程池和 `batch_size < num_envs` 的异步 `send/recv`；PufferLib 用共享内存把 Pokémon Red 推到 6000 步/秒以上；RLlib 有子环境级重启和健康探针。
- **Isaac-RL（同一游戏，Repentance 1.7.9b）**：LuaSocket + 每实例端口 + `MC_INPUT_ACTION` 覆盖，8 帧/动作，六个隐藏窗口的进程 18.3–20.0 steps/s；私有 exe 副本改同长度存档目录字符串并关 Steam Cloud；不静默重启；12,381,564 次决策、2284 局 GRU 后 Boss 零胜。它给出了"未加速实时多开"的吞吐基线。

## 2. 本项目的吞吐账

30 Hz 逻辑、4 帧一决策时，单 worker 实时上限 7.5 决策/秒；Isaac-RL 的 6 实例实测约为理论值（6 × 3.75 = 22.5）的 80–90%。据此估算（不含重置与训练开销）：

| 配置 | 决策/秒 | 决策/天 |
|---|---|---|
| 1 worker 实时（现状） | 7.5 | 6.5 × 10⁵ |
| 8 worker 实时 | 约 50–60 | 4–5 × 10⁶ |
| 8 worker × 3× 时钟（若验收通过） | 约 150–180 | 1.3–1.5 × 10⁷ |

Isaac-RL 在 1.2 × 10⁷ 次决策后仍零胜，说明吞吐只是必要条件；单worker按理想7.5决策/秒达到1.2×10⁷约需18.5天（未计训练/重置），并非做不到；多实例主要缩短墙钟时间。表中8-worker数据是容量估算，不是本机实测。

## 3. 实施方案（按顺序）

### 3.1 worker 池与异步聚合

- 每个 worker = 一个 `isaac-ng.exe --luadebug` 进程 + 独立端口（`ISAAC_RL_PORT` 已支持）+ 独立 turbo 控制块（按 pid 命名已支持）。
- Python侧可以先复用 `SubprocVecEnv` 作为基准（其Python子进程各持有一个游戏连接），或用线程/`selectors` 实现固定N个槽位的 `VecEnv`。当前SB3 PPO的 `step_wait` 必须返回同一批N个环境且保持槽位对应；不能直接按到达顺序返回不足N个或换序的结果，否则会错配action/observation/done及rollout。部分批次/乱序采样需要独立collector与策略版本管理，不是替换这一方法即可。[SB3 VecEnv契约](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- 端口按 worker 序号分配（27015 + i），启动前检查占用。

### 3.2 存档与配置隔离（不改 EXE）

- 静态证据：存档根目录由 `FUN_009a9510`（RVA `0x5A9510`）决定，顺序是 `LoadLibraryA("userenv")` + `GetProcAddress("GetUserProfileDirectoryA")` → 失败或目录不存在时 `getenv("USERPROFILE")` → 再退到 `HOMEDRIVE`+`HOMEPATH`；然后拼 `Documents/My Games/Binding of Isaac Repentance+/`。`savedatapath.txt` 是它写出的信息文件（原文注明"purely informational"）。
- 方案：turbo DLL 在启动即注入模式下钩住 `userenv!GetUserProfileDirectoryA`（与 `_access` 探针同样按导出名定位），返回 `ISAAC_RL_PROFILE_DIR` 指定的每 worker 目录；API钩子须保留buffer长度协商、返回值和错误码语义，并限制影响范围；进程全局API修改可能影响Steam/叠加层。同改 `USERPROFILE` 仅是待验证后备，不等于完成隔离。每个目录预置一份 `options.ini`（`EnableMods=1`、`EnableDebugConsole=1`、`PauseOnFocusLost=0`、`SteamCloud=0`）和当前存档副本。mods 目录在游戏安装目录下，各实例只读共享。
- 与 Isaac-RL 的差别：它复制并改写 exe，本项目不动 exe，只在运行期改一个 API 的返回值。
- 验收：两个 worker 同时运行，先确认各自 `savedatapath.txt` 指向不同目录；再通过两个worker的实际存档/日志写入及文件访问记录验证不互相覆盖、不写真实用户存档。信息文件和 `SteamCloud=0` 配置本身不足以证明全部路径与云同步均隔离。

### 3.3 加速时钟

- 已有虚拟时钟（每次 `Manager::Update` 推进 1/60 s，读时钟不推进）。2×/3×仅作为候选测试档位，不能把另一游戏的3×经验当作本游戏安全上限，并按 [L1_FEASIBILITY_PLAN.md §4](L1_FEASIBILITY_PLAN.md) 的阶段 2 协议做 A/A 与 A/B 轨迹对照；只有对照通过的倍率才允许进入训练。
- 已知约束：开局淡入的 3 s 真实时间保持（`Game_Update` 第 366–410 行）不受虚拟时钟影响，rewind 重置不经过它。

### 3.4 崩溃监督器

- 每个 worker 记录 pid + 创建时间做身份核对（Isaac-RL 做法），`poll` 存活；结束顺序 `WM_CLOSE → 等待 → TerminateProcess`（PySC2 顺序）。
- 结局分类：`clean_exit`、`exit_crash_after_shutdown`（退出后异常；仅保留完整且已落盘的样本，不伪装正常退出）、`crash`（附转储路径）、`launch_error`。启动失败只重试并计数，不计为死亡；同一 worker 连续失败超过阈值就停止并保留证据（RLlib 的容忍度参数思路）。
- 重启带冷却（首版 5 s），并在探针日志里保留每次失败的 `last_fail_text`。
- 不静默：每次重启写进训练报告。

### 3.5 观测传输

- Lua 5.3 自带 `string.pack`：把 actor 需要的 12 维玩家向量、128×14 实体表、mask 和 5×9×15 地形打成小端 float32 二进制（约 10 KB，其中地形 2.7 KB），JSON 只在诊断/校准通道保留。
- 先量再改：现在每步 JSON 约 23 KB，4逻辑帧/动作即7.5条观测/秒时，单worker约170 KB/s，本机回环不是瓶颈；8 worker 时再评估。

### 3.6 窗口与焦点

- `PauseOnFocusLost=0` 已设；worker 启动后用 `ShowWindow(SW_HIDE)`/`SW_MINIMIZE` 收起窗口（Isaac-RL 的隐藏窗口继续模拟），可视化验收再显示。

### 3.7 确定性与随机化

- ALE 的教训：完全确定性会让策略记住开环动作序列。固定种子留给校准与评估；训练分布用随机种子，必要时再用留出测试决定是否加入动作延迟/重复；不默认照搬ALE的0.25 sticky-action概率。这与 [PROJECT_SPEC.md §7](PROJECT_SPEC.md) 的留出集要求一致。

## 4. 不采用的做法

- 引擎级 headless / 非实时步进（SC2）：Isaac 没有此开关，只能跳渲染 + 阻塞主线程。
- 复制并改写 exe（Isaac-RL 的存档隔离）：违反本项目"不改 EXE"的边界，用 §3.2 的钩子替代。
- 屏幕抓取 + 实时按键（EldenRL/Sekiro）：已有结构化观测与输入覆盖，不退回。
- Mesa/llvmpipe 软渲染：把渲染搬到 CPU，和多开争 CPU；跳渲染更省。
- Docker/云规模（Dota）：Windows 游戏无此路径；本项目的规模决策按 PROJECT_SPEC §6 以实测吞吐决定。
- 写进程内存做 reset（SoulsGym）：Lua API 已能做等价操作。

## 5. 验收顺序

1. 崩溃监督器 + 结局分类先落地（它也是崩溃取证循环的基础）。
2. 存档隔离钩子：两个 worker 并行 20 分钟无互相覆盖。
3. worker 池 + 异步聚合：4 worker 实时吞吐与单 worker 线性比对。
4. 二进制观测通道：与 JSON 逐字段一致性测试。
5. 时钟 2×/3× 的 A/A、A/B 对照，通过后才允许训练使用。
