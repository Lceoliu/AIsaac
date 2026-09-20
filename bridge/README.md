# IsaacRLBridge：原版 J460 的同步训练桥接

设计见 [ENV_ARCHITECTURE.md §0.3](../docs/ENV_ARCHITECTURE.md)。原引擎同步采样、原生worker修复与PPO存取档已验收。2026-09-21默认训练入口升级为实体注意力 + 地图CNN + 因果Transformer；保留MLP对照，不把短程接入当作战斗达标。

## Transformer v1

### 并行训练（2026-09-21）

入口 `bridge/python/train_parallel.py`：2或4个独立游戏进程，独立端口、profile、options、日志及Mod存档目录，一个MaskablePPO学习器。原生只读资源共用，桥接Mod复制到每个runtime；`SteamCloud=0`仅写入worker配置。J460首先动态调用`GetUserProfileDirectoryA`，仅改USERPROFILE环境变量不足以隔离，因此启动期按worker重定向该API，不修改系统用户/注册表。

采样复用SB3 VecEnv接口，通过线程池并发等待各游戏TCP连接；并不是同一个游戏里的四个假环境。先验证过SubprocVecEnv的2实例路径，但4实例因重复加载PyTorch及观测缓冲区分配而内存不足，改为单Python运行时，游戏进程仍独立。失败报告保留。

```powershell
$env:PYTHONPATH="D:\Projects\fortune\Isaac\rl\bridge\python;D:\Projects\fortune\Isaac\rl\runs\cuda-deps;D:\Projects\fortune\Isaac\rl\runs\training-deps"
# 2/4实例等预算短测：每次PPO更新总计256条transition，epochs=4，batch_size=8
python D:\Projects\fortune\Isaac\rl\bridge\python\train_parallel.py --workers 2 --device cuda --updates 2 --out D:\Projects\fortune\Isaac\rl\runs\parallel-2
python D:\Projects\fortune\Isaac\rl\bridge\python\train_parallel.py --workers 4 --device cuda --updates 2 --out D:\Projects\fortune\Isaac\rl\runs\parallel-4
# 正式训练：所有worker累计至少512个完整回合，不是512步；每个rollout完成更新后判断停止
python D:\Projects\fortune\Isaac\rl\bridge\python\train_parallel.py --workers 4 --device cuda --episodes 512 --episode-frames 3600 --out D:\Projects\fortune\Isaac\rl\runs\parallel-512
```

`--episode-frames`为单局逻辑帧上限，30帧/秒；默认3600帧，即用户确认的120秒。死亡/清房提前结束，超时单独记为truncated。`--updates`仅供短测，会覆盖episodes停止条件。`--checkpoint`可接续模型和优化器，但512局计数从此次运行开始，旧run不重复计入。每8次更新及结束保存checkpoint；`episodes.jsonl`逐局记录worker、回报、长度和结局，`resources.jsonl`逐秒记录学习器与引擎RSS/CPU，`report.json`记录完整状态。RSS合计包含共享页，不是独占物理内存。

本机CPU版PyTorch不能调用4060；CUDA版PyTorch2.7.1+cu128安装在忽略的`runs/cuda-deps`，不替换全局环境。安装参考：[官方版本矩阵](https://pytorch.org/get-started/previous-versions/)。接口参考：[SB3 VecEnv](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)、[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)。实测状态见架构文档§0.4，未完成512局前不得记为训练验收完成。

安装/复用 `python/requirements-training.txt` 中固定版本的训练依赖（本机位于 `rl/runs/training-deps`，未安装到全局）。游戏mods中的isaac_rl_bridge是指向本仓库Mod目录的junction，无需复制到自身；新worker会读到更新的schema=3。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = 'D:\Projects\fortune\Isaac\rl\bridge\python;D:\Projects\fortune\Isaac\rl\runs\training-deps'
python D:\Projects\fortune\Isaac\rl\bridge\python\train_monstro.py --launch --model transformer --steps 512 --out D:\Projects\fortune\Isaac\rl\runs\transformer-smoke
# 与训练完全相同的历史窗口、CNN、因果注意力及动作mask；不再使用规则控制器：
$workerPid = [int](Read-Host '输入训练命令打印的游戏 PID')
python D:\Projects\fortune\Isaac\rl\bridge\python\monstro_rollout.py --pid $workerPid --render-mode visible --checkpoint D:\Projects\fortune\Isaac\rl\runs\transformer-smoke\ppo_monstro.zip --episodes 3 --frames 960 --out D:\Projects\fortune\Isaac\rl\runs\transformer-eval
```

默认64步历史、每2逻辑帧决策、256个实体/激光线段容量；超容量报错，不能静默截断。`--history`和`--entity-capacity`属于checkpoint输入规格；评估会从checkpoint恢复。`--model mlp`保留原两层64单元网络/4帧动作间隔。两个模型不能直接互载checkpoint。`--pid`与`--launch`二选一；结束后进入安全房、保留worker，实际退出验收另发WM_CLOSE并记录退出码。

新模型将完整原始历史作为每条PPO样本的一部分，因此不是给普通PPO偷偷添加一个训练时缺失的缓存。因果遮罩、padding、重置隔离、实体置换、动作mask、存取档均有测试：`python -m unittest test_transformer`。完整数据字段和边界见架构文档§0.3。

本轮通过38项离线测试、原引擎512步Transformer训练及3回合checkpoint可视化评估；参数397万，完整验收均正常退出。最终checkpoint为 `../runs/l1/20260921-transformer/transformer-final/ppo_monstro.zip`。另实测199颗同时可见弹进入Actor，没有被旧128槽限制截掉。仍无Boss击杀，尚未完成收敛与泛化训练。证据与失败实验统一见架构文档§0.3和 `../runs/l1/20260921-transformer/acceptance-summary.json`。


## 正式入口：Isaac / 零道具 / Monstro / 空地形

新会话使用 `isaac_bridge.training.IsaacTrainingEnv`。它复用底层 `env.py` 的传输，并在 `connect()` 中消费服务端主动发送的初始观测，防止 reset/step 响应错位。原 `env.py` 保留不变作为传输层；旧 feasibility 脚本尚未切换到这个入口，不要用它们代替本节验收。

```powershell
Set-Location D:\Projects\fortune\Isaac\rl\bridge\python
$env:PYTHONPATH = 'D:\Projects\fortune\Isaac\rl\runs\training-deps'
python -m unittest -v test_training test_monstro_gym test_rendering
python monstro_rollout.py --launch --frames 2700 --policies idle,track,track --out D:\Projects\fortune\Isaac\rl\runs\l1\monstro-full-episodes
$workerPid = [int](Read-Host '输入上一条命令打印的游戏 PID')
python train_monstro.py --pid $workerPid --model mlp --steps 512 --episode-frames 960 --out D:\Projects\fortune\Isaac\rl\runs\l1\monstro-ppo-smoke
# 可视化验收（不改变逻辑时钟）：
python monstro_rollout.py --pid $workerPid --render-mode visible --frames 120 --policies track --out D:\Projects\fortune\Isaac\rl\runs\l1\monstro-visible
```

`monstro_rollout.py` / `train_monstro.py` 的 `--launch` 现在先挂起创建进程、注入原生控制层，再用 `--set-stage=1` 直接开局。默认只在本worker排除 `NvCamera32.dll` / `nvspcap.dll` 录屏/照片叠加层，不改系统驱动、Steam设置或游戏EXE；用于处理已复现的NVIDIA退出故障。默认同时启用已注册物理容器的绝对路径解析修复，嵌套归档仍走原生搜索。默认 `--render-mode headless`，从启动阶段关闭渲染，**virtual_clock=false**；`--render-mode visible` 保留真实渲染。两种启动方式都使用隔离；不默认启用字体跳过、文件重试或探针转储。已有游戏使用 `--pid`（与 `--launch` 二选一），但无法追溯移除已加载的叠加层，需要重启worker才能应用启动隔离。正常结束回到安全房，保留游戏进程；验收脚本另外发送WM_CLOSE并检查实际退出码。连接等待默认30秒，可用 `--connect-timeout` 调整，尚无自动重启监督器。

- 场景定义：`scenarios/monstro_empty.lua`；固定种子 `9AM0 7PRP`，Isaac，移除全部道具，玩家 `(320,380)`，原版 Monstro `(320,220)`。
- 使用固定种子的空出生房，检查内部没有障碍；通过 `Game:Spawn` 指定房间种子生成 Monstro，不使用随机 `goto`。这是战斗训练场，不模拟完整 Boss 房奖励和转层。
- 当前 `reset_monstro()` 仅首次执行 `restart 0 → remove * → SetStartSeed → stage 1 → rewind`，建立无道具的原生房间入口快照；此后每局只执行 `rewind` 并重建本房间 Monstro。`_reset_seeded_start()` 只保留给没有已建立快照的安全退出路径，不作为每局重置。不能退回测试启动模式下曾静默换随机种子的 `seed` 控制台命令。
- 建场后等待 Monstro 可见，并验证玩家身份、心数、主动道具、敌人数和清房标记；战斗中只发合法动作，不修改位置、速度或伤害。
- 当前 v2 所有回合均先经过原生 rewind，再由引擎自然生成 Monstro，HP 为 250；房间种子仍为 `2979258163`。旧版首次未 rewind 的 312.5 HP 结果是历史记录，不混用；未强写 Boss 血量，也未确认短暂血量缩放差异的根因。
- 结果：`../runs/l1/20260919-monstro-isolated/report.json`（5 次 × 600 帧）与 `../runs/l1/20260919-monstro-isolated-stress/report.json`（20 次 × 480 帧），共 25 次建场、12600 逻辑帧，两次退出码均为 0，逐步逻辑帧增量均为 4。脚本策略未清房，不是 RL 学习成绩。
- 按用户许可停用另外 8 个 Mod，仅加载 RL bridge；原状态在 `../runs/l1/20260919-175736-stage2-live/resume-20260919-182112/protocol-fix/mod-isolation.json`。

## 当前：rewind、密集奖励与地形输入（2026-09-20，v2）

- 用户确认奖励：实际受伤每次 −1；敌人实际扣血结算每次 +0.05；另加 `实际扣血 / 该敌人初始 MaxHP`；清房 +1；死亡不额外扣分。时间上限仍独立截断。
- `combat` 每个逻辑帧统计已结算伤害。当前命中计数是“同一个敌人在同一帧有实际扣血”的一次结算，同帧多发命中会合并；无敌期伤害请求不算命中/受伤。当前单 Boss 空房不区分环境伤害来源；推广到其他房间前需补来源归因。计数用于奖励，**不进入 actor 张量**。
- `terrain` 为 5×9×15 浮点张量：房内区域、当前玩家可通行、碰撞类别/5、坑、潜在尖刺危险。覆盖所有格子，包括空地；步步更新。是格心可通行图，不是完整连续碰撞/路径规划器。当前适配 1×1 房型；飞行可跨坑/石头，墙仍阻挡。尖刺通道尚不编码伸缩相位；火/毒液等危险仍需单独完善。
- 当前观测版本 `monstro-combat-v2`；SB3 将这张浮点地形表展平，与玩家/实体表合并输入双层 64 单元 MLP，并非 CNN。旧 v1 checkpoint 不能直接加载到新观测空间；新版从头训练。
- 实机受伤一次、对 250 HP Boss 造成 10 伤害，奖励为 **−0.91**；无敌期再次尝试受伤奖励 **0**。受伤、死亡、清房后 rewind 均恢复 Isaac 六点心数、无道具、Monstro 250 HP 和零奖励计数，房间重置耗时约 **0.272–0.274 秒**（3次测量，不是性能分布）。Lua 后生成的 Boss 不在入口快照里，所以 rewind 后仍需本地建场。
- 20 项离线测试通过。新版实机 PPO **512 步 / 16 epochs / 74.27 秒**，参数 L2 改变 **0.6133**，存取档与重载动作一致；4个结束回合全部死亡，回报为 −5.552 / −4.744 / −5.744 / −5.230。奖励已生效，但没有战斗达标结论。
- 另用规则控制器实测3回合：正奖励步骤 **24 / 23 / 25**，Boss 受伤比例 **36.4% / 35.0% / 36.4%**；不是 RL 学习成绩。三局在同一个 PID18788 内完成，重置类型 initialize / rewind / rewind；敌弹标记 **1642/1642** 正确；渲染调用增量和虚拟 tick 增量均为0。正常清理保留存活 worker。
- 地形物理探针：从 x=260 向右走24逻辑帧，空地/飞行越坑/飞行越石头到 x≈435.07；地面石头/坑停在 x≈290，障碍中心 x=320。与掩码一致。测试坑由运行时生成后显式设为坑碰撞，非自然房间采样；飞行由测试道具提供，rewind 后确认移除。首次探针错误及飞行石头掩码缺陷均保留证据，修复后同输入通过。
- 证据：[本轮结果](../runs/l1/20260920-rewind/live-summary.json)、[PPO 报告](../runs/l1/20260920-rewind/dense-ppo/report.json)、[地形探针](../runs/l1/20260920-rewind/terrain-probe.json)。该轮尚未完成原生崩溃修复；后续worker修复及实战结果见下节。这不是独立无窗口引擎。


## 弹幕输入覆盖核对（2026-09-20）

`main.lua::build_obs`逐个枚举当前房间实体，`entity_record`把Visible=true的敌弹（Type9）及眼泪（Type2）位置放入obs。`actor_observation`确实把位置归一化后写入Agent实体表，而不是只放进调试info。

实机PID30060：300个同逻辑帧快照，对照独立控制台枚举、桥接obs、实际actor张量中的类型/数量；122个快照有敌弹，累计947条敌弹记录，单帧最多18颗敌弹/20个总实体，漏项0，原生与桥接坐标误差超过0.001的记录0；游戏正常退出0。已验证部署Mod与仓库代码一致。证据：`../runs/l1/20260920-resolver-fix/projectile-audit-final/report.json`。

范围限制：Visible是引擎标志，不是屏幕像素可见性证明；默认每4逻辑帧采一次，不保证捕获采样间生成并消失的弹；128槽由所有实体共享，超限报错而非悄悄截断。激光目前只有通用实体字段，缺长度/方向/端点，特效危险受白名单限制。当前actor不输入原生速度，也没有历史帧或循环记忆，不能据单帧位置可靠判断弹道；这些边界不能用本次Monstro样本的0漏项推成全游戏完备性。

## 崩溃排查状态

2026-09-20 18时更新：worker新增进程级NVIDIA捕获叠加层隔离；恢复叠加层的反向对照重新出现同一 `0xDEDEDEDE` / `nvwgf2um+0x91F091` 退出故障，隔离模式已通过逻辑更新与字体渲染门禁。完整实机验收与边界统一维护在 [NATIVE_CRASH_ANALYSIS.md](../docs/NATIVE_CRASH_ANALYSIS.md)；本节以下是排查过程记录。其中"栈候选含 `Game::Update` 地址"一说已被纠正：`0x2FC3A4` 属于 `Game::Render`（`0x2FBC10`），三条 A 类链都在整帧渲染路径里。

已有转储显示 `Shader stack empty` 后在 RVA `0x0061C423` 读取地址 `0xC`，位于字体绘制的 shader 栈恢复路径。**停用其他 Mod 没有解决崩溃**：PID 30604 在先前测试结束后于 19:24 崩溃；本次启动的 PID 13508、11008 也在进入战斗前崩溃。三份新转储均为相同 RVA/空指针读取，未加载 Turbo、NetFix 或 REPENTOGON 模块。证据在 `../runs/l1/20260919-training-loop/new-crashes.json` 和对应日志。此前 25 次建场与 12600 帧通过仍有效，但不能作为长期稳定结论。

上一轮 Computer Use 捕获到 Windows 锁屏，阻挡菜单操作；用户已自行关闭 ToDesk 断开自动锁屏。本轮解锁后首次冷启动 PID 29048 仍崩溃，新转储位置为 RVA `0x668CB5`：资源归档流对象为空。PID 22060 随后启动失败，转储 RVA 为 `0x668CD6`。PID 27476 成功直接开局并完成两次真实 PPO，但在 22:04:37 的清理后阶段再次出现 `Shader stack empty`，转储确认 RVA `0x61C423`、读取地址 `0xC`。这两类崩溃不能混为一个根因，训练验收成功也不等于长期稳定。随后 PID 48584 重启成功，完成种子修复的实机回归。只读探针与失败证据保存在 `../runs/l1/20260919-training-loop/resume-20260919-214224/`。未改游戏 EXE、显卡驱动或系统安全设置；不能宣称锁屏或其他 Mod 是已确认根因。

### 2026-09-20 当前worker验收（已完成）

进程级叠加层隔离、挂起启动的正式入口、物理容器路径修复、rewind后玩家资源模板已接入。最终路径修复通过20次冷启动→至少120次逻辑更新→正常退出0；不启用虚拟时钟、文件重试或字体跳过。房间门使用Close+Bar，Gym拒绝跨房状态，避免把逃入空房误记为Boss胜利。此前2048步PPO的“1次win”证据不足，已撤回有效击杀结论；不能用它声称战斗能力达标。最终实战结果见根因文档§0.4及 `../runs/l1/20260920-resolver-fix/`。

rewind不再被假定一定恢复完整玩家资源：已观察到2/6红血、0炸弹；场景脚本仅在episode开始补齐红血、恢复1炸弹/0钥匙/0金币，不在战斗中回血，仍不重新启动游戏或重建楼层。

### 历史：2026-09-20 崩溃回归

- 上次保留的 PID48584 后来也在同一字体渲染位置崩溃。新 PID32412 连续战斗第 4 回合再次崩溃，训练器收到连接错误；另一进程 PID42132 则完成 20 回合（335.37秒）及正常清理。**无代码修复的基线也能偶尔通过20回合，因此单次通过不能作为原生崩溃已修复的证据。**
- 已修正 Python 的二次故障：`MonstroGymEnv.step/reset` 记录传输失败；`close` 不再向已断开的 worker 发安全房重置，保留首个错误。采样器与 PPO 报告明确写 `worker_error` / `broken_connection_closed`，不把进程故障伪装成死亡、截断或安全清理。
- 16项离线测试通过；受控故障验证旧版会被清理异常覆盖，修改后保留原始异常；正常实机3次9帧回合仍通过。此修复是**崩溃传播与清理修复，不是 J460 原生崩溃根因修复**。
- 用户确认默认无渲染训练、另保留可视化验收。PID42132 无渲染20回合完成（350.61秒），期间原渲染调用增量0、跳过20950次、虚拟 tick 增量0；随后正式入口512步 PPO 完成（74.96秒），参数 L2 变化0.4119135，模型存取档和再次执行通过，期间渲染调用增量仍为0。结束的4个学习回合全部死亡，不算战斗能力达标。
- 相同种子、相同静止输入9逻辑帧的模式对照：可视化18次渲染调用 → 无渲染0次/跳过18次 → 恢复可视化18次 → 重新关闭0次；四次均准确推进9逻辑帧，未推进虚拟时钟。这证明运行中能避开该渲染入口，不证明消除了所有崩溃或视觉/无渲染逐帧轨迹等价。
- 第一版无渲染门禁曾因“整个运行的 manager counter mismatch 必须为0”断言失败，错误记录保留。分阶段复核显示两种模式每次 restart 都增加1，9帧战斗步骤内均为0；20次重置加清理重置恰好21次。不能把重置引起的计数器归零误报为战斗丢帧。
- ProcDump 从启动阶段接管导致进程停在早期 WOW64 异常，已撤销该测试并停止该测试子进程；改为游戏启动后附加。一次 first-chance 监控只观察到进程退出，没有捕获完整异常转储，不能宣称已拿到完整调用栈。
- 更多战斗数据和当前 RL 方法见 [ENV_ARCHITECTURE.md §0.2](../docs/ENV_ARCHITECTURE.md)。敌弹 `projectile=true` 已在 v2 修复并实机确认。

## 历史：Gymnasium / PPO 接入（2026-09-19，v1 稀疏奖励）

- `python/isaac_bridge/monstro_gym.py`：`MonstroGymEnv`，动作 `MultiDiscrete([9,5,2,2])`，显式可见字段编码为玩家向量、128 槽实体表与 mask。实体排序不依赖内部 ID，种子、内部速度、AI 状态、事件计数、隐藏敌人数与 `info` 不进策略输入。该编码只适用于当前空地形课程，不是全游戏表示。
- `step()`：胜利 +1、死亡 -1、其他 0；超时独立返回 `truncated`，最后一步按剩余逻辑帧数推进。终止后必须 reset；close 尝试回安全出生房，再释放连接。
- `python/monstro_rollout.py`：无动作/可见状态跟踪射击的完整回合验收，输出每步原始观测、动作、奖励与终止原因。跟踪控制器只是诊断基线，不是学习策略。
- `python/train_monstro.py`：使用现成 Stable-Baselines3 PPO，记录回合与优化次数、检测参数变化、保存 checkpoint、重新加载并核对推理动作后送入引擎。
- **离线通过**：11 项测试，包括 SB3 `check_env`、胜负/超时/重置、隐藏字段与 ID 不进入策略张量、实际种子不符时拒绝返回“固定场景”、PPO 更新及存取档。`offline-unit-ppo.zip` 仍仅为单元测试产物。
- **实机通过**：三个自然死亡回合（484 / 412 / 500 逻辑帧）均返回 -1 并可继续重置；两次 9 帧边界测试按 4+4+1 推进，返回 `truncated=True`；另以显式 NPC Kill 验证清房 +1 和清房后重置，**这个正向协议测试不算策略击杀**。
- **真实 PPO**：首次 512 步已更新并存取档，但发现实际种子变化，因此只记为接入验收。修复后第二次固定种子 512 步 / 2048 战斗逻辑帧 / 16 个训练 epoch，参数 L2 变化 `0.3841785377871189`，用时 `74.8268 s`，四个结束的回合均为死亡。模型重新加载后动作一致，并成功再次送入游戏。原先 `optimizer_updates` 字段误用 SB3 的 epoch 计数，已改名 `training_epochs`，不再当成 minibatch 优化步数。
- 固定场景与模型证据：`../runs/l1/20260919-training-loop/resume-20260919-214224/fixed-seed-rollout/`、`ppo-fixed-seed/report.json` 和 `ppo-fixed-seed/ppo_monstro.zip`。逐次实际种子 `9AM0 7PRP`、房间种子 `2979258163`、stage_type=0、Isaac 六格血量、零道具、Monstro HP=312.5；这是引擎自然生成的 HP，不是脚本改血。此前 250 HP 的可变种子样本分开保留，不据此推断是难度差异。

当前依赖隔离安装在 `rl/runs/training-deps`，没有改变全局 Python 环境；可按 `python/requirements-training.txt` 安装到自己的虚拟环境。复现命令统一维护在上方“正式入口”，避免两套命令漂移。

固定 Isaac 种子与 Gymnasium 的 Python RNG seed 分开处理；不承诺引擎轨迹逐帧确定性。接口采用 [Gymnasium 的终止/截断语义](https://gymnasium.farama.org/api/env/)，训练器使用 SB3，不另写 PPO。

预览注意：不要在战斗房直接断开后再等待人工暂停，游戏会恢复实时运行。一次独立预览因此被 Monstro 杀死并返回主菜单；正式 `monstro_smoke.py` 的正常结束路径会先 `reset_safe()`，不受这个预览失误影响。

## 目录

| 路径 | 内容 |
|---|---|
| `mod/isaac_rl_bridge/main.lua` | mod 本体：TCP 服务、观测白名单、`MC_INPUT_ACTION` 注入、控制台重置、特权 `info` |
| `mod/isaac_rl_bridge/metadata.xml` | mod 元数据 |
| `python/isaac_bridge/env.py` | `IsaacBridgeEnv`（reset/step 五元组）、`Action`、`TrajectoryRecorder` |
| `python/isaac_bridge/launch.py` | 以 `--luadebug` 和每实例端口启动 `isaac-ng.exe` |
| `python/smoke_test.py` | 探针：连接、重置、随机动作、记录 JSONL、打印吞吐 |

## 安装（2026-09-19 已按用户要求完成）

1. `D:\Steam\steamapps\common\The Binding of Isaac Rebirth\mods\isaac_rl_bridge` 是指向 `mod/isaac_rl_bridge` 的目录联接（Junction），游戏目录里没有复制文件；删除联接即可还原。这是用户明确授权的唯一游戏目录改动。
2. `options.ini`（`Documents\My Games\Binding of Isaac Repentance+\`）已改 `EnableMods=1`、`PauseOnFocusLost=0`，改前备份在 `rl/runs/l1/backup/`；`EnableDebugConsole=1` 原本就是；`VSync=1` 未动（加速层跳过渲染后它不再限速）。
3. Steam 已登录后由用户手动启动游戏（必须带 `--luadebug`）。

## 运行探针

```powershell
Set-Location -LiteralPath 'D:\Projects\fortune\Isaac\rl\bridge\python'
$env:ISAAC_RL_PORT = '27015'
$env:ISAAC_RL_PRIVILEGED = '1'
python -m isaac_bridge.launch --port 27015 --privileged
# 等游戏进入标题/存档并开始一局（restart 需要已在游戏内）；然后：
python smoke_test.py --port 27015 --room "goto d.12" --steps 200 --out runs\probe.jsonl
```

游戏日志 `log.txt` 里应出现 `[IsaacRLBridge] listening on 127.0.0.1:27015`；若出现 `luasocket unavailable`，说明没有传 `--luadebug`。

## 协议

每行一个 JSON 对象。游戏发送 `{"type":"hello"}` 后，在每个决策边界发送 `{"type":"obs","event":"step|reset|query","seq":n,"obs":{...}}` 并阻塞等待指令：

| 指令 | 含义 |
|---|---|
| `{"cmd":"step","move":0-8,"shoot":0-4,"bomb":0/1,"item":0/1,"pill":0/1,"drop":0/1,"repeat":k}` | 保持该按键 k 个逻辑帧后回传观测 |
| `{"cmd":"reset","commands":["restart"],"settle":2}` | 执行控制台命令，等 `MC_POST_NEW_ROOM` 再过 settle 帧回传观测 |
| `{"cmd":"exec","command":"debug 4"}` | 执行控制台命令，回 `ok` |
| `{"cmd":"info"}` | 回特权真值（敌人 HP/AI 状态/冷却、玩家射击冷却、种子） |
| `{"cmd":"obs"}` | 立即重发观测 |
| `{"cmd":"control","enabled":false}` | 交还键盘 |
| `{"cmd":"close"}` | 断开 |

观测字段与白名单理由见设计文档 §3.3；`info` 只能用于奖励、校准和评估。0.2.0 起 `obs` 另带 `logic_frames`（mod 数到的 `MC_POST_UPDATE` 次数）与 `events`（自上次 reset 起的 `damage/tears/npc_deaths/clears` 计数，来自 `MC_ENTITY_TAKE_DMG`、`MC_POST_FIRE_TEAR`、`MC_POST_NPC_DEATH`、`MC_PRE_SPAWN_CLEAN_AWARD`，均只计数不改返回值）。

## L1 加速层与可行性实验

`isaac_bridge.turbo` 控制 `rl/turbo/isaac_turbo.dll`（`launch_turbo` 启动并注入、`attach` 附着、`TurboControl.set_turbo/request_render`）；`feasibility/stage1_control.py`、`stage2_equivalence.py`、`stage3_cost.py` 是三阶段实验脚本，用法与判据见 [L1_FEASIBILITY_PLAN.md](../docs/L1_FEASIBILITY_PLAN.md) §4–§5。`test_turbo_control.py` 不需要游戏，验证 Python 与 DLL 的共享内存布局一致。

## 阶段 0 要回答的问题

见设计文档 §3.5 的 8 条探针；每条的答案（含 `log.txt` 片段或 JSONL 样本）写回 `rl/docs/ENV_ARCHITECTURE.md`，之后才冻结契约 v1。
