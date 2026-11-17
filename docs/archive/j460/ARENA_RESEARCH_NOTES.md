<!-- 2026-09-20 由研究子任务生成的原始调研笔记：每条事实附 URL，[未找到] 表示没有查到可引用原文。综合结论见 TRAINING_ARENA_DESIGN.md。 -->
# 闭源商业游戏引擎 → RL 训练环境:已有项目做法调研

日期 2026-09-20。目标读者:Isaac Repentance+ RL 项目(Windows x86 exe、30 Hz 逻辑、OpenGL、Lua mod、无引擎源码)。
标注约定:**[事实]** = 有 URL 可查的原文;**[推断]** = 我的推论;**[未找到]** = 本次检索没有拿到可引用的数字/原文,不编造。
引用格式:每条事实后跟来源 URL。引号内为原文(英文原样)。

---

## 1. StarCraft II:SC2LE / PySC2 / s2client-proto

**引擎侧能力(Blizzard 提供)**
- [事实] Blizzard 官方发布 "Self contained headless linux StarCraft II builds"(3.17–4.10 各版本);接口为 protobuf,定义在 `s2clientprotocol/sc2api.proto`。https://github.com/Blizzard/s2client-proto
- [事实] SC2LE 论文:"we also provide a limited headless build that runs on Linux";"the StarCraft II API does not currently render RGB pixels. Rather, it generates a set of 'feature layers'"(2017 时)。https://ar5iv.labs.arxiv.org/html/1708.04782 (arXiv:1708.04782)
- [事实] 引擎步进速率:"StarCraft II updates the simulation 16 (at 'normal speed') or 22.4 (at 'fast speed') times per second"。同上 URL。
- [事实] proto:`RequestCreateGame` 有 `optional bool realtime = 6; // If set, the game plays in real time.`;`RequestStep` 有 `optional uint32 count = 1; // Number of game loops to simulate for the next frame.`;`InterfaceOptions` 含 `raw`、`score`、`feature_layer`(SpatialCameraSetup,"Omit to disable")、`render`、`raw_affects_selection`、`raw_crop_to_playable_area`;另有注释 "Max simulation_loop is (1<<19) before 'end of time' will occur"。https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/sc2api.proto
  - 工具摘录还给出注释 "In realtime the request will only return once the simulation game loop has reached this value. When not realtime this value is ignored."——我记忆中它挂在 `RequestObservation.game_loop` 上而非 `RequestStep`,本次未逐字核对字段归属,仅作参考。
- [事实] Linux 渲染路径(PySC2 `run_configs/platforms.py`):优先 `-eglpath libEGL.so`,其次 `-osmesapath libOSMesa.so`(软渲染),都没有则加 `-headlessNoRender` 并打印 "No GL library found, so RGB rendering will be disabled. For software rendering install libosmesa."。Windows 用 `Support64/SC2_x64.exe`。https://github.com/google-deepmind/pysc2/blob/master/pysc2/run_configs/platforms.py
- [推断] "headless" 是引擎自带的特性开关(无 GL 库时仍能跑逻辑并输出 feature layers),不是外部 hook。闭源 Isaac 没有等价开关,只能靠 DLL 跳过渲染。

**PySC2 步进语义**
- [事实] `SC2Env(step_mul=...)`:"How many game steps per agent step (action/observation). None means use the map default.";`realtime`:"Whether to use realtime mode. In this mode the game simulation automatically advances (at 22.4 gameloops per second) rather than being stepped manually. The number of game loops advanced with each call to step() won't necessarily match the step_mul specified.";`game_steps_per_episode`:"0 means no limit"。https://github.com/google-deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
- [事实] SC2LE 实验:"We act at a fix rate every 8 game steps, equivalent to about three actions per second or 180 APM";"The optimisation process runs 64 asynchronous threads using shared RMSProp"。https://ar5iv.labs.arxiv.org/html/1708.04782
- [事实] 吞吐:"single-threaded speed of a ladder map varies from 200–700 game steps per wall-clock second, which is more than an order of magnitude faster than real-time";CollectMineralShards 小地图 "1600–2000 game steps per wall-clock second"。同上。
- [事实] PySC2 README:"PySC2 should work on MacOS and Windows systems running Python 3.8+, but has only been thoroughly tested on Linux";需要 3.16.1+ 带 API 的游戏;安装路径由 `SC2PATH` 覆盖。https://github.com/google-deepmind/pysc2

**多实例 / 端口 / 临时目录 / 崩溃**
- [事实] 进程启动(`pysc2/lib/sc_process.py`):命令行 `-listen <host> -port <port> -dataDir <run_config.data_dir> -tempDir <self._tmp_dir>`,加 `-displayMode 0/1`、`-verbose`、`-dataVersion`;端口 `portpicker.pick_unused_port()`(除非 `FLAGS.sc2_port`);每实例临时目录 `tempfile.mkdtemp(prefix='sc-', dir=run_config.tmp_dir)`;存活检查 `self._proc.poll()`;关闭顺序 `terminate()` → 每秒 10 次轮询等待 → `kill()` → `wait()`。https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/sc_process.py
- [事实] 一个 `SC2Env` 内部:`self._ports = portspicker.pick_unused_ports(self._num_agents * 2)`(注释 "Reserve a whole bunch of ports for the weird multiplayer implementation");每个 agent interface 启一个 SC2 进程:`self._sc2_procs = [self._run_config.start(extra_ports=self._ports, want_rgb=interface.HasField("render")) for interface in self._interface_options]`;多个 controller 用 `self._parallel.run((c.step, step_mul) for c in self._controllers)` 同步推进。https://github.com/google-deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
- [事实] `run_parallel.RunParallel` 是 `futures.ThreadPoolExecutor` 线程池:"A thread pool for running a set of functions synchronously in parallel","mainly intended for use where the functions have a barrier and none will return until all have been called"。它服务于一个 env 内的多 controller,不是"多 env 并行"的机制。https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/run_parallel.py
- [推断] PySC2 的"多环境并行"= 每个训练进程各建一个 `SC2Env`(各自 pick 端口、各自 tmp 目录),由上层框架(A3C 64 线程等)组织;PySC2 本身不提供 vectorized env。
- [事实] 回合重启:单人单地图时 `self._controllers[0].restart()`(RestartGame),否则 `_create_join()` 重新建局/加入;`step()` 中没有对 `ProtocolError`/`ConnectionError` 的捕获——异常直接向上抛,PySC2 不自动重启崩溃的游戏进程。https://github.com/google-deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
- [事实] 连接层:`timeout_seconds = timeout_seconds or FLAGS.sc2_timeout`(默认 360 s),每秒重试一次连接 websocket,失败抛 `ConnectError('Failed to connect to the SC2 websocket. Is it up?')`;`@catch_game_end` 只在 'Game has already ended' 时返回 None;`step(self, count=1)`。https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/remote_controller.py

**AlphaStar 规模**
- [事实] DeepMind 博客(2019-01):"16 TPUs for each agent";"The AlphaStar league was run for 14 days";"Each agent experienced up to 200 years of real-time StarCraft play";"a population of agents learning from many thousands of parallel instances of StarCraft II";平均 APM "around 280",观察到动作延迟 "350ms on average"。https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/
- [未找到] Nature 论文 Methods 里"每个 agent 16,000 场并发对局 / 16 个 actor task"这一数字只出现在搜索引擎摘要中;Nature 页面因 cookie 跳转无法抓取(https://www.nature.com/articles/s41586-019-1724-z),本次未能对原文核实,不作为事实引用。

---

## 2. Dota 2 / OpenAI Five

来源主要为论文 PDF https://cdn.openai.com/dota-2.pdf(arXiv:1912.06680 https://arxiv.org/abs/1912.06680);OpenAI 博客 https://openai.com/index/openai-five/ 本次返回 403,未能直接引用。

**游戏如何被驱动(Appendix K "Dota 2 Gym Environment")**
- [事实] "Dota 2 includes a scripting API designed for building bots. The provided API is exposed through Lua and has methods for querying the visible state of the game as well as submitting actions for bots to take."
- [事实] "we implemented a helper process in Go that we load into Dota 2 through an attached debugger that exposes a gRPC server. This gRPC server implements methods to configure a game and perform an environment step."
- [事实] lockstep 阻塞:"When the step method is called in the gRPC server, it gets dispatched to the Lua code and then the method blocks until an observation arrives back from Lua ... The game blocks until an action is available."
- [事实] 隔离与多实例:"Putting the game environment behind a gRPC server allowed us to package the game into a Docker image and easily run many isolated game instances per machine. It also allowed us to easily setup, reset, and use the environment from anywhere where Docker is running."
- [事实] 脚注 12:"Originally the Lua scripting API was used to iterate and gather the visible game state, however this was somewhat slow and our final system used an all-in-one game state collection method that was added through cooperation with Valve"。
- [未找到] 论文正文没有出现 "headless"、"dedicated server"、"host_timescale" 字样(对提取文本 grep 无命中);网上流传的"headless 模式 + Docker VM 6 倍速、总体快 300 倍"说法只见于搜索摘要/二手文章,本次未找到 OpenAI 原文,不引用。

**时间步与速度(Appendix L / Sec. 3.2)**
- [事实] "The Dota 2 game engine runs at 30 steps per second ... we downsample to every 4th frame, which we call frameskip. This yields an effective observation and action rate of 7.5 frames per second.";"OpenAI Five selects an action every fourth frame, yielding approximately 20,000 steps per episode"。
- [事实] 不是严格 lockstep 推理:"we reduce our computational requirements by allowing the game and the machine learning model to run concurrently by asynchronously issuing actions with an action offset"(动作作用于 T+1 观测)。
- [事实] 游戏跑半速:"'Rollout' worker machines run self-play games. They run these games at approximately 1/2 real time, because we found that we could run slightly more than twice as many games in parallel at this speed, increasing total throughput."
- [事实] 脚注 8:"Rollout machines produce 7.5 steps per second; they send data every 256 steps, or 34 seconds of game play. Because our rollout games run at approximately half-speed, this means they push data approximately once per minute."
- [推断] 半速 + 多开 = 用 CPU 吞吐换单局延迟;引擎本身没被"加速",只是 CPU 时间片被更多实例分摊。

**Rapid 规模(Sec. 3.2、Fig. 2、Appendix C 表)**
- [事实] "The entire system runs on our custom distributed training platform called Rapid, running on Google Cloud Platform.";四类机器 Rollout CPU / Forward Pass GPU / Optimizer GPU / Controller(Redis:"The controller also stores all metadata about the state of the system, for stopping and restarting training runs");"The rollout machines run the game engine but not the policy; they communicate with a separate pool of GPU machines which run forward passes in larger batches of approximately 60."
- [事实] Appendix C 表:Number of rollout CPUs — Rerun 51,200;OpenAI Five 80,000↔172,800;Baseline 6,400。Number of rollout GPUs — 512 / 500↔1,440 / 64。Number of optimizer GPUs — 512 / 480↔1,536 / 64。Frameskip 4,Samples per segment 16,LSTM unroll 16。
- [事实] 成本结构:优化 GPU "roughly 30% of the cost by dollars spent",forward-pass GPU 30%,"actual rollouts CPUs running the selfplay games (30%)",其余 10%;总优化算力 "770±50 PFlops/s·days"。
- [事实] 崩溃处理只有一句间接记载:"rather than keeping track of every change and every time something crashed and needed to be restarted; we estimate this does not add more than 5% error"(Appendix A)。[未找到] 论文没有描述 rollout worker 的自动重启机制。
- [事实] 版本变更靠 "surgery":大约每两周一次,10 个月内 "over twenty surgeries"。https://ar5iv.labs.arxiv.org/html/1912.06680
- [事实](二手)Wikipedia:"By 2018, OpenAI Five had played around 180 years worth of games in reinforcement learning running on 256 GPUs and 128,000 CPU cores"。https://en.wikipedia.org/wiki/OpenAI_Five

---

## 3. Souls 系列:SoulsGym / SoulsAI / EldenRL / Sekiro

**SoulsGym(Dark Souls III,Windows)**
- [事实] "SoulsGym uses the games as simulations that are modified at runtime by reading and writing into the game memory";需要 "a running instance of the game";Windows 为主,Linux 走 Proton/Wine 需 "speed-hack proxy DLL";Iudex Gundyr 环境 "has been solved with a 45% winrate";图像观测列为 future work。https://github.com/amacati/SoulsGym
- [事实] speedhack 实现:向游戏进程注入 DLL(远程线程调 `LoadLibrary`),"reroutes the game's calls to Windows' performance timer functions to custom timers";速度值通过 NamedPipe 运行时更新;API `set_game_speed(value: float)`("Can't be lower than 0");限制 "only a limited number of clients can connect to the pipe"。https://soulsgym.readthedocs.io/en/dev/core/speedhack.html
- [事实] "The game runs at 3x speed to accelerate training."https://soulsgym.readthedocs.io/en/latest/ ;作者博客:Duelling Double DQN,"a win rate of about 45% within a few days of training"。https://amacati.github.io/posts/2023/05/soulsgym/
- [事实] 基类 `SoulsEnv`:`game_speed` 默认 1.0,`assert game_speed > 0`,警告 "Setting game_speed too high might result in unstable behaviour",建议范围 "[1, 3]";"During an episode the game is paused per default"——每步 `resume()` → 推进 → `pause()`;`step_size = 0.1  # Seconds between each step`(按游戏内时间计,循环 `while self.game.timed(self.game.time, t_start) < ...`);异常 `GameStateError`("Player is not loaded into the game")、`ResetNeeded`、`InvalidPlayerStateError`。https://github.com/amacati/SoulsGym/blob/master/soulsgym/envs/soulsenv.py
- [事实] Iudex 环境 reset:写内存传送 `self.game.player_pose = self.game.data.coordinates[self.ENV_ID]["fog_wall"]`,`sleep(0.2)` 等相机稳定,`reset_player_hp()` / `reset_boss_hp("iudex")`,`self.game.game_speed = 3  # Increase game speed to speed up recovery`,`self.game.time = 0`,`HARD_RESET_INTERVAL = 900  # Reset the environment every 15 minutes`,`IUDEX_MAX_HP = 1037`;观测 14 项(phase、双方 HP/SP、pose、animation 与其 duration、lock_on)。https://github.com/amacati/SoulsGym/blob/master/soulsgym/envs/darksouls3/iudex.py
- [事实] 使用要求:"Dark Souls III has to be open before launching the script",并有反作弊封号警告。https://soulsgym.readthedocs.io/en/dev/getting_started/gym.html
- [未找到] SoulsGym 没有给出 steps/s 吞吐数字。
- [事实] SoulsAI(分布式):"Since the achievable speed ups with SoulsGym environments are limited, SoulsAI enables improved training times by allowing multiple worker nodes to sample simultaneously.";Redis 中心服务器,DQN/PPO,"training can take several days";每个客户端各装一份游戏。https://github.com/amacati/SoulsAI
- [推断] Souls 路线的天花板:单机单实例 + 3x speedhack;要多实例只能多台 PC。原因之一是反作弊/Steam 单实例,原文没明说。

**EldenRL(Elden Ring,Windows 11,纯屏幕)**
- [事实] "All of our observation are derived from screen capturing the game with CV2.";要求窗口化 1920x1080 置于左上;实时运行,吞吐 "~2.4fps on a i9 13900k CPU for the training (and the game running normally)";reset 靠 `WalkToBoss.py` 从篝火走到 boss(PvP 走匹配);"When the code is `Waiting for loading screen...` you will need to ... kill the character to trigger a loading screen";模型每 500 步保存,"about 20 deaths"。https://github.com/ocram444/EldenRL

**DQN_play_sekiro(Sekiro,纯屏幕)**
- [事实] "The window size I set is 96*86";血量靠图像抓取(`find_blood_location.py`);按键用 `directkeys.py`;"In order to shorten the time of restart the game, we need game modifier"(用第三方修改器原地复活);[未找到] 无吞吐数字。https://github.com/analoganddigital/DQN_play_sekiro

---

## 4. 向量化环境基础设施

- [事实] Gymnasium `AsyncVectorEnv`:"runs multiple environments in parallel" 用 "multiprocessing processes, and pipes for communication";`shared_memory=True` 时 "observations from the worker processes are communicated back through shared variables";`daemon`、自定义 `worker`("a high chance to shoot yourself in the foot")、`autoreset_mode`。https://gymnasium.farama.org/api/vector/async_vector_env/
- [事实] SB3 `SubprocVecEnv`:"distributing each environment to its own process";"if your environment is not IO bound, the number of environments should not exceed the number of logical cores on your CPU";start method "Defaults to 'forkserver' on available platforms, and 'spawn' otherwise"(Windows 需 `if __name__ == "__main__":`);自动 reset,末观测在 `infos[env_idx]["terminal_observation"]`。https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- [事实] EnvPool(C++ 线程池 + pybind11):Atari FPS — 12 核笔记本 49,439;32 核工作站 200,428;96 核 TPU-VM 373,169;256 核 DGX-A100 1,069,922(DGX 上为 gym.vector_env 的 "14.9x / 19.6x");async API `envpool.make(..., num_envs=64, batch_size=16)`、`env.async_reset()`、`env.send(action, env_id)`、`env.recv()`。https://github.com/sail-sg/envpool ;论文摘要:"one million frames per second ... on Atari environments and three million frames per second on MuJoCo",笔记本 "2.8x that of the Python subprocess"。https://arxiv.org/abs/2206.10558
- [事实] PufferLib 0.7 博客:"sending environment data through multiprocessing.Array instead of through Pipe yields an additional ~20% performance improvement";"Pokemon Red training at over 6000 steps/second on a single desktop, up from around 3000"(PyBoy);"CleanRL trains Atari 65% faster just by switching to PufferLib's vectorization";"Python's multiprocessing caps around 10,000 steps per second per worker"。[未找到] 该页未给出核数/每 worker 环境数。https://pufferai.github.io/dev/build/html/rst/blog.html
- [事实] Sample Factory:`--num_envs_per_worker` "in high-throughput configurations this should be in 10-30 range for Atari/VizDoom";`--worker_num_splits` 双缓冲采样;`--async_rl`;`--set_workers_cpu_affinity`;`--force_envs_single_thread`。[未找到] 无崩溃重启参数。https://www.samplefactory.dev/02-configuration/cfg-params/
- [事实] Ray RLlib 容错:`restart_failed_sub_environments` — "If True and any sub-environment (within a vectorized env) throws any error during env stepping, the EnvRunner tries to restart the faulty sub-environment.";`restart_failed_env_runners`("restart the lost EnvRunner(s) as an identical copy")、`max_num_env_runner_restarts`、`delay_between_env_runner_restarts_s`、`num_consecutive_env_runner_failures_tolerance`、`env_runner_health_probe_timeout_s`。https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.fault_tolerance.html ;概览 https://docs.ray.io/en/latest/rllib/rllib-fault-tolerance.html

---

## 5. 已有 Binding of Isaac RL 尝试

**iamyanbo/Isaac-RL(Repentance 1.7.9b,Windows)** https://github.com/iamyanbo/Isaac-RL
- [事实] 桥接:"LuaSocket TCP on localhost, one port per game, newline JSON, monotonic sequence and request IDs";"The game waits at each action boundary while Python performs inference or optimization."(同步 lockstep)。
- [事实] mod 源码:`pcall(require, "socket")`,`socket.tcp()` 连 127.0.0.1(默认端口 9999),`client:settimeout(120)`;在 `MC_POST_UPDATE` 内 `client:receive("*l")` 阻塞等命令(源码注释 "Blocking at a step boundary freezes simulation during inference/optimization.");动作经 `MC_INPUT_ACTION` 覆盖输入;每动作帧数 `remaining = math.max(1, math.min(30, tonumber(cmd.frames) or 4))`;reset 用 `Isaac.ExecuteCommand("restart 0")` 或 `Isaac.ExecuteCommand("seed " .. cmd.seed)`;其余回调 MC_POST_GAME_STARTED / MC_POST_NEW_ROOM / MC_ENTITY_TAKE_DMG / MC_POST_NPC_DEATH / MC_POST_RENDER / MC_PRE_MOD_UNLOAD。https://github.com/iamyanbo/Isaac-RL/blob/main/mod/main.lua
- [事实] 多实例与存档隔离:"six separately saved normal-speed game processes (instances 0/1/2/4/5/6), one shared CPU PPO learner";"The private executable changes exactly one same-length save-directory string; it does not change gameplay code or Steam licensing.";"SteamCloud=0 is set in the private save directory";哈希与目录记录在 `runtime/installation.json`。实例→端口:0/1/2/4/5/6 → 9999/10000/10001/10003/10004/10005,实例 3 留作评估。https://github.com/iamyanbo/Isaac-RL/blob/main/OPERATIONS.md
- [事实] 速度:"Training runs at normal simulation speed. An experiment invoking `Game:Update()` between renders changed player displacement per update; that approach was removed.";窗口:"Extra game windows are hidden after startup and continue simulating normally."
- [事实] 吞吐:"18.3–20.0 steps/s" (6 worker),"10.19 steps/s on three";"36.7–38.7 s per 768 transitions";CPU PPO 更新 "0.378 s median";每动作 "8 game updates per action"。
- [推断] 30 Hz / 8 帧 = 3.75 决策/s/实例 的实时上限,6 实例理论 22.5 steps/s,实测 18–20 → 他们完全被实时时钟卡住,而非算力;这正是本项目"虚拟时钟"要解决的点。
- [事实] 崩溃:"Process or bridge errors can still end a run and must be diagnosed, not silently restarted.";运维记录里出现过 "Worker 1 native log had resource-decompression assertions/minidump.";用 PID+创建时间核对进程身份,`WM_CLOSE` 优雅关闭。https://github.com/iamyanbo/Isaac-RL/blob/main/OPERATIONS.md
- [事实] 结果:PPO + 256 单元 GRU("combat_gru_v4"),观测 "10 × 16 × 28 spatial grid plus 392 normalized scalar/entity/door features",动作 9 移动×5 射击×4 道具;"final 12,381,564 decisions","zero logged boss wins in 2,284 GRU episodes"。
- [事实] 启动注意:"Do not use `--set-stage=1` ... causes seeded resets to produce a different seed. Normal menu startup is automated instead."

**其他**
- [事实] 2-X/binding-of-isaac-ai:Afterbirth+ 的 Lua mod,需 `--luadebug` 启动,按 R 开关 agent;README 未说明算法/状态读取/吞吐。https://github.com/2-X/binding-of-isaac-ai
- [未找到] 除上述两个,未检索到其他公开的 Isaac RL 仓库(GitHub topic 页只列出 mod 开发工具类项目)。

---

## 6. 通用"游戏当 RL 环境"工程实践

**确定性步进 / lockstep**
- [事实] SC2 非 realtime 模式按 `RequestStep.count` 手动推进(见 §1);Dota 2 用 gRPC↔Lua 双向阻塞(§2);Isaac-RL 在 `MC_POST_UPDATE` 内阻塞(§5)。
- [事实] ALE:"given a particular emulator state s and a joystick input a there is a unique resulting next state s'";完全确定性导致 "agents that simply memorize an effective sequence of actions"(open-loop);建议 sticky actions,概率 0.25 重复上一动作;常用 frame skip k=4/5。https://ar5iv.labs.arxiv.org/html/1709.06009 (arXiv:1709.06009);ALE 文档:"Atari games are entirely deterministic",`frameskip`、`repeat_action_probability`、`full_action_space`。https://ale.farama.org/environments/

**save-state 式 reset**
- [事实] ALE C++:`cloneState` — "This makes a copy of the environment state. By defualt this copy does *not* include pseudorandomness ... If `include_prng` is set to true, then the pseudorandom number generator is also serialized.";`restoreState` 反向操作;`cloneSystemState` 含 RNG "suitable for serialization"。https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/ale_interface.hpp
- [事实] PyBoy:`save_state`/`load_state` 接受文件或内存对象("Remember to `seek` the in-memory buffer to the beginning before calling `PyBoy.load_state`");`tick(count, render)` 的 render=False 视为 "frameskipping";`set_emulation_speed(0)` = "unlimited";`window="null"` 无头;`pyboy.memory[addr]` 直接读写。https://docs.pyboy.dk/index.html
- [事实] stable-retro(libretro 内核):集成包自带 "savestates at the beginning of levels",奖励/结束条件来自 data.json / scenario.json 的内存地址。https://github.com/Farama-Foundation/stable-retro
- [推断] save-state 依赖模拟器能序列化整机状态;真机原生 exe 没有这个能力,替代方案是"游戏内命令 reset"(Isaac `restart`/`seed`)或"写内存回滚"(SoulsGym 传送+回血)。

**多实例 / 无头 / 软渲染**
- [事实] Dota:半速多开提升总吞吐(§2);SC2:单线程 200–700 步/s(§1);EnvPool:按核数近线性扩展(§4)。
- [事实] Mesa 环境变量:`LIBGL_ALWAYS_SOFTWARE` "if set to true, always use software rendering";`GALLIUM_DRIVER` 选 `softpipe`/`llvmpipe`;`LP_NUM_THREADS` "The default value is the number of CPU cores present."。https://docs.mesa3d.org/envvars.html ;llvmpipe "is multithreaded to take advantage of multiple CPU cores (up to 32 at this time)"。https://docs.mesa3d.org/drivers/llvmpipe.html
- [事实] Windows 上的 Mesa:mesa-dist-win 提供预编译 llvmpipe,"intended as fallback when hardware acceleration is not possible. It can only handle very light gaming with good performance.";按应用部署 = 在 exe 旁放 opengl32.dll(工具用符号链接);Mesa ≥21.3 默认暴露 OpenGL 4.5 兼容上下文。https://github.com/pal1000/mesa-dist-win
- [事实] Xvfb 用法(MineRL 文档):`xvfb-run python3 <your_script.py>` 或 `MINERL_HEADLESS=1`。https://minerl.readthedocs.io/en/v0.4.4/tutorials/first_agent.html (搜索摘要称 xvfb-run 与 NVIDIA 驱动不兼容,见 https://github.com/minerllabs/minerl/issues/224,本次未核实原文)
- [推断] llvmpipe 会把渲染成本搬到 CPU,与"多开吃 CPU"冲突;若能像本项目 DLL 那样直接跳过 draw call,比软渲染更省。

**文件系统隔离**
- [事实] SC2:每进程 `-dataDir` + `-tempDir`(mkdtemp `sc-` 前缀)(§1);Dota:Docker 镜像隔离实例(§2);Isaac-RL:私有 exe 副本改存档目录字符串 + `SteamCloud=0`(§5)。

**崩溃与重启**
- [事实] RLlib 有 sub-environment 级重启与 EnvRunner 健康探针(§4);PySC2 不自动重启,只抛 `ConnectError`(§1);OpenAI Five 承认 "crashed and needed to be restarted" 但未描述机制(§2);Isaac-RL 明确选择"不静默重启"(§5)。

---

## 7. 对本项目(Windows-only、闭源、30 Hz、OpenGL、Lua bridge)的适用性

**直接适用**
- lockstep 阻塞桥(Dota gRPC↔Lua、Isaac-RL `MC_POST_UPDATE` 阻塞):与现有 mod 一致,已验证可行。
- 每实例一个进程 + 各自端口(PySC2 portpicker、Isaac-RL 9999+n):直接照搬;Python 侧用线程/asyncio 聚合 N 个 TCP 客户端即可,不需要 SubprocVecEnv(env 本身已在别的进程)。
- EnvPool 式 async 批处理(`batch_size < num_envs`,先到先推理):适用于把 N 个游戏进程的观测按到达顺序攒批推理,避免最慢实例拖全批。
- 存档目录隔离:Isaac-RL 的"私有 exe 改同长度目录字符串 + SteamCloud=0"是已在同一游戏上跑通的做法;SC2 的 `-dataDir/-tempDir` 思路等价(Windows 也可用 junction/环境变量,视引擎读取方式而定——需在游戏外副本上验证,不动安装目录)。
- speedhack(SoulsGym:hook Windows 计时器函数,3x,NamedPipe 调速,建议 ≤3x 且"过高不稳定"):与本项目"虚拟时钟"同一思路,SoulsGym 的经验给出了安全倍率区间与失稳警告;本项目时钟尚未验证,应先以 1x 对照再逐步提速。
- 游戏内命令 reset(Isaac-RL `restart 0` / `seed N`):替代 save-state;现有 rewind/console 方案同类。
- 隐藏窗口 + 继续模拟(Isaac-RL "hidden windows ... continue simulating normally"):Windows 上 headless 的现实替代。
- 崩溃监督:RLlib 的 sub-env 重启语义 + PySC2 的 `poll()`/`terminate→kill` 顺序 + Isaac-RL 的 PID+创建时间身份核对,可组合成一个简单 supervisor(需自建,无现成 Windows 工具)。
- 确定性教训(ALE):seed 固定的 Isaac 会被 open-loop 记忆;评估时按 ALE 做法引入随机化(随机 seed / sticky action)。
- 吞吐基线:Isaac-RL 6 实例实时 ≈18–20 steps/s(8 帧/动作)可作为"未加速"的对照数字。

**不适用 / 需替代**
- SC2 headless Linux 二进制、`realtime=False` 引擎级步进、`-headlessNoRender`:都是引擎自带开关,Isaac 没有;只能靠 DLL 跳渲染 + 阻塞主线程模拟。
- Dota 的 Valve 定制"all-in-one 状态采集"与 Docker/GCP 规模(5 万–17 万 rollout CPU):无厂商合作、Windows 无 Docker 化游戏方案;状态采集只能靠 Lua API 自己拼。
- ALE/PyBoy/libretro 的 cloneState/save_state:原生 exe 无整机快照;不可用。
- Xvfb / Mesa llvmpipe(Linux):Xvfb 是 X11 概念,Windows 无;mesa-dist-win 的 opengl32.dll 可作为"无 GPU 机器上让游戏起得来"的备选,但把渲染搬到 CPU,不如直接跳过渲染。
- EnvPool 本体:要求环境以 C++ 实现并编译进 pool,无法包住外部 exe;只借鉴其 send/recv 批处理模式。
- Gymnasium AsyncVectorEnv / SB3 SubprocVecEnv 的 shared_memory 优化:观测来自 TCP JSON,瓶颈不在进程间拷贝而在游戏实时时钟;价值有限。
- SoulsGym 的"写内存传送 + 回血"reset:Isaac 的 Lua API 已能做等价操作(换房/回血),不必读写进程内存。
- EldenRL/Sekiro 的屏幕抓取 + 实时按键(2.4 fps):本项目已有结构化状态和输入覆盖,无需退回此路线。

---

## 8. 未找到 / 未核实清单
- AlphaStar Nature 论文中的"16,000 concurrent matches"(未能抓取原文)。
- OpenAI Five 博客原文(403);"headless + Docker 6x 速度 / 300x"说法无原文。
- SoulsGym、DQN_play_sekiro 的 steps/s 数字(未公布)。
- PufferLib "6000 steps/s" 的核数与 worker 配置(博客未给)。
- Sample Factory 的崩溃重启参数(文档未列)。
