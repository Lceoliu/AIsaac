# 首次正式训练启动包（gpu57，GPU 4–7，K=2 同步）

本文是首次正式训练的操作手册：启动命令、参数、检查点与磁盘、日志位置、停止与恢复、失败时守护脚本的行为、开训后头两小时要看的指标。

- 适用的树：`$T` = `mf9b60d6`（提交 f9b60d6），即最终训练树：门禁 3 道（2026-11-12 17:31 / 17:44 / 17:57，EXT 三项 ok、native skips 0）、A8、追溯重跑、run_w3（看门狗复验和自然结束）和启动彩排 rehearsal1 都已通过。
- 启动由协调方下指令（用户已经授权给协调方）。

## 1. 开训前确认

| 项 | 怎么确认 |
|---|---|
| 树已部署 | `ls /root/AIsaac/bm1/trees/$T/bridge/python/train_tok.py` |
| 游戏库来自合并后的树 | `md5sum /root/AIsaac/bm1/home/isaac-abplus/tools/abp_turbo.c` 与树里的 `bridge/engine/abp_turbo.c` 相同（ea617f1f…） |
| 门禁和 A8 都绿 | `reports/bm1.log` 里该树有 3 次 `gate … rc=0`（每行带 `EXT actor ok EXT learn ok EXT visible ok; native skips 0`；原生组件没编出来或其测试被 skip 时 gate 记 rc=9），还有 `job a8-… rc=0` |
| 脚本是最新的 | `sha1sum train_guard.sh xnew2.sh` 与仓库一致 |
| GPU 4–7 上没有别人的进程 | `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv`；守护脚本启动时也会查 |
| 没有别的任务占着锁 | `ls /root/AIsaac/bm1/locks/`；守护脚本会以"training"为目的等待并拿独占锁 |

## 2. 启动命令

```sh
cd /root/AIsaac/bm1
T=mf9b60d6
N=main1
RD=/root/AIsaac/train/$N
mkdir -p $RD
cat > $RD/train_args <<'EOF'
--characters 0,1,2
--run-seconds 3600 --cap-truncate 1
--stall-progress items --floor-stall-seconds 120 --stall-truncate 1
--reward heal=0.5,resource=0.05
--pickup-force 0
--teach-buffer 8000 --teacher-v2-buffer 4000
--lr 2e-4 --lr-final 2e-5 --lr-schedule cosine --lr-warmup 0.01
EOF
HOURS=5000 CKPT_EVERY=100 CKPT_KEEP=3 CKPT_MILESTONE=400 CKPT_ROTATE=delete MEM_GB=176 \
  setsid nohup ./train_guard.sh $N $T env/gpu57-k2-s0dl.env env/gpu57-k2-s1dl.env \
  > $RD/guard.out 2>&1 < /dev/null &
```

- 用户已定：`--stall-truncate 1`（无进展也按截断处理），学习率 2e-4 → 2e-5 余弦、预热 1%。
- 从零开始：不设 `INIT`，不加载任何权重。
- 起来后先看 `$RD/guard.log` 的前几行：
  - "guard start"那一行有树、GPU、CPU、176g、hours 5000；
  - "train_args:"那一行是上面的参数；
  - "attempt 1: fresh start"。

## 3. 参数

**训练命令** = `xnew2.sh` 里的固定配置（与 long2 相同），加上 `train_args`（写在后面的同名参数生效）。

| 参数 | 值 | 说明 |
|---|---|---|
| 模式 | `--mode run` | 一局可以跨楼层 |
| 角色 | `--characters 0,1,2` | Isaac、Magdalene、Cain 等权重。实际比例：约 30% 的新开局严格按权重；约 70% 从存档点开局，沿用存档点的角色，所以会略偏向活得久的角色。episodes.jsonl 的 `char0` 可以直接统计。 |
| 整局上限 | `--run-seconds 3600 --cap-truncate 1` | 到点按截断处理：不给 timeout 惩罚、不算终止、用最后一条记录的价值自举。 |
| 无进展 | `--stall-progress items --floor-stall-seconds 120` | 新房间、清房、打到怪、资源增加、获得收集品/饰品/药丸/卡牌，都算进展。120 秒无进展判 stall。 |
| 无进展的处理 | `--stall-truncate 1` | 按截断处理（不给惩罚、不算终止、用最后一条记录的价值自举）。原因是模型的输入里看不到 stall 计时。 |
| 奖励 | `--reward heal=0.5,resource=0.05` | 其余项不变 |
| 强制拾取 | `--pickup-force 0` | 关闭。起始道具底座（`--start-build-mode pedestal`）照常生成，捡不捡由策略决定。 |
| PLR | 不开 | 配置里本来就没有 |
| 在线评测 | 不做 | 训练曲线用 `train_summary.py`（第 8 节）汇总训练对局；里程碑检查点在训练结束后做 512 种子的留出评测（eval_tok.py，三个角色各自） |
| 缓冲区 | `--teach-buffer 8000 --teacher-v2-buffer 4000` | 每条记录在 trainer 里约占 1.33 MB/rank。每条记录的期望使用次数由 `--teach-reuse` 决定，缓冲区大小只决定记录有多新。 |
| 学习率 | `--lr 2e-4 --lr-final 2e-5 --lr-schedule cosine --lr-warmup 0.01` | 前 1% 从 0 线性升到峰值，之后按余弦降到终值。进度按全程累计计算，恢复后连续，两个分片相同。 |
| 训练总量 | `HOURS=5000` | 每个分片 5000 游戏小时，两片合计 10000（ISAAC_GAME_HOURS_TOTAL）。每轮每片约 1.21 游戏小时，共约 4125 轮。按 340–390× 估算墙钟约 26–29 小时。到点后守护进程正常结束（rc 0）。 |
| 同步 | K=2 | 梯度取平均、学习率不乘系数；每一步做一次哈希检查，确定性检查开启 |
| 绑核 | `gpu57-k2-s0dl/s1dl.env` | 每片 driver 1 个物理核，learner 2 个 |
| PageStore 回收 | `COLLECT_EVERY=4`（守护脚本默认） | 每 4 次 hand-off 回收一次；检查点之前和运行结束时强制回收 |
| 容器内存上限 | `MEM_GB=176` | 是上限，不是预留。main1 在 144g 下于第 368 轮触顶（缓冲区填满后稳定在 138–143 GB，worker PSS 是 long2 的两倍），协调方 2026-11-13 改为 176g；主机 251 GB，其他租户合计十几 GB |

**在这个配置下不起作用的参数**（命令行与 long2 相同、保留，只在这里注明）：

- `--start-hp`；
- `--groups-file` 的分组（`scaling2_groups.json`）：run 模式下各组只决定起点；进度里 `run win / hurt` 的分组统计仍按组给出；
- `--item-tokens / --item-shared / --item-slots`：block-memory 模型读的是可见帧，不读旧的 ROW；
- `--episode-seconds / --floor-seconds`：run 模式用 `--run-seconds`。

## 4. 检查点与磁盘

| 项 | 设置 |
|---|---|
| 保存间隔 | `CKPT_EVERY=100`：每 100 轮（约 242 个合计游戏小时，墙钟约 40 分钟）。崩溃时最多重跑 100 轮。实测每次保存耗时 0.3–0.43 s，collect 没有可见停顿。 |
| 保留 | 最近 3 个 `update-*.pt`，加上每 400 轮一个里程碑（永久保留，约 10 个），加 `last.pt`（rank 0）和 `last-rank1.pt` |
| 轮换 | `CKPT_ROTATE=delete`：轮换下来的旧检查点直接删除。这是用户只对这一类文件的批准，其他任何删除仍先进 `/root/AIsaac/trash`。每次删除都记在 `checkpoint_rotation.log`。 |

**磁盘估算**（实测：每个检查点 133 MB；episodes.jsonl 每轮每片约 26 KB；progress.csv 每轮约 5.6 KB）：

- 检查点：(3 + 约 10 个里程碑) × 133 MB ≈ 1.7 GB，加 last 两个 0.27 GB，合计约 2 GB；
- episodes.jsonl：约 110 MB/片，超过 256 MB 会轮转成 `episodes.<k>.jsonl`，全部保留；
- progress.csv：约 23 MB/片；
- 内存采样：约 30 MB；
- 合计约 2.5 GB，相对 gpu57 的约 320 GB 空余可以忽略。
- 守护脚本在可用空间低于 100 GB 时每小时警告一次，低于 50 GB 时停机。

## 5. 日志和进度

都在 `$RD = /root/AIsaac/train/<N>/` 下：

| 文件 | 内容 |
|---|---|
| `guard.log` | 每次尝试的开始、结束、原因、恢复；train_args；各类警告和停机原因（最重要的一个） |
| `<N>-s0/progress.csv`、`<N>-s1/progress.csv` | 每轮一行，列见第 8 节 |
| `<N>-s{0,1}/episodes.jsonl` | 每局一行：done、cap、stall、broken、char0、ret 等 |
| `<N>-s0/checkpoints/`、`checkpoint_rotation.log` | 检查点 |
| `<N>-s{0,1}.mem.attempt<k>.csv` | 每 30 s 一行：trainer、workers、游戏进程的 RSS/PSS，cgroup |
| `<N>-s{r}.stdout / .stderr`（`.attempt<k>` 是之前尝试的） | trainer 输出 |
| `foreign_gpu.log` | 其他租户出现在 GPU 4–7 上的进程 |
| `launcher.log` | 容器启动输出 |
| `reports/bm1.log` | 锁、全局事件 |

## 6. 停止与恢复

- **停止**：`touch $RD/STOP`。两个 rank 会在下一次 collect 时一起结束、保存 `last.pt`，守护进程结束，不再恢复。通常 1 分钟内完成。
- **恢复**：先把 STOP 移进 trash（`mv $RD/STOP /root/AIsaac/trash/$(date +%Y%m%d)/STOP-<N>-$(date +%H%M%S)`），再运行与第 2 节**完全相同**的命令。守护进程会从最新的、能完整读出的检查点继续：
  - 游戏小时、学习率、强制拾取概率都接着走；
  - worker 的随机流不重放；
  - teacher 和 CF 缓冲区从空开始（恢复后约 3 轮没有 CF 样本，约 6 轮 teacher 持有量偏低）。
- **改参数**：编辑 `$RD/train_args`，下一次尝试生效（守护进程每次尝试前读取并记入 guard.log）。守护脚本自己的参数（DISK_MIN_GB、MAX_RETRY、BACKOFF 等）写在 `$RD/guard.env` 里，同样每次尝试前读取。

## 7. 失败时守护脚本的行为

| 情形 | 行为 |
|---|---|
| 达到 HOURS（rc 0，没有 STOP，且两个 rank 中最大的 game_hours ≥ HOURS × 0.999） | 训练完成，守护进程结束（guard.log：`game hours reached`） |
| rc 0 但游戏小时没到（例如 worker 全部退出，trainer 照常结束） | 原因 early_end，警告并恢复一次；连续第二次就停下等人 |
| 某个 rank 的 stdout 里 `B52 cpp actor driver:` 不是 on（编译失败会静默回落 Python 路径） | 当场按容器 id kill，原因 driver_off，不恢复，停下等人；train_watch 记事件。`REQUIRE_DRIVER=0` 才能在没有 driver 时跑 |
| 一个 trainer 退出或被杀 | 另一个 rank 的看门狗发现后自己结束并保存；守护进程以原因 other 恢复（等待 BACKOFF × 次数） |
| 同一个原因连续失败超过 MAX_RETRY（3）次，或同一轮失败 3 次 | 停下等人 |
| 两个 rank 的哈希不一致 | 自动用 rank 0 的状态修复；连续 3 次检查都不一致就报错，守护进程停下等人 |
| CheckpointError，或检查点重算时发生编译 | 停下等人 |
| CUDA out of memory | 恢复；如果当时有别人的进程在我们的卡上，第二次就停 |
| 容器内存超过上限的 90% | guard.log 每 10 分钟警告一次（progress 的 ctr_mem_frac） |
| 内核在内存上限处杀掉 trainer（rc ≠ 0，cgroup oom_kill > 0） | 恢复一次，第二次停下；日志写明是内存上限，不是别人的进程 |
| 内核杀掉游戏或 worker 进程（trainer 还在） | 增长一次只警告；连续 3 轮在增长，或一次尝试累计达到 20，就停下 rank：原因 cgroup_mem，警告后自动续跑（缓冲区从空开始，内存回到约 52 GB）；两次内存停机相隔不足 100 轮，或全程超过 3 次，才停下等人（2026-11-13） |
| 断局（游戏或 worker 出错导致一局中断） | 按截断处理，单独记一行；有就警告；连续 3 轮都有，或一次尝试累计达到 20，就停下 |
| 单个 worker 进程死掉 | 当前没有提示：少一个 worker 继续跑，collect 变慢，唯一痕迹是 progress 的 errors 加 1000（trainer 侧的检测由 cpp-actor 在下个维护窗口加） |
| 活着的 worker 少于 4 个或全死，或 worker 都活着但一局也产不出（collect 卡死） | 看门狗：任一 rank 超过 600 s 没有新的 progress 行（尝试刚开始时放宽到 1800 s）就判卡死 → SIGINT（trainer 存 last.pt）→ 180 s 后如果同一个容器（按 id）还在就 kill → 原因 stall，恢复；同一轮第二次卡死就停下 |
| 预检失败（Xid、GPU 缺失） | 停下，系统层面什么都不碰 |
| 可用磁盘低于 50 GB | 停下 |

守护脚本只管我们自己的容器，系统层面什么都不做：不重置 GPU，不碰别人的进程。

## 8. 指标和汇总

**离线汇总**：`python3 /root/AIsaac/bm1/train_summary.py $RD --window 400 [--csv out.csv]`。只读运行目录，不碰训练进程。按每 400 轮一个窗口给出：

- 新开局（episodes.jsonl 的 `start` 为 fresh）的平均最深层数、新进入的房间数、Boss 清除数、收集品数、受伤；
- 结束方式占比：死亡、stall 截断、cap 截断、断局；
- 以上按角色（char0）再分别算一遍；
- 全体对局的零进展占比；
- 窗口内 KL 和各动作头熵的中位数、合计 ×实时、容器内存。

episodes 里没有"清房数"，用"新进入的房间数"代替。

**头两小时要看的指标**

progress.csv 的列（括号里是预期值）：

| 列 | 看什么 |
|---|---|
| `ctr_mem_gb` / `ctr_mem_frac` | 随缓冲区填充上涨：归档填满后基线约 85–88 GB，之后每条缓冲记录约 1.2–1.3 MB（每 rank）；teach 8000 约第 100 轮填满（约 105–111 GB），v2 4000 约第 600 轮后填满，稳态约 116 GB（long2 拟合，协调方 2026-11-12；彩排 u32 为 87.6 GB）。之后不应持续上涨 |
| `block_gc_s` | 每 4 轮一次完整回收，约 2 s，其余轮接近 0，平均约 0.5 s；不应随时间增长 |
| `x_real_time` | 每片约 170–200，合计约 340–390（long2 从 422 降到 342，一半是缓冲区增大造成的，现在缓冲区有上限） |
| `kl` | 学习率 2e-4 时预计会比 C99（常数 1e-4 时 0.012–0.03）更高，注意是否明显越界 |
| `ep_stall` / `ep_stall_zero` / `ep_cap` | 每轮 stall 局数、零进展 stall 局数、到上限结束的局数。long2 里零进展 stall 占全部局 31% → 44%，新口径下应明显下降 |
| `lr` / `shard_frac` | 预热约 41 轮后到 2e-4，之后余弦下降 |
| `shard_mismatches`、`block_steady_compiles`、`b42_learn_ckpt_compile_redo`、`broken_total`、`ctr_oom_kills` | 都应为 0 |
| `teach_dropped` | 累计值，含义是搜索预算放不下而放弃的受伤候选点，不是丢数据；随策略变好会下降 |
| `b42_learn_ckpt_segments` | 每轮的确定性检查次数 |
| `errors` | 应为 0；每加 1000 表示一个 worker 进程死了 |

**事件驱动的回报**（gpu57 上的只读监视脚本 + 本机的后台等待；没有事件时不唤醒，不轮询）。启动训练后紧接着在 gpu57 上运行：

```sh
setsid nohup python3 /root/AIsaac/bm1/train_watch.py $RD > $RD/train_watch.out 2>&1 < /dev/null &
```

它每分钟只读一次 guard.log 和 progress/episodes 文件，把事件逐行追加到 `$RD/events.log`（`<序号> <时间> <类型> <内容>`，类型有 guard、u300、milestone、alert、end）。本机的等待命令只取上次之后的新行。监视脚本重启后会从 `train_watch.state` 接着走，已经报过的不会重报。事件包括：

- 第 300 轮（约 2 小时）：内存的实际数值（ctr_mem_gb，预期 105–112 GB；teach_held、v2_held；trainer 和 workers 的 PSS）、block_gc_s 是否封顶、×实时走势、KL 和各头熵、学习率是否按曲线走、三个角色的实际比例、哈希不一致和稳态编译计数；
- 每个里程碑（每 400 轮）：train_summary.py 的窗口表，加 KL 中位数、熵、×实时、内存；按开局构建数（各 rank 新开局的不同 (group, seed)）的角色分布；`checkpoint_rotation.log` 的新行（第一次真实轮转在第 400 轮）；
- 守护脚本的任何恢复、警告（内存 90%、OOM、断局、worker 相关、卡死）和停下等人：立即；
- 训练到点结束：立即。

**只报不停的警戒线**（协调方 2026-11-12 定；触发时回报消息的第一行标出；停不停由协调方定，硬故障仍按第 7 节处理）：

| 警戒线 | 条件 |
|---|---|
| KL / 熵 | 最近 50 轮 KL 中位数 > 0.05，或任一动作头的熵在 100 轮内掉了一半以上 |
| 学习停滞 | 到合计 2500 游戏小时，全体对局的零进展占比没有比第 400 轮时下降，且新开局的平均清房数（这里用新进入房间数）没有上升 |
| 内存 | ctr_mem_gb > 160 GB（上限 176），或第 800 轮之后仍以每 100 轮 2 GB 以上的速度上涨 |
| 速度 | ×实时（两片合计）的 100 轮中位数 < 300 |
| v2_idle（2026-11-13） | 某个 rank 的 v2_held ≥ 256，且连续 20 轮 v2_loss_pi 为空（teacher-v2 / CF 记录只进缓冲、不参与训练）。这属于"静默关闭"：以后的门禁里按硬错误处理，main1 本次运行中仍只报告 |

## 9. 开训后的待办

- 观测格式 v3（Repentance 兼容）：设计已定稿，第一步已在树里，不改行为；其余在开训后并行做。
- trainer 侧检测 worker 死亡、新增 `workers_live` 列、纯 Python 路径全死时 rc=0 的问题：归 cpp-actor。
- 绑核细调（learner 只留 1 个核）、worker 数扫描、双卡 learner、切分均衡、CPU 记账。
- PageStore 增量标记，进一步降低回收开销。
- CF 价值目标的折扣不一致：目标按 0.999 计算并从 value_global 自举，但回归到的是 0.995 的主价值头。从 B34（2026-10-16）起就是这样。
