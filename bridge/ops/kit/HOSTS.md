# 主机登记（迁移套件 v1）

不含任何凭据。连接方式只写名字；8x3090 的密码只经协调会话的 helper 使用，gpu57 用 PC 的密钥。两台机器之间没有免密关系，数据经 PC 中转（`kit_transfer.py`）。
每台机器的套件参数（ROOT、锁、CPU、冒烟用的 GPU）在 `hosts/<hostname>.env`。

## dgene-8x3090（10.19.136.184）

| 项 | 内容 |
|---|---|
| 连接 | 用户 runaiwusheng，密码登录，只经协调会话的 helper（`kit_transfer.py` 按路径导入，`KIT_HELPER_3090`）；认证失败一次就停 |
| 系统 | Ubuntu，hostname `dgene`，Docker 29.3.1（overlay2，BuildKit/buildx 0.25），nvidia runtime，cgroup v2 |
| CPU / 内存 | 2x Xeon Gold 6146，48 线程；NUMA node0 = CPU 0-11,24-35，node1 = 12-23,36-47；125 GiB |
| GPU | 8x RTX 3090，驱动 580.178.04；GPU 0、6、7 是其他租户的 |
| 事件 | 2026-11-11 01:40 UTC GPU 5 掉卡（Xid 79），07:46 UTC 整机重启后 8 张卡都在；GPU 5 不用 |
| ROOT | `~/aisaac-bench-20261019/cpu-wide`（旧布局，容器里挂成 `/bench`）；套件存档 `<ROOT>/kit/` |
| 锁 | 两把 bm1 读写锁：`<ROOT>/bm1/locks`（node0，cpp-actor 的计时运行）和 `<ROOT>/bm1/locks-node1`（node1，A1 的计时运行）；套件的每个重活在两把锁里都拿共享锁（会话名 `kit`），任一把有独占（持有或排队）就等 |
| 分道（2026-11-11 起） | cpp-actor：node0 CPU + GPU 1、2；A1：node1 CPU（12-23,36-47）+ GPU 3、4；套件的构建/存档：node1 CPU，不用 GPU；体检冒烟用 GPU 3 或 4，只在空闲且拿着两把共享锁时 |
| 规则 | 共享机器（约 19 个别人的容器）：不碰别人的容器/镜像/卷，不 prune；不联网下载；不 --privileged、不改 seccomp/sysctl/驱动、不重启；我们的容器 `aisaac-bench-*`、镜像 `aisaac-bench:*`；kit 容器带内存上限；删除 = `gio trash` 或移到 `trash/` |
| 磁盘 | / 约 750 GB 空闲（2026-11-11） |
| 已有套件 | `kit-20261111.1`（在此构建；镜像和存档文件都在） |
| 验收基线 | 未跑（kit_accept 不在 v1 这一步） |

## gpu57（10.19.136.57，ssh 端口 15654）

| 项 | 内容 |
|---|---|
| 连接 | root，PC 的密钥（`ssh -p 15654 root@10.19.136.57`）；认证失败一次就停 |
| 系统 | 裸机，共享实验平台（约 46 个别人的 `ins_*` 容器、harbor、dbcloudlab）；Ubuntu 20.04，Docker 24.0.7 + nvidia runtime，cgroup v1；有 gzip、pigz，没有 zstd |
| CPU / 内存 | 2x E5-2667 v3，32 线程；node0 = 0-7,16-23，node1 = 8-15,24-31；251 GB |
| GPU | 8x RTX 3090，全在 node0；驱动 560.28.03（低于 cu128 的 570.26，靠 CUDA 小版本兼容运行，体检给 WARN） |
| ROOT | `/root/AIsaac`：`bm1/`（两个会话的运行目录）、`home/`、`checkpoints/`、`trash/`；套件 `kit/`，数据 `data/<版本>/` |
| 锁 | `/root/AIsaac/bm1/locks`，整机一把读写锁（A1、cpp-actor、kit 共用） |
| 分道 | GPU 0 别人的；A1 用 1、2（5 做探针）；cpp-actor 用 3、4；套件体检冒烟用 6 或 7（空闲时）；套件的重活用 node1 CPU |
| 规则 | 我们是宿主 root，只能写 `/root/AIsaac/`、镜像 `aisaac-bench:*`、容器 `aisaac-bench-*`；不 apt、不在宿主 pip、不 sysctl/systemctl、不 docker prune、不碰别的租户和 /remote-home、不用他们的 Harbor；不放宽容器安全；每次只允许一个计时运行；删除 = 移到 `/root/AIsaac/trash/` |
| 磁盘 | / 约 346 GB 空闲（2026-11-11） |
| 已有套件 | `kit-20261111.1`（并排安装，现有的 `aisaac-bench:20261019` 和 venv 不动） |
| 验收基线 | 未跑 |

## host 2（单卡 3090，第三处冗余候选）

见记忆里的 host 2 条目；还没加进 `kit_transfer.py` 的 HOSTS，加进去就是多一个条目。
