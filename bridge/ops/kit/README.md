# 迁移套件（kit v1）

设计：`temp/multi-gpu-design-20261111/MIGRATION_KIT.md`（未进 git）。产物有版本号 `kit-YYYYMMDD.N`，生成后不再修改：

| 产物 | 位置 | 内容 |
|---|---|---|
| 镜像 `aisaac-bench:kit-<版本>` | 各主机 Docker | 基础镜像 `aisaac-bench:20261019` + `/opt/aisaac/venv`（torch 2.7.1+cu128）+ `/opt/aisaac/pydeps`；PATH/PYTHONPATH 已设好，不挂 /bench 也能 `import torch` |
| 镜像存档 `aisaac-bench-kit-<版本>.tar.gz` | `<ROOT>/kit/` | `docker save | pigz`，只做一次，之后迁移直接拷这个文件 |
| 数据包 `aisaac-data-kit-<版本>.tar` | `<ROOT>/kit/` | runtime、tools、payloads/b45-payloads2、checkpoints/c99-segment2-final.pt、FILES.sha256 |
| 清单 `kit-<版本>.manifest.json` | `<ROOT>/kit/`，并抄进本目录 `MANIFEST.json` | 镜像 ID、两个文件的 sha256 和字节数、数据文件数、torch、最低驱动 |

游戏文件有版权：只在我们自己的机器之间传，不进镜像、不进 git、不进任何镜像仓库。

## 文件

| 文件 | 在哪跑 | 做什么 |
|---|---|---|
| `kit_build.sh <版本> <ops 提交>` | 构建机（8x3090） | 离线构建镜像（`kit_ctx.py` 生成构建上下文并修正 venv 脚本里的绝对路径，Dockerfile 没有 RUN）；新旧环境包集合和 `pip freeze` 必须一致；打数据包；存镜像文件；写清单 |
| `kit_transfer.py <源> <目标> <版本>` | PC | 经 PC 中转：分段、可续传（先在两端比对已传前缀的 sha256）、60 s 无数据就停（rc 2）、两端各算 sha256 比对 |
| `kit_transfer.py ops <主机> <提交>` | PC | 把某个提交的 `bridge/ops/kit` 和 `bm1/common.sh`、`bm1/cuda_smoke.py` 部署到 `<ROOT>/kit/ops/<提交>/`（`current` 指向它） |
| `kit_install.sh <版本>` | 目标机 | 核对存档 sha256、`docker load`、核对镜像 ID、解数据包到 `<ROOT>/data/<版本>/`、核对文件数/字节数/每个文件的 sha256、设只读 |
| `kit_doctor.sh <版本> [--no-smoke]` | 任意机器 | 只读体检，每项一行 PASS/WARN/FAIL，有 FAIL 时 rc 1；最后在一张空卡上用新镜像（不挂任何目录）跑导入测试和 CUDA 冒烟 |
| `hosts/<hostname>.env` | — | 每台机器的 ROOT、锁目录、套件用的 CPU、冒烟可用的 GPU、构建源路径 |
| `HOSTS.md`、`MANIFEST.json` | — | 主机登记、每个套件版本的清单 |
| `Dockerfile` / `Dockerfile.from-scratch` | — | 前者是套件镜像；后者只作记录（基础镜像丢了时才用），不构建 |

规则：重活（构建、存档、load、解包、冒烟）都先拿 bm1 读写锁的共享锁（会话名 `kit`，经 `bm1/common.sh`），所以不会和计时运行（独占）重叠；锁日志写在 `<ROOT>/kit/lockshim/reports/bm1.log`。重活绑在 `KIT_CPUSET`。删除一律移到 trash。

## 新机器上线（命令）

在 PC 上（`rl/bridge/ops/kit` 目录里）：

```sh
# 0. 新机器：先写 hosts/<它的 hostname>.env，在 kit_transfer.py 的 HOSTS 里加一条（连接方式、ROOT、pin），提交
# 1. 部署套件脚本
python -u kit_transfer.py ops <新主机> <提交>
# 2. 传清单、数据包、镜像存档（源 = 任一已有该版本的主机）；中断了就再跑同一条命令，从断点续传
python -u kit_transfer.py dgene-8x3090 <新主机> kit-20261111.1 | tee transfer.log
```

在新机器上：

```sh
cd <ROOT>/kit/ops/current/bridge/ops/kit
bash kit_install.sh kit-20261111.1
bash kit_doctor.sh kit-20261111.1          # 全部 PASS 或只有可接受的 WARN；FAIL 逐条处理或报用户
```

然后（不在这一步）：写 `bm1/env/<主机>.env`（`BM1_IMAGE=aisaac-bench:kit-...`）、部署代码、`kit_accept.sh`、登记 HOSTS.md。

## 出新版本

在构建机上：`python -u kit_transfer.py ops dgene-8x3090 <提交>`，然后
`cd <ROOT>/kit/ops/current/bridge/ops/kit && nohup setsid bash kit_build.sh kit-YYYYMMDD.N <提交> > <ROOT>/kit/logs/build-....log 2>&1 &`。
把 `<ROOT>/kit/kit-<版本>.manifest.json` 的内容加进 `MANIFEST.json`。存档至少放在两台远端机器上；旧版本只留上一个，其余移到 trash。

## 镜像的用法（以后切换时）

`docker run ... aisaac-bench:kit-<版本>`：解释器 `/opt/aisaac/venv/bin/python`（PATH 里第一个 `python`），`PYTHONPATH=/opt/aisaac/pydeps`。
bm1 的 `run.sh` 会用 `-e PYTHONPATH=...` 覆盖 PYTHONPATH：切换时把 `BM1_PY=/opt/aisaac/venv/bin/python`、`PYTHONPATH_EXTRA=/opt/aisaac/pydeps`，游戏 runtime 挂 `<ROOT>/data/<版本>/runtime`。
