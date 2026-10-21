# AIsaac

用强化学习玩《以撒的结合：Afterbirth+》。

AIsaac 直接在原版 Linux 游戏引擎上训练，让同一个策略学习战斗、探索、拾取道具和使用资源。我们想解决的不只是如何打赢一个房间，而是如何在随机生成的地图和道具组合中，把一整局游戏玩下去。

## 工作方式

- **原版游戏环境。** 使用 Lua 桥接和原生加速层控制游戏，保留原版的敌人行为、碰撞和道具效果；多个游戏进程并行采样。
- **结构化观测。** 将玩家状态、房间地形、可见实体、道具和已探索地图编码为 token，不直接使用屏幕图像。
- **带记忆的 Transformer。** 空间 Transformer 处理当前观察，时间 Transformer 结合近期决策和房间访问摘要，输出移动、射击及道具使用动作。
- **PPO 与分支搜索。** 策略通过游戏奖励学习，也可以从同一引擎状态 fork 出不同动作分支，把搜索结果用于辅助训练。

## 实验结果

C99 第二段最终模型在 512 个评估种子上的结果：

| 指标 | 结果 |
| --- | ---: |
| 每局平均通过的楼层数 | **1.953** |
| 到达第 4 层及以上 | **147 / 512（28.7%）** |
| 累计训练游戏时间 | 58,506.2 小时 |

这里的楼层数按完成并离开的楼层计。评估条件及其他实验结果见[实验记录](bridge/abplus/EXPERIMENTS.md)。

## 运行

训练需要 Linux x86-64、Python 3.10+、支持 CUDA 的 PyTorch，以及自己的 Steam 版 *The Binding of Isaac: Rebirth + Afterbirth+*（Linux v1.06.T1）。游戏需要已登录的 Steam 客户端和 X / OpenGL 环境。

```bash
git clone https://github.com/Lceoliu/AIsaac.git
cd AIsaac
python -m venv .venv
source .venv/bin/activate
pip install -r bridge/python/requirements-training.txt
export PYTHONPATH="$PWD/bridge/python"
```

游戏运行目录需另外准备：默认位于 `~/isaac-abplus`，`runtime/` 放置带 Lua 桥接加载入口的游戏副本，`tools/` 放置 `run_instance.sh`、编译好的 `libabp_turbo.so` 和渲染桩表。对应源码在 `bridge/engine/`；Python 依赖安装不会完成这一步。

环境配置好后，可以从一个小规模整局训练开始：

```bash
python bridge/python/train_tok.py \
  --groups-file bridge/abplus/catalog/scaling2_groups.json \
  --mode run --items --item-tokens 1 --global-token 1 --temporal 1 \
  --workers 8 --game-hours 100 --out runs/example
```

评估保存的模型：

```bash
python bridge/python/eval_tok.py \
  --checkpoint runs/example/checkpoints/last.pt \
  --groups-file bridge/abplus/catalog/scaling2_groups.json \
  --mode run --workers 8 --seeds 2147490000:512 \
  --out runs/example/eval
```

这是基础训练示例；C99 使用的搜索教师和训练参数可参考 `bridge/ops/hpc_c99.sh`。完整参数通过两个脚本各自的 `--help` 查看。

AIsaac 是非官方研究项目，运行时需要自行提供游戏文件。
