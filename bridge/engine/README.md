# 引擎改造层与原生组件

更新：2026-11-17。这里是 AB+ v1.06 Linux 版的加速层源码和训练用的原生组件，以本目录为准（工作区 `analysis/scripts/abplus/` 下的旧位置 2026-10-10 起只是副本）。部署时由 `bridge/ops/bm1/deploy.sh abphome` 编出 `libabp_turbo.so`：`gcc -shared -fPIC -O2 -o tools/libabp_turbo.so abp_turbo.c -ldl`。

| 文件 | 作用 |
|---|---|
| `abp_turbo.c` | `LD_PRELOAD` 进游戏的控制层。虚拟时钟、渲染跳过 / 空 GL、随机数重播、fork 克隆、原生输入、原生精简观测和 step、ROW 编码器、看门狗。环境变量开关写在文件头 |
| `stub_render_h.txt` | 渲染路径的打桩表，worker 实例的默认（B14、B15） |
| `stub_render_g.txt` | 较早的打桩表，评估默认用它 |
| `null_render_e.txt` | 空 GL 调用表（`ABP_NULLGL_LIST`） |
| `run_instance.sh` | 从 `$HOME/isaac-abplus/runtime`（自备的游戏文件副本）起一个隔离实例。每个实例有自己的存档目录，关 Steam 云、关 mod、静音 |
| `abp_visible.c` | visible-v2 帧的编码，worker 经 ctypes 调用（B45；`isaac_bridge/visible_native.py`） |
| `abp_actor.cpp` | block-memory 一套的 C++ actor：页存储、ActorDriver、lane 服务（B52、B54、B58；`isaac_bridge/actor_native.py`） |
| `abp_core.h`、`abp_lanes.h`、`abp_lineage.h` | C++ actor 的无 torch 核心：页与 arena、lane 状态机、谱系与快照（B58c、B58d） |
| `abp_learn.c` | block learner 的主机侧打包，在 GIL 之外运行（B60；`isaac_bridge/learn_native.py`） |
| `abp_schema_v3.h` | visible-v3 常量，由 `bridge/python/tools/gen_schema_v3_h.py` 生成，不要手改 |
| `core_tests/` | C++ 核心的单元测试（bm1 的 `tsan_core.sh` 用 TSAN 跑它们） |

引擎层的实验记在 [EXPERIMENTS.md](../abplus/EXPERIMENTS.md)：A7、A8、A19、B9、B11、B12、B14、B15。
