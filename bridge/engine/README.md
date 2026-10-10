# 引擎改造层（2026-10-10 起随仓库管理）

`abp_turbo.c`（LD_PRELOAD 层：虚拟时钟、fork 克隆、原生观测 / 步进、行编码器、看门狗等）、打桩表 `stub_render_h.txt`（当前默认）/ `stub_render_g.txt`、GL 空桩表 `null_render_e.txt`、实例启动脚本 `run_instance.sh`（`$HOME/isaac-abplus` 布局；HPC 用 `ABP_HOME` 版，见 `ops/`）。

历史位置 `analysis/scripts/abplus/`（不在本仓库内）从 10-10 起只是副本；以这里为准。各主机沙箱的 `tools/` 由 `ops/sync_*.sh` 从这里同步并编译。
