#!/bin/bash
# Sync the workspace's AB+ bridge tree and abp_turbo.c (+ stub lists) to a sandbox on the HPC node (/101063/AIsaac/<name>,
# default ws1) and rebuild the preload library there. The engine runtime is GPT's unpacked image
# (/101063/AIsaac/hpc-smoke-20261009/engine-image/opt/isaac), linked as <sandbox>/runtime. usage: sync_hpc.sh [name]
set -e
NAME=${1:-ws1}
ROOT=/d/Projects/fortune/Isaac
HOST=root@10.15.171.204
SSHP="-p 30575"; SCPP="-P 30575"
D=/101063/AIsaac/$NAME
cd "$ROOT/rl/bridge"
tar -cf - --exclude=__pycache__ --exclude='*.pyc' python abplus/abp_bridge.lua abplus/catalog | \
  ssh -o BatchMode=yes $SSHP $HOST "mkdir -p $D/tools $D/runs $D/instances $D/home $D/tmp $D/cache $D/bridge && cd $D && tar -xf - && cp abplus/abp_bridge.lua bridge/abp_bridge.lua && [ -e runtime ] || ln -s /101063/AIsaac/hpc-smoke-20261009/engine-image/opt/isaac runtime"
scp -q $SCPP "$ROOT/rl/bridge/engine/abp_turbo.c" "$ROOT/rl/bridge/engine/stub_render_g.txt" "$ROOT/rl/bridge/engine/stub_render_h.txt" "$ROOT/rl/bridge/engine/null_render_e.txt" "$ROOT/rl/bridge/engine/run_instance.sh" $HOST:$D/tools/
ssh -o BatchMode=yes $SSHP $HOST "cd $D/tools && sed -i 's/\r\$//' stub_render_g.txt stub_render_h.txt null_render_e.txt run_instance.sh && cp /101063/AIsaac/hpc-smoke-20261009/run_instance.hpc.sh run_instance.sh && chmod +x run_instance.sh && gcc -shared -fPIC -O2 -Wall -Wno-format-truncation -o libabp_turbo.so.new abp_turbo.c -ldl && mv libabp_turbo.so.new libabp_turbo.so && sha256sum libabp_turbo.so | cut -c1-16"
