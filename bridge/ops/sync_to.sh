#!/bin/bash
# Sync the workspace's AB+ bridge tree and abp_turbo.c (+ stub lists) to a sandbox on host 2 (~/isaac-abplus/<name>,
# default fk2) and rebuild the preload library there. usage: sync_h1.sh [name]
set -e
NAME=${1:-fk2}
ROOT=/d/Projects/fortune/Isaac
HOST=rhythmo@10.19.131.132
SSHP=""; SCPP=""
cd "$ROOT/rl/bridge"
tar -cf - --exclude=__pycache__ --exclude='*.pyc' python abplus/abp_bridge.lua abplus/catalog | \
  ssh -o BatchMode=yes $SSHP $HOST "mkdir -p ~/isaac-abplus/$NAME/tools ~/isaac-abplus/$NAME/runs && cd ~/isaac-abplus/$NAME && tar -xf - && cp abplus/abp_bridge.lua abp_bridge.lua.new && mv abp_bridge.lua.new abp_bridge.lua"
scp -q $SCPP "$ROOT/analysis/scripts/abplus/abp_turbo.c" "$ROOT/analysis/scripts/abplus/stub_render_g.txt" "$ROOT/analysis/scripts/abplus/stub_render_h.txt" $HOST:isaac-abplus/$NAME/tools/
ssh -o BatchMode=yes $SSHP $HOST "cd ~/isaac-abplus/$NAME/tools && sed -i 's/\r\$//' stub_render_g.txt stub_render_h.txt && gcc -shared -fPIC -O2 -Wall -Wno-format-truncation -o libabp_turbo.so.new abp_turbo.c -ldl && mv libabp_turbo.so.new libabp_turbo.so && sha256sum libabp_turbo.so | cut -c1-16"
