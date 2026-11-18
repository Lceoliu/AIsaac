#!/bin/bash
# Create sandbox ws15 (B39 / C102: no-active starts) from the repo (branch $1, default b39-no-active), copying the
# node-specific parts from ws13 (engine tools: libabp_turbo.so of the obs-v2 abp_turbo.c, = ws9's; run_instance.sh,
# stub lists; the Steam-free engine image) with an Xvfb display of its own (:95), so the sandbox runs on host A or B
# without touching the other sandboxes' displays. The bridge Lua is the worktree's (goexplore_abplus.default_bridge_lua:
# src/bridge/abplus/abp_bridge.lua, abp-0.2.18); bridge/ gets ws9's copy (the same file) for ABP_BRIDGE_LUA.
set -e
A=/101063/AIsaac; W=$A/ws15; C=${1:-b39-no-active}; F=$A/ws13
git -C $A/repo fetch -q origin
mkdir -p $W/home $W/tmp $W/cache $W/runs $W/instances $W/tools $W/bridge
[ -d $W/src ] || git -C $A/repo worktree add --detach $W/src origin/$C
git -C $W/src log --oneline -1
cd $W
[ -L python ] || ln -s src/bridge/python python
[ -L abplus ] || ln -s src/bridge/abplus abplus
[ -L runtime ] || ln -s $(readlink $F/runtime) runtime
# (goexplore_abplus.default_preload: <src>/bridge/tools/libabp_turbo.so, as ws9 / ws13 have it)
[ -L src/bridge/tools ] || ln -s $W/tools src/bridge/tools
cp -n $A/ws9/bridge/abp_bridge.lua bridge/
cp -n $F/tools/*.txt $F/tools/run_instance.sh tools/
if cmp -s $F/src/bridge/engine/abp_turbo.c src/bridge/engine/abp_turbo.c && [ -f $F/tools/libabp_turbo.so ]; then
  cp -n $F/tools/libabp_turbo.so tools/; echo "engine unchanged since ws13: libabp_turbo.so copied"
else
  gcc -shared -fPIC -O2 -o tools/libabp_turbo.so src/bridge/engine/abp_turbo.c -ldl && echo "libabp_turbo.so rebuilt"
fi
for f in hpc_env.sh hpc_guard.sh run_evals.py; do
  sed -e 's#/101063/AIsaac/ws13#/101063/AIsaac/ws15#g' -e 's#:94#:95#g' $F/$f > $f
done
chmod +x hpc_guard.sh
cp src/bridge/ops/hpc_c102.sh . && chmod +x hpc_c102.sh
grep -n "ABP_HOME=\|DISPLAY\|Xvfb" hpc_env.sh | head -n 4
cmp src/bridge/abplus/abp_bridge.lua bridge/abp_bridge.lua && echo "bridge copy = the worktree's"
ls -la | sed -n 2,20p
