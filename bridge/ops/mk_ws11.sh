#!/bin/bash
# Create sandbox ws11 on HPC128 from the repo (abplus-training @ $1, default the branch head), copying the node-specific
# parts from ws4 (engine tools, Xvfb :94 environment, guard). Same layout as ws4 on HPC128 (ops README recipe).
set -e
A=/101063/AIsaac; W=$A/ws11; C=${1:-abplus-training}
git -C $A/repo fetch -q origin
mkdir -p $W/home $W/tmp $W/cache $W/runs $W/instances $W/tools $W/bridge
[ -d $W/src ] || git -C $A/repo worktree add --detach $W/src origin/$C
git -C $W/src log --oneline -1
cd $W
[ -L python ] || ln -s src/bridge/python python
[ -L abplus ] || ln -s src/bridge/abplus abplus
[ -L runtime ] || ln -s $(readlink $A/ws4/runtime) runtime
cp -n $A/ws4/bridge/* bridge/ 2>/dev/null || true
cp -n $A/ws4/tools/*.txt $A/ws4/tools/run_instance.sh tools/
if cmp -s $A/ws4/src/bridge/engine/abp_turbo.c src/bridge/engine/abp_turbo.c && [ -f $A/ws4/tools/libabp_turbo.so ]; then
  cp -n $A/ws4/tools/libabp_turbo.so tools/; echo "engine unchanged since ws4: libabp_turbo.so copied"
else
  gcc -shared -fPIC -O2 -o tools/libabp_turbo.so src/bridge/engine/abp_turbo.c -ldl && echo "libabp_turbo.so rebuilt"
fi
for f in hpc_env.sh hpc_guard.sh run_evals.py; do sed 's#/101063/AIsaac/ws4#/101063/AIsaac/ws11#g' $A/ws4/$f > $f; done
chmod +x hpc_guard.sh
grep -n "DISPLAY\|Xvfb" hpc_env.sh | head -n 3
ls -la | sed -n 2,20p
