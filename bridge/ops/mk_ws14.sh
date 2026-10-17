#!/bin/bash
# Create sandbox ws14 on HPC160 (B38 test sandbox: --learner-device) from the repo at $1 (default the branch
# b38-learner-device, pushed to /101063/AIsaac/repo.git from the PC), copying the node-specific parts from ws9 (the C99
# sandbox: engine image, bridge Lua abp-0.2.18, stub lists, libabp_turbo.so when abp_turbo.c is unchanged). Its own
# Xvfb :95 (C99 keeps :93). Same layout as ws9 (incl. the src/bridge/tools symlink).
set -e
A=/101063/AIsaac; W=$A/ws14; C=${1:-b38-learner-device}
git -C $A/repo fetch -q origin
mkdir -p $W/home $W/tmp $W/cache $W/runs $W/instances $W/tools $W/bridge
[ -d $W/src ] || git -C $A/repo worktree add --detach $W/src origin/$C
git -C $W/src log --oneline -1
cd $W
[ -L python ] || ln -s src/bridge/python python
[ -L abplus ] || ln -s src/bridge/abplus abplus
[ -L runtime ] || ln -s $(readlink $A/ws9/runtime) runtime
[ -L src/bridge/tools ] || ln -s $W/tools src/bridge/tools
cp -n $A/ws9/bridge/* bridge/ 2>/dev/null || true
cp -n $A/ws9/tools/*.txt $A/ws9/tools/run_instance.sh tools/
if cmp -s $A/ws9/src/bridge/engine/abp_turbo.c src/bridge/engine/abp_turbo.c && [ -f $A/ws9/tools/libabp_turbo.so ]; then
  cp -n $A/ws9/tools/libabp_turbo.so tools/; echo "engine unchanged since ws9: libabp_turbo.so copied"
else
  gcc -shared -fPIC -O2 -o tools/libabp_turbo.so src/bridge/engine/abp_turbo.c -ldl && echo "libabp_turbo.so rebuilt"
fi
sed -e 's#/101063/AIsaac/ws9#/101063/AIsaac/ws14#g' -e 's#:93#:95#g' $A/ws9/hpc_env.sh > hpc_env.sh
grep -n "ABP_HOME=\|DISPLAY\|Xvfb" hpc_env.sh | head -n 4
ls -la | sed -n 2,20p
