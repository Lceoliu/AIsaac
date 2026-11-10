#!/bin/sh
# bm1 one-time / per-tree setup on the host:
#   deploy.sh abphome <tree>   ABP_HOME = bm1/home/isaac-abplus (run_instance.sh starts the game from
#                              $HOME/isaac-abplus, HOME=/bench/bm1/home in the container): runtime -> the shared
#                              read-only runtime, tools/ from the tree's bridge/engine (libabp_turbo.so built in the
#                              container: gcc -shared -fPIC -O2 ... -ldl, as the 4090 / mk_ws*.sh), instances/ its own
#   deploy.sh tree <name> <tar>  unpack a git-archive tar (bridge/...) as trees/<name> (never over an existing one)
#   (lane.sh: LANE=cpp -> bm1/cpp/home/isaac-abplus and bm1/cpp/trees; LANE=a1 -> bm1/home, bm1/trees as before)
set -eu
. "$(dirname "$0")/lane.sh"
case $1 in
  abphome)
    T=$LD/trees/$2/bridge/engine
    A=$LD/home/isaac-abplus
    mkdir -p $A/tools $A/instances
    [ -e $A/runtime ] || ln -s $RUNTIME_LINK $A/runtime
    for f in run_instance.sh null_render_e.txt stub_render_g.txt stub_render_h.txt abp_turbo.c; do cp $T/$f $A/tools/; done
    chmod +x $A/tools/run_instance.sh
    [ -f $BM/../home/isaac-abplus/tools/stub_render_f.txt ] && cp $BM/../home/isaac-abplus/tools/stub_render_f.txt $A/tools/
    $OPS/dock.sh build-turbo ${GPU:-${GPUS_OK%%,*}} sh -c "cd $LC/home/isaac-abplus/tools && gcc -shared -fPIC -O2 -o libabp_turbo.so abp_turbo.c -ldl"
    ls -la $A/tools
    ;;
  tree)
    D=$LD/trees/$2
    [ -e $D ] && { echo "deploy.sh: $D exists" >&2; exit 2; }
    mkdir -p $D && tar -xf $3 -C $D && gio trash $3
    ls $D/bridge
    ;;
esac
