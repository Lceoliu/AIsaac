#!/bin/sh
# bm1 one-time / per-tree setup on the host:
#   deploy.sh abphome <tree>   ABP_HOME = bm1/home/isaac-abplus (run_instance.sh starts the game from
#                              $HOME/isaac-abplus, HOME=$BM1_IN/home in the container): runtime -> the shared
#                              read-only runtime, tools/ from the tree's bridge/engine (libabp_turbo.so built in the
#                              container: gcc -shared -fPIC -O2 ... -ldl, as the 4090 / mk_ws*.sh), instances/ its own
#   deploy.sh tree <name> <tar>  unpack a git-archive tar (bridge/...) as trees/<name> (never over an existing one)
set -eu
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
case $1 in
  abphome)
    T=$BM/trees/$2/bridge/engine
    A=$BM/home/isaac-abplus
    mkdir -p $A/tools $A/instances
    # (docker: the runtime is mounted read-only three levels above bm1/home/isaac-abplus; direct: a link to BM1_GAME_RUNTIME)
    if [ ! -e $A/runtime ]; then if [ "$BM1_MODE" = docker ]; then ln -s "$BM1_RUNTIME_LINK" $A/runtime; else ln -s "$BM1_GAME_RUNTIME" $A/runtime; fi; fi
    for f in run_instance.sh null_render_e.txt stub_render_g.txt stub_render_h.txt abp_turbo.c; do cp $T/$f $A/tools/; done
    chmod +x $A/tools/run_instance.sh
    [ -f "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" ] && cp "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" $A/tools/
    $BM/run.sh build-turbo ${GPU:-1} sh -c "cd $BM1_IN/home/isaac-abplus/tools && gcc -shared -fPIC -O2 -o libabp_turbo.so abp_turbo.c -ldl"
    ls -la $A/tools
    ;;
  tree)
    D=$BM/trees/$2
    [ -e $D ] && { echo "deploy.sh: $D exists" >&2; exit 2; }
    mkdir -p $D && tar -xf $3 -C $D && trash $3
    ls $D/bridge
    ;;
esac
