#!/bin/sh
# bm1 one-time / per-tree setup on the host:
#   [ABHOME=home] deploy.sh abphome <tree>   ABP_HOME = bm1/$ABHOME/isaac-abplus (default bm1/home; ab2's arm A, tree
#                              m1ae82d3: ABHOME=home-c99, its own libabp_turbo.so) (run_instance.sh starts the game from
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
    A=$BM/${ABHOME:-home}/isaac-abplus
    # (2026-11-18, Codex round 3: every step passes its failure on; the installed library is replaced only by a build
    # that succeeded (run.sh's rc 0: it passes gcc's on) and left a fresh non-empty file, by one rename; any failure
    # leaves the installed library as it was)
    fail() { echo "deploy.sh abphome: $*; the installed libabp_turbo.so left as it was" >&2; exit 1; }
    mkdir -p $A/tools $A/instances || fail "mkdir"
    # (docker: the runtime is mounted read-only three levels above bm1/home/isaac-abplus; direct: a link to BM1_GAME_RUNTIME)
    if [ ! -e $A/runtime ]; then if [ "$BM1_MODE" = docker ]; then ln -s "$BM1_RUNTIME_LINK" $A/runtime; else ln -s "$BM1_GAME_RUNTIME" $A/runtime; fi || fail "the runtime link"; fi
    for f in run_instance.sh null_render_e.txt stub_render_g.txt stub_render_h.txt abp_turbo.c abp_v3_native.c abp_schema_v3.h abp_step_line.h; do
      cp $T/$f $A/tools/ || fail "copying $f"
    done
    chmod +x $A/tools/run_instance.sh || fail "chmod"
    if [ -f "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" ]; then cp "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" $A/tools/ || fail "copying stub_render_f.txt"; fi
    NEW=$A/tools/libabp_turbo.so.new
    if [ -e $NEW ]; then trash $NEW || fail "a stale $NEW"; fi
    rc=0
    $BM/run.sh build-turbo ${GPU:-1} sh -c "cd $BM1_IN/${ABHOME:-home}/isaac-abplus/tools && gcc -shared -fPIC -O2 -o libabp_turbo.so.new abp_turbo.c -ldl" || rc=$?
    if [ $rc != 0 ] || [ ! -s $NEW ] || [ $NEW -ot $A/tools/abp_turbo.c ]; then
      have=$([ -s $NEW ] && echo 'a new file left' || echo 'no new library')
      [ -e $NEW ] && trash $NEW
      fail "the build failed (run.sh rc $rc, $have)"
    fi
    mv -fT $NEW $A/tools/libabp_turbo.so || fail "installing the new library"
    ls -la $A/tools
    ;;
  tree)
    D=$BM/trees/$2
    [ -e $D ] && { echo "deploy.sh: $D exists" >&2; exit 2; }
    mkdir -p $D && tar -xf $3 -C $D && trash $3
    ls $D/bridge
    ;;
esac
