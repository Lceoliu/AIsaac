#!/bin/sh
# bm1 one-time / per-tree setup on the host:
#   [ABHOME=home] deploy.sh abphome <tree>   ABP_HOME = bm1/$ABHOME/isaac-abplus (default bm1/home; ab2's arm A, tree
#                              m1ae82d3: ABHOME=home-c99, its own libabp_turbo.so) (run_instance.sh starts the game from
#                              $HOME/isaac-abplus, HOME=$BM1_IN/home in the container): runtime -> the shared
#                              read-only runtime, tools/ from the tree's bridge/engine (libabp_turbo.so built in the
#                              container: gcc -shared -fPIC -O2 ... -ldl, as the 4090 / mk_ws*.sh), instances/ its own
#   deploy.sh tree <name> <tar>  unpack a git-archive tar (bridge/...) as trees/<name> (never over an existing one)
#   [OPS_ACCEPT=f1,f2] deploy.sh ops <tree>   bm1's own scripts from trees/<tree>/bridge/ops/bm1 (2026-11-18, after a
#                              wholesale copy nearly dropped the host's GPU 5 guard):
#                              - every file written with LF line ends (a tree made on Windows can carry CRLF);
#                              - a host file that differs from the copy this command deployed last time
#                                (.ops_deployed/<file>; no record yet: from the new one) is a host edit not in the repo:
#                                nothing is written, the files are listed, exit 3 -- unless named in OPS_ACCEPT;
#                              - the new run.sh and train_guard.sh must call gpu_forbid (else exit 4, nothing written);
#                              - afterwards every file compared with the tree's (line ends aside), and gpu_forbid
#                                checked again in the deployed run.sh / train_guard.sh (exit 5 on a mismatch).
set -eu
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
case $1 in
  abphome)
    T=$BM/trees/$2/bridge/engine
    A=$BM/${ABHOME:-home}/isaac-abplus
    mkdir -p $A/tools $A/instances
    # (docker: the runtime is mounted read-only three levels above bm1/home/isaac-abplus; direct: a link to BM1_GAME_RUNTIME)
    if [ ! -e $A/runtime ]; then if [ "$BM1_MODE" = docker ]; then ln -s "$BM1_RUNTIME_LINK" $A/runtime; else ln -s "$BM1_GAME_RUNTIME" $A/runtime; fi; fi
    for f in run_instance.sh null_render_e.txt stub_render_g.txt stub_render_h.txt abp_turbo.c abp_v3_native.c abp_schema_v3.h abp_step_line.h; do cp $T/$f $A/tools/; done
    chmod +x $A/tools/run_instance.sh
    [ -f "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" ] && cp "$BM1_GAME_RUNTIME/../tools/stub_render_f.txt" $A/tools/
    # (2026-11-18, Codex round 2: a failed build exits non-zero and leaves no old library in place to pass for the new
    # one; built under a new name, checked, then moved over the old one)
    [ -e $A/tools/libabp_turbo.so.new ] && trash $A/tools/libabp_turbo.so.new
    $BM/run.sh build-turbo ${GPU:-1} sh -c "cd $BM1_IN/${ABHOME:-home}/isaac-abplus/tools && gcc -shared -fPIC -O2 -o libabp_turbo.so.new abp_turbo.c -ldl" || true
    if [ ! -s $A/tools/libabp_turbo.so.new ] || [ $A/tools/libabp_turbo.so.new -ot $A/tools/abp_turbo.c ]; then
      [ -e $A/tools/libabp_turbo.so ] && trash $A/tools/libabp_turbo.so
      echo "deploy.sh abphome: libabp_turbo.so not built (the old one moved to trash)" >&2
      exit 1
    fi
    mv -f $A/tools/libabp_turbo.so.new $A/tools/libabp_turbo.so
    ls -la $A/tools
    ;;
  ops)
    O=$BM/trees/$2/bridge/ops/bm1; K=$BM/.ops_deployed
    [ -d $O ] || { echo "deploy.sh ops: no $O" >&2; exit 2; }
    tmp=$(mktemp -d); bad=""; n=0
    lf() { tr -d '' < "$1"; }
    for f in $(cd $O && find . -type f | sed 's#^\./##' | sort); do
      n=$((n+1))
      [ -e "$BM/$f" ] || continue
      if [ -e "$K/$f" ]; then base="$K/$f"; else base="$O/$f"; fi
      lf "$BM/$f" > $tmp/host; lf "$base" > $tmp/base
      if ! cmp -s $tmp/host $tmp/base; then
        case ",${OPS_ACCEPT:-}," in *",$f,"*) ;; *) bad="$bad $f" ;; esac
      fi
    done
    if [ -n "$bad" ]; then
      echo "deploy.sh ops: host edits not in the repo (the host copy differs from the one last deployed) -- nothing written:" >&2
      for f in $bad; do echo "  $f" >&2; done
      echo "  (commit them to the repo, or name them in OPS_ACCEPT=f1,f2 to overwrite)" >&2
      rm -rf $tmp; exit 3
    fi
    for f in run.sh train_guard.sh; do
      grep -q '^ *gpu_forbid ' $O/$f || { echo "deploy.sh ops: $O/$f does not call gpu_forbid -- nothing written" >&2; rm -rf $tmp; exit 4; }
    done
    for f in $(cd $O && find . -type f | sed 's#^\./##' | sort); do
      mkdir -p "$(dirname "$BM/$f")" "$(dirname "$K/$f")"
      lf "$O/$f" > "$BM/$f.new"
      if [ -e "$BM/$f" ]; then chmod --reference="$BM/$f" "$BM/$f.new"; else case $f in *.sh) chmod +x "$BM/$f.new" ;; esac; fi
      mv "$BM/$f.new" "$BM/$f"
      cp -p "$BM/$f" "$K/$f"
    done
    m=0
    for f in $(cd $O && find . -type f | sed 's#^\./##'); do
      lf "$O/$f" > $tmp/new; cmp -s $tmp/new "$BM/$f" || { m=$((m+1)); echo "deploy.sh ops: MISMATCH $f" >&2; }
    done
    rm -rf $tmp
    for f in run.sh train_guard.sh; do grep -q '^ *gpu_forbid ' $BM/$f || { echo "deploy.sh ops: the deployed $f has no gpu_forbid" >&2; m=$((m+1)); }; done
    echo "deploy.sh ops: $n files from trees/$2, $m mismatches"
    log "deploy.sh ops $2: $n files, $m mismatches"
    [ $m = 0 ] || exit 5
    ;;
  tree)
    D=$BM/trees/$2
    [ -e $D ] && { echo "deploy.sh: $D exists" >&2; exit 2; }
    mkdir -p $D && tar -xf $3 -C $D && trash $3
    ls $D/bridge
    ;;
esac
