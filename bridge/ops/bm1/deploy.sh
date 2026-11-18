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
#                              - probe the staged GPU behavior, then switch with rollback and retain previous versions.
set -eu
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. "$BM/common.sh"
# Exercise the actual candidate, with an allowed-device positive control and
# a forbidden-device attempt. The guard gets a stub runner so a missing
# guard cannot pass just because its child runner also refuses the device.
ops_probe_gpu() (
  set -eu
  p=$1/probe
  runner=$p/runner; guard=$p/guard
  mkdir -p "$p/bin" "$runner" "$guard"
  cp "$1/new/common.sh" "$1/new/run.sh" "$runner/"
  cp "$1/new/common.sh" "$1/new/train_guard.sh" "$guard/"
  printf 'BM1_IN=/bench\nBM1_PY=python3\nBM1_MODE=docker\nBM1_IMAGE=probe\nBM1_DOCKER_BASE=.\nBM1_GAME_RUNTIME=.\nBM1_GPUS_OK=0,1\nBM1_GPU_FORBID_REQUIRED=1\nBM1_GPU_FORBID_UUID=GPU-bad\nBM1_GPU_EXPECT=3\nBM1_TRAIN=$BM/train\nBM1_TRAIN_IN=/bench/train\nBM1_DISK=.\n' > "$runner/bm1.env"
  cp "$runner/bm1.env" "$guard/bm1.env"
  cat > "$p/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  *--query-gpu=index,uuid*) printf '0, GPU-good\n1, GPU-bad\n2, GPU-outside\n' ;;
  -L) printf 'GPU 0: good\nGPU 1: bad\nGPU 2: outside\n' ;;
  *--query-gpu=memory.used*) echo 0 ;;
esac
EOF
  cat > "$p/bin/docker" <<'EOF'
#!/bin/sh
case "$1" in
  run) printf '%s\n' "$@" > "$OPS_STARTED";;
esac
EOF
  cat > "$p/bin/df" <<'EOF'
#!/bin/sh
echo 999
EOF
  cat > "$p/bin/python3" <<'EOF'
#!/bin/sh
case "$*" in
  *'float(sys.argv[1] or 0)'*) echo 0;;
esac
EOF
  for cmd in journalctl dmesg; do printf '#!/bin/sh\nexit 0\n' > "$p/bin/$cmd"; done
  cat > "$guard/run.sh" <<'EOF'
#!/bin/sh
printf '%s\n' "$2" > "$OPS_STARTED"
EOF
  chmod +x "$p/bin/"* "$guard/run.sh"
  export PATH="$p/bin:$PATH"
  run_probe() {
    # Reset execution-affecting variables inherited from the deployment shell.
    env -u GPUS_OK -u BM1_GPU_FORBID -u RUNENV -u DOCK_ENV -u LOCKFILE \
      -u NO_RESUME -u RUN_PARENT BM="$1" OPS_STARTED="$2" NOLOCK=1 HOURS=0 \
      timeout 15 sh "$3" "$4" "$5" "$6"
  }
  for mode in run train_guard; do
    for gpu in 0 1 2; do
      marker=$p/$mode-$gpu.start
      rc=0
      if [ "$mode" = run ]; then
        run_probe "$runner" "$marker" "$runner/run.sh" probe "$gpu" true > "$marker.log" 2>&1 || rc=$?
      else
        lane=$p/lane-$gpu.env
        printf 'LANE_GPUS=%s\nLANE_CPUSET=\nLANE_MEMS=\nLANE_LEARNER_CPUS=\nLANE_DRIVER_CPUS=\nLANE_MEM=1g\nLANE_NAME=probe\nLANE_PORT=41000\n' "$gpu" > "$lane"
        run_probe "$guard" "$marker" "$guard/train_guard.sh" "probe-$gpu" tree "$lane" > "$marker.log" 2>&1 || rc=$?
      fi
      if [ "$gpu" = 0 ]; then
        if [ "$rc" != 0 ] || ! grep -q 'GPU-good' "$marker"; then
          echo "$mode: allowed GPU did not reach the launcher (rc $rc)" >&2; cat "$marker.log" >&2; exit 1
        fi
      elif [ "$gpu" = 1 ] && { [ "$rc" != 5 ] || [ -e "$marker" ] || ! grep -q 'request includes forbidden device' "$marker.log"; }; then
        echo "$mode: forbidden GPU did not get the expected refusal (rc $rc)" >&2; cat "$marker.log" >&2; exit 1
      elif [ "$gpu" = 2 ] && { [ "$rc" != 2 ] || [ -e "$marker" ]; }; then
        echo "$mode: out-of-lane GPU was not refused (rc $rc)" >&2; cat "$marker.log" >&2; exit 1
      fi
    done
  done
)

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
    [ -d "$O" ] || { echo "deploy.sh ops: no $O" >&2; exit 2; }
    lock=$BM/.ops-deploy.lock
    mkdir "$lock" || { echo "deploy.sh ops: another deployment owns $lock" >&2; exit 2; }
    stage= archive= switching=0 baseline=0
    finish_ops() {
      rc=$?
      trap - EXIT HUP INT TERM
      if [ "$switching" = 1 ]; then
        set +e
        failed=0
        if [ "$baseline" = 1 ]; then
          if [ ! -e "$stage/baseline" ] && [ -e "$K" ]; then mv -T "$K" "$stage/baseline" || failed=1; fi
          if [ -e "$archive/.ops_deployed" ]; then mv -T "$archive/.ops_deployed" "$K" || failed=1; fi
        fi
        while IFS= read -r f; do
          if [ ! -e "$stage/new/$f" ] && [ -e "$BM/$f" ]; then
            mv -T "$BM/$f" "$stage/new/$f" || failed=1
          fi
          if [ -e "$archive/files/$f" ]; then
            mv -T "$archive/files/$f" "$BM/$f" || failed=1
          fi
        done < "$stage/files"
        if [ "$failed" = 1 ]; then
          echo "deploy.sh ops: rollback needs attention: $archive and $stage" >&2
          exit 6  # keep the lock and both recovery directories
        fi
        if [ -f "$stage/dirs" ]; then
          sort -r "$stage/dirs" | while IFS= read -r dir; do rmdir "$dir" || :; done
        fi
        echo 'deploy.sh ops: rolled back; deployment baseline unchanged' >&2
      fi
      if [ -n "$stage" ]; then
        if [ -n "$archive" ]; then mv -T "$stage" "$archive/staging" || :; fi
        if [ -d "$stage" ]; then echo "deploy.sh ops: staging left at $stage" >&2; fi
      fi
      rmdir "$lock"
      exit "$rc"
    }
    trap finish_ops EXIT
    trap 'exit 5' HUP INT TERM
    trash_day=${BM1_TRASH:-$BM/trash}/$(date +%F)
    mkdir -p "$trash_day"
    archive=$(mktemp -d "$trash_day/ops.XXXXXX")
    stage=$(mktemp -d "$BM/.ops-stage.XXXXXX")
    # Renames of live files and of the baseline must stay atomic and reversible.
    [ "$(stat -c %d "$archive")" = "$(stat -c %d "$stage")" ] || {
      echo 'deploy.sh ops: BM1_TRASH must be on the same filesystem as BM' >&2; exit 2;
    }
    mkdir "$stage/new" "$stage/baseline" "$archive/files"
    (cd "$O" && find . -type f | sed 's#^\./##' | LC_ALL=C sort) > "$stage/files"
    lf() { sed -z 's/\r\n/\n/g' "$1"; }
    bad=0
    while IFS= read -r f; do
      case "$f" in bm1.env|reports/*|.ops*|trees/*) echo "deploy.sh ops: reserved path $f" >&2; exit 2 ;; esac
      if [ -L "$BM/$f" ] || { [ -e "$BM/$f" ] && [ ! -f "$BM/$f" ]; }; then
        echo "deploy.sh ops: not a regular live file: $f" >&2; exit 2
      fi
      if [ -f "$BM/$f" ]; then
        if [ -f "$K/$f" ]; then base=$K/$f; else base=$O/$f; fi
        lf "$BM/$f" > "$stage/host"
        lf "$base" > "$stage/base"
        if ! cmp -s "$stage/host" "$stage/base"; then
          case ",${OPS_ACCEPT:-}," in *",$f,"*) ;; *) echo "host edit: $f" >&2; bad=1 ;; esac
        fi
      fi
      mkdir -p "$(dirname "$stage/new/$f")" "$(dirname "$stage/baseline/$f")" "$(dirname "$archive/files/$f")"
      lf "$O/$f" > "$stage/new/$f"
      if [ -f "$BM/$f" ]; then chmod --reference="$BM/$f" "$stage/new/$f"; else
        case "$f" in *.sh) chmod +x "$stage/new/$f" ;; esac
      fi
      cp -p "$stage/new/$f" "$stage/baseline/$f"
    done < "$stage/files"
    [ "$bad" = 0 ] || { echo 'deploy.sh ops: host edits; no live files written (use OPS_ACCEPT=f1,f2)' >&2; exit 3; }
    if ! ops_probe_gpu "$stage"; then
      echo "deploy.sh ops: staged GPU behavior failed; no live files written" >&2; exit 4
    fi
    ops_mkdir() (
      [ -L "$1" ] && exit 1
      [ -d "$1" ] && exit 0
      ops_mkdir "$(dirname "$1")" && mkdir "$1" && printf '%s\n' "$1" >> "$stage/dirs"
    )
    switching=1
    while IFS= read -r f; do
      ops_mkdir "$(dirname "$BM/$f")"
      if [ -e "$BM/$f" ]; then mv -T "$BM/$f" "$archive/files/$f"; fi
      mv -T "$stage/new/$f" "$BM/$f"
    done < "$stage/files"
    while IFS= read -r f; do
      cmp -s "$stage/baseline/$f" "$BM/$f" || { echo "deploy.sh ops: MISMATCH $f" >&2; exit 5; }
    done < "$stage/files"
    baseline=1
    if [ -e "$K" ]; then mv -T "$K" "$archive/.ops_deployed"; fi
    mv -T "$stage/baseline" "$K"
    switching=0
    printf 'deploy.sh ops: %s files from trees/%s; previous versions: %s\n' "$(wc -l < "$stage/files")" "$2" "$archive"
    ;;
  tree)
    D=$BM/trees/$2
    [ -e $D ] && { echo "deploy.sh: $D exists" >&2; exit 2; }
    mkdir -p $D && tar -xf $3 -C $D && trash $3
    ls $D/bridge
    ;;
esac
