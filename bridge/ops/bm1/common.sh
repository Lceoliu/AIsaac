# bm1 common (sourced by every host-side script). Host-independent: every path, the lane and the run mode come from
# the environment file $BM/bm1.env (a copy of one of env/*.env, made at deployment), never from this file.
#   BM            the bm1 directory on the host (default: the directory of the calling script)
# Variables of bm1.env (env/dgene-8x3090.env documents each):
#   BM1_MODE docker | direct, BM1_IN, BM1_PY, BM1_GAME_RUNTIME, BM1_CPUSET, BM1_MEMS, BM1_GPUS_OK, BM1_MEM, BM1_PIDS,
#   BM1_LOCK, BM1_SESSION, BM1_GPU_EXPECT, BM1_IMAGE, BM1_DOCKER_BASE
# Functions: take / give (the lane lock), capmode, preflight, sigint_trainer, trash, log. (Callers may run with set -e:
# every grep -c here ends with || true, a zero count being no error.)
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
if [ -f "$BM/bm1.env" ]; then . "$BM/bm1.env"; else echo "bm1: $BM/bm1.env missing (copy one of $BM/env/*.env)" >&2; exit 2; fi
# (the older per-script variables still override: CPUSET, MEMS, GPUS_OK, MEM, PIDS, LOCK, SESSION)
BM1_CPUSET=${CPUSET-${BM1_CPUSET:-}}; BM1_MEMS=${MEMS-${BM1_MEMS:-}}; BM1_GPUS_OK=${GPUS_OK:-${BM1_GPUS_OK:-}}
BM1_MEM=${MEM:-${BM1_MEM:-}}; BM1_PIDS=${PIDS:-${BM1_PIDS:-4096}}
LOCK=${LOCK:-${BM1_LOCK:-$BM/LOCK_node0}}; SESSION=${SESSION:-${BM1_SESSION:-bm1-a1}}
R=$BM/reports; LOG=$R/bm1.log
mkdir -p "$R"

log() { echo "$(date '+%F %T') $*" >> $LOG; }

# The lane lock. Two kinds:
#   * BM1_LOCKDIR unset: the old single-file lane lock $LOCK (noclobber), take = take_shared = exclusive.
#   * BM1_LOCKDIR set (gpu57: one lock for the whole host, shared by every session): a reader-writer lock.
#     take_excl "<purpose>" (timed runs): writes $BM1_LOCKDIR/X at once -- no new shared holder starts from then on
#     (writer first) -- then waits until no live shared holder is left. take_shared "<purpose>" (gates, builds, smokes,
#     probes): waits while X exists, then writes $BM1_LOCKDIR/S.<session>.<pid>; shared holders run together.
#     Each file's first line is "<session> <time> <purpose>" (the old format), its second "pid <host pid of the holding
#     script>": a file whose pid is gone is stale and removed by the next taker (logged). Checks and writes happen under
#     a short mutex ($BM1_LOCKDIR/.mutex, mkdir: atomic; a mutex older than 60 s is stale). Waiting is a sleep loop in
#     the job script; the wait in seconds goes into bm1.log.
#   take "<purpose>" is take_excl when the purpose says 计时运行 (a timed run), else take_shared. give releases what
#   this script holds.
_mutex() {
  while ! mkdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; do
    a=$(( $(date +%s) - $(stat -c %Y "$BM1_LOCKDIR/.mutex" 2>/dev/null || date +%s) ))
    [ "$a" -gt 60 ] && { rmdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; log "lock: stale mutex ($a s) removed"; }
    sleep 1
  done
}
_unmutex() { rmdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; }
_stale_sweep() {   # (under the mutex) lock files whose holder pid is gone
  for f in "$BM1_LOCKDIR"/X "$BM1_LOCKDIR"/S.*; do
    [ -f "$f" ] || continue
    p=$(sed -n 's/^pid \([0-9]*\).*/\1/p' "$f")
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      log "lock: stale $(basename "$f") removed (holder pid ${p:-?} gone): $(head -1 "$f")"
      rm -f "$f"
    fi
  done
}
_live_shared() { ls "$BM1_LOCKDIR"/S.* 2>/dev/null | wc -l; }
take_excl() {
  if [ -z "${BM1_LOCKDIR:-}" ]; then _take_file "$1"; return; fi
  mkdir -p "$BM1_LOCKDIR"; t0=$(date +%s)
  while :; do   # 1: the writer's claim (no other writer)
    _mutex; _stale_sweep
    if [ ! -f "$BM1_LOCKDIR/X" ]; then
      printf '%s\npid %s waiting\n' "$SESSION $(date '+%F %T') $1" "$$" > "$BM1_LOCKDIR/X"; _unmutex; break
    fi
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  while :; do   # 2: the shared holders under way end (none starts meanwhile)
    _mutex; _stale_sweep; n=$(_live_shared)
    if [ "$n" = 0 ]; then printf '%s\npid %s\n' "$SESSION $(date '+%F %T') $1" "$$" > "$BM1_LOCKDIR/X"; _unmutex; break; fi
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  LOCKFILE=$BM1_LOCKDIR/X; LOCKLINE=$(head -1 "$LOCKFILE")
  log "lock: exclusive taken after $(( $(date +%s) - t0 )) s: $LOCKLINE"
}
take_shared() {
  if [ -z "${BM1_LOCKDIR:-}" ]; then _take_file "$1"; return; fi
  mkdir -p "$BM1_LOCKDIR"; t0=$(date +%s)
  f=$BM1_LOCKDIR/S.$SESSION.$$
  while :; do
    _mutex; _stale_sweep
    if [ ! -f "$BM1_LOCKDIR/X" ]; then printf '%s\npid %s\n' "$SESSION $(date '+%F %T') $1" "$$" > "$f"; _unmutex; break; fi
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  LOCKFILE=$f; LOCKLINE=$(head -1 "$f")
  log "lock: shared taken after $(( $(date +%s) - t0 )) s: $LOCKLINE"
}
_take_file() {
  while ! ( set -C; echo "$SESSION $(date '+%F %T') $1" > "$LOCK" ) 2>/dev/null; do sleep 30; done
  LOCKFILE=$LOCK; LOCKLINE=$(cat "$LOCK")
}
take() { case "$1" in *计时运行*) take_excl "$1" ;; *) take_shared "$1" ;; esac; }
give() {
  [ -n "${LOCKFILE:-}" ] && [ "$(head -1 "$LOCKFILE" 2>/dev/null)" = "${LOCKLINE:-}" ] && rm -f "$LOCKFILE"
  LOCKLINE=; LOCKFILE=
}

# capmode <tree>: the graph capture mode a run of <tree> uses (ISAAC_CAPTURE_MODE from ENVS, else the tree's default)
capmode() {
  m=$(echo "${ENVS:-}" | grep -o 'ISAAC_CAPTURE_MODE=[a-z_]*' | cut -d= -f2)
  [ -n "$m" ] || m=$(grep -o "ISAAC_CAPTURE_MODE', '[a-z_]*'" $BM/trees/$1/bridge/python/isaac_bridge/block_training.py 2>/dev/null | cut -d"'" -f3)
  echo "${m:-global (no switch in this tree)}"
}

# preflight: before a GPU job. Every GPU nvidia-smi -L lists without an error line, as many as BM1_GPU_EXPECT when set,
# and no NVRM Xid in the kernel log beyond the count recorded by `preflight_baseline`. A failure is logged, the queue
# stop file $BM/STOP_QUEUE written (every later script refuses until it is removed by hand), and the caller exits 3.
xid_count() { ( journalctl -k -b --no-pager 2>/dev/null || dmesg 2>/dev/null ) | grep -c 'NVRM: Xid' || true; }   # (this boot's kernel log; a non-root user in adm reads it too)
preflight_baseline() { xid_count > $BM/.xid_baseline; log "preflight baseline: $(cat $BM/.xid_baseline) Xid lines in the kernel log"; }
preflight() {
  if [ -f $BM/STOP_QUEUE ]; then echo "bm1: $BM/STOP_QUEUE exists ($(cat $BM/STOP_QUEUE)); nothing starts" >&2; exit 3; fi
  L_=$(nvidia-smi -L 2>&1); n_=$(echo "$L_" | grep -c '^GPU ' || true); e_=$(echo "$L_" | grep -ciE 'error|unable|unknown' || true)
  x_=$(xid_count); b_=$(cat $BM/.xid_baseline 2>/dev/null || echo "$x_")
  why=
  [ "$e_" -gt 0 ] && why="nvidia-smi -L reports an error"
  [ -n "${BM1_GPU_EXPECT:-}" ] && [ "$n_" -lt "$BM1_GPU_EXPECT" ] && why="${why:+$why; }$n_ GPUs listed, $BM1_GPU_EXPECT expected"
  [ "$x_" -gt "$b_" ] && why="${why:+$why; }$((x_-b_)) new Xid line(s) in the kernel log"
  if [ -n "$why" ]; then
    echo "$(date '+%F %T') $why" > $BM/STOP_QUEUE
    log "PREFLIGHT FAIL ($1): $why -- queue stopped ($BM/STOP_QUEUE)"
    exit 3
  fi
}

# sigint_trainer <runner name> <tag>: SIGINT to that run's train_tok main process (matched by its --out)
sigint_trainer() {
  pat="^$BM1_PY -u train_tok.py .*--out $BM1_IN/reports/$2( |\$)"
  if [ "$BM1_MODE" = docker ]; then docker exec "aisaac-bench-bm1-a1-$1" pkill -INT -f "$pat"; else pkill -INT -f "$pat"; fi
}

# trash <path>: gio trash where it exists, else a move into ${BM1_TRASH:-$BM/trash}/<time>/ (nothing is deleted)
trash() {
  [ -e "$1" ] || return 0
  if command -v gio > /dev/null 2>&1 && gio trash "$1" 2>/dev/null; then return 0; fi
  d=${BM1_TRASH:-$BM/trash}/$(date +%m%d-%H%M%S); mkdir -p "$d" && mv "$1" "$d/"
}

# gpus_free <g[,g...]>: every listed GPU without another process (memory used under BM1_GPU_FREE_MB, default 500 MiB,
# and no compute process); a busy GPU is logged and the caller exits 4 (shared hosts: never start on a used card)
gpus_free() {
  for g in $(echo "$1" | tr ',' ' '); do
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    p=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ -z "$m" ] || [ "$m" -ge "${BM1_GPU_FREE_MB:-500}" ] || [ "$p" -gt 0 ]; then
      log "GPU $g BUSY (memory used ${m:-?} MiB, ${p} compute process(es)): not started"
      exit 4
    fi
  done
}

# host_snapshot <label>: every GPU's memory / utilisation, the load average and the five busiest processes, one bm1.log
# line (a timed run's start and end: a run disturbed by other tenants can be told)
host_snapshot() {
  g=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%s:%sMiB/%s%% ", $1, $2, $3}')
  t=$(ps -eo pcpu=,comm= --sort=-pcpu 2>/dev/null | head -5 | awk '{printf "%s %s%%; ", $2, $1}')
  log "$1 host: gpus [$g] load [$(cut -d' ' -f1-3 /proc/loadavg)] top cpu [$t]"
}
