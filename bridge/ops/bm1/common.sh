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
# (two lanes of one host = two bm1 directories, each with its own bm1.env: the container name prefix, the runner's
# PYTHONPATH and, in docker mode, the runtime link of deploy.sh abphome come from it too; the defaults are lane a1's)
BM1_NAME=${BM1_NAME:-aisaac-bench-bm1-a1}; BM1_PYTHONPATH=${BM1_PYTHONPATH:-$BM1_IN/pydeps}
BM1_RUNTIME_LINK=${BM1_RUNTIME_LINK:-../../../home/isaac-abplus/runtime}
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
#     Bounded wait (2026-11-11): a shared taker that has to wait leaves $BM1_LOCKDIR/W.<session>.<pid> ("pid", "since
#     <epoch s>"); a writer about to claim X first lets every shared taker that was already waiting before the writer
#     itself arrived take its shared lock (then claims X and waits for them to end, as always). So back-to-back timed
#     runs let the non-timed jobs waiting between them in at least once: X1, the waiting shared jobs, X2. A timed run
#     still never overlaps a shared job.
#     Containers (2026-11-11): run.sh adds "container <name>" to the lock file it runs under ($LOCKFILE, exported by
#     take_*) before it starts the container. A file is stale only when its holder pid is gone AND none of its
#     containers is still running: a job script that dies while its container goes on (shardk.sh under set -u, c2flipc)
#     keeps its lock until the container ends. (BM1_CONTAINER_ALIVE: a shell snippet testing "$1" instead of docker ps;
#     the regression test's stand-in.)
# (every variable the lock functions set is named _lk_*: sh has no locals, and a caller's f, n, p, t0 ... must survive.
#  2026-11-11: _stale_sweep's loop variable f overwrote take_shared's own file name, so shared holders wrote one common
#  literal "S.*" file, and the first holder to finish released the others' claim)
_mutex() {
  while ! mkdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; do
    _lk_a=$(( $(date +%s) - $(stat -c %Y "$BM1_LOCKDIR/.mutex" 2>/dev/null || date +%s) ))
    [ "$_lk_a" -gt 60 ] && { rmdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; log "lock: stale mutex ($_lk_a s) removed"; }
    sleep 1
  done
}
_unmutex() { rmdir "$BM1_LOCKDIR/.mutex" 2>/dev/null; }
_container_alive() {   # <name>: the container is running
  if [ -n "${BM1_CONTAINER_ALIVE:-}" ]; then ( set -- "$1"; eval "$BM1_CONTAINER_ALIVE" ); return; fi
  docker ps -q --filter "name=^/$1\$" 2>/dev/null | grep -q .
}
_stale_sweep() {   # (under the mutex) lock files whose holder pid is gone and whose containers have all ended
  for _lk_f in "$BM1_LOCKDIR"/X "$BM1_LOCKDIR"/S.* "$BM1_LOCKDIR"/W.*; do
    [ -f "$_lk_f" ] || continue
    _lk_p=$(sed -n 's/^pid \([0-9]*\).*/\1/p' "$_lk_f")
    if [ -z "$_lk_p" ] || ! kill -0 "$_lk_p" 2>/dev/null; then
      _lk_c=""
      for _lk_n in $(sed -n 's/^container \(.*\)/\1/p' "$_lk_f"); do
        _container_alive "$_lk_n" && { _lk_c=$_lk_n; break; }
      done
      if [ -n "$_lk_c" ]; then
        _lk_o="$BM1_LOCKDIR/.orphan.$(basename "$_lk_f")"   # (logged once; not a lock file name: X, S.*, W.*)
        [ -f "$_lk_o" ] || { log "lock: $(basename "$_lk_f")'s holder pid ${_lk_p:-?} gone but its container $_lk_c runs: kept: $(head -1 "$_lk_f")"; : > "$_lk_o"; }
        continue
      fi
      log "lock: stale $(basename "$_lk_f") removed (holder pid ${_lk_p:-?} gone, no container of it running): $(head -1 "$_lk_f")"
      rm -f "$_lk_f" "$BM1_LOCKDIR/.orphan.$(basename "$_lk_f")"
    fi
  done
}
_live_shared() { ls "$BM1_LOCKDIR"/S.* 2>/dev/null | wc -l; }
_older_waiters() {   # (under the mutex) shared takers waiting since before epoch $1
  _lk_c=0
  for _lk_w in "$BM1_LOCKDIR"/W.*; do
    [ -f "$_lk_w" ] || continue
    _lk_s=$(sed -n 's/^since \([0-9]*\).*/\1/p' "$_lk_w")
    [ -n "$_lk_s" ] && [ "$_lk_s" -lt "$1" ] && _lk_c=$((_lk_c+1))
  done
  echo $_lk_c
}
take_excl() {
  if [ -z "${BM1_LOCKDIR:-}" ]; then _take_file "$1"; return; fi
  mkdir -p "$BM1_LOCKDIR"; _lk_t0=$(date +%s); _lk_y=0
  while :; do   # 1: the writer's claim (no other writer; shared takers waiting since before this writer go first)
    _mutex; _stale_sweep
    if [ ! -f "$BM1_LOCKDIR/X" ]; then
      _lk_o=$(_older_waiters "$_lk_t0")
      if [ "$_lk_o" = 0 ]; then
        printf '%s\npid %s waiting\n' "$SESSION $(date '+%F %T') $1" "$$" > "$BM1_LOCKDIR/X"; _unmutex; break
      fi
      [ "$_lk_y" = 0 ] && log "lock: exclusive yields to $_lk_o shared taker(s) waiting since before it: $1"; _lk_y=1
    fi
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  while :; do   # 2: the shared holders under way end (none starts meanwhile)
    _mutex; _stale_sweep; _lk_n=$(_live_shared)
    if [ "$_lk_n" = 0 ]; then printf '%s\npid %s\n' "$SESSION $(date '+%F %T') $1" "$$" > "$BM1_LOCKDIR/X"; _unmutex; break; fi
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  LOCKFILE=$BM1_LOCKDIR/X; LOCKLINE=$(head -1 "$LOCKFILE"); export LOCKFILE
  log "lock: exclusive taken after $(( $(date +%s) - _lk_t0 )) s: $LOCKLINE"
}
take_shared() {
  if [ -z "${BM1_LOCKDIR:-}" ]; then _take_file "$1"; return; fi
  mkdir -p "$BM1_LOCKDIR"; _lk_t0=$(date +%s)
  _lk_sf=$BM1_LOCKDIR/S.$SESSION.$$; _lk_wf=$BM1_LOCKDIR/W.$SESSION.$$
  while :; do
    _mutex; _stale_sweep
    if [ ! -f "$BM1_LOCKDIR/X" ]; then
      printf '%s\npid %s\n' "$SESSION $(date '+%F %T') $1" "$$" > "$_lk_sf"; rm -f "$_lk_wf"; _unmutex; break
    fi
    # (waiting: the marker a later writer yields to)
    [ -f "$_lk_wf" ] || printf 'pid %s\nsince %s\n%s\n' "$$" "$(date +%s)" "$SESSION $(date '+%F %T') $1" > "$_lk_wf"
    _unmutex; sleep ${BM1_LOCK_POLL:-30}
  done
  LOCKFILE=$_lk_sf; LOCKLINE=$(head -1 "$_lk_sf"); export LOCKFILE
  log "lock: shared taken after $(( $(date +%s) - _lk_t0 )) s: $LOCKLINE"
}
_take_file() {
  while ! ( set -C; echo "$SESSION $(date '+%F %T') $1" > "$LOCK" ) 2>/dev/null; do sleep 30; done
  LOCKFILE=$LOCK; LOCKLINE=$(cat "$LOCK")
}
take() { case "$1" in *计时运行*) take_excl "$1" ;; *) take_shared "$1" ;; esac; }
give() {
  [ -n "${LOCKFILE:-}" ] && [ "$(head -1 "$LOCKFILE" 2>/dev/null)" = "${LOCKLINE:-}" ] && rm -f "$LOCKFILE" "$(dirname "$LOCKFILE")/.orphan.$(basename "$LOCKFILE")"
  LOCKLINE=; LOCKFILE=
}
# lock_container <name>: the container a job is about to start, registered in the lock file it runs under (run.sh)
lock_container() {
  [ -n "${LOCKFILE:-}" ] && [ -f "$LOCKFILE" ] && [ -n "${BM1_LOCKDIR:-}" ] || return 0
  _mutex; [ -f "$LOCKFILE" ] && echo "container $1" >> "$LOCKFILE"; _unmutex
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
  if [ "$BM1_MODE" = docker ]; then docker exec "$BM1_NAME-$1" pkill -INT -f "$pat"; else pkill -INT -f "$pat"; fi
}

# trash <path>: nothing is deleted. BM1_TRASH_GIO=1 (default): gio trash where it exists (the account's own trash),
# else a move into ${BM1_TRASH:-$BM/trash}/<date>/<time>/. BM1_TRASH_GIO=0 (gpu57: our files only under ~/AIsaac; gio
# put them in the host root's ~/.local/share/Trash): always the move.
trash() {
  [ -e "$1" ] || return 0
  if [ "${BM1_TRASH_GIO:-1}" != 0 ] && command -v gio > /dev/null 2>&1 && gio trash "$1" 2>/dev/null; then return 0; fi
  d=${BM1_TRASH:-$BM/trash}/$(date +%Y%m%d)/$(date +%H%M%S); mkdir -p "$d" && mv "$1" "$d/"
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
