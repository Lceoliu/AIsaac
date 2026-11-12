#!/bin/sh
# bm1: a training run of K synchronised shards (BM1 C) under a guard that resumes it after a failure, unattended.
#   usage: [HOURS=<the run's game hours in all, resumes included (ISAAC_GAME_HOURS_TOTAL)>] [INIT=<checkpoint path as the container sees it>] [ENVS="-e K=V ..."]
#          [MAX_RETRY=3] [BACKOFF=120] [DISK_MIN_GB=50] [MEM_GB=128] train_guard.sh <name> <tree> <lane 0 file> <lane 1 file> [...]
# Run directory /root/AIsaac/train/<name> (the container's /bench/train/<name>): rank r writes <name>-s<r>/ (progress.csv,
# episodes.jsonl rotated past 256 MB, checkpoints/), guard.log (every start, end, cause, resume), foreign_gpu.log (other
# processes seen on our GPUs), STOP (touch it: the ranks end together at their next collect, the guard ends, no resume),
# guard.env (optional: sourced before every attempt, e.g. DISK_MIN_GB / DISK_WARN_GB / MAX_RETRY / BACKOFF /
# CKPT_ROTATE=trash|delete: the rotated checkpoints moved to /root/AIsaac/trash/<date>/ (default) or deleted -- only
# update-<6 digits>.pt of the run's checkpoints directories, each logged in checkpoint_rotation.log; the newest
# CKPT_KEEP (5) and every CKPT_MILESTONE-th (250) never).
# The container's memory limit MEM_GB (default 128; a limit, nothing reserved; not the lanes' sum): past 90 % of it a
# warning in guard.log (every 10 min while it lasts) and the ranks' progress (ctr_mem_gb / ctr_mem_limit_gb /
# ctr_mem_frac / ctr_oom_kills); <name>-s<r>.mem.attempt<n>.csv: each rank's memory every MEM_SAMPLE_S (30) s (m0_mem.py).
# Takes the host's exclusive lock for the whole run (purpose: training), released at its end.
# Each attempt: preflight (every GPU listed, no new Xid), the disk's free space, the newest readable checkpoint of rank 0
# (update-*.pt / last.pt: a zip that tests whole; else the one before), one container (shardk_in.sh, the K ranks).
# After an attempt ends: rc 0 without STOP -- the run's HOURS reached: the guard ends; else the cause from the ranks'
# stdout and the guard's own checks --
#   stop, no resume: the STOP file; a CheckpointError or a compile during a checkpoint's recomputation; the ranks'
#     hashes differing in consecutive checks; preflight failing (Xid / a GPU missing); free disk under DISK_MIN_GB;
#     the same update failing MAX_RETRY times; an out-of-memory a second time while another tenant's process was on
#     our GPUs (the first: resumed once); the kernel killing a rank at the container's memory limit (cgroup oom_kill,
#     shardk_in's <name>.oom_kills) a second time (the first: resumed once); the kernel's OOM kills in the container
#     (game / worker processes) growing in OOM_KILL_STREAK (3) updates in a row or reaching OOM_KILL_MAX (20) in one
#     attempt (the monitor: the ranks stopped at once, .oom_stop; a single growth: a warning); episodes breaking off
#     (progress broken_episodes: a game or worker failed) in BROKEN_STREAK (3) updates in a row or BROKEN_MAX (20) in
#     one attempt (likewise, .broken_stop; any: a warning); no progress (the monitor: a rank without a new progress row
#     for STALL_S, 600 s, or STARTUP_STALL_S, 1800 s, before its first row of the attempt: the trainers SIGINTed, the
#     container killed STALL_KILL_S, 180 s, later; cause stall, resumed) twice at the same update;
#   else: resumed after BACKOFF x attempt seconds, MAX_RETRY attempts per cause.
# Nothing system-wide: no GPU reset, no other process touched.
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
NAME=$1; TREE=$2; shift 2
K=$#
RD=/root/AIsaac/train/$NAME; RDC=/bench/train/$NAME
mkdir -p $RD
GL=$RD/guard.log
glog() { echo "$(date '+%F %T') $*" >> $GL; log "train $NAME: $*"; }
lane() { ( . "$1"; eval echo "\$$2" ); }
G=""; PAIRS=""; CPUS=""; LCPUS=""; DCPUS=""; MEMG=0; CS=""; i=0
for f in "$@"; do
  pair=""
  for g in $(echo $(lane $f LANE_GPUS) | tr ',' ' '); do G="${G:+$G,}$g"; pair="${pair:+$pair,}$i"; i=$((i+1)); done
  PAIRS="$PAIRS $pair"
  c=$(lane $f LANE_CPUSET); CPUS="${CPUS}${c}|"; [ -n "$c" ] && CS="${CS:+$CS,}$c"
  LCPUS="${LCPUS}$(lane $f LANE_LEARNER_CPUS)|"; DCPUS="${DCPUS}$(lane $f LANE_DRIVER_CPUS)|"
  MEMG=$((MEMG + $(lane $f LANE_MEM | tr -d 'gG')))
done
CPUS=${CPUS%|}; LCPUS=${LCPUS%|}; DCPUS=${DCPUS%|}
LANESUM=$MEMG; MEMG=${MEM_GB:-128}
NB=$(lane $1 LANE_NAME); PB=$(lane $1 LANE_PORT)
CTR="$BM1_NAME-train-$NAME"
take_excl "training $NAME ($TREE, K $K, GPU $G) 计时运行"
for m_ in .guard_end .stop_sent .oom_stop .broken_stop .stall .attempt_start; do   # (an earlier guard's markers: moved aside, nothing deleted)
  [ -f $RD/$m_ ] && { mkdir -p ${BM1_TRASH:-/root/AIsaac/trash}/$(date +%Y%m%d); mv $RD/$m_ ${BM1_TRASH:-/root/AIsaac/trash}/$(date +%Y%m%d)/$NAME$m_-$(date +%H%M%S); }
done
glog "guard start: tree $TREE K $K GPUs $G CPUs $CPUS learner $LCPUS driver $DCPUS memory limit ${MEMG}g (the lanes' sum ${LANESUM}g) hours ${HOURS:-125}"
# (other tenants' processes on our GPUs, every 30 s: our container's pids from docker top)
( while [ ! -f $RD/.guard_end ]; do
    ours=$(docker top $CTR -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')
    for g in $(echo $G | tr ',' ' '); do
      for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        case " $ours " in *" $p "*) ;; *) echo "$(date '+%F %T') GPU $g: pid $p ($(ps -o comm= -p $p 2>/dev/null)) not ours" >> $RD/foreign_gpu.log ;; esac
      done
    done
    # (disk: under DISK_WARN_GB a warning every hour -- the trash's size with it; it is on the same disk)
    fg_=$(df -BG --output=avail /root/AIsaac | tail -1 | tr -dc 0-9)
    if [ "$fg_" -lt "${DISK_WARN_GB:-100}" ] && [ $(( $(date +%s) - ${warned_:-0} )) -ge 3600 ]; then
      warned_=$(date +%s)
      glog "WARNING free disk ${fg_} GB < ${DISK_WARN_GB:-100} GB (trash $(du -sh ${BM1_TRASH:-/root/AIsaac/trash} 2>/dev/null | cut -f1); stop at ${DISK_MIN_GB:-50} GB)"
    fi
    # (the container's memory past 90 % of its limit: rank 0's newest progress row (the container's cgroup, every rank's
    #  the same), a warning every 10 min while it lasts)
    cm_=$(python3 -c "import csv, sys; r = list(csv.DictReader(open(sys.argv[1]))); x = r[-1] if r else {}; print(x.get('ctr_mem_frac', ''), x.get('ctr_mem_gb', ''), x.get('update', ''), x.get('ctr_oom_kills', ''))" $RD/$NAME-s0/progress.csv 2>/dev/null)
    set -- $cm_
    # (the kernel's OOM kills in the container (a game or worker process: the ranks may go on), per new progress row:
    #  a growth a warning; growing in OOM_KILL_STREAK (3) updates in a row, or OOM_KILL_MAX (20) in one attempt: the
    #  ranks' stop file and .oom_stop -- the guard stops, no resume)
    if [ $# -ge 4 ] && [ "$3" != "${oku_:-}" ] && [ ! -f $RD/.oom_stop ]; then
      oku_=$3
      if [ "$4" -lt "${okl_:-0}" ] 2>/dev/null; then okl_=0; oks_=0; fi   # (a new attempt: its new container's count)
      if [ "$4" -gt "${okl_:-0}" ] 2>/dev/null; then
        oks_=$(( ${oks_:-0} + 1 ))
        glog "WARNING the kernel's OOM kills in the container at its memory limit ${MEMG}g: $4 in this attempt (update $3, growing $oks_ update(s) in a row)"
        if [ $oks_ -ge ${OOM_KILL_STREAK:-3} ] || [ "$4" -ge ${OOM_KILL_MAX:-20} ]; then
          echo "the container's memory at its limit ${MEMG}g, the kernel killing game or worker processes ($4 kills in this attempt, growing $oks_ update(s) in a row)" > $RD/.oom_stop
          docker exec $CTR touch /tmp/shards-$NAME/stop 2>/dev/null
          glog "STOP (memory): $(cat $RD/.oom_stop); the ranks' stop file written"
        fi
      else oks_=0; fi
      okl_=$4
    fi
    # (episodes that broke off -- a game or worker failed -- every rank's newest progress row: any a warning; some in
    #  BROKEN_STREAK (3) updates in a row, or BROKEN_MAX (20) in one attempt (the ranks' broken_total): the ranks'
    #  stop file and .broken_stop -- the guard stops, no resume)
    bk_=$(python3 -c "
import csv, sys
u, e, t = '', 0, 0
for p in sys.argv[1:]:
    try:
        r = list(csv.DictReader(open(p)))
    except OSError:
        continue
    if r and r[-1].get('broken_episodes', '') != '':
        u = u or r[-1]['update']; e += int(float(r[-1]['broken_episodes'])); t += int(float(r[-1]['broken_total']))
print(u, e, t)" $(for r in $(seq 0 $((K-1))); do echo $RD/$NAME-s$r/progress.csv; done) 2>/dev/null)
    set -- $bk_
    if [ $# -ge 3 ] && [ "$1" != "${bku_:-}" ] && [ ! -f $RD/.broken_stop ]; then
      bku_=$1
      if [ "$2" -gt 0 ]; then
        bks_=$(( ${bks_:-0} + 1 ))
        glog "WARNING episodes broke off (a game or worker failed): $2 in update $1, $3 in this attempt ($bks_ update(s) in a row)"
        if [ $bks_ -ge ${BROKEN_STREAK:-3} ] || [ "$3" -ge ${BROKEN_MAX:-20} ]; then
          echo "episodes breaking off: games or workers failing ($3 in this attempt, $bks_ update(s) in a row)" > $RD/.broken_stop
          docker exec $CTR touch /tmp/shards-$NAME/stop 2>/dev/null
          glog "STOP (broken episodes): $(cat $RD/.broken_stop); the ranks' stop file written"
        fi
      else bks_=0; fi
    fi
    set -- $cm_   # (the memory row again: the block above set its own)
    if [ $# -ge 3 ] && awk "BEGIN {exit !($1 > 0.9)}" && [ $(( $(date +%s) - ${mwarned_:-0} )) -ge 600 ]; then
      mwarned_=$(date +%s)
      glog "WARNING container memory $(printf %.1f $2) GB = $(awk "BEGIN {printf \"%.0f\", 100 * $1}") % of its limit ${MEMG}g (update $3)"
    fi
    # (worker processes alive, every rank's newest progress row (workers_live): a drop a warning, nothing more -- the
    #  run goes on with the others, slower; none left ends the rank's run with an error)
    wl_=$(for r in $(seq 0 $((K-1))); do python3 -c "import csv, sys; r = list(csv.DictReader(open(sys.argv[1]))); print((r[-1].get('workers_live') or '-') if r else '-', r[-1]['update'] if r else '-')" $RD/$NAME-s$r/progress.csv 2>/dev/null || echo "- -"; done | tr '\n' ' ')
    set -- $wl_
    r_=0; while [ $# -ge 2 ]; do
      eval "pw_=\${wlp_$r_:-}"
      if [ "$1" != "-" ] && [ -n "$pw_" ] && [ "$1" -lt "$pw_" ] 2>/dev/null; then
        glog "WARNING rank $r_: worker processes alive $1 (were $pw_) at update $2 -- the run goes on with the others, slower"
      fi
      [ "$1" != "-" ] && eval "wlp_$r_=$1"
      shift 2; r_=$((r_+1))
    done
    # (no progress: while the attempt's container runs, a rank without a new progress row for STALL_S (600 s, about 14
    #  updates' time) -- STARTUP_STALL_S (1800 s) before its first row of the attempt: .stall, the trainers SIGINTed (they
    #  save last.pt), the container killed STALL_KILL_S (180 s) later if still there)
    if [ -f $RD/.attempt_start ] && [ ! -f $RD/.stall ] && [ "$(docker inspect -f '{{.State.Running}}' $CTR 2>/dev/null)" = true ]; then
      a_=$(cat $RD/.attempt_start); now_=$(date +%s)
      for r in $(seq 0 $((K-1))); do
        m_=$(stat -c %Y $RD/$NAME-s$r/progress.csv 2>/dev/null || echo 0)
        if [ "$m_" -lt "$a_" ]; then lim_=${STARTUP_STALL_S:-1800}; since_=$a_; what_="no progress row of the attempt yet"
        else lim_=${STALL_S:-600}; since_=$m_; what_="no new progress row"; fi
        if [ $(( now_ - since_ )) -ge $lim_ ]; then
          echo "rank $r: $what_ for $(( now_ - since_ )) s (limit $lim_ s)" > $RD/.stall
          glog "STALL: $(cat $RD/.stall): the trainers SIGINTed, the container killed in ${STALL_KILL_S:-180} s if still there"
          docker exec $CTR pkill -INT -f "train_tok.py .*--out $RDC/$NAME-s" 2>/dev/null
          ( sleep ${STALL_KILL_S:-180}; [ "$(docker inspect -f '{{.State.Running}}' $CTR 2>/dev/null)" = true ] && {
              docker kill $CTR > /dev/null 2>&1; glog "STALL: the container $CTR killed (still there ${STALL_KILL_S:-180} s after the SIGINT)"; } ) &
          break
        fi
      done
    fi
    sleep 30
  done ) &
MON=$!
# (STOP while the ranks run: their stop file inside the container, so they end together at the next collect)
( while [ ! -f $RD/.guard_end ]; do
    if [ -f $RD/STOP ] && [ ! -f $RD/.stop_sent ]; then
      docker exec $CTR touch /tmp/shards-$NAME/stop 2>/dev/null && { : > $RD/.stop_sent; glog "STOP: the ranks' stop file written"; }
    fi
    sleep 5
  done ) &
STW=$!
newest_ckpt() {   # rank 0's newest checkpoint whose zip tests whole (as the container sees it), else empty
  for c in $(ls -t $RD/$NAME-s0/checkpoints/update-*.pt $RD/$NAME-s0/checkpoints/last.pt 2>/dev/null); do
    if python3 -c "import sys, zipfile; sys.exit(0 if zipfile.ZipFile(sys.argv[1]).testzip() is None else 1)" $c 2>/dev/null; then
      echo $RDC/${c#$RD/}; return
    fi
    glog "checkpoint $c unreadable: skipped"
  done
}
attempt=0; last_cause=""; same=0; oom_foreign=0; last_upd=""; upd_fail=0
while :; do
  [ -f $RD/guard.env ] && . $RD/guard.env   # (overrides read before every attempt: DISK_MIN_GB, MAX_RETRY, BACKOFF, ...)
  [ -f $RD/STOP ] && { glog "STOP before an attempt: the guard ends"; break; }
  ( preflight "train $NAME" ) || { glog "preflight failed (Xid or a GPU missing: $(cat $BM/STOP_QUEUE 2>/dev/null)): stopped, no resume"; break; }
  free_gb=$(df -BG --output=avail /root/AIsaac | tail -1 | tr -dc 0-9)
  [ "$free_gb" -lt "${DISK_MIN_GB:-50}" ] && { glog "free disk ${free_gb} GB < ${DISK_MIN_GB:-50}: stopped, no resume"; break; }
  ck=$(newest_ckpt)
  if [ -n "$ck" ]; then EX3="-e EXTRA3=--resume=$ck"; from="resume from $ck"
  elif [ -n "${INIT:-}" ]; then EX3="-e EXTRA3=--init=$INIT"; from="start from INIT $INIT"
  else EX3=""; from="fresh start"; fi
  attempt=$((attempt+1))
  for r in $(seq 0 $((K-1))); do   # (each attempt's ranks' stdout / stderr kept: xnew2.sh writes them afresh)
    for x in stdout stderr; do [ -f $RD/$NAME-s$r.$x ] && mv $RD/$NAME-s$r.$x $RD/$NAME-s$r.$x.attempt$((attempt-1)); done
  done
  glog "attempt $attempt: $from"
  date +%s > $RD/.attempt_start   # (the monitor's no-progress clock)
  MP=""
  for r in $(seq 0 $((K-1))); do   # (each rank's memory by process kind and the container's cgroup, every MEM_SAMPLE_S s)
    python3 $BM/m0_mem.py $RD/$NAME-s$r.mem.attempt$attempt.csv ${MEM_SAMPLE_S:-30} " --out $RDC/$NAME-s$r " > $RD/$NAME-s$r.mem.attempt$attempt.log 2>&1 &
    MP="$MP $!"
  done
  DE="-e CODE=$TREE -e OUTDIR=$RDC -e ISAAC_BLOCK_TIMING=1 -e EXTRA=--learner-device=cuda:1 -e EXTRA2=--checkpoint-every=${CKPT_EVERY:-25} $EX3"
  DE="$DE -e NAME_BASE=$NB -e PORT_BASE=$PB -e SHARD_PORT=${SHARD_PORT:-29400} -e ISAAC_SHARD_TIMEOUT=${SHARD_TIMEOUT:-1800}"
  DE="$DE -e ISAAC_GAME_HOURS_TOTAL=${HOURS:-125} -e ISAAC_LEARN_CKPT_DET=1 -e ISAAC_CKPT_KEEP=${CKPT_KEEP:-5} -e ISAAC_CKPT_MILESTONE=${CKPT_MILESTONE:-250}"
  DE="$DE -e ISAAC_CKPT_TRASH=/bench/trash -e ISAAC_CKPT_ROTATE=${CKPT_ROTATE:-trash} -e ISAAC_EPISODES_ROTATE_MB=${EPISODES_MB:-256} ${ENVS:-}"
  CPUSET="${CS:-${BM1_TIMED_CPUSET:-$BM1_CPUSET}}" MEMS="" MEM="${MEMG}g" PIDS=$((${BM1_PIDS:-4096} * K)) GPUS_OK="$G" \
    DOCK_ENV="$DE" $BM/run.sh train-$NAME $G sh $BM1_IN/shardk_in.sh $NAME 315360000 ${HOURS:-125} "$(echo $PAIRS)" "$CPUS" "$LCPUS" "$DCPUS" >> $RD/launcher.log 2>&1
  rc=$?
  kill $MP 2>/dev/null
  stall_=""; [ -f $RD/.stall ] && { stall_=$(cat $RD/.stall); mv $RD/.stall $RD/.stall.attempt$attempt; }
  cgk=$(cat $RD/$NAME.oom_kills 2>/dev/null); [ -f $RD/$NAME.oom_kills ] && mv $RD/$NAME.oom_kills $RD/$NAME.oom_kills.attempt$attempt
  outs=$(for r in $(seq 0 $((K-1))); do tail -c 200000 $RD/$NAME-s$r.stdout $RD/$NAME-s$r.stderr 2>/dev/null; done)
  upd=$(python3 -c "import csv, sys; r = list(csv.DictReader(open(sys.argv[1]))); print(r[-1]['update'] if r else '')" $RD/$NAME-s0/progress.csv 2>/dev/null)
  if [ -f $RD/.broken_stop ]; then glog "attempt $attempt ended rc=$rc at update ${upd:-?}: $(cat $RD/.broken_stop): stopped, no resume (a person looks)"; break; fi
  if [ -f $RD/.oom_stop ]; then glog "attempt $attempt ended rc=$rc at update ${upd:-?}: $(cat $RD/.oom_stop): stopped, no resume (a person looks)"; break; fi
  if [ -f $RD/STOP ]; then glog "attempt $attempt ended rc=$rc at update ${upd:-?} after STOP: the guard ends"; break; fi
  if [ $rc = 0 ]; then
    case "$cgk" in [1-9]*) glog "WARNING the kernel killed $cgk process(es) at the container's memory limit ${MEMG}g in attempt $attempt (the ranks went on)" ;; esac
    glog "attempt $attempt ended rc=0 at update ${upd:-?}: the run's ${HOURS:-125} game hours reached, the guard ends"; break
  fi
  cause=other
  [ -f $RD/.attempt_start ] && mv $RD/.attempt_start $RD/.attempt_start.prev
  echo "$outs" | grep -q "CheckpointError\|compiled during the backward recomputation" && cause=checkpoint
  echo "$outs" | grep -q "state hashes of the ranks differed" && cause=hash
  echo "$outs" | grep -q "out of memory" && cause=oom
  case "$cgk" in [1-9]*) cause=cgroup_oom ;; esac   # (the kernel's OOM kill at the container's limit)
  [ $rc = 4 ] && cause=gpu_busy   # (run.sh: a GPU had another process at the start)
  [ -n "$stall_" ] && cause=stall   # (the monitor's no-progress watchdog ended it)
  glog "attempt $attempt ended rc=$rc at update ${upd:-?}: cause $cause"
  case $cause in
    checkpoint|hash) glog "cause $cause: stopped, no resume (a person looks)"; break ;;
    oom) if [ -s $RD/foreign_gpu.log ] && [ $oom_foreign -ge 1 ]; then glog "out of memory again with other tenants on our GPUs: stopped"; break; fi
         [ -s $RD/foreign_gpu.log ] && oom_foreign=$((oom_foreign+1)) ;;
    stall) glog "no progress: $stall_"
         if [ -n "$upd" ] && [ "$upd" = "${stall_upd:-}" ]; then glog "stalled again at update $upd: stopped, no resume (a person looks)"; break; fi
         stall_upd=$upd ;;
    cgroup_oom) cg_oom=$(( ${cg_oom:-0} + 1 ))
         glog "a rank killed by the kernel at the container's memory limit ${MEMG}g (cgroup oom_kill $cgk; the memory limit, not another tenant's process) -- the $cg_oom. time"
         [ $cg_oom -ge 2 ] && { glog "the container's memory limit a second time: stopped, no resume (a person looks)"; break; } ;;
  esac
  if [ -n "$upd" ] && [ "$upd" = "$last_upd" ]; then upd_fail=$((upd_fail+1)); else upd_fail=1; last_upd=$upd; fi
  [ $upd_fail -ge ${MAX_RETRY:-3} ] && { glog "update $upd failed $upd_fail times: stopped, no resume"; break; }
  if [ "$cause" = "$last_cause" ]; then same=$((same+1)); else same=1; last_cause=$cause; fi
  [ $same -gt ${MAX_RETRY:-3} ] && { glog "cause $cause $same times in a row: stopped, no resume"; break; }
  w=$(( ${BACKOFF:-120} * same ))
  glog "resuming in $w s (cause $cause, $same in a row)"
  t0=$(date +%s); while [ $(( $(date +%s) - t0 )) -lt $w ] && [ ! -f $RD/STOP ]; do sleep 5; done
done
touch $RD/.guard_end
kill $MON $STW 2>/dev/null
give
glog "guard end"
