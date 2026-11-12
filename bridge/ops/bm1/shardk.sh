#!/bin/sh
# bm1 BM1 C: K shards as one training run (train_tok + isaac_bridge.shard_sync), timed like prof.sh. One container
# (the K trainers share its loopback and /dev/shm: NCCL between them without any network; the container's security
# settings as every bm1 run's), its GPUs every lane's, its memory limit the lanes' sum, its CPUs the lanes' union.
# Inside, shardk_in.sh starts rank r with lane r's two GPUs (actor, learner) and CPUs (taskset).
#   usage: [UPDATES=24] [HOURS=30.0] [LIMIT=4500] [ENVS="-e K=V ..."] [SHARD_PORT=29400] [NOLOCK=1]
#          shardk.sh <tag> <tree> <lane 0 file> <lane 1 file> [...]
#   lane files as shard2.sh's: LANE_GPUS (two: actor, learner), LANE_CPUSET (empty: no pinning), LANE_MEM, LANE_NAME,
#   LANE_PORT (each rank's instances' name / port base); optional LANE_LEARNER_CPUS / LANE_DRIVER_CPUS (that rank's
#   dedicated learner / driver CPUs, inside its LANE_CPUSET)
# Per rank r (tag <tag>-s<r>): reports/<tag>-s<r>/ (progress.csv ...), .stdout / .stderr, .mem.csv / .threads.csv (M0,
# this trainer's processes only), .gpu (its two GPUs); shared: <tag>.mpstat / .allgpu / .topcpu / .cpufreq (linked
# under each rank's name for summ.py); then summ.py per rank, and in bm1.log the ranks' u10+ x_real and their sum.
# Takes the host's exclusive lock (a timed run); the run stops (SIGINT to every trainer) when the slowest rank has
# UPDATES rows (a stop file the ranks agree on; SIGINT only if they have not ended 600 s later).
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TAG=$1; TREE=$2; shift 2
UPDATES=${UPDATES:-24}; HOURS=${HOURS:-30.0}; LIMIT=${LIMIT:-4500}
R=$BM/reports; LOG=$R/bm1.log
lane() { ( . "$1"; eval echo "\$$2" ); }
K=$#
[ $K -ge 2 ] || { echo "shardk.sh: at least two lane files" >&2; exit 2; }
G=""; PAIRS=""; CPUS=""; LCPUS=""; DCPUS=""; MEMG=0; CS=""; i=0; r=0
for f in "$@"; do
  for v in LANE_GPUS LANE_MEM LANE_NAME LANE_PORT; do [ -n "$(lane $f $v)" ] || { echo "shardk.sh: $v missing in $f" >&2; exit 2; }; done
  lg=$(lane $f LANE_GPUS); pair=""
  for g in $(echo $lg | tr ',' ' '); do G="${G:+$G,}$g"; pair="${pair:+$pair,}$i"; i=$((i+1)); done
  PAIRS="$PAIRS $pair"
  c=$(lane $f LANE_CPUSET); CPUS="${CPUS}${c}|"; [ -n "$c" ] && CS="${CS:+$CS,}$c"
  LCPUS="${LCPUS}$(lane $f LANE_LEARNER_CPUS)|"; DCPUS="${DCPUS}$(lane $f LANE_DRIVER_CPUS)|"
  MEMG=$((MEMG + $(lane $f LANE_MEM | tr -d 'gG')))
  [ -e $R/$TAG-s$r ] && { echo "shardk.sh: $R/$TAG-s$r exists (never overwritten)" >&2; exit 2; }
  r=$((r+1))
done
CPUS=${CPUS%|}; LCPUS=${LCPUS%|}; DCPUS=${DCPUS%|}
if [ "${NOLOCK:-0}" = 1 ]; then :; else take_excl "timed shards $TAG ($TREE, K $K, GPU $G) 计时运行"; fi
log "$TAG start: tree $TREE K $K GPUs $G (rank pairs$PAIRS) CPUs ${CPUS:-unpinned} learner CPUs ${LCPUS:-none} driver CPUs ${DCPUS:-none} memory ${MEMG}g updates $UPDATES hours $HOURS"
host_snapshot "$TAG start"
( while :; do echo "$(date +%s) $(awk '/^cpu MHz/ {printf "%d ", $4}' /proc/cpuinfo)"; sleep 5; done ) > $R/$TAG.cpufreq &
S1=$!
( while :; do echo "== $(date +%s)"; top -b -n 2 -d 2 -w 160 2>/dev/null | awk '/^top -/ {k++} k == 2' | sed -n '8,15p'; sleep 28; done ) > $R/$TAG.topcpu &
S2=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $R/$TAG.allgpu &
S3=$!
{ command -v mpstat > /dev/null && exec mpstat -P ALL 5; } > $R/$TAG.mpstat &
S4=$!
SP=""; r=0
for f in "$@"; do
  M=" --out $BM1_IN/reports/$TAG-s$r "
  python3 $BM/m0_mem.py $R/$TAG-s$r.mem.csv 5 "$M" $R/$TAG-s$r.threads.csv > $R/$TAG-s$r.mem.log 2>&1 &
  SP="$SP $!"
  nvidia-smi -i $(lane $f LANE_GPUS) --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,index --format=csv,noheader -l 1 > $R/$TAG-s$r.gpu &
  SP="$SP $!"
  for x in mpstat allgpu topcpu cpufreq; do ln -sf $TAG.$x $R/$TAG-s$r.$x; done
  r=$((r+1))
done
NB=$(lane $1 LANE_NAME); PB=$(lane $1 LANE_PORT)
CPUSET="${CS:-${BM1_TIMED_CPUSET:-$BM1_CPUSET}}" MEMS="" MEM="${MEMG}g" PIDS=$((${BM1_PIDS:-4096} * K)) GPUS_OK="$G" \
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 -e EXTRA=--learner-device=cuda:1 -e NAME_BASE=$NB -e PORT_BASE=$PB -e SHARD_PORT=${SHARD_PORT:-29400} ${ENVS:-}" \
  $BM/run.sh shards-$TAG $G sh $BM1_IN/shardk_in.sh $TAG $LIMIT $HOURS "$(echo $PAIRS)" "$CPUS" "$LCPUS" "$DCPUS" > $R/$TAG.launcher.log 2>&1 &
DP=$!
while kill -0 $DP 2>/dev/null; do
  sleep 15
  n=999999; r=0
  while [ $r -lt $K ]; do
    m=0; [ -f $R/$TAG-s$r/progress.csv ] && m=$(($(wc -l < $R/$TAG-s$r/progress.csv)-1))
    [ $m -lt $n ] && n=$m
    r=$((r+1))
  done
  if [ $n -ge $UPDATES ]; then
    log "$TAG: every rank has $n updates: the ranks' stop file (they end together at the next collect)"
    docker exec "$BM1_NAME-shards-$TAG" touch /tmp/shards-$TAG/stop
    t_s=$(date +%s)
    while kill -0 $DP 2>/dev/null && [ $(( $(date +%s) - t_s )) -lt 600 ]; do sleep 5; done
    if kill -0 $DP 2>/dev/null; then
      log "$TAG: not ended 600 s after the stop file: SIGINT to the trainers"
      docker exec "$BM1_NAME-shards-$TAG" pkill -INT -f "train_tok.py .*--out $BM1_IN/reports/$TAG-s"
    fi
    break
  fi
done
wait $DP
rc=$?
kill $S1 $S2 $S3 $S4 $SP 2>/dev/null
host_snapshot "$TAG end"
r=0
while [ $r -lt $K ]; do
  [ -d $R/$TAG-s$r/checkpoints ] && trash $R/$TAG-s$r/checkpoints
  python3 $BM/summ.py $R $TAG-s$r "$(lane $(eval echo \${$((r+1))}) LANE_CPUSET)" $(lane $(eval echo \${$((r+1))}) LANE_GPUS) > $R/$TAG-s$r.summary.txt 2>&1
  r=$((r+1))
done
xr=$(r=0; while [ $r -lt $K ]; do awk '/^  u10\+/ {print $NF}' $R/$TAG-s$r.summary.txt; r=$((r+1)); done | tr '\n' ' ')
log "$TAG u10+ x_real per rank: $xr= $(echo $xr | awk '{s=0; for (i=1;i<=NF;i++) s+=$i; print s}') in all"
log "$TAG done rc=$rc; $(grep -h 'BM1 C shard 0: ' $R/$TAG-s0.stdout 2>/dev/null | tail -1 | cut -c1-200)"
[ "${NOLOCK:-0}" = 1 ] || give
