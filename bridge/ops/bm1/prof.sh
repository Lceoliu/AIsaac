#!/bin/sh
# bm1 port of the 4090's prof2.sh + b62_step0.sh P24: a timed profile of xnew2.sh's configuration, GPU-synced phase
# timers (ISAAC_BLOCK_TIMING=1), stopped by SIGINT to train_tok after N updates (or the game hours / time limit).
# Host side; takes the lane lock (timed run: nothing else on this lane meanwhile). Samples while it runs:
#   <tag>.gpu      nvidia-smi of its GPU, 1 s (util, memory, power, SM clock)
#   <tag>.allgpu   nvidia-smi of every GPU, 5 s (the other tenants' load)
#   <tag>.mpstat   mpstat -P ALL 5 (every CPU: this lane's and the others')
# Afterwards the run's checkpoints are moved to the trash (speed runs keep none).
#   usage: [GPU=1] [UPDATES=24] [HOURS=30.0] [LIMIT=4500] [ENVS="-e K=V ..."] prof.sh <tag> <tree>
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TAG=$1; TREE=$2
GPU=${GPU:-1}; UPDATES=${UPDATES:-24}; HOURS=${HOURS:-30.0}; LIMIT=${LIMIT:-4500}
R=$BM/reports; LOG=$R/bm1.log
mkdir -p $R
[ -e $R/$TAG ] && { echo "prof.sh: $R/$TAG exists (never overwritten)" >&2; exit 2; }
if [ "${NOLOCK:-0}" = 1 ]; then LOCKLINE="(held by the caller: ${LOCKS_HELD:-?})"; else take "timed $TAG ($TREE, GPU $GPU) 计时运行"; fi
echo "$(date '+%F %T') $TAG start: tree $TREE capture mode $(capmode $TREE) GPU $GPU updates $UPDATES hours $HOURS lock: $LOCKLINE" >> $LOG
host_snapshot "$TAG start"
# (a timed run's own CPUs / memory nodes when bm1.env sets them: gpu57's timed runs use every thread)
# (LANE_CPUS=1: the caller's CPUSET / MEMS kept as given, shard2.sh's lanes; empty: no pinning)
[ -n "${BM1_TIMED_CPUSET+x}" ] && [ "${LANE_CPUS:-0}" != 1 ] && export CPUSET="$BM1_TIMED_CPUSET" MEMS="${BM1_TIMED_MEMS:-}"
SUMM_CPUS=${CPUSET:-${BM1_TIMED_CPUSET:-$BM1_CPUSET}}
torch_line=$(grep -h -m1 '' $BM/.torch_build 2>/dev/null)
log "$TAG software: driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i ${GPU%%,*} 2>/dev/null) ${torch_line:-torch build: see the run's stdout}"
# CPU frequency (read-only /proc/cpuinfo, every 5 s): every CPU's MHz on one line
( while :; do echo "$(date +%s) $(awk '/^cpu MHz/ {printf "%d ", $4}' /proc/cpuinfo)"; sleep 5; done ) > $R/$TAG.cpufreq &
S7=$!
# (each CPU's current frequency every second, kHz, from cpufreq's scaling_cur_freq in CPU order; read only)
FQ=$(n=0; while [ -r /sys/devices/system/cpu/cpu$n/cpufreq/scaling_cur_freq ]; do echo /sys/devices/system/cpu/cpu$n/cpufreq/scaling_cur_freq; n=$((n+1)); done)
( [ -n "$FQ" ] && while :; do echo "$(date +%s.%N | cut -c1-14) $(awk '{printf "%d ", $1}' $FQ)"; sleep 1; done ) > $R/$TAG.freq1 &
S8=$!
# (every 30 s, the host's top CPU processes over a 2 s window (top's second frame), every tenant's: the run's outside load)
( while :; do echo "== $(date +%s)"; top -b -n 2 -d 2 -w 160 2>/dev/null | awk '/^top -/ {k++} k == 2' | sed -n '8,15p'; sleep 28; done ) > $R/$TAG.topcpu &
S9=$!
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,index --format=csv,noheader -l 1 > $R/$TAG.gpu &
S1=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $R/$TAG.allgpu &
S2=$!
{ command -v mpstat > /dev/null && exec mpstat -P ALL 5; } > $R/$TAG.mpstat &
S3=$!
# M0 (host /proc, matched by this run's --out: K trainers on one host each get their own): memory by process kind and
# the container's cgroup, cores by process kind, the trainer's per-thread ticks (5 s). (b57_threads.py is not used: given
# a match string it matched its own command line, base-m77657fc)
M=" --out $BM1_IN/reports/$TAG"
python3 $BM/m0_mem.py $R/$TAG.mem.csv 5 "$M" $R/$TAG.threads.csv > $R/$TAG.mem.log 2>&1 &
S4=$!
S5=
# (the host's learner cores, bm1.env BM1_LEARNER_CPUS: added unless the run names its own ISAAC_LEARNER_CPUS, runs as a
#  shard2 lane, or says LEARNER_PIN=0)
LP=""
if [ -n "${BM1_LEARNER_CPUS:-}" ] && [ "${LANE_CPUS:-0}" != 1 ] && [ "${LEARNER_PIN:-1}" != 0 ]; then
  case "${ENVS:-}" in *ISAAC_LEARNER_CPUS=*) ;; *) LP="-e ISAAC_LEARNER_CPUS=$BM1_LEARNER_CPUS" ;; esac
fi
log "$TAG learner CPUs: $(echo "${ENVS:-} $LP" | grep -o 'ISAAC_LEARNER_CPUS=[^ ]*' | head -1 || true) ${LP:+(bm1.env BM1_LEARNER_CPUS)}"
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 -e ISAAC_BLOCK_TIMELINE=$BM1_IN/reports/$TAG.timeline.csv ${ENVS:-} $LP" $BM/run.sh prof-$TAG $GPU sh $BM1_IN/xnew2.sh $TAG $LIMIT $HOURS &
DP=$!
# progress.csv row times (1 s): each update's end on the host clock, to place the 1 s GPU / CPU samples in each cycle
( k=0; while kill -0 $DP 2>/dev/null; do
    n=0; [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
    [ $n != $k ] && { echo "$(date +%s.%N) $n" >> $R/$TAG.rows; k=$n; }
    sleep 1
  done ) &
S6=$!
while kill -0 $DP 2>/dev/null; do
  sleep 15
  n=0
  [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
  if [ $n -ge $UPDATES ]; then
    echo "$(date '+%F %T') $TAG: $n updates, SIGINT to train_tok" >> $LOG
    sigint_trainer prof-$TAG $TAG
    break
  fi
done
wait $DP
rc=$?
kill $S1 $S2 $S3 $S4 $S5 $S6 $S7 $S8 $S9 2>/dev/null
host_snapshot "$TAG end"
[ -d $R/$TAG/checkpoints ] && trash $R/$TAG/checkpoints
echo "$(date '+%F %T') $TAG done rc=$rc rows $(($(wc -l < $R/$TAG/progress.csv 2>/dev/null || echo 1)-1)); $(grep -h -m1 "BM1 graph capture mode" $R/$TAG.stdout 2>/dev/null)" >> $LOG
python3 $BM/summ.py $R $TAG "$SUMM_CPUS" $GPU > $R/$TAG.summary.txt 2>&1   # (still inside the lock)
[ "${NOLOCK:-0}" = 1 ] || give
