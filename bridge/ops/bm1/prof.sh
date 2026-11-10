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
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
. $BM/lock.sh
TAG=$1; TREE=$2
GPU=${GPU:-1}; UPDATES=${UPDATES:-24}; HOURS=${HOURS:-30.0}; LIMIT=${LIMIT:-4500}
R=$BM/reports; LOG=$R/bm1.log
mkdir -p $R
[ -e $R/$TAG ] && { echo "prof.sh: $R/$TAG exists (never overwritten)" >&2; exit 2; }
take "timed $TAG ($TREE, GPU $GPU) 计时运行"
echo "$(date '+%F %T') $TAG start: tree $TREE GPU $GPU updates $UPDATES hours $HOURS lock: $LOCKLINE" >> $LOG
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader -l 1 > $R/$TAG.gpu &
S1=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 > $R/$TAG.allgpu &
S2=$!
mpstat -P ALL 5 > $R/$TAG.mpstat &
S3=$!
C=aisaac-bench-bm1-a1-prof-$TAG
# M0 (host /proc, matched by this run's --out: K trainers on one host each get their own): memory by process kind and
# the container's cgroup, cores by process kind, the trainer's per-thread ticks (5 s). (b57_threads.py is not used: given
# a match string it matched its own command line, base-m77657fc)
M=" --out /bench/bm1/reports/$TAG"
python3 $BM/m0_mem.py $R/$TAG.mem.csv 5 "$M" $R/$TAG.threads.csv > $R/$TAG.mem.log 2>&1 &
S4=$!
S5=
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 ${ENVS:-}" $BM/dock.sh prof-$TAG $GPU sh /bench/bm1/xnew2.sh $TAG $LIMIT $HOURS &
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
    docker exec $C pkill -INT -f '^/bench/home/isaac-abplus/.venv/bin/python -u train_tok.py'
    break
  fi
done
wait $DP
rc=$?
kill $S1 $S2 $S3 $S4 $S5 $S6 2>/dev/null
[ -d $R/$TAG/checkpoints ] && gio trash $R/$TAG/checkpoints
echo "$(date '+%F %T') $TAG done rc=$rc rows $(($(wc -l < $R/$TAG/progress.csv 2>/dev/null || echo 1)-1))" >> $LOG
python3 $BM/summ.py $R $TAG "${CPUSET:-0-11,24-35}" $GPU > $R/$TAG.summary.txt 2>&1   # (still inside the lock)
give
