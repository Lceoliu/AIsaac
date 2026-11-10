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
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 ${ENVS:-}" $BM/dock.sh prof-$TAG $GPU sh /bench/bm1/xnew2.sh $TAG $LIMIT $HOURS &
DP=$!
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
kill $S1 $S2 $S3 2>/dev/null
[ -d $R/$TAG/checkpoints ] && gio trash $R/$TAG/checkpoints
echo "$(date '+%F %T') $TAG done rc=$rc rows $(($(wc -l < $R/$TAG/progress.csv 2>/dev/null || echo 1)-1))" >> $LOG
give
