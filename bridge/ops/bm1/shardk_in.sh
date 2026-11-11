#!/bin/sh
# bm1 BM1 C, inside the one container of shardk.sh: K trainers (xnew2.sh) as the ranks of one training run.
#   usage: shardk_in.sh <tag> <limit_s> <game_hours> "<rank 0 gpus> <rank 1 gpus> ..." "<rank 0 cpus>|<rank 1 cpus>|..."
#   e.g.   shardk_in.sh c2 4500 30.0 "0,1 2,3" "0-7,16-23|8-15,24-31"     (container GPUs: the order run.sh got them)
# Each rank: CUDA_VISIBLE_DEVICES its two cards (cuda:0 actor, cuda:1 learner), its CPUs by taskset (empty: none), its
# instances' NAME / PORT (NAME_BASE / PORT_BASE + 1000 * rank), ISAAC_SHARD_* and the rendezvous on 127.0.0.1:SHARD_PORT;
# out dir reports/<tag>-s<rank>. A rank that ends with an error: the others get GRACE s (default 90) to end through
# their own watchdog (shard_sync), then SIGINT, then 120 s later SIGKILL. Exit code: the first non-zero one, else 0.
TAG=$1; LIMIT=$2; HOURS=$3; PAIRS=$4; CPUS=${5:-}
K=$(echo $PAIRS | wc -w)
export ISAAC_SHARD_WORLD=$K MASTER_ADDR=127.0.0.1 MASTER_PORT=${SHARD_PORT:-29400} ISAAC_SHARD_DIR=/tmp/shards-$TAG
mkdir -p $ISAAC_SHARD_DIR
r=0; PIDS=""
for pair in $PAIRS; do
  cs=$(echo "$CPUS" | cut -d'|' -f$((r+1)))
  ( export CUDA_VISIBLE_DEVICES=$pair ISAAC_SHARD_RANK=$r NAME=${NAME_BASE:-b42new}s$r PORT=$((${PORT_BASE:-39200} + 1000 * r))
    if [ -n "$cs" ]; then exec taskset -c $cs sh $BM1_IN/xnew2.sh $TAG-s$r $LIMIT $HOURS
    else exec sh $BM1_IN/xnew2.sh $TAG-s$r $LIMIT $HOURS; fi ) &
  PIDS="$PIDS $!"
  r=$((r+1))
done
echo "shardk_in: $K ranks started (pids$PIDS), rendezvous 127.0.0.1:$MASTER_PORT"
rc=0; first=""
alive() { st=$(sed 's/.*) //' /proc/$1/stat 2>/dev/null | cut -d' ' -f1); [ -n "$st" ] && [ "$st" != Z ]; }
while :; do
  alive=""
  for p in $PIDS; do
    if alive $p; then alive="$alive $p"
    else wait $p; c=$?; [ $c != 0 ] && [ -z "$first" ] && { first=$(date +%s); rc=$c; echo "shardk_in: pid $p ended rc=$c"; }; fi
  done
  PIDS=$alive
  [ -z "$PIDS" ] && break
  if [ -n "$first" ]; then
    el=$(( $(date +%s) - first ))
    if [ $el -ge $(( ${GRACE:-90} + 120 )) ]; then pkill -KILL -f "train_tok.py .*--out $BM1_IN/reports/$TAG-s"
    elif [ $el -ge ${GRACE:-90} ]; then pkill -INT -f "train_tok.py .*--out $BM1_IN/reports/$TAG-s"; fi
  fi
  sleep 2
done
exit $rc
