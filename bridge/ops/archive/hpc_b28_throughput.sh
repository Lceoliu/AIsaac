#!/bin/bash
# B28 (2026-10-14): throughput vs worker count on HPC128 (sandbox ws10 = B24 code, GPU free): the actor is GPU-latency
# bound (64 workers idle 0.75 waiting for a 2.4 ms inference round trip while the CPU load is 28/160), so more workers
# per batch should raise x almost linearly until the engine CPU or the inference batch saturates.
# usage: hpc_b28_throughput.sh <checkpoint> [worker counts...]   (each count runs ~7 min, then the last 3 progress lines)
source /101063/AIsaac/ws10/hpc_env.sh
R=$ABP_HOME/runs; CK=$1; shift
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 0.6:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8"
for N in "$@"; do
  tag=b28-w$N
  PYTHONPATH=. timeout 420 $PY train_tok.py $COMMON --workers $N --game-hours 5000 --checkpoint-every 100000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $CK --name hb28 --port 49800 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "== workers $N: tracebacks $(grep -c Traceback $R/$tag.log); load $(cut -d' ' -f1-3 /proc/loadavg); game processes $(pgrep -c isaac.x64)"
  grep "^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-120
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  pkill -KILL -f "[f]rom multiprocessing" 2>/dev/null; pkill -KILL isaac.x64 2>/dev/null; sleep 10
  rm -rf $R/$tag   # (no checkpoints written; the log stays)
done
