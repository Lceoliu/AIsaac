#!/bin/bash
# B29 (2026-10-14): actor latency knobs in a real run on HPC128 (sandbox ws10), 64 workers, the B28 configuration
# otherwise; each configuration runs 7 minutes (timeout 420), then the last 3 progress lines and the actor's columns.
# usage: hpc_b29_knobs.sh <checkpoint> "<tag>|<extra train_tok arguments>" ...
source /101063/AIsaac/ws10/hpc_env.sh
R=$ABP_HOME/runs; CK=$1; shift
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 0.6:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8"
for spec in "$@"; do
  tag=b29-${spec%%|*}; extra=${spec#*|}
  PYTHONPATH=. timeout 420 $PY train_tok.py $COMMON --workers 64 --game-hours 5000 --checkpoint-every 100000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 $extra \
    --resume $CK --name hb29 --port 49800 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "== $tag ($extra): tracebacks $(grep -c Traceback $R/$tag.log); load $(cut -d' ' -f1-3 /proc/loadavg)"
  grep "^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-120
  cp $R/$tag/progress.csv $R/$tag.progress.csv 2>/dev/null
  pkill -KILL -f "[f]rom multiprocessing" 2>/dev/null; pkill -KILL isaac.x64 2>/dev/null; sleep 10
  rm -rf $R/$tag   # (no checkpoints written; the log and progress.csv stay)
done
