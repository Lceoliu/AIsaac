#!/bin/bash
# B24 (fused item/global attention + actor-side value_global + --item-slots) GPU checks on HPC160 sandbox ws8
# (repo at the B24 commit), while C94 keeps running on the same node (so timings are under C94's load; compare with
# an old-path smoke run at the same time, hpc_c94.sh smoke).
# usage: hpc_b24.sh test | smoke <checkpoint> [extra args]
source /101063/AIsaac/ws8/hpc_env.sh
R=$ABP_HOME/runs
if [ "$1" = test ]; then
  PYTHONPATH=. timeout 1800 $PY tests/test_fused_tokens_smoke.py 2>&1 | tail -n 12
  exit 0
fi
if [ "$1" = smoke ]; then
  shift; CK=$1; shift
  tag=${TAG:-smoke-b24}
  COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
    --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
    --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
    --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8"
  PYTHONPATH=. timeout 1800 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsb24 --port 49700 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "fused\|memory:\|item tokens\|global token\|shared\|convert" $R/$tag.log | head -n 8 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | cut -c1-160
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/$tag/progress.csv')))
print([(r['update'], r['kl'][:6], r.get('ev','')[:5], r.get('update_s',r.get('update_time',''))[:5], r.get('pickups_free'), r.get('pickup_imit','')[:6]) for r in rows])"
  exit 0
fi
