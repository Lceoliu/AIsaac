#!/bin/bash
# Smokes of the predictive-memory switches on host 2's one GPU (sandbox ~/isaac-abplus/pred): C99's flags from segment2-final,
# 8 workers, learner on the actor's GPU; real engine, real CUDA, the teacher and the counterfactual lanes on. Not a
# throughput measurement (that is the 3-GPU gate, M4). usage: host2_pred_smoke.sh <tag> [train_tok arguments]
cd /home/rhythmo/isaac-abplus/pred/python || exit 1
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
R=../runs
CK=/home/rhythmo/isaac-abplus/b35/ck/segment2-final.pt
tag=$1; shift
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --obs-v2 1 \
  --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 \
  --cf-distill-steps 8 --cf-random 1 --cf-advantage 0"
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
export OMP_NUM_THREADS=4
PYTHONPATH=. timeout ${SMOKE_TIMEOUT:-1500} $PY train_tok.py $COMMON $FAST --overlap --decoupled 1 --workers ${WORKERS:-8} --game-hours ${HOURS:-1.5} \
  --checkpoint-every 100000 --rollout ${ROLLOUT:-8192} --minibatch 2048 --micro 1024 --learner-fast 1 --resume $CK "$@" \
  --name "ps$tag" --port ${PORT:-49700} --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
echo "exit $?; tracebacks $(grep -c Traceback $R/$tag.log)"
grep -i "aux (tok_aux)\|privileged critic\|aux: --resume" $R/$tag.log | cut -c1-200
grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 4 | cut -c1-220
