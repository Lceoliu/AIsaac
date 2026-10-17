#!/bin/bash
# B38 step 4 (the learner's fixed cost per part) on host A, sandbox ws14, GPUs 2 and 3 only (C99 on 0 / 1), no
# searches. The replays use the C99-size job saved in step 3 (ws14/runs/b38-job/job.pt, 32,768 decisions) and start
# from C99's segment1-final.pt; no actor, no games (abplus_b38_replay.py).
# usage: hpc_b38_step4.sh rep <tag> <reps> [extra train_tok flags...]   a replay into ws14/runs/<tag> (GPU 2)
#        hpc_b38_step4.sh prof <tag> [extra flags...]                   the same, the 3rd update under the torch
#                                                                       profiler (ws14/runs/<tag>/prof)
source /101063/AIsaac/ws14/hpc_env.sh > /dev/null
export CUDA_VISIBLE_DEVICES=${B38_GPUS:-2,3}
R=$ABP_HOME/runs
CK=/101063/AIsaac/ws9/runs/c99-scratch/checkpoints/segment1-final.pt
BASE="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher-queue 12 \
  --archive-size 12 --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --item-tokens 1 --global-token 1 --item-shared 1 \
  --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal --pickup-force 1:0 \
  --reward heal=0.5,resource=0.25 --item-slots 8 --temporal 1 --temporal-width 256 --temporal-layers 4 \
  --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 --obs-v2 1 --cf 0"
RP="--workers 64 --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK --learner-process 1 \
  --overlap --decoupled 1 --learner-device cuda:0 --name hb38r --port 49850"
mode=$1; tag=$2; shift 2
[ -e "$R/$tag" ] && { echo "$R/$tag exists"; exit 1; }
if [ "$mode" = prof ]; then
  export ISAAC_RL_B38_TORCHPROF=$R/$tag/prof ISAAC_RL_B38_TORCHPROF_AT=3
  reps=4
else
  reps=$1; shift
fi
PYTHONPATH=. timeout -s INT 1500 $PY abplus_b38_replay.py --job $R/b38-job/job.pt --reps $reps -- $BASE $RP "$@" \
  --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
echo "$tag: exit $? tracebacks $(grep -c Traceback $R/$tag.log)"
grep "^ranks" $R/$tag.log | cut -c1-400
