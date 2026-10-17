#!/bin/bash
# C101 "minibatch 8192" A/B (user 10-18): the C99 recipe (COMMON / FAST below = hpc_c99.sh's) continued from C99's
# segment1-final.pt (u20902, 25,379 h) for 10,000 game hours with --minibatch 8192 --micro 8192 (12 optimizer steps per
# update instead of 24; LR unchanged 1e-4), B38 4-GPU layout (two actor processes on the 1st and 3rd visible GPUs, the
# learner process on the 2nd, --learner-fast 2, overlap + decoupled). Control = C99 segment 2 (minibatch 4096) over the
# same hours from the same checkpoint: compare the 128-seed curves at equal game hours (B38 step 4 measured update
# 2.21 s vs 2.77 s on the replay bench and clip_frac 0.09 vs 0.18). usage: hpc_c101.sh smoke | run | watch
# Run with CUDA_VISIBLE_DEVICES=0,1,2,3 (actors 0 and 2, learner 1, watcher 3). Sandbox ws9 (C99's), own --out / name / ports.
source /101063/AIsaac/ws9/hpc_env.sh
R=$ABP_HOME/runs
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
if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-c101}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --learner-device cuda:1 --learner-process 1 --overlap --decoupled 1     --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2     --workers 8 --game-hours 1.5 --checkpoint-every 100000 --rollout 8192 --minibatch 8192 --micro 8192 "$@"     --name hsm1 --port 49700 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  exit 0
fi
if [ "$1" = run ]; then
  CK=${2:-$R/c99-scratch/checkpoints/segment1-final.pt}
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST --learner-device cuda:1 --learner-process 1 --overlap --decoupled 1     --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2     --workers 64 --game-hours 10000 --checkpoint-every 1000 --rollout 32768 --minibatch 8192 --micro 8192     --resume "$CK" --name hc101 --port 49100 --out $R/c101-mb8192 > $R/c101-mb8192.log 2>&1 < /dev/null &
  sleep 300
  date +%T; grep -i "devices (B38)" $R/c101-mb8192.log | tail -n 1 | cut -c1-160; tail -n 3 $R/c101-mb8192.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c101-mb8192.log)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c101-mb8192/evals
  while true; do
    for ck in $(ls $R/c101-mb8192/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c101-mb8192/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json         --mode run --items --workers 8 --seeds 2147490000:128 --name hc101e --port 49400 --out "$R/c101-mb8192/evals/$tag"         > "$R/c101-mb8192/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c101-mb8192/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c101-mb8192/evals/$tag/summary.json" "$tag" >> "$R/c101-mb8192/evals.jsonl"
    done
    [ -f "$R/c101-mb8192/stop-evals" ] && break
    sleep 120
  done
fi
