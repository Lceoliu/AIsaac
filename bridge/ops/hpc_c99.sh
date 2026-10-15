#!/bin/bash
# C99 "the new architecture from scratch" (user 10-15/16): the B33 default model (spatial transformer + B31 temporal
# attention 256x4 / W 256 / rooms 128, item tokens (8 slots), global token, shared table; no GRU, no gates: the temporal
# side is part of the network from the first step) trained from zero in run mode on
# HPC160 (sandbox ws9, 64 workers, B29 actor flags), with the environment design of the items line (pedestal start
# builds, forced pickup 1 -> 0 over the run, heal / resource rewards). No --resume / --init: random weights.
# Purpose: does the architecture itself learn faster than the inherited lineage (C77 / C87 curves at equal game hours),
# and does it pass the inherited 1.7 floors? usage: hpc_c99.sh smoke | run | watch
# 2026-10-16 (B35, user-approved): --obs-v2 1 (pickup prices, heart split, room type; ROW v2). Needs the sandbox's
# abp_bridge.lua = abp-0.2.18 and a libabp_turbo.so rebuilt from this commit's abp_turbo.c (abp_native_entity_bytes 148,
# abp_row_encode version 7); with an older library the bridge falls back to the Lua observation and FastRow is off (slow).
# 2026-10-16 (B34, user-approved, rl/docs/COUNTERFACTUAL_TEACHER.md): the counterfactual branch teacher (--cf 1) with
# the design's starting values: 6 choice points per worker per game hour, 4 branches x 180 game s (taken, target,
# 2 sticky high-temperature), margin 0.3, weight cap 2, the improving branch's first 8 records distilled, 1 random
# point per game hour, no advantage replacement; its points run in forked children of the workers (--teacher-thread 1
# --teacher-fork 1 above), its records through teacher v2's ring and loss (--teacher-v2-* learner defaults); raise
# --cf-points to 10-12 once the throughput allows (progress.csv cf_*).
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
  tag=${TAG:-smoke-c99}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --name hsm9 --port 49600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "temporal\|memory:\|pedestal\|pickup" $R/$tag.log | head -n 6 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  exit 0
fi
if [ "$1" = run ]; then
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST --workers 64 --game-hours 25000 --checkpoint-every 1000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --name hc99 --port 49000 --out $R/c99-scratch > $R/c99-scratch.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c99-scratch.log | cut -c1-300; tail -n 3 $R/c99-scratch.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c99-scratch.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c99-scratch/evals
  while true; do
    for ck in $(ls $R/c99-scratch/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c99-scratch/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc99e --port 49300 --out "$R/c99-scratch/evals/$tag" \
        > "$R/c99-scratch/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c99-scratch/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c99-scratch/evals/$tag/summary.json" "$tag" >> "$R/c99-scratch/evals.jsonl"
    done
    [ -f "$R/c99-scratch/stop-evals" ] && break
    sleep 120
  done
fi
