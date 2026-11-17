#!/bin/bash
# 10-12: moved to HPC160 sandbox ws6 (same code, 4a2c1b3): HPC128's GPU was held by a foreign 87 GB process (CUDA OOM at launch).
# C92 "item tokens + global token" on HPC128 (sandbox ws4, worktree moved to the B21/B22 commit): C90's final weights
# (memory line, 36,000 own game hours, 1.32 floors) resumed with --item-tokens 1 --global-token 1 (B21/B22, gates start
# at zero = same outputs as C90 at the start) and items put into the training distribution: start builds on half of
# the floors (C70's setting) + heal/resource rewards (C73's). 12,000 game hours; control = C91 at the same point.
# usage: hpc_c92.sh smoke [extra args] | run | watch
source /101063/AIsaac/ws6/hpc_env.sh
R=$ABP_HOME/runs
CK=/101063/AIsaac/ws4/runs/c90-memory-cont/checkpoints/last.pt
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --start-build-prob 0.5 --start-build-max 3 --reward heal=0.5,resource=0.25"
if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-c92}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsm2 --port 47600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "memory:\|item tokens\|global token\|CUDA graphs\|eager" $R/$tag.log | head -n 6 | cut -c1-200
  grep "^finished\|^u[0-9]* \|^updates" $R/$tag.log | tail -n 4 | cut -c1-220
  exit 0
fi
if [ "$1" = run ]; then
  mkdir -p $R/c92-items/checkpoints && cp $CK $R/c92-items/checkpoints/from-c90-last.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 48 --game-hours 12000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c92-items/checkpoints/from-c90-last.pt --name hc92 --port 47000 --out $R/c92-items > $R/c92-items.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c92-items.log | cut -c1-300; grep -i "memory:\|item tokens\|global token" $R/c92-items.log | cut -c1-160; tail -n 3 $R/c92-items.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c92-items.log)"
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c92-items/evals
  while true; do
    for ck in $(ls $R/c92-items/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c92-items/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc92e --port 47300 --out "$R/c92-items/evals/$tag" \
        > "$R/c92-items/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c92-items/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c92-items/evals/$tag/summary.json" "$tag" >> "$R/c92-items/evals.jsonl"
    done
    [ -f "$R/c92-items/stop-evals" ] && break
    sleep 120
  done
fi
