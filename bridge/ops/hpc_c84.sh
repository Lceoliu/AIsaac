#!/bin/bash
# C84 "teacher v2 at scale" on HPC128 (sandbox ws3 = repo abplus-training 6b299c3, 128-core quota, H20): C77's
# configuration (C67 final + death teacher, forked searches, fast learner, 48 workers for the smaller quota) plus
# --teacher-v2 1 --teacher-v2-random 60 (B19 whole-branch distillation), 12,000 game hours: the A/B against C77 at
# equal game hours (acceptance: v2_prior rises, v2_improving_share falls; 128-seed floors vs C77's 0.9-1.05).
# usage: hpc_c84.sh run|watch
source /101063/AIsaac/ws3/hpc_env.sh
R=$ABP_HOME/runs
if [ "$1" = run ]; then
  mkdir -p $R/c84-teacher2/checkpoints && cp $R/c67-last.pt $R/c84-teacher2/checkpoints/from-c67-last.pt
  PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 48 \
    --game-hours 12000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 \
    --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --teacher-v2 1 --teacher-v2-random 60 \
    --resume $R/c84-teacher2/checkpoints/from-c67-last.pt --name hc84 --port 43000 --out $R/c84-teacher2 \
    > $R/c84-teacher2.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c84-teacher2.log | cut -c1-300; tail -n 3 $R/c84-teacher2.log | cut -c1-220; grep -c Traceback $R/c84-teacher2.log
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c84-teacher2/evals
  while true; do
    for ck in $(ls $R/c84-teacher2/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c84-teacher2/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc84e --port 43300 --out "$R/c84-teacher2/evals/$tag" \
        > "$R/c84-teacher2/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c84-teacher2/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c84-teacher2/evals/$tag/summary.json" "$tag" >> "$R/c84-teacher2/evals.jsonl"
    done
    [ -f "$R/c84-teacher2/stop-evals" ] && break
    sleep 120
  done
fi
