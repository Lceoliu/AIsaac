#!/bin/bash
# C91 = the GRU-memory line continued on HPC160 (sandbox ws5 = repo abplus-training b18479e, 160-core quota): C90's
# final checkpoint (C87 + C90 = 36,000 own game hours, 1.32 floors) resumed for 50,000 more game hours with C77's
# node configuration (64 workers) + --memory gru. User 2026-10-12: "记忆线在 HPC 上再放大 5 万小时".
# usage: hpc_c91.sh run | watch
source /101063/AIsaac/ws5/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32"
if [ "$1" = run ]; then
  mkdir -p $R/c91-memory-cont2/checkpoints && cp /101063/AIsaac/ws4/runs/c90-memory-cont/checkpoints/last.pt $R/c91-memory-cont2/checkpoints/from-c90-last.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 64 --game-hours 50000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c91-memory-cont2/checkpoints/from-c90-last.pt --name hc91 --port 46000 --out $R/c91-memory-cont2 > $R/c91-memory-cont2.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c91-memory-cont2.log | cut -c1-300; grep "memory:" $R/c91-memory-cont2.log | cut -c1-160; tail -n 3 $R/c91-memory-cont2.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c91-memory-cont2.log)"
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c91-memory-cont2/evals
  while true; do
    for ck in $(ls $R/c91-memory-cont2/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c91-memory-cont2/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc91e --port 46300 --out "$R/c91-memory-cont2/evals/$tag" \
        > "$R/c91-memory-cont2/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c91-memory-cont2/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c91-memory-cont2/evals/$tag/summary.json" "$tag" >> "$R/c91-memory-cont2/evals.jsonl"
    done
    [ -f "$R/c91-memory-cont2/stop-evals" ] && break
    sleep 120
  done
fi
