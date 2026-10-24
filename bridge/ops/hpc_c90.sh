#!/bin/bash
# C90 = C87 continued on HPC128 (sandbox ws4): the GRU-memory line (C77 configuration + --memory gru) resumed from
# C87's final checkpoint (weights + Adam) for 24,000 more game hours, to see whether memory keeps rising where C77
# plateaued (1.05-1.25 after 20,000 gh). Same parameters as C87. usage: hpc_c90.sh run | watch
source /101063/AIsaac/ws4/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32"
if [ "$1" = run ]; then
  mkdir -p $R/c90-memory-cont/checkpoints && cp $R/c87-memory/checkpoints/last.pt $R/c90-memory-cont/checkpoints/from-c87-last.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 48 --game-hours 24000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c90-memory-cont/checkpoints/from-c87-last.pt --name hc90 --port 45000 --out $R/c90-memory-cont > $R/c90-memory-cont.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c90-memory-cont.log | cut -c1-300; tail -n 3 $R/c90-memory-cont.log | cut -c1-220; grep -c Traceback $R/c90-memory-cont.log
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c90-memory-cont/evals
  while true; do
    for ck in $(ls $R/c90-memory-cont/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c90-memory-cont/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc90e --port 45300 --out "$R/c90-memory-cont/evals/$tag" \
        > "$R/c90-memory-cont/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c90-memory-cont/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c90-memory-cont/evals/$tag/summary.json" "$tag" >> "$R/c90-memory-cont/evals.jsonl"
    done
    [ -f "$R/c90-memory-cont/stop-evals" ] && break
    sleep 120
  done
fi
