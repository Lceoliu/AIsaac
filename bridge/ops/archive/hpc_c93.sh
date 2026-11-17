#!/bin/bash
# C93 = C92 continued on HPC128 (sandbox ws4): the item-token / global-token line (C92 final: +0.06 floors over C91 at
# equal updates, builds help monotonically) resumed for 12,000 more game hours, same parameters, --micro 4096 (the GPU
# is ours alone again). usage: hpc_c93.sh run | watch
source /101063/AIsaac/ws4/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --start-build-prob 0.5 --start-build-max 3 --reward heal=0.5,resource=0.25"
if [ "$1" = run ]; then
  mkdir -p $R/c93-items-cont/checkpoints && cp $R/c92-items/checkpoints/last.pt $R/c93-items-cont/checkpoints/from-c92-last.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 48 --game-hours 12000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c93-items-cont/checkpoints/from-c92-last.pt --name hc93 --port 48000 --out $R/c93-items-cont > $R/c93-items-cont.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c93-items-cont.log | cut -c1-300; tail -n 3 $R/c93-items-cont.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c93-items-cont.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c93-items-cont/evals
  while true; do
    for ck in $(ls $R/c93-items-cont/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c93-items-cont/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc93e --port 48300 --out "$R/c93-items-cont/evals/$tag" \
        > "$R/c93-items-cont/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c93-items-cont/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c93-items-cont/evals/$tag/summary.json" "$tag" >> "$R/c93-items-cont/evals.jsonl"
    done
    [ -f "$R/c93-items-cont/stop-evals" ] && break
    sleep 120
  done
fi
