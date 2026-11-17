#!/bin/bash
# C92 continued on the re-provisioned HPC128 (10-12 22:xx): resumes the ws6 (HPC160) run's last checkpoint in ws4 so
# C91 gets HPC160's GPU back. Same parameters as hpc_c92.sh run, --micro 2048 (the new container's H20 shares 75 GB
# with a foreign process; the learner reserved 17 GB at --micro 4096), --game-hours counted from the resume point.
# usage: hpc_c92_resume.sh run <ws6 last.pt> <remaining game hours> | watch
source /101063/AIsaac/ws4/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --start-build-prob 0.5 --start-build-max 3 --reward heal=0.5,resource=0.25"
if [ "$1" = run ]; then
  SRC=$2; HOURS=${3:-11000}
  mkdir -p $R/c92-items/checkpoints && cp "$SRC" $R/c92-items/checkpoints/from-ws6-last.pt
  [ -f $(dirname $(dirname "$SRC"))/evals.jsonl ] && cp $(dirname $(dirname "$SRC"))/evals.jsonl $R/c92-items/evals.jsonl
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 48 --game-hours $HOURS --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 2048 --learner-fast 1 \
    --resume $R/c92-items/checkpoints/from-ws6-last.pt --name hc92 --port 47000 --out $R/c92-items > $R/c92-items.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c92-items.log | cut -c1-300; grep -i "memory:\|item tokens\|global token" $R/c92-items.log | cut -c1-160; tail -n 3 $R/c92-items.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c92-items.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
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
