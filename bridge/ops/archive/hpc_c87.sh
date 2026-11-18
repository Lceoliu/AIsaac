#!/bin/bash
# C87 "GRU memory at scale" on HPC128 (sandbox ws4 = repo abplus-training 13f917c, 128-core quota, H20): C77's
# configuration (C67 final + death teacher, forked searches, fast learner, 48 workers) plus B20 memory
# (--memory gru --memory-dim 128 --memory-bptt 32), 12,000 game hours: the A/B against C77 / C84 at equal game hours.
# usage: hpc_c87.sh smoke [extra args] | run | watch
source /101063/AIsaac/ws4/hpc_env.sh
R=$ABP_HOME/runs
A_C67=$R/c67-last.pt; [ -f $A_C67 ] || cp /101063/AIsaac/ws3/runs/c67-last.pt $A_C67
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32"
if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-mem}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $A_C67 --name hsm --port 44600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "memory\|CUDA graphs\|eager" $R/$tag.log | head -n 5 | cut -c1-200
  grep "^finished\|^u[0-9]* \|^updates" $R/$tag.log | tail -n 4 | cut -c1-220
  exit 0
fi
if [ "$1" = run ]; then
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 48 --game-hours 12000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $A_C67 --name hc87 --port 44000 --out $R/c87-memory > $R/c87-memory.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c87-memory.log | cut -c1-300; tail -n 3 $R/c87-memory.log | cut -c1-220; grep -c Traceback $R/c87-memory.log
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c87-memory/evals
  while true; do
    for ck in $(ls $R/c87-memory/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c87-memory/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc87e --port 44300 --out "$R/c87-memory/evals/$tag" \
        > "$R/c87-memory/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c87-memory/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c87-memory/evals/$tag/summary.json" "$tag" >> "$R/c87-memory/evals.jsonl"
    done
    [ -f "$R/c87-memory/stop-evals" ] && break
    sleep 120
  done
fi
