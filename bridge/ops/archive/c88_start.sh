#!/bin/bash
# C88 on the 3090 (sandbox mem = workspace 10-11 04:2x, repo 23383cd): C67 final + death teacher (forked searches, fast
# learner, C85's teacher settings without --archive-plr) + B20 GRU memory (--memory gru --memory-dim 128 --memory-bptt 32),
# 3,000 game hours: the host-scale replicate of C87 (HPC128). Controls: C75 0.90 / C80 0.77 / C81 0.90 (no memory),
# C83 1.08 (teacher v2). usage: c88_start.sh run|watch

S=/home/rhythmo/isaac-abplus/mem
R=$S/runs
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
CK=/home/rhythmo/isaac-abplus/ev/runs/c67-last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.5 --learner-fast 1 --teacher-fork 1 --teacher-slots 8 --memory gru --memory-dim 128 --memory-bptt 32"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S/python
mkdir -p $R
if [ "$1" = run ]; then
  bash ~/isaac-abplus/guard.sh once | head -n 1
  mkdir -p $R/c88-memory/checkpoints && cp $CK $R/c88-memory/checkpoints/from-c67-last.pt
  PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
    --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $R/c88-memory/checkpoints/from-c67-last.pt --name h2m --port 49600 --out $R/c88-memory > $R/c88-memory.log 2>&1 < /dev/null &
  sleep 10
  setsid nohup bash $S/c88_start.sh watch > $R/watch-c88.log 2>&1 < /dev/null &
  sleep 240
  date +%T; grep "memory:" $R/c88-memory.log | cut -c1-160; tail -n 3 $R/c88-memory.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c88-memory.log)"
  bash ~/isaac-abplus/guard.sh once | head -n 1
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c88-memory/evals
  while true; do
    for ck in $(ls $R/c88-memory/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c88-memory/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 4 --seeds 2147490000:32 --name h2me --port 49300 --out "$R/c88-memory/evals/$tag" \
        > "$R/c88-memory/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c88-memory/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c88-memory/evals/$tag/summary.json" "$tag" >> "$R/c88-memory/evals.jsonl"
    done
    [ -f "$R/c88-memory/stop-evals" ] && break
    sleep 60
  done
fi
