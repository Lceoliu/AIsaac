#!/bin/bash
# C89 on the 3080 (sandbox mv2 = workspace 10-11 05:2x, repo aa3d615): C83's configuration (C67 final + hurt/death
# teacher + teacher v2 whole-branch distillation, thread searches, fast learner) plus B20 GRU memory
# (--memory gru --memory-dim 128 --memory-bptt 32), 3,000 game hours. The direct test of the thesis' explanation for
# C83/C84 (routes found by the search are not internalised because the policy has no memory): compare with C83 (1.08,
# v2 without memory) and C88 (memory without v2). usage: c89_start.sh run|watch
S=/home/eolc/isaac-abplus/mv2
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.3 --teacher-slots 8 --learner-fast 1 --teacher-v2 1 --teacher-v2-random 60 --memory gru --memory-dim 128 --memory-bptt 32"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $S/python
mkdir -p $R
if [ "$1" = run ]; then
  bash ~/isaac-abplus/guard.sh once | head -n 1
  mkdir -p $R/c89-v2mem/checkpoints && cp $CK $R/c89-v2mem/checkpoints/from-c67-last.pt
  PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
    --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $R/c89-v2mem/checkpoints/from-c67-last.pt --name h1vm --port 36600 --out $R/c89-v2mem > $R/c89-v2mem.log 2>&1 < /dev/null &
  sleep 10
  setsid nohup bash $S/c89_start.sh watch > $R/watch-c89.log 2>&1 < /dev/null &
  sleep 240
  date +%T; grep "memory:" $R/c89-v2mem.log | cut -c1-160; tail -n 3 $R/c89-v2mem.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c89-v2mem.log)"
  bash ~/isaac-abplus/guard.sh once | head -n 1
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c89-v2mem/evals
  while true; do
    for ck in $(ls $R/c89-v2mem/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c89-v2mem/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 4 --seeds 2147490000:32 --name h1vme --port 36300 --out "$R/c89-v2mem/evals/$tag" \
        > "$R/c89-v2mem/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c89-v2mem/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c89-v2mem/evals/$tag/summary.json" "$tag" >> "$R/c89-v2mem/evals.jsonl"
    done
    [ -f "$R/c89-v2mem/stop-evals" ] && break
    sleep 60
  done
fi
