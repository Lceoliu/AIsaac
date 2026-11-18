#!/bin/bash
# C71 on host 2 (sandbox dp = workspace of 10-08 12:40): C67's final weights resumed (LR 1e-4), run mode + items +
# charge + teacher as C68/C69/C70, no branches / lab / start builds, but --archive-deep 4: archive entries drawn with
# weight 4^(floor-1) and the shallowest floor evicted first (C68 spent 71% of its decisions on floor 1, deaths are on
# floors 2-3). usage: c71_start.sh smoke|run
S=/home/rhythmo/isaac-abplus/dp
R=$S/runs
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --archive-deep 4 \
    --resume /home/rhythmo/isaac-abplus/ev/runs/c67-last.pt --name dsm --port 45900 --out $R/smoke-deep \
    > $R/smoke-deep.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-deep.log; tail -n 2 $R/smoke-deep.log | cut -c1-200
  python3 -c "
import json, collections
c=collections.Counter()
for l in open('$R/smoke-deep/episodes.jsonl'):
    r=json.loads(l); c[r.get('stage0')]+=1
print('episodes by start stage', dict(c))
"
  exit 0
fi
mkdir -p $R/c71-deep/checkpoints && cp /home/rhythmo/isaac-abplus/ev/runs/c67-last.pt $R/c71-deep/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --archive-deep 4 \
  --resume $R/c71-deep/checkpoints/from-c67-last.pt --name h2d --port 45000 --out $R/c71-deep > $R/c71-deep.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/rhythmo/isaac-abplus/ev#/home/rhythmo/isaac-abplus/dp#g; s#c69-lab#c71-deep#g; s#--port 39300#--port 45300#; s#--name h2le#--name h2de#' \
  /home/rhythmo/isaac-abplus/ev/runs/watch-c69-lab.sh > $R/watch-c71-deep.sh
grep -n "port\|name" $R/watch-c71-deep.sh | head -n 3
nohup bash $R/watch-c71-deep.sh > $R/watch-c71-deep.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c71-deep.log | cut -c1-300; tail -n 3 $R/c71-deep.log | cut -c1-220; grep -c Traceback $R/c71-deep.log
bash ~/isaac-abplus/guard.sh once | head -n 1
