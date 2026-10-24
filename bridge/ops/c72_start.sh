#!/bin/bash
# C72 on host 1 (sandbox sa = the workspace after the stat-aug + observation changes of 10-08): C67's final weights
# resumed (LR 1e-4), run mode + items + charge + teacher as C68-C71, no branches / lab / start builds, but
# --stat-aug-prob 0.5 (default ranges: damage x0.7-2.5, tears x0.7-2, range x0.8-1.5, shot speed x0.8-1.3, speed
# x0.9-1.3, log-uniform) -- the user's "augment the base stats for generalisation". usage: c72_start.sh smoke|run
S=/home/eolc/isaac-abplus/sa
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --stat-aug-prob 1.0 \
    --resume /home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt --name ssm --port 33900 --out $R/smoke-aug \
    > $R/smoke-aug.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-aug.log; tail -n 2 $R/smoke-aug.log | cut -c1-200
  python3 -c "
import json
rs=[json.loads(l) for l in open('$R/smoke-aug/episodes.jsonl')]
print('episodes', len(rs), 'with stats0', sum('stats0' in r for r in rs))
for r in rs[:10]: print(r.get('seed'), r.get('stage0'), r.get('stats0'))
"
  tail -n 1 $R/smoke-aug/progress.csv | tr ',' '\n' | tail -n 3
  exit 0
fi
mkdir -p $R/c72-aug/checkpoints && cp /home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt $R/c72-aug/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --stat-aug-prob 0.5 \
  --resume $R/c72-aug/checkpoints/from-c67-last.pt --name h1s --port 33600 --out $R/c72-aug > $R/c72-aug.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/sa#g; s#c67-run#c72-aug#g; s#--port 30900#--port 33300#; s#--name h1re#--name h1se#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c72-aug.sh
grep -n "port\|name\|OMP" $R/watch-c72-aug.sh | head -n 4
nohup bash $R/watch-c72-aug.sh > $R/watch-c72-aug.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c72-aug.log | cut -c1-300; tail -n 3 $R/c72-aug.log | cut -c1-220; grep -c Traceback $R/c72-aug.log
bash ~/isaac-abplus/guard.sh once | head -n 1
