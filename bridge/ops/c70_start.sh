#!/bin/bash
# C70 on host 1 (sandbox bld = workspace of 10-08 11:30 incl. start builds): C67's final weights resumed (LR 1e-4),
# run mode + items + charge + teacher as C68/C69, no branches, no lab, but --start-build-prob 0.5: half the episodes
# start with a random build of 1..3 eligible collectibles transplanted into the clone (build diversity without labels).
# usage: c70_start.sh smoke|run
S=/home/eolc/isaac-abplus/bld
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --start-build-prob 1.0 --start-build-max 3 \
    --resume /home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt --name bsm --port 31900 --out $R/smoke-build \
    > $R/smoke-build.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-build.log; tail -n 3 $R/smoke-build.log | cut -c1-200
  python3 -c "
import json
n=b=0; its=0
for l in open('$R/smoke-build/episodes.jsonl'):
    r=json.loads(l); n+=1; b+=bool(r.get('build0')); its+=len(r.get('build0') or [])
print('episodes',n,'with build0',b,'mean build size',its/max(b,1))
print([r.get('build0') for r in map(json.loads,open('$R/smoke-build/episodes.jsonl'))][:8])
"
  exit 0
fi
mkdir -p $R/c70-build/checkpoints && cp /home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt $R/c70-build/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --start-build-prob 0.5 --start-build-max 3 \
  --resume $R/c70-build/checkpoints/from-c67-last.pt --name h1b --port 31600 --out $R/c70-build > $R/c70-build.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/bld#g; s#c67-run#c70-build#g; s#--port 30900#--port 31300#; s#--name h1re#--name h1be#' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c70-build.sh
grep -n "port\|name" $R/watch-c70-build.sh | head -n 3
nohup bash $R/watch-c70-build.sh > $R/watch-c70-build.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c70-build.log | cut -c1-300; tail -n 3 $R/c70-build.log | cut -c1-220; grep -c Traceback $R/c70-build.log
bash ~/isaac-abplus/guard.sh once | head -n 1
