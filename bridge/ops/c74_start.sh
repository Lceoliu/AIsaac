#!/bin/bash
# C74 "characters" on host 1 (sandbox ch1 = the workspace of 10-08 23:30, bridge abp-0.2.17): C67's final weights via
# --init (the character table starts at zero = the same outputs; the optimiser restarts, LR 1e-4 constant), run mode +
# items + charge + teacher as C68-C73, floor starts as a random character (Isaac weight 3, the other 12 supported
# characters weight 1 each; SCALING_THESIS.md S4). usage: c74_start.sh smoke|run
S=/home/eolc/isaac-abplus/ch1
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
CHARS="0:3,1,2,3,4,5,6,7,8,9,13,14,15"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --characters "$CHARS" \
    --init $CK --name csm --port 35900 --out $R/smoke-chars > $R/smoke-chars.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-chars.log; head -n 3 $R/smoke-chars.log | cut -c1-200; tail -n 2 $R/smoke-chars.log | cut -c1-200
  python3 -c "
import json, collections
rs=[json.loads(l) for l in open('$R/smoke-chars/episodes.jsonl')]
c=collections.Counter(r.get('char0') for r in rs)
print('episodes', len(rs), 'by character', dict(c)); print(list(rs[0].keys()))
"
  exit 0
fi
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --characters "$CHARS" \
  --init $CK --name h1c --port 35600 --out $R/c74-chars > $R/c74-chars.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/ch1#g; s#c67-run#c74-chars#g; s#--port 30900#--port 35300#; s#--name h1re#--name h1ce#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c74-chars.sh
grep -n "port\|name\|OMP" $R/watch-c74-chars.sh | head -n 4
nohup bash $R/watch-c74-chars.sh > $R/watch-c74-chars.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 4 $R/c74-chars.log | cut -c1-300; tail -n 3 $R/c74-chars.log | cut -c1-220; grep -c Traceback $R/c74-chars.log
bash ~/isaac-abplus/guard.sh once | head -n 1
