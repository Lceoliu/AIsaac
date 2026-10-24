#!/bin/bash
# C76 "gamma only" (host 2 sandbox hz): C67 final resumed, run mode as C68-C73, --gamma 0.999 WITHOUT the heal /
# resource terms: isolates the horizon from the reward terms of C73 (whose continuation C73b drifted to taking items
# and timing out). usage: c76_start.sh smoke|run [host]
HOSTN=${2:-2}
if [ "$HOSTN" = 1 ]; then S=/home/eolc/isaac-abplus/hz; PY=/home/eolc/isaac-rl/.venv/bin/python; CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt; WATCH=/home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh; SUB='s#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/hz#g; s#c67-run#c76-gamma#g; s#--port 30900#--port 34400#; s#--name h1re#--name h1ge#'
else S=/home/rhythmo/isaac-abplus/hz; PY=/home/rhythmo/isaac-abplus/.venv/bin/python; CK=/home/rhythmo/isaac-abplus/ev/runs/c67-last.pt; WATCH=/home/rhythmo/isaac-abplus/ev/runs/watch-c69-lab.sh; SUB='s#/home/rhythmo/isaac-abplus/ev#/home/rhythmo/isaac-abplus/hz#g; s#c69-lab#c76-gamma#g; s#--port 39300#--port 46400#; s#--name h2le#--name h2ge#'; fi
R=$S/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --gamma 0.999 \
    --resume $CK --name hsm --port 34900 --out $R/smoke-gamma > $R/smoke-gamma.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-gamma.log; tail -n 2 $R/smoke-gamma.log | cut -c1-200
  python3 -c "
import json
rs=[json.loads(l) for l in open('$R/smoke-gamma/episodes.jsonl')]
print('episodes', len(rs), 'healed', sum(r.get('healed',0) for r in rs), 'gathered', sum(r.get('gathered',0) for r in rs))
for r in rs[:6]: print(r.get('seed'), r.get('done'), r.get('ret'), r.get('healed'), r.get('gathered'))
"
  head -n 1 $R/smoke-gamma/progress.csv | tr ',' '\n' | grep -n "healed\|gathered"
  exit 0
fi
mkdir -p $R/c76-gamma/checkpoints && cp $CK $R/c76-gamma/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --gamma 0.999 \
  --resume $R/c76-gamma/checkpoints/from-c67-last.pt --name hzg --port 34700 --out $R/c76-gamma > $R/c76-gamma.log 2>&1 < /dev/null &
sleep 10
sed -e "$SUB" -e 's#^cd #export OMP_NUM_THREADS=4\ncd #' $WATCH > $R/watch-c76-gamma.sh
grep -n "port\|name\|OMP" $R/watch-c76-gamma.sh | head -n 4
nohup bash $R/watch-c76-gamma.sh > $R/watch-c76-gamma.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c76-gamma.log | cut -c1-300; tail -n 3 $R/c76-gamma.log | cut -c1-220; grep -c Traceback $R/c76-gamma.log
bash ~/isaac-abplus/guard.sh once | head -n 1
