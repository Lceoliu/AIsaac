#!/bin/bash
# C73 "horizon" on host 2 (sandbox hz = the workspace of 10-08 17:30): C67's final weights resumed (LR 1e-4), run mode +
# items + charge + teacher as C68-C72, but --gamma 0.999 (value horizon ~1000 decisions = 2.2 game minutes instead of
# 27 s) and the heal / resource reward terms (SCALING_THESIS.md S3). usage: c73_start.sh smoke|run  [host: 1|2]
HOSTN=${2:-2}
if [ "$HOSTN" = 1 ]; then S=/home/eolc/isaac-abplus/hz; PY=/home/eolc/isaac-rl/.venv/bin/python; CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt; WATCH=/home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh; SUB='s#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/hz#g; s#c67-run#c73-horizon#g; s#--port 30900#--port 34300#; s#--name h1re#--name h1he#'
else S=/home/rhythmo/isaac-abplus/hz; PY=/home/rhythmo/isaac-abplus/.venv/bin/python; CK=/home/rhythmo/isaac-abplus/ev/runs/c67-last.pt; WATCH=/home/rhythmo/isaac-abplus/ev/runs/watch-c69-lab.sh; SUB='s#/home/rhythmo/isaac-abplus/ev#/home/rhythmo/isaac-abplus/hz#g; s#c69-lab#c73-horizon#g; s#--port 39300#--port 46300#; s#--name h2le#--name h2he#'; fi
R=$S/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.4 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --gamma 0.999 --reward heal=0.5,resource=0.25 \
    --resume $CK --name hsm --port 34900 --out $R/smoke-horizon > $R/smoke-horizon.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-horizon.log; tail -n 2 $R/smoke-horizon.log | cut -c1-200
  python3 -c "
import json
rs=[json.loads(l) for l in open('$R/smoke-horizon/episodes.jsonl')]
print('episodes', len(rs), 'healed', sum(r.get('healed',0) for r in rs), 'gathered', sum(r.get('gathered',0) for r in rs))
for r in rs[:6]: print(r.get('seed'), r.get('done'), r.get('ret'), r.get('healed'), r.get('gathered'))
"
  head -n 1 $R/smoke-horizon/progress.csv | tr ',' '\n' | grep -n "healed\|gathered"
  exit 0
fi
mkdir -p $R/c73-horizon/checkpoints && cp $CK $R/c73-horizon/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --gamma 0.999 --reward heal=0.5,resource=0.25 \
  --resume $R/c73-horizon/checkpoints/from-c67-last.pt --name hzr --port 34600 --out $R/c73-horizon > $R/c73-horizon.log 2>&1 < /dev/null &
sleep 10
sed -e "$SUB" -e 's#^cd #export OMP_NUM_THREADS=4\ncd #' $WATCH > $R/watch-c73-horizon.sh
grep -n "port\|name\|OMP" $R/watch-c73-horizon.sh | head -n 4
nohup bash $R/watch-c73-horizon.sh > $R/watch-c73-horizon.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c73-horizon.log | cut -c1-300; tail -n 3 $R/c73-horizon.log | cut -c1-220; grep -c Traceback $R/c73-horizon.log
bash ~/isaac-abplus/guard.sh once | head -n 1
