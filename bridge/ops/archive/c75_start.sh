#!/bin/bash
# C75 "death teacher" on host 1 (sandbox dt = the workspace of 10-09 15:00): C67's final weights resumed (LR 1e-4),
# run mode + items + charge + teacher as C68-C73, plus --teacher-death-depths 2,4,8,16,32 (margin 15): the fatal hurt
# of every episode is searched, deeper (SCALING_THESIS.md S5). usage: c75_start.sh smoke|run
S=/home/eolc/isaac-abplus/dt
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.6 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 \
    --resume $CK --name dsm --port 36900 --out $R/smoke-death > $R/smoke-death.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-death.log; tail -n 2 $R/smoke-death.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-death/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('hurts','searches','avoidable','death_searches','death_avoidable','death_s','teach_s')})
"
  exit 0
fi
mkdir -p $R/c75-death/checkpoints && cp $CK $R/c75-death/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 \
  --resume $R/c75-death/checkpoints/from-c67-last.pt --name h1d --port 36600 --out $R/c75-death > $R/c75-death.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/dt#g; s#c67-run#c75-death#g; s#--port 30900#--port 36300#; s#--name h1re#--name h1de#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c75-death.sh
grep -n "port\|name\|OMP" $R/watch-c75-death.sh | head -n 4
nohup bash $R/watch-c75-death.sh > $R/watch-c75-death.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c75-death.log | cut -c1-300; tail -n 3 $R/c75-death.log | cut -c1-220; grep -c Traceback $R/c75-death.log
bash ~/isaac-abplus/guard.sh once | head -n 1
