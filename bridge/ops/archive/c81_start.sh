#!/bin/bash
# C81 on host 2 (sandbox lf = the HPC ws1 tree): C67 final + death teacher + gamma 0.999 (C75 + C76 combined), with the
# B18 fast learner and threaded searches. usage: c81_start.sh smoke|run

S=/home/rhythmo/isaac-abplus/lf
R=$S/runs
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
CK=/home/rhythmo/isaac-abplus/ev/runs/c67-last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.5 --learner-fast 1 --gamma 0.999"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.6 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $CK --name l2sm --port 47900 --out $R/smoke-lf2 > $R/smoke-lf2.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-lf2.log; tail -n 2 $R/smoke-lf2.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-lf2/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','searches','death_searches','teach_s','update_s','collect_s')})
"
  exit 0
fi
mkdir -p $R/c81-deathgamma/checkpoints && cp $CK $R/c81-deathgamma/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --resume $R/c81-deathgamma/checkpoints/from-c67-last.pt --name h2g --port 47600 --out $R/c81-deathgamma > $R/c81-deathgamma.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/rhythmo/isaac-abplus/ev#/home/rhythmo/isaac-abplus/lf#g; s#c69-lab#c81-deathgamma#g; s#--port 39300#--port 47300#; s#--name h2le#--name h2ge#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/rhythmo/isaac-abplus/ev/runs/watch-c69-lab.sh > $R/watch-c81-deathgamma.sh
nohup bash $R/watch-c81-deathgamma.sh > $R/watch-c81-deathgamma.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c81-deathgamma.log | cut -c1-300; tail -n 3 $R/c81-deathgamma.log | cut -c1-220; grep -c Traceback $R/c81-deathgamma.log
bash ~/isaac-abplus/guard.sh once | head -n 1
