#!/bin/bash
# C82 on host 1: C80 (C67 + death teacher + fast learner + threaded searches) continued for 3,000 more game hours
# (6,000 in all at the hosts' scale). usage: c82_start.sh smoke|run

S=/home/eolc/isaac-abplus/lf
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/lf/runs/c80-fastlearner/checkpoints/last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.5 --learner-fast 1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.6 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $CK --name lsm --port 38900 --out $R/smoke-lf3 > $R/smoke-lf3.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-lf3.log; tail -n 2 $R/smoke-lf3.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-lf3/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','searches','death_searches','teach_s','update_s','collect_s')})
"
  exit 0
fi
mkdir -p $R/c82-cont/checkpoints && cp $CK $R/c82-cont/checkpoints/from-c80-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --resume $R/c82-cont/checkpoints/from-c80-last.pt --name h1k --port 38600 --out $R/c82-cont > $R/c82-cont.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/lf#g; s#c67-run#c82-cont#g; s#--port 30900#--port 38300#; s#--name h1re#--name h1ke#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c82-cont.sh
nohup bash $R/watch-c82-cont.sh > $R/watch-c82-cont.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c82-cont.log | cut -c1-300; tail -n 3 $R/c82-cont.log | cut -c1-220; grep -c Traceback $R/c82-cont.log
bash ~/isaac-abplus/guard.sh once | head -n 1
