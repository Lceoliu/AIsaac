#!/bin/bash
# C80 on host 1 (sandbox lf = the HPC ws1 tree of 10-10 01:50: B18 fast learner + threaded searches + nullgl): C75's
# setting (C67 final + death teacher) with --learner-fast 1 --teacher-thread 1 --teacher-share 0.5: does the fast
# learner reproduce C75's result (0.90 floors) on the 3080Ti? usage: c80_start.sh smoke|run
S=/home/eolc/isaac-abplus/lf
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.5 --learner-fast 1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.6 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $CK --name lsm --port 37900 --out $R/smoke-lf > $R/smoke-lf.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-lf.log; tail -n 2 $R/smoke-lf.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-lf/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','searches','death_searches','teach_s','update_s','collect_s')})
"
  exit 0
fi
mkdir -p $R/c80-fastlearner/checkpoints && cp $CK $R/c80-fastlearner/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --resume $R/c80-fastlearner/checkpoints/from-c67-last.pt --name h1l --port 37600 --out $R/c80-fastlearner > $R/c80-fastlearner.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/lf#g; s#c67-run#c80-fastlearner#g; s#--port 30900#--port 37300#; s#--name h1re#--name h1le#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c80-fastlearner.sh
nohup bash $R/watch-c80-fastlearner.sh > $R/watch-c80-fastlearner.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c80-fastlearner.log | cut -c1-300; tail -n 3 $R/c80-fastlearner.log | cut -c1-220; grep -c Traceback $R/c80-fastlearner.log
bash ~/isaac-abplus/guard.sh once | head -n 1
