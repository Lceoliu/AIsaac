#!/bin/bash
# C83 "teacher v2" on host 1 (sandbox tv = the workspace of 10-10 13:10 with B19): C67's final weights resumed (LR 1e-4),
# run mode + items + charge, hurt teacher + death teacher + teacher v2 (whole-branch distillation, SCALING_THESIS.md
# S5b), searches in the worker thread, B18 fast learner. The acceptance test: v2_prior rises and v2_improving_share
# falls over training; then the 128-seed run eval against C75 (0.90). usage: c83_start.sh smoke|run
S=/home/eolc/isaac-abplus/tv
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.3 --teacher-slots 8 --learner-fast 1 --teacher-v2 1 --teacher-v2-random 60"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 1500 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 \
    --game-hours 1.5 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $CK --name tsm --port 39900 --out $R/smoke-tv > $R/smoke-tv.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-tv.log; tail -n 2 $R/smoke-tv.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-tv/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','collect_s','update_s','searches','death_searches','v2_points','v2_improving_share','v2_records','v2_held','v2_step','v2_loss_pi','v2_loss_vf','v2_prior','v2_gain0_mean','v2_spread_mean','v2_lane_calls','v2_cpu_per_point','v2_dropped') if k in r})
"
  exit 0
fi
mkdir -p $R/c83-teacher2/checkpoints && cp $CK $R/c83-teacher2/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --resume $R/c83-teacher2/checkpoints/from-c67-last.pt --name h1t --port 39600 --out $R/c83-teacher2 > $R/c83-teacher2.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/tv#g; s#c67-run#c83-teacher2#g; s#--port 30900#--port 39300#; s#--name h1re#--name h1te#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c83-teacher2.sh
nohup bash $R/watch-c83-teacher2.sh > $R/watch-c83-teacher2.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c83-teacher2.log | cut -c1-300; tail -n 3 $R/c83-teacher2.log | cut -c1-220; grep -c Traceback $R/c83-teacher2.log
bash ~/isaac-abplus/guard.sh once | head -n 1
