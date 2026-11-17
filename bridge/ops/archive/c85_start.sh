#!/bin/bash
# C85 on the 3090 (sandbox plr = workspace of 10-10 18:00): C67 final + death teacher (forked searches, fast learner) +
# --archive-plr 1 (SCALING_THESIS.md S5b: archive starts drawn by the rank of |G - V0|, the regret proxy; C78 in the
# plan). The control is C80/C75 (same without PLR). usage: c85_start.sh smoke|run

S=/home/rhythmo/isaac-abplus/plr
R=$S/runs
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
CK=/home/rhythmo/isaac-abplus/ev/runs/c67-last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-share 0.5 --learner-fast 1 --teacher-fork 1 --teacher-slots 8 --archive-plr 1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 900 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 4 \
    --game-hours 0.6 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --resume $CK --name psm --port 48900 --out $R/smoke-plr > $R/smoke-plr.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-plr.log; tail -n 2 $R/smoke-plr.log | cut -c1-200
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/smoke-plr/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','searches','death_searches','teach_s','update_s','collect_s')})
"
  exit 0
fi
mkdir -p $R/c85-plr/checkpoints && cp $CK $R/c85-plr/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --resume $R/c85-plr/checkpoints/from-c67-last.pt --name h2p --port 48600 --out $R/c85-plr > $R/c85-plr.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/rhythmo/isaac-abplus/ev#/home/rhythmo/isaac-abplus/plr#g; s#c69-lab#c85-plr#g; s#--port 39300#--port 48300#; s#--name h2le#--name h2pe#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/rhythmo/isaac-abplus/ev/runs/watch-c69-lab.sh > $R/watch-c85-plr.sh
nohup bash $R/watch-c85-plr.sh > $R/watch-c85-plr.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c85-plr.log | cut -c1-300; tail -n 3 $R/c85-plr.log | cut -c1-220; grep -c Traceback $R/c85-plr.log
bash ~/isaac-abplus/guard.sh once | head -n 1
