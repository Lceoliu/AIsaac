#!/bin/bash
# C68 on host 1 (sandbox fk2 = full workspace incl. Phase B2): C67's final weights resumed (optimiser kept, LR 1e-4),
# run mode + items + charge, with the branch teacher on as recommended by B2 plus the branch archive (the policy
# practises from "took the item" states) and fewer episodes per start state (more floors), 3,000 game hours.
S=/home/eolc/isaac-abplus/fk2
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
PYTHONPATH=. timeout 900 $PY abplus_probe_fast_row.py --groups-file ../abplus/catalog/scaling2_groups.json --floor --items --run --floors 2 \
  --seeds range:2147600500:4 --name c68p --port 30980 --out $R/c68-fastrow > $R/c68-fastrow.log 2>&1 < /dev/null
grep -h SUMMARY $R/c68-fastrow.log | cut -c1-200
mkdir -p $R/c68-branch/checkpoints && cp $R/c67-run/checkpoints/last.pt $R/c68-branch/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 \
  --branch-share 0.3 --branch-kinds item,item_left --branch-seconds 60 --branch-crn 0 --branch-snap-every 0 \
  --branch-noise 0.25 --branch-replicate-max 2 --choice-heads 4 --branch-uncert 0 --choice-adv-coef 0 --branch-archive 1 \
  --resume $R/c68-branch/checkpoints/from-c67-last.pt --name h1r --port 30600 --out $R/c68-branch > $R/c68-branch.log 2>&1 < /dev/null &
sleep 10
sed -e 's#c67-run#c68-branch#g' $R/watch-c67-run.sh > $R/watch-c68-branch.sh
nohup bash $R/watch-c68-branch.sh > $R/watch-c68-branch.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 4 $R/c68-branch.log | cut -c1-300; tail -n 3 $R/c68-branch.log | cut -c1-220; grep -c Traceback $R/c68-branch.log
bash ~/isaac-abplus/guard.sh once | head -n 1
