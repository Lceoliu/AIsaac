#!/bin/bash
# C86 "characters, second try" on the 3080 (sandbox ch2 = repo abplus-training): C74 lost 0.27 floors on Isaac with 80%
# other characters and --init; this time Isaac keeps 60% (weights 0:18 + 12 characters x 1), with the death teacher,
# forked searches and the fast learner (C75/C80 setting); --init C67 (the character table must start: resume keeps
# pchar off). Compare Isaac's 128-seed floors with C80 (0.77) / C75 (0.90). usage: c86_start.sh smoke|run
S=/home/eolc/isaac-abplus/ch2
R=$S/runs
PY=/home/eolc/isaac-rl/.venv/bin/python
CK=/home/eolc/isaac-abplus/fk2/runs/c67-run/checkpoints/last.pt
EXTRA="--teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 --teacher-share 0.3 --teacher-slots 8 --learner-fast 1 --characters 0:18,1,2,3,4,5,6,7,8,9,13,14,15"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
mkdir -p $R
if [ "$1" = smoke ]; then
  PYTHONPATH=. timeout 1500 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 \
    --game-hours 1.5 --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
    --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
    --init $CK --name c2sm --port 49900 --out $R/smoke-ch2 > $R/smoke-ch2.log 2>&1 < /dev/null
  grep -c Traceback $R/smoke-ch2.log; tail -n 2 $R/smoke-ch2.log | cut -c1-200
  python3 -c "
import csv, json, collections
rows=list(csv.DictReader(open('$R/smoke-ch2/progress.csv')))
r=rows[-1]; print('updates', len(rows), {k: r.get(k) for k in ('errors','searches','death_searches','char_starts') if k in r})
rs=[json.loads(l) for l in open('$R/smoke-ch2/episodes.jsonl')]; print('episodes', len(rs), 'by character', dict(collections.Counter(x.get('char0') for x in rs)))
"
  exit 0
fi
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 $EXTRA \
  --init $CK --name h1c2 --port 49600 --out $R/c86-chars2 > $R/c86-chars2.log 2>&1 < /dev/null &
sleep 10
sed -e 's#/home/eolc/isaac-abplus/fk2#/home/eolc/isaac-abplus/ch2#g; s#c67-run#c86-chars2#g; s#--port 30900#--port 49300#; s#--name h1re#--name h1c2e#; s#^cd #export OMP_NUM_THREADS=4\ncd #' \
  /home/eolc/isaac-abplus/fk2/runs/watch-c67-run.sh > $R/watch-c86-chars2.sh
nohup bash $R/watch-c86-chars2.sh > $R/watch-c86-chars2.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c86-chars2.log | cut -c1-300; tail -n 3 $R/c86-chars2.log | cut -c1-220; grep -c Traceback $R/c86-chars2.log
bash ~/isaac-abplus/guard.sh once | head -n 1
