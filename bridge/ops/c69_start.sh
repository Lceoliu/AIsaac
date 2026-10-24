#!/bin/bash
# C69 on host 2 (sandbox ev = full workspace incl. Phase C): C67's final weights resumed with the build lab on
# (lab episodes = practice with transplanted items, 30% of worker time, cost-balanced panel), branches off, otherwise
# as C68 (run mode, items, 16 workers, LR 1e-4, episodes per state 8, 3,000 game hours). The pair C68 (branches) vs
# C69 (lab) from the same weights.
S=/home/rhythmo/isaac-abplus/ev
R=$S/runs
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ~/isaac-abplus/guard.sh once | head -n 1
cd $S/python
PYTHONPATH=. timeout 900 $PY abplus_probe_fast_row.py --groups-file ../abplus/catalog/scaling2_groups.json --floor --items --run --floors 2 \
  --seeds range:2147600600:4 --name c69p --port 39980 --out $R/c69-fastrow > $R/c69-fastrow.log 2>&1 < /dev/null
grep -h SUMMARY $R/c69-fastrow.log | cut -c1-200
mkdir -p $R/c69-lab/checkpoints && cp $R/c67-last.pt $R/c69-lab/checkpoints/from-c67-last.pt
PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 16 \
  --game-hours 3000 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --lab-share 0.3 --lab-costs 1 \
  --resume $R/c69-lab/checkpoints/from-c67-last.pt --name h2l --port 39000 --out $R/c69-lab > $R/c69-lab.log 2>&1 < /dev/null &
sleep 10
cat > $R/watch-c69-lab.sh <<EOF
#!/bin/bash
cd $S/python
mkdir -p $R/c69-lab/evals
while true; do
  for ck in \$(ls $R/c69-lab/checkpoints/update-*.pt 2>/dev/null | sort); do
    tag=\$(basename "\$ck" .pt)
    [ -e "$R/c69-lab/evals/\$tag" ] && continue
    PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "\$ck" --groups-file ../abplus/catalog/scaling2_groups.json \\
      --mode run --items --workers 4 --seeds 2147490000:32 --name h2le --port 39300 --out "$R/c69-lab/evals/\$tag" \\
      > "$R/c69-lab/evals/\$tag.log" 2>&1 < /dev/null
    [ -f "$R/c69-lab/evals/\$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c69-lab/evals/\$tag/summary.json" "\$tag" >> "$R/c69-lab/evals.jsonl"
  done
  [ -f "$R/c69-lab/stop-evals" ] && break
  sleep 60
done
EOF
nohup bash $R/watch-c69-lab.sh > $R/watch-c69-lab.log 2>&1 < /dev/null &
sleep 240
date +%T; head -n 3 $R/c69-lab.log | cut -c1-300; tail -n 3 $R/c69-lab.log | cut -c1-220; grep -c Traceback $R/c69-lab.log
bash ~/isaac-abplus/guard.sh once | head -n 1
