#!/bin/bash
# C77 "ten game years" on the HPC node (sandbox ws1, 160-core cgroup quota, H20): C67's final weights resumed (LR 1e-4),
# run mode + items + charge + teacher + death teacher (C75's setting), 64 workers (96 saturate the quota), rollout
# 32768 / minibatch 4096 / micro 4096, the B18 fast learner (no CPU pinning: it starved the actor thread), searches in a
# worker thread with a 15% share and 16 slots over all workers (a 50% share without slots stalled the collection:
# GIL contention in the workers, fork storms), synchronous (overlap makes the actor's inference 5-10x slower on the
# shared GPU), 87,600 game hours. usage: hpc_c77.sh run|watch
source /101063/AIsaac/ws1/hpc_env.sh
R=$ABP_HOME/runs
if [ "$1" = run ]; then
  mkdir -p $R/c77-tenyears/checkpoints && cp $R/c77-prev-last.pt $R/c77-tenyears/checkpoints/from-prev-last.pt
  PYTHONPATH=. nohup $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 64 \
    --game-hours 87600 --epochs 3 --checkpoint-every 600 --teacher --teacher-queue 12 --archive-size 12 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 \
    --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
    --resume $R/c77-tenyears/checkpoints/from-prev-last.pt --name hc77 --port 41000 --out $R/c77-tenyears \
    > $R/c77-tenyears.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c77-tenyears.log | cut -c1-300; tail -n 3 $R/c77-tenyears.log | cut -c1-220; grep -c Traceback $R/c77-tenyears.log
  uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c77-tenyears/evals
  while true; do
    for ck in $(ls $R/c77-tenyears/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c77-tenyears/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc77e --port 41300 --out "$R/c77-tenyears/evals/$tag" \
        > "$R/c77-tenyears/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c77-tenyears/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c77-tenyears/evals/$tag/summary.json" "$tag" >> "$R/c77-tenyears/evals.jsonl"
    done
    [ -f "$R/c77-tenyears/stop-evals" ] && break
    sleep 120
  done
fi
