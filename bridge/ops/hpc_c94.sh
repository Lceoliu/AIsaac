#!/bin/bash
# C94 "pedestal builds + forced pickup + shared item table" (B23) on HPC160 (sandbox ws7 = repo at the B23 commit,
# 64 workers, after C91): resumes C93's latest checkpoint (memory + item tokens + global token, items line) with the
# start builds spawned as pedestals in the start room, forced walking to pedestals in cleared rooms with probability
# 1 -> 0 over the run, no extra reward, and one shared collectible table. Control = C93 continuing on HPC128.
# usage: hpc_c94.sh smoke [extra args] | run <resume checkpoint> | watch
source /101063/AIsaac/ws7/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25"
if [ "$1" = smoke ]; then
  shift; CK=$1; shift
  tag=${TAG:-smoke-c94}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsm4 --port 49600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "memory:\|item tokens\|global token\|shared\|pedestal\|pickup" $R/$tag.log | head -n 8 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/$tag/progress.csv')))
print([(r['update'], r.get('start_pedestals'), r.get('pickup_force_p'), r.get('pickup_forced_eps'), r.get('pickups_forced'), r.get('pickups_free'), r.get('pickup_imit'), r.get('forced_share'), r['kl'][:6]) for r in rows])"
  grep -c build_pedestals $R/$tag/episodes.jsonl
  exit 0
fi
if [ "$1" = run ]; then
  CK=$2
  mkdir -p $R/c94-pickup/checkpoints && cp "$CK" $R/c94-pickup/checkpoints/from-c93.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 64 --game-hours 12000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c94-pickup/checkpoints/from-c93.pt --name hc94 --port 49000 --out $R/c94-pickup > $R/c94-pickup.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c94-pickup.log | cut -c1-300; grep -i "shared\|pedestal\|pickup" $R/c94-pickup.log | head -n 4 | cut -c1-160; tail -n 3 $R/c94-pickup.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c94-pickup.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c94-pickup/evals
  while true; do
    for ck in $(ls $R/c94-pickup/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c94-pickup/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc94e --port 49300 --out "$R/c94-pickup/evals/$tag" \
        > "$R/c94-pickup/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c94-pickup/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c94-pickup/evals/$tag/summary.json" "$tag" >> "$R/c94-pickup/evals.jsonl"
    done
    [ -f "$R/c94-pickup/stop-evals" ] && break
    sleep 120
  done
fi
