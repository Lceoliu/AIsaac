#!/bin/bash
# C95 "B24 fused attention" = C94 continued at the fused layout on HPC160 (sandbox ws8 = repo at the B24 commit, 64 workers,
# after C94): resumes C94's final checkpoint (memory + item tokens + global token + shared table, B23 pedestal builds)
# converted to --item-tokens-fused 1 --item-slots 8 (approximate conversion, gates kept), forced pickup continuing C94 schedule (p 0.6 -> 0; C94
# schedule ended at 0), everything else = C94. Purpose: the speed-up at 64 workers (C94 ~600x) and whether the items line
# keeps improving at the fused layout. usage: hpc_c95.sh smoke <checkpoint> [extra args] | run <resume checkpoint> | watch
source /101063/AIsaac/ws8/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 0.6:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8"
if [ "$1" = smoke ]; then
  shift; CK=$1; shift
  tag=${TAG:-smoke-c95}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsm5 --port 49600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
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
  mkdir -p $R/c95-fused/checkpoints && cp "$CK" $R/c95-fused/checkpoints/from-c94-u58800.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 64 --game-hours 12000 --checkpoint-every 1000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c95-fused/checkpoints/from-c94-u58800.pt --name hc95 --port 49000 --out $R/c95-fused > $R/c95-fused.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c95-fused.log | cut -c1-300; grep -i "shared\|pedestal\|pickup" $R/c95-fused.log | head -n 4 | cut -c1-160; tail -n 3 $R/c95-fused.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c95-fused.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c95-fused/evals
  while true; do
    for ck in $(ls $R/c95-fused/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c95-fused/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc95e --port 49300 --out "$R/c95-fused/evals/$tag" \
        > "$R/c95-fused/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c95-fused/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c95-fused/evals/$tag/summary.json" "$tag" >> "$R/c95-fused/evals.jsonl"
    done
    [ -f "$R/c95-fused/stop-evals" ] && break
    sleep 120
  done
fi
