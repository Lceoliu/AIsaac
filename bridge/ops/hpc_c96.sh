#!/bin/bash
# C96 "B30 temporal attention" = C95 continued with the temporal transformer (docs/TEMPORAL_ATTENTION.md: 256x4 over the
# last 256 steps + room tokens, readout/feedback/value_global gates zero at the start = bit-identical to C95 at the
# start) + the B29 actor speed flags, on HPC160 (sandbox ws9 = repo at the B30 commit, 64 workers, after C95):
# resumes C95's final checkpoint, forced pickup off (C95's schedule ended at 0), everything else = C95.
# Purpose: speed (target >= C95's ~1,100x) and whether the items line (plateau 1.45-1.55) moves with temporal context;
# watch the gates t_gate_read / t_gate_feed / t_gate_vg. usage: hpc_c96.sh smoke <checkpoint> [extra] | run <ckpt> | watch
source /101063/AIsaac/ws9/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory gru --memory-dim 128 --memory-bptt 32 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 0:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --actor-compile max-autotune-no-cudagraphs --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
# (smoke: the compile flags are dropped below: a first compile takes ~18 min; the smoke checks the model, the run checks speed)
if [ "$1" = smoke ]; then
  shift; CK=$1; shift
  tag=${TAG:-smoke-c96}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsm6 --port 49600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
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
  mkdir -p $R/c96-temporal/checkpoints && cp "$CK" $R/c96-temporal/checkpoints/from-c95.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON --workers 64 --game-hours 12000 --checkpoint-every 1000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c96-temporal/checkpoints/from-c95.pt --name hc96 --port 49000 --out $R/c96-temporal > $R/c96-temporal.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c96-temporal.log | cut -c1-300; grep -i "shared\|pedestal\|pickup" $R/c96-temporal.log | head -n 4 | cut -c1-160; tail -n 3 $R/c96-temporal.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c96-temporal.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c96-temporal/evals
  while true; do
    for ck in $(ls $R/c96-temporal/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c96-temporal/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc96e --port 49300 --out "$R/c96-temporal/evals/$tag" \
        > "$R/c96-temporal/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c96-temporal/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c96-temporal/evals/$tag/summary.json" "$tag" >> "$R/c96-temporal/evals.jsonl"
    done
    [ -f "$R/c96-temporal/stop-evals" ] && break
    sleep 120
  done
fi
