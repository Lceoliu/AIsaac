#!/bin/bash
# C98 "temporal attention WITHOUT the GRU" (B32 ablation, user 10-15) on HPC128 (sandbox ws11 = repo at the B32 commit,
# 48 workers, GPU shared with a foreign process): resumes C96's latest checkpoint with the GRU tensors dropped
# (--memory off: mem_gru / mem_out / gtok.mem_in), everything else = C96 (temporal 256x4, W 256, rooms 128, B29 actor
# flags). Control = C96 itself over the same game hours. Watch: floors at equal hours, update_s (no 32-step GRU loop),
# the gates (does g1 open once the GRU is gone?). usage: hpc_c98.sh smoke <ckpt> [extra] | run <ckpt> | watch
source /101063/AIsaac/ws11/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 --memory off \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 0:0 --reward heal=0.5,resource=0.25 --item-tokens-fused 1 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256"
# B29 actor speed flags (run only: the smoke checks the model, the run checks speed)
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
if [ "$1" = smoke ]; then
  shift; CK=$1; shift
  tag=${TAG:-smoke-c98}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --resume $CK --name hsm8 --port 48600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "B32\|temporal\|dropped" $R/$tag.log | head -n 6 | cut -c1-220
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  python3 -c "
import csv
rows=list(csv.DictReader(open('$R/$tag/progress.csv')))
print([(r['update'], r['kl'][:6], r.get('update_s','')[:5], r.get('infer_ms','')[:5], r.get('t_gate_read'), r.get('t_gate_feed'), r.get('t_gate_vg'), r.get('pickups_free')) for r in rows])"
  exit 0
fi
if [ "$1" = run ]; then
  CK=$2
  mkdir -p $R/c98-nogru/checkpoints && cp "$CK" $R/c98-nogru/checkpoints/from-c96.pt
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST --workers 48 --game-hours 3000 --checkpoint-every 600 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --resume $R/c98-nogru/checkpoints/from-c96.pt --name hc98 --port 48000 --out $R/c98-nogru > $R/c98-nogru.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c98-nogru.log | cut -c1-300; grep -i "B32\|dropped" $R/c98-nogru.log | head -n 3 | cut -c1-220; tail -n 3 $R/c98-nogru.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c98-nogru.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c98-nogru/evals
  while true; do
    for ck in $(ls $R/c98-nogru/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c98-nogru/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc98e --port 48300 --out "$R/c98-nogru/evals/$tag" \
        > "$R/c98-nogru/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c98-nogru/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c98-nogru/evals/$tag/summary.json" "$tag" >> "$R/c98-nogru/evals.jsonl"
    done
    [ -f "$R/c98-nogru/stop-evals" ] && break
    sleep 120
  done
fi
