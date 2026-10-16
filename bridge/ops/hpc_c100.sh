#!/bin/bash
# C100 "C99 scaled up" (user 10-16) on HPC128 (sandbox ws13 = repo at the C99 commit or later, 48 workers, GPU shared
# with a foreign process): the C99 recipe from scratch with a bigger model: spatial transformer width 192 -> 256,
# 4 -> 6 layers, 4 -> 8 heads (head dim 32 kept); temporal transformer 256 x 4 -> 256 x 6, 8 heads. ~10.7M parameters
# vs C99's ~7M. Everything else = C99 (no GRU, no gates, obs v2, counterfactual teacher, death teacher, B29 flags,
# 25,000 game hours). Compare the 128-seed curves at equal game hours. usage: hpc_c100.sh smoke | run | resume <ckpt> | watch
source /101063/AIsaac/ws13/hpc_env.sh
R=$ABP_HOME/runs
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
  --width 256 --layers 6 --heads 8 --temporal 1 --temporal-width 256 --temporal-layers 6 --temporal-heads 8 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --obs-v2 1 \
  --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 \
  --cf-distill-steps 8 --cf-random 1 --cf-advantage 0"
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-c100}
  PYTHONPATH=. timeout 1500 $PY train_tok.py $COMMON --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
    --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 "$@" \
    --name hsm10 --port 48600 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "temporal\|memory:\|pedestal\|pickup" $R/$tag.log | head -n 6 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  exit 0
fi
if [ "$1" = run ]; then
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST --workers 48 --game-hours 25000 --checkpoint-every 1000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
    --name hc100 --port 48000 --out $R/c100-scale > $R/c100-scale.log 2>&1 < /dev/null &
  sleep 300
  date +%T; head -n 3 $R/c100-scale.log | cut -c1-300; tail -n 3 $R/c100-scale.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c100-scale.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = resume ]; then   # (B36 restart: same flags, --resume the given checkpoint into the same --out)
  CK=$2
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST --workers 48 --game-hours 25000 --checkpoint-every 1000     --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1     --resume "$CK" --name hc100 --port 48000 --out $R/c100-scale >> $R/c100-scale.log 2>&1 < /dev/null &
  sleep 300
  date +%T; tail -n 3 $R/c100-scale.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $R/c100-scale.log)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then
  export OMP_NUM_THREADS=4
  mkdir -p $R/c100-scale/evals
  while true; do
    for ck in $(ls $R/c100-scale/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$R/c100-scale/evals/$tag" ] && continue
      PYTHONPATH=. timeout 5000 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
        --mode run --items --workers 8 --seeds 2147490000:128 --name hc100e --port 48300 --out "$R/c100-scale/evals/$tag" \
        > "$R/c100-scale/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$R/c100-scale/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$R/c100-scale/evals/$tag/summary.json" "$tag" >> "$R/c100-scale/evals.jsonl"
    done
    [ -f "$R/c100-scale/stop-evals" ] && break
    sleep 120
  done
fi
