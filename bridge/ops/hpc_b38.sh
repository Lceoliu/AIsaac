#!/bin/bash
# B38 (train_tok --learner-device) tests on HPC160, sandbox ws14 (mk_ws14.sh), GPUs 1 and 2 only (GPU 0 runs C99:
# CUDA_VISIBLE_DEVICES=1,2, so cuda:0 = GPU 1 is the actor's, cuda:1 = GPU 2 the learner's). One run at a time, each
# ends by itself (--game-hours) or by SIGINT at its time limit (the trainer saves last.pt and closes its sampler);
# afterwards the game processes whose ABP_WATCH_PID names a dead pid are ended (as hpc_guard.sh does).
# usage: hpc_b38.sh smoke [a b c]   8 workers, 1.5 game hours, the C99 recipe from scratch (hpc_c99.sh smoke)
#        hpc_b38.sh tput  [a b c]   64 workers, ~60 updates (73 game hours), the C99 run flags + FAST, resumed from
#                                   C99's segment1-final.pt into ws14/runs/b38-tput-<v> (never pre-created)
# variants: a one GPU, synchronous (as C99); b two GPUs + --overlap --decoupled 1; c two GPUs, synchronous;
#           d one GPU + --overlap --decoupled 1 (reference: the B11 / B13 contention)
source /101063/AIsaac/ws14/hpc_env.sh
export CUDA_VISIBLE_DEVICES=1,2
R=$ABP_HOME/runs
CK=/101063/AIsaac/ws9/runs/c99-scratch/checkpoints/segment1-final.pt
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --obs-v2 1 \
  --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 \
  --cf-distill-steps 8 --cf-random 1 --cf-advantage 0"
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
variant() {
  case $1 in
    a) echo "" ;;
    b) echo "--learner-device cuda:1 --overlap --decoupled 1" ;;
    c) echo "--learner-device cuda:1" ;;
    d) echo "--overlap --decoupled 1" ;;
  esac
}
gpumon() {   # GPUs 1 and 2 (physical numbering) and the load every 10 s
  while true; do
    echo "$(date +%T) $(nvidia-smi -i 1,2 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')load $(cut -d' ' -f1 /proc/loadavg)"
    sleep 10
  done
}
orphans() {
  sleep 30
  n=0
  for p in $(pgrep isaac.x64); do
    w=$( (tr '\000' '\012' < /proc/$p/environ) 2>/dev/null | sed -n 's/^ABP_WATCH_PID=//p')
    if [ -n "$w" ] && ! kill -0 "$w" 2>/dev/null; then kill -KILL $p 2>/dev/null; n=$((n+1)); fi
  done
  echo "$(date +%T) orphan games ended: $n; games left $(pgrep -c isaac.x64)"
}
run_one() {   # tag timeout flags...
  tag=$1; to=$2; shift 2
  [ -e "$R/$tag" ] && { echo "$R/$tag exists: skipped"; return; }
  gpumon > $R/$tag.gpu 2>&1 &
  mon=$!
  echo "$(date +%T) start $tag: $*"
  PYTHONPATH=. timeout -s INT $to $PY train_tok.py $COMMON "$@" --name hb38 --port 49800 --out $R/$tag \
    > $R/$tag.log 2>&1 < /dev/null
  echo "$(date +%T) $tag exit $? tracebacks $(grep -c Traceback $R/$tag.log)"
  kill $mon
  grep "^devices\|^learner device\|^finished" $R/$tag.log | cut -c1-250
  grep "^u[0-9]* " $R/$tag.log | tail -n 2 | cut -c1-220
  orphans
}
mode=$1; shift
vs=${*:-a b c}
for v in $vs; do
  if [ "$mode" = smoke ]; then
    run_one b38-smoke-$v 1500 --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
      --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 $(variant $v)
  elif [ "$mode" = tput ]; then
    run_one b38-tput-$v 2700 $FAST --workers 64 --game-hours 73 --checkpoint-every 100000 \
      --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant $v)
  fi
done
echo "$(date +%T) done"
