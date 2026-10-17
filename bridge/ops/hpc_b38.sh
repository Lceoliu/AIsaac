#!/bin/bash
# B38 (train_tok --learner-device) tests on HPC160, sandbox ws14 (mk_ws14.sh), GPUs 1 and 2 only (GPU 0 runs C99:
# CUDA_VISIBLE_DEVICES=1,2, so cuda:0 = GPU 1 is the actor's, cuda:1 = GPU 2 the learner's). One run at a time, each
# ends by itself (--game-hours) or by SIGINT at its time limit (the trainer saves last.pt and closes its sampler);
# afterwards the game processes whose ABP_WATCH_PID names a dead pid are ended (as hpc_guard.sh does).
# usage: hpc_b38.sh smoke [a b c]   8 workers, 1.5 game hours, the C99 recipe from scratch (hpc_c99.sh smoke)
#        hpc_b38.sh tput  [a b c]   64 workers, ~60 updates (73 game hours), the C99 run flags + FAST, resumed from
#                                   C99's segment1-final.pt into ws14/runs/b38-tput-<v> (never pre-created)
#        hpc_b38.sh tput2 [a b c]   the same without the searches (no --teacher, --cf 0): 2026-10-17 09:13-09:33 the
#                                   full C99 flags (searches in forked children) of a second 64-worker trainer drove the
#                                   node's load to ~175 and C99 to ~120x until the run was stopped
# variants: a one GPU, synchronous (as C99); b two GPUs + --overlap --decoupled 1; c two GPUs, synchronous;
#           d one GPU + --overlap --decoupled 1 (reference: the B11 / B13 contention);
#           e two GPUs + --overlap, plain ratio, no recompute pass (--decoupled 0 --recompute-values off);
#           f as b with Python's default thread switch interval (--switch-interval 0.005);
#           g (B38 step 2) two GPUs, the learner in its own process (--learner-process 1) + --overlap --decoupled 1;
#           h (B38 step 2) as g, synchronous
# step 2 (no searches anywhere: no --teacher, --cf 0, as long as C99 runs on this host):
#        hpc_b38.sh smoke2 [g h a b]  8 workers, 1.5 game hours from scratch
#        hpc_b38.sh tput3 [a b g]     as tput2 (a second series: a, b and g in one window)
#        hpc_b38.sh fail              g with ISAAC_RL_B38_FAIL_AT=3: the learner process raises before update 3; the
#                                     trainer must stop with that error (last.pt by the learner process, no hang)
#        hpc_b38.sh int               g stopped by SIGINT after 240 s (the learner process finishes, saves last.pt)
#        hpc_b38.sh kill              g; the trainer (actor) process SIGKILLed after its 3rd update: the learner process
#                                     must notice, save last.pt and end; the dead trainer's helpers and games are ended
# a load guard ends a run (SIGINT) when the node's 1-minute load stays above LOADMAX (default 85) for 30 s
source /101063/AIsaac/ws14/hpc_env.sh
# (2026-10-17 evening, the rebuilt host A with 4 GPUs: C99 on GPUs 0 / 1, its watcher on 2 now and then: ours 2 and 3;
#  B38_GPUS overrides)
export CUDA_VISIBLE_DEVICES=${B38_GPUS:-2,3}
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
LITE=$(echo "$COMMON" | sed -e 's/--teacher --teacher-queue 12/--teacher-queue 12/' -e 's/--cf 1 /--cf 0 /')
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
variant() {
  case $1 in
    a) echo "" ;;
    b) echo "--learner-device cuda:1 --overlap --decoupled 1" ;;
    c) echo "--learner-device cuda:1" ;;
    d) echo "--overlap --decoupled 1" ;;
    e) echo "--learner-device cuda:1 --overlap --recompute-values off" ;;
    f) echo "--learner-device cuda:1 --overlap --decoupled 1 --switch-interval 0.005" ;;
    g) echo "--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1" ;;
    h) echo "--learner-device cuda:1 --learner-process 1" ;;
    k) echo "--learner-device cuda:1,cuda:0 --learner-process 1 --overlap --decoupled 1" ;;
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
  for p in $(games); do   # (environ read with a time limit: a forking game holds its mm lock)
    w=$(timeout 2 cat /proc/$p/environ 2>/dev/null | tr '\000' '\012' | sed -n 's/^ABP_WATCH_PID=//p')
    if [ -n "$w" ] && ! kill -0 "$w" 2>/dev/null; then kill -KILL $p 2>/dev/null; n=$((n+1)); fi
  done
  echo "$(date +%T) orphan games ended: $n; games left $(games | wc -l)"
}
games() {   # game pids from /proc/<pid>/comm (pgrep reads files that block while a process forks)
  for d in /proc/[0-9]*; do [ "$(cat $d/comm 2>/dev/null)" = isaac.x64 ] && echo ${d#/proc/}; done
}
run_one() {   # tag timeout flags...
  tag=$1; to=$2; shift 2
  [ -e "$R/$tag" ] && { echo "$R/$tag exists: skipped"; return; }
  gpumon > $R/$tag.gpu 2>&1 &
  mon=$!
  echo "$(date +%T) start $tag: $*"
  PYTHONPATH=. timeout -s INT $to $PY train_tok.py $BASEFLAGS "$@" --name hb38 --port 49800 --out $R/$tag \
    > $R/$tag.log 2>&1 < /dev/null &
  tp=$!
  high=0
  while kill -0 $tp 2>/dev/null; do   # (the load guard: SIGINT to timeout, which hands it to the trainer)
    sleep 10
    l=$(cut -d. -f1 /proc/loadavg)
    if [ "$l" -gt "${LOADMAX:-90}" ]; then high=$((high + 1)); else high=0; fi
    if [ $high -ge 3 ]; then
      echo "$(date +%T) load $l above ${LOADMAX:-90} for 30 s: $tag stopped"; kill -INT $tp; high=-100000
    fi
  done
  wait $tp; rc=$?
  echo "$(date +%T) $tag exit $rc tracebacks $(grep -c Traceback $R/$tag.log)"
  kill $mon
  grep "^devices\|^learner device\|^finished" $R/$tag.log | cut -c1-250
  grep "^u[0-9]* " $R/$tag.log | tail -n 2 | cut -c1-220
  orphans
}
mode=$1; shift
vs=${*:-a b c}
# (no searches unless asked for: the full flags of a second trainer next to C99 are what hurt C99 on 10-17)
BASEFLAGS=$LITE
case $mode in tput|smoke)
  [ "$B38_ALLOW_SEARCHES" = 1 ] || { echo "$mode runs the searches (--teacher, --cf 1): B38_ALLOW_SEARCHES=1 needed"; exit 1; }
  BASEFLAGS=$COMMON ;;
esac
SMOKE="--workers 8 --checkpoint-every 100000 --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1"
# B38 step 3a (two learner ranks; only GPUs 2 and 3 here, so the functional runs put rank 1 on the actor's GPU:
# ISAAC_RL_B38_SHARED_GPU=1, a test aid; the speed of two ranks is measured without an actor: dump + replay)
#   hpc_b38.sh smoke3      variant k (ranks on cuda:1 and the actor's cuda:0), 8 workers, 1.5 game hours
#   hpc_b38.sh fail3       k with ISAAC_RL_B38_FAIL_AT=3 (rank 1 raises)
#   hpc_b38.sh dump        g with 64 workers, ISAAC_RL_B38_DUMP: the job of update 8 saved to runs/b38-job
#   hpc_b38.sh replay      abplus_b38_replay.py on that job: 1 rank vs 2 ranks (8 updates each), then the
#                          one-update weight comparisons (1 rank micro 2048 twice, 2 ranks, 1 rank micro 4096)
if [ "$mode" = smoke3 ]; then
  export ISAAC_RL_B38_SHARED_GPU=1
  run_one b38-smoke3-k 1500 $SMOKE --game-hours 1.5 $(variant k)
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = fail3 ]; then
  export ISAAC_RL_B38_SHARED_GPU=1 ISAAC_RL_B38_FAIL_AT=3
  run_one b38-fail3-k 900 $SMOKE --game-hours 1.5 $(variant k)
  unset ISAAC_RL_B38_FAIL_AT
  grep -n "injected\|^RuntimeError\|^finished\|learner process\|terminated" $R/b38-fail3-k.log | cut -c1-200 | head -n 12
  ls -la $R/b38-fail3-k/checkpoints/
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = dump ]; then
  export ISAAC_RL_B38_DUMP=$R/b38-job ISAAC_RL_B38_DUMP_AT=8
  run_one b38-dump-g 1200 $FAST --workers 64 --game-hours 13 --checkpoint-every 100000 \
    --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant g)
  unset ISAAC_RL_B38_DUMP ISAAC_RL_B38_DUMP_AT
  ls -la $R/b38-job
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = replay ]; then
  RP="--workers 64 --rollout 32768 --minibatch 4096 --learner-fast 1 --resume $CK --learner-process 1 --overlap --decoupled 1"
  rp() {   # tag reps devices micro
    [ -e "$R/$1" ] && { echo "$R/$1 exists: skipped"; return; }
    PYTHONPATH=. timeout -s INT 1200 $PY abplus_b38_replay.py --job $R/b38-job/job.pt --reps $2 -- $BASEFLAGS $RP \
      --learner-device $3 --micro $4 --name hb38r --port 49850 --out $R/$1 > $R/$1.log 2>&1 < /dev/null
    echo "$(date +%T) $1: exit $? tracebacks $(grep -c Traceback $R/$1.log)"; grep "^ranks" $R/$1.log | cut -c1-400
  }
  rp b38-rep-r1 8 cuda:0 4096
  rp b38-rep-r2 8 cuda:0,cuda:1 4096
  rp b38-rep-r1b 8 cuda:1 4096
  rp b38-eq-1a 1 cuda:0 2048
  rp b38-eq-1b 1 cuda:0 2048
  rp b38-eq-2 1 cuda:0,cuda:1 4096
  rp b38-eq-1c 1 cuda:0 4096
  PYTHONPATH=. $PY -c "
import sys, torch
R = sys.argv[1]
ck = {t: torch.load(f'{R}/b38-eq-{t}/checkpoints/last.pt', map_location='cpu', weights_only=False)['model'] for t in ('1a', '1b', '2', '1c')}
def d(a, b):
    return max(float((ck[a][k].float() - ck[b][k].float()).abs().max()) for k in ck[a] if ck[a][k].is_floating_point())
base = torch.load('$CK', map_location='cpu', weights_only=False)['model']
step = max(float((ck['1a'][k].float() - base[k].float()).abs().max()) for k in base if base[k].is_floating_point())
print(f'one update from the same job: |1 rank micro 2048 - again| {d("1a", "1b"):.3g}, |2 ranks - 1 rank micro 2048| {d("2", "1a"):.3g}, |2 ranks - 1 rank micro 4096| {d("2", "1c"):.3g}, |1 rank 2048 - 4096| {d("1a", "1c"):.3g}; the update itself moved weights by up to {step:.3g}')
" $R
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = fail ]; then
  export ISAAC_RL_B38_FAIL_AT=3
  run_one b38-fail-g 900 $SMOKE --game-hours 1.5 $(variant g)
  unset ISAAC_RL_B38_FAIL_AT
  grep -n "injected\|^RuntimeError\|Traceback\|^finished\|learner process" $R/b38-fail-g.log | cut -c1-200 | head -n 12
  ls -la $R/b38-fail-g/checkpoints/
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = int ]; then
  run_one b38-int-g 240 $SMOKE --game-hours 50 $(variant g)
  grep -n "Traceback\|^finished\|learner process\|KeyboardInterrupt" $R/b38-int-g.log | cut -c1-200 | head -n 12
  ls -la --time-style=+%T $R/b38-int-g/checkpoints/
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = kill ]; then
  tag=b38-kill-g
  [ -e "$R/$tag" ] && { echo "$R/$tag exists"; exit 1; }
  PYTHONPATH=. timeout -s INT 900 $PY train_tok.py $BASEFLAGS $SMOKE --game-hours 50 $(variant g) --name hb38 \
    --port 49800 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null &
  tp=$!
  for i in $(seq 1 300); do grep -q "^u5 " $R/$tag.log && break; sleep 2; done
  tr_=$(cat /proc/$tp/task/*/children | tr -d " ")   # (timeout's child: the trainer)
  kids=$(cat /proc/$tr_/task/*/children)
  lp=$(sed -n 's/.*learner process (B38 step 2): pid \([0-9]*\).*/\1/p' $R/$tag.log | head -n 1)
  echo "$(date +%T) trainer $tr_, learner process $lp, children: $kids; SIGKILL to the trainer"
  kill -KILL $tr_
  for i in $(seq 1 60); do [ -e /proc/$lp ] || break; sleep 1; done
  echo "$(date +%T) learner process $([ -e /proc/$lp ] && echo STILL ALIVE || echo ended)"
  for k in $kids; do [ "$k" != "$lp" ] && kill -KILL $k 2>/dev/null; done   # (the dead trainer's worker helpers)
  wait $tp
  grep -n "trainer is gone\|Traceback\|learner process" $R/$tag.log | cut -c1-200 | head
  ls -la --time-style=+%T $R/$tag/checkpoints/
  orphans
  echo "$(date +%T) done"; exit 0
fi
for v in $vs; do
  if [ "$mode" = smoke2 ]; then
    run_one b38-smoke2-$v 1500 $SMOKE --game-hours 1.5 $(variant $v)
  elif [ "$mode" = smoke ]; then
    run_one b38-smoke-$v 1500 --workers 8 --game-hours 1.5 --checkpoint-every 100000 \
      --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1 $(variant $v)
  elif [ "$mode" = tput ] || [ "$mode" = tput2 ] || [ "$mode" = tput3 ]; then
    run_one b38-$mode-$v 2700 $FAST --workers 64 --game-hours 73 --checkpoint-every 100000 \
      --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant $v)
  fi
done
echo "$(date +%T) done"
