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
#        hpc_b38.sh tput4 [g p]       B38 step 4: g and p (= g + --learner-fast 2) in one window
#        hpc_b38.sh fail              g with ISAAC_RL_B38_FAIL_AT=3: the learner process raises before update 3; the
#                                     trainer must stop with that error (last.pt by the learner process, no hang)
#        hpc_b38.sh int               g stopped by SIGINT after 240 s (the learner process finishes, saves last.pt)
#        hpc_b38.sh kill              g; the trainer (actor) process SIGKILLed after its 3rd update: the learner process
#                                     must notice, save last.pt and end; the dead trainer's helpers and games are ended
# step 5 (--actor-processes 2; variants: p one actor process, q two actor processes on the first visible GPU (a test
# layout: two CUDA processes on one GPU), t6 two actors on cuda:0 / cuda:2 with the learner on cuda:1):
#        hpc_b38.sh smoke5 [p q]      8 workers, 1.5 game hours from scratch (q: 2 x 4)
#        hpc_b38.sh corr5 [p q]       32 workers (q: 2 x 16), 30 updates from segment1-final.pt: the learning numbers
#        hpc_b38.sh fail5             q, the learner raises before update 3: both actors stop, last.pt saved
#        hpc_b38.sh int5              q, SIGINT after 240 s
#        hpc_b38.sh kill5             q, the second actor process SIGKILLed after update 5: the run ends cleanly
#        hpc_b38.sh tput6 [p6 t6]     the 4-GPU measurement (C99 stopped; B38_GPUS=0,1,2,3 B38_SMI=0,1,2,3
#                                     B38_C99_STOPPED=1): 64 workers, 60 updates, no searches; p6 = one actor on cuda:0
#        hpc_b38.sh smoke6-full [t6]  the full C99 flags (searches) with two actors, 8 workers (B38_ALLOW_SEARCHES=1)
#        hpc_b38.sh tput6-full [t6]   the full C99 flags, 64 workers, 60 updates (B38_ALLOW_SEARCHES=1, C99 stopped)
# a load guard ends a run (SIGINT) when the node's 1-minute load stays above LOADMAX (default 90) for 30 s
source /101063/AIsaac/ws14/hpc_env.sh
# (since the rebuilt host A, 10-17 evening: ours are GPUs 2 and 3; B38_GPUS / B38_SMI override)
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
    p) echo "--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2" ;;
    q) echo "--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:0" ;;
    p6) echo "--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2" ;;
    t6) echo "--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2" ;;
  esac
}
gpumon() {   # GPUs 1 and 2 (physical numbering) and the load every 10 s
  while true; do
    echo "$(date +%T) $(nvidia-smi -i ${B38_SMI:-2,3} --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')load $(cut -d' ' -f1 /proc/loadavg)"
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
case $mode in tput|smoke|smoke6-full|tput6-full)
  [ "$B38_ALLOW_SEARCHES" = 1 ] || { echo "$mode runs the searches (--teacher, --cf 1): B38_ALLOW_SEARCHES=1 needed"; exit 1; }
  BASEFLAGS=$COMMON ;;
esac
case $mode in tput6|tput6-full)
  [ "$B38_C99_STOPPED" = 1 ] || { echo "$mode is the 4-GPU measurement: only while C99 is stopped (B38_C99_STOPPED=1, B38_GPUS=0,1,2,3, B38_SMI=0,1,2,3)"; exit 1; }
esac
SMOKE="--workers 8 --checkpoint-every 100000 --rollout 8192 --minibatch 2048 --micro 1024 --learner-fast 1"
if [ "$mode" = fail5 ]; then
  export ISAAC_RL_B38_FAIL_AT=3
  run_one b38-fail5-q 900 $SMOKE --game-hours 1.5 $(variant q)
  unset ISAAC_RL_B38_FAIL_AT
  grep -n "injected\|^RuntimeError\|^finished\|actor process\|terminated" $R/b38-fail5-q.log | cut -c1-200 | head -n 12
  ls -la --time-style=+%T $R/b38-fail5-q/checkpoints/
  orphans
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = int5 ]; then
  run_one b38-int5-q 240 $SMOKE --game-hours 50 $(variant q)
  grep -n "^finished\|actor process\|terminated\|did not end" $R/b38-int5-q.log | cut -c1-200 | head -n 12
  ls -la --time-style=+%T $R/b38-int5-q/checkpoints/
  orphans
  echo "$(date +%T) done"; exit 0
fi
if [ "$mode" = kill5 ]; then
  tag=b38-kill5-q
  [ -e "$R/$tag" ] && { echo "$R/$tag exists"; exit 1; }
  PYTHONPATH=. timeout -s INT 900 $PY train_tok.py $BASEFLAGS $SMOKE --game-hours 50 $(variant q) --name hb38 \
    --port 49800 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null &
  tp=$!
  for i in $(seq 1 300); do grep -q "^u5 " $R/$tag.log && break; sleep 2; done
  a1=$(sed -n 's/.*actor process 1 (B38 step 5): pid \([0-9]*\).*/\1/p' $R/$tag.log | head -n 1)
  echo "$(date +%T) SIGKILL to the second actor process $a1"
  kill -KILL $a1
  wait $tp; rc=$?; echo "$(date +%T) trainer exit $rc"
  grep -n "^RuntimeError\|^finished\|actor process 1 ended\|terminated" $R/$tag.log | cut -c1-200 | head
  ls -la --time-style=+%T $R/$tag/checkpoints/
  orphans
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
  elif [ "$mode" = smoke5 ]; then
    run_one b38-smoke5-$v 1500 $SMOKE --game-hours 1.5 $(variant $v)
  elif [ "$mode" = smoke6-full ]; then
    run_one b38-smoke6full-$v 1500 $SMOKE --game-hours 1.5 $(variant $v)
  elif [ "$mode" = corr5 ]; then
    run_one b38-corr5-$v 2400 $FAST --workers 32 --game-hours 36.5 --checkpoint-every 100000 \
      --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant $v)
  elif [ "$mode" = tput6 ] || [ "$mode" = tput6-full ]; then
    run_one b38-$mode-$v 2700 $FAST --workers 64 --game-hours 73 --checkpoint-every 100000 \
      --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant $v)
  elif [ "$mode" = tput ] || [ "$mode" = tput2 ] || [ "$mode" = tput3 ] || [ "$mode" = tput4 ]; then
    run_one b38-$mode-$v${B38_WORKERS:+-w$B38_WORKERS} 2700 $FAST --workers ${B38_WORKERS:-64} --game-hours 73 --checkpoint-every 100000 \
      --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 --resume $CK $(variant $v)
  fi
done
echo "$(date +%T) done"
