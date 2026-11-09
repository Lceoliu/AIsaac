#!/bin/sh
# B52 profiles (GPU_LOCK protocol, guards): the cpp actor driver (16 updates), the Python path of the same tree
# (16 updates, the merged baseline), the CF-off control (8 updates), the nsys python-gil run (hand-offs 2-7), and one
# update under the torch profiler for the learner kernel durations (condition 6). usage: run_b52p.sh <code dir> [which]
# which: all (default) | smoke | cpp | py | nocf | nsys | kern (smoke is not part of all)
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b52}
WHICH=${2:-all}
LOG=$R/b52.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
want() { { [ "$WHICH" = all ] && [ "$1" != smoke ]; } || [ "$WHICH" = "$1" ]; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B52 profiles ($CODE, $WHICH): cpp 16 upd, python 16 upd, CF off 8 upd, nsys GIL, learner kernels" > $R/GPU_LOCK
echo "$(date +%T) b52p start ($CODE, $WHICH)" >> $LOG
guard
cd $ROOT
if want smoke; then   # 2 updates with the driver: does the integration run at all
  CODE=$CODE ./prof2.sh prof-b52-smoke 1800 2.5
  echo "$(date +%T) prof-b52-smoke done rc=$? $(grep -m1 'B52 cpp actor driver' $R/prof-b52-smoke.stdout | cut -c1-120); $(grep -E '^u[0-9]+ ' $R/prof-b52-smoke.stdout | tail -1 | cut -c1-100)" >> $LOG
  sleep 5; guard
fi
if want cpp; then
  CODE=$CODE ./prof2.sh prof-b52-cpp 4500 18.3
  echo "$(date +%T) prof-b52-cpp done rc=$? $(grep -c 'collect' $R/prof-b52-cpp/progress.csv 2>/dev/null) rows; driver: $(grep -m1 'B52 cpp actor driver' $R/prof-b52-cpp.stdout | cut -c1-120)" >> $LOG
  sleep 5; guard
fi
if want py; then
  ISAAC_BLOCK_CPP_ACTOR=0 CODE=$CODE ./prof2.sh prof-b52-py 4500 18.3
  echo "$(date +%T) prof-b52-py (python path, same tree) done rc=$?" >> $LOG
  sleep 5; guard
fi
if want nocf; then
  EXTRA="--cf-points 0" CODE=$CODE ./prof2.sh prof-b52-nocf 2400 8.6
  echo "$(date +%T) prof-b52-nocf done rc=$?" >> $LOG
  sleep 5; guard
fi
if want nsys; then
  rm -f $R/GPU_LOCK   # (run_b50m.sh takes the lock itself)
  PYS='' TRIGGER=nvtx $ROOT/$CODE/bridge/ops/run_b50m.sh prof-b52m $CODE 1800 10.3 nvtx,python-gil
  echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B52 profiles ($CODE): continuing" > $R/GPU_LOCK
  echo "$(date +%T) prof-b52m (nsys) done rc=$?" >> $LOG
  guard
fi
if want kern; then
  ISAAC_RL_B38_TORCHPROF=$R/b52-kern ISAAC_RL_B38_TORCHPROF_AT=3 CODE=$CODE ./prof2.sh prof-b52-kern 1500 4.9
  echo "$(date +%T) prof-b52-kern (torch profiler on update 3) done rc=$?" >> $LOG
  python3 $ROOT/$CODE/bridge/ops/b52_kernels.py $R/b52-kern/trace.json 10 > $R/b52-kernels.txt 2>&1
  echo "$(date +%T) kernels: $(head -1 $R/b52-kernels.txt)" >> $LOG
  sleep 5; guard
fi
rm -f $R/GPU_LOCK
echo "$(date +%T) b52p lock released" >> $LOG
