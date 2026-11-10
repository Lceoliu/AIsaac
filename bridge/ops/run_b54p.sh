#!/bin/sh
# B54 profiles (GPU_LOCK protocol, guards): the cpp actor driver with the lane rounds on it (16 updates), the Python
# path of the same tree (16 updates), the CF-off control (8 updates). usage: run_b54p.sh <code dir> [which]
# which: all (default: cpp, py, nocf) | smoke | cpp | py | nocf
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b54}
WHICH=${2:-all}
LOG=$R/b54.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
want() { { [ "$WHICH" = all ] && [ "$1" != smoke ]; } || [ "$WHICH" = "$1" ]; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
if [ "$WHICH" != smoke ] && [ -f $R/b54-gate.failed ]; then echo "$(date +%T) b54p $WHICH refused: the full gate failed ($(tr '\n' ' ' < $R/b54-gate.failed))" >> $LOG; exit 3; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B54 profiles ($CODE, $WHICH): cpp 16 upd, python 16 upd, CF off 8 upd" > $R/GPU_LOCK
echo "$(date +%T) b54p start ($CODE, $WHICH)" >> $LOG
guard
cd $ROOT
if want smoke; then   # 2 updates with the driver: does the integration run at all
  CODE=$CODE ./prof2.sh prof-b54-smoke 1800 3.7
  echo "$(date +%T) prof-b54-smoke done rc=$? $(grep -m1 'B54 cpp actor driver' $R/prof-b54-smoke.stdout | cut -c1-120); $(grep -E '^u[0-9]+ ' $R/prof-b54-smoke.stdout | tail -1 | cut -c1-100)" >> $LOG
  sleep 5; guard
fi
if want cpp; then
  CODE=$CODE ./prof2.sh prof-b54-cpp 4500 18.3
  echo "$(date +%T) prof-b54-cpp done rc=$? $(grep -c 'collect' $R/prof-b54-cpp/progress.csv 2>/dev/null) rows; driver: $(grep -m1 'B54 cpp actor driver' $R/prof-b54-cpp.stdout | cut -c1-120)" >> $LOG
  sleep 5; guard
fi
if want py; then
  ISAAC_BLOCK_CPP_ACTOR=0 CODE=$CODE ./prof2.sh ${PYTAG:-prof-b54-py} 4500 18.3
  echo "$(date +%T) ${PYTAG:-prof-b54-py} (python path, same tree) done rc=$?" >> $LOG
  sleep 5; guard
fi
if want nocf; then
  EXTRA="--cf-points 0" CODE=$CODE ./prof2.sh prof-b54-nocf 2400 8.6
  echo "$(date +%T) prof-b54-nocf done rc=$?" >> $LOG
  sleep 5; guard
fi
rm -f $R/GPU_LOCK
echo "$(date +%T) b54p lock released" >> $LOG
