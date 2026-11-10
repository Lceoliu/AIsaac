#!/bin/sh
# B58c/d step 0: the driver's per-phase time with the native consumer on and off (ISAAC_BLOCK_CPP_CONSUMER=1 / 0),
# 24 updates each, no nsys; the off arm is also the Python-lane CF baseline for B58d. Each arm takes GPU_LOCK itself
# (waits while it is held or a trainer / pytest runs), so another session can use the GPU between the arms.
# usage: run_b58c0.sh <code dir> [arms: "on off"]
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58c0}
ARMS=${2:-on off}
LOG=$R/b58.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
waitlock() { while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; }
echo "$(date +%T) c0 start ($CODE, arms $ARMS)" >> $LOG
cd $ROOT
for A in $ARMS; do
  while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done
  echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B58c/d step 0: consumer $A, 24 updates, no nsys ($CODE)" > $R/GPU_LOCK
  guard
  C=1; [ "$A" = off ] && C=0
  ISAAC_BLOCK_CPP_CONSUMER=$C CODE=$CODE ./prof2.sh prof-b58c0-$A 6000 29.0
  rc=$?
  NC=$(grep -m1 'B58 native consumer' $R/prof-b58c0-$A.stdout | cut -c1-80)
  TB=$(grep -cE 'File "[^"]*train_tok\.py"' $R/prof-b58c0-$A.stdout $R/prof-b58c0-$A.stderr 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  echo "$(date +%T) c0 arm $A done rc=$rc, $(($(wc -l < $R/prof-b58c0-$A/progress.csv 2>/dev/null || echo 1)-1)) updates, tracebacks $TB, $NC" >> $LOG
  sleep 5; guard
  rm -f $R/GPU_LOCK
done
$ROOT/.venv/bin/python $ROOT/$CODE/bridge/ops/b58_ab_summary.py $(for A in $ARMS; do echo $R/prof-b58c0-$A; done) > $R/prof-b58c0.summary.txt 2>&1
echo "$(date +%T) c0 done (summary rc=$?)" >> $LOG
