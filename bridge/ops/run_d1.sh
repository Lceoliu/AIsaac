#!/bin/sh
# D1 (the CF regression after the B60 merge): consumer off, 16 updates, ISAAC_BLOCK_NATIVE_PACK=0 against the default,
# no nsys; mid-run the machine's cores and the trainer's threads by name. Each arm takes GPU_LOCK itself.
# usage: run_d1.sh <code dir> [arms: "pack0 pack1"]
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-d1}
ARMS=${2:-pack0 pack1}
LOG=$R/b58.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
waitlock() { while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; }
echo "$(date +%T) d1 start ($CODE, arms $ARMS)" >> $LOG
cd $ROOT
for A in $ARMS; do
  while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done
  echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') D1 CF diagnosis: $A, consumer off, 16 updates ($CODE)" > $R/GPU_LOCK
  guard
  P=1; [ "$A" = pack0 ] && P=0
  ( sleep 300; { echo "cores $(nproc)"; T=$(pgrep -f 'train_tok.py' | head -1); echo "trainer pid $T threads $(ls /proc/$T/task 2>/dev/null | wc -l)";
    for t in /proc/$T/task/*; do cat $t/comm; done | sort | uniq -c | sort -rn | head -20; echo "processes: $(pgrep -c isaac) isaac, $(ps -e --no-headers | wc -l) total"; } > $R/prof-d1-$A.threads.txt 2>&1 ) &
  ISAAC_BLOCK_CPP_CONSUMER=0 ISAAC_BLOCK_NATIVE_PACK=$P CODE=$CODE ./prof2.sh prof-d1-$A 4500 19.3
  rc=$?
  wait
  echo "$(date +%T) d1 arm $A done rc=$rc, $(($(wc -l < $R/prof-d1-$A/progress.csv 2>/dev/null || echo 1)-1)) updates" >> $LOG
  sleep 5; guard
  rm -f $R/GPU_LOCK
done
$ROOT/.venv/bin/python $ROOT/$CODE/bridge/ops/b58_ab_summary.py $(for A in $ARMS; do echo $R/prof-d1-$A; done) > $R/prof-d1.summary.txt 2>&1
echo "$(date +%T) d1 done (summary rc=$?)" >> $LOG
