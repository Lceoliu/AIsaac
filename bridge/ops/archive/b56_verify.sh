#!/bin/sh
# B56 GPU verification of ISAAC_FX_CACHE=1 (run after the cpp-actor B54 chain; <= 30 min with the lock):
#   1. a short game profile with the switch on (prof2.sh, production Inductor cache dir), stopped by SIGINT once
#      6 updates are in progress.csv (hard limit 1500 s): block_steady_compiles = 0 with Flex and real batches,
#      update / collect / x real time vs the same-tree profiles (prof-b51-unb, prof-b52-py);
#   2. P1b in the same lock: b55_compile_probe.py --fx-cache on --steady --why-miss on the now-populated cache
#      (warm): why the 8 learner grad graphs' AOT cache entries miss their FX graphs.
#   usage: CODE=<candidate dir> sh b56_verify.sh
set -u
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
TAG=prof-b56-fx
LOG=$R/b56.log
OUT=$R/b56
mkdir -p $OUT
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ] || [ "$(pgrep -fc '[r]un_b54')" != "0" ]; do sleep 30; done
echo "a1 $(date '+%F %T') B56 FX cache profile + P1b guard-miss diagnosis ($CODE)" > $R/GPU_LOCK
guard() { echo "$(date '+%F %T') guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[b]55_compile_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
T0=$(date +%s)
echo "$(date '+%F %T') b56 start ($CODE)" >> $LOG
guard
cd $ROOT
( ISAAC_FX_CACHE=1 CODE=$CODE ./prof2.sh $TAG 1500 8.0 ) &
P=$!
while kill -0 $P 2>/dev/null; do
  sleep 15
  n=$(($(wc -l < $R/$TAG/progress.csv 2>/dev/null || echo 1)-1))
  if [ $n -ge 6 ]; then
    echo "$(date '+%F %T') $n updates: SIGINT" >> $LOG
    pkill -INT -f "[t]rain_tok.py.*$TAG"
    break
  fi
done
wait $P
echo "$(date '+%F %T') profile done, $(grep -h 'B42 compiled Spatial\|B44 Flex' $R/$TAG.stdout | tr '\n' ' ')" >> $LOG
sleep 10; guard
LEFT=$((1740-($(date +%s)-T0)))
[ $LEFT -gt 480 ] && LEFT=480
if [ $LEFT -ge 240 ]; then
  cd $ROOT/$CODE/bridge/ops
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 timeout $LEFT $PY b55_compile_probe.py $R/prof-b48-pf/checkpoints/last.pt \
    $OUT/why-miss.json --fx-cache on --steady --why-miss > $OUT/why-miss.log 2>&1
  echo "$(date '+%F %T') P1b probe rc=$? (timeout $LEFT) $(grep -h 'B55 spatial\|B55 STATS' $OUT/why-miss.log | tr '\n' ' ')" >> $LOG
else
  echo "$(date '+%F %T') P1b probe skipped: $LEFT s left of the 30 min" >> $LOG
fi
sleep 5; guard
grep -q "B56 FX cache" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK
echo "$(date '+%F %T') b56 done, GPU_LOCK released, $(( $(date +%s)-T0 )) s" >> $LOG
