#!/bin/sh
# B56 second GPU pass (<= 25 min with the lock; after the cpp-actor B54 chain):
#   1. warm-cache game profile (ISAAC_FX_CACHE=1, production Inductor cache dir, filled by prof-b56-fx): 6 updates,
#      then SIGINT to the train_tok main process only; the warm warm-up in production and update vs the baselines;
#   2. P1b as in B55: the probe cold in its own fresh cache dir (Spatial only, Triton from the production dir),
#      then warm with --why-miss (Spatial only, no steady check: already verified in prof-b56-fx).
#   usage: CODE=<candidate dir> sh b56_verify2.sh
set -u
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
TAG=prof-b56-fxw
LOG=$R/b56.log
OUT=$R/b56
mkdir -p $OUT
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ] || [ "$(pgrep -fc '[r]un_b54')" != "0" ]; do sleep 30; done
echo "a1 $(date '+%F %T') B56 pass 2: warm FX cache profile + P1b cold/warm probe ($CODE)" > $R/GPU_LOCK
guard() { echo "$(date '+%F %T') guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc 'python.* [b]55_compile_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
T0=$(date +%s)
echo "$(date '+%F %T') b56 pass 2 start ($CODE)" >> $LOG
guard
cd $ROOT
( ISAAC_FX_CACHE=1 CODE=$CODE ./prof2.sh $TAG 1200 8.0 ) &
P=$!
while kill -0 $P 2>/dev/null; do
  sleep 15
  n=0
  [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
  if [ $n -ge 6 ]; then
    echo "$(date '+%F %T') $n updates: SIGINT to the train_tok main process" >> $LOG
    pkill -INT -f "^$PY -u train_tok.py"
    break
  fi
done
wait $P
echo "$(date '+%F %T') profile done, $(grep -h 'B42 compiled Spatial\|B44 Flex' $R/$TAG.stdout | tr '\n' ' ')" >> $LOG
sleep 10; guard
cd $ROOT/$CODE/bridge/ops
FX=$HOME/.cache/isaac-b56-probe-$(date +%H%M%S)
mkdir -p $FX
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
TORCHINDUCTOR_CACHE_DIR=$FX TRITON_CACHE_DIR=$HOME/.cache/isaac-b42-inductor/triton/0 timeout 720 \
  $PY b55_compile_probe.py $R/prof-b48-pf/checkpoints/last.pt $OUT/p1b-cold.json --fx-cache on --no-reader > $OUT/p1b-cold.log 2>&1
echo "$(date '+%F %T') P1b cold rc=$? $(grep -h 'B55 spatial' $OUT/p1b-cold.log)" >> $LOG
guard
LEFT=$((1470-($(date +%s)-T0)))
[ $LEFT -gt 420 ] && LEFT=420
if [ $LEFT -ge 200 ]; then
  TORCHINDUCTOR_CACHE_DIR=$FX TRITON_CACHE_DIR=$HOME/.cache/isaac-b42-inductor/triton/0 timeout $LEFT \
    $PY b55_compile_probe.py $R/prof-b48-pf/checkpoints/last.pt $OUT/p1b-warm.json --fx-cache on --no-reader --why-miss > $OUT/p1b-warm.log 2>&1
  echo "$(date '+%F %T') P1b warm rc=$? (timeout $LEFT) $(grep -h 'B55 spatial' $OUT/p1b-warm.log)" >> $LOG
else
  echo "$(date '+%F %T') P1b warm skipped: $LEFT s left of the 25 min" >> $LOG
fi
sleep 5; guard
grep -q "B56 pass 2" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK
echo "$(date '+%F %T') b56 pass 2 done, GPU_LOCK released, $(( $(date +%s)-T0 )) s" >> $LOG
