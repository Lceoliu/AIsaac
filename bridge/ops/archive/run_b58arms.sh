#!/bin/sh
# Step 0 of phase 3: each B53 ablation arm (A detach, B random-2 unbiased, C gradreplay-2, D random-4 unbiased) started
# for ~2 updates with the merged tree, in its own directory (not the arms' run directories): the C++ actor path must be
# on for every arm (driver on, its per-collect line printed), none may fall back. GPU_LOCK held for the four runs.
# usage: run_b58arms.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b58}
LOG=$R/b58.log
OPS=$ROOT/$CODE/bridge/ops/b53
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') step 0: the four B53 arms, ~2 updates each, C++ path check ($CODE)" > $R/GPU_LOCK
echo "$(date +%T) b58arms start ($CODE)" >> $LOG
guard
cd $ROOT/$CODE/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null ISAAC_BLOCK_TIMING=1
for ARM in A B C D; do
  OUT=$R/b58-arm-$ARM
  ARGS=$($PY $OPS/b53_arm.py $OPS/arms.json $ARM 2.5 $OUT | sed "s/--port [0-9]*/--port 3926$(echo $ARM | tr ABCD 0123)/")
  eval xvfb-run -a -s \"-screen 0 640x480x24 -nolisten tcp -noreset\" timeout --signal=INT --kill-after=60s 1500 \
    $PY -u train_tok.py $ARGS > $OUT.stdout 2> $OUT.stderr
  rc=$?
  DRV=$(grep -m1 'B52 cpp actor driver' $OUT.stdout | cut -c1-90)
  NB=$(grep -c 'B54 driver collect' $OUT.stdout)
  UP=$(($(wc -l < $OUT/progress.csv 2>/dev/null || echo 1)-1))
  TB=$(grep -cE 'File "[^"]*train_tok\.py"' $OUT.stdout $OUT.stderr | awk -F: '{s+=$2} END {print s+0}')
  OK=no; echo "$DRV" | grep -q 'driver: on' && [ "$NB" -ge 2 ] && [ "$UP" -ge 1 ] && [ "$TB" = 0 ] && OK=yes
  echo "$(date +%T) arm $ARM: rc=$rc, C++ path $OK ($DRV; $NB driver collect lines; $UP updates; train_tok tracebacks $TB); flags: $(echo "$ARGS" | grep -o -- '--block-history-[a-z]* [^ ]*' | tr '\n' ' ')" >> $LOG
  sleep 5; guard
done
rm -f $R/GPU_LOCK
echo "$(date +%T) b58arms lock released" >> $LOG
