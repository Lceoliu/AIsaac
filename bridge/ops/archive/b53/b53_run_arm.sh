#!/bin/sh
# B53: run (or resume) one ablation arm on the 4090 -- prepared, not started by B53.
#   usage: b53_run_arm.sh <A|B|C|D> <total_game_hours> [wall_limit_s]
#   env:   CODE=<candidate dir under $ROOT> (required: the code to train with)
#          DUMP_AT=<update>   one learner dump (ISAAC_LEARNER_DUMP) at that update, for the late B49 re-measure
# The arm's run directory is $ROOT/reports/b53/arm-<arm> (its own progress.csv, checkpoints every 25 updates).
# A rerun of the same command resumes from checkpoints/last.pt with the remaining game hours (b53_arm.py).
# GPU_LOCK: waits until the lock is free and nothing else trains / tests, then holds it for the whole run.
set -u
ARM=$1; TOTAL=$2; LIMIT=${3:-172800}
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
OUT=$ROOT/reports/b53/arm-$ARM
OPS=$ROOT/$CODE/bridge/ops/b53
PY=$ROOT/.venv/bin/python
LOG=$ROOT/reports/b53/b53.log
mkdir -p $ROOT/reports/b53
guard() { echo "$(date '+%F %T') [$ARM] guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ]; do sleep 60; done
echo "b53-arm-$ARM $(date '+%F %T') B53 ablation arm $ARM, $TOTAL game hours ($CODE)" > $R/GPU_LOCK
guard
ARGS=$($PY $OPS/b53_arm.py $OPS/arms.json $ARM $TOTAL $OUT) || { echo "$(date '+%F %T') [$ARM] b53_arm.py failed" >> $LOG; rm -f $R/GPU_LOCK; exit 1; }
if [ "$ARGS" = "DONE" ]; then echo "$(date '+%F %T') [$ARM] nothing left ($TOTAL game hours done)" >> $LOG; rm -f $R/GPU_LOCK; exit 0; fi
mkdir -p $OUT
cd $ROOT/$CODE/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
if [ -n "${DUMP_AT:-}" ]; then export ISAAC_LEARNER_DUMP=$OUT/b49-dump ISAAC_LEARNER_DUMP_AT=$DUMP_AT; fi
echo "$PY -u train_tok.py $ARGS" > $OUT/command-$(date +%Y%m%d-%H%M%S).txt
echo "$(date '+%F %T') [$ARM] start: $ARGS" >> $LOG
eval xvfb-run -a -s \"-screen 0 640x480x24 -nolisten tcp -noreset\" timeout --signal=INT --kill-after=120s $LIMIT \
  $PY -u train_tok.py $ARGS >> $OUT/stdout.log 2>> $OUT/stderr.log
rc=$?
echo "$(date '+%F %T') [$ARM] train_tok rc=$rc, game hours $(tail -1 $OUT/progress.csv 2>/dev/null | cut -d, -f3)" >> $LOG
sleep 10; guard
grep -q "b53-arm-$ARM" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK
exit $rc
