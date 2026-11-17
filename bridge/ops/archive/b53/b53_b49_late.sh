#!/bin/sh
# B53: the late B49 re-measure in one command, on an arm's own learner dump.
#   usage: b53_b49_late.sh <A|B|C|D> [advance]
#   env:   CODE=<candidate dir under $ROOT>
# Without 'advance': the arm's dump (made during its run with DUMP_AT=<update>, b53_run_arm.sh) is analysed:
#   b49_grad_stats.py (q_j concentration, sampling rules vs the full first-order gradient) and b48_coverage.py.
# With 'advance': no dump yet -- the arm is resumed for 3 more updates with a dump at the first of them (this trains
#   the arm 3 updates further on its own trajectory), then analysed as above.
# GPU_LOCK held for the analysis (b53_run_arm.sh takes it for the advance).
set -u
ARM=$1; ADV=${2:-}
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
OUT=$ROOT/reports/b53/arm-$ARM
OPS=$ROOT/$CODE/bridge/ops
PY=$ROOT/.venv/bin/python
LOG=$ROOT/reports/b53/b53.log
DUMP=$OUT/b49-dump/learner_inputs.pt
if [ ! -f $DUMP ]; then
  [ "$ADV" = "advance" ] || { echo "no dump at $DUMP (rerun the arm with DUMP_AT=<update>, or pass 'advance')"; exit 2; }
  LAST=$(tail -1 $OUT/progress.csv | cut -d, -f1)
  HOURS=$(tail -1 $OUT/progress.csv | cut -d, -f3)
  STEP=$($PY -c "print($HOURS + 3*32768/27000)")   # (3 more updates of 32768 decisions)
  DUMP_AT=$((LAST+1)) CODE=$CODE sh $OPS/b53/b53_run_arm.sh $ARM $STEP
  [ -f $DUMP ] || { echo "the advance made no dump ($DUMP)"; exit 1; }
fi
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ]; do sleep 30; done
echo "b53-b49-$ARM $(date '+%F %T') B53 late B49 re-measure, arm $ARM" > $R/GPU_LOCK
echo "$(date '+%F %T') [$ARM] late B49 on $DUMP" >> $LOG
cd $OPS
timeout 5400 $PY b49_grad_stats.py $DUMP --chunks 40 --draws 8 --json $OUT/b49-late.json > $OUT/b49-late.log 2>&1
echo "$(date '+%F %T') [$ARM] b49 rc=$?" >> $LOG
PYTHONPATH=$ROOT/$CODE/bridge/python CUDA_VISIBLE_DEVICES= timeout 3600 $PY b48_coverage.py $DUMP --json $OUT/coverage-late.json > $OUT/coverage-late.log 2>&1
echo "$(date '+%F %T') [$ARM] coverage rc=$?" >> $LOG
grep -q "b53-b49-$ARM" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK
