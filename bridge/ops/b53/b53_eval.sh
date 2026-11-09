#!/bin/sh
# B53 evaluation protocol: one checkpoint on the 512 held-out run seeds (arms.json eval), run mode, sampled actions
# (the trained stochastic policy), 16 workers. Repeats of the same checkpoint are separate output directories.
#   usage: b53_eval.sh <checkpoint> <out_dir> [extra eval_tok args]
#   env:   CODE=<candidate dir under $ROOT>
# GPU_LOCK: waits until free (and nothing trains / tests), holds it for the evaluation.
set -u
CK=$1; OUTD=$2; shift 2
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
LOG=$ROOT/reports/b53/b53.log
mkdir -p $ROOT/reports/b53
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 30; done
echo "b53-eval $(date '+%F %T') B53 held-out evaluation $CK -> $OUTD" > $R/GPU_LOCK
cd $ROOT/$CODE/bridge/python
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
SEEDS=$($PY -c "import json;print(json.load(open('$ROOT/$CODE/bridge/ops/b53/arms.json'))['eval']['seeds'])")
echo "$(date '+%F %T') [eval] start $CK -> $OUTD seeds $SEEDS" >> $LOG
xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT 14400 \
  $PY eval_tok.py --checkpoint $CK --groups-file ../abplus/catalog/scaling2_groups.json --groups normal:16 \
  --mode run --run-seconds 1800 --seeds $SEEDS --port 39800 --name b53e --out $OUTD "$@" > $OUTD.stdout 2> $OUTD.stderr
rc=$?
echo "$(date '+%F %T') [eval] done rc=$rc $OUTD" >> $LOG
grep -q "b53-eval" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK
exit $rc
