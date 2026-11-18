#!/bin/sh
# B52 GPU gate: the holder probe, the cpp actor tests, the slot / lane integration tests (GPU_LOCK protocol, guards).
# usage: run_b52t.sh <code dir> [extra pytest args]
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b52}
LOG=$R/b52.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B52 gates: holder probe + cpp actor tests ($CODE, ~10 min)" > $R/GPU_LOCK
echo "$(date +%T) b52t start ($CODE)" >> $LOG
guard
if [ -d $ROOT/b52-probe ] && [ ! -f $R/b52-holder-probe.log ]; then
  (cd $ROOT/b52-probe && $PY probe_test.py > $R/b52-holder-probe.log 2>&1; echo "$(date +%T) holder probe rc=$? $(grep -E 'PROBE|iterations' $R/b52-holder-probe.log | tr '\n' ' ')" >> $LOG)
fi
cd $ROOT/$CODE/bridge/python
ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 1500 $PY -m pytest tests/test_b52_cpp_actor.py tests/test_block_slot_integration.py tests/test_block_lanes.py -q -p no:cacheprovider -x $2 > $R/tests-b52t.log 2>&1
echo "$(date +%T) b52t tests rc=$? $(tail -1 $R/tests-b52t.log)" >> $LOG
guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b52t lock released" >> $LOG
tail -40 $R/tests-b52t.log
