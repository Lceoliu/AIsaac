#!/bin/sh
# B54 GPU gate: the cpp actor tests (B52 + B54 lanes), the slot / lane integration tests (GPU_LOCK protocol, guards).
# usage: run_b54t.sh <code dir> [extra pytest args]
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b54}
LOG=$R/b54.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B54 gates: cpp actor + lane driver tests ($CODE, ~10 min)" > $R/GPU_LOCK
echo "$(date +%T) b54t start ($CODE)" >> $LOG
guard
if false; then
  (cd $ROOT/b54-probe && $PY probe_test.py > $R/b54-holder-probe.log 2>&1; echo "$(date +%T) holder probe rc=$? $(grep -E 'PROBE|iterations' $R/b54-holder-probe.log | tr '\n' ' ')" >> $LOG)
fi
cd $ROOT/$CODE/bridge/python
ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 1500 $PY -m pytest tests/test_b52_cpp_actor.py tests/test_b54_driver_lanes.py tests/test_block_slot_integration.py tests/test_block_lanes.py -q -p no:cacheprovider -x $2 > $R/tests-b54t.log 2>&1
echo "$(date +%T) b54t tests rc=$? $(tail -1 $R/tests-b54t.log)" >> $LOG
guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b54t lock released" >> $LOG
tail -40 $R/tests-b54t.log
