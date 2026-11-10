#!/bin/sh
# B54 full gate (GPU_LOCK protocol, guards): the train_tok entry gate (tests/test_b52_entry.py: the production
# construction path to the first collect and update, driver on and ISAAC_BLOCK_CPP_ACTOR=0), the whole test suite of
# the candidate tree (cpp driver on, the extension built into ~/.cache/isaac-actor), then the A8 play digest (8 seeds,
# 300 steps). A failed part writes $R/b54-gate.failed: run_b54p.sh then refuses the profiles. usage: run_b54g.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b54}
LOG=$R/b54.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B54 full gate: all tests + A8 play digest ($CODE, ~1.8 h)" > $R/GPU_LOCK
echo "$(date +%T) b54g start ($CODE)" >> $LOG
rm -f $R/b54-gate.failed
guard
cd $ROOT/$CODE/bridge/python
ISAAC_B52_ENTRY=1 ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua timeout 3600 \
  $PY -m pytest tests/test_b52_entry.py -q -p no:cacheprovider --basetemp=$R/b54-entry > $R/tests-b54-entry.log 2>&1
echo "$(date +%T) b54g entry gate rc=$? $(tail -1 $R/tests-b54-entry.log | cut -c1-120)" >> $LOG
tail -1 $R/tests-b54-entry.log | grep -q " passed" && ! tail -1 $R/tests-b54-entry.log | grep -qE "failed|skipped|error" || echo "entry gate" >> $R/b54-gate.failed
guard
CUBLAS_WORKSPACE_CONFIG=:4096:8 ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 3600 $PY -m pytest tests -v --tb=long -p no:cacheprovider > $R/tests-b54-full.log 2>&1
echo "$(date +%T) b54g full tests rc=$? $(tail -1 $R/tests-b54-full.log | cut -c1-120)" >> $LOG
# (B58c: -v keeps the test order; a failed run's whole log is kept with a time stamp, e.g. for the flaky capture error
# f370e1d only mitigates)
if ! { tail -1 $R/tests-b54-full.log | grep -q " passed" && ! tail -1 $R/tests-b54-full.log | grep -qE "failed|error"; }; then
  echo "full tests" >> $R/b54-gate.failed
  mkdir -p $R/failed && cp $R/tests-b54-full.log $R/failed/tests-b54-full-$(date +%m%d-%H%M%S).log
fi
guard
( export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
  timeout 1800 xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" $PY abplus_probe_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147500000:8 --steps 300 --out $R/b54-a8-play > $R/b54-a8-play.log 2>&1 )
echo "$(date +%T) b54g a8 rc=$? $(tail -1 $R/b54-a8-play.log | cut -c1-200)" >> $LOG
tail -1 $R/b54-a8-play.log | grep -q '"seeds": 8, "ok": 8' || echo "A8" >> $R/b54-gate.failed
[ -f $R/b54-gate.failed ] && echo "$(date +%T) b54g GATE FAILED: $(tr '\n' ' ' < $R/b54-gate.failed)" >> $LOG
sleep 3
guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b54g lock released" >> $LOG
