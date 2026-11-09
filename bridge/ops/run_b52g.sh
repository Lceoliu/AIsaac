#!/bin/sh
# B52 full gate (GPU_LOCK protocol, guards): the whole test suite of the candidate tree (cpp driver on, the extension
# built into ~/.cache/isaac-actor), then the A8 play digest (8 seeds, 300 steps). usage: run_b52g.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b52}
LOG=$R/b52.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then echo "LOCKED: $(cat $R/GPU_LOCK)"; exit 2; fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B52 full gate: all tests + A8 play digest ($CODE, ~1.5 h)" > $R/GPU_LOCK
echo "$(date +%T) b52g start ($CODE)" >> $LOG
guard
cd $ROOT/$CODE/bridge/python
CUBLAS_WORKSPACE_CONFIG=:4096:8 ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 3600 $PY -m pytest tests -q -p no:cacheprovider > $R/tests-b52-full.log 2>&1
echo "$(date +%T) b52g full tests rc=$? $(tail -1 $R/tests-b52-full.log | cut -c1-120)" >> $LOG
guard
( export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
  timeout 1800 xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" $PY abplus_probe_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147500000:8 --steps 300 --out $R/b52-a8-play > $R/b52-a8-play.log 2>&1 )
echo "$(date +%T) b52g a8 rc=$? $(tail -1 $R/b52-a8-play.log | cut -c1-200)" >> $LOG
sleep 3
guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b52g lock released" >> $LOG
