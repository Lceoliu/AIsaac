#!/bin/sh
# B52 condition 6 (GPU_LOCK protocol, waits for the lock, guards): the learner's long kernels on the B48 learner dump,
# torch profiler in the main thread (b52_learner_kernels.py), then b52_kernels.py. usage: run_b52k.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b52}
LOG=$R/b52.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
echo "$(date +%T) b52k queued behind: $(cat $R/GPU_LOCK 2>/dev/null)" >> $LOG
while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 30; done
sleep 10
[ -f $R/GPU_LOCK ] && exec "$0" "$@"
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B52 learner kernel durations (fixed-input replay of the B48 dump, ~20 min)" > $R/GPU_LOCK
echo "$(date +%T) b52k start ($CODE)" >> $LOG
guard
cd $ROOT/$CODE/bridge/ops
timeout 2400 $PY b52_learner_kernels.py $R/b48-dump/learner_inputs.pt $R/b52-kern2 --warm 2 > $R/b52-kern2.log 2>&1
echo "$(date +%T) b52k replay rc=$? $(tail -1 $R/b52-kern2.log | cut -c1-120)" >> $LOG
python3 b52_kernels.py $R/b52-kern2/trace.json 10 > $R/b52-kernels.txt 2>&1
echo "$(date +%T) b52k kernels: $(head -1 $R/b52-kernels.txt | cut -c1-160)" >> $LOG
guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b52k lock released" >> $LOG
