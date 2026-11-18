#!/bin/sh
# B57 measurement (no behaviour change): after the B54 runs (after_b54e.sh) and with the GPU lock free:
#  1. nsys nvtx + python-gil over hand-offs 2..7 of the cpp path (prof-b57m, run_b50m.sh), per-thread CPU sampled
#     from /proc meanwhile (b57_threads.py);
#  2. one collect under cProfile (prof-b57-cprof: the consumer's functions, ISAAC_BLOCK_ACTOR_PROFILE at hand-off 3);
#  3. the analysis (CPU only, after the lock is released): b57_nsys.py -> prof-b57m.analysis.txt.
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
CODE=${1:-candidate-b54}
LOG=$R/b57.log
OPS=$ROOT/$CODE/bridge/ops
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
waitlock() { while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; }
echo "$(date +%T) b57 queued: waiting for after_b54e.sh and the lock" >> $LOG
while pgrep -f '[a]fter_b54e.sh' > /dev/null; do sleep 30; done
waitlock
# 1. nsys (run_b50m.sh takes and releases the lock itself)
python3 $OPS/b57_threads.py $R/prof-b57m.threads.csv 1.0 > $R/prof-b57m.threads.log 2>&1 &
PYS='' TRIGGER=nvtx NSYS_RANGE=2:7 $OPS/run_b50m.sh prof-b57m $CODE 2400 10.3 nvtx,python-gil
echo "$(date +%T) b57 nsys done rc=$? $(ls -la $R/prof-b57m.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
wait
# 2. cProfile of one collect
waitlock
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B57 cProfile of one collect (prof-b57-cprof, 5 updates)" > $R/GPU_LOCK
guard
cd $ROOT
ISAAC_BLOCK_ACTOR_PROFILE=$R/b57-cprof ISAAC_BLOCK_ACTOR_PROFILE_AT=3 CODE=$CODE ./prof2.sh prof-b57-cprof 1800 6.1
echo "$(date +%T) b57 cprofile done rc=$?" >> $LOG
sleep 5; guard
rm -f $R/GPU_LOCK
echo "$(date +%T) b57 lock released" >> $LOG
# 3. analysis (no GPU)
python3 $OPS/b57_nsys.py $R/prof-b57m.nsys-rep $R/prof-b57m.threads.csv > $R/prof-b57m.analysis.txt 2>&1
echo "$(date +%T) b57 analysis rc=$? (b57 end)" >> $LOG
