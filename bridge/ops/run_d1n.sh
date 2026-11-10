#!/bin/sh
# B58d d1, after the timing arms: (not timed) the extension built, lane.submit's longest occurrence in prof-b58c-m
# dissected; (timed) a 6-update nsys nvtx + python-gil profile with ISAAC_BLOCK_CPP_LANES=1 and the per-thread CPU
# sampler, its analysis; (not timed) the full gate (entry + all tests + A8) with ISAAC_BLOCK_CPP_LANES=1.
# GPU_WANT_CPP only while waiting for the lock. usage: run_d1n.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58d1n}
C=$ROOT/$CODE
OPS=$C/bridge/ops
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
take() { while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done; echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') $1" > $R/GPU_LOCK; }
fresh() { for x in "$@"; do if ls -d $R/$x* > /dev/null 2>&1; then d=$R/failed/old-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/$x* $d/; echo "$(date +%T) d1n: earlier $x* moved to $d" >> $LOG; fi; done; }
until grep -qE 'd1t D (done|unusable)' $R/b58.log; do sleep 30; done
echo "$(date +%T) d1n start ($CODE)" >> $LOG
take "B58d d1n: build, lane.submit dissection (not a timing run)"
( cd $C/bridge/python && timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p); print(str(a.REASON)[-3000:] if p is None else '')
" > $R/d1n-build.log 2>&1 )
python3 $OPS/b58_maxrange.py $R/failed/old-1110-222315/prof-b58c-m.sqlite lane.submit 3 > $R/d1n-submit-max.txt 2>&1 || \
  python3 $OPS/b58_maxrange.py $R/prof-b58c-m.sqlite lane.submit 3 > $R/d1n-submit-max.txt 2>&1
rm -f $R/GPU_LOCK
echo "$(date +%T) d1n build: $(grep -c BUILD $R/d1n-build.log) ok line(s)" >> $LOG
fresh prof-d1-m
waitlock
python3 $OPS/b57_threads.py $R/prof-d1-m.threads.csv 1.0 > $R/prof-d1-m.threads.log 2>&1 &
( sleep 240; sh $OPS/threads_snapshot.sh > $R/prof-d1-m.threadnames.txt 2>&1 ) &
ISAAC_BLOCK_CPP_LANES=1 PYS='' TRIGGER=nvtx NSYS_RANGE=2:5 $OPS/run_b50m.sh prof-d1-m $CODE 1800 7.4 nvtx,python-gil
echo "$(date +%T) d1n nsys rc=$?" >> $LOG
wait
python3 $OPS/b57_nsys.py $R/prof-d1-m.nsys-rep $R/prof-d1-m.threads.csv > $R/prof-d1-m.analysis.txt 2>&1
echo "$(date +%T) d1n analysis rc=$?" >> $LOG
take "B58d d1n: full gate with CPP_LANES=1 (not a timing run)"
rm -f $R/GPU_LOCK   # (run_b54g.sh takes the lock itself; the take above only waited our turn)
ISAAC_BLOCK_CPP_LANES=1 $OPS/run_b54g.sh $CODE > $R/d1n-gate.launcher.log 2>&1
if [ -f $R/b54-gate.failed ]; then echo "$(date +%T) d1n stops: full gate failed ($(tr '\n' ' ' < $R/b54-gate.failed))" >> $LOG; exit 1; fi
echo "$(date +%T) d1n gate: $(tail -1 $R/tests-b54-entry.log | cut -c1-50) | $(tail -1 $R/tests-b54-full.log | cut -c1-70) | a8 $(tail -1 $R/b54-a8-play.log | grep -o '"seeds": 8, "ok": [0-9]*')" >> $LOG
echo "$(date +%T) d1n done" >> $LOG
