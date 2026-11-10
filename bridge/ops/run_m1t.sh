#!/bin/sh
# d2 M1 (not timing runs), after the B58e chain: the extension built, test_b58f_m1 alone, then the driver / lane tests
# (incl. B58e's), then the core TSAN (lineage + lanes state machine). Nothing timed; the M1 timing waits for review.
# GPU_WANT_CPP only while waiting for the lock. usage: run_m1t.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58f}
C=$ROOT/$CODE
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
take() { while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done; echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') $1" > $R/GPU_LOCK; }
passed() { tail -1 $1 | grep -q " passed" && ! tail -1 $1 | grep -qE "failed|error"; }
until grep -qE 'b58e (done|stops)' $LOG; do sleep 30; done
echo "$(date +%T) m1t start ($CODE)" >> $LOG
take "d2 M1: build + tests (not a timing run)"
cd $C/bridge/python
timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p); print(str(a.REASON)[-6000:] if p is None else '')
" > $R/m1t-build.log 2>&1
if ! grep -q "BUILD /" $R/m1t-build.log; then rm -f $R/GPU_LOCK; echo "$(date +%T) m1t stops: build failed" >> $LOG; exit 1; fi
timeout 900 $PY -m pytest tests/test_b58f_m1.py -v --tb=long -p no:cacheprovider -p no:logging > $R/tests-m1t-new.log 2>&1
echo "$(date +%T) m1t new tests: $(tail -1 $R/tests-m1t-new.log | cut -c1-100)" >> $LOG
if ! passed $R/tests-m1t-new.log; then rm -f $R/GPU_LOCK; echo "$(date +%T) m1t stops: new tests failed" >> $LOG; exit 1; fi
ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 2400 $PY -m pytest tests/test_b58e_batch.py tests/test_b58d_lanes.py tests/test_b58c_lineage.py tests/test_b54_driver_lanes.py tests/test_b52_cpp_actor.py tests/test_b58_native_consumer.py tests/test_b58_lifetime.py tests/test_b54_lane_bookkeeping.py -q --tb=long -p no:cacheprovider -p no:logging > $R/tests-m1t-drv.log 2>&1
rm -f $R/GPU_LOCK
echo "$(date +%T) m1t driver tests: $(tail -1 $R/tests-m1t-drv.log | cut -c1-100)" >> $LOG
take "d2 M1: core TSAN (CPU only; held so that no timing run shares the CPU)"
sh $C/bridge/ops/run_core_tsan.sh $CODE > $R/m1t-tsan.launcher.log 2>&1
rc=$?
rm -f $R/GPU_LOCK
echo "$(date +%T) m1t tsan rc=$rc: $(tr '\n' ' ' < $R/core-tsan.log | cut -c1-400)" >> $LOG
echo "$(date +%T) m1t done" >> $LOG
