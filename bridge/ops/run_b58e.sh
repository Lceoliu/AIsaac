#!/bin/sh
# B58e: (not timed) the extension built, test_b58e_batch alone, then the driver / entry tests; (nsys, not a timing
# run) 6 updates with nvtx + python-gil, the main thread's GIL acquisitions and held seconds per cycle, the processes by
# name / parent sampled meanwhile; (not timed) the full gate (entry + all tests + A8); then, back to back in one quiet
# stretch, two arms of 24 updates each, no nsys: E (batched records, the default) and A (ISAAC_BLOCK_CPP_BATCH=0: a
# record per step, the d1 configuration), every 30 s the isaac processes and MemAvailable sampled; the summaries.
# GPU_WANT_CPP only while waiting for the lock. usage: run_b58e.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58e}
C=$ROOT/$CODE
OPS=$C/bridge/ops
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
take() { while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done; echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') $1" > $R/GPU_LOCK; }
fresh() { for x in "$@"; do if ls -d $R/$x* > /dev/null 2>&1; then d=$R/failed/old-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/$x* $d/; echo "$(date +%T) b58e: earlier $x* moved to $d" >> $LOG; fi; done; }
procs() { ps -eo ppid=,comm= | awk '{c[$1" "$2]++} END {for (k in c) print c[k], k}' | sort -rn | head -25; }
passed() { tail -1 $1 | grep -q " passed" && ! tail -1 $1 | grep -qE "failed|error"; }
echo "$(date +%T) b58e start ($CODE)" >> $LOG
take "B58e: build + tests (not a timing run)"
cd $C/bridge/python
timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p); print(str(a.REASON)[-4000:] if p is None else '')
" > $R/b58e-build.log 2>&1
timeout 900 $PY -m pytest tests/test_b58e_batch.py -v --tb=long -p no:cacheprovider -p no:logging > $R/tests-b58e-new.log 2>&1
echo "$(date +%T) b58e new tests: $(tail -1 $R/tests-b58e-new.log | cut -c1-100)" >> $LOG
if ! passed $R/tests-b58e-new.log; then rm -f $R/GPU_LOCK; echo "$(date +%T) b58e stops: new tests failed" >> $LOG; exit 1; fi
ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 2400 $PY -m pytest tests/test_b52_entry.py tests/test_b58d_lanes.py tests/test_b58c_lineage.py tests/test_b54_driver_lanes.py tests/test_b52_cpp_actor.py tests/test_b58_native_consumer.py tests/test_b58_lifetime.py tests/test_b54_lane_bookkeeping.py -q -p no:cacheprovider -p no:logging > $R/tests-b58e-drv.log 2>&1
rm -f $R/GPU_LOCK
echo "$(date +%T) b58e driver + entry tests: $(tail -1 $R/tests-b58e-drv.log | cut -c1-100)" >> $LOG
if ! passed $R/tests-b58e-drv.log; then echo "$(date +%T) b58e stops: driver / entry tests failed" >> $LOG; exit 1; fi
cd $ROOT
fresh prof-e-m
waitlock
python3 $OPS/b57_threads.py $R/prof-e-m.threads.csv 1.0 > $R/prof-e-m.threads.log 2>&1 &
( sleep 240; sh $OPS/threads_snapshot.sh > $R/prof-e-m.threadnames.txt 2>&1; procs > $R/prof-e-m.procs.txt 2>&1 ) &
PYS='' TRIGGER=nvtx NSYS_RANGE=2:5 $OPS/run_b50m.sh prof-e-m $CODE 1800 7.4 nvtx,python-gil
echo "$(date +%T) b58e nsys rc=$?" >> $LOG
wait
python3 $OPS/b57_nsys.py $R/prof-e-m.nsys-rep $R/prof-e-m.threads.csv > $R/prof-e-m.analysis.txt 2>&1
echo "$(date +%T) b58e analysis rc=$?: $(grep 'GIL acquisitions' $R/prof-e-m.analysis.txt | cut -c1-120)" >> $LOG
while :; do   # (run_b54g.sh takes the lock itself; exit 2: someone else had it)
  waitlock; $OPS/run_b54g.sh $CODE > $R/b58e-gate.launcher.log 2>&1; rc=$?; [ $rc -eq 2 ] || break; sleep 30
done
if [ -f $R/b54-gate.failed ]; then echo "$(date +%T) b58e stops: full gate failed ($(tr '\n' ' ' < $R/b54-gate.failed))" >> $LOG; exit 1; fi
echo "$(date +%T) b58e gate: $(tail -1 $R/tests-b54-entry.log | cut -c1-50) | $(tail -1 $R/tests-b54-full.log | cut -c1-70) | a8 $(tail -1 $R/b54-a8-play.log | grep -o '"seeds": 8, "ok": [0-9]*')" >> $LOG
for A in batch1 batch0; do
  fresh prof-e-$A
  take "计时运行 B58e: $A, 24 updates ($CODE)"
  B=1; [ "$A" = batch0 ] && B=0
  ( sleep 400; sh $OPS/threads_snapshot.sh > $R/prof-e-$A.threadnames.txt 2>&1; procs > $R/prof-e-$A.procs.txt 2>&1 ) &
  ( echo "t,isaac,mem_avail_kb" > $R/prof-e-$A.samples.csv
    while [ -f $R/GPU_LOCK ] && grep -q "B58e: $A" $R/GPU_LOCK; do
      echo "$(date +%s),$(pgrep -c isaac),$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" >> $R/prof-e-$A.samples.csv; sleep 30
    done ) &
  ISAAC_BLOCK_CPP_BATCH=$B CODE=$CODE ./prof2.sh prof-e-$A 6000 29.0
  rc=$?
  rm -f $R/GPU_LOCK
  wait
  echo "$(date +%T) b58e arm $A done rc=$rc, $(($(wc -l < $R/prof-e-$A/progress.csv 2>/dev/null || echo 1)-1)) updates, $(grep -m1 'B58e batched' $R/prof-e-$A.stdout | cut -c1-80)" >> $LOG
done
{ $PY $OPS/cf_rates.py $R/prof-e-batch1 $R/prof-e-batch0
  $PY $OPS/b58_ab_summary.py $R/prof-e-batch1 $R/prof-e-batch0
  for A in batch1 batch0; do
    awk -F, -v a=$A 'NR>1 { if ($2 > p) p = $2; if (m == "" || $3 < m) m = $3; n++ } END { printf "%s: %d samples, isaac processes peak %d, MemAvailable min %.1f GiB\n", a, n, p, m/1048576 }' $R/prof-e-$A.samples.csv
    head -4 $R/prof-e-$A.threadnames.txt
  done; } > $R/prof-e.summary.txt 2>&1
echo "$(date +%T) b58e done (summary rc=$?)" >> $LOG
