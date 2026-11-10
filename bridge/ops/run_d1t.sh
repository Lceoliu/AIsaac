#!/bin/sh
# B58d d1 timing: first (not a timing run) the extension built and the d1 / B58c / B54 driver tests; then, back to
# back in one quiet stretch, three arms of 24 updates each, no nsys: A ISAAC_BLOCK_CPP_LANES=1 (the native lane service
# on the driver thread), B =0 (the B58c configuration: native lineage, the B54 Python lane service), C
# ISAAC_BLOCK_CPP_LINEAGE=0 (B58a+b on the same tree); every 30 s the isaac processes and MemAvailable sampled; then the
# summaries (cf_rates.py, b58_ab_summary.py, the samples' peak / minimum).
# GPU_WANT_CPP only while waiting for the lock. usage: run_d1t.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58d1}
C=$ROOT/$CODE
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
take() { while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done; echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') $1" > $R/GPU_LOCK; }
fresh() { for x in "$@"; do if ls -d $R/$x* > /dev/null 2>&1; then d=$R/failed/old-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/$x* $d/; echo "$(date +%T) d1t: earlier $x* moved to $d" >> $LOG; fi; done; }
echo "$(date +%T) d1t start ($CODE)" >> $LOG
take "B58d d1: build + tests (not a timing run)"
cd $C/bridge/python
timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p); print(str(a.REASON)[-4000:] if p is None else '')
" > $R/d1t-build.log 2>&1
ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 1800 $PY -m pytest tests/test_b58d_lanes.py tests/test_b58c_lineage.py tests/test_b54_driver_lanes.py tests/test_b52_cpp_actor.py tests/test_b58_native_consumer.py tests/test_b58_lifetime.py tests/test_b54_lane_bookkeeping.py -q -p no:cacheprovider -p no:logging > $R/tests-d1t.log 2>&1
rm -f $R/GPU_LOCK
echo "$(date +%T) d1t tests: $(tail -1 $R/tests-d1t.log | cut -c1-100)" >> $LOG
if ! tail -1 $R/tests-d1t.log | grep -q " passed" || tail -1 $R/tests-d1t.log | grep -qE "failed|error"; then
  echo "$(date +%T) d1t stops: tests failed" >> $LOG; exit 1
fi
cd $ROOT
for A in lanes1 lanes0 lin0; do
  fresh prof-d1t-$A
  take "计时运行 B58d d1: $A, 24 updates ($CODE)"
  L=0; LIN=1; [ "$A" = lanes1 ] && L=1; [ "$A" = lin0 ] && LIN=0
  ( sleep 400; sh $C/bridge/ops/threads_snapshot.sh > $R/prof-d1t-$A.threadnames.txt 2>&1 ) &
  ( echo "t,isaac,mem_avail_kb" > $R/prof-d1t-$A.samples.csv
    while [ -f $R/GPU_LOCK ] && grep -q "d1: $A" $R/GPU_LOCK; do
      echo "$(date +%s),$(pgrep -c isaac),$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" >> $R/prof-d1t-$A.samples.csv; sleep 30
    done ) &
  ISAAC_BLOCK_CPP_LANES=$L ISAAC_BLOCK_CPP_LINEAGE=$LIN CODE=$CODE ./prof2.sh prof-d1t-$A 6000 29.0
  rc=$?
  rm -f $R/GPU_LOCK
  wait
  echo "$(date +%T) d1t arm $A done rc=$rc, $(($(wc -l < $R/prof-d1t-$A/progress.csv 2>/dev/null || echo 1)-1)) updates, $(grep -m1 'B58d native lanes' $R/prof-d1t-$A.stdout | cut -c1-80)" >> $LOG
done
{ $PY $C/bridge/ops/cf_rates.py $R/prof-d1t-lanes1 $R/prof-d1t-lanes0 $R/prof-d1t-lin0
  $PY $C/bridge/ops/b58_ab_summary.py $R/prof-d1t-lanes1 $R/prof-d1t-lanes0 $R/prof-d1t-lin0
  for A in lanes1 lanes0 lin0; do
    awk -F, -v a=$A 'NR>1 { if ($2 > p) p = $2; if (m == "" || $3 < m) m = $3; n++ } END { printf "%s: %d samples, isaac processes peak %d, MemAvailable min %.1f GiB\n", a, n, p, m/1048576 }' $R/prof-d1t-$A.samples.csv
    head -4 $R/prof-d1t-$A.threadnames.txt
  done; } > $R/prof-d1t.summary.txt 2>&1
echo "$(date +%T) d1t done (summary rc=$?)" >> $LOG
