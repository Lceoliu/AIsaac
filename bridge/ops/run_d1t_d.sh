#!/bin/sh
# B58d d1 timing, arm D (after A, B, C of run_d1t.sh; measurement only, its correctness untested): ISAAC_BLOCK_CPP_LANES=1
# + ISAAC_BLOCK_CPP_LANE_THREAD=1 (the native lane service on a thread of its own), 24 updates. A time limit of 1.5 x
# arm A's wall time; past it, or on a crash, the outputs go to failed/ and the lock is released (no debugging here).
# Self-checks for the summary: train_tok tracebacks, cf_dropped, lane answers (native) vs the workers' lane calls.
# usage: run_d1t_d.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58d1}
C=$ROOT/$CODE
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
A=lanethread
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
until grep -qE 'd1t done|d1t stops' $R/b58.log 2>/dev/null && [ "$(sed -n '/d1t: restarted with three arms/,$p' $R/b58.log | grep -cE 'd1t done|d1t stops')" != "0" ]; do sleep 30; done
if sed -n '/d1t: restarted with three arms/,$p' $R/b58.log | grep -q 'd1t stops'; then echo "$(date +%T) d1t D: not run (the chain stopped)" >> $LOG; exit 1; fi
# arm A's wall time (its samples' first and last time) x 1.5
DUR=$(awk -F, 'NR==2 {a=$1} NR>1 {b=$1} END {print b-a}' $R/prof-d1t-lanes1.samples.csv)
LIM=$(( (DUR > 600 ? DUR : 600) * 3 / 2 ))
if ls -d $R/prof-d1t-$A* > /dev/null 2>&1; then d=$R/failed/old-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/prof-d1t-$A* $d/; fi
# arm D's own tree: arm A-C's candidate + the lane-thread fix (lane_timing pauses the lane thread), built before the
# timed run (not a timing run)
D=${CODE}d
[ -d $ROOT/$D ] || cp -a $C $ROOT/$D
[ -f $ROOT/d1t-d-tree.tgz ] && tar xzf $ROOT/d1t-d-tree.tgz -C $ROOT/$D && mv $ROOT/d1t-d-tree.tgz $R/failed/d1t-d-tree-deployed.tgz
while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B58d d1 arm D: build (not a timing run)" > $R/GPU_LOCK
( cd $ROOT/$D/bridge/python && timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p); print(str(a.REASON)[-3000:] if p is None else '')
" > $R/d1t-d-build.log 2>&1 )
rm -f $R/GPU_LOCK
CODE=$D; C=$ROOT/$D
while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') 计时运行 B58d d1: $A (arm D, measurement only), 24 updates, limit ${LIM}s ($CODE)" > $R/GPU_LOCK
echo "$(date +%T) d1t D start (limit ${LIM} s = 1.5 x arm A's ${DUR} s)" >> $LOG
cd $ROOT
( echo "t,isaac,mem_avail_kb" > $R/prof-d1t-$A.samples.csv
  while [ -f $R/GPU_LOCK ] && grep -q "d1: $A" $R/GPU_LOCK; do
    echo "$(date +%s),$(pgrep -c isaac),$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" >> $R/prof-d1t-$A.samples.csv; sleep 30
  done ) &
( sleep 400; sh $C/bridge/ops/threads_snapshot.sh > $R/prof-d1t-$A.threadnames.txt 2>&1 ) &
ISAAC_BLOCK_CPP_LANES=1 ISAAC_BLOCK_CPP_LANE_THREAD=1 CODE=$CODE timeout --signal=INT --kill-after=60s $LIM ./prof2.sh prof-d1t-$A $LIM 29.0
rc=$?
N=$(($(wc -l < $R/prof-d1t-$A/progress.csv 2>/dev/null || echo 1)-1))
TB=$(grep -cE 'File "[^"]*train_tok\.py"' $R/prof-d1t-$A.stdout $R/prof-d1t-$A.stderr 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
rm -f $R/GPU_LOCK
wait
if [ $rc -ne 0 ] || [ "$N" -lt 24 ] || [ "$TB" != "0" ]; then
  d=$R/failed/d1t-D-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/prof-d1t-$A* $d/
  echo "$(date +%T) d1t D unusable: rc=$rc, $N updates, $TB tracebacks; outputs in $d" >> $LOG
  echo "$(date +%T) d1t D done" >> $LOG; exit 1
fi
{ echo "== arm D ($A): measurement only, correctness untested"
  echo "self-checks: tracebacks $TB, $(grep -m1 'B58d native lanes' $R/prof-d1t-$A.stdout | cut -c1-80)"
  $PY $C/bridge/ops/cf_rates.py $R/prof-d1t-$A
  $PY $C/bridge/ops/b58_ab_summary.py $R/prof-d1t-$A
  awk -F, 'NR==1 {for(i=1;i<=NF;i++) h[$i]=i} NR>1 {last=$0} END {split(last, v, ","); printf "u%s totals: cf_dropped %s, cf_lane_calls %s\n", v[h["update"]], v[h["cf_dropped"]], v[h["cf_lane_calls"]]}' $R/prof-d1t-$A/progress.csv
  awk -F, 'NR==1 {for(i=1;i<=NF;i++) h[$i]=i} NR>1 {s+=$h["lt_rt_s_n"]} END {printf "native lane answers (sum of lt_rt_s_n) %d\n", s}' $R/prof-d1t-$A/progress.csv
  awk -F, 'NR>1 { if ($2 > p) p = $2; if (m == "" || $3 < m) m = $3; n++ } END { printf "%d samples, isaac processes peak %d, MemAvailable min %.1f GiB\n", n, p, m/1048576 }' $R/prof-d1t-$A.samples.csv
  head -4 $R/prof-d1t-$A.threadnames.txt; } > $R/prof-d1t-D.summary.txt 2>&1
echo "$(date +%T) d1t D done" >> $LOG
