#!/bin/sh
# B58d plan (not timing runs): why lane rows do not ride the main steps. After the d1 nsys / gate chain: the extension
# built, then 6 updates with ISAAC_BLOCK_CPP_LANES=0 (B54 merging on) and 6 with =1 (d1), each printing per collect the
# lane-only rows at the main version / an older one, merged rows put back, lane jobs queued inside / outside a main step.
# usage: run_merge_probe.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58d1m}
C=$ROOT/$CODE
PY=$ROOT/.venv/bin/python
LOG=$R/b58.log
trap 'rm -f $R/GPU_WANT_CPP' EXIT
waitlock() { touch $R/GPU_WANT_CPP; while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; rm -f $R/GPU_WANT_CPP; }
until grep -qE 'd1n (done|stops)' $R/b58.log; do sleep 30; done
while :; do waitlock; [ -f $R/GPU_LOCK ] || break; done
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B58d merge probe: build + 2 x 6 updates (not a timing run)" > $R/GPU_LOCK
echo "$(date +%T) mp start ($CODE)" >> $LOG
( cd $C/bridge/python && timeout 900 $PY -c "
from isaac_bridge import actor_native as a
p=a.build(True); print('BUILD', p)
" > $R/mp-build.log 2>&1 )
cd $ROOT
for A in lanes0 lanes1; do
  if ls -d $R/prof-mp-$A* > /dev/null 2>&1; then d=$R/failed/old-$(date +%m%d-%H%M%S); mkdir -p $d; mv $R/prof-mp-$A* $d/; fi
  L=0; [ "$A" = lanes1 ] && L=1
  ISAAC_BLOCK_CPP_LANES=$L CODE=$CODE ./prof2.sh prof-mp-$A 1800 7.4
  echo "$(date +%T) mp arm $A rc=$?" >> $LOG
done
rm -f $R/GPU_LOCK
for A in lanes0 lanes1; do
  awk -F, -v a=$A 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $h["update"]>=3 {cur+=$h["drv_lo_cur_rows"]; old+=$h["drv_lo_old_rows"]; mr+=$h["drv_merged_rows"]; pb+=$h["drv_putback_rows"]; ins+=$h["drv_enq_in_step"]; out+=$h["drv_enq_outside"]; n++}
    END {printf "%s (u3+, %d collects): lane-only rows at the main version %d, at an older one %d (%.1f%% current); merged rows %d, put back %d; jobs queued inside a main step %d, outside %d\n", a, n, cur, old, 100*cur/(cur+old+1e-9), mr, pb, ins, out}' $R/prof-mp-$A/progress.csv
done > $R/mp.summary.txt 2>&1
echo "$(date +%T) mp done" >> $LOG
