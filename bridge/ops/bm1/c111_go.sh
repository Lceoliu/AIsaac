#!/bin/sh
# gpu57 C111 (2026-11-16, the coordinator): after an arm's smoke chain (c111_smoke.sh <name>: STATUS_<name> "chain over"),
# the go / no-go of its 400-hour run: x_real (c111/gate_<name>.txt, the median after u2) >= 30 and the D1 lr-0 check
# written (c108/d1/<name>-smoke/d1_diag.jsonl: lines, every |dlogp| mean_abs < 0.005) -> the run c111/<name> (HOURS 400,
# CKPT_EVERY 100, c108_guard.sh, the arm's tree and lane), its c111_after.sh and its midway readings (c111_mid.sh); else
# "NO-GO" (a person looks). Status lines appended to c111/STATUS_<name>.
# (MID_ES=1 in the environment: the midway readings with the early stop, c111_mid.sh earlystop -- REF)
#   usage: [MID_ES=1] setsid nohup sh c111_go.sh <name> <tree> <lane file> <eval gpu> <cpuset> <mems> > /root/AIsaac/train/c111/go_<name>.out 2>&1 < /dev/null &
N=$1; T=$2; LANE=$3; G=$4; CS=$5; MS=$6
P=/root/AIsaac/train/c111; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" | tee -a $P/STATUS_$N; echo "$(date '+%F %T') c111_go $N: $*" >> $BM/reports/bm1.log; }
while ! grep -qE "chain over|FAILED" $P/STATUS_$N 2>/dev/null; do sleep 60; done
grep -q FAILED $P/STATUS_$N && { say "NO-GO: the smoke chain failed"; exit 1; }
X=$(sed -n 's/.*x_real: median after u2 \([0-9.]*\).*/\1/p' $P/gate_$N.txt)
D1=$(python3 -c "
import json,sys
L=[json.loads(l) for l in open('/root/AIsaac/train/c108/d1/$N-smoke/d1_diag.jsonl')]
m=max(r['all']['mean_abs'] for r in L)
print(len(L), '%.6f' % m, 'ok' if L and m < 0.005 else 'bad')" 2>/dev/null)
say "smoke x_real $X, D1 (lines, max mean |dlogp|) $D1"
if [ -z "$X" ] || [ "$(python3 -c "print(1 if float('$X') >= 30 else 0)")" != 1 ]; then say "NO-GO: x_real $X < 30 (a person looks)"; exit 1; fi
case "$D1" in *ok) ;; *) say "NO-GO: D1 $D1 (a person looks)"; exit 1 ;; esac
cd $BM
mkdir -p $P/$N
HOURS=400 CKPT_EVERY=100 MEM_GB=96 setsid nohup sh $BM/c108_guard.sh c111 $N $T $LANE > $P/$N.launch.out 2>&1 < /dev/null &
sleep 20
setsid nohup sh $BM/c111_after.sh $N $G $CS $MS > $P/after_$N.out 2>&1 < /dev/null &
setsid nohup sh $BM/c111_mid.sh $N ${MID_ES:+earlystop $G $CS $MS} > $P/mid_$N.out 2>&1 < /dev/null &
say "GO: run c111/$N started ($(tail -1 $P/$N/guard.log 2>/dev/null | cut -c1-100))"
