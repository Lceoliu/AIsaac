#!/bin/sh
# gpu57 C110 (2026-11-15): after one C110 arm (c108_guard.sh c110 <name> ...), as ab4_after.sh. When its guard.log says
# the 400 game hours were reached: the arm's 100-update window table (ab2_win.py <dir> 100 -> win100_<name>.txt), then
# the held-out evaluation of its last.pt exactly as C109's (ev_ab2.sh PARENT=c110, kind new: tree maa1e98a's eval_tok.py
# through evtools/eval_ent.py, entropy.json, seeds 2147490000:128, 14 workers, TOK_EVAL_DEVICE=cuda) on the arm's actor
# GPU and CPUs. Short of its hours: neither (a person looks). Status in /root/AIsaac/train/c110/AFTER_DONE_<name> (the
# fp32 chain, c110_fp32.sh, waits for it: the lane is free then).
# (the c45a583 arms: EV_TREE=mc45a583 EV_ABHOME=home-c45 in the environment -> ev_ab2.sh: their own tree's eval_tok.py)
#   usage: setsid nohup sh c110_after.sh <name> <eval gpu> <cpuset> <mems> > /root/AIsaac/train/c110/after_<name>.out 2>&1 < /dev/null &
N=$1; G=$2; CS=$3; MS=$4
P=/root/AIsaac/train/c110
BM=/root/AIsaac/bm1
while pgrep -f "c108_guard.sh c110 $N " > /dev/null; do sleep 60; done
echo "$(date '+%F %T') c110_after $N: training ended" >> $BM/reports/bm1.log
st=""
if grep -q "game hours reached" $P/$N/guard.log; then
  python3 $BM/ab2_win.py $P/$N/$N-s0 100 > $P/win100_$N.txt 2>&1
  PARENT=c110 sh $BM/ev_ab2.sh $N new $G $CS $MS 14 $N-c128-last,last.pt,2147490000:128 > $P/eval_$N.out 2>&1
else st=" $N short of its hours ($(tail -1 $P/$N/guard.log | cut -c1-120));"; fi
n=$(ls $P/eval/$N-c128-last/summary.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') evaluation done: $n/1 summary.json;$st" > $P/AFTER_DONE_$N
echo "$(date '+%F %T') c110_after $N: evaluation done, $n/1 summary.json;$st" >> $BM/reports/bm1.log
