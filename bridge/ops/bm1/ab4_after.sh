#!/bin/sh
# gpu57 ab4 (C109, 2026-11-15): after one ab4 arm (c108_guard.sh ab4 <name> ...), as ab3_after.sh, for that arm alone.
# When its guard.log says the 400 game hours were reached: the 100-update window table of the arm, and once (the first
# arm to end writes them) the comparison tables of ab2-bm, ab3-lr and ab2-c99old up to u330 (ab2_win.py <dir> 100 330);
# then the held-out evaluation of the arm's last.pt (ev_ab2.sh with PARENT=ab4, kind new as ab2-bm's: seeds
# 2147490000:128, 14 workers) on the arm's actor GPU and CPUs. Short of its hours: neither (a person looks).
# Its status in /root/AIsaac/train/ab4/AFTER_DONE_<name>.
#   usage: setsid nohup sh ab4_after.sh <name> <eval gpu> <cpuset> <mems> > /root/AIsaac/train/ab4/after_<name>.out 2>&1 < /dev/null &
N=$1; G=$2; CS=$3; MS=$4
P=/root/AIsaac/train/ab4
BM=/root/AIsaac/bm1
while pgrep -f "c108_guard.sh ab4 $N " > /dev/null; do sleep 60; done
echo "$(date '+%F %T') ab4_after $N: training ended" >> $BM/reports/bm1.log
st=""
if grep -q "game hours reached" $P/$N/guard.log; then
  python3 $BM/ab2_win.py $P/$N/$N-s0 100 > $P/win100_$N.txt 2>&1
  [ -f $P/win100_ab2bm_u330.txt ] || python3 $BM/ab2_win.py /root/AIsaac/train/ab2/ab2-bm/ab2-bm-s0 100 330 > $P/win100_ab2bm_u330.txt 2>&1
  [ -f $P/win100_ab3lr.txt ] || python3 $BM/ab2_win.py /root/AIsaac/train/ab3/ab3-lr/ab3-lr-s0 100 330 > $P/win100_ab3lr.txt 2>&1
  [ -f $P/win100_ab2c99old_u330.txt ] || python3 $BM/ab2_win.py /root/AIsaac/train/ab2/ab2-c99old/ab2-c99old-s0 100 330 > $P/win100_ab2c99old_u330.txt 2>&1
  PARENT=ab4 sh $BM/ev_ab2.sh $N new $G $CS $MS 14 $N-c128-last,last.pt,2147490000:128 > $P/eval_$N.out 2>&1
else st=" $N short of its hours ($(tail -1 $P/$N/guard.log | cut -c1-120));"; fi
n=$(ls $P/eval/$N-c128-last/summary.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') evaluation done: $n/1 summary.json;$st" > $P/AFTER_DONE_$N
echo "$(date '+%F %T') ab4_after $N: evaluation done, $n/1 summary.json;$st" >> $BM/reports/bm1.log
