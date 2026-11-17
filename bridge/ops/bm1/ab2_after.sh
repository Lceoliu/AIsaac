#!/bin/sh
# gpu57 ab2 (C107, 2026-11-14): after the pair, as ab1_after.sh. Waits for ab2_pair.sh ab2 to end; then for each arm
# whose guard.log says the 1,200 game hours were reached: its window table (ab2_win.py, 200 updates) and the held-out
# evaluations of update-000500.pt and last.pt (ev_ab2.sh, seeds 2147490000:128) -- arm A (old: C99's protocol, the CPU)
# on GPU 4 + node0's CPUs, arm B (new: as ev_ab1, 14 workers) on GPU 6 + node1's, the two arms in parallel. An arm short
# of its hours gets neither (a person looks). Its status in /root/AIsaac/train/ab2/AFTER_DONE at the end.
#   usage: setsid nohup sh ab2_after.sh > /root/AIsaac/train/ab2/after.out 2>&1 < /dev/null &
P=/root/AIsaac/train/ab2
BM=/root/AIsaac/bm1
while pgrep -f "ab2_pair.sh ab2 " > /dev/null; do sleep 60; done
echo "$(date '+%F %T') ab2_after: training ended" >> $BM/reports/bm1.log
st=""
if grep -q "game hours reached" $P/ab2-c99old/guard.log; then
  python3 $BM/ab2_win.py $P/ab2-c99old/ab2-c99old-s0 > $P/win_c99old.txt 2>&1
  sh $BM/ev_ab2.sh ab2-c99old old 4 0-7,16-23 0 8 c99old-c128-u0500,update-000500.pt,2147490000:128 c99old-c128-last,last.pt,2147490000:128 > $P/eval_c99old.out 2>&1 &
  E1=$!
else st="$st ab2-c99old short of its hours ($(tail -1 $P/ab2-c99old/guard.log | cut -c1-120));"; E1=""; fi
sleep 30
if grep -q "game hours reached" $P/ab2-bm/guard.log; then
  python3 $BM/ab2_win.py $P/ab2-bm/ab2-bm-s0 > $P/win_bm.txt 2>&1
  sh $BM/ev_ab2.sh ab2-bm new 6 8-15,24-31 1 14 bm-c128-u0500,update-000500.pt,2147490000:128 bm-c128-last,last.pt,2147490000:128 > $P/eval_bm.out 2>&1 &
  E2=$!
else st="$st ab2-bm short of its hours ($(tail -1 $P/ab2-bm/guard.log | cut -c1-120));"; E2=""; fi
[ -n "$E1" ] && wait $E1
[ -n "$E2" ] && wait $E2
n=$(ls $P/eval/*/summary.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') evaluations done: $n/4 summary.json;$st" > $P/AFTER_DONE
echo "$(date '+%F %T') ab2_after: evaluations done, $n/4 summary.json;$st" >> $BM/reports/bm1.log
