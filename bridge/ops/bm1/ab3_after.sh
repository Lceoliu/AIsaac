#!/bin/sh
# gpu57 ab3 (C108 D3, 2026-11-15): after ab3-lr (c108_guard.sh ab3 ...), as ab2_after.sh. When its guard.log says the
# 400 game hours were reached: the 100-update window tables of ab3-lr and, for comparison, the first 330 updates of
# ab2-bm and ab2-c99old (ab2_win.py <dir> 100 330: ab3-lr's 400 game hours are ~330 updates); then the held-out
# evaluation of ab3-lr's last.pt (ev_ab2.sh with PARENT=ab3, kind new as ab2-bm's: seeds 2147490000:128, 14 workers,
# GPU 6 + node1's CPUs). Short of its hours: neither (a person looks). Its status in /root/AIsaac/train/ab3/AFTER_DONE.
#   usage: setsid nohup sh ab3_after.sh > /root/AIsaac/train/ab3/after.out 2>&1 < /dev/null &
P=/root/AIsaac/train/ab3
BM=/root/AIsaac/bm1
while pgrep -f "c108_guard.sh ab3 ab3-lr" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') ab3_after: training ended" >> $BM/reports/bm1.log
st=""
if grep -q "game hours reached" $P/ab3-lr/guard.log; then
  python3 $BM/ab2_win.py $P/ab3-lr/ab3-lr-s0 100 > $P/win100_ab3lr.txt 2>&1
  python3 $BM/ab2_win.py /root/AIsaac/train/ab2/ab2-bm/ab2-bm-s0 100 330 > $P/win100_ab2bm_u330.txt 2>&1
  python3 $BM/ab2_win.py /root/AIsaac/train/ab2/ab2-c99old/ab2-c99old-s0 100 330 > $P/win100_ab2c99old_u330.txt 2>&1
  PARENT=ab3 sh $BM/ev_ab2.sh ab3-lr new 6 8-15,24-31 1 14 ab3lr-c128-last,last.pt,2147490000:128 > $P/eval_ab3lr.out 2>&1
else st=" ab3-lr short of its hours ($(tail -1 $P/ab3-lr/guard.log | cut -c1-120));"; fi
n=$(ls $P/eval/*/summary.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') evaluations done: $n/1 summary.json;$st" > $P/AFTER_DONE
echo "$(date '+%F %T') ab3_after: evaluations done, $n/1 summary.json;$st" >> $BM/reports/bm1.log
