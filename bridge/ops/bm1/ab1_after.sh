#!/bin/sh
# gpu57 ab1 (2026-11-14): after the pair. Waits for ab1_pair.sh ab1 to end; when both guard.logs say the 1200 game
# hours were reached, the held-out evaluations of u0500 and last.pt, the two runs in parallel (force: GPU 4 + node0's
# CPUs, noforce: GPU 6 + node1's; evtools/ev_ab1.sh, 14 workers, seeds 2147490000:128); else none. Its status in
# /root/AIsaac/train/ab1/AFTER_DONE at the end (the operator's waiter reads only that file).
#   usage: setsid nohup sh ab1_after.sh > /root/AIsaac/train/ab1/after.out 2>&1 < /dev/null &
P=/root/AIsaac/train/ab1
BM=/root/AIsaac/bm1
while pgrep -f "ab1_pair.sh ab1 " > /dev/null; do sleep 60; done
ok=1
for g in force noforce; do grep -q "game hours reached" $P/ab1-$g/guard.log || ok=0; done
if [ $ok = 0 ]; then
  echo "$(date '+%F %T') training did not reach its game hours: no evaluation ($(tail -1 $P/ab1-force/guard.log | cut -c1-120) | $(tail -1 $P/ab1-noforce/guard.log | cut -c1-120))" > $P/AFTER_DONE
  exit 0
fi
echo "$(date '+%F %T') ab1_after: training ended, evaluations start" >> $BM/reports/bm1.log
sh $BM/evtools/ev_ab1.sh ab1-force 4 0-7,16-23 0 14 force-c128-u0500,update-000500.pt,2147490000:128 force-c128-last,last.pt,2147490000:128 > $P/eval_force.out 2>&1 &
E1=$!
sleep 30
sh $BM/evtools/ev_ab1.sh ab1-noforce 6 8-15,24-31 1 14 noforce-c128-u0500,update-000500.pt,2147490000:128 noforce-c128-last,last.pt,2147490000:128 > $P/eval_noforce.out 2>&1 &
E2=$!
wait $E1; wait $E2
n=$(ls $P/eval/*/summary.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') evaluations done: $n/4 summary.json" > $P/AFTER_DONE
echo "$(date '+%F %T') ab1_after: evaluations done, $n/4 summary.json" >> $BM/reports/bm1.log
