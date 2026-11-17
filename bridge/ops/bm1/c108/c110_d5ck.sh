#!/bin/sh
# C110 D5 by action head at trained weights (2026-11-15, the coordinator; after the gate arm in priority): waits for the
# first of c110/AFTER_DONE_c110-qm (lane a: GPUs 4,5, node0) / AFTER_DONE_c110-g (lane b: GPUs 6,7, node1), then on that
# lane c108/d5_run.sh with D5_HEADS=1 D5_SUB=c110/d5h: bm-u0250-heads, bm-last-heads (block tree maa1e98a-d5, ab2-bm's
# u250 / last, lr 0, 16 updates), then c99old-u0250-heads (m1ae82d3-d5, lr 0, about 17 updates). Status in
# /root/AIsaac/train/c110/D5CK_STATUS.
#   usage: setsid nohup sh c110_d5ck.sh > /root/AIsaac/train/c110/d5ck_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c110; BM=/root/AIsaac/bm1
first=""
while [ -z "$first" ]; do
  for n in c110-qm c110-g; do [ -f $P/AFTER_DONE_$n ] && { first=$n; break; }; done
  [ -n "$first" ] || sleep 60
done
case $first in
  c110-qm) G=4,5; C=0-7,16-23; M=0 ;;
  c110-g) G=6,7; C=8-15,24-31; M=1 ;;
esac
echo "$(date '+%F %T') $first over; D5 heads at trained weights on GPUs $G" >> $P/D5CK_STATUS
D5_HEADS=1 D5_SUB=c110/d5h GPUS=$G CPUS=$C MEMS=$M sh $BM/c108/d5_run.sh bm-u0250-heads:new:bm-u0250.pt \
  bm-last-heads:new:bm-last.pt c99old-u0250-heads:old:c99old-u0250.pt > $P/d5ck_run.out 2>&1
echo "$(date '+%F %T') done: $(grep -c 'grad lines' $P/d5ck_run.out) runs ($(grep 'grad lines' $P/d5ck_run.out | tr '\n' ';'))" >> $P/D5CK_STATUS
