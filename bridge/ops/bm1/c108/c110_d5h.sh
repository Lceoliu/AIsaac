#!/bin/sh
# C110 D5 by action head (2026-11-15, the coordinator; throwaway): waits until both c45 arms are over (their AFTER_DONE
# files: training and evaluation, GPUs 4 and 5 free), then c108/d5_run.sh with D5_HEADS=1 D5_SUB=c110/d5h on GPUs 4,5,
# node0's CPUs: bm-fresh-heads (block tree maa1e98a-d5, fresh init, lr 0, 16 updates), then c99old-fresh-heads
# (m1ae82d3-d5, fresh, lr 0, about 17 updates). Status in /root/AIsaac/train/c110/D5H_STATUS.
#   usage: setsid nohup sh c110_d5h.sh > /root/AIsaac/train/c110/d5h_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c110; BM=/root/AIsaac/bm1
while [ ! -f $P/AFTER_DONE_c110-r45 ] || [ ! -f $P/AFTER_DONE_c110-q45 ]; do sleep 60; done
echo "$(date '+%F %T') c45 arms over; D5 heads on GPUs 4,5" >> $P/D5H_STATUS
D5_HEADS=1 D5_SUB=c110/d5h GPUS=4,5 CPUS=0-7,16-23 MEMS=0 sh $BM/c108/d5_run.sh bm-fresh-heads:new:fresh c99old-fresh-heads:old:fresh > $P/d5h_run.out 2>&1
echo "$(date '+%F %T') done: $(grep -c 'grad lines' $P/d5h_run.out) runs ($(grep 'grad lines' $P/d5h_run.out | tr '\n' ';'))" >> $P/D5H_STATUS
