#!/bin/sh
# gpu57 C112 L2 on lane a (2026-11-16, the coordinator: lane b's GPUs 6,7 taken by another tenant since 13:31): l2old
# then l2blk, one after the other on env/gpu57-c110-a.env (GPUs 4,5), 60 game hours each (c112_arm.sh). Before each arm
# GPUs 4 and 5 must show no compute process at all (nothing of ours runs then): otherwise "GPUs 4/5 busy", no wait, no
# other GPUs, c112/HALT, stop. An arm FAILED: HALT, stop. Lines in c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_l2a.sh > /root/AIsaac/train/c112/l2a.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw; LA=env/gpu57-c110-a.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_l2a: $*" >> $BM/reports/bm1.log; }
hit() { sed -n 's/.*SUMMARY.*diag_hit \([0-9.]*\).*/\1/p' $P/STATUS_$1 2>/dev/null | tail -1; }
sum() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed 's/.*SUMMARY //'; }
busy() {
  for g in 4 5; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$u")
    [ "$n" = 0 ] || { echo "GPU $g: $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ' | tr '\n' ' ')"; }
  done
}
cd $BM
for n in l2old l2blk; do
  b=$(busy)
  if [ -n "$b" ]; then touch $P/HALT; say "ladder stopped: GPUs 4/5 busy before $n ($b): no wait, no other GPUs"; exit 1; fi
  say "$n on lane a (GPUs 4,5 free)"
  sh $BM/c112_arm.sh $n $T $LA > $P/arm_$n.out 2>&1
  if grep -qE "FAILED|HALT" $P/STATUS_$n; then touch $P/HALT; say "ladder stopped: $n $(grep -E 'FAILED|HALT' $P/STATUS_$n | tail -1 | cut -c21-200)"; exit 1; fi
  say "$n over: $(sum $n)"
done
say "ladder over (L2 on lane a): last-10 diag_hit l1old $(hit l1old), l1blk $(hit l1blk), l2old $(hit l2old), l2blk $(hit l2blk)"
