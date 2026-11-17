#!/bin/sh
# gpu57 C112 (2026-11-16, the coordinator): the reward ladder's order. Tree maa1e98a-rw; arms l1old, l1blk, l2old, l2blk
# (c112_arm.sh). Waits for C111 VN0's evaluation (AFTER_DONE_vn0: lane b, GPUs 6+7, free), runs c112_d6.sh first when
# it exists and D6_DONE does not, then reads REF's early-stop outcome (c111/MID_ref):
#   REF stopped and evaluated: L1 old on lane b and L1 block on lane a together, then L2 old (b) and L2 block (a);
#   REF not stopped: all four on lane b, one after the other (REF untouched).
# The old stack missing L1 (last 10 updates' diag_hit <= 0.5) or any arm FAILED: c112/HALT, the running block arm
# stopped (STOP), no further arm; "ladder stopped". Lines in c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_ladder.sh > /root/AIsaac/train/c112/ladder.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; C=/root/AIsaac/train/c111; BM=/root/AIsaac/bm1; T=maa1e98a-rw
LA=env/gpu57-c110-a.env; LB=env/gpu57-c110-b.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_ladder: $*" >> $BM/reports/bm1.log; }
arm() { setsid nohup sh $BM/c112_arm.sh $1 $T $2 > $P/arm_$1.out 2>&1 < /dev/null & }
wait_arm() { while ! grep -qE "arm over|FAILED|HALT" $P/STATUS_$1 2>/dev/null; do sleep 60; done; }
hit() { sed -n 's/.*SUMMARY.*diag_hit \([0-9.]*\).*/\1/p' $P/STATUS_$1 2>/dev/null | tail -1; }
failed() { grep -qE "FAILED|HALT" $P/STATUS_$1; }
halt() {
  touch $P/HALT
  for n in l1blk l2blk; do
    [ -d $P/$n/$n-s0 ] && touch $P/$n/STOP
    [ -d /root/AIsaac/train/c112smoke/${n}s ] && touch /root/AIsaac/train/c112smoke/${n}s/STOP
  done
  say "ladder stopped: $*"; exit 1
}
passed_l1() { python3 -c "print(1 if float('${1:-0}') > 0.5 else 0)"; }
say "waiting for C111 VN0's evaluation (AFTER_DONE_vn0)"
while [ ! -f $C/AFTER_DONE_vn0 ]; do sleep 60; done
say "VN0: $(cut -c1-120 $C/AFTER_DONE_vn0)"
if [ -f $BM/c112_d6.sh ] && [ ! -f $P/D6_DONE ]; then
  say "D6 test first (lane b, CPU only)"; sh $BM/c112_d6.sh > $P/d6.out 2>&1; say "D6: $(tail -1 $P/D6_DONE 2>/dev/null)"
fi
while ! grep -qE "early stop: run ended|no early stop|ended first|c111_mid over" $C/MID_ref 2>/dev/null; do sleep 60; done
if grep -q "early stop: run ended" $C/MID_ref; then
  say "REF stopped and evaluated: L arms in pairs (old on lane b, block on lane a)"
  arm l1old $LB; arm l1blk $LA; wait_arm l1old
  failed l1old && halt "l1old FAILED"
  h=$(hit l1old); [ "$(passed_l1 $h)" = 1 ] || halt "the old stack missed L1 (last-10 diag_hit ${h:-?})"
  wait_arm l1blk; failed l1blk && halt "l1blk FAILED"
  arm l2old $LB; arm l2blk $LA
  wait_arm l2old; failed l2old && halt "l2old FAILED"
  wait_arm l2blk; failed l2blk && halt "l2blk FAILED"
else
  say "REF not stopped: L arms one after the other on lane b"
  arm l1old $LB; wait_arm l1old; failed l1old && halt "l1old FAILED"
  h=$(hit l1old); [ "$(passed_l1 $h)" = 1 ] || halt "the old stack missed L1 (last-10 diag_hit ${h:-?})"
  for n in l1blk l2old l2blk; do arm $n $LB; wait_arm $n; failed $n && halt "$n FAILED"; done
fi
say "ladder over: last-10 diag_hit l1old $(hit l1old), l1blk $(hit l1blk), l2old $(hit l2old), l2blk $(hit l2blk)"
