#!/bin/sh
# gpu57 C112 l2ego-head (2026-11-16, the coordinator): waits for c112_rw4_tests2.sh's verdict (c112/rw4_tests2/RESULT);
# "RW4 TESTS OK": waits for a free pair (4+5 or 6+7: no compute process on either card), then L2 block with
# ISAAC_BLOCK_EGO=head (tree maa1e98a-rw4; smoke, coef checks, D1 in maa1e98a-rw4-d1 < 0.001, 60 gh, seed 1000).
# Anything else: stops. Lines in c112/STATUS_head (the host watcher reports them), c112/LADDER and bm1.log.
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw4; TD=maa1e98a-rw4-d1
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_head; echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_head: $*" >> $BM/reports/bm1.log; }
free() { for g in $1; do u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 1; done; return 0; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
cd $BM
say "head: waiting for the rw4 tests (round 2)"
while ! grep -q "^RW4 TESTS" $P/rw4_tests2/RESULT 2>/dev/null; do sleep 30; done
r=$(tail -1 $P/rw4_tests2/RESULT)
[ "$r" = "RW4 TESTS OK" ] || { say "head: rw4 tests round 2: $r -- not started (a person looks)"; exit 1; }
say "head: rw4 tests round 2 ok; waiting for a free pair (4+5 or 6+7)"
while :; do
  if free "4 5"; then lane=env/gpu57-c110-a.env; g="4,5"; break; fi
  if free "6 7"; then lane=env/gpu57-c110-b.env; g="6,7"; break; fi
  sleep 60
done
[ -e $P/HALT ] && mv $P/HALT $P/HALT.$(date +%H%M)
say "head: l2ego-head started on GPUs $g ($lane), tree $T, 60 game hours, D1 in $TD"
D1_TREE=$TD ARM_HOURS=60 sh $BM/c112_arm.sh l2ego-head $T $lane > $P/arm_l2ego-head.out 2>&1
if grep -qE "FAILED|HALT" $P/STATUS_l2ego-head; then say "head: ladder stopped: l2ego-head $(grep -E 'FAILED|HALT' $P/STATUS_l2ego-head | tail -1 | cut -c21-200)"; exit 1; fi
h=$(val l2ego-head diag_hit); l=$(val l2ego-head diag_left); r=$(val l2ego-head diag_right)
v=$(python3 -c "print('PASS' if ${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15 else 'no pass')")
say "head: l2ego-head $v: $(grep SUMMARY $P/STATUS_l2ego-head | tail -1 | sed 's/.*SUMMARY //')"
