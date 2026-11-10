#!/bin/sh
# cpp lane, B58e acceptance on the 8x3090 (host side): gate3 (entry + full tests + A8) of the tree, then, only when it
# passed, the same-tree timing pair on GPU 4: ISAAC_BLOCK_CPP_BATCH=1 then =0, 24 updates each (prof.sh: P24, 30.0
# game hours, M0 samplers). Before each heavy step it waits while node0's lock says a timed run (A1's baseline).
#   usage: chain_b58e.sh <tree>
set -u
OPSD=$(dirname "$0")
export LANE=cpp
. $OPSD/lane.sh
TREE=$1
R=$LD/reports; LOG=$R/bm1.log
mkdir -p $R
quiet0() { while grep -q '计时运行' $BM/LOCK_node0 2>/dev/null; do sleep 30; done; }
quiet0
GPU=4 $OPSD/gate3.sh $TREE gate3-$TREE
# (the coordinator's known items of the mainline gate on this host do not block the arms: test_b48 [False-cuda] /
# unused_prefetch, test_b54 pressure_restores, test_b60 spatial_segment[cuda] (conftest's leftover-thread errors) and
# test_b52's test_driver_pressure_pages_complete_before_records (timing-sensitive). Anything else stops the chain.)
KNOWN='test_b48.*(False-cuda|unused_prefetch)|test_b54_driver_lanes.py::test_driver_lanes_pressure_restores|test_b60.*spatial_segment|test_b52_cpp_actor.py::test_driver_pressure_pages_complete_before_records'
if [ -f $R/gate3-$TREE.failed ]; then
  other=$(grep -E '^(FAILED|ERROR) ' $R/gate3-$TREE.full.log | grep -vE "$KNOWN")
  if [ "$(cat $R/gate3-$TREE.failed)" = "full" ] && [ -z "$other" ] && grep -qE '^(FAILED|ERROR) ' $R/gate3-$TREE.full.log; then
    echo "$(date '+%F %T') chain_b58e: gate3 failures are known items only ($(grep -cE '^(FAILED|ERROR) ' $R/gate3-$TREE.full.log)): the arms go on" >> $LOG
  else
    echo "$(date '+%F %T') chain_b58e stops: gate3 failed ($(tr '\n' ' ' < $R/gate3-$TREE.failed); other: $(echo "$other" | head -3 | tr '\n' ' '))" >> $LOG; exit 1
  fi
fi
for B in 1 0; do
  quiet0
  GPU=4 UPDATES=24 HOURS=30.0 ENVS="-e ISAAC_BLOCK_CPP_BATCH=$B" $OPSD/prof.sh e3-batch$B-$TREE $TREE
done
echo "$(date '+%F %T') chain_b58e done ($TREE)" >> $LOG
