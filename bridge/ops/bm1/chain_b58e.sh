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
if [ -f $R/gate3-$TREE.failed ]; then echo "$(date '+%F %T') chain_b58e stops: gate3 failed ($(tr '\n' ' ' < $R/gate3-$TREE.failed))" >> $LOG; exit 1; fi
for B in 1 0; do
  quiet0
  GPU=4 UPDATES=24 HOURS=30.0 ENVS="-e ISAAC_BLOCK_CPP_BATCH=$B" $OPSD/prof.sh e3-batch$B-$TREE $TREE
done
echo "$(date '+%F %T') chain_b58e done ($TREE)" >> $LOG
