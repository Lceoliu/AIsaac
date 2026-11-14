#!/bin/sh
# gpu57 ab1 (2026-11-14, the coordinator): two independent K=1 training runs at once under one exclusive bm1 lock.
# train_guard.sh takes X for itself, so a second guard would wait for the first; here the pair takes X once and starts
# both guards with NOLOCK=1 (their containers are registered in X through the exported LOCKFILE: run.sh lock_container).
#   usage: [HOURS= CKPT_EVERY= CKPT_KEEP= CKPT_MILESTONE= CKPT_ROTATE= MEM_GB= ...: train_guard.sh's, both runs]
#          ab1_pair.sh <parent under /root/AIsaac/train> <tree> <name 1> <lane file 1> <name 2> <lane file 2>
#   each run directory /root/AIsaac/train/<parent>/<name> must exist with its train_args (and must not hold a run yet)
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
PARENT=$1; TREE=$2; N1=$3; L1=$4; N2=$5; L2=$6
for n in $N1 $N2; do
  [ -f /root/AIsaac/train/$PARENT/$n/train_args ] || { echo "ab1_pair: /root/AIsaac/train/$PARENT/$n/train_args missing" >&2; exit 2; }
  [ -e /root/AIsaac/train/$PARENT/$n/$n-s0 ] && { echo "ab1_pair: /root/AIsaac/train/$PARENT/$n/$n-s0 exists" >&2; exit 2; }
done
take_excl "training pair $PARENT ($N1 + $N2, $TREE) 计时运行"
log "ab1_pair $PARENT: $N1 ($L1) + $N2 ($L2), tree $TREE, hours ${HOURS:-?}"
RUN_PARENT=$PARENT NOLOCK=1 $BM/train_guard.sh $N1 $TREE $L1 > /root/AIsaac/train/$PARENT/$N1/guard.out 2>&1 < /dev/null &
P1=$!
RUN_PARENT=$PARENT NOLOCK=1 $BM/train_guard.sh $N2 $TREE $L2 > /root/AIsaac/train/$PARENT/$N2/guard.out 2>&1 < /dev/null &
P2=$!
wait $P1; r1=$?
wait $P2; r2=$?
log "ab1_pair $PARENT: guards ended ($N1 rc $r1, $N2 rc $r2)"
give
