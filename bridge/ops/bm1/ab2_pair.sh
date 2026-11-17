#!/bin/sh
# gpu57 ab2 (C107, 2026-11-14, the coordinator): two independent K=1 training runs at once under one exclusive bm1 lock,
# as ab1_pair.sh, but each run with its own tree and its own guard settings: <run dir>/guard.pre (KEY=VALUE lines, read
# with set -a before that run's train_guard.sh starts: e.g. arm A's BASE=xold_c99.sh LEARNER_DEV= NO_RESUME=1). The
# settings both runs share come from the caller's environment (HOURS, CKPT_EVERY, MEM_GB, ...).
#   usage: ab2_pair.sh <parent under /root/AIsaac/train> <name 1> <tree 1> <lane file 1> <name 2> <tree 2> <lane file 2>
#   each run directory /root/AIsaac/train/<parent>/<name> must exist with its train_args (and must not hold a run yet)
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
PARENT=$1; N1=$2; T1=$3; L1=$4; N2=$5; T2=$6; L2=$7
for n in $N1 $N2; do
  [ -f /root/AIsaac/train/$PARENT/$n/train_args ] || { echo "ab2_pair: /root/AIsaac/train/$PARENT/$n/train_args missing" >&2; exit 2; }
  [ -e /root/AIsaac/train/$PARENT/$n/$n-s0 ] && { echo "ab2_pair: /root/AIsaac/train/$PARENT/$n/$n-s0 exists" >&2; exit 2; }
done
take_excl "training pair $PARENT ($N1 $T1 + $N2 $T2) 计时运行"
log "ab2_pair $PARENT: $N1 ($T1, $L1) + $N2 ($T2, $L2), hours ${HOURS:-?}"
one() {   # <name> <tree> <lane file>
  D=/root/AIsaac/train/$PARENT/$1
  ( [ -f $D/guard.pre ] && { set -a; . $D/guard.pre; set +a; }
    export RUN_PARENT=$PARENT NOLOCK=1
    exec $BM/train_guard.sh $1 $2 $3 ) > $D/guard.out 2>&1 < /dev/null
}
one $N1 $T1 $L1 &
P1=$!
one $N2 $T2 $L2 &
P2=$!
wait $P1; r1=$?
wait $P2; r2=$?
log "ab2_pair $PARENT: guards ended ($N1 rc $r1, $N2 rc $r2)"
give
