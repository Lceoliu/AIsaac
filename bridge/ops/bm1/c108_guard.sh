#!/bin/sh
# gpu57 C108 (2026-11-15, the coordinator): one training run (train_guard.sh) under a SHARED bm1 lock instead of the
# exclusive one -- a run whose speed is not a result (D3, ab3-lr), so that not-timed jobs (D1's minutes-long lr-0 checks
# on GPUs 4+5) can run beside it. Its own guard settings from <run dir>/guard.pre (KEY=VALUE lines, set -a), as ab2_pair.sh;
# the rest (HOURS, CKPT_EVERY, MEM_GB, ...) from the caller's environment.
#   usage: c108_guard.sh <parent under /root/AIsaac/train> <name> <tree> <lane file>
#   the run directory /root/AIsaac/train/<parent>/<name> must exist with its train_args (and must not hold a run yet)
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
PARENT=$1; N=$2; T=$3; L=$4
D=${BM1_TRAIN:-/root/AIsaac/train}/$PARENT/$N   # (bm1.env BM1_TRAIN; gpu57 default)
[ -f $D/train_args ] || { echo "c108_guard: $D/train_args missing" >&2; exit 2; }
[ -e $D/$N-s0 ] && { echo "c108_guard: $D/$N-s0 exists" >&2; exit 2; }
take_shared "training $PARENT/$N ($T, $L), not timed"
log "c108_guard $PARENT/$N: $T $L, hours ${HOURS:-?}"
( [ -f $D/guard.pre ] && { set -a; . $D/guard.pre; set +a; }
  export RUN_PARENT=$PARENT NOLOCK=1
  exec $BM/train_guard.sh $N $T $L ) > $D/guard.out 2>&1 < /dev/null
r=$?
log "c108_guard $PARENT/$N: guard ended rc $r"
give
