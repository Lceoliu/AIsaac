#!/bin/sh
# bm1: one untimed container job under the lane lock; output to reports/<purpose>-<mmdd-HHMMSS>.log.
#   usage: [GPU=1] [ENVS="-e K=V"] [LOCK_EXCL=1: the exclusive lock, a short measurement] job.sh <purpose> <command in the container (one string, run by sh -c)>
# Prints the log's path; the container's exit code goes to bm1.log and to <log>.rc (2026-11-18: callers that take only
# the path -- `L=$(job.sh ... | tail -1)` -- read the code there instead of losing it).
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
P=$1; CMD=$2; GPU=${GPU:-1}
R=$BM/reports; LOG=$R/bm1.log
if [ "${LOCK_EXCL:-0}" = 1 ]; then take_excl "$P (GPU $GPU, measurement, exclusive)"; else take "$P (GPU $GPU, not timed)"; fi
L=$R/$P-$(date +%m%d-%H%M%S).log
DOCK_ENV="${ENVS:-}" $BM/run.sh $P $GPU sh -c "$CMD" > $L 2>&1
rc=$?
echo "$(date '+%F %T') job $P rc=$rc ($L)" >> $LOG
echo $rc > $L.rc
give
echo $L
