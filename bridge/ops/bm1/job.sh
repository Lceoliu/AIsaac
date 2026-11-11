#!/bin/sh
# bm1: one untimed container job under the lane lock; output to reports/<purpose>-<mmdd-HHMMSS>.log.
#   usage: [GPU=1] [ENVS="-e K=V"] job.sh <purpose> <command in the container (one string, run by sh -c)>
set -u
. "$(dirname "$0")/lane.sh"
. $OPS/lock.sh
P=$1; CMD=$2; GPU=${GPU:-1}
R=$LD/reports; LOG=$R/bm1.log
take "$P (GPU $GPU, not timed)"
L=$R/$P-$(date +%m%d-%H%M%S).log
DOCK_ENV="${ENVS:-}" $OPS/dock.sh $P $GPU sh -c "$CMD" > $L 2>&1
echo "$(date '+%F %T') job $P rc=$? ($L)" >> $LOG
give
echo $L
