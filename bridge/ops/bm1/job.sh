#!/bin/sh
# bm1: one untimed container job under the lane lock; output to reports/<purpose>-<mmdd-HHMMSS>.log.
#   usage: [GPU=1] [ENVS="-e K=V"] [LOCK_EXCL=1: the exclusive lock, a short measurement] job.sh <purpose> <command in the container (one string, run by sh -c)>
# Prints the unique log path and returns the container exit code, also saved atomically in <log>.rc.
# Do not pipe through tail when the exit status is needed.
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
P=$1; CMD=$2; GPU=${GPU:-1}
R=$BM/reports; LOG=$R/bm1.log
if [ "${LOCK_EXCL:-0}" = 1 ]; then take_excl "$P (GPU $GPU, measurement, exclusive)"; else take "$P (GPU $GPU, not timed)"; fi
L=$(mktemp --suffix=.log "$R/$P-$(date +%m%d-%H%M%S)-XXXXXX") || { give; exit 1; }
DOCK_ENV="${ENVS:-}" "$BM/run.sh" "$P" "$GPU" sh -c "$CMD" > "$L" 2>&1
rc=$?
echo "$(date '+%F %T') job $P rc=$rc ($L)" >> $LOG
if ! printf "%s\n" "$rc" > "$L.rc.new" || ! mv -fT "$L.rc.new" "$L.rc"; then
  echo "job.sh: cannot publish exit status for $L" >&2
  give
  exit 1
fi
give
printf "%s\n" "$L"
exit "$rc"
