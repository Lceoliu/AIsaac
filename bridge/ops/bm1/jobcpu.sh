#!/bin/sh
# bm1: job.sh's CPU-only variant (2026-11-16, the coordinator): one untimed container job under the shared lock via
# runcpu.sh (no GPU), output to reports/<purpose>-<mmdd-HHMMSS>.log, a bm1.log line. Host CPU guard (2026-11-17, the
# coordinator: the load average is meaningless with ~100 processes stuck in D state in the GPU driver; this job itself
# adds about 25 % of the host): the whole host's busy share from /proc/stat (1 - (idle + iowait) / total). Before the
# start: a 30-s sample, busy < 65 % to start, else wait. While the job runs (60-s samples): busy > 90 % for 5 minutes
# in a row pauses its container (docker pause, ours only) until a sample under 80 % (logged).
#   usage: [ENVS="-e K=V"] [CPUSET=.. MEMS=.. MEM=..] jobcpu.sh <purpose> <command in the container (sh -c)>
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
P=$1; CMD=$2
R=$BM/reports; LOG=$R/bm1.log
st() { head -1 /proc/stat | awk '{t=0; for (i=2;i<=NF;i++) t+=$i; print t, $5+$6}'; }
busy() {   # seconds -> busy percent over that window
  a=$(st); sleep $1; b=$(st)
  echo "$a $b" | awk '{dt=$3-$1; di=$4-$2; if (dt<=0) print 0; else printf "%d\n", 100*(1-di/dt)}'
}
w=0
while :; do
  u=$(busy 30)
  [ "$u" -lt 65 ] && break
  [ $w = 0 ] && log "jobcpu $P: host CPU busy ${u}% >= 65%, waiting"; w=1
  sleep 30
done
log "jobcpu $P: host CPU busy ${u}% < 65%: starting"
take "$P (CPU only, not timed)"
L=$R/$P-$(date +%m%d-%H%M%S).log
DOCK_ENV="${ENVS:-}" $BM/runcpu.sh $P sh -c "$CMD" > $L 2>&1 &
J=$!
C="$BM1_NAME-cpu-$P"; high=0; paused=0
while kill -0 $J 2>/dev/null; do
  u=$(busy 60)
  if [ $paused = 0 ]; then
    if [ "$u" -gt 90 ]; then high=$((high + 1)); else high=0; fi
    if [ $high -ge 5 ]; then docker pause "$C" > /dev/null 2>&1 && { paused=1; log "jobcpu $P: host CPU busy > 90% for 5 min (now ${u}%): container paused"; }; fi
  elif [ "$u" -lt 80 ]; then
    docker unpause "$C" > /dev/null 2>&1; paused=0; high=0; log "jobcpu $P: host CPU busy ${u}% < 80%: container resumed"
  fi
done
wait $J; rc=$?
echo "$(date '+%F %T') jobcpu $P rc=$rc ($L)" >> $LOG
give
echo $L
