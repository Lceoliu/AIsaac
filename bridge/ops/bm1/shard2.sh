#!/bin/sh
# bm1: two shards timed at once, each with its learner on its own second GPU (--learner-device=cuda:1). The two lanes
# come from two lane files (no hardware written here), each setting:
#   LANE_GPUS   the shard's two GPUs, actor first, e.g. 1,2      LANE_CPUSET / LANE_MEMS   its CPUs / memory node
#   LANE_LOCK   its lane lock file                               LANE_NAME / LANE_PORT     the instances' name / port base
#   LANE_MEM    the per-run memory limit (one shard's measured peak 44.9 GB on the 8x3090: 50g there)
#   (env/dgene-lane0.env, env/dgene-lane1.env: the 8x3090's node0 / node1 lanes of M1b)
# Takes lane 0's lock then lane 1's (fixed order: never holds the second while waiting for the first), runs prof.sh
# twice concurrently with NOLOCK=1 (tags <tag>-s0 / <tag>-s1), releases both.
# Isolation: docker mode, each shard is its own container (pid namespace: kill_abplus scans only its own /proc; its own
# /dev/shm, /tmp and X server; --network none). Direct mode shares the host's: names and port bases must differ (they
# do: LANE_NAME neither the other plus digits, LANE_PORT ranges disjoint), and kill_abplus matches instances/<name>/data
# so the shards do not kill each other's games; the PNG cache in /dev/shm is per name and port.
# Safety line (M1b): the host's MemAvailable below 10 GiB, or a memory-limit kill (docker: the container's memory.events;
# direct: a memguard line in bm1.log) stops both shards (SIGINT to their trainers), logged; no retry.
#   usage: [UPDATES=24] [HOURS=30.0] [LIMIT=4500] shard2.sh <tag> <tree> <lane0 file> <lane1 file>
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TAG=$1; TREE=$2; LF0=$3; LF1=$4
lane() { ( . "$1"; eval echo "\$$2" ); }
for f in $LF0 $LF1; do for v in LANE_GPUS LANE_LOCK LANE_NAME LANE_PORT LANE_MEM; do
  [ -n "$(lane $f $v)" ] || { echo "shard2.sh: $v missing in $f" >&2; exit 2; }; done; done
L0F=$(lane $LF0 LANE_LOCK); L1F=$(lane $LF1 LANE_LOCK)
for L in $L0F $L1F; do
  while ! ( set -C; echo "$SESSION $(date '+%F %T') timed shard pair $TAG ($TREE) 计时运行" > $L ) 2>/dev/null; do sleep 30; done
done
L0=$(cat $L0F); L1=$(cat $L1F)
log "shard pair $TAG start: tree $TREE; locks: $L0 | $L1"
X="-e EXTRA=--learner-device=cuda:1"
P_=""
for k in 0 1; do
  f=$LF0; [ $k = 1 ] && f=$LF1
  G=$(lane $f LANE_GPUS)
  NOLOCK=1 LANE_CPUS=1 LOCKS_HELD="${LOCKS_HELD:-both lanes}" GPU=$G GPUS_OK=$G CPUSET=$(lane $f LANE_CPUSET) MEMS=$(lane $f LANE_MEMS) \
    MEM=$(lane $f LANE_MEM) ENVS="$X -e NAME=$(lane $f LANE_NAME) -e PORT=$(lane $f LANE_PORT) ${ENVS:-}" \
    $BM/prof.sh $TAG-s$k $TREE &
  P_="$P_ $!"
done
( mg0=$(grep -c 'memguard .* \(over the limit\|OOM kill\)' $LOG 2>/dev/null)
  while :; do
    sleep 5
    alive=0; for p in $P_; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 1 ] || break
    av=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    oom=0
    if [ "$BM1_MODE" = docker ]; then
      for k in 0 1; do
        id=$(docker inspect -f '{{.Id}}' aisaac-bench-bm1-a1-prof-$TAG-s$k 2>/dev/null) || continue
        n=$(cat /sys/fs/cgroup/system.slice/docker-$id.scope/memory.events /sys/fs/cgroup/memory/docker/$id/memory.oom_control 2>/dev/null | awk '/^oom_kill / {print $2; exit}')   # (cgroup v2 / v1)
        [ "${n:-0}" -gt 0 ] && oom="$oom s$k:$n"
      done
    else
      mg=$(grep -c 'memguard .* \(over the limit\|OOM kill\)' $LOG 2>/dev/null)
      [ "$mg" -gt "$mg0" ] && oom="memguard"
    fi
    if [ "$av" -lt 10240 ] || [ "$oom" != 0 ]; then
      log "shard pair $TAG SAFETY STOP: MemAvailable ${av} MiB, memory-limit kills:$oom"
      for k in 0 1; do sigint_trainer prof-$TAG-s$k $TAG-s$k 2>/dev/null; done
      break
    fi
  done ) &
WD=$!
wait $P_
kill $WD 2>/dev/null
log "shard pair $TAG done"
[ "$(cat $L1F 2>/dev/null)" = "$L1" ] && rm -f $L1F
[ "$(cat $L0F 2>/dev/null)" = "$L0" ] && rm -f $L0F
