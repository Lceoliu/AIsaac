#!/bin/sh
# bm1: two shards timed at once, each with its learner on its own second GPU (--learner-device=cuda:1):
#   shard 0 = GPUs 1,2 + node0 (CPU 0-11,24-35, memory node 0); shard 1 = GPUs 4,5 + node1 (CPU 12-23,36-47, node 1).
# Takes LOCK_node0 then LOCK_node1 (both lanes: nothing else on either meanwhile; node1 is cpp-actor's lane, so the
# coordinator schedules this), runs prof.sh twice concurrently with NOLOCK=1 (tags <tag>-s0 / <tag>-s1), releases both.
# Per-shard isolation (the single-trainer inventory, bm1-single-trainer-inventory.md):
#   * each shard is its own container: its own pid namespace (kill_abplus / stop_abplus scan only their own /proc),
#     its own /dev/shm (the PNG cache, the teacher flock files), its own /tmp (xvfb-run -a: its own X server), and
#     --network none (its own loopback: ports cannot collide); names and port bases are still distinct (s0a / s1b,
#     39200 / 39600) so that logs and instance directories tell them apart;
#   * HOME (/bench/bm1/home) is shared: instances/<name>/ differ by name (s0a0.. vs s1b0..; neither name is the other
#     plus digits); the caches under it (isaac-actor, learn-native, visible, Inductor, Triton) are content-addressed
#     and written atomically (mkstemp + rename);
#   * memory: MEM=50g per container (2 x 50 + the other tenants ~18 within 125 GiB; one shard's measured peak 43.9 GB);
#   * threads: OMP_NUM_THREADS=1 per container (dock.sh), cpusets disjoint, the learner's devices by
#     CUDA_VISIBLE_DEVICES (the container sees only its two GPUs as cuda:0 / cuda:1).
#   usage: [UPDATES=24] [HOURS=30.0] [LIMIT=4500] shard2.sh <tag> <tree>
set -u
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
TAG=$1; TREE=$2
R=$BM/reports; LOG=$R/bm1.log
SESSION=${SESSION:-bm1-a1}
for L in LOCK_node0 LOCK_node1; do   # (fixed order: never holds node1 while waiting for node0)
  while ! ( set -C; echo "$SESSION $(date '+%F %T') timed shard pair $TAG ($TREE) 计时运行" > $BM/$L ) 2>/dev/null; do sleep 30; done
done
L0=$(cat $BM/LOCK_node0); L1=$(cat $BM/LOCK_node1)
echo "$(date '+%F %T') shard pair $TAG start: tree $TREE; locks: $L0 | $L1" >> $LOG
X="-e EXTRA=--learner-device=cuda:1"
NOLOCK=1 LOCKS_HELD="node0+node1" GPU=1,2 CPUSET=0-11,24-35 MEMS=0 GPUS_OK=1,2,3 MEM=50g \
  ENVS="$X -e NAME=s0a -e PORT=39200 ${ENVS:-}" $BM/prof.sh $TAG-s0 $TREE &
P0=$!
NOLOCK=1 LOCKS_HELD="node0+node1" GPU=4,5 CPUSET=12-23,36-47 MEMS=1 GPUS_OK=4,5 MEM=50g \
  ENVS="$X -e NAME=s1b -e PORT=39600 ${ENVS:-}" $BM/prof.sh $TAG-s1 $TREE &
P1=$!
wait $P0 $P1
echo "$(date '+%F %T') shard pair $TAG done" >> $LOG
[ "$(cat $BM/LOCK_node1 2>/dev/null)" = "$L1" ] && rm -f $BM/LOCK_node1
[ "$(cat $BM/LOCK_node0 2>/dev/null)" = "$L0" ] && rm -f $BM/LOCK_node0
