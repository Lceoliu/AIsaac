#!/bin/sh
# bm1: the batched acceptance of G (native visible block + native step), W-a (compact V2Log rows) and B (the workers'
# CPU by part), plus the read-only smaps survey (2b), in one machine window. Host side, this lane's bm1 directory.
# Holds the exclusive lock for the whole batch (the training runs measure CPU); every container through run.sh.
#   usage: [GPU=6] [GATE=0|1] accept_g.sh <tree>        (the tree deployed with deploy.sh tree first)
# Steps (expected minutes on gpu57, without other tenants):
#   1  the lane home's libabp_turbo.so from the tree (deploy.sh abphome), compile warnings kept       ~2
#   2  unit tests: test_w_a_v2log_compact, test_b59_lane_arena, test_b58d_lanes, test_b58_native_consumer,
#      test_broken_episode (the native consumer's broken-episode snapshot, N_SNAP 35)                      ~4
#   3  frame probes (abplus_probe_frame_exact): reference clone mode 0, candidate clone ...            ~16
#        a mode 1 (native block + native step), 16 room seeds, --visible-json: payloads equal but for
#          the JSON key order, digests equal, the native block used (status: native / step counts)
#        b mode 1, 4 floor seeds, the same
#        c mode 2 (both built, the Lua one sent), 16 room seeds: payloads byte-equal, 0 mismatches
#        d mode 2, 4 floor seeds
#        (cpu_ms_per_decision_ref / _fast of a / b: G's saving per decision on the same states)
#   4  A8 (abplus_probe_play, 8 seeds, 300 steps) with ISAAC_RL_NATIVE_VISIBLE=1 and =2: 8/8 ok each     ~2
#   5  training, one shard untimed (node1, 12 updates each; ABP_CLONE_STATS and ISAAC_RL_WORKER_CPU=1 in all):  ~27
#        R0 baseline (G off, W-a off), with the smaps survey at update 6 (root + 3 clones)
#        R1 candidate: ISAAC_RL_NATIVE_VISIBLE=1, ISAAC_RL_V2LOG_COMPACT=1
#        R2 check: ISAAC_RL_NATIVE_VISIBLE=2, 6 updates: native visible checks > 0, mismatches 0 (clone_stats)
#   6  (GATE=1) gate3 on the tree                                                                      ~40
# Total ~50 min, ~90 min with GATE=1. Results: reports/acceptg-<tree>-<time>/ (summary.txt: one line per check).
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TREE=$1
GPU=${GPU:-6}
export GPUS_OK=${BM1_TIMED_GPUS_OK:-$BM1_GPUS_OK}
R=$BM/reports
TAG=acceptg-$TREE-$(date +%m%d-%H%M%S)
D=$R/$TAG
mkdir -p $D
S=$D/summary.txt
T=$BM1_IN/trees/$TREE
[ -d $BM/trees/$TREE ] || { echo "accept_g.sh: no tree $TREE (deploy.sh tree first)" >&2; exit 2; }
[ -f $BM/STOP_QUEUE ] && { echo "accept_g.sh: STOP_QUEUE set" >&2; exit 1; }
GAME="-e ABP_HOME=$BM1_IN/home/isaac-abplus -e ABP_BRIDGE_LUA=$T/bridge/abplus/abp_bridge.lua -e LIBGL_ALWAYS_SOFTWARE=1 -e SDL_AUDIODRIVER=dummy -e ALSOFT_DRIVERS=null -e OMP_NUM_THREADS=1"
XV="xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset'"
note() { echo "$(date '+%F %T') $*" | tee -a $S; }
take_excl "$TAG: G / W-a / B acceptance batch (GPU $GPU) 计时运行"
note "$TAG start: tree $TREE GPU $GPU"
host_snapshot "$TAG start"

# 1 the library
GPU=$GPU sh $BM/deploy.sh abphome $TREE > $D/abphome.log 2>&1
note "1 abphome rc=$? warnings $(grep -c 'warning' $D/abphome.log) ($(ls -la $BM/home/isaac-abplus/tools/libabp_turbo.so 2>/dev/null | awk '{print $5}') bytes)"

# 2 unit tests
DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e ISAAC_VISIBLE_PAYLOADS=${BM1_PAYLOADS:-$BM1_IN/payloads}/b45-payloads2" $BM/run.sh acceptg-unit $GPU sh -c \
  "cd $T/bridge/python && timeout 1200 $BM1_PY -m pytest tests/test_w_a_v2log_compact.py tests/test_b59_lane_arena.py tests/test_b58d_lanes.py tests/test_b58_native_consumer.py tests/test_broken_episode.py -q -p no:cacheprovider" > $D/unit.log 2>&1
note "2 unit rc=$? $(tail -1 $D/unit.log)"

# 3 frame probes
ST="return tostring(ABP_NATIVE_VIS.mode)..' native='..ABP_NATIVE_VIS.native..' step='..ABP_NATIVE_VIS.step..' fallbacks='..ABP_NATIVE_VIS.fallbacks..' checks='..ABP_NATIVE_VIS.checks..' mismatches='..ABP_NATIVE_VIS.mismatches"
probe() {   # name, mode, extra args
  n=$1; m=$2; shift 2
  DOCK_ENV="$GAME" $BM/run.sh acceptg-$n $GPU sh -c "cd $T/bridge/python && timeout 1800 $XV $BM1_PY abplus_probe_frame_exact.py --out $BM1_IN/reports/$TAG/$n --fast-lua 'ABP_NATIVE_VIS.mode = $m; return \"ok\"' --status-lua \"$ST\" $*" > $D/$n.log 2>&1
  rc=$?
  note "3 $n rc=$rc $(python3 -c "import json,sys; s=json.load(open('$D/$n/summary.json')); print({k: s.get(k) for k in ('tested','decisions','payload_equal','visible_reordered','digests_equal','cpu_ms_per_decision_ref','cpu_ms_per_decision_fast')})" 2>/dev/null) fast clones: $(python3 $BM/vis_status.py $D/$n/rows.json 2>/dev/null)"
}
probe room1 1 --group normal --seeds range:2147500000:16 --visible-json
probe floor1 1 --floor --seeds range:2147600000:4 --visible-json
probe room2 2 --group normal --seeds range:2147500000:16
probe floor2 2 --floor --seeds range:2147600000:4

# 4 A8 with the native block
for m in 1 2; do
  DOCK_ENV="$GAME -e ISAAC_RL_NATIVE_VISIBLE=$m" $BM/run.sh acceptg-a8m$m $GPU sh -c \
    "cd $T/bridge/python && timeout 1800 $XV $BM1_PY abplus_probe_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147500000:8 --steps 300 --out $BM1_IN/reports/$TAG/a8m$m" > $D/a8m$m.log 2>&1
  note "4 A8 mode $m rc=$? $(tail -1 $D/a8m$m.log | grep -o '"seeds": [0-9]*, "ok": [0-9]*')"
done

# 5 training runs (prof.sh under this batch's lock: NOLOCK; one shard on node1 as cload.sh)
CSPY=$BM/trees/$TREE/bridge/ops/clone_stats.py
train() {   # name, updates, extra -e flags
  n=$1; u=$2; shift 2
  mkdir -p $D/$n-stats
  NOLOCK=1 LOCKS_HELD="$TAG" LANE_CPUS=1 CPUSET=8-15,24-31 MEMS=1 LEARNER_PIN=0 DRIVER_PIN=0 GPU=$GPU UPDATES=$u \
    ENVS="-e ABP_CLONE_STATS=$BM1_IN/reports/$TAG/$n-stats -e ISAAC_RL_WORKER_CPU=1 $*" sh $BM/prof.sh $TAG-$n $TREE
  python3 $CSPY $D/$n-stats > $D/$n-clone-stats.txt 2>&1
  note "5 $n done: $(tail -1 $R/$TAG-$n.summary.txt 2>/dev/null | cut -c1-200); $(head -1 $D/$n-clone-stats.txt)"
}
# (the smaps survey while R0 runs: once progress.csv has 6 rows, the root game process of the run and three of its
#  clones, /proc/<pid>/smaps and smaps_rollup copied; read only)
( P=$R/$TAG-r0/progress.csv
  while [ ! -f $P ] || [ $(($(wc -l < $P)-1)) -lt 6 ]; do sleep 5; [ -f $D/r0.done ] && exit 0; done
  CID=$(docker inspect -f '{{.Id}}' $BM1_NAME-prof-$TAG-r0 2>/dev/null)
  mkdir -p $D/smaps
  for pid in $(pgrep -x isaac.x64); do
    grep -q "$CID" /proc/$pid/cgroup 2>/dev/null || continue
    pp=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null)
    kind=clone; [ "$(cat /proc/$pp/comm 2>/dev/null)" = isaac.x64 ] || kind=root
    echo "$pid $pp $kind" >> $D/smaps/pids.txt
  done
  for pid in $(awk '$3 == "root" {print $1}' $D/smaps/pids.txt | head -1) $(awk '$3 == "clone" {print $1}' $D/smaps/pids.txt | head -3); do
    cp /proc/$pid/smaps $D/smaps/$pid.smaps 2>/dev/null; cp /proc/$pid/smaps_rollup $D/smaps/$pid.rollup 2>/dev/null
  done
  echo "$(date '+%F %T') smaps survey: $(ls $D/smaps | wc -l) files" >> $S ) &
SM=$!
train r0 12
touch $D/r0.done; wait $SM
train r1 12 -e ISAAC_RL_NATIVE_VISIBLE=1 -e ISAAC_RL_V2LOG_COMPACT=1
train r2 6 -e ISAAC_RL_NATIVE_VISIBLE=2
note "5 r2 native visible check: $(head -1 $D/r2-clone-stats.txt)"

# 6 the gate (optional; under this batch's lock)
if [ "${GATE:-0}" = 1 ]; then
  give
  GPU=$GPU sh $BM/gate3.sh $TREE gate3-$TREE-$TAG
  note "6 gate3: $(grep "gate3-$TREE-$TAG" $R/bm1.log | tail -1 | cut -c1-160)"
else
  give
fi
note "$TAG end"
host_snapshot "$TAG end"
