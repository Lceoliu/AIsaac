#!/bin/sh
# bm1: the 4090's B54 full gate (run_b54g.sh) in the container, under the lane lock (not timed): the train_tok entry
# gate (tests/test_b52_entry.py, ISAAC_B52_ENTRY=1), the whole test suite (-v, CUBLAS workspace, the b45 payloads), the
# A8 play digest (abplus_probe_play.py, 8 seeds, 300 steps). Logs under reports/, a failed part in <tag>.failed.
#   usage: [LANE=cpp] [GPU=4] gate3.sh <tree> [tag]
set -u
. "$(dirname "$0")/lane.sh"
. $OPS/lock.sh
TREE=$1; TAG=${2:-gate3-$1-$(date +%m%d-%H%M%S)}
GPU=${GPU:-${GPUS_OK%%,*}}
R=$LD/reports; LOG=$R/bm1.log
mkdir -p $R/failed
PY=/bench/home/isaac-abplus/.venv/bin/python
T=$LC/trees/$TREE
GAME="-e ABP_HOME=$LC/home/isaac-abplus -e ABP_BRIDGE_LUA=$T/bridge/abplus/abp_bridge.lua -e LIBGL_ALWAYS_SOFTWARE=1 -e SDL_AUDIODRIVER=dummy -e ALSOFT_DRIVERS=null"
take "gate3 $TREE: entry + full tests + A8 (GPU $GPU, not timed)"
echo "$(date '+%F %T') $TAG start: tree $TREE GPU $GPU lock: $LOCKLINE" >> $LOG
rm -f $R/$TAG.failed
DOCK_ENV="-e ISAAC_B52_ENTRY=1 $GAME ${ENVS:-}" $OPS/dock.sh gate3-entry $GPU sh -c \
  "cd $T/bridge/python && timeout 3600 $PY -m pytest tests/test_b52_entry.py -q -p no:cacheprovider --basetemp=$LC/reports/$TAG-entry" > $R/$TAG.entry.log 2>&1
echo "$(date '+%F %T') $TAG entry rc=$? $(tail -1 $R/$TAG.entry.log | cut -c1-160)" >> $LOG
tail -1 $R/$TAG.entry.log | grep -q " passed" && ! tail -1 $R/$TAG.entry.log | grep -qE "[0-9]+ (failed|skipped|errors?)( |,|$)" || echo "entry" >> $R/$TAG.failed
DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e ISAAC_VISIBLE_PAYLOADS=/bench/bm1/payloads/b45-payloads2 ${ENVS:-}" $OPS/dock.sh gate3-full $GPU sh -c \
  "cd $T/bridge/python && timeout 3600 $PY -m pytest tests -v --tb=long -p no:cacheprovider" > $R/$TAG.full.log 2>&1
echo "$(date '+%F %T') $TAG full rc=$? $(tail -1 $R/$TAG.full.log | cut -c1-160)" >> $LOG
tail -1 $R/$TAG.full.log | grep -q " passed" && ! tail -1 $R/$TAG.full.log | grep -qE "[0-9]+ (failed|errors?)( |,|$)" || { echo "full" >> $R/$TAG.failed; cp $R/$TAG.full.log $R/failed/; }
DOCK_ENV="-e OMP_NUM_THREADS=1 $GAME" $OPS/dock.sh gate3-a8 $GPU sh -c \
  "cd $T/bridge/python && timeout 1800 xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset' $PY abplus_probe_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147500000:8 --steps 300 --out $LC/reports/$TAG-a8" > $R/$TAG.a8.log 2>&1
echo "$(date '+%F %T') $TAG a8 rc=$? $(tail -1 $R/$TAG.a8.log | cut -c1-200)" >> $LOG
tail -1 $R/$TAG.a8.log | grep -q '"seeds": 8, "ok": 8' || echo "A8" >> $R/$TAG.failed
[ -f $R/$TAG.failed ] && echo "$(date '+%F %T') $TAG GATE FAILED: $(tr '\n' ' ' < $R/$TAG.failed)" >> $LOG || echo "$(date '+%F %T') $TAG gate passed" >> $LOG
give
