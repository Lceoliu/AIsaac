#!/bin/sh
# bm1: the full test gate of a tree (as the 4090's b62_step0.sh G), in the container, under the lane lock.
# Log: reports/tests-full-<tree>-<mmdd-HHMM>.log, never overwritten; failures copied to reports/failed/.
#   usage: [GPU=1] [RUNS=1] gate.sh <tree> [pytest args...]
set -u
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
. $BM/lock.sh
TREE=$1; shift
GPU=${GPU:-1}; RUNS=${RUNS:-1}
R=$BM/reports; LOG=$R/bm1.log
mkdir -p $R/failed
i=1
while [ $i -le $RUNS ]; do
  take "full gate $TREE run $i/$RUNS (GPU $GPU, not timed)"
  GL=$R/tests-full-$TREE-$(date +%m%d-%H%M%S).log
  DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e ISAAC_VISIBLE_PAYLOADS=/bench/bm1/payloads/b45-payloads2 ${ENVS:-}" \
    $BM/dock.sh gate-$TREE $GPU sh -c "cd /bench/bm1/trees/$TREE/bridge/python && timeout 3000 /bench/home/isaac-abplus/.venv/bin/python -m pytest tests -q -p no:cacheprovider $*" > $GL 2>&1
  rc=$?
  echo "$(date '+%F %T') gate $TREE run $i/$RUNS (capture mode $(capmode $TREE)) rc=$rc $(tail -1 $GL | cut -c1-200) ($GL)" >> $LOG
  [ $rc != 0 ] && cp $GL $R/failed/
  give
  i=$((i+1))
done
