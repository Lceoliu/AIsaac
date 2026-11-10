#!/bin/sh
# bm1: B62 capture probes (bridge/ops/b62_capture_probe.py) of a tree, in the container, under the lane lock (not timed).
#   usage: [GPU=1] probe.sh <tree> [case ...]
set -u
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
. $BM/lock.sh
TREE=$1; shift
GPU=${GPU:-1}
R=$BM/reports; LOG=$R/bm1.log
take "capture probe $TREE (GPU $GPU, not timed)"
PL=$R/b62-capture-probe-$TREE-$(date +%m%d-%H%M%S).log
DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8" $BM/dock.sh probe-$TREE $GPU sh -c \
  "cd /bench/bm1/trees/$TREE/bridge/python && timeout 1800 /bench/home/isaac-abplus/.venv/bin/python ../ops/b62_capture_probe.py $*" > $PL 2>&1
echo "$(date '+%F %T') probe $TREE rc=$? ($PL)" >> $LOG
give
