#!/bin/sh
# bm1: B62 capture probes (bridge/ops/b62_capture_probe.py) of a tree, in the container, under the lane lock (not timed).
#   usage: [GPU=1] probe.sh <tree> [case ...]
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TREE=$1; shift
GPU=${GPU:-1}
R=$BM/reports; LOG=$R/bm1.log
take "capture probe $TREE (GPU $GPU, not timed)"
PL=$R/b62-capture-probe-$TREE-$(date +%m%d-%H%M%S).log
DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8" $BM/run.sh probe-$TREE $GPU sh -c \
  "cd $BM1_IN/trees/$TREE/bridge/python && timeout 1800 $BM1_PY ../ops/b62_capture_probe.py $*" > $PL 2>&1
echo "$(date '+%F %T') probe $TREE rc=$? ($PL)" >> $LOG
give
