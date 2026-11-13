#!/bin/sh
# bm1: the full test gate of a tree (as the 4090's b62_step0.sh G), in the container, under the lane lock.
# Log: reports/tests-full-<tree>-<mmdd-HHMM>.log, never overwritten; failures copied to reports/failed/.
#   usage: [GPU=1] [RUNS=1] gate.sh <tree> [pytest args...]
# rc 9: the tests passed but a native component did not build / load (native_check.py) or its tests skipped.
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TREE=$1; shift
GPU=${GPU:-1}; RUNS=${RUNS:-1}
R=$BM/reports; LOG=$R/bm1.log
mkdir -p $R/failed
i=1
while [ $i -le $RUNS ]; do
  take "full gate $TREE run $i/$RUNS (GPU $GPU, not timed)"
  GL=$R/tests-full-$TREE-$(date +%m%d-%H%M%S).log
  DOCK_ENV="-e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e ISAAC_VISIBLE_PAYLOADS=${BM1_PAYLOADS:-$BM1_IN/payloads}/b45-payloads2 ${ENVS:-}" \
    $BM/run.sh gate-$TREE $GPU sh -c "cd $BM1_IN/trees/$TREE/bridge/python && $BM1_PY $BM1_IN/native_check.py && timeout 3000 $BM1_PY -m pytest tests -q -rs -p no:cacheprovider $*" > $GL 2>&1
  rc=$?
  # (2026-11-12, the coordinator: a native component's tests skip when it is not built, so a failed build passed
  #  silently: native_check.py's EXT lines all ok, and no skip for a native component / CUDA / the step graph -- else
  #  the gate fails, rc 9)
  nat=$(grep -a '^EXT ' $GL | paste -sd ' ' | cut -c1-200)
  nskip=$(grep -a -c -E '^SKIPPED.*(extension unavailable|C visible encoder|gcc unavailable|CUDA|GPU|ISAAC_BLOCK_STEP_GRAPH=0|memfd|gloo)' $GL)
  if [ $rc = 0 ] && { grep -a -q '^EXT .* FAIL' $GL || [ "$(grep -a -c '^EXT .* ok' $GL)" -lt 3 ] || [ "$nskip" -gt 0 ]; }; then rc=9; fi
  echo "$(date '+%F %T') gate $TREE run $i/$RUNS (capture mode $(capmode $TREE)) rc=$rc $(tail -1 $GL | cut -c1-200); $nat; native skips $nskip ($GL)" >> $LOG
  [ $rc != 0 ] && cp $GL $R/failed/
  give
  i=$((i+1))
done
