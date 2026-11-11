#!/bin/sh
# bm1 (measurement, not timed): Nsight Systems with nvtx + python-gil over hand-offs RANGE of a run (split by default:
# actor GPU + learner GPU, EXTRA=--learner-device=cuda:1), then b57_nsys.py: GIL held / waited per thread and per
# cycle, the main thread's GIL acquisitions (Python entries) per cycle, its ranges (c.wait, h.*), the learner's
# segments. Lane-aware (lane.sh), under the lane lock; export and analysis inside it.
#   usage: [LANE=cpp] [GPU=4,5] [RANGE=2:5] [STOP_ROWS=6] [ENVS="-e K=V"] nsys_gil.sh <tag> <tree>
set -u
. "$(dirname "$0")/lane.sh"
. $OPS/lock.sh
TAG=$1; TREE=$2
GPU=${GPU:-4,5}; RANGE=${RANGE:-2:5}; STOP_ROWS=${STOP_ROWS:-6}
R=$LD/reports; LOG=$R/bm1.log
NSYS=/bench/bm1/tools/nsys-2023.4.4/opt/nvidia/nsight-systems/2023.4.4/bin/nsys
[ -e $R/$TAG ] && { echo "nsys_gil.sh: $R/$TAG exists" >&2; exit 2; }
take "nsys gil $TAG ($TREE, GPU $GPU, not timed)"
echo "$(date '+%F %T') nsys gil $TAG start: tree $TREE GPU $GPU range $RANGE lock: $LOCKLINE" >> $LOG
C=$NAME-nsys-$TAG
NS="$NSYS profile -t nvtx,python-gil --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop -o $LC/reports/$TAG.nsys-rep --force-overwrite true"
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 -e ISAAC_BLOCK_NVTX=1 -e ISAAC_NSYS_RANGE=$RANGE -e ISAAC_NSYS_TRIGGER=nvtx -e EXTRA=${SPLIT_EXTRA:---learner-device=cuda:1} ${ENVS:-}" \
  $OPS/dock.sh nsys-$TAG $GPU sh -c "NSYS_PREFIX='$NS' sh $OPC/xnew2.sh $TAG 3000 30.0" &
DP=$!
while kill -0 $DP 2>/dev/null; do
  sleep 10
  n=0; [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
  if [ $n -ge $STOP_ROWS ]; then
    docker exec $C pkill -INT -f '^/bench/home/isaac-abplus/.venv/bin/python -u train_tok.py'
    break
  fi
done
wait $DP
echo "$(date '+%F %T') nsys gil $TAG run rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
$OPS/dock.sh nsys-export-$TAG ${GPU%%,*} sh -c "$NSYS export --type sqlite --force-overwrite true -o $LC/reports/$TAG.sqlite $LC/reports/$TAG.nsys-rep" > $R/$TAG.export.log 2>&1
python3 $LD/trees/$TREE/bridge/ops/b57_nsys.py $R/$TAG.sqlite > $R/$TAG.analysis.txt 2>&1
echo "$(date '+%F %T') nsys gil $TAG analysis rc=$? $(grep 'GIL acquisitions' $R/$TAG.analysis.txt | cut -c1-120)" >> $LOG
[ -d $R/$TAG/checkpoints ] && gio trash $R/$TAG/checkpoints
give
