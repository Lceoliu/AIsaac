#!/bin/sh
# bm1 (measurement, not timed): Nsight Systems with nvtx + python-gil over hand-offs RANGE of a run (split by default:
# actor GPU + learner GPU, EXTRA=--learner-device=cuda:1), then b57_nsys.py: GIL held / waited per thread and per
# cycle, the main thread's GIL acquisitions (Python entries) per cycle, its ranges (c.wait, h.*), the learner's
# segments. The lane from bm1.env, under the lane lock; export and analysis inside it.
#   usage: [GPU=a,b] [RANGE=2:5] [STOP_ROWS=6] [ENVS="-e K=V"] nsys_gil.sh <tag> <tree>   (lane: bm1.env; BM1_NSYS)
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TAG=$1; TREE=$2
GPU=${GPU:-$BM1_GPUS_OK}; RANGE=${RANGE:-2:5}; STOP_ROWS=${STOP_ROWS:-6}
NSYS=${BM1_NSYS:?BM1_NSYS (nsys as the runner sees it) in bm1.env}
[ -e $R/$TAG ] && { echo "nsys_gil.sh: $R/$TAG exists" >&2; exit 2; }
export PIDS=${PIDS:-16384}   # (nsys puts threads into every traced process: 4096 pids were not enough here, A1)
take "nsys gil $TAG ($TREE, GPU $GPU, not timed)"
echo "$(date '+%F %T') nsys gil $TAG start: tree $TREE GPU $GPU range $RANGE lock: $LOCKLINE" >> $LOG
NS="$NSYS profile -t nvtx,python-gil --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop -o $BM1_IN/reports/$TAG.nsys-rep --force-overwrite true"
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 -e ISAAC_BLOCK_NVTX=1 -e ISAAC_NSYS_RANGE=$RANGE -e ISAAC_NSYS_TRIGGER=nvtx -e EXTRA=${SPLIT_EXTRA:---learner-device=cuda:1} ${ENVS:-}" \
  $BM/run.sh nsys-$TAG $GPU sh -c "NSYS_PREFIX='$NS' sh $BM1_IN/xnew2.sh $TAG 3000 30.0" &
DP=$!
while kill -0 $DP 2>/dev/null; do
  sleep 10
  n=0; [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
  if [ $n -ge $STOP_ROWS ]; then
    sigint_trainer nsys-$TAG $TAG
    break
  fi
done
wait $DP
echo "$(date '+%F %T') nsys gil $TAG run rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
$BM/run.sh nsys-export-$TAG ${GPU%%,*} sh -c "$NSYS export --type sqlite --force-overwrite true -o $BM1_IN/reports/$TAG.sqlite $BM1_IN/reports/$TAG.nsys-rep" > $R/$TAG.export.log 2>&1
python3 $BM/trees/$TREE/bridge/ops/b57_nsys.py $R/$TAG.sqlite > $R/$TAG.analysis.txt 2>&1
echo "$(date '+%F %T') nsys gil $TAG analysis rc=$? $(grep 'GIL acquisitions' $R/$TAG.analysis.txt | cut -c1-120)" >> $LOG
[ -d $R/$TAG/checkpoints ] && trash $R/$TAG/checkpoints
give
