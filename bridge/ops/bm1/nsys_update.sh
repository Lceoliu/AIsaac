#!/bin/sh
# bm1 (BM1 step 4, measurement, not timed): Nsight Systems (cuda + nvtx) over a window of hand-offs of a split run
# (actor GPU / learner GPU), then the learner's sections split into GPU time and host waiting
# (bridge/ops/b62_learner_sections.py). Under the lane lock; the export and the analysis inside it too.
#   usage: [GPU=1,2] [RANGE=10:12] [STOP_ROWS=13] nsys_update.sh <tag> <tree>
set -u
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
. $BM/lock.sh
TAG=$1; TREE=$2
GPU=${GPU:-1,2}; RANGE=${RANGE:-10:12}; STOP_ROWS=${STOP_ROWS:-13}
R=$BM/reports; LOG=$R/bm1.log
NSYS=/bench/bm1/tools/nsys-2023.4.4/opt/nvidia/nsight-systems/2023.4.4/bin/nsys
[ -e $R/$TAG ] && { echo "nsys_update.sh: $R/$TAG exists" >&2; exit 2; }
take "nsys $TAG ($TREE, GPU $GPU, not timed)"
C=aisaac-bench-bm1-a1-nsys-$TAG
NS="$NSYS profile -t cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=graph --capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop -o /bench/bm1/reports/$TAG.nsys-rep --force-overwrite true"
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 -e ISAAC_BLOCK_NVTX=1 -e ISAAC_NSYS_RANGE=$RANGE -e ISAAC_NSYS_TRIGGER=nvtx -e EXTRA=--learner-device=cuda:1" \
  PIDS=16384 $BM/dock.sh nsys-$TAG $GPU sh -c "NSYS_PREFIX='$NS' sh /bench/bm1/xnew2.sh $TAG 3000 30.0" &
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
echo "$(date '+%F %T') nsys $TAG run rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
$BM/dock.sh nsys-export-$TAG ${GPU%%,*} sh -c "$NSYS export --type sqlite --force-overwrite true -o /bench/bm1/reports/$TAG.sqlite /bench/bm1/reports/$TAG.nsys-rep" > $R/$TAG.export.log 2>&1
python3 $BM/trees/$TREE/bridge/ops/b62_learner_sections.py $R/$TAG.sqlite > $R/$TAG.sections.txt 2>&1
echo "$(date '+%F %T') nsys $TAG analysis rc=$? $(tail -1 $R/$TAG.sections.txt)" >> $LOG
[ -d $R/$TAG/checkpoints ] && gio trash $R/$TAG/checkpoints
give
