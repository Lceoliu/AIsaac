#!/bin/sh
# bm1 (BM1 decision a): a short run ended by SIGINT at a chosen moment, to check the shutdown: the return code, no
# std::terminate, last.pt saved, where the interrupt landed (the traceback's frames, or train_tok's "BM1 shutdown"
# line when it arrived during the shutdown itself). Not timed; under the lane lock.
#   ENVS="-e K=V ..." goes to the run (e.g. -e ISAAC_BLOCK_CPP_LANE_THREAD=1: the lane thread's shutdown).
#   usage: [GPU=1,2] [SPLIT=1] [AFTER_ROW=2] [FRAC=0.8] [HOURS=30.0] sigint_smoke.sh <tag> <tree>
#   HOURS small (e.g. 2.5: the loop ends cleanly after the third hand-off) puts a SIGINT a few seconds after row 2 into
#   the shutdown's own wait for the learner's last update (the path that skipped the join and the save before BM1 a)
#   SIGINT goes FRAC x (that row's update_s) seconds after progress row AFTER_ROW appears (with the learner on its own
#   GPU the actor then usually waits for the learner: wait_learner).
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
TAG=$1; TREE=$2
GPU=${GPU:-1,2}; SPLIT=${SPLIT:-1}; AFTER_ROW=${AFTER_ROW:-2}; FRAC=${FRAC:-0.8}
R=$BM/reports; LOG=$R/bm1.log
[ -e $R/$TAG ] && { echo "sigint_smoke.sh: $R/$TAG exists" >&2; exit 2; }
take "sigint smoke $TAG ($TREE, GPU $GPU, not timed)"
X=""; [ "$SPLIT" = 1 ] && X="-e EXTRA=--learner-device=cuda:1"
DOCK_ENV="-e CODE=$TREE -e ISAAC_BLOCK_TIMING=1 $X ${ENVS:-}" $BM/run.sh sig-$TAG $GPU sh $BM1_IN/xnew2.sh $TAG 3000 ${HOURS:-30.0} &
DP=$!
while kill -0 $DP 2>/dev/null; do
  n=0; [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
  [ $n -ge $AFTER_ROW ] && break
  sleep 0.5
done
u=$(tail -1 $R/$TAG/progress.csv | cut -d, -f6)
d=$(python3 -c "print(round(float('$u')*$FRAC, 2))")
sleep $d
sigint_trainer sig-$TAG $TAG
t_int=$(date '+%T')
wait $DP
rc=$?
ck=no; [ -f $R/$TAG/checkpoints/last.pt ] && ck=yes
term=$(grep -c 'terminate called' $R/$TAG.stdout $R/$TAG.stderr 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
segv=$(grep -c 'Segmentation fault\|dumped core' $R/$TAG.stdout $R/$TAG.stderr 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
held=$(grep -h 'BM1 shutdown' $R/$TAG.stdout | head -1 | cut -c1-120)
where=$(grep -h -B1 '^KeyboardInterrupt' $R/$TAG.stdout $R/$TAG.stderr 2>/dev/null | grep -v '^KeyboardInterrupt' | grep -v '^--' | tail -2 | tr '\n' ' ' | cut -c1-200)
rows=$(($(wc -l < $R/$TAG/progress.csv)-1))
echo "$(date '+%F %T') sigint smoke $TAG: SIGINT at $t_int ($d s after row $AFTER_ROW, update_s $u); rc=$rc, terminate $term, segv $segv, last.pt $ck, rows $rows; held: ${held:-none}; landed: ${where:-?}" >> $LOG
[ -d $R/$TAG/checkpoints ] && trash $R/$TAG/checkpoints
give
