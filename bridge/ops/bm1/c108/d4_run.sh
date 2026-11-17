#!/bin/sh
# C108 D4b (2026-11-15, the coordinator): frozen-weight dumps of representations + labels for linear probes. Each run:
# a checkpoint copy in c108/d1/inputs, --lr 0 --lr-final 0, two updates (block: --max-updates 2; old: --game-hours 2.5),
# the throwaway tree copy with d4hook (ISAAC_D4_DUMP=1) -> /root/AIsaac/train/c108/d4/<tag>/d4_u*.npz. One container
# job each (job.sh: shared lock), GPUs 4(+5), node0's CPUs, sequential; a tag whose directory exists is skipped.
#   usage: d4_run.sh <tag>:<new|old>:<input checkpoint name in c108/d1/inputs> ...
set -u
BM=/root/AIsaac/bm1
D=/root/AIsaac/train/c108/d4
mkdir -p $D
echo "$(date '+%F %T') d4_run: $*" >> $BM/reports/bm1.log
for spec in "$@"; do
  tag=$(echo "$spec" | cut -d: -f1); kind=$(echo "$spec" | cut -d: -f2); ck=$(echo "$spec" | cut -d: -f3)
  [ -e $D/$tag ] && { echo "skip $tag (exists)"; continue; }
  t0=$(date +%s)
  if [ "$kind" = new ]; then
    { cat /root/AIsaac/train/ab2/ab2-bm/train_args; echo "--lr 0 --lr-final 0"; } > $D/$tag.args
    L=$(GPU=4,5 MEM=80g sh $BM/job.sh d4-$tag "cd /bench && export CODE=maa1e98a-d4 OUTDIR=/bench/train/c108/d4 NAME=d4n PORT=41500 FLAGS_FILE=/bench/train/c108/d4/$tag.args EXTRA=--learner-device=cuda:1 EXTRA2=--max-updates=2 EXTRA3=--resume=/bench/train/c108/d1/inputs/$ck ISAAC_D4_DUMP=1 && sh /bench/bm1/xnew2.sh $tag 3600 1200" | tail -1)
  else
    { cat /root/AIsaac/train/ab2/ab2-c99old/train_args; echo "--lr 0 --lr-final 0"; } > $D/$tag.args
    L=$(GPU=4 MEM=80g sh $BM/job.sh d4-$tag "cd /bench && export CODE=m1ae82d3-d4 OUTDIR=/bench/train/c108/d4 NAME=d4o PORT=41700 FLAGS_FILE=/bench/train/c108/d4/$tag.args EXTRA3=--resume=/bench/train/c108/d1/inputs/$ck ISAAC_D4_DUMP=1 && sh /bench/bm1/xold_c99.sh $tag 3600 2.5" | tail -1)
  fi
  cp "$L" $D/$tag.job.log 2>/dev/null
  echo "$(date '+%F %T') $tag ($kind, $ck): $(( $(date +%s) - t0 )) s, $(ls $D/$tag/d4_u*.npz 2>/dev/null | wc -l) dumps ($L)"
done
echo "D4_RUN DONE"
