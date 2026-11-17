#!/bin/sh
# C108 D1 (2026-11-15, the coordinator): actor vs learner log-probabilities at fixed weights. Each run: a trained
# checkpoint (a copy in /root/AIsaac/train/c108/d1/inputs/, the ab1 / ab2 files only read), --lr 0 --lr-final 0, five
# updates, the throwaway tree copy with d1diag (ISAAC_D1_DIAG=1) -> /root/AIsaac/train/c108/d1/<tag>/ (d1_diag.jsonl,
# d1_raw_u*.npz, progress.csv, checkpoints/last.pt). One container job each (job.sh: shared lock), GPUs 4(+5),
# node0's CPUs, sequential; a tag whose directory exists is skipped.
# (C110, 2026-11-15: D1_GPUS (default 4,5), D1_CPUSET, D1_MEMS for the new kind -- the fp32 check on whichever lane is free;
#  D1_CODE (default maa1e98a-d1) and D1_ARGS (default ab2-bm's train_args) for the new kind -- Q@maa: maa1e98a-nv-d1)
#   usage: d1_run.sh <tag>:<new|old>:<input checkpoint name in inputs/>[:<extra flags, comma-separated>][:<env K=V,...>] ...
set -u
BM=/root/AIsaac/bm1
D=/root/AIsaac/train/c108/d1
echo "$(date '+%F %T') d1_run: $*" >> $BM/reports/bm1.log
for spec in "$@"; do
  tag=$(echo "$spec" | cut -d: -f1); kind=$(echo "$spec" | cut -d: -f2); ck=$(echo "$spec" | cut -d: -f3)
  xf=$(echo "$spec" | cut -d: -f4 | tr ',' ' '); xe=$(echo "$spec" | cut -d: -f5 | tr ',' ' ')
  [ -e $D/$tag ] && { echo "skip $tag (exists)"; continue; }
  [ -f $D/inputs/$ck ] || { echo "skip $tag: no $D/inputs/$ck"; continue; }
  t0=$(date +%s)
  if [ "$kind" = new ]; then
    { cat ${D1_ARGS:-/root/AIsaac/train/ab2/ab2-bm/train_args}; echo "--lr 0 --lr-final 0 $xf"; } > $D/$tag.args
    L=$(env GPU=${D1_GPUS:-4,5} ${D1_CPUSET:+CPUSET=$D1_CPUSET} ${D1_MEMS:+MEMS=$D1_MEMS} MEM=80g sh $BM/job.sh d1-$tag "cd /bench && export $xe CODE=${D1_CODE:-maa1e98a-d1} OUTDIR=/bench/train/c108/d1 NAME=d1n PORT=41500 FLAGS_FILE=/bench/train/c108/d1/$tag.args EXTRA=--learner-device=cuda:1 EXTRA2=--max-updates=5 EXTRA3=--resume=/bench/train/c108/d1/inputs/$ck ISAAC_D1_DIAG=1 ISAAC_BLOCK_TIMING=1 && sh /bench/bm1/xnew2.sh $tag 3600 1200" | tail -1)
  else
    { cat /root/AIsaac/train/ab2/ab2-c99old/train_args; echo "--lr 0 --lr-final 0 $xf"; } > $D/$tag.args
    L=$(GPU=4 MEM=80g sh $BM/job.sh d1-$tag "cd /bench && export $xe CODE=m1ae82d3-d1 OUTDIR=/bench/train/c108/d1 NAME=d1o PORT=41700 FLAGS_FILE=/bench/train/c108/d1/$tag.args EXTRA3=--resume=/bench/train/c108/d1/inputs/$ck ISAAC_D1_DIAG=1 && sh /bench/bm1/xold_c99.sh $tag 3600 6" | tail -1)
  fi
  cp "$L" $D/$tag.job.log 2>/dev/null
  echo "$(date '+%F %T') $tag ($kind, $ck): $(( $(date +%s) - t0 )) s, $(wc -l < $D/$tag/d1_diag.jsonl 2>/dev/null || echo 0) diag lines ($L)"
done
echo "D1_RUN DONE"
