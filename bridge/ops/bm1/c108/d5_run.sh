#!/bin/sh
# C109 D5 (2026-11-15, the coordinator): gradient consistency at frozen weights (d5hook, ISAAC_D5_GRAD=1). Each run:
# --lr 0 --lr-final 0, sixteen updates (v2; block: --max-updates 16; the old tree: --game-hours 19.8, about 17 updates,
# its budget incremental on --resume; the summary takes the first 16). Kind "check" (block): ISAAC_D5_CHECK=1 instead,
# one update (d5hook's same-batch donated-buffer check -> d5_check.json), the throwaway tree copy -> /root/AIsaac/train/c109/d5/<tag>/ (d5_grad.jsonl, d5_gram.json). Block runs
# with --micro 2048 (a memory knob only: the same minibatch and step; the hook's extra backward passes keep the part's
# graph). One container job each (job.sh: shared lock) on GPUs $GPUS (default 6,7) and CPUs $CPUS (8-15,24-31), node
# $MEMS (1), sequential; a tag whose directory exists is skipped.
# (C110, 2026-11-15: D5_SUB=<dir under /root/AIsaac/train, default c109/d5> the output; D5_HEADS=1 adds ISAAC_D5_HEADS=1
#  (d5hook v3: the policy term per action head -> d5h_grad.jsonl / d5h_gram.json))
#   usage: d5_run.sh <tag>:<new|old|check>:<input checkpoint in c108/d1/inputs, or "fresh"> ...
set -u
BM=/root/AIsaac/bm1
SUB=${D5_SUB:-c109/d5}
D=/root/AIsaac/train/$SUB
HF=""; [ "${D5_HEADS:-0}" = 1 ] && HF="ISAAC_D5_HEADS=1"
GP=${GPUS:-6,7}; CP=${CPUS:-8-15,24-31}; MP=${MEMS:-1}
G1=$(echo $GP | cut -d, -f1)
mkdir -p $D
echo "$(date '+%F %T') d5_run (GPUs $GP, CPUs $CP): $*" >> $BM/reports/bm1.log
for spec in "$@"; do
  tag=$(echo "$spec" | cut -d: -f1); kind=$(echo "$spec" | cut -d: -f2); ck=$(echo "$spec" | cut -d: -f3)
  [ -e $D/$tag ] && { echo "skip $tag (exists)"; continue; }
  if [ "$ck" = fresh ]; then RES=""; else
    [ -f /root/AIsaac/train/c108/d1/inputs/$ck ] || { echo "skip $tag: no input $ck"; continue; }
    RES="EXTRA3=--resume=/bench/train/c108/d1/inputs/$ck"; fi
  t0=$(date +%s)
  if [ "$kind" = new ] || [ "$kind" = check ]; then
    if [ "$kind" = new ]; then MU=16; FL=ISAAC_D5_GRAD=1; else MU=1; FL=ISAAC_D5_CHECK=1; fi
    { cat /root/AIsaac/train/ab2/ab2-bm/train_args; echo "--lr 0 --lr-final 0 --micro 2048"; } > $D/$tag.args
    L=$(GPU=$GP CPUSET=$CP MEMS=$MP MEM=80g sh $BM/job.sh d5-$tag "cd /bench && export CODE=maa1e98a-d5 OUTDIR=/bench/train/$SUB NAME=d5n PORT=41500 FLAGS_FILE=/bench/train/$SUB/$tag.args EXTRA=--learner-device=cuda:1 EXTRA2=--max-updates=$MU $RES $FL $HF && sh /bench/bm1/xnew2.sh $tag 5400 1200" | tail -1)
  else
    { cat /root/AIsaac/train/ab2/ab2-c99old/train_args; echo "--lr 0 --lr-final 0"; } > $D/$tag.args
    L=$(GPU=$G1 CPUSET=$CP MEMS=$MP MEM=80g sh $BM/job.sh d5-$tag "cd /bench && export CODE=m1ae82d3-d5 OUTDIR=/bench/train/$SUB NAME=d5o PORT=41700 FLAGS_FILE=/bench/train/$SUB/$tag.args $RES ISAAC_D5_GRAD=1 $HF && sh /bench/bm1/xold_c99.sh $tag 5400 19.8" | tail -1)
  fi
  cp "$L" $D/$tag.job.log 2>/dev/null
  echo "$(date '+%F %T') $tag ($kind, $ck): $(( $(date +%s) - t0 )) s, $(cat $D/$tag/d5_grad.jsonl $D/$tag/d5h_grad.jsonl 2>/dev/null | wc -l) grad lines ($L)"
done
echo "D5_RUN DONE"
