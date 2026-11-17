#!/bin/sh
# ab1 held-out evaluations (the coordinator, 2026-11-14; C106): as evtools/ev_main1.sh (C99's protocol: tree maa1e98a's
# eval_tok.py --mode run --items, sampled actions, run 1800 s, floor stall 60 s on events as a terminal end, Isaac, no
# build, no forced pick-up; through evtools/eval_ent.py: + entropy.json, config.json), for one ab1 run: its checkpoints
# under /root/AIsaac/train/ab1/<run>/<run>-s0/checkpoints. One container job per evaluation under the bm1 lock (job.sh:
# shared lock), inference on cuda:0 of the given GPU, the container on the given CPUs / memory node; sequential; a
# finished one (summary.json) is skipped. Writes only /root/AIsaac/train/ab1/eval/<tag>/ (new) and bm1/reports.
#   usage: ev_ab1.sh <run> <gpu> <cpuset> <mems> <workers> <tag>,<checkpoint file>,<first:count> ...
set -u
BM=/root/AIsaac/bm1
RUN=$1; G=$2; CS=$3; MS=$4; W=$5; shift 5
HOST=/root/AIsaac/train/ab1/eval
mkdir -p $HOST
echo "$(date '+%F %T') ev_ab1: $RUN GPU $G CPUs $CS workers $W: $*" >> $BM/reports/bm1.log
for spec in "$@"; do
  tag=$(echo "$spec" | cut -d, -f1); ck=$(echo "$spec" | cut -d, -f2); seeds=$(echo "$spec" | cut -d, -f3)
  [ -f $HOST/$tag/summary.json ] && { echo "skip $tag (done)"; continue; }
  [ -e $HOST/$tag ] && { echo "skip $tag: $HOST/$tag exists without summary.json (a failed run: look)"; continue; }
  [ -f /root/AIsaac/train/ab1/$RUN/$RUN-s0/checkpoints/$ck ] || { echo "skip $tag: no checkpoint $ck"; continue; }
  t0=$(date +%s)
  L=$(GPU=$G CPUSET=$CS MEMS=$MS ENVS="-e TOK_EVAL_DEVICE=cuda" sh $BM/job.sh ev-$RUN "cd /bench/bm1/trees/maa1e98a/bridge/python && export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null && xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset' \$BM1_PY -u /bench/bm1/evtools/eval_ent.py --checkpoint /bench/train/ab1/$RUN/$RUN-s0/checkpoints/$ck --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers $W --seeds $seeds --name abev --port 41300 --out /bench/train/ab1/eval/$tag" | tail -1)
  cp "$L" $HOST/$tag.log 2>/dev/null
  echo "$(date '+%F %T') $tag $ck $seeds: $(( $(date +%s) - t0 )) s, summary $( [ -f $HOST/$tag/summary.json ] && echo yes || echo NO ) ($L)"
done
echo "EV_AB1 DONE $RUN"
