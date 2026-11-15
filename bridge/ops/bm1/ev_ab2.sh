#!/bin/sh
# ab2 held-out evaluations (C107, 2026-11-14, the coordinator): 128 seeds, sampled actions, no forced pick-up, no build,
# Isaac, run mode with items. One container job per evaluation under the bm1 lock (job.sh: shared lock), the container
# on the given GPU / CPUs / memory node; sequential; a finished one (summary.json) is skipped. Writes only
# /root/AIsaac/train/ab2/eval/<tag>/ (new), <tag>.log beside it, and bm1/reports.
#   kind old (arm A, ab2-c99old): C99's own protocol -- tree m1ae82d3's eval_tok.py with hpc_c99.sh "watch"'s flags
#     (--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 --seeds <s> --name hc99e
#     --port 49300) and its environment (OMP_NUM_THREADS=4, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, inference
#     on the CPU: no TOK_EVAL_DEVICE, as C99's watcher), ABP_HOME home-c99 (its own libabp_turbo.so); run 1800 s and
#     floor stall 60 s are that eval_tok's defaults. <workers> is ignored (8, as C99).
#   kind new (arm B, ab2-bm): exactly ev_ab1.sh -- tree maa1e98a's eval_tok.py through evtools/eval_ent.py (+ entropy.json,
#     config.json), --mode run --items --workers <workers>, TOK_EVAL_DEVICE=cuda, --name abev --port 41300.
#   usage: ev_ab2.sh <run> <old|new> <gpu> <cpuset> <mems> <workers> <tag>,<checkpoint file>,<first:count> ...
set -u
BM=/root/AIsaac/bm1
RUN=$1; KIND=$2; G=$3; CS=$4; MS=$5; W=$6; shift 6
PAR=${PARENT:-ab2}   # (C108, 2026-11-15: PARENT=ab3 for ab3's runs; ab2 as before)
HOST=/root/AIsaac/train/$PAR/eval
mkdir -p $HOST
echo "$(date '+%F %T') ev_ab2: $RUN ($KIND) GPU $G CPUs $CS workers $W: $*" >> $BM/reports/bm1.log
X="xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset'"
for spec in "$@"; do
  tag=$(echo "$spec" | cut -d, -f1); ck=$(echo "$spec" | cut -d, -f2); seeds=$(echo "$spec" | cut -d, -f3)
  [ -f $HOST/$tag/summary.json ] && { echo "skip $tag (done)"; continue; }
  [ -e $HOST/$tag ] && { echo "skip $tag: $HOST/$tag exists without summary.json (a failed run: look)"; continue; }
  [ -f /root/AIsaac/train/$PAR/$RUN/$RUN-s0/checkpoints/$ck ] || { echo "skip $tag: no checkpoint $ck"; continue; }
  CK=/bench/train/$PAR/$RUN/$RUN-s0/checkpoints/$ck; O=/bench/train/$PAR/eval/$tag
  t0=$(date +%s)
  if [ "$KIND" = old ]; then
    L=$(GPU=$G CPUSET=$CS MEMS=$MS sh $BM/job.sh ev-$RUN "cd /bench/bm1/trees/m1ae82d3/bridge/python && export HOME=/bench/bm1/home-c99 ABP_HOME=/bench/bm1/home-c99/isaac-abplus ABP_BRIDGE_LUA=/bench/bm1/trees/m1ae82d3/bridge/abplus/abp_bridge.lua OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null PYTHONPATH=.:\$PYTHONPATH && $X \$BM1_PY -u eval_tok.py --checkpoint $CK --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 --seeds $seeds --name hc99e --port 49300 --out $O" | tail -1)
  else
    L=$(GPU=$G CPUSET=$CS MEMS=$MS ENVS="-e TOK_EVAL_DEVICE=cuda" sh $BM/job.sh ev-$RUN "cd /bench/bm1/trees/maa1e98a/bridge/python && export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null && $X \$BM1_PY -u /bench/bm1/evtools/eval_ent.py --checkpoint $CK --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers $W --seeds $seeds --name abev --port 41300 --out $O" | tail -1)
  fi
  cp "$L" $HOST/$tag.log 2>/dev/null
  echo "$(date '+%F %T') $tag $ck $seeds: $(( $(date +%s) - t0 )) s, summary $( [ -f $HOST/$tag/summary.json ] && echo yes || echo NO ) ($L)"
done
echo "EV_AB2 DONE $RUN"
