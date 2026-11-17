#!/bin/sh
# C108 D2 (2026-11-15, the coordinator): the records D2 reads, on GPUs 4(+5) and node0's CPUs, one container job each
# (job.sh: shared lock), sequential, into /root/AIsaac/train/c108/d2/<tag> (a tag whose directory exists is skipped):
#   prod-bm / prod-c99old: the production path -- the committed trees maa1e98a / m1ae82d3 (no D1 hooks), ab2's own
#     train_args plus --lr 0 --lr-final 0 --dump-rollouts 2, resumed from the copies in c108/d1/inputs; two updates
#     (block: --max-updates 2; the old tree: --game-hours 2.5, its --resume budget being incremental); rollouts/ holds
#     the two rollouts handed to the learner (rows, pending, actions, logp, ...).
#   ev-bm / ev-c99old: the eval path -- each tree's own eval_tok.py --record, seeds 2147490000:12, sampled, as the C107
#     evaluations (bm: TOK_EVAL_DEVICE=cuda, 12 workers; c99old: C99's watch flags, the CPU, 8 workers).
#   analysis: d2_actions.py table (all four) + draw (both eval records, 12 episodes), and which video tools the image has.
set -u
BM=/root/AIsaac/bm1
D=/root/AIsaac/train/c108/d2
mkdir -p $D
echo "$(date '+%F %T') d2_run: $*" >> $BM/reports/bm1.log
X="xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset'"
EVC="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --seeds 2147490000:12"
for tag in "$@"; do
  [ -e $D/$tag ] && { echo "skip $tag (exists)"; continue; }
  t0=$(date +%s)
  case $tag in
    prod-bm)
      { cat /root/AIsaac/train/ab2/ab2-bm/train_args; echo "--lr 0 --lr-final 0 --dump-rollouts 2"; } > $D/$tag.args
      L=$(GPU=4,5 MEM=80g sh $BM/job.sh d2-$tag "cd /bench && export CODE=maa1e98a OUTDIR=/bench/train/c108/d2 NAME=d2n PORT=41500 FLAGS_FILE=/bench/train/c108/d2/$tag.args EXTRA=--learner-device=cuda:1 EXTRA2=--max-updates=2 EXTRA3=--resume=/bench/train/c108/d1/inputs/bm-last.pt && sh /bench/bm1/xnew2.sh $tag 3600 1200" | tail -1) ;;
    prod-c99old)
      { cat /root/AIsaac/train/ab2/ab2-c99old/train_args; echo "--lr 0 --lr-final 0 --dump-rollouts 2"; } > $D/$tag.args
      L=$(GPU=4 MEM=80g sh $BM/job.sh d2-$tag "cd /bench && export CODE=m1ae82d3 OUTDIR=/bench/train/c108/d2 NAME=d2o PORT=41700 FLAGS_FILE=/bench/train/c108/d2/$tag.args EXTRA3=--resume=/bench/train/c108/d1/inputs/c99old-last.pt && sh /bench/bm1/xold_c99.sh $tag 3600 2.5" | tail -1) ;;
    ev-bm)
      L=$(GPU=4 MEM=64g ENVS="-e TOK_EVAL_DEVICE=cuda" sh $BM/job.sh d2-$tag "cd /bench/bm1/trees/maa1e98a/bridge/python && export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null && $X \$BM1_PY -u eval_tok.py --checkpoint /bench/train/c108/d1/inputs/bm-last.pt $EVC --workers 12 --name d2ev --port 41900 --record /bench/train/c108/d2/$tag/record --out /bench/train/c108/d2/$tag" | tail -1) ;;
    ev-c99old)
      L=$(GPU=4 MEM=64g sh $BM/job.sh d2-$tag "cd /bench/bm1/trees/m1ae82d3/bridge/python && export HOME=/bench/bm1/home-c99 ABP_HOME=/bench/bm1/home-c99/isaac-abplus ABP_BRIDGE_LUA=/bench/bm1/trees/m1ae82d3/bridge/abplus/abp_bridge.lua OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null PYTHONPATH=.:\$PYTHONPATH && $X \$BM1_PY -u eval_tok.py --checkpoint /bench/train/c108/d1/inputs/c99old-last.pt $EVC --workers 8 --name d2eo --port 42100 --record /bench/train/c108/d2/$tag/record --out /bench/train/c108/d2/$tag" | tail -1) ;;
    analysis)
      mkdir -p $D/analysis
      L=$(GPU=4 MEM=32g sh $BM/job.sh d2-$tag "cd /bench/train/c108/d2/analysis && { which ffmpeg xwd import gst-launch-1.0 2>&1; \$BM1_PY -c 'import imageio' 2>&1 | tail -1; } > tools.txt; \$BM1_PY -u /bench/bm1/c108/d2_actions.py table actions.json bm-prod=/bench/train/c108/d2/prod-bm/rollouts c99old-prod=/bench/train/c108/d2/prod-c99old/rollouts bm-eval=/bench/train/c108/d2/ev-bm/record c99old-eval=/bench/train/c108/d2/ev-c99old/record && \$BM1_PY -u /bench/bm1/c108/d2_actions.py draw pictures ab2-bm-last /bench/train/c108/d2/ev-bm/record 12 && \$BM1_PY -u /bench/bm1/c108/d2_actions.py draw pictures ab2-c99old-last /bench/train/c108/d2/ev-c99old/record 12" | tail -1) ;;
    *) echo "unknown tag $tag"; continue ;;
  esac
  cp "$L" $D/$tag.job.log 2>/dev/null
  echo "$(date '+%F %T') $tag: $(( $(date +%s) - t0 )) s ($L)"
done
echo "D2_RUN DONE"
