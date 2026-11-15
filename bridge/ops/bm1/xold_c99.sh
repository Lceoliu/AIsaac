#!/bin/sh
# ab2 (C107, 2026-11-14) arm A: C99's own command line, run inside the container like xnew2.sh (shardk_in.sh BASE=xold_c99.sh).
# The trainer flags are bridge/ops/hpc_c99.sh's "run" branch at 1ae82d3 word for word -- COMMON, FAST, then
# "--rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1" -- with only --workers 16 (C99: 64) and
# --game-hours $HOURS (C99: 25000) in it; --checkpoint-every 1000 stays as C99's and the run's own changes come after it
# (a later flag wins): FLAGS_FILE (the run directory's train_args, e.g. --pickup-force 1:0.952), then EXTRA..EXTRA3 (the
# guard: --checkpoint-every=<CKPT_EVERY>; with LEARNER_DEV empty no --learner-device, which this tree does not have).
# Environment as C99's hpc_env.sh where it matters to the trainer: OMP / MKL / OpenBLAS threads 1,
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; its own ABP_HOME (bm1/$ABHOME/isaac-abplus, default home-c99:
# libabp_turbo.so built from 1ae82d3's abp_turbo.c, abp_row_encode version 7; the shared bm1/home one is of another
# layout and FastRow would silently turn off) and HOME=bm1/$ABHOME (run_instance.sh starts the game from
# $HOME/isaac-abplus); the tree's own bridge Lua (abp-0.2.18). One xvfb-run per trainer (C99: one shared Xvfb).
#   usage (in the container): CODE=<tree, m1ae82d3> [NAME= PORT=] sh /bench/bm1/xold_c99.sh <tag> [limit_s=7200] [game_hours=125]
BM=${BM1_IN:-/bench/bm1}
TAG=${1:-xold-c99}
OUT=${OUTDIR:-$BM/reports}/$TAG
LIMIT=${2:-7200}
HOURS=${3:-125}
PY=${BM1_PY:-/bench/home/isaac-abplus/.venv/bin/python}
mkdir -p $BM/reports
cd $BM/trees/${CODE:?CODE=<tree under bm1/trees>}/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOME=$BM/${ABHOME:-home-c99} ABP_HOME=$BM/${ABHOME:-home-c99}/isaac-abplus ABP_BRIDGE_LUA=$BM/trees/$CODE/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
COMMON="--groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --obs-v2 1 \
  --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 \
  --cf-distill-steps 8 --cf-random 1 --cf-advantage 0"
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
CMD="$PY -u train_tok.py $COMMON $FAST --workers 16 --game-hours $HOURS --checkpoint-every 1000 \
  --rollout 32768 --minibatch 4096 --micro 4096 --learner-fast 1 \
  --name ${NAME:-hc99} --port ${PORT:-49000} --out $OUT $( [ -n "${FLAGS_FILE:-}" ] && cat "$FLAGS_FILE" ) ${EXTRA:-} ${EXTRA2:-} ${EXTRA3:-}"
echo "$CMD" > $OUT.command.txt
exec xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s $LIMIT $CMD > $OUT.stdout 2> $OUT.stderr
