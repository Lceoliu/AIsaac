#!/bin/sh
# bm1 port of the 4090's xnew2.sh (B42 X_new, block-memory v1, A1 slot/paged path), run inside the container
# (dock.sh). The train_tok configuration is the 4090's, flag for flag: workers 16, rollout 32768, minibatch 8192,
# micro 8192, teacher / CF / aux / priv-critic budgets, overlap, decoupled, AMP bf16, one learner in-process.
# Only paths differ: the tree under /bench/bm1/trees/$CODE, the shared venv, ABP_HOME=/bench/bm1/home/isaac-abplus (run_instance.sh uses $HOME/isaac-abplus), the
# tree's own bridge Lua (identical to the 4090's candidate-a1 copy, md5 6cdc633b of the LF text).
#   usage (in the container): CODE=<tree> [NAME=b42new PORT=39200] sh /bench/bm1/xnew2.sh <tag> [limit_s=7200] [game_hours=125]
#   (NAME / PORT: the instances' name and port base, distinct per trainer when several run on one host)
#   NSYS_PREFIX: a profiler command put before the trainer (nsys_update.sh); empty for every timed run
#   EXTRA, EXTRA2, EXTRA3: flags added at the end (one each: docker -e values carry no spaces; a later flag wins)
#   FLAGS_FILE: a file of more flags (any number, whitespace-separated: train_guard.sh's <run dir>/train_args), put after
#   the configuration's own flags (so they win) and before EXTRA..EXTRA3 (which win over them)
BM=${BM1_IN:-/bench/bm1}   # (bm1 as the runner sees it: run.sh passes BM1_IN; the 8x3090 container default)
TAG=${1:-xnew-a1}
OUT=${OUTDIR:-$BM/reports}/$TAG   # (OUTDIR: a training run's own directory, e.g. /bench/train/<name>)
LIMIT=${2:-7200}
HOURS=${3:-125}
PY=${BM1_PY:-/bench/home/isaac-abplus/.venv/bin/python}
mkdir -p $BM/reports
cd $BM/trees/${CODE:?CODE=<tree under bm1/trees>}/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$BM/home/isaac-abplus ABP_BRIDGE_LUA=$BM/trees/$CODE/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
CMD="$PY -u train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items \
 --epochs 3 --teacher --teacher-queue 12 --archive-size 12 --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 \
 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.001 \
 --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
 --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
 --teach-batch 512 --teach-reuse 8 --teach-weight 8 --teach-buffer 60000 --teach-coef 1 --danger-coef 1 --teach-merge 1 \
 --teacher-v2-batch 256 --teacher-v2-reuse 4 --teacher-v2-weight 1 --teacher-v2-coef 1 --teacher-v2-vf 0.25 --teacher-v2-buffer 30000 \
 --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 --cf-distill-steps 8 --cf-random 1 --cf-advantage 0 \
 --aux-predict 1 --aux-horizons 1,8 --aux-coef 0.1 --priv-critic 1 \
 --memory off --temporal 0 --block-memory 1 --block-size v1 --block-slots 1 --width 192 --layers 4 --heads 4 \
 --actor-bf16 1 --graph-entities 128 \
 --workers 16 --rollout 32768 --minibatch 8192 --micro 8192 --game-hours $HOURS --checkpoint-every 1000 \
 --overlap --decoupled 1 --learner-fast 0 --actor-processes 1 --learner-process 0 --amp bf16 \
 --port ${PORT:-39200} --name ${NAME:-b42new} --seed 1000 --out $OUT  $( [ -n "${FLAGS_FILE:-}" ] && cat "$FLAGS_FILE" ) ${EXTRA:-} ${EXTRA2:-} ${EXTRA3:-}"
echo "$CMD" > $OUT.command.txt
exec xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s $LIMIT ${NSYS_PREFIX:-} $CMD > $OUT.stdout 2> $OUT.stderr
