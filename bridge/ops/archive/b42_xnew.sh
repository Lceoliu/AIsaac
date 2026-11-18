#!/bin/sh
# B42 X_new(4090): block-memory v1 (A1 slot/paged path), fresh random init, same budget as xold.sh.
# Identical: mode/items/epochs/teacher(+queue/archive/slots/death depths/share)/start builds/pickup/reward/item slots,
# injury+CF batch/reuse/weight/buffer/coef, CF points/branches/seconds/target/margin/cap/distill/random,
# aux-predict+horizons+coef, priv-critic, workers/rollout/minibatch/micro, overlap/decoupled, AMP, learner process 0.
# Architecture-only differences: block memory (visible-v2 input, Spatial->4 latents->Reader/Writer) instead of
# temporal/obs-v2, and each side's own actor/learner implementation knobs.
ROOT=/home/linux/isaac-block-memory-v1
TAG=${1:-xnew-a1}
OUT=$ROOT/reports/b42-speed/$TAG
LIMIT=${2:-7200}
HOURS=${3:-125}
mkdir -p $ROOT/reports/b42-speed
cd $ROOT/candidate-a1/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
CMD="$ROOT/.venv/bin/python -u train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items \
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
 --port 39200 --name b42new --seed 1000 --out $OUT $EXTRA"
echo "$CMD" > $OUT.command.txt
exec xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s $LIMIT $CMD > $OUT.stdout 2> $OUT.stderr
