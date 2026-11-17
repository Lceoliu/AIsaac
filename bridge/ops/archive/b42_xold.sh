#!/bin/sh
# B42 X_old(4090): c45a583 predictive-aux, fresh random init, 16 workers, single 4090.
ROOT=/home/linux/isaac-block-memory-v1
OUT=$ROOT/reports/b42-speed/xold-c45a583
mkdir -p $ROOT/reports/b42-speed
cd $ROOT/baseline-c45a583/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/baseline-c45a583/bridge/abplus/abp_bridge.lua
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
 --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 --obs-v2 1 \
 --actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32 \
 --workers 16 --rollout 32768 --minibatch 8192 --micro 8192 --game-hours 125 --checkpoint-every 1000 \
 --overlap --decoupled 1 --learner-fast 2 --actor-processes 1 --learner-process 0 --amp bf16 \
 --port 39100 --name b42old --seed 1000 --out $OUT"
echo "$CMD" > $OUT.command.txt
exec xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s 7200 $CMD > $OUT.stdout 2> $OUT.stderr
