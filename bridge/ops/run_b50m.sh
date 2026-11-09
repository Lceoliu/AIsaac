#!/bin/sh
# B50 phase 0: Nsight Systems profile of the B47 code (NVTX ranges, python-gil) over hand-offs 2..7 of the
# prof-f4 short profile (16 workers, rollout 32768, micro 8192, cf 6/180, teacher share 0.1). GPU_LOCK protocol
# and guards as run_b4*.sh. usage: run_b50m.sh <tag> <code dir> [limit s] [game hours] [nsys trace list]
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
NSYS=/usr/local/cuda/bin/nsys
TAG=${1:-prof-b50m}
CODE=${2:-candidate-b50m}
LIMIT=${3:-1500}
HOURS=${4:-10.3}
TRACE=${5:-cuda,nvtx,python-gil}
OUT=$R/$TAG
LOG=$R/b50.log
guard() { echo "$(date +%T) guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, $(pgrep -fc '[a]bplus_probe') probe, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
if [ -f $R/GPU_LOCK ]; then
  if ! grep -q '^cpp-actor' $R/GPU_LOCK; then echo "$(date +%T) $TAG: GPU_LOCK held by: $(cat $R/GPU_LOCK)" >> $LOG; exit 2; fi
fi
echo "cpp-actor $(date '+%Y-%m-%d %H:%M:%S') B50 phase 0 nsys profile $TAG ($CODE, trace $TRACE)" > $R/GPU_LOCK
echo "$(date +%T) $TAG start ($CODE, trace $TRACE, hours $HOURS)" >> $LOG
guard
cd $ROOT/$CODE/bridge/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
export ISAAC_BLOCK_TIMING=1 ISAAC_BLOCK_NVTX=1 ISAAC_NSYS_RANGE=${NSYS_RANGE:-2:7} ISAAC_NSYS_TRIGGER=${TRIGGER:-nvtx}
if [ "${TRIGGER:-nvtx}" = cupti ]; then CAP="--capture-range=cudaProfilerApi"; else CAP="--capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0"; fi
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
 --port 39200 --name b42new --seed 1000 --out $OUT $EXTRA"
PYS=${PYS-"--python-sampling=true --python-sampling-frequency=${PYSAMP:-250}"}   # PYS="" : no Python sampling
NS="$NSYS profile -t $TRACE $PYS \
 --sample=none --cpuctxsw=none --cuda-graph-trace=${GRAPH_TRACE:-graph} \
 $CAP --capture-range-end=stop -o $R/$TAG.nsys-rep --force-overwrite true"
echo "$NS $CMD" > $OUT.command.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 1 > $R/$TAG.gpu &
SMI=$!
xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s $LIMIT $NS $CMD > $OUT.stdout 2> $OUT.stderr
RC=$?
kill $SMI
echo "$(date +%T) $TAG done rc=$RC $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
sleep 5; guard
rm -f $R/GPU_LOCK
echo "$(date +%T) $TAG lock released" >> $LOG
