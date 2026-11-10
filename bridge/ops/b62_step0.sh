#!/bin/sh
# B62 step 0 on the merged tree (exp/block-memory-v1 = B53-B60 + cpp-actor step 0), C++ actor path. Stages, each under
# its own GPU_LOCK (taken atomically after 2 idle minutes, as b60_gpu.sh):
#   G   the full test gate;
#   P   a 16-update profile (prof2.sh, 18.3 game hours as run_b54p.sh's cpp profile), switch interval 500 us;
#   N   Nsight Systems nvtx + python-gil over hand-offs 2..7 (run_b50m.sh's command, no Python sampling), per-thread
#       CPU samples (b57_threads.py) meanwhile; the analysis (b57_nsys.py) after the lock is released;
#   W100, W50  16-update profiles with ISAAC_SWITCH_INTERVAL_US=100 / 50.
#   usage: CODE=<candidate dir> sh b62_step0.sh G P N W100 W50
set -u
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
NSYS=/usr/local/cuda/bin/nsys
LOG=$R/b62.log
OPS=$ROOT/$CODE/bridge/ops
idle() { [ ! -f $R/GPU_LOCK ] && [ "$(pgrep -fc '[c]hain_b')" = "0" ] && [ "$(pgrep -fc '[r]un_b5')" = "0" ] && \
  [ "$(pgrep -fc '[t]rain_tok.py')" = "0" ] && [ "$(pgrep -fc '[p]ytest')" = "0" ] && [ "$(pgrep -fc '[e]val_tok.py')" = "0" ]; }
guard() { echo "$(date '+%F %T') guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
take() {
  while true; do
    ok=0
    while [ $ok -lt 4 ]; do if idle; then ok=$((ok+1)); else ok=0; fi; [ $ok -lt 4 ] && sleep 30; done
    if ( set -C; echo "a1 $(date '+%F %T') B62 step 0 stage $1 ($CODE)" > $R/GPU_LOCK ) 2>/dev/null; then break; fi
    sleep 30
  done
  echo "$(date '+%F %T') stage $1 start ($CODE)" >> $LOG
  guard
}
give() { sleep 5; guard; grep -q "B62 step 0 stage $1" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK; echo "$(date '+%F %T') stage $1 done, lock released" >> $LOG; }
prof() {   # prof <tag> [env assignments...]: a 16-update profile of the C++ path
  TAG=$1; shift
  cd $ROOT
  ( env "$@" CODE=$CODE ./prof2.sh $TAG 4500 18.3 )
  echo "$(date '+%F %T') $TAG done rc=$? $(grep -m1 'cpp actor driver' $R/$TAG.stdout | cut -c1-100)" >> $LOG
}
for S in "$@"; do
  take $S
  case $S in
  G)
    cd $ROOT/$CODE/bridge/python
    CUBLAS_WORKSPACE_CONFIG=:4096:8 ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 OMP_NUM_THREADS=1 timeout 1500 $PY -m pytest \
      tests -q -p no:cacheprovider > $R/tests-b62-full.log 2>&1
    echo "$(date '+%F %T') G full gate rc=$? $(tail -1 $R/tests-b62-full.log)" >> $LOG
    ;;
  P) prof prof-b62-cpp ISAAC_SWITCH_INTERVAL_US=500 ;;
  W100) prof prof-b62-sw100 ISAAC_SWITCH_INTERVAL_US=100 ;;
  W50) prof prof-b62-sw50 ISAAC_SWITCH_INTERVAL_US=50 ;;
  N)
    TAG=prof-b62-gil
    OUT=$R/$TAG
    python3 $OPS/b57_threads.py $R/$TAG.threads.csv 1.0 > $R/$TAG.threads.log 2>&1 &
    THR=$!
    cd $ROOT/$CODE/bridge/python
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
    export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
    export ISAAC_BLOCK_TIMING=1 ISAAC_BLOCK_NVTX=1 ISAAC_NSYS_RANGE=2:7 ISAAC_NSYS_TRIGGER=nvtx
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
     --workers 16 --rollout 32768 --minibatch 8192 --micro 8192 --game-hours 10.3 --checkpoint-every 1000 \
     --overlap --decoupled 1 --learner-fast 0 --actor-processes 1 --learner-process 0 --amp bf16 \
     --port 39200 --name b62gil --seed 1000 --out $OUT"
    NS="$NSYS profile -t nvtx,python-gil --sample=none --cpuctxsw=none --cuda-graph-trace=graph \
      --capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop \
      -o $R/$TAG.nsys-rep --force-overwrite true"
    echo "$NS $CMD" > $OUT.command.txt
    xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s 2400 $NS $CMD \
      > $OUT.stdout 2> $OUT.stderr
    echo "$(date '+%F %T') N nsys done rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
    kill $THR 2>/dev/null
    ;;
  esac
  give $S
  if [ $S = N ]; then   # (CPU only, after the lock)
    python3 $OPS/b57_nsys.py $R/prof-b62-gil.nsys-rep $R/prof-b62-gil.threads.csv > $R/prof-b62-gil.analysis.txt 2>&1
    echo "$(date '+%F %T') N analysis rc=$?" >> $LOG
  fi
done
