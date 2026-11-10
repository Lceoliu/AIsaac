#!/bin/sh
# B62 step 0 on the merged tree (exp/block-memory-v1 = B53-B60 + cpp-actor step 0), C++ actor path. Stages, each under
# its own GPU_LOCK (taken atomically after 2 idle minutes, as b60_gpu.sh):
#   G   the full test gate;
#   P   a 16-update profile (prof2.sh, 18.3 game hours as run_b54p.sh's cpp profile), switch interval 500 us;
#   N   Nsight Systems nvtx + python-gil over hand-offs 2..7 (run_b50m.sh's command, no Python sampling), per-thread
#       CPU samples (b57_threads.py) meanwhile; the analysis (b57_nsys.py) after the lock is released;
#   W100, W50  16-update profiles with ISAAC_SWITCH_INTERVAL_US=100 / 50.
#   T   unit tests (gate); O / OC  bench op counts by calling frame (OC: Inductor cpp_wrapper); H  host preparation
#       by function (bench, serial, cProfile); Q  a 5-update profile
#       with the learner's update 4 under cProfile.
#   usage: CODE=<candidate dir> sh b62_step0.sh G P N W100 W50 | T O OC Q
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
  # (yield to cpp-actor first, between stages only: while it asks for the GPU (GPU_WANT_CPP) or its scratch-b58c
  # runner waits, never take the lock)
  y0=$(date +%s); why=''
  while true; do
    if [ -f $R/GPU_WANT_CPP ]; then why='GPU_WANT_CPP'
    elif pgrep -f '[s]cratch-b58c/run_' > /dev/null; then why="scratch-b58c ($(pgrep -fa '[s]cratch-b58c/run_' | head -1 | cut -c1-60))"
    else break; fi
    sleep 15
  done
  [ -n "$why" ] && echo "$(date '+%F %T') stage $1: yielded to cpp-actor, last seen: $why, $(( $(date +%s)-y0 )) s" >> $LOG
  while true; do
    ok=0
    while [ $ok -lt 4 ]; do if idle; then ok=$((ok+1)); else ok=0; fi; [ $ok -lt 4 ] && sleep 30; done
    case $1 in P*|Y*|W*|N*|Q) U_=' 计时运行' ;; *) U_='' ;; esac
    if ( set -C; echo "a1 $(date '+%F %T') B62 step 0 stage $1 ($CODE)$U_ ${PTAG:-}" > $R/GPU_LOCK ) 2>/dev/null; then break; fi
    sleep 30
  done
  echo "$(date '+%F %T') stage $1 start ($CODE)" >> $LOG
  guard
}
give() { sleep 5; guard; grep -q "B62 step 0 stage $1" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK; echo "$(date '+%F %T') stage $1 done, lock released" >> $LOG; }
prof() {   # prof <tag> [env assignments...]: a 16-update profile of the C++ path (PROF_HOURS / PROF_LIMIT: shorter)
  TAG=$1; shift
  cd $ROOT
  ( env "$@" CODE=$CODE ./prof2.sh $TAG ${PROF_LIMIT:-4500} ${PROF_HOURS:-18.3} )
  echo "$(date '+%F %T') $TAG done rc=$? $(grep -m1 'cpp actor driver' $R/$TAG.stdout | cut -c1-100)" >> $LOG
}
for S in "$@"; do
  take $S
  case $S in
  G)
    cd $ROOT/$CODE/bridge/python
    CUBLAS_WORKSPACE_CONFIG=:4096:8 ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 OMP_NUM_THREADS=1 timeout 1500 $PY -m pytest \
      tests -q -p no:cacheprovider > $R/tests-b62-full-$CODE-$(date +%m%d-%H%M).log 2>&1
    rc=$?
    GL=$(ls -t $R/tests-b62-full-$CODE-*.log | head -1)
    echo "$(date '+%F %T') G full gate rc=$rc $(tail -1 $GL) ($GL)" >> $LOG
    if [ $rc != 0 ]; then mkdir -p $R/failed && cp $GL $R/failed/; fi   # (B62: gate logs never overwritten; failures kept)
    ;;
  P) prof prof-b62-cpp ISAAC_SWITCH_INTERVAL_US=500 ;;
  A8)   # (the actor probe's play digest, 8 seeds, not timed)
    cd $ROOT/$CODE/bridge/python
    ( export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
      timeout 1800 xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" $PY abplus_probe_play.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal --seeds range:2147500000:8 --steps 300 --out $R/$CODE-a8-play > $R/$CODE-a8-play.log 2>&1 )
    echo "$(date '+%F %T') A8 rc=$? $(tail -1 $R/$CODE-a8-play.log | cut -c1-200)" >> $LOG
    ;;
  L)   # (lean checkpoint vs the B44-B60 call on CUDA, 20 + 20 pairs, B48 spread rule)
    cd $OPS
    CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 600 $PY b60_prefetch_spread.py $ROOT/$CODE $R/b62-lean-spread.json       --pairs 20 --lean-compare > $R/b62-lean-spread.log 2>&1
    echo "$(date '+%F %T') L lean spread rc=$? $(tail -1 $R/b62-lean-spread.log | cut -c1-300)" >> $LOG
    ;;
  P24|Y24)   # (24 updates, stopped by SIGINT to the train_tok main process; Y24: ISAAC_CUDA_BLOCKING_SYNC=$SYNC)
    TAG=${PTAG:-prof-b62-$S${SYNC:+-$SYNC}}
    # (the sampler attaches once the trainer itself runs: started earlier it matched the xvfb-run wrapper, B62 D1)
    ( until pgrep -f "^$PY -u train_tok.py" > /dev/null; do sleep 1; done
      exec python3 $OPS/b57_threads.py $R/$TAG.threads.csv 1.0 > $R/$TAG.threads.log 2>&1 ) &
    THR=$!
    cd $ROOT
    ( ISAAC_CUDA_BLOCKING_SYNC=${SYNC:-} CODE=$CODE ./prof2.sh $TAG 4500 ${HOURS24:-30.0} ) &
    PP=$!
    while kill -0 $PP 2>/dev/null; do
      sleep 15
      n=0
      [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
      if [ $n -ge ${UPDATES:-24} ]; then
        echo "$(date '+%F %T') $TAG: $n updates, SIGINT to the train_tok main process" >> $LOG
        pkill -INT -f "^$PY -u train_tok.py"
        break
      fi
    done
    wait $PP
    kill $THR 2>/dev/null
    echo "$(date '+%F %T') $TAG done, $(grep -h 'B62 CUDA scheduling' $R/$TAG.stdout | head -1)" >> $LOG
    ;;
  NB)   # (the bench's second rep under nsys cuda + nvtx: B2 measures 1 and 2, bench口径)
    TAG=b62-bench-cuda
    cd $OPS
    ISAAC_BLOCK_NVTX=1 ISAAC_BENCH_NVTX_REP=1 OMP_NUM_THREADS=1 timeout 840 /usr/local/cuda/bin/nsys profile -t cuda,nvtx       --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=B60bench -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0       --capture-range-end=stop -o $R/$TAG.nsys-rep --force-overwrite true $PY b44_learner_bench.py       $R/b48-dump/learner_inputs.pt --mode packed --reps 2 --variants n1p1b0 > $R/$TAG.log 2>&1
    echo "$(date '+%F %T') NB nsys done rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
    ;;
  NC)   # (nsys cuda + nvtx over hand-offs 2..4, the C++ path: the GPU idle on the learner's critical path)
    TAG=${NCTAG:-prof-b62-cuda}
    OUT=$R/$TAG
    cd $ROOT/$CODE/bridge/python
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export ABP_HOME=$ROOT/formal/abplus ABP_BRIDGE_LUA=$ROOT/candidate-a1/bridge/abplus/abp_bridge.lua
    export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null
    export ISAAC_BLOCK_TIMING=1 ISAAC_BLOCK_NVTX=1 ISAAC_NSYS_RANGE=2:4 ISAAC_NSYS_TRIGGER=nvtx
    NS="/usr/local/cuda/bin/nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=graph       --capture-range=nvtx --nvtx-capture=B50window -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop       -o $R/$TAG.nsys-rep --force-overwrite true"
    xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp -noreset" timeout --signal=INT --kill-after=60s 1800 $NS       $PY -u $(cat $R/prof-b62-gil.command.txt | sed 's/.*train_tok.py/train_tok.py/; s/--name b62gil/--name b62cuda/; s#--out [^ ]*#--out '$OUT'#; s/--game-hours 10.3/--game-hours 6.1/')       > $OUT.stdout 2> $OUT.stderr
    echo "$(date '+%F %T') NC nsys done rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
    if [ ! -s $R/$TAG.nsys-rep ] && [ -f $R/$TAG.qdstrm ]; then   # (the importer failed in the run: once more, in the lock)
      /usr/local/cuda/nsight-systems-2023.4.4/host-linux-x64/QdstrmImporter -i $R/$TAG.qdstrm > $R/$TAG.import.log 2>&1
      echo "$(date '+%F %T') NC re-import rc=$? $(ls -la $R/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes" >> $LOG
    fi
    python3 $OPS/b62_gpu_idle.py $R/$TAG.nsys-rep > $R/$TAG.idle.txt 2>&1   # (in the lock: no CPU work after it)
    echo "$(date '+%F %T') NC analysis rc=$? $(tr '
' ' ' < $R/$TAG.idle.txt | cut -c1-500)" >> $LOG
    ;;
  T)   # (the B60 / B62 unit tests and the B48 prefetch test on this host)
    cd $ROOT/$CODE/bridge/python
    CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 timeout 300 $PY -m pytest tests/test_b60_native_pack.py       tests/test_b48_prefetch.py tests/test_block_packed_learner.py -q -p no:cacheprovider > $R/tests-b62-t-$CODE-$(date +%m%d-%H%M).log 2>&1
    rc=$?
    TL=$(ls -t $R/tests-b62-t-$CODE-*.log | head -1)
    echo "$(date '+%F %T') T tests rc=$rc $(tail -1 $TL) ($TL)" >> $LOG
    [ $rc != 0 ] && mkdir -p $R/failed && cp $TL $R/failed/
    if [ $rc != 0 ]; then give $S; echo "$(date '+%F %T') stage T failed: later stages not run" >> $LOG; exit 1; fi
    ;;
  O|OL|OC)   # (op count by calling frame, C packing pipelined; OL: lean checkpoint; OC: Inductor cpp_wrapper)
    cd $OPS
    W=0; [ $S = OC ] && W=1
    LN=0; [ $S = OL ] && LN=1
    ISAAC_LEARN_CKPT_LEAN=$LN ISAAC_INDUCTOR_CPP_WRAPPER=$W OMP_NUM_THREADS=1 timeout 840 $PY b44_learner_bench.py       $R/b48-dump/learner_inputs.pt --mode packed --reps 2 --variants n1p1b0 --pipeline --opcount       --json $R/b62-opcount-$S.json > $R/b62-opcount-$S.log 2>&1
    echo "$(date '+%F %T') $S opcount rc=$?" >> $LOG
    ;;
  H)   # (host preparation by function: C packing, serial preparation, cProfile of the last rep)
    cd $OPS
    OMP_NUM_THREADS=1 timeout 840 $PY b44_learner_bench.py $R/b48-dump/learner_inputs.pt --mode packed --reps 2       --variants n1p0b0 --cprofile --json $R/b62-host.json > $R/b62-host.log 2>&1
    echo "$(date '+%F %T') H host profile rc=$?" >> $LOG
    ;;
  Q)   # (a 5-update C++ path profile with the learner's update 4 under cProfile)
    PROF_HOURS=6.1 PROF_LIMIT=1500 prof prof-b62-cprof ISAAC_RL_B38_TORCHPROF=$R/prof-b62-cprof-learner ISAAC_RL_LEARNER_CPROFILE=1       ISAAC_RL_B38_TORCHPROF_AT=4
    ;;
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
  if [ $S = NB ]; then   # (CPU only, after the lock)
    T_=b62-bench-cuda
    python3 $OPS/b62_gpu_idle.py $R/$T_.nsys-rep > $R/$T_.idle.txt 2>&1
    echo "$(date '+%F %T') $S analysis rc=$? $(tail -3 $R/$T_.idle.txt | tr '
' ' ' | cut -c1-400)" >> $LOG
  fi
  if [ $S = N ]; then   # (CPU only, after the lock)
    python3 $OPS/b57_nsys.py $R/prof-b62-gil.nsys-rep $R/prof-b62-gil.threads.csv > $R/prof-b62-gil.analysis.txt 2>&1
    echo "$(date '+%F %T') N analysis rc=$?" >> $LOG
  fi
done
