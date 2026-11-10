#!/bin/sh
# B60 GPU stages, each <= 15 min under its own GPU_LOCK (taken only after the lock has been free, with no cpp-actor
# chain / run script, train_tok, pytest or eval_tok, for 2 minutes in a row):
#   A  the B60 CPU tests on this host (the compiled C library) + the fixed-input learner bench (b48 dump) variants
#      n0p0b0 (Python packing, serial: the 10.95 s reference) n0p0b1 (+1 busy Python thread) n0p1b0 (Python,
#      pipelined) n1p0b0 (C packing)
#      n1p1b0 (C packing, pipelined as the learner) n1p1b1 (+1 busy thread) n1p1b1i100 (switch interval 100 us,
#      diagnosis; the others 500 us as train_tok), then one op-counting rep;
#   B  the full test gate;
#   C  a 6-update short profile (prof2.sh) with ISAAC_LEARN_OPCOUNT=0: the learner's prefetch hits / drops across
#      recompute groups and epochs, and the uc_* (thread CPU) columns; SIGINT to the train_tok main process only.
#   T  stage A's tests only (gates the stages after it); O  the op count by function (bench, C packing, pipelined).
#   usage: CODE=<candidate dir> sh b60_gpu.sh A|T|B|C [more stages]
set -u
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
LOG=$R/b60.log
OUT=$R/b60
mkdir -p $OUT
idle() { [ ! -f $R/GPU_LOCK ] && [ "$(pgrep -fc '[c]hain_b')" = "0" ] && [ "$(pgrep -fc '[r]un_b5')" = "0" ] && \
  [ "$(pgrep -fc '[t]rain_tok.py')" = "0" ] && [ "$(pgrep -fc '[p]ytest')" = "0" ] && [ "$(pgrep -fc '[e]val_tok.py')" = "0" ]; }
guard() { echo "$(date '+%F %T') guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
take() {
  ok=0
  while [ $ok -lt 4 ]; do if idle; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
  echo "a1 $(date '+%F %T') B60 stage $1 ($CODE)" > $R/GPU_LOCK
  echo "$(date '+%F %T') stage $1 start ($CODE)" >> $LOG
  guard
}
give() { sleep 5; guard; grep -q "B60 stage $1" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK; echo "$(date '+%F %T') stage $1 done, lock released" >> $LOG; }
for S in "$@"; do
  take $S
  cd $ROOT/$CODE/bridge/python
  export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
  case $S in
  A)
    timeout 240 $PY -m pytest tests/test_b60_native_pack.py tests/test_b48_prefetch.py -q -p no:cacheprovider \
      > $R/tests-b60-native.log 2>&1
    echo "$(date '+%F %T') A tests rc=$? $(tail -1 $R/tests-b60-native.log)" >> $LOG
    cd $ROOT/$CODE/bridge/ops
    timeout 780 $PY b44_learner_bench.py $R/b48-dump/learner_inputs.pt --mode packed --reps 2 \
      --variants n0p0b0,n0p0b1,n0p1b0,n1p0b0,n1p1b0,n1p1b1,n1p1b1i100 --pipeline --opcount --json $OUT/bench.json > $OUT/bench.log 2>&1
    rc=$?
    echo "$(date '+%F %T') A bench rc=$rc" >> $LOG
    if [ $rc != 0 ] || ! grep -q " passed" $R/tests-b60-native.log || grep -q " failed" $R/tests-b60-native.log; then
      give $S; echo "$(date '+%F %T') stage A failed: later stages not run" >> $LOG; exit 1
    fi
    ;;
  T)   # (stage A's tests only, gating B and C)
    CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 240 $PY -m pytest tests/test_b60_native_pack.py tests/test_b48_prefetch.py       -q -p no:cacheprovider > $R/tests-b60-native.log 2>&1
    rc=$?
    echo "$(date '+%F %T') T tests rc=$rc $(tail -1 $R/tests-b60-native.log)" >> $LOG
    if [ $rc != 0 ]; then give $S; echo "$(date '+%F %T') stage T failed: later stages not run" >> $LOG; exit 1; fi
    ;;
  O)   # (the op count by function, C packing, pipelined: one rep to compile, then the counting rep)
    cd $ROOT/$CODE/bridge/ops
    timeout 780 $PY b44_learner_bench.py $R/b48-dump/learner_inputs.pt --mode packed --reps 1 --variants n1p1b0       --pipeline --opcount --json $OUT/opcount.json > $OUT/opcount.log 2>&1
    echo "$(date '+%F %T') O opcount rc=$?" >> $LOG
    ;;
  B)
    CUBLAS_WORKSPACE_CONFIG=:4096:8 ISAAC_VISIBLE_PAYLOADS=$R/b45-payloads2 timeout 840 $PY -m pytest tests -q \
      -p no:cacheprovider > $R/tests-b60-full.log 2>&1
    echo "$(date '+%F %T') B full gate rc=$? $(tail -1 $R/tests-b60-full.log)" >> $LOG
    ;;
  C)
    TAG=${PROF_TAG:-prof-b60}
    cd $ROOT
    ( CODE=$CODE ./prof2.sh $TAG 840 8.0 ) &
    P=$!
    while kill -0 $P 2>/dev/null; do
      sleep 15
      n=0
      [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
      if [ $n -ge 6 ]; then
        echo "$(date '+%F %T') $n updates: SIGINT to the train_tok main process" >> $LOG
        pkill -INT -f "^$PY -u train_tok.py"
        break
      fi
    done
    wait $P
    echo "$(date '+%F %T') C profile done, $(grep -h 'B42 compiled Spatial\|B56 compile cache' $R/$TAG.stdout | tr '\n' ' ' | cut -c1-400)" >> $LOG
    ;;
  esac
  give $S
done
