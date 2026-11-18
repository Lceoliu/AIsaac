#!/bin/sh
# B61a: 200x confirmation before training, no code change. Fixed code (a91d189's tree), the C++ path, everything as
# prof-b62-cpp; 24 updates per run (stopped by SIGINT to the train_tok main process once progress.csv has 24 rows),
# switch interval 500 / 50 us alternating: 500 50 500 50 500 50. Each run under its own GPU_LOCK (atomic, after 2 idle
# minutes; cpp-actor first).   usage: CODE=<candidate dir> sh b61a.sh 500a 50a 500b 50b 500c 50c
set -u
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
PY=$ROOT/.venv/bin/python
LOG=$R/b61a.log
idle() { [ ! -f $R/GPU_LOCK ] && [ "$(pgrep -fc '[c]hain_b')" = "0" ] && [ "$(pgrep -fc '[r]un_b5')" = "0" ] && \
  [ "$(pgrep -fc '[t]rain_tok.py')" = "0" ] && [ "$(pgrep -fc '[p]ytest')" = "0" ] && [ "$(pgrep -fc '[e]val_tok.py')" = "0" ]; }
guard() { echo "$(date '+%F %T') guard: $(pgrep -fc '[t]rain_tok.py') train_tok, $(pgrep -c isaac) isaac, $(pgrep -fc '[p]ytest') pytest, gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $LOG; }
take() {
  while true; do
    ok=0
    while [ $ok -lt 4 ]; do if idle; then ok=$((ok+1)); else ok=0; fi; [ $ok -lt 4 ] && sleep 30; done
    if ( set -C; echo "a1 $(date '+%F %T') B61a run $1 ($CODE)" > $R/GPU_LOCK ) 2>/dev/null; then break; fi
    sleep 30
  done
  echo "$(date '+%F %T') run $1 start ($CODE)" >> $LOG
  guard
}
give() { sleep 5; guard; grep -q "B61a run $1" $R/GPU_LOCK 2>/dev/null && rm -f $R/GPU_LOCK; echo "$(date '+%F %T') run $1 done, lock released" >> $LOG; }
for S in "$@"; do
  US=${S%[abc]}
  TAG=prof-b61a-$S
  take $S
  cd $ROOT
  ( ISAAC_SWITCH_INTERVAL_US=$US CODE=$CODE ./prof2.sh $TAG 4500 30.0 ) &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 15
    n=0
    [ -f $R/$TAG/progress.csv ] && n=$(($(wc -l < $R/$TAG/progress.csv)-1))
    if [ $n -ge 24 ]; then
      echo "$(date '+%F %T') $TAG: $n updates, SIGINT to the train_tok main process" >> $LOG
      pkill -INT -f "^$PY -u train_tok.py"
      break
    fi
  done
  wait $P
  echo "$(date '+%F %T') $TAG done ($US us), $(grep -m1 'cpp actor driver' $R/$TAG.stdout | cut -c1-80)" >> $LOG
  give $S
done
