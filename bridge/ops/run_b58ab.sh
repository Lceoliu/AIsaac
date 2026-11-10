#!/bin/sh
# B58a+b chain (after the step 0 chain, chain_b58s0.sh): native-consumer tests -> smoke (3 updates, the native consumer
# on) -> nsys nvtx + python-gil over hand-offs 2..5 of a 6-update profile with the per-thread CPU sampler (B57 tools)
# -> full gate (entry + all tests + A8). Each stage takes the lock itself; stops at the first failed gate.
# usage: run_b58ab.sh <code dir>
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58ab}
C=$ROOT/$CODE
OPS=$C/bridge/ops
LOG=$R/b58.log
waitlock() { while [ -f $R/GPU_LOCK ] || [ "$(pgrep -fc '[t]rain_tok.py')" != "0" ] || [ "$(pgrep -fc '[p]ytest')" != "0" ] || [ "$(pgrep -fc '[e]val_tok.py')" != "0" ]; do sleep 20; done; sleep 10; }
stage() {
  n=$1; shift
  while :; do waitlock; "$@" > $R/b58ab-$n.launcher.log 2>&1; rc=$?; [ $rc -eq 2 ] || break; sleep 30; done
  echo "$(date +%T) ab stage $n rc=$rc" >> $LOG
}
echo "$(date +%T) ab queued: waiting for the step 0 chain" >> $LOG
while pgrep -f '[c]hain_b58s0.sh' > /dev/null; do sleep 30; done
echo "$(date +%T) ab chain start ($CODE)" >> $LOG
stage tests $OPS/run_b58t.sh $CODE
if ! tail -1 $R/tests-b58t.log | grep -q " passed" || tail -1 $R/tests-b58t.log | grep -qE "failed|error"; then
  echo "$(date +%T) ab stops: tests failed ($(tail -1 $R/tests-b58t.log | cut -c1-120))" >> $LOG; exit 1
fi
stage smoke env SMOKETAG=prof-b58ab-smoke $OPS/run_b54p.sh $CODE smoke
ROWS=$(($(wc -l < $R/prof-b58ab-smoke/progress.csv 2>/dev/null || echo 0)-1))
TB=$(grep -cE 'File "[^"]*train_tok\.py"' $R/prof-b58ab-smoke.stdout $R/prof-b58ab-smoke.stderr 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
NC=$(grep -m1 'B58 native consumer' $R/prof-b58ab-smoke.stdout | cut -c1-80)
if [ "$ROWS" -lt 2 ] || [ "$TB" != "0" ] || ! echo "$NC" | grep -q ': on'; then
  echo "$(date +%T) ab stops: smoke failed (updates $ROWS, train_tok tracebacks $TB, $NC)" >> $LOG; exit 1
fi
echo "$(date +%T) ab smoke ok: $ROWS updates, $NC" >> $LOG
# nsys + per-thread CPU (run_b50m.sh takes the lock itself)
waitlock
python3 $OPS/b57_threads.py $R/prof-b58ab-m.threads.csv 1.0 > $R/prof-b58ab-m.threads.log 2>&1 &
PYS='' TRIGGER=nvtx NSYS_RANGE=2:5 $OPS/run_b50m.sh prof-b58ab-m $CODE 1800 7.4 nvtx,python-gil
echo "$(date +%T) ab nsys rc=$?" >> $LOG
wait
python3 $OPS/b57_nsys.py $R/prof-b58ab-m.nsys-rep $R/prof-b58ab-m.threads.csv > $R/prof-b58ab-m.analysis.txt 2>&1
echo "$(date +%T) ab analysis rc=$?" >> $LOG
stage gate $OPS/run_b54g.sh $CODE
if [ -f $R/b54-gate.failed ]; then echo "$(date +%T) ab stops: full gate failed ($(tr '\n' ' ' < $R/b54-gate.failed))" >> $LOG; exit 1; fi
echo "$(date +%T) ab gate: $(tail -1 $R/tests-b54-entry.log | cut -c1-50) | $(tail -1 $R/tests-b54-full.log | cut -c1-70) | a8 $(tail -1 $R/b54-a8-play.log | grep -o '"seeds": 8, "ok": [0-9]*')" >> $LOG
echo "$(date +%T) ab chain done" >> $LOG
