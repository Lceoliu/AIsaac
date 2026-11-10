#!/bin/sh
# B58c: the torch-free core (engine/abp_core.h) under ThreadSanitizer, random interleavings (CPU only, no GPU lock).
# usage: run_core_tsan.sh <code dir> [iterations] -> reports/b42-speed/core-tsan.log; exit 0 when every seed passes
ROOT=/home/linux/isaac-block-memory-v1
R=$ROOT/reports/b42-speed
CODE=${1:-candidate-b58c}
N=${2:-20000}
cd $ROOT/$CODE/bridge/engine/core_tests || exit 1
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_core.cpp -o test_abp_core || exit 1
RC=0
: > $R/core-tsan.log
for s in 1 2 3 4 5; do
  setarch $(uname -m) -R ./test_abp_core $N $s > out$s.txt 2>&1; rc=$?
  echo "seed $s rc=$rc tsan warnings $(grep -c 'WARNING: ThreadSanitizer' out$s.txt): $(grep -E 'part' out$s.txt | tr '\n' ' ')" >> $R/core-tsan.log
  [ $rc -eq 0 ] || RC=1
done
setarch $(uname -m) -R ./test_abp_core $N 9 nostamp > outns.txt 2>&1
echo "control without the stamp rule (must fail): rc=$? tsan warnings $(grep -c 'WARNING: ThreadSanitizer' outns.txt), $(grep -E 'part 1' outns.txt)" >> $R/core-tsan.log
# (B58c) the native lineage: logs / snapshots / sidecar / contexts / injury sources with a concurrent GC
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lineage.cpp -o test_abp_lineage || exit 1
for s in 1 2 3 4 5; do
  setarch $(uname -m) -R ./test_abp_lineage $N $s > lout$s.txt 2>&1; rc=$?
  echo "lineage seed $s rc=$rc tsan warnings $(grep -c 'WARNING: ThreadSanitizer' lout$s.txt): $(grep -E '^lineage' lout$s.txt)" >> $R/core-tsan.log
  [ $rc -eq 0 ] || RC=1
done
setarch $(uname -m) -R ./test_abp_lineage $N 9 nostamp > loutns.txt 2>&1
echo "lineage control without the stamp rule (must fail): rc=$? tsan warnings $(grep -c 'WARNING: ThreadSanitizer' loutns.txt), $(grep -E '^lineage' loutns.txt)" >> $R/core-tsan.log
# (B58d) the lane service's state machine with workers, the driver's captures and a reader
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lanes.cpp -o test_abp_lanes || exit 1
for s in 1 2 3; do
  setarch $(uname -m) -R ./test_abp_lanes 300 $s > nout$s.txt 2>&1; rc=$?
  echo "lanes seed $s rc=$rc tsan warnings $(grep -c 'WARNING: ThreadSanitizer' nout$s.txt): $(grep -E '^lanes' nout$s.txt)" >> $R/core-tsan.log
  [ $rc -eq 0 ] || RC=1
done
setarch $(uname -m) -R ./test_abp_lanes 300 9 replyfirst > noutrf.txt 2>&1
echo "lanes control, the reply before the answer (must fail): rc=$? tsan warnings $(grep -c 'WARNING: ThreadSanitizer' noutrf.txt), $(grep -E '^lanes' noutrf.txt)" >> $R/core-tsan.log
exit $RC
