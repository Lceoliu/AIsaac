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
exit $RC
