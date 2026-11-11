#!/bin/sh
# bm1, inside the container (job.sh): the torch-free cores under ThreadSanitizer, as the 4090's run_core_tsan.sh:
# abp_core (5 seeds + the no-stamp control), abp_lineage (5 seeds + its control), abp_lanes (3 seeds + the
# reply-first control). Controls must fail; the log names every run's rc and TSAN warning count.
#   usage (in the container): sh tsan_core.sh <tree dir> <log> [iterations=20000]
T=$1; LOGF=$2; N=${3:-20000}
cd $T/bridge/engine/core_tests || exit 1
RC=0
: > $LOGF
run() {   # run <name> <out> <args...>: the TSAN binary with ASLR off (TSAN's shadow map)
  n=$1; o=$2; shift 2
  if [ "${NO_SETARCH:-0}" = 1 ]; then ./$n "$@" > $o 2>&1; else setarch $(uname -m) -R ./$n "$@" > $o 2>&1; fi; r=$?
  echo "$n $* rc=$r tsan warnings $(grep -c 'WARNING: ThreadSanitizer' $o): $(grep -E '^(part|lineage|lanes)' $o | tr '\n' ' ' | cut -c1-200)" >> $LOGF
  return $r
}
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_core.cpp -o test_abp_core || { echo "build core failed" >> $LOGF; exit 1; }
for s in 1 2 3 4 5; do run test_abp_core out$s.txt $N $s || RC=1; done
run test_abp_core outns.txt $N 9 nostamp && echo "CONTROL core nostamp did not fail" >> $LOGF
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lineage.cpp -o test_abp_lineage || { echo "build lineage failed" >> $LOGF; exit 1; }
for s in 1 2 3 4 5; do run test_abp_lineage lout$s.txt $N $s || RC=1; done
run test_abp_lineage loutns.txt $N 9 nostamp && echo "CONTROL lineage nostamp did not fail" >> $LOGF
g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lanes.cpp -o test_abp_lanes || { echo "build lanes failed" >> $LOGF; exit 1; }
for s in 1 2 3; do run test_abp_lanes nout$s.txt 300 $s || RC=1; done
run test_abp_lanes noutrf.txt 300 9 replyfirst && echo "CONTROL lanes replyfirst did not fail" >> $LOGF
echo "tsan_core rc=$RC" >> $LOGF
exit $RC
