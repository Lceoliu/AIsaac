#!/bin/sh
# gpu57 C112 rw4 checks, second round (2026-11-16, the coordinator: ISAAC_BLOCK_EGO=head added), CPU only (jobcpu.sh;
# waits for any other CPU-only container first):
#   1. the four block CPU test files in maa1e98a-rw4 with the switches unset, EGO=res, =query, =head and
#      ENTITY_SCALE_FIX=1; passing = no FAILED and no ERROR line but the known teardown error of
#      test_packed_learner_spatial_matches_padded[cpu] (it occurs on unmodified maa1e98a too);
#   2. default-off identity again (run_b2.py block stack, 4 payload streams, random + l2blk, CPU; maa1e98a-b2 vs rw4);
#   3. EGO=head forward on the same streams (random init): runs; arrays compared (differ).
# Output c112/rw4_tests2/; last line of RESULT: "RW4 TESTS OK" or "RW4 TESTS FAILED: ...".
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; O=/bench/train/c112/rw4_tests2
mkdir -p $P/rw4_tests2; rm -f $P/rw4_tests2/RESULT
while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
cd $BM
CPUSET=0-7,16-23 MEMS=0 MEM=24g sh $BM/jobcpu.sh c112-rw4-tests2 "
R=/bench/bm1/trees/maa1e98a-rw4/bridge/python; B=/bench/bm1/trees/maa1e98a-b2/bridge/python; fail=''
KNOWN='ERROR tests/test_block_slot_integration.py::test_packed_learner_spatial_matches_padded[cpu]'
cd \$R
for e in off res query head fix; do
  case \$e in off) env=;; res) env=ISAAC_BLOCK_EGO=res;; query) env=ISAAC_BLOCK_EGO=query;; head) env=ISAAC_BLOCK_EGO=head;; fix) env=ISAAC_ENTITY_SCALE_FIX=1;; esac
  env \$env nice -n 10 \$BM1_PY -m pytest -p no:cacheprovider -q tests/test_block_slot_integration.py tests/test_block_packed_learner.py tests/test_b52_entry.py tests/test_block_adv_alignment.py > $O/pytest_\$e.txt 2>&1; rc=\$?
  other=\$(grep -E '^(FAILED|ERROR) ' $O/pytest_\$e.txt | grep -vxF \"\$KNOWN\" | wc -l)
  echo \"pytest \$e: \$(tail -1 $O/pytest_\$e.txt) rc \$rc; failures other than the known teardown error: \$other\" >> $O/RESULT
  { [ \$rc = 0 ] || [ \$rc = 1 ]; } && [ \$other = 0 ] || fail=\"\$fail pytest-\$e\"
done
A='--block-input payloads --payloads c112_b2/payloads --payload-files 4 --stacks block --device cpu --keep-caps 1 --cheap 1'
cd \$B && \$BM1_PY c112_b2/run_b2.py --out $O/ref --weights random,l2blk \$A > $O/run_ref.txt 2>&1 || fail=\"\$fail run-ref\"
cd \$R && \$BM1_PY c112_b2/run_b2.py --out $O/off --weights random,l2blk \$A > $O/run_off.txt 2>&1 || fail=\"\$fail run-off\"
\$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/off exact > $O/cmp_off.txt 2>&1 || fail=\"\$fail identity\"
echo \"identity (switches unset vs maa1e98a): \$(tail -1 $O/cmp_off.txt)\" >> $O/RESULT
env ISAAC_BLOCK_EGO=head \$BM1_PY c112_b2/run_b2.py --out $O/head --weights random \$A > $O/run_head.txt 2>&1 || fail=\"\$fail run-head\"
\$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/head differ > $O/cmp_head.txt 2>&1
echo \"head forward: \$(grep -E ' (p_tok|z|o|zr) ' $O/cmp_head.txt | sed 's/.*npz //' | tr '\n' ';')\" >> $O/RESULT
if [ -z \"\$fail\" ]; then echo 'RW4 TESTS OK' >> $O/RESULT; else echo \"RW4 TESTS FAILED:\$fail\" >> $O/RESULT; fi
"
tail -1 $P/rw4_tests2/RESULT 2>/dev/null
