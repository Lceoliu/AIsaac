#!/bin/sh
# gpu57 C112 rw4 checks before the EGO arms (2026-11-16, the coordinator), one container job on lane a (GPU 4, CPU work):
#   1. the block CPU tests (test_block_slot_integration, test_block_packed_learner, test_b52_entry,
#      test_block_adv_alignment) in tree maa1e98a-rw4 with the switches unset, ISAAC_BLOCK_EGO=res, =query and
#      ISAAC_ENTITY_SCALE_FIX=1;
#   2. default-off identity: c112_b2/run_b2.py (block stack, 4 payload streams, weights random + l2blk, CPU) in
#      maa1e98a-b2 (= maa1e98a) and in maa1e98a-rw4 with the switches unset: every captured array bitwise equal;
#   3. EGO=res / =query / ENTITY_SCALE_FIX=1 forward on the same streams (random init): runs; arrays compared (differ).
# Output c112/rw4_tests/; the last line of c112/rw4_tests/RESULT: "RW4 TESTS OK" or "RW4 TESTS FAILED: ...".
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; O=/bench/train/c112/rw4_tests
mkdir -p $P/rw4_tests; rm -f $P/rw4_tests/RESULT
cd $BM
GPU=4 CPUSET=0-7,16-23 MEMS=0 MEM=64g ENVS="-e CUDA_VISIBLE_DEVICES=" sh $BM/job.sh c112-rw4-tests "
R=/bench/bm1/trees/maa1e98a-rw4/bridge/python; B=/bench/bm1/trees/maa1e98a-b2/bridge/python; fail=''
cd \$R
for e in off res query fix; do
  case \$e in off) env=;; res) env=ISAAC_BLOCK_EGO=res;; query) env=ISAAC_BLOCK_EGO=query;; fix) env=ISAAC_ENTITY_SCALE_FIX=1;; esac
  env \$env nice -n 10 \$BM1_PY -m pytest -p no:cacheprovider -q tests/test_block_slot_integration.py tests/test_block_packed_learner.py tests/test_b52_entry.py tests/test_block_adv_alignment.py > $O/pytest_\$e.txt 2>&1; rc=\$?
  echo \"pytest \$e: \$(tail -1 $O/pytest_\$e.txt) rc \$rc\" >> $O/RESULT; [ \$rc = 0 ] || fail=\"\$fail pytest-\$e\"
done
A='--block-input payloads --payload-files 4 --stacks block --device cpu --keep-caps 1'
cd \$B && \$BM1_PY c112_b2/run_b2.py --out $O/ref --weights random,l2blk \$A > $O/run_ref.txt 2>&1 || fail=\"\$fail run-ref\"
cd \$R && \$BM1_PY c112_b2/run_b2.py --out $O/off --weights random,l2blk \$A > $O/run_off.txt 2>&1 || fail=\"\$fail run-off\"
\$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/off exact > $O/cmp_off.txt 2>&1 || fail=\"\$fail identity\"
echo \"identity (switches unset vs maa1e98a): \$(tail -1 $O/cmp_off.txt)\" >> $O/RESULT
for e in res query fix; do
  case \$e in res) env=ISAAC_BLOCK_EGO=res;; query) env=ISAAC_BLOCK_EGO=query;; fix) env=ISAAC_ENTITY_SCALE_FIX=1;; esac
  env \$env \$BM1_PY c112_b2/run_b2.py --out $O/\$e --weights random \$A > $O/run_\$e.txt 2>&1 || fail=\"\$fail run-\$e\"
  \$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/\$e differ > $O/cmp_\$e.txt 2>&1
  echo \"\$e forward: \$(grep -c . $O/cmp_\$e.txt) lines; z/o max|diff| \$(grep -E ' (z|o) ' $O/cmp_\$e.txt | sed -n 's/.*max|diff| //p' | tr '\n' ' ')\" >> $O/RESULT
done
if [ -z \"\$fail\" ]; then echo 'RW4 TESTS OK' >> $O/RESULT; else echo \"RW4 TESTS FAILED:\$fail\" >> $O/RESULT; fi
"
tail -1 $P/rw4_tests/RESULT 2>/dev/null
