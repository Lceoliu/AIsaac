#!/bin/sh
# gpu57 C112 rw4 + ISAAC_BLOCK_EGO=latent checks (2026-11-17, the coordinator), CPU only (jobcpu.sh), after B6:
#   1. 12 block CPU test files in maa1e98a-rw4 with the switches unset and with EGO=latent, and the same files (but
#      test_block_adv_alignment, rw4-only) in unmodified maa1e98a as the baseline. Passing: no FAILED / ERROR line in
#      off or latent that is not also in the baseline (the known teardown error of
#      test_packed_learner_spatial_matches_padded[cpu] among them);
#   2. default-off identity (run_b2.py block, 4 streams, random + l2blk, CPU; maa1e98a-b2 vs rw4): bit-equal;
#   3. EGO=latent forward (random init): runs; arrays compared (differ).
# Then, when OK: B6 for EGO=latent and EGO=latent + ENTITY_SCALE_FIX=1 (same seeds, same streams).
# Lines in c112/STATUS_rw4t4 / STATUS_b6; output c112/rw4_tests4/ (RESULT's last line "RW4 TESTS OK" / "... FAILED").
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; O=/bench/train/c112/rw4_tests4; R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
say() { f=$1; shift; echo "$(date '+%F %T') $*" >> $P/$f; echo "$(date '+%F %T') c112_rw4_tests4: $*" >> $BM/reports/bm1.log; }
st() { say STATUS_rw4t4 "$*"; }
mkdir -p $P/rw4_tests4; rm -f $P/rw4_tests4/RESULT
st "waiting for B6 (6 settings) and the CPU slot"
while ! grep -q "B6 over" $P/STATUS_b6 2>/dev/null; do sleep 30; done
while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
cd $BM
st "tests started"
CPUSET=0-7,16-23 MEMS=0 MEM=24g sh $BM/jobcpu.sh c112-rw4-tests4 "
B0=/bench/bm1/trees/maa1e98a/bridge/python; B=/bench/bm1/trees/maa1e98a-b2/bridge/python; fail=''
T='tests/test_block_memory.py tests/test_block_slot_integration.py tests/test_block_packed_learner.py tests/test_b52_entry.py tests/test_block_loss_only.py tests/test_b51_history_scale.py tests/test_block_relative_history.py tests/test_visible_input.py tests/test_block_training.py tests/test_b58c_lineage.py tests/test_b60_native_pack.py'
cd \$B0 && nice -n 10 \$BM1_PY -m pytest -p no:cacheprovider -q \$T > $O/pytest_base.txt 2>&1; echo \"pytest base (maa1e98a, 11 files): \$(tail -1 $O/pytest_base.txt)\" >> $O/RESULT
grep -E '^(FAILED|ERROR) ' $O/pytest_base.txt | sed 's/ - .*//' | sort > $O/base_bad.txt
cd $R
for e in off latent; do
  case \$e in off) env=;; latent) env=ISAAC_BLOCK_EGO=latent;; esac
  env \$env nice -n 10 \$BM1_PY -m pytest -p no:cacheprovider -q \$T tests/test_block_adv_alignment.py > $O/pytest_\$e.txt 2>&1; rc=\$?
  grep -E '^(FAILED|ERROR) ' $O/pytest_\$e.txt | sed 's/ - .*//' | sort > $O/\${e}_bad.txt
  new=\$(comm -23 $O/\${e}_bad.txt $O/base_bad.txt | grep -c .)
  echo \"pytest \$e (12 files): \$(tail -1 $O/pytest_\$e.txt) rc \$rc; failures not in the baseline: \$new\" >> $O/RESULT
  comm -23 $O/\${e}_bad.txt $O/base_bad.txt | sed 's/^/  new: /' >> $O/RESULT
  [ \$new = 0 ] || fail=\"\$fail pytest-\$e\"
done
A='--block-input payloads --payloads c112_b2/payloads --payload-files 4 --stacks block --device cpu --keep-caps 1 --cheap 1'
cd \$B && \$BM1_PY c112_b2/run_b2.py --out $O/ref --weights random,l2blk \$A > $O/run_ref.txt 2>&1 || fail=\"\$fail run-ref\"
cd $R && \$BM1_PY c112_b2/run_b2.py --out $O/off --weights random,l2blk \$A > $O/run_off.txt 2>&1 || fail=\"\$fail run-off\"
\$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/off exact > $O/cmp_off.txt 2>&1 || fail=\"\$fail identity\"
echo \"identity (switches unset vs maa1e98a): \$(tail -1 $O/cmp_off.txt)\" >> $O/RESULT
env ISAAC_BLOCK_EGO=latent \$BM1_PY c112_b2/run_b2.py --out $O/latent --weights random \$A > $O/run_latent.txt 2>&1 || fail=\"\$fail run-latent\"
\$BM1_PY c112_rw4/cmp_caps.py $O/ref $O/latent differ > $O/cmp_latent.txt 2>&1
echo \"latent forward: \$(grep -E ' (p_tok|sp_mean|z|o|zr) ' $O/cmp_latent.txt | sed 's/.*npz //' | tr '\n' ';')\" >> $O/RESULT
if [ -z \"\$fail\" ]; then echo 'RW4 TESTS OK' >> $O/RESULT; else echo \"RW4 TESTS FAILED:\$fail\" >> $O/RESULT; fi
"
r=$(tail -1 $P/rw4_tests4/RESULT 2>/dev/null)
st "tests over: ${r:-no RESULT}"
[ "$r" = "RW4 TESTS OK" ] || exit 1
B6="--stack block --payloads c112_b2/payloads"
for s in "ego_latent|-e ISAAC_BLOCK_EGO=latent" "ego_latent_scalefix|-e ISAAC_BLOCK_EGO=latent -e ISAAC_ENTITY_SCALE_FIX=1"; do
  n=${s%%|*}; e=${s#*|}
  while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
  say STATUS_b6 "B6 $n started ($e)"
  L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g ENVS="$e" sh $BM/jobcpu.sh c112-b6-$n "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b6.py --out /bench/train/c112/b6_out/$n $B6; echo b6 rc \$?" | tail -1)
  say STATUS_b6 "B6 $n done: $(tail -1 $L) ($L)"
done
say STATUS_b6 "B6 latent over"
