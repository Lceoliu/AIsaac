#!/bin/sh
# gpu57 C112 D6 (2026-11-16, the coordinator): the block learner's advantage / return / old value / old logp / action
# alignment test (tests/test_block_adv_alignment.py, packed and per-length learner, 2 seeds, 2 batch configurations) in
# the throwaway tree maa1e98a-d6 (maa1e98a + that test file), CPU only, nice 10, as a job on GPU 6's lane (job.sh needs
# the lane's GPU free; the test never touches it: CUDA_VISIBLE_DEVICES empty). Called by c112_ladder.sh once VN0's
# evaluation is done. Result: c112/d6_tests.txt; one line in c112/D6_DONE.
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1
cd $BM
L=$(GPU=6 CPUSET=8-15,24-31 MEMS=1 MEM=32g ENVS="-e CUDA_VISIBLE_DEVICES=" sh $BM/job.sh d6-tests "cd /bench/bm1/trees/maa1e98a-d6/bridge/python && nice -n 10 \$BM1_PY -m pytest -p no:cacheprovider -rA -q tests/test_block_adv_alignment.py > /bench/train/c112/d6_tests.txt 2>&1; echo pytest rc \$? >> /bench/train/c112/d6_tests.txt" | tail -1)
echo "$(date '+%F %T') D6: $(grep -E '(passed|failed|error)' $P/d6_tests.txt | tail -1) $(tail -1 $P/d6_tests.txt) ($L)" >> $P/D6_DONE
