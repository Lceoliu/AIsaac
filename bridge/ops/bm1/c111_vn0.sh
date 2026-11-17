#!/bin/sh
# gpu57 C111 VN0 (2026-11-16, the coordinator): after HD (c111/AFTER_DONE_hd: trained and evaluated, lane b free):
#   1. B45's byte-equality tests against the deployed C encoder, CPU only (CUDA_VISIBLE_DEVICES empty) in one container
#      job on GPU 6's lane: tree maa1e98a's tests/test_visible_native.py (synthetic edge cases, 300 random
#      observations, the recorded payload streams bm1/payloads/b45-payloads2 via ISAAC_VISIBLE_PAYLOADS) and
#      tests/test_compact_records.py -> c111/b45_tests.txt;
#   2. the VN0 smoke chain (c111_smoke.sh vn0: ab2-bm's train_args, ENVS ISAAC_VISIBLE_NATIVE=0, every other optimisation
#      on, lane b), then c111_go.sh vn0 (x_real >= 30 and D1 -> the 400-hour run).
#   usage: setsid nohup sh c111_vn0.sh > /root/AIsaac/train/c111/vn0_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c111; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" | tee -a $P/STATUS_vn0; echo "$(date '+%F %T') c111_vn0: $*" >> $BM/reports/bm1.log; }
say "waiting for AFTER_DONE_hd"
while [ ! -f $P/AFTER_DONE_hd ]; do sleep 60; done
cd $BM
L=$(GPU=6 CPUSET=8-15,24-31 MEMS=1 MEM=32g ENVS="-e CUDA_VISIBLE_DEVICES= -e ISAAC_VISIBLE_PAYLOADS=/bench/bm1/payloads/b45-payloads2" sh $BM/job.sh b45-tests "cd /bench/bm1/trees/maa1e98a/bridge/python && \$BM1_PY -m pytest -p no:cacheprovider -rA -q tests/test_visible_native.py tests/test_compact_records.py > /bench/train/c111/b45_tests.txt 2>&1; echo pytest rc \$? >> /bench/train/c111/b45_tests.txt" | tail -1)
say "B45 tests: $(tail -2 $P/b45_tests.txt | tr '\n' ' ') ($L)"
setsid nohup sh $BM/c111_go.sh vn0 maa1e98a env/gpu57-c110-b.env 6 8-15,24-31 1 > $P/go_vn0.out 2>&1 < /dev/null &
sh $BM/c111_smoke.sh vn0 maa1e98a maa1e98a-d1 env/gpu57-c110-b.env 6,7 8-15,24-31 1 on > $P/chain_vn0.out 2>&1
