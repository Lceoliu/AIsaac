#!/bin/sh
# gpu57 C112 B4 (2026-11-16, the coordinator): variance share V, within-stream step change, action log-prob state
# dependence (and the probes) for every tap and weights set, CPU only (jobcpu.sh, one container at a time; waits for
# the rw4 tests round 2 and any other CPU container first). Script: maa1e98a-rw4 c112_b2/run_b4.py. About 5000 decisions.
#   ego:        ISAAC_BLOCK_EGO=res, block, random (EGO=res) + l2ego-res
#   base_block: unset, block, l2blk + ab2-bm + c110-g
#   base_old:   unset, old, l2old + c110-qm + random (old, seed 1000)
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_b4; echo "$(date '+%F %T') c112_b4: $*" >> $BM/reports/bm1.log; }
cd $BM; mkdir -p $P/b4_out
say "B4 waiting for the rw4 tests round 2 and the CPU slot"
while ! grep -q "^RW4 TESTS" $P/rw4_tests2/RESULT 2>/dev/null; do sleep 30; done
A="--block-input payloads --payloads c112_b2/payloads --max-decisions 5000 --cheap 1 --device cpu --threads 8"
run() {   # name env args
  while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
  say "$1 started"
  L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g ENVS="$2" sh $BM/jobcpu.sh c112-b4-$1 "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b4.py --out /bench/train/c112/b4_out/$1 $3 $A; echo b4 rc \$?" | tail -1)
  say "$1 done: $(tail -1 $L) ($L)"
}
run ego "-e ISAAC_BLOCK_EGO=res" "--stacks block --weights random,l2egores --random-config /bench/train/c112/l2ego-res/l2ego-res-s0/checkpoints/last.pt"
run base_block "" "--stacks block --weights l2blk,ab2bm,c110g"
run base_old "" "--stacks old --weights l2old,c110qm,random"
say "B4 over"
