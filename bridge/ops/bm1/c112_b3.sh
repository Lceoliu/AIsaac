#!/bin/sh
# gpu57 C112 B3 (2026-11-16, the coordinator): the EGO probe on CPU only (jobcpu.sh), tree maa1e98a-rw4,
# c112_b2/run_b2.py (B3 version: Reader layers, pool, logit(right) - logit(left) sensitivity), about 5000 decisions:
#   ego:  ISAAC_BLOCK_EGO=res, block stack, weights random (EGO=res, seed 1000) + l2ego-res final -> c112/b3_out/ego
#   base: switches unset, block + old stacks, weights l2blk + l2old final -> c112/b3_out/base
# One after the other (one CPU-only container at a time). Lines in c112/STATUS_b3 and bm1.log.
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_b3; echo "$(date '+%F %T') c112_b3: $*" >> $BM/reports/bm1.log; }
cd $BM; mkdir -p $P/b3_out
R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
A="--block-input payloads --payloads c112_b2/payloads --max-decisions 5000 --cheap 1 --device cpu --threads 8"
say "ego (EGO=res; random + l2ego-res) on CPU"
L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g ENVS="-e ISAAC_BLOCK_EGO=res" sh $BM/jobcpu.sh c112-b3-ego "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b2.py --out /bench/train/c112/b3_out/ego --stacks block --weights random,l2egores --random-config /bench/train/c112/l2ego-res/l2ego-res-s0/checkpoints/last.pt $A; echo b3 rc \$?" | tail -1)
say "ego done: $(tail -1 $L) ($L)"
say "base (switches unset; l2blk + l2old) on CPU"
L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g sh $BM/jobcpu.sh c112-b3-base "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b2.py --out /bench/train/c112/b3_out/base --stacks block,old --weights l2blk,l2old $A; echo b3 rc \$?" | tail -1)
say "base done: $(tail -1 $L) ($L)"
say "B3 over"
