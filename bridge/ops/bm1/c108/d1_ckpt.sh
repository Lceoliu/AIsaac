#!/bin/sh
# C108 D1: after d2_run.sh, the bit-identity check of the lr-0 runs' weights (ckpt_same.py), one job on GPU 4.
while pgrep -f "^sh /root/AIsaac/bm1/c108/d2_run.sh" > /dev/null; do sleep 30; done
I=/bench/train/c108/d1
GPU=4 MEM=16g sh /root/AIsaac/bm1/job.sh d1-ckpt "\$BM1_PY /bench/bm1/c108/ckpt_same.py $I/inputs/bm-last.pt $I/a-bm-last/checkpoints/last.pt $I/inputs/bm-u0250.pt $I/a-bm-u0250/checkpoints/last.pt $I/inputs/c99old-last.pt $I/b-c99old-last/checkpoints/last.pt > $I/ckpt_same.txt 2>&1"
