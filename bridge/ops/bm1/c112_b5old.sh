#!/bin/sh
# gpu57 C112 B5 old stack on all 16 workers of qm-smoke 007 (2026-11-17: the 5-worker subset had 750 enemy decisions in
# 2 workers, the enemy targets were skipped). CPU only (jobcpu.sh). Lines in c112/STATUS_b5.
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_b5; echo "$(date '+%F %T') c112_b5old: $*" >> $BM/reports/bm1.log; }
cd $BM
while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
say "old_all started again (the first attempt's command lost \$BM1_PY to quoting)"
L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g sh $BM/jobcpu.sh c112-b5-oldall "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b5.py --out /bench/train/c112/b5_out/old_all --stacks old --weights c110qm,c99old,random --old-workers 0-15 --max-decisions 0 --cs 0.1 --device cpu --threads 8; echo b5 rc \$?" | tail -1)
say "old_all done: $(tail -1 $L) ($L)"
