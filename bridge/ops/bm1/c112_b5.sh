#!/bin/sh
# gpu57 C112 B5 (2026-11-17, the coordinator): can long-trained representations linearly decode the nearest vulnerable
# enemy and the nearest open door relative to the player? CPU only (jobcpu.sh / runcpu.sh: no GPU, no driver call), one
# container at a time. Script maa1e98a-rw4 c112_b2/run_b5.py.
#   1. counts only (B45 payload streams): fewer than 2000 decisions with a vulnerable enemy -> stop (a person decides);
#   2. block: ab2-bm, c110-g, random (10000 decisions, whole streams, enemy-rich first);
#   3. old: c110-qm, ab2-c99old, random (qm-smoke 007, 5 workers x 2000 steps).
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_b5; echo "$(date '+%F %T') c112_b5: $*" >> $BM/reports/bm1.log; }
cd $BM; mkdir -p $P/b5_out
run() {   # name args
  while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
  say "$1 started"
  L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g sh $BM/jobcpu.sh c112-b5-$1 "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b5.py --out /bench/train/c112/b5_out/$1 $2; echo b5 rc \$?" | tail -1)
  say "$1 done: $(tail -1 $L) ($L)"
}
run counts "--stacks block --counts-only 1 --payloads c112_b2/payloads"
n=$(python3 -c "
import json
d=json.load(open('$P/b5_out/counts/counts.json'))
def find(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if 'enem' in k.lower() and isinstance(v,(int,float)) and 'mean' not in k.lower() and 'both' not in k.lower(): return int(v)
        for v in x.values():
            r=find(v)
            if r is not None: return r
print(find(d))" 2>/dev/null)
say "counts: decisions with a vulnerable enemy ${n:-?} ($(python3 -c "import json;print(json.dumps(json.load(open('$P/b5_out/counts/counts.json')))[:300])" 2>/dev/null))"
if [ -z "$n" ] || [ "$n" = None ] || [ "$n" -lt 2000 ]; then say "B5 stopped: fewer than 2000 enemy decisions (or unreadable counts): a person decides"; exit 1; fi
run block "--stacks block --weights ab2bm,c110g,random --max-decisions 10000 --cs 0.1 --payloads c112_b2/payloads --device cpu --threads 8"
run old "--stacks old --weights c110qm,c99old,random --old-workers 0-15 --max-decisions 10000 --cs 0.1 --device cpu --threads 8"
say "B5 over"
