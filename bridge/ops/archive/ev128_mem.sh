#!/bin/bash
# 128-seed run-mode evals (CPU) of checkpoints that need the B20 memory code: runs from the mv2 sandbox's python
# (repo >= 13f917c). Same outputs/layout as bld/ev128.sh. usage: ev128_mem.sh <tag> <checkpoint> [<tag> <checkpoint> ...]
export OMP_NUM_THREADS=4
cd /home/eolc/isaac-abplus/mv2/python
mkdir -p /home/eolc/isaac-abplus/bld/runs/ev128
while [ $# -ge 2 ]; do
  tag=$1; ck=$2; shift 2
  [ -f /home/eolc/isaac-abplus/bld/runs/ev128/$tag/summary.json ] && continue
  PYTHONPATH=. timeout 7200 /home/eolc/isaac-rl/.venv/bin/python eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
    --mode run --items --workers 4 --seeds 2147490000:128 --name e128m --port 32400 --out /home/eolc/isaac-abplus/bld/runs/ev128/$tag \
    > /home/eolc/isaac-abplus/bld/runs/ev128/$tag.log 2>&1 < /dev/null
  [ -f /home/eolc/isaac-abplus/bld/runs/ev128/$tag/summary.json ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" /home/eolc/isaac-abplus/bld/runs/ev128/$tag/summary.json "$tag" >> /home/eolc/isaac-abplus/bld/runs/ev128/evals.jsonl
done
