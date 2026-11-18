#!/bin/bash
# 512-seed run-mode evals on the 3090 (CPU, sandbox mem2 = workspace with B20/B21/B22 code): the decisive numbers for
# the final checkpoints (128 seeds give +-0.07 floors; 512 give +-0.035). Seeds 2147490000:512 (the first 128 are the
# usual ones). usage: ev512_3090.sh <tag> <checkpoint> [<tag> <checkpoint> ...]   out: ~/isaac-abplus/mem2/runs/ev512/<tag>
export OMP_NUM_THREADS=4
S=/home/rhythmo/isaac-abplus/mem2
PY=/home/rhythmo/isaac-abplus/.venv/bin/python
cd $S/python
mkdir -p $S/runs/ev512
while [ $# -ge 2 ]; do
  tag=$1; ck=$2; shift 2
  [ -f $S/runs/ev512/$tag/summary.json ] && continue
  PYTHONPATH=. timeout 14400 $PY eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json \
    --mode run --items --workers 10 --seeds 2147490000:512 --name e512 --port 33300 --out $S/runs/ev512/$tag \
    > $S/runs/ev512/$tag.log 2>&1 < /dev/null
  [ -f $S/runs/ev512/$tag/summary.json ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" $S/runs/ev512/$tag/summary.json "$tag" >> $S/runs/ev512/evals.jsonl
done
