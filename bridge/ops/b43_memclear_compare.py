"""B43 memory-clearing evaluation: per-group difference of two eval_tok.py summaries on the same seeds.

  python eval_tok.py --checkpoint CK --seeds S:N ... --out runs/eval-full
  python eval_tok.py --checkpoint CK --seeds S:N ... --memory-clear older:8 --out runs/eval-older8
  python eval_tok.py --checkpoint CK --seeds S:N ... --memory-clear recent:8 --out runs/eval-recent8
  python b43_memclear_compare.py runs/eval-full runs/eval-older8 [runs/eval-recent8 ...]

Prints, per group, the clear/death rates, clear-without-hit and hurt-related keys of each run and
their difference to the first (the uncleared baseline). Refuses summaries of different seeds or modes.
"""
import json
import sys
from pathlib import Path

runs = [json.loads((Path(a) / 'summary.json').read_text()) for a in sys.argv[1:]]
base = runs[0]
for r in runs[1:]:
    if r.get('seeds') != base.get('seeds') or r.get('mode') != base.get('mode'):
        raise SystemExit('compare evaluations of the same seeds and mode')
keys = ('clear', 'floor_clear', 'death', 'clear_without_hit', 'seconds_per_clear', 'episodes')
for name in base.get('groups', {}):
    print(f'== {name}')
    for path, r in zip(sys.argv[1:], runs):
        g = r['groups'].get(name, {})
        tag = r.get('memory_clear') or 'full'
        vals = {k: g[k] for k in keys if k in g}
        diff = {k: round(g[k]-base['groups'][name][k], 4) for k in vals if k in base['groups'][name]
                and isinstance(g[k], (int, float)) and k != 'episodes'}
        print(f'  {tag:>12} {r.get("memory_clear_fill", "")}: {json.dumps(vals)}'
              + ('' if r is base else f'  diff {json.dumps(diff)}'))
