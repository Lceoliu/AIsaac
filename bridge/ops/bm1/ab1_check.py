#!/usr/bin/env python3
"""ab1 (2026-11-14): the training episodes' check of a K=1 run directory (read-only, standard library).
  ab1_check.py <run dir>/<name>-s0 [first_update last_update]
Per episode file rows: forced-pickup episodes / steps, done counts, the no-progress ends (stall 1) with their done and
ret + 0.002 x decisions (the time term put back: -1 = the timeout penalty; rooms / damage / hurt / heal / gathered 0
only, so no other term), build0 (start builds), char0, start kinds."""
import collections
import json
import os
import sys

d = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
eps = []
for name in sorted(os.listdir(d)):
    if name.startswith('episodes.jsonl'):
        with open(os.path.join(d, name)) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if lo <= int(e.get('update', 0)) <= hi:
                    eps.append(e)
n = len(eps)
print(f'{d}: {n} episodes, updates {lo}..{min(hi, max([e.get("update", 0) for e in eps] or [0]))}')
if not n:
    sys.exit(1)
pf = [e for e in eps if e.get('pickup_forced')]
print(f'pickup_forced episodes {len(pf)} ({len(pf) / n:.3f}), forced_steps sum {sum(e.get("forced_steps", 0) for e in eps)}'
      f', episodes with forced_steps > 0 {sum(1 for e in eps if e.get("forced_steps", 0) > 0)}')
print('done', dict(sorted(collections.Counter(e['done'] for e in eps).items())),
      ' stall 1:', dict(sorted(collections.Counter(e['done'] for e in eps if e.get('stall')).items())))
clean = [e for e in eps if e.get('stall') and e['done'] == 3 and not e.get('rooms') and not e.get('damage')
         and not e.get('hurt') and not e.get('healed') and not e.get('gathered')]
res = collections.Counter(round(e['ret'] + 0.002 * e['decisions'], 2) for e in clean)
print(f'stall ends without any other reward term: {len(clean)}; ret + 0.002 x decisions: {dict(res.most_common(6))}')
cap = [e for e in eps if e['done'] == 3 and not e.get('stall')]
print(f'done 3 without stall (the cap): {len(cap)}')
b = [e for e in eps if e.get('build0')]
print(f'build0 non-empty {len(b)} ({len(b) / n:.3f}); fresh starts {sum(1 for e in eps if e.get("start") != "archive")},'
      f' with build0 {sum(1 for e in b if e.get("start") != "archive")}')
print('char0', dict(collections.Counter(e.get('char0') for e in eps)), ' start', dict(collections.Counter(e.get('start') for e in eps)))
print('items per episode', round(sum(e.get('items', 0) for e in eps) / n, 3), ' rooms cleared >= 1 share',
      round(sum(1 for e in eps if e.get('rooms', 0) >= 1) / n, 3))
