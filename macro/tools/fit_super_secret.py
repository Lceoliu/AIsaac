"""Fit the super secret room weights (secret.SUPER_SECRET_*_WEIGHTS) on generated floors.

Conditional logit over the candidate cells of each floor's visible map (secret and super secret
rooms hidden) with two categorical features: `delta` (depth minus the deepest visible dead end,
clipped to [-2, 3]) and `boss_delta` (depth minus the boss room's depth, clipped to [-3, 1]).
usage: python tools/fit_super_secret.py [runs] [seed]
"""
import collections
import math
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.dataset import iter_floors
from isaac_macro.roomconfig import default_room_config
from isaac_macro.secret import super_secret_candidates

DELTA = (-2, 3)
BOSS = (-3, 1)
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rc = default_room_config()
data = []                       # per floor: (list of candidate feature keys, index of the truth)
stats = collections.Counter()
for rec, floor in iter_floors(rc, runs, seed):
    stats['floors'] += 1
    if not rec['super_secret']:
        stats['no_super_secret'] += 1
        continue
    truth = rec['super_secret'][0]
    cands = super_secret_candidates(floor.visible())
    if truth not in cands:
        stats['truth_not_candidate'] += 1
        continue
    order = list(cands)
    keys = [(('d', max(DELTA[0], min(DELTA[1], cands[c]['delta']))),
             ('b', max(BOSS[0], min(BOSS[1], cands[c]['boss_delta'])))) for c in order]
    data.append((keys, order.index(truth)))
    stats['candidates'] += len(order)

params = [('d', k) for k in range(DELTA[0], DELTA[1] + 1)] + [('b', k) for k in range(BOSS[0], BOSS[1] + 1)]
fixed = {('d', 0), ('b', 0)}
w = {p: 0.0 for p in params}
for it in range(800):
    grad = {p: 0.0 for p in params}
    ll = 0.0
    for keys, t in data:
        z = [math.exp(sum(w[f] for f in k)) for k in keys]
        s = sum(z)
        ll += math.log(z[t] / s)
        for f in keys[t]:
            grad[f] += 1
        for k, zi in zip(keys, z):
            for f in k:
                grad[f] -= zi / s
    for p in params:
        if p not in fixed:
            w[p] += 0.5 * grad[p] / len(data)
dw = {k: round(math.exp(w[('d', k)]), 4) for k in range(DELTA[0], DELTA[1] + 1)}
bw = {k: round(math.exp(w[('b', k)]), 4) for k in range(BOSS[0], BOSS[1] + 1)}
p_truth = 0.0
for keys, t in data:
    z = [dw[k[0][1]] * bw[k[1][1]] for k in keys]
    p_truth += z[t] / sum(z)
print(dict(stats))
print('mean candidates', round(stats['candidates'] / len(data), 2), 'log-lik/floor', round(ll / len(data), 4))
print('SUPER_SECRET_DELTA_WEIGHTS =', dw)
print('SUPER_SECRET_BOSS_WEIGHTS =', bw)
print('mean P(truth)', round(p_truth / len(data), 4))
