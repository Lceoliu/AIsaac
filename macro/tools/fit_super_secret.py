"""Fit the super secret room weights (secret.SUPER_SECRET_DELTA_WEIGHTS) on generated floors.

Conditional logit over the candidate cells of each floor's visible map (secret and super secret
rooms hidden), one weight per clipped delta. usage: python tools/fit_super_secret.py [runs] [seed]
"""
import collections
import math
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.dataset import iter_floors
from isaac_macro.roomconfig import default_room_config
from isaac_macro.secret import super_secret_candidates

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rc = default_room_config()
data = []                       # per floor: (list of candidate keys, index of the truth or -1)
stats = collections.Counter()
for rec, floor in iter_floors(rc, runs, seed):
    stats['floors'] += 1
    if not rec['super_secret']:
        stats['no_super_secret'] += 1
        continue
    truth = rec['super_secret'][0]
    cands = super_secret_candidates(floor.visible())
    keys = {c: max(-2, min(3, f['delta'])) for c, f in cands.items()}
    if truth not in keys:
        stats['truth_not_candidate'] += 1
        continue
    order = list(keys)
    data.append(([keys[c] for c in order], order.index(truth)))
    stats['candidates'] += len(order)

levels = [-2, -1, 0, 1, 2, 3]
w = {k: 0.0 for k in levels}            # log-weights, w[0] fixed at 0
for it in range(500):
    grad = {k: 0.0 for k in levels}
    ll = 0.0
    for keys, t in data:
        z = [math.exp(w[k]) for k in keys]
        s = sum(z)
        ll += w[keys[t]] - math.log(s)
        grad[keys[t]] += 1
        for k, zi in zip(keys, z):
            grad[k] -= zi / s
    for k in levels:
        if k != 0:
            w[k] += 0.5 * grad[k] / len(data)
weights = {k: round(math.exp(w[k]), 4) for k in levels}
top1 = sum(1 for keys, t in data if weights[keys[t]] >= max(weights[k] for k in keys)
           and sum(1 for k in keys if weights[k] == weights[keys[t]]) == 1)
p_truth = sum(weights[keys[t]] / sum(weights[k] for k in keys) for keys, t in data) / len(data)
print(dict(stats))
print('mean candidates', round(stats['candidates'] / len(data), 2), 'log-lik/floor', round(ll / len(data), 4))
print('weights', weights)
print('mean P(truth)', round(p_truth, 4), 'unique-argmax hits', round(top1 / len(data), 4))
