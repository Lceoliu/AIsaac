"""Evaluate the secret room posterior on generated floors (exact ground truth).

For each floor: hide the secret and super secret rooms, compute P(secret cell), and count the bombs
needed when testing cells in posterior order. Variants:
  oracle_ss   - super secret room position known (exact posterior of the placement rules)
  prior_ss    - super secret room marginalised with secret.super_secret_prior
  no_layouts  - as prior_ss, but special-room layouts unknown (all door slots assumed present)
  most_nb     - baseline: most neighbouring rooms first (the usual player rule), ties random
Super secret room: ss_prior (layouts of the visible rooms known, i.e. the player can identify a
room's layout and hence its door slots), ss_prior_nolayout (door slots unknown), ss_given_secret
(layouts known, secret room already found).
usage: python tools/eval_secret.py [runs] [seed]
       python tools/eval_secret.py --engine <dump.jsonl>   (floors dumped from the engine by
                                                            rl/bridge/python/abplus_probe_floors.py)
"""
import collections
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.dataset import iter_floors
from isaac_macro.floor import Floor, FloorRoom
from isaac_macro.levelgen import GRID, TRAVEL, index
from isaac_macro.roomconfig import default_room_config
from isaac_macro.secret import (expected_bombs, hit_within, secret_posterior, super_secret_posterior,
                                super_secret_prior)

rc = default_room_config()


def engine_floors(path):
    import json
    for line in open(path):
        rec = json.loads(line)
        if rec['stage'] > 8:
            continue
        floor = Floor.from_engine_dump(rec, rc)
        yield dict(secret=[r.safe_grid_index for r in floor.rooms if r.type == 7],
                   super_secret=[r.safe_grid_index for r in floor.rooms if r.type == 8]), floor


if len(sys.argv) > 2 and sys.argv[1] == '--engine':
    source = engine_floors(sys.argv[2])
else:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    source = iter_floors(rc, runs, seed)
agg = collections.defaultdict(list)
stats = collections.Counter()
calib = collections.defaultdict(lambda: [0.0, 0, 0])   # predicted-probability bin -> [sum p, hits, n]


def unknown_layouts(floor: Floor, normal_too: bool = False) -> Floor:
    rooms = [FloorRoom(r.index, r.x, r.y, r.shape, r.type, r.variant, r.subtype,
                       0xFF if r.type != 1 or normal_too else r.layout_doors) for r in floor.rooms]
    return Floor(rooms, floor.stage, floor.stage_type, floor.curses, 0, floor.start)


def most_neighbours(floor: Floor) -> dict:
    out = {}
    for i in range(GRID * GRID):
        if floor.grid[i] >= 0:
            continue
        n = sum(1 for dx, dy in TRAVEL if (j := index(i % GRID + dx, i // GRID + dy)) >= 0 and floor.grid[j] >= 0)
        if n:
            out[i] = float(n)
    return out


for rec, floor in source:
    stats['floors'] += 1
    if not rec['secret']:
        stats['no_secret'] += 1
        continue
    if len(rec['secret']) > 1:
        stats['two_secret'] += 1
        continue
    truth = rec['secret'][0]
    vis = floor.visible()
    variants = {
        'oracle_ss': secret_posterior(vis, rec['super_secret'][0] if rec['super_secret'] else None),
        'prior_ss': secret_posterior(vis),
        'no_layouts': secret_posterior(unknown_layouts(vis)),
        'most_nb': most_neighbours(vis),
    }
    for name, post in variants.items():
        b = expected_bombs(post, truth)
        if b != b:
            stats[name + '_miss'] += 1
            continue
        agg[name].append(b)
        agg[name + '_p'].append(post[truth] / sum(post.values()))
        agg[name + '_first'].append(hit_within(post, truth, 1))
        agg[name + '_le2'].append(hit_within(post, truth, 2))
    if rec['super_secret']:
        ss = rec['super_secret'][0]
        for name, post in (('ss_given_secret', super_secret_posterior(vis, truth)),
                           ('ss_prior', super_secret_prior(vis)),
                           ('ss_prior_nolayout', super_secret_prior(unknown_layouts(vis, True)))):
            b = expected_bombs(post, ss)
            if b != b:
                stats[name + '_miss'] += 1
                continue
            agg[name].append(b)
            agg[name + '_p'].append(post[ss])
            agg[name + '_first'].append(hit_within(post, ss, 1))
            agg[name + '_le2'].append(hit_within(post, ss, 2))
    for c, p in variants['oracle_ss'].items():
        k = min(int(p * 10), 9)
        calib[k][0] += p
        calib[k][1] += c == truth
        calib[k][2] += 1

print(dict(stats))
print(f"{'variant':17s} {'floors':>6s} {'P(truth)':>9s} {'1st bomb':>9s} {'<=2 bombs':>9s} {'E[bombs]':>9s}")
for name in ('oracle_ss', 'prior_ss', 'no_layouts', 'most_nb', 'ss_prior', 'ss_prior_nolayout',
             'ss_given_secret'):
    v = agg[name]
    if not v:
        continue
    m = lambda k: sum(agg[k]) / len(agg[k])
    print(f"{name:17s} {len(v):6d} {m(name + '_p'):9.3f} {m(name + '_first'):9.3f} {m(name + '_le2'):9.3f} {m(name):9.2f}")
print('calibration of oracle_ss (bin: mean predicted / observed frequency / cells):')
for k in sorted(calib):
    s, h, n = calib[k]
    print(f'  [{k / 10:.1f},{(k + 1) / 10:.1f}): {s / n:.3f} / {h / n:.3f} / {n}')
