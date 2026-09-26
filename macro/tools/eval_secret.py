"""Evaluate hidden-room inference on floors with exact ground truth.

For each floor: hide the secret and super secret rooms, compute P(cell), and count the bombs needed
when testing cells in that order (tied cells in random order). Methods:

  secret room
    most_nb        baseline: most neighbouring rooms first (the usual player rule)
    wall_slot      player rule: walls where a recognised layout has a door slot but no door first
    rules          exact placement rules, super secret room marginalised (secret.secret_posterior)
    rules_no_lay   as rules, special-room layouts unknown
    rules_ss_known as rules, super secret room position known
    rules+layouts  joint posterior with normal-room layout evidence (secret.hidden_posterior)
  super secret room
    ss_rules       depth-weighted candidates (secret.super_secret_prior)
    ss_nolayout    as ss_rules, normal-room door slots unknown
    ss+layouts     marginal of the joint posterior with layout evidence
    ss|secret      joint posterior with layout evidence, secret room already found

usage: python tools/eval_secret.py [runs] [seed]
       python tools/eval_secret.py --engine <dump.jsonl>   (floors dumped from the engine by
                                                            rl/bridge/python/abplus_probe_floors.py)
"""
import collections
import json
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.dataset import iter_floors
from isaac_macro.floor import Floor, FloorRoom
from isaac_macro.levelgen import GRID, TRAVEL, index
from isaac_macro.roomconfig import default_room_config
from isaac_macro.secret import (LayoutEvidence, expected_bombs, hidden_posterior, hit_within,
                                secret_posterior, super_secret_prior, wall_slot_heuristic)

rc = default_room_config()
_evidence = {}


def evidence(floor):
    key = (floor.stage, floor.stage_type)
    if key not in _evidence:
        _evidence[key] = LayoutEvidence(rc, floor.stage, floor.stage_type)
    return _evidence[key]


def engine_floors(path):
    for line in open(path):
        rec = json.loads(line)
        if rec['stage'] > 8:
            continue
        floor = Floor.from_engine_dump(rec, rc)
        yield dict(secret=[r.safe_grid_index for r in floor.rooms if r.type == 7],
                   super_secret=[r.safe_grid_index for r in floor.rooms if r.type == 8]), floor


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


if len(sys.argv) > 2 and sys.argv[1] == '--engine':
    source = engine_floors(sys.argv[2])
else:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    source = iter_floors(rc, runs, seed)

agg = collections.defaultdict(list)
stats = collections.Counter()
calib = collections.defaultdict(lambda: [0.0, 0, 0])   # predicted-probability bin -> [sum p, hits, n]


def score(name, post, truth):
    if not post or truth not in post:
        stats[name + ' miss'] += 1
        agg[name].append(float('nan'))
        return
    agg[name].append(expected_bombs(post, truth))
    agg[name + '_p'].append(post[truth] / sum(post.values()))
    agg[name + '_1'].append(hit_within(post, truth, 1))
    agg[name + '_2'].append(hit_within(post, truth, 2))
    agg[name + '_3'].append(hit_within(post, truth, 3))


for rec, floor in source:
    stats['floors'] += 1
    if len(rec['secret']) != 1 or len(rec['super_secret']) != 1:
        stats['skipped (not exactly one secret and one super secret room)'] += 1
        continue
    truth, ss = rec['secret'][0], rec['super_secret'][0]
    vis = floor.visible()
    ev = evidence(floor)
    joint_s, joint_ss = hidden_posterior(vis, ev)
    score('most_nb', most_neighbours(vis), truth)
    score('wall_slot', wall_slot_heuristic(vis), truth)
    score('rules', secret_posterior(vis), truth)
    score('rules_no_lay', secret_posterior(unknown_layouts(vis)), truth)
    score('rules_ss_known', secret_posterior(vis, ss), truth)
    score('rules+layouts', joint_s, truth)
    score('ss_rules', super_secret_prior(vis), ss)
    score('ss_nolayout', super_secret_prior(unknown_layouts(vis, True)), ss)
    score('ss+layouts', joint_ss, ss)
    # super secret room once the secret room is known: condition the joint posterior on it
    _, ss_given = hidden_posterior(vis, ev, secret_cell=truth)
    score('ss|secret', ss_given, ss)
    for c, p in joint_s.items():
        k = min(int(p * 10), 9)
        calib[k][0] += p
        calib[k][1] += c == truth
        calib[k][2] += 1

print(dict(stats))
print(f"{'method':15s} {'floors':>6s} {'P(truth)':>8s} {'1 bomb':>7s} {'<=2':>6s} {'<=3':>6s} {'E[bombs]':>8s}")
for name in ('most_nb', 'wall_slot', 'rules_no_lay', 'rules', 'rules_ss_known', 'rules+layouts',
             'ss_nolayout', 'ss_rules', 'ss+layouts', 'ss|secret'):
    v = [x for x in agg[name] if x == x]
    if not v:
        continue
    m = lambda k: sum(agg[k]) / len(agg[k])
    print(f"{name:15s} {len(v):6d} {m(name + '_p'):8.3f} {m(name + '_1'):7.3f} {m(name + '_2'):6.3f} "
          f"{m(name + '_3'):6.3f} {sum(v) / len(v):8.2f}")
print('calibration of rules+layouts (bin: mean predicted / observed / cells):')
for k in sorted(calib):
    s, h, n = calib[k]
    print(f'  [{k / 10:.1f},{(k + 1) / 10:.1f}): {s / n:.3f} / {h / n:.3f} / {n}')
