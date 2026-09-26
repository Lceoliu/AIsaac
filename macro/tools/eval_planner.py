"""Rule planner v0 (isaac_macro/planner.py) on generated floors: transitions and hidden rooms found.
usage: python tools/eval_planner.py [runs] [seed]
"""
import collections
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.dataset import iter_floors
from isaac_macro.planner import Explorer
from isaac_macro.roomconfig import default_room_config

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rc = default_room_config()
agg = collections.defaultdict(list)
for rec, floor in iter_floors(rc, runs, seed):
    for bombs in (0, 1, 2, 3, 5):
        r = Explorer(floor, bombs=bombs).run()
        k = f'bombs={bombs}'
        agg[k + ' transitions'].append(r.transitions)
        agg[k + ' coverage'].append(r.rooms_visited / r.rooms_total)
        agg[k + ' boss'].append(r.reached_boss)
        agg[k + ' bombs_used'].append(r.bombs_used)
        if r.has_secret:
            agg[k + ' secret'].append(r.found_secret)
        if r.has_super_secret:
            agg[k + ' super_secret'].append(r.found_super_secret)
    agg['rooms'].append(floor and len(floor.rooms))
n = len(agg['rooms'])
print(f'{n} floors, mean rooms {sum(agg["rooms"]) / n:.1f}')
print(f"{'start bombs':>11s} {'transitions':>11s} {'coverage':>8s} {'boss':>5s} {'bombs used':>10s} {'secret':>7s} {'super':>6s}")
for bombs in (0, 1, 2, 3, 5):
    k = f'bombs={bombs}'
    m = lambda s: sum(agg[k + s]) / len(agg[k + s])
    print(f"{bombs:11d} {m(' transitions'):11.1f} {m(' coverage'):8.3f} {m(' boss'):5.2f} {m(' bombs_used'):10.2f} "
          f"{m(' secret'):7.3f} {m(' super_secret'):6.3f}")
