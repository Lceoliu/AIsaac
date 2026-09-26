"""Consistency check of the J460 (Repentance+) port on whole generated runs (main route).

This is not a validation against the game: it checks that every floor of many runs generates without
errors and satisfies the invariants the placement rules imply, and prints the distributions (room
counts, special rooms, curses, stage types, bosses) for a plausibility read.
usage: python tools/check_rep_runs.py [runs] [seed] [last_stage]
"""
import collections
import random
import sys
import time
import traceback

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.floor import Floor
from isaac_macro.rep.level import RepGameContext
from isaac_macro.rep.rng import Seeds
from isaac_macro.rep.roomconfig import default_room_config
from isaac_macro.rep.run import iter_run

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
last_stage = int(sys.argv[3]) if len(sys.argv) > 3 else 11
TYPE_NAMES = {2: 'shop', 4: 'treasure', 5: 'boss', 6: 'miniboss', 7: 'secret', 8: 'supersecret', 9: 'arcade',
              10: 'curse_room', 11: 'challenge', 12: 'library', 13: 'sacrifice', 18: 'bedroom', 19: 'bedroom',
              20: 'vault', 21: 'dice', 24: 'planetarium', 29: 'ultrasecret'}

rc = default_room_config()
rnd = random.Random(seed)
stats = collections.Counter()
per_stage = collections.defaultdict(collections.Counter)
errors = collections.Counter()
examples = []
t0 = time.time()
floors = 0
for _ in range(runs):
    start = rnd.getrandbits(32) or 1
    for debug in (True, False):
        ctx = RepGameContext()
        try:
            for level in iter_run(rc, Seeds(start), ctx, last_stage=last_stage, debug_start=debug,
                                  cathedral=rnd.random() < 0.5):
                floors += 1
                ps = per_stage[(level.stage, level.stage_type)]
                ps['floors'] += 1
                ps['rooms'] += len(level.rooms)
                ps['attempts'] += level.attempts
                ps['curse'] += bool(level.curses)
                ps['labyrinth'] += bool(level.curses & 2)
                types = collections.Counter(d.config.type for d in level.rooms)
                for t, name in TYPE_NAMES.items():
                    ps[name] += types[t] > 0
                # invariants
                grid_rooms = [d for d in level.rooms if d.grid_index >= 0]
                floor = Floor.from_level(level)
                cells = collections.Counter(c for r in floor.rooms for c in r.cells)
                if any(n > 1 for n in cells.values()):
                    over = {floor.grid[c] for c, n in cells.items() if n > 1}
                    shared = [r for r in floor.rooms if any(cells[c] > 1 for c in r.cells)]
                    all_sins = all(ctx.flag(b) for b in range(9, 15))
                    if any(r.type == 10 for r in shared) and all_sins:
                        stats['curse room reused (all sins seen, engine quirk)'] += 1
                    else:
                        errors['overlapping cells'] += 1
                if types[5] < 1:
                    errors['no boss room'] += 1
                if types[8] != 1 and level.stage <= 8:
                    errors['super secret count != 1'] += 1
                if types[7] < 1 and level.stage <= 8:
                    errors['no secret room'] += 1
                reach = floor.distances()
                unreachable = [r for r in floor.rooms if r.index not in reach and r.type != 29
                               and not (r.type == 10 and all(ctx.flag(b) for b in range(9, 15)))]
                if unreachable:
                    errors['unreachable room'] += 1
                if floor.room_at(floor.start) is None:
                    errors['no start room'] += 1
                stats['rooms'] += len(grid_rooms)
        except NotImplementedError as exc:
            errors['NotImplemented: ' + str(exc)[:60]] += 1
        except Exception as exc:  # noqa: BLE001
            errors[type(exc).__name__ + ': ' + str(exc)[:80]] += 1
            if len(examples) < 3:
                examples.append((start, debug, traceback.format_exc()))

dt = time.time() - t0
print(f'{runs * 2} runs, {floors} floors, {dt:.1f} s ({dt / max(floors, 1) * 1000:.1f} ms/floor)')
print('errors / invariant violations:', dict(errors) or 'none')
print('notes:', {k: v for k, v in stats.items() if k != 'rooms'} or 'none')
for e in examples:
    print('example', e[0], 'debug' if e[1] else 'normal')
    print(e[2][-1500:])
cols = ['floors', 'rooms', 'attempts', 'curse', 'labyrinth', 'shop', 'treasure', 'planetarium', 'secret',
        'supersecret', 'ultrasecret', 'curse_room', 'miniboss', 'challenge', 'arcade', 'vault', 'library',
        'sacrifice', 'dice', 'bedroom']
print(f"{'stage.type':>10s} {'floors':>6s} {'rooms':>6s} {'tries':>6s} {'curse':>6s} {'XL':>5s} "
      f"{'shop':>5s} {'treas':>5s} {'plan':>5s} {'sec':>5s} {'ssec':>5s} {'usec':>5s} {'crs':>5s} {'mini':>5s} "
      f"{'chal':>5s} {'arc':>5s} {'vault':>5s} {'lib':>5s} {'sac':>5s} {'dice':>5s} {'bed':>5s}")
for key in sorted(per_stage):
    ps = per_stage[key]
    n = ps['floors']
    f = lambda k: ps[k] / n
    print(f"{key[0]:>7d}.{key[1]:<2d} {n:6d} {f('rooms'):6.1f} {f('attempts'):6.2f} {f('curse'):6.3f} "
          f"{f('labyrinth'):5.3f} {f('shop'):5.2f} {f('treasure'):5.2f} {f('planetarium'):5.2f} {f('secret'):5.2f} "
          f"{f('supersecret'):5.2f} {f('ultrasecret'):5.2f} {f('curse_room'):5.2f} {f('miniboss'):5.2f} {f('challenge'):5.2f} "
          f"{f('arcade'):5.2f} {f('vault'):5.2f} {f('library'):5.2f} {f('sacrifice'):5.2f} {f('dice'):5.2f} "
          f"{f('bedroom'):5.2f}")
