"""Compare engine floor dumps (rl/bridge/python/abplus_probe_floors.py) with the offline generator.

For every dumped floor the generator runs with the dump's stage, stage type and stage seed, the
dump's player resources, and the run state flags of the previous dump of the same run (the flags
a floor starts with). Every room is compared in list order: grid index, type, variant, subtype,
shape and the three per-room seeds. Also reported: which room types were covered, and Labyrinth
floors.
usage: python tools/compare_engine_floors.py <dump.jsonl> [--show N]
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, __file__.rsplit('tools', 1)[0])
from isaac_macro.floor import ROOM_TYPE_NAMES
from isaac_macro.level import GameContext, Player, generate_floor
from isaac_macro.roomconfig import default_room_config

FIELDS = ('grid', 'type', 'variant', 'subtype', 'shape', 'spawn_seed', 'decoration_seed', 'award_seed')
# Run flags that level generation writes (choose_boss and the miniboss room); other bits change in play.
GEN_FLAGS = sum(1 << b for b in (0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 27, 28, 32, 35, 36, 42,
                                 43))


def ours(rc, rec, flags):
    p = rec['player']
    ctx = GameContext(state_flags=flags, player=Player(hearts=p['hearts'], max_hearts=p['max_hearts'],
                                                       soul_hearts=p['soul_hearts'], keys=p['keys'],
                                                       coins=p['coins'], active_item=105))
    level = generate_floor(rc, ctx, rec['stage'], rec['stage_type'], rec['stage_seed'])
    rooms = [dict(grid=d.grid_index, type=d.type, variant=d.variant, subtype=d.config.subtype,
                  shape=d.shape, spawn_seed=d.spawn_seed, decoration_seed=d.decoration_seed,
                  award_seed=d.award_seed) for d in level.rooms]
    return level, rooms, ctx.state_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--show', type=int, default=3)
    args = ap.parse_args()
    rc = default_room_config()
    per_stage = collections.defaultdict(collections.Counter)
    covered = collections.Counter()        # room type -> rooms compared on floors that match
    rooms_total = rooms_ok = 0
    shown = 0
    prev_flags = {}
    for line in open(args.dump):
        rec = json.loads(line)
        flags_in = prev_flags.get(rec['run'], 0) if rec['step'] else 0
        engine_flags = sum(1 << f for f in rec['flags'])
        prev_flags[rec['run']] = engine_flags
        st = per_stage[(rec['stage'], rec['stage_type'])]
        st['floors'] += 1
        try:
            level, mine, flags_after = ours(rc, rec, flags_in)
        except NotImplementedError:
            st['not translated'] += 1
            continue
        engine = [{k: r[k] for k in FIELDS} for r in rec['rooms']]
        ok = mine == engine
        st['rooms ok'] += ok
        st['curses ok'] += level.curses == rec['curses']
        st['flags ok'] += (engine_flags & GEN_FLAGS) == (flags_after & GEN_FLAGS)
        st['labyrinth'] += bool(rec['curses'] & 2)
        st['labyrinth ok'] += bool(rec['curses'] & 2) and ok
        rooms_total += len(engine)
        rooms_ok += sum(1 for a, b in zip(mine, engine) if a == b)
        if ok:
            for r in engine:
                covered[r['type']] += 1
        elif shown < args.show:
            shown += 1
            print(f"MISMATCH run {rec['run']} step {rec['step']} ({rec['command']}) stage "
                  f"{rec['stage']}.{rec['stage_type']} seed {rec['stage_seed']} flags_in {flags_in:#x}:")
            for i in range(max(len(mine), len(engine))):
                a = mine[i] if i < len(mine) else None
                b = engine[i] if i < len(engine) else None
                if a != b:
                    print('  room', i, 'ours  ', a)
                    print('  room', i, 'engine', b)
    print(f"{'stage.type':>10s} {'floors':>6s} {'rooms ok':>8s} {'curses ok':>9s} {'flags ok':>8s} {'labyrinth':>9s}")
    tot = collections.Counter()
    for key in sorted(per_stage):
        st = per_stage[key]
        tot.update(st)
        lab = f"{st['labyrinth ok']}/{st['labyrinth']}" if st['labyrinth'] else '-'
        print(f"{key[0]:>7d}.{key[1]:<2d} {st['floors']:6d} {st['rooms ok']:8d} {st['curses ok']:9d} "
              f"{st['flags ok']:8d} {lab:>9s}" + (f"  (not translated: {st['not translated']})"
                                                  if st['not translated'] else ''))
    print(f"{'total':>10s} {tot['floors']:6d} {tot['rooms ok']:8d} {tot['curses ok']:9d} {tot['flags ok']:8d} "
          f"{tot['labyrinth ok']}/{tot['labyrinth']:<7d}")
    print(f'rooms identical: {rooms_ok}/{rooms_total}')
    print('room types compared on identical floors:',
          ', '.join(f'{ROOM_TYPE_NAMES.get(t, t)} {n}' for t, n in sorted(covered.items())))


if __name__ == '__main__':
    main()
