"""Held-out rooms of the goal-conditioned line (GOAL_CONDITIONED_DESIGN.md M2 / E-GOTO: ~20 % of the room variants
held out to measure generalisation). Splits C39's 491 1x1 normal rooms (catalog/scaling_normal_rooms.json) by the
solve class of catalog/room_classes.json (turret / walker / ...), 20 % of each class, numpy default_rng(20260930);
writes catalog/goal_rooms_train.json and catalog/goal_rooms_holdout.json (same format as the source).

C44 (user decision 2026-09-30): the same for the Basement I normal rooms of the other shapes (catalog/
scaling2_normal_big_rooms.json, C41's 358: IH, IV, 1x2, IIV, 2x1, IIH, 2x2 and the four L shapes), 20 % of each shape of
catalog/abplus_basement1_rooms.json held out, numpy default_rng(20260931): catalog/goal_big_rooms_train.json and
catalog/goal_big_rooms_holdout.json.

usage: python room_holdout.py   (from anywhere; paths are relative to this file)"""
import collections
import json
from pathlib import Path

import numpy as np

CATALOG = Path(__file__).resolve().parents[2] / 'bridge' / 'abplus' / 'catalog'
SEED, SHARE = 20260930, 0.2

source = json.loads((CATALOG / 'scaling_normal_rooms.json').read_text(encoding='utf8'))
classes = json.loads((CATALOG / 'room_classes.json').read_text(encoding='utf8'))['rooms']
by_class = collections.defaultdict(list)
for v in sorted(source['normal']):
    by_class[classes.get(str(v), {}).get('solve', 'unknown')].append(v)
rng = np.random.default_rng(SEED)
held = []
for name in sorted(by_class):
    rooms = by_class[name]
    k = int(round(SHARE * len(rooms)))
    held += [int(v) for v in rng.permutation(rooms)[:k]]
    print(f'{name}: {len(rooms)} rooms, {k} held out')
held = sorted(held)
train = sorted(set(source['normal']) - set(held))
note = (f"split of scaling_normal_rooms.json (C39's 491 1x1 normal rooms) for the goal-conditioned line: {SHARE:.0%} of each "
        f"solve class of room_classes.json held out, numpy default_rng({SEED}) (rl/docs/goal_conditioned/room_holdout.py)")
for name, rooms, what in (('goal_rooms_train.json', train, 'training rooms'), ('goal_rooms_holdout.json', held, 'held-out rooms')):
    out = dict(source, description=f'{what} ({len(rooms)}): {note}', normal=rooms,
               sources=dict(source.get('sources', {}), split=note))
    (CATALOG / name).write_text(json.dumps(out, indent=1) + '\n', encoding='utf8')
    print(name, len(rooms))


SHAPES = {2: 'IH', 3: 'IV', 4: '1x2', 5: 'IIV', 6: '2x1', 7: 'IIH', 8: '2x2', 9: 'LTL', 10: 'LTR', 11: 'LBL', 12: 'LBR'}
BIG_SEED = SEED + 1
big = json.loads((CATALOG / 'scaling2_normal_big_rooms.json').read_text(encoding='utf8'))
shape_of = {r['variant']: r['shape'] for r in json.loads((CATALOG / 'abplus_basement1_rooms.json').read_text(encoding='utf8'))['normal']}
by_shape = collections.defaultdict(list)
for v in sorted(big['normal']):
    by_shape[shape_of[v]].append(v)
rng = np.random.default_rng(BIG_SEED)
held = []
for shape in sorted(by_shape):
    rooms = by_shape[shape]
    k = int(round(SHARE * len(rooms)))
    held += [int(v) for v in rng.permutation(rooms)[:k]]
    print(f'{SHAPES[shape]}: {len(rooms)} rooms, {k} held out')
held = sorted(held)
train = sorted(set(big['normal']) - set(held))
note = (f"split of scaling2_normal_big_rooms.json (C41's 358 Basement I normal rooms of the other shapes) for the goal-conditioned "
        f"line (C44): {SHARE:.0%} of each shape held out, numpy default_rng({BIG_SEED}) (rl/docs/goal_conditioned/room_holdout.py)")
for name, rooms, what in (('goal_big_rooms_train.json', train, 'training rooms'), ('goal_big_rooms_holdout.json', held, 'held-out rooms')):
    out = dict(big, description=f'{what} ({len(rooms)}): {note}', normal=rooms, sources=dict(big.get('sources', {}), split=note))
    (CATALOG / name).write_text(json.dumps(out, indent=1) + '\n', encoding='utf8')
    print(name, len(rooms))
