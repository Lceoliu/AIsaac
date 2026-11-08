"""B45: the C visible-frame encoder (bridge/engine/abp_visible.c via isaac_bridge.visible_native) writes
structured_input.observation_frame's bytes exactly: synthetic edge cases (a full entity list with overflow in every
class, an empty room, no inventory / slots, two bodies, curse masks, a hidden pedestal), random observations, and the
recorded real payload streams of abplus_probe_visible_frame.py (ISAAC_VISIBLE_PAYLOADS=<dir of payloads_*.npz>)."""
import glob
import os
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_bridge import visible_native
from isaac_bridge.abplus_lean import ENTITY, PLAYER, DOOR, MAP_ROOM, LeanDecoder
from isaac_bridge.structured_input import SCHEMA, observation_frame

LIB = visible_native.load()
pytestmark = pytest.mark.skipif(LIB is None, reason='no C visible encoder (gcc unavailable or disabled)')


def _obs(rng, n_players=1, n_ent=20, n_doors=4, n_grid=6, n_map=12, n_inv=5, n_slots=4, curses=0, width=15, height=9):
    players = np.zeros(n_players, PLAYER)
    for k in PLAYER.names:
        players[k] = rng.normal(0, 30, n_players)
    players['x'] = rng.uniform(60, 60+40*width, n_players)
    players['y'] = rng.uniform(140, 140+40*height, n_players)
    for k in ('hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal', 'golden', 'coins', 'bombs', 'keys'):
        players[k] = rng.integers(0, 24, n_players)
    cells = []
    for i in range(width*height):
        cells.append([i, 60+40*(i % width)+rng.choice([0, 0.5]), 140+40*(i//width), int(rng.integers(0, 7)),
                      int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(0, 2)),
                      int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(0, 2))])
    doors = np.zeros(n_doors, DOOR)
    doors['open'] = rng.integers(0, 2, n_doors); doors['locked'] = rng.integers(0, 2, n_doors)
    doors['seen'] = rng.integers(0, 4, n_doors); doors['target_type'] = rng.integers(0, 30, n_doors)
    doors['x'] = rng.uniform(40, 80+40*width, n_doors).astype(np.float32)
    doors['y'] = rng.uniform(120, 180+40*height, n_doors).astype(np.float32)
    grid = [[int(rng.integers(0, 300)), int(rng.choice([2, 4, 6, 14, 16, 17, 18, 21])), int(rng.integers(0, 256)),
             int(rng.integers(0, 256)), 0, float(rng.uniform(60, 60+40*width)), float(rng.uniform(140, 140+40*height))]
            for _ in range(n_grid)]
    ent = np.zeros(n_ent, ENTITY)
    ent['type'] = rng.choice([1, 2, 3, 4, 5, 6, 7, 9, 10, 20, 1000, 1022, 1023], n_ent)
    ent['variant'] = rng.choice([0, 100, 350, 7], n_ent); ent['subtype'] = rng.integers(0, 600, n_ent)
    for k in ('x', 'y', 'vx', 'vy', 'size', 'size_mx', 'size_my', 'cdmg', 'hp', 'height', 'fall', 'scale', 'l_cos',
              'l_sin', 'l_len', 'l_ex', 'l_ey', 'l_radius'):
        ent[k] = rng.normal(100, 200, n_ent)
    for k in ('coll', 'gcoll', 'aframe', 'age', 'champion', 'l_circle', 'price'):
        ent[k] = rng.integers(-1000, 4000, n_ent)
    ent['kind'] = rng.integers(0, 7, n_ent); ent['flags'] = rng.integers(0, 16, n_ent)
    ent['anim'] = rng.integers(0, 9, n_ent); ent['flip'] = rng.integers(0, 2, n_ent)
    ent['tflags'] = rng.integers(-2**62, 2**62, n_ent)
    mp = np.zeros(n_map, MAP_ROOM)
    mp['idx'] = rng.integers(0, 169, n_map); mp['shape'] = rng.integers(0, 14, n_map)
    mp['type'] = rng.integers(0, 8, n_map); mp['flags'] = rng.integers(0, 8, n_map)
    room = (0, 0, 0, 0, int(mp['idx'][0]) if n_map else 3, 1, 0, 0, 60.0, 140.0, 60.0+40*width, 140.0+40*height)
    inv = [[int(rng.integers(0, n_players)), int(rng.integers(1, 700)), int(rng.integers(1, 4)), 1]
           for _ in range(n_inv)]
    slots = [[0, 1, 0, int(rng.integers(0, 600)), 1, 1, float(rng.integers(0, 12)), 6, 1, 1, 1, 0, 0.5]
             for _ in range(n_slots)]
    visible = dict(schema=SCHEMA, curses=curses, stage=int(rng.integers(1, 12)), capabilities=1, inventory=inv,
                   slots=slots, bodies=[dict(controller=i-1, controller_known=int(i > 0), animation=int(rng.integers(0, 15)),
                                             animation_known=1) for i in range(n_players)])
    return SimpleNamespace(players=players, entities=ent, doors=doors, grid=grid, room=room, map=mp,
                           terrain=dict(width=width, height=height, cells=cells), visible=visible)


def _check(o, enc):
    ref = observation_frame(o)
    got = enc.frame(o)
    assert got == ref, (len(got), len(ref))


@pytest.mark.parametrize('case', ['full', 'empty', 'two_bodies_cursed', 'large_room', 'single_cell'])
def test_edge_cases_match_reference_bytes(case):
    rng = np.random.default_rng(7)
    enc = LIB.encoder()
    if case == 'full':      # > 128 entities: every class overflows, kept order by priority then distance then index
        o = _obs(rng, n_ent=260)
        o.entities['type'][:40] = 5; o.entities['kind'][:40] = 5
        o.entities['x'][:20] = o.players['x'][0]   # distance ties: the index decides
    elif case == 'empty':   # no entities, doors, grid, map, inventory or slots
        o = _obs(rng, n_ent=0, n_doors=0, n_grid=0, n_map=0, n_inv=0, n_slots=0)
    elif case == 'two_bodies_cursed':   # Curse of the Unknown (8) masks health; the Blind (64) hides pedestals
        o = _obs(rng, n_players=2, curses=(1 << 31) | 8 | 64)
        o.entities['type'][:5] = 5; o.entities['variant'][:5] = 100
    elif case == 'large_room':
        o = _obs(rng, n_ent=128, width=28, height=16, n_grid=40, n_map=60)
    else:
        o = _obs(rng, n_ent=3, width=1, height=1, n_grid=1, n_doors=1)
    _check(o, enc)
    assert enc.native == 1 and enc.reference == 0


def test_random_observations_match_reference_bytes():
    enc = LIB.encoder()
    rng = np.random.default_rng(11)
    for i in range(300):
        o = _obs(rng, n_players=int(rng.integers(1, 3)), n_ent=int(rng.integers(0, 200)), n_doors=int(rng.integers(0, 8)),
                 n_grid=int(rng.integers(0, 30)), n_map=int(rng.integers(0, 40)), n_inv=int(rng.integers(0, 60)),
                 n_slots=int(rng.integers(0, 8)), curses=int(rng.integers(0, 2**31)),
                 width=int(rng.integers(1, 29)), height=int(rng.integers(1, 17)))
        _check(o, enc)
    assert enc.reference == 0


def test_rejected_canvas_values_raise_the_reference_error():
    rng = np.random.default_rng(3)
    o = _obs(rng)
    o.terrain['cells'][0][3] = 3.3   # collision / 6 * 6 is not an integer: outside the lossless schema
    with pytest.raises(ValueError):
        observation_frame(o)
    enc = LIB.encoder()
    with pytest.raises(ValueError):
        enc.frame(o)
    assert enc.reference == 1


PAYLOADS = os.environ.get('ISAAC_VISIBLE_PAYLOADS', '')


@pytest.mark.skipif(not PAYLOADS, reason='ISAAC_VISIBLE_PAYLOADS: recorded payload streams')
def test_recorded_payload_streams_match_reference_bytes():
    files = sorted(glob.glob(os.path.join(PAYLOADS, 'payloads_*.npz')))
    assert files
    total = 0
    for f in files:
        d = np.load(f)
        blob, at = d['blob'].tobytes(), 0
        dec, enc = LeanDecoder(), LIB.encoder()
        for n in d['lengths']:
            o = dec.decode(blob[at:at+n]); at += int(n)
            if o.visible is None:
                continue
            assert enc.frame(o) == observation_frame(o), (f, total)
            total += 1
        assert enc.reference == 0
    print('recorded decisions compared', total)
    assert total > 1000
