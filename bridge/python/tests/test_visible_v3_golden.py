"""visible-v3 golden frames (V2 / F3, package C, 2026-11-17; tests/fixtures/visible_v3, from abplus_probe_v3_golden.py
on AB+, 184 golden17): 19 real frames -- every RoomShape 1-12, the off-grid Homunculus part 228.10, a
player in a door frame, a coin 600 frames old, a devil room's negative prices, a shooting player, the Forgotten with
the Soul (Forgotten controlled; Soul controlled). Each: the raw block (abp_bridge.lua visible_v3_raw), the reference encoder's bytes, the engine values read
apart. Here: the bytes are reproduced from the raw block (the reference encoder unchanged), and every body's and sent
entity's position in the frame is an engine Position minus (GetGridPosition(0) - (20, 20)) read apart."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import schema_v3 as V3
from isaac_bridge import visible_producer as VP

FIX = Path(__file__).resolve().parent / 'fixtures' / 'visible_v3'
MANIFEST = json.loads((FIX / 'manifest.json').read_text())
NAMES = [f['name'] for f in MANIFEST['frames']]


def test_the_cases_are_there():
    assert len(NAMES) == 19 and not MANIFEST['errors']
    assert {f'shape-{s}' for s in ('1x1', 'IH', 'IV', '1x2', 'IIV', '2x1', 'IIH', '2x2', 'LTL', 'LTR', 'LBL', 'LBR')} <= set(NAMES)
    assert {'offgrid', 'doorframe', 'longlived', 'devilprice', 'shooting', 'forgotten', 'forgotten-soul'} <= set(NAMES)


@pytest.mark.parametrize('name', NAMES)
def test_bytes_and_positions(name):
    blob = json.loads((FIX / f'{name}.raw.json').read_text())
    payload = (FIX / f'{name}.v3.bin').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == next(f['sha256'] for f in MANIFEST['frames'] if f['name'] == name)
    assert VP.observation_frame_v3(blob['raw'], stall_share=blob['stall_share'], stall_known=blob['stall_known']) == payload
    eng = json.loads((FIX / f'{name}.engine.json').read_text())
    ox, oy = eng['g0'][0] - 20, eng['g0'][1] - 20
    f = VP.unpack_frame_v3(payload)
    bodies = [(p['x'], p['y']) for p in eng['players']] + [(p['sub']['x'], p['sub']['y']) for p in eng['players'] if p.get('sub')]
    ents = [(e['x'], e['y']) for e in eng['entities']]
    for row in f['P']:
        if not row[V3.col(V3.P_COLS, 'position_known')]:   # (schema batch 3: no room position, written 0)
            assert row[0] == row[1] == 0
            continue
        assert any(abs(row[0] - (x - ox)) < 1e-3 and abs(row[1] - (y - oy)) < 1e-3 for x, y in bodies)
    for row in f['E']:
        assert any(abs(row[3] - (x - ox)) < 2e-3 and abs(row[4] - (y - oy)) < 2e-3 for x, y in ents)
    assert int(f['P'][0][V3.col(V3.P_COLS, 'grid_w')]) == eng['gw'] and int(f['P'][0][V3.col(V3.P_COLS, 'grid_h')]) == eng['gh']


def _frame(name):
    return VP.unpack_frame_v3((FIX / f'{name}.v3.bin').read_bytes())


def test_the_special_cases():
    E, P = V3.E_COLS, V3.P_COLS
    off = _frame('offgrid')['E']
    ys = off[:, V3.col(E, 'y')]
    assert any(int(r[0]) == 228 and int(r[1]) == 10 and r[V3.col(E, 'y')] >= 9 * 40 for r in off)   # below the 2x1 grid
    assert not any(int(r[0]) == 1000 and int(r[1]) == 41 for r in off)                         # the drops: decorative
    door = _frame('doorframe')['P'][0]
    col, row = int(door[0] // 40), int(door[1] // 40)
    assert row == 0 or col == 0 or col == int(door[V3.col(P, 'grid_w')]) - 1 or row == int(door[V3.col(P, 'grid_h')]) - 1
    assert max(_frame('longlived')['E'][:, V3.col(E, 'age')]) >= 600
    prices = [r[V3.col(E, 'price')] for r in _frame('devilprice')['E'] if int(r[0]) == 5]
    assert min(prices) < 0
    assert _frame('shooting')['P'][0][V3.col(P, 'fire_dir')] >= 0
    fg = _frame('forgotten')
    P0 = fg['P']
    assert len(P0) == 2 and [int(r[V3.col(P, 'ptype')]) - 2 for r in P0] == [16, 17]
    assert [int(r[V3.col(P, 'body_role')]) for r in P0] == [1, 3] and [int(r[V3.col(P, 'linked_body')]) for r in P0] == [2, 1]
    assert [int(r[V3.col(P, 'position_known')]) for r in P0] == [1, 0] and not fg['local'][1].any()
    soul = _frame('forgotten-soul')
    assert [int(r[V3.col(P, 'ptype')]) - 2 for r in soul['P']] == [17, 16]      # (the room entity is the Soul now)
    assert [int(r[V3.col(P, 'position_known')]) for r in soul['P']] == [1, 0]
    assert any(int(r[0]) == 3 and int(r[1]) == 900 for r in soul['E'])           # the Forgotten's body, a familiar
    assert not any(int(r[0]) == 1000 and int(r[1]) == 114 for r in soul['E'])   # the chain: decorative


def test_worker_row_and_stall_backfill():
    """The worker side (ISAAC_RL_VISIBLE_V3=1): write_frame_v3 puts the frame into a ROW (visible_schema = the v3
    VERSION) and v3_backfill_stall writes P 40 / 53 of every body after encode_row's progress update, nothing else."""
    from isaac_bridge.tok_obs import ROW
    payload = (FIX / 'forgotten.v3.bin').read_bytes()   # (two bodies)
    row = np.zeros(1, ROW)
    VP.write_frame_v3(row, payload)
    assert int(row['visible_schema'][0]) == V3.VERSION and int(row['visible_bytes'][0]) == len(payload)
    VP.v3_backfill_stall(row, 0.25, 1.0)
    got = bytes(row['visible_frame'][0, :len(payload)])
    a, b = VP.unpack_frame_v3(payload), VP.unpack_frame_v3(got)
    s, k = V3.col(V3.P_COLS, 'stall_share'), V3.col(V3.P_COLS, 'stall_known')
    assert list(b['P'][:, s]) == [0.25, 0.25] and list(b['P'][:, k]) == [1.0, 1.0]
    mask = np.ones(V3.PF, bool)
    mask[[s, k]] = False
    assert np.array_equal(a['P'][:, mask], b['P'][:, mask]) and np.array_equal(a['E'], b['E'])
    assert np.array_equal(a['canvas'], b['canvas']) and np.array_equal(a['local'], b['local'])


def test_stall_columns_in_run_mode_too():
    """--mode run's EpisodeState (run=True) is a whole-floor state (st.floor), so encode_row backfills the stall
    columns there as on whole floors: known 1 with a stall rule, 0 without one."""
    from isaac_bridge.tok_obs import EpisodeState
    assert EpisodeState(100, run=True, stall=120).floor and EpisodeState(100, floor=True, stall=120).floor
    assert not EpisodeState(100).floor
    src = (Path(VP.__file__).parent / 'tok_obs.py').read_text(encoding='utf-8')
    assert 'known = st.floor and st.stall > 0' in src and 'v3_backfill_stall(row' in src
