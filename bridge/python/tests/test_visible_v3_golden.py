"""visible-v3 golden frames (V2 / F3, package C, 2026-11-17; tests/fixtures/visible_v3, from abplus_probe_v3_golden.py
on AB+, 184 golden17; batch 5, 2026-11-18: the raw version field and header[6] rewritten offline, nothing else -- tools/golden_v3_diff.py): 19 real frames -- every RoomShape 1-12, the off-grid Homunculus part 228.10, a
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


def _identity_check(f, eng):
    """Every sent entity (E row) matched to exactly one engine entity of the same type, variant, subtype (or a masked
    subtype: identity_known 0) at its position in the canvas frame, no engine entity used twice; and every visible
    engine entity that is not a player nor a listed decorative effect sent. Returns (unmatched rows, missing ones)."""
    ox, oy = eng['g0'][0] - 20, eng['g0'][1] - 20
    ik = V3.col(V3.E_COLS, 'identity_known')
    used, unmatched = set(), []
    for r in f['E']:
        c = [k for k, e in enumerate(eng['entities']) if k not in used and int(r[0]) == e['type'] and
             int(r[1]) == e['variant'] and (int(r[2]) == e['subtype'] or not r[ik]) and
             abs(r[3] - np.float32(e['x'] - ox)) < 1e-3 and abs(r[4] - np.float32(e['y'] - oy)) < 1e-3]
        if len(c) == 1:
            used.add(c[0])
        else:
            unmatched.append((int(r[0]), int(r[1]), int(r[2]), len(c)))
    missing = [(e['type'], e['variant'], e['subtype']) for k, e in enumerate(eng['entities']) if k not in used and
               e['type'] != 1 and e['visible'] and not (e['type'] == 1000 and e['variant'] in VP.V3_DECORATIVE_WHY)]
    return unmatched, missing


@pytest.mark.parametrize('name', NAMES)
def test_entity_identity_unique_and_complete(name):
    """(2026-11-18, Codex round 2: the position check alone did not test identity, a unique match or a missing row)
    The E rows against the engine's own entity list read apart (engine.json): see _identity_check. The check bites: a
    frame with one E row taken out has a missing entity."""
    eng = json.loads((FIX / f'{name}.engine.json').read_text())
    f = _frame(name)
    assert _identity_check(f, eng) == ([], [])
    if len(f['E']):
        assert _identity_check(dict(f, E=f['E'][1:]), eng)[1]


@pytest.mark.parametrize('name', NAMES)
def test_the_native_encoder_frames(name):
    """<name>.native.bin: abp_v3_native.c's own frame of the same state (184 golden-b5, tree c2-9e941f3, the native
    encoder in the game), equal to the reference encoder's frame of the raw block with the stall columns 0 (the native
    side writes them 0, the worker backfills them). Taken again with the golden probe when the schema changes."""
    blob = json.loads((FIX / f'{name}.raw.json').read_text())
    assert (FIX / f'{name}.native.bin').read_bytes() == VP.observation_frame_v3(blob['raw'])


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


def _lean_v3(frame, terrain):
    """A lean payload (abplus_lean's layout) with the native v3 frame (flag 64), one player, and (terrain) a 15 x 9
    terrain block (flag 2): enough for encode_row."""
    import struct
    from isaac_bridge import abplus_lean as al
    b = bytearray(al._HEADER.pack(al.MAGIC, 1234, 1200, (2 if terrain else 0) | 64, 1, 0, 0))
    b += al._ROOM.pack(1, 0, 0, 0, 60, 2, 0, 0, 40.0, 120.0, 600.0, 400.0)
    b += al._TOTALS.pack(0.0, 0.0, 0.0, 0.0)
    pl = np.zeros(1, al.PLAYER)
    pl['hearts'], pl['x'], pl['y'] = 6, 300, 250
    b += pl.tobytes() + struct.pack('<H', 0)
    if terrain:
        cells = [[i, 40 + 40 * (i % 15), 120 + 40 * (i // 15), 0, 1, 1, 0, 0, 0, 0] for i in range(135)]
        t, g = json.dumps(dict(width=15, height=9, cells=cells)).encode(), b'[]'
        b += struct.pack('<II', 1, len(t)) + t + struct.pack('<I', len(g)) + g
    return bytes(b) + struct.pack('<I', len(frame)) + frame


@pytest.mark.parametrize('mode, known', [('run', 1.0), ('floor', 1.0), ('floor-no-stall', 0.0), ('room', 0.0)])
def test_stall_columns_by_episode_mode(mode, known, monkeypatch):
    """encode_row itself (2026-11-18: a behaviour test in place of the source check): --mode run's EpisodeState
    (run=True) is a whole-floor state, so the stall columns are backfilled there as on whole floors -- the share of the
    stall limit used (t - progress_t) / stall and known 1 with a stall rule; 0 / 0 without one or in a room episode."""
    from isaac_bridge.abplus_lean import LeanDecoder
    from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 2)   # (a v3 worker: a mode 0 one refuses v3 records, round 9)
    blob = json.loads((FIX / 'forgotten.raw.json').read_text())
    frame = VP.observation_frame_v3(blob['raw'])
    st = dict(run=EpisodeState(1000, run=True, stall=120), floor=EpisodeState(1000, floor=True, stall=120),
              room=EpisodeState(1000, stall=120), **{'floor-no-stall': EpisodeState(1000, floor=True)})[mode]
    dec, row = LeanDecoder(), np.zeros(1, ROW)
    for t in range(4):
        encode_row(dec.decode(_lean_v3(frame, terrain=t == 0)), st, row, t)
    f = VP.unpack_frame_v3(bytes(row['visible_frame'][0, :int(row['visible_bytes'][0])]))
    share = f['P'][:, V3.col(V3.P_COLS, 'stall_share')]
    assert list(f['P'][:, V3.col(V3.P_COLS, 'stall_known')]) == [known, known]
    assert np.allclose(share, (3 - st.progress_t) / 120 if known else 0.0)


def test_a_handed_back_record_reads_like_a_native_one():
    """A record the native side handed back (a cancelled lethal hit: only the raw block, flag 32) goes through the
    reference encoder in the worker (visible_producer.write_v3, as encode_row calls it): the same frame format as a
    native frame, read by the same decoder, with the stall columns backfilled the same way."""
    from types import SimpleNamespace
    from isaac_bridge.tok_obs import ROW
    blob = json.loads((FIX / 'forgotten.raw.json').read_text())
    native = (FIX / 'forgotten.native.bin').read_bytes()               # (the native encoder's own frame: stall 0)
    rows = []
    for o in (SimpleNamespace(visible=blob['raw'], visible_v3_frame=None),
              SimpleNamespace(visible=None, visible_v3_frame=native)):
        row = np.zeros(1, ROW)
        assert VP.write_v3(row, o)
        VP.v3_backfill_stall(row, 0.5, 1.0)
        rows.append(bytes(row['visible_frame'][0, :int(row['visible_bytes'][0])]))
        assert int(row['visible_schema'][0]) == V3.VERSION
    assert rows[0] == rows[1]
    f = VP.unpack_frame_v3(rows[0])
    assert list(f['P'][:, V3.col(V3.P_COLS, 'stall_share')]) == [0.5, 0.5]
    assert not VP.write_v3(np.zeros(1, ROW), SimpleNamespace(visible={'schema': 'isaac-visible-prlmens-g-v2'},
                                                             visible_v3_frame=None))

@pytest.mark.parametrize('name', NAMES)
def test_body_order_in_every_golden_frame(name):
    """Batch 4's body order on the real frames (the reference bytes; 184's v3_native_golden_cmp.py checks the native
    encoder's frames the same way): the forgotten / forgotten-soul frames hold the Forgotten / the Soul at row 0
    (position_known 1) and the other right after it."""
    assert VP.v3_body_order(_frame(name)) == []
