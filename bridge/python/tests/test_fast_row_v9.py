"""abp_row_encode version 9 (2026-11-20, the native ROW encoder for ROW schema 8) against encode_row, on synthetic
whole-floor episodes of lean payloads: the visible-v3 frame (flag 64), the rewind count (flag 128, a restore in the
middle), EV_NOHIT, --stall-progress events / pickups / items / first, items (inventory blocks, an active item used),
a death at the end. Every combination runs (1) FastRow in check mode (each fast record also through decode +
encode_row on copies, the ROW slots and the EpisodeStates compared) and (2) a pure encode_row reference over the same
payloads, whose records and done codes FastRow's must equal byte for byte.

Linux only: ISAAC_RL_TEST_TURBO_SO=<a libabp_turbo.so built from this tree's abp_turbo.c>; skipped otherwise."""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import test_obs_v2_smoke as V   # noqa: E402  (its world / terrain builders)

from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402

SO = os.environ.get('ISAAC_RL_TEST_TURBO_SO', '')
pytestmark = pytest.mark.skipif(not SO, reason='ISAAC_RL_TEST_TURBO_SO not set (Linux, a libabp_turbo.so of this tree)')
FIX = HERE / 'fixtures' / 'visible_v3'
N = 30
STALL = 6


def _raw():
    for p in (FIX / 'forgotten.raw.json', HERE / 'fixtures' / 'forgotten.raw.json'):
        if p.exists():
            return json.loads(p.read_text())['raw']
    hits = list(HERE.rglob('forgotten.raw.json'))
    return json.loads(hits[0].read_text())['raw']


def payload(k, hurt_at, rw_counts, inv_at, coin_at, active_used_at, dead_at, v3, frame):
    p, d, e, room = V.world(0)
    p['x'], p['y'] = 300.0 + 2.0 * k, 280.0 - 1.0 * k
    p['coins'] = 17 + (1 if k >= coin_at else 0)
    p['active'], p['active_charge'] = 45.0, (0.0 if k >= active_used_at else 6.0)
    p['dead'] = 1.0 if k == dead_at else 0.0
    flags = (1 if k >= 9 else 0)   # the room's clear flag from record 9
    flags |= 2 if k % 7 == 0 else 0   # the same terrain block now and then (the fast path takes it)
    flags |= 16 if k in inv_at else 0
    flags |= 64 if v3 else 0
    rw = rw_counts[k]
    flags |= 128 if rw is not None else 0
    damage_taken = 1.0 if k >= hurt_at else 0.0
    monsters_hp = max(30.0 - 4.0 * k, 0.0)
    out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
    out += struct.pack('<iiiiiiiiffff', *room)
    out += struct.pack('<dddd', damage_taken, monsters_hp, monsters_hp, 1.0)
    out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
    rec = np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)
    out += rec[:, :148].tobytes()
    if flags & 2:
        tj, gj = V.terrain_json()
        out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
    if flags & 16:   # an inventory block: n, layout, max charges, trinkets, pill, effect, card, curses; then the items
        items = [(1, 1), (45, 1)] + ([(7, 1)] if k > 0 else [])
        out += AL.INV_HEADER.pack(len(items), 1, 6, 0, 0, 3 if k > 0 else 0, 0, 0, 0)
        arr = np.zeros(len(items), AL.INV_ITEM)
        arr['id'], arr['count'] = [i for i, _ in items], [c for _, c in items]
        out += arr.tobytes()
    if flags & 64:
        out += struct.pack('<I', len(frame)) + frame
    if flags & 128:
        out += struct.pack('<i', rw)
    return out


def state(stall_mode, items, rewind):
    pk, it = TO.stall_flags(stall_mode)
    return TO.EpisodeState(10 ** 6, floor=True, stall=STALL, items=items, stall_pickups=pk, stall_items=it,
                           stall_first=TO.stall_first(stall_mode), rewind=rewind)


def scenario(rewind_case, hurt):
    rw = [None] * N
    if rewind_case == 'on':
        for k in range(N):   # the bridge's count: at the first records, then whenever it changed (a restore at 14)
            rw[k] = 0 if k < 14 else 1
            if k not in (0, 1, 2, 14, 20, 21):
                rw[k] = None
    elif rewind_case == 'twice':   # two restores in a row (14 and 15)
        for k in range(N):
            rw[k] = 0 if k < 14 else 1 if k == 14 else 2
    return dict(hurt_at=5 if hurt else 10 ** 6, rw_counts=rw, inv_at={0, 17}, coin_at=11, active_used_at=12,
                dead_at=N - 1)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(TO, 'FAST_ROW', True)
    monkeypatch.setattr(TO, '_LIBS', {})
    fn = TO.fast_row_function(SO)
    assert fn is not None, 'the library is not of version 9 / this layout'
    return fn


@pytest.mark.parametrize('v3', [0, 2])
@pytest.mark.parametrize('stall_mode', ['events', 'pickups', 'items', 'first'])
@pytest.mark.parametrize('items', [False, True])
@pytest.mark.parametrize('rewind_case', ['off', 'on', 'twice'])
@pytest.mark.parametrize('nohit,hurt', [(False, False), (True, False), (True, True)])
def test_fast_row_v9_equals_encode_row(fast, monkeypatch, v3, stall_mode, items, rewind_case, nohit, hurt):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', v3)
    monkeypatch.setenv('ISAAC_RL_NOHIT', '1' if nohit else '0')
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', True)
    for k_ in list(TO.FAST_STATS):
        TO.FAST_STATS[k_] = 0
    frame = VP.observation_frame_v3(_raw()) if v3 else None
    sc = scenario(rewind_case, hurt)
    rewind = rewind_case != 'off'
    pls = [payload(k, v3=bool(v3), frame=frame, **sc) for k in range(N)]
    # the reference: decode + encode_row on every record
    dec_r, st_r = AL.LeanDecoder(), state(stall_mode, items, rewind)
    ref = np.zeros(N, TO.ROW)
    ref_done = []
    for k in range(N):
        ref_done.append(TO.encode_row(dec_r.decode(pls[k]), st_r, ref[k:k + 1], k))
    ref['first'] = [1] + [0] * (N - 1)
    # FastRow (two slots used in turn, as a worker does; copied out after each record)
    dec, st = AL.LeanDecoder(), state(stall_mode, items, rewind)
    fr = TO.FastRow(fast, st, dec, 0, 0, 0)
    slots = np.zeros(2, TO.ROW)
    got = np.zeros(N, TO.ROW)
    done = []
    for k in range(N):
        s = slots[k & 1:(k & 1) + 1]
        item = dec.decode(pls[k]) if k == 0 else pls[k]
        done.append(fr.encode(item, s, s.ctypes.data, k))
        got[k] = s[0]
    fr._to_st()
    tag = f'fast {fr.fast} slow {fr.slow} reasons {fr.reasons} stats {TO.FAST_STATS}'
    assert TO.FAST_STATS['check_mismatch'] == 0, tag
    assert fr.fast >= N // 2, tag
    for k in range(N):
        bad = [nm for nm in TO.ROW.names if nm != 'visible_frame' and got[nm][k].tobytes() != ref[nm][k].tobytes()]
        nb = int(ref['visible_bytes'][k])
        if got['visible_frame'][k][:nb].tobytes() != ref['visible_frame'][k][:nb].tobytes():
            bad.append('visible_frame')
        assert not bad, f'record {k}: {bad} ({tag})'
    assert done == ref_done, f'{done} vs {ref_done}'
    diff = TO._state_diff(st, st_r)
    assert not diff, diff
    if rewind:
        assert fr.reasons.get(-11, 0) >= 1, tag   # the restore went through encode_row
    if v3:
        assert TO.FAST_STATS['fast'] > 0 and int(got['visible_schema'][N // 2]) == VP.V3.VERSION


def test_fast_row_v9_hands_back_what_encode_row_refuses(fast, monkeypatch):
    """A worker not configured for v3 and a v3 frame; a v3 worker and no frame; a frame with a bad header: the C side
    hands the record back and encode_row raises as it always did."""
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', False)
    frame = VP.observation_frame_v3(_raw())
    bad = bytearray(frame)
    bad[0:4] = bytes(4)
    sc = scenario('off', False)
    for m, fr_bytes, exc in ((0, frame, ValueError), (2, None, VP.VisibleV3Fatal), (2, bytes(bad), VP.VisibleV3Fatal)):
        monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', m)
        dec, st = AL.LeanDecoder(), state('events', False, False)
        fr = TO.FastRow(fast, st, dec, 0, 0, 0)
        row = np.zeros(1, TO.ROW)
        first = payload(0, v3=bool(m), frame=frame, **sc)
        fr.encode(dec.decode(first), row, row.ctypes.data, 0)
        p1 = payload(1, v3=fr_bytes is not None, frame=fr_bytes, **sc)
        with pytest.raises(exc):
            fr.encode(p1, row, row.ctypes.data, 1)
        assert fr.fast == 0 and fr.reasons.get(-13, 0) == 1, fr.reasons


def test_fast_row_v9_trailing_bytes_handed_back(fast, monkeypatch):
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', False)
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 0)
    sc = scenario('off', False)
    dec, st = AL.LeanDecoder(), state('events', False, False)
    fr = TO.FastRow(fast, st, dec, 0, 0, 0)
    row = np.zeros(1, TO.ROW)
    fr.encode(dec.decode(payload(0, v3=False, frame=None, **sc)), row, row.ctypes.data, 0)
    with pytest.raises(ValueError, match='trailing'):
        fr.encode(payload(1, v3=False, frame=None, **sc) + b'\0', row, row.ctypes.data, 1)
    assert fr.fast == 0 and fr.reasons.get(-1, 0) == 1
