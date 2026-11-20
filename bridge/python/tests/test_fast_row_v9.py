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


def payload(k, hurt_at, rw_counts, inv_at, coin_at, active_used_at, dead_at, v3, frame, stage_at=10 ** 6,
            raw_block=None):
    p, d, e, room = V.world(0)
    if k >= stage_at:   # (the next floor: the same room index, another stage -- a run's EV_EXIT, a floor's end)
        room = room[:5] + (room[5] + 1,) + room[6:]
    p['x'], p['y'] = 300.0 + 2.0 * k, 280.0 - 1.0 * k
    p['coins'] = 17 + (1 if k >= coin_at else 0)
    p['active'], p['active_charge'] = 45.0, (0.0 if k >= active_used_at else 6.0)
    p['dead'] = 1.0 if k == dead_at else 0.0
    flags = (1 if k >= 9 else 0)   # the room's clear flag from record 9
    flags |= 2 if k % 7 == 0 else 0   # the same terrain block now and then (the fast path takes it)
    flags |= 16 if k in inv_at else 0
    flags |= 64 if v3 else 0
    flags |= 32 if raw_block is not None else 0
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
    if flags & 32:
        text = json.dumps(raw_block).encode()
        out += struct.pack('<I', len(text)) + text
    if flags & 64:
        out += struct.pack('<I', len(frame)) + frame
    if flags & 128:
        out += struct.pack('<i', rw)
    return out


def state(stall_mode, items, rewind, run=False):
    pk, it = TO.stall_flags(stall_mode)
    return TO.EpisodeState(10 ** 6, floor=True, run=run, stall=STALL, items=items, stall_pickups=pk, stall_items=it,
                           stall_first=TO.stall_first(stall_mode), rewind=rewind)


def scenario(rewind_case, hurt, stage_at=10 ** 6):
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
                dead_at=N - 1, stage_at=stage_at)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(TO, 'FAST_ROW', True)
    monkeypatch.setattr(TO, '_LIBS', {})
    fn = TO.fast_row_function(SO)
    assert fn is not None, 'the library is not of version 9 / this layout'
    return fn


@pytest.mark.parametrize('run', [False, True])
@pytest.mark.parametrize('v3', [0, 2])
@pytest.mark.parametrize('stall_mode', ['events', 'pickups', 'items', 'first'])
@pytest.mark.parametrize('items', [False, True])
@pytest.mark.parametrize('rewind_case', ['off', 'on', 'twice'])
@pytest.mark.parametrize('nohit,hurt', [(False, False), (True, False), (True, True)])
def test_fast_row_v9_equals_encode_row(fast, monkeypatch, tmp_path, run, v3, stall_mode, items, rewind_case, nohit,
                                       hurt):
    monkeypatch.setenv('ISAAC_RL_FAST_ROW_CHECK_DIR', str(tmp_path))   # (check mode: every process's counts)
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', v3)
    monkeypatch.setenv('ISAAC_RL_NOHIT', '1' if nohit else '0')
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', True)
    for k_ in list(TO.FAST_STATS):
        TO.FAST_STATS[k_] = 0
    frame = VP.observation_frame_v3(_raw()) if v3 else None
    sc = scenario(rewind_case, hurt, stage_at=24)   # (the next floor at 24: run -- EV_EXIT, handed back; a floor's end)
    rewind = rewind_case != 'off'
    pls = [payload(k, v3=bool(v3), frame=frame, **sc) for k in range(N)]
    # the reference: decode + encode_row on every record
    dec_r, st_r = AL.LeanDecoder(), state(stall_mode, items, rewind, run)
    ref = np.zeros(N, TO.ROW)
    ref_done = []
    for k in range(N):
        ref_done.append(TO.encode_row(dec_r.decode(pls[k]), st_r, ref[k:k + 1], k))
    ref['first'] = [1] + [0] * (N - 1)
    # FastRow (two slots used in turn, as a worker does; copied out after each record)
    dec, st = AL.LeanDecoder(), state(stall_mode, items, rewind, run)
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
    if run:
        assert fr.reasons.get(-8, 0) == 1 and int(got['events'][24]) & TO.EV_EXIT, tag   # the next floor: encode_row's
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


def _seq(fast, monkeypatch, n, env_vis, mode, raw_block=None, frame=None):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', mode)
    if env_vis:
        monkeypatch.setenv('ISAAC_NO_VISIBLE', '1')
    else:
        monkeypatch.delenv('ISAAC_NO_VISIBLE', raising=False)
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', False)
    sc = scenario('off', False)
    pls = [payload(k, v3=frame is not None, frame=frame, raw_block=raw_block, **sc) for k in range(n)]
    dec_r, st_r = AL.LeanDecoder(), state('events', False, False)
    ref = np.zeros(n, TO.ROW)
    for k in range(n):
        TO.encode_row(dec_r.decode(pls[k]), st_r, ref[k:k + 1], k)
    ref['first'] = [1] + [0] * (n - 1)
    dec, st = AL.LeanDecoder(), state('events', False, False)
    fr = TO.FastRow(fast, st, dec, 0, 0, 0)
    got = np.zeros(n, TO.ROW)
    for k in range(n):
        s_ = got[k:k + 1]
        fr.encode(dec.decode(pls[k]) if k == 0 else pls[k], s_, s_.ctypes.data, k)
    bad = [nm for nm in TO.ROW.names if got[nm].tobytes() != ref[nm].tobytes()]
    return fr, bad, got


def test_fast_row_v9_visible_mode_1_raw_blocks_go_to_encode_row(fast, monkeypatch):
    """(A1's review, B) a mode 1 worker: every record carries a raw v3 block (flag 32): the C path hands each back
    (-13), the reference encoder writes the frame; the records equal encode_row's."""
    fr, bad, got = _seq(fast, monkeypatch, 8, False, 1, raw_block=_raw())
    assert not bad and fr.fast == 0 and fr.reasons.get(-13, 0) == 7, (bad, fr.reasons)
    assert int(got['visible_schema'][3]) == VP.V3.VERSION


def test_fast_row_v9_no_visible_writes_schema_0(fast, monkeypatch):
    """(A1's review, B) ISAAC_NO_VISIBLE=1 (--obs-visible 0) in a v3 worker: the frames are not written (RC_VIS bit 1),
    visible_schema 0, by the C path as by encode_row."""
    fr, bad, got = _seq(fast, monkeypatch, 8, True, 2, frame=VP.observation_frame_v3(_raw()))
    assert not bad and fr.fast == 7, (bad, fr.fast, fr.reasons)
    assert int(got['visible_schema'][3]) == 0 and int(got['visible_bytes'][3]) == 0


# ---- (Codex round 23-1) every hand-back writes nothing: the ROW, ctx and the EpisodeState's arrays pre-filled with
# non-zero bytes, each code provoked on a record after a real first one, all three compared after the call
_PV = 96   # player 0's doubles (abplus_lean: header 16, room 48, totals 32)


def _patch(pl, at, fmt, value):
    b = bytearray(pl)
    struct.pack_into(fmt, b, at, value)
    return bytes(b)


def _prefilled_call(fr, pl, row, t=1):
    raw = row.view(np.uint8).reshape(-1)
    raw[:] = 0x5A
    for a in (fr.st.inv_arr, fr.st.pitem, fr.st.pinv):
        a.view(np.uint8)[:] = 0x33
    fr.ctx[TO.RC_T], fr.ctx[TO.RC_FIRST] = t, 0
    ctx0, raw0 = fr.ctx.copy(), raw.copy()
    ext0 = [a.copy() for a in (fr.st.inv_arr, fr.st.pitem, fr.st.pinv)]
    g, pd, m = fr._ptrs
    code = fr.fn(pl, len(pl), row.ctypes.data, fr._ctx_p, g, pd, m, fr._offs_p, fr._refs_p, fr._ext_p)
    same = (np.array_equal(raw, raw0), fr.ctx.tobytes() == ctx0.tobytes(),
            all(a.tobytes() == b.tobytes() for a, b in zip(ext0, (fr.st.inv_arr, fr.st.pitem, fr.st.pinv))))
    return code, same


def _base(fast, monkeypatch, run=False, rewind=True, items=True):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 0)
    monkeypatch.setattr(TO, 'FAST_ROW_CHECKING', False)
    sc = scenario('twice' if rewind else 'off', False)
    pls = [payload(k, v3=False, frame=None, **sc) for k in range(3)]
    dec, st = AL.LeanDecoder(), state('events', items, rewind, run)
    fr = TO.FastRow(fast, st, dec, 0, 0, 0)
    row = np.zeros(1, TO.ROW)
    fr.encode(dec.decode(pls[0]), row, row.ctypes.data, 0)
    assert fr._ptrs[0], 'the first record did not set the C path up'
    return fr, pls[1], row


def test_every_hand_back_writes_nothing(fast, monkeypatch):
    fr, pl, row = _base(fast, monkeypatch)
    n_doors = pl[14]
    ents_at = _PV + 8 * 39 * pl[13] + 16 * n_doors
    cases = [
        (-1, _patch(pl, 0, '<I', 0)),                                   # another magic
        (-1, pl + b'\0'),                                              # a trailing byte
        (-2, _patch(pl, 12, '<B', pl[12] | 8)),                         # a map block flag
        (-3, _patch(pl, ents_at, '<H', 65)),                            # 65 old entities
        (-4, _patch(pl, 32, '<i', struct.unpack_from('<i', pl, 32)[0] + 1)),   # another room
        (-5, _patch(pl, _PV + 8 * 39 + 4, '<f', 1.5)),                  # a door record changed
        (-11, _patch(pl, len(pl) - 4, '<i', 7)),                        # the restore count went up
        (-13, _patch(pl, 12, '<B', pl[12] | 32)),                       # a visible block flag
        (-14, _patch(pl, _PV + 8 * 24, '<d', float('nan'))),            # active: NaN (items)
    ]
    for name, i in (('x', 1), ('soul', 8), ('black', 9), ('bombs', 15), ('weapons', 37), ('ptype', 38)):
        for v in (float('nan'), float('inf'), -float('inf')):
            cases.append((-15, _patch(pl, _PV + 8 * i, '<d', v)))
    cases.append((-15, _patch(pl, _PV + 8 * 1, '<d', 1e300)))           # a finite x out of a long's range
    for want, p_ in cases:
        code, same = _prefilled_call(fr, p_, row)
        assert code == want and all(same), (want, code, same)
    for want, edit in ((-6, lambda c: c.__setitem__(TO.RC_STAGE0, np.nan)), (-12, lambda c: c.__setitem__(TO.RC_CK_NONE, 1))):
        keep = fr.ctx.copy()
        edit(fr.ctx)
        code, same = _prefilled_call(fr, pl, row)
        fr.ctx[:] = keep
        assert code == want and all(same), (want, code, same)
    frr, plr, rowr = _base(fast, monkeypatch, run=True)
    code, same = _prefilled_call(frr, _patch(plr, 36, '<i', struct.unpack_from('<i', plr, 36)[0] + 1), rowr)
    assert code == -8 and all(same), (code, same)


@pytest.mark.parametrize('field,i', [('x', 1), ('soul', 8), ('black', 9), ('bombs', 15), ('weapons', 37), ('ptype', 38)])
@pytest.mark.parametrize('v', [float('nan'), float('inf'), 1e300])
def test_bad_numbers_go_to_encode_row(fast, monkeypatch, field, i, v):
    """(Codex round 23-1) a value encode_row turns into an integer, not safe for the C side: handed back (-15);
    FastRow then raises what encode_row raises, or (a large finite value) writes encode_row's record."""
    fr, pl, row = _base(fast, monkeypatch, rewind=False, items=False)
    bad = _patch(pl, _PV + 8 * i, '<d', v)
    dec_r, st_r = AL.LeanDecoder(), state('events', False, False)
    ref = np.zeros(1, TO.ROW)
    TO.encode_row(dec_r.decode(payload(0, v3=False, frame=None, **scenario('off', False))), st_r, ref, 0)
    try:
        TO.encode_row(dec_r.decode(bad), st_r, ref, 1)
        ref['first'] = 0
        want_exc = None
    except Exception as exc:   # noqa: BLE001
        want_exc = type(exc)
    if want_exc is not None:
        with pytest.raises(want_exc):
            fr.encode(bad, row, row.ctypes.data, 1)
    else:
        fr.encode(bad, row, row.ctypes.data, 1)
        assert row.tobytes() == ref.tobytes()
    assert fr.reasons.get(-15) == 1
