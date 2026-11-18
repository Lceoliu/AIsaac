"""visible-v3 in the worker and the acceptance scripts (2026-11-18, Codex round 2, the must-fix items): a worker
configured for v3 writes a v3 frame for every record or stops; the bridge's mode is checked after connecting; the C ROW
fast path never takes a payload with a v3 frame or raw block (by the payload's flags, not the worker's environment);
the reference encoder refuses a short grid like the native one; the probes / tools / deploy script fail on a real
difference."""
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from isaac_bridge import abplus_lean as al
from isaac_bridge import schema_v3 as V3
from isaac_bridge import visible_producer as VP
from isaac_bridge import tok_obs
from isaac_bridge.tok_obs import ROW, EpisodeState, FastRow, write_visible

HERE = Path(__file__).resolve().parent
PY = HERE.parent
FIX = HERE / 'fixtures' / 'visible_v3'
BM1 = PY.parent / 'ops' / 'bm1'
RAW = json.loads((FIX / 'forgotten.raw.json').read_text())['raw']


def lean(raw=None, frame=None, v2=None):
    """A lean payload (abplus_lean's layout): one player, no entities; flag 32 with raw (a v3 raw block) or v2 (a v2
    visible block), flag 64 with frame (a native v3 frame)."""
    flags = (32 if raw is not None or v2 is not None else 0) | (64 if frame is not None else 0)
    b = bytearray(al._HEADER.pack(al.MAGIC, 1234, 1200, flags, 1, 0, 0))
    b += al._ROOM.pack(*([0] * 8), 0.0, 0.0, 600.0, 400.0)
    b += al._TOTALS.pack(0.0, 0.0, 0.0, 0.0)
    pl = np.zeros(1, al.PLAYER)
    pl['hearts'] = 6
    b += pl.tobytes() + struct.pack('<H', 0)
    block = raw if raw is not None else v2
    if block is not None:
        text = json.dumps(block).encode()
        b += struct.pack('<I', len(text)) + text
    if frame is not None:
        b += struct.pack('<I', len(frame)) + frame
    return bytes(b)


@pytest.fixture
def mode(monkeypatch):
    def set_(m):
        monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', m)
    return set_


# ------------------------------------------------------------------ item 2: a v3 worker writes v3 or stops
def test_configured_mode_is_0_to_3():
    assert [VP.configured_mode(v) for v in ('', ' ', '0', '1', '2', '3', None)] == [0, 0, 0, 1, 2, 3, 0]
    for bad in ('4', '-1', 'on', '2.0', 'true'):
        with pytest.raises(ValueError, match='0-3'):
            VP.configured_mode(bad)


@pytest.mark.parametrize('m', [1, 2, 3])
def test_a_v3_worker_refuses_a_record_without_a_v3_frame(mode, m):
    mode(m)
    v2 = {'schema': 'isaac-visible-prlmens-g-v2', 'bodies': [], 'inventory': [], 'capabilities': 1}
    for payload, why in ((lean(v2=v2), "schema 'isaac-visible-prlmens-g-v2'"), (lean(), 'no visible block')):
        with pytest.raises(ValueError, match=why):
            write_visible(al.LeanDecoder().decode(payload), np.zeros(1, ROW))
    for payload in (lean(raw=RAW), lean(frame=VP.observation_frame_v3(RAW))):   # a v3 record: written
        row = np.zeros(1, ROW)
        assert write_visible(al.LeanDecoder().decode(payload), row)
        assert int(row['visible_schema'][0]) == V3.VERSION and int(row['visible_bytes'][0]) > 0
    src = Path(tok_obs.__file__).read_text(encoding='utf-8')   # (encode_row's own first step)
    assert 'v3 = write_visible(o, row)' in src[src.index('def encode_row('):]


def test_a_v2_worker_still_takes_v2_records(mode):
    mode(0)
    row = np.zeros(1, ROW)
    assert not write_visible(al.LeanDecoder().decode(lean()), row)
    assert int(row['visible_schema'][0]) == 0


def test_the_bridge_mode_is_checked(mode):
    ok = lambda m, native=True: dict(type='ok', cmd='visible_v3', mode=m, enabled=m != 0, native_available=native)
    for m in range(4):
        assert VP.check_bridge_mode(ok(m), m) == m
    with pytest.raises(ValueError, match='bridge runs mode 0, this worker is configured for mode 2'):
        VP.check_bridge_mode(ok(0), 2)
    with pytest.raises(ValueError, match='bridge runs mode 2, this worker is configured for mode 0'):
        VP.check_bridge_mode(ok(2), 0)
    for m in (2, 3):
        with pytest.raises(ValueError, match='native encoder'):
            VP.check_bridge_mode(ok(m, native=False), m)
    assert VP.check_bridge_mode(ok(1, native=False), 1) == 1   # (mode 1 needs no native encoder)
    unknown = dict(type='error', msg='unknown cmd')   # a bridge without the command: mode 0
    assert VP.check_bridge_mode(unknown, 0) == 0
    with pytest.raises(ValueError):
        VP.check_bridge_mode(unknown, 1)
    with pytest.raises(ValueError, match='did not answer'):
        VP.check_bridge_mode(dict(type='error', cmd='visible_v3', msg='visible_v3: mode must be 0-3, got 4'), 0)
    mode(3)   # (the default: the worker's configured mode)
    assert VP.check_bridge_mode(ok(3)) == 3


def test_needs_decode_by_the_payload():
    frame = VP.observation_frame_v3(RAW)
    assert VP.needs_decode(lean(frame=frame)) and VP.needs_decode(lean(raw=RAW, frame=frame))
    assert VP.needs_decode(lean(raw=RAW))
    assert not VP.needs_decode(lean()) and not VP.needs_decode(lean(v2={'schema': 'isaac-visible-prlmens-g-v2'}))


@pytest.mark.parametrize('m', [0, 2])
def test_fastrow_never_lets_the_c_path_take_a_v3_payload(mode, monkeypatch, m):
    """FastRow.encode on a raw payload with a v3 frame (flag 64) or raw block decodes it and hands the LeanObs to
    encode_row, whatever the worker's own mode (before: decided by ISAAC_RL_VISIBLE_V3 at import, so a v2-configured
    worker let abp_row_encode take the step and drop the frame); a payload without one still goes to the C path."""
    mode(m)
    calls, slow = [], []

    def fn(*a):
        calls.append(a)
        return 0
    monkeypatch.setattr(tok_obs, 'encode_row', lambda o, st, row, t: slow.append(o) or 0)
    fr = FastRow(fn, EpisodeState(1000), al.LeanDecoder())
    fr._ptrs = (1, 1, 1)   # (a grid as if the C path were set up)
    frame = VP.observation_frame_v3(RAW)
    row = np.zeros(1, ROW)
    for payload in (lean(raw=RAW, frame=frame), lean(frame=frame), lean(raw=RAW)):
        fr.encode(payload, row, row.ctypes.data, 1)
    assert not calls and fr.slow == 3 and len(slow) == 3
    assert slow[0].visible_v3_frame == frame and slow[1].visible_v3_frame == frame and slow[2].visible == RAW
    fr._ptrs = (1, 1, 1)   # (a slow record resets them: _from_st)
    fr.encode(lean(), row, row.ctypes.data, 2)   # no v3 in it: the C path
    assert len(calls) == 1 and fr.fast == 1


def test_the_c_row_path_hands_a_v3_frame_back():
    src = (PY.parent / 'engine' / 'abp_turbo.c').read_text(encoding='utf-8')
    at = src.index('int abp_row_encode(const unsigned char *pl')
    assert 'if (flags & 64) return -9;' in src[at:at + 600]


# ------------------------------------------------------------------ item 4: a short grid
def test_a_short_grid_is_refused():
    raw = json.loads(json.dumps(RAW))
    n = int(raw['gw']) * int(raw['gh'])
    assert len(raw['cells']) >= n
    raw['cells'] = raw['cells'][:n - 1]
    with pytest.raises(ValueError, match=f'grid size: {n - 1} cells'):
        VP.observation_frame_v3(raw)
    raw['cells'] = RAW['cells'][:n]
    VP.observation_frame_v3(raw)
