"""(Codex round 23, items 2 and 3) FastRow's failure paths, with stand-ins for abp_row_encode (no library needed):
check mode (ISAAC_RL_FAST_ROW=2) stops on a bad record -- a "success" that did not write the ROW, a record the C side
took and encode_row refuses, a hand-back that wrote, an unknown hand-back code -- with FastRowFatal (always fatal: the
worker exits V3_FATAL_EXIT, the trainer fails), never trains on it; the switch on requires the library (missing /
another layout: FastRowFatal, also at the trainer's start); a forked child's last-record mismatch reaches its parent
(exit V3_FATAL_EXIT, its final counts written before os._exit); the fr_* counters never iterate a dict another thread
grows (round 23-3: "dictionary changed size during iteration" ended a floor worker)."""
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from isaac_bridge import abplus_lean as al   # noqa: E402
from isaac_bridge import tok_obs   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_obs import ROW, EpisodeState, FastRow, FastRowFatal   # noqa: E402


def lean(x=300.0, k=1):
    """A lean payload of test_obs_v2_smoke's world (player 0 at x; the terrain block at k = 0), no visible block."""
    import test_obs_v2_smoke as V
    p, d, e, room = V.world(0)
    p['x'] = x
    flags = 2 if k == 0 else 0
    out = struct.pack('<IIIBBBB', al.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
    out += struct.pack('<iiiiiiiiffff', *room)
    out += struct.pack('<dddd', 0.0, 30.0, 30.0, 1.0)
    out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
    out += np.ascontiguousarray(e).view(np.uint8).reshape(len(e), al.ENTITY.itemsize)[:, :148].tobytes()
    if flags & 2:
        tj, gj = V.terrain_json()
        out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
    return out


@pytest.fixture
def check_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 0)
    monkeypatch.setattr(tok_obs, 'FAST_ROW', True)
    monkeypatch.setattr(tok_obs, 'FAST_ROW_CHECKING', True)
    monkeypatch.setenv('ISAAC_RL_FAST_ROW_CHECK_DIR', str(tmp_path))
    tok_obs.child_start('worker')
    return tmp_path


def _fr(fn):
    st = EpisodeState(1000)
    fr = FastRow(fn, st, al.LeanDecoder())
    row = np.zeros(1, ROW)
    fr.encode(fr.decoder.decode(lean(k=0)), row, row.ctypes.data, 0)   # (the first record: encode_row's)
    fr._ptrs = (1, 1, 1)   # (as if the grid were set up: the stand-in is offered every record)
    return fr, row


def test_a_success_that_wrote_nothing_stops_the_run(check_mode):
    fr, row = _fr(lambda *a: 0)
    with pytest.raises(FastRowFatal, match='FastRow check: record 1'):
        fr.encode(lean(302.0), row, row.ctypes.data, 1)
    assert tok_obs.FAST_STATS['check_mismatch'] == 1
    rep = json.loads((check_mode / f'fastrow-{os.getpid()}.json').read_text())
    assert rep['check_mismatch'] == 1 and rep['role'] == 'worker'
    assert list(check_mode.glob('fastrow-mismatch-*.npz'))


def test_a_record_encode_row_refuses_stops_the_run(check_mode):
    """The C side "took" a record whose x is NaN; encode_row raises on it (round(): the patch's cell)."""
    fr, row = _fr(lambda *a: 0)
    with pytest.raises(FastRowFatal, match='encode_row raised: ValueError'):
        fr.encode(lean(float('nan')), row, row.ctypes.data, 1)


def test_a_hand_back_that_wrote_stops_the_run(check_mode):
    def fn(pl, n, addr, *rest):
        ROW_view = (np.ctypeslib.as_array((np.ctypeslib.ctypes.c_uint8 * ROW.itemsize).from_address(addr)))
        ROW_view[0] ^= 0xFF
        return -2
    fr, row = _fr(fn)
    with pytest.raises(FastRowFatal, match=r'handed record 1 back \(-2\) after writing ROW'):
        fr.encode(lean(302.0), row, row.ctypes.data, 1)


def test_an_unknown_hand_back_code_is_fatal(check_mode, monkeypatch):
    for checking in (True, False):
        monkeypatch.setattr(tok_obs, 'FAST_ROW_CHECKING', checking)
        fr, row = _fr(lambda *a: -99)
        with pytest.raises(FastRowFatal, match='returned -99'):
            fr.encode(lean(302.0), row, row.ctypes.data, 1)


def test_the_switch_on_requires_the_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tok_obs, 'FAST_ROW', True)
    monkeypatch.setattr(tok_obs, 'FAST_ROW_CHECKING', False)
    monkeypatch.setattr(tok_obs, '_LIBS', {})
    with pytest.raises(FastRowFatal, match='no libabp_turbo.so'):
        tok_obs.fast_row_function(str(tmp_path / 'missing.so'))
    bad = tmp_path / 'not_a_library.so'
    bad.write_bytes(b'not an ELF file')
    with pytest.raises(FastRowFatal):
        tok_obs.fast_row_function(str(bad))
    monkeypatch.setattr(tok_obs, 'FAST_ROW_CHECKING', True)
    monkeypatch.delenv('ISAAC_RL_FAST_ROW_CHECK_DIR', raising=False)
    with pytest.raises(FastRowFatal, match='needs ISAAC_RL_FAST_ROW_CHECK_DIR'):
        tok_obs.fast_row_function(str(bad))
    monkeypatch.setattr(tok_obs, 'FAST_ROW', False)
    assert tok_obs.fast_row_function(str(tmp_path / 'missing.so')) is None   # (off: as before)


def test_fastrow_fatal_is_always_fatal():
    assert issubclass(FastRowFatal, VP.VisibleV3Fatal) and FastRowFatal.always_fatal
    from isaac_bridge.tok_floor import ChildFatal, RewindFatal
    assert ChildFatal.always_fatal and RewindFatal.always_fatal


def test_counters_have_fixed_keys_and_survive_a_writer_thread(check_mode):
    """Round 23-3's interleaving: another thread counting (teacher v2, a cf thread) while the worker's main thread
    reads fast_counts() and writes a report. Fixed keys: nothing is added, nothing iterated mid-change."""
    with pytest.raises(KeyError):
        tok_obs._bump('slow_-99')   # (no key is ever added)
    stop, errors = [False], []

    def writer():
        keys = [k for k in tok_obs._STAT_KEYS]
        i = 0
        while not stop[0]:
            tok_obs._bump(keys[i % len(keys)])
            i += 1

    th = threading.Thread(target=writer)
    th.start()
    t0 = time.time()
    try:
        while time.time() - t0 < 0.5:
            try:
                tok_obs.fast_counts()
                tok_obs._check_report(True)
            except Exception as exc:   # noqa: BLE001
                errors.append(repr(exc))
                break
    finally:
        stop[0] = True
        th.join()
    assert not errors, errors
    assert set(tok_obs.FAST_STATS) == set(tok_obs._STAT_KEYS)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='os.fork: Linux')
def test_a_forked_childs_last_record_mismatch_reaches_the_parent(check_mode):
    """The cf / search child pattern of tok_floor: child_start, the work, the final counts before os._exit (no
    atexit); a FastRowFatal in it -> exit V3_FATAL_EXIT (the parent raises ChildFatal), its final report says so."""
    fr, row = _fr(lambda *a: 0)
    tok_obs._bump('children_forked')
    pid = os.fork()
    if pid == 0:
        code = 0
        tok_obs.child_start('cf-child')
        try:
            fr.encode(lean(302.0), row, row.ctypes.data, 1)   # (its last record: a mismatch)
        except VP.VisibleV3Fatal as exc_:
            code = VP.V3_FATAL_EXIT if getattr(exc_, 'always_fatal', False) else 1
        except BaseException:
            code = 1
        finally:
            try:
                tok_obs.fast_row_final()
            except BaseException:
                code = VP.V3_FATAL_EXIT
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == VP.V3_FATAL_EXIT
    rep = json.loads((check_mode / f'fastrow-{pid}.json').read_text())
    assert rep['final'] and rep['role'] == 'cf-child' and rep['check_mismatch'] == 1 and rep['ppid'] == os.getpid()
    assert tok_obs.FAST_STATS['children_forked'] == 1
