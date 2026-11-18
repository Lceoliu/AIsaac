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


def test_the_connect_check_with_the_clients_recv():
    """abplus.check_visible_v3 (AbplusTrainingEnv.connect's check) with a client whose _recv raises on an error reply,
    as the real one does: an old bridge's 'unknown cmd' is mode 0 (passes for a mode 0 client, refused otherwise); a
    mismatch is a BridgeError naming both modes."""
    from isaac_bridge.abplus import check_visible_v3
    from isaac_bridge.env import BridgeError

    class Client:
        def __init__(self, reply):
            self.reply, self.sent = reply, []

        def _send(self, msg):
            self.sent.append(msg)

        def _recv(self):
            if self.reply.get('type') == 'error':
                raise BridgeError(self.reply.get('msg', 'error'))
            return self.reply
    c = Client(dict(type='ok', cmd='visible_v3', mode=2, native_available=True))
    assert check_visible_v3(c, 2) == 2 and c.sent == [{'cmd': 'visible_v3'}]
    with pytest.raises(BridgeError, match='bridge runs mode 2, this worker is configured for mode 0'):
        check_visible_v3(c, 0)
    old = Client(dict(type='error', msg='unknown cmd'))
    assert check_visible_v3(old, 0) == 0
    with pytest.raises(BridgeError, match='bridge runs mode 0, this worker is configured for mode 1'):
        check_visible_v3(old, 1)
    src = Path(al.__file__).with_name('abplus.py').read_text(encoding='utf-8')
    assert 'check_visible_v3(self)' in src[src.index('    def connect(self):'):]


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


# ------------------------------------------------------------------ item 5: real differences fail the scripts
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_native_probe_verdict():
    probe = _load(PY / 'abplus_probe_v3_native.py', 'probe_v3_native')
    good = {'P': dict(compared=5, equal=5), 'E': dict(compared=5, equal=5), 'missing': 1}
    assert probe.failures(good, {'lethal hit cancelled': 1}, ['lethal hit cancelled']) is None
    assert probe.failures({**good, 'E': dict(compared=5, equal=4)}, {}, [])['differ'] == ['E']
    assert probe.failures({'missing': 3}, {}, [])['nothing_compared']
    assert probe.failures({'P': dict(compared=0, equal=0)}, {}, [])['nothing_compared']
    assert probe.failures(good, {'v3: grid size': 2}, ['lethal hit cancelled'])['reasons_not_allowed'] == ['v3: grid size']


def test_drop_probe_verdict():
    probe = _load(PY / 'abplus_probe_drop.py', 'probe_drop')
    case = dict(paths_agree=True, expected_change=True)
    assert probe.summary_ok([case, case])
    assert not probe.summary_ok([])
    assert not probe.summary_ok([case, dict(case, paths_agree=False)])
    assert not probe.summary_ok([case, dict(case='x', error='RuntimeError')])


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def test_native_golden_cmp_exit_code(tmp_path):
    tool = PY / 'tools' / 'v3_native_golden_cmp.py'
    names = 'header,canvas,minimap,P,local,E,N,S'
    for f in ('forgotten', 'shooting'):
        shutil.copy(FIX / f'{f}.v3.bin', tmp_path / f'{f}.v3.bin')
        shutil.copy(FIX / f'{f}.v3.bin', tmp_path / f'{f}.native.bin')
    r = _run([sys.executable, str(tool), str(tmp_path), names])
    assert r.returncode == 0 and 'PASS' in r.stdout, r.stdout + r.stderr
    b = bytearray((tmp_path / 'shooting.native.bin').read_bytes())
    b[V3.FIXED * 4 + 4 * 3] ^= 0x10   # a P float of body 0
    (tmp_path / 'shooting.native.bin').write_bytes(bytes(b))
    r = _run([sys.executable, str(tool), str(tmp_path), names])
    assert r.returncode == 1 and 'FAIL' in r.stdout and 'shooting P differs' in r.stdout
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert _run([sys.executable, str(tool), str(empty), names]).returncode == 1


def test_accept_checks():
    ac = _load(BM1 / 'accept_checks.py', 'accept_checks')
    assert ac.frame_ok(dict(tested=4, payload_equal=4, digests_equal=4))
    assert not ac.frame_ok(dict(tested=4, payload_equal=3, digests_equal=4))
    assert not ac.frame_ok(dict(tested=4, payload_equal=4, digests_equal=3))
    assert not ac.frame_ok(dict(tested=0, payload_equal=0, digests_equal=0))
    line = 'files 12, exit slots lost 0, native visible checks %d mismatches %d'
    assert ac.r2_ok(line % (500, 0)) and not ac.r2_ok(line % (500, 1)) and not ac.r2_ok(line % (0, 0))
    assert not ac.r2_ok('no such line')


SH = shutil.which('sh')


@pytest.mark.skipif(SH is None, reason='no sh')
@pytest.mark.parametrize('build_ok', [True, False])
def test_deploy_abphome_fails_on_a_failed_build(tmp_path, build_ok):
    """deploy.sh abphome with a stub run.sh (the container build): a build that writes no library exits non-zero and
    moves the old library to trash (none left to pass for the new one); a good build replaces it."""
    bm = tmp_path / 'bm1'
    eng = bm / 'trees' / 't' / 'bridge' / 'engine'
    eng.mkdir(parents=True)
    for f in ('run_instance.sh', 'null_render_e.txt', 'stub_render_g.txt', 'stub_render_h.txt', 'abp_turbo.c',
              'abp_v3_native.c', 'abp_schema_v3.h', 'abp_step_line.h'):
        (eng / f).write_text('x')
    shutil.copy(BM1 / 'deploy.sh', bm / 'deploy.sh')
    tools = bm / 'home' / 'isaac-abplus' / 'tools'
    tools.mkdir(parents=True)
    (tools.parent / 'runtime').mkdir()   # (present: no link made)
    (tools / 'libabp_turbo.so').write_text('OLD')
    (bm / 'common.sh').write_text(
        'BM1_MODE=direct\nBM1_GAME_RUNTIME=/nonexistent\nBM1_IN=%s\n'
        'trash() { [ -e "$1" ] || return 0; d=$BM/trash/x; mkdir -p "$d" && mv "$1" "$d/"; }\n' % bm.as_posix())
    # (the failure as on the host: the container's gcc fails, run.sh still returns 0)
    build = 'echo NEW > %s/libabp_turbo.so.new' % tools.as_posix() if build_ok else 'echo "gcc: error" >&2; exit 0'
    (bm / 'run.sh').write_text('#!/bin/sh\n%s\n' % build)
    os.chmod(bm / 'run.sh', 0o755)
    r = _run([SH, (bm / 'deploy.sh').as_posix(), 'abphome', 't'], env=dict(os.environ, BM=bm.as_posix()))
    lib = tools / 'libabp_turbo.so'
    if build_ok:
        assert r.returncode == 0, r.stderr
        assert lib.read_text().strip() == 'NEW' and not (tools / 'libabp_turbo.so.new').exists()
    else:
        assert r.returncode != 0 and 'not built' in r.stderr
        assert not lib.exists() and (bm / 'trash' / 'x' / 'libabp_turbo.so').read_text() == 'OLD'
