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
        with pytest.raises(VP.VisibleV3Fatal, match=why):
            write_visible(al.LeanDecoder().decode(payload), np.zeros(1, ROW))
    for payload in (lean(raw=RAW), lean(frame=VP.observation_frame_v3(RAW))):   # a v3 record: written
        row = np.zeros(1, ROW)
        assert write_visible(al.LeanDecoder().decode(payload), row)
        assert int(row['visible_schema'][0]) == V3.VERSION and int(row['visible_bytes'][0]) > 0
    src = Path(tok_obs.__file__).read_text(encoding='utf-8')   # (encode_row's own first step)
    assert 'v3 = write_visible(o, row)' in src[src.index('def encode_row('):]


def _lean_v2():
    """A lean payload with a 15 x 9 terrain block (flag 2) and a real v2 visible block (flag 32), one player."""
    from isaac_bridge import schema_v2
    b = bytearray(al._HEADER.pack(al.MAGIC, 1234, 1200, 2 | 32, 1, 0, 0))
    b += al._ROOM.pack(1, 0, 0, 0, 60, 2, 0, 0, 40.0, 120.0, 600.0, 400.0)
    b += al._TOTALS.pack(0.0, 0.0, 0.0, 0.0)
    pl = np.zeros(1, al.PLAYER)
    pl['hearts'], pl['x'], pl['y'] = 6, 300, 250
    b += pl.tobytes() + struct.pack('<H', 0)
    cells = [[i, 60 + 40 * (i % 15), 140 + 40 * (i // 15), 0, 1, 1, 0, 0, 0, 0] for i in range(135)]
    t, g = json.dumps(dict(width=15, height=9, cells=cells)).encode(), b'[]'
    b += struct.pack('<II', 1, len(t)) + t + struct.pack('<I', len(g)) + g
    vis = json.dumps(dict(schema=schema_v2.SCHEMA, curses=0, capabilities=1, stage=1, inventory=[], slots=[],
                          bodies=[dict(controller=0, controller_known=1, animation=0, animation_known=1)])).encode()
    return bytes(b) + struct.pack('<I', len(vis)) + vis


def test_a_v2_worker_still_takes_v2_records(mode):
    """(2026-11-18, round 3: a real v2 block, not only a record without one) A v2 worker writes the v2 frame of a v2
    record (visible_schema = schema_v2.VERSION, the frame's bytes) and visible_schema 0 for a record without a block;
    a v3 worker refuses the same v2 record."""
    from isaac_bridge import schema_v2
    from isaac_bridge.visible_native import frame
    mode(0)
    o = al.LeanDecoder().decode(_lean_v2())
    row = np.zeros(1, ROW)
    assert not write_visible(o, row)
    want = frame(o)
    assert int(row['visible_schema'][0]) == schema_v2.VERSION and int(row['visible_bytes'][0]) == len(want) > 0
    assert bytes(row['visible_frame'][0, :len(want)]) == want
    row = np.zeros(1, ROW)
    assert not write_visible(al.LeanDecoder().decode(lean()), row)
    assert int(row['visible_schema'][0]) == 0
    mode(2)
    with pytest.raises(VP.VisibleV3Fatal, match='prlmens-g-v2'):
        write_visible(al.LeanDecoder().decode(_lean_v2()), np.zeros(1, ROW))


def test_the_bridge_mode_is_checked(mode):
    ok = lambda m, native=True: dict(type='ok', cmd='visible_v3', mode=m, enabled=m != 0, native_available=native)
    for m in range(4):
        assert VP.check_bridge_mode(ok(m), m) == m
    with pytest.raises(VP.VisibleV3Fatal, match='bridge runs mode 0, this worker is configured for mode 2'):
        VP.check_bridge_mode(ok(0), 2)
    with pytest.raises(VP.VisibleV3Fatal, match='bridge runs mode 2, this worker is configured for mode 0'):
        VP.check_bridge_mode(ok(2), 0)
    for m in (2, 3):
        with pytest.raises(VP.VisibleV3Fatal, match='native encoder'):
            VP.check_bridge_mode(ok(m, native=False), m)
    assert VP.check_bridge_mode(ok(1, native=False), 1) == 1   # (mode 1 needs no native encoder)
    unknown = dict(type='error', msg='unknown cmd')   # a bridge without the command: mode 0
    assert VP.check_bridge_mode(unknown, 0) == 0
    with pytest.raises(VP.VisibleV3Fatal):
        VP.check_bridge_mode(unknown, 1)
    with pytest.raises(VP.VisibleV3Fatal, match='did not answer'):
        VP.check_bridge_mode(dict(type='error', cmd='visible_v3', msg='visible_v3: mode must be 0-3, got 4'), 0)
    for bad in ('x', None, [1]):   # (a malformed answer: a protocol error, not a ValueError)
        with pytest.raises(VP.VisibleV3Fatal, match='answered the mode query'):
            VP.check_bridge_mode(dict(type='ok', cmd='visible_v3', mode=bad, native_available=True), 1)
    mode(3)   # (the default: the worker's configured mode)
    assert VP.check_bridge_mode(ok(3)) == 3


def test_the_connect_check_with_the_clients_recv():
    """abplus.check_visible_v3 (AbplusTrainingEnv.connect's check) with a client whose _recv raises on an error reply,
    as the real one does: an old bridge's 'unknown cmd' is mode 0 (passes for a mode 0 client, refused otherwise); a
    mismatch is a VisibleV3Fatal naming both modes (2026-11-18, round 3: not a BridgeError, which the samplers retry)
    and the connection is closed."""
    from isaac_bridge.abplus import check_visible_v3
    from isaac_bridge.env import BridgeError, BridgeErrorReply

    class Sock:
        closed = False

        def close(self):
            self.closed = True

    class Client:
        def __init__(self, reply):
            self.reply, self.sent, self._sock = reply, [], Sock()

        def _send(self, msg):
            self.sent.append(msg)

        def _recv(self):
            if self.reply.get('type') == 'error':
                raise BridgeErrorReply(self.reply)
            return self.reply
    c = Client(dict(type='ok', cmd='visible_v3', mode=2, native_available=True))
    assert check_visible_v3(c, 2) == 2 and c.sent == [{'cmd': 'visible_v3'}]
    sock = c._sock
    with pytest.raises(VP.VisibleV3Fatal, match='bridge runs mode 2, this worker is configured for mode 1'):
        check_visible_v3(c, 1)
    assert sock.closed and c._sock is None and not issubclass(VP.VisibleV3Fatal, (BridgeError, OSError, RuntimeError,
                                                                                  ValueError))
    old = Client(dict(type='error', msg='unknown cmd'))   # (mode 0: no query at all -- 05f1622's protocol)
    assert check_visible_v3(old, 0) == 0 and old.sent == [] and not old._sock.closed
    with pytest.raises(VP.VisibleV3Fatal, match='bridge runs mode 0, this worker is configured for mode 1'):
        check_visible_v3(old, 1)
    src = Path(al.__file__).with_name('abplus.py').read_text(encoding='utf-8')
    assert 'check_visible_v3(self)' in src[src.index('    def connect(self):'):]


def _socket_client(answer, binary=False):
    """A client on one end of a socket pair with its real _send / _recv: IsaacBridgeEnv's, or (binary) the
    AbplusTrainingEnv's of obs_format 2; answer: bytes written by the other end first, None: EOF on our read."""
    import socket
    from isaac_bridge.env import IsaacBridgeEnv
    from isaac_bridge.abplus import AbplusTrainingEnv
    ours, theirs = socket.socketpair()
    c = object.__new__(AbplusTrainingEnv if binary else IsaacBridgeEnv)
    c._sock, c._buf = ours, b''
    if binary:
        c.obs_format, c.validate_obs = 2, False
    if answer is not None:
        theirs.sendall(answer)
    else:   # (EOF on our read; our query still goes out)
        theirs.shutdown(socket.SHUT_WR)
    return c, theirs


@pytest.mark.parametrize('binary', [False, True])
def test_the_connect_check_retries_transport_errors_as_before(binary):
    """(2026-11-18, round 4) Only a real error answer of the bridge ({"type": "error"}: BridgeErrorReply) is read as
    the answer -- 'unknown cmd' is mode 0, another error a protocol error (VisibleV3Fatal); the game closing the
    connection during the query (EOF) or no connection stays a plain BridgeError, retried by the samplers as before
    05f1622 (before: any BridgeError became the answer, and an EOF a fatal error, in mode 0 too)."""
    from isaac_bridge.abplus import check_visible_v3
    from isaac_bridge.env import BridgeError, BridgeErrorReply
    c, other = _socket_client(binary=binary, answer=b'{"type":"error","msg":"unknown cmd"}\n')
    with pytest.raises(VP.VisibleV3Fatal, match='bridge runs mode 0, this worker is configured for mode 1'):
        check_visible_v3(c, 1)
    c, other = _socket_client(binary=binary, answer=b'{"type":"error","msg":"something else"}\n')
    with pytest.raises(VP.VisibleV3Fatal, match='did not answer'):
        check_visible_v3(c, 2)
    for mode in (1, 2):
        c, other = _socket_client(None, binary)   # the game gone during the query
        with pytest.raises(BridgeError, match='connection closed by game') as got:
            check_visible_v3(c, mode)
        assert not isinstance(got.value, (VP.VisibleV3Fatal, BridgeErrorReply))
    c, other = _socket_client(binary=binary, answer=b'')
    c._sock = None   # not connected
    with pytest.raises(BridgeError, match='not connected'):
        check_visible_v3(c, 1)
    c, other = _socket_client(None, binary)   # mode 0: no query sent, nothing read (05f1622's protocol)
    assert check_visible_v3(c, 0) == 0 and c._buf == b''
    c, other = _socket_client(binary=binary, answer=b'{"type":"error","msg":"x"}\n')   # the reply kept on the exception
    with pytest.raises(BridgeErrorReply) as got:
        c._recv()
    assert got.value.reply == {'type': 'error', 'msg': 'x'} and isinstance(got.value, BridgeError)


def test_needs_decode_by_the_payload():
    frame = VP.observation_frame_v3(RAW)
    assert VP.needs_decode(lean(frame=frame)) and VP.needs_decode(lean(raw=RAW, frame=frame))
    assert VP.needs_decode(lean(raw=RAW))
    assert not VP.needs_decode(lean()) and not VP.needs_decode(lean(v2={'schema': 'isaac-visible-prlmens-g-v2'}))


@pytest.mark.parametrize('m', [0, 2])
def test_fastrow_hands_what_the_c_path_returns_to_encode_row(mode, monkeypatch, m):
    """(2026-11-20, abp_row_encode version 9: the C path takes a v3 frame itself, after the same checks as write_visible)
    FastRow.encode gives every raw payload to the C path; whatever it hands back (a negative code: a raw v3 block, a
    frame in a worker not configured for v3, a v3 worker's record without a frame, a frame it refuses) is decoded and
    goes through encode_row, the reference, with its checks. Before (f3264e0 .. version 8): a v3 payload never reached
    the C path."""
    mode(m)
    calls, slow = [], []

    def fn(*a):   # (the C path's answer: hand back a payload with a v3 frame or raw block, take the others)
        calls.append(a)
        return -13 if (a[0][12] & (32 | 64)) else 0
    monkeypatch.setattr(tok_obs, 'encode_row', lambda o, st, row, t: slow.append(o) or 0)
    fr = FastRow(fn, EpisodeState(1000), al.LeanDecoder())
    fr._ptrs = (1, 1, 1)   # (a grid as if the C path were set up)
    frame = VP.observation_frame_v3(RAW)
    row = np.zeros(1, ROW)
    for payload in (lean(raw=RAW, frame=frame), lean(frame=frame), lean(raw=RAW)):
        fr._ptrs = (1, 1, 1)   # (a slow record resets them: _from_st)
        fr.encode(payload, row, row.ctypes.data, 1)
    assert len(calls) == 3 and fr.slow == 3 and len(slow) == 3 and fr.reasons == {-13: 3}
    assert slow[0].visible_v3_frame == frame and slow[1].visible_v3_frame == frame and slow[2].visible == RAW
    fr._ptrs = (1, 1, 1)
    fr.encode(lean(), row, row.ctypes.data, 2)   # taken by the C path (in a v3 worker the real one hands it back)
    assert len(calls) == 4 and fr.fast == 1


def test_the_c_row_path_checks_a_v3_frame_as_write_visible():
    src = (PY.parent / 'engine' / 'abp_turbo.c').read_text(encoding='utf-8')
    at = src.index('int abp_row_encode(const unsigned char *pl')
    body = src[at:at + 6000]
    assert 'if (flags & 32) return -13;' in body and 'if (frame && !v3_mode) return -13;' in body
    assert 're_v3_frame_ok(frame, frame_bytes, &bodies)' in body and '} else if (v3_mode) {' in body
    ok = src[src.index('static int re_v3_frame_ok'):at]
    for want in ('ABP_V3_MAGIC', 'ABP_V3_VERSION_WORD', 'ABP_V3_CAP_BODY', 'ABP_V3_CAP_ENTITY', 'nf != want'):
        assert want in ok


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
    """failures() on the probe's per-record accounting (2026-11-18, Codex round 3): every decoded record is both
    blocks (compared), an allowed raw-only hand-back, or another missing frame (a failure); each section compared once
    per record with both blocks; Codex's false pass (1 compared, the rest missing, no reason reported) fails."""
    import os
    before = os.environ.get('ISAAC_RL_VISIBLE_V3')
    probe = _load(PY / 'abplus_probe_v3_native.py', 'probe_v3_native')
    assert os.environ.get('ISAAC_RL_VISIBLE_V3') == before   # (no environment written at import)
    acct = lambda both, ok=0, other=0: dict(both=both, raw_only_allowed=ok, other_missing=other, other_first=[])
    good = {'records': acct(5, 1), 'frame': dict(compared=5, equal=5), 'P': dict(compared=5, equal=5),
            'E': dict(compared=5, equal=5)}
    assert probe.failures(good, {'lethal hit cancelled': 1}, ['lethal hit cancelled']) is None
    assert probe.failures({**good, 'E': dict(compared=5, equal=4)}, {}, [])['differ'] == ['E']
    assert probe.failures({**good, 'records': acct(5, 1, 1)}, {}, [])['other_missing'] == 1
    assert probe.failures({'records': acct(1, 0, 9999), 'P': dict(compared=1, equal=1)}, {}, [])   # (Codex's case)
    assert probe.failures({'records': acct(0), 'P': dict(compared=0, equal=0)}, {}, [])['nothing_compared']
    assert probe.failures({**good, 'P': dict(compared=4, equal=4)}, {}, [])['compared_short'] == ['P']
    assert probe.failures(good, {'v3: grid size': 2}, ['lethal hit cancelled'])['reasons_not_allowed'] == ['v3: grid size']


def test_native_probe_raw_only_records_must_encode():
    """(2026-11-18, Codex round 4) A raw-only record counts as an allowed hand-back only when the reference encoder
    encodes its block (as the worker's write_v3): a lethal-hit hand-back whose block is of another version or has a
    short grid is another missing frame. And the verdict needs the whole frame compared, whatever --sections lists."""
    probe = _load(PY / 'abplus_probe_v3_native.py', 'probe_v3_native_r4')
    allowed = {'lethal hit cancelled'}
    why = 'lethal hit cancelled'
    assert probe.classify_raw_only(RAW, 6, why, 5, allowed) == ('raw_only_allowed', why)
    old = dict(RAW, version=2026111801)
    short = dict(RAW, cells=RAW['cells'][:int(RAW['gw']) * int(RAW['gh']) - 1])
    for raw in (old, short):
        kind, w = probe.classify_raw_only(raw, 6, why, 5, allowed)
        assert kind == 'other_missing' and 'the reference encoder' in w
    assert probe.classify_raw_only(RAW, 7, why, 5, allowed)[0] == 'other_missing'      # (the counter moved by 2)
    assert probe.classify_raw_only(RAW, 6, 'v3: grid size', 5, allowed)[0] == 'other_missing'
    assert probe.classify_raw_only(None, 6, why, 5, allowed)[0] == 'other_missing'
    acct = dict(both=5, raw_only_allowed=1, other_missing=0, other_first=[])
    header_only = {'records': acct, 'header': dict(compared=5, equal=5)}   # (--sections header: no 'frame')
    assert probe.failures(header_only, {}, [])['compared_short'] == ['frame (not compared)']
    assert probe.failures({**header_only, 'frame': dict(compared=5, equal=5)}, {}, []) is None


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
    """v3_native_golden_cmp.py (2026-11-18, Codex rounds 3 / 4): PASS only for the frozen frame set (the fixtures'
    manifest), a run manifest without collection errors, both headers strictly valid for this schema, and the frames
    byte for byte equal but for the stall words. Codex's -0.0 for +0.0 at word 2460 (equal as decoded values), a
    changed size word, an appended float, another magic, a collection error, one frame only, a missing file: FAIL."""
    tool = PY / 'tools' / 'v3_native_golden_cmp.py'
    frozen = json.loads((FIX / 'manifest.json').read_text())
    names = [f['name'] for f in frozen['frames']]

    def setup(d, which=None, errors=(), native=None):
        d.mkdir()
        (d / 'manifest.json').write_text(json.dumps(dict(
            frames=[dict(name=n, native=(native or {}).get(n, 'frame')) for n in (which or names)],
            errors=list(errors))))
        for n in (which or names):
            shutil.copy(FIX / f'{n}.v3.bin', d / f'{n}.v3.bin')   # (stall 0.5 / 1, as the fixtures)
            raw = json.loads((FIX / f'{n}.raw.json').read_text())['raw']
            (d / f'{n}.native.bin').write_bytes(VP.observation_frame_v3(raw))   # (stall 0, as the native side)
        return d
    run = lambda d: _run([sys.executable, str(tool), str(d)], cwd=str(PY))
    good = setup(tmp_path / 'good')
    r = run(good)
    assert r.returncode == 0 and 'PASS' in r.stdout and f'frames {len(names)}/{len(names)} equal' in r.stdout, r.stdout + r.stderr

    def broken(how):
        d = setup(tmp_path / how.replace(' ', '_'))
        p_ = d / 'forgotten.native.bin'
        b = bytearray(p_.read_bytes())
        if how == 'minus zero':
            assert struct.unpack_from('<f', b, 4 * 2460)[0] == 0.0
            struct.pack_into('<f', b, 4 * 2460, -0.0)
        elif how == 'size word':
            struct.pack_into('<f', b, 4, struct.unpack_from('<f', b, 4)[0] + 1)
        elif how == 'float appended':
            b += bytes(4)
        elif how == 'magic':
            struct.pack_into('<f', b, 0, V3.MAGIC + 1)
        p_.write_bytes(bytes(b))
        return d
    for how in ('minus zero', 'size word', 'float appended', 'magic'):
        r = run(broken(how))
        assert r.returncode == 1 and 'FAIL' in r.stdout, (how, r.stdout)
    r = run(setup(tmp_path / 'errors', errors=[dict(name='shape 3', error='RuntimeError')]))
    assert r.returncode == 1 and 'collection error' in r.stdout
    r = run(setup(tmp_path / 'one', which=['forgotten']))
    assert r.returncode == 1 and 'expected set' in r.stdout
    # (round 5) the directory reused, this run's native encoder answered nil for a frame: the earlier run's
    # .native.bin is still there, byte-equal -- the manifest's mark fails it; a manifest without marks fails too
    r = run(setup(tmp_path / 'reused', native={'shooting': 'nil:v3: grid size'}))
    assert r.returncode == 1 and "no native frame from this run for ['shooting']" in r.stdout
    d = setup(tmp_path / 'unmarked')
    m = json.loads((d / 'manifest.json').read_text())
    (d / 'manifest.json').write_text(json.dumps(dict(m, frames=[dict(name=f['name']) for f in m['frames']])))
    assert run(d).returncode == 1
    d = setup(tmp_path / 'missing')
    (d / 'shooting.native.bin').unlink()
    r = run(d)
    assert r.returncode == 1 and 'shooting: a file is missing' in r.stdout
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert run(empty).returncode == 1


def test_the_golden_probe_moves_an_earlier_run_aside(tmp_path):
    """abplus_probe_v3_golden.clear_previous (round 5): the files a run writes, left by an earlier run in the same
    directory, go to <out>/trash/<time>/ (never deleted) before this run writes its own; other files stay."""
    probe = _load(PY / 'abplus_probe_v3_golden.py', 'probe_v3_golden')
    for f in ('a.raw.json', 'a.v3.bin', 'a.engine.json', 'a.native.bin', 'manifest.json', 'run.log', 'STATUS'):
        (tmp_path / f).write_text(f)
    assert probe.clear_previous(tmp_path) == 5
    left = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert left == ['STATUS', 'run.log']
    moved = sorted(p.name for p in (tmp_path / 'trash').rglob('*') if p.is_file())
    assert moved == ['a.engine.json', 'a.native.bin', 'a.raw.json', 'a.v3.bin', 'manifest.json']
    assert probe.clear_previous(tmp_path) == 0


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
@pytest.mark.parametrize('case', ['ok', 'gcc fails', 'fails after a new file', 'gpu refused', 'install fails',
                                  'copy fails'])
def test_deploy_abphome(tmp_path, case):
    """deploy.sh abphome with a stub run.sh that passes its command's exit code on, as the real one (docker: exec; direct:
    exit $rc) (2026-11-18, Codex round 3): only a build with rc 0 that left a fresh non-empty library replaces the
    installed one, by a rename; a failed build (with or without a file left behind), a GPU check that refuses (gcc
    never ran), a failed install or a failed copy exit non-zero and leave the installed library as it was."""
    bm = tmp_path / 'bm1'
    eng = bm / 'trees' / 't' / 'bridge' / 'engine'
    eng.mkdir(parents=True)
    for f in ('run_instance.sh', 'null_render_e.txt', 'stub_render_g.txt', 'stub_render_h.txt', 'abp_turbo.c',
              'abp_v3_native.c', 'abp_schema_v3.h', 'abp_step_line.h'):
        if not (case == 'copy fails' and f == 'abp_step_line.h'):
            (eng / f).write_text('x')
    shutil.copy(BM1 / 'deploy.sh', bm / 'deploy.sh')
    tools = bm / 'home' / 'isaac-abplus' / 'tools'
    tools.mkdir(parents=True)
    (tools.parent / 'runtime').mkdir()   # (present: no link made)
    lib = tools / 'libabp_turbo.so'
    if case == 'install fails':   # (the install target cannot be replaced: a non-empty directory in its place)
        lib.mkdir()
        (lib / 'keep').write_text('OLD')
    else:
        lib.write_text('OLD')
    (bm / 'common.sh').write_text(
        'BM1_MODE=direct\nBM1_GAME_RUNTIME=/nonexistent\nBM1_IN=%s\n'
        'trash() { [ -e "$1" ] || return 0; d=$BM/trash/x; mkdir -p "$d" && mv "$1" "$d/"; }\n' % bm.as_posix())
    new = tools.as_posix() + '/libabp_turbo.so.new'
    build = {'ok': f'echo NEW > {new}; exit 0', 'install fails': f'echo NEW > {new}; exit 0',
             'gcc fails': 'echo "gcc: error" >&2; exit 1', 'fails after a new file': f'echo PART > {new}; exit 1',
             'gpu refused': 'echo "GPU 1 BUSY: not started" >&2; exit 4', 'copy fails': f'echo NEW > {new}; exit 0'}[case]
    (bm / 'run.sh').write_text('#!/bin/sh\n%s\n' % build)
    os.chmod(bm / 'run.sh', 0o755)
    r = _run([SH, (bm / 'deploy.sh').as_posix(), 'abphome', 't'], env=dict(os.environ, BM=bm.as_posix()))
    if case == 'ok':
        assert r.returncode == 0, r.stderr
        assert lib.read_text().strip() == 'NEW' and not (tools / 'libabp_turbo.so.new').exists()
        return
    assert r.returncode != 0 and 'left as it was' in r.stderr, (r.returncode, r.stderr)
    if case == 'install fails':
        assert (lib / 'keep').read_text() == 'OLD'
    else:
        assert lib.read_text() == 'OLD'
    if case in ('gcc fails', 'fails after a new file', 'gpu refused'):
        assert not (tools / 'libabp_turbo.so.new').exists()


# ------------------------------------------------------------------ Codex round 9 (the hardening batch's review)
def test_a_mode_0_worker_refuses_v3_records_before_the_row(mode, monkeypatch):
    """Item 1: a worker not configured for v3 sends no mode query (05f1622's protocol without the query) and so must
    refuse a v3 record itself -- before the ROW, as an ordinary protocol error (ValueError: the episode's handlers take
    it, the worker goes on) -- a valid native frame, a bad one, a raw v3 block; and FastRow sends such a payload to
    encode_row, where the same refusal holds. Before: a valid frame was written (visible_schema the v3 VERSION), a bad
    one ended the worker."""
    mode(0)
    good = VP.observation_frame_v3(RAW)
    bad = bytearray(good)
    bad[0:4] = bytes(4)   # (another magic)
    for payload in (lean(frame=good), lean(frame=bytes(bad)), lean(raw=RAW)):
        row = np.zeros(1, ROW)
        with pytest.raises(ValueError, match='not configured for v3') as got:
            write_visible(al.LeanDecoder().decode(payload), row)
        assert not isinstance(got.value, VP.VisibleV3Fatal)
        assert int(row['visible_schema'][0]) == 0 and int(row['visible_bytes'][0]) == 0
    calls = []   # (version 9: the C path hands a frame in a mode 0 worker back, -13; encode_row refuses it)
    fr = FastRow(lambda *a: calls.append(a) or -13, EpisodeState(1000), al.LeanDecoder())
    fr._ptrs = (1, 1, 1)
    row = np.zeros(1, ROW)
    with pytest.raises(ValueError, match='not configured for v3'):
        fr.encode(lean(frame=good), row, row.ctypes.data, 1)
    assert len(calls) == 1 and int(row['visible_schema'][0]) == 0
    mode(2)   # (a v3 worker writes the same valid frame)
    row = np.zeros(1, ROW)
    assert write_visible(al.LeanDecoder().decode(lean(frame=good)), row)
    assert int(row['visible_schema'][0]) == V3.VERSION


def test_a_whole_payload_of_bad_structure_is_fatal_in_v3_mode(mode):
    """Item 2: the inner structure of a whole payload (read_lean_raw read all its bytes -- a connection lost half way is
    a BridgeError there): the v3 frame or the visible block running past the end, trailing bytes -- VisibleV3Fatal in a
    v3 worker (not retried), the ordinary ValueError in mode 0."""
    frame = VP.observation_frame_v3(RAW)
    good = lean(frame=frame)
    for bad in (good[:-1], good + bytes(1), lean(raw=RAW)[:-3]):
        mode(2)
        with pytest.raises(VP.VisibleV3Fatal):
            al.LeanDecoder().decode(bad)
        mode(0)
        with pytest.raises(ValueError) as got:
            al.LeanDecoder().decode(bad)
        assert not isinstance(got.value, VP.VisibleV3Fatal)


def test_the_mode_answer_is_taken_as_the_protocol_has_it():
    """Item 2: the mode an integer 0-3 as it is (2.9 is no 2, inf no OverflowError, True / '2' no mode), native_available
    a boolean."""
    ok = dict(type='ok', cmd='visible_v3', native_available=True)
    assert VP.check_bridge_mode(dict(ok, mode=2), 2) == 2
    for bad in (2.9, 2.0, float('inf'), float('nan'), True, '2', -1, 4, None):
        with pytest.raises(VP.VisibleV3Fatal, match='answered the mode query'):
            VP.check_bridge_mode(dict(ok, mode=bad), 2)
    for bad in (1, 'yes', None):
        with pytest.raises(VP.VisibleV3Fatal, match='native_available'):
            VP.check_bridge_mode(dict(ok, mode=2, native_available=bad), 2)
