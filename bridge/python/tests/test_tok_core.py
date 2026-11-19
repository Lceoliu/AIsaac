"""P package (2026-11-19): abp_tok_core.h through isaac_bridge.tok_native -- the v3 check, the fold staging and CRN
against the Python reference on tools/p_golden.py's cases (generated here), and the pipe reader's framing. Skipped where
the module cannot be built (Windows, no g++)."""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))

from isaac_bridge import tok_native   # noqa: E402

M = tok_native.load()
pytestmark = pytest.mark.skipif(M is None, reason=f'tok_native: {tok_native.REASON}')


@pytest.fixture(scope='module')
def golden(tmp_path_factory):
    import p_golden
    out = tmp_path_factory.mktemp('golden') / 'g.npz'
    p_golden.main([str(out)])
    return np.load(out)


def test_check_and_fold_equal_the_reference(golden):
    import p_golden_cpp as G
    consts = tok_native.v3_consts()
    names = sorted({k.split('/')[0] for k in golden.files})
    bad = {}
    for cn in names:
        b = G.check_one(M, consts, golden, cn) if cn.startswith('check') else (
            G.fold_one(M, golden, cn) + [f'prefix:{x}' for x in G.fold_one(M, golden, cn, prefix=True)])
        if b:
            bad[cn] = b
    assert len(names) > 80 and not bad, bad


def test_crn_bit_for_bit():
    from isaac_bridge.tok_policy import crn_uniforms
    rng = np.random.default_rng(3)
    keys, t = rng.integers(-2 ** 62, 2 ** 62, 2000), rng.integers(-3, 10 ** 6, 2000)
    assert np.array_equal(M.crn(keys, t, 29).view(np.uint32), crn_uniforms(keys, t, 29).view(np.uint32))


def msg(payload):
    return struct.pack('!i', len(payload)) + payload


def test_reader_frames_records_split_and_joined():
    r = M.MsgReader(3)
    data = msg(b'0') + msg(b'1') + msg(b'0')
    for i in range(len(data)):   # one byte a read
        r.feed(1, data[i:i + 1])
    assert [(k, w, s) for k, w, s, _ in r.take()] == [(0, 1, 2), (0, 1, 3), (0, 1, 2)]
    r.feed(2, msg(b'1') + msg(b'x') + msg(b'0'))   # (nothing after the end)
    assert [(k, w, s) for k, w, s, _ in r.take()] == [(0, 2, 5), (1, 2, -1)]


def test_reader_fatal_and_its_reason():
    r = M.MsgReader(4)
    r.feed(0, msg(b'F') + msg(b'R' + b'visible-v3: a bad frame'))
    assert r.take() == [(2, 0, -1, b'visible-v3: a bad frame')]
    r.feed(1, msg(b'F'))
    assert r.take() == []
    r.feed(1, msg(b'R')[:3])          # (the reason's header half there)
    r.feed(1, msg(b'Rwhy')[0:0])
    assert r.take() == []
    r.eof(1)                          # (the reason never came)
    (k, w, s, t), = r.take()
    assert (k, w) == (2, 1) and b'never came' in t
    r.feed(2, msg(b'F'))
    r.r_timeout(2)
    (k, w, s, t), = r.take()
    assert (k, w) == (2, 2) and b'timeout' in t
    r.feed(3, msg(b'F') + msg(b'0'))  # an F not followed by its R
    (k, w, s, t), = r.take()
    assert (k, w) == (3, 3)


@pytest.mark.parametrize('data', [struct.pack('!i', 0), struct.pack('!i', -5), struct.pack('!i', 1 << 21) + b'0',
                                  msg(b'z'), msg(b'00')])
def test_reader_refuses_what_no_worker_sends(data):
    r = M.MsgReader(1)
    r.feed(0, data)
    (k, w, s, t), = r.take()
    assert (k, w) == (3, 0)
    r.feed(0, msg(b'0'))   # (closed: nothing more)
    assert r.take() == []


def test_reader_eof():
    r = M.MsgReader(2)
    r.eof(0)
    assert r.take() == [(4, 0, -1, b'')]
    r.feed(1, msg(b'0')[:3])
    r.eof(1)
    (k, w, s, t), = r.take()
    assert (k, w) == (3, 1)
