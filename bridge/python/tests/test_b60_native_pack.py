"""B60: the learner's host packing in C (learn_native, bridge/engine/abp_learn.c) gives the Python reference's bytes:
frame decodes (ROW records and frame arenas, several parts, one-row parts, empty lists), invalid headers (the
reference's error), the slab concatenation (other dtypes, strided pieces), page gathers, and a whole prepared learner
part (_prepare_packed: every slab region byte for byte, and the same plans / index metadata).

Each test runs on the compiled library when gcc is available ('c') and on an emulation of its four functions in
Python over the same addresses ('emulated': checks the wrapper's address and size arithmetic where no compiler is)."""
import ctypes
import types

import numpy as np
import pytest
import torch

from test_visible_input import synth_rows
from isaac_bridge import block_learner, block_paged as bp, learn_native
from isaac_bridge import structured_input as si
from isaac_bridge.tok_obs import FrameArena, compact_rows


def _i64(addr, n):
    return np.ctypeslib.as_array((ctypes.c_int64*n).from_address(addr)) if n else np.zeros(0, np.int64)


class Emulated:
    """abp_learn.c's four functions in Python, reading and writing the same addresses."""
    path = 'emulated'

    def scan(self, m, head, schema, size, avail, version, fixed, pf, pl, ef, nf, sf, frame_bytes, counts):
        head, schema, size, avail = _i64(head, m), _i64(schema, m), _i64(size, m), _i64(avail, m)
        out = _i64(counts, 5*m)
        for r in range(m):
            if schema[r] != version or avail[r] < 32:
                return r
            h = np.frombuffer(ctypes.string_at(int(head[r]), 24), '<f4')
            c = []
            for x in h[2:6]:
                if not np.isfinite(x) or x < 0 or x > frame_bytes:
                    return r
                c.append(int(x))
            p, e, n, s = c
            expected = fixed+p*(pf+pl)+e*ef+n*nf+s*sf
            if not (h[0] == 61107 and float(h[1]) == float(expected) and size[r] == expected*4 and p > 0 and e <= 128
                    and size[r] <= frame_bytes and size[r] <= avail[r]):
                return r
            out[5*r:5*r+5] = (p, p, e, n, s)
        return -1

    def gather(self, m, data, counts, header, room_n, map_n, fixed, widths, room, map_, *outs):
        data, counts, widths = _i64(data, m), _i64(counts, 5*m), _i64(widths, 5)
        at = [0]*5
        for r in range(m):
            f = int(data[r])
            ctypes.memmove(room+4*r*room_n, f+4*header, 4*room_n)
            ctypes.memmove(map_+4*r*map_n, f+4*(header+room_n), 4*map_n)
            off = fixed
            for k in range(5):
                c, w = int(counts[5*r+k]), int(widths[k])
                if c:
                    ctypes.memmove(outs[k]+4*at[k]*w, f+4*off, 4*c*w)
                at[k] += c
                off += c*w

    def concat_(self, n, src, nbytes, dst, zero):
        src, nbytes = _i64(src, n), _i64(nbytes, n)
        at = dst
        for i in range(n):
            if nbytes[i]:
                ctypes.memmove(at, int(src[i]), int(nbytes[i]))
            at += int(nbytes[i])
        if zero > 0:
            ctypes.memset(at, 0, zero)

    def pages_(self, n, ids, rows_per_page, n_pages, pages, row_bytes, out):
        ids, pages = _i64(ids, n), _i64(pages, n_pages)
        for i in range(n):
            if ids[i] < 0 or ids[i]//rows_per_page >= n_pages:
                return i
            ctypes.memmove(out+i*row_bytes, int(pages[ids[i]//rows_per_page])+int(ids[i] % rows_per_page)*row_bytes,
                           row_bytes)
        return -1


def _variants():
    out = ['emulated']
    if learn_native.build() is not None:
        out.insert(0, 'c')
    return out


@pytest.fixture(params=_variants())
def lib(request, monkeypatch):
    monkeypatch.setattr(learn_native, 'ENABLED', True)
    monkeypatch.setitem(learn_native._LIB, 'lib',
                        learn_native.Library(learn_native.build()) if request.param == 'c' else Emulated())
    return request.param


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert x.tobytes() == y.tobytes(), k


def _parts(seed):
    rng = np.random.default_rng(seed)
    rows = [synth_rows(rng, n) for n in (5, 1, 3)]
    arenas = [FrameArena(64) for _ in rows]   # (64 bytes: grows, offsets not from 0 in later parts)
    compact = [(compact_rows(r, a), a.buf) for r, a in zip(rows, arenas)]
    return rows, compact


def test_decode_equals_reference_bytes(lib):
    rows, compact = _parts(3)
    for parts in (rows, compact, [rows[0], compact[1], rows[2]], [compact[1]], [rows[1]]):
        ref = si.decode_frames_compact(parts, packed=True)
        got = learn_native.decode(parts)
        assert len(ref) == len(got)
        for a, b in zip(ref, got):
            _same(a, b)
    # (lists empty in every row: zero-length outputs of the reference's shapes)
    one = rows[1].copy()
    ref = si.decode_frames_compact([one], packed=True)[0]
    assert all(ref['c_'+k].shape[1] == si.RAGGED[k] for k in si.RAGGED)


def test_invalid_header_raises_the_reference_error(lib):
    rows, compact = _parts(4)
    bad = rows[0].copy()
    f = bad['visible_frame'][2].view('<f4')
    f[0] = 1.0   # (wrong magic)
    with pytest.raises(ValueError, match='invalid visible-v2'):
        si.decode_frames_compact([bad], packed=True)
    with pytest.raises(ValueError, match='invalid visible-v2'):
        learn_native.decode([bad])
    old = rows[0].copy()
    old['visible_schema'][1] = 7
    with pytest.raises(ValueError, match='legacy observations'):
        learn_native.decode([old])


def test_concat_equals_reference_assignment(lib):
    rng = np.random.default_rng(5)
    pieces = [rng.standard_normal((n, 7)).astype(np.float32) for n in (3, 0, 5, 1)]
    pieces.append(rng.standard_normal((4, 14)).astype(np.float32)[:, ::2])   # (strided)
    pieces.append(rng.standard_normal((2, 7)))                               # (float64: numpy's conversion)
    n = sum(len(x) for x in pieces)
    for zero in (0, 1):
        ref = np.full((n+zero, 7), 9.0, np.float32)
        at = 0
        for x in pieces:
            ref[at:at+len(x)] = x
            at += len(x)
        if zero:
            ref[at:] = 0
        got = np.full((n+zero, 7), 9.0, np.float32)
        learn_native.concat(pieces, got, zero_rows=zero)
        assert got.tobytes() == ref.tobytes()
    ints = [np.arange(6, dtype=np.int64).reshape(3, 2), np.arange(4, dtype=np.int32).reshape(2, 2)]
    out = np.zeros((5, 2), np.int64)
    learn_native.concat(ints, out)
    np.testing.assert_array_equal(out, np.concatenate([ints[0], ints[1].astype(np.int64)]))
    with pytest.raises(ValueError):
        learn_native.concat([np.zeros((4, 7), np.float32)], np.zeros((3, 7), np.float32))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_pages_equal_store_gather(lib, dtype):
    P = 8
    pages = [torch.randn(P, 3, 5).to(dtype) for _ in range(4)]
    ids = np.array([0, 31, 9, 9, 17, 2, 24], np.int64)
    ref = torch.empty((len(ids), 3, 5), dtype=dtype)
    bp.PagedStore._gather(types.SimpleNamespace(page=P), pages, ids, (3, 5), dtype, ref)
    got = torch.empty_like(ref)
    assert learn_native.pages(pages, ids, P, got)
    assert got.view(torch.uint8).numpy().tobytes() == ref.view(torch.uint8).numpy().tobytes()
    assert learn_native.pages(pages, ids[:0], P, got[:0])
    with pytest.raises(IndexError):
        learn_native.pages(pages, np.array([32]), P, got[:1])
    assert not learn_native.pages(pages, ids, P, torch.empty((len(ids), 3, 5), dtype=torch.float64))


def _fresh(chunks):
    for c in chunks:
        c.__dict__.pop('_host', None)
        c.snapshot.store.decoded = {}
    bp.clear_decode_cache()


def test_prepared_part_equals_reference_byte_for_byte(lib, monkeypatch):
    from test_block_packed_learner import _rollout_and_teacher
    m, chunks, teach = _rollout_and_teacher('cpu', seed=41)
    m.train()
    batch = chunks+teach
    preps = []
    for native in (False, True):
        monkeypatch.setattr(block_learner, 'NATIVE_PACK', native)
        _fresh(batch)
        gen = torch.Generator().manual_seed(11)
        P = block_learner._prepare_packed(m, batch, True, generator=gen, version=2, history_mode='random',
                                          max_rows=4096, packed=True)
        raw = P['slab'].base.numpy()
        regions = {k: bytes(raw[at:at+n]) for k, (at, n, _, _) in P['slab'].layout.items()}
        preps.append((P, regions))
    (pa, ra), (pb, rb) = preps
    assert ra.keys() == rb.keys() and len(ra) > 8
    for k in ra:
        assert ra[k] == rb[k], k
    for k in ('names_i', 'sizes_i', 'n_main', 'n_hist', 'n_tail', 'n_new', 'n_old_total', 'plans'):
        assert pa[k] == pb[k], k
    norm = (lambda rb: [(r[0], np.asarray(r[1]).tolist(), *r[2:]) for r in rb])
    assert norm(pa['rbuckets']) == norm(pb['rbuckets'])
    assert pa['n_hist'] > 0 and pa['n_tail'] >= 0


def test_prefetch_queue_order_skips_and_drop(monkeypatch):
    """B60 queue: parts prepared in submission order (grad and no-grad), each taken by its own call; a call for a
    later part gives up the parts queued ahead of it; at most DEPTH queued; drop() empties the queue."""
    from test_block_packed_learner import _rollout_and_teacher
    from isaac_bridge.block_training import forward_bucketed
    m, chunks, teach = _rollout_and_teacher('cpu', seed=42)
    m.train()
    monkeypatch.setattr(block_learner, 'PACKED', True)
    Q = block_learner.PREFETCH
    Q.drop()
    a, b = chunks[:len(chunks)//2], chunks[len(chunks)//2:]
    kw = dict(version=1, max_rows=4096)
    assert Q.submit(m, a, grad=True, **kw) and Q.submit(m, teach, grad=False, train_history=False, version=1)
    assert not Q.submit(m, b, **kw)   # (DEPTH 2)
    h0, d0 = Q.hits, Q.misses
    forward_bucketed(m, a, **kw)
    with torch.no_grad():
        forward_bucketed(m, teach, train_history=False, version=1)
    assert Q.hits-h0 == 2 and Q.job is None
    assert Q.submit(m, a, **kw) and Q.submit(m, b, **kw)
    forward_bucketed(m, b, **kw)   # (a, queued ahead of b, is given up)
    assert Q.hits-h0 == 3 and Q.misses-d0 == 1 and Q.job is None
    assert Q.submit(m, a, **kw)
    Q.drop()
    assert Q.job is None and Q.misses-d0 == 2
