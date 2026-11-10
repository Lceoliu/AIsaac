"""B60: the learner's host packing in C (learn_native, bridge/engine/abp_learn.c) gives the Python reference's bytes:
frame decodes (ROW records and frame arenas, several parts, one-row parts, empty lists), invalid headers (the
reference's error), the slab concatenation (other dtypes, strided pieces), and a whole prepared learner
part (_prepare_packed: every slab region byte for byte, and the same plans / index metadata).

Each test runs on the compiled library when gcc is available ('c') and on an emulation of its functions in
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
    """abp_learn.c's functions in Python, reading and writing the same addresses."""
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


def _same_value(x, y, where):
    """Recursive equality of prepared-part values (numpy arrays and tensors by dtype, shape and bytes)."""
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), where
    elif isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape, where
        assert torch.equal(x, y), where
    elif isinstance(x, dict):
        assert isinstance(y, dict) and x.keys() == y.keys(), where
        for k in x:
            _same_value(x[k], y[k], f'{where}.{k}')
    elif isinstance(x, (list, tuple)):
        assert type(x) is type(y) and len(x) == len(y), where
        for i, (a, b) in enumerate(zip(x, y)):
            _same_value(a, b, f'{where}[{i}]')
    elif isinstance(x, torch.device):
        assert x == y, where
    else:
        assert x == y and type(x) is type(y), where


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
        with block_learner.PREP_LOCK:
            P = block_learner._prepare_packed(m, batch, True, generator=gen, version=2, history_mode='random',
                                              max_rows=4096, packed=True)
        raw = P['slab'].base.numpy()
        regions = {k: bytes(raw[at:at+n]) for k, (at, n, _, _) in P['slab'].layout.items()}
        preps.append((P, regions))
    (pa, ra), (pb, rb) = preps
    assert ra.keys() == rb.keys() and len(ra) > 8
    for k in ra:
        assert ra[k] == rb[k], k
    # every field of the prepared part outside the slab: plans, shapes, indices, counts, draws, scales
    assert pa.keys() == pb.keys()
    for k in pa:
        if k == 'slab':
            continue
        if k in ('cfg', 'memory', 'store'):
            assert pa[k] is pb[k], k
            continue
        _same_value(pa[k], pb[k], k)
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


def test_decode_outside_prep_lock_is_refused():
    from test_block_packed_learner import _rollout_and_teacher
    _, chunks, _ = _rollout_and_teacher('cpu', seed=44)
    _fresh(chunks)
    with pytest.raises(RuntimeError, match='PREP_LOCK'):
        block_learner._host_decode(chunks, True)
    with block_learner.PREP_LOCK:
        assert len(block_learner._host_decode(chunks, True)) == len(chunks)


def test_cache_eviction_during_fill_keeps_the_bytes(lib, monkeypatch):
    """The decode caches dropped (as PagedStore.collect / a new update do, from other threads) right before each
    C concatenation: the preparation's own references keep every array alive, the slab is the reference's."""
    import gc
    from test_block_packed_learner import _rollout_and_teacher
    m, chunks, teach = _rollout_and_teacher('cpu', seed=45)
    m.train()
    batch = chunks+teach
    store = batch[0].snapshot.store
    concat = learn_native.concat

    def evicting(pieces, out, zero_rows=0):
        store.decoded = {}
        bp.clear_decode_cache()
        for c in batch:
            c.__dict__.pop('_host', None)
        gc.collect()
        return concat(pieces, out, zero_rows)
    got = []
    for native in (False, True):
        monkeypatch.setattr(block_learner, 'NATIVE_PACK', native)
        monkeypatch.setattr(learn_native, 'concat', evicting if native else concat)
        _fresh(batch)
        with block_learner.PREP_LOCK:
            P = block_learner._prepare_packed(m, batch, True, generator=torch.Generator().manual_seed(3), version=2,
                                              max_rows=4096, packed=True)
        raw = P['slab'].base.numpy()
        got.append({k: bytes(raw[at:at+n]) for k, (at, n, _, _) in P['slab'].layout.items()})
    assert got[0] == got[1]


def test_epoch_permutations_and_history_draws_equal_the_serial_order(monkeypatch):
    """B60 EpochPerms with the prefetch queue running (the next epoch's first part queued while this epoch's last part
    runs, the update's first part before the first epoch): with a fixed seed, the three permutations and every part's
    history draws (both from the default CPU generator, as torch.randperm(n < 30000, device=cuda) is) equal the
    serial learner's, element for element."""
    from test_block_packed_learner import _rollout_and_teacher
    from isaac_bridge.block_training import forward_bucketed
    m, chunks, _ = _rollout_and_teacher('cpu', seed=43)
    m.train()
    monkeypatch.setattr(block_learner, 'PACKED', True)
    units = list(chunks)
    n = len(units)
    mb = max(2, n//3)
    rec = []
    orig = block_learner._prepare_packed

    def spy(*a, **k):
        P = orig(*a, **k)
        rec.append([x.tolist() for x in P['selected']])
        return P
    monkeypatch.setattr(block_learner, '_prepare_packed', spy)
    kw = dict(version=1, max_rows=4096)

    def parts_of(ph):
        return [[units[i] for i in ph[a:a+mb]] for a in range(0, n, mb)]

    def run(ahead):
        rec.clear()
        torch.manual_seed(5)
        perms = []
        Q = block_learner.PREFETCH
        if not ahead:
            for e in range(3):
                p_ = torch.randperm(n)
                perms.append(p_.tolist())
                for part in parts_of(p_.numpy()):
                    forward_bucketed(m, part, **kw)
        else:
            ep = block_learner.EpochPerms(n, 'cpu')
            h0 = Q.hits
            Q.submit(m, parts_of(ep.get(0)[1])[0], **kw)
            for e in range(3):
                p_, ph = ep.get(e)
                perms.append(p_.tolist())
                ps = parts_of(ph)
                for k, part in enumerate(ps):
                    forward_bucketed(m, part, **kw)
                    if k+1 < len(ps):
                        Q.submit(m, ps[k+1], **kw)
                    elif e+1 < 3:
                        Q.submit(m, parts_of(ep.get(e+1)[1])[0], **kw)
            assert Q.job is None and Q.hits-h0 == 3*len(parts_of(ph))
        return perms, [list(r) for r in rec]
    serial, queued = run(False), run(True)
    assert serial[0] == queued[0]
    assert serial[1] == queued[1] and any(any(len(x) for x in r) for r in serial[1])


def test_update_parameters_bit_identical_native_on_off(monkeypatch):
    """B60 equivalence (B44 / B48 standard): a fixed-input update (three parts, loss, backward, Adam step each, on
    one CPU thread) with the packed host layout gives bit-identical losses and parameters with the C packing on
    and off."""
    from test_block_packed_learner import _rollout_and_teacher
    from test_block_slot_integration import _losses
    from isaac_bridge.block_training import forward_bucketed, source_rows
    from isaac_bridge.structured_input import from_rows
    if learn_native.load() is None:   # (the emulation is checked by the slab tests; this one runs the real C)
        monkeypatch.setitem(learn_native._LIB, 'lib', Emulated())
        monkeypatch.setattr(learn_native, 'ENABLED', True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(block_learner, 'PACKED', True)
    monkeypatch.setattr(block_learner, 'HOST_PACKED', True)
    try:
        res = []
        for native in (False, True):
            monkeypatch.setattr(block_learner, 'NATIVE_PACK', native)
            m_, ch_, te_ = _rollout_and_teacher('cpu', seed=46)
            batch = ch_+te_
            _fresh(batch)
            idx = np.array_split(np.arange(len(batch)), 3)
            parts = [[batch[i] for i in ix] for ix in idx]
            m_.train()
            torch.manual_seed(0)
            opt = torch.optim.Adam(m_.main_parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(9)
            losses = []
            for part in parts:
                out, _ = forward_bucketed(m_, part, generator=gen, version=3, max_rows=4096)
                rows = source_rows([c.framed() for c in part]); pend = np.concatenate([c.pending for c in part])
                loss = _losses(m_, out, from_rows(rows, 'cpu', pend), 3)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.detach().clone())
            res.append((losses, {k: v.detach().clone() for k, v in m_.state_dict().items()}))
        (la, pa), (lb, pb) = res
        for x, y in zip(la, lb):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert pa.keys() == pb.keys()
        for k in pa:
            assert torch.equal(pa[k], pb[k]), k
    finally:
        torch.set_num_threads(threads)


def test_unaligned_arena_offsets_fall_back_to_the_reference(lib, monkeypatch):
    """An arena frame whose offset is not a multiple of 4 (the reference reads its header at the offset and its
    floats at offset // 4 * 4): the whole batch is decoded by the reference."""
    rows, compact = _parts(6)
    recs, buf = compact[0]
    shifted = recs.copy()
    shifted['visible_off'][1] += 2
    pad = np.zeros(len(buf)+8, np.uint8)
    pad[:len(buf)] = buf
    calls = []
    ref = si.decode_frames_compact

    def spy(parts, packed=False):
        calls.append(len(parts))
        return ref(parts, packed)
    monkeypatch.setattr(si, 'decode_frames_compact', spy)
    try:
        want = ref([(shifted, pad), compact[1]], True)
    except ValueError as e:   # (the shifted frame's header is then garbage: the reference's own error)
        with pytest.raises(ValueError, match=str(e)[:20]):
            learn_native.decode([(shifted, pad), compact[1]])
        assert calls == [2]
        return
    got = learn_native.decode([(shifted, pad), compact[1]])
    assert calls == [2]
    for a, b in zip(want, got):
        _same(a, b)


def test_ctypes_signatures_match_the_c_source():
    """Every exported abp_learn_* function: the ctypes argument list has the C parameter list's length, a pointer
    (c_void_p) exactly where the C parameter is a pointer and c_long elsewhere (no compiler needed: the C library's
    tests only run where gcc is, B60 stage A once failed on a 15-for-14 scan signature)."""
    import re
    src = learn_native.SOURCE.read_text()
    found = {}
    for m in re.finditer(r'^\s*(?:long|void|int)\s+(abp_learn_\w+)\s*\(([^)]*)\)\s*\{', src, re.M | re.S):
        params = [x.strip() for x in m.group(2).split(',') if x.strip() and x.strip() != 'void']
        found[m.group(1)] = params
    assert set(found) - {'abp_learn_version'} == set(learn_native.SIGNATURES)
    for name, (args, _) in learn_native.SIGNATURES.items():
        params = found[name]
        assert len(args) == len(params), (name, len(args), len(params))
        for i, (t, p_) in enumerate(zip(args, params)):
            want = ctypes.c_void_p if '*' in p_ else ctypes.c_long
            assert t is want, (name, i, p_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_epoch_perms_on_cuda_keep_the_serial_order_whichever_generator():
    """EpochPerms on CUDA, whichever generator torch.randperm(2400, device=cuda) draws from in this torch (probed:
    the default CPU generator or the CUDA one; 2.7.1 on the 4090 left the CPU state unchanged): with a fixed seed,
    the lazily drawn permutations and CPU draws interleaved as the learner does (each epoch's parts' history draws,
    the next epoch's permutation drawn right after the last part) equal the serial loop's, element for element."""
    torch.manual_seed(7)
    cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.randperm(2400, device='cuda')
    torch.cuda.synchronize()
    uses_cpu = not torch.equal(torch.get_rng_state(), cpu0)
    uses_cuda = not torch.equal(torch.cuda.get_rng_state(), cuda0)
    print(f'torch {torch.__version__}: randperm(2400, cuda) draws from cpu={uses_cpu} cuda={uses_cuda}')
    assert uses_cpu or uses_cuda
    n, parts = 2400, 4

    def history():   # (a part's history draw: torch.randperm on the CPU without a generator)
        return torch.randperm(9).tolist()

    torch.manual_seed(11)
    serial = []
    for e in range(3):
        serial.append(torch.randperm(n, device='cuda').cpu().tolist())
        serial.append([history() for _ in range(parts)])
    torch.manual_seed(11)
    lazy = []
    ep = block_learner.EpochPerms(n, 'cuda')
    ep.get(0)
    for e in range(3):
        lazy.append(ep.get(e)[0].cpu().tolist())
        draws = [history() for _ in range(parts)]
        if e+1 < 3:
            ep.get(e+1)
        lazy.append(draws)
    assert lazy == serial


def test_lean_checkpoint_same_outputs_and_gradients(monkeypatch):
    """B62 N5: the learner's Spatial checkpoint without the determinism metadata and RNG stashing gives the default
    checkpoint's outputs and every gradient bit for bit (CPU, one thread)."""
    from test_block_packed_learner import _rollout_and_teacher
    from isaac_bridge.block_training import forward_bucketed
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(block_learner, 'PACKED', True)
    monkeypatch.setattr(block_learner, 'SPATIAL_CKPT', 1)
    try:
        res = []
        for kw in (dict(use_reentrant=False), block_learner.LEAN_CKPT_KW):
            monkeypatch.setattr(block_learner, 'CKPT_KW', kw)
            m_, ch_, te_ = _rollout_and_teacher('cpu', seed=47)
            m_.train()
            out, _ = forward_bucketed(m_, ch_+te_, generator=torch.Generator().manual_seed(4), version=2, max_rows=4096)
            loss = sum((o.float()*(i+1)).mean() for i, o in enumerate(out))
            loss.backward()
            res.append(([o.detach().clone() for o in out],
                        {n: p.grad.clone() for n, p in m_.named_parameters() if p.grad is not None}))
        (oa, ga), (ob, gb) = res
        for x, y in zip(oa, ob):
            assert torch.equal(x, y)
        assert ga.keys() == gb.keys() and len(ga) > 10
        for k in ga:
            assert torch.equal(ga[k], gb[k]), k
    finally:
        torch.set_num_threads(threads)
