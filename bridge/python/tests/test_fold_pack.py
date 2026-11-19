"""(P package, 2026-11-19) fold_pack: GraphActor(fold=True)'s host side as a pure function -- the bytes it writes equal
the packing GraphActor did before (kept here word for word as the reference), on random records, every combination of
memory / temporal / crn / given uniforms / rollout rows, calls of every size; fold_defined covers exactly the bytes it
writes. CPU only (numpy buffers in place of the pinned ones)."""
import numpy as np
import pytest

from isaac_bridge.tok_policy import ROW_BYTES, FoldLayout, crn_uniforms, fold_defined, fold_pack, fold_views
from isaac_bridge.tok_obs import ROW

HEADS = (9, 9, 2, 3, 6)


def _old_pack(lay, stage, cs, rows, pending, widx, reset, dest, ukeys, sizes):
    """GraphActor._pack_fold before the P package (5a212ec), on explicit buffers."""
    H = sum(HEADS)
    cw = lay.cw
    c_widx, c_keep, c_dup, c_cf = (cs[:, 0:8].view(np.int64)[:, 0], cs[:, 8:12].view(np.float32)[:, 0],
                                   cs[:, 12:16].view(np.float32)[:, 0], cs[:, 16:20].view(np.float32)[:, 0])
    c_cu = cs[:, 20:20 + 4 * H].view(np.float32)
    c_dest = cs[:, cw - 8:cw].view(np.int64)[:, 0] if lay.t_on else None
    dt_e, off_e = ROW.fields['ent'][:2]
    off_e, per_e, end_e = off_e, dt_e.itemsize // dt_e.shape[0], off_e + dt_e.itemsize
    n = len(rows)
    most = int(rows['n_ent'].max())
    g = int(np.searchsorted(sizes, most))
    src = np.ascontiguousarray(rows).view(np.uint8).reshape(n, ROW_BYTES)
    cut = off_e + per_e * sizes[g]
    stage[:n, :cut] = src[:, :cut]
    stage[:n, end_e:ROW_BYTES] = src[:, end_e:]
    stage[:n, ROW_BYTES:] = np.ascontiguousarray(pending, np.int64).view(np.uint8).reshape(n, 8 * len(HEADS))
    if lay.mem or lay.t_on:
        c_widx[:n] = widx
        c_widx[n:] = widx[0]
    if lay.mem:
        c_keep[:n] = 1.0 if reset is None else (~np.asarray(reset, bool)).astype(np.float32)
    if dest is not None:
        c_dest[:n] = dest
    c_dup[:n] = 0.0
    c_dup[n:] = 1.0
    c_cf[:] = 0.0
    crn_rows = 0
    if lay.crn and 'crn' in rows.dtype.names:
        crn = rows['crn']
        if crn.any():
            sel = np.flatnonzero(crn)
            c_cu[sel] = crn_uniforms(crn[sel], rows['t'][sel], H)
            c_cf[sel] = 1.0
            crn_rows = len(sel)
    if ukeys is not None:
        c_cu[:n] = crn_uniforms(ukeys[0], ukeys[1], H)
        c_cf[:n] = 1.0
    return g, crn_rows


def _rows(rng, k):
    r = np.frombuffer(rng.integers(0, 256, k * ROW_BYTES, dtype=np.uint8).tobytes(), ROW).copy()
    r['n_ent'] = rng.integers(0, 65, k)
    r['crn'] = np.where(rng.random(k) < 0.3, rng.integers(1, 2 ** 40, k), 0)
    r['t'] = rng.integers(0, 5000, k)
    return r


@pytest.mark.parametrize('mem,t_on,crn', [(0, 1, 1), (0, 1, 0), (1, 0, 1), (1, 1, 1), (0, 0, 0)])
def test_fold_pack_equals_the_old_packing(mem, t_on, crn):
    rng = np.random.default_rng(mem * 4 + t_on * 2 + crn)
    n = 16
    sizes, rsizes = (16, 32, 64, 128), (8, 16)
    lay = FoldLayout(n, HEADS, mem, t_on, crn, sizes, rsizes)
    sb = ROW_BYTES + 8 * len(HEADS)
    a_st, b_st = np.full((n, sb), 0xAB, np.uint8), np.full((n, sb), 0xAB, np.uint8)
    a_c, b_c = np.full((n, lay.cw), 0xCD, np.uint8), np.full((n, lay.cw), 0xCD, np.uint8)
    cv = fold_views(lay, b_c)
    for call in range(60):
        k = int(rng.integers(1, n + 1))
        rows = _rows(rng, k)
        pend = rng.integers(0, 9, (k, len(HEADS)))
        widx = rng.permutation(64)[:k] if (mem or t_on) else None
        reset = (rng.random(k) < 0.2) if (mem and rng.random() < 0.7) else None
        dest = rng.integers(0, 10 ** 6, k) if (t_on and rng.random() < 0.7) else None
        ukeys = (rng.integers(1, 2 ** 40, k), rng.integers(0, 999, k)) if rng.random() < 0.2 else None
        eb = int(rng.choice(sizes)) if rng.random() < 0.5 else None
        g0, crn0 = _old_pack(lay, a_st, a_c, rows, pend, widx, reset, dest, ukeys, sizes)
        res = fold_pack(lay, rows, pend, widx, reset, dest, ukeys, stage=b_st, ctrl=b_c, cv=cv if call % 2 else None,
                        eb=eb)
        assert np.array_equal(a_st, b_st) and np.array_equal(a_c, b_c), call   # (the whole buffers: stale bytes too)
        assert res['crn_rows'] == crn0
        assert res['g'] == (sizes.index(eb) if eb is not None else g0)
        r = rsizes[int(np.searchsorted(rsizes, k))]
        assert res['r'] == r and res['graph'] == res['g'] + rsizes.index(r) * len(sizes)
        # fold_defined: a pack into buffers of other stale bytes agrees on exactly the defined ones
        sm, cm = fold_defined(lay, rows, dest=dest is not None, ukeys=ukeys is not None)
        c_st, c_c = np.full((n, sb), 0x11, np.uint8), np.full((n, lay.cw), 0x22, np.uint8)
        fold_pack(lay, rows, pend, widx, reset, dest, ukeys, stage=c_st, ctrl=c_c, eb=eb)
        assert np.array_equal(c_st[:k][sm], b_st[:k][sm]) and np.array_equal(c_c[cm], b_c[cm])
        assert not np.any(c_st[:k][~sm] != 0x11) and not np.any(c_c[~cm] != 0x22)
