"""(P package, 2026-11-19) Golden samples for the C++ driver's host side: the same inputs through the Python reference
(v3_input.check_batch / reject_code / bucket, tok_policy.fold_pack / fold_defined), the expected outputs saved with
them. The C++ side (frame_counts_v3, the fold control-block packer) must reproduce every expected array; the defined-byte
masks say which bytes of the staging and control buffers are compared (the others are stale by design).

check cases (case kind 'check'): headers [B, 8] float32, versions [B] float64, sizes [B] float64, p0 [B, 2] float32,
  p_max -> code, record (reject_code), and where accepted counts [B, 4] int64, trunc_n, trunc_s (n_max N_MAX_V3,
  s_max S_MAX_V3), e_bucket (E_BUCKETS of the largest e). Covers every E bucket boundary (16 / 17 / 32 / 33 / 64 / 65 /
  128), N / S past their caps, every REJECT reason at the first / a middle / the last record, two reasons in one batch
  (precedence), the P0 pass after the headers, p_max.
fold cases (kind 'fold'): a layout (n, heads, mem, t_on, crn, sizes, rsizes), the records (ROW bytes; random but for
  the visible frame past its first 512 bytes), pending, widx, reset, dest, ukeys, eb; the buffers start filled with
  stage0_byte / ctrl0_byte -> the buffers after fold_pack (all n rows), the masks (stage: the call's k rows), g / r /
  graph / crn_rows.

   usage: python tools/p_golden.py <out.npz>   (also writes <out>.json: the cases' list and what each holds)
          python tools/p_golden.py --verify <file.npz>   (the Python reference against the file: the template of the
          C++ side's check -- replace check_one / fold_one's reference calls with the driver's)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from isaac_bridge import schema_v3 as V3   # noqa: E402
from isaac_bridge import v3_input as VI   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402
from isaac_bridge.tok_policy import ROW_BYTES, FoldLayout, fold_defined, fold_pack   # noqa: E402

HEADS = (9, 9, 2, 3, 6)


def headers_for(counts, rng):
    """Valid header words [B, 8] for counts [B, 4] (p, e, n, s)."""
    counts = np.asarray(counts, np.int64)
    words = V3.frame_floats(*counts.T)
    b = len(counts)
    return (np.column_stack((np.full(b, V3.MAGIC), words, counts, np.full(b, V3.VERSION_WORD),
                             rng.integers(-10, 10, b))).astype(np.float32), (words * 4).astype(np.float64))


def check_case(h, v, s, p0, p_max):
    code, at = VI.reject_code(h, v, s, p0=p0, p_max=p_max)
    out = dict(headers=np.asarray(h, np.float32), versions=np.asarray(v, np.float64), sizes=np.asarray(s, np.float64),
               p0=np.asarray(p0, np.float32), p_max=np.int64(p_max), code=np.int64(code), record=np.int64(at))
    try:
        counts, trunc = VI.check_batch(h, v, s, p_max, VI.N_MAX_V3, VI.S_MAX_V3, p0=p0)
        assert code == 0, (code, at)
        out.update(counts=counts, trunc_n=np.int64(trunc['n']), trunc_s=np.int64(trunc['s']),
                   e_bucket=np.int64(VI.bucket(int(counts[:, 1].max()), VI.E_BUCKETS)) if len(counts) else np.int64(-1))
    except Exception:   # noqa: BLE001
        assert code != 0, 'check_batch refused, reject_code accepted'
    return out


def check_cases(rng):
    cases = {}
    ones = lambda b: np.ones((b, 2), np.float32)   # noqa: E731
    for e in (0, 16, 17, 32, 33, 64, 65, 127, 128):   # the E bucket boundaries (the batch's largest e)
        c = np.column_stack([np.ones(5, np.int64), rng.integers(0, min(e, 16) + 1, 5), rng.integers(0, 9, 5),
                             rng.integers(0, 9, 5)])
        c[2, 1] = e
        h, s = headers_for(c, rng)
        cases[f'check_e{e}'] = check_case(h, np.full(5, V3.VERSION, np.float64), s, ones(5), VI.P_MAX)
    for nm, col, vals in (('n', 2, (VI.N_MAX_V3, VI.N_MAX_V3 + 1, VI.N_MAX_V3 + 20)),
                          ('s', 3, (VI.S_MAX_V3, VI.S_MAX_V3 + 1, VI.S_MAX_V3 + 9))):
        c = np.column_stack([np.ones(4, np.int64), np.full(4, 3), np.full(4, 4), np.full(4, 2)])
        c[1:, col] = vals
        h, s = headers_for(c, rng)
        cases[f'check_trunc_{nm}'] = check_case(h, np.full(4, V3.VERSION, np.float64), s, ones(4), VI.P_MAX)
    defects = [('schema', V3.VERSION - 1), ('length', 4), ('length', V3.FRAME_BYTES + 4), ('length', 10001),
               ('length', np.nan), (V3.H_MAGIC, V3.MAGIC + 1), (V3.H_VERSION, V3.VERSION_WORD + 1),
               (V3.H_PRODUCER, 0.5), (V3.H_PRODUCER, np.inf), (V3.H_P, 0), (V3.H_P, V3.CAP_BODY + 1),
               (V3.H_E, -1), (V3.H_E, V3.CAP_ENTITY + 1), (V3.H_N, -1), (V3.H_S, -1), (V3.H_SIZE, 1.5),
               (V3.H_SIZE, 10000), ('role', 0.999), ('known', 0), ('known', np.nan)]
    for j, (field, value) in enumerate(defects):
        for i in (0, 3, 7):
            c = np.column_stack([np.ones(8, np.int64), rng.integers(0, 20, 8), rng.integers(0, 9, 8),
                                 rng.integers(0, 9, 8)])
            h, s = headers_for(c, rng)
            v, p0 = np.full(8, V3.VERSION, np.float64), ones(8)
            dst, idx = {'schema': (v, i), 'length': (s, i), 'role': (p0, (i, 0)), 'known': (p0, (i, 1))}.get(
                field, (h, (i, field)))
            dst[idx] = value
            cases[f'check_bad{j:02d}_{field}_at{i}'] = check_case(h, v, s, p0, VI.P_MAX)
    c = np.column_stack([np.ones(8, np.int64), np.full(8, 5), np.full(8, 1), np.full(8, 1)])   # precedence
    h, s = headers_for(c, rng)
    v, p0 = np.full(8, V3.VERSION, np.float64), ones(8)
    p0[1, 0] = 0
    cases['check_order_p0_only'] = check_case(h.copy(), v.copy(), s.copy(), p0.copy(), VI.P_MAX)
    h[4, V3.H_MAGIC] = -1
    v[5] = V3.VERSION - 1
    cases['check_order_headers_before_p0'] = check_case(h, v, s, p0, VI.P_MAX)
    c[:, 0] = [1, 2, 1, 1, 2, 1, 1, 1]
    h, s = headers_for(c, rng)
    cases['check_p_max'] = check_case(h, np.full(8, V3.VERSION, np.float64), s, ones(8), 1)
    return cases


def fold_cases(rng):
    cases = {}
    for name, mem, t_on, crn in (('t', 0, 1, 0), ('t_crn', 0, 1, 1), ('m', 1, 0, 1), ('mt', 1, 1, 1)):
        lay = FoldLayout(16, HEADS, mem, t_on, crn, (16, 32, 64, 128), (8, 16))
        for k in (1, 7, 8, 9, 16):
            rows = np.frombuffer(rng.integers(0, 256, k * ROW_BYTES, dtype=np.uint8).tobytes(), ROW).copy()
            fa = ROW.fields['visible_frame'][1]   # (the frame past its first 512 bytes zero: the file compresses)
            rows.view(np.uint8).reshape(k, ROW_BYTES)[:, fa + 512:fa + V3.FRAME_BYTES] = 0
            rows['n_ent'] = rng.integers(0, 65, k)
            rows['crn'] = np.where(rng.random(k) < 0.4, rng.integers(1, 2 ** 40, k), 0)
            rows['t'] = rng.integers(0, 5000, k)
            pend = rng.integers(0, 9, (k, len(HEADS))).astype(np.int64)
            widx = rng.permutation(64)[:k].astype(np.int64)
            reset = rng.random(k) < 0.3 if mem else None
            dest = rng.integers(0, 10 ** 6, k).astype(np.int64) if t_on else None
            ukeys = (rng.integers(1, 2 ** 40, k), rng.integers(0, 999, k)) if k == 9 else None
            eb = 32 if k == 7 else None
            stage = np.full((lay.n, ROW_BYTES + 8 * lay.nh), 0xAB, np.uint8)
            ctrl = np.full((lay.n, lay.cw), 0xCD, np.uint8)
            res = fold_pack(lay, rows, pend, widx if (mem or t_on) else None, reset, dest, ukeys, stage=stage,
                            ctrl=ctrl, eb=eb)
            sm, cm = fold_defined(lay, rows, dest=dest is not None, ukeys=ukeys is not None)
            c = dict(layout=np.array([lay.n, mem, t_on, crn], np.int64), heads=np.array(HEADS, np.int64),
                     sizes=np.array(lay.sizes, np.int64), rsizes=np.array(lay.rsizes, np.int64),
                     rows=rows.view(np.uint8).reshape(k, ROW_BYTES), pending=pend, widx=widx,
                     stage0_byte=np.uint8(0xAB), ctrl0_byte=np.uint8(0xCD), stage=stage, ctrl=ctrl,
                     stage_mask=sm, ctrl_mask=cm,
                     g=np.int64(res['g']), r=np.int64(res['r']), graph=np.int64(res['graph']),
                     crn_rows=np.int64(res['crn_rows']), eb=np.int64(-1 if eb is None else eb))
            if reset is not None:
                c['reset'] = reset
            if dest is not None:
                c['dest'] = dest
            if ukeys is not None:
                c['ukey'], c['ut'] = ukeys
            cases[f'fold_{name}_k{k}'] = c
    return cases


def check_one(z, cn):
    """A check case's inputs through the reference; the mismatches' names."""
    h, v, s, p0, pm = (z[f'{cn}/{k}'] for k in ('headers', 'versions', 'sizes', 'p0', 'p_max'))
    got = check_case(h, v, s, p0, int(pm))
    want = {k.split('/', 1)[1]: z[k] for k in z.files if k.startswith(cn + '/')}
    return [k for k in want if not np.array_equal(np.asarray(got.get(k)), want[k], equal_nan=k in ('headers',
                                                                                                    'sizes', 'p0'))]


def fold_one(z, cn):
    """A fold case's inputs through the reference (buffers from the start bytes); the mismatches' names."""
    g = lambda k: z[f'{cn}/{k}']   # noqa: E731
    has = lambda k: f'{cn}/{k}' in z.files   # noqa: E731
    n, mem, t_on, crn = (int(x) for x in g('layout'))
    lay = FoldLayout(n, tuple(int(x) for x in g('heads')), mem, t_on, crn, g('sizes'), g('rsizes'))
    rows = g('rows').copy().view(ROW)[:, 0]
    stage = np.full(g('stage').shape, g('stage0_byte'), np.uint8)
    ctrl = np.full(g('ctrl').shape, g('ctrl0_byte'), np.uint8)
    eb = int(g('eb'))
    res = fold_pack(lay, rows, g('pending'), g('widx') if (mem or t_on) else None, g('reset') if has('reset') else None,
                    g('dest') if has('dest') else None, (g('ukey'), g('ut')) if has('ukey') else None, stage=stage,
                    ctrl=ctrl, eb=None if eb < 0 else eb)
    k = len(rows)
    bad = [] if np.array_equal(stage[:k][g('stage_mask')], g('stage')[:k][g('stage_mask')]) else ['stage']
    bad += [] if np.array_equal(ctrl[g('ctrl_mask')], g('ctrl')[g('ctrl_mask')]) else ['ctrl']
    return bad + [x for x in ('g', 'r', 'graph', 'crn_rows') if res[x] != int(g(x))]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv[0] == '--verify':
        z = np.load(argv[1])
        names = sorted({k.split('/')[0] for k in z.files})
        bad = {cn: (check_one if cn.startswith('check') else fold_one)(z, cn) for cn in names}
        bad = {k: v for k, v in bad.items() if v}
        print(f'{len(names)} cases, {len(bad)} with mismatches' + (f': {bad}' if bad else ''))
        return 1 if bad else 0
    out = argv[0]
    rng = np.random.default_rng(20261119)
    cases = {**check_cases(rng), **fold_cases(rng)}
    flat, index = {}, {}
    for cn, c in cases.items():
        index[cn] = sorted(c)
        for k, a in c.items():
            flat[f'{cn}/{k}'] = np.asarray(a)
    np.savez_compressed(out, **flat)
    meta = dict(schema_version=V3.VERSION, row_bytes=ROW_BYTES, n_max=VI.N_MAX_V3, s_max=VI.S_MAX_V3,
                e_buckets=list(VI.E_BUCKETS), p_max=VI.P_MAX, reject=VI.REJECT, cases=index)
    with open(os.path.splitext(out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{len(cases)} cases -> {out}')


if __name__ == '__main__':
    sys.exit(main())
