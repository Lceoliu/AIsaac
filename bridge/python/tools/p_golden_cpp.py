"""(P package, 2026-11-19) The golden samples of tools/p_golden.py through the C++ side (isaac_bridge.tok_native:
bridge/engine/abp_tok_core.h): every check case's code / record / counts / truncations / E bucket, every fold case's
defined staging and control bytes and g / r / graph / crn_rows must equal the file's (the Python reference's). Also
the frame-prefix rule of the driver (prefix = each record's visible_bytes): the bytes it copies are the reference's,
every one of them; and CRN bit for bit against tok_policy.crn_uniforms.

   usage: python tools/p_golden_cpp.py <golden.npz>     exit 0: every case equal"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from isaac_bridge import tok_native   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402
from isaac_bridge.tok_policy import FoldLayout, crn_uniforms   # noqa: E402


def check_one(m, consts, z, cn):
    g = lambda k: z[f'{cn}/{k}']   # noqa: E731
    code, rec, counts, tn, ts, eb = m.check_batch(consts, g('headers'), g('versions'), g('sizes'), g('p0'),
                                                  int(g('p_max')))
    bad = [k for k, v in (('code', code), ('record', rec)) if v != int(g(k))]
    if int(g('code')) == 0:
        if not np.array_equal(counts, g('counts')):
            bad.append('counts')
        bad += [k for k, v in (('trunc_n', tn), ('trunc_s', ts), ('e_bucket', eb)) if v != int(g(k))]
    return bad


def fold_one(m, z, cn, prefix=False):
    g = lambda k: z[f'{cn}/{k}']   # noqa: E731
    has = lambda k: f'{cn}/{k}' in z.files   # noqa: E731
    n, mem, t_on, crn = (int(x) for x in g('layout'))
    lay = FoldLayout(n, tuple(int(x) for x in g('heads')), mem, t_on, crn, g('sizes'), g('rsizes'))
    L = tok_native.fold_layout(lay)
    rows = np.ascontiguousarray(g('rows'))
    k = len(rows)
    stage = np.full(g('stage').shape, g('stage0_byte'), np.uint8)
    ctrl = np.full(g('ctrl').shape, g('ctrl0_byte'), np.uint8)
    pf = None
    sm = g('stage_mask').copy()
    if prefix:   # (the driver's rule: past the entity span, each record's bytes up to its frame's end only)
        fa, vb = ROW.fields['visible_frame'][1], ROW.fields['visible_bytes'][1]
        nb = rows[:, vb:vb + 4].copy().view(np.int32)[:, 0].astype(np.int64)
        cap = L['row_bytes'] - fa   # (A-4's take_prefix rule: up to the frame's end in whole floats; outside 0 .. cap whole)
        pf = np.where((nb >= 0) & (nb <= cap), fa - L['end_e'] + ((nb + 3) & ~3), L['row_bytes'] - L['end_e'])
        for i in range(k):
            sm[i, L['end_e'] + pf[i]:L['row_bytes']] = False
    r = m.fold_pack(L, rows, g('pending'), g('widx') if (mem or t_on) else None,
                    g('reset').astype(np.uint8) if has('reset') else None, g('dest') if has('dest') else None,
                    g('ukey') if has('ukey') else None, g('ut') if has('ut') else None, int(g('eb')), stage, ctrl, pf)
    bad = [] if np.array_equal(stage[:k][sm], g('stage')[:k][sm]) else ['stage']
    bad += [] if np.array_equal(ctrl[g('ctrl_mask')], g('ctrl')[g('ctrl_mask')]) else ['ctrl']
    return bad + [x for x, v in zip(('g', 'r', 'graph', 'crn_rows'), r) if v != int(g(x))]


def main(argv=None):
    argv = argv or sys.argv[1:]
    m = tok_native.load(verbose=True)
    if m is None:
        print('tok_native:', tok_native.REASON)
        return 2
    z = np.load(argv[0])
    consts = tok_native.v3_consts()
    names = sorted({k.split('/')[0] for k in z.files})
    bad = {}
    for cn in names:
        b = check_one(m, consts, z, cn) if cn.startswith('check') else fold_one(m, z, cn)
        if not cn.startswith('check'):
            b += [f'prefix:{x}' for x in fold_one(m, z, cn, prefix=True)]
        if b:
            bad[cn] = b
    rng = np.random.default_rng(7)
    keys, t = rng.integers(-2 ** 62, 2 ** 62, 4000), rng.integers(-5, 10 ** 6, 4000)
    crn_ok = np.array_equal(m.crn(keys, t, 29).view(np.uint32), crn_uniforms(keys, t, 29).view(np.uint32))
    print(f'{len(names)} cases ({sum(c.startswith("check") for c in names)} check, '
          f'{sum(not c.startswith("check") for c in names)} fold, each also with the prefix rule), '
          f'{len(bad)} with mismatches' + (f': {bad}' if bad else '') + f'; crn {"equal" if crn_ok else "DIFFERS"}')
    return 1 if (bad or not crn_ok) else 0


if __name__ == '__main__':
    sys.exit(main())
