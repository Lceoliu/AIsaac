"""P package (Codex round 18-3): p_audit's regression test on a real training run's logs and dumps -- the clean data must
reconcile (exit 0) and each of Codex's three counterexamples must fail (exit 1):
  (a) the last rollout's last closed decision of a worker made open, its reward 999;
  (b) the first decision of a worker deleted from the first rollout;
  (c) a reply deleted from the middle of a worker's log, and a later decision's reward changed.
The mutated copies go to <scratch> (the unchanged rollouts linked); the tool under test is any p_audit.py (an old tree's
too: the same test shows it pass the counterexamples).

   usage: python tools/p_audit_neg.py <audit dir> <rollouts dir> <scratch dir> [--tool tools/p_audit.py] [--out f.json]"""
import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np


def run(tool, audit, rollouts, out):
    p = subprocess.run([sys.executable, tool, audit, rollouts, '--out', out], capture_output=True, text=True)
    try:
        r = json.load(open(out))
    except Exception:
        r = {}
    return p.returncode, r.get('problems', [])[:3], p.stderr[-400:] if p.returncode not in (0, 1) else ''


def copy_rollouts(src, dst, replace):
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, 'rollout-*'))):
        q = os.path.join(dst, os.path.basename(p))
        if os.path.basename(p) in replace:
            np.savez(q, **replace[os.path.basename(p)])
        else:
            os.symlink(p, q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('audit')
    ap.add_argument('rollouts')
    ap.add_argument('scratch')
    ap.add_argument('--tool', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p_audit.py'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if os.path.exists(a.scratch):
        raise SystemExit(f'{a.scratch} exists')
    os.makedirs(a.scratch)
    npz = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    first, last = npz[0], npz[-1]
    res = dict(tool=a.tool)

    # clean
    res['clean'] = run(a.tool, a.audit, a.rollouts, os.path.join(a.scratch, 'clean.json'))

    # (a) the last rollout: a worker's last decision, closed, made open with reward 999
    z = dict(np.load(last))
    w = next((w for w in range(len(z['length'])) if z['length'][w] > 0 and z['open'][w] == -1), None)
    how = 'closed last decision made open'
    if w is None:   # (every worker's last decision open: the one before it made open -- also a wrong open)
        w = next(w for w in range(len(z['length'])) if z['length'][w] > 1)
        how = 'the decision before the open one made open'
    k = int(z['length'][w]) - 1 if how.startswith('closed') else int(z['length'][w]) - 2
    z['open'] = z['open'].copy()
    z['open'][w] = k
    z['reward'] = z['reward'].copy()
    z['reward'][w, k] = 999.0
    copy_rollouts(a.rollouts, os.path.join(a.scratch, 'ra'), {os.path.basename(last): z})
    res['a'] = dict(worker=w, k=k, how=how,
                    result=run(a.tool, a.audit, os.path.join(a.scratch, 'ra'), os.path.join(a.scratch, 'a.json')))

    # (b) the first rollout: a worker's first decision deleted
    z = dict(np.load(first))
    w = next(w for w in range(len(z['length'])) if z['length'][w] > 2)
    for f, v in list(z.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == len(z['length']) and f not in ('length', 'open'):
            v = v.copy()
            v[w, :-1] = v[w, 1:]
            z[f] = v
    z['length'] = z['length'].copy()
    z['length'][w] -= 1
    z['open'] = z['open'].copy()
    if z['open'][w] >= 0:
        z['open'][w] -= 1
    copy_rollouts(a.rollouts, os.path.join(a.scratch, 'rb'), {os.path.basename(first): z})
    res['b'] = dict(worker=w, result=run(a.tool, a.audit, os.path.join(a.scratch, 'rb'),
                                         os.path.join(a.scratch, 'b.json')))

    # (c) a reply deleted from the middle of worker 0's log; a later decision's reward changed (the last rollout)
    ad = os.path.join(a.scratch, 'audit_c')
    shutil.copytree(a.audit, ad)
    ents = []
    with open(os.path.join(a.audit, 'w000.pkl'), 'rb') as f:
        while True:
            try:
                ents.append(pickle.load(f))
            except EOFError:
                break
    rs = [i for i, e in enumerate(ents) if e[0] == 'r' and e[3] is not None]
    drop = rs[len(rs) // 3]
    dropped = ents[drop]
    with open(os.path.join(ad, 'w000.pkl'), 'wb') as f:
        for i, e in enumerate(ents):
            if i != drop:
                pickle.dump(e, f)
    z = dict(np.load(last))
    k = next((k for k in range(int(z['length'][0])) if k != int(z['open'][0])), 0)
    z['reward'] = z['reward'].copy()
    z['reward'][0, k] += 5.0
    copy_rollouts(a.rollouts, os.path.join(a.scratch, 'rc'), {os.path.basename(last): z})
    res['c'] = dict(dropped=list(dropped[1:3]), k=k,
                    result=run(a.tool, ad, os.path.join(a.scratch, 'rc'), os.path.join(a.scratch, 'c.json')))

    res['pass'] = bool(res['clean'][0] == 0 and all(res[x]['result'][0] == 1 for x in 'abc'))
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
