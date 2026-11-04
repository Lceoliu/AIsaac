"""CPU tests of the offline world-model tool (abplus_world_model.py, rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 5). No game, no GPU.
Plain script:  cd rl/bridge/python && python -X utf8 tests/test_world_model_smoke.py [<stream dir with ROW records>]
Checks:
  (a) segments: an episode change, a gap in t, a first record cut; events from the stream's fields;
  (b) windows against a brute force on a hand-made log: event targets and masks at a real end, at an unknown end, padding;
      fields at t + k; the held-out sets are disjoint and a held-out seed never trains;
  (c) AUC against the pair count, with ties; the verdict's line;
  (d) causality: an input at t + 1 never changes an output at <= t (wm, latent), and base reads only its own frame;
  (e) learning: an event the current frame shows is learned by wm and base; one that earlier frames show, by wm only;
  (f) optional, with a real log: events from the stream == tok_aux.event_rows on the logged ROW records.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402,F401  (its path set-up, gymnasium stand-in)

import numpy as np   # noqa: E402
import torch   # noqa: E402
from types import SimpleNamespace   # noqa: E402

import abplus_world_model as W   # noqa: E402
from isaac_bridge import tok_stream as L   # noqa: E402
from isaac_bridge.tok_aux import EVENTS, STATE   # noqa: E402

FAIL = []
WIDTH, HEADS = 8, 2


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def bf(x):
    """float array -> bfloat16 raw bits as int16."""
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).view(torch.int16).numpy()


def make_log(root, episodes, width=WIDTH, heads=HEADS):
    """episodes: list of dict(worker, seed, episode, n, done (close_done of the last), terminal, t0, hurt_at, z (optional fn)). Written
    through the real writer's format: parts of raw records, one worker per file."""
    import zstandard
    root = Path(root)
    root.mkdir(parents=True)
    dt = L.stream_dtype(width, heads)
    (root / 'format.json').write_text(json.dumps(dict(width=width, heads=heads)))
    by = {}
    for ep in episodes:
        n = ep['n']
        r = np.zeros(n, dt)
        r['worker'], r['seed'], r['episode'] = ep['worker'], ep['seed'], ep['episode']
        r['t'] = np.arange(n) + ep.get('t0', 0)
        r['first'][0] = int(ep.get('t0', 0) == 0)
        r['room'] = 7
        r['stage'] = 1
        r['update'] = ep.get('update', 5)
        r['state'][:, 0] = np.arange(n) / 100.0
        r['close_hurt'][list(ep.get('hurt_at', ()))] = 1.0
        r['close_done'][-1] = ep.get('done', 0)
        r['terminal'][-1] = int(ep.get('terminal', False))
        if 'z' in ep:
            r['z'] = bf(ep['z'])
        if 'action' in ep:
            r['action'] = ep['action']
        by.setdefault(ep['worker'], []).append(r)
    for w, parts in by.items():
        with open(root / f'stream-w{w:03d}-p0000.zst', 'wb') as f:
            f.write(zstandard.ZstdCompressor().compress(np.concatenate(parts).tobytes()))
    return root


def run_prep(roots, out, **kw):
    W.prep(SimpleNamespace(roots=roots, out=out, update_min=kw.get('update_min', 0), update_max=kw.get('update_max', 1 << 31), min_length=kw.get('min_length', 2)))
    return W.Data(out)


def case_a():
    dt = L.stream_dtype(WIDTH, HEADS)
    r = np.zeros(10, dt)
    r['episode'] = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    r['t'] = [0, 1, 2, 0, 1, 3, 4, 0, 1, 0]
    r['first'] = [1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    check('(a) segments', W.split_segments(r) == [(0, 3), (3, 5), (5, 7), (7, 9), (9, 10)], str(W.split_segments(r)))
    r = np.zeros(6, dt)
    r['room'] = [3, 3, 255, 4, 4, 5]
    r['close_ev'] = [0, 1, 0, 2, 8, 16]
    r['close_hurt'] = [0, 2, 0, 0, 0, 0]
    r['close_done'] = [0, 0, 0, 0, 0, 2]
    r['state'][:, W.S_COL['coins']] = [0, 0, 0, .02, .02, .02]
    e = W.stream_events(r)
    c = W.E_COL
    check('(a) hurt, clear', e[:, c['hurt']].tolist() == [0, 1, 0, 0, 0, 0] and e[:, c['clear']].tolist() == [0, 1, 0, 0, 0, 0])
    check('(a) room: across a record without a map', e[:, c['room']].tolist() == [0, 0, 1, 1, 1, 0], str(e[:, c['room']].astype(int).tolist()))
    check('(a) pickup: a gain, an item', e[:, c['pickup']].tolist() == [0, 0, 1, 0, 0, 1], str(e[:, c['pickup']].astype(int).tolist()))
    check('(a) new room, floor, death', e[:, c['new_room']].tolist() == [0, 0, 0, 1, 0, 0] and e[:, c['floor']].tolist() == [0, 0, 0, 0, 1, 0]
          and e[:, c['death']].tolist() == [0, 0, 0, 0, 0, 1])


def brute(n, hurt_at, real_end, start, L_, horizons):
    """The module's definition, written directly."""
    K, ev, m_ev, st, m_st = len(horizons), np.zeros((L_, len(horizons)), bool), np.zeros((L_, len(horizons)), bool), np.zeros((L_, len(horizons))), np.zeros((L_, len(horizons)), bool)
    for i in range(L_):
        t = start + i
        if t >= n:
            continue
        for ki, k in enumerate(horizons):
            last_step = t + k - 1
            if last_step < n or real_end:
                m_ev[i, ki] = True
                ev[i, ki] = any(t <= h <= last_step for h in hurt_at)
            if t + k < n:
                m_st[i, ki] = True
                st[i, ki] = (t + k) / 100.0
    return ev, m_ev, st, m_st


def case_b(tmp):
    root = make_log(tmp / 'b', [dict(worker=0, seed=11, episode=0, n=40, done=2, terminal=True, hurt_at=(5, 30, 39)),
                                dict(worker=0, seed=12, episode=1, n=30, done=0, terminal=False, hurt_at=(2, 29)),
                                dict(worker=1, seed=13, episode=0, n=3, done=3, terminal=True, hurt_at=(1,))])
    d = run_prep([root], tmp / 'bd')
    check('(b) segments and ends', d.segs[:, 1].tolist() == [40, 30, 3] and d.segs[:, 6].tolist() == [1, 0, 1], str(d.segs.tolist()))
    hz = (1, 4, 16)
    good = True
    cases = [(0, 40, (5, 30, 39), True), (1, 30, (2, 29), False), (2, 3, (1,), True)]
    for seg, n, hurt, end in cases:
        for start in (0, 1, 10, n - 2):
            if start < 0 or start >= n:
                continue
            w = d.window(seg, start, 8, hz)
            ev, m_ev, st, m_st = brute(n, hurt, end, start, 8, hz)
            hi = W.E_COL['hurt']
            good &= np.array_equal(w['m_ev'], m_ev) and np.array_equal(w['t_ev'][:, :, hi] & m_ev, ev) and np.array_equal(w['m_st'], m_st)
            good &= bool(np.allclose(w['t_st'][:, :, 0], st, atol=1e-5))
            good &= int(w['live'].sum()) == min(8, n - start)
    check('(b) windows == brute force (real end, unknown end, padding)', good)
    # the split: disjoint; a held-out seed's every episode is out
    eps = [dict(worker=0, seed=100 + (i % 20), episode=i, n=5, done=2, terminal=True) for i in range(200)]
    d2 = run_prep([make_log(tmp / 'b2', eps)], tmp / 'b2d')
    tr, he, hs = d2.split(1)
    check('(b) the split is a partition', len(tr) + len(he) + len(hs) == 200 and not (set(tr) & set(he) or set(tr) & set(hs) or set(he) & set(hs)))
    check('(b) held-out seeds never train', not set(d2.segs[hs, 2]) & set(d2.segs[tr, 2]) and len(hs) > 0)
    check('(b) held-out episodes: training seeds', len(he) > 0 and set(d2.segs[he, 2]) <= set(d2.segs[tr, 2]))
    # --update-min drops older records
    d3 = run_prep([make_log(tmp / 'b3', [dict(worker=0, seed=1, episode=0, n=9, update=3), dict(worker=0, seed=2, episode=1, n=7, update=8)])], tmp / 'b3d', update_min=5)
    check('(b) --update-min', d3.meta['n'] == 7)


def case_c():
    rng = np.random.default_rng(0)
    for name, s in (('continuous', rng.random(300)), ('ties', rng.integers(0, 4, 300).astype(float))):
        y = rng.random(300) < 0.3
        pos, neg = s[y], s[~y]
        want = float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
        got = W.auc(s, y)
        check(f'(c) AUC == pair count, {name}', abs(got - want) < 1e-9, f'{got:.6f} {want:.6f}')
    def res(a, b):
        return {k: dict(events={e: dict(auc=a, brier=b) for e in W.VERDICT_EVENTS}) for k in ('32', '64')}
    check('(c) verdict passes above the line', W.verdict(res(.8, .1), res(.5, .2))['ok'])
    v = W.verdict(res(.74, .1), res(.5, .2))
    check('(c) verdict: AUC below 0.75 fails', not v['ok'] and len(v['failed']) == 3)
    v = W.verdict(res(.9, .2), res(.5, .2))
    check('(c) verdict: Brier equal to the baseline fails', not v['ok'] and len(v['failed']) == 3)


META = dict(width=WIDTH, heads=HEADS, action_card=[3, 3])
NORM = dict(mu=[0.0] * len(STATE), sd=[1.0] * len(STATE))


def fake_batch(B=2, T=16):
    return dict(z=torch.from_numpy(bf(np.random.default_rng(1).normal(size=(B, T, WIDTH)))), g=torch.from_numpy(bf(np.random.default_rng(2).normal(size=(B, T, WIDTH)))),
                state=torch.from_numpy(np.random.default_rng(3).normal(size=(B, T, len(STATE))).astype(np.float32)), action=torch.zeros(B, T, HEADS, dtype=torch.int8),
                live=torch.ones(B, T))


def case_d():
    for arm in ('wm', 'latent', 'base'):
        torch.manual_seed(0)
        m = W.ARMS[arm](META, NORM, width=32, layers=2, heads=4, window=16, K=3).eval()
        b = fake_batch()
        torch.manual_seed(5)
        (e0, s0), _ = m(b)
        b2 = {k: v.clone() for k, v in b.items()}
        b2['state'][:, 9] += 3.0
        b2['action'][:, 9] = 2
        torch.manual_seed(5)
        (e1, s1), _ = m(b2)
        check(f'(d) {arm}: outputs up to t unchanged by t + 1', bool(torch.equal(e0[:, :9], e1[:, :9]) and torch.equal(s0[:, :9], s1[:, :9])))
        check(f'(d) {arm}: the changed step itself is seen', not torch.equal(e0[:, 9], e1[:, 9]))
        if arm == 'base':
            check('(d) base: later steps do not remember it', bool(torch.equal(e0[:, 10:], e1[:, 10:])))
        else:
            check(f'(d) {arm}: later steps remember it', not torch.equal(e0[:, 10:], e1[:, 10:]))


def learned(tmp, name, arm, data, horizons, steps=600):
    out = tmp / f'run-{name}-{arm}'
    W.train(SimpleNamespace(data=str(data), arm=arm, out=str(out), steps=steps, batch=16, window=32, width=32, layers=2, heads=4, lr=2e-3, horizons=horizons, eval_every=steps,
                            eval_windows=200, eval_batch=16, seed=1, device='cpu'))
    return json.loads((out / 'results.json').read_text())['final']['episodes']


def case_e(tmp):
    # 'now': frame t shows the hurt of step t: the current frame is enough (k = 1).
    # 'past': a pulse in frame t, the hurt at step t + 8; 'a hurt within 16 steps' needs the pulses of the last 8 frames, which only
    # a memory has (the current frame shows one of them)
    rng = np.random.default_rng(0)
    for name, lag, rate, hz, key in (('now', 0, 0.15, '1,4', '1'), ('past', 8, 0.03, '1,16', '16')):
        eps = []
        for i in range(120):
            n = 96
            h = rng.random(n) < rate
            z = rng.normal(0, 0.3, (n, WIDTH))
            z[:, 0] = np.roll(h, -lag) * 2.0   # (frame t shows the hurt of step t + lag)
            if lag:
                z[-lag:, 0] = 0.0
            eps.append(dict(worker=i % 4, seed=1000 + i, episode=i, n=n, done=2, terminal=True, hurt_at=np.flatnonzero(h).tolist(), z=z))
        d = tmp / f'e-{name}-data'
        run_prep([make_log(tmp / f'e-{name}', eps, width=WIDTH)], d)
        a = learned(tmp, name, 'wm', d, hz)[key]['events']['hurt']['auc']
        b = learned(tmp, name, 'base', d, hz)[key]['events']['hurt']['auc']
        if lag:
            check(f'(e) {name}: wm reads the past, base cannot', a > b + 0.1, f'wm {a:.3f} base {b:.3f}')
        else:
            check(f'(e) {name}: both read the frame', a > 0.9 and b > 0.9, f'wm {a:.3f} base {b:.3f}')


def case_f(root):
    from isaac_bridge.tok_aux import event_rows, boundaries, state_rows
    root = Path(root)
    fmt = json.loads((root / 'format.json').read_text())
    row_rate = fmt['row_rate'] if 'row_rate' in fmt else 0.02
    rows = {}
    for w, r in L.iter_records(root, 'row'):
        rows.setdefault(w, []).append(r)
    agree = {e: [0, 0, 0] for e in EVENTS}
    n_checked = 0
    same_state = True
    for w, r_parts in sorted(rows.items()):
        rec = np.concatenate([x for _, x in L.iter_records(root, 'stream', worker=w)])
        rr = np.concatenate(r_parts)
        # (the ROW set: the stream records of the episodes under row_rate, in log order; seeds repeat, so (seed, t) is no key)
        sel = rec[L.episode_hash(rec['worker'], rec['seed'], rec['episode']) < int(row_rate * L.SCALE)]
        if len(sel) != len(rr) or not np.array_equal(sel['t'], rr['row']['t']):
            check(f'(f) worker {w}: the ROW records line up with the stream', False, f'{len(sel)} {len(rr)}')
            continue
        for a, b in W.split_segments(sel):
            seg, row = sel[a:b], rr['row'][a:b]
            if len(seg) < 2:
                continue
            same_state &= bool(np.allclose(state_rows(row), seg['state']))
            mine = W.stream_events(seg)
            bnd = boundaries(len(seg), seg['terminal'].astype(bool), np.zeros(len(seg), bool), row['first'])
            theirs = event_rows(row, len(seg), seg['close_hurt'], seg['close_done'], seg['close_ev'], bnd)
            for i, e in enumerate(EVENTS):
                agree[e][0] += int((mine[:, i] == theirs[:, i]).sum())
                agree[e][1] += len(seg)
                agree[e][2] += int(theirs[:, i].sum())
            n_checked += 1
    check('(f) segments with ROW records found', n_checked > 0, str(n_checked))
    check('(f) the stream state == state_rows(ROW)', same_state)
    for e in EVENTS:
        v = agree[e]
        check(f'(f) {e} == event_rows', v[1] > 0 and v[0] / v[1] > 0.9995, f'{v[0]} of {v[1]}, positives {v[2]}')


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        case_a()
        case_b(tmp)
        case_c()
        case_d()
        case_e(tmp)
        if len(sys.argv) > 1:
            case_f(sys.argv[1])
    print('FAILED ' + ', '.join(FAIL) if FAIL else 'ALL OK')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
