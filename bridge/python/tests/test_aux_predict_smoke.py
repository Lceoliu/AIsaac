"""CPU tests of the predictive auxiliary loss (train_tok --aux-predict, isaac_bridge/tok_aux.py, rl/docs/
PREDICTIVE_MEMORY_DESIGN.md section 3). No game, no GPU. Plain script:
  cd rl/bridge/python && python -X utf8 tests/test_aux_predict_smoke.py [--quick]
Checks:
  (a) target tables by hand: small sequences with an episode end (a death), a rollout end, an open decision, a worker
      error (CLOSE_BROKE), a branch's truncation, a first record without a terminal: every value and mask spelled out;
  (b) the same tables against an independent brute-force reference on random sequences;
  (c) the events of a step: hurt, clear, new room, floor, death from close_*; a room change from the minimap as
      TemporalAttention.step decides it (a hidden map neither moves nor replaces the last one; none across episodes),
      pickups from the player columns or EV_ITEM; the NPC count counts real tokens only;
  (d) normalisation: running statistics == batch statistics over two rollouts; masked rows hold zeros; clip;
  (e) part_loss: the parts' shares add up to the minibatch's means (one part == any split), masked and zero-weight rows
      do not count, hand values of MSE / BCE / Huber;
  (f) the model: keep_o changes no output, o_seq is the chunk's o_t with gradient; L_aux's gradient reaches the temporal
      transformer and the spatial encoder, not the policy, value or danger heads; the head is not in the model's state;
  (g) off == the base commit, bit for bit: train_tok's real learn() under the CPU shim (tests/aux_learn_driver.py) in
      this tree and in an export of the base commit (AUX_BASE_REF, default abplus-training), --learner-fast 1 and 2:
      the rollout (real Episodes.arrive, store), the weights and Adam's state after two updates, progress.csv's columns
      and values (timing columns aside);
  (h) on: the same rollout as off (the actor does not see the head; close_* are the only new arrays), the same
      'model' keys, aux_* columns present and finite, the training loss falls on a repeated rollout; the time per update
      printed (indicative, CPU);
  (i) --resume: a checkpoint with the head restores it (head, Adam, normalisation); one without it (a C99 checkpoint)
      starts the head fresh with the policy exact; flags: --aux-predict without --temporal, with --learner-compile, a
      bad --aux-horizons are refused.
--quick skips (g) to (i).
"""
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402  (its path set-up, gymnasium stand-in, records)

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_aux as A   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge.tok_obs import EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM, KIND_COLUMN, N_ROOM_TYPE, ROW   # noqa: E402

FAIL = []
EV = {n: i for i, n in enumerate(A.EVENTS)}
ST_ = {n: i for i, n in enumerate(A.STATE)}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def fake_roll(n, cap):
    """A rollout's host arrays (what tok_aux reads), zero; rows' player column 0 = t * 0.01, room_type = t: identifiable."""
    r = types.SimpleNamespace(n=n, cap=cap, rows=np.zeros((n, cap), ROW), length=np.zeros(n, np.int64),
                              open=np.full(n, -1, np.int64), terminal=np.zeros((n, cap), bool),
                              boot=np.zeros((n, cap), bool), reward=np.zeros((n, cap), np.float32),
                              close_hurt=np.zeros((n, cap), np.float32), close_done=np.zeros((n, cap), np.uint8),
                              close_ev=np.zeros((n, cap), np.uint8))
    r.rows['player'][:, :, 0] = np.arange(cap)[None] * 0.01
    r.rows['room_type'][:] = np.arange(cap)[None]
    return r


def case_a():
    H = (1, 2, 4)
    r = fake_roll(5, 12)
    # worker 0: an episode dies at decision 3 (hurt), the next starts at 4; no open decision
    r.length[0] = 10
    r.rows['first'][0, [0, 4]] = 1
    r.terminal[0, 3] = True
    r.close_done[0, 3] = 2
    r.close_hurt[0, 3] = 1.0
    r.reward[0, :10] = [1, 2, 3, 4, 10, 20, 30, 40, 50, 60]
    # worker 1: 6 decisions, the last one open
    r.length[1], r.open[1] = 6, 5
    r.rows['first'][1, 0] = 1
    r.reward[1, :5] = 1
    # worker 2: a worker error at decision 2 (the next episode's first record closed it)
    r.length[2] = 6
    r.rows['first'][2, [0, 3]] = 1
    r.terminal[2, 2] = True
    r.close_done[2, 2] = A.CLOSE_BROKE
    r.reward[2, :6] = 1
    # worker 3: a branch's truncated end at decision 2
    r.length[3] = 3
    r.rows['first'][3, 0] = 1
    r.boot[3, 2] = True
    r.reward[3, :3] = 1
    # worker 4: another episode's first record at 3 without a terminal before it
    r.length[4] = 6
    r.rows['first'][4, [0, 3]] = 1
    r.reward[4, :6] = 1
    tb = A.rollout_tables(r, H)
    me, ms, ret, ev, st = tb['m_ev'], tb['m_state'], tb['ret'], tb['ev'], tb['state']
    Kx = {h: i for i, h in enumerate(H)}
    X = ST_['x']

    def val(w, t, h):
        k = Kx[h]
        return bool(me[w, t, k]), bool(ms[w, t, k]), float(ret[w, t, k]), ev[w, t, k], st[w, t, k]
    ok = []
    # (decision, horizon) -> event valid, state valid, return, death in the window, x of the state target
    for (w, t, h), (e_, s_, rr, dd, xx) in {
            (0, 0, 1): (True, True, 1, False, 0.01),
            (0, 3, 1): (True, False, 4, True, 0),        # the death's step: the state after it does not exist
            (0, 2, 2): (True, False, 7, True, 0),        # the window ends at the death: complete, the state masked
            (0, 1, 4): (True, False, 9, True, 0),        # truncated at the death: rewards 2 + 3 + 4
            (0, 3, 4): (True, False, 4, True, 0),
            (0, 0, 4): (True, False, 10, True, 0),       # steps 0..3: the death is the window's last step
            (0, 4, 4): (True, True, 100, False, 0.08),   # the next episode
            (0, 6, 4): (True, False, 180, False, 0),     # the window fits, row 10 does not exist
            (0, 7, 4): (False, False, 0, False, 0),      # runs off the rollout
            (0, 9, 1): (True, False, 60, False, 0),      # the step closed, no row after it
            (1, 3, 2): (True, True, 2, False, 0.05),     # the state target may be the open decision's row
            (1, 4, 1): (True, True, 1, False, 0.05),
            (1, 4, 2): (False, False, 0, False, 0),      # needs the open decision's step
            (1, 5, 1): (False, False, 0, False, 0),      # the open decision itself
            (2, 0, 2): (True, True, 2, False, 0.02),     # before the error
            (2, 0, 4): (False, False, 0, False, 0),      # the window reaches an end of unknown outcome
            (2, 2, 1): (False, False, 0, False, 0),
            (2, 3, 2): (True, True, 2, False, 0.05),     # after it
            (3, 1, 2): (False, False, 0, False, 0),      # reaches a branch's truncation
            (3, 2, 1): (False, False, 0, False, 0),      # the truncated record itself
            (3, 0, 2): (True, True, 2, False, 0.02),
            (4, 1, 2): (False, False, 0, False, 0),      # steps 1, 2: the next stored record starts another episode
            (4, 0, 2): (True, True, 2, False, 0.02)}.items():
        g_e, g_s, g_r, g_ev, g_st = val(w, t, h)
        good = g_e == e_ and g_s == s_ and abs(g_r - rr) < 1e-6 and bool(g_ev[EV['death']]) == dd and \
            abs(float(g_st[X]) - xx) < 1e-6
        if not good:
            print('   case', (w, t, h), 'got', g_e, g_s, g_r, bool(g_ev[EV['death']]), float(g_st[X]))
        ok.append(good)
    check('(a) hand-built sequences: values and masks at the episode, rollout and unknown ends', all(ok), f'{len(ok)} cases')
    check('(a) the death\'s hurt is seen (it is not in any stored row)',
          bool(ev[0, 2, Kx[2], EV['hurt']]) and not ev[0, 0, Kx[2], EV['hurt']])
    check('(a) past the longest worker\'s rows nothing is valid', not me[1, 6:].any() and not ms[1, 6:].any())


def brute(r, w, horizons, events):
    """An independent reference: walks each window step by step."""
    L, K = int(r.length[w]), len(horizons)
    U = L - 1 if r.open[w] == L - 1 else L
    out = dict(me=np.zeros((L, K), bool), ms=np.zeros((L, K), bool), ret=np.zeros((L, K)), ev=np.zeros((L, K, events.shape[1]), bool))

    def is_end(i):   # the step of decision i leaves the episode; returns None / 'real' / 'hard'
        nf = i + 1 < L and r.rows['first'][w, i + 1] != 0
        if r.boot[w, i] or r.close_done[w, i] == A.CLOSE_BROKE:
            return 'hard'
        if r.terminal[w, i]:
            return 'real'
        return 'hard' if nf else None
    for t in range(U):
        if r.boot[w, t]:
            continue
        for ki, h in enumerate(horizons):
            tot, evs, valid, ended = 0.0, np.zeros(events.shape[1], bool), True, False
            for i in range(t, t + h):
                if i >= U:
                    valid = False
                    break
                tot += float(r.reward[w, i])
                evs |= events[i]
                en = is_end(i)
                if en == 'hard':
                    valid = False
                    ended = True
                    break
                if en == 'real':
                    ended = True
                    break
            out['me'][t, ki] = valid
            if valid:
                out['ret'][t, ki], out['ev'][t, ki] = tot, evs
            out['ms'][t, ki] = not ended and t + h < L
    return out


def random_roll(rng, n=6, cap=70):
    r = fake_roll(n, cap)
    for w in range(n):
        L = int(rng.integers(0, cap - 2))
        r.length[w] = L
        r.reward[w, :L] = rng.normal(size=L)
        r.close_hurt[w, :L] = rng.random(L) < 0.1
        r.close_ev[w, :L] = rng.integers(0, 128, L) * (rng.random(L) < 0.2)
        if not L:
            continue
        r.rows['first'][w, 0] = 1
        i = 0
        while i < L - 1:
            i += int(rng.integers(1, 25))
            if i >= L:
                break
            kind = rng.integers(0, 5)
            if kind == 0:   # a time-out
                r.terminal[w, i], r.close_done[w, i] = True, 3
            elif kind == 1:   # a death
                r.terminal[w, i], r.close_done[w, i] = True, 2
            elif kind == 2:   # an error
                r.terminal[w, i], r.close_done[w, i] = True, A.CLOSE_BROKE
            elif kind == 3:   # a truncation
                r.boot[w, i] = True
            if kind <= 3 and i + 1 < L:
                r.rows['first'][w, i + 1] = 1
            elif kind == 4 and i + 1 < L:   # another episode without a terminal
                r.rows['first'][w, i + 1] = 1
        if rng.random() < 0.5:
            r.open[w] = L - 1
            r.terminal[w, L - 1] = r.boot[w, L - 1] = False
            r.close_done[w, L - 1] = 0
    return r


def case_b():
    rng = np.random.default_rng(3)
    H = (1, 3, 8, 32)
    bad = 0
    n_ok = 0
    for _ in range(150):
        r = random_roll(rng)
        tb = A.rollout_tables(r, H)
        for w in range(r.n):
            L = int(r.length[w])
            if not L:
                continue
            U = L - 1 if r.open[w] == L - 1 else L
            bnd = A.boundaries(L, r.terminal[w], r.boot[w], r.rows['first'][w, :L])
            evs = A.event_rows(r.rows[w, :L], U, r.close_hurt[w], r.close_done[w], r.close_ev[w], bnd)
            b = brute(r, w, H, evs)
            same = (tb['m_ev'][w, :L] == b['me']).all() and (tb['m_state'][w, :L] == b['ms']).all() and \
                np.allclose(tb['ret'][w, :L], b['ret'], atol=1e-5) and (tb['ev'][w, :L] == b['ev']).all()
            for t, ki in zip(*np.nonzero(b['ms'])):
                same = same and tb['state'][w, t, ki, ST_['room_type']] == t + H[ki]
            bad += not same
            n_ok += 1
    check('(b) the tables == a brute-force reference on random sequences', bad == 0, f'{n_ok} workers, {bad} differ')


def case_c():
    r = fake_roll(1, 16)
    L = 12
    r.length[0] = L
    rows = r.rows[0]
    rows['first'][[0, 8]] = 1
    # rooms: cell 10 at 0..2, cell 11 at 3, hidden at 4 and 5, cell 11 again at 6, cell 12 at 7; the next episode starts at
    # 8 in cell 13 (no room change across the end), hidden at 9, cell 13 at 10
    cells = [10, 10, 10, 11, 11, None, 11, 12, 13, None, 13, 13]
    for t, c in enumerate(cells):
        if c is not None:
            rows['map'][t, 3, c // 13, c % 13] = 1
    r.terminal[0, 7] = True
    r.close_done[0, 7] = 2
    r.close_hurt[0, [1, 7]] = 1.0
    r.close_ev[0, 1] = EV_CLEARED
    r.close_ev[0, 2] = EV_NEW_ROOM        # (the record at 3: a first visit)
    r.close_ev[0, 9] = EV_EXIT
    r.close_ev[0, 10] = EV_ITEM
    rows['player'][:, 7] = 3 / 12         # health units
    rows['player'][:, 10] = 5 / 50        # coins
    rows['player'][5:, 7] = 4 / 12        # a heart at 4 -> 5
    rows['player'][6:, 10] = 6 / 50       # a coin at 5 -> 6
    rows['player'][7:, 10] = 5 / 50       # a coin spent at 6 -> 7: no pickup
    bnd = A.boundaries(L, r.terminal[0], r.boot[0], rows['first'])
    ev = A.event_rows(rows, L, r.close_hurt[0], r.close_done[0], r.close_ev[0], bnd)
    want = {n: np.zeros(L, bool) for n in A.EVENTS}
    want['hurt'][[1, 7]] = True
    want['clear'][1] = True
    want['new_room'][2] = True
    want['room'][[2, 6, 9]] = True        # 2 -> 3 (map and bit), 6 -> 7, and the exit's step 9 -> 10
    want['floor'][9] = True
    want['death'][7] = True
    want['pickup'][[4, 5, 10]] = True
    for j, n in enumerate(A.EVENTS):
        check(f'(c) event {n}', (ev[:, j] == want[n]).all(), f'{np.flatnonzero(ev[:, j]).tolist()}')
    # 'room' at 2: from the map and the bit together; the moved map alone at 6 -> 7 counted above
    x = np.zeros(1, ROW)
    x['n_ent'] = 3
    x['ent'][0, :5, KIND_COLUMN + 6] = 1.0    # (columns past n_ent hold older values)
    x['ent'][0, 1, KIND_COLUMN + 6] = 0.0
    x['player'][0, [0, 1, 4, 5, 8, 9, 10]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    x['phearts'][0] = (0.8, 1.0, 0.9)
    x['room_type'], x['stage'] = 5, 2
    s = A.state_rows(x)[0]
    want_s = dict(x=0.1, y=0.2, hearts=0.3, soul=0.4, black=0.8, eternal=1.0, bone=0.9, coins=0.7, bombs=0.5, keys=0.6,
                  enemies=2, room_type=5, stage=2)
    check('(c) state fields, the NPC count over the real tokens only',
          all(abs(s[ST_[k]] - v) < 1e-6 for k, v in want_s.items()), str(s.tolist()))


def case_d():
    rng = np.random.default_rng(5)
    H = (1, 4)
    a, b = random_roll(rng), random_roll(rng)
    norm = A.AuxNorm(len(A.REG_COLS), len(H))
    device = 'cpu'
    all_raw, all_ret = [], [[] for _ in H]
    for r in (a, b):
        flat = np.flatnonzero((np.arange(r.cap)[None] < r.length[:, None]).reshape(-1))
        weight = np.ones(len(flat), np.float32)
        tb = A.rollout_tables(r, H)
        all_raw.append(tb['state_raw'])
        for ki in range(len(H)):
            all_ret[ki].append(tb['ret'][..., ki][tb['m_ev'][..., ki]])
        tg, log = A.build_targets(r, flat, weight, H, norm, device)
    raw = np.concatenate(all_raw).astype(np.float64)[:, A.REG_COLS]
    check('(d) running mean / std == the batch statistics over both rollouts',
          np.allclose(norm.mean, raw.mean(0), atol=1e-6) and
          np.allclose(norm.state_std(), np.sqrt(np.maximum(raw.var(0), 1e-4)), atol=1e-6))
    rs = np.array([np.sqrt(max((np.concatenate(v) ** 2).mean(), 1e-4)) for v in all_ret])
    check('(d) a scale\'s return std == its batch second moment', np.allclose(norm.ret_std(), rs, atol=1e-6))
    check('(d) masked rows hold zeros', float((tg['state'] * (1 - tg['m_state'])[..., None]).abs().sum()) == 0.0 and
          float((tg['ret'] * (1 - tg['m_ev'])).abs().sum()) == 0.0 and float((tg['ev'] * (1 - tg['m_ev'])[..., None]).sum()) == 0.0)
    check('(d) state targets clipped', float(tg['state'].abs().max()) <= A.CLIP)
    check('(d) the log has the masked shares in [0, 1]', all(0.0 <= log[f'aux_m{x}_{h}'] <= 1.0 for x in 'se' for h in H) and
          log['aux_ms_4'] >= log['aux_ms_1'], str({k: round(v, 3) for k, v in log.items() if k.startswith('aux_m')}))


def case_e():
    torch.manual_seed(9)
    K, S, E, n = 3, len(A.REG_COLS), len(A.EVENTS), 40
    head = A.make_head(8, K, hidden=16)
    o = torch.randn(n, 8)
    tg = dict(state=torch.randn(n, K, S), room=torch.randint(0, N_ROOM_TYPE, (n, K)), m_state=(torch.rand(n, K) < 0.7).float(),
              ev=(torch.rand(n, K, E) < 0.3).float(),
              ret=torch.randn(n, K) * 3, m_ev=(torch.rand(n, K) < 0.8).float())
    w = (torch.rand(n) < 0.9).float()
    sel = torch.arange(n)
    den = A.minibatch_den(tg, sel, w)
    one, st1, ev1 = A.part_loss(head, o, tg, sel, w, den)
    tot, st2, ev2 = 0.0, 0.0, 0.0
    for lo, hi in ((0, 7), (7, 25), (25, 40)):
        p = sel[lo:hi]
        l_, s_, e_ = A.part_loss(head, o[p], tg, p, w[p], den)
        tot, st2, ev2 = tot + l_, st2 + s_, ev2 + e_
    check('(e) the parts\' shares add up to the minibatch\'s means', abs(float(one - tot)) < 1e-5 and
          float((st1 - st2).abs().max()) < 1e-5 and float((ev1 - ev2).abs().max()) < 1e-5)
    tg2 = {k: v.clone() for k, v in tg.items()}
    tg2['state'] = tg2['state'] + 100 * (1 - tg['m_state'])[..., None] * (1 - 0)
    tg2['room'] = torch.where(tg['m_state'] > 0, tg2['room'], (tg2['room'] + 1) % N_ROOM_TYPE)
    tg2['ret'] = tg2['ret'] + 100 * (1 - tg['m_ev'])
    tg2['ev'] = torch.where(tg['m_ev'][..., None] > 0, tg2['ev'], 1 - tg2['ev'])
    l2, _, _ = A.part_loss(head, o, tg2, sel, w, den)
    check('(e) masked targets do not count', abs(float(l2 - one)) < 1e-5)
    o3 = o.clone()
    o3[w == 0] = 50.0
    l3, _, _ = A.part_loss(head, o3, tg, sel, w, den)
    check('(e) zero-weight rows do not count', abs(float(l3 - one)) < 1e-5)
    # hand values: a head that predicts zeros
    zero = A.make_head(8, 1, hidden=4)
    for q in zero.parameters():
        torch.nn.init.zeros_(q)
    t1 = dict(state=torch.full((2, 1, S), 2.0), room=torch.tensor([[5], [7]]), m_state=torch.tensor([[1.0], [0.0]]),
              ev=torch.zeros((2, 1, E)),
              ret=torch.tensor([[3.0], [0.5]]), m_ev=torch.ones(2, 1))
    t1['ev'][0, 0, 0] = 1.0
    w1 = torch.ones(2)
    d1 = A.minibatch_den(t1, torch.arange(2), w1)
    _, s1, e1 = A.part_loss(zero, torch.zeros(2, 8), t1, torch.arange(2), w1, d1)
    want_ret = (0.5 * 0.25 + (3.0 - 0.5)) / 2   # Huber(delta 1): 0.5 r^2 below 1, |r| - 0.5 above
    ce0 = float(np.log(N_ROOM_TYPE))
    want_reg = (S * 4.0 + ce0) / (S + 1)   # 12 columns of MSE 4, the room type's CE at uniform logits as the 13th
    check('(e) hand values: state mean with the room type\'s CE as one column, BCE at logit 0, Huber, the CE logged alone',
          abs(float(s1[0, 0]) - want_reg) < 1e-5 and abs(float(s1[1, 0]) - np.log(2.0)) < 1e-5 and
          abs(float(s1[2, 0]) - want_ret) < 1e-5 and abs(float(s1[3, 0]) - ce0) < 1e-5,
          f'{s1[:, 0].tolist()} want {want_reg:.4f}, {np.log(2.0):.4f}, {want_ret}, {ce0:.4f}')
    check('(e) per-event BCE shares average to the events\' mean', abs(float(e1.mean()) - float(s1[1, 0])) < 1e-5)


def case_f():
    rng = np.random.default_rng(41)
    cfg = dict(TS.SMALL, **TS.BASE, **TS.FUSED, **TS.TK)
    torch.manual_seed(5)
    m = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(m, 6)
    seq = TS.make_seq(rng)
    pend = [np.zeros((TS.NW, len(m.heads)), np.int64) for _ in seq]
    ra, _ = TS.new_rolls(m)
    st = TP.TemporalState(m, TS.NW, 'cpu', torch.float32)
    for t in range(16):
        rows = seq[t]
        b = TP.to_batch(rows, pend[t], 'cpu')
        td = dict(mode='step', state=st, slot=torch.arange(TS.NW))
        with torch.no_grad():
            a_, lp, v, vg = m.act(b, need_vg=True, temporal=td)
        import train_tok as TT
        TT.store(ra, np.arange(TS.NW, dtype=np.int64), rows, pend[t], a_.numpy(), lp.numpy(), v.numpy(), 0, None, vg.numpy(),
                 TP.temporal_out(td['out']))
    plan = TP.temporal_chunks(ra.length, np.arange(ra.cap)[None] < ra.length[:, None], 8, None)
    flat, tch = plan['pos'], plan['tch']
    td_ = TP.temporal_inputs(ra, tch)
    trow = TT.temporal_rows(ra, flat, ra.cap, 'cpu')
    table = m.tatt.room_table(ra.t_o, td_)
    rows_all = ra.rows.reshape(-1)[flat]
    pend_all = ra.pending.reshape(-1, len(m.heads))[flat]
    b = TP.to_batch(rows_all, pend_all, 'cpu')

    def chunk(keep):
        d = dict(mode='chunk', plan=TP.temporal_part(td_, np.arange(len(tch)), tch[:, 1], trow['ep'], table),
                 o_prev=trow['o_prev'], has=trow['has'])
        if keep:
            d['keep_o'] = True
        return m(b, need_vg=True, temporal=d), d
    with torch.no_grad():
        o0, d0 = chunk(False)
        o1, d1 = chunk(True)
    same = all(torch.equal(x, y) for x, y in zip(o0, o1))
    check('(f) keep_o changes no output, off leaves no o_seq', same and 'o_seq' not in d0)
    check('(f) o_seq has the chunk\'s rows, o_t\'s width', d1['o_seq'].shape == (len(flat), TS.TK['temporal_width']))
    # o_seq == the stored o_t: the same function (actor step by step, learner chunk)
    check('(f) o_seq == the actor\'s stored o_t', float((d1['o_seq'] - trow['o']).abs().max()) < 1e-4,
          f"max |diff| {float((d1['o_seq'] - trow['o']).abs().max()):.1e}")
    head = A.make_head(TS.TK['temporal_width'], 2, hidden=16)
    m.zero_grad(set_to_none=True)
    _, d2 = chunk(True)
    n = len(flat)
    tg = dict(state=torch.randn(n, 2, len(A.REG_COLS)), room=torch.randint(0, N_ROOM_TYPE, (n, 2)), m_state=torch.ones(n, 2),
              ev=torch.zeros(n, 2, len(A.EVENTS)), ret=torch.randn(n, 2), m_ev=torch.ones(n, 2))
    w = torch.ones(n)
    idx = torch.arange(n)
    L, _, _ = A.part_loss(head, d2['o_seq'], tg, idx, w, A.minibatch_den(tg, idx, w))
    L.backward()
    named = dict(m.named_parameters())

    def has_g(prefix):
        return any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n_, p_ in named.items() if n_.startswith(prefix))
    for pre in ('tatt.inp', 'tatt.layers', 'tatt.norm', 'tatt.feed', 'blocks', 'player', 'entity', 'gtok.tok'):
        check(f'(f) L_aux reaches {pre}', has_g(pre))
    for pre in ('pi.', 'value.', 'danger.', 'tatt.read', 'gtok.value'):
        check(f'(f) L_aux does not reach {pre}', not has_g(pre))
    check('(f) every head parameter has a gradient', all(q.grad is not None for q in head.parameters()))
    check('(f) the head is not part of the policy\'s state', not any('aux' in k for k in m.state_dict()))
    # learnable targets (a fixed function of each worker and step): the loss through o_t, the temporal transformer and
    # the spatial encoder falls far below the targets' variance
    g = torch.Generator().manual_seed(4)
    tg = dict(state=torch.randn(TS.NW, 2, len(A.REG_COLS), generator=g)[np.asarray(flat // ra.cap)],
              room=torch.randint(0, N_ROOM_TYPE, (TS.NW, 2), generator=g)[np.asarray(flat // ra.cap)], m_state=torch.ones(n, 2), ev=(torch.rand(TS.NW, 2, len(A.EVENTS), generator=g) < 0.5).float()[np.asarray(flat // ra.cap)],
              ret=torch.randn(TS.NW, 2, generator=g)[np.asarray(flat // ra.cap)], m_ev=torch.ones(n, 2))
    opt = torch.optim.Adam(list(head.parameters()) + list(m.parameters()), lr=2e-3)
    den = A.minibatch_den(tg, idx, w)
    hist = []
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        _, d3 = chunk(True)
        L, st_, _ = A.part_loss(head, d3['o_seq'], tg, idx, w, den)
        L.backward()
        opt.step()
        hist.append([float(x) for x in st_[:3].sum(1)] + [float(st_[3].sum())])
    a_, b_ = np.array(hist[0]), np.array(hist[-1])
    check('(f) learnable targets: state, events\' BCE, return and the room type\'s CE all fall clearly through o_t',
          b_[0] < 0.5 * a_[0] and b_[1] < 0.7 * a_[1] and b_[2] < 0.5 * a_[2] and b_[3] < 0.5 * a_[3],
          f'{a_.round(3)} -> {b_.round(3)}')


def case_arrays():
    """The close arrays' way to the learner process and between actor processes (what --learner-process 1 and
    --actor-processes 2 do with a rollout's host arrays; the allocator here plain numpy, memfds need Linux)."""
    import train_tok as TT

    def alloc(shape, dtype):
        return np.zeros(shape, dtype)
    r0, r1 = TT.Rollout(2, 8, 3, close=True), TT.Rollout(2, 8, 3, close=True)
    full = TT.full_from_local(r0, 4, 0, alloc)
    TT.slice_into(r1, full, 2)
    check('(arrays) full_from_local takes the close arrays', {'close_hurt', 'close_done', 'close_ev'} <= set(full))
    r0.close_hurt[1, 3], r1.close_done[0, 2], r1.close_ev[1, 5] = 1.5, 2, 8
    check('(arrays) what each actor writes lands in the shared arrays',
          full['close_hurt'][1, 3] == 1.5 and full['close_done'][2, 2] == 2 and full['close_ev'][3, 5] == 8)
    parts = [(0, TT.rollout_scalars(r0), None), (2, TT.rollout_scalars(r1), None)]
    view, _ = TT.merge_rollouts(full, parts, {}, 'cpu')
    check('(arrays) the merged view reads them', view.close_hurt[1, 3] == 1.5 and view.close_ev[3, 5] == 8)
    single = TT.Rollout(2, 8, 3, close=True)
    arrays = {k: v for k, v in vars(single).items() if isinstance(v, np.ndarray)}
    v2 = TT.rollout_view(arrays, TT.rollout_scalars(single), None)
    check('(arrays) the learner process\'s view reads them', v2.close_done is single.close_done)
    off = TT.Rollout(2, 8, 3)
    check('(arrays) off: no close arrays anywhere', off.close_hurt is None and off.close_done is None and off.close_ev is None
          and not any(k.startswith('close_') and isinstance(v, np.ndarray) for k, v in vars(off).items()))
    # Episodes.arrive: an open decision carried over a hand-over is closed in the new rollout
    rw = dict(room=1.0, explore=0.5, boss=2.0, exit=3.0, damage=1.0, hurt=0.75, death=2.0, timeout=1.0, time=0.002,
              heal=0.0, resource=0.0, clear=1.0)
    a, b = TT.Rollout(1, 8, 3, close=True), TT.Rollout(1, 8, 3, close=True)
    book = TT.Episodes(1, rw, True, run=True)
    x = np.zeros(1, ROW)
    x['first'] = 1
    pend = np.zeros((1, 3), np.int64)
    idx = np.zeros(1, np.int64)
    book.arrive(a, x, idx, pend)
    TT.store(a, idx, x, pend[idx], np.zeros((1, 3), np.int64), np.zeros(1, np.float32), np.zeros(1, np.float32), 0)
    b.carry(a)
    x = np.zeros(1, ROW)
    x['done'], x['hurt'], x['events'] = 2, 1.0, 1
    book.arrive(b, x, idx, pend)
    check('(arrays) a carried decision is closed in the new rollout: the death, its hurt, its events',
          b.close_done[0, 0] == 2 and b.close_hurt[0, 0] == 1.0 and b.close_ev[0, 0] == 1 and bool(b.terminal[0, 0]))
    # a broken episode
    a = TT.Rollout(1, 8, 3, close=True)
    x = np.zeros(1, ROW)
    x['first'] = 1
    book.arrive(a, x, idx, pend)
    TT.store(a, idx, x, pend[idx], np.zeros((1, 3), np.int64), np.zeros(1, np.float32), np.zeros(1, np.float32), 0)
    x = np.zeros(1, ROW)
    x['first'] = 1
    book.arrive(a, x, idx, pend)
    check('(arrays) an episode broken off: marked, terminal', a.close_done[0, 0] == A.CLOSE_BROKE and bool(a.terminal[0, 0]))


def export_base(ref, dst):
    py = HERE.parent.parent.parent
    tar = subprocess.run(['git', '-C', str(py), 'archive', ref, 'bridge/python', 'bridge/abplus/catalog', 'bridge/abplus/abp_bridge.lua'],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dst, filter='data')


def driver(tree, out, *extra):
    cmd = [sys.executable, '-X', 'utf8', str(HERE / 'aux_learn_driver.py'), '--tree', str(tree), '--out', str(out)] + list(extra)
    env = dict(os.environ, OMP_NUM_THREADS='2')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=900)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-3000:])
        raise RuntimeError(f'driver failed: {extra}')
    return torch.load(Path(out) / 'result.pt', weights_only=False), p.stdout


TIMING = ('collect_s', 'update_s', 'wall_s', 'x_real_time', 'infer_ms', 'infer_upd_ms', 'cpu_update', 'cpu_collect',
          'u_log_prev', 'u_ckpt_prev', 'c_trainer_cores')


def same_tensors(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def same_opt(a, b):
    if len(a['state']) != len(b['state']):
        return False
    for k in a['state']:
        for n in a['state'][k]:
            x, y = a['state'][k][n], b['state'][k][n]
            if torch.is_tensor(x) and not torch.equal(x, y):
                return False
    return True


def timing_col(k):
    return k in TIMING or k.startswith('u_') or k.startswith('p_') or k.startswith('w_')


def case_g_h_i():
    tmp = Path(tempfile.mkdtemp(prefix='aux-test-', dir=os.environ.get('AUX_TEST_TMP') or None))
    ref = os.environ.get('AUX_BASE_REF', 'abplus-training')
    base = tmp / 'base'
    export_base(ref, base)
    here_tree = HERE.parent.parent.parent
    results = {}
    for fast in (1, 2):
        r_b, _ = driver(base, tmp / f'b{fast}', '--updates', '2', '--fast', str(fast))
        r_n, _ = driver(here_tree, tmp / f'n{fast}', '--updates', '2', '--fast', str(fast))
        results[fast] = r_n
        same_r = list(r_b['rollout']) == list(r_n['rollout']) and all(
            np.array_equal(r_b['rollout'][k], r_n['rollout'][k]) for k in r_b['rollout'])
        check(f'(g) off == {ref} [--learner-fast {fast}]: the rollout (Episodes.arrive, store), bit for bit', same_r)
        check(f'(g) off == {ref} [--learner-fast {fast}]: the weights before', same_tensors(r_b['before'], r_n['before']))
        check(f'(g) off == {ref} [--learner-fast {fast}]: the weights after two updates, bit for bit',
              same_tensors(r_b['weights'], r_n['weights']))
        check(f'(g) off == {ref} [--learner-fast {fast}]: Adam\'s state', same_opt(r_b['opt'], r_n['opt']))
        cb, cn = r_b['progress'], r_n['progress']
        check(f'(g) off == {ref} [--learner-fast {fast}]: progress.csv\'s columns', list(cb[0]) == list(cn[0]))
        diff = sorted({k for x, y in zip(cb, cn) for k in x if x[k] != y.get(k) and not timing_col(k)})
        check(f'(g) off == {ref} [--learner-fast {fast}]: progress.csv\'s values (timing aside)', not diff, str(diff[:8]))
        # the update really moved the weights
        check(f'(g) [--learner-fast {fast}] the update moves the weights', not same_tensors(r_n['before'], r_n['weights']))
    for fast in (1, 2):
        r_on, out_on = driver(here_tree, tmp / f'on{fast}', '--updates', '2', '--fast', str(fast), '--aux', '1')
        r_off = results[fast]
        extra = sorted(set(r_on['rollout']) - set(r_off['rollout']))
        same_roll = all(np.array_equal(r_off['rollout'][k], r_on['rollout'][k]) for k in r_off['rollout'])
        check(f'(h) on [--learner-fast {fast}]: the same rollout as off; only close_* added', same_roll and
              extra == ['close_done', 'close_ev', 'close_hurt'], str(extra))
        check(f'(h) on [--learner-fast {fast}]: the model\'s keys and start weights as off',
              same_tensors(r_off['before'], r_on['before']))
        check(f'(h) on [--learner-fast {fast}]: the policy is trained differently from off',
              not same_tensors(r_off['weights'], r_on['weights']))
        row = r_on['progress'][-1]
        cols = [f'aux_{x}_{h}' for x in ('reg', 'ev', 'ret', 'room_ce', 'ms', 'me', 'ev0') for h in (1, 4, 16)] + \
            ['aux_loss', 'aux_grad_norm', 'u_aux'] + [f'aux_rate_{e}_{h}' for e in A.EVENTS for h in (1, 4, 16)] + \
            [f'aux_bce_{e}_{h}' for e in A.EVENTS for h in (1, 4, 16)]
        missing = [c for c in cols if c not in row or row[c] == '']
        check(f'(h) on [--learner-fast {fast}]: progress.csv has the aux columns', not missing, str(missing[:6]))
        fin = all(np.isfinite(float(row[c])) for c in cols if c not in missing)
        check(f'(h) on [--learner-fast {fast}]: they are finite', fin)
        check(f'(h) on [--learner-fast {fast}]: masked shares grow with the scale',
              float(row['aux_me_16']) >= float(row['aux_me_1']) and float(row['aux_ms_16']) >= float(row['aux_ms_1']),
              f"events {row['aux_me_1']} .. {row['aux_me_16']}, state {row['aux_ms_1']} .. {row['aux_ms_16']}")
        print(f"info: CPU shim, one update of the tiny test model [--learner-fast {fast}]: off {r_off['learn_s']:.3f} s, "
              f"on {r_on['learn_s']:.3f} s (x{r_on['learn_s'] / r_off['learn_s']:.2f}; the tiny model's budget, not the "
              f"real one: the time gate is the CUDA run), tables {float(row['u_aux']) * 1000:.1f} ms")
    r_long, _ = driver(here_tree, tmp / 'long', '--updates', '12', '--aux', '1', '--steps', '96')
    p0, p1 = float(r_long['progress'][0]['aux_loss']), float(r_long['progress'][-1]['aux_loss'])
    check('(h) on: L_aux falls over 12 updates of the same rollout', p1 < p0, f'{p0:.3f} -> {p1:.3f}')
    ev = [float(r_long['progress'][-1][f'aux_ev_{h}']) for h in (1, 4, 16)]
    ev0 = [float(r_long['progress'][-1][f'aux_ev0_{h}']) for h in (1, 4, 16)]
    check('(h) on: the events\' BCE ends near or below the base rates\' (a sane scale)',
          all(a_ < 2 * b_ + 0.05 for a_, b_ in zip(ev, ev0)), f'bce {ev} base {ev0}')
    # ---- (i)
    ck_on = r_long['last']
    r_res, out_res = driver(here_tree, tmp / 'res', '--updates', '1', '--aux', '1', '--steps', '96', '--resume', ck_on)
    ck = torch.load(ck_on, map_location='cpu', weights_only=False)
    check('(i) a checkpoint keeps the head, its Adam and the normalisation apart from the model',
          set(ck['aux']) == {'head', 'opt', 'norm'} and not any('aux' in k for k in ck['model']))
    check('(i) --resume with the head: no fresh start', 'it starts fresh' not in out_res)
    check('(i) the resumed loss goes on from the end, not from the start',
          float(r_res['progress'][-1]['aux_loss']) < 0.9 * p0,
          f"{p0:.3f} at the start, {p1:.3f} at the end, {float(r_res['progress'][-1]['aux_loss']):.3f} resumed")
    ck_c99 = results[1]['last']
    r_fresh, out_fresh = driver(here_tree, tmp / 'c99', '--updates', '1', '--aux', '1', '--resume', ck_c99)
    c0 = torch.load(ck_c99, map_location='cpu', weights_only=False)
    check('(i) --resume of a checkpoint without the head (C99): it starts fresh', 'it starts fresh' in out_fresh and
          'aux' not in c0)
    check('(i) the policy\'s weights resumed exact', same_tensors(c0['model'], r_fresh['before']))
    refused = 0
    tree = here_tree / 'bridge' / 'python'
    for i_, (extra, needle) in enumerate(((['--aux-predict', '1'], 'needs --temporal 1'),
                                          (['--aux-predict', '1', '--temporal', '1', '--learner-compile', 'default'],
                                           'learner-compile'),
                                          (['--aux-predict', '1', '--temporal', '1', '--aux-horizons', '8,1'], 'aux-horizons'))):
        code = ('import sys; sys.path.insert(0, "tests"); import test_memory_smoke; import torch; '
                'sys.path.insert(0, "."); import train_tok; '
                f'train_tok.main(["--groups-file", "x", "--out", {str(tmp / f"ref{i_}")!r}, "--mode", "run", "--items"] + {extra!r})')
        p = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], capture_output=True, text=True, encoding='utf-8',
                           errors='replace', cwd=str(tree))
        refused += p.returncode != 0 and needle in p.stderr
    check('(i) the bad flag combinations are refused', refused == 3, f'{refused} of 3')
    print('info: test directory', tmp)


def main():
    case_a()
    case_b()
    case_c()
    case_d()
    case_e()
    case_f()
    case_arrays()
    if '--quick' not in sys.argv:
        case_g_h_i()
    print('RESULT', 'all checks passed' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
