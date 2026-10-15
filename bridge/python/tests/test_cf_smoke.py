"""CPU smoke test of B34 (2026-10-16): the counterfactual branch teacher (isaac_bridge/tok_cf.py, train_tok --cf 1,
rl/docs/COUNTERFACTUAL_TEACHER.md) and the lanes' temporal state (TemporalState.copy_slot_ / reset_slot_, the stored
outputs of imitation records: tok_policy temporal 'stored' m / 'tail').

No game, no GPU needed. Plain script:
  cd rl/bridge/python && python tests/test_cf_smoke.py
Checks:
  (a) choice-point detection on synthetic records: every trigger type (pickup classes, pedestal, locked door with /
      without a key, shop / treasure doors, a cleared room with 1 / 2 doors), first sight only (the room's seen set),
      the order and targets;
  (b) the quota: the token bucket's rate and capacity, "unseen trigger types first" (a fresh type replaces the
      episode's oldest pending point of a type seen before), skips; Quota.choose;
  (c) the target intent: the move toward a pickup / a door, arrival (one fewer of the class, the class gone, the room
      changed), the cap, the other heads untouched; the sticky intent's high-temperature re-picks (eps 0 = teacher v2's
      draws);
  (d) branch scoring arithmetic: returns_to_go (discount, the cut's bootstrap), taken_score (cap bootstrap with the
      value or value_global, a terminal end, exit_end, the heal / resource extra), gains_of / gain_reward, judge (the
      margin and the replicate's spread), cf_search with stand-in branches (the replicate only above the margin, the
      improvement test, the intents: target + sticky up to K - 1);
  (e) the distillation records: records_of with steps (the first k only), weight min(gain, cap), kind / trigger /
      intent / j / n meta; t_pack / t_split round trip;
  (f) LanePolicy: the request's sequence number, a stale answer skipped, value_global, the temporal begin, V2Stop;
  (g) TemporalState.copy_slot_ exactness (every per-slot tensor, between states, several slots), reset_slot_;
  (h) a temporal model's lane step from a copied slot == the parent's continuation, step by step (eager, float32):
      logits, value, value_global, o, the state after; the stored outputs (o_prev, has, o) of the lane steps reproduce
      the lane's logits in 'stored' mode, and as a 'tail' behind other records (m = 0 rows: as without temporal);
  (i) the shared blocks (cf_sizes / attach_cf) attach with the expected shapes.
"""
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402  (its path set-up: the package, the gymnasium stand-in)
import test_temporal_smoke as TS   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_cf as CF   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import tok_teacher2 as T2   # noqa: E402
from isaac_bridge.tok_branch import parse_reward, step_reward   # noqa: E402
from isaac_bridge.tok_obs import EV_CLEARED, EV_EXIT, EV_NEW_ROOM, ROW   # noqa: E402
from isaac_bridge.tok_pickup import move_toward   # noqa: E402

FAIL = []
FLOOR_REWARD = 'damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002,heal=0.5,resource=0.25'


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def empty_row():
    r = np.zeros(1, ROW)
    return r


def put_ent(r, j, typ, var, sub=0, dx=0.0, dy=0.0, item=0):
    e = r[0]
    e['ent_id'][j] = (typ, var, sub)
    e['ent'][j, 0], e['ent'][j, 1], e['ent'][j, 2] = dx, dy, np.hypot(dx, dy) * 200 / 300
    e['ent_item'][j, 0] = item
    r['n_ent'] = max(int(r['n_ent'][0]), j + 1)


def put_door(r, j, dx=0.0, dy=0.0, open_=1.0, locked=0.0, rtype=1):
    d = r[0]['doors'][j]
    d[0], d[1], d[2], d[3], d[4] = dx, dy, open_, locked, rtype / 30
    r['n_doors'] = max(int(r['n_doors'][0]), j + 1)


def names(found):
    return [CF.TRIGGERS[t] for t, _ in found]


def test_detect():
    rng = np.random.default_rng(1)
    r = empty_row()
    for j, (var, cls) in enumerate(((10, 'heart'), (20, 'coin'), (30, 'key'), (40, 'bomb'), (60, 'chest'),
                                    (70, 'pill'), (300, 'card'), (350, 'trinket'))):
        put_ent(r, j, 5, var, 1, dx=0.1 * (j + 1))
    put_ent(r, 8, 5, 100, 0, item=0)   # an empty pedestal: not a trigger
    put_ent(r, 9, 10, 0, 0)            # a monster
    seen = set()
    f = CF.detect(r, seen, rng)
    check('(a) every pickup class found once, in TRIGGERS order',
          names(f) == ['heart', 'coin', 'key', 'bomb', 'chest', 'pill', 'card', 'trinket'], str(names(f)))
    check('(a) pickup targets are (ent, class)', all(tg == ('ent', CF.TRIGGERS[t]) for t, tg in f))
    check('(a) first sight only: the same record again finds nothing', CF.detect(r, seen, rng) == [])
    put_ent(r, 8, 5, 100, 0, item=45)
    f = CF.detect(r, seen, rng)
    check('(a) a pedestal holding an item', names(f) == ['pedestal'] and f[0][1] == ('ent', 'pedestal'), str(f))
    r2 = empty_row()
    put_ent(r2, 0, 5, 100, 7)   # (no items mode: subtype != 0)
    put_ent(r2, 1, 5, 360, 0)   # red chest
    check('(a) pedestal by subtype, red chest', names(CF.detect(r2, set(), rng)) == ['chest', 'pedestal'])
    # doors
    r3 = empty_row()
    put_door(r3, 0, dx=-0.5, dy=0.0, open_=0.0, locked=1.0)
    put_door(r3, 1, dx=0.2, dy=-0.6, rtype=2)
    put_door(r3, 2, dx=0.6, dy=0.1, rtype=4)
    r3['player'][0, 9] = 0.0
    f = CF.detect(r3, set(), rng)
    check('(a) shop / treasure doors (no key: no locked trigger)', names(f) == ['shop', 'treasure'] and
          f[0][1] == ('door', 1) and f[1][1] == ('door', 2), str(f))
    r3['player'][0, 9] = 0.1
    f = CF.detect(r3, set(), rng)
    check('(a) a locked door with a key: its target', 'locked' in names(f) and dict(
        (CF.TRIGGERS[t], tg) for t, tg in f)['locked'] == ('door', 0), str(f))
    r4 = empty_row()
    put_door(r4, 0, open_=1.0)
    r4['events'] = EV_CLEARED
    check('(a) cleared with one door: none', CF.detect(r4, set(), rng) == [])
    put_door(r4, 1, open_=1.0)
    put_door(r4, 2, open_=0.0)
    f = CF.detect(r4, set(), rng)
    check('(a) cleared with >= 2 doors: an open door as target', names(f) == ['cleared'] and f[0][1][0] == 'door'
          and f[0][1][1] in (0, 1), str(f))
    check('(a) random has a TRIGGERS index', CF.TI['random'] == len(CF.TRIGGERS) - 1)


def test_quota():
    q = CF.Quota(6.0, 4)
    per_point = 3600 * 30 / 4 / 6   # decisions per point at 6 per game hour
    q.tick(int(per_point) - 10)
    check('(b) no token before 1 / 6 game hour', q.budget < 1.0, f'{q.budget:.3f}')
    q.tick(20)
    check('(b) one token after 1 / 6 game hour', 1.0 <= q.budget < 1.01, f'{q.budget:.3f}')
    q.tick(10 ** 6)
    check('(b) the bucket holds CF_BURST tokens at most', q.budget == CF.CF_BURST)
    q.begin_episode()
    ok, victim, fresh = q.offer(CF.TI['coin'])
    q.pending.append(dict(trig=CF.TI['coin'], fresh=fresh, tag='c1'))
    check('(b) accepted with a token, fresh', ok and victim is None and fresh)
    ok, victim, fresh = q.offer(CF.TI['coin'])
    q.pending.append(dict(trig=CF.TI['coin'], fresh=fresh, tag='c2'))
    check('(b) a repeated type: accepted with the second token, not fresh', ok and not fresh and q.budget == 0.0)
    ok, victim, fresh = q.offer(CF.TI['key'])
    check('(b) no token, an unseen type: replaces the oldest pending point of a seen type',
          ok and fresh and victim is not None and victim['tag'] == 'c2', str(victim))
    q.pending.append(dict(trig=CF.TI['key'], fresh=fresh, tag='k'))
    ok, victim, fresh = q.offer(CF.TI['bomb'])
    check('(b) no token, unseen type, but no stale pending point: skipped', not ok and victim is None and fresh)
    ok, victim, fresh = q.offer(CF.TI['coin'])
    check('(b) no token, a seen type: skipped', not ok and not fresh)
    check('(b) counters', q.skipped == 2 and q.replaced == 1, f'{q.skipped} {q.replaced}')
    q.begin_episode()
    q.types.add('coin')
    found = [(CF.TI['coin'], ('ent', 'coin')), (CF.TI['key'], ('ent', 'key'))]
    check('(b) choose: the first type new to the episode', q.choose(found)[0] == CF.TI['key'])
    q.types.add('key')
    check('(b) choose: else the first', q.choose(found)[0] == CF.TI['coin'])


def test_intents():
    r = empty_row()
    put_ent(r, 0, 5, 20, 1, dx=0.5, dy=0.0)   # a coin to the right
    put_ent(r, 1, 5, 20, 1, dx=-0.9, dy=0.0)  # a farther one to the left
    r['map'][0, 3, 5, 5] = 1
    it = CF.TargetIntent(('ent', 'coin'), h=5)
    a = (7, 3, 1, 0, 1)
    out = it.compose(0, a, r)
    check('(c) target: the move toward the nearest of the class, the other heads the policy\'s',
          out == (move_toward(0.5, 0.0), 3, 1, 0, 1), str(out))
    r_b = r.copy()
    r_b['n_ent'] = 1   # one coin fewer: picked up
    check('(c) target: arrival when one of the class is gone', it.compose(1, a, r_b) == a and it.arrived)
    check('(c) target: after arrival the policy', it.compose(2, a, r) == a)
    it.reset()
    check('(c) target: reset', not it.arrived and it.compose(0, a, r)[0] == move_toward(0.5, 0.0))
    check('(c) target: the cap', it.compose(5, a, r) == a)
    rd = empty_row()
    put_door(rd, 0, dx=0.0, dy=-0.7)
    rd['map'][0, 3, 5, 5] = 1
    itd = CF.TargetIntent(('door', 0), h=50)
    check('(c) door target: up', itd.compose(0, a, rd)[0] == move_toward(0.0, -0.7) == 1)
    rd2 = rd.copy()
    rd2['map'][0, 3, 5, 5], rd2['map'][0, 3, 5, 6] = 0, 1   # the room changed
    check('(c) door target: arrival at the room change', itd.compose(1, a, rd2) == a and itd.arrived)
    rd3 = rd.copy()
    rd3['events'] = EV_NEW_ROOM
    itd.reset()
    itd.compose(0, a, rd)
    check('(c) door target: arrival at EV_NEW_ROOM', itd.compose(1, a, rd3) == a and itd.arrived)
    # sticky: eps 0 = teacher v2's draws; eps 1 = every re-pick uniform (its own rng)
    s0 = T2.Intent('sticky', 0, 12, 3.0, seed=5)
    s1 = T2.Intent('sticky', 0, 12, 3.0, seed=5, eps=0.0)
    seq0 = [s0.compose(j, (4, 0, 0, 0, 0))[0] for j in range(12)]
    seq1 = [s1.compose(j, (4, 0, 0, 0, 0))[0] for j in range(12)]
    check('(c) sticky eps 0: as teacher v2', seq0 == seq1 == [4] * 12)
    s2 = T2.Intent('sticky', 0, 40, 1.0, seed=5, eps=1.0)
    seq2 = [s2.compose(j, (4, 0, 0, 0, 0))[0] for j in range(40)]
    check('(c) sticky eps 1: uniform re-picks', len(set(seq2)) >= 5, str(seq2))
    s2.reset()
    check('(c) sticky reset: the same draws', [s2.compose(j, (4, 0, 0, 0, 0))[0] for j in range(40)] == seq2)
    pt = T2.V2Point(target=('ent', 'coin'))
    cfg = type('C', (), dict(cf_branches=4, teacher_v2_sticky=4.0, cf_sticky_eps=0.25))()
    its = CF.intents(pt, cfg, np.random.default_rng(0), 30)
    check('(c) intents: target + 2 sticky for K = 4', [i.kind for i in its] == ['target', 'sticky', 'sticky'] and
          all(i.h == 30 for i in its))
    pt.target = None
    check('(c) intents: 3 sticky without a target', [i.kind for i in CF.intents(pt, cfg, np.random.default_rng(0), 30)]
          == ['sticky'] * 3)


class FakeLog(T2.V2Log):
    pass


def make_log(n, rng, rw):
    log = T2.V2Log()
    rows = np.zeros(n, ROW)
    for t in range(n):
        r = rows[t:t + 1]
        r['damage'] = rng.random() * 0.1
        r['hurt'] = float(rng.random() < 0.1)
        r['events'] = 0
        r['player'][0, 7] = (4 + (t // 7)) / 12          # health units grow every 7 records
        r['player'][0, 10] = (t // 5) / 50               # coins grow every 5 records
        log.add(r, 0, t, rw=rw)
        log.value.append(float(rng.normal()))
        log.vg.append(float(rng.normal()))
    return log


def test_scoring():
    rw = parse_reward(FLOOR_REWARD)
    rng = np.random.default_rng(3)
    G = T2.returns_to_go([1.0, 0.0, 2.0], 0.5, 4.0)
    check('(d) returns_to_go: discount and bootstrap', np.allclose(G, [1 + 0.5 * (0 + 0.5 * (2 + 0.5 * 4)),
                                                                      0.5 * (2 + 0.5 * 4), 2 + 0.5 * 4, 4.0]), str(G))
    check('(d) returns_to_go: an end (0)', np.allclose(T2.returns_to_go([1.0], 0.9, 0.0), [1.0, 0.0]))
    n = 60
    log = make_log(n, rng, rw)
    g = 0.99
    d, cap = 10, 20

    def manual(stop, vals, extra=True, boot=True):
        s_, disc = 0.0, 1.0
        for i in range(1, stop - d + 1):
            t = d + i
            s_ += disc * (step_reward(log.damage[t], log.hurt[t], log.done[t], log.events[t], rw, True)
                          + (log.gain_r[t] if extra else 0.0))
            disc *= g
        return s_ + (disc * vals[stop] if boot else 0.0)
    sc, m = T2.taken_score(log, d, cap, g, rw, True)
    check('(d) taken: the cap bootstraps from the value (no extra)',
          m == cap and abs(sc - manual(d + cap, log.value, extra=False)) < 1e-9, f'{sc:.6f}')
    sc, m = T2.taken_score(log, d, cap, g, rw, True, values=log.vg, extra=log.gain_r)
    check('(d) taken: value_global and the heal / resource extra', abs(sc - manual(d + cap, log.vg)) < 1e-9)
    check('(d) the log\'s gain_r: heal 0.5 per unit, resource 0.25 per coin',
          log.gain_r[7] == 0.5 and log.gain_r[5] == 0.25 and log.gain_r[6] == 0.0 and log.gain_r[0] == 0.0,
          str(log.gain_r[:11]))
    log.done[d + 5] = 2
    sc, m = T2.taken_score(log, d, cap, g, rw, True, values=log.vg, extra=log.gain_r)
    check('(d) taken: a death ends the window without a bootstrap',
          m == 5 and abs(sc - manual(d + 5, log.vg, boot=False)) < 1e-9)
    log.done[d + 5] = 0
    log.events[d + 7] = EV_EXIT
    sc, m = T2.taken_score(log, d, cap, g, rw, True, values=log.vg, exit_end=True, extra=log.gain_r)
    check('(d) taken: exit_end stops at EV_EXIT (its reward, then its value)',
          m == 7 and abs(sc - manual(d + 7, log.vg)) < 1e-9)
    sc2, m2 = T2.taken_score(log, d, cap, g, rw, True, values=log.vg, exit_end=False, extra=log.gain_r)
    check('(d) taken: without exit_end the run goes on', m2 == cap)
    sc, m = T2.taken_score(log, n - 5, cap, g, rw, True)
    check('(d) taken: the log breaking off bootstraps from its last value', m == 4)
    r0, r1 = np.zeros(1, ROW), np.zeros(1, ROW)
    r0['player'][0, 7], r1['player'][0, 7] = 5 / 12, 3 / 12      # health lost: no heal term
    r0['player'][0, 8], r1['player'][0, 8] = 0.1, 0.3             # 1 -> 3 bombs
    check('(d) gain_reward: gains only', T2.gain_reward(T2.gains_of(r0), T2.gains_of(r1), rw) == 0.5)
    # judge and cf_search with stand-ins
    b, gain, spread, imp = CF.judge(1.0, [0.5, 2.0, 1.2], 1.9, 0.3)
    check('(d) judge: improving (gain 1.0 > margin, > spread 0.1)', b == 1 and abs(gain - 1.0) < 1e-12 and
          abs(spread - 0.1) < 1e-12 and imp)
    check('(d) judge: spread above the gain', not CF.judge(1.0, [2.0], 0.5, 0.3)[3])
    check('(d) judge: no replicate, not improving', not CF.judge(1.0, [2.0], None, 0.3)[3])
    check('(d) judge: gain below the margin', not CF.judge(1.0, [1.2], 1.2, 0.3)[3])

    class FakeS:
        closed = False

        def close(self):
            FakeS.closed = True
    calls = []

    def fake_restore(pt, d_, ctx):
        return FakeS(), 10.0

    def fake_branch(S, ctx, pt, d_, hp0, intent, policy, reseed=None, keep=True, trace=None):
        calls.append((intent.kind, reseed, keep))
        base = {'target': 2.0, 'sticky': 0.5}[intent.kind]
        sc_ = base - 0.1 if reseed is not None else base
        return dict(score=sc_, n=30, end=T2.END_CAP, rows=[], pend=[], act=[], ret=np.zeros(0), value=[], logp=[])
    cfg = type('C', (), dict(cf_branches=4, teacher_v2_sticky=4.0, cf_sticky_eps=0.25, cf_target_seconds=20.0,
                             frames_per_decision=4, cf_margin=0.3, cf_distill_steps=8))()
    ctx = type('X', (), dict(cfg=cfg, cap=cap, gamma=g, rw=rw, run=True, exit_end=False, boot_vg=True, gains=True,
                             stats=dict(replicates=0)))()
    log2 = make_log(n, rng, rw)
    pt = T2.V2Point(at=20, offset=5, log=log2, target=('ent', 'coin'), bank=None)
    res = CF.cf_search(pt, ctx, None, np.random.default_rng(0), branch=fake_branch, restore_fn=fake_restore)
    taken_exp = T2.taken_score(log2, 20, cap, g, rw, True, values=log2.vg, extra=log2.gain_r)[0]
    check('(d) cf_search: taken from the log (value_global, extra)', abs(res['taken'] - taken_exp) < 1e-12)
    check('(d) cf_search: 3 branches (target, 2 sticky) + the best one\'s replicate',
          [c[0] for c in calls] == ['target', 'sticky', 'sticky', 'target'] and calls[3][1] is not None and
          calls[0][2] == 8 and calls[3][2] is False, str(calls))
    imp_exp = (2.0 - taken_exp) > 0.3 and (2.0 - taken_exp) > 0.1
    check('(d) cf_search: best = target, gain, spread, improvement test', res['best']['intent'].kind == 'target' and
          abs(res['gain'] - (2.0 - taken_exp)) < 1e-12 and abs(res['spread'] - 0.1) < 1e-12 and
          res['improving'] == imp_exp and FakeS.closed, f"gain {res['gain']:.3f} improving {res['improving']}")
    calls.clear()
    pt_low = T2.V2Point(at=20, offset=5, log=log2, target=None, bank=None)
    ctx.cfg.cf_margin = 50.0
    res = CF.cf_search(pt_low, ctx, None, np.random.default_rng(0), branch=fake_branch, restore_fn=fake_restore)
    check('(d) cf_search: below the margin no replicate, not improving',
          len(calls) == 3 and res['rep'] is None and not res['improving'])
    ctx.cfg.cf_margin = 0.3
    check('(d) cf_search: the room\'s entry record is not searched',
          CF.cf_search(T2.V2Point(at=5, offset=5, log=log2, target=None), ctx, None, np.random.default_rng(0),
                       branch=fake_branch, restore_fn=fake_restore) is None)


def test_records():
    rng = np.random.default_rng(4)
    nk = 8
    rows = [M.synth_rows(rng, 1)[0] for _ in range(nk)]
    best = dict(n=40, score=2.5, rows=rows, pend=[(1, 2, 0, 0, 0)] * nk, act=[(3, 0, 1, 0, 0)] * nk,
                ret=np.arange(nk, dtype=np.float64), value=[0.1] * nk, logp=[-1.5] * nk,
                intent=CF.TargetIntent(('ent', 'coin'), 150))
    res = dict(best=best, gain=3.1, spread=0.2, taken=-0.6, k=0, rep=2.3)
    pt = T2.V2Point(kind=T2.KIND_CF, pid=77, version=5, trig=CF.TI['key'])
    r_, m_ = T2.records_of(res, None, pt, 3, 2.0, steps=5)
    VI = T2.VI
    check('(e) records: the first `steps` only', len(r_) == 5 and m_.shape == (5, T2.V2_META))
    check('(e) records: weight min(gain, cap)', np.all(m_[:, VI['weight']] == 2.0))
    check('(e) records: meta', np.all(m_[:, VI['kind']] == T2.KIND_CF) and np.all(m_[:, VI['trig']] == CF.TI['key'] + 1)
          and np.all(m_[:, VI['intent']] == T2.INTENT_TARGET) and list(m_[:, VI['j']]) == [0, 1, 2, 3, 4] and
          np.all(m_[:, VI['n']] == 40) and np.all(m_[:, VI['point']] == 77) and np.all(m_[:, VI['worker']] == 3)
          and np.all(m_[:, VI['ret']] == np.arange(5)) and np.all(m_[:, VI['act0']] == 3))
    check('(e) records: the rows as kept', all(np.array_equal(np.atleast_1d(r_[j]).view(np.uint8),
                                                              np.atleast_1d(rows[j]).view(np.uint8)) for j in range(5)))
    res['gain'] = 0.7
    r_, m_ = T2.records_of(res, None, pt, 3, 2.0, steps=50)
    check('(e) records: weight = gain under the cap; all kept when steps > kept', len(r_) == nk and
          np.all(m_[:, VI['weight']] == 0.7))
    pt2 = T2.V2Point(kind=T2.KIND_HURT, pid=1, version=0)
    res['best']['intent'] = T2.Intent('hold', 4, 4)
    r_, m_ = T2.records_of(res, None, pt2, 0, 2.0)
    check('(e) teacher v2 records: trig 0, all kept', len(r_) == nk and np.all(m_[:, VI['trig']] == 0))
    D = 6
    op, has, o = rng.normal(size=(3, D)).astype(np.float32), np.array([True, False, True]), \
        rng.normal(size=(3, D)).astype(np.float32)
    t = CF.t_pack(op, has, o)
    op2, has2, o2 = CF.t_split(t, D)
    check('(e) t_pack / t_split round trip, valid column', np.array_equal(op, op2) and np.array_equal(has, has2) and
          np.array_equal(o, o2) and np.all(t[:, -1] == 1.0) and t.shape == (3, 2 * D + CF.LANE_T_PAD))


def test_lane_policy():
    from multiprocessing import Pipe
    a, b = Pipe()
    io = np.zeros((2, T2.LANE_IO), np.int32)
    out = np.zeros((2, T2.LANE_OUT), np.float64)
    lt = np.zeros((2, 2 * 4 + CF.LANE_T_PAD), np.float32)
    pol = T2.LanePolicy(a, io, out, 1, lambda: False, lane_t=lt)
    pol.tbegin(17)
    check('(f) tbegin: bank row + 1; None: -1', io[1, 11] == 18 and (pol.tbegin(None) or io[1, 11] == -1))
    pol.request(None, (1, 2, 0, 0, 0))
    b.recv_bytes()
    s1 = int(io[1, 10])
    check('(f) request: the action under way and a sequence number', tuple(io[1, :5]) == (1, 2, 0, 0, 0) and s1 > 0)
    out[1, 4] = s1 ^ 0x55   # a stale answer first
    b.send_bytes(b'a')

    def late(seq_ref, vals):
        time.sleep(0.15)
        io[1, 5:10] = vals
        out[1, :4] = (0.25, -1.5, 9, 0.75)
        lt[1] = 1.0
        out[1, 4] = seq_ref()
        b.send_bytes(b'a')
    th = threading.Thread(target=late, args=(lambda: s1, (3, 1, 0, 0, 0)))
    th.start()
    ans = pol.answer()
    th.join()
    check('(f) a stale answer is skipped; the answer, value_global, t_used',
          ans == ((3, 1, 0, 0, 0), 0.25, -1.5) and pol.vg == 0.75 and pol.calls == 1 and np.all(pol.t_used == 1.0),
          str(ans))
    # a leftover reply of an older request (two replies for one request): skipped by the next answer
    b.send_bytes(b'a')   # (out[1, 4] still holds s1)
    pol.request(None, (0, 0, 0, 0, 0))
    b.recv_bytes()
    s2 = int(io[1, 10])
    th = threading.Thread(target=late, args=(lambda: s2, (5, 0, 0, 0, 0)))
    th.start()
    ans = pol.answer()
    th.join()
    check('(f) a leftover reply of the request before is skipped', ans[0] == (5, 0, 0, 0, 0) and s2 != s1)
    out[1, 4] = 0   # a stand-in actor that echoes nothing: accepted
    pol.request(None, (0, 0, 0, 0, 0))
    b.recv_bytes()
    b.send_bytes(b'a')
    check('(f) no echo (0): accepted', pol.answer()[0] == (5, 0, 0, 0, 0))
    stop = [False]
    pol2 = T2.LanePolicy(a, io, out, 1, lambda: stop[0])
    pol2.request(None, (0, 0, 0, 0, 0))
    b.recv_bytes()
    stop[0] = True
    try:
        pol2.answer()
        ok = False
    except T2.V2Stop:
        ok = True
    check('(f) stop: V2Stop', ok and pol2.begin is None)


def small_model(seed=5):
    cfg = dict(TS.SMALL, memory=0, **TS.BASE, **TS.FUSED, **TS.TK)
    torch.manual_seed(seed)
    m = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(m, seed + 1)
    return m


def test_copy_slot():
    m = small_model()
    rng = np.random.default_rng(11)
    nw = TS.NW
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, nw, m.heads) for _ in seq]
    st = TP.TemporalState(m, nw, 'cpu', torch.float32)
    with torch.no_grad():
        for t in range(12):
            m(TP.to_batch(seq[t], pend[t], 'cpu'), need_vg=True,
              temporal=dict(mode='step', state=st, slot=torch.arange(nw)))
    other = TP.TemporalState(m, 6, 'cpu', torch.float32)
    for t_ in (other.kv, other.prev_o):
        t_.normal_()
    other.steps.fill_(99)
    other.copy_slot_([0, 2], [5, 1], other=st)
    same = all(torch.equal(t_.select(ax, 5), u_.select(ax, 0)) and torch.equal(t_.select(ax, 1), u_.select(ax, 2))
               for (t_, ax), (u_, _) in zip(other._tensors(), st._tensors()))
    check('(g) copy_slot_: every per-slot tensor equal (between states, two slots)', same)
    check('(g) copy_slot_: other slots untouched', int(other.steps[0]) == 99 and int(other.steps[3]) == 99)
    st2 = TP.TemporalState(m, nw, 'cpu', torch.float32)
    st2.copy_slot_(torch.tensor([1]), torch.tensor([0]))
    st.copy_slot_([1], [0])
    check('(g) copy_slot_: within a state', all(torch.equal(t_.select(ax, 0), t_.select(ax, 1))
                                                 for t_, ax in st._tensors()))
    other.reset_slot_([5])
    check('(g) reset_slot_: zeros', all(not t_.select(ax, 5).any() for t_, ax in other._tensors()) and
          int(other.steps[1]) == int(st.steps[2]))
    try:
        TP.TemporalState(m, 2, 'cpu', torch.bfloat16).copy_slot_([0], [0], other=st)
        bad = False
    except ValueError:
        bad = True
    check('(g) copy_slot_: another dtype refused', bad)


def test_lane_continuation():
    m = small_model(7)
    rng = np.random.default_rng(12)
    nw, T1, T = TS.NW, 15, TS.T2
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, nw, m.heads) for _ in seq]
    parent = TP.TemporalState(m, nw, 'cpu', torch.float32)
    bank = TP.TemporalState(m, 8, 'cpu', torch.float32)
    lane = TP.TemporalState(m, nw, 'cpu', torch.float32)
    par = dict(lg=[], v=[], vg=[], o=[])
    with torch.no_grad():
        for t in range(T1):
            m(TP.to_batch(seq[t], pend[t], 'cpu'), need_vg=True,
              temporal=dict(mode='step', state=parent, slot=torch.arange(nw)))
        bank.copy_slot_([0], [6], other=parent)   # the snapshot before record T1's step (worker 0)
        for t in range(T1, T):   # the parent goes on (worker 0 alone, as the actor may step it)
            td = dict(mode='step', state=parent, slot=torch.tensor([0]))
            lg, v, _, vg = m(TP.to_batch(seq[t][0:1], pend[t][0:1], 'cpu'), need_vg=True, temporal=td)
            par['lg'].append(lg)
            par['v'].append(v)
            par['vg'].append(vg)
            par['o'].append(td['out']['o'])
        # the lane: dirty first (another history), then the snapshot copied into its slot 1
        for t in range(8):
            m(TP.to_batch(seq[t], pend[t], 'cpu'), need_vg=True,
              temporal=dict(mode='step', state=lane, slot=torch.arange(nw)))
        lane.copy_slot_([6], [1], other=bank)
        lan = dict(lg=[], v=[], vg=[], o=[], op=[], has=[])
        rows_l = []
        for t in range(T1, T):
            sl = torch.tensor([1])
            op = lane.prev_o.index_select(0, sl).clone()
            has = (lane.steps.index_select(0, sl) > 0) & torch.from_numpy(seq[t][0:1]['first'] == 0)
            td = dict(mode='step', state=lane, slot=sl)
            lg, v, _, vg = m(TP.to_batch(seq[t][0:1], pend[t][0:1], 'cpu'), need_vg=True, temporal=td)
            for k_, x_ in (('lg', lg), ('v', v), ('vg', vg), ('o', td['out']['o']), ('op', op), ('has', has)):
                lan[k_].append(x_)
            rows_l.append(seq[t][0])
    dmax = max(float((torch.cat(par[k]) - torch.cat(lan[k])).abs().max()) for k in ('lg', 'v', 'vg', 'o'))
    check('(h) the lane step from the copied slot == the parent\'s continuation (logits, value, value_global, o)',
          dmax <= 1e-6, f'max diff {dmax:.2e} over {T - T1} steps')
    check('(h) the states after: parent slot 0 == lane slot 1', all(
        torch.allclose(t_.select(ax, 0).float(), u_.select(ax, 1).float(), atol=1e-6)
        for (t_, ax), (u_, _) in zip(parent._tensors(), lane._tensors())))
    # the imitation records' stored outputs reproduce the lane's logits (the learner's 'stored' readout)
    R = np.stack(rows_l)
    P = np.concatenate([p_[0:1] for p_ in pend[T1:T]])
    O, OP, H = torch.cat(lan['o']), torch.cat(lan['op']), torch.cat(lan['has'])
    with torch.no_grad():
        lg_s, v_s, _, vg_s = m(TP.to_batch(R, P, 'cpu'), need_vg=True, temporal=dict(mode='stored', o=O, o_prev=OP,
                                                                                    has=H))
        lg_m, _, _, _ = m(TP.to_batch(R, P, 'cpu'), need_vg=True,
                          temporal=dict(mode='stored', o=O, o_prev=OP, has=H, m=torch.ones(len(R))))
    ds = float((lg_s - torch.cat(lan['lg'])).abs().max())
    check('(h) stored outputs (o_prev, has, o) reproduce the lane\'s logits', ds < 1e-4 and
          float((vg_s - torch.cat(lan['vg'])).abs().max()) < 1e-4 and torch.equal(lg_m, lg_s), f'max diff {ds:.2e}')
    # as a tail behind other records: 2 'stored' rows (k), then a teacher-like row (m 0) and the rest (m 1)
    k = 2
    extra = M.synth_rows(np.random.default_rng(1), 1)
    R2 = np.concatenate([R[:k], extra, R[k:]])
    P2 = np.concatenate([P[:k], M.synth_pending(np.random.default_rng(2), 1, m.heads), P[k:]])
    nt = len(R2) - k
    D = O.shape[1]
    tail = dict(o_prev=torch.cat([torch.zeros(1, D), OP[k:]]), has=torch.cat([torch.zeros(1, dtype=torch.bool), H[k:]]),
                o=torch.cat([torch.zeros(1, D), O[k:]]), m=torch.cat([torch.zeros(1), torch.ones(nt - 1)]))
    with torch.no_grad():
        lg_t = m(TP.to_batch(R2, P2, 'cpu'), need_vg=True,
                 temporal=dict(mode='stored', o=O[:k], o_prev=OP[:k], has=H[:k], tail=tail))[0]
        lg_plain = m(TP.to_batch(extra, P2[k:k + 1], 'cpu'), need_vg=True)[0]
    dt = float((torch.cat([lg_t[:k], lg_t[k + 1:]]) - lg_s).abs().max())
    d_e = float((lg_t[k] - lg_plain[0]).abs().max())
    check('(h) tail: the stored rows past k as the lane\'s, an m = 0 row as without temporal', dt < 1e-4 and d_e < 1e-4,
          f'{dt:.2e} / {d_e:.2e}')


def test_blocks():
    from multiprocessing import shared_memory
    W, D = 3, 8
    blocks = [shared_memory.SharedMemory(create=True, size=s_) for s_ in CF.cf_sizes(W, D)]
    try:
        got = CF.attach_cf([b_.name for b_ in blocks], W, D)
        _, main_vg, snap, lane_t, v2_t = got
        check('(i) attach_cf (temporal): shapes', main_vg.shape == (W,) and snap.shape == (W,) and snap.dtype == np.int64
              and lane_t.shape == (W, 2 * D + 2) and v2_t.shape == (W, T2.V2_RING, 2 * D + 2))
        for b_ in got[0]:
            b_.close()
    finally:
        for b_ in blocks:
            b_.close()
            b_.unlink()
    blocks = [shared_memory.SharedMemory(create=True, size=s_) for s_ in CF.cf_sizes(W, 0)]
    try:
        got = CF.attach_cf([b_.name for b_ in blocks], W, 0)
        check('(i) attach_cf (no temporal): two blocks, no lane_t', len(blocks) == 2 and got[3] is None and got[4] is None)
        for b_ in got[0]:
            b_.close()
    finally:
        for b_ in blocks:
            b_.close()
            b_.unlink()
    cfg = type('C', (), dict(memory_dim=0, cf_queue=8))()
    check('(i) names offset / bank rows', CF.cf_names_offset(cfg) == 6 and CF.bank_rows(cfg) == 8 + CF.CF_BANK_EXTRA)
    cfg.memory_dim = 64
    check('(i) names offset with memory', CF.cf_names_offset(cfg) == 9)


if __name__ == '__main__':
    t0 = time.perf_counter()
    for fn in (test_detect, test_quota, test_intents, test_scoring, test_records, test_lane_policy, test_copy_slot,
               test_lane_continuation, test_blocks):
        fn()
    print(f'{len(FAIL)} failed' + (': ' + ', '.join(FAIL) if FAIL else '') + f'  ({time.perf_counter() - t0:.1f} s)')
    sys.exit(1 if FAIL else 0)
