"""The long run's reward terms (2026-11-19, the user / the coordinator; Codex round 13's rules; train_tok NEW_TERMS,
Episodes.arrive; tok_obs.nohit_step): item / trinket (ids never rewarded in the episode, the start build not rewarded;
a passive only on a rise of the held collectibles' count, the active slot / a trinket slot changing to a new id),
depth (the exit and boss terms x 1 + depth x (n - 1); the exit only to a floor deeper than any reached), nohit (the
worker's EV_NOHIT: a room that had monsters, cleared, never hurt in it, once per room; not on a death). Each is 0 by
default and then every reward is the old one. Codex's CPU counterexamples as regressions."""
import numpy as np
import pytest

import train_tok as T
from isaac_bridge.tok_obs import EV_BOSS, EV_CLEARED, EV_EXIT, EV_NOHIT, EpisodeState, ROW, nohit_step

OLD = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
           heal=0.0, resource=0.0, clear=1.0)


def _row(first=0, inv=(), active=0, trinkets=(0, 0), n=None, stage=1, events=0, hurt=0.0, damage=0.0, done=0):
    r = np.zeros(1, ROW)
    r['first'], r['done'], r['seed'], r['group'] = first, done, 7, 0
    for k, v in enumerate(inv):
        r['inv'][0, k] = (v, 1)
    r['pitem'][0, 0] = active
    r['pitem'][0, 1], r['pitem'][0, 2] = trinkets
    r['pinv'][0, 6] = (len(inv) if n is None else n) / 16   # (the held collectibles' count, distinct ids)
    r['stage'] = stage
    r['events'], r['hurt'], r['damage'] = events, hurt, damage
    return r


class Run:
    def __init__(self, **new):
        self.rw = dict(OLD, **new)
        self.roll = T.Rollout(1, 64, close=True)
        self.book = T.Episodes(1, self.rw, True, run=True, items=True)
        self.pending = np.zeros((1, 3), np.int64)
        self.rewards = []

    def rec(self, **kw):
        rows = _row(**kw)
        k_before = int(self.roll.open[0])
        go = self.book.arrive(self.roll, rows, np.array([0]), self.pending)
        if k_before >= 0:
            self.rewards.append(float(self.roll.reward[0, k_before]))
        if go[0]:
            k = int(self.roll.length[0])
            self.roll.open[0], self.roll.length[0] = k, k + 1
        return self


def _old(hurt=0.0, damage=0.0, events=0, done=0):
    """The reward before the new terms (Episodes.arrive's formula, run mode, no gains)."""
    rw = OLD
    r = rw['damage'] * damage - rw['hurt'] * hurt - rw['time'] + (-rw['death'] if done == 2 else 0.0)
    return (r + (rw['room'] * ((events & 1) > 0) + rw['explore'] * ((events & 2) > 0) + rw['boss'] * ((events & 4) > 0))
            + rw['exit'] * ((events & EV_EXIT) > 0))


def _extra(r, olds):
    return [a - b for a, b in zip(r.rewards, olds)]


def test_defaults_are_the_old_rewards_bit_for_bit():
    r = Run()
    assert not r.book.new_on
    steps = [dict(events=EV_CLEARED, damage=0.5), dict(hurt=1.0), dict(events=EV_BOSS | EV_CLEARED | EV_NOHIT, stage=3),
             dict(events=EV_EXIT, stage=4, inv=(9,)), dict(active=12, trinkets=(3, 0))]
    r.rec(first=1, inv=(1,))
    for s in steps:
        r.rec(**s)
    want = [_old(s.get('hurt', 0.0), s.get('damage', 0.0), s.get('events', 0)) for s in steps]
    assert r.rewards == [float(np.float32(x)) for x in want]   # (exactly, as Rollout stores it)
    r.rec(done=2)
    assert 'r' not in r.book.finished[0] and 'items_new' not in r.book.finished[0]


def test_items_count_rise_active_trinket_and_the_farming_ways():
    r = Run(item=1.0, trinket=0.3)
    r.rec(first=1, inv=(10,), active=0)                        # the start build: not rewarded
    r.rec(inv=(20, 10))                                        # a passive taken (count 1 -> 2): +1
    r.rec(inv=(20, 10), active=30)                             # an active taken: +1
    r.rec(inv=(20, 10), active=0)                              # put down ...
    r.rec(inv=(20, 10), active=30)                             # ... and taken again: 0
    r.rec(inv=(20, 10), active=31)                             # a second active (swap): +1
    r.rec(inv=(20, 10), active=30)                             # swapped back: 0
    r.rec(inv=(20, 10), active=30, trinkets=(5, 0))            # a trinket: +0.3
    r.rec(inv=(20, 10), active=30, trinkets=(0, 0))            # dropped (or smelted) ...
    r.rec(inv=(20, 10), active=30, trinkets=(5, 0))            # ... and taken again: 0
    r.rec(inv=(41, 42), active=30, trinkets=(5, 0))            # D4: the whole build rerolled, no count rise: 0
    r.rec(inv=(43, 41, 42), active=30, trinkets=(5, 0))        # a rerolled pedestal taken: +1
    r.rec(inv=(43, 41, 42), active=30, trinkets=(5, 0), n=3)   # nothing: 0
    olds = [_old()] * 12
    assert _extra(r, olds) == pytest.approx([1, 1, 0, 0, 1, 0, 0.3, 0, 0, 0, 1, 0], abs=1e-5)
    r.rec(done=2)
    f = r.book.finished[0]
    assert f['items_new'] == 4 and f['trinkets_new'] == 1


def test_items_inv_count_down_and_up_pays_once():
    """(Codex: inv count 0 -> 1 -> 0 -> 1) an item lost and gained again: one reward."""
    r = Run(item=1.0)
    r.rec(first=1, inv=(10,))
    r.rec(inv=(20, 10))     # +1
    r.rec(inv=(10,))        # lost
    r.rec(inv=(20, 10))     # back: 0
    assert _extra(r, [_old()] * 3) == pytest.approx([1, 0, 0], abs=1e-5)


def test_item_set_starts_again_each_episode():
    r = Run(item=1.0)
    r.rec(first=1, inv=(10,))
    r.rec(inv=(20, 10))
    r.rec(done=2, inv=(20, 10))
    r.rec(first=1, inv=(10,))
    r.rec(inv=(20, 10))
    assert r.rewards[-1] - _old() == pytest.approx(1.0, abs=1e-5)


def test_depth_scales_and_pays_the_exit_once_per_new_floor():
    r = Run(depth=0.25)
    r.rec(first=1, stage=4)
    r.rec(stage=4, events=EV_BOSS)          # floor 4: boss x 1.75
    r.rec(stage=5, events=EV_EXIT)          # left floor 4 (deeper than reached): exit x 1.75
    r.rec(stage=4, events=EV_EXIT)          # back to 4: no extra
    r.rec(stage=5, events=EV_EXIT)          # 5 again: no extra
    r.rec(stage=6, events=EV_EXIT)          # left 5: x 2.0
    ev = (EV_BOSS, EV_EXIT, EV_EXIT, EV_EXIT, EV_EXIT)
    assert _extra(r, [_old(events=e) for e in ev]) == pytest.approx([2 * 0.75, 3 * 0.75, 0, 0, 3 * 1.0], abs=1e-5)


def test_nohit_on_the_workers_bit_not_on_a_death():
    r = Run(nohit=0.5)
    r.rec(first=1)
    r.rec(events=EV_CLEARED | EV_NOHIT)                  # +0.5
    r.rec(events=EV_CLEARED)                             # (no bit: the worker said no) 0
    r.rec(events=EV_CLEARED | EV_NOHIT, done=2)          # cleared and died in the same step: 0
    olds = [_old(events=EV_CLEARED), _old(events=EV_CLEARED), _old(events=EV_CLEARED, done=2)]
    assert _extra(r, olds) == pytest.approx([0.5, 0, 0], abs=1e-5)


def _st():
    st = EpisodeState(100, floor=True, run=True)
    st.nohit_on = True
    return st


def _walk(st, steps):
    """steps: (room, next_floor, hurt, damage, monsters_hp, cleared); the EV_NOHIT of each (encode_row's order)."""
    out = []
    for room, nf, hurt, damage, hp, cleared in steps:
        ev = nohit_step(st, room, nf, hurt, damage, hp, EV_CLEARED if cleared else 0)
        out.append(bool(ev & EV_NOHIT))
        st.room = room
    return out


def test_nohit_worker_rules():
    # an empty room whose clear flag comes a frame late: no monsters ever -- no reward
    assert _walk(_st(), [(1, False, 0, 0, 0, False), (1, False, 0, 0, 0, True)]) == [False, False]
    # a room with monsters, cleared, never hurt: once; cleared again (respawn / challenge): no second time
    assert _walk(_st(), [(1, False, 0, 0, 5, False), (1, False, 0, 0.5, 0, True), (1, False, 0, 0, 3, False),
                         (1, False, 0, 0, 0, True)]) == [False, True, False, False]
    # hurt in room 1, out and back in, cleared: no reward (every visit counts)
    assert _walk(_st(), [(1, False, 0, 0, 5, False), (1, False, 1, 0, 5, False), (2, False, 0, 0, 0, False),
                         (1, False, 0, 0, 5, False), (1, False, 0, 0, 0, True)]) == [False] * 5
    # hurt in A, the step that moved to B carries it: A is charged, B stays eligible
    st = _st()
    assert _walk(st, [(1, False, 0, 0, 5, False), (2, False, 1, 0, 5, False), (2, False, 0, 0, 0, True)]) == \
        [False, False, True]
    assert st.nohit_rooms[(0, 1)][1] and not st.nohit_rooms[(0, 2)][1]
    # the next floor's room with the same index is another room
    st = _st()
    assert _walk(st, [(1, False, 1, 0, 5, False), (1, True, 0, 0, 5, False), (1, False, 0, 0, 0, True)]) == \
        [False, False, True]


def test_nohit_without_the_visible_map():
    """(curse of the lost: ROW map cleared) the worker keys rooms by the engine's index, no map: the bit still comes."""
    st = _st()
    st.map[:] = 0
    assert _walk(st, [(7, False, 0, 0, 4, False), (7, False, 0, 1.0, 0, True)]) == [False, True]


def test_new_terms_refused_where_they_cannot_be_computed(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv('ISAAC_RL_VISIBLE_V3', raising=False)
    monkeypatch.delenv('ISAAC_RL_NOHIT', raising=False)
    cases = ((['--reward', 'item=1'], 'floor or run only'),
             (['--mode', 'run', '--reward', 'nohit=0.5'], '--frame v3 only'),
             (['--mode', 'run', '--reward', 'item=1', '--block-memory', '1'], 'not with --block-memory'))
    for i, (extra, msg) in enumerate(cases):
        with pytest.raises(SystemExit):
            T.main(['--groups-file', str(tmp_path / 'x.json'), '--out', str(tmp_path / f'o{i}')] + extra)
        assert msg in capsys.readouterr().err, extra


class _LoopEpisodes(T.Episodes):
    """The per-record loop new_terms was before A-1 (2026-11-19, eadef6e / 20b0fce), word for word but for reading the
    record before from held_prev / held_ok: the reference the vectorized one must equal bit for bit."""

    @staticmethod
    def _held1(r):
        return (int(r['pitem'][0]), int(r['pitem'][1]), int(r['pitem'][2]), int(round(float(r['pinv'][6]) * 16)))

    def new_terms(self, rows, sel, w, ev, hu, d, held_all=None):   # (held_all: A-5's argument, not used here)
        rw = self.rw
        out = np.zeros((len(w), 4))
        r_ = rows[sel]
        stage = r_['stage'].astype(np.int64)
        for j in range(len(w)):
            i = int(w[j])
            r = r_[j]
            if (rw.get('item') or rw.get('trinket')) and self.held_ok[i]:
                a0, t0, t1, n0 = (int(x) for x in self.held_prev[i])
                a, ta, tb, n = self._held1(r)
                seen = self.seen_c[i]
                got_c = 0
                if a > 0 and a != a0 and a not in seen:
                    seen.add(a)
                    got_c += 1
                rise = n - n0 - (1 if got_c else 0)
                if rise > 0:
                    for v, c in r['inv']:
                        v = int(v)
                        if rise <= 0:
                            break
                        if v > 0 and c > 0 and v != a and v not in seen:
                            seen.add(v)
                            got_c += 1
                            rise -= 1
                got_t = 0
                for tv, tp in ((ta, t0), (tb, t1)):
                    if tv > 0 and tv != tp and tv not in self.seen_t[i]:
                        self.seen_t[i].add(tv)
                        got_t += 1
                out[j, 0] = rw.get('item', 0.0) * got_c
                out[j, 1] = rw.get('trinket', 0.0) * got_t
                self.counts[i, 0] += got_c
                self.counts[i, 1] += got_t
            if rw.get('depth'):
                if self.run and (ev[j] & EV_EXIT):
                    if stage[j] > self.stage_max[i]:
                        out[j, 2] += rw['exit'] * rw['depth'] * max(int(self.stage_prev[i]) - 1, 0)
                if ev[j] & 4:
                    out[j, 2] += rw['boss'] * rw['depth'] * max(int(stage[j]) - 1, 0)
            if rw.get('nohit') and (ev[j] & EV_NOHIT) and d[j] != 2:
                out[j, 3] = rw['nohit']
                self.counts[i, 2] += 1
        return out


def _stream(seed, W=6, steps=300, rate=1.0):
    """Random record batches (distinct workers per batch) with holdings that change the ways a run does: passives
    taken / lost / rerolled, the active taken / dropped / swapped, trinkets, floors up and down, bosses, no-hit clears,
    deaths and new episodes; the count column now and then off the ids (and between sixteenths). rate: scales how
    often the holdings change."""
    rng = np.random.default_rng(seed)
    st = [dict(inv=[], a=0, t=[0, 0], stage=1, new=True) for _ in range(W)]
    out = []
    for _ in range(steps):
        ws = rng.choice(W, size=int(rng.integers(1, W + 1)), replace=False)
        rows = np.zeros(len(ws), ROW)
        for j, i in enumerate(ws):
            s = st[i]
            r = rows[j]
            r['seed'], r['group'] = 7, 0
            if s['new']:
                s.update(inv=list(rng.choice(np.arange(1, 9), size=int(rng.integers(0, 3)), replace=False)),
                         a=int(rng.integers(0, 3)) * int(rng.integers(20, 25)), t=[int(rng.integers(0, 2)) * 50, 0],
                         stage=int(rng.integers(1, 3)), new=False)
                r['first'] = 1
            else:
                u = rng.random() / rate
                if u < 0.15:
                    s['inv'].insert(0, int(rng.integers(1, 40)))       # a passive (sometimes one held already)
                elif u < 0.2 and s['inv']:
                    s['inv'].pop(int(rng.integers(len(s['inv']))))    # lost
                elif u < 0.23:
                    s['inv'] = [int(x) for x in rng.integers(1, 40, len(s['inv']))]   # rerolled
                elif u < 0.33:
                    s['a'] = int(rng.choice([0, s['a'], int(rng.integers(20, 30))]))
                elif u < 0.4:
                    s['t'][int(rng.integers(2))] = int(rng.choice([0, int(rng.integers(50, 56))]))
                ev = 0
                if rng.random() < 0.3:
                    ev |= EV_CLEARED
                if rng.random() < 0.08:
                    ev |= EV_BOSS
                if rng.random() < 0.15:
                    ev |= EV_NOHIT
                if rng.random() < 0.1:
                    s['stage'] = max(1, s['stage'] + int(rng.choice([1, 1, 1, -1])))
                    ev |= EV_EXIT
                r['events'] = ev
                r['hurt'] = float(rng.random() < 0.1)
                r['damage'] = float(rng.random() < 0.3) * 0.5
                if rng.random() < 0.04:
                    r['done'] = int(rng.choice([2, 3]))
                    s['new'] = True
            ids = list(dict.fromkeys(s['inv']))[:16]
            for k, v in enumerate(ids):
                r['inv'][k] = (v, 1)
            n = len(set(s['inv']) | ({s['a']} - {0}))
            if rng.random() < 0.05:
                n += int(rng.integers(-1, 2))
            r['pinv'][6] = (max(n, 0) + (0.5 if rng.random() < 0.03 else 0.0)) / 16
            r['pitem'][0], r['pitem'][1], r['pitem'][2] = s['a'], s['t'][0], s['t'][1]
            r['stage'] = s['stage']
        out.append((rows, ws))
    return out


def _play(cls, stream, rw, W=6):
    roll = T.Rollout(W, 512, close=True)
    book = cls(W, rw, True, run=True, items=True)
    pending = np.zeros((W, 3), np.int64)
    for rows, ws in stream:
        go = book.arrive(roll, rows, ws.astype(np.int64), pending)
        for i in ws[go]:
            k = int(roll.length[i])
            roll.open[i], roll.length[i] = k, k + 1
    return roll, book


@pytest.mark.parametrize('seed', range(6))
def test_vectorized_new_terms_equal_the_per_record_loop(seed):
    """A-1 (2026-11-19): the vectorized new_terms -- only records whose holdings changed go through the id sets -- gives
    every decision's reward, each term's share, the counts and the episodes' lines of the per-record loop, bit for bit."""
    rw = dict(OLD, heal=0.5, resource=0.05, item=1.0, trinket=0.3, depth=0.25, nohit=0.5)
    s = _stream(seed)
    ra, ba = _play(T.Episodes, s, rw)
    rb, bb = _play(_LoopEpisodes, s, rw)
    n = int(ra.length.max())
    assert np.array_equal(ra.length, rb.length) and n > 20
    assert np.array_equal(ra.reward[:, :n], rb.reward[:, :n])
    assert np.array_equal(ba.term, bb.term) and np.array_equal(ba.counts, bb.counts)
    assert ba.finished == bb.finished and len(ba.finished) > 3
    assert sum(f.get('items_new', 0) for f in ba.finished) > 0 and sum(f.get('nohit_rooms', 0) for f in ba.finished) > 0
