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
