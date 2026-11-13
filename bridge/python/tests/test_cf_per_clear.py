"""--cf-per-clear X (2026-11-13, tok_cf.Quota's earned bucket): a choice point needs a token of the time bucket (cfg.cf_points
per game hour, as before) and one earned by the worker's main episode clearing rooms (X per EV_CLEARED, capacity
CF_BURST); a replacement needs an earned token too. X = 0: the quota bit for bit as before (a copy of the old class
below is the reference). Branches never earn (only tok_floor's main loop calls earn)."""
import inspect
import random

from isaac_bridge import tok_cf as CF
from isaac_bridge import tok_floor, tok_sampler


class OldQuota:
    """tok_cf.Quota as of 657b198 (the reference for X = 0), with the two refund forms tok_floor used then."""

    def __init__(self, points_per_hour, fpd, burst=CF.CF_BURST):
        self.rate = float(points_per_hour) * fpd / 30.0 / 3600.0
        self.burst = max(1.0, float(burst))
        self.budget = 0.0
        self.types = set()
        self.pending = []
        self.skipped = self.replaced = 0

    def tick(self, n=1):
        self.budget = min(self.burst, self.budget + self.rate * n)

    def begin_episode(self):
        self.types = set()
        self.pending = []

    def offer(self, trig):
        name = CF.TRIGGERS[trig]
        fresh = name not in self.types
        self.types.add(name)
        if self.budget >= 1.0:
            self.budget -= 1.0
            return True, None, fresh
        if fresh:
            stale = [p for p in self.pending if not p['fresh']]
            if stale:
                victim = stale[0]
                self.pending.remove(victim)
                self.replaced += 1
                return True, victim, fresh
        self.skipped += 1
        return False, None, fresh


def test_off_is_the_old_quota_bit_for_bit():
    rnd = random.Random(7)
    for trial in range(40):
        old, new = OldQuota(6.0, 4), CF.Quota(6.0, 4, per_clear=0.0)
        for step in range(3000):
            r = rnd.random()
            if r < 0.6:
                n = rnd.randint(1, 900)
                old.tick(n)
                new.tick(n)
            elif r < 0.62:
                old.begin_episode()
                new.begin_episode()
            elif r < 0.64:
                new.earn(rnd.randint(1, 3))   # (off: nothing)
            else:
                trig = rnd.randrange(len(CF.TRIGGERS))
                a, b = old.offer(trig), new.offer(trig)
                assert (a[0], a[2]) == (b[0], b[2]) and (a[1] is None) == (b[1] is None)
                if a[0]:
                    tag = (trial, step)
                    old.pending.append(dict(trig=trig, fresh=a[2], tag=tag, charged=a[1] is None))
                    new.pending.append(dict(trig=trig, fresh=b[2], tag=tag, charge=new.charge))
                    if rnd.random() < 0.1:   # no bank row: the old code's budget += 1.0 (charged ones), now refund()
                        p_old, p_new = old.pending.pop(), new.pending.pop()
                        if p_old['charged']:
                            old.budget += 1.0
                        new.refund(p_new['charge'])
                    elif rnd.random() < 0.05 and old.pending:   # the actor refused: old min(burst, budget + 1)
                        p_old, p_new = old.pending.pop(0), new.pending.pop(0)
                        if p_old['charged']:
                            old.budget = min(old.burst, old.budget + 1.0)
                        new.refund(p_new['charge'])
            assert old.budget == new.budget, (trial, step, old.budget, new.budget)
            assert (old.skipped, old.replaced) == (new.skipped, new.replaced)
            assert [p['tag'] for p in old.pending] == [p['tag'] for p in new.pending]
        assert new.earned == 0.0 and new.skipped_earned == 0


def test_no_clears_no_points():
    q = CF.Quota(6.0, 4, per_clear=0.25)
    q.tick(10 ** 6)   # (the time bucket full)
    q.begin_episode()
    for trig in ('coin', 'key', 'pedestal', 'random', 'cleared'):
        ok, victim, fresh = q.offer(CF.TI[trig])
        assert not ok and victim is None and fresh
    assert q.skipped == q.skipped_earned == 5 and q.budget == CF.CF_BURST


def test_points_follow_the_clears():
    q = CF.Quota(6.0, 4, per_clear=0.25)
    q.tick(10 ** 6)
    q.begin_episode()
    q.earn(3)
    assert q.earned == 0.75 and not q.offer(CF.TI['coin'])[0]
    q.earn(1)   # 4 clears: one token
    ok, victim, fresh = q.offer(CF.TI['coin'])
    assert ok and victim is None and q.charge == (1.0, 1.0) and q.earned == 0.0 and q.budget == CF.CF_BURST - 1.0
    q.earn(100)   # capped at CF_BURST
    assert q.earned == CF.CF_BURST and q.earned_total == 0.25 * 104


def test_the_time_cap_still_holds():
    q = CF.Quota(6.0, 4, per_clear=1.0)
    q.earn(2)
    q.begin_episode()
    q.types.update(CF.TRIGGERS)   # (every type seen: no replacement)
    ok, _, _ = q.offer(CF.TI['coin'])
    assert not ok and q.skipped == 1 and q.skipped_earned == 0 and q.earned == 2.0   # (no time token)
    per_point = 3600 * 30 / 4 / 6
    q.tick(int(per_point) + 5)
    assert q.offer(CF.TI['coin'])[0] and q.earned == 1.0


def test_a_replacement_needs_an_earned_token():
    q = CF.Quota(6.0, 4, per_clear=1.0)
    q.begin_episode()
    q.pending.append(dict(trig=CF.TI['coin'], fresh=False, tag='stale'))
    q.types.add('coin')
    ok, victim, fresh = q.offer(CF.TI['key'])   # (no earned token: not even a replacement)
    assert not ok and victim is None and q.skipped_earned == 1 and len(q.pending) == 1
    q.earn(1)
    ok, victim, fresh = q.offer(CF.TI['bomb'])   # (no time token, an unseen type: replaces, one earned token)
    assert ok and victim['tag'] == 'stale' and q.charge == (0.0, 1.0) and q.earned == 0.0
    q.refund(q.charge)   # (no bank row: the earned token back, the time bucket untouched)
    assert q.earned == 1.0 and q.budget == 0.0


def test_only_the_main_episodes_earn():
    src = inspect.getsource(tok_floor)
    assert src.count('cf_quota.earn(') == 1 and src.count("mine[ST['clears']] += 1") == 1
    at = src.index('cf_quota.earn(')
    main = src.index("cf_quota.begin_episode()")   # (the main episodes' loop: the quota's episode start)
    for name in ('def branch_episode', 'def lab_episode'):   # (branches / lab episodes: before the main loop)
        assert src.index(name) < main < at
    assert 'earn(' not in inspect.getsource(CF.cf_search)
    assert "per_clear=getattr(cfg, 'cf_per_clear', 0.0)" in src
    assert "cf_quota.refund(charge_)" in src and "cf_quota.refund(point_b['charge'])" in src
    assert 'cf_quota.budget += 1.0' not in src
    # (only a fought clear earns: the earn call sits under 'if fought_:', every EV_CLEARED counts in clears)
    blk = src[src.index('cleared_, fought_ = clear_ctr.step('):src.index('if cfon and not done:   # B34')]
    assert blk.index('if cleared_:') < blk.index("mine[ST['clears']] += 1") < blk.index('if fought_:') <         blk.index("mine[ST['clears_fought']] += 1") < blk.index('cf_quota.earn(1)')
    assert 'clear_ctr = ClearCounter()' in src


def test_a_clear_without_a_fight_does_not_earn():
    """tok_obs.ClearCounter: EV_CLEARED counts as cleared always, as fought only after damage in the room since
    entering it; a room change resets the sum."""
    from isaac_bridge.tok_obs import ClearCounter, EV_CLEARED, EV_NEW_ROOM
    c = ClearCounter()
    assert c.step(True, 0.0, EV_NEW_ROOM) == (False, False)
    assert c.step(False, 0.0, EV_CLEARED) == (True, False)          # (an empty room's late clear flag: not fought)
    assert c.step(True, 0.0, EV_NEW_ROOM) == (False, False)
    assert c.step(False, 0.4, 0) == (False, False)
    assert c.step(False, 0.6, EV_CLEARED) == (True, True)           # (monsters hurt in this room: fought)
    assert c.step(True, 0.0, 0) == (False, False)                   # (the next room: the sum starts again)
    assert c.step(False, 0.0, EV_CLEARED) == (True, False)
    # the bucket: X per fought clear only (tok_floor calls earn on fought ones)
    q = CF.Quota(6.0, 4, per_clear=0.5)
    c = ClearCounter()
    for new_room, dmg, ev in ((True, 0, 0), (False, 0, EV_CLEARED), (True, 0, 0), (False, 1.0, EV_CLEARED)):
        cleared, fought = c.step(new_room, dmg, ev)
        if fought:
            q.earn(1)
    assert q.earned == 0.5


def test_plumbing():
    assert tok_sampler.TokSamplerConfig().cf_per_clear == 0.0
    st = tok_sampler.STATS
    for k in ('clears', 'clears_fought', 'cf_earned', 'cf_skip_earned'):
        assert k in st and st.index(k) < st.index('np_runs')
    assert list(st[-2:]) == ['np_runs', 'np_removed']   # (tests/test_eval_ablate_smoke.py)
    import train_tok
    src = inspect.getsource(train_tok)
    assert "'--cf-per-clear', type=float, default=0.0" in src and 'cf_per_clear=args.cf_per_clear' in src
    assert "for k_cl in ('clears', 'clears_fought'):" in src and "row[k_cl] = tot_cl - L.get(k_cl + '_prev', 0.0)" in src
