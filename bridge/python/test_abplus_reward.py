"""combat-v2 (isaac_bridge/abplus_reward.py) on synthetic observations; no engine needed."""
import unittest

from isaac_bridge.abplus_reward import (COMPONENTS, CombatV2, clear_bonus, describe, health_value)


def obs(hearts=6, soul=0, bombs=1, damage=0.0, blocking=100.0, points=16):
    return {'players': [{'hearts': hearts, 'soul': soul, 'eternal': 0, 'bone': 0, 'bombs': bombs}],
            'combat': {'player_damage': damage, 'blocking_hp': blocking, 'blocking_points': points}}


class HealthCurveTest(unittest.TestCase):
    def test_first_half_heart_costs_one_and_the_last_about_four(self):
        costs = [health_value(h) - health_value(h - 1) for h in range(6, 0, -1)]
        self.assertAlmostEqual(costs[0], 1.0, places=9)
        self.assertAlmostEqual(costs[-1], 4.0876, places=4)
        self.assertEqual(costs, sorted(costs))           # every further half heart costs more
        self.assertEqual([round(c, 4) for c in costs], [1.0, 1.0645, 1.2949, 1.8018, 2.6959, 4.0876])
        self.assertEqual(describe()['hurt'], [round(c, 3) for c in costs])

    def test_extra_health_costs_one_per_half_heart(self):
        self.assertAlmostEqual(health_value(9) - health_value(8), 1.0)
        self.assertEqual(health_value(-2), 0.0)


class CombatV2Test(unittest.TestCase):
    def episode(self, kind='normal', **start):
        r = CombatV2()
        r.reset(obs(**start), kind)
        return r

    def test_hurt_follows_the_curve_from_the_current_health(self):
        r = self.episode()
        self.assertAlmostEqual(r.step(obs(hearts=5, damage=1), 'running', 2)['hurt'], -1.0)
        self.assertAlmostEqual(r.step(obs(hearts=3, damage=3), 'running', 2)['hurt'],
                               -(health_value(5) - health_value(3)))

    def test_uncounted_self_damage_is_free_but_lowers_the_next_price(self):
        r = self.episode()
        free = r.step(obs(hearts=4, damage=0), 'running', 2)   # health lost, no counted damage
        self.assertEqual(free['hurt'], 0.0)
        hit = r.step(obs(hearts=3, damage=1), 'running', 2)
        self.assertAlmostEqual(hit['hurt'], -(health_value(4) - health_value(3)))

    def test_lethal_hit_charges_the_remaining_health_and_death(self):
        # The bridge cancels the lethal hit: health stays, player_damage grows by the hit's amount.
        r = self.episode(hearts=1)
        step = r.step(obs(hearts=1, damage=2), 'death', 2)
        self.assertAlmostEqual(step['hurt'], -health_value(1))
        self.assertAlmostEqual(step['hurt'], -4.0876, places=4)
        self.assertEqual(step['death'], -5.0)

    def test_time_bombs_timeout(self):
        r = self.episode()
        step = r.step(obs(bombs=0), 'running', 2)
        self.assertAlmostEqual(step['time'], -2 / 30 / 170)
        self.assertAlmostEqual(step['bomb'], -0.08)
        self.assertEqual(r.step(obs(bombs=1), 'running', 2)['bomb'], 0.0)   # a pickup is not a reward
        self.assertEqual(r.step(obs(bombs=1), 'time_limit', 2)['timeout'], -1.0)

    def test_clear_bonus_from_room_points_kill_points_and_floor(self):
        self.assertAlmostEqual(clear_bonus('normal', 24), 0.32)
        self.assertAlmostEqual(clear_bonus('boss', 16), 3.08)
        self.assertAlmostEqual(clear_bonus('arena', 16), 3.08)
        self.assertGreater(clear_bonus('boss', 0), clear_bonus('normal', 200))
        r = self.episode(kind='normal', points=24)
        self.assertAlmostEqual(r.step(obs(blocking=0), 'win', 2)['clear'], 0.32)

    def test_progress_total_is_the_start_potential_whatever_the_outcome(self):
        for outcome in ('win', 'death', 'time_limit'):
            r = self.episode(kind='boss', blocking=140)
            r.step(obs(blocking=100), 'running', 2)
            spawn = r.step(obs(blocking=130), 'running', 2)          # spawned or regrown HP
            self.assertAlmostEqual(spawn['progress'], -30 / 70)
            r.step(obs(blocking=90, damage=1 if outcome == 'death' else 0), outcome, 2)
            self.assertAlmostEqual(r.totals['progress'], 140 / 70, msg=outcome)

    def test_totals_track_every_component(self):
        r = self.episode()
        steps = [r.step(obs(hearts=5, damage=1, bombs=0, blocking=60), 'running', 2),
                 r.step(obs(hearts=5, damage=1, bombs=0, blocking=0), 'win', 2)]
        for k in COMPONENTS:
            self.assertAlmostEqual(r.totals[k], sum(s[k] for s in steps))
        self.assertEqual(len(r.totals_array()), len(COMPONENTS))


if __name__ == '__main__':
    unittest.main()


# ---- combat-v3 -------------------------------------------------------------------------------
from isaac_bridge.abplus_reward import COMPONENTS_V3, V3, CombatV3, describe_v3, idle_cost, time_cost


def obs3(hearts=6, bombs=1, damage=0.0, blocking=30.0, count=3):
    return {'players': [{'hearts': hearts, 'soul': 0, 'eternal': 0, 'bone': 0, 'bombs': bombs}],
            'combat': {'player_damage': damage, 'blocking_hp': blocking, 'blocking_points': 24,
                       'blocking_count': count}}


def play(r, steps, final=None, outcome='running', **state):
    """steps of 2 frames with a constant state, then one more step with `final`/`outcome`."""
    for _ in range(steps):
        r.step(obs3(**state), 'running', 2)
    return r.step(final if final is not None else obs3(**state), outcome, 2)


class CombatV3Test(unittest.TestCase):
    def episode(self, kind='normal', **start):
        r = CombatV3()
        r.reset(obs3(**start), kind)
        return r

    def total(self, r):
        return sum(r.totals.values())

    def test_time_price_is_the_integral_of_the_curve(self):
        r = self.episode()
        play(r, 899, count=3)      # 900 steps of 2 logic frames = 60 s, nothing hit
        self.assertAlmostEqual(-r.totals['time'], time_cost(0, 60, 3), places=6)
        self.assertAlmostEqual(time_cost(0, 60, 3), (1 / 120 + 3 / 60) * (60 + 60 ** 3 / (3 * 3600)))
        self.assertAlmostEqual(time_cost(0, 60, 1), 2.0)        # one enemy for a minute
        self.assertAlmostEqual(time_cost(0, 120, 1), 7.0)       # ... for the whole deadline (0.025 * 280)

    def test_twenty_seconds_without_a_hit_cost_ten_then_ten_per_minute(self):
        r = self.episode()
        play(r, 298)
        self.assertEqual(r.totals['idle'], 0.0)                  # 19.93 s
        step = r.step(obs3(), 'running', 2)                      # reaches 20 s
        self.assertAlmostEqual(step['idle'], -10.0, places=6)
        play(r, 1499)                                            # to 120 s
        self.assertAlmostEqual(r.totals['idle'], -idle_cost(0, 120), places=6)
        self.assertAlmostEqual(idle_cost(0, 120), 10.0 + 100 / 6)

    def test_every_hit_restarts_the_twenty_seconds(self):
        steady = self.episode()                                  # a tear every 15 s: never stalled
        hp = 30.0
        for step in range(1, 1801):
            if step % 225 == 0:
                hp -= 3.5
            steady.step(obs3(blocking=hp), 'running', 2)
        self.assertEqual(steady.totals['idle'], 0.0)
        once = self.episode()                                    # one tear at 10 s, then nothing
        play(once, 148)
        once.step(obs3(blocking=26.5), 'running', 2)
        play(once, 1649, blocking=26.5)
        self.assertAlmostEqual(once.t, 120.0, places=6)
        self.assertAlmostEqual(once.totals['idle'], -(10.0 + (120 - 30) / 6), places=6)

    def test_hits_and_kills_telescope_over_spawns(self):
        r = self.episode(blocking=30.0, count=3)
        step = r.step(obs3(blocking=26.5, count=3), 'running', 2)
        self.assertAlmostEqual(step['hit'], 3.5 / 20)
        step = r.step(obs3(blocking=46.5, count=5), 'running', 2)  # two enemies spawn
        self.assertAlmostEqual(step['hit'], -1.0)
        self.assertAlmostEqual(step['kill'], -2.0)
        r.step(obs3(blocking=0.0, count=0), 'win', 2)
        self.assertAlmostEqual(r.totals['hit'], 30.0 / 20)       # the start HP, whatever spawned
        self.assertAlmostEqual(r.totals['kill'], 3.0)
        self.assertEqual(r.totals['clear'], 3.0)

    def test_clear_bonus_by_kind_and_bomb_price(self):
        self.assertEqual(V3['clear'], {'normal': 3.0, 'boss': 6.0, 'arena': 6.0})
        r = self.episode('boss', blocking=250.0, count=1)
        self.assertAlmostEqual(r.step(obs3(bombs=0, blocking=190.0, count=1), 'running', 2)['bomb'], -0.1)
        self.assertEqual(r.step(obs3(bombs=0, blocking=0.0, count=0), 'win', 2)['clear'], 6.0)

    def test_clear_at_any_health_beats_an_untouched_timeout_which_beats_death(self):
        # Clear at 118 s, having lost 5 of 6 half hearts at once and hit first at 59 s.
        clear = self.episode()
        clear.step(obs3(hearts=1, damage=5), 'running', 2)
        play(clear, 882, hearts=1, damage=5)
        clear.step(obs3(hearts=1, damage=5, blocking=10.0, count=1), 'running', 2)
        play(clear, 884, final=obs3(hearts=1, damage=5, blocking=0.0, count=0), outcome='win', hearts=1, damage=5,
             blocking=10.0, count=1)
        self.assertAlmostEqual(clear.t, 118.0, places=6)
        # Untouched timeout: never hit, never hurt.
        timeout = self.episode()
        play(timeout, 1799, outcome='time_limit')
        self.assertAlmostEqual(timeout.t, 120.0, places=6)
        # Death at 30 s without a hit (the bridge cancels the lethal hit: damage grows, health stays).
        death = self.episode()
        play(death, 449, final=obs3(hearts=6, damage=6), outcome='death')
        self.assertGreater(self.total(clear), self.total(timeout))
        self.assertGreater(self.total(timeout), self.total(death))
        self.assertAlmostEqual(timeout.totals['timeout'], -20.0)

    def test_death_is_never_cheaper_than_waiting_for_the_timeout(self):
        # From the same state (3 enemies alive, nothing hit), dying at t costs at least what staying costs.
        for steps in (15, 600, 1200, 1790):
            stay = self.episode()
            play(stay, steps)
            stay_cost = self.total(stay)
            play(stay, 1799 - steps - 1, outcome='time_limit')
            self.assertAlmostEqual(stay.t, 120.0, places=6)
            die = self.episode()
            play(die, steps - 1, final=obs3(damage=6), outcome='death')
            self.assertLess(self.total(die), self.total(stay), steps)
            self.assertLess(self.total(die) - self.total(stay), -(V3['death'] - V3['timeout']) + 1e-6)
            self.assertLess(stay_cost, 0)

    def test_totals_and_describe(self):
        r = self.episode()
        r.step(obs3(blocking=26.5), 'running', 2)
        self.assertEqual(list(r.totals), list(COMPONENTS_V3))
        self.assertEqual(r.totals_array().shape, (len(COMPONENTS_V3),))
        self.assertEqual(describe_v3()['timeout'], -20.0)
        self.assertEqual(CombatV3.scale, 0.25)


# ---- combat-v4 (stage one) ---------------------------------------------------------------------
from isaac_bridge.abplus_reward import COMPONENTS_V4, V4, CombatV4, describe_v4


def obs4(bombs=1, damage=0.0, events=0, blocking=30.0, count=3):
    return {'players': [{'hearts': 6, 'soul': 0, 'eternal': 0, 'bone': 0, 'bombs': bombs}],
            'combat': {'player_damage': damage, 'player_damage_events': events, 'blocking_hp': blocking,
                       'blocking_points': 24, 'blocking_count': count}}


class CombatV4Test(unittest.TestCase):
    def episode(self, kind='normal', **start):
        r = CombatV4()
        r.reset(obs4(**start), kind)
        return r

    def total_of(self, steps, kind='normal', **start):
        """steps: list of (obs kwargs, outcome); returns the episode total."""
        r = self.episode(kind, **start)
        for kwargs, outcome in steps:
            r.step(obs4(**kwargs), outcome, 2)
        self.last = r
        return sum(r.totals.values())

    def test_the_design_trajectories(self):
        hide = self.total_of([({}, 'running')] * 1799 + [({}, 'time_limit')])
        self.assertEqual(hide, 0.0)                                # nothing negative to avoid
        one_hit = self.total_of([({'blocking': 26.5}, 'running')] + [({'blocking': 26.5}, 'time_limit')])
        self.assertAlmostEqual(one_hit, 3.5 / 20)
        # Attack for 10 s, deal 60 HP to a boss, get hit three times, die.
        attack = self.total_of([({'blocking': 250.0 - 60.0 * k / 149, 'events': k // 50, 'damage': k // 50, 'count': 1},
                            'running') for k in range(149)] +
                          [({'blocking': 190.0, 'events': 3, 'damage': 3, 'count': 1}, 'death')],
                          'boss', blocking=250.0, count=1)
        self.assertGreater(attack, 0.0)
        self.assertAlmostEqual(self.last.totals['death'], -0.5)
        self.assertAlmostEqual(self.last.totals['hurt'], -0.3)
        clear = self.total_of([({'blocking': 10.0, 'count': 1}, 'running'), ({'blocking': 0.0, 'count': 0}, 'win')])
        self.assertAlmostEqual(clear, 30 / 20 + 3 * 0.25 + 30.0)

    def test_every_clear_beats_every_non_clear(self):
        worst_clear = self.episode(bombs=3, blocking=30.0, count=3)
        worst_clear.step(obs4(bombs=0, damage=6, events=6, blocking=0.0, count=0), 'win', 2)
        self.assertAlmostEqual(worst_clear.totals['clear_health'], -3.0)
        self.assertAlmostEqual(worst_clear.totals['clear_bombs'], -1.5)
        # The best non-clear: the largest room of the mixture (boss room 1047, 415 HP) down to 1 HP.
        best_miss = self.episode('boss', blocking=415.0, count=1)
        best_miss.step(obs4(blocking=1.0, count=1), 'time_limit', 2)
        self.assertGreater(sum(worst_clear.totals.values()), sum(best_miss.totals.values()) + 5)
        self.assertGreater(V4['clear'] - 6 * V4['lambda_health'] - 3 * V4['lambda_bomb'], 415 / V4['hit_hp'] + 1)

    def test_hurt_is_flat_per_event_and_spawns_telescope(self):
        r = self.episode()
        self.assertAlmostEqual(r.step(obs4(events=2, damage=4), 'running', 2)['hurt'], -0.2)
        step = r.step(obs4(events=2, damage=4, blocking=50.0, count=5), 'running', 2)  # two enemies spawn
        self.assertAlmostEqual(step['hit'], -1.0)
        self.assertAlmostEqual(step['kill'], -0.5)
        r.step(obs4(events=2, damage=4, blocking=0.0, count=0), 'win', 2)
        self.assertAlmostEqual(r.totals['hit'], 30 / 20)
        self.assertAlmostEqual(r.totals['kill'], 0.75)
        self.assertEqual(list(r.totals), list(COMPONENTS_V4))
        self.assertEqual(CombatV4.scale, 0.1)
        self.assertIn('truncation', describe_v4()['timeout'])


from isaac_bridge.abplus_geometry import CELL, TEAR_RADIUS  # noqa: E402
from isaac_bridge.abplus_reward import COMPONENTS_V5, V5, CombatV5, describe_v5  # noqa: E402

TAU = 13 + TEAR_RADIUS


def obs5(bombs=1, damage=0.0, events=0, dealt=0.0, kills=0, x=100.0, y=300.0, targets=((300.0, 300.0),),
         blocking=30.0, count=3):
    """A player at (x, y) (tear range 260) and lineage NPCs of size 13 at targets."""
    return {'players': [{'bombs': bombs, 'pos': [x, y], 'range': 260.0}],
            'entities': [{'pos': list(t), 'size': 13.0, 'lineage': True, 'blocking': True, 'enemy': True}
                         for t in targets],
            'combat': {'player_damage': damage, 'player_damage_events': events, 'lineage_damage': dealt,
                       'lineage_kills': kills, 'lineage_hp': 30.0 - dealt, 'lineage_count': len(targets),
                       'blocking_hp': blocking, 'blocking_count': count}}


class CombatV5Test(unittest.TestCase):
    def episode(self, kind='normal', **start):
        r = CombatV5()
        r.reset(obs5(**start), kind)
        return r

    def run_steps(self, r, steps):
        for kwargs, outcome in steps:
            r.step(obs5(**kwargs), outcome, 2)
        return r

    def test_hits_and_kills_count_lineage_damage_only(self):
        r = self.episode()
        self.assertAlmostEqual(r.step(obs5(dealt=3.5), 'running', 2)['hit'], 3.5 / V5['hit_hp'])   # one tear
        # A spawn (doors-blocking HP and count up, lineage untouched) costs nothing any more.
        step = r.step(obs5(dealt=3.5, blocking=60.0, count=8), 'running', 2)
        self.assertEqual((step['hit'], step['kill']), (0.0, 0.0))
        # A kill whose death leaves new NPCs is positive at that step.
        step = r.step(obs5(dealt=13.0, kills=1, blocking=62.0, count=11), 'running', 2)
        self.assertAlmostEqual(step['hit'] + step['kill'], 9.5 / V5['hit_hp'] + 0.25)

    def test_alignment_potential_telescopes(self):
        # Walk from 3 cells off the firing band into it, wait, back off: the discounted sum of the
        # shaping is gamma^T Phi(s_T) - Phi(s_0) whatever the path.
        ys = [300 + TAU + 120 - 4 * k for k in range(31)] + [300.0] * 20 + [300 + 3 * k for k in range(20)]
        r = self.episode(y=ys[0])
        g, discounted = 1.0, 0.0
        for y in ys[1:]:
            discounted += g * r.step(obs5(y=y), 'running', 2)['align']
            g *= V5['gamma']
        phi0 = -V5['align'] * 120 / CELL
        phi_t = -V5['align'] * max(0.0, ys[-1] - 300 - TAU) / CELL
        self.assertAlmostEqual(discounted, g * phi_t - phi0, places=9)
        # One cell closer pays about 0.2, 0.6 of a tear.
        r = self.episode(y=300 + TAU + 80)
        self.assertAlmostEqual(r.step(obs5(y=300 + TAU + 40), 'running', 2)['align'], 0.2, places=2)

    def test_clear_and_death_end_the_potential_truncation_keeps_it(self):
        far = 300 + TAU + 100                        # 2.5 cells off the horizontal band: Phi = -0.5
        self.assertAlmostEqual(self.episode(y=far).step(obs5(y=far), 'death', 2)['align'], 0.5)
        self.assertAlmostEqual(self.episode(y=far).step(obs5(y=far), 'win', 2)['align'], 0.5)
        self.assertAlmostEqual(self.episode(y=far).step(obs5(y=far), 'time_limit', 2)['align'],
                               0.5 * (1 - V5['gamma']))

    def test_every_clear_beats_every_non_clear_of_the_room(self):
        far = 300 + TAU + 200
        hide = self.run_steps(self.episode(y=far), [({'y': far}, 'running')] * 1799 + [({'y': far}, 'time_limit')])
        self.assertEqual(hide.totals['hit'] + hide.totals['kill'], 0.0)
        # The worst clear (3 bombs used, 5 half hearts lost in 5 hits) against the best miss of the
        # same 30 HP room (29.9 HP dealt, 2 kills, aligned at the deadline), from the same start.
        clear = self.episode(y=far, bombs=3)
        self.run_steps(clear, [({'y': far, 'bombs': 0, 'damage': 5, 'events': 5, 'dealt': 30.0, 'kills': 3}, 'win')])
        miss = self.run_steps(self.episode(y=far), [({'y': 300.0, 'dealt': 29.9, 'kills': 2}, 'time_limit')])
        self.assertGreater(sum(clear.totals.values()), sum(miss.totals.values()) + 20)

    def test_totals_and_describe(self):
        r = self.run_steps(self.episode(), [({'dealt': 30.0, 'kills': 3}, 'win')])
        self.assertEqual(list(r.totals), list(COMPONENTS_V5))
        self.assertAlmostEqual(r.totals['hit'], 30.0 / V5['hit_hp'])
        self.assertAlmostEqual(r.totals['clear'], 30.0)
        self.assertEqual(CombatV5.scale, 0.1)
        self.assertEqual(V5['hit_hp'], 5.0)
        self.assertEqual(CombatV5(hit_hp=10.0).hit_hp, 10.0)
        d = describe_v5(align=0.4, gamma=0.999)
        self.assertIn('0.4', d['align'])
        self.assertIn('0.999', d['align'])
        self.assertIn('truncation', d['timeout'])
