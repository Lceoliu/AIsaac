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
