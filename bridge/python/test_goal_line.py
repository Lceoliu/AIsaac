"""Goal-conditioned line M0 (rl/docs/GOAL_CONDITIONED_DESIGN.md), engine-free parts: the walking-distance field and goal
sampling (abplus_nav), the GOTO reward's discounted shaping and early-failure cost (abplus_reward.GotoReward), the item mask
by availability, the group modes and the option plans (abplus_worker)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_bridge import abplus_nav as nav
from isaac_bridge.abplus_groups import load_groups
from isaac_bridge.abplus_reward import GotoReward, remaining_time_cost
from isaac_bridge.transformer_obs import PLAYER_FIELDS

W, H = 15, 9
X0, Y0 = 40.0, 120.0   # cell 0 centre of a 1x1 room


def room_obs(blocked=(), hazards=(), player=(3, 4), fires=()):
    """A 15 x 9 room grid: the wall ring not walkable, `blocked` interior cells solid, `hazards` walkable hazards."""
    cells = []
    for index in range(W * H):
        row, col = divmod(index, W)
        inside = 0 < row < H - 1 and 0 < col < W - 1
        walkable = inside and (col, row) not in blocked
        cells.append([index, X0 + 40 * col, Y0 + 40 * row, 0 if walkable else 4, int(inside), int(walkable), 0,
                      int((col, row) in hazards), int(not walkable), 0])
    px, py = X0 + 40 * player[0], Y0 + 40 * player[1]
    return {'terrain': {'width': W, 'height': H, 'version': None, 'cells': cells},
            'players': [{'pos': [px, py]}],
            'entities': [{'type': 33, 'pos': [X0 + 40 * c, Y0 + 40 * r]} for c, r in fires],
            'doors': [{'slot': 2, 'open': True, 'pos': [X0 + 40 * (W - 1), Y0 + 40 * 4]}],
            'room': {'gw': W, 'gh': H}}


class NavTest(unittest.TestCase):
    def test_straight_distance(self):
        obs = room_obs()
        field = nav.goal_field(obs, (X0 + 40 * 8, Y0 + 40 * 4))
        self.assertAlmostEqual(nav.goal_distance(X0 + 40 * 3, Y0 + 40 * 4, field), 5 * 40.0)

    def test_detour_around_wall(self):
        wall = {(6, r) for r in range(1, 8) if r != 1}   # a wall with a gap at the top
        obs = room_obs(blocked=wall)
        field = nav.goal_field(obs, (X0 + 40 * 9, Y0 + 40 * 4))
        d = nav.goal_distance(X0 + 40 * 3, Y0 + 40 * 4, field)
        self.assertGreater(d, 6 * 40.0 + 40)   # longer than the straight line through the wall
        self.assertTrue(all(c not in field['field'] for c in wall))

    def test_hazard_cost_and_fire_block(self):
        obs = room_obs(hazards={(5, 4)})
        target = (X0 + 40 * 7, Y0 + 40 * 4)
        clean = nav.goal_distance(X0 + 40 * 3, Y0 + 40 * 4, nav.goal_field(room_obs(), target))
        spiky = nav.goal_distance(X0 + 40 * 3, Y0 + 40 * 4, nav.goal_field(obs, target))
        self.assertGreater(spiky, clean - 1e-6)   # goes around or pays
        fire = nav.goal_field(room_obs(fires=[(5, 4)]), target)
        self.assertNotIn((5, 4), fire['field'])

    def test_unreachable_is_none(self):
        box = {(c, r) for c in (8, 10) for r in range(1, 8)} | {(9, 1), (9, 7)}
        obs = room_obs(blocked=box - {(9, 4)} | {(9, 3), (9, 5)})
        field = nav.goal_field(obs, (X0 + 40 * 9, Y0 + 40 * 4))
        self.assertIsNone(nav.goal_distance(X0 + 40 * 3, Y0 + 40 * 4, field))

    def test_door_target_inside(self):
        obs = room_obs()
        x, y = nav.door_target(obs, 2)
        self.assertEqual((x, y), (X0 + 40 * (W - 2), Y0 + 40 * 4))

    def strata(self, obs, draws=80):
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(draws):
            got = nav.sample_goal(obs, rng)
            self.assertIsNotNone(got)
            _, kind, d = got
            self.assertGreaterEqual(d, nav.MIN_GOAL_CELLS * 40)
            seen.add(kind)
        return seen

    def test_sample_goal_strata(self):
        # one rock in the way: going round it never walks away from the goal (a detour); a long wall with a gap at one
        # end: heading straight at a goal behind it walks into a dead end
        self.assertEqual(self.strata(room_obs(blocked={(6, 4)})) & {'straight', 'detour'}, {'straight', 'detour'})
        self.assertIn('dead_end', self.strata(room_obs(blocked={(6, r) for r in range(2, 8)})))

    def test_sample_goal_weights(self):
        # C43: no weights = the uniform draw and its random numbers unchanged; weights draw among the strata present
        wall = room_obs(blocked={(6, r) for r in range(2, 8)})
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        for _ in range(30):
            self.assertEqual(nav.sample_goal(wall, r1), nav.sample_goal(wall, r2, weights=None))
        rng = np.random.default_rng(6)
        kinds = {nav.sample_goal(wall, rng, weights={'straight': 0, 'detour': 0, 'dead_end': 1})[1] for _ in range(40)}
        self.assertEqual(kinds, {'dead_end'})
        rng = np.random.default_rng(7)
        draws = [nav.sample_goal(wall, rng, weights={'straight': 1, 'detour': 2, 'dead_end': 2})[1] for _ in range(300)]
        self.assertLess(draws.count('straight'), 100)
        # a stratum the room lacks is skipped (an open room has no dead end): the weights of the others decide
        rng = np.random.default_rng(8)
        open_kinds = {nav.sample_goal(room_obs(), rng, weights={'straight': 1, 'detour': 0, 'dead_end': 5})[1] for _ in range(40)}
        self.assertEqual(open_kinds, {'straight'})


    def test_door_approach_aligns_then_enters(self):
        obs = room_obs()
        inside = nav.door_target(obs, 2)                   # the right door's inside cell
        field = nav.goal_field(obs, inside)
        field['exit'] = tuple(obs['doors'][0]['pos'])
        ex, ey = field['exit']
        # at the inside cell 15 px below the door's axis: on through the door with a diagonal correction (up-right, code
        # 2); nearly on the axis: straight right (3); pushed into the door cell already: never back into the room
        self.assertEqual(nav.descent_move(inside[0], ey + 15, field), 2)
        self.assertEqual(nav.descent_move(inside[0], ey + 2, field), 3)
        self.assertEqual(nav.descent_move(inside[0] + 25, ey - 3, field), 3)
        # further away: down the field towards the inside cell
        self.assertIn(nav.descent_move(X0 + 40 * 3, Y0 + 40 * 4, field), (3, 2, 4))


class GotoRewardTest(unittest.TestCase):
    def obs(self, damage=0.0):
        return {'combat': {'player_damage': damage}}

    def test_shaping_telescopes_to_minus_phi0(self):
        """Discounted shaping of an episode = -Phi(s0) whatever the path (Phi = 0 at the terminal)."""
        r = GotoReward(gamma=0.99)
        for path in ([200, 160, 120, 80, 40], [200, 240, 200, 160]):
            r.reset(self.obs(), path[0])
            total, discount = 0.0, 1.0
            for i, d in enumerate(path[1:] + [None]):
                outcome = 'running' if d is not None else 'time_limit'
                parts = r.step(self.obs(), outcome, d, 10)
                total += discount * parts['shaping']
                discount *= r.gamma
            self.assertAlmostEqual(total, -r.potential(path[0]), places=9)

    def test_failure_cost_is_the_discounted_time(self):
        gamma, c, n = 0.999, 0.1, 37
        self.assertAlmostEqual(remaining_time_cost(c, gamma, n), sum(c * gamma ** k for k in range(1, n + 1)), places=12)
        r = GotoReward(gamma=gamma)
        r.reset(self.obs(), 100)
        parts = r.step(self.obs(), 'exit', None, n)
        self.assertAlmostEqual(parts['fail'], -remaining_time_cost(r.time, gamma, n))
        self.assertEqual(parts['goal'], 0.0)

    def test_goal_hurt_and_time(self):
        r = GotoReward()
        r.reset(self.obs(0.0), 80)
        parts = r.step(self.obs(2.0), 'goal', None, 5)
        self.assertEqual(parts['goal'], r.goal)
        self.assertEqual(parts['hurt'], -2 * r.hurt)
        self.assertEqual(parts['time'], -r.time)
        self.assertEqual(parts['fail'], 0.0)
        self.assertAlmostEqual(r.scale * r.goal, 1.0)       # trained units: +1, -0.01, -0.1 (user table)
        self.assertAlmostEqual(r.scale * r.time, 0.01)
        self.assertAlmostEqual(r.scale * r.hurt, 0.1)
        self.assertAlmostEqual(r.scale * r.alpha, 0.02)


class MaskAndGroupsTest(unittest.TestCase):
    def test_active_ready_index(self):
        from isaac_bridge.gpu_ppo import ACTIVE_READY
        self.assertEqual(PLAYER_FIELDS.index('active_ready'), ACTIVE_READY)

    def test_item_mask(self):
        import torch
        from isaac_bridge.gpu_ppo import action_masks
        nvec = (9, 5, 2, 2)
        bombs = torch.tensor([True, False])
        never = action_masks(nvec, bombs)
        self.assertFalse(never[:, 17].any())
        ready = action_masks(nvec, bombs, item_ok=torch.tensor([True, False]))
        self.assertEqual(ready[:, 17].tolist(), [True, False])
        self.assertTrue(ready[:, 16].all())   # no item is always allowed

    def test_group_modes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'rooms.json').write_text(json.dumps({'weights': {'normal': 1}, 'normal': [1, 2], 'boss': []}))
            (d / 'groups.json').write_text(json.dumps({'groups': [
                {'name': 'single', 'tasks': 'rooms.json', 'share': 1},
                {'name': 'chain', 'tasks': 'rooms.json', 'share': 1, 'mode': 'chain', 'goals': 1},
                {'name': 'empty', 'tasks': 'rooms.json', 'share': 1, 'mode': 'goto_empty', 'goals': 4}]}))
            groups = load_groups(d / 'groups.json')
            self.assertNotIn('mode', groups[0])
            self.assertEqual((groups[1]['mode'], groups[1]['goals']), ('chain', 1))
            self.assertEqual(groups[2]['goto_seconds'], 8.0)
            (d / 'bad.json').write_text(json.dumps({'groups': [{'name': 'x', 'tasks': 'rooms.json', 'share': 1,
                                                                'mode': 'teleport'}]}))
            with self.assertRaises(ValueError):
                load_groups(d / 'bad.json')

    def test_option_plans(self):
        from isaac_bridge.abplus_worker import Worker
        w = Worker.__new__(Worker)
        w.config = {'groups': [dict(mode='combat'), dict(mode='combat_goto', goals=2), dict(mode='goto_empty', goals=3),
                               dict(mode='chain', goals=1)]}
        tasks = lambda plan: [(o['task'], o['source']) for o in plan]
        self.assertEqual(tasks(w.option_plan(0, {})), [('combat', 'single')])
        self.assertEqual(tasks(w.option_plan(1, {})), [('combat', 'single')] + [('goto_position', 'goto')] * 2)
        self.assertEqual(tasks(w.option_plan(2, {})), [('goto_position', 'goto')] * 3)
        chain = w.option_plan(3, {'chain': (1, 71, 2, 72, 62, 215)})
        self.assertEqual(tasks(chain), [('combat', 'chain'), ('goto_position', 'goto'), ('goto_door', 'goto'),
                                        ('combat', 'chain')])
        self.assertEqual((chain[2]['slot'], chain[2]['room'], chain[3]['room'], chain[3]['variant']), (2, 72, 72, 215))


class OptionSequenceTest(unittest.TestCase):
    def test_next_only_after_success(self):
        from isaac_bridge.abplus_options import OptionSequence
        seq = OptionSequence(dict(mode='chain', goals=1), 7, {'chain': (1, 71, 2, 72, 62, 215)})
        self.assertEqual(len(seq.plan), 4)
        for outcome in ('death', 'time_limit', 'exit', 'wrong_door', 'hurt'):
            self.assertFalse(seq.next(outcome))
            self.assertEqual(seq.index, 0)
        self.assertTrue(seq.next('win'))
        self.assertTrue(seq.next('goal'))
        self.assertTrue(seq.next('goal'))
        self.assertEqual(seq.plan[seq.index]['task'], 'combat')
        self.assertFalse(seq.next('win'))   # the last option


if __name__ == '__main__':
    unittest.main()
