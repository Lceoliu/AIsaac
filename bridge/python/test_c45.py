"""C45 (goal-hp3, user decisions 2026-10-01): the scripted A* expert's labels (transformer_obs.expert_label): GOTO frames the
scripted navigator's move, COMBAT frames only after STUCK_FRAMES pushing against the terrain (then the approach moves), the
terrain test; goal-hp3 adds no network input (a goal-hp2 policy's parameters and outputs carry over); the frame layout."""
import unittest

import numpy as np
import torch

from isaac_bridge.abplus_nav import descent_move, goal_field
from isaac_bridge.abplus_worker import frame_layout, observation_options
from isaac_bridge.transformer_obs import CELL, STUCK_FRAMES, VisibleHistory, terrain_channels
from test_c44 import policy, room
from test_goal_model import heads, random_windows, static_heads


def history(task='combat', nav=None):
    h = VisibleHistory(**observation_options('goal-hp3'))
    h.goal_state = dict(task=task, target=None, distance=None, source='single', nav_state={'nav': nav} if nav else None)
    return h


class ExpertLabelTest(unittest.TestCase):
    def setUp(self):
        self.obs = room(9, 15, rocks={(4, 7)})
        self.obs['entities'] = []
        self.full = terrain_channels(self.obs, (9, 15))
        self.x, self.y = CELL * 6, CELL * 4   # the cell left of the rock

    def test_blocked_by_terrain(self):
        self.assertTrue(VisibleHistory.blocked_by_terrain(self.obs, self.full, self.x, self.y, 3))    # right: the rock
        self.assertFalse(VisibleHistory.blocked_by_terrain(self.obs, self.full, self.x, self.y, 7))   # left: open
        self.assertTrue(VisibleHistory.blocked_by_terrain(self.obs, self.full, self.x, self.y, 2))    # up-right: via the rock
        self.assertTrue(VisibleHistory.blocked_by_terrain(self.obs, self.full, CELL * 1, CELL * 4, 7))  # the wall

    def step(self, h, pos, move, approach, dt=4):
        h.previous_action[:] = 0
        h.previous_action[move] = 1
        frame = {'approach': np.asarray(approach, np.float32)}
        label = h.expert_label(self.obs, {'pos': list(pos)}, self.full, dt, frame)
        h.previous = {'players': [{'pos': list(pos)}]}
        return label

    def test_combat_label_after_three_seconds_against_the_rock(self):
        h = history('combat')
        h.previous = {'players': [{'pos': [self.x, self.y]}]}
        approach = [0, 1, 0, 0, 0, 1, 0, 0, 0]   # up or down goes round
        labels = [self.step(h, (self.x, self.y), 3, approach) for _ in range(STUCK_FRAMES // 4 + 2)]
        first = next(i for i, l in enumerate(labels) if l.any())
        self.assertEqual(first, int(np.ceil(STUCK_FRAMES / 4)) - 1)   # 4 frames a decision: 23 decisions = 92 frames
        np.testing.assert_allclose(labels[-1], np.asarray(approach) / 2)
        # moving again clears it
        self.assertFalse(self.step(h, (self.x - 20, self.y), 7, approach).any())
        self.assertEqual(h.stuck_frames, 0)

    def test_no_combat_label_when_not_blocked_by_terrain(self):
        h = history('combat')
        x = CELL * 4   # open floor all round: standing still while pushing left (an enemy, a knock-back) is no terrain block
        h.previous = {'players': [{'pos': [x, self.y]}]}
        labels = [self.step(h, (x, self.y), 7, [0, 1, 0, 0, 0, 0, 0, 0, 0]) for _ in range(40)]
        self.assertFalse(any(l.any() for l in labels))

    def test_goto_label_is_the_scripted_move(self):
        target = (CELL * 9, CELL * 4)   # behind the rock
        nav = goal_field(self.obs, target)
        h = history('goto_position', nav)
        label = self.step(h, (self.x, self.y), 0, [0] * 9)
        self.assertEqual(label.sum(), 1.0)
        self.assertEqual(int(label.argmax()), descent_move(self.x, self.y, nav))
        self.assertNotEqual(int(label.argmax()), 3)   # not straight into the rock


class Goal3ModelTest(unittest.TestCase):
    def test_expert_move_is_no_input(self):
        s2 = VisibleHistory(**observation_options('goal-hp2')).space
        s3 = VisibleHistory(**observation_options('goal-hp3')).space
        self.assertEqual(set(s3.spaces) - set(s2.spaces), {'expert_move'})
        p2, p3 = policy(s2, 7, lr_groups=True), policy(s3, 7, lr_groups=True)
        self.assertEqual(list(p2.state_dict()), list(p3.state_dict()))
        p3.load_state_dict(p2.state_dict())
        rng = np.random.default_rng(9)
        obs = random_windows(s3, 3, ('combat', 'goto_position', 'goto_door'), rng)
        obs['room_bits'] = torch.as_tensor(rng.integers(0, 256, tuple(obs['room_bits'].shape)), dtype=torch.float32)
        with torch.no_grad():
            for path in (heads, static_heads):
                a = path(p2, {k: v for k, v in obs.items() if k in s2.spaces})
                b = path(p3, obs)
                self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_frame_layout(self):
        dtype, keys = frame_layout('goal-hp3')
        self.assertIn('expert_move', keys)
        self.assertEqual(dtype.fields['expert_move'][0].shape, (9,))
        self.assertNotIn('expert_move', frame_layout('goal-hp2')[1])


if __name__ == '__main__':
    unittest.main()
