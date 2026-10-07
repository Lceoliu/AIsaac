"""COMBAT options that leave their room (user decision 2026-09-30, EXPERIMENTS.md A13): with combat_multi_room the option
goes on in other rooms and is won only by its own room's clear (there: the room's clear; elsewhere: that room's
descriptor); another room's clear is not a win; without it a room change is an error, as in every earlier run; the history
restarts at a room change but keeps the option's clock; a COMBAT won from outside its room ends the option plan."""
import unittest

import numpy as np

from isaac_bridge.abplus_options import OptionSequence
from isaac_bridge.env import BridgeError
from isaac_bridge.monstro_gym import MonstroGymEnv
from isaac_bridge.transformer_obs import VisibleHistory

COMBAT = dict(player_damage_events=0, enemy_damage_events=0, enemy_damage_fraction=0.0)


def obs(frames, room, clear=False, dead=False):
    return dict(logic_frames=frames, room=dict(room_idx=room, clear=clear), players=[dict(dead=dead)], combat=dict(COMBAT),
                nav={})


class FakeBridge:
    """Plays a script of (room, clear, dead) per step, 4 frames each; lua answers the descriptor query from `cleared`."""
    action_repeat = 4
    stop_clear = True

    def __init__(self, script, cleared=()):
        self.script, self.frames, self.cleared, self.queries = list(script), 100, set(cleared), []

    def step(self, action, repeat=None):
        room, clear, dead = self.script.pop(0)
        self.frames += repeat
        return obs(self.frames, room, clear, dead), 0.0, False, False, {}

    def lua(self, code):
        self.queries.append(code)
        room = int(code.split('GetRoomByIdx(')[1].split(')')[0])
        return 'true' if room in self.cleared else 'false'


class Env(MonstroGymEnv):
    def __init__(self, script, cleared=(), multi_room=True, max_frames=40):
        super().__init__(max_episode_frames=max_frames, bridge=FakeBridge(script, cleared))
        self.combat_multi_room = multi_room
        self.raw_obs, self.finished, self.elapsed_frames = obs(100, 71), False, 0
        self.begin_option('combat', max_frames)
        self.room_changes = 0

    def encode_observation(self, o):
        return None

    def on_room_change(self):
        self.room_changes += 1


ACTION = np.array([0, 0, 0, 0])


def play(env):
    outcomes = []
    while True:
        _, _, terminated, truncated, info = env.step(ACTION)
        outcomes.append(info['outcome'])
        if terminated or truncated:
            return outcomes


class CombatRoomsTest(unittest.TestCase):
    def test_own_room_clear_wins_as_before(self):
        env = Env([(71, False, False), (71, True, False)])
        self.assertEqual(play(env), ['running', 'win'])
        self.assertEqual((env.room_changes, env.bridge.queries), (0, []))   # no descriptor query in its own room

    def test_a_clear_room_elsewhere_is_no_win(self):
        # out through a blown door into the (clear) start room, on until the deadline: time_limit, never 'win'
        env = Env([(71, False, False)] + [(84, True, False)] * 9)
        out = play(env)
        self.assertEqual(out[-1], 'time_limit')
        self.assertNotIn('win', out)
        self.assertEqual(env.room_changes, 1)

    def test_another_room_turning_clear_is_no_win(self):
        env = Env([(58, False, False), (58, True, False), (58, True, False)], max_frames=12)
        self.assertEqual(play(env), ['running', 'running', 'time_limit'])

    def test_own_room_cleared_by_its_descriptor_from_elsewhere(self):
        # left during the clear countdown: the engine clears the room behind the player (A13)
        env = Env([(58, False, False), (58, False, False)], cleared={71})
        self.assertEqual(play(env), ['win'])

    def test_back_in_its_room_and_cleared(self):
        env = Env([(58, False, False), (71, False, False), (71, True, False)])
        self.assertEqual(play(env), ['running', 'running', 'win'])
        self.assertEqual(env.room_changes, 2)

    def test_death_elsewhere(self):
        env = Env([(58, False, False), (58, False, True)])
        self.assertEqual(play(env), ['running', 'death'])

    def test_room_change_without_multi_room_is_an_error(self):
        env = Env([(58, False, False)], multi_room=False)
        with self.assertRaises(BridgeError):
            env.step(ACTION)

    def test_room_clear_keeps_the_option_clock(self):
        h = VisibleHistory(4, 8)
        h.origin, h.previous, h.combat = 1234, {'x': 1}, {'start': 5}
        h.frames.append({'t': 0})
        h.room_clear()
        self.assertEqual((h.origin, h.previous, h.combat, len(h.frames)), (1234, None, None, 0))
        h.clear()
        self.assertIsNone(h.origin)

    def test_combat_won_from_outside_ends_the_plan(self):
        info = dict(chain=(1, 71, 1, 58, 5, 6))
        seq = OptionSequence(dict(mode='chain', goals=1), 7, info)
        seq.option = dict(seq.plan[0], room=71)
        self.assertFalse(seq.next('win', 58))
        self.assertEqual(seq.index, 0)
        self.assertTrue(seq.next('win', 71))
        self.assertEqual(seq.index, 1)


if __name__ == '__main__':
    unittest.main()
