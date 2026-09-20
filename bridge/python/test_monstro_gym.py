import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from isaac_bridge.monstro_gym import MonstroGymEnv, actor_observation, ENTITY_SLOTS, TERRAIN_SHAPE
from isaac_bridge.env import BridgeError


FIXTURE = Path(__file__).resolve().parent / "fixtures/monstro_initial_obs.json"


class RecordedBridge:
    action_repeat = 4

    def __init__(self):
        self.original = json.loads(FIXTURE.read_text(encoding="utf-8-sig"))
        self.mode = "running"
        self.closed = False
        self.safe = False

    def connect(self): pass

    def reset_monstro(self, seed):
        self.obs = copy.deepcopy(self.original)
        return self.obs, {"hidden_seed": seed}

    def step(self, action, repeat):
        self.obs = copy.deepcopy(self.obs)
        self.obs["logic_frames"] += repeat
        self.obs["players"][0]["dead"] = self.mode == "death"
        if self.mode == 'death': self.obs['combat']['player_damage_events'] += 1
        self.obs["room"]["clear"] = self.mode in ("win", "death")
        return self.obs, 0, False, False, {"hidden_seed": 1234}

    def reset_safe(self): self.safe = True
    def close(self): self.closed = True


class GymTest(unittest.TestCase):
    def setUp(self):
        self.bridge = RecordedBridge()
        self.env = MonstroGymEnv(bridge=self.bridge, max_episode_frames=9)

    def tearDown(self): self.env.close()

    def test_sb3_contract(self):
        check_env(self.env, warn=True)

    def test_adjacent_empty_room_is_not_a_boss_win(self):
        self.env.reset()
        self.bridge.mode = 'win'
        self.bridge.obs = copy.deepcopy(self.bridge.obs)
        self.bridge.obs['room']['room_idx'] += 1
        with self.assertRaisesRegex(BridgeError, 'arena room changed'):
            self.env.step([0, 0, 0, 0])

    def test_time_limit_exact_remainder(self):
        self.env.reset()
        results = [self.env.step([0, 0, 0, 0]) for _ in range(3)]
        self.assertEqual([r[4]["logic_frames_advanced"] for r in results], [4, 4, 1])
        self.assertEqual(results[-1][1:4], (0., False, True))
        with self.assertRaises(RuntimeError): self.env.step([0, 0, 0, 0])
        self.env.reset()
        self.assertFalse(self.env.step([0, 0, 0, 0])[3])

    def test_win_and_death(self):
        for mode, reward in [("win", 1.), ("death", -1.)]:
            self.env.reset()
            self.bridge.mode = mode
            _, r, terminal, truncated, info = self.env.step([0, 0, 0, 0])
            self.assertEqual((r, terminal, truncated, info["outcome"]), (reward, True, False, mode))

    def test_actor_ignores_hidden_fields(self):
        raw = copy.deepcopy(self.bridge.original)
        before = actor_observation(raw)
        raw["seed"] = 999
        raw["room"]["alive"] = 666
        raw["events"] = {"damage": 999}
        raw['combat']['enemy_damage'] = 999
        for e in raw["entities"]:
            e.update(id=999-e["id"], hp=999, state=999, pcool=999, vel=[99, 99])
        raw["entities"].reverse()
        for key, tensor in actor_observation(raw).items():
            np.testing.assert_array_equal(before[key], tensor)

    def test_overflow_not_silently_truncated(self):
        raw = copy.deepcopy(self.bridge.original)
        raw["entities"] = raw["entities"][:1] * (ENTITY_SLOTS + 1)
        with self.assertRaises(ValueError): actor_observation(raw)

    def test_dense_damage_reward_and_no_repeated_credit(self):
        self.env.reset()
        self.bridge.obs['combat'].update(player_damage_events=1, enemy_damage_events=1,
                                         enemy_damage=10, enemy_damage_fraction=.04)
        # Change the next observation, not the previous observation stored by Gym.
        self.env.raw_obs = copy.deepcopy(self.bridge.original)
        _, reward, _, _, info = self.env.step([0,0,0,0])
        self.assertAlmostEqual(reward, -.91)
        self.assertEqual(info['reward_components'], dict(hurt=-1., hit=.05, damage=.04, clear=0.))
        self.assertEqual(self.env.step([0,0,0,0])[1], 0.)

    def test_attempted_damage_without_settled_loss_is_not_penalized(self):
        self.env.reset()
        self.bridge.obs['events']['damage'] += 1
        self.assertEqual(self.env.step([0,0,0,0])[1], 0.)

    def test_terrain_is_in_actor_input_and_updates(self):
        raw = copy.deepcopy(self.bridge.original)
        before = actor_observation(raw)
        self.assertEqual(before['terrain'].shape, TERRAIN_SHAPE)
        cell = next(c for c in raw['terrain']['cells'] if c[5]==1)
        row,col = divmod(cell[0],raw['terrain']['width'])
        cell[3],cell[5] = 3,0
        after = actor_observation(raw)
        self.assertEqual(before['terrain'][1,row,col],1)
        self.assertEqual(after['terrain'][1,row,col],0)
        self.assertAlmostEqual(float(after['terrain'][2,row,col]),.6)

    def test_no_invalid_action_or_use_before_reset(self):
        with self.assertRaises(RuntimeError): self.env.step([0, 0, 0, 0])
        self.env.reset()
        with self.assertRaises(ValueError): self.env.step([9, 0, 0, 0])

    def test_close_releases_only_after_safe_reset(self):
        self.env.reset()
        self.env.close()
        self.assertTrue(self.bridge.safe and self.bridge.closed)
        self.assertEqual(self.env.cleanup_outcome, "safe_room_and_disconnected")

    def test_worker_crash_is_not_masked_by_cleanup_reset(self):
        self.env.reset()
        original = ConnectionResetError("worker crashed during step")
        self.bridge.step = Mock(side_effect=original)
        self.bridge.reset_safe = Mock(side_effect=ConnectionAbortedError("dead worker"))
        with self.assertRaises(ConnectionResetError) as caught:
            try:
                self.env.step([0, 0, 0, 0])
            finally:
                self.env.close()
        self.assertIs(caught.exception, original)
        self.bridge.reset_safe.assert_not_called()
        self.assertTrue(self.bridge.closed)
        self.assertEqual(self.env.cleanup_outcome, "broken_connection_closed")

    def test_reset_transport_failure_closes_without_second_reset(self):
        self.bridge.reset_monstro = Mock(side_effect=ConnectionAbortedError("worker died on reset"))
        self.bridge.reset_safe = Mock()
        with self.assertRaises(ConnectionAbortedError): self.env.reset()
        self.env.close()
        self.bridge.reset_safe.assert_not_called()
        self.assertTrue(self.bridge.closed)

    def test_ppo_update_and_checkpoint_on_recorded_fixture(self):
        # This is an offline integration test, never a live-engine training result.
        torch.set_num_threads(1)
        self.bridge.mode = "win"
        model = PPO("MultiInputPolicy", self.env, n_steps=8, batch_size=8, n_epochs=1,
                    seed=0, device="cpu", policy_kwargs={"net_arch": [16]})
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        model.learn(total_timesteps=16)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.policy.state_dict().items()))
        out = Path(__file__).resolve().parents[2] / "runs/l1/20260919-training-loop/offline-unit-ppo"
        model.save(out)
        restored = PPO.load(out, device="cpu")
        obs, _ = self.env.reset()
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                      restored.predict(obs, deterministic=True)[0])


if __name__ == "__main__": unittest.main()
