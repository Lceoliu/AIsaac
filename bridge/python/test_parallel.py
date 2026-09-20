import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
import torch

from isaac_bridge.parallel import prepare_worker, EngineVecEnv
from isaac_bridge.transformer_obs import TransformerMonstroEnv
from isaac_bridge.transformer_policy import CombatTransformer
from test_transformer import bridge_v3
from train_parallel import EpisodeLog


def fake_worker():
    return Monitor(TransformerMonstroEnv(bridge=bridge_v3(), max_episode_frames=4, history=4, entity_capacity=16))


class ParallelTest(unittest.TestCase):
    def test_steps_are_dispatched_concurrently(self):
        barrier = threading.Barrier(2, timeout=5)
        def factory():
            env = fake_worker()
            original_step = env.step
            def step(action):
                barrier.wait()
                return original_step(action)
            env.step = step
            return env
        env = EngineVecEnv([factory, factory])
        try:
            env.reset()
            env.step(np.zeros((2, 3), dtype=np.int64))
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA build/device required')
    def test_cuda_shared_learner_has_finite_updated_parameters(self):
        env = EngineVecEnv([fake_worker, fake_worker])
        try:
            model = MaskablePPO('MultiInputPolicy', env, device='cuda', n_steps=4, batch_size=4, n_epochs=1,
                policy_kwargs={'features_extractor_class': CombatTransformer, 'net_arch': [],
                               'normalize_images': False}, seed=2)
            before = {k: v.detach().clone() for k,v in model.policy.named_parameters()}
            model.learn(8)
            self.assertTrue(all(torch.isfinite(v).all() for v in model.policy.parameters()))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.policy.named_parameters()))
        finally:
            env.close()

    def test_worker_writable_paths_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]/'runs') as temp:
            root = Path(temp)
            game, source = root/'game', root/'source'
            (game/'resources').mkdir(parents=True)
            (game/'mods/isaac_rl_bridge').mkdir(parents=True)
            (game/'mods/isaac_rl_bridge/main.lua').write_text('test')
            (game/'mods/other').mkdir()
            (game/'isaac-ng.exe').write_bytes(b'fixture')
            source.mkdir()
            original = 'SteamCloud=1\nPauseOnFocusLost=1\nEnableMods=0\nTryImportSave=1\nSaveCommandHistory=1\n'
            (source/'options.ini').write_text(original)
            (source/'inputconfigs.dat').write_bytes(b'fixture')
            a = prepare_worker(root/'a', game, source)
            b = prepare_worker(root/'b', game, source)
            self.assertNotEqual(a, b)
            (a[2]/'options.ini').write_text('changed')
            self.assertIn('SteamCloud=0', (b[2]/'options.ini').read_text())
            self.assertEqual((source/'options.ini').read_text(), original)
            self.assertTrue((a[0]/'isaac-ng.exe').is_file())
            self.assertTrue((a[0]/'mods/other/disable.it').exists())
            self.assertFalse((a[0]/'mods/isaac_rl_bridge').is_symlink())

    def test_episode_count_is_not_vector_step_count(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]/'runs') as temp:
            cb = EpisodeLog(Path(temp)/'episodes.jsonl')
            cb.num_timesteps = 512
            running = {'step_start': 1., 'step_end': 2., 'worker': 0}
            cb.locals = {'infos': [running, dict(running, worker=1)]}
            cb._on_step()
            self.assertEqual(len(cb.episodes), 0)
            done = dict(running, episode={'r': -1., 'l': 174}, outcome='death', elapsed_frames=348)
            cb.locals = {'infos': [done, dict(done, worker=1, outcome='time_limit')]}
            cb._on_step()
            self.assertEqual(len(cb.episodes), 2)
            self.assertEqual(cb.overlap_steps, 2)
            records = [json.loads(x) for x in cb.path.read_text().splitlines()]
            self.assertEqual([x['worker'] for x in records], [0, 1])

    def test_parallel_masks_history_reset_and_shared_learner(self):
        env = EngineVecEnv([fake_worker, fake_worker])
        try:
            model = MaskablePPO('MultiInputPolicy', env, n_steps=4, batch_size=4, n_epochs=1,
                policy_kwargs={'features_extractor_class': CombatTransformer, 'net_arch': [],
                               'normalize_images': False}, seed=2)
            model.learn(8)
            self.assertEqual(model.num_timesteps, 8)
            self.assertEqual(len(model.ep_info_buffer), 4)
            obs = env.reset()
            actions, _ = model.predict(obs, action_masks=get_action_masks(env))
            self.assertEqual(actions.shape, (2, 3))
            self.assertTrue(np.all(actions[:, 2] == 0))
            env.step(actions)
            obs, _, dones, infos = env.step(actions)
            self.assertTrue(dones.all())
            self.assertEqual(obs['history_mask'].sum(), 2)
            self.assertTrue(all(info['TimeLimit.truncated'] for info in infos))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
