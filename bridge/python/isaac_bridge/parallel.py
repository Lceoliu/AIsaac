"""Isolated native engines with concurrent TCP stepping, one Python runtime."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import time
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from .transformer_obs import TransformerMonstroEnv


def prepare_worker(root, game_dir, profile_source):
    root, game_dir, profile_source = map(Path, (root, game_dir, profile_source))
    runtime = root/'runtime'
    runtime.mkdir(parents=True)
    # Assets are shared read-only by convention; writable data/mod state is not linked.
    import _winapi
    _winapi.CreateJunction(str(game_dir/'resources'), str(runtime/'resources'))
    for source in game_dir.glob('*.a'):
        os.link(source, runtime/source.name)
    for source in [game_dir/'isaac-ng.exe', *game_dir.glob('*.dll')]:
        os.link(source, runtime/source.name)
    for source in game_dir.glob('*.ini'):
        shutil.copyfile(source, runtime/source.name)
    if (game_dir/'steam_appid.txt').exists():
        shutil.copyfile(game_dir/'steam_appid.txt', runtime/'steam_appid.txt')
    (runtime/'data').mkdir()
    (runtime/'mods').mkdir()
    shutil.copytree(game_dir/'mods/isaac_rl_bridge', runtime/'mods/isaac_rl_bridge')
    # Steam repopulates subscribed mods even in a fresh runtime. Preserve the
    # disabled markers before its startup sync, rather than enabling them anew.
    for mod in (game_dir/'mods').iterdir():
        if mod.is_dir() and mod.name != 'isaac_rl_bridge':
            target = runtime/'mods'/mod.name
            target.mkdir()
            (target/'disable.it').touch()
            if (mod/'metadata.xml').exists():
                shutil.copyfile(mod/'metadata.xml', target/'metadata.xml')
    profile = root/'profile'
    savedata = profile/'Documents/My Games/Binding of Isaac Repentance+'
    savedata.mkdir(parents=True)
    options = (profile_source/'options.ini').read_text(encoding='utf-8')
    for key, value in {'SteamCloud': 0, 'PauseOnFocusLost': 0, 'EnableMods': 1,
                       'TryImportSave': 0, 'SaveCommandHistory': 0}.items():
        options = re.sub(rf'(?m)^{key}=.*$', f'{key}={value}', options)
    (savedata/'options.ini').write_text(options, encoding='utf-8')
    shutil.copyfile(profile_source/'inputconfigs.dat', savedata/'inputconfigs.dat')
    return runtime, profile, savedata


class WorkerMonitor(Monitor):
    def __init__(self, port, rank, out, episode_frames, history, capacity):
        env = TransformerMonstroEnv(port=port, max_episode_frames=episode_frames,
                                   history=history, entity_capacity=capacity)
        env.bridge.connect_timeout = 60
        super().__init__(env, filename=str(Path(out)/'monitor.csv'))
        self.rank = rank

    def step(self, action):
        start = time.perf_counter()
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(worker=self.rank, step_start=start, step_end=time.perf_counter())
        return obs, reward, terminated, truncated, info

    def snapshot(self):
        raw = self.env.raw_obs
        return {'worker': self.rank, 'port': self.env.bridge.port,
                'logic_frames': raw['logic_frames'], 'player_pos': raw['players'][0]['pos'],
                'history_length': len(self.env.history.frames)}

    def action_masks(self):
        return self.env.action_masks()


def make_worker(port, rank, out, episode_frames, history, capacity):
    return WorkerMonitor(port, rank, out, episode_frames, history, capacity)


class EngineVecEnv(DummyVecEnv):
    """SB3's buffer/reset contract, but native engine I/O runs concurrently.

    The expensive simulators already live in separate OS processes. A second
    Python/PyTorch process per connection only duplicates memory on Windows.
    Each task exclusively owns one environment/socket until step_wait completes.
    """
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.pool = ThreadPoolExecutor(max_workers=self.num_envs)

    def step_async(self, actions):
        self.pending = [self.pool.submit(self._step_one, i, action) for i, action in enumerate(actions)]

    def _step_one(self, i, action):
        obs, reward, terminated, truncated, info = self.envs[i].step(action)
        done = terminated or truncated
        info['TimeLimit.truncated'] = truncated and not terminated
        if done:
            info['terminal_observation'] = obs
            obs, self.reset_infos[i] = self.envs[i].reset()
        return obs, reward, done, info

    def step_wait(self):
        for i, pending in enumerate(self.pending):
            obs, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = pending.result()
            self._save_obs(i, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self):
        pending = [self.pool.submit(env.reset, seed=self._seeds[i], options=self._options[i])
                   for i, env in enumerate(self.envs)]
        for i, future in enumerate(pending):
            obs, self.reset_infos[i] = future.result()
            self._save_obs(i, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self):
        self.pool.shutdown(wait=True)
        super().close()
