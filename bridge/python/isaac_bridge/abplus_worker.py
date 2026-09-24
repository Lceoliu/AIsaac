"""AB+ rollout worker: one environment slot of AbplusFrameVecEnv (learner side: abplus_vec.py).

Runs in its own process and never imports torch. It owns two AB+ instances: the active one plays
the current episode while the other is reset for the next episode in a background thread, so an
episode boundary is a switch, not a ~0.1 s reset that would stall every environment of the batch.

Per step the learner sends 13 bytes (b'S' + joint, bomb, item as int32). The worker advances the
active instance two logic frames through the bridge, encodes the newest frame exactly as
VisibleHistory does, writes it as a FRAME_DTYPE record into shared memory (plus the next episode's
first frame when the episode ended) and answers with one byte. Reward: combat-v2 (abplus_reward:
health curve, death, time, bombs, room clear points, timeout, potential progress; the component
totals of a finished episode go to META['components']) or combat-v1 (the bridge's legacy
components, a win adds 2 plus a [0, 1] speed bonus). In both the 120 s deadline ends the episode
as a termination, not a truncation.

Instance recycling: an instance that has played config['recycle_episodes'] episodes (default 200,
0 = never) is restarted in its background preparation thread, while the other instance plays, so
the batch does not wait for the restart. It was added for a leak of ~0.76 MiB per training episode
that also slowed resets; the cause was render-lite stubbing ImageManager::apply_frame_images, which
recycles transparent render batches (fixed in tools/stub_render_f.txt, analysis/docs/
ABP_LINUX_REVERSE_ENGINEERING.md 15.11). Recycling stays as a guard against slower leaks.
"""
from __future__ import annotations

import os
import signal
import struct
import subprocess
import threading
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from .abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from .abplus_reward import COMPONENTS, CombatV2
from .abplus_tasks import TASKS, TaskSampler
from .combat_reward import combat_v1_reward

# Must equal sim_vec.FRAME_DTYPE (checked by abplus_vec at import; sim_vec imports torch).
FRAME_DTYPE = np.dtype([
    ('player', 'f4', (23,)), ('player_anim', 'i4', (32,)), ('active_kind', 'i4', (1,)),
    ('entities', 'f4', (256, 31)), ('entity_kind', 'i4', (256, 3)), ('entity_anim', 'i4', (256, 32)),
    ('entity_mask', 'f4', (256,)), ('terrain', 'f4', (7, 9, 15)), ('previous_action', 'f4', (4,)),
    ('time', 'f4'), ('history_mask', 'f4'), ('reward', 'f4'), ('done', 'i4'), ('truncated', 'i4'),
    ('outcome', 'i4'), ('elapsed', 'u4'), ('layout', 'u4'), ('count', 'u4')])
FRAME_KEYS = ('player', 'player_anim', 'active_kind', 'entities', 'entity_kind', 'entity_anim',
              'entity_mask', 'terrain', 'previous_action', 'time', 'history_mask')
# Per-environment side channel: the episode that stepped, the episode that starts after a done,
# and timings for diagnostics.
META_DTYPE = np.dtype([
    ('seed', 'i8'), ('start', 'f4', (2,)), ('reset_seed', 'i8'), ('reset_start', 'f4', (2,)),
    ('step_ms', 'f4'), ('switch_wait_ms', 'f4'), ('reset_ms', 'f4'), ('errors', 'i4'), ('episodes', 'i4'),
    ('task', 'i4'), ('reset_task', 'i4'), ('components', 'f4', (len(COMPONENTS),)), ('recycles', 'i4')])
REWARD_PROFILES = ('combat-v1', 'combat-v2')
OUTCOMES = ('running', 'death', 'win', 'time_limit', 'error')
FULL_START = (6.0, 1.0)  # player half-hearts, Boss HP fraction
MAX_EPISODE_FRAMES = 3600  # 120 s at 30 logic frames/s
RECYCLE_EPISODES = 200     # restart an AB+ process after this many episodes (memory leak; 0 = never)


def sample_start(seed, config):
    """gpu_env.sample_start, bit for bit (checked by abplus_vec)."""
    if not config:
        return FULL_START
    rng = np.random.default_rng([int(seed), 0x5EED])
    player = float(rng.integers(config['player_hp_min'], 6)) if rng.random() < config['player_hp_prob'] else 6.0
    boss = float(rng.uniform(config['boss_hp_min'], 1.0)) if rng.random() < config['boss_hp_prob'] else 1.0
    return player, boss


def episode_seed(config, index, episode):
    """FrameChunk's schedule: env i plays base_seed + i + num_envs * k in its k-th episode."""
    return config['base_seed'] + index + config['num_envs'] * episode


class FrameEnv(AbplusTransformerEnv):
    """AB+ environment that returns only the newest frame; the learner keeps the window."""

    def encode_observation(self, obs):
        return self.history.encode(obs)


class Instance:
    """One AB+ process with its bridge; prepare() resets it for an episode in a background thread."""

    def __init__(self, name, port, config):
        self.name, self.port, self.config = name, port, config
        self.proc = self.env = None
        self.thread = None
        self.ready = threading.Event()
        self.result = self.error = None
        self.episodes = 0   # episodes prepared since this process started
        self.recycles = 0
        self.launch()

    def launch(self):
        self.episodes = 0
        self.proc = launch_abplus(self.name, self.port, self.config['mode'], nice=self.config['nice'])
        self.env = FrameEnv(port=self.port, max_episode_frames=MAX_EPISODE_FRAMES, deadline=True)
        self.env.bridge.binary_obs = self.config.get('binary_obs', True)
        spec = self.config.get('tasks')
        self.env.bridge.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss']) if spec else None

    def relaunch(self):
        self.close()
        time.sleep(1.0)
        self.launch()

    def prepare(self, seed, start, recycle=False):
        """Reset for an episode in a background thread; recycle=True first restarts the process."""
        def run():
            try:
                if recycle:
                    self._stop_process()
                    time.sleep(1.0)
                    self.launch()
                    self.recycles += 1
                self.episodes += 1
                t = time.perf_counter()
                frame, info = self.env.reset(options={'arena_seed': int(seed), 'start': start})
                self.result = (int(seed), tuple(start), frame, info, self.env.history.last_rows,
                               1000 * (time.perf_counter() - t))
            except BaseException:
                self.error = traceback.format_exc()
            finally:
                self.ready.set()
        self.ready.clear()
        self.result = self.error = None
        self.thread = threading.Thread(target=run, name=f'prepare-{self.name}', daemon=True)
        self.thread.start()

    def take(self):
        self.ready.wait()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f'{self.name} reset failed:\n{self.error}')
        return self.result

    def _stop_process(self):
        """Close the bridge and end the game; waits for the exit so the port is free again."""
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass
        finally:
            if self.proc is not None:
                stop_abplus(self.proc, self.name)
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    self.proc.wait(timeout=20)
            self.proc = self.env = None

    def close(self):
        if self.thread is not None and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=30)
        self._stop_process()


def write_frame(row, frame, reward, done, truncated, outcome, elapsed, layout, count):
    for key in FRAME_KEYS:
        row[key] = frame[key]
    row['reward'] = reward
    row['done'] = int(done)
    row['truncated'] = int(truncated)
    row['outcome'] = outcome
    row['elapsed'] = elapsed
    row['layout'] = layout
    row['count'] = count


class Worker:
    def __init__(self, index, config, step_row, reset_row, meta):
        self.index, self.config = index, config
        self.step_row, self.reset_row, self.meta = step_row, reset_row, meta
        name = f"{config['name']}{index}"
        port = config['port'] + 2 * index
        self.instances = [Instance(name + 'a', port, config), Instance(name + 'b', port + 1, config)]
        self.active = 0
        self.episode = 0
        self.seed = None
        self.start = FULL_START
        self.layout = 0
        self.task = 0
        self.elapsed = 0
        self.last_frame = None
        self.recycle_after = int(config.get('recycle_episodes', RECYCLE_EPISODES))
        self.profile = config.get('reward_profile', 'combat-v1')
        if self.profile not in REWARD_PROFILES:
            raise ValueError(f'unknown reward profile {self.profile!r}')
        self.reward = CombatV2() if self.profile == 'combat-v2' else None

    def schedule(self, episode):
        seed = episode_seed(self.config, self.index, episode)
        return seed, sample_start(seed, self.config['start_randomization'])

    def _begin(self, instance_index, result):
        seed, start, frame, info, rows, reset_ms = result
        self.active = instance_index
        self.seed, self.start = seed, start
        self.layout = int(info.get('room_variant', info.get('arena', {}).get('variant', 0)))
        self.task = TASKS.index(info.get('task', 'arena'))
        self.elapsed = 0
        self.last_frame = frame
        if self.reward is not None:
            self.reward.reset(self.instances[instance_index].env.raw_obs, TASKS[self.task])
        return frame, rows, reset_ms

    def reset(self, seed, base_seed, start):
        """First episode of a (new) collection generation: explicit seed, schedule afterwards."""
        self.config = {**self.config, 'base_seed': int(base_seed)}
        self.episode = 0
        first = self.instances[0]
        for instance in self.instances:
            if instance.thread is not None:
                try:
                    instance.take()
                except RuntimeError:
                    instance.relaunch()
        start = sample_start(seed, self.config['start_randomization']) if start is None else start
        first.prepare(seed, start)
        frame, rows, reset_ms = self._begin(0, first.take())
        write_frame(self.step_row, frame, 0.0, False, False, 0, 0, self.layout, rows)
        self.instances[1].prepare(*self.schedule(1))
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        self.meta['reset_ms'] = reset_ms
        self.meta['episodes'] = 0

    def step(self, joint, bomb, item):
        active = self.instances[self.active]
        t = time.perf_counter()
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        try:
            # The learner masks the bomb from the same frame; a mismatch only drops the bomb.
            if bomb and not active.env.action_masks()[46]:
                bomb = 0
            frame, reward, terminated, truncated, info = active.env.step(np.array([joint, bomb, item]))
            outcome = OUTCOMES.index(info['outcome'])
            elapsed = int(info['elapsed_frames'])
            if self.reward is not None:
                reward = float(sum(self.reward.step(active.env.raw_obs, info['outcome'], elapsed - self.elapsed).values()))
            else:
                reward = float(combat_v1_reward(reward, outcome, elapsed))
            done = bool(terminated or truncated)
            # combat-v1: the 120 s deadline is part of the task, so it terminates without bootstrap.
            write_frame(self.step_row, frame, reward, done, False, outcome, elapsed, self.layout,
                        active.env.history.last_rows)
            self.last_frame = frame
            self.elapsed = elapsed
        except Exception:
            # Engine or bridge failure: end the episode as a truncation (value bootstrap) on the
            # last good frame, replace the process, continue with the standby instance.
            self.meta['errors'] += 1
            print(f'[worker {self.index}] {active.name} failed:\n{traceback.format_exc()}', flush=True)
            write_frame(self.step_row, self.last_frame, 0.0, True, True, OUTCOMES.index('error'),
                        self.elapsed, self.layout, int(self.last_frame['entity_mask'].sum()))
            done = True
            active.relaunch()
        self.meta['step_ms'] = 1000 * (time.perf_counter() - t)
        if done:
            if self.reward is not None:
                self.meta['components'] = self.reward.totals_array()
            self.episode += 1
            standby_index = 1 - self.active
            standby = self.instances[standby_index]
            t = time.perf_counter()
            for attempt in range(3):
                try:
                    result = standby.take()
                    break
                except RuntimeError:
                    print(f'[worker {self.index}] standby reset failed (attempt {attempt}):\n'
                          f'{traceback.format_exc()}', flush=True)
                    self.meta['errors'] += 1
                    standby.relaunch()
                    standby.prepare(*self.schedule(self.episode))
            else:
                raise RuntimeError(f'worker {self.index}: standby instance keeps failing')
            self.meta['switch_wait_ms'] = 1000 * (time.perf_counter() - t)
            old = self.active
            frame, rows, reset_ms = self._begin(standby_index, result)
            write_frame(self.reset_row, frame, 0.0, False, False, 0, 0, self.layout, rows)
            self.meta['reset_seed'] = self.seed
            self.meta['reset_start'] = self.start
            self.meta['reset_task'] = self.task
            self.meta['reset_ms'] = reset_ms
            self.meta['episodes'] = self.episode
            self.meta['recycles'] = sum(instance.recycles for instance in self.instances)
            # The instance that just finished prepares the episode after the one now starting,
            # restarting its process first when it has played recycle_after episodes.
            finished = self.instances[old]
            recycle = bool(self.recycle_after) and finished.episodes >= self.recycle_after
            finished.prepare(*self.schedule(self.episode + 1), recycle=recycle)

    def close(self):
        for instance in self.instances:
            instance.close()


def worker_main(index, config, shm_name, conn):
    """Process entry. Protocol (bytes): R + seed, base_seed:int64 + start:2f32 (NaN = sample) -> k;
    S + joint, bomb, item:int32 -> k; Q -> exit. Any failure answers E + traceback."""
    current = os.nice(0)
    if config.get('nice', 0) > current:
        os.nice(config['nice'] - current)
    shm = shared_memory.SharedMemory(name=shm_name)
    n = config['num_envs']
    frames = np.ndarray((2, n), dtype=FRAME_DTYPE, buffer=shm.buf)
    meta = np.ndarray((n,), dtype=META_DTYPE, buffer=shm.buf, offset=2 * n * FRAME_DTYPE.itemsize)
    worker = None
    try:
        worker = Worker(index, config, frames[0, index:index + 1][0], frames[1, index:index + 1][0],
                        meta[index:index + 1][0])
        conn.send_bytes(b'r')
        while True:
            msg = conn.recv_bytes()
            kind = msg[:1]
            try:
                if kind == b'S':
                    worker.step(*struct.unpack('<3i', msg[1:13]))
                elif kind == b'R':
                    seed, base_seed, p, b = struct.unpack('<qqff', msg[1:25])
                    worker.reset(seed, base_seed, None if np.isnan(p) else (p, b))
                elif kind == b'Q':
                    break
                else:
                    raise ValueError(f'unknown command {msg[:1]!r}')
                conn.send_bytes(b'k')
            except Exception:
                conn.send_bytes(b'E' + traceback.format_exc().encode('utf8', 'replace'))
    except Exception:
        try:
            conn.send_bytes(b'E' + traceback.format_exc().encode('utf8', 'replace'))
        except OSError:
            pass
    finally:
        if worker is not None:
            worker.close()
        del frames, meta
        shm.close()
