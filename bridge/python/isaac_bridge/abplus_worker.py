"""AB+ rollout worker: one environment slot of AbplusFrameVecEnv (learner side: abplus_vec.py).

Runs in its own process and never imports torch. It owns two AB+ instances: the active one plays
the current episode while the other is reset for the next episode in a background thread, so an
episode boundary is a switch, not a ~0.1 s reset that would stall every environment of the batch.

Per step the learner sends 13 bytes (b'S' + joint, bomb, item as int32). The worker advances the
active instance two logic frames through the bridge, encodes the newest frame exactly as
VisibleHistory does, writes it as a FRAME_DTYPE record into shared memory (plus the next episode's
first frame when the episode ended) and answers with one byte. Reward: combat-v3 or combat-v2 (abplus_reward:
health curve, death, time, bombs, room clear points, timeout, potential progress; the component
totals of a finished episode go to META['components']) or combat-v1 (the bridge's legacy
components, a win adds 2 plus a [0, 1] speed bonus). In all of them the 120 s deadline ends the episode
as a termination, not a truncation. combat-v3 frames carry the room state input too
(FRAME_DTYPE_COMBAT, transformer_obs.COMBAT_FIELDS), and training can randomise the bombs the player
starts with (sample_bombs) and draw rooms by Prioritized Level Replay (PlrTaskChooser reads the
learner's room distribution from shared memory). Every finished episode reports EPISODE_STATS
(behaviour: first hit, longest time without a hit, longest stay in one spot, cells, bombs used).

Instance recycling: an instance that has played config['recycle_episodes'] episodes (default 200,
0 = never) is replaced in its background preparation thread, while the other instance plays, so
the batch does not wait. It was added for a leak of ~0.76 MiB per training episode that also slowed
resets; the cause was render-lite stubbing ImageManager::apply_frame_images, which recycles
transparent render batches (fixed in tools/stub_render_f.txt, analysis/docs/
ABP_LINUX_REVERSE_ENGINEERING.md 15.11). Recycling stays as a guard against slower leaks.

Every start of an AB+ process needs a running, logged-in Steam client (steam_watch.py). A recycle
therefore starts the replacement under the instance's second identity (name + 'x', port +
2 * num_envs) and stops the old process only once the new one serves the bridge. With no Steam
client running the recycle is deferred; a replacement that exits or does not come up within
STARTUP_S is counted as a start failure and the old process keeps playing. Either way the next
attempt comes RETRY_EPISODES episodes later. META counts both for the learner's alerts.
"""
from __future__ import annotations

import math
import os
import signal
import socket
import struct
import subprocess
import threading
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from .abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from .abplus_reward import COMPONENTS, COMPONENTS_V3, COMPONENTS_V4, COMPONENTS_V5, REWARDS
from .abplus_tasks import TASKS, Task, TaskSampler
from .transformer_obs import COMBAT_FIELDS, COMBAT_FIELDS_V4, ENTITY_FLAGS, FACTORED_NVEC
from .combat_reward import combat_v1_reward
from .steam_watch import steam_running

# Must equal sim_vec.FRAME_DTYPE (checked by abplus_vec at import; sim_vec imports torch).
FRAME_DTYPE = np.dtype([
    ('player', 'f4', (23,)), ('player_anim', 'i4', (32,)), ('active_kind', 'i4', (1,)),
    ('entities', 'f4', (256, 31)), ('entity_kind', 'i4', (256, 3)), ('entity_anim', 'i4', (256, 32)),
    ('entity_mask', 'f4', (256,)), ('terrain', 'f4', (7, 9, 15)), ('previous_action', 'f4', (4,)),
    ('time', 'f4'), ('history_mask', 'f4'), ('reward', 'f4'), ('done', 'i4'), ('truncated', 'i4'),
    ('outcome', 'i4'), ('elapsed', 'u4'), ('layout', 'u4'), ('count', 'u4')])
FRAME_KEYS = ('player', 'player_anim', 'active_kind', 'entities', 'entity_kind', 'entity_anim',
              'entity_mask', 'terrain', 'previous_action', 'time', 'history_mask')
# combat-v3: the simulator record plus the room state input (not part of the simulator ABI).
FRAME_DTYPE_COMBAT = np.dtype(FRAME_DTYPE.descr + [('combat', 'f4', (len(COMBAT_FIELDS),))])
FRAME_KEYS_COMBAT = FRAME_KEYS + ('combat',)
EPISODE_STATS = ('first_hit_s', 'longest_no_hit_s', 'longest_stationary_s', 'cells', 'bombs_used')
# Per-environment side channel: the episode that stepped, the episode that starts after a done,
# and timings for diagnostics.
META_DTYPE = np.dtype([
    ('seed', 'i8'), ('start', 'f4', (2,)), ('reset_seed', 'i8'), ('reset_start', 'f4', (2,)),
    ('step_ms', 'f4'), ('switch_wait_ms', 'f4'), ('reset_ms', 'f4'), ('errors', 'i4'), ('episodes', 'i4'),
    ('task', 'i4'), ('reset_task', 'i4'),
    ('components', 'f4', (max(len(COMPONENTS), len(COMPONENTS_V3), len(COMPONENTS_V4), len(COMPONENTS_V5)),)),
    ('recycles', 'i4'), ('start_failures', 'i4'), ('recycle_deferrals', 'i4'), ('level', 'i4'), ('reset_level', 'i4'),
    ('bombs', 'i4'), ('reset_bombs', 'i4'), ('stats', 'f4', (len(EPISODE_STATS),))])
REWARD_PROFILES = ('combat-v1', 'combat-v2', 'combat-v3', 'combat-v4', 'combat-v5')
# Room state input per profile, and the profiles whose 120 s deadline is an observed termination
# (combat-v4/v5 truncate at the deadline instead and observe no elapsed time).
COMBAT_LAYOUTS = {'combat-v3': COMBAT_FIELDS, 'combat-v4': COMBAT_FIELDS_V4, 'combat-v5': COMBAT_FIELDS_V4}
DEADLINE_PROFILES = ('combat-v1', 'combat-v2', 'combat-v3')
TRUNCATING_PROFILES = ('combat-v4', 'combat-v5')
# combat-v5 frames: firing geometry (entity flags, fire_distance, the auxiliary labels) and the
# factored previous action; 'hit' carries the step's hit reward for the learner's diagnostics.
GEOMETRY_PROFILES = ('combat-v5',)
GEOMETRY_FIELDS = [('entity_flags', 'f4', (256, len(ENTITY_FLAGS))), ('fire_distance', 'f4'), ('aim_label', 'f4'),
                   ('approach', 'f4', (9,)), ('hit', 'f4')]
GEOMETRY_KEYS = ('entity_flags', 'fire_distance', 'aim_label', 'approach')
OUTCOMES = ('running', 'death', 'win', 'time_limit', 'error')
FULL_START = (6.0, 1.0)  # player half-hearts, Boss HP fraction
MAX_EPISODE_FRAMES = 3600  # 120 s at 30 logic frames/s
RECYCLE_EPISODES = 200     # restart an AB+ process after this many episodes (memory leak; 0 = never)
RETRY_EPISODES = 50        # a deferred or failed recycle is tried again this many episodes later
STARTUP_S = 30.0           # a replacement process must serve the bridge within this time


def sample_start(seed, config):
    """gpu_env.sample_start, bit for bit (checked by abplus_vec)."""
    if not config:
        return FULL_START
    rng = np.random.default_rng([int(seed), 0x5EED])
    player = float(rng.integers(config['player_hp_min'], 6)) if rng.random() < config['player_hp_prob'] else 6.0
    boss = float(rng.uniform(config['boss_hp_min'], 1.0)) if rng.random() < config['boss_hp_prob'] else 1.0
    return player, boss


def frame_layout(profile):
    """Frame record and the observation keys it carries for a reward profile."""
    fields = COMBAT_LAYOUTS.get(profile)
    if not fields:
        return FRAME_DTYPE, FRAME_KEYS
    combat = [('combat', 'f4', (len(fields),))]
    if profile not in GEOMETRY_PROFILES:
        return np.dtype(FRAME_DTYPE.descr + combat), FRAME_KEYS_COMBAT
    # The simulator record with the one-hot factored previous action in place of (joint, bomb, item, valid).
    base = [('previous_action', 'f4', (sum(FACTORED_NVEC),)) if field[0] == 'previous_action' else field
            for field in FRAME_DTYPE.descr]
    return np.dtype(base + combat + GEOMETRY_FIELDS), FRAME_KEYS_COMBAT + GEOMETRY_KEYS


def observation_options(profile):
    """VisibleHistory / AbplusTransformerEnv options of a reward profile."""
    geometry = profile in GEOMETRY_PROFILES
    return dict(deadline=profile in DEADLINE_PROFILES, combat_state=COMBAT_LAYOUTS.get(profile, False),
                **(dict(geometry=True, factored_actions=True) if geometry else {}))


def sample_bombs(seed, config):
    """Bombs at the start of an episode: 0 with probability zero_prob, else 1..max; no config: 1."""
    if not config:
        return 1
    rng = np.random.default_rng([int(seed), 0xB0B5])
    return 0 if rng.random() < config['zero_prob'] else int(rng.integers(1, config['max'] + 1))


def level_index(levels, info):
    """Index of the room an episode plays in the PLR level list (-1 when there is none)."""
    if not levels:
        return -1
    task = info.get('task', 'arena')
    key = ('arena', 0) if task == 'arena' else (task, int(info.get('room_variant', -1)))
    return levels.get(key, -1)


class PlrTaskChooser:
    """TaskSampler's choose() over the learner's room distribution (plr.PrioritizedLevels)."""

    def __init__(self, levels, probabilities):
        self.levels, self.probabilities = [tuple(level) for level in levels], probabilities

    def choose(self, seed, retry=0):
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x9E7, int(retry)])
        p = np.clip(np.array(self.probabilities, np.float64), 0.0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
        i = min(int(np.searchsorted(np.cumsum(p), rng.random(), side='right')), len(p) - 1)
        kind, variant = self.levels[i]
        return Task(kind, int(variant), int(rng.integers(4)))


class EpisodeStats:
    """EPISODE_STATS of one episode from the raw observations (a hit: doors-blocking HP fell)."""

    def __init__(self, obs):
        p = obs['players'][0]
        self.blocking = float(obs['combat']['blocking_hp'])
        self.t = self.last_hit = self.anchor_t = 0.0
        self.first_hit = -1.0
        self.no_hit = self.stationary = 0.0
        self.anchor = tuple(p['pos'])
        self.cells = {self.cell(p['pos'])}
        self.bombs0 = self.bombs = int(p['bombs'])

    @staticmethod
    def cell(pos):
        return int(pos[0] // 40), int(pos[1] // 40)

    def step(self, obs, frames):
        self.t += frames / 30
        p = obs['players'][0]
        hp = float(obs['combat']['blocking_hp'])
        if hp < self.blocking - 1e-9:
            if self.first_hit < 0:
                self.first_hit = self.t
            self.no_hit = max(self.no_hit, self.t - self.last_hit)
            self.last_hit = self.t
        self.blocking = hp
        pos = tuple(p['pos'])
        if math.dist(pos, self.anchor) > 48:
            self.stationary = max(self.stationary, self.t - self.anchor_t)
            self.anchor, self.anchor_t = pos, self.t
        self.cells.add(self.cell(pos))
        self.bombs = int(p['bombs'])

    def array(self):
        return np.array([self.first_hit, max(self.no_hit, self.t - self.last_hit),
                         max(self.stationary, self.t - self.anchor_t), len(self.cells), self.bombs0 - self.bombs],
                        np.float32)


def episode_seed(config, index, episode):
    """FrameChunk's schedule: env i plays base_seed + i + num_envs * k in its k-th episode."""
    return config['base_seed'] + index + config['num_envs'] * episode


class FrameEnv(AbplusTransformerEnv):
    """AB+ environment that returns only the newest frame; the learner keeps the window."""

    def encode_observation(self, obs):
        return self.history.encode(obs)


def wait_listening(proc, port, timeout):
    """True once the game serves the bridge on port (bound after its first logic frame); False when
    the process exits first (e.g. the DRM stub handing over to steam.sh) or the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


class Instance:
    """One AB+ process with its bridge; prepare() resets it for an episode in a background thread.

    The process runs under one of two identities (instance name and port) so that a recycle can
    start the replacement before the old process stops."""

    def __init__(self, name, port, config, alt_port=None):
        self.names = (name, name + 'x')
        self.ports = (port, port if alt_port is None else alt_port)
        self.slot = 0
        self.config = config
        self.proc = self.env = None
        self.thread = None
        self.ready = threading.Event()
        self.result = self.error = None
        self.episodes = 0   # episodes prepared since this process started
        self.recycles = self.start_failures = self.recycle_deferrals = 0
        self.launch()

    @property
    def name(self):
        return self.names[self.slot]

    @property
    def port(self):
        return self.ports[self.slot]

    def _env(self, port):
        env = FrameEnv(port=port, max_episode_frames=MAX_EPISODE_FRAMES,
                       **observation_options(self.config.get('reward_profile')))
        env.bridge.binary_obs = self.config.get('binary_obs', True)
        env.bridge.lineage_mode = int(self.config.get('lineage_mode', env.bridge.lineage_mode))
        spec = self.config.get('tasks')
        if self.config.get('plr_probabilities') is not None:
            env.bridge.tasks = PlrTaskChooser(self.config['plr_levels'], self.config['plr_probabilities'])
        else:
            env.bridge.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss']) if spec else None
        return env

    def launch(self):
        self.episodes = 0
        self.proc = launch_abplus(self.name, self.port, self.config['mode'], nice=self.config['nice'])
        self.env = self._env(self.port)

    def _replace(self):
        """Start the other identity; stop the current process only when the new one is up."""
        slot = 1 - self.slot
        name, port = self.names[slot], self.ports[slot]
        proc = launch_abplus(name, port, self.config['mode'], nice=self.config['nice'])
        if not wait_listening(proc, port, STARTUP_S):
            self._end(proc, name, None)
            return False
        old = (self.proc, self.name, self.env)
        self.slot, self.proc, self.env = slot, proc, self._env(port)
        self.episodes = 0
        self._end(*old)
        return True

    def _recycle(self):
        retry_at = max(0, int(self.config.get('recycle_episodes', RECYCLE_EPISODES)) - RETRY_EPISODES)
        if self.ports[0] == self.ports[1]:        # no second identity: stop, then start
            self._stop_process()
            time.sleep(1.0)
            self.launch()
            self.recycles += 1
        elif not steam_running():
            self.recycle_deferrals += 1
            self.episodes = retry_at
        elif self._replace():
            self.recycles += 1
        else:
            self.start_failures += 1
            self.episodes = retry_at

    def relaunch(self):
        self.close()
        time.sleep(1.0)
        self.launch()

    def prepare(self, seed, start, bombs=1, recycle=False):
        """Reset for an episode in a background thread; recycle=True first replaces the process."""
        def run():
            try:
                if recycle:
                    self._recycle()
                self.episodes += 1
                t = time.perf_counter()
                frame, info = self.env.reset(options={'arena_seed': int(seed), 'start': start, 'bombs': int(bombs)})
                self.result = (int(seed), tuple(start), int(bombs), frame, info, self.env.history.last_rows,
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

    @staticmethod
    def _end(proc, name, env):
        """Close a bridge and end its game; waits for the exit so the port is free again."""
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        finally:
            if proc is not None:
                stop_abplus(proc, name)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait(timeout=20)

    def _stop_process(self):
        self._end(self.proc, self.name, self.env)
        self.proc = self.env = None

    def close(self):
        if self.thread is not None and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=30)
        self._stop_process()


def write_frame(row, frame, reward, done, truncated, outcome, elapsed, layout, count, hit=0.0):
    names = row.dtype.names
    for key in FRAME_KEYS + ('combat',) + GEOMETRY_KEYS:
        if key in names:
            row[key] = frame[key]
    if 'hit' in names:
        row['hit'] = hit
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
        alt = 2 * config['num_envs']   # second identities use the port block after the first
        self.instances = [Instance(name + 'a', port, config, port + alt),
                          Instance(name + 'b', port + 1, config, port + 1 + alt)]
        self.active = 0
        self.episode = 0
        self.seed = None
        self.start = FULL_START
        self.bombs = 1
        self.layout = 0
        self.task = 0
        self.level = -1
        self.levels = {tuple(level): i for i, level in enumerate(config.get('plr_levels') or ())}
        self.elapsed = 0
        self.last_frame = None
        self.stats = None
        self.recycle_after = int(config.get('recycle_episodes', RECYCLE_EPISODES))
        self.profile = config.get('reward_profile', 'combat-v1')
        if self.profile not in REWARD_PROFILES:
            raise ValueError(f'unknown reward profile {self.profile!r}')
        self.reward = (REWARDS[self.profile](**config.get('reward_options', {})) if self.profile in REWARDS
                       else None)

    def schedule(self, episode):
        seed = episode_seed(self.config, self.index, episode)
        return (seed, sample_start(seed, self.config['start_randomization']),
                sample_bombs(seed, self.config.get('start_bombs')))

    def _begin(self, instance_index, result):
        seed, start, bombs, frame, info, rows, reset_ms = result
        self.active = instance_index
        self.seed, self.start, self.bombs = seed, start, bombs
        self.layout = int(info.get('room_variant', info.get('arena', {}).get('variant', 0)))
        self.task = TASKS.index(info.get('task', 'arena'))
        self.level = level_index(self.levels, info)
        self.elapsed = 0
        self.last_frame = frame
        raw = self.instances[instance_index].env.raw_obs
        self.stats = EpisodeStats(raw)
        if self.reward is not None:
            self.reward.reset(raw, TASKS[self.task])
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
        first.prepare(seed, start, sample_bombs(seed, self.config.get('start_bombs')))
        frame, rows, reset_ms = self._begin(0, first.take())
        write_frame(self.step_row, frame, 0.0, False, False, 0, 0, self.layout, rows)
        self.instances[1].prepare(*self.schedule(1))
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        self.meta['level'] = self.level
        self.meta['bombs'] = self.bombs
        self.meta['reset_ms'] = reset_ms
        self.meta['episodes'] = 0

    def step(self, joint, bomb, item):
        active = self.instances[self.active]
        t = time.perf_counter()
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        self.meta['level'] = self.level
        self.meta['bombs'] = self.bombs
        try:
            # The learner masks the bomb from the same frame; a mismatch only drops the bomb.
            if bomb and not active.env.action_masks()[46]:
                bomb = 0
            frame, reward, terminated, truncated, info = active.env.step(np.array([joint, bomb, item]))
            outcome = OUTCOMES.index(info['outcome'])
            elapsed = int(info['elapsed_frames'])
            self.stats.step(active.env.raw_obs, elapsed - self.elapsed)
            hit = 0.0
            if self.reward is not None:
                parts = self.reward.step(active.env.raw_obs, info['outcome'], elapsed - self.elapsed)
                reward = self.reward.scale * float(sum(parts.values()))
                hit = float(parts.get('hit', 0.0))
            else:
                reward = float(combat_v1_reward(reward, outcome, elapsed))
            done = bool(terminated or truncated)
            # combat-v1..v3: the 120 s deadline is part of the task, so it terminates without bootstrap;
            # combat-v4 truncates there (the learner bootstraps the value of the last frame).
            cut = self.profile in TRUNCATING_PROFILES and info['outcome'] == 'time_limit'
            write_frame(self.step_row, frame, reward, done, cut, outcome, elapsed, self.layout,
                        active.env.history.last_rows, hit)
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
                totals = self.reward.totals_array()
                self.meta['components'] = np.pad(totals, (0, len(self.meta['components']) - len(totals)))
            self.meta['stats'] = self.stats.array()
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
            self.meta['reset_level'] = self.level
            self.meta['reset_bombs'] = self.bombs
            self.meta['reset_ms'] = reset_ms
            self.meta['episodes'] = self.episode
            self.meta['recycles'] = sum(instance.recycles for instance in self.instances)
            self.meta['start_failures'] = sum(instance.start_failures for instance in self.instances)
            self.meta['recycle_deferrals'] = sum(instance.recycle_deferrals for instance in self.instances)
            # The instance that just finished prepares the episode after the one now starting,
            # restarting its process first when it has played recycle_after episodes.
            finished = self.instances[old]
            recycle = bool(self.recycle_after) and finished.episodes >= self.recycle_after
            seed, start, bombs = self.schedule(self.episode + 1)
            finished.prepare(seed, start, bombs, recycle=recycle)

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
    frame_dtype, _ = frame_layout(config.get('reward_profile'))
    frames = np.ndarray((2, n), dtype=frame_dtype, buffer=shm.buf)
    meta = np.ndarray((n,), dtype=META_DTYPE, buffer=shm.buf, offset=2 * n * frame_dtype.itemsize)
    plr_shm = None
    if config.get('plr_shm'):
        # The learner rewrites the room distribution after every rollout (float64 per level).
        plr_shm = shared_memory.SharedMemory(name=config['plr_shm'])
        config = {**config, 'plr_probabilities': np.ndarray((len(config['plr_levels']),), np.float64,
                                                             buffer=plr_shm.buf)}
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
        config.pop('plr_probabilities', None)
        shm.close()
        if plr_shm is not None:
            plr_shm.close()
