"""AB+ source for GpuMaskablePPO: rollout worker processes around one exclusive CUDA learner.

Drop-in for gpu_env.GpuFrameVecEnv with the same chunk contract (pinned double-buffered
TransferSlots, advance_and_upload, reset frames packed in the prefix of reset_frames), so
FrameSampler, GpuHistoryRolloutBuffer and segment PPO run unchanged; the learner is the only
CUDA user. Each worker process (abplus_worker.py) owns one environment slot and two AB+ instances.

Per step the learner makes one batched policy call per chunk (chunks=1: all environments at
once), sends every worker 13 bytes over a pipe and copies back only the newest FRAME_DTYPE record
(~73 KB) per environment from shared memory into the chunk's pinned slot.
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import wait
import struct
import time

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from . import abplus_worker as W
from .gpu_env import FULL_START, GpuFrameVecEnv, TransferSlot, sample_start, validate_start_randomization
from .sim_vec import FRAME_DTYPE
from .transformer_obs import ENTITY_CAPACITY, HISTORY, VisibleHistory

if W.FRAME_DTYPE != FRAME_DTYPE:
    raise ImportError('abplus_worker.FRAME_DTYPE differs from sim_vec.FRAME_DTYPE')
_probe = dict(boss_hp_prob=0.5, boss_hp_min=0.1, player_hp_prob=0.25, player_hp_min=3)
if any(W.sample_start(s, _probe) != sample_start(s, _probe) for s in range(64)):
    raise ImportError('abplus_worker.sample_start differs from gpu_env.sample_start')


class AbplusChunk:
    """A group of workers stepped together; one policy call and one upload per step."""

    def __init__(self, owner, start, stop):
        self.owner, self.start, self.stop, self.n = owner, start, stop, stop - start
        self.conns = owner.conns[start:stop]
        self.copy_stream = torch.cuda.Stream(device=owner.device)
        self.compute_stream = torch.cuda.Stream(device=owner.device)
        self.ready = torch.cuda.Event()
        self.slots = [TransferSlot(self.n, owner.device) for _ in range(2)]
        self.episodes = np.zeros(self.n, np.int64)
        self.returns = np.zeros(self.n)
        self.lengths = np.zeros(self.n, np.int64)
        self.seeds = [owner.base_seed + i for i in range(start, stop)]
        self.actions = [[] for _ in range(self.n)]
        self.starts = [FULL_START] * self.n
        self.batch = self  # TrainingSession.save() records batch.states()
        self.exchange_s = 0.0
        self.exchanges = 0
        self.cycle_s = 0.0
        self.cycles = 0
        self.last_advance = None

    def states(self):
        """In-progress episodes; AB+ replays a room exactly from its seed and actions."""
        return [dict(seed=int(s), start=list(st), steps=len(a))
                for s, st, a in zip(self.seeds, self.starts, self.actions)]

    def _exchange(self, messages):
        for conn, msg in zip(self.conns, messages):
            conn.send_bytes(msg)
        pending = {conn: i for i, conn in enumerate(self.conns)}
        errors = []
        while pending:
            for conn in wait(list(pending)):
                i = pending.pop(conn)
                try:
                    reply = conn.recv_bytes()
                except EOFError:
                    reply = b'Eworker process exited'
                if reply[:1] != b'k':
                    errors.append(f'[env {self.start + i}] {reply[1:].decode("utf8", "replace")}')
        if errors:
            raise RuntimeError('AB+ worker failure:\n' + '\n'.join(errors))

    def reset(self, seeds, starts=None):
        """starts=None: workers sample the training start from the seed (gpu_env.sample_start)."""
        self.seeds = list(map(int, seeds))
        self.actions = [[] for _ in range(self.n)]
        messages = []
        for i, seed in enumerate(self.seeds):
            player, boss = (float('nan'), float('nan')) if starts is None else map(float, starts[i])
            messages.append(b'R' + struct.pack('<qqff', seed, self.owner.base_seed, player, boss))
        self._exchange(messages)
        rows = slice(self.start, self.stop)
        self.slots[0].frames[:] = self.owner.staging[0, rows]
        self.starts = [tuple(map(float, s)) for s in self.owner.meta['start'][rows]]
        self.episodes.fill(0)
        self.returns.fill(0)
        self.lengths.fill(0)

    def advance(self, slot):
        # Never read a pinned action before its D2H copy ends, nor overwrite a frame that DMA reads.
        slot.action_ready.synchronize()
        if slot.used:
            slot.upload_done.synchronize()
        actions = slot.actions.numpy().tolist()
        t = time.perf_counter()
        if self.last_advance is not None and t - self.last_advance < 5.0:  # not across a PPO update
            self.cycle_s += t - self.last_advance
            self.cycles += 1
        self.last_advance = t
        self._exchange([b'S' + struct.pack('<3i', *a) for a in actions])
        self.exchange_s += time.perf_counter() - t
        self.exchanges += 1
        rows = slice(self.start, self.stop)
        frames = slot.frames
        frames[:] = self.owner.staging[0, rows]
        meta = self.owner.meta[rows].copy()
        for trace, action in zip(self.actions, actions):
            trace.append(action)
        rewards = frames['reward'].copy()
        dones = frames['done'].astype(bool)
        self.returns += rewards
        self.lengths += 1
        infos = [dict(outcome=W.OUTCOMES[r['outcome']], elapsed_frames=int(r['elapsed']), layout=int(r['layout']),
                      seed=int(m['seed']), task=W.TASKS[int(m['task'])],
                      **{'TimeLimit.truncated': bool(r['truncated'])})
                 for r, m in zip(frames, meta)]
        terminal = np.flatnonzero(dones)
        for i in terminal:
            infos[i]['episode'] = {'r': float(self.returns[i]), 'l': int(self.lengths[i])}
            if self.owner.reward_profile == 'combat-v2':
                infos[i]['reward_components'] = {k: round(float(v), 4)
                                                 for k, v in zip(W.COMPONENTS, meta['components'][i])}
            infos[i]['episode_start'] = {'player_hp': float(meta['start'][i][0]),
                                         'boss_hp_fraction': float(meta['start'][i][1])}
            self.returns[i] = 0
            self.lengths[i] = 0
            self.episodes[i] += 1
            seed = int(meta['reset_seed'][i])
            if self.owner.training_seeds and seed >= 2 ** 31:
                raise ValueError('Training seed entered held-out namespace')
            self.seeds[i] = seed
            self.actions[i] = []
            self.starts[i] = tuple(map(float, meta['reset_start'][i]))
        if len(terminal):
            # Pack only reset rows in the pinned prefix, as FrameChunk does.
            slot.reset_frames[:len(terminal)] = self.owner.staging[1, self.start + terminal]
        return rewards, dones, infos

    def advance_and_upload(self, slot):
        result = self.advance(slot)
        count = int(result[1].sum())
        with torch.cuda.stream(self.copy_stream):
            if slot.used:
                self.copy_stream.wait_event(slot.consumed)
            slot.device.copy_(slot.host, non_blocking=True)
            if count:
                slot.reset_device[:count].copy_(slot.reset_host[:count], non_blocking=True)
            slot.upload_done.record(self.copy_stream)
        slot.used = True
        return result


class AbplusFrameVecEnv(GpuFrameVecEnv):
    """Rollout source of n AB+ environment slots for GpuMaskablePPO (room mixture or Monstro arena;
    combat-v2 or combat-v1 reward, both with the 120 s deadline observation).

    Subclasses GpuFrameVecEnv only for GpuMaskablePPO's type check: every method is overridden and
    GpuFrameVecEnv.__init__ (the Rust simulator) never runs.
    """

    def __init__(self, n=8, seed=1, chunks=1, device='cuda', reward_profile='combat-v2', start_randomization=None,
                 mode='exact', name='tr', port=27400, nice=19, history=HISTORY, capacity=ENTITY_CAPACITY,
                 startup_timeout=600, tasks=None, binary_obs=True, recycle_episodes=W.RECYCLE_EPISODES):
        if reward_profile not in W.REWARD_PROFILES:
            raise ValueError(f'AB+ reward profiles are {W.REWARD_PROFILES} (deadline observation)')
        if not 1 <= chunks <= n:
            raise ValueError('chunks must be 1..envs')
        if (history, capacity) != (HISTORY, ENTITY_CAPACITY):
            raise ValueError('The frame record is fixed at 64 frames x 256 entities')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('AbplusFrameVecEnv feeds GpuMaskablePPO and requires CUDA')
        self.base_seed, self.capacity, self.history, self.generation = seed, capacity, history, 0
        self.reward_profile = reward_profile
        # Training-only; evaluation constructs this env without it and therefore starts at full HP.
        self.start_randomization = validate_start_randomization(start_randomization)
        self.record_states = False
        self.training_seeds = False
        self.closed = True
        VecEnv.__init__(self, n, VisibleHistory(history, capacity, deadline=True).space, spaces.MultiDiscrete([45, 2, 2]))
        frame_bytes = 2 * n * W.FRAME_DTYPE.itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes + n * W.META_DTYPE.itemsize)
        self.staging = np.ndarray((2, n), dtype=W.FRAME_DTYPE, buffer=self.shm.buf)
        self.meta = np.ndarray((n,), dtype=W.META_DTYPE, buffer=self.shm.buf, offset=frame_bytes)
        self.meta[...] = np.zeros((), W.META_DTYPE)
        # tasks: abplus_tasks spec {'weights', 'normal', 'boss'}; None trains the Monstro arena only.
        config = dict(num_envs=n, base_seed=seed, mode=mode, name=name, port=port, nice=nice,
                      start_randomization=self.start_randomization, tasks=tasks, binary_obs=binary_obs,
                      reward_profile=reward_profile, recycle_episodes=recycle_episodes)
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        self.closed = False
        try:
            for i in range(n):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=W.worker_main, args=(i, config, self.shm.name, child),
                                   name=f'abplus-worker-{i}')
                proc.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(proc)
            deadline = time.monotonic() + startup_timeout
            pending = set(self.conns)
            while pending:
                ready = wait(list(pending), timeout=max(0.0, deadline - time.monotonic()))
                if not ready:
                    raise TimeoutError(f'{len(pending)} AB+ workers did not start within {startup_timeout} s')
                for conn in ready:
                    reply = conn.recv_bytes()
                    if reply[:1] != b'r':
                        raise RuntimeError(f'AB+ worker failed to start:\n{reply[1:].decode("utf8", "replace")}')
                    pending.discard(conn)
            boundaries = np.linspace(0, n, chunks + 1, dtype=int)
            self.chunks = [AbplusChunk(self, int(boundaries[i]), int(boundaries[i + 1])) for i in range(chunks)]
            self.executor = ThreadPoolExecutor(max_workers=chunks, thread_name_prefix='abplus-frame')
        except BaseException:
            self.close()
            raise

    def reset(self):
        torch.cuda.synchronize(self.device)
        if self._seeds[0] is not None:
            self.base_seed = int(self._seeds[0])
        futures = [self.executor.submit(c.reset, [self._seeds[i] if self._seeds[i] is not None else self.base_seed + i
                                                  for i in range(c.start, c.stop)]) for c in self.chunks]
        for future in futures:
            future.result()
        self._reset_seeds()
        self.generation += 1
        # One conventional SB3 reset observation, released by the collector at once.
        obs = {k: np.zeros((self.num_envs, *s.shape), s.dtype) for k, s in self.observation_space.spaces.items()}
        for c in self.chunks:
            for k, v in obs.items():
                source = np.ones(c.n, np.float32) if k == 'remaining_time' else c.slots[0].frames[k]
                v[c.start:c.stop, 0] = source
        return obs

    def diagnostics(self):
        """Worker timings (last step) and exchange latency, for the training log."""
        m = self.meta
        exchanges = sum(c.exchanges for c in self.chunks)
        cycles = sum(c.cycles for c in self.chunks)
        return {'step_cycle_ms': 1000 * sum(c.cycle_s for c in self.chunks) / max(1, cycles),
                'worker_step_ms': float(m['step_ms'].mean()), 'worker_step_ms_max': float(m['step_ms'].max()),
                'switch_wait_ms_max': float(m['switch_wait_ms'].max()), 'reset_ms_mean': float(m['reset_ms'].mean()),
                'worker_errors': int(m['errors'].sum()), 'instance_recycles': int(m['recycles'].sum()),
                'instance_start_failures': int(m['start_failures'].sum()),
                'recycle_deferrals': int(m['recycle_deferrals'].sum()),
                'exchange_ms': 1000 * sum(c.exchange_s for c in self.chunks) / max(1, exchanges)}

    def step_async(self, actions):
        raise NotImplementedError('Use the GpuMaskablePPO chunk collector')

    def step_wait(self):
        raise NotImplementedError('Use the GpuMaskablePPO chunk collector')

    def close(self):
        if self.closed:
            return
        self.closed = True
        executor = getattr(self, 'executor', None)
        if executor is not None:
            executor.shutdown(wait=True)
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        for conn in self.conns:
            try:
                conn.send_bytes(b'Q')
            except OSError:
                pass
        for proc in self.procs:
            proc.join(timeout=90)
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        for conn in self.conns:
            conn.close()
        del self.staging, self.meta
        self.shm.close()
        self.shm.unlink()

    def get_attr(self, name, indices=None):
        return [None if name == 'render_mode' else getattr(self, name) for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        raise NotImplementedError('Configure the source through the constructor')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
