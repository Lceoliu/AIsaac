"""AB+ source for GpuMaskablePPO: rollout worker processes around one exclusive CUDA learner.

Drop-in for gpu_env.GpuFrameVecEnv with the same chunk contract (pinned double-buffered
TransferSlots, advance_and_upload, reset frames packed in the prefix of reset_frames), so
FrameSampler, GpuHistoryRolloutBuffer and segment PPO run unchanged; the learner is the only
CUDA user. Each worker process (abplus_worker.py) owns one environment slot and two AB+ instances.

Per step the learner makes one batched policy call per chunk (chunks=1: all environments at
once), sends every worker 13 bytes over a pipe and copies back only the newest FRAME_DTYPE record
(~73 KB) per environment from shared memory into the chunk's pinned slot.

Duel (abplus_duel.py, bridge abp-0.2.9): one worker process per duel game serves two consecutive slots, the player (2k)
and the duel NPC (2k + 1); it gets both slots' actions in one 25-byte message and writes both frames. The policy acts
for both (self-play); infos carry the slot's side.
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
from .abplus_duel import SIDES, duel_worker_main
from .abplus_groups import BUDGETS, slot_groups
from .gpu_env import FULL_START, GpuFrameVecEnv, TransferSlot, sample_start, validate_start_randomization
from .plr import PrioritizedLevels
from .sim_vec import FRAME_DTYPE
from .transformer_obs import ENTITY_CAPACITY, FACTORED_NVEC, HISTORY, VisibleHistory

if W.FRAME_DTYPE != FRAME_DTYPE:
    raise ImportError('abplus_worker.FRAME_DTYPE differs from sim_vec.FRAME_DTYPE')
_probe = dict(boss_hp_prob=0.5, boss_hp_min=0.1, player_hp_prob=0.25, player_hp_min=3)
if any(W.sample_start(s, _probe) != sample_start(s, _probe) for s in range(64)):
    raise ImportError('abplus_worker.sample_start differs from gpu_env.sample_start')


def step_messages(actions, agents=1):
    """The workers' step messages for the slots' (joint, bomb, item) rows: b'S' + int32s, one per worker of `agents`
    consecutive slots (duel: 2)."""
    return [b'S' + struct.pack(f'<{3 * agents}i', *[int(x) for a in actions[j:j + agents] for x in a])
            for j in range(0, len(actions), agents)]


class AbplusChunk:
    """A group of workers stepped together; one policy call and one upload per step."""

    def __init__(self, owner, start, stop):
        self.owner, self.start, self.stop, self.n = owner, start, stop, stop - start
        # one worker per owner.agents slots (duel: 2); chunk boundaries fall between workers
        self.agents = owner.agents
        self.conns = owner.conns[start // self.agents:stop // self.agents]
        self.copy_stream = torch.cuda.Stream(device=owner.device)
        # High priority: with asynchronous training the learner's kernels share the GPU.
        self.compute_stream = torch.cuda.Stream(device=owner.device, priority=-1)
        self.ready = torch.cuda.Event()
        self.slots = [TransferSlot(self.n, owner.device, owner.frame_dtype) for _ in range(2)]
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
                    errors.append(f'[worker {self.start // self.agents + i}] {reply[1:].decode("utf8", "replace")}')
        if errors:
            raise RuntimeError('AB+ worker failure:\n' + '\n'.join(errors))

    def reset(self, seeds, starts=None):
        """starts=None: workers sample the training start from the seed (gpu_env.sample_start)."""
        self.seeds = list(map(int, seeds))
        self.actions = [[] for _ in range(self.n)]
        messages = []
        for i in range(0, self.n, self.agents):   # a worker's first slot (a duel worker draws its own first seed)
            player, boss = (float('nan'), float('nan')) if starts is None else map(float, starts[i])
            messages.append(b'R' + struct.pack('<qqff', self.seeds[i], self.owner.base_seed, player, boss))
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
        self._exchange(step_messages(actions, self.agents))
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
                      seed=int(m['seed']), task=W.TASKS[int(m['task'])], level=int(m['level']), group=int(m['group']),
                      replay=int(m['replay']), rss_mib=float(m['rss_mib']), instance=int(m['instance']),
                      instance_episodes=int(m['instance_episodes']),
                      **({'option': W.GOAL_TASKS[int(m['option'])], 'source': W.SOURCES[int(m['source'])]}
                         if self.owner.goal_line else {}),
                      **{'TimeLimit.truncated': bool(r['truncated'])})
                 for r, m in zip(frames, meta)]
        if self.agents == 2:   # duel: slot 2k is the player, 2k + 1 the duel NPC
            for i, info in enumerate(infos):
                info['side'] = SIDES[(self.start + i) % 2]
        terminal = np.flatnonzero(dones)
        components = self.owner.components
        for i in terminal:
            infos[i]['episode'] = {'r': float(self.returns[i]), 'l': int(self.lengths[i])}
            if self.owner.goal_line:   # the option's reward: combat-hp2 or GotoReward
                names = W.COMPONENTS_HP if infos[i]['option'] == 'combat' else W.COMPONENTS_GOTO
                infos[i]['reward_components'] = {k: round(float(v), 4) for k, v in zip(names, meta['components'][i])}
                if meta['seq_end'][i]:   # C44: the option sequence ended here (the Room Buffer's unit)
                    infos[i]['sequence'] = dict(ok=bool(meta['seq_ok'][i]), hurt=round(float(meta['seq_hurt'][i]), 2),
                                                steps=int(meta['seq_steps'][i]))
            elif components:
                infos[i]['reward_components'] = {k: round(float(v), 4)
                                                 for k, v in zip(components, meta['components'][i])}
            infos[i]['episode_start'] = {'player_hp': float(meta['start'][i][0]),
                                         'boss_hp_fraction': float(meta['start'][i][1]),
                                         'bombs': int(meta['bombs'][i]),
                                         **({'stats': {k: round(float(v), 4) for k, v in zip(W.STAT_KEYS, meta['stats_mod'][i])}}
                                            if meta['stats_mod'][i].any() else {})}
            infos[i]['episode_stats'] = {k: round(float(v), 2) for k, v in zip(W.EPISODE_STATS, meta['stats'][i])}
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
                 startup_timeout=600, tasks=None, binary_obs=True, recycle_episodes=W.RECYCLE_EPISODES,
                 recycle_rss_mib=W.RECYCLE_RSS_MIB,
                 start_bombs=None, plr_levels=None, reward_options=None, lineage_mode=None,
                 max_episode_frames=W.MAX_EPISODE_FRAMES, invincible=False, miss_cap=0, target=None,
                 hurt_ends=False, groups=None, budget='steps', duel=None, frames_per_decision=2, room_buffer=None,
                 stat_noise=None):
        """groups (abplus_groups.load_groups) replace tasks / target / max_episode_frames per episode; budget
        splits the steps between them (abplus_groups.BUDGETS); with plr_levels (plr.group_levels) each group draws
        its rooms by PLR. duel: a duel spec file's contents (abplus_duel, catalog/duel_rooms.json): n slots are n / 2
        duel games, the player in slot 2k and the duel NPC in 2k + 1. frames_per_decision: logic frames per action
        (C39). room_buffer (C39): dict(capacity, fresh_share) for a per-group Room Buffer seed table in shared memory
        (set_room_buffer)."""
        if reward_profile not in W.REWARD_PROFILES:
            raise ValueError(f'AB+ reward profiles are {W.REWARD_PROFILES} (deadline observation)')
        self.duel = dict(duel) if duel else None
        self.agents = 2 if self.duel else 1
        if self.duel and (groups or plr_levels or tasks or target or invincible):
            raise ValueError('the duel brings its own rooms and NPC (no groups, tasks, target, PLR or invincibility)')
        if n % self.agents:
            raise ValueError('duel slots come in pairs: the number of environments must be even')
        if not 1 <= chunks <= n // self.agents:
            raise ValueError('chunks must be 1..envs (duel: 1..games)')
        if (history, capacity) != (HISTORY, ENTITY_CAPACITY):
            raise ValueError('The frame record is fixed at 64 frames x 256 entities')
        if groups and (tasks or target):
            raise ValueError('parallel task groups bring their own rooms and targets (no tasks or target)')
        if groups and budget not in BUDGETS:
            raise ValueError(f'budget must be one of {BUDGETS}')
        self.groups = [dict(g) for g in groups] if groups else None
        self.budget = budget if groups else None
        # slots budget: the group of every environment slot (interleaved; abplus_groups.slot_groups)
        self.slot_groups = slot_groups([g['share'] for g in self.groups], n) if self.groups else None
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
        self.components = {'combat-v2': W.COMPONENTS, 'combat-v3': W.COMPONENTS_V3,
                           'combat-v4': W.COMPONENTS_V4, 'combat-v5': W.COMPONENTS_V5,
                           'combat-hitrate': W.COMPONENTS_HR, 'combat-hitrate-walk': W.COMPONENTS_HRW,
                           'combat-hitrate-miss': W.COMPONENTS_HRM,
                           'combat-hitrate-fire': W.COMPONENTS_HRF,
                           'combat-hitrate-hurt': W.COMPONENTS_HRH, 'combat-hp': W.COMPONENTS_HP,
                           'combat-hp2': W.COMPONENTS_HP, 'combat-hp2-camera': W.COMPONENTS_HP,
                           'goal-hp': W.COMPONENTS_HP, 'goal-hp2': W.COMPONENTS_HP,
                           'goal-hp3': W.COMPONENTS_HP}.get(reward_profile)
        self.goal_line = reward_profile in W.GOAL_PROFILES
        if self.components and (reward_options or {}).get('hurt_rest'):
            self.components = self.components + ('rest',)   # abplus_reward.COMPONENTS_HRM_REST
        # combat-v3 frames add the room state input (abplus_worker.FRAME_DTYPE_COMBAT).
        self.frame_dtype, _ = W.frame_layout(reward_profile)
        # start_bombs {'zero_prob', 'max'}: training bomb start (abplus_worker.sample_bombs); None: 1.
        self.start_bombs = dict(start_bombs) if start_bombs else None
        options = W.observation_options(reward_profile)
        # combat-v5: the policy's heads are (move, shoot, bomb, item); the workers keep the joint layout.
        self.factored_actions = bool(options.get('factored_actions'))
        VecEnv.__init__(self, n, VisibleHistory(history, capacity, **options).space,
                        spaces.MultiDiscrete(FACTORED_NVEC if self.factored_actions else [45, 2, 2]))
        frame_bytes = 2 * n * self.frame_dtype.itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes + n * W.META_DTYPE.itemsize)
        self.staging = np.ndarray((2, n), dtype=self.frame_dtype, buffer=self.shm.buf)
        self.meta = np.ndarray((n,), dtype=W.META_DTYPE, buffer=self.shm.buf, offset=frame_bytes)
        self.meta[...] = np.zeros((), W.META_DTYPE)
        # plr_levels: the rooms Prioritized Level Replay draws from (plr.mixture_levels); the learner
        # writes their probabilities with set_level_probabilities(), workers read them per episode.
        self.plr_levels = [list(level) for level in plr_levels] if plr_levels else None
        self.plr_shm = self.level_probabilities = None
        if self.plr_levels:
            if not tasks and not self.groups:
                raise ValueError('plr_levels need the tasks spec (its kind weights) or groups')
            self.plr_shm = shared_memory.SharedMemory(create=True, size=8 * len(self.plr_levels))
            self.level_probabilities = np.ndarray((len(self.plr_levels),), np.float64, buffer=self.plr_shm.buf)
            # The workers' first episodes start before the learner writes PLR's distribution: the
            # mixture's kind weights over rooms not played yet (not uniform over all rooms).
            weights = {g['name']: g['share'] for g in self.groups} if self.groups else tasks['weights']
            self.level_probabilities[:] = PrioritizedLevels(self.plr_levels, weights).probabilities()
        # C39 Room Buffer table: seeds (G, K) int64, start probabilities (G, K), fresh start probability (G,).
        self.buffer_shm = self.buffer_arrays = None
        if room_buffer:
            if not self.groups or plr_levels:
                raise ValueError('the Room Buffer needs parallel task groups and no PLR')
            g, k = len(self.groups), int(room_buffer['capacity'])
            self.buffer_shm = shared_memory.SharedMemory(create=True, size=16 * g * k + 8 * g)
            self.buffer_arrays = (np.ndarray((g, k), np.int64, buffer=self.buffer_shm.buf),
                                  np.ndarray((g, k), np.float64, buffer=self.buffer_shm.buf, offset=8 * g * k),
                                  np.ndarray((g,), np.float64, buffer=self.buffer_shm.buf, offset=16 * g * k))
            self.buffer_arrays[0][:] = -1
            self.buffer_arrays[1][:] = 0.0
            self.buffer_arrays[2][:] = 1.0     # every episode fresh until the learner writes the first table
        # tasks: abplus_tasks spec {'weights', 'normal', 'boss'}; None trains the Monstro arena only.
        config = dict(num_envs=n, base_seed=seed, mode=mode, name=name, port=port, nice=nice,
                      start_randomization=self.start_randomization, tasks=tasks, binary_obs=binary_obs,
                      reward_profile=reward_profile, recycle_episodes=recycle_episodes,
                      recycle_rss_mib=recycle_rss_mib,
                      start_bombs=self.start_bombs, plr_levels=self.plr_levels,
                      plr_shm=self.plr_shm.name if self.plr_shm else None,
                      reward_options=dict(reward_options or {}),
                      # The hit-rate test: 180 s episodes, an invincible player; combat-hitrate-miss's
                      # miss cap (bridge abp-0.2.6, 0 = none).
                      max_episode_frames=int(max_episode_frames), invincible=bool(invincible),
                      miss_cap=int(miss_cap), target=dict(target) if target else None,
                      hurt_ends=bool(hurt_ends),   # C37: the first damage ends the episode (abplus_worker)
                      frames_per_decision=int(frames_per_decision),   # C39
                      stat_noise=dict(stat_noise) if stat_noise else None,   # C41 (abplus_worker.sample_stats)
                      room_buffer=(dict(shm=self.buffer_shm.name, groups=len(self.groups), capacity=int(room_buffer['capacity']),
                                        fresh_share=float(room_buffer['fresh_share'])) if self.buffer_shm else None),
                      groups=self.groups, budget=self.budget, slot_groups=self.slot_groups,
                      duel_spec=self.duel, num_games=n // self.agents,
                      **({'lineage_mode': int(lineage_mode)} if lineage_mode is not None else {}))
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        self.closed = False
        try:
            for i in range(n // self.agents):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=duel_worker_main if self.duel else W.worker_main,
                                   args=(i, config, self.shm.name, child), name=f'abplus-worker-{i}')
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
            boundaries = np.linspace(0, n // self.agents, chunks + 1, dtype=int) * self.agents
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
                'memory_recycles': int(m['memory_recycles'].sum()),
                'instance_rss_max_mib': float(m['rss_mib'].max()),
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
        if self.plr_shm is not None:
            self.level_probabilities = None
            self.plr_shm.close()
            self.plr_shm.unlink()
        if self.buffer_shm is not None:
            self.buffer_arrays = None
            self.buffer_shm.close()
            self.buffer_shm.unlink()

    def transfer_actions(self, actions):
        """Factored policy actions (move, shoot, bomb, item) -> the workers' (joint, bomb, item)."""
        if not self.factored_actions:
            return actions
        return torch.stack([actions[:, 0] * 5 + actions[:, 1], actions[:, 2], actions[:, 3]], -1)

    def set_level_probabilities(self, p):
        """Room distribution of the next episodes (PLR); workers read it at every episode start."""
        p = np.asarray(p, np.float64)
        if self.level_probabilities is None or p.shape != self.level_probabilities.shape:
            raise ValueError('set_level_probabilities needs plr_levels and one probability per level')
        self.level_probabilities[:] = p / p.sum()

    def set_room_buffer(self, seeds, probs, fresh):
        """The Room Buffer table the workers read at every episode boundary (C39)."""
        if self.buffer_arrays is None:
            raise ValueError('set_room_buffer needs room_buffer')
        s, p, f = self.buffer_arrays
        s[:], p[:], f[:] = seeds, probs, fresh

    def get_attr(self, name, indices=None):
        return [None if name == 'render_mode' else getattr(self, name) for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        raise NotImplementedError('Configure the source through the constructor')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
