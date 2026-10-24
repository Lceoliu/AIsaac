"""The duel arena (user request 2026-09-28, EXPERIMENTS.md A6): the player against a second "Isaac", both driven by the
same policy, each through its own first-person observation, both paid by the same reward.

The second Isaac is the bridge's duel NPC (abp_bridge.lua abp-0.2.9): a Pacer that moves by the player's movement law,
shoots projectiles on the player's fire delay with a tear's speed, spawn point, range and hit radius, has six hits of HP
and the player's damage cooldown. The bridge reports both sides' shots, hits, misses and damage the same way
(obs['duel']).

Views (duel_views): each side sees the room as the player sees it in every other task, so the policy's input format,
action heads and reward code are unchanged.
  - The player's view is the observation with the combat counters taken from the duel's player side.
  - The NPC's view swaps the roles: players[0] is the NPC (its position, Size, hits left as half hearts, the player's
    stats, which the NPC's are calibrated to), the player becomes an entity with the NPC's entity record (the same kind,
    flags and collision fields as the NPC has in the player's view), the NPC's projectiles become own tears and the
    player's tears the other side's projectiles, each with that kind's collision fields, and the combat counters are
    the NPC side's.
  - In both views the other side's HP is in NPC HP units (a player half heart = HP_UNIT), so a hit is the same number
    of HP (3.5) on either side, and the combat fields (blocking_hp, lineage_damage, tear_hits, tear_misses,
    miss_units, miss_streak, player_damage) mean the same thing for both.
Both views run through their own VisibleHistory; the reward is one instance of the run's reward class per side
(combat-hitrate-miss: hit per 1.25 HP, kill, hit-rate time price, miss penalty; with hurt_ends the first damage to
either side ends the episode: 'hurt' for the side that took it, 'win' for the other).

Training: one AB+ instance serves two environment slots of AbplusFrameVecEnv (slot 2k the player, 2k + 1 the NPC);
the learner's policy acts for both (self-play with the current weights). DuelWorker keeps abplus_worker.Worker's
active / standby instance switch, recycling and error handling.
"""
from __future__ import annotations

import math
import os
import struct
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from .abplus import AbplusTrainingEnv, RoomUnusable
from .abplus_reward import REWARDS
from .abplus_tasks import TASKS, TaskSampler
from .abplus_worker import (FULL_START, META_DTYPE, OUTCOMES, RECYCLE_EPISODES, EpisodeStats, Instance,
                            frame_layout, observation_options, write_frame)
from .env import Action, BridgeError
from .transformer_obs import ENTITY_CAPACITY, HISTORY, VisibleHistory

SIDES = ('player', 'npc')
HP_UNIT = 3.5                 # NPC HP per player half heart (21 HP, 6 half hearts: six hits each)
# Collision fields of a shot seen as one's own tear or as the other side's projectile (AB+ player tears and duel
# projectiles, A6), with their scale.
TEAR_FIELDS = dict(coll=4, gcoll=4, cdmg=3.5, scale=1.0206242799759)
PROJECTILE_FIELDS = dict(coll=4, gcoll=4, cdmg=0.0, scale=1.0)
TYPE_TEAR, TYPE_PROJECTILE = 2, 9
MOVE_UNITS = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))


def _shot(e, kind):
    """A shot's entity record as kind ('tear' or 'projectile')."""
    rec = {k: v for k, v in e.items() if k != 'projectile'}
    if kind == 'tear':
        rec.update(type=TYPE_TEAR, variant=0, subtype=0, **TEAR_FIELDS)
    else:
        rec.update(type=TYPE_PROJECTILE, variant=0, subtype=0, projectile=True, **PROJECTILE_FIELDS)
    return rec


def _combat(own, damage_scale, hurt_scale, other_hp, other_dead):
    """A view's combat counters from its side's duel counters: damage dealt x damage_scale is in NPC HP units, damage taken
    x hurt_scale in half hearts; other_hp: the other side's HP in NPC HP units."""
    alive = not other_dead
    damage = float(own['damage']) * damage_scale
    return dict(player_damage_events=float(own['hurt']), player_damage=float(own['hurt_amount']) * hurt_scale,
                enemy_damage_events=float(own['hits']), enemy_damage=damage, enemy_damage_fraction=damage / (6 * HP_UNIT),
                blocking_hp=other_hp if alive else 0.0, blocking_points=0.0, blocking_count=float(alive),
                lineage_damage=damage, lineage_kills=float(other_dead), lineage_count=float(alive),
                lineage_hp=other_hp if alive else 0.0, tear_hits=float(own['hits']), blocked_hits=0.0,
                tear_misses=float(own['misses']), miss_units=float(own['miss_units']),
                miss_streak=float(own['miss_streak']), credits=[])


def duel_views(obs):
    """(the player's view, the NPC's view) of a duel observation (module docstring)."""
    d = obs.get('duel')
    if not d or not d.get('active'):
        raise BridgeError('duel observation without an active duel NPC')
    player = obs['players'][0]
    entities = obs['entities']
    npc_rec = next((e for e in entities if e['id'] == d['npc'] and e['type'] == 11), None)
    if npc_rec is None:
        raise BridgeError(f"duel NPC {d['npc']} is not in the observation")
    player_dead, npc_dead = bool(player.get('dead')), bool(d['dead'])
    player_hp = float(player['hearts']) * HP_UNIT     # the player as the NPC sees it, in NPC HP units
    npc_hp = float(d['hp'])
    # The player's view: the observation itself, with the duel's player-side counters.
    mine = dict(obs, combat=_combat(d['player'], 1.0, 1.0, npc_hp, npc_dead),
                events=dict(damage=float(d['player']['hurt']), tears=float(d['player']['shots']),
                            npc_deaths=float(npc_dead), clears=0.0))
    # The NPC's view.
    npc_player = dict(player, id=d['npc'], pos=list(d['pos']), size=float(d['size']), hearts=npc_hp / HP_UNIT,
                      max_hearts=6, soul=0, black=0, bone=0, eternal=0, golden=0, lives=0, coins=0, bombs=0, keys=0,
                      can_fly=False, active=0, active_charge=0, active_ready=False, invulnerable=d['iframes'] > 0,
                      controls=True, aframe=0, flip=False, dead=npc_dead, ptype=0)
    npc_player.pop('vel', None)
    other = dict(npc_rec, id=player['id'], pos=list(player['pos']), size=float(player['size']), age=int(obs['logic_frames']))
    swapped = [other]
    for e in entities:
        if e is npc_rec:
            continue
        if e['type'] == TYPE_PROJECTILE:
            swapped.append(_shot(e, 'tear'))
        elif e['type'] == TYPE_TEAR:
            swapped.append(_shot(e, 'projectile'))
        else:
            swapped.append(e)
    theirs = dict(obs, players=[npc_player], entities=swapped,
                  combat=_combat(d['npc_side'], HP_UNIT, 1.0 / HP_UNIT, player_hp, player_dead),
                  events=dict(damage=float(d['npc_side']['hurt']), tears=float(d['npc_side']['shots']),
                              npc_deaths=float(player_dead), clears=0.0))
    return mine, theirs


def duel_outcomes(views, elapsed, max_frames, hurt_ends):
    """Per side: 'hurt' / 'win' (hurt_ends: the first damage to either side ends the episode), else 'death' / 'win', else
    'time_limit' at the deadline, else 'running'."""
    hurt = [v['combat']['player_damage_events'] > 0 for v in views]
    dead = [bool(v['players'][0].get('dead')) for v in views]
    if hurt_ends and any(hurt):
        return ['hurt' if h else 'win' for h in hurt]
    if any(dead):
        return ['death' if x else 'win' for x in dead]
    if elapsed >= max_frames:
        return ['time_limit', 'time_limit']
    return ['running', 'running']


GLIDE = 1.0 / (1.0 - 0.8803 ** 2)   # px a released walker still slides per px/frame of speed (the movement law)


class ScriptedDuellist:
    """The evaluation's fixed opponent: walks to where its glide ends on the other side's row or column, shoots along it
    once aligned and not sliding sideways (a shot inherits 1.2 x the speed), walks closer while farther than reach; never
    dodges. (duel_probe's align-and-shoot overshoots and its sliding shots curve off: 0 hits in most duels.)
    One instance per side and episode: it keeps the last position to estimate its own speed."""

    def __init__(self, band=12.0, reach=200.0, slide=1.5):
        self.band, self.reach, self.slide = band, reach, slide
        self.previous = None

    def reset(self):
        self.previous = None

    def __call__(self, view, frames=2):
        """(move, shoot) for a view; frames: logic frames since the previous call."""
        me = view['players'][0]['pos']
        other = next(e for e in view['entities'] if e.get('lineage'))['pos']
        vx, vy = ((me[0] - self.previous[0]) / frames, (me[1] - self.previous[1]) / frames) if self.previous else (0.0, 0.0)
        self.previous = tuple(me)
        rx, ry = other[0] - me[0], other[1] - me[1]
        if abs(rx) <= self.band and abs(vx) < self.slide:
            return (0 if abs(ry) <= self.reach else (5 if ry > 0 else 1)), (1 if ry < 0 else 3)
        if abs(ry) <= self.band and abs(vy) < self.slide:
            return (0 if abs(rx) <= self.reach else (3 if rx > 0 else 7)), (4 if rx < 0 else 2)
        dx, dy = rx - GLIDE * vx, ry - GLIDE * vy     # from where a stop now would end
        if abs(dx) <= abs(dy):
            return (0 if abs(dx) <= self.band else (3 if dx > 0 else 7)), 0
        return (0 if abs(dy) <= self.band else (5 if dy > 0 else 1)), 0


class DuelEnv:
    """One AB+ instance playing duels: the bridge with the duel spec, one VisibleHistory per side."""

    def __init__(self, port, max_episode_frames, duel, tasks, hurt_ends=True, lineage_mode=3, miss_cap=20,
                 binary_obs=True, history=HISTORY, entity_capacity=ENTITY_CAPACITY, **obs_options):
        b = self.bridge = AbplusTrainingEnv(port=port)
        b.action_repeat = 2
        b.binary_obs, b.lineage_mode, b.miss_cap, b.invincible = bool(binary_obs), int(lineage_mode), int(miss_cap), False
        b.duel, b.tasks, b.target = dict(duel), tasks, None
        self.histories = [VisibleHistory(history, entity_capacity, **obs_options) for _ in SIDES]
        self.observation_space = self.histories[0].space
        self.max_episode_frames, self.hurt_ends = int(max_episode_frames), bool(hurt_ends)
        self.connected, self.finished, self.transport_failed = False, True, False
        self.raw_obs = self.views = None
        self.elapsed_frames = 0

    @property
    def history(self):   # Instance.prepare reads history.last_rows
        return self.histories[0]

    def reset(self, seed):
        """frames (player, NPC), info. The room, arm and cells come from the seed (TaskSampler, abplus.duel_cells)."""
        if not self.connected:
            self.bridge.connect()
            self.connected = True
        for h in self.histories:
            h.clear()
        try:
            obs, info = self.bridge.reset_monstro(int(seed))
        except (OSError, BridgeError):
            self.transport_failed, self.finished = True, True
            raise
        self.transport_failed, self.finished = False, False
        self.raw_obs, self.elapsed_frames = obs, 0
        self.views = duel_views(obs)
        frames = [h.encode(v) for h, v in zip(self.histories, self.views)]
        return frames, {**info, 'arena_seed': int(seed), 'outcomes': ['running', 'running']}

    def step(self, moves, shoots):
        """One decision (2 logic frames) of both sides: moves, shoots = (player, NPC). Returns frames, outcomes (per side),
        terminated, truncated, info."""
        if self.finished:
            raise RuntimeError('reset() is required before step() or after a terminal transition')
        repeat = min(self.bridge.action_repeat, self.max_episode_frames - self.elapsed_frames)
        previous = self.raw_obs
        for h, m, s in zip(self.histories, moves, shoots):
            h.set_previous_action(int(m) * 5 + int(s), 0, 0)
        self.bridge.duel_action = (int(moves[1]), int(shoots[1]))
        try:
            obs, _, _, _, info = self.bridge.step(Action(move=int(moves[0]), shoot=int(shoots[0])), repeat=repeat)
        except (OSError, BridgeError):
            self.transport_failed, self.finished = True, True
            raise
        advanced = obs['logic_frames'] - previous['logic_frames']
        if advanced != repeat:
            raise RuntimeError(f'Expected {repeat} logic frames, received {advanced}')
        if obs['room']['room_idx'] != previous['room']['room_idx'] or obs['room']['clear']:
            self.finished = True
            raise BridgeError(f"duel room changed or cleared: {previous['room']['room_idx']} -> {obs['room']['room_idx']}")
        self.raw_obs = obs
        self.elapsed_frames += advanced
        self.views = duel_views(obs)
        outcomes = duel_outcomes(self.views, self.elapsed_frames, self.max_episode_frames, self.hurt_ends)
        terminated = any(o in ('hurt', 'win', 'death') for o in outcomes)
        truncated = not terminated and outcomes[0] == 'time_limit'
        self.finished = terminated or truncated
        frames = [h.encode(v) for h, v in zip(self.histories, self.views)]
        return frames, outcomes, terminated, truncated, {**info, 'outcomes': outcomes, 'elapsed_frames': self.elapsed_frames,
                                                         'logic_frames_advanced': advanced}

    def close(self):
        if self.connected:
            try:
                if not self.transport_failed:
                    self.bridge.reset_safe()
            finally:
                self.bridge.close()
                self.connected, self.finished = False, True


def duel_tasks(spec):
    """TaskSampler of a duel spec file: the normal rooms of its arms (terrains), drawn by the arms' weights."""
    return TaskSampler(spec['weights'], spec['normal'], spec['boss'], spec['duel'].get('arms'))


class DuelInstance(Instance):
    """abplus_worker.Instance whose game plays duels (DuelEnv)."""

    def _env(self, port):
        c = self.config
        return DuelEnv(port, int(c.get('max_episode_frames', 900)), c['duel_spec']['duel'], duel_tasks(c['duel_spec']),
                       hurt_ends=bool(c.get('hurt_ends', True)), lineage_mode=int(c.get('lineage_mode', 3)),
                       miss_cap=int(c.get('miss_cap', 20)), binary_obs=c.get('binary_obs', True),
                       **observation_options(c.get('reward_profile')))

    def prepare(self, seed, start=FULL_START, bombs=0, group=-1, recycle=False):
        import threading

        def run():
            try:
                if recycle:
                    self._recycle()
                self.episodes += 1
                t = time.perf_counter()
                frames, info = self.env.reset(int(seed))
                rows = [h.last_rows for h in self.env.histories]
                self.result = (int(seed), FULL_START, 0, frames, info, rows, 1000 * (time.perf_counter() - t), -1)
            except BaseException:
                self.error = traceback.format_exc()
            finally:
                self.ready.set()
        self.ready.clear()
        self.result = self.error = None
        self.thread = threading.Thread(target=run, name=f'prepare-{self.name}', daemon=True)
        self.thread.start()


def duel_seed(config, game, episode):
    """Episode seed of game k's e-th episode (abplus_worker.episode_seed over games instead of slots)."""
    return config['base_seed'] + game + config['num_games'] * episode


class DuelWorker:
    """One duel game (two AB+ instances, active + standby) serving environment slots 2k (player) and 2k + 1 (NPC)."""

    def __init__(self, game, config, step_rows, reset_rows, metas):
        self.game, self.config = game, config
        self.step_rows, self.reset_rows, self.metas = step_rows, reset_rows, metas
        name = f"{config['name']}{game}"
        port = config['port'] + 2 * game
        alt = 2 * config['num_games']
        self.instances = [DuelInstance(name + 'a', port, config, port + alt),
                          DuelInstance(name + 'b', port + 1, config, port + 1 + alt)]
        self.active = 0
        self.episode = 0
        self.seed = None
        self.layout = 0
        self.elapsed = 0
        self.last_frames = None
        self.recycle_after = int(config.get('recycle_episodes', RECYCLE_EPISODES))
        self.profile = config.get('reward_profile', 'combat-hitrate-miss')
        options = config.get('reward_options', {})
        self.rewards = [REWARDS[self.profile](**options) for _ in SIDES]
        self.stats = [None, None]

    def schedule(self, episode):
        return (duel_seed(self.config, self.game, episode),)

    def _begin(self, index, result):
        seed, _, _, frames, info, rows, reset_ms, _ = result
        self.active = index
        self.seed = seed
        self.layout = int(info.get('room_variant', 0))
        self.elapsed = 0
        self.last_frames = frames
        env = self.instances[index].env
        for s, (reward, view) in enumerate(zip(self.rewards, env.views)):
            reward.reset(view, 'normal')
            self.stats[s] = EpisodeStats(view)
        return frames, rows, reset_ms

    def _meta(self, **values):
        for meta in self.metas:
            for k, v in values.items():
                meta[k] = v

    def _write_reset(self, rows_out, frames, rows):
        env = self.instances[self.active].env
        for s in range(2):
            write_frame(rows_out[s], frames[s], 0.0, False, False, 0, 0, self.layout, rows[s], raw=env.views[s])

    def reset(self, base_seed):
        """First episode of a (new) collection generation; every episode's seed is duel_seed(base_seed, game, episode)."""
        self.config = {**self.config, 'base_seed': int(base_seed)}
        self.episode = 0
        for instance in self.instances:
            if instance.thread is not None:
                try:
                    instance.take()
                except RuntimeError:
                    instance.relaunch()
        first = self.instances[0]
        first.prepare(*self.schedule(0))
        frames, rows, reset_ms = self._begin(0, first.take())
        self._write_reset(self.step_rows, frames, rows)
        self.instances[1].prepare(*self.schedule(1))
        self._meta(seed=self.seed, start=FULL_START, task=TASKS.index('normal'), level=-1, bombs=0, group=-1,
                   reset_group=-1, reset_ms=reset_ms, episodes=0)

    def step(self, actions):
        """actions: ((joint, bomb, item) of the player's slot, of the NPC's slot)."""
        active = self.instances[self.active]
        env = active.env
        t = time.perf_counter()
        self._meta(seed=self.seed, start=FULL_START, task=TASKS.index('normal'), level=-1, bombs=0, group=-1)
        moves = [int(a[0]) // 5 for a in actions]
        shoots = [int(a[0]) % 5 for a in actions]
        try:
            frames, outcomes, terminated, truncated, info = env.step(moves, shoots)
            elapsed = int(info['elapsed_frames'])
            done = bool(terminated or truncated)
            for s in range(2):
                view = env.views[s]
                self.stats[s].step(view, elapsed - self.elapsed)
                parts = self.rewards[s].step(view, outcomes[s], elapsed - self.elapsed)
                reward = self.rewards[s].scale * float(sum(parts.values()))
                write_frame(self.step_rows[s], frames[s], reward, done, bool(truncated), OUTCOMES.index(outcomes[s]),
                            elapsed, self.layout, env.histories[s].last_rows, float(parts.get('hit', 0.0)), None, raw=view)
            self.last_frames = frames
            self.elapsed = elapsed
        except Exception:
            # Engine or bridge failure: both sides end as a truncation on their last good frame; the process is replaced
            # and the standby instance plays on.
            for meta in self.metas:
                meta['errors'] += 1
            print(f'[duel {self.game}] {active.name} failed:\n{traceback.format_exc()}', flush=True)
            for s in range(2):
                write_frame(self.step_rows[s], self.last_frames[s], 0.0, True, True, OUTCOMES.index('error'),
                            self.elapsed, self.layout, int(self.last_frames[s]['entity_mask'].sum()))
            done = True
            active.relaunch()
        step_ms = 1000 * (time.perf_counter() - t)
        self._meta(step_ms=step_ms)
        if not done:
            return
        for meta, reward, stats in zip(self.metas, self.rewards, self.stats):
            totals = reward.totals_array()
            meta['components'] = np.pad(totals, (0, len(meta['components']) - len(totals)))
            meta['stats'] = stats.array()
        self.episode += 1
        standby_index = 1 - self.active
        standby = self.instances[standby_index]
        t = time.perf_counter()
        for attempt in range(3):
            try:
                result = standby.take()
                break
            except RuntimeError:
                print(f'[duel {self.game}] standby reset failed (attempt {attempt}):\n{traceback.format_exc()}', flush=True)
                for meta in self.metas:
                    meta['errors'] += 1
                standby.relaunch()
                standby.prepare(*self.schedule(self.episode))
        else:
            raise RuntimeError(f'duel {self.game}: standby instance keeps failing')
        switch_ms = 1000 * (time.perf_counter() - t)
        old = self.active
        frames, rows, reset_ms = self._begin(standby_index, result)
        self._write_reset(self.reset_rows, frames, rows)
        self._meta(switch_wait_ms=switch_ms, reset_seed=self.seed, reset_start=FULL_START,
                   reset_task=TASKS.index('normal'), reset_level=-1, reset_bombs=0, reset_group=-1, reset_ms=reset_ms,
                   episodes=self.episode,
                   recycles=sum(i.recycles for i in self.instances),
                   start_failures=sum(i.start_failures for i in self.instances),
                   recycle_deferrals=sum(i.recycle_deferrals for i in self.instances))
        finished = self.instances[old]
        recycle = bool(self.recycle_after) and finished.episodes >= self.recycle_after
        finished.prepare(*self.schedule(self.episode + 1), recycle=recycle)

    def close(self):
        for instance in self.instances:
            instance.close()


def duel_worker_main(game, config, shm_name, conn):
    """Process entry of duel game k (slots 2k, 2k + 1). Protocol as abplus_worker.worker_main: R + seed, base_seed:int64 +
    start:2f32 (seed and start ignored: duel_seed, full health) -> k; S + (joint, bomb, item) x 2:int32 -> k; Q -> exit;
    failures answer E + traceback."""
    current = os.nice(0)
    if config.get('nice', 0) > current:
        os.nice(config['nice'] - current)
    shm = shared_memory.SharedMemory(name=shm_name)
    n = config['num_envs']
    frame_dtype, _ = frame_layout(config.get('reward_profile'))
    frames = np.ndarray((2, n), dtype=frame_dtype, buffer=shm.buf)
    meta = np.ndarray((n,), dtype=META_DTYPE, buffer=shm.buf, offset=2 * n * frame_dtype.itemsize)
    worker = None
    try:
        slots = (2 * game, 2 * game + 1)
        worker = DuelWorker(game, config, [frames[0, i:i + 1][0] for i in slots], [frames[1, i:i + 1][0] for i in slots],
                            [meta[i:i + 1][0] for i in slots])
        conn.send_bytes(b'r')
        while True:
            msg = conn.recv_bytes()
            kind = msg[:1]
            try:
                if kind == b'S':
                    a = struct.unpack('<6i', msg[1:25])
                    worker.step((a[0:3], a[3:6]))
                elif kind == b'R':
                    _, base_seed, _, _ = struct.unpack('<qqff', msg[1:25])   # the game's own seeds (duel_seed)
                    worker.reset(base_seed)
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
