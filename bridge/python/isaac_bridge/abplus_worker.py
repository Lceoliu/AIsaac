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
(behaviour: first hit, longest time without a hit, longest stay in one spot, cells, bombs used,
tears fired and tears that hit a lineage NPC). The hit-rate test (combat-hitrate) plays with an
invincible player (config 'invincible') and a longer deadline (config 'max_episode_frames').
combat-hitrate-fire frames also carry 'credit': the part of the step's reward that belongs to the
steps its tears were fired in (credit[j - 1]: j steps back), which the learner moves there.
config 'hurt_ends' (C37, user decision 2026-09-28): the first damage the player takes ends the episode
as a failure, outcome 'hurt', terminal (no bootstrap); evaluation (abplus_eval) keeps the game's rules.
The reward sees that step with outcome 'hurt' (combat-hitrate-miss hurt_rest charges the rest of the deadline).
config 'groups' (parallel task groups, abplus_groups.py): every episode plays one group's rooms, target and
deadline; config 'budget' and 'slot_groups' decide the group of each slot's next episode (GroupScheduler),
at the episode boundary. META 'group' is the group of the episode that stepped, 'reset_group' the next one's.
With PLR (config 'plr_levels': (group, room, arm) levels, plr.group_levels) each group draws its rooms by PLR
(GroupPlrChooser over that group's levels).
config 'room_buffer' (C39, room_buffer.py): the learner's per-group seed table in shared memory; at an episode boundary
the slot plays a buffer seed (replay) or its own next training seed (fresh) of the group the step budget chose, fresh
or replay by a second step budget per group (GroupScheduler over (fresh, replay), shares fresh_share and the rest).
An episode that is fresh only because the group's buffer is still empty stays outside that budget (the first rollout
of a run: counted, it made the next rollouts replay almost only, C39 smoke test). META 'replay' says whether the stepping
episode replays.
config 'frames_per_decision' (C39; default 2): the logic frames each action is held for. combat-hp frames
(C39) carry remaining_time (the deadline of the episode's group), which the learner reads as it is.

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
from .abplus_geometry import blocked_moves
from .abplus_groups import GroupScheduler, sample_group_hp
from .abplus_reward import (COMPONENTS, COMPONENTS_HP, COMPONENTS_HR, COMPONENTS_HRF, COMPONENTS_HRH, COMPONENTS_HRM,
                            COMPONENTS_HRM_REST, COMPONENTS_HRW, COMPONENTS_V3, COMPONENTS_V4, COMPONENTS_V5, CREDIT_STEPS,
                            REWARDS)
from .abplus_tasks import TASKS, Task, TaskSampler
from .abplus_options import GOTO_RADIUS, GOTO_SECONDS, GROUP_MODES, OptionSequence
from .abplus_reward import COMPONENTS_GOTO, GotoReward
from .transformer_obs import (BIG_TERRAIN, COMBAT_FIELDS, COMBAT_FIELDS_HITRATE, COMBAT_FIELDS_MISS, COMBAT_FIELDS_V4,
                              ENTITY_FLAGS, FACTORED_NVEC, GOAL_FIELDS, GOAL_TASKS, ROOM_CANVAS, SOURCES)
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
EPISODE_STATS = ('first_hit_s', 'longest_no_hit_s', 'longest_stationary_s', 'cells', 'bombs_used', 'shots',
                 'tear_hits', 'misses')
# Per-environment side channel: the episode that stepped, the episode that starts after a done,
# and timings for diagnostics.
META_DTYPE = np.dtype([
    ('seed', 'i8'), ('start', 'f4', (2,)), ('reset_seed', 'i8'), ('reset_start', 'f4', (2,)),
    ('step_ms', 'f4'), ('switch_wait_ms', 'f4'), ('reset_ms', 'f4'), ('errors', 'i4'), ('episodes', 'i4'),
    ('task', 'i4'), ('reset_task', 'i4'),
    ('components', 'f4', (max(len(COMPONENTS), len(COMPONENTS_V3), len(COMPONENTS_V4), len(COMPONENTS_V5),
                               len(COMPONENTS_HR), len(COMPONENTS_HRW), len(COMPONENTS_HRM), len(COMPONENTS_HRM_REST),
                               len(COMPONENTS_HRF), len(COMPONENTS_HRH), len(COMPONENTS_HP)),)),
    ('recycles', 'i4'), ('start_failures', 'i4'), ('recycle_deferrals', 'i4'), ('level', 'i4'), ('reset_level', 'i4'),
    ('bombs', 'i4'), ('reset_bombs', 'i4'), ('stats', 'f4', (len(EPISODE_STATS),)),
    ('group', 'i4'), ('reset_group', 'i4'),   # parallel task groups (abplus_groups), -1 without
    ('replay', 'i4'),   # C39: 1 when the stepping episode replays a Room Buffer seed
    # C39 t3r, written at the end of an episode: the resident memory of the process that played it, which instance
    # (0 a, 1 b) and its episodes since that process started; recycles started by the memory cap.
    ('rss_mib', 'f4'), ('instance', 'i4'), ('instance_episodes', 'i4'), ('memory_recycles', 'i4'),
    ('stats_mod', 'f4', (5,)),    # C41: the stepping episode's stat offsets (STAT_KEYS)
    # goal line: the stepping option's task (GOAL_TASKS) and sample source (SOURCES), and the next one's
    ('option', 'i4'), ('source', 'i4'), ('reset_option', 'i4'), ('reset_source', 'i4'),
    # C44 (Room Buffer for the goal groups), written when an option ends: whether its option sequence ended with it, whether
    # every option of the sequence succeeded (a plan cut short because no goal could be drawn counts as done), the half
    # hearts the sequence lost and its decisions
    ('seq_end', 'i4'), ('seq_ok', 'i4'), ('seq_hurt', 'f4'), ('seq_steps', 'i4')])
REWARD_PROFILES = ('combat-v1', 'combat-v2', 'combat-v3', 'combat-v4', 'combat-v5', 'combat-hitrate',
                   'combat-hitrate-walk', 'combat-hitrate-miss', 'combat-hitrate-fire', 'combat-hitrate-hurt',
                   'combat-hp', 'combat-hp2', 'combat-hp2-camera', 'goal-hp', 'goal-hp2', 'goal-hp3')
# Room state input per profile, and the profiles whose 120 s deadline is an observed termination
# (combat-v4/v5 and the hit-rate test truncate at the deadline instead and observe no elapsed time).
COMBAT_LAYOUTS = {'combat-v3': COMBAT_FIELDS, 'combat-v4': COMBAT_FIELDS_V4, 'combat-v5': COMBAT_FIELDS_V4,
                  'combat-hitrate': COMBAT_FIELDS_HITRATE, 'combat-hitrate-walk': COMBAT_FIELDS_HITRATE,
                  'combat-hitrate-miss': COMBAT_FIELDS_MISS, 'combat-hitrate-fire': COMBAT_FIELDS_MISS,
                  'combat-hitrate-hurt': COMBAT_FIELDS_MISS, 'combat-hp': COMBAT_FIELDS_V4, 'combat-hp2': COMBAT_FIELDS_V4,
                  'combat-hp2-camera': COMBAT_FIELDS_V4, 'goal-hp': COMBAT_FIELDS_V4, 'goal-hp2': COMBAT_FIELDS_V4,
                  'goal-hp3': COMBAT_FIELDS_V4}
# combat-hp (C39) observes the remaining time of its group's deadline, which ends the episode as a termination.
DEADLINE_PROFILES = ('combat-v1', 'combat-v2', 'combat-v3', 'combat-hp', 'combat-hp2', 'combat-hp2-camera', 'goal-hp',
                     'goal-hp2', 'goal-hp3')
# Frames that carry remaining_time (the deadline differs per group); for the others the learner derives it as
# 1 - time / 120 (gpu_env.decode_frame).
STORED_DEADLINE_PROFILES = ('combat-hp', 'combat-hp2', 'combat-hp2-camera', 'goal-hp', 'goal-hp2', 'goal-hp3')
# C41 (combat-hp2): rooms of every Basement shape: the terrain on transformer_obs.BIG_TERRAIN, positions in 1x1-room units.
BIG_ROOM_PROFILES = ('combat-hp2',)
# C41 as C39's continuation (combat-hp2-camera): the terrain is the camera's 15x9 window, positions in 1x1-room units; the
# frames and the network are combat-hp's.
CAMERA_PROFILES = ('combat-hp2-camera', 'goal-hp', 'goal-hp2', 'goal-hp3')
# Goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md): the observation's goal fields (camera view as base, so a room
# of another shape entered through a door still encodes), COMBAT options rewarded as combat-hp2, GOTO options by GotoReward.
GOAL_PROFILES = ('goal-hp', 'goal-hp2', 'goal-hp3')
# C44 (goal-hp2): goal-hp's observation with the player's position from the camera window (window_position) and the whole
# room on the 16 x 28 canvas (room_bits) for the model's full-room branch; a 1x1 room encodes as goal-hp plus room_bits.
ROOM_BITS_PROFILES = ('goal-hp2', 'goal-hp3')
# C45 (goal-hp3): goal-hp2 plus 'expert_move', the scripted A* expert's move (transformer_obs.STUCK_FRAMES) for the learner's
# imitation loss; not a network input, so a goal-hp2 checkpoint migrates without new parameters.
EXPERT_PROFILES = ('goal-hp3',)
GOAL_FIELDS_DTYPE = [('goal', 'f4', (GOAL_FIELDS,)), ('goal_map', 'f4', (2, 9, 15)), ('goal_distance', 'f4'),
                     ('source', 'f4')]
GOAL_KEYS = ('goal', 'goal_map', 'goal_distance', 'source')
# Group modes (a group's 'mode') and the option sequences: abplus_options (GROUP_MODES, OptionSequence).
TRUNCATING_PROFILES = ('combat-v4', 'combat-v5', 'combat-hitrate', 'combat-hitrate-walk', 'combat-hitrate-miss',
                       'combat-hitrate-fire', 'combat-hitrate-hurt')
# combat-v5 frames: firing geometry (entity flags, fire_distance, the auxiliary labels) and the
# factored previous action; 'hit' carries the step's hit reward for the learner's diagnostics.
# The hit-rate test keeps combat-v5's observation and heads; combat-hitrate-walk measures d_fire
# (the critic's fire_distance, the approach label) as the walking distance (WALK_PROFILES).
GEOMETRY_PROFILES = ('combat-v5', 'combat-hitrate', 'combat-hitrate-walk', 'combat-hitrate-miss', 'combat-hitrate-fire',
                     'combat-hitrate-hurt', 'combat-hp', 'combat-hp2', 'combat-hp2-camera', 'goal-hp', 'goal-hp2',
                     'goal-hp3')
WALK_PROFILES = ('combat-hitrate-walk', 'combat-hitrate-miss', 'combat-hitrate-fire', 'combat-hitrate-hurt', 'combat-hp',
                 'combat-hp2', 'combat-hp2-camera', 'goal-hp', 'goal-hp2', 'goal-hp3')
# combat-hitrate-fire: per step, the reward to move to the steps the tears were fired in (1..CREDIT_STEPS back).
CREDIT_PROFILES = ('combat-hitrate-fire',)
CREDIT_FIELDS = [('credit', 'f4', (CREDIT_STEPS,))]
GEOMETRY_FIELDS = [('entity_flags', 'f4', (256, len(ENTITY_FLAGS))), ('fire_distance', 'f4'), ('aim_label', 'f4'),
                   ('approach', 'f4', (9,)), ('hit', 'f4'),
                   # C30: the moves the terrain stops dead (abplus_geometry.blocked_moves), for --block-moves masks;
                   # not an observation.
                   ('move_block', 'f4', (9,))]
GEOMETRY_KEYS = ('entity_flags', 'fire_distance', 'aim_label', 'approach')
# append only: frames store the index. Goal line: goal (a GOTO option reached its goal), exit (left through a door on a
# position task), wrong_door.
OUTCOMES = ('running', 'death', 'win', 'time_limit', 'error', 'hurt', 'goal', 'exit', 'wrong_door')
FULL_START = (6.0, 1.0)  # player half-hearts, Boss HP fraction
MAX_EPISODE_FRAMES = 3600  # 120 s at 30 logic frames/s
RECYCLE_EPISODES = 200     # restart an AB+ process after this many episodes (memory leak; 0 = never)
RETRY_EPISODES = 50        # a deferred or failed recycle is tried again this many episodes later
RECYCLE_RSS_MIB = 0        # C39 t3r: also restart it once its resident memory reaches this many MiB (0 = never)
MEMORY_RETRY_EPISODES = 10  # a deferred or failed recycle for memory is tried again this many episodes later
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
    if profile in BIG_ROOM_PROFILES:   # C41: the terrain canvas of every room shape
        base = [('terrain', 'f4', (7, *BIG_TERRAIN)) if field[0] == 'terrain' else field for field in base]
    credit = CREDIT_FIELDS if profile in CREDIT_PROFILES else []
    deadline = [('remaining_time', 'f4')] if profile in STORED_DEADLINE_PROFILES else []
    goal = GOAL_FIELDS_DTYPE if profile in GOAL_PROFILES else []
    room = [('room_bits', 'f4', ROOM_CANVAS)] if profile in ROOM_BITS_PROFILES else []
    expert = [('expert_move', 'f4', (9,))] if profile in EXPERT_PROFILES else []
    keys = (FRAME_KEYS_COMBAT + GEOMETRY_KEYS + (GOAL_KEYS if goal else ()) + (('room_bits',) if room else ())
            + (('expert_move',) if expert else ()))
    return np.dtype(base + combat + GEOMETRY_FIELDS + credit + deadline + goal + room + expert), keys


def observation_options(profile):
    """VisibleHistory / AbplusTransformerEnv options of a reward profile."""
    geometry = profile in GEOMETRY_PROFILES
    return dict(deadline=profile in DEADLINE_PROFILES, combat_state=COMBAT_LAYOUTS.get(profile, False),
                **(dict(geometry='walk' if profile in WALK_PROFILES else True, factored_actions=True)
                   if geometry else {}),
                **(dict(terrain_shape=BIG_TERRAIN, room_scale='fixed') if profile in BIG_ROOM_PROFILES else {}),
                **(dict(room_scale='camera') if profile in CAMERA_PROFILES else {}),
                **(dict(goal=True) if profile in GOAL_PROFILES else {}),
                **(dict(window_position=True, room_bits=True) if profile in ROOM_BITS_PROFILES else {}),
                **(dict(expert=True) if profile in EXPERT_PROFILES else {}))


STAT_KEYS = ('speed', 'damage', 'shot_speed', 'tears', 'range')


def sample_stats(seed, spec):
    """C41 (user decisions 2026-09-29): the player's stat offsets of an episode (STAT_KEYS; tears in shots per second,
    range in Repentance units of 40 px), each uniform in [-a, a] for the spec's half widths {stat: a}; a pure function of
    the seed, so a replayed seed replays its stats. No spec: None (the base stats)."""
    if not spec:
        return None
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x57A75])
    widths = [float(spec.get(k, 0.0)) for k in STAT_KEYS]
    return tuple(float(rng.uniform(-a, a)) if a > 0 else 0.0 for a in widths)


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


class GroupPlrChooser:
    """TaskSampler's choose() over one parallel group's PLR levels ((group, room, arm), plr.group_levels): the
    learner's probabilities restricted to that group's levels."""

    def __init__(self, levels, probabilities, indices):
        self.levels, self.probabilities = [tuple(level) for level in levels], probabilities
        self.indices = np.asarray(indices, np.int64)
        if not len(self.indices):
            raise ValueError('a PLR group needs levels')

    def choose(self, seed, retry=0):
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x9E8, int(retry)])
        p = np.clip(np.asarray(self.probabilities, np.float64)[self.indices], 0.0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
        i = min(int(np.searchsorted(np.cumsum(p), rng.random(), side='right')), len(p) - 1)
        _, variant, arm = self.levels[int(self.indices[i])]
        return Task('normal', int(variant), int(rng.integers(4)), int(arm))


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
        self.shots0 = self.shots = self.fired(obs)
        self.hits0 = self.hits = int(obs['combat'].get('tear_hits', 0))
        self.misses0 = self.misses = int(obs['combat'].get('tear_misses', 0))

    @staticmethod
    def fired(obs):
        return int(obs.get('events', {}).get('tears', 0))

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
        self.shots = self.fired(obs)
        self.hits = int(obs['combat'].get('tear_hits', 0))
        self.misses = int(obs['combat'].get('tear_misses', 0))

    def array(self):
        return np.array([self.first_hit, max(self.no_hit, self.t - self.last_hit),
                         max(self.stationary, self.t - self.anchor_t), len(self.cells), self.bombs0 - self.bombs,
                         self.shots - self.shots0, self.hits - self.hits0, self.misses - self.misses0], np.float32)


def recycle_threshold(base, index):
    """Episodes an instance of slot `index` plays before its process is replaced: base .. 2 base - 1, spread over the slots
    (0: never)."""
    return base + (index * 61) % base if base > 0 else 0


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
        self.memory_retry_at = 0   # episodes of this process before which the memory cap starts no recycle
        self.launch()

    @property
    def name(self):
        return self.names[self.slot]

    @property
    def port(self):
        return self.ports[self.slot]

    def rss_mib(self):
        """Resident memory of the game process in MiB (0 when it cannot be read; run_instance.sh execs the game, so
        the pid is the game's)."""
        try:
            with open(f'/proc/{self.proc.pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except (AttributeError, OSError, ValueError):
            pass
        return 0.0

    def _env(self, port):
        frames = int(self.config.get('max_episode_frames', MAX_EPISODE_FRAMES))
        env = FrameEnv(port=port, max_episode_frames=frames, deadline_s=frames / 30,
                       frames_per_decision=int(self.config.get('frames_per_decision', 2)),
                       **observation_options(self.config.get('reward_profile')))
        env.bridge.binary_obs = self.config.get('binary_obs', True)
        env.bridge.lineage_mode = int(self.config.get('lineage_mode', env.bridge.lineage_mode))
        env.combat_multi_room = self.config.get('reward_profile') in GOAL_PROFILES   # goal line: COMBAT may leave its room
        env.bridge.invincible = bool(self.config.get('invincible', False))
        env.bridge.miss_cap = int(self.config.get('miss_cap', 0))
        env.bridge.target = self.config.get('target')   # single-enemy aiming arena (C22)
        spec = self.config.get('tasks')
        groups = self.config.get('groups')
        if groups and self.config.get('plr_probabilities') is not None:
            # Parallel task groups with PLR: each group draws its (room, arm) levels by PLR; use_group() before a reset.
            levels = [tuple(level) for level in self.config['plr_levels']]
            self.samplers = [GroupPlrChooser(levels, self.config['plr_probabilities'],
                                             [i for i, level in enumerate(levels) if level[0] == g['name']])
                             for g in groups]
            env.bridge.tasks, env.bridge.target = self.samplers[0], groups[0].get('target')
        elif groups:
            # Parallel task groups: each group's own mixture (target arms included); use_group() before a reset.
            self.samplers = [TaskSampler(g['spec']['weights'], g['spec']['normal'], g['spec']['boss'],
                                         (g.get('target') or {}).get('arms')) for g in groups]
            env.bridge.tasks, env.bridge.target = self.samplers[0], groups[0].get('target')
        elif self.config.get('plr_probabilities') is not None:
            env.bridge.tasks = PlrTaskChooser(self.config['plr_levels'], self.config['plr_probabilities'])
        else:
            # Tier 6 (C28): the target's arms choose the normal rooms (abplus_tasks.target_arm).
            arms = (self.config.get('target') or {}).get('arms')
            env.bridge.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss'], arms) if spec else None
        return env

    def launch(self):
        self.episodes = self.memory_retry_at = 0
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
        self.episodes = self.memory_retry_at = 0
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
            self.memory_retry_at = retry_at + MEMORY_RETRY_EPISODES
        elif self._replace():
            self.recycles += 1
        else:
            self.start_failures += 1
            self.episodes = retry_at
            self.memory_retry_at = retry_at + MEMORY_RETRY_EPISODES

    def relaunch(self):
        self.close()
        time.sleep(1.0)
        self.launch()

    def use_group(self, group):
        """The rooms, target and deadline of a parallel task group for the next reset."""
        g = self.config['groups'][group]
        self.env.bridge.tasks, self.env.bridge.target = self.samplers[group], g.get('target')
        self.env.max_episode_frames = int(g['frames'])
        self.env.history.deadline_s = float(g['seconds'])   # remaining_time (combat-hp)
        mode = g.get('mode', 'combat')   # goal line: how the group's episodes start
        self.env.bridge.reset_mode = {'chain': 'chain', 'goto_empty': 'goto_room'}.get(mode)
        if mode == 'chain':
            self.env.bridge.chain_rooms = frozenset(int(v) for v in g['spec']['normal'])

    def prepare(self, seed, start, bombs=1, group=-1, replay=0, recycle=False):
        """Reset for an episode in a background thread; recycle=True first replaces the process; group >= 0:
        an episode of that parallel task group; replay (C39): 1 a Room Buffer seed, 0 a fresh seed, -1 a fresh seed
        outside the fresh/replay budget (the buffer was empty)."""
        def run():
            try:
                if recycle:
                    self._recycle()
                if group >= 0:
                    self.use_group(group)
                self.episodes += 1
                t = time.perf_counter()
                stats = sample_stats(seed, self.config.get('stat_noise'))   # C41
                options = {'arena_seed': int(seed), 'start': start, 'bombs': int(bombs)}
                if stats is not None:
                    options['stats'] = stats
                frame, info = self.env.reset(options=options)
                self.result = (int(seed), tuple(start), int(bombs), frame, info, self.env.history.last_rows,
                               1000 * (time.perf_counter() - t), int(group), int(replay), stats)
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


class _OptionStepped(Exception):
    """Worker.step: the goal line's option_step wrote the step (skips the combat-only path, keeps the error handling)."""


# Shield in training (floor-clear mandate, A17): with ABP_SHIELD_TRAIN=1 a frame's move_block is the collision shield's
# dangerous moves (isaac_bridge.abplus_shield.danger on the last two observations), unless every move is dangerous; with
# --block-moves the sampler masks them, the same rule the floor runner's --shield applies at test time.
SHIELD_TRAIN = os.environ.get('ABP_SHIELD_TRAIN') == '1'


def write_frame(row, frame, reward, done, truncated, outcome, elapsed, layout, count, hit=0.0, credit=None, raw=None,
                prev=None):
    names = row.dtype.names
    for key in FRAME_KEYS + ('combat',) + GEOMETRY_KEYS + ('remaining_time',) + GOAL_KEYS + ('room_bits', 'expert_move'):
        if key in names:
            row[key] = frame[key]
    if 'hit' in names:
        row['hit'] = hit
    if 'move_block' in names:   # raw: the observation of this frame (None: nothing blocked)
        if SHIELD_TRAIN:
            from .abplus_shield import danger
            bad = danger(prev, raw) if raw is not None else [False] * 9
            row['move_block'] = 0.0 if all(bad) else np.asarray(bad, np.float32)
        else:
            row['move_block'] = blocked_moves(raw) if raw is not None else 0.0
    if 'credit' in names:   # every row, so a reset or error row never keeps the last step's credit
        row['credit'] = 0.0 if credit is None else credit
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
        # Staggered by slot (C39 t3 OOM, 09-29 03:06): the slots start together, so one threshold for all of them recycled
        # a dozen processes within one rollout, each replacement running next to the process it replaces.
        base = int(config.get('recycle_episodes', RECYCLE_EPISODES))
        self.recycle_after = recycle_threshold(base, index)
        self.recycle_rss = float(config.get('recycle_rss_mib', RECYCLE_RSS_MIB))
        self.profile = config.get('reward_profile', 'combat-v1')
        if self.profile not in REWARD_PROFILES:
            raise ValueError(f'unknown reward profile {self.profile!r}')
        # goal line: reward_options = {'combat': {...}, 'goto': {...}}; COMBAT options use combat-hp2, GOTO options GotoReward
        self.goal_line = self.profile in GOAL_PROFILES
        options = config.get('reward_options', {})
        if self.goal_line:
            self.reward = REWARDS[self.profile](**options.get('combat', {}))
            self.goto_reward = GotoReward(**options.get('goto', {}))
        else:
            self.reward = REWARDS[self.profile](**options) if self.profile in REWARDS else None
        self.seq, self.option = None, None
        self.hurt_ends = bool(config.get('hurt_ends', False))
        self.damage0 = 0.0   # combat.player_damage when the episode started
        # Parallel task groups: the group of the episode being played and its decisions so far.
        groups = config.get('groups')
        self.scheduler = (GroupScheduler([g['share'] for g in groups], config.get('budget', 'steps'),
                                         (config.get('slot_groups') or [0] * (index + 1))[index]) if groups else None)
        self.group, self.decisions = -1, 0
        self.stats_mod = np.zeros(len(STAT_KEYS), np.float32)   # C41
        self.replay = 0
        # C39 Room Buffer: seeds (G, K), start probabilities (G, K), (log only) (worker_main maps them); per group a step
        # budget over (fresh, replay).
        self.buffer = config.get('buffer_arrays')
        self.kinds = None
        if self.buffer is not None and groups:
            f = float(config['room_buffer']['fresh_share'])
            self.kinds = [GroupScheduler([f, 1.0 - f], 'steps') for _ in groups]

    def schedule(self, episode):
        """(seed, start, bombs, group, replay) of this slot's episode-th episode: the group from the step budget, then
        with a Room Buffer a replayed buffer seed (replay 1) or the slot's own fresh seed (0; -1 when the group's buffer is
        empty, outside the fresh/replay budget). A pure function of the fresh seed and the slot's budget."""
        seed = episode_seed(self.config, self.index, episode)
        group = self.scheduler.choose(seed) if self.scheduler is not None else -1
        replay = 0
        if self.buffer is not None and group >= 0:
            seeds, probs, _ = self.buffer
            p = np.clip(np.asarray(probs[group], np.float64), 0.0, None)
            if p.sum() <= 0:
                replay = -1
            elif self.kinds[group].choose(seed) == 1:
                rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0xB0FF])
                i = min(int(np.searchsorted(np.cumsum(p / p.sum()), rng.random(), side='right')), len(p) - 1)
                if int(seeds[group, i]) >= 0:
                    seed, replay = int(seeds[group, i]), 1
        return (seed, self.start_of(seed, group), sample_bombs(seed, self.config.get('start_bombs')), group, replay)

    def start_of(self, seed, group):
        """(player half hearts, boss HP fraction) of an episode: the run's start randomisation, the half hearts replaced by
        the group's start_hp when it has one (C45, abplus_groups.sample_group_hp)."""
        start = sample_start(seed, self.config['start_randomization'])
        hp = sample_group_hp(seed, self.group_spec(group).get('start_hp')) if group >= 0 else None
        return (hp, start[1]) if hp is not None else start

    def _begin(self, instance_index, result):
        seed, start, bombs, frame, info, rows, reset_ms, group, replay, stats = result
        self.stats_mod = np.asarray(stats if stats is not None else (0.0,) * len(STAT_KEYS), np.float32)
        self.active = instance_index
        self.group, self.decisions, self.replay = group, 0, replay
        if self.kinds is not None and group >= 0 and replay >= 0:
            self.kinds[group].start(replay)
        if self.scheduler is not None:
            self.scheduler.start(group)
            if self.reward is not None and hasattr(self.reward, 'deadline_s'):
                # hurt_rest and a death term charge the rest of this group's deadline
                self.reward.deadline_s = float(self.config['groups'][group]['seconds'])
        self.seed, self.start, self.bombs = seed, start, bombs
        self.layout = int(info.get('room_variant', info.get('arena', {}).get('variant', 0)))
        self.task = TASKS.index(info.get('task', 'arena'))
        if self.levels and group >= 0:   # parallel groups with PLR: (group, room, arm) levels
            self.level = self.levels.get((self.config['groups'][group]['name'], int(info.get('room_variant', -1)),
                                          int(info.get('target_arm', -1))), -1)
        else:
            self.level = level_index(self.levels, info)
        self.elapsed = 0
        self.last_frame = frame
        raw = self.instances[instance_index].env.raw_obs
        self.stats = EpisodeStats(raw)
        self.damage0 = float(raw['combat']['player_damage']) if 'combat' in raw else 0.0
        if self.goal_line:
            self.seq = OptionSequence(self.group_spec(group), seed, info, (self.reward, self.goto_reward))
            started = self.start_option(self.instances[instance_index].env, first=True)
            if started is None:
                raise RuntimeError(f'seed {seed}: the first option found no goal')
            frame, rows = started
            self.last_frame = frame
            return frame, rows, reset_ms
        if self.reward is not None:
            self.reward.reset(raw, TASKS[self.task])
        return frame, rows, reset_ms

    # ------------------------------------------------------------------ goal line: options
    def group_spec(self, group):
        return self.config['groups'][group] if group >= 0 else {}

    def option_plan(self, group, info):
        """The options of the sequence this reset started (abplus_options.option_plan)."""
        from .abplus_options import option_plan
        return option_plan(self.group_spec(group), info)

    def start_option(self, env, first=False, room_changed=False):
        """Start the sequence's current option in place (OptionSequence.start) and reset this worker's per-option state;
        None when a GOTO option finds no goal."""
        started = self.seq.start(env, first=first, room_changed=room_changed)
        if started is None:
            return None
        self.option = self.seq.option
        if 'variant' in self.option:
            self.layout = int(self.option['variant'])
        self.stats = EpisodeStats(env.raw_obs)
        self.elapsed = 0
        return started

    def option_codes(self):
        """(GOAL_TASKS index, SOURCES index) of the current option."""
        return GOAL_TASKS.index(self.option['task']), SOURCES.index(self.option['source'])

    def option_step(self, active, frame, info, terminated, truncated):
        """One decision of the current option: its outcome, reward and step row. When the option ended with a win or a
        reached goal and the plan goes on, the next option starts in place (its first frame in the reset row) and False
        is returned; True when the sequence ended (the caller switches to the standby instance)."""
        raw = active.env.raw_obs
        option = self.option
        outcome = self.seq.outcome(raw, info['outcome'])
        elapsed = int(info['elapsed_frames'])
        self.stats.step(raw, elapsed - self.elapsed)
        reward, parts, hit = self.seq.step(raw, outcome, elapsed - self.elapsed)
        done = bool(terminated or truncated)
        write_frame(self.step_row, frame, reward, done, False, OUTCOMES.index(outcome), elapsed, self.layout,
                    active.env.history.last_rows, hit, None, raw=raw, prev=getattr(active.env.history, 'before_previous', None))
        self.last_frame = frame
        self.elapsed = elapsed
        if not done:
            return False
        totals = self.seq.totals()
        self.meta['components'] = np.pad(totals, (0, len(self.meta['components']) - len(totals)))
        self.meta['stats'] = self.stats.array()
        success = outcome in ('win', 'goal')
        self.meta['seq_end'], self.meta['seq_ok'] = 1, int(success and self.seq.index + 1 >= len(self.seq.plan))
        self.meta['seq_hurt'] = max(0.0, float(raw['combat']['player_damage']) - self.damage0)
        self.meta['seq_steps'] = self.decisions
        if self.seq.next(outcome, raw['room']['room_idx']):
            started = self.start_option(active.env, room_changed=option['task'] == 'goto_door')
            if started is None:
                self.meta['seq_ok'] = 1   # the plan went on but no goal could be drawn: every option played succeeded
            else:
                self.meta['seq_end'] = 0
                first, rows = started
                write_frame(self.reset_row, first, 0.0, False, False, 0, 0, self.layout, rows, raw=raw)
                self.last_frame = first
                self.meta['reset_seed'], self.meta['reset_start'] = self.seed, self.start
                self.meta['reset_task'], self.meta['reset_level'] = self.task, self.level
                self.meta['reset_bombs'], self.meta['reset_group'] = self.bombs, self.group
                self.meta['reset_option'], self.meta['reset_source'] = self.option_codes()
                self.meta['reset_ms'] = 0.0
                return False
        return True

    def option_outcome(self, raw, outcome):
        return self.seq.outcome(raw, outcome)

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
        if self.scheduler is not None:
            self.scheduler.clear_inflight()   # the episodes prepared for the old generation are dropped
        for kinds in self.kinds or ():
            kinds.clear_inflight()
        group = self.scheduler.choose(seed) if self.scheduler is not None else -1
        start = self.start_of(seed, group) if start is None else start
        # A generation's first episode is the learner's fresh seed (outside the budget while the group's buffer is empty).
        empty = self.buffer is not None and group >= 0 and not np.asarray(self.buffer[1][group]).sum() > 0
        first.prepare(seed, start, sample_bombs(seed, self.config.get('start_bombs')), group, -1 if empty else 0)
        frame, rows, reset_ms = self._begin(0, first.take())
        write_frame(self.step_row, frame, 0.0, False, False, 0, 0, self.layout, rows, raw=first.env.raw_obs)
        self.instances[1].prepare(*self.schedule(1))
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        self.meta['level'] = self.level
        self.meta['bombs'] = self.bombs
        self.meta['group'] = self.meta['reset_group'] = self.group
        self.meta['replay'] = int(self.replay == 1)
        self.meta['stats_mod'] = self.stats_mod
        self.meta['reset_ms'] = reset_ms
        if self.goal_line:
            self.meta['option'], self.meta['source'] = self.option_codes()
            self.meta['reset_option'], self.meta['reset_source'] = self.meta['option'], self.meta['source']
        self.meta['episodes'] = 0

    def step(self, joint, bomb, item):
        active = self.instances[self.active]
        t = time.perf_counter()
        self.meta['seed'] = self.seed
        self.meta['start'] = self.start
        self.meta['task'] = self.task
        self.meta['level'] = self.level
        self.meta['bombs'] = self.bombs
        self.meta['group'] = self.group
        self.meta['replay'] = int(self.replay == 1)
        self.meta['stats_mod'] = self.stats_mod
        if self.goal_line:
            self.meta['option'], self.meta['source'] = self.option_codes()
        self.decisions += 1
        try:
            # The learner masks the bomb from the same frame; a mismatch only drops the bomb.
            if bomb and not active.env.action_masks()[46]:
                bomb = 0
            frame, reward, terminated, truncated, info = active.env.step(np.array([joint, bomb, item]))
            if self.goal_line:
                done = self.option_step(active, frame, info, terminated, truncated)
                raise _OptionStepped
            if (self.hurt_ends and info['outcome'] == 'running'
                    and float(active.env.raw_obs['combat']['player_damage']) > self.damage0):
                # C37: the first damage ends the episode as a failure (terminal): the rest of the room's
                # rewards are lost; the reward charges the rest of the deadline only with hurt_rest.
                # The instance is reset for a later episode.
                info = {**info, 'outcome': 'hurt'}
                terminated, truncated = True, False
            outcome = OUTCOMES.index(info['outcome'])
            elapsed = int(info['elapsed_frames'])
            self.stats.step(active.env.raw_obs, elapsed - self.elapsed)
            hit, credit = 0.0, None
            if self.reward is not None:
                parts = self.reward.step(active.env.raw_obs, info['outcome'], elapsed - self.elapsed)
                reward = self.reward.scale * float(sum(parts.values()))
                hit = float(parts.get('hit', parts.get('damage', 0.0)))   # combat-hp: the monster damage
                if hasattr(self.reward, 'credit'):
                    credit = self.reward.scale * self.reward.credit
            else:
                reward = float(combat_v1_reward(reward, outcome, elapsed))
            done = bool(terminated or truncated)
            # combat-v1..v3: the 120 s deadline is part of the task, so it terminates without bootstrap;
            # combat-v4 truncates there (the learner bootstraps the value of the last frame).
            cut = self.profile in TRUNCATING_PROFILES and info['outcome'] == 'time_limit'
            write_frame(self.step_row, frame, reward, done, cut, outcome, elapsed, self.layout,
                        active.env.history.last_rows, hit, credit, raw=active.env.raw_obs,
                        prev=getattr(active.env.history, 'before_previous', None))
            self.last_frame = frame
            self.elapsed = elapsed
        except _OptionStepped:
            pass
        except Exception:
            # Engine or bridge failure: end the episode as a truncation (value bootstrap) on the
            # last good frame, replace the process, continue with the standby instance.
            self.meta['errors'] += 1
            print(f'[worker {self.index}] {active.name} failed:\n{traceback.format_exc()}', flush=True)
            write_frame(self.step_row, self.last_frame, 0.0, True, True, OUTCOMES.index('error'),
                        self.elapsed, self.layout, int(self.last_frame['entity_mask'].sum()))
            self.meta['seq_end'] = 0   # C44: an engine failure says nothing about the seed (the Room Buffer skips it)
            done = True
            active.relaunch()
        self.meta['step_ms'] = 1000 * (time.perf_counter() - t)
        if done:
            if self.scheduler is not None:
                self.scheduler.finish(self.group, self.decisions)
            if self.kinds is not None and self.group >= 0 and self.replay >= 0:
                self.kinds[self.group].finish(self.replay, self.decisions)
            if self.reward is not None and not self.goal_line:   # the goal line wrote its option's totals
                totals = self.reward.totals_array()
                self.meta['components'] = np.pad(totals, (0, len(self.meta['components']) - len(totals)))
                self.meta['stats'] = self.stats.array()
            elif not self.goal_line:
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
            write_frame(self.reset_row, frame, 0.0, False, False, 0, 0, self.layout, rows,
                        raw=self.instances[standby_index].env.raw_obs)
            self.meta['reset_seed'] = self.seed
            self.meta['reset_start'] = self.start
            self.meta['reset_task'] = self.task
            self.meta['reset_level'] = self.level
            self.meta['reset_bombs'] = self.bombs
            self.meta['reset_group'] = self.group
            self.meta['reset_ms'] = reset_ms
            if self.goal_line:
                self.meta['reset_option'], self.meta['reset_source'] = self.option_codes()
            self.meta['episodes'] = self.episode
            self.meta['recycles'] = sum(instance.recycles for instance in self.instances)
            self.meta['start_failures'] = sum(instance.start_failures for instance in self.instances)
            self.meta['recycle_deferrals'] = sum(instance.recycle_deferrals for instance in self.instances)
            # The instance that just finished prepares the episode after the one now starting,
            # restarting its process first when it has played recycle_after episodes.
            finished = self.instances[old]
            recycle = bool(self.recycle_after) and finished.episodes >= self.recycle_after
            # C39 t3r (09-29 14:59, out of memory on 31 GB): some processes grew by 100-200 MiB a minute while others
            # stayed at ~290 MiB, so a cap on resident memory bounds each process whatever the cause.
            rss = finished.rss_mib()
            self.meta['rss_mib'], self.meta['instance'], self.meta['instance_episodes'] = rss, old, finished.episodes
            if (not recycle and self.recycle_rss > 0 and rss >= self.recycle_rss
                    and finished.episodes >= finished.memory_retry_at):
                recycle = True
                self.meta['memory_recycles'] += 1
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
    frame_dtype, _ = frame_layout(config.get('reward_profile'))
    frames = np.ndarray((2, n), dtype=frame_dtype, buffer=shm.buf)
    meta = np.ndarray((n,), dtype=META_DTYPE, buffer=shm.buf, offset=2 * n * frame_dtype.itemsize)
    buffer_shm = None
    if config.get('room_buffer'):
        # C39: the learner rewrites the Room Buffer table after every rollout.
        spec = config['room_buffer']
        g, k = int(spec['groups']), int(spec['capacity'])
        buffer_shm = shared_memory.SharedMemory(name=spec['shm'])
        config = {**config, 'buffer_arrays': (
            np.ndarray((g, k), np.int64, buffer=buffer_shm.buf),
            np.ndarray((g, k), np.float64, buffer=buffer_shm.buf, offset=8 * g * k),
            np.ndarray((g,), np.float64, buffer=buffer_shm.buf, offset=16 * g * k))}
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
        config.pop('buffer_arrays', None)
        if worker is not None:   # its reset() copies the config: drop every view of the buffer table
            worker.buffer = None
            worker.config.pop('buffer_arrays', None)
        shm.close()
        if plr_shm is not None:
            plr_shm.close()
        if buffer_shm is not None:
            try:
                buffer_shm.close()
            except BufferError:   # a view still held somewhere: the process ends anyway
                pass
