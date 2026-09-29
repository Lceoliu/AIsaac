"""Player-visible, raw observation windows. No learned features are cached.

The fixed Gym allocation is a capacity, not a top-k selector. Overflow raises;
the network compacts valid entities before attention. Each PPO sample contains
its entire causal context, so shuffled minibatches cannot mix episode memories.
"""
from collections import deque
import math

import numpy as np
from gymnasium import spaces

from .abplus_geometry import CELL, D_FIRE_MAX, MOVES, fire_geometry
from .hit_rate import HitRate
from .monstro_gym import MonstroGymEnv

SCHEMA = 'monstro-transformer-v3'
DEADLINE_SCHEMA = 'monstro-transformer-v4-deadline'
HISTORY = 64
ENTITY_CAPACITY = 256
ANIMATION_BYTES = 32
# combat-v3 room state (VisibleHistory(combat_state=True), AB+ bridge abp-0.2.2): what the reward's
# time price and first-minute penalty depend on beyond the remaining time.
COMBAT_FIELDS = ('blocking_count', 'engaged', 'blocking_fraction', 'since_hit')
# combat-v4 has no stall or time penalty, so it keeps only the room aggregates.
COMBAT_FIELDS_V4 = ('blocking_count', 'blocking_fraction')
# The hit-rate test (combat-hitrate) prices time by the running tear hit rate (hit_rate.HitRate).
COMBAT_FIELDS_HITRATE = ('blocking_count', 'blocking_fraction', 'hit_rate')
# combat-hitrate-miss adds the current run of misses (bridge abp-0.2.5 miss_streak): the next miss costs
# its length + 1 times the miss price.
COMBAT_FIELDS_MISS = COMBAT_FIELDS_HITRATE + ('miss_streak',)
# combat-v5 (VisibleHistory(geometry=True, factored_actions=True)): per-NPC flags from bridge
# abp-0.2.3, the critic's d_fire input (fire_distance = d_fire / 40, at most 15) and the heads the
# 45-way joint action was split into.
ENTITY_FLAGS = ('lineage', 'blocking')
FIRE_DISTANCE_MAX = D_FIRE_MAX / CELL
FACTORED_NVEC = (9, 5, 2, 2)
# C41 (combat-hp2): the terrain canvas of the largest Basement room grid (2x2: 28 x 16 cells; a smaller room fills its
# top-left corner) and the interior of a 1x1 room in pixels (top_left (60, 140), bottom_right (580, 420)): positions,
# velocities and sizes in those units keep a 1x1 room's values and let a larger room reach about 2.
BIG_TERRAIN = (16, 28)
ROOM_1X1 = (520.0, 280.0)
# C41 (combat-hp2-camera, the continuation of C39): a 1x1 room's grid is the terrain view; in a larger room the view is
# the window of that size around the player, stopping at the room's edges, as the game's camera does.
CAMERA_VIEW = (9, 15)
PLAYER_FIELDS = ('x', 'y', 'vx', 'vy', 'motion_valid', 'size', 'hearts', 'max_hearts',
                 'soul', 'bombs', 'keys', 'coins', 'damage', 'speed', 'shot_speed',
                 'fire_delay_max', 'range', 'can_fly', 'active_charge', 'active_ready',
                 'animation_frame', 'flip', 'dt')
ENTITY_FIELDS = ('dx', 'dy', 'vx', 'vy', 'motion_valid', 'size', 'size_x', 'size_y',
                 'height', 'scale', 'collision', 'grid_collision', 'damage', 'enemy',
                 'boss', 'boss_hp', 'has_boss_hp', 'animation_frame', 'flip',
                 'end_dx', 'end_dy', 'angle_sin', 'angle_cos', 'laser_length', 'ring_radius', 'circle',
                 'airborne', 'body_visible', 'shadow_dx', 'shadow_dy', 'shadow_valid')


def monstro_visual(entity, player_pos, width, height):
    """Features derived from *current* ANM2 only; never NPC.State/TargetPosition.

    The Monstro shadow layer has (0,0) offset at every frame. Ground Position
    is therefore the visible shadow position, including while the body is offscreen.
    """
    if (entity['type'], entity['variant']) != (20, 0):
        return [0, 1, 0, 0, 0]
    anim, frame = entity.get('anim', ''), entity['aframe']  # AB+: no anim when no known name matched
    airborne = ((anim == 'Walk' and 7 <= frame < 23) or
                (anim == 'JumpUp' and frame >= 10) or (anim == 'JumpDown' and frame < 33))
    body = not ((anim == 'JumpUp' and frame >= 11) or (anim == 'JumpDown' and frame < 29))
    x,y = entity['pos']
    return [airborne, body, (x-player_pos[0])/width, (y-player_pos[1])/height, 1]


def animation_bytes(name):
    data = (name or '').encode('utf-8')
    if len(data) > ANIMATION_BYTES:
        raise ValueError(f'Animation name exceeds {ANIMATION_BYTES} bytes: {name!r}')
    result = np.zeros(ANIMATION_BYTES, np.uint8)
    result[:len(data)] = list(data)
    return result


def terrain_channels(obs, shape=(9, 15)):
    """The room's grid as 7 channels on a (rows, cols) canvas: (9, 15), a 1x1 room's grid, requires one; a larger canvas
    (C41: BIG_TERRAIN) takes any room that fits, from the top-left corner, the rest zero (outside)."""
    terrain = obs['terrain']
    height, width = terrain['height'], terrain['width']
    if tuple(shape) == (9, 15) and (height, width) != (9, 15):
        raise ValueError('Transformer curriculum requires a 15x9 room')
    if height > shape[0] or width > shape[1]:
        raise ValueError(f'room grid {width}x{height} exceeds the terrain canvas {shape[1]}x{shape[0]}')
    result = np.zeros((7, *shape), np.float32)
    positions = []
    for index, x, y, collision, inside, walkable, pit, hazard, solid, destructible in terrain['cells']:
        row, col = divmod(index, width)
        result[:, row, col] = (inside, walkable, solid, pit, destructible, hazard, 0)
        positions.append((index, x, y))
    for door in obs['doors']:
        if not door['open']:
            index, _, _ = min(positions, key=lambda c: (c[1]-door['pos'][0])**2 + (c[2]-door['pos'][1])**2)
            row, col = divmod(index, width)
            result[6, row, col] = 1
    return result


def camera_origin(obs, height, width, view=CAMERA_VIEW):
    """(row, col) of the top-left cell of the view-sized window of a height x width room grid that follows the player:
    centred on the player's cell, stopping at the room's edges; (0, 0) in a room of the view's size."""
    rows, cols = view
    x0 = y0 = None
    for index, x, y, *_ in obs['terrain']['cells']:
        if index == 0:
            x0, y0 = x, y
            break
    px, py = obs['players'][0]['pos']
    row, col = int(round((py - y0) / CELL)), int(round((px - x0) / CELL))
    return min(max(row - rows // 2, 0), height - rows), min(max(col - cols // 2, 0), width - cols)


def entity_key(entity):
    # Index is for tracking only. Age continuity rejects reused indices after respawn.
    return entity['id'], entity['type'], entity['variant'], entity['subtype']


def factored_to_joint(actions):
    """Policy actions (move, shoot, bomb, item) -> the bridge's (move*5 + shoot, bomb, item)."""
    a = np.asarray(actions)
    return np.stack([a[..., 0] * 5 + a[..., 1], a[..., 2], a[..., 3]], -1)


def factored_masks(joint_mask):
    """TransformerMonstroEnv.action_masks (45 joint + bomb + item) -> the factored heads' masks."""
    m = np.asarray(joint_mask, bool)
    return np.concatenate([np.ones(9 + 5, bool), m[45:47], m[47:49]])


class VisibleHistory:
    def __init__(self, history=HISTORY, capacity=ENTITY_CAPACITY,deadline=False,combat_state=False,
                 geometry=False,factored_actions=False,deadline_s=120.0,terrain_shape=(9, 15),room_scale='room'):
        self.deadline=deadline
        # C41: the terrain canvas and the units of positions, velocities and sizes. 'room' divides by this room's width
        # and height (every earlier run); 'fixed' by a 1x1 room's (ROOM_1X1), the terrain on the canvas (combat-hp2);
        # 'camera' by a 1x1 room's, the terrain the CAMERA_VIEW window around the player (combat-hp2-camera: a 1x1 room
        # encodes exactly as with 'room', so C39's network continues).
        self.terrain_shape=tuple(int(v) for v in terrain_shape)
        if room_scale not in ('room','fixed','camera'):
            raise ValueError(f'room_scale {room_scale!r}')
        if room_scale=='camera' and self.terrain_shape!=CAMERA_VIEW:
            raise ValueError(f'the camera view is {CAMERA_VIEW}, not {self.terrain_shape}')
        self.room_scale=room_scale
        # remaining_time = 1 - elapsed / deadline_s (combat-v1..v3: the 120 s task; combat-hp: each group's deadline,
        # set per episode by the worker, C39).
        self.deadline_s=float(deadline_s)
        # combat_state: False, True (COMBAT_FIELDS) or the tuple of COMBAT_FIELDS to observe.
        self.combat_fields=(COMBAT_FIELDS if combat_state is True else tuple(combat_state)) if combat_state else ()
        self.combat_state=bool(self.combat_fields)
        self.combat=None
        # combat-v5: per-NPC lineage/blocking flags (bridge abp-0.2.3), d_fire / 40 for the critic,
        # and the auxiliary head's labels (abplus_geometry.fire_geometry; geometry='walk' measures
        # d_fire as the walking distance); the previous action as the one-hot of the factored heads
        # (move 9, shoot 5, bomb 2, item 2).
        self.geometry, self.factored_actions = geometry, factored_actions
        self.d_fire = None
        self.history, self.capacity = history, capacity
        self.frames = deque(maxlen=history)
        self.previous = None
        self.origin = None
        self.previous_action = np.zeros(sum(FACTORED_NVEC) if factored_actions else 4, np.float32)
        h, n = history, capacity
        self.space = spaces.Dict({
            'player': spaces.Box(-np.inf, np.inf, (h, len(PLAYER_FIELDS)), np.float32),
            # int32 categories, not uint8 images: SB3 must not transpose [time, entity, name].
            'player_anim': spaces.Box(0, 255, (h, ANIMATION_BYTES), np.int32),
            'active_kind': spaces.Box(0, 4095, (h, 1), np.int32),
            'entities': spaces.Box(-np.inf, np.inf, (h, n, len(ENTITY_FIELDS)), np.float32),
            'entity_kind': spaces.Box(0, 65535, (h, n, 3), np.int32),
            'entity_anim': spaces.Box(0, 255, (h, n, ANIMATION_BYTES), np.int32),
            'entity_mask': spaces.Box(0, 1, (h, n), np.float32),
            'terrain': spaces.Box(0, 1, (h, 7, *self.terrain_shape), np.float32),
            'previous_action': spaces.Box(0, 44, (h, 4), np.float32),
            'time': spaces.Box(0, np.inf, (h,), np.float32),
            'history_mask': spaces.Box(0, 1, (h,), np.float32),
        })
        if deadline:self.space.spaces['remaining_time']=spaces.Box(0,1,(h,),np.float32)
        # Doors-blocking NPCs alive / 10, whether any of them lost HP yet this episode, their HP as
        # a fraction of the start, seconds since they last lost HP (or since the start) / 60.
        if self.combat_state:self.space.spaces['combat']=spaces.Box(0,np.inf,(h,len(self.combat_fields)),np.float32)
        if factored_actions:self.space.spaces['previous_action']=spaces.Box(0,1,(h,sum(FACTORED_NVEC)),np.float32)
        if geometry:
            self.space.spaces['entity_flags']=spaces.Box(0,1,(h,n,len(ENTITY_FLAGS)),np.float32)
            self.space.spaces['fire_distance']=spaces.Box(0,FIRE_DISTANCE_MAX,(h,),np.float32)
            self.space.spaces['aim_label']=spaces.Box(0,4,(h,),np.float32)
            self.space.spaces['approach']=spaces.Box(0,1,(h,len(MOVES)),np.float32)

    def clear(self):
        self.frames.clear()
        self.previous = None
        self.origin = None
        self.combat = None
        self.d_fire = None
        self.previous_action[:] = 0

    def set_previous_action(self, joint, bomb, item):
        """The action just sent to the bridge, in this history's previous_action format."""
        if self.factored_actions:
            move, shoot = divmod(int(joint), 5)
            self.previous_action[:] = 0
            for offset, value in zip((0, 9, 14, 16), (move, shoot, int(bomb), int(item))):
                self.previous_action[offset + value] = 1
        else:
            self.previous_action[:] = joint, bomb, item, 1

    def encode(self, obs):
        """Append obs to the history and return only its frame (window row, no padding copy).

        The GPU learner keeps the window on the device, so rollout workers need just this.
        """
        # Shared sim/native contract: cosmetic player/tear/blood-shot animation
        # is not calibrated in the simulator. Exclude it on BOTH backends,
        # rather than let the policy distinguish domains from placeholder art.
        obs = {**obs, 'players': [{**p, 'anim': '', 'aframe': 0, 'flip': False,
                                  'active_charge': p['active_charge'] if p['active'] else 0}
                                  for p in obs['players']],
               'entities': [e if e['type'] in (20, 4) else
                            {**e, 'anim': '', 'aframe': 0, 'flip': False}
                            for e in obs['entities']]}
        if obs.get('combat_schema') != 3:
            raise ValueError('Transformer requires bridge combat_schema=3; deploy the matching Mod')
        if self.previous is None:
            self.origin = obs['logic_frames']
        p = obs['players'][0]
        left, top = obs['room']['top_left']
        right, bottom = obs['room']['bottom_right']
        width, height = (right-left, bottom-top) if self.room_scale == 'room' else ROOM_1X1
        dt = obs['logic_frames']-self.previous['logic_frames'] if self.previous is not None else 0
        if self.previous is not None and dt <= 0:
            raise ValueError('Observation time must advance; clear history on reset')
        pv = (np.asarray(p['pos'])-self.previous['players'][0]['pos']) / dt if dt else np.zeros(2)
        frame = {k: np.zeros(v.shape[1:], dtype=v.dtype) for k, v in self.space.spaces.items()}
        frame['player'][:] = ((p['pos'][0]-left)/width, (p['pos'][1]-top)/height,
            pv[0]/width, pv[1]/height, bool(dt), p['size']/width, p['hearts']/6,
            p['max_hearts']/6, p['soul']/6, p['bombs']/10, p['keys']/10, p['coins']/100,
            p['damage']/10, p['speed'], p['shot_speed'], p['fire_delay_max']/30,
            p['range']/width, p['can_fly'], p['active_charge']/12, p['active_ready'],
            p['aframe']/60, p['flip'], dt/30)
        frame['player_anim'] = animation_bytes(p['anim'])
        frame['active_kind'][0] = p['active']
        previous = {entity_key(e): e for e in self.previous['entities']} if dt else {}
        rows = []
        for e in obs['entities']:
            old = previous.get(entity_key(e))
            valid = old is not None and e['age'] == old['age'] + dt
            velocity = (np.asarray(e['pos'])-old['pos'])/dt if valid else np.zeros(2)
            # Curved lasers become their observed polyline segments, never just a chord.
            laser = e.get('laser')
            if laser and not laser['circle']:
                points = laser['samples']
                if len(points) < 2:
                    points = [e['pos'], laser['end']]
                segments = list(zip(points[:-1], points[1:]))
            else:
                segments = [(e['pos'], e['pos'])]
            for start, end in segments:
                dx, dy = end[0]-start[0], end[1]-start[1]
                angle = math.atan2(dy, dx) if laser else 0
                row = [(start[0]-p['pos'][0])/width, (start[1]-p['pos'][1])/height,
                    velocity[0]/width, velocity[1]/height, valid, e['size']/width,
                    *e['size_multi'], e.get('height', 0)/height, e.get('scale', 1),
                    e['coll']/5, e['gcoll']/7, e['cdmg']/10, bool(e.get('enemy')),
                    bool(e.get('boss')), e.get('boss_hp', 0), 'boss_hp' in e,
                    e['aframe']/60, e['flip'],
                    (end[0]-p['pos'][0])/width if laser else 0,
                    (end[1]-p['pos'][1])/height if laser else 0,
                    math.sin(angle) if laser else 0, math.cos(angle) if laser else 0,
                    math.hypot(dx, dy)/width if laser else 0,
                    laser['radius']/width if laser and laser['circle'] else 0,
                    bool(laser and laser['circle'])]
                row.extend(monstro_visual(e,p['pos'],width,height))
                flags = [float(bool(e.get(k))) for k in ENTITY_FLAGS]
                rows.append((row, (e['type'], e['variant'], e['subtype']), animation_bytes(e.get('anim', '')), flags))
        if len(rows) > self.capacity:
            raise ValueError(f'Visible entity/laser segment overflow: {len(rows)} > {self.capacity}; increase capacity, never truncate')
        for i, (row, kind, anim, flags) in enumerate(rows):
            frame['entities'][i] = row
            frame['entity_kind'][i] = kind
            frame['entity_anim'][i] = anim
            frame['entity_mask'][i] = 1
            if self.geometry:
                frame['entity_flags'][i] = flags
        if self.geometry:
            # geometry='walk': the walking distance to a firing position (combat-hitrate-walk).
            g = fire_geometry(obs, self.d_fire, walk=self.geometry == 'walk')
            self.d_fire = g['d_fire']
            frame['fire_distance'] = np.float32(min(g['d_fire'], D_FIRE_MAX) / CELL)
            frame['aim_label'] = np.float32(g['aim'])
            frame['approach'][:] = g['approach']
        # Binary observations (abplus_obs) mark unchanged terrain with a version: reuse its channels. The camera view
        # keeps the whole room's channels and takes the window around the player every frame.
        camera = self.room_scale == 'camera'
        shape = (obs['terrain']['height'], obs['terrain']['width']) if camera else self.terrain_shape
        version = obs['terrain'].get('version')
        if version is None:
            channels = terrain_channels(obs, shape)
        else:
            key = (version, tuple((d['pos'][0], d['pos'][1], bool(d['open'])) for d in obs['doors']))
            if key != getattr(self, '_terrain_key', None):
                self._terrain_key, self._terrain_value = key, terrain_channels(obs, shape)
            channels = self._terrain_value
        if camera:
            row, col = camera_origin(obs, *shape)
            channels = channels[:, row:row + CAMERA_VIEW[0], col:col + CAMERA_VIEW[1]]
        frame['terrain'] = channels
        frame['previous_action'] = self.previous_action.copy()
        frame['time'] = np.float32((obs['logic_frames']-self.origin)/30)
        if self.deadline:frame['remaining_time']=np.float32(max(0,1-frame['time']/self.deadline_s))
        if self.combat_state:frame['combat'][:]=self.combat_features(obs['combat'],float(frame['time']),obs)
        frame['history_mask'] = np.float32(1)
        self.frames.append(frame)
        self.previous = obs
        self.last_rows = len(rows)
        return frame

    def combat_features(self, combat, t, obs=None):
        """COMBAT_FIELDS; 'engaged' and the last hit follow combat-v3's definition (blocking HP fell);
        'hit_rate' is the running tear hit rate the hit-rate test's time price uses (obs needed)."""
        hp, count = float(combat['blocking_hp']), float(combat['blocking_count'])
        rated = 'hit_rate' in self.combat_fields
        if self.combat is None:
            self.combat = dict(start=hp, last=hp, engaged=False, hit=0.0, rate=0.0)
            if rated:
                self.hit_rate = HitRate()
                self.hit_rate.reset(obs)
        elif rated:
            self.combat['rate'] = self.hit_rate.update(obs)
        c = self.combat
        if hp < c['last'] - 1e-9:
            c['engaged'], c['hit'] = True, t
        c['last'] = hp
        values = dict(blocking_count=min(count, 30.0) / 10, engaged=float(c['engaged']),
                      blocking_fraction=min(hp / max(1.0, c['start']), 3.0), since_hit=min(t - c['hit'], 120.0) / 60,
                      hit_rate=c['rate'], miss_streak=min(float(combat.get('miss_streak', 0)), 100.0) / 20)
        return tuple(values[k] for k in self.combat_fields)

    def append(self, obs):
        self.encode(obs)
        result = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.space.spaces.items()}
        # Right padding: causal attention always has a real first key, even at episode start.
        for i, record in enumerate(self.frames):
            for key in result:
                result[key][i] = record[key]
        return result


class TransformerMonstroEnv(MonstroGymEnv):
    def __init__(self, *args, history=HISTORY, entity_capacity=ENTITY_CAPACITY, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge.action_repeat = 2
        self.history = VisibleHistory(history, entity_capacity)
        self.observation_space = self.history.space
        self.action_space = spaces.MultiDiscrete([45, 2, 2])

    def reset(self, *, seed=None, options=None):
        self.history.clear()
        return super().reset(seed=seed, options=options)

    def encode_observation(self, obs):
        return self.history.append(obs)

    def action_masks(self):
        p = self.raw_obs['players'][0]
        return np.asarray([True]*45 + [True, p['bombs'] > 0] + [True, p['active_ready']], bool)

    def decode_action(self, action):
        joint, bomb, item = (int(v) for v in action)
        mask = self.action_masks()
        if not mask[45+bomb] or not mask[47+item]:
            raise ValueError('Requested unavailable bomb/active item')
        self.history.set_previous_action(joint, bomb, item)
        move, shoot = divmod(joint, 5)
        return [move, shoot, bomb, item]
