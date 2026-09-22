"""Player-visible, raw observation windows. No learned features are cached.

The fixed Gym allocation is a capacity, not a top-k selector. Overflow raises;
the network compacts valid entities before attention. Each PPO sample contains
its entire causal context, so shuffled minibatches cannot mix episode memories.
"""
from collections import deque
import math

import numpy as np
from gymnasium import spaces

from .monstro_gym import MonstroGymEnv

SCHEMA = 'monstro-transformer-v3'
HISTORY = 64
ENTITY_CAPACITY = 256
ANIMATION_BYTES = 32
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
    anim, frame = entity['anim'], entity['aframe']
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


def terrain_channels(obs):
    terrain = obs['terrain']
    if (terrain['height'], terrain['width']) != (9, 15):
        raise ValueError('Transformer curriculum requires a 15x9 room')
    result = np.zeros((7, 9, 15), np.float32)
    positions = []
    for index, x, y, collision, inside, walkable, pit, hazard, solid, destructible in terrain['cells']:
        row, col = divmod(index, 15)
        result[:, row, col] = (inside, walkable, solid, pit, destructible, hazard, 0)
        positions.append((index, x, y))
    for door in obs['doors']:
        if not door['open']:
            index, _, _ = min(positions, key=lambda c: (c[1]-door['pos'][0])**2 + (c[2]-door['pos'][1])**2)
            row, col = divmod(index, 15)
            result[6, row, col] = 1
    return result


def entity_key(entity):
    # Index is for tracking only. Age continuity rejects reused indices after respawn.
    return entity['id'], entity['type'], entity['variant'], entity['subtype']


class VisibleHistory:
    def __init__(self, history=HISTORY, capacity=ENTITY_CAPACITY):
        self.history, self.capacity = history, capacity
        self.frames = deque(maxlen=history)
        self.previous = None
        self.origin = None
        self.previous_action = np.zeros(4, np.float32)
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
            'terrain': spaces.Box(0, 1, (h, 7, 9, 15), np.float32),
            'previous_action': spaces.Box(0, 44, (h, 4), np.float32),
            'time': spaces.Box(0, np.inf, (h,), np.float32),
            'history_mask': spaces.Box(0, 1, (h,), np.float32),
        })

    def clear(self):
        self.frames.clear()
        self.previous = None
        self.origin = None
        self.previous_action[:] = 0

    def append(self, obs):
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
        width, height = right-left, bottom-top
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
                rows.append((row, (e['type'], e['variant'], e['subtype']), animation_bytes(e['anim'])))
        if len(rows) > self.capacity:
            raise ValueError(f'Visible entity/laser segment overflow: {len(rows)} > {self.capacity}; increase capacity, never truncate')
        for i, (row, kind, anim) in enumerate(rows):
            frame['entities'][i] = row
            frame['entity_kind'][i] = kind
            frame['entity_anim'][i] = anim
            frame['entity_mask'][i] = 1
        frame['terrain'] = terrain_channels(obs)
        frame['previous_action'] = self.previous_action.copy()
        frame['time'] = np.float32((obs['logic_frames']-self.origin)/30)
        frame['history_mask'] = np.float32(1)
        self.frames.append(frame)
        self.previous = obs
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
        self.history.previous_action[:] = joint, bomb, item, 1
        move, shoot = divmod(joint, 5)
        return [move, shoot, bomb, item]
