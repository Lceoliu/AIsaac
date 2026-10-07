"""Decoder of abp_bridge.lua's lean observation (abp-0.2.15, pack_lean): fixed-layout records read as numpy arrays.

The layout is described above pack_lean in abp_bridge.lua. A LeanObs holds views into the payload (players, doors and
entities as structured arrays) and the terrain dict of the last terrain block (cached per connection, as ObsDecoder
does for format 2). Nothing here builds per-entity Python objects.
"""
import json
import struct

import numpy as np

MAGIC = 0x33504241  # "ABP3"
_HEADER = struct.Struct('<IIIBBBB')
_ROOM = struct.Struct('<iiiiiiiiffff')
_TOTALS = struct.Struct('<dddd')
PLAYER_FIELDS = ('id', 'x', 'y', 'vx', 'vy', 'size', 'hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal',
                 'golden', 'lives', 'coins', 'bombs', 'keys', 'damage', 'fire_delay_max', 'shot_speed', 'range',
                 'speed', 'luck', 'can_fly', 'active', 'active_charge', 'active_ready', 'invulnerable', 'controls',
                 'head_dir', 'fire_dir', 'move_dir', 'aframe', 'dead', 'fire_delay', 'damage_cooldown',
                 # 2026-10-07 (charge): the charge counter of a charged weapon (Entity_Player +0x2634) and the weapon
                 # types as bits (bit w: HasWeaponType(w), w = 0 .. 10); see pack_lean in abp_bridge.lua
                 'charge', 'weapons')
PLAYER = np.dtype([(name, '<f8') for name in PLAYER_FIELDS])
DOOR = np.dtype([('slot', 'u1'), ('open', 'u1'), ('locked', 'u1'), ('seen', 'u1'), ('x', '<f4'), ('y', '<f4'),
                 ('target_type', '<i4')])
ENTITY = np.dtype([('id', '<i8'), ('type', '<i4'), ('variant', '<i4'), ('subtype', '<i4'), ('x', '<f8'), ('y', '<f8'),
                   ('vx', '<f8'), ('vy', '<f8'), ('size', '<f4'), ('size_mx', '<f4'), ('size_my', '<f4'),
                   ('coll', '<i4'), ('gcoll', '<i4'), ('cdmg', '<f4'), ('aframe', '<i4'), ('age', '<i4'),
                   ('kind', 'u1'), ('flags', 'u1'), ('anim', 'u1'), ('flip', 'u1'), ('champion', '<i4'),
                   ('hp', '<f4'), ('height', '<f4'), ('fall', '<f4'), ('scale', '<f4')])
MAP_ROOM = np.dtype([('idx', 'u1'), ('shape', 'u1'), ('type', 'u1'), ('display', 'u1'), ('flags', 'u1')])
# 2026-10-06 (items, Phase A): the inventory block (flag 16, abp_bridge.lua lean_inventory; only when the bridge's
# lean_items is on): a header, then n x (collectible id, count), newest first
INV_HEADER = struct.Struct('<BBHHHHHHI')   # n, layout version, active MaxCharges, trinket 0, trinket 1, pill color,
#                                           pill effect + 1 when identified (0 unknown / none), card, curses
INV_ITEM = np.dtype([('id', '<u2'), ('count', 'u1')])
assert PLAYER.itemsize == 304 and DOOR.itemsize == 16 and ENTITY.itemsize == 108 and MAP_ROOM.itemsize == 5
assert INV_HEADER.size == 18 and INV_ITEM.itemsize == 3
_LASER = struct.Struct('<qBddddddq')
# entity kinds (as format 2) and flag bits
KIND_TEAR, KIND_PROJECTILE, KIND_LASER, KIND_BOMB, KIND_PICKUP, KIND_NPC = 1, 2, 3, 4, 5, 6
FLAG_ENEMY, FLAG_VULNERABLE, FLAG_BOSS, FLAG_BLOCKING = 1, 2, 4, 8


class LeanObs:
    __slots__ = ('logic_frames', 'game_frame', 'clear', 'paused', 'room', 'damage_taken', 'monsters_hp',
                 'blocking_hp', 'blocking_count', 'players', 'doors', 'entities', 'lasers', 'terrain', 'grid',
                 'terrain_version', 'terrain_changed', 'map', 'map_version', 'terrain_key', 'inv', 'inv_head',
                 'inv_version')

    @property
    def dead(self):
        return bool(self.players['dead'][0])


class LeanDecoder:
    """Stateful per connection: keeps the last terrain block and the last map block."""

    def __init__(self):
        self.terrain_version = None
        self.terrain = None
        self.grid = None
        self.terrain_raw = None   # the last terrain block's bytes (without its version)
        self.terrain_key = 0      # changes when the terrain block's content does (not at a resend of the same)
        self.map = np.zeros((0,), MAP_ROOM)
        self.map_version = 0
        self.inv = np.zeros((0,), INV_ITEM)   # the last inventory block (items mode): held ids, newest first
        self.inv_head = None                  # its header (INV_HEADER fields), None before the first block
        self.inv_version = 0

    def decode(self, payload):
        magic, logic_frames, game_frame, flags, n_players, n_doors, n_lasers = _HEADER.unpack_from(payload, 0)
        if magic != MAGIC:
            raise ValueError(f'bad lean observation magic {magic:#x}')
        off = _HEADER.size
        o = LeanObs()
        o.logic_frames, o.game_frame = logic_frames, game_frame
        o.clear, o.paused = bool(flags & 1), bool(flags & 4)
        o.room = _ROOM.unpack_from(payload, off)
        off += _ROOM.size
        o.damage_taken, o.monsters_hp, o.blocking_hp, o.blocking_count = _TOTALS.unpack_from(payload, off)
        off += _TOTALS.size
        o.players = np.frombuffer(payload, PLAYER, n_players, off)
        off += PLAYER.itemsize * n_players
        o.doors = np.frombuffer(payload, DOOR, n_doors, off)
        off += DOOR.itemsize * n_doors
        n = struct.unpack_from('<H', payload, off)[0]
        off += 2
        o.entities = np.frombuffer(payload, ENTITY, n, off)
        off += ENTITY.itemsize * n
        lasers = []
        for _ in range(n_lasers):
            idx, circle, radius, angle, length, width, ex, ey, k = _LASER.unpack_from(payload, off)
            off += _LASER.size
            samples = np.frombuffer(payload, '<f8', 2 * k, off).reshape(k, 2)
            off += 16 * k
            lasers.append(dict(id=idx, circle=bool(circle), radius=radius, angle=angle, length=length, width=width,
                               end=(ex, ey), samples=samples))
        o.lasers = lasers
        o.terrain_changed = bool(flags & 2)
        if o.terrain_changed:
            version, n = struct.unpack_from('<II', payload, off)
            n2 = struct.unpack_from('<I', payload, off + 8 + n)[0]
            raw = payload[off + 4:off + 12 + n + n2]   # the block without its version
            if raw != self.terrain_raw:   # the bridge resends an unchanged terrain every 30 logic frames: parsed once
                terrain = json.loads(payload[off + 8:off + 8 + n])
                self.grid = json.loads(payload[off + 12 + n:off + 12 + n + n2])
                self.terrain_raw, self.terrain = raw, terrain
                self.terrain_key += 1
            off += 12 + n + n2
            self.terrain['version'] = version
            self.terrain_version = version
        if flags & 8:
            n = payload[off]
            self.map = np.frombuffer(payload, MAP_ROOM, n, off + 1)
            self.map_version += 1
            off += 1 + MAP_ROOM.itemsize * n
        if flags & 16:
            head = INV_HEADER.unpack_from(payload, off)
            self.inv_head = head
            self.inv = np.frombuffer(payload, INV_ITEM, head[0], off + INV_HEADER.size)
            self.inv_version += 1
            off += INV_HEADER.size + INV_ITEM.itemsize * head[0]
        if off != len(payload):
            raise ValueError(f'lean observation has {len(payload) - off} trailing bytes')
        o.terrain, o.grid, o.terrain_version = self.terrain, self.grid, self.terrain_version
        o.terrain_key = self.terrain_key
        o.map, o.map_version = self.map, self.map_version
        o.inv, o.inv_head, o.inv_version = self.inv, self.inv_head, self.inv_version
        return o


def read_lean(bridge, decoder):
    """The next lean observation of a bridge client (after a step, obs or play command in lean mode)."""
    return decoder.decode(read_lean_raw(bridge))


def read_lean_raw(bridge):
    """The next lean observation's payload, not decoded (tok_obs.FastRow takes it as it is)."""
    while True:
        line = bridge._read_line()
        if line.startswith(b'L '):
            return bridge._read_exact(int(line.split(b' ')[1]))
        msg = json.loads(line.decode('utf-8'))
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg', 'error'))


def lean_logic_frames(payload):
    """The logic frame count in a raw lean payload's header."""
    return _LOGIC_FRAMES.unpack_from(payload, 4)[0]


_LOGIC_FRAMES = struct.Struct('<I')
