"""Decoder of abp_bridge.lua's binary observation (format 2, abp-0.2.10: abp-0.2.7's layout with monster_damage at
the end of the combat block (abp-0.2.8-hp) and the duel block at the end (abp-0.2.9); abp-0.2.8 changed only the
lineage rule).

Produces the same dict as json.loads of the format-1 message: same keys, same key presence
(anim only when an animation matched, boss_hp only for bosses, per-kind extras), same values
(binary doubles are exact where JSON printed %.14g). Every consumer (IsaacTrainingEnv,
MonstroGymEnv, VisibleHistory, evaluation) therefore runs unchanged.

Integers travel as int64: the engine reports some unsigned values (a player's Index is
0xFFFFFFFF right after a restart), which the JSON path printed as large numbers.

Terrain and grid arrive only when they may have changed (see pack_obs); between frames the
cached dicts are reused, and terrain['version'] lets VisibleHistory reuse its terrain channels.
The dynamic part of the terrain hazard flag (whitelisted effects with contact damage within
size + 28.3 of a cell centre) is recomputed here as terrain_record does in Lua, with the
distance in float32 like the engine's Vector.Distance.
"""
import json
import struct

import numpy as np

MAGIC = 0x32504241  # "ABP2"
ENTITY_EFFECT = 1000
KINDS = {1: 'tear', 2: 'projectile', 3: 'laser', 4: 'bomb', 5: 'pickup', 6: 'npc'}

_HEADER = struct.Struct('<IIIBqqqq')
# abp-0.2.1 adds blocking_hp, blocking_points; abp-0.2.2 blocking_count; abp-0.2.3 the lineage counters;
# abp-0.2.4 tear_hits, blocked_hits; abp-0.2.5 tear_misses, miss_units, miss_streak; abp-0.2.8-hp monster_damage
_COMBAT = struct.Struct('<dddddddddddddddddd')
# abp-0.2.7: after the combat block, a count and per fire frame (frame, damage, kills, miss units)
_CREDIT = struct.Struct('<qddd')
_ROOM = struct.Struct('<qqqqddddBqqqqqq')
_ROOM_DATA = struct.Struct('<qq')
_DOOR = struct.Struct('<BBBddq')
_TERRAIN = struct.Struct('<IB')
_PLAYER = struct.Struct('<qddd' + 'q' * 11 + 'dddddd' + 'BqqB' + 'BB' + 'qqq' + 'qB' + 'Bq')
_ENTITY_HEAD = struct.Struct('<qqqqdddddqqdB')
_ENTITY_TAIL = struct.Struct('<qBqB')
_VEL = struct.Struct('<dd')
_TPH = struct.Struct('<ddd')
_LASER = struct.Struct('<Bddddddq')
_NPC = struct.Struct('<BBBqBdBB')  # abp-0.2.3: + lineage, blocking
# abp-0.2.9 duel block (after a flag byte): active, dead, NPC index, position, size, HP, max HP, iframes, the player's
# iframes, fire cooldown, velocity, move, shoot; then per side (the player, the NPC) the counters of DUEL_SIDE_KEYS
_DUEL = struct.Struct('<BBqdddddqqqddBB')
_DUEL_SIDE = struct.Struct('<dddddddd')
DUEL_SIDE_KEYS = ('shots', 'hits', 'misses', 'miss_units', 'miss_streak', 'damage', 'hurt', 'hurt_amount')
_U8 = struct.Struct('<B')
_U16 = struct.Struct('<H')
_U32 = struct.Struct('<I')


class ObsDecoder:
    """Stateful per connection: holds the last terrain/grid block."""

    def __init__(self):
        self.terrain_version = None
        self.terrain = None
        self.grid = None
        self._cells = None

    def decode(self, payload):
        off = 0
        (magic, logic_frames, game_frame, paused, damage, tears, npc_deaths,
         clears) = _HEADER.unpack_from(payload, off)
        if magic != MAGIC:
            raise ValueError(f'bad observation magic {magic:#x}')
        off += _HEADER.size
        c = _COMBAT.unpack_from(payload, off)
        off += _COMBAT.size
        n = _U16.unpack_from(payload, off)[0]
        off += 2
        credits = []
        for _ in range(n):
            credits.append(list(_CREDIT.unpack_from(payload, off)))
            off += _CREDIT.size
        r = _ROOM.unpack_from(payload, off)
        off += _ROOM.size
        room = dict(type=r[0], shape=r[1], gw=r[2], gh=r[3], top_left=[r[4], r[5]], bottom_right=[r[6], r[7]],
                    clear=bool(r[8]), alive=r[9], frame=r[10], stage=r[11], stage_type=r[12], curses=r[13],
                    room_idx=r[14])
        has_data = payload[off]
        off += 1
        if has_data:
            room['variant'], room['subtype'] = _ROOM_DATA.unpack_from(payload, off)
            off += _ROOM_DATA.size
            n = _U16.unpack_from(payload, off)[0]
            off += 2
            room['name'] = payload[off:off + n].decode('utf8', 'replace')
            off += n
        count = payload[off]
        off += 1
        doors = []
        for _ in range(count):
            slot, is_open, locked, x, y, target = _DOOR.unpack_from(payload, off)
            off += _DOOR.size
            doors.append(dict(slot=slot, open=bool(is_open), locked=bool(locked), pos=[x, y], target_type=target))
        version, present = _TERRAIN.unpack_from(payload, off)
        off += _TERRAIN.size
        if present:
            n = _U32.unpack_from(payload, off)[0]
            off += 4
            terrain = json.loads(payload[off:off + n])
            off += n
            n = _U32.unpack_from(payload, off)[0]
            off += 4
            self.grid = json.loads(payload[off:off + n])
            off += n
            terrain['version'] = version
            self.terrain_version, self.terrain = version, terrain
            self._cells = np.array([(cell[1], cell[2]) for cell in terrain['cells']], np.float32).reshape(-1, 2)
        elif version != self.terrain_version:
            raise ValueError(f'terrain version {version} arrived without its block (cached {self.terrain_version})')
        players = []
        count = payload[off]
        off += 1
        for _ in range(count):
            p = _PLAYER.unpack_from(payload, off)
            off += _PLAYER.size
            rec = dict(id=p[0], pos=[p[1], p[2]], size=p[3], hearts=p[4], max_hearts=p[5], soul=p[6], black=p[7],
                       bone=p[8], eternal=p[9], golden=p[10], lives=p[11], coins=p[12], bombs=p[13], keys=p[14],
                       damage=p[15], fire_delay_max=p[16], shot_speed=p[17], range=p[18], speed=p[19], luck=p[20],
                       can_fly=bool(p[21]), active=p[22], active_charge=p[23], active_ready=bool(p[24]),
                       invulnerable=bool(p[25]), controls=bool(p[26]), head_dir=p[27], fire_dir=p[28],
                       move_dir=p[29], aframe=p[30], flip=bool(p[31]), dead=bool(p[32]), ptype=p[33])
            if payload[off]:
                rec['vel'] = list(_VEL.unpack_from(payload, off + 1))
                off += 1 + _VEL.size
            else:
                off += 1
            players.append(rec)
        count = _U16.unpack_from(payload, off)[0]
        off += 2
        entities = []
        for _ in range(count):
            h = _ENTITY_HEAD.unpack_from(payload, off)
            off += _ENTITY_HEAD.size
            n = payload[off]
            anim = payload[off + 1:off + 1 + n].decode('utf8', 'replace')
            off += 1 + n
            aframe, flip, age, kind = _ENTITY_TAIL.unpack_from(payload, off)
            off += _ENTITY_TAIL.size
            rec = dict(id=h[0], type=h[1], variant=h[2], subtype=h[3], pos=[h[4], h[5]], size=h[6],
                       size_multi=[h[7], h[8]], coll=h[9], gcoll=h[10], cdmg=h[11])
            if h[12]:
                rec['anim'] = anim
            rec.update(aframe=aframe, flip=bool(flip), age=age)
            if payload[off]:
                rec['vel'] = list(_VEL.unpack_from(payload, off + 1))
                off += 1 + _VEL.size
            else:
                off += 1
            if kind in (1, 2):
                height, fall, scale = _TPH.unpack_from(payload, off)
                off += _TPH.size
                if kind == 2:
                    rec['projectile'] = True
                rec.update(height=height, fall=fall, scale=scale)
            elif kind == 3:
                circle, radius, angle, length, width, ex, ey, n = _LASER.unpack_from(payload, off)
                off += _LASER.size
                samples = [list(_VEL.unpack_from(payload, off + 16 * i)) for i in range(n)]
                off += 16 * n
                rec['laser'] = {'circle': bool(circle), 'radius': radius, 'angle': angle, 'length': length,
                                'width': width, 'end': [ex, ey], 'samples': samples}
            elif kind == 4:
                rec['bomb'] = True
            elif kind == 5:
                rec['pickup'] = True
            elif kind == 6:
                enemy, vulnerable, boss, champion, has_hp, hp, lineage, blocking = _NPC.unpack_from(payload, off)
                off += _NPC.size
                rec.update(enemy=bool(enemy), vulnerable=bool(vulnerable), boss=bool(boss), champion=champion)
                if has_hp:
                    rec['boss_hp'] = hp
                rec.update(lineage=bool(lineage), blocking=bool(blocking))
            entities.append(rec)
        duel = None
        if payload[off]:   # abp-0.2.9
            d = _DUEL.unpack_from(payload, off + 1)
            off += 1 + _DUEL.size
            sides = []
            for _ in range(2):
                sides.append(dict(zip(DUEL_SIDE_KEYS, _DUEL_SIDE.unpack_from(payload, off))))
                off += _DUEL_SIDE.size
            duel = dict(active=bool(d[0]), dead=bool(d[1]), npc=d[2], pos=[d[3], d[4]], size=d[5], hp=d[6], max_hp=d[7],
                        iframes=d[8], player_iframes=d[9], cooldown=d[10], vel=[d[11], d[12]], move=d[13], shoot=d[14],
                        player=sides[0], npc_side=sides[1])
        else:
            off += 1
        if off != len(payload):
            raise ValueError(f'observation payload has {len(payload) - off} trailing bytes')
        obs = dict(combat_schema=3, engine='abplus-1.06', game_frame=game_frame, paused=bool(paused),
                    logic_frames=logic_frames,
                    events=dict(damage=damage, tears=tears, npc_deaths=npc_deaths, clears=clears),
                    players=players, entities=entities, room=room, grid=self.grid, doors=doors,
                    terrain=self._terrain_with_hazards(entities),
                    combat=dict(player_damage_events=c[0], player_damage=c[1], enemy_damage_events=c[2],
                                enemy_damage=c[3], enemy_damage_fraction=c[4], blocking_hp=c[5],
                                blocking_points=c[6], blocking_count=c[7], lineage_damage=c[8],
                                lineage_kills=c[9], lineage_count=c[10], lineage_hp=c[11],
                                tear_hits=c[12], blocked_hits=c[13], tear_misses=c[14], miss_units=c[15],
                                miss_streak=c[16], credits=credits, monster_damage=c[17]))
        if duel is not None:
            obs['duel'] = duel
        return obs

    def _terrain_with_hazards(self, entities):
        hazards = [e for e in entities if e['type'] == ENTITY_EFFECT and e['cdmg'] > 0]
        if not hazards or self.terrain is None:
            return self.terrain
        cells = self._cells
        near = np.zeros(len(cells), bool)
        for e in hazards:
            d = cells - np.asarray(e['pos'], np.float32)
            distance = np.sqrt((d * d).sum(axis=1, dtype=np.float32))
            near |= distance.astype(np.float64) <= e['size'] + 28.3
        if not near.any():
            return self.terrain
        rows = [list(cell) for cell in self.terrain['cells']]
        for i in np.flatnonzero(near):
            rows[i][7] = 1
        return dict(self.terrain, cells=rows, version=(self.terrain_version, tuple(np.flatnonzero(near).tolist())))


def compare(expected, got, path='obs', tolerance=1e-12):
    """First difference between a JSON observation and a decoded binary one, or None."""
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return path, expected, got
        keys = (set(expected) | set(got)) - {'version'}
        for k in sorted(keys, key=str):
            if k not in expected or k not in got:
                return f'{path}.{k}', expected.get(k, '<absent>'), got.get(k, '<absent>')
            d = compare(expected[k], got[k], f'{path}.{k}', tolerance)
            if d:
                return d
        return None
    if isinstance(expected, list):
        if not isinstance(got, list) or len(expected) != len(got):
            return path + '#len', len(expected) if isinstance(expected, list) else expected, \
                len(got) if isinstance(got, list) else got
        for i, (a, b) in enumerate(zip(expected, got)):
            d = compare(a, b, f'{path}[{i}]', tolerance)
            if d:
                return d
        return None
    if isinstance(expected, bool) or isinstance(got, bool):
        return None if expected == got and type(expected) is type(got) else (path, expected, got)
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        if expected == got or abs(expected - got) <= tolerance * max(1.0, abs(expected), abs(got)):
            return None
        return path, expected, got
    return None if expected == got else (path, expected, got)
