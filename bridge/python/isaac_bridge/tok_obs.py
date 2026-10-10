"""Observation record of the token policy (EXPERIMENTS.md C55 on): one fixed-size ROW per decision, written by a
sampler worker from the bridge's lean observation (abplus_lean.LeanObs) into shared memory.

What the policy reads is what the player sees: the player's own state, every visible entity (position relative to
the player, the engine's velocity, size, kind, type / variant / subtype, the flags of pack_lean, a boss's health
bar), the doors, the room's grid (terrain_channels' seven channels) and a 9 x 9 cell patch of it around the player.
An NPC's hit points (other than a boss's), its AI state and the game's RNG are not in it. The step's reward terms
(half hearts lost, share of the room's monster HP removed, the events of a whole-floor episode) and the episode's end
ride along in the same record but are not policy input.

For whole floors (tok_floor): a door also says whether the room behind it was visited and is clear (what the minimap
shows), the floor's exit (a trapdoor or stairs, a grid entity) is one more entity token of its own type, and the
minimap itself is a 13 x 13 image of the level grid with the rooms the game shows there (pack_lean's map block).
"""
import ctypes
import os
import struct

import numpy as np

from .transformer_obs import terrain_channels

ENT_CAP = 64
# 2026-10-08 (abp-0.2.16, the policy's view of lasers and tear flags): entity columns 0 .. 32 are those of before (same
# meaning, same order); appended are FLAG_F flag columns and LASER_F laser columns, 0 for entities they do not apply to.
#   flag columns (FLAG_COLUMN + j): 1 when the entity's flag word (abplus_lean 'tflags': a tear's or a laser's
#     TearFlags, a projectile's ProjectileFlags) has a bit of TEAR_FLAG_MASKS[j] (tears, lasers) or PROJ_FLAG_MASKS[j]
#     (projectiles). Bit numbers: AB+'s own resources/scripts/enums.lua (TearFlags, ProjectileFlags); the engine's
#     getters (Entity_Tear::GetTearFlags +0x758, Entity_Projectile::GetProjectileFlags +0x760) return the uint64 that
#     luabridge pushes as a Lua integer.
#   laser columns (LASER_COLUMN + 0 .. 6): cos and sin of the laser's angle, LaserLength / 300 (0: an unbounded beam
#     such as Brimstone's), its end point relative to the player / 200 (GetEndPoint; a circle laser's own position), circle
#     (1 / 0), Radius / 100. A laser's width is its Size (column 7).
ENT_F0 = 33
FLAG_F = 16
LASER_F = 7
FLAG_COLUMN = ENT_F0
LASER_COLUMN = ENT_F0 + FLAG_F
ENT_F = ENT_F0 + FLAG_F + LASER_F
_B = [1 << k for k in range(64)]
# j: tear (and laser) TearFlags | projectile ProjectileFlags
TEAR_FLAG_MASKS = np.array([
    _B[0],                     # 0 TEAR_SPECTRAL          | NO_WALL_COLLIDE (13)
    _B[1],                     # 1 TEAR_PIERCING          | GHOST (4)
    _B[2],                     # 2 TEAR_HOMING            | SMART (0)
    _B[3],                     # 3 TEAR_SLOW              | SLOWED (24)
    _B[4],                     # 4 TEAR_POISON            | ACID_GREEN, ACID_RED (2, 8)
    _B[5],                     # 5 TEAR_FREEZE            | GOO, RED_CREEP, CREEP_BROWN (3, 10, 14)
    _B[6] | _B[18],            # 6 TEAR_SPLIT, QUADSPLIT  | BURST, BURST3 (16, 29)
    _B[7],                     # 7 TEAR_GROW              | ACCELERATE (27)
    _B[8],                     # 8 TEAR_BOMBERANG         | BOOMERANG (6)
    _B[9],                     # 9 TEAR_PERSISTENT        | HIT_ENEMIES (7)
    _B[10] | _B[26] | _B[44],  # 10 TEAR_WIGGLE, SPIRAL, BIG_SPIRAL | WIGGLE, SINE_VELOCITY, MEGA_WIGGLE, SAWTOOTH (5, 21-23)
    _B[12],                    # 11 TEAR_EXPLOSIVE (Ipecac) | EXPLODE (1)
    _B[16],                    # 12 TEAR_ORBIT            | ORBIT_CW, ORBIT_CCW (11, 12)
    _B[19],                    # 13 TEAR_BOUNCE           | CURVE_LEFT, CURVE_RIGHT, TURN_HORIZONTAL (18-20)
    _B[22],                    # 14 TEAR_BURN             | FIRE (15)
    _B[24]], np.int64)         # 15 TEAR_KNOCKBACK        | CANT_HIT_PLAYER (31)
PROJ_FLAG_MASKS = np.array([
    _B[13], _B[4], _B[0], _B[24], _B[2] | _B[8], _B[3] | _B[10] | _B[14], _B[16] | _B[29], _B[27], _B[6], _B[7],
    _B[5] | _B[21] | _B[22] | _B[23], _B[1], _B[11] | _B[12], _B[18] | _B[19] | _B[20], _B[15], _B[31]], np.int64)
assert len(TEAR_FLAG_MASKS) == len(PROJ_FLAG_MASKS) == FLAG_F
PLAYER_F = 31
DOOR_CAP = 8
DOOR_F = 7
EXIT_TYPE = 1023      # entity type id of the floor's exit (trapdoor / stairs): no engine entity has it
EV_CLEARED, EV_NEW_ROOM, EV_BOSS = 1, 2, 4   # bits of ROW['events']
# 2026-10-06 (Phase A: items and runs past Basement I; only with EpisodeState(run=True) / (items=True))
EV_EXIT = 8           # run mode: the next floor was reached (the episode goes on)
EV_ITEM = 16          # items: a collectible was gained (a held id's count went up)
EV_ACTIVE = 32        # items: the active item was used (its charge went down, or a used-up active is gone)
EV_PILL = 64          # items: the pill / card in pocket slot 0 was used (the slot emptied)
# Item ids (user decision 2026-10-06: sized for Repentance / Repentance+ with headroom): collectibles 0..1023 (AB+ max
# 552, Repentance+ 732), trinkets / pill effects / cards 0..255. ITEM_UNKNOWN: a pedestal under Curse of the Blind.
N_COLLECTIBLE, N_TRINKET, N_PILL_EFFECT, N_CARD, N_PILL_COLOR = 1024, 256, 256, 256, 16
ITEM_UNKNOWN = 1023
INV_CAP = 16          # held collectibles in a record (newest first)
PINV_F = 12           # pinv: active charge share, max charge / 12, pill held, pill identified, card held, trinket held,
#                       held collectibles / 16, stage / 12, curses darkness, lost, unknown, blind
PCHARGE_F = 12        # 2026-10-07 (charge): min(charge, 120) / 60, min(charge / max(MaxFireDelay, 1), 3) / 3, the weapon
#                       type bits 1 .. 10 (tears, Brimstone, Technology, Mom's Knife, Dr. Fetus, Epic Fetus, Monstro's Lung,
#                       Ludovico, Tech X, bone); charge: the charge bar's counter (abplus_lean PLAYER 'charge')
PATCH = 9
GRID = (7, 16, 28)
MAP = (8, 13, 13)     # shown, visited, clear, the player's room, boss, treasure, shop, another special room
# cells a room of each RoomShape covers, as offsets from its GridIndex (the top left of its bounding box)
SHAPE_CELLS = {4: (0, 13), 5: (0, 13), 6: (0, 1), 7: (0, 1), 8: (0, 1, 13, 14), 9: (1, 13, 14), 10: (0, 13, 14),
               11: (0, 1, 14), 12: (0, 1, 13)}
CELL = 40.0
ROW = np.dtype([
    ('player', np.float32, (PLAYER_F,)),
    ('ent', np.float32, (ENT_CAP, ENT_F)),
    ('ent_id', np.int16, (ENT_CAP, 3)),      # type (< 1024), variant mod 1024, subtype mod 256
    ('doors', np.float32, (DOOR_CAP, DOOR_F)),
    ('patch', np.uint8, (GRID[0], PATCH, PATCH)),
    ('grid', np.uint8, GRID),
    ('map', np.uint8, MAP),
    ('n_ent', np.int32),
    ('n_doors', np.int32),
    ('t', np.int32),                         # decisions since the episode's start
    ('group', np.int32),
    ('episode', np.int64),
    ('seed', np.int64),
    ('hurt', np.float32),                    # half hearts lost during the step that led here
    ('damage', np.float32),                  # share of the room's starting monster HP removed during that step
    ('done', np.uint8),                      # 0 running, 1 room clear, 2 death, 3 time limit
    ('first', np.uint8),
    ('bombs', np.uint8),
    ('events', np.uint8),                    # whole-floor episodes: EV_* of the step that led here
    # 2026-10-06 (Phase A). The fields above are unchanged (same offsets, same bytes); these are zero unless the
    # episode has items (EpisodeState.items) or is a run (EpisodeState.run: 'stage' only)
    ('ent_item', np.int16, (ENT_CAP, 2)),    # per entity: collectible id of a pedestal (pickup variant 100; ITEM_UNKNOWN
    #                                          under Curse of the Blind), trinket id of a trinket (variant 350)
    ('inv', np.int16, (INV_CAP, 2)),         # held collectibles, newest first: id, count
    ('pitem', np.int16, (6,)),               # active item id, trinket 0, trinket 1, pill color, pill effect + 1 when
    #                                          identified, card / rune id
    ('pinv', np.float32, (PINV_F,)),
    ('stage', np.uint8),                     # the level's stage of this record
    # 2026-10-07 (charge; appended, the fields above keep their offsets): player 0's charged-weapon state (PCHARGE_F),
    # written in every mode; a policy input only for TokPolicy(charge=True)
    ('pcharge', np.float32, (PCHARGE_F,)),
    # 2026-10-07 (counterfactual branches, tok_branch.py; appended, the fields above keep their offsets): 0 in every
    # ordinary record; in a branch episode's records the worker's point id and the branch's number (1, 2, ...). Never
    # a policy input.
    ('point', np.int64),
    ('branch', np.int16),
    # 2026-10-07 (Phase B2, common random numbers for the policy; appended): 0 in every ordinary record (the actor
    # samples with fresh uniforms); in a branch episode's records the key of its branch pair (tok_branch.crn_key): the
    # actor's sampling uniforms of such a record are tok_policy.crn_uniforms(key, ROW 't', head slot), the same in
    # every branch of the pair. Never a policy input.
    ('crn', np.int64),
    # 2026-10-08 (the build lab, tok_lab.py; appended, the fields above keep their offsets): 0 in every ordinary record;
    # in a lab episode's records the job id of the build being benchmarked and the panel state's number + 1. Never a
    # policy input.
    ('lab', np.int64),
    ('lab_s', np.int16),
    # 2026-10-08 (character randomisation, bridge abp-0.2.17; appended, the fields above keep their offsets): player 0's
    # PlayerType (abplus_lean 'ptype', clipped to 0 .. N_CHAR - 1), written in every mode; a policy input only for
    # TokPolicy(pchar=True)
    ('pchar', np.int16),
], align=True)
N_CHAR = 32           # 2026-10-08: PlayerType ids the policy's character input has room for (AB+ 0 .. 17, Repentance+ 0 .. 40
#                       would need more)
DONE_BRANCH = 4       # ROW 'done' of a branch episode's last record when it ends for the branch's own reasons (the
#                       floor's exit or a boss cleared in run mode, or the branch's time cap): a truncation, the
#                       trainer bootstraps from the value of that record (train_tok: the boot decisions)
KIND_COLUMN = 16      # entity columns 16..22: one-hot of pack_lean's kind (0 other, 1 tear, 2 projectile, 3 laser,
#                       4 bomb, 5 pickup, 6 NPC)


class EpisodeState:
    """What the encoder keeps between the records of one episode."""

    def __init__(self, limit, floor=False, stall=0, run=False, items=False):
        self.limit = max(int(limit), 1)
        self.floor = floor or run  # a whole-floor episode: it ends at the floor's exit, not at a room's clear
        self.run = run             # a run (2026-10-06): the floor's exit is an event (EV_EXIT), the episode goes on
        self.items = items         # items (2026-10-06): ent_item, inv, pitem, pinv and the item events are written
        self.inv_version = None    # items: the decoder's inventory block last taken in
        self.inv_counts = None     # items: {id: count} of that block
        self.inv_arr = np.zeros((INV_CAP, 2), np.int16)
        self.pitem = np.zeros(6, np.int16)
        self.pinv = np.zeros(PINV_F, np.float32)
        self.max_charge = 0
        self.blind = False
        self.prev_active = None    # items: (active id, charge) at the record before
        self.pocket = (0, 0)       # items: (pill color, card) of the last block
        self.stall = int(stall)    # whole floors: decisions without progress (a new room, a cleared room, a monster
        #                            hurt) that end the episode
        self.stage0 = None
        self.progress_t = 0
        self.hp0 = None
        self.previous = None      # (damage taken, monsters' HP) at the record before
        self.version = None
        self.grid = None
        self.padded = None
        self.origin = (0.0, 0.0)
        self.exits = []            # (x, y) of the room's trapdoors and stairs
        self.map_version = None
        self.map = np.zeros(MAP, np.uint8)
        self.room = None           # whole floors: the room the record before was in, was it clear, rooms seen so far
        self.was_clear = True
        self.visited = set()


def _terrain(o, st):
    # keyed by the terrain's content (abplus_lean: the same block resent keeps its key), not its version number
    version = (o.terrain_key, o.doors.tobytes())
    if st.version == version:
        return
    st.version = version
    doors = [dict(open=bool(d['open']), pos=(float(d['x']), float(d['y']))) for d in o.doors]
    st.grid = terrain_channels(dict(terrain=o.terrain, doors=doors), GRID[1:]).astype(np.uint8)
    half = PATCH // 2
    st.padded = np.pad(st.grid, ((0, 0), (half, half), (half, half)))
    for cell in o.terrain['cells']:
        if cell[0] == 0:
            st.origin = (float(cell[1]), float(cell[2]))
            break
    st.exits = [(float(g[5]), float(g[6])) for g in (o.grid or []) if g[1] in (17, 18)]


def _minimap(o, st):
    if st.map_version == o.map_version:
        return
    st.map_version = o.map_version
    m = st.map
    m[:] = 0
    for r in o.map:
        kind = int(r['type'])
        special = 4 if kind == 5 else 5 if kind == 4 else 6 if kind == 2 else 7 if kind > 1 else -1
        for cell in SHAPE_CELLS.get(int(r['shape']), (0,)):
            cell += int(r['idx'])
            if not 0 <= cell < 169:
                continue
            y, x = divmod(cell, 13)
            m[0, y, x] = 1
            m[1, y, x], m[2, y, x], m[3, y, x] = r['flags'] & 1, (r['flags'] >> 1) & 1, (r['flags'] >> 2) & 1
            if special >= 0:
                m[special, y, x] = 1


def encode_row(o, st, row, t):
    """Write the lean observation `o` into row (a length-1 view of a ROW array). Sets everything but episode, seed,
    group and first. Returns the done code."""
    p = o.players[0]
    px, py = float(p['x']), float(p['y'])
    room = o.room
    inv_events = _inventory(o, st) if st.items else 0   # first: ent_item needs st.blind
    tlx, tly, brx, bry = room[8:12]
    w, h = max(brx - tlx, 1.0), max(bry - tly, 1.0)
    units = p['hearts'] + p['soul'] + p['eternal'] + 2 * p['bone']
    row['player'][0] = (
        (px - tlx) / w, (py - tly) / h, p['vx'] / 10, p['vy'] / 10, p['hearts'] / 12, p['soul'] / 12,
        p['max_hearts'] / 12, units / 12, min(p['bombs'], 10) / 10, min(p['keys'], 10) / 10, min(p['coins'], 50) / 50,
        p['damage'] / 10, p['fire_delay_max'] / 20, p['shot_speed'] / 2, p['range'] / 500, p['speed'] / 2,
        p['luck'] / 5, p['can_fly'], p['invulnerable'], max(-1.0, min(1.0, p['fire_delay'] / 20)),
        min(p['damage_cooldown'], 120) / 60, float(p['active'] != 0), p['active_ready'], float(o.clear),
        min(t / st.limit, 1.0), p['size'] / 20, room[2] / 28, room[3] / 16, min(room[6], 10) / 10,
        min(o.blocking_count, 10) / 10,
        # whole floors: how much of the time without a new or a cleared room is used up (as of the record before)
        min((t - st.progress_t) / st.stall, 1.0) if st.floor and st.stall > 0 and st.stage0 is not None else 0.0)
    row['bombs'] = min(int(p['bombs']), 255)
    weapons = int(p['weapons'])
    row['pcharge'][0] = (min(p['charge'], 120) / 60, min(p['charge'] / max(p['fire_delay_max'], 1.0), 3.0) / 3) + \
        tuple(float((weapons >> w) & 1) for w in range(1, 11))
    row['pchar'] = min(max(int(p['ptype']), 0), N_CHAR - 1)   # 2026-10-08 (abp_row_encode version 6)
    e = o.entities
    n = len(e)
    out = row['ent'][0]
    if n:
        dx = (e['x'] - px).astype(np.float32)
        dy = (e['y'] - py).astype(np.float32)
        dist = np.hypot(dx, dy)
        if n > ENT_CAP:   # the nearest ones
            keep = np.argsort(dist)[:ENT_CAP]
            e, dx, dy, dist, n = e[keep], dx[keep], dy[keep], dist[keep], ENT_CAP
        out[:n] = 0.0
        kind = np.minimum(e['kind'], 6)
        flags = e['flags']
        out[:n, 0], out[:n, 1], out[:n, 2] = dx / 200, dy / 200, dist / 300
        out[:n, 3], out[:n, 4] = e['vx'] / 10, e['vy'] / 10
        out[:n, 5], out[:n, 6] = (e['x'] - tlx) / w, (e['y'] - tly) / h
        out[:n, 7], out[:n, 8], out[:n, 9] = e['size'] / 20, e['size_mx'], e['size_my']
        out[:n, 10], out[:n, 11] = e['coll'] / 4, e['gcoll'] / 8
        out[:n, 12], out[:n, 13] = e['cdmg'] > 0, np.minimum(e['cdmg'], 4) / 2
        out[:n, 14], out[:n, 15] = np.minimum(e['aframe'], 120) / 30, np.minimum(e['age'], 360) / 90
        out[np.arange(n), KIND_COLUMN + kind] = 1.0
        out[:n, 23], out[:n, 24] = flags & 1, (flags >> 1) & 1
        out[:n, 25], out[:n, 26] = (flags >> 2) & 1, (flags >> 3) & 1
        out[:n, 27], out[:n, 28], out[:n, 29] = e['hp'], e['height'] / 30, e['fall'] / 10
        out[:n, 30], out[:n, 31], out[:n, 32] = e['scale'], e['anim'] / 8, e['flip']
        # 2026-10-08 (abp-0.2.16): flag and laser columns (abp_turbo.c abp_row_encode version 5)
        word = e['tflags']
        if word.any():
            masks = np.where((kind == 2)[:, None], PROJ_FLAG_MASKS[None], TEAR_FLAG_MASKS[None])
            out[:n, FLAG_COLUMN:FLAG_COLUMN + FLAG_F] = (word[:, None] & masks) != 0
        lz = np.flatnonzero(e['kind'] == 3)
        if len(lz):
            L, el = LASER_COLUMN, e[lz]
            out[lz, L], out[lz, L + 1], out[lz, L + 2] = el['l_cos'], el['l_sin'], el['l_len'] / 300
            out[lz, L + 3] = (el['l_ex'].astype(np.float64) - px).astype(np.float32) / 200
            out[lz, L + 4] = (el['l_ey'].astype(np.float64) - py).astype(np.float32) / 200
            out[lz, L + 5], out[lz, L + 6] = el['l_circle'], el['l_radius'] / 100
        ids = row['ent_id'][0]
        ids[:n, 0] = np.clip(e['type'], 0, 1023)
        ids[:n, 1] = e['variant'] % 1024
        ids[:n, 2] = e['subtype'] % 256
        if st.items:   # abp_turbo.c abp_row_encode's ent_item
            pickup = e['type'] == 5
            it = row['ent_item'][0]
            it[:n, 0] = np.where(pickup & (e['variant'] == 100),
                                 ITEM_UNKNOWN if st.blind else np.clip(e['subtype'], 0, 1023), 0)
            it[:n, 1] = np.where(pickup & (e['variant'] == 350), np.minimum(e['subtype'] & 0x7fff, 255), 0)
    d = o.doors
    k = min(len(d), DOOR_CAP)
    if k:
        doors = row['doors'][0]
        doors[:k, 0], doors[:k, 1] = (d['x'][:k] - px) / 200, (d['y'][:k] - py) / 200
        doors[:k, 2], doors[:k, 3], doors[:k, 4] = d['open'][:k], d['locked'][:k], d['target_type'][:k] / 30
        doors[:k, 5], doors[:k, 6] = d['seen'][:k] & 1, (d['seen'][:k] >> 1) & 1
    row['n_doors'] = k
    _terrain(o, st)
    for ex, ey in st.exits:   # the floor's exit as an entity token
        if n >= ENT_CAP:
            break
        out[n] = 0.0
        out[n, 0], out[n, 1], out[n, 2] = (ex - px) / 200, (ey - py) / 200, np.hypot(ex - px, ey - py) / 300
        out[n, 5], out[n, 6], out[n, 7], out[n, KIND_COLUMN] = (ex - tlx) / w, (ey - tly) / h, 1.0, 1.0
        row['ent_id'][0, n] = (EXIT_TYPE, 0, 0)
        if st.items:
            row['ent_item'][0, n] = 0
        n += 1
    row['n_ent'] = n
    row['grid'] = st.grid
    if st.floor:
        _minimap(o, st)
        row['map'] = st.map
    else:
        row['map'] = 0
    col = int(round((px - st.origin[0]) / CELL))
    r = int(round((py - st.origin[1]) / CELL))
    col, r = min(max(col, 0), GRID[2] - 1), min(max(r, 0), GRID[1] - 1)
    row['patch'] = st.padded[:, r:r + PATCH, col:col + PATCH]
    events = 0
    room_idx = room[4]
    # a run's next floor (2026-10-06): a room change even where the new start room has the old room's index
    next_floor = st.run and st.stage0 is not None and room[5] != st.stage0
    if next_floor:
        st.visited = set()
    if st.previous is None:
        st.hp0 = max(float(o.monsters_hp), 1.0)
        row['hurt'] = row['damage'] = 0.0
        st.visited.add(room_idx)
    elif room_idx != st.room or next_floor:   # whole floors: another room; its monsters are a new total
        row['hurt'] = o.damage_taken - st.previous[0]
        row['damage'] = 0.0
        st.hp0 = max(float(o.monsters_hp), 1.0)
        if room_idx not in st.visited:
            st.visited.add(room_idx)
            events |= EV_NEW_ROOM
    else:
        row['hurt'] = o.damage_taken - st.previous[0]
        row['damage'] = min(max(0.0, st.previous[1] - o.monsters_hp) / st.hp0, 1.0)
        if o.clear and not st.was_clear:
            events |= EV_CLEARED | (EV_BOSS if room[0] == 5 else 0)
    if next_floor:   # the floor's exit reached: an event of its own, not a new room of the old floor
        events = (events & ~EV_NEW_ROOM) | EV_EXIT
        st.stage0 = room[5]
    st.room, st.was_clear = room_idx, bool(o.clear)
    row['events'] = events
    st.previous = (o.damage_taken, o.monsters_hp)
    if st.floor:
        if st.stage0 is None:
            st.stage0, st.progress_t = room[5], t
        if events or row['damage'][0] > 0:   # a new room, a cleared room, or a monster hurt during the step
            st.progress_t = t
        stalled = st.stall > 0 and t - st.progress_t >= st.stall
        done = 2 if o.dead else 1 if (room[5] != st.stage0 and not st.run) else \
            3 if (t >= st.limit or stalled) else 0
    else:
        done = 2 if o.dead else 1 if o.clear else 3 if t >= st.limit else 0
    row['t'], row['done'] = t, done
    if st.items or st.run:
        _items_step(st, row, room[5], float(p['active']), float(p['active_charge']), inv_events)
    return done


def _inventory(o, st):
    """Items: take in the decoder's inventory block when it is new (st.inv_version); returns the EV_ITEM / EV_PILL bits
    of the change (none at the episode's first block)."""
    if o.inv_version == st.inv_version or o.inv_head is None:
        return 0
    first = st.inv_version is None
    st.inv_version = o.inv_version
    n, _, max_charge, tr0, tr1, pill, effect, card, curses = o.inv_head
    counts = {int(i): int(c) for i, c in zip(o.inv['id'], o.inv['count'])}
    events = 0
    if not first:
        old = st.inv_counts or {}
        if any(c > old.get(i, 0) for i, c in counts.items()):
            events |= EV_ITEM
        if (st.pocket[0] != 0 and pill == 0) or (st.pocket[1] != 0 and card == 0):
            events |= EV_PILL
    st.inv_counts = counts
    st.pocket = (pill, card)
    k = min(len(o.inv), INV_CAP)
    st.inv_arr[:] = 0
    st.inv_arr[:k, 0] = np.minimum(o.inv['id'][:k], N_COLLECTIBLE - 1)
    st.inv_arr[:k, 1] = o.inv['count'][:k]
    st.max_charge = max_charge
    st.blind = bool(curses & 64)
    st.pitem[1:] = (min(tr0 & 0x7fff, N_TRINKET - 1), min(tr1 & 0x7fff, N_TRINKET - 1), min(pill, N_PILL_COLOR - 1),
                    min(effect, N_PILL_EFFECT - 1), min(card, N_CARD - 1))
    st.pinv[:] = (0.0, max_charge / 12, pill != 0, effect != 0, card != 0, (tr0 | tr1) != 0, min(len(counts), 32) / 16,
                  0.0, (curses & 1) != 0, (curses & 4) != 0, (curses & 8) != 0, (curses & 64) != 0)
    return events


def _items_step(st, row, stage, active, charge, events):
    """Items / run fields of a record from the episode's inventory state and player 0's active item (FastRow does the
    same after abp_row_encode, from the payload): ROW 'stage', and with items 'inv', 'pitem', 'pinv' and EV_ACTIVE."""
    row['stage'] = min(max(int(stage), 0), 255)
    if not st.items:
        return
    a = int(active)
    if st.prev_active is not None:
        pa, pc = st.prev_active
        if (a == pa and a != 0 and charge < pc) or (pa != 0 and a == 0):
            events |= EV_ACTIVE
    st.prev_active = (a, charge)
    st.pitem[0] = min(max(a, 0), N_COLLECTIBLE - 1)
    st.pinv[0] = min(charge / st.max_charge, 1.0) if st.max_charge > 0 else 0.0
    st.pinv[7] = stage / 12
    row['inv'] = st.inv_arr
    row['pitem'] = st.pitem
    row['pinv'] = st.pinv
    if events:
        row['events'] |= events


# ---- FastRow (2026-10-04): encode_row in C for the sampler workers ----
# abp_turbo's abp_row_encode (analysis/scripts/abplus/abp_turbo.c) writes the same ROW bytes as LeanDecoder.decode +
# encode_row straight from a lean payload (abplus_probe_fast_row.py compares every record of real episodes). It does
# the common case only (a payload whose terrain block, if any, is the same as the last one); a record it hands back (a
# map block, a changed terrain block, the episode's first record, a room change, more than ENT_CAP entities, other door
# records) goes through decode + encode_row here, and the EpisodeState and the decoder are kept in step both ways.
# ISAAC_RL_FAST_ROW=0 switches it off.
FAST_ROW = os.environ.get('ISAAC_RL_FAST_ROW', '1') != '0'
_ROW_FIELDS = ('player', 'ent', 'ent_id', 'doors', 'patch', 'grid', 'map', 'n_ent', 'n_doors', 't', 'hurt', 'damage',
               'done', 'bombs', 'events', 'episode', 'seed', 'group', 'first', 'ent_item', 'pcharge', 'pchar')
_ROW_OFFS_ITEMS = np.array([ROW.fields[k][1] for k in _ROW_FIELDS], np.int64)
_ROW_OFFS = _ROW_OFFS_ITEMS.copy()
_ROW_OFFS[_ROW_FIELDS.index('ent_item')] = -1   # without items abp_row_encode leaves ent_item alone (version 3)
_ROW_VERSION = 6   # abp_row_encode_version's layout (4: 2026-10-07, the 38-double player record and pcharge; 5:
#                    2026-10-08, abp-0.2.16's 144-byte entity records and ENT_F = 56; 6: 2026-10-08, abp-0.2.17's
#                    39-double player record (PlayerType) and pchar)
# ctx slots (abp_turbo.c RC_*)
(RC_T, RC_LIMIT, RC_FLOOR, RC_STALL, RC_PROGRESS, RC_STAGE0, RC_HP0, RC_PREV_DMG, RC_PREV_HP, RC_HAVE_PREV, RC_ROOM,
 RC_WAS_CLEAR, RC_ORIGIN_X, RC_ORIGIN_Y, RC_N_EXITS, RC_EXITS) = range(16)
RC_TVERSION = RC_EXITS + 16
(RC_TSEEN, RC_HURT, RC_EPISODE, RC_SEED, RC_GROUP, RC_FIRST, RC_RUN, RC_ITEMS, RC_COUNT) = \
    range(RC_TVERSION + 1, RC_TVERSION + 10)
# player 0's active item and charge, and the room's stage, in a raw lean payload (abplus_lean: header 16, room 48,
# totals 32, then the player's doubles: active is field 24)
_PAYLOAD_ACTIVE = struct.Struct('<dd')
_PAYLOAD_STAGE = struct.Struct('<i')
_LIBS = {}


def _address(b):
    """The address of a bytes object's buffer (no copy)."""
    return ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value


def fast_row_function(path):
    """abp_row_encode of the libabp_turbo.so at path (ctypes), None when off, missing or of another layout."""
    if not FAST_ROW or not path or not os.path.isfile(path):
        return None
    if path not in _LIBS:
        fn = None
        try:
            lib = ctypes.CDLL(path, mode=getattr(os, 'RTLD_LOCAL', 0))
            version = lib.abp_row_encode_version
            version.restype = ctypes.c_int
            if version() == (_ROW_VERSION << 16) | (RC_COUNT << 8) | len(_ROW_FIELDS) and ENT_CAP == 64 and \
                    ENT_F == 56 and ENT_F0 == 33 and PCHARGE_F == 12 and N_CHAR == 32 and \
                    DOOR_CAP == 8 and DOOR_F == 7 and PATCH == 9 and GRID == (7, 16, 28) and MAP == (8, 13, 13):
                fn = lib.abp_row_encode
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_char_p, ctypes.c_long] + [ctypes.c_void_p] * 7
        except (OSError, AttributeError):
            fn = None
        _LIBS[path] = fn
    return _LIBS[path]


class FastRow:
    """One episode's records through abp_row_encode (fn from fast_row_function). encode(item, row, row_address, t):
    item is a raw lean payload (bytes) or a LeanObs; returns the done code like encode_row and sets self.hurt (half
    hearts were lost in the step that led here), self.obs (the LeanObs when the record went through encode_row, else
    None). Writes episode, seed, group and first too (set_meta)."""

    def __init__(self, fn, st, decoder, episode=0, seed=0, group=0):
        self.fn, self.st, self.decoder = fn, st, decoder
        self.ctx = np.zeros(RC_COUNT, np.float64)
        self._ctx_p = self.ctx.ctypes.data
        self._offs_p = (_ROW_OFFS_ITEMS if st.items else _ROW_OFFS).ctypes.data
        self.post = st.items or st.run   # items / run fields written here after abp_row_encode
        self.refs = np.zeros(4, np.int64)   # the last terrain block's bytes, the door records of the grid: address, length
        self._refs_p = self.refs.ctypes.data
        self._keep = (None, None)
        self._ptrs = (None, None, None)
        self.hurt = False
        self.obs = None
        self.fast = self.slow = 0
        self.reasons = {}   # abp_row_encode's hand-back codes: how often
        c = self.ctx
        c[RC_LIMIT], c[RC_FLOOR], c[RC_STALL] = st.limit, float(st.floor), st.stall
        self.set_meta(episode, seed, group)
        self._from_st()

    @property
    def room(self):
        """The room index of the last record encoded (its lean observation's room[4])."""
        return int(self.ctx[RC_ROOM])

    def set_meta(self, episode, seed, group):
        c = self.ctx
        c[RC_EPISODE], c[RC_SEED], c[RC_GROUP] = episode, seed, group
        self.meta = (episode, seed, group)

    def encode(self, item, row, row_address, t):
        c = self.ctx
        if type(item) is bytes:
            c[RC_T] = t
            c[RC_FIRST] = t == 0
            g, pd, m = self._ptrs
            done = self.fn(item, len(item), row_address, self._ctx_p, g, pd, m, self._offs_p, self._refs_p) \
                if g else -7
            if done >= 0:
                self.fast += 1
                self.hurt = c[RC_HURT] != 0
                self.obs = None
                if c[RC_TSEEN]:   # the same terrain block again: the decoder's version follows it
                    d = self.decoder
                    d.terrain_version = int(c[RC_TVERSION])
                    d.terrain['version'] = d.terrain_version
                if self.post:   # encode_row's _items_step on the payload's values
                    active, charge = _PAYLOAD_ACTIVE.unpack_from(item, 96 + 8 * 24)
                    _items_step(self.st, row, _PAYLOAD_STAGE.unpack_from(item, 36)[0], active, charge, 0)
                return done
            self.reasons[done] = self.reasons.get(done, 0) + 1
            item = self.decoder.decode(item)
        self.slow += 1
        self._to_st()
        done = encode_row(item, self.st, row, t)
        row['episode'], row['seed'], row['group'], row['first'] = self.meta[0], self.meta[1], self.meta[2], t == 0
        self.hurt = bool(row['hurt'][0] > 0)
        self.obs = item
        self._from_st()
        return done

    def _to_st(self):
        c, st = self.ctx, self.st
        if c[RC_HAVE_PREV]:
            st.previous = (float(c[RC_PREV_DMG]), float(c[RC_PREV_HP]))
            st.room = int(c[RC_ROOM])
            st.was_clear = bool(c[RC_WAS_CLEAR])
            st.progress_t = int(c[RC_PROGRESS])

    def _from_st(self):
        c, st = self.ctx, self.st
        c[RC_HAVE_PREV] = st.previous is not None and st.room is not None and st.hp0 is not None
        if c[RC_HAVE_PREV]:
            c[RC_PREV_DMG], c[RC_PREV_HP] = st.previous
            c[RC_ROOM], c[RC_WAS_CLEAR], c[RC_HP0] = st.room, bool(st.was_clear), st.hp0
        c[RC_PROGRESS] = st.progress_t
        c[RC_STAGE0] = np.nan if st.stage0 is None else st.stage0
        c[RC_RUN] = float(st.run)
        c[RC_ITEMS] = (2.0 if st.blind else 1.0) if st.items else 0.0
        c[RC_ORIGIN_X], c[RC_ORIGIN_Y] = st.origin
        c[RC_N_EXITS] = min(len(st.exits), 8)
        for i, (ex, ey) in enumerate(st.exits[:8]):
            c[RC_EXITS + 2 * i], c[RC_EXITS + 2 * i + 1] = ex, ey
        terrain = self.decoder.terrain_raw
        doors = st.version[1] if st.version is not None else None
        arrays = (st.grid, st.padded, st.map)
        if len(st.exits) <= 8 and type(terrain) is bytes and type(doors) is bytes and \
                all(a is not None and a.dtype == np.uint8 and a.flags['C_CONTIGUOUS'] for a in arrays) and \
                st.grid.shape == GRID and st.padded.shape == (GRID[0], GRID[1] + PATCH - 1, GRID[2] + PATCH - 1) and \
                st.map.shape == MAP:
            self._keep = (terrain, doors, arrays)
            self.refs[:] = (_address(terrain), len(terrain), _address(doors), len(doors))
            self._ptrs = tuple(a.ctypes.data for a in arrays)
        else:   # every record goes through encode_row
            self._keep = (None, None)
            self._ptrs = (None, None, None)
