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
import sys
import struct

import numpy as np

from .transformer_obs import terrain_channels
from . import visible_producer as _VP

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
# (2026-11-19, the long run's nohit term; only with ISAAC_RL_NOHIT=1, set by train_tok --reward nohit=...: encode_row,
#  the Python path a visible-v3 worker takes for every record) a room cleared that had monsters (their HP > 0 at a
#  record in it, or damage dealt in it) and in which the episode was never hurt -- the engine's room index on the
#  floor instance, a hurt step that changed rooms charged to the room left -- once per room and episode
EV_NOHIT = 128
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
    # 2026-10-13 (B23, tok_pickup.py; appended, the fields above keep their offsets): 0 in every record unless the
    # trainer's --start-build-mode pedestal / --pickup-force is on. pforce: the move (1 .. 8) the worker forces at this
    # record (the actor substitutes it for the sampled move and answers shoot 0; the learner's flag of a forced
    # decision), 0 = the policy's own; pmode: 1 in every record of an episode drawn as forced; bped: the collectible ids
    # of the episode's start build spawned as pedestals (0 = none). Never a policy input.
    ('pforce', np.uint8),
    ('pmode', np.uint8),
    ('bped', np.int16, (3,)),
    # 2026-10-16 (B35, observation v2, bridge abp-0.2.18; appended, the fields above keep their offsets), written in
    # every mode; policy inputs only for TokPolicy(obs_prices / obs_hearts / obs_room_type) (train_tok --obs-v2 1):
    #   ent_price: per entity its price (abplus_lean ENTITY 'price': EntityPickup.Price of a pickup, 0 otherwise),
    #     clipped to int16; the policy's PRICE_F price columns are price_columns() of it;
    #   phearts: player 0's black half hearts / 12, eternal hearts (0 / 1), bone hearts / 6 (PHEART_F; red and soul
    #     half hearts / 12 are already player columns 4 and 5);
    #   room_type: the room's RoomType (the lean room block's room:GetType()) clipped to 0 .. N_ROOM_TYPE - 1.
    ('ent_price', np.int16, (ENT_CAP, 1)),
    ('phearts', np.float32, (3,)),
    ('room_type', np.uint8),
    # (2026-11-20, reward-rewind-design.md V2; in room_type's alignment padding: every other offset and the itemsize
    #  unchanged) bits 0-6: Glowing Hour Glass restores since the record before (the game's restore count, abp_native_
    #  rewinds; 0 at an episode's first record and when the count went down); bit 7 (REWIND_CHECKPOINT): this record
    #  changed rooms or floors without a rewind -- the game stored its rewind state on the way (Game::ChangeRoom /
    #  StartStageTransition), the reward ledger's checkpoint. Written by every encoder path (FastRow's fast records: 0).
    #  Dumps without row_schema >= 8 carry padding bytes here: treat them as unknown (0).
    ('rewind', np.uint8),
    # Full new visible contract, independently versioned. Legacy fields above are
    # kept for old policies and reward/statistics only; new Spatial never reads them.
    ('visible_schema', np.int32),
    ('visible_bytes', np.int32),
    ('visible_frame', np.uint8, (131072,)),
], align=True)
# B45 (2026-11-09): compact records. ROW keeps a 128-KiB visible_frame budget (the workers' shared-memory slots and
# rings); stored decisions use ROWC: every ROW field but visible_frame at the same offset, plus visible_off, the byte
# offset of the record's frame (visible_bytes long) in a FrameArena. A rollout's ~11 KB frames then cost ~11 KB each,
# not 128 KiB (B42: 151,880-byte records, ~5 GB per 32,768-decision rollout).
ROW_FRAME_AT = ROW.fields['visible_frame'][1]
assert ROW.fields['rewind'][1] == 20793 and ROW.itemsize == 151880   # (in the padding: no other offset moved)
REWIND_CHECKPOINT = 128
REWIND_COUNT = 127
ROW_SCHEMA = 8   # (2026-11-20) the ROW layout revision dumps record ('row_schema'); 8: the rewind field
# (2026-11-20, the memory ledger) ISAAC_RL_RING_COMPACT=1 (default off): the workers' shared teacher and V2 rings
# (tok_sampler's teach_rows, tok_teacher2's v2_rows: whole ROW slots, 151,880 bytes) are written and read only up to
# each record's frame end -- ROW_FRAME_AT + visible_bytes, the bytes every reader uses -- so a slot's frame-budget
# pages past it are never touched and never become resident (a read of a shared-memory hole allocates a page too).
# A record taken from a ring then has zeros past its frame end where the whole-slot copy had the slot's stale bytes
# of earlier records (no reader reads them: tests/test_ring_compact.py). (Codex round 25) What it saves is resident
# shared pages, not address space: a page a longer frame (or a whole-slot access) once touched stays resident. Every
# process reads the switch from its own environment; the workers refuse a value other than the trainer's
# (tok_sampler.check_ring_mode). Diagnostic dumps of the rings (B38 dump_job) differ past the frames between the modes.
RING_COMPACT = os.environ.get('ISAAC_RL_RING_COMPACT', '0') == '1'
# (2026-11-20, the memory ledger) The sampler's workers do not load torch: they did (about 220 MB private per process:
# libtorch_cuda, cuBLAS / cuBLASLt's static data, ...) only for names they took from structured_input, a torch module
# that re-exports them -- tok_sampler.attach's VERSION, write_visible's write_frame, visible_native's VERSION /
# FRAME_BYTES / SCHEMA / observation_frame -- now taken from their own torch-free modules (schema_v2, visible_producer:
# the same objects), and splitmix64's finaliser lives here (mix64). Every worker reports whether it holds torch with its
# ring confirmation (the trainer logs how many do) and in its loop (stats torch_held, progress worker_torch, a warning
# line: never the run's end). ISAAC_RL_WORKER_LEAN=1 (default off) enforces it at the start: the trainer refuses a
# worker holding torch (or of the other setting) before any sampling. The trainer and the learner load torch as
# before. (tests/test_worker_lean.py)
WORKER_LEAN = os.environ.get('ISAAC_RL_WORKER_LEAN', '0') == '1'


def mix64(z):
    """splitmix64's finaliser on a uint64 array (wrapping arithmetic). (tok_policy._mix64 is this function: here so a
    worker's tok_branch.crn_key takes it without importing torch.)"""
    with np.errstate(over='ignore'):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))
ROWC = np.dtype([(name, ROW.fields[name][0]) for name in ROW.names if name != 'visible_frame'] +
                [('visible_off', np.int64)], align=True)
assert all(ROWC.fields[n][1] == ROW.fields[n][1] for n in ROW.names if n != 'visible_frame')
assert ROWC.fields['visible_off'][1] >= ROW_FRAME_AT


class FrameArena:
    """Append-only byte store of visible frames (8-byte aligned starts); grows by reallocation. Records hold their
    frame's offset (ROWC visible_off). Readers take .buf once the writer is done (a rollout's chunks: after it).
    buf (A-6, 2026-11-19): a given byte array (e.g. shared memory the learner process maps) used as it is -- never
    reallocated (a full arena raises: size it for the worst case; untouched pages of a memfd cost nothing)."""
    ALIGN = 8

    def __init__(self, nbytes=1 << 20, buf=None):
        self.fixed = buf is not None
        self.buf = buf if buf is not None else np.zeros(max(int(nbytes), 1024), np.uint8)
        self.used = 0
        self.peak = 0   # (A-6, Codex round 16: the high-water mark -- reset keeps the pages written, shared memory too)

    def _room(self, need):
        if need <= len(self.buf):
            return
        if self.fixed:
            raise RuntimeError(f'FrameArena: {need} bytes past a fixed arena of {len(self.buf)}')
        grown = np.zeros(max(need, int(len(self.buf) * 1.5)), np.uint8)
        grown[:self.used] = self.buf[:self.used]
        self.buf = grown

    def reset(self):
        self.used = 0

    def append(self, rows):
        """Frames of ROW records `rows` (their visible_bytes) appended; returns their offsets (int64)."""
        sizes = np.asarray(rows['visible_bytes'], np.int64)
        padded = (sizes+self.ALIGN-1)//self.ALIGN*self.ALIGN
        offs = self.used+np.concatenate(([0], np.cumsum(padded)[:-1])).astype(np.int64)
        need = self.used+int(padded.sum())
        self._room(need)
        frames = rows['visible_frame']
        for i in range(len(sizes)):
            n = int(sizes[i])
            if n:
                self.buf[offs[i]:offs[i]+n] = frames[i, :n]
        self.used = need
        self.peak = max(self.peak, need)
        return offs


    def append_from(self, buf, offs, sizes):
        """Frames of another arena's records (their offsets / sizes in buf) appended; returns the new offsets."""
        sizes = np.asarray(sizes, np.int64)
        padded = (sizes+self.ALIGN-1)//self.ALIGN*self.ALIGN
        new = self.used+np.concatenate(([0], np.cumsum(padded)[:-1])).astype(np.int64)
        need = self.used+int(padded.sum())
        self._room(need)
        for o, n, d in zip(np.asarray(offs, np.int64), sizes, new):
            self.buf[d:d+n] = buf[o:o+n]
        self.used = need
        self.peak = max(self.peak, need)
        return new


FRAME_W = ROW.fields['visible_frame'][0].itemsize   # (the frame field's width: 131,072 bytes)


def row_frame_end(row1):
    """The bytes of a 1-record ROW array every reader uses: every field before the frame, then visible_bytes of it.
    (Codex round 25) A record without a frame (visible_schema 0: its visible_bytes may be a reused slot's) ends at the
    frame; a record with one whose length is outside [0, FRAME_W] is refused (ValueError), never clamped."""
    if int(row1['visible_schema'][0]) == 0:
        return ROW_FRAME_AT
    vb = int(row1['visible_bytes'][0])
    if not 0 <= vb <= FRAME_W:
        raise ValueError(f'ring record: visible_bytes {vb} outside [0, {FRAME_W}] '
                         f'(visible_schema {int(row1["visible_schema"][0])})')
    return ROW_FRAME_AT + vb


def ring_put(dst1, src1):
    """A record (src1, a 1-record ROW array) into a ring slot (dst1, the slot's 1-record view): with RING_COMPACT its
    bytes up to its frame end only, otherwise the whole record (as before)."""
    if not RING_COMPACT:
        dst1[0] = src1[0]
        return
    n = row_frame_end(src1)
    dst1.view(np.uint8).reshape(-1)[:n] = src1.view(np.uint8).reshape(-1)[:n]


def ring_take(ring, i, slot):
    """A copy (a ROW scalar) of the record in ring[i, slot]: with RING_COMPACT its bytes up to its frame end and zeros
    past them (the slot's tail is not read), otherwise the whole slot (ring[i, slot].copy(), as before). A torn read
    (the slot reused while it is copied, its length included) is the caller's to detect: the slot's generation
    (tok_sampler.ring_open / ring_publish), read before and after."""
    if not RING_COMPACT:
        return ring[i, slot].copy()
    src1 = ring[i, slot:slot + 1]
    n = row_frame_end(src1)
    out = np.zeros(1, ROW)
    out.view(np.uint8)[:n] = src1.view(np.uint8).reshape(-1)[:n]
    return out[0]


def compact_rows(rows, arena):
    """ROW records -> ROWC records whose frames are appended to `arena` (a FrameArena)."""
    rows = np.ascontiguousarray(rows)
    out = np.zeros(len(rows), ROWC)
    out.view(np.uint8).reshape(len(rows), ROWC.itemsize)[:, :ROW_FRAME_AT] =         rows.view(np.uint8).reshape(len(rows), ROW.itemsize)[:, :ROW_FRAME_AT]
    out['visible_off'] = arena.append(rows)
    return out


def take_prefix(src, slots, out):
    """(A-4, 2026-11-19) src[slots] into out[:len(slots)] (ROW arrays; out reused between calls), each record's bytes
    up to the end of its frame only -- every field before visible_frame and the frame's visible_bytes (rounded up to
    whole floats); the frame field's tail past it keeps what out held (never read: decode_frames_device / the lazy
    decode read a frame within its length, compact_rows / store copy it no further). A record whose visible_bytes is
    outside 0 .. the frame field is copied whole (the checks downstream refuse it). Returns out[:len(slots)]; the
    same values as src[slots] on every field byte and each frame (tests/test_take_prefix.py)."""
    n = len(slots)
    dst = out[:n]
    sb = src.reshape(-1).view(np.uint8).reshape(-1, ROW.itemsize)
    db = dst.view(np.uint8).reshape(n, ROW.itemsize)
    nb = np.asarray(src['visible_bytes'], np.int64)[slots]
    cap = ROW.itemsize - ROW_FRAME_AT
    for i, (s_, k) in enumerate(zip(np.asarray(slots, np.int64).tolist(), nb.tolist())):
        e = ROW_FRAME_AT + ((k + 3) & ~3) if 0 <= k <= cap else ROW.itemsize
        db[i, :e] = sb[s_, :e]
    return dst


def compact_copy(rows):
    """(ROWC records, their own frame bytes) of ROW records: a self-contained compact copy (teacher records)."""
    arena = FrameArena(int(np.asarray(rows['visible_bytes'], np.int64).sum())+64)
    return compact_rows(rows, arena), arena.buf[:arena.used]


def expand_rows(rows, frames):
    """ROW records of ROWC records `rows` and their frame bytes `frames` (tok_stream, tools); ROW records pass."""
    if 'visible_frame' in rows.dtype.names:
        return rows
    out = np.zeros(len(rows), ROW)
    out.view(np.uint8).reshape(len(rows), ROW.itemsize)[:, :ROW_FRAME_AT] =         np.ascontiguousarray(rows).view(np.uint8).reshape(len(rows), ROWC.itemsize)[:, :ROW_FRAME_AT]
    for i, (o, n) in enumerate(zip(rows['visible_off'], rows['visible_bytes'])):
        out['visible_frame'][i, :n] = frames[o:o+n]
    return out


PHEART_F = 3          # 2026-10-16 (B35): ROW 'phearts'
PRICE_F = 4           # 2026-10-16 (B35): price_columns() per entity
N_ROOM_TYPE = 32      # 2026-10-16 (B35): room types the policy's room-type table has room for (AB+ 0 .. 23, Repentance+
#                       0 .. 29)
N_CHAR = 32           # 2026-10-08: PlayerType ids the policy's character input has room for (AB+ 0 .. 17, Repentance+ 0 .. 40
#                       would need more)
DONE_BRANCH = 4       # ROW 'done' of a branch episode's last record when it ends for the branch's own reasons (the
#                       floor's exit or a boss cleared in run mode, or the branch's time cap): a truncation, the
#                       trainer bootstraps from the value of that record (train_tok: the boot decisions)
KIND_COLUMN = 16      # entity columns 16..22: one-hot of pack_lean's kind (0 other, 1 tear, 2 projectile, 3 laser,
#                       4 bomb, 5 pickup, 6 NPC)
PRICE_COINS = 15.0    # 2026-10-16 (B35): a coin price's scale (AB+ Room::GetShopItemPrice: 15 a collectible, 5 a pickup,
#                       3 a heart; halved by Steam Sale)


def price_columns(price):
    """2026-10-16 (B35): the policy's price columns of ROW 'ent_price' values (an array of any shape, numpy or torch;
    float32 out with one more trailing axis of PRICE_F): coin price / PRICE_COINS (0 unless the price is in coins),
    heart containers the price takes (1 at -1, 2 at -2, else 0), soul-heart price (1 at -3: three soul hearts, else 0),
    for sale (price != 0; a free shop item, -1000, is for sale at no cost). Price semantics: AB+ Entity_Pickup::SetPrice
    and Room::GetShopItemPrice (decompiled, analysis/abplus-linux): > 0 coins; devil-deal prices -1 (one heart
    container), -2 (two), -3 (three soul hearts, when the player has no heart containers); -1000 free (Steam Sale)."""
    if isinstance(price, np.ndarray):
        p = price.astype(np.float32)
        cols = [np.maximum(p, 0) / np.float32(PRICE_COINS), (p == -1) + 2 * (p == -2), p == -3, p != 0]
        return np.stack([c.astype(np.float32) for c in cols], -1)
    import torch
    p = price.float()
    cols = [p.clamp(min=0) / PRICE_COINS, (p == -1).float() + 2 * (p == -2).float(), (p == -3).float(),
            (p != 0).float()]
    return torch.stack(cols, -1)


def black_half_hearts(soul, mask):
    """2026-10-16 (B35): player 0's black half hearts from GetSoulHearts (soul half hearts, black ones included) and
    GetBlackHearts (AB+: a bit mask, bit k set when soul-heart slot k, half hearts 2k + 1 and 2k + 2, is black;
    Entity_Player::IsBlackHeart reads bit (i - 1) >> 1 of +0x25c4). abp_turbo.c abp_row_encode computes the same."""
    soul, mask = int(soul), int(mask) & 0xFFFFFFFF
    return sum(min(2, soul - 2 * k) for k in range(32) if (mask >> k) & 1 and 2 * k < soul)


class ClearCounter:
    """(2026-11-13, tok_floor's main episodes; --cf-per-clear) a room cleared after a fight: step(new_room, damage,
    events) per record returns (cleared: EV_CLEARED, fought: cleared and the main episode hurt the room's monsters
    since it entered the room -- the records' damage summed, a room change resetting it). An EV_CLEARED without a
    hurt monster (an empty room whose clear flag came a frame late, say) is cleared but not fought."""

    def __init__(self):
        self.damage = 0.0

    def step(self, new_room, damage, events):
        if new_room:
            self.damage = 0.0
        self.damage += float(damage)
        cleared = bool(int(events) & EV_CLEARED)
        return cleared, cleared and self.damage > 0


def stall_flags(mode):
    """(stall_pickups, stall_items) of EpisodeState for --stall-progress `mode` ('events', 'pickups', 'items', 'first'):
    the one rule the floor workers' episodes (tok_floor) and the teacher's branches (tok_teacher2.play_branch) share.
    'first' (2026-11-20, reward-rewind-design.md section 3.3 v1): EpisodeState.stall_first instead (stall_first(mode))."""
    return mode in ('pickups', 'items'), mode == 'items'


def stall_first(mode):
    """--stall-progress first: progress is only what this episode does for the first time, never rolled back by a
    rewind -- a room entered, a room cleared, a room's monsters hurt (each per (floor instance, room index)), an id held
    (the items mode's seen sets), a floor reached; a repeated clear (a rewind, D7) or a resource is not progress."""
    return mode == 'first'


class EpisodeState:
    """What the encoder keeps between the records of one episode."""

    def __init__(self, limit, floor=False, stall=0, run=False, items=False, stall_pickups=False, stall_items=False,
                 stall_first=False, rewind=False):
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
        # B40 (2026-10-18, --stall-progress pickups): a gain in player 0's coins, keys, bombs or health units is progress
        # too; prev_res: those four (floats) at the record before (None: none yet)
        self.stall_pickups = bool(stall_pickups)
        # (2026-11-12, --stall-progress items: stall_pickups and also a collectible gained or a trinket / pill / card
        #  acquired -- _inventory's inv_gain; inventory blocks always take the Python path, abp_row_encode hands them back)
        self.stall_items = bool(stall_items)
        self.inv_gain = False
        # (2026-11-13: an acquisition is progress only the first time the episode holds that id -- swapping two active
        #  items on a pedestal back and forth raised a collectible's count on every swap and the stall never came.
        #  Ids held so far this episode, the first inventory block's included: collectibles, trinkets, pill colors, cards)
        self.seen_items, self.seen_tr, self.seen_pill, self.seen_card = set(), set(), set(), set()
        self.prev_res = None
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
        # (2026-11-19) EV_NOHIT: per (floor instance, room index) [had monsters, hurt in it, rewarded]
        self.nohit_on = os.environ.get('ISAAC_RL_NOHIT') == '1'
        self.nohit_rooms = {}
        self.floor_n = 0
        # (2026-11-20, reward-rewind-design.md) rewinds: the game's restore count at the record before (None: not
        # known yet), the floor instance (+1 per floor reached), the state a rewind restores (taken at the episode's
        # first record and at every checkpoint record: visited, nohit_rooms, floor_n, floor_k, stage0), rewinds seen
        # (Codex round 20: the behaviour only with rewind on -- off, the count is not looked at, ROW 'rewind' is 0 and
        #  every event, hurt, damage, state update and the stall are b53f14c's)
        self.rewind_on = bool(rewind)
        self.rw_count = None
        self.floor_k = 0
        self.ck = None
        self.rewinds = 0
        # --stall-progress first (stall_first): never rolled back
        self.stall_first = bool(stall_first)
        self.ever_rooms, self.ever_cleared, self.ever_hurt = set(), set(), set()
        self.prev_maxc = 0   # items: the active item's max charge at the record before (EV_ACTIVE)


def _checkpoint(st):
    st.ck = (set(st.visited), {k: list(v) for k, v in st.nohit_rooms.items()}, st.floor_n, st.floor_k, st.stage0)


def _restore(st):
    if st.ck is not None:
        v, nh, st.floor_n, st.floor_k, st.stage0 = st.ck
        st.visited, st.nohit_rooms = set(v), {k: list(x) for k, x in nh.items()}


def nohit_step(st, room_idx, next_floor, hurt, damage, monsters_hp, events):
    """EV_NOHIT's bookkeeping for one record (encode_row, before st.room moves on): the hurt of the step charged to the
    room left when the step changed rooms (else to this one), this room marked as having had monsters when their HP is
    > 0 or damage was dealt; EV_NOHIT added to `events` on the clear of such a room never hurt in, once per room.
    Rooms are (floor instance, the engine's room index): no visible map involved. Returns the events."""
    left = (st.floor_n, st.room)
    if next_floor:
        st.floor_n += 1
    here = (st.floor_n, room_idx)
    nh = st.nohit_rooms
    if st.room is not None and hurt > 0:
        nh.setdefault(left if (room_idx != st.room or next_floor) else here, [False, False, False])[1] = True
    e = nh.setdefault(here, [False, False, False])
    if monsters_hp > 0 or damage > 0:
        e[0] = True
    if events & EV_CLEARED and e[0] and not e[1] and not e[2]:
        events |= EV_NOHIT
        e[2] = True
    return events


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


V3_FRAMES = [0, 0]   # (seq-v1 step 3) this process's v3 frames written: native, the reference encoder's


def obs_visible_conflict(obs_visible, v3_raw):
    """(seq-v1, the coordinator 2026-11-18) --obs-visible 0 (no visible frame at all) and ISAAC_RL_VISIBLE_V3 set to a
    mode (a v3 worker writes a v3 frame for every record) contradict each other: write_visible would take
    --obs-visible 0 first and write nothing. train_tok / eval_tok refuse the pair at their argument parsing: the
    message, or None when the two agree."""
    from .visible_producer import configured_mode
    mode = configured_mode(v3_raw)
    if not obs_visible and mode:
        return (f'--obs-visible 0 (the workers write no visible frame) and ISAAC_RL_VISIBLE_V3={v3_raw} (mode {mode}: '
                f'a v3 frame for every record) contradict each other: set one of them')
    return None


def write_visible(o, row):
    """encode_row's visible frame: v3 (the native frame or the raw block's reference frame), else v2, else none
    (visible_schema 0); a worker configured for v3 (visible_producer.VISIBLE_V3_MODE) refuses a record without a v3
    frame. --obs-visible 0 (ISAAC_NO_VISIBLE=1) writes no frame at all, v3 or v2 (visible_schema 0). True when a v3
    frame was written (its stall columns are encode_row's to fill)."""
    vis = getattr(o, 'visible', None)
    v3 = False
    is_v3 = getattr(o, 'visible_v3_frame', None) is not None or (isinstance(vis, dict) and
                                                                vis.get('schema') == 'isaac-visible-raw-v3')
    if is_v3 and not _VP.VISIBLE_V3_MODE:   # (2026-11-18, Codex round 9: a mode 0 worker, which sends no mode query,
        # refuses a v3 record before the ROW -- an ordinary protocol error: the episode's handlers take it, as 05f1622
        # took a bridge in another mode -- never a v3 frame in an old arm's ROW, never the worker's end; seq-v1's
        # --obs-visible 0 below does not hide it)
        raise ValueError('visible-v3: a v3 record (%s) in a worker not configured for v3 (ISAAC_RL_VISIBLE_V3=0): '
                         'the game runs another visible mode' % ('native frame' if getattr(o, 'visible_v3_frame',
                                                                                          None) is not None
                                                                 else 'raw v3 block'))
    if os.environ.get('ISAAC_NO_VISIBLE') == '1':   # (train_tok / eval_tok --obs-visible 0: no frame, v3 or v2)
        row['visible_schema'] = 0
    elif is_v3:
        # visible-v3 (package C): the native frame as it came, else (mode 1, or the native side handed back) the
        # reference encoder's frame of the raw block; the stall columns after the progress update below
        from .visible_producer import write_v3
        native = getattr(o, 'visible_v3_frame', None) is not None
        v3 = write_v3(row, o)
        if v3:
            V3_FRAMES[0 if native else 1] += 1
    elif _VP.VISIBLE_V3_MODE:   # (2026-11-18, Codex round 2: a v3 worker writes a v3 frame for every record or stops)
        raise _VP.VisibleV3Fatal('visible-v3: this worker is configured for mode %d, the record has no v3 frame and %s'
                         % (_VP.VISIBLE_V3_MODE, 'no visible block' if vis is None else 'a visible block of schema %r'
                            % (vis.get('schema') if isinstance(vis, dict) else type(vis).__name__)))
    elif vis is not None:
        from .visible_producer import write_frame   # (structured_input re-exports it, importing torch: not here)
        from .visible_native import frame   # B45: the C encoder (observation_frame's bytes) when available
        write_frame(row,frame(o))
    else:
        row['visible_schema'] = 0
    return v3


def encode_row(o, st, row, t):
    """Write the lean observation `o` into row (a length-1 view of a ROW array). Sets everything but episode, seed,
    group and first. Returns the done code."""
    v3 = write_visible(o, row)
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
    # 2026-10-16 (B35, abp_row_encode version 7): the heart split and the room's type
    row['phearts'][0] = (black_half_hearts(p['soul'], p['black']) / 12, p['eternal'], p['bone'] / 6)
    row['room_type'] = min(max(int(room[0]), 0), N_ROOM_TYPE - 1)
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
        row['ent_price'][0][:n, 0] = np.clip(e['price'], -32768, 32767)   # 2026-10-16 (B35)
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
        row['ent_price'][0, n] = 0
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
    # (2026-11-20) Glowing Hour Glass restores since the record before (the bridge's count block; None: not sent)
    rw = getattr(o, 'rewinds', None) if st.rewind_on else None
    rewound = 0
    if rw is not None:
        if st.rw_count is not None and st.previous is not None and rw > st.rw_count:
            rewound = min(rw - st.rw_count, REWIND_COUNT)
        st.rw_count = rw
    if rewound:   # the game went back to its last store: so does what a rewind restores here
        _restore(st)
        st.rewinds += rewound
    # a run's next floor (2026-10-06): a room change even where the new start room has the old room's index
    next_floor = st.run and st.stage0 is not None and room[5] != st.stage0
    if next_floor:
        st.visited = set()
        st.floor_k += 1
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
    if rewound:   # (2026-11-20) the rewind's own step: no event, no hurt, no damage (its jumps are not this timeline's)
        events = 0
        row['hurt'] = row['damage'] = 0.0
    elif st.nohit_on:   # (2026-11-19) EV_NOHIT
        events = nohit_step(st, room_idx, next_floor, float(row['hurt'][0]), float(row['damage'][0]),
                            float(o.monsters_hp), events)
    changed = st.room is not None and (room_idx != st.room or next_floor)
    st.room, st.was_clear = room_idx, bool(o.clear)
    row['events'] = events
    st.previous = (o.damage_taken, o.monsters_hp)
    # (2026-11-20) the rewind field (bit 7: a checkpoint -- a room or floor change, not a rewind)
    row['rewind'] = (rewound | (REWIND_CHECKPOINT if changed and not rewound else 0)) if st.rewind_on else 0
    if st.floor:
        if st.stage0 is None:
            st.stage0, st.progress_t = room[5], t
        if st.rewind_on and (st.ck is None or (changed and not rewound)):   # what a rewind restores: here, the first
            _checkpoint(st)
        if st.stall_first:   # (--stall-progress first: what this episode does for the first time; never rolled back)
            key = (st.floor_k, room_idx)
            prog = bool(events & EV_EXIT)
            if not rewound:
                if key not in st.ever_rooms:
                    st.ever_rooms.add(key)
                    prog = True
                if events & EV_CLEARED and key not in st.ever_cleared:
                    st.ever_cleared.add(key)
                    prog = True
                if row['damage'][0] > 0 and key not in st.ever_hurt:
                    st.ever_hurt.add(key)
                    prog = True
                prog = prog or st.inv_gain
            if prog:
                st.progress_t = t
        elif events or row['damage'][0] > 0:   # a new room, a cleared room, or a monster hurt during the step
            st.progress_t = t
        if st.stall_pickups and not st.stall_first:   # B40: or a resource gained (coins, keys, bombs, health units; abp_row_encode the same)
            res = (float(p['coins']), float(p['keys']), float(p['bombs']), float(units))
            if st.prev_res is not None and any(a > b for a, b in zip(res, st.prev_res)):
                st.progress_t = t
            st.prev_res = res
        if st.stall_items and st.inv_gain and not st.stall_first:   # (--stall-progress items: an item, trinket, pill or card acquired)
            st.progress_t = t
        stalled = st.stall > 0 and t - st.progress_t >= st.stall
        done = 2 if o.dead else 1 if (room[5] != st.stage0 and not st.run) else \
            3 if (t >= st.limit or stalled) else 0
    else:
        done = 2 if o.dead else 1 if o.clear else 3 if t >= st.limit else 0
    if v3:   # P 40 / 53: the share of the stall limit used up, with this record's progress counted
        from .visible_producer import v3_backfill_stall
        known = st.floor and st.stall > 0
        v3_backfill_stall(row, min((t - st.progress_t) / st.stall, 1.0) if known else 0.0, 1.0 if known else 0.0)
    row['t'], row['done'] = t, done
    if st.items or st.run:
        # (2026-11-20) a rewind's step: its inventory jumps are not events; the use of the hour glass is (EV_ACTIVE)
        _items_step(st, row, room[5], float(p['active']), float(p['active_charge']),
                    EV_ACTIVE if rewound else inv_events)
    return done


def _inventory(o, st):
    """Items: take in the decoder's inventory block when it is new (st.inv_version); returns the EV_ITEM / EV_PILL bits
    of the change (none at the episode's first block)."""
    st.inv_gain = False
    if o.inv_version == st.inv_version or o.inv_head is None:
        return 0
    first = st.inv_version is None
    st.inv_version = o.inv_version
    n, _, max_charge, tr0, tr1, pill, effect, card, curses = o.inv_head
    counts = {int(i): int(c) for i, c in zip(o.inv['id'], o.inv['count'])}
    events = 0
    now_items = {i for i, c in counts.items() if c > 0}
    now_tr = {min(tr0 & 0x7fff, N_TRINKET - 1), min(tr1 & 0x7fff, N_TRINKET - 1)} - {0}
    if not first:
        old = st.inv_counts or {}
        if any(c > old.get(i, 0) for i, c in counts.items()):
            events |= EV_ITEM
        if (st.pocket[0] != 0 and pill == 0) or (st.pocket[1] != 0 and card == 0):
            events |= EV_PILL
        # (--stall-progress items: something acquired for the first time this episode -- a collectible, a trinket, a
        #  pill color or a card the episode has not held before; one swapped back in is not progress)
        st.inv_gain = bool(now_items - st.seen_items) or bool(now_tr - st.seen_tr) or \
            (pill != 0 and pill not in st.seen_pill) or (card != 0 and card not in st.seen_card)
    st.seen_items |= now_items
    st.seen_tr |= now_tr
    if pill != 0:
        st.seen_pill.add(pill)
    if card != 0:
        st.seen_card.add(card)
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
        # (2026-11-20, reward-rewind-design.md section 6: a real use only -- the same item's charge went down, or a
        #  single-use item without charges is gone; not the empty slot of a pedestal swap's pickup animation, which
        #  counted as a use before)
        if (a == pa and a != 0 and charge < pc) or (pa != 0 and a == 0 and st.prev_maxc == 0):
            events |= EV_ACTIVE
    st.prev_active = (a, charge)
    st.prev_maxc = st.max_charge
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
# (2026-11-20, version 9 -- the coordinator: the native ROW encoder for ROW schema 8 back, OFF by default until the
#  paired runs) ISAAC_RL_FAST_ROW: 0 / unset off (every record through decode + encode_row, as since f3264e0); 1 on;
#  2 on and checked: every record the fast path takes is also encoded by decode + encode_row on copies of the decoder
#  and the EpisodeState, the two ROW slots and the states compared byte for byte (FastRow._check).
FAST_ROW_MODE = os.environ.get('ISAAC_RL_FAST_ROW', '0')
FAST_ROW = FAST_ROW_MODE in ('1', '2')
FAST_ROW_CHECKING = FAST_ROW_MODE == '2'   # (this process's: configure_worker sets it in a canary worker)
# (Codex round 24, C: the canary) ISAAC_RL_FAST_ROW=1 and ISAAC_RL_FAST_ROW_CHECK_WORKERS=<worker indices, e.g. 0>: those
# workers run check mode (every fast record also through encode_row, a difference stops the run), the others mode 1
FAST_ROW_CHECK_WORKERS = {int(x) for x in os.environ.get('ISAAC_RL_FAST_ROW_CHECK_WORKERS', '').split(',') if x.strip()}


def configure_worker(index):
    """A sampler worker's FastRow mode, at its start: check mode when ISAAC_RL_FAST_ROW=2, or when the switch is on and
    the worker is one of ISAAC_RL_FAST_ROW_CHECK_WORKERS (the canary); its reports say which worker it is."""
    global FAST_ROW_CHECKING
    _WORKER[0] = int(index)
    if FAST_ROW and int(index) in FAST_ROW_CHECK_WORKERS:
        FAST_ROW_CHECKING = True
_ROW_FIELDS = ('player', 'ent', 'ent_id', 'doors', 'patch', 'grid', 'map', 'n_ent', 'n_doors', 't', 'hurt', 'damage',
               'done', 'bombs', 'events', 'episode', 'seed', 'group', 'first', 'ent_item', 'pcharge', 'pchar',
               'ent_price', 'phearts', 'room_type', 'rewind', 'visible_schema', 'visible_bytes', 'visible_frame',
               'stage', 'inv', 'pitem', 'pinv')
_ROW_OFFS_ITEMS = np.array([ROW.fields[k][1] for k in _ROW_FIELDS], np.int64)
_ROW_OFFS = _ROW_OFFS_ITEMS.copy()
_ROW_OFFS[_ROW_FIELDS.index('ent_item')] = -1   # without items abp_row_encode leaves ent_item alone (version 3)


def _row_offs(items, run):
    """abp_row_encode's offs for an episode: ent_item, inv, pitem, pinv written with items only, stage with items or a
    run (encode_row's _items_step), every other field always."""
    o = _ROW_OFFS_ITEMS.copy()
    if not items:
        for k in ('ent_item', 'inv', 'pitem', 'pinv'):
            o[_ROW_FIELDS.index(k)] = -1
    if not (items or run):
        o[_ROW_FIELDS.index('stage')] = -1
    return o


_ROW_VERSION = 11   # (11: Codex round 24-1: the record before's active compared as a double, bombs < 0 handed back) (10: 2026-11-20, Codex round 23-1: -15, the integer conversions encode_row raises on) (9: 2026-11-20, ROW schema 8: the v3 frame, the rewind count, EV_NOHIT, --stall-progress first,
#                    _items_step, the ext pointers) (8: 2026-11-20, the rewind count block, flag 128, handed back: -10)
#                    abp_row_encode_version's layout (4: 2026-10-07, the 38-double player record and pcharge; 5:
#                    2026-10-08, abp-0.2.16's 144-byte entity records and ENT_F = 56; 6: 2026-10-08, abp-0.2.17's
#                    39-double player record (PlayerType) and pchar; 7: 2026-10-16 (B35), abp-0.2.18's 148-byte entity
#                    records (price) and ent_price, phearts, room_type)
# ctx slots (abp_turbo.c RC_*)
(RC_T, RC_LIMIT, RC_FLOOR, RC_STALL, RC_PROGRESS, RC_STAGE0, RC_HP0, RC_PREV_DMG, RC_PREV_HP, RC_HAVE_PREV, RC_ROOM,
 RC_WAS_CLEAR, RC_ORIGIN_X, RC_ORIGIN_Y, RC_N_EXITS, RC_EXITS) = range(16)
RC_TVERSION = RC_EXITS + 16
(RC_TSEEN, RC_HURT, RC_EPISODE, RC_SEED, RC_GROUP, RC_FIRST, RC_RUN, RC_ITEMS, RC_STALL_MODE, RC_HAVE_RES,
 RC_PREV_RES) = range(RC_TVERSION + 1, RC_TVERSION + 12)
# B40 (2026-10-18): RC_STALL_MODE (1: --stall-progress pickups), RC_HAVE_RES, RC_PREV_RES .. + 3 (coins, keys, bombs,
# health units at the record before). Version 9 (abp_turbo.c's comment): the worker's visible mode, the rewind count,
# the nohit room, --stall-progress first's room keys, _items_step's state; out: a v3 frame written, the state changed
(RC_VIS, RC_REWIND_ON, RC_RW_HAVE, RC_RW_COUNT, RC_CK_NONE, RC_NOHIT_ON) = range(RC_PREV_RES + 4, RC_PREV_RES + 10)
RC_NH = RC_NOHIT_ON + 1
RC_FIRST_MODE = RC_NH + 3
RC_EVER = RC_FIRST_MODE + 1
RC_POST = RC_EVER + 3
(RC_PREV_A_HAVE, RC_PREV_A, RC_PREV_C, RC_PREV_MAXC, RC_MAX_CHARGE, RC_OUT_V3, RC_OUT_DIRTY) = range(RC_POST + 1,
                                                                                                 RC_POST + 8)
RC_COUNT = RC_OUT_DIRTY + 1   # abp_turbo.c's RC_COUNT 67
# player 0's active item and charge, and the room's stage, in a raw lean payload (abplus_lean: header 16, room 48,
# totals 32, then the player's doubles: active is field 24)
_PAYLOAD_ACTIVE = struct.Struct('<dd')
_PAYLOAD_STAGE = struct.Struct('<i')
_LIBS = {}


def _address(b):
    """The address of a bytes object's buffer (no copy)."""
    return ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value


_TORCH_WARNED = []


def check_worker_torch(index):
    """1 when this worker process holds torch (its stats torch_held; the trainer's progress worker_torch counts the
    workers), else 0; the first time, a warning line with the worker's index. Never the run's end (the coordinator:
    a memory cost, not a data error -- an arm stopped would resume with its buffers emptied); the start's check with
    ISAAC_RL_WORKER_LEAN on (tok_sampler.confirm_rings) refuses before any sampling. Cheap (one dict lookup): the
    workers call it every loop."""
    if 'torch' not in sys.modules:
        return 0
    if not _TORCH_WARNED:
        _TORCH_WARNED.append(1)
        print(f'tok worker {index}: WARNING torch was loaded in this worker process (a worker-side import of a torch '
              f'module: about 220 MB more; progress worker_torch)', file=sys.stderr, flush=True)
    return 1


class FastRowFatal(_VP.VisibleV3Fatal):
    """(2026-11-20, Codex round 23-2) A FastRow failure that must stop the run, never a bad episode to retry: the
    switch on (ISAAC_RL_FAST_ROW 1 / 2) without a library of this layout, an unknown hand-back code, and in check mode
    a record the two encoders disagree on (or one the C side took and encode_row refuses), a hand-back that wrote, a
    report not written. always_fatal: the worker ends with V3_FATAL_EXIT and the trainer fails, in any visible mode."""
    always_fatal = True


# abp_row_encode's hand-back codes (abp_turbo.c; -7: not offered -- no grid / terrain pointers yet); any other
# negative return is a FastRowFatal
_KNOWN_CODES = (-1, -2, -3, -4, -5, -6, -7, -8, -11, -12, -13, -14, -15)


def fast_row_function(path):
    """abp_row_encode of the libabp_turbo.so at path (ctypes); None when the switch is off (ISAAC_RL_FAST_ROW unset /
    0). Switched on (1 / 2) it is required (Codex round 23-2: an explicit feature never degrades silently): a missing
    library, one of another version / layout, or check mode without ISAAC_RL_FAST_ROW_CHECK_DIR raise FastRowFatal.
    (f3264e0 to 2026-11-20: always None -- the version 7 encoder could not carry the visible-v3 frame.)"""
    if not FAST_ROW:
        return None
    if FAST_ROW_CHECKING and not os.environ.get('ISAAC_RL_FAST_ROW_CHECK_DIR'):
        raise FastRowFatal('ISAAC_RL_FAST_ROW=2 (check mode) needs ISAAC_RL_FAST_ROW_CHECK_DIR (every process\'s counts)')
    if not path or not os.path.isfile(path):
        raise FastRowFatal(f'ISAAC_RL_FAST_ROW={FAST_ROW_MODE} but no libabp_turbo.so at {path!r}')
    if path not in _LIBS:
        want = (_ROW_VERSION << 16) | (RC_COUNT << 8) | len(_ROW_FIELDS)
        try:
            lib = ctypes.CDLL(path, mode=getattr(os, 'RTLD_LOCAL', 0))
            version = lib.abp_row_encode_version
            version.restype = ctypes.c_int
            got = version()
        except (OSError, AttributeError) as exc:
            raise FastRowFatal(f'ISAAC_RL_FAST_ROW={FAST_ROW_MODE}: {path} has no abp_row_encode ({exc})') from None
        if got != want or not (ENT_CAP == 64 and ENT_F == 56 and ENT_F0 == 33 and PCHARGE_F == 12 and N_CHAR == 32 and
                               PHEART_F == 3 and N_ROOM_TYPE == 32 and N_COLLECTIBLE == 1024 and INV_CAP == 16 and
                               PINV_F == 12 and ROW.fields['visible_frame'][0].itemsize == 131072 and DOOR_CAP == 8
                               and DOOR_F == 7 and PATCH == 9 and GRID == (7, 16, 28) and MAP == (8, 13, 13)):
            raise FastRowFatal(f'ISAAC_RL_FAST_ROW={FAST_ROW_MODE}: {path} is abp_row_encode layout {got:#x}, this '
                               f'tree needs {want:#x} (version {_ROW_VERSION}): build the game home from this tree')
        fn = lib.abp_row_encode
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_char_p, ctypes.c_long] + [ctypes.c_void_p] * 8
        _LIBS[path] = fn
    return _LIBS[path]


class FastRow:
    """One episode's records through abp_row_encode (fn from fast_row_function). encode(item, row, row_address, t):
    item is a raw lean payload (bytes) or a LeanObs; returns the done code like encode_row and sets self.hurt (half
    hearts were lost in the step that led here), self.obs (the LeanObs when the record went through encode_row, else
    None). Writes episode, seed, group and first too (set_meta).
    (Version 9, 2026-11-20) the EpisodeState is brought up to date from the C side's slots before a record goes through
    encode_row and after a fast record that changed it (RC_OUT_DIRTY: the rewind count, the nohit room, the first-mode
    keys, the active item's state); st.inv_gain (read only inside encode_row, reset by _inventory) is not kept.
    tag: where the episode is played ('start' / 'archive' -- a main episode's fresh or archive start --, 'branch' -- a
    branch point's --, 'lab', 'cf' / 'teacher' -- a cf / teacher v2 branch (tok_teacher2.play_branch) --, 'room'), for
    check mode's boundary counts."""

    def __init__(self, fn, st, decoder, episode=0, seed=0, group=0, tag='main'):
        self.fn, self.st, self.decoder = fn, st, decoder
        self.tag = tag if tag in _TAGS else 'other'
        self.ctx = np.zeros(RC_COUNT, np.float64)
        self._ctx_p = self.ctx.ctypes.data
        self._offs = _row_offs(st.items, st.run)
        self._offs_p = self._offs.ctypes.data
        self.refs = np.zeros(4, np.int64)   # the last terrain block's bytes, the door records of the grid: address, length
        self._refs_p = self.refs.ctypes.data
        # (version 9) _items_step's arrays, written in place by abp_row_encode: inv, pitem, pinv
        arrs = (st.inv_arr, st.pitem, st.pinv)
        assert st.inv_arr.dtype == np.int16 and st.inv_arr.shape == (INV_CAP, 2) and st.pitem.dtype == np.int16 and \
            st.pitem.shape == (6,) and st.pinv.dtype == np.float32 and st.pinv.shape == (PINV_F,) and \
            all(a.flags['C_CONTIGUOUS'] for a in arrs)
        self.ext = np.array([a.ctypes.data for a in arrs], np.int64)
        self._ext_p = self.ext.ctypes.data
        self._keep = (None, None)
        self._ptrs = (None, None, None)
        self.hurt = False
        self.obs = None
        self.fast = self.slow = 0
        self.reasons = {}   # abp_row_encode's hand-back codes: how often
        self._code = None   # (check mode's boundary counts) this record's hand-back code, payload; the last two paths
        self._payload = None
        self._paths = (None, None)
        self._last_rw = 0
        self._last_pchar = None
        c = self.ctx
        c[RC_LIMIT], c[RC_FLOOR], c[RC_STALL] = st.limit, float(st.floor), st.stall
        c[RC_STALL_MODE] = float(st.stall_pickups)   # B40
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
        self._code = None
        self._payload = item if type(item) is bytes else None
        if type(item) is bytes:
            c[RC_T] = t
            c[RC_FIRST] = t == 0
            g, pd, m = self._ptrs
            if FAST_ROW_CHECKING and g:
                return self._check(item, row, row_address, t)
            done = self.fn(item, len(item), row_address, self._ctx_p, g, pd, m, self._offs_p, self._refs_p,
                           self._ext_p) if g else -7
            if done >= 0:
                self._fast_done(row, item)
                return done
            self._handed(done)
            item = self.decoder.decode(item)
        else:
            _bump('slow_obs')
        return self._slow(item, row, t)

    def _handed(self, done):
        if done not in _KNOWN_CODES:   # (Codex round 23-2: not a silent hand-back)
            raise FastRowFatal(f'abp_row_encode returned {done}: not one of its hand-back codes {_KNOWN_CODES}')
        self._code = done
        self.reasons[done] = self.reasons.get(done, 0) + 1
        _bump('slow_%d' % done)

    def _fast_done(self, row, payload):
        c = self.ctx
        self.fast += 1
        _bump('fast')
        self.hurt = c[RC_HURT] != 0
        self.obs = None
        if c[RC_TSEEN]:   # the same terrain block again: the decoder's version follows it
            d = self.decoder
            d.terrain_version = int(c[RC_TVERSION])
            d.terrain['version'] = d.terrain_version
        if c[RC_OUT_V3]:
            V3_FRAMES[0] += 1
        dirty = int(c[RC_OUT_DIRTY])
        if dirty:
            self._to_st()
            for bit, name in ((1, 'fast_rw'), (2, 'fast_nohit_state'), (4, 'fast_first_prog'), (8, 'fast_active_state')):
                if dirty & bit:
                    _bump(name)
        ev = int(row['events'][0])
        if ev & EV_NOHIT:
            _bump('fast_ev_nohit')
        if ev & EV_ACTIVE:
            _bump('fast_ev_active')
        if FAST_ROW_CHECKING:
            self._track(True, row, payload)

    def _slow(self, item, row, t):
        self.slow += 1
        _bump('slow')
        self._to_st()
        done = encode_row(item, self.st, row, t)
        row['episode'], row['seed'], row['group'], row['first'] = self.meta[0], self.meta[1], self.meta[2], t == 0
        self.hurt = bool(row['hurt'][0] > 0)
        self.obs = item
        self._from_st()
        if FAST_ROW_CHECKING:
            self._track(False, row, None)
        return done

    def _track(self, fast, row, payload):
        """(check mode; Codex rounds 23 / 24: by the real conditions) how often the boundaries the acceptance asks for
        were met: a fast record with more than one body; a character change on a fast / a slow record; a raw payload
        with exactly 64 old entities and the floor's exit tokens (fast) / exactly 65 (handed back, -3); a same-room boss
        clear on a fast record; a map or inventory block (the payload's flags 8 / 16) on the record right after a
        rewind; a rewind on a death / a time-limit (cap) / a stall record, apart; fast -> slow -> fast within an episode
        by where it is played (tag: a fresh start, an archive start, a cf branch, a teacher branch, ...)."""
        p2, p1 = self._paths
        if fast and p1 is False and p2 is True:
            _bump('hit_fsf_' + self.tag)
        self._paths = (p1, fast)
        rw = int(row['rewind'][0]) & REWIND_COUNT
        done = int(row['done'][0])
        if rw:
            _bump('hit_rewind')
            if done == 2:
                _bump('hit_rewind_death')
            elif done == 3:
                _bump('hit_rewind_cap' if int(row['t'][0]) >= self.st.limit else 'hit_rewind_stall')
        pl = payload if payload is not None else self._payload
        flags = pl[12] if pl is not None and len(pl) > 12 else 0
        if self._last_rw and flags & (8 | 16):
            _bump('hit_block_after_rewind')
        pc = int(row['pchar'][0])
        if self._last_pchar is not None and pc != self._last_pchar:
            _bump('hit_char_change_fast' if fast else 'hit_char_change_slow')
        if pl is not None and len(pl) >= 16:
            n_pl, n_dr = pl[13], pl[14]
            at = 96 + 8 * 39 * n_pl + 16 * n_dr
            n_ent = struct.unpack_from('<H', pl, at)[0] if len(pl) >= at + 2 else -1
            exits = self.ctx[RC_N_EXITS] > 0
            if fast and n_pl > 1:
                _bump('hit_multi_player')
            if fast and n_ent == ENT_CAP and exits:
                _bump('hit_ent64_exit')
            if not fast and n_ent == ENT_CAP + 1 and exits and self._code == -3:
                _bump('hit_ent65_exit')
        if fast and int(row['events'][0]) & EV_BOSS:
            _bump('hit_boss_clear')
        self._last_rw, self._last_pchar = rw, pc

    def _check(self, item, row, row_address, t):
        """(ISAAC_RL_FAST_ROW=2) the record through abp_row_encode and, on copies of the decoder and the EpisodeState
        as they stand before it, through decode + encode_row; the ROW slots, the done codes, the hurt flags and the
        states compared. (Codex round 23-2) A difference -- or a record the C side took and encode_row refuses, or a
        hand-back that changed the ROW, ctx or the EpisodeState's arrays -- is dumped, reported and raised
        (FastRowFatal: the run stops; never trained on, never retried)."""
        import copy
        c = self.ctx
        self._to_st()
        ref_st, ref_dec = copy.deepcopy(self.st), copy.deepcopy(self.decoder)
        raw = row.view(np.uint8).reshape(-1)
        before = raw.copy()
        ctx0 = self.ctx.copy()
        ext0 = [self.st.inv_arr.copy(), self.st.pitem.copy(), self.st.pinv.copy()]
        g, pd, m = self._ptrs
        done = self.fn(item, len(item), row_address, self._ctx_p, g, pd, m, self._offs_p, self._refs_p, self._ext_p)
        if done < 0:   # handed back: nothing may have been written; then encode_row on the real state
            wrote = [n for n, ok in (('ROW', np.array_equal(raw, before)),
                                     ('ctx', self.ctx.tobytes() == ctx0.tobytes()),
                                     ('inventory arrays', all(a.tobytes() == b.tobytes() for a, b in
                                                              zip(ext0, (self.st.inv_arr, self.st.pitem,
                                                                         self.st.pinv))))) if not ok]
            if wrote:
                _bump('check_mismatch')
                _check_dump(item, before, raw.copy(), None, t, [f'hand-back {done} wrote ' + ', '.join(wrote)])
                _check_report(True)
                raise FastRowFatal(f'FastRow check: abp_row_encode handed record {t} back ({done}) after writing '
                                   f'{", ".join(wrote)}')
            self._handed(done)
            return self._slow(self.decoder.decode(item), row, t)
        self._fast_done(row, item)
        self._to_st()
        got = raw.copy()
        raw[:] = before
        v3_counts = list(V3_FRAMES)
        try:
            ref_done = encode_row(ref_dec.decode(item), ref_st, row, t)
            row['episode'], row['seed'], row['group'], row['first'] = self.meta[0], self.meta[1], self.meta[2], t == 0
            ref_hurt = bool(row['hurt'][0] > 0)
            want = raw.copy()
            err = None
        except Exception as e:   # (encode_row refused a record the fast path took)
            ref_done, ref_hurt, want, err = None, None, None, f'{type(e).__name__}: {e}'
        V3_FRAMES[:] = v3_counts
        raw[:] = got
        _bump('check_records')
        bad = []
        if err is not None:
            bad.append('encode_row raised: ' + err)
        else:
            if not np.array_equal(got, want):
                a_, b_ = got.view(ROW)[0], want.view(ROW)[0]
                bad.append('row fields ' + ','.join(k for k in ROW.names if a_[k].tobytes() != b_[k].tobytes()))
            if ref_done != done:
                bad.append(f'done {done} vs {ref_done}')
            if ref_hurt != self.hurt:
                bad.append(f'hurt {self.hurt} vs {ref_hurt}')
        sd = _state_diff(self.st, ref_st)
        if sd:
            bad.append('state ' + ','.join(sd))
        if ref_dec.terrain_version != self.decoder.terrain_version:
            bad.append('decoder terrain_version')
        if bad:
            _bump('check_mismatch')
            _check_dump(item, before, got, want, t, bad)
            _check_report(True)
            raise FastRowFatal(f'FastRow check: record {t} of episode {self.meta[0]} ({self.tag}): ' + '; '.join(bad))
        _check_report()
        return done

    def _to_st(self):
        c, st = self.ctx, self.st
        if c[RC_HAVE_PREV]:
            st.previous = (float(c[RC_PREV_DMG]), float(c[RC_PREV_HP]))
            st.room = int(c[RC_ROOM])
            st.was_clear = bool(c[RC_WAS_CLEAR])
            st.progress_t = int(c[RC_PROGRESS])
        if c[RC_HAVE_RES]:   # B40
            st.prev_res = tuple(float(v) for v in c[RC_PREV_RES:RC_PREV_RES + 4])
        # (version 9)
        if c[RC_RW_HAVE]:
            st.rw_count = int(c[RC_RW_COUNT])
        if st.room is not None:
            if st.nohit_on:
                e = st.nohit_rooms.setdefault((st.floor_n, st.room), [False, False, False])
                e[0], e[1], e[2] = bool(c[RC_NH]), bool(c[RC_NH + 1]), bool(c[RC_NH + 2])
            if st.stall_first:
                key = (st.floor_k, st.room)
                for j, s_ in enumerate((st.ever_rooms, st.ever_cleared, st.ever_hurt)):
                    if c[RC_EVER + j]:
                        s_.add(key)
        if c[RC_PREV_A_HAVE]:
            st.prev_active = (int(c[RC_PREV_A]), float(c[RC_PREV_C]))
            st.prev_maxc = int(c[RC_PREV_MAXC])

    def _from_st(self):
        c, st = self.ctx, self.st
        c[RC_HAVE_PREV] = st.previous is not None and st.room is not None and st.hp0 is not None
        if c[RC_HAVE_PREV]:
            c[RC_PREV_DMG], c[RC_PREV_HP] = st.previous
            c[RC_ROOM], c[RC_WAS_CLEAR], c[RC_HP0] = st.room, bool(st.was_clear), st.hp0
        c[RC_PROGRESS] = st.progress_t
        c[RC_HAVE_RES] = st.prev_res is not None   # B40
        if st.prev_res is not None:
            c[RC_PREV_RES:RC_PREV_RES + 4] = st.prev_res
        c[RC_STAGE0] = np.nan if st.stage0 is None else st.stage0
        c[RC_RUN] = float(st.run)
        c[RC_ITEMS] = (2.0 if st.blind else 1.0) if st.items else 0.0
        c[RC_ORIGIN_X], c[RC_ORIGIN_Y] = st.origin
        c[RC_N_EXITS] = min(len(st.exits), 8)
        for i, (ex, ey) in enumerate(st.exits[:8]):
            c[RC_EXITS + 2 * i], c[RC_EXITS + 2 * i + 1] = ex, ey
        # (version 9) the worker's visible mode (write_visible), the rewind count, the nohit room, the first-mode keys,
        # _items_step's state
        c[RC_VIS] = (1.0 if _VP.VISIBLE_V3_MODE else 0.0) + (2.0 if os.environ.get('ISAAC_NO_VISIBLE') == '1' else 0.0)
        c[RC_REWIND_ON] = float(st.rewind_on)
        c[RC_RW_HAVE] = st.rw_count is not None
        c[RC_RW_COUNT] = st.rw_count if st.rw_count is not None else 0
        c[RC_CK_NONE] = st.ck is None
        c[RC_NOHIT_ON] = float(st.nohit_on)
        nh = st.nohit_rooms.get((st.floor_n, st.room)) if (st.nohit_on and st.room is not None) else None
        c[RC_NH:RC_NH + 3] = [float(x) for x in nh] if nh is not None else (0.0, 0.0, 0.0)
        c[RC_FIRST_MODE] = float(st.stall_first)
        key = (st.floor_k, st.room)
        c[RC_EVER:RC_EVER + 3] = (key in st.ever_rooms, key in st.ever_cleared, key in st.ever_hurt) \
            if (st.stall_first and st.room is not None) else (0.0, 0.0, 0.0)
        c[RC_POST] = 2.0 if st.items else 1.0 if st.run else 0.0
        c[RC_PREV_A_HAVE] = st.prev_active is not None
        if st.prev_active is not None:
            c[RC_PREV_A], c[RC_PREV_C] = st.prev_active
        c[RC_PREV_MAXC], c[RC_MAX_CHARGE] = st.prev_maxc, st.max_charge
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


# (version 9 / 10) this process's FastRow counters, under fixed keys only (Codex round 23-3: a thread adding a key while
# fast_counts or a report read the dict ended the worker): fast / slow records, the hand-back codes, what the fast
# records did; in check mode the records compared, the mismatches, the children forked and the boundary counts
_TAGS = ('start', 'archive', 'branch', 'lab', 'cf', 'teacher', 'room', 'other')
_STAT_KEYS = (('fast', 'slow', 'slow_obs') + tuple('slow_%d' % c for c in _KNOWN_CODES) +
              ('fast_rw', 'fast_nohit_state', 'fast_first_prog', 'fast_active_state', 'fast_ev_nohit', 'fast_ev_active',
               'check_records', 'check_mismatch', 'children_forked', 'hit_multi_player', 'hit_char_change_fast',
               'hit_char_change_slow', 'hit_ent64_exit', 'hit_ent65_exit', 'hit_boss_clear', 'hit_block_after_rewind',
               'hit_rewind', 'hit_rewind_death', 'hit_rewind_cap', 'hit_rewind_stall') +
              tuple('hit_fsf_' + t for t in _TAGS))
FAST_STATS = dict.fromkeys(_STAT_KEYS, 0)
_CHECK_DUMPS = [0]
_CHECK_LAST = [0.0]
_ROLE = ['worker']
_WORKER = [-1]
import threading as _threading   # noqa: E402
_REPORT_LOCK = [_threading.Lock()]   # (Codex round 24-3: one report at a time per process; a forked child: its own)
_FINAL = [False]                     # (no periodic report after the final one)


def _bump(key, n=1):
    """One of FAST_STATS' fixed keys up by n (an unknown key is a KeyError: no key is ever added)."""
    FAST_STATS[key] += n


def fast_counts():
    """(fast, slow, slow by hand-back group: block, room / floor, visible, rewind, other, check mode's records compared,
    mismatches) of this process's FastRow records so far -- tok_sampler's fr_* worker stats (the canary's compared /
    mismatches in progress.csv). Reads fixed keys; never iterates the dict."""
    s = FAST_STATS
    g = lambda *codes: sum(s['slow_%d' % c] for c in codes)   # noqa: E731
    return [s['fast'], s['slow'], g(-2), g(-4, -5, -6, -8), g(-13), g(-11, -12),
            g(-1, -3, -7, -14, -15) + s['slow_obs'], s['check_records'], s['check_mismatch']]


def child_start(role):
    """(check mode) In a forked child (a cf point, a search): its own counts from zero, under its role, its own report
    lock (the parent's may have been held by another thread at the fork) and no final yet."""
    _REPORT_LOCK[0] = _threading.Lock()
    _FINAL[0] = False
    _ROLE[0] = role
    for k in _STAT_KEYS:
        FAST_STATS[k] = 0
    _CHECK_LAST[0] = 0.0


def note_child(pid, role):
    """(check mode, in the parent right after a fork) the child's pid and role appended to
    ISAAC_RL_FAST_ROW_CHECK_DIR/fastrow-children-<parent pid>.txt and counted (children_forked): tools/fastrow_verify.py
    requires a final report of every child listed. Raises FastRowFatal when it cannot be written."""
    _bump('children_forked')
    d = os.environ.get('ISAAC_RL_FAST_ROW_CHECK_DIR')
    try:
        with _REPORT_LOCK[0]:
            with open(os.path.join(d, f'fastrow-children-{os.getpid()}.txt'), 'a') as f:
                f.write(f'{int(pid)} {role}\n')
    except (OSError, TypeError) as exc:
        raise FastRowFatal(f'FastRow check: the child {pid} could not be listed: {exc}') from None


def fast_row_final():
    """(check mode) This process's final counts (final: true), before it ends -- a worker's loop's end, a forked
    child's os._exit (which runs no atexit). Raises FastRowFatal when they cannot be written."""
    if FAST_ROW_CHECKING:
        _check_report(True, final=True)


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and \
            a.dtype == b.dtype and a.tobytes() == b.tobytes()
    try:
        return bool(a == b)
    except Exception:
        return False


def _state_diff(st, ref):
    """The EpisodeState attributes that differ (inv_gain: encode_row's own per-record flag, not kept by FastRow)."""
    return [k for k in vars(ref) if k != 'inv_gain' and not _same(getattr(st, k, None), getattr(ref, k))]


def _check_dump(payload, before, got, want, t, bad):
    d = os.environ.get('ISAAC_RL_FAST_ROW_CHECK_DIR')
    if not d or _CHECK_DUMPS[0] >= 20:
        return
    _CHECK_DUMPS[0] += 1
    try:
        np.savez(os.path.join(d, f'fastrow-mismatch-{os.getpid()}-{_CHECK_DUMPS[0]}.npz'),
                 payload=np.frombuffer(payload, np.uint8), before=before, got=got,
                 want=want if want is not None else np.zeros(0, np.uint8), t=t, bad=np.array(bad))
    except OSError:
        pass


def _check_report(force=False, final=False):
    """(check mode) This process's counts to ISAAC_RL_FAST_ROW_CHECK_DIR/fastrow-<pid>.json (at most every 30 s
    unless forced), with its pid, parent, role and whether final. A failure to write raises FastRowFatal (Codex round
    23-2: the counts must reach the check)."""
    d = os.environ.get('ISAAC_RL_FAST_ROW_CHECK_DIR')
    if not FAST_ROW_CHECKING:
        return
    if not d:
        raise FastRowFatal('ISAAC_RL_FAST_ROW=2 without ISAAC_RL_FAST_ROW_CHECK_DIR')
    import time as _time
    now = _time.time()
    if not force and now - _CHECK_LAST[0] < 30:
        return
    with _REPORT_LOCK[0]:   # (round 24-3: serialised; a final one is never overwritten by a later periodic one)
        if _FINAL[0] and not final:
            return
        _CHECK_LAST[0] = now
        try:
            import json as _json
            tmp = os.path.join(d, f'.fastrow-{os.getpid()}-{_threading.get_ident()}.json')
            with open(tmp, 'w') as f:
                _json.dump(dict({k: FAST_STATS[k] for k in _STAT_KEYS}, pid=os.getpid(), ppid=os.getppid(),
                                role=_ROLE[0], worker=_WORKER[0], final=bool(final), time=now), f)
            os.replace(tmp, os.path.join(d, f'fastrow-{os.getpid()}.json'))
        except OSError as exc:
            raise FastRowFatal(f'FastRow check: the counts could not be written to {d}: {exc}') from None
        if final:
            _FINAL[0] = True
