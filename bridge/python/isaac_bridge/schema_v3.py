"""visible-v3: the frozen interface between the producers (package C: Lua bridge, abp_turbo.c, abp_visible.c, the
producer module, the C++ actor) and the model / learner (package A). The single source of truth of the design
D:/Projects/fortune/Isaac/temp/rep-compat-schema-20261112/DESIGN.md (2026-11-12), artefacts F1, F2 and F4:

  F1  the frame: schema string, version, header, sizes, the room canvas words and their bit layouts, the minimap
      bits, every P / E / N / S column (name, kind, raw unit), known flags, the embedding tables' namespaces and the
      out-of-range guard rule, the AB+ -> canonical TearFlags table, the hidden projectile bits, the visible
      entity-flag mask, code conventions, producer ids; plus (the coordinator, 2026-11-12) P 40 stall_share and
      P 53 stall_known (the floor's stall rule: an environment quantity, not a game one)
  F2  the action space: heads and their order, sizes, the ButtonAction names per head, action_code with drop, the
      step line, pending [n, 6], CRN columns 22
  F4  the record formats: V2_FIELDS (pend0..5 / act0..5, V2_META 31), TEACH_META, ACTION_F 6

engine/abp_schema_v3.h is generated from this module (tools/gen_schema_v3_h.py; tests/test_schema_v3.py asserts the
committed header equals the generated one). Nothing here changes v2 behaviour: v2 lives in schema_v2.py /
structured_input.py as before; v3 is a new path chosen by a switch.

Freeze rule: after the coordinator's review, any change of a column number, a size, the version or the head order is
reported to the coordinator first, who tells the other package; neither side changes them on its own.

Wire rule (DESIGN 2.0): every value is a raw game unit as float32 (integers exact to 2^24, bit words <= 24 bits, flag
words in 16-bit chunks); categorical codes are integers, 0 = none / not known; reserved columns are 0. The model
applies SCALE (below) itself. Known flags (G3): a field the game version lacks is written as its true value there
(e.g. AB+ broken hearts = 0) with its known flag 1; a field that exists but the producer cannot read is 0 with known 0.
"""

# ------------------------------------------------------------------------------------------------- F1: identity
SCHEMA = 'isaac-visible-prlmens-g-v3'
VERSION = 2026111201
FRAME_BYTES = 131072
MAGIC = 61107
LAYOUT_GEN = 3        # header[6]
HEADER = 8
# header[0..7]: magic, size (floats), p, e, n, s, LAYOUT_GEN, producer id (diagnostics only, never a model input)
H_MAGIC, H_SIZE, H_P, H_E, H_N, H_S, H_GEN, H_PRODUCER = range(8)
PRODUCER = {'abplus': 0, 'rep_j746': 1, 'rep_plus': 2, 'recorder': 3}

# ------------------------------------------------------------------------------------------------- F1: room canvas
CANVAS_H, CANVAS_W = 16, 28
ROOM_WORDS = 5        # per cell: word 0 bits, word 1 bits, grid type, grid variant, grid state (raw)
# word 0, LSB first (20 bits): (name, bits)
ROOM_W0 = (('inside', 1), ('collision', 3), ('solid', 1), ('pit', 1), ('destructible', 1), ('hazard', 1),
           ('door', 1), ('door_open', 1), ('door_locked', 1), ('door_target', 6), ('visited', 1), ('clear', 1),
           ('exit', 1))
# word 1, LSB first (23 bits)
ROOM_W1 = (('door_variant', 6), ('door_state', 4), ('red_room', 1), ('reserved_a', 6), ('reserved_b', 6))
ROOM_RAW = ('grid_type', 'grid_variant', 'grid_state')   # words 2..4
# decoded channels (RC 21): word 0's 13, word 1's 5, the 3 raw words; the model's scale of each (codes /64, DESIGN
# 2.2 and Q6: scalar-coded)
ROOM_CHANNELS = tuple(n for n, _ in ROOM_W0) + tuple(n for n, _ in ROOM_W1) + ROOM_RAW
RC = len(ROOM_CHANNELS)
ROOM_DECODE_SCALE = {'collision': 6, 'door_target': 64, 'door_variant': 64, 'door_state': 15, 'reserved_a': 64,
                     'reserved_b': 64, 'grid_type': 64, 'grid_variant': 256, 'grid_state': 256}   # others 1
LOCAL_SIDE = 9
LOCAL = LOCAL_SIDE * LOCAL_SIDE * ROOM_WORDS   # floats per body's local patch (same 5-word cells)

# ------------------------------------------------------------------------------------------------- F1: minimap
MAP_SIDE = 13
# one float per cell, LSB first (16 bits): the icon is the raw RoomType when shown or visited, else 0
MAP_BITS = (('shown', 1), ('visited', 1), ('clear', 1), ('current', 1), ('icon', 6), ('reserved', 6))
MAP_ICONS = 64        # the icon's embedding rows (Embedding(64, 8)); a code >= 64 -> 63

FIXED = HEADER + CANVAS_H * CANVAS_W * ROOM_WORDS + MAP_SIDE * MAP_SIDE

# ------------------------------------------------------------------------------------------------- F1: tokens
PF, EF, NF, SF = 192, 64, 8, 24
BODY = PF + LOCAL     # floats per body (P token + its local patch)
RAGGED = {'body': PF, 'local': LOCAL, 'entity': EF, 'inventory': NF, 'slots': SF}
CAP_ENTITY = 128      # entities per frame at most (priority classes; the overflow counts P 176..180)
ENTITY_CLASSES = ('interactable', 'monster', 'familiar', 'projectile', 'other')


def frame_floats(p, e, n, s):
    """The size check (header[1]): FIXED + p x (PF + LOCAL) + e x EF + n x NF + s x SF."""
    return FIXED + p * BODY + e * EF + n * NF + s * SF


# Column kinds: 'num' raw number (SCALE divides it in the model), 'known' a 0/1 known flag, 'bit' one bit of a bit
# block (0/1 on the wire), 'word16' a 16-bit flag word (decoded into 16 bits), 'bits8' a word decoded into 8 bits,
# 'cat:<namespace>' a categorical code (the `cat` table), 'id' a raw entity id (`identity`), 'item:<family>' an item
# id (`item`), 'anim' an anim hash code (`anim`), 'reserved' 0.

def _cols(spec, width):
    cols = [('reserved', 'reserved', '', 0.0)] * width
    for c, name, kind, unit, scale in spec:
        assert cols[c][0] == 'reserved', c
        cols[c] = (name, kind, unit, float(scale))
    return tuple(cols)


def _block(start, n, prefix, kind, unit='', scale=1.0):
    return [(start + i, f'{prefix}{i}', kind, unit, scale) for i in range(n)]


# P: one token per body (DESIGN 2.5). (col, name, kind, raw unit, model scale)
P_SPEC = [
    (0, 'x', 'num', 'px from the room top-left', 520), (1, 'y', 'num', 'px from the room top-left', 280),
    (2, 'vx', 'num', 'px/frame', 10), (3, 'vy', 'num', 'px/frame', 10), (4, 'size', 'num', 'px', 20),
    (5, 'hearts', 'num', 'red half hearts (0 under Curse of the Unknown)', 12),
    (6, 'max_hearts', 'num', 'half hearts', 12), (7, 'soul', 'num', 'soul half hearts', 12),
    (8, 'bone', 'num', 'bone hearts', 6), (9, 'eternal', 'num', 'eternal hearts', 2),
    (10, 'golden', 'num', 'golden hearts', 2), (11, 'broken', 'num', 'broken hearts (AB+: 0)', 12),
    (12, 'rotten', 'num', 'rotten hearts (AB+: 0)', 12), (13, 'extra_lives', 'num', 'lives', 4),
    (14, 'coins', 'num', 'coins', 100), (15, 'bombs', 'num', 'bombs', 100), (16, 'keys', 'num', 'keys', 100),
    (17, 'giga_bombs', 'num', 'giga bombs (AB+: 0)', 10), (18, 'golden_bomb', 'num', '0/1 (AB+: 0)', 1),
    (19, 'golden_key', 'num', '0/1 (AB+: 0)', 1), (20, 'poop_mana', 'num', 'poop mana (AB+: 0)', 9),
    (21, 'soul_charge', 'num', 'GetEffectiveSoulCharge (AB+: 0)', 99),
    (22, 'blood_charge', 'num', 'GetEffectiveBloodCharge (AB+: 0)', 99),
    (23, 'damage', 'num', 'Damage', 10), (24, 'max_fire_delay', 'num', 'MaxFireDelay frames', 20),
    (25, 'shot_speed', 'num', 'ShotSpeed', 2), (26, 'range', 'num', 'tear range px (AB+ 260 h / -23.75)', 500),
    (27, 'move_speed', 'num', 'MoveSpeed', 2), (28, 'luck', 'num', 'Luck', 5),
    (29, 'can_fly', 'num', '0/1', 1), (30, 'invulnerable', 'num', '0/1 (damage cooldown > 0)', 1),
    (31, 'controls', 'num', '0/1 ControlsEnabled', 1), (32, 'dead', 'num', '0/1', 1),
    (33, 'head_dir', 'num', 'Direction -1..3', 4), (34, 'fire_dir', 'num', 'Direction -1..3', 4),
    (35, 'move_dir', 'num', 'Direction -1..3', 4), (36, 'sprite_frame', 'num', 'sprite frame', 60),
    (37, 'weapon_charge', 'num', 'charge counter frames (known: P 51)', 60),
    (38, 'body_index', 'num', 'index in the frame', 4), (39, 'linked_body', 'num', 'partner index + 1, 0 none', 4),
    (40, 'stall_share', 'num', "decisions since the last progress / the stall limit, 0..1 (known: P 53)", 1),
    (48, 'health_known', 'known', 'not Curse of the Unknown', 1),
    (49, 'controller_known', 'known', 'ControllerIndex present', 1),
    (50, 'animation_known', 'known', 'animation name known', 1),
    (51, 'weapon_charge_known', 'known', 'P 37 read', 1),
    (52, 'bone_slots_known', 'known', 'IsBoneHeart readable (P 128..151)', 1),
    (53, 'stall_known', 'known', 'the stall rule exists here (AB+ training 1; a producer without it 0)', 1),
    (56, 'ptype', 'cat:ptype', 'PlayerType + 2 (-1 POSSESSOR -> 1)', 1),
    (57, 'body_role', 'cat:body_role', '1 main, 2 twin, 3 sub-player, 4 co-op baby, 5 co-op ghost', 1),
    (58, 'controller', 'cat:controller', 'ControllerIndex + 2', 1),
    (59, 'animation', 'cat:animation', 'index in the 14-name list + 1 (v2 order)', 1),
    (60, 'stage', 'cat:stage', 'LevelStage', 1), (61, 'stage_type', 'cat:stage_type', 'StageType + 1', 1),
    (62, 'room_type', 'cat:room_type', 'RoomType', 1), (63, 'difficulty', 'cat:difficulty', 'Difficulty + 1', 1),
    (64, 'game_version', 'cat:game_version', '1 AB+, 2 Rep, 3 Rep+', 1),
    (65, 'control_scheme', 'cat:control_scheme', 'Rep+ JacobEsauControls + 1, else 0', 1),
] + _block(72, 32, 'curse', 'bit', 'GetCurses bit i') \
  + _block(104, 24, 'black_slot', 'bit', 'GetBlackHearts bit i') \
  + _block(128, 24, 'bone_slot', 'bit', 'IsBoneHeart(i)') \
  + _block(152, 24, 'weapon', 'bit', 'HasWeaponType(i)') \
  + [(176 + i, f'overflow_{c}', 'num', 'entities dropped past the cap', 128) for i, c in enumerate(ENTITY_CLASSES)]
P_COLS = _cols(P_SPEC, PF)

# E: one token per entity (DESIGN 2.6)
E_SPEC = [
    (0, 'type', 'id', 'EntityType (identity)', 1), (1, 'variant', 'id', 'variant (identity)', 1),
    (2, 'subtype', 'id', 'subtype (identity; 0 for a pedestal under Curse of the Blind)', 1),
    (3, 'x', 'num', 'px from the room top-left', 520), (4, 'y', 'num', 'px from the room top-left', 280),
    (5, 'vx', 'num', 'px/frame', 20), (6, 'vy', 'num', 'px/frame', 20), (7, 'size', 'num', 'px', 20),
    (8, 'size_mx', 'num', 'SizeMulti x', 20), (9, 'size_my', 'num', 'SizeMulti y', 20),
    (10, 'coll', 'num', 'EntityCollisionClass', 1), (11, 'gcoll', 'num', 'GridCollisionClass', 1),
    (12, 'cdmg', 'num', 'CollisionDamage', 1), (13, 'sprite_frame', 'num', 'frame', 60),
    (14, 'age', 'num', 'FrameCount', 60), (15, 'kind', 'cat:entity_kind',
                                             '0 other, 1 tear, 2 projectile, 3 laser, 4 bomb, 5 pickup, 6 npc, '
                                             '7 knife, 8 familiar, 9 effect', 1),
    (16, 'npc_flags', 'bits8', '1 IsEnemy, 2 IsVulnerableEnemy, 4 IsBoss, 8 CanShutDoors', 1),
    (17, 'anim', 'num', 'short animation list index', 1), (18, 'flip_x', 'num', '0/1', 1),
    (19, 'champion', 'cat:champion', 'ChampionColorIdx + 2', 1),
    (20, 'boss_hp', 'num', 'visible boss bar fraction', 1), (21, 'height', 'num', 'Height', 30),
    (22, 'fall', 'num', 'FallingSpeed', 30), (23, 'scale', 'num', 'Scale', 30),
    (24, 'l_cos', 'num', 'laser direction cos', 1), (25, 'l_sin', 'num', 'laser direction sin', 1),
    (26, 'l_len', 'num', 'laser length px', 520), (27, 'l_ex', 'num', 'laser end x px', 520),
    (28, 'l_ey', 'num', 'laser end y px', 520), (29, 'l_radius', 'num', 'laser radius px', 520),
    (30, 'l_circle', 'num', '0/1', 1), (31, 'price', 'num', 'shop price (coins)', 100),
    (32, 'identity_known', 'known', 'type / variant / subtype readable', 1),
    (33, 'flags_known', 'bits8', '1 tear namespace, 2 projectile, 4 entity flags, 8 anim_id', 1),
] + _block(34, 8, 'tear_w', 'word16', 'TearFlags bits 16k..16k+15, canonical (Rep TEARFLAG) numbering') \
  + _block(42, 4, 'proj_w', 'word16', 'ProjectileFlags bits 16k..16k+15 (PROJ_HIDDEN masked)') \
  + _block(46, 4, 'eflag_w', 'word16', 'EntityFlags bits 16k..16k+15 AND ENTITY_VISIBLE') \
  + [(50, 'anim_id', 'anim', 'hash(anm2 file, animation) mod 4095 + 1, 0 unknown', 1)]
E_COLS = _cols(E_SPEC, EF)

# N: inventory (DESIGN 2.7)
N_SPEC = [(0, 'owner', 'num', 'body_index', 4), (1, 'id', 'item:collectible', 'CollectibleType', 1),
          (2, 'count', 'num', 'GetCollectibleNum', 4), (3, 'id_known', 'known', '', 1),
          (4, 'display_order', 'num', 'HUD order', 32), (5, 'display_order_known', 'known', '', 1)]
N_COLS = _cols(N_SPEC, NF)

# S: slots (DESIGN 2.7)
S_SPEC = [(0, 'owner', 'num', 'body_index', 4),
          (1, 'family', 'cat:slot_family', '1 active, 2 trinket, 3 pill, 4 card, 5 poop spell, 6 crafting bag item, '
                                           '7 crafting output', 1),
          (2, 'slot', 'num', 'active slot / pocket index / poop queue position', 4),
          (3, 'id', 'item:by_family', 'id in the family namespace (trinket & 0x7fff; pill effect + 1 if identified)',
           1),
          (4, 'occupied', 'num', '0/1', 1), (5, 'selected', 'num', '0/1 what the button uses now', 1),
          (6, 'charge', 'num', 'active charge', 12), (7, 'max_charge', 'num', 'MaxCharges', 12),
          (8, 'id_known', 'known', '', 1), (9, 'selected_known', 'known', '', 1), (10, 'charge_known', 'known', '', 1),
          (11, 'pill_colour', 'item:pill_colour', 'PillColor & 0x7ff', 1),
          (12, 'battery', 'num', 'GetBatteryCharge', 12), (13, 'golden', 'num', '0/1 trinket 0x8000 / PILL_GOLD', 1),
          (14, 'giant', 'num', '0/1 pill 0x800', 1), (15, 'display_order', 'num', 'pocket order', 4),
          (16, 'display_order_known', 'known', '', 1),
          (17, 'charge_type', 'cat:charge_type', 'ItemConfig ChargeType + 1', 1),
          (18, 'sub_charge', 'num', 'GetActiveSubCharge', 12), (19, 'sub_charge_known', 'known', '', 1)]
S_COLS = _cols(S_SPEC, SF)


def col(cols, name):
    """The column number of `name` in P_COLS / E_COLS / N_COLS / S_COLS."""
    for i, c in enumerate(cols):
        if c[0] == name:
            return i
    raise KeyError(name)


def scale_vector(cols):
    """The model's divisor per column (1 for every non-numeric / reserved column)."""
    return tuple(c[3] if c[1] == 'num' and c[3] else 1.0 for c in cols)


# ------------------------------------------------------------------------------------------------- F1: tables
# identity (existing, 65536 x 24): the raw id; < 0 or >= 65535 (or NaN) -> IDENTITY_UNKNOWN; 0 the padding row
IDENTITY_ROWS, IDENTITY_UNKNOWN = 65536, 65535
# item (new, 8192 x 24): family namespaces (base, size); id 0 -> row 0 (none); 0 < id < size -> base + id; any other
# id (negative, too large, NaN) -> ITEM_UNKNOWN. A pill effect is written as effect + 1 (0: unidentified / none).
ITEM_ROWS, ITEM_UNKNOWN = 8192, 8191
ITEM_NS = {'collectible': (0, 2048), 'trinket': (2048, 512), 'card': (2560, 256), 'pill_effect': (2816, 128),
           'pill_colour': (2944, 64), 'poop_spell': (3008, 64), 'crafting': (3072, 128)}   # 3200..8190 reserved
# the S family code -> its item namespace (S col 3)
FAMILY_ITEM_NS = {1: 'collectible', 2: 'trinket', 3: 'pill_effect', 4: 'card', 5: 'poop_spell', 6: 'crafting',
                  7: 'collectible'}
# cat (new, 1024 x 16): code 0 -> row 0 (none); else base + min(code, size - 1); a negative or NaN code -> base +
# size - 1 (the namespace's last row: out of range / unknown)
CAT_ROWS = 1024
CAT_NS = {'ptype': (0, 64), 'body_role': (64, 16), 'controller': (80, 16), 'animation': (96, 32), 'stage': (128, 32),
          'stage_type': (160, 16), 'room_type': (176, 64), 'difficulty': (240, 16), 'game_version': (256, 16),
          'control_scheme': (272, 16), 'entity_kind': (288, 16), 'champion': (304, 64), 'slot_family': (368, 16),
          'charge_type': (384, 16)}   # 400..1023 free
# code conventions: +2 (so -1 = 1 and 0 = 2; 0 stays "none") for these
CAT_PLUS2 = ('ptype', 'controller', 'champion')
# anim (new, 4096 x 16): hash(file, animation) mod 4095 + 1, 0 unknown (collisions accepted)
ANIM_ROWS = 4096
GAME_VERSION = {'abplus': 1, 'rep': 2, 'rep_plus': 3}
BODY_ROLE = {'main': 1, 'twin': 2, 'sub': 3, 'coop_baby': 4, 'coop_ghost': 5}
SLOT_FAMILY = {'active': 1, 'trinket': 2, 'pill': 3, 'card': 4, 'poop_spell': 5, 'crafting_item': 6,
               'crafting_output': 7}
ENTITY_KIND = {'other': 0, 'tear': 1, 'projectile': 2, 'laser': 3, 'bomb': 4, 'pickup': 5, 'npc': 6, 'knife': 7,
               'familiar': 8, 'effect': 9}


def _bad(v):
    return v != v   # NaN


def identity_index(v):
    """identity row of a raw entity id (the model does the same with torch.where)."""
    if _bad(v) or v < 0 or v >= IDENTITY_UNKNOWN:
        return IDENTITY_UNKNOWN
    return int(v)


def item_index(ns, v):
    base, size = ITEM_NS[ns]
    if _bad(v):
        return ITEM_UNKNOWN
    v = int(v)
    if v == 0:
        return 0
    return base + v if 0 < v < size else ITEM_UNKNOWN


def cat_index(ns, code):
    base, size = CAT_NS[ns]
    if _bad(code) or code < 0:
        return base + size - 1
    code = int(code)
    return 0 if code == 0 else base + min(code, size - 1)


def map_icon_index(code):
    if _bad(code) or code < 0:
        return MAP_ICONS - 1
    return min(int(code), MAP_ICONS - 1)


# ------------------------------------------------------------------------------------------------- F1: flags
FLAG_WORD_BITS = 16
TEAR_WORDS, PROJ_WORDS, EFLAG_WORDS = 8, 4, 4     # 128, 64 and 64 bits
# AB+ TearFlags bit -> canonical (Rep TEARFLAG(n)) bit, by name (DESIGN 2.6; MIGAN = MULLIGAN, BOMBERANG = BOOMERANG)
TEAR_AB_TO_CANON = {**{b: b for b in range(30)}, 30: 35, 31: 30, 32: 31, 33: 32, 43: 33, 60: 127,
                    **{b: b + 2 for b in range(34, 43)}, **{b: b + 2 for b in range(44, 60)}}
# ProjectileFlags bits hidden from the actor (Q5: future behaviour): CHANGE_FLAGS_AFTER_TIMEOUT 32,
# CHANGE_VELOCITY_AFTER_TIMEOUT 33; numbering identical on AB+ and Rep
PROJ_HIDDEN = (32, 33)
# EntityFlags shown (Rep numbering = AB+ numbering on 0..37): the visible-status mask
ENTITY_VISIBLE = (5, 6, 7, 8, 9, 10, 11, 12, 24, 29, 34, 49, 50, 51, 52, 54, 56, 57, 58)


def mask_bits(bits):
    m = 0
    for b in bits:
        m |= 1 << b
    return m


ENTITY_VISIBLE_MASK = mask_bits(ENTITY_VISIBLE)
PROJ_VISIBLE_MASK = ((1 << 64) - 1) & ~mask_bits(PROJ_HIDDEN)


def flag_words(value, n_words):
    """A flag integer as n 16-bit words (word k: bits 16k..16k+15), the wire's encoding."""
    return [float((int(value) >> (FLAG_WORD_BITS * k)) & 0xffff) for k in range(n_words)]


def tear_canonical(ab_value):
    """An AB+ TearFlags integer remapped to the canonical numbering."""
    out = 0
    for b, c in TEAR_AB_TO_CANON.items():
        if (int(ab_value) >> b) & 1:
            out |= 1 << c
    return out


# ------------------------------------------------------------------------------------------------- F2: actions
HEADS = ('move', 'shoot', 'bomb', 'item', 'pill', 'drop')
HEAD_SIZES = (9, 5, 2, 2, 2, 2)
N_HEADS = len(HEADS)                    # pending / actions int64 per decision
N_LOGITS = sum(HEAD_SIZES)              # pi outputs, the pending one-hot, the CRN uniform columns
CRN_COLS = N_LOGITS
DROP_BIAS = -4.0                        # pi.bias of drop = 1 at initialisation (TokPolicy USE_BIAS)
# ButtonAction names per head value (mapped by NAME: Rep+ renumbered ButtonAction)
MOVE = ((), ('UP',), ('UP', 'RIGHT'), ('RIGHT',), ('RIGHT', 'DOWN'), ('DOWN',), ('DOWN', 'LEFT'), ('LEFT',),
        ('LEFT', 'UP'))
SHOOT = ((), ('SHOOTUP',), ('SHOOTRIGHT',), ('SHOOTDOWN',), ('SHOOTLEFT',))
BUTTON = {'bomb': 'BOMB', 'item': 'ITEM', 'pill': 'PILLCARD', 'drop': 'DROP'}
# AB+ / Rep J746 ButtonAction numbers (Rep+: by name, values unverified)
BUTTON_ACTION_ABP = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'SHOOTLEFT': 4, 'SHOOTRIGHT': 5, 'SHOOTUP': 6,
                     'SHOOTDOWN': 7, 'BOMB': 8, 'ITEM': 9, 'PILLCARD': 10, 'DROP': 11}
# action_code = m + 9 s + 45 b + 90 i + 180 p + 360 d (720 codes)
ACTION_CODE_MUL = (1, 9, 45, 90, 180, 360)
N_ACTION_CODES = 720
# the step line: 'S rep m s b i p d' (the 7th number absent = 0); JSON step key 'drop'
STEP_LINE_FIELDS = ('rep', 'move', 'shoot', 'bomb', 'item', 'pill', 'drop')


def action_code(m, s, b, i, p, d=0):
    return m * 1 + s * 9 + b * 45 + i * 90 + p * 180 + d * 360


def action_of_code(code):
    out = []
    for size in HEAD_SIZES:
        out.append(code % size)
        code //= size
    return tuple(out)


# ------------------------------------------------------------------------------------------------- F4: records
ACTION_F = N_HEADS     # tok_sampler: the applied action's fields per record
# teacher v2 / CF imitation records (tok_teacher2.V2_FIELDS in v3): the action under way and the applied one (6
# each), then v2's fields unchanged
V2_FIELDS = tuple(f'pend{k}' for k in range(N_HEADS)) + tuple(f'act{k}' for k in range(N_HEADS)) + (
    'weight', 'ret', 'gain', 'spread', 'taken', 'best', 'j', 'n', 'depth', 'hold', 'intent', 'kind', 'point',
    'worker', 'version', 'value', 'logp', 'rep', 'trig')
V2_META = len(V2_FIELDS)
# injury teacher records (tok_sampler TEACH_META in v3): the action under way (all 6 heads), nine danger bits, the
# move taken, the depth, the safe moves, 0
TEACH_FIELDS = tuple(f'pend{k}' for k in range(N_HEADS)) + tuple(f'danger{k}' for k in range(9)) + (
    'move', 'depth', 'safe', 'pad')
TEACH_META = len(TEACH_FIELDS)
