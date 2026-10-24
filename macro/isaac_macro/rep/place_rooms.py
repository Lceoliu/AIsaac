"""Level::place_rooms of Repentance+ v1.9.7.17 (J460, Windows x86), RVA 0x339370 (VA 0x739370-0x73D9A9).

Ported from the re-decompiled corpus analysis/j460/exports/j460-fixed/{generator,ranges} and, for every
call whose arguments Ghidra loses (stack arguments after log calls, register arguments, float returns),
from a capstone disassembly of rl/runs/human/20260923-session01/worker/runtime/isaac-ng.exe (read only),
by diffing against the engine-validated AB+ translation isaac_macro.level.Level.place_rooms. Addresses
are RVAs (VA = RVA + 0x400000) unless marked VA. Static translation only: nothing here has been checked
against the running J460 engine.

Functions (RVA -> Python)
  Level::place_rooms                        0x339370  place_rooms
  Level::place_room (5 args, 4th unread)    0x338AB0  place_room
  RoomDescriptor seed init (name ours)      0x028940  init_seeds
  Level::build_secret_room_index_blacklist  0x338D70  secret_room_blacklist (same as AB+ 0x337210)
  ultra secret blacklist (inline)           VA 0x73C9E2-0x73CAF8  ultra_secret_blacklist
  extra-room blacklist (name ours)          0x352460  end_room_blacklist
  Game::GetPlanetariumChance                0x34DBD0  planetarium_chance
  strange door floor (name ours)            0x34F030  strange_door_floor
  barren bedroom allowed (name ours)        0x34B5D0  barren_bedroom_allowed (normal-run reduction of
                                                      0x34B5D0 / 0x34B710 / 0x34EA50, see there)
  Devil's Crown treasure check (name ours)  0x34D8D0  inside place_room
  RoomConfig GetRandomRoom + stage fallback 0x42CDD0  get_random_room2
  PlayerManager / Entity_Player queries     0x04BFB0 AnyoneHasTrinket, 0x5BE080 FirstCollectibleOwner,
                                            0x5BE6B0 FirstTrinketOwner, 0x5BEA80 GetTrinketMultiplier,
                                            0x5BEB30 (any full health), 0x5BE630 (first player of a type),
                                            0x5BF930 (Cain with Birthright), 0x3CB6E0 GetTrinketMultiplier,
                                            0x371550 HasTrinket, 0x371620 (golden trinket), 0x3CAFE0
                                            GetHealthType, 0x3DB6B0 (lost-like) -> single-player helpers below
  Boss ids and boss room layouts come from rep/bosspool.py: BossPool.get_boss_id (RVA 0x22830) and
  pick_boss_room (RVA 0x339080). Generator calls go to rep/levelgen.py (see "APIs used" at the end).

J460 order (the level RNG is Level+0x182D4, triple 35; "draw" = one Next of it; a room "goes back" =
LevelGenerator::AddEndRoom, inlined: the dead end is appended to the dead-end list again; "place" =
place_room with the level RNG state as seed unless noted; every GetRandomRoom decays the weight of its
pick unless noted; stage' = stage + 1 on stage types 4/5, else stage)
  1 Boss: GetBossId(stage, type) (BossPool pool RNG, not the level RNG); draw; picker(boss, draw);
    GetNewBossRoom(force 0) -> fail if none; place. Level+0x18314 (last boss) and +0x18330 (first boss)
    = list index.
  2 Void (stage 12): Random(4)+5 extra bosses from ids 0..69 minus {55, 63, 6, 62} and the 37-entry
    table VA 0xBAB5D0.. (= AB+'s), Fisher-Yates shuffled with the level RNG; per boss: draw, picker,
    GetNewBossRoom(force 1); none -> fail for the first five, else stop; place.
  3 Curse of the Labyrinth: save the level RNG; GetBossId(stage + 1, type); draw; picker; GetNewBossRoom;
    place; restore (seed, triple 35).
  4 The last boss room's cells and their 4 neighbours become blocked (LevelGenerator 0x5B1860): no
    ultra secret room, secret room or CreateRandomEndRoom cell next to it. Skipped when the Void's last
    GetNewBossRoom failed.
  5 Super secret room (type 8, not challenge-filtered): draw s; local RNG(s, triple 9); 2 rooms with
    Luna (589) else 1; per room: pick (stage 0) with the local RNG, GetNewEndRoom, none -> fail for the
    first, else stop; place with the local seed.
  6 secret seed = draw.
  7 Shop (type 2), when stage < 7, or stage 7/8 with Silver Dollar (110), or Sheol with Wicked Crown
    (161), or Cathedral with Holy Crown (155); not filtered; fewer than 3 victory laps. Level: draw s,
    local RNG(s, 19): a = 1+U(151)+U(153), b = 1+U(152)+U(154), level = Random(a)+Random(b), replaced by
    the maximum a+b-2 unless (next draw odd and hard/greedier); next draw's low byte: 0 -> level 11
    (unless Store Key (83) x2+), 1 with level >= 2 -> level 10; golden crown/Silver Dollar or Store Key
    x3+ -> one more draw: &3 == 0 -> 10, 11 -> 0, below 4 -> +1 (4 -> 10 without a draw); Tainted
    Keeper +100. Pick (stage 0, subtype = level) with a draw; GetNewEndRoom -> fail if none; place.
    Crown multiplier > 2 (golden + Mom's Box) shows the shop icon (DisplayFlags |= 4).
  8 Treasure (type 4) under the same stage/trinket rule (Bloody Crown 111 on stage 7/8), not filtered,
    twice on a Labyrinth floor: draw %100 == 0 -> options, else d = 100 // max(Golden Horse Shoe (82)
    multiplier, 1), draw, (d == 0 or draw % d < 15) with the trinket -> options, else special daily 19
    -> options; subtype options 1 / 3 (Pay To Win 112), else 0 / 2; pick with a draw from the current
    stage file, falling back to stage 0 (Womb..Cathedral and the alt-path files have treasure rooms);
    GetNewEndRoom -> fail if none; place; crown icons as the shop. The level RNG is set back to its
    state after the first treasure room. (Filtered or off-stage: a challenge-only mirror-floor case.)
  9 Planetarium (type 24): chance; pick with a draw (stage 0); GetNewEndRoom (may be none); filtered
    (24 or 4) -> goes back; else, if a room was found: draw f; chance > f and U(406 The Planetarium)
    -> place, else back.
 10 stage < 11:
    a Dice (21) / sacrifice (13): draw %50 == 0 or (draw %5 == 0 and keys > 1) -> dice unless filtered;
      pick with a draw; GetNewEndRoom; sacrifice filtered -> back; draw %7 == 0 -> place; else draw,
      &3 == 0 and (a player with red+soul >= max or lost-like, or a Lazarus 8/11) -> place, else back.
    b Library (12): subtype Random(1 + library level 0..4 from U(151..154)); pick with a draw, no weight
      decay; filtered -> back; draw %20 == 0 -> place; else draw, &3 == 0 and state flag 8 -> place.
    c Curse (10): draw s; local RNG(s, 66); 2 rooms with Voodoo Head (599) else 1; per room: pick
      (subtype 1 first with Voodoo Head, then subtype 0) with local draws, GetNewEndRoom; filtered ->
      back; else one local draw, a second when it is odd, and the room is ALWAYS placed (local seed).
    d Miniboss (6): stage' >= 3: draw %10 == 0 and not flag 32 -> draw, Ultra Pride (subtype 14); else
      super = Random(d) == 0 (d = 80, 30 with U(4 The Womb), 10 with U(33 Everything is Terrible), 5 when
      Manager+0x330+0x334 != 0; no
      draw on stage' < 3), list of unseen sins (flags 9..14 -> subtypes 2/9, 3/10, 1/8, 0/7, 5/12,
      6/13), subtype = list[Random(len)]; pick with a draw, difficulty exactly 1 (stage < 7) or 10,
      else a second pick with a draw, difficulty 1..10; no sin left -> the curse room's layout and dead
      end are reused (as AB+). Filtered -> back; draw &3 == 0, or (draw %3 == 0 and stage' == 1) ->
      place and set the sin's state flag (9..14, Ultra Pride 32), else back.
    e Challenge (11): subtype 1 on stages 2/4/6/8 (Level+8, set by Level::Init VA 0x744CF0), else 0;
      pick with a draw; filtered -> back; draw odd and stage' < 3 -> back; nobody at full health (or
      lost-like) -> back; stage' < 2 -> back; else place.
    f Arcade (9) / vault (20): Cain with Birthright (619) = cbr; draw %10 == 0 or (draw %3 == 0 and
      keys > 1) -> vault (unless filtered): cbr -> arcade subtype 1, else a vault pick; otherwise cbr ->
      arcade subtype 1 (subtype 0 if none), else arcade subtype 0; one draw per pick. Placed when its
      type is not filtered, (arcade with coins >= 5 or cbr, or vault with keys >= 2) and (stage' in
      2/4/6/8, or cbr with stage' <= 10); else back.
 11 stage < 7, bedroom: draw even (or dirty bedroom not allowed) and Isaac's (18) not filtered ->
    Isaac's, else Barren (19); pick with a draw; draw %50 == 0 or (draw %5 == 0 and low health) ->
    place, else back.
 12 Downpour/Dross II (mirror world): pick subtype 34, difficulty 0 from the current stage file with a
    draw, GetNewEndRoom -> fail if none, place; 1 extra room. Mines/Ashpit II with Knife Piece 1 (626):
    local RNG(level seed, 34) one step (no level draw), pick subtype 10 difficulty 0 from the current
    stage file then Mines (29), GetNewEndRoom -> fail if none, place with the local seed; 2 extra rooms.
    Depths II (or I on a Labyrinth floor), not alt path, with Manager+0x2C7 (the byte of achievement
    635, RepGameContext.strange_door, as for generate_dungeon's (6,5) block): 1 extra room.
 13 Secret room (type 7, not filtered): 1 room, 2 with Fragmented Card (102), +1 with Luna; seeds:
    the secret seed, then RNG(seed, triple 1) steps; per room: pick with that seed from the current
    stage file then stage 0 (the alt-path files have secret rooms); blacklist 0x338D70 from the rooms
    placed so far; GetNewSecretRoom (skips blocked cells; none -> no room, no failure); place with the
    same seed. Fragmented Card multiplier 2 shows the first, 3+ the first two (DisplayFlags |= 4).
 14 Dark Room (stage 11 type 0): draw; U(390 The Forgotten) -> pick variants 3..9 from stage 0, else
    GetRoom(0, 1, 3);
    GetNewEndRoom -> fail if none; place with the draw.
 15 Ultra secret room (type 29, not filtered): seed = RNG(secret seed, triple 71) one step (no level
    draw); pick from the current stage file then stage 0; none -> skipped; blacklist: door targets
    (ignore_narrow) of every placed room slot without a layout door, or of every slot of boss / super
    secret / secret / curse rooms; GetNewUltraSecretRoom (none -> no room); place with the seed; when
    placed, the descriptor's +0xB4 = 99.
 16 Every remaining room (non-dead-ends, then unused dead ends): min difficulty 1 when the level RNG
    state & 7 == 0 and the minimum is above 1; the start room is GetRoom(0, 1, 2) (Dark Room 16/1/0,
    Chest 17/1/0); others: pick with a draw from the current stage file, subtype 0, variant >= 1 on
    stage 11, the room's shape and doors -> fail if none; place.
 17 One draw per generator room that was not in that list (every special room, secret rooms included).
 18 Extra rooms (step 12): local RNG(level seed, 41) shuffles the (descriptor, room) pairs of step 16;
    skipping the start room, the first `extra` pairs get a subtype 1 DEFAULT layout (min difficulty 1
    when local state & 7 == 0 and the minimum is above 1; retry with 1..10) from the current stage
    file, else the family file (Dross -> Downpour 27, Ashpit -> Mines 29, Necropolis/Dank -> Depths 7);
    the descriptor keeps its seeds and takes the layout and its doors. Past the list,
    CreateRandomEndRoom (1x1, blacklist 0x352460) makes a room (none -> fail), placed with a local
    draw. Rooms still missing -> fail.
 19 Surprise miniboss (Greed): as AB+ (flags 15/16, coins >= 20 -> 1.05, flags 30/31 +0.1/+0.02 and
    cleared, counter Game+0x2656C min 5 x 0.05 and zeroed, x 1/3; two draws; secret room on stage > 4,
    shop on stage > 3; Flags |= 0x10).
 20 Off-grid rooms (GridRooms): -2 error (3), -6 black market (22), -4 crawlspace (16): seeds from 3
    draws + pick with a draw; -5 boss rush: seeds, a draw, then GetRoom(0, 17, 0) with Broken Shovel 1
    (550) else a pick; -7 Mega Satan (boss subtype 55); -8 (-9 in the Blue Womb file) stage 13 DEFAULT
    subtype 1 difficulty 0; -13 secret shop: draw s, local RNG(s, 19), level Random(1+U(153)) +
    Random(1+U(152)+U(154)), seeds and pick from the local RNG; -18 angel shop: draw s, local RNG(s,
    20), seeds and a subtype 1 angel room (15); Corpse II (or I on a Labyrinth floor): a draw, seeds,
    -10 Mother (boss subtype 88, current stage file).

Differences from AB+ (Level::place_rooms 0x337740), with evidence
  - Bosses: BossPool (bosspools.xml, own per-pool RNGs) replaces choose_boss/choose_double_trouble,
    so boss ids no longer consume the level RNG; every layout pick is one draw + the picker's
    RNG(draw, 39) (VA 0x739475-0x7394C3, 0x73975A, 0x739926-0x73997D), which AB+ used for the first
    boss only (its Labyrinth and Void layouts drew straight from the level RNG).
  - The last boss room blocks its surroundings (VA 0x739A0D).
  - Super secret room: a local RNG (triple 9, VA 0x739A8F) and a second room with Luna (VA 0x739ADE);
    AB+ picked with a draw and placed with the level seed.
  - Shop: the level comes from a local RNG with random/maximum rules, Store Key, crowns, Silver Dollar
    and Tainted Keeper (VA 0x739CCA-0x73A1A1); AB+: unlocks + Random(256) 10/11. Wicked/Holy Crown
    shops and treasure rooms on Sheol/Cathedral are new.
  - Treasure rooms come from the current stage file first (GetRandomRoom2 VA 0x73A7AD); the special
    daily check applies to both Labyrinth rooms (AB+: the first only).
  - New: planetarium (VA 0x73A8F7-0x73AB45), ultra secret room (VA 0x73C902-0x73CB5D), mirror /
    mineshaft entrance / strange-door extra rooms (VA 0x73C416-0x73C69E, 0x73CDDC-0x73D150), off-grid
    secret shop, angel shop, Mother room; Luna's second secret room; the draws of step 17.
  - Sacrifice: lost-like players and Lazarus count as healthy (VA 0x73ADA9-0x73ADDA).
  - Curse room: always placed when not filtered (VA 0x73B30B-0x73B38F); AB+: Random(2) == 0 or
    (Random(4) == 0 and devil room visited). Local RNG (triple 66) instead of level draws; Voodoo Head.
  - Miniboss: chosen by subtype (0..14) instead of variant ranges, difficulty exactly 1/10 first, no
    super-sin draw before stage' 3 (AB+ always drew it), the 1-in-5 condition is Manager+0x330+0x334
    != 0 (AB+: six achievements), stage' instead of stage for the Ultra Pride and stage-1 tests.
  - Challenge room: boss challenge subtype on even stages (AB+: any subtype).
  - Vault: needs keys >= 2 (AB+: coins >= 5 and arcade not filtered); Cain's Birthright.
  - Bedroom: the dirty bedroom also needs 0x34B5D0.
  - Second secret room placed with its own seed (AB+: the secret seed); secret rooms from the stage
    file first; Fragmented Card / Luna icons.
  - Normal rooms: subtype 0 (AB+: any) and the 1-in-8 minimum difficulty 1 (VA 0x73CC08).
  - Boss rush with Broken Shovel draws too (VA 0x73D530); stage 13 room difficulty 0 (AB+: 1..10).
  - place_room: limit 507 rooms (AB+ 128), fourth seed, Devil's Crown and Corpse backdrop flags,
    dimension field (Level+0x1830C).

Modes: Greed, challenges and daily runs (Game+0x26630) raise NotImplementedError; co-op / online is
reduced to one player (below); special seeds are not modelled (their permanent / banned curse masks,
0x2F9400 / 0x2F95A0, are taken as 0, as in RepLevel.get_curses).

Not settled (guesses are marked so)
  - Challenge parameters are the default entry (no room filter, +0x74 = 0, +0x80 = 0) - assumed for
    challenge 0. The branches that read them are ported anyway (room filter, +0x80 == 2 mirror
    treasure) except the +0x74 boss skip, which raises.
  - Names: Manager+0x330/+0x334 (RepGameContext.super_sin_counter), Game+0x26558/+0x2655C (treasure
    rooms entered / planetariums entered, from the formula), descriptor +0x64 ("boss death seed",
    guess) and +0xB4 (99 on the ultra secret room; DeliriumDistance is a guess).
  - Multiplayer (co-op babies, Jacob & Esau, The Forgotten's soul, "first owner" preferences) is reduced
    to one player; Entity_Player+0x2EF8/+0x2EF0 (a special trinket slot) and smelted trinkets are not
    modelled; Rainbow Worm and the other GetTrinketMultiplier special cases are not modelled.
  - barren_bedroom_allowed assumes a normal run (no challenge/daily, not Greed, dimension 0); the
    Mausoleum II Mom-room check (Level+0x18300, only reachable when regenerating that floor) returns
    the same result as the flag 47 check it is combined with.
  - The trinket id | 0x8000 golden encoding of RepPlayer.trinkets follows the engine's slot encoding.

APIs used: rep/levelgen.py LevelGenerator.get_new_boss_room(shape, doors, force), block_room_area(room),
get_new_end_room(shape, doors), add_end_room(room), get_new_secret_room(blacklist),
get_new_ultra_secret_room(blacklist), create_random_end_room(1, blacklist), get_remaining_rooms(),
`rooms`; GenRoom .x .y .shape .doors .depth; rep/bosspool.py BossPool.get_boss_id(stage, stage_type,
ctx=BossContext), pick_boss_room(rc, boss_id, seed, ctx, level_stage_id=, curses=, void_level=),
BossContext. Level::generate_dungeon (rep/level.py) clears / commits the BossPool level blacklist
around the attempts (VA 0x74176C / 0x741C22).
"""
from __future__ import annotations

from dataclasses import dataclass


from ..level import (CURSE_LABYRINTH, Player, ROOM_ARCADE, ROOM_BARREN, ROOM_BLACK_MARKET, ROOM_BOSS,
                     ROOM_BOSSRUSH, ROOM_CHALLENGE, ROOM_CHEST, ROOM_CURSE, ROOM_DEFAULT, ROOM_DICE,
                     ROOM_DUNGEON, ROOM_ERROR, ROOM_ISAACS, ROOM_LIBRARY, ROOM_MINIBOSS, ROOM_SACRIFICE,
                     ROOM_SECRET, ROOM_SHOP, ROOM_SUPERSECRET, ROOM_TREASURE, RoomDesc)
from ..rng import RNG
from .bosspool import BossContext, pick_boss_room
from .levelgen import GRID, PLACEMENT, door_target, index
from .roomconfig import SHAPE_ANY, stage_id

from ..f32 import F32  # noqa: E402
ANY_VARIANT = 0xFFFFFFFF

ROOM_ANGEL, ROOM_PLANETARIUM, ROOM_ULTRASECRET = 15, 24, 29

# xorshift triples (s_Shifts index; VA = 0xB1F4C8 + 12 * index)
LEVEL_SHIFT = 35         # (5, 9, 7)   the level RNG
DESC_SHIFT = 12          # (2, 7, 9)   place_room's seed RNG (VA 0xB1F558), Greed rolls
DESC_EXTRA_SHIFT = 66    # (11, 7, 12) the fourth descriptor seed (VA 0xB1F7E0)
CORPSE_SHIFT = 7         # (1, 21, 20) Corpse backdrop roll (VA 0xB1F51C)
SUPERSECRET_SHIFT = 9    # (2, 5, 15)  VA 0xB1F534
SHOP_SHIFT = 19          # (3, 3, 26)  VA 0xB1F5AC (shop and secret shop levels)
ANGEL_SHOP_SHIFT = 20    # (3, 3, 28)  VA 0xB1F5B8
CURSE_SHIFT = 66         # (11, 7, 12) VA 0xB1F7E0
MINESHAFT_SHIFT = 34     # (5, 7, 22)  VA 0xB1F660
SECRET_SHIFT = 1         # (1, 5, 16)  VA 0xB1F4D4
ULTRASECRET_SHIFT = 71   # (13, 3, 17) VA 0xB1F81C
EXTRA_ROOM_SHIFT = 41    # (5, 21, 12) VA 0xB1F6B4

MAX_GRID_ROOMS = 0x1FB   # place_room: "[warn] Could not place room, limit of %d exceeded!"
DIMENSION_NORMAL = 0     # Level+0x1830C during generate_dungeon

# The Void's extra bosses: ids 0..69 minus these (VA 0x739600) and the table VA 0xBAB5D0.. (37 ids)
VOID_SKIPPED_BOSSES = (0x37, 0x3F, 6, 0x3E)
VOID_EXCLUDED_BOSSES = (60, 43, 50, 23, 67, 69, 61, 57, 44, 64, 56, 65, 47, 45, 43, 50, 52, 9, 10, 18,
                        32, 36, 27, 13, 14, 20, 17, 28, 2, 1, 21, 3, 4, 34, 37, 29, 26)

# float32 constants (.rdata)
PLANET_BASE = F32(0.009999999776482582)     # VA 0xBAA06C
PLANET_PER_SKIP = F32(0.20000000298023224)  # VA 0xBAA198
ONE = F32(1.0)                              # VA 0xBAA454
PLANET_ITEM = F32(0.15000000596046448)      # VA 0xBAA154
PLANET_SAUSAGE = F32(0.0689999982714653)    # VA 0xBAA0F0
PLANET_LENS_ONE = F32(0.18000000715255737)  # VA 0xBAA17C (golden xor Mom's Box)
PLANET_LENS_BOTH = F32(0.33000001311302185) # VA 0xBAA220
PLANET_LENS_NONE = F32(0.09000000357627869) # VA 0xBAA10C
GREED_RICH = F32(1.0499999523162842)        # VA 0xBAA47C
GREED_FLAG30 = F32(0.10000000149011612)     # VA 0xBAA120
GREED_FLAG31 = F32(0.019999999552965164)    # VA 0xBAA08C
GREED_STEP = F32(0.05000000074505806)       # VA 0xBAA0D0
GREED_SCALE = F32(0.3333333432674408)       # VA 0xBAA230

# collectibles / trinkets / player types (resources/scripts/enums.lua)
C_CRYSTAL_BALL, C_MAGIC_8_BALL, C_MOMS_BOX, C_BROKEN_SHOVEL_1 = 158, 194, 439, 550
C_LUNA, C_VOODOO_HEAD, C_BIRTHRIGHT, C_KNIFE_PIECE_1, C_SAUSAGE = 589, 599, 619, 626, 669
T_GOLDEN_HORSE_SHOE, T_STORE_KEY, T_FRAGMENTED_CARD = 82, 83, 102
T_SILVER_DOLLAR, T_BLOODY_CROWN, T_PAY_TO_WIN = 110, 111, 112
T_DEVILS_CROWN, T_TELESCOPE_LENS, T_HOLY_CROWN, T_WICKED_CROWN = 146, 152, 155, 161
TRINKET_GOLDEN = 0x8000
P_CAIN, P_LAZARUS, P_LAZARUS2, P_KEEPER_B, P_JACOB2_B = 2, 8, 11, 33, 39
# Entity_Player::GetHealthType (0x3CAFE0, jump table VA 0x7CB018 / 0x7CB02C): 1 soul, 2 lost, 3 coin, 4 bone
HEALTH_TYPE = {4: 1, 10: 2, 12: 1, 14: 3, 16: 4, 17: 1, 24: 1, 25: 1, 31: 2, 33: 3, 35: 1, 36: 1, 40: 2}

# achievements (achievements.xml ids; Unlocked(id), inline as Manager+0x4C+id)
A_THE_WOMB, A_EVERYTHING_IS_TERRIBLE = 4, 0x21
A_STORE_1, A_STORE_2, A_STORE_3, A_STORE_4 = 0x97, 0x98, 0x99, 0x9A   # Store Upgrade lv.1-4
A_THE_FORGOTTEN = 0x186        # Manager+0x1D2: the Dark Room tomb variants 3..9
A_PLANETARIUM = 0x196          # Manager+0x1E2 "The Planetarium"

# GameStateFlag (enums.lua)
F_BOOK_PICKED_UP = 8
SIN_FLAGS = ((9, 2, 9), (10, 3, 10), (11, 1, 8), (12, 0, 7), (13, 5, 12), (14, 6, 13))  # flag, sin, super
F_GREED_SPAWNED, F_SUPERGREED_SPAWNED = 15, 16
F_DONATION_SLOT_BLOWN, F_SHOPKEEPER_KILLED = 30, 31
F_ULTRAPRIDE_SPAWNED = 32
F_BACKWARDS_PATH_INIT = 47
MB_ULTRA_PRIDE = 14

# GridRooms (off-grid descriptors: list index 0x1FA - idx, reset_room_list 0x338650)
ROOM_ERROR_IDX, ROOM_DUNGEON_IDX, ROOM_BOSSRUSH_IDX, ROOM_BLACK_MARKET_IDX = -2, -4, -5, -6
ROOM_MEGA_SATAN_IDX, ROOM_BLUE_WOOM_IDX, ROOM_THE_VOID_IDX, ROOM_SECRET_EXIT_IDX = -7, -8, -9, -10
ROOM_SECRET_SHOP_IDX, ROOM_ANGEL_SHOP_IDX = -13, -18
BOSS_MEGA_SATAN, BOSS_MOTHER = 0x37, 0x58

DISPLAY_ICON = 4                                   # RoomDescriptor DisplayFlags (+0x3C)
FLAG_SURPRISE_MINIBOSS, FLAG_DEVIL_TREASURE, FLAG_ALT_BACKDROP = 0x10, 0x800, 0x1000   # Flags (+0x44)
TREASURE_KNIFE_PIECE = 0x22     # treasure subtype excluded from Devil's Crown (0x34D8D0)
STAGE_ID_CORPSE = 0x21
STAGE_ID_MINES, STAGE_ID_BLUE_WOMB = 0x1D, 0x0D
EXTRA_ROOM_FAMILY = {0x1C: 0x1B, 0x1E: 0x1D, 8: 7, 9: 7}   # VA 0x73CE79-0x73CF4B


@dataclass
class RepRoomDesc(RoomDesc):
    """RoomDescriptor (0xB8 bytes, Level+0x14 + 0xB8 * list index). J460 offsets: GridIndex +0x0,
    SafeGridIndex +0x4, ListIndex +0x8, dimension +0xC, Data +0x10, doors +0x18, DisplayFlags +0x3C,
    Flags +0x44, seeds +0x58 (decoration) +0x5C (spawn) +0x60 (award) +0x64."""
    display_flags: int = 0
    dimension: int = DIMENSION_NORMAL
    boss_death_seed: int = 0   # +0x64 = RNG(award seed, triple 66).Next(); the name is a guess
    field_b4: int = 0          # +0xB4, 99 on the ultra secret room; meaning not settled


@dataclass
class RepPlayer(Player):
    """The J460 player state place_rooms reads beyond the AB+ Player. `trinkets` holds trinket ids
    with the engine's golden bit (id | 0x8000); `lost_curse` is NullItemID 112 (LOST_CURSE), which with
    PLAYER_JACOB2_B makes a player count as healthy (0x3DB6B0). A plain Player works too (no curse)."""
    lost_curse: bool = False


@dataclass(frozen=True)
class ChallengeParams:
    """The ChallengeParam fields place_rooms reads (Manager::GetChallengeParams 0x306940)."""
    room_filter: frozenset = frozenset()   # +0x18 std::set<int>: room types that are never placed
    skip_boss_on_stage_11: bool = False    # +0x74
    field_80: int = 0                      # +0x80 (2: mirror-floor treasure room when filtered)


NORMAL_RUN_PARAMS = ChallengeParams()


# ---------------------------------------------------------------------------------------- player
def has_collectible(p: Player, item: int) -> bool:
    """Entity_Player::HasCollectible for one player (an active item counts)."""
    return item in p.collectibles or p.active_item == item


def trinket_multiplier(p: Player, trinket: int) -> int:
    """Entity_Player::GetTrinketMultiplier (0x3CB6E0) without the special cases: 1 per held copy, 2 per
    golden one (id | 0x8000), +1 with Mom's Box when any."""
    n = 0
    for t in p.trinkets:
        if (t & 0x7FFF) == trinket:
            n += 2 if t & TRINKET_GOLDEN else 1
    if n > 0 and has_collectible(p, C_MOMS_BOX):
        n += 1
    return n


def has_trinket(p: Player, trinket: int) -> bool:
    """HasTrinket(t, false) (0x371550) = GetTrinketMultiplier(t) > 0; also AnyoneHasTrinket (0x04BFB0)
    and FirstTrinketOwner (0x5BE6B0) != 0 for one player."""
    return trinket_multiplier(p, trinket) > 0


def has_golden_trinket(p: Player, trinket: int) -> bool:
    """0x371620: a held copy with the golden bit."""
    return any((t & 0x7FFF) == trinket and t & TRINKET_GOLDEN for t in p.trinkets)


def effective_max_hearts(p: Player) -> int:
    """Entity_Player::GetEffectiveMaxHearts (0x3CB060): bone hearts count unless health type 1/2."""
    if HEALTH_TYPE.get(p.player_type, 0) in (1, 2):
        return p.max_hearts
    return p.max_hearts + p.bone_hearts * 2


def lost_like(p: Player) -> bool:
    """0x3DB6B0: the LOST_CURSE effect or PLAYER_JACOB2_B."""
    return bool(getattr(p, 'lost_curse', False)) or p.player_type == P_JACOB2_B


def full_health(p: Player) -> bool:
    """PlayerManager 0x5BEB30(false) for one player: red + soul >= max, or lost-like."""
    return p.hearts + p.soul_hearts >= p.max_hearts or lost_like(p)


def low_health(p: Player) -> bool:
    """The bedroom test (VA 0x73C280-0x73C2EF): (red < 2 and no soul) or (no effective containers
    and soul < 3)."""
    return (p.hearts < 2 and p.soul_hearts <= 0) or (effective_max_hearts(p) <= 0 and p.soul_hearts <= 2)


# ---------------------------------------------------------------------------------------- context
def challenge_params(ctx) -> ChallengeParams:
    if ctx.challenge or getattr(ctx, 'special_run', False):
        raise NotImplementedError('challenge / daily challenge parameters (GetChallengeParams 0x306940)')
    return NORMAL_RUN_PARAMS


def boss_context(ctx) -> BossContext:
    """rep/bosspool.py's view of RepGameContext (built per call: the state flags change during a run)."""
    return BossContext(achievements=ctx.achievements, manager_state=2,
                       daily_id=int(bool(getattr(ctx, 'special_run', False))), is_debug=False,
                       special_daily_id=getattr(ctx, 'special_daily_id', 0), challenge=ctx.challenge,
                       difficulty=ctx.difficulty, state_flags=ctx.state_flags)


def _flag(ctx, bit: int) -> bool:
    """Game::GetStateFlag (0x217A0) / the bitset test 0x23250: Game+0x26548, bit clamped to [0, 52]."""
    return bool((ctx.state_flags >> min(max(bit, 0), 52)) & 1)


def _set_flag(ctx, bit: int, value: bool) -> None:
    if value:
        ctx.state_flags |= 1 << bit
    else:
        ctx.state_flags &= ~(1 << bit)


def _stage_prime(level) -> int:
    return level.stage + 1 if level.stage_type in (4, 5) else level.stage


# ---------------------------------------------------------------------------------------- rooms
def init_seeds(rng: RNG) -> tuple[int, int, int, int]:
    """0x028940 (RoomDescriptor, thiscall): +0x58, +0x5C, +0x60 = three Next of `rng`; +0x64 = one step
    of RNG(third, triple 66), which does not advance `rng`."""
    deco, spawn, award = rng.next(), rng.next(), rng.next()
    return deco, spawn, award, RNG(award, DESC_EXTRA_SHIFT).next()


def get_random_room2(level, seed: int, reduce_weight: bool, stage: int, fallback_stage: int, rtype: int,
                     shape: int, min_difficulty: int, max_difficulty: int, doors: int, subtype: int):
    """0x42CDD0: GetRandomRoom on `stage` (any variant, mode -1); when it finds nothing, the same draw on
    `fallback_stage`. Its 7th, 8th and 13th stack arguments are never read."""
    rc, curses, void = level.rc, level.get_curses(), level.stage == 12
    room = rc.get_random_room(seed, reduce_weight, stage, rtype, shape, 0, ANY_VARIANT, min_difficulty,
                              max_difficulty, doors, subtype, void_level=void, curses=curses)
    if room is None:
        room = rc.get_random_room(seed, reduce_weight, fallback_stage, rtype, shape, 0, ANY_VARIANT,
                                  min_difficulty, max_difficulty, doors, subtype, void_level=void, curses=curses)
    return room


def _pick(level, seed: int, rtype: int, subtype: int = -1, stage: int = 0, reduce_weight: bool = True,
          min_variant: int = 0, max_variant: int = ANY_VARIANT, min_difficulty: int = 1,
          max_difficulty: int = 10, shape: int = SHAPE_ANY, doors: int = 0):
    """RoomConfig::GetRandomRoom (0x42C7D0) as place_rooms calls it (mode -1)."""
    return level.rc.get_random_room(seed, reduce_weight, stage, rtype, shape, min_variant, max_variant,
                                    min_difficulty, max_difficulty, doors, subtype,
                                    void_level=level.stage == 12, curses=level.get_curses())


def place_room(level, gr, config, seed: int) -> RepRoomDesc | None:
    """Level::place_room (0x338AB0, ret 0x14: room, data, seed, an unread 4th argument, out descriptor).
    Returns the descriptor, or None when 507 rooms are placed already (the engine logs and returns 0)."""
    if config is None:
        raise AssertionError('Level::place_room data is NULL!')
    if len(level.rooms) >= MAX_GRID_ROOMS:
        return None
    deco, spawn, award, extra = init_seeds(RNG(seed, DESC_SHIFT))
    gi = index(gr.x, gr.y)
    desc = RepRoomDesc(len(level.rooms), gi, gi + (config.shape == 9), config, gr.doors, deco, spawn, award,
                       config.shape, dimension=DIMENSION_NORMAL, boss_death_seed=extra)
    p, ctx = level.ctx.player, level.ctx
    # Devil's Crown (0x34D8D0): treasure rooms other than subtype 34, not visited (VisitedCount +0x40
    # is 0 after reset_room_list); in Greed only the room at Level+0x18370
    if has_trinket(p, T_DEVILS_CROWN) and config.type == ROOM_TREASURE and config.subtype != TREASURE_KNIFE_PIECE:
        if ctx.greed:
            raise NotImplementedError("Devil's Crown in Greed mode (Level+0x18370)")
        desc.flags |= FLAG_DEVIL_TREASURE
    # Corpse: 1 in 5 default rooms without pits (entity types 3000-3009) take the alternate backdrop;
    # the stage type comes from 0x34F690 (Level.stage_type outside Red Redemption)
    if config.type == ROOM_DEFAULT and stage_id(level.stage, level.stage_type, greed=ctx.greed) == STAGE_ID_CORPSE:
        if RNG(deco, CORPSE_SHIFT).next() % 5 == 0 and not any(
                3000 <= e.type < 3010 for sp in config.spawns for e in sp.entries):
            desc.flags |= FLAG_ALT_BACKDROP
    for ox, oy in PLACEMENT[config.shape]:
        cell = index(gr.x + ox, gr.y + oy)
        if cell < 0:
            raise AssertionError('[Level] place_room: invalid room pos/size')
        level.grid[cell] = desc.list_index
    level.rooms.append(desc)
    return desc


def _offgrid(level, idx: int, rng: RNG) -> RepRoomDesc:
    """GetRoomByIdx(idx) (0x340BC0) + InitSeeds(rng): an off-grid descriptor (Data set by the caller)."""
    deco, spawn, award, extra = init_seeds(rng)
    desc = RepRoomDesc(0x1FA - idx, idx, idx, None, 0, deco, spawn, award, 0, boss_death_seed=extra)
    level.offgrid_rooms[idx] = desc
    return desc


def _set_offgrid_config(desc: RepRoomDesc, config) -> None:
    desc.config = config
    desc.shape = config.shape if config is not None else 0


# ---------------------------------------------------------------------------------------- blacklists
def _desc_xy(desc) -> tuple[int, int]:
    return desc.grid_index % GRID, desc.grid_index // GRID


def secret_room_blacklist(level) -> set[int]:
    """Level::build_secret_room_index_blacklist (0x338D70), identical to AB+ 0x337210: on stage 11 the
    four cells around the start room; for every placed room, the outside cells of its bounding box
    (w = width / 13, h = height / 7) for each slot without a layout door, or every slot of a boss,
    super secret or secret room (1x1: slots 0-3, shape 4 skips 5/7, shape 6 skips 4/6)."""
    out: set[int] = set()
    if level.stage == 0xB:
        for off in (1, -1, 13, -13):
            c = level.start_index + off
            if 0 <= c < GRID * GRID:
                out.add(c)
    for desc in level.rooms:
        cfg = desc.config
        if cfg is None or desc.grid_index < 0:
            continue
        x, y = _desc_xy(desc)
        wx, hy = cfg.width // 13, cfg.height // 7
        cells = [(x - 1, y), (x, y - 1), (x + wx, y), (x, y + hy), (x - 1, y + 1), (x + 1, y - 1),
                 (x + wx, y + 1), (x + 1, y + hy)]
        for slot in range(8):
            if cfg.shape == 1 and slot >= 4:
                continue
            if cfg.shape == 4 and slot in (5, 7):
                continue
            if cfg.shape == 6 and slot in (4, 6):
                continue
            ci = index(*cells[slot])
            if ((not (cfg.doors >> slot) & 1) or cfg.type in (ROOM_BOSS, ROOM_SUPERSECRET, ROOM_SECRET)) and ci >= 0:
                out.add(ci)
    return out


def ultra_secret_blacklist(level) -> set[int]:
    """Inline at VA 0x73C9E2-0x73CAF8: for every placed room and slot without a layout door (every slot
    of boss, super secret, secret and curse rooms), the door target (ignore_narrow) when on the grid."""
    out: set[int] = set()
    for desc in level.rooms:
        cfg = desc.config
        if cfg is None or desc.grid_index < 0:
            continue
        x, y = _desc_xy(desc)
        for slot in range(8):
            if (cfg.doors >> slot) & 1 and cfg.type not in (ROOM_BOSS, ROOM_SUPERSECRET, ROOM_SECRET, ROOM_CURSE):
                continue
            tgt = door_target(x, y, cfg.shape, slot, True)
            if tgt is not None and 0 <= tgt[0] < GRID and 0 <= tgt[1] < GRID:
                out.add(tgt[0] + GRID * tgt[1])
    return out


def end_room_blacklist(level) -> set[int]:
    """0x352460: for every placed room and slot, unless the layout has the door and the room is a
    default room of subtype 0, the door target (ignore_narrow); an off-grid target adds -1."""
    out: set[int] = set()
    for desc in level.rooms:
        cfg = desc.config
        if cfg is None or desc.grid_index < 0:
            continue
        x, y = _desc_xy(desc)
        for slot in range(8):
            if (cfg.doors >> slot) & 1 and cfg.type == ROOM_DEFAULT and cfg.subtype == 0:
                continue
            tgt = door_target(x, y, cfg.shape, slot, True)
            if tgt is not None:
                out.add(index(*tgt))
    return out


# ---------------------------------------------------------------------------------------- odds
def planetarium_chance(level) -> float:
    """Game::GetPlanetariumChance (0x34DBD0), float32 like the engine."""
    ctx, p, stage = level.ctx, level.ctx.player, level.stage
    sp = _stage_prime(level)
    owner = has_trinket(p, T_TELESCOPE_LENS)                 # FirstTrinketOwner(152)
    lens = owner                                             # owner->HasTrinket(152)
    golden = owner and has_golden_trinket(p, T_TELESCOPE_LENS)
    momsbox = owner and has_collectible(p, C_MOMS_BOX)
    if stage > 10 or (stage > 6 and not lens) or (stage > 8 and not golden and not momsbox):
        return F32(0)
    c = PLANET_BASE
    lens_part = False
    if ctx.planetarium_visits == 0:
        if ROOM_TREASURE not in challenge_params(ctx).room_filter:
            visited = sp - 1 if ctx.treasure_rooms_visited is None else ctx.treasure_rooms_visited
            if visited < sp - 1:
                c = F32(F32(F32(sp - visited - 1) * PLANET_PER_SKIP) + PLANET_BASE)
                if has_collectible(p, C_CRYSTAL_BALL):
                    c = F32(c + ONE)
        if has_collectible(p, C_MAGIC_8_BALL):
            c = F32(c + PLANET_ITEM)
        if has_collectible(p, C_CRYSTAL_BALL):
            c = F32(c + PLANET_ITEM)
        if has_collectible(p, C_SAUSAGE):
            c = F32(c + PLANET_SAUSAGE)
        if lens:
            c = F32(c + PLANET_ITEM)
            lens_part = True
    elif lens:
        lens_part = True
    if lens_part:
        if golden != momsbox:
            c = F32(c + PLANET_LENS_ONE)
        elif golden:
            c = F32(c + PLANET_LENS_BOTH)
        else:
            c = F32(c + PLANET_LENS_NONE)
    c = c if c > PLANET_BASE else PLANET_BASE                # maxss
    return c if c < ONE else ONE                             # minss


def strange_door_floor(level) -> bool:
    """0x34F030: Depths II (or Depths I on a Labyrinth floor), not the alt path, not Greed, not an Ascent
    floor."""
    ctx = level.ctx
    if ctx.greed or (1 <= level.stage <= 6 and ctx.ascent) or level.stage_type in (4, 5):
        return False
    return level.stage == 6 or (level.stage == 5 and bool(level.get_curses() & CURSE_LABYRINTH))


def barren_bedroom_allowed(level) -> bool:
    """0x34B5D0 for a normal run (no challenge or daily: ChallengeParams +0x80 is neither 2 nor 3; not
    Greed; dimension 0; stage < 7 as the bedroom needs). 0x34B710 then reduces to "not 0x34EA50", i.e.
    not (stage'' == 6 on the alt path with state flag 47), stage'' = stage + 1 on a Labyrinth floor. The
    Mom-room check of 0x34B5D0 (Level+0x18300) also requires 0x34EA50, so it cannot change the result."""
    ctx = level.ctx
    if 1 <= level.stage <= 6 and ctx.ascent:
        return False
    stage2 = level.stage + 1 if level.get_curses() & CURSE_LABYRINTH else level.stage
    ea50 = (not ctx.greed and stage2 == 6 and level.stage_type in (4, 5) and _flag(ctx, F_BACKWARDS_PATH_INIT))
    return not ea50


def _unlocked(ctx, achievement: int) -> bool:
    return ctx.unlocked(achievement)


def _shop_subtype(level, rng: RNG) -> tuple[int, bool, bool]:
    """VA 0x739CD7-0x73A1A1. Returns (subtype, Wicked Crown owner seen, Holy Crown owner seen); the
    owners are only looked up on the modifier path and drive the icon after placement."""
    ctx, p, stage, st = level.ctx, level.ctx.player, level.stage, level.stage_type
    a = (2 if _unlocked(ctx, A_STORE_1) else 1) + _unlocked(ctx, A_STORE_3)
    b = (2 if _unlocked(ctx, A_STORE_2) else 1) + _unlocked(ctx, A_STORE_4)
    loc = RNG(rng.next(), SHOP_SHIFT)
    level_ = loc.random_int(a)
    level_ += loc.random_int(b)
    if (loc.next() & 1) == 0 or ctx.difficulty not in (1, 3):
        level_ = a - 2 + b
    store_key = trinket_multiplier(p, T_STORE_KEY)
    low = loc.next() & 0xFF
    wicked = holy = False
    path = 'mods'
    if low == 0 and store_key < 2:
        level_ = 11
    elif low == 1 and level_ >= 2:
        level_ = 10
        path = 'keeper'
    elif level_ == 10:
        path = 'keeper'
    elif store_key > 2:
        path = 'bump'
    if path == 'mods':                                      # VA 0x73A036
        path = 'keeper'
        if stage == 10 and st == 0:
            wicked = has_trinket(p, T_WICKED_CROWN)
            if wicked and trinket_multiplier(p, T_WICKED_CROWN) > 1:
                path = 'bump'
        if path == 'keeper' and stage == 10 and st == 1:
            holy = has_trinket(p, T_HOLY_CROWN)
            if holy and trinket_multiplier(p, T_HOLY_CROWN) > 1:
                path = 'bump'
        if path == 'keeper' and stage in (7, 8) and trinket_multiplier(p, T_SILVER_DOLLAR) > 1:
            path = 'bump'
    if path == 'bump':                                      # VA 0x73A139
        if level_ == 4:
            level_ = 10
        elif (loc.next() & 3) == 0:
            level_ = 10
        elif level_ == 11:
            level_ = 0
        elif level_ < 4:
            level_ += 1
    if p.player_type == P_KEEPER_B:                         # VA 0x739FB5: Tainted Keeper shops
        level_ += 100
    return level_, wicked, holy


def _crown_icon(level, wicked: bool, holy: bool) -> bool:
    p, stage, st = level.ctx.player, level.stage, level.stage_type
    return stage == 10 and ((st == 0 and wicked and trinket_multiplier(p, T_WICKED_CROWN) > 2)
                            or (st == 1 and holy and trinket_multiplier(p, T_HOLY_CROWN) > 2))


def _treasure_subtype(level, rng: RNG) -> tuple[int, int]:
    """VA 0x73A424-0x73A745. Returns (subtype, pick seed); consumes the level RNG like the engine."""
    ctx, p = level.ctx, level.ctx.player
    options = rng.next() % 100 == 0
    if not options:
        d = 100 // max(trinket_multiplier(p, T_GOLDEN_HORSE_SHOE), 1)
        s = rng.next()
        if (d == 0 or s % d < 15) and has_trinket(p, T_GOLDEN_HORSE_SHOE):
            options = True
        elif getattr(ctx, 'special_daily_id', 0) == 0x13:
            options = True
    pay = has_trinket(p, T_PAY_TO_WIN)
    sub = (3 if pay else 1) if options else (2 if pay else 0)
    return sub, rng.next()


# ---------------------------------------------------------------------------------------- place_rooms
def place_rooms(level, gen, min_difficulty: int, max_difficulty: int) -> bool:
    """Level::place_rooms (0x339370). `level` is a rep.level.RepLevel, `gen` a rep.levelgen.LevelGenerator
    after Generate. Fills level.rooms / level.grid; also leaves on `level`: boss_list_index (Level+0x18314),
    first_boss_list_index (+0x18330), shop_index, secret_index, offgrid_rooms {GridRooms idx: descriptor}.
    Returns False where the engine does (the caller resets the room list and retries)."""
    ctx, rng, stage, st = level.ctx, level.rng, level.stage, level.stage_type
    p = ctx.player
    params = challenge_params(ctx)
    if ctx.greed:
        raise NotImplementedError('Greed mode floors use generate_greed_dungeon (RVA 0x343550)')
    bosspool = getattr(level, 'bosspool', None)
    if bosspool is None:
        raise ValueError('J460 place_rooms needs the run\'s BossPool (Game+0x1AF70): pass bosspool= to RepLevel')
    sid = stage_id(stage, st, greed=ctx.greed)         # GetStageID(stage, type, -1), VA 0x7393DF
    curses = level.get_curses()
    excluded = params.room_filter.__contains__
    sp = _stage_prime(level)

    level.boss_list_index = -1
    level.first_boss_list_index = -1
    level.shop_index = -1
    level.secret_index = -1
    level.offgrid_rooms = {}
    miniboss_missing = True                             # [ebp-0x130] low byte

    def back(gr) -> None:
        if gr is not None:
            gen.add_end_room(gr)

    def boss_room(boss_id: int):
        seed = rng.next()
        return pick_boss_room(level.rc, boss_id, seed, boss_context(ctx), level_stage_id=sid, curses=curses,
                              void_level=stage == 12)

    # ---- 1 boss room
    if params.skip_boss_on_stage_11 and stage == 11:
        raise NotImplementedError('challenge parameter +0x74 (no boss room on stage 11)')
    boss_id = bosspool.get_boss_id(stage, st, ctx=boss_context(ctx))
    room = boss_room(boss_id)
    if room is None:                                    # "[warn] could not find matching boss room"
        return False
    last_boss = gen.get_new_boss_room(room.shape, room.doors, False)
    if last_boss is None:
        return False
    level.boss_list_index = level.first_boss_list_index = len(level.rooms)
    place_room(level, last_boss, room, rng.seed)
    # ---- 2 the Void's extra bosses
    if stage == 12:
        extra = rng.random_int(4) + 5
        pool = [b for b in range(0x46) if b not in VOID_SKIPPED_BOSSES and b not in VOID_EXCLUDED_BOSSES]
        i = len(pool) - 1
        while i > 0:
            j = rng.random_int(i + 1)
            pool[i], pool[j] = pool[j], pool[i]
            i -= 1
        for k in range(extra):
            room = boss_room(pool[k])
            if room is None:
                return False
            last_boss = gen.get_new_boss_room(room.shape, room.doors, True)
            if last_boss is None:
                if k < 5:
                    return False
                break
            level.boss_list_index = len(level.rooms)
            place_room(level, last_boss, room, rng.seed)
    # ---- 3 the Labyrinth's second boss room
    if curses & CURSE_LABYRINTH:
        saved = rng.seed
        room = boss_room(bosspool.get_boss_id(stage + 1, st, ctx=boss_context(ctx)))
        if room is None:
            return False
        last_boss = gen.get_new_boss_room(room.shape, room.doors, False)
        level.boss_list_index = len(level.rooms)
        if last_boss is None:
            return False
        place_room(level, last_boss, room, rng.seed)
        rng.set_seed(saved, LEVEL_SHIFT)
    # ---- 4 block the last boss room's surroundings
    if last_boss is not None:
        gen.block_room_area(last_boss)
    # ---- 5 super secret room(s)
    if not excluded(ROOM_SUPERSECRET):
        loc = RNG(rng.next(), SUPERSECRET_SHIFT)
        for i in range(2 if has_collectible(p, C_LUNA) else 1):
            s = loc.next()
            cfg = _pick(level, s, ROOM_SUPERSECRET)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                if i == 0:
                    return False
                break
            place_room(level, gr, cfg, s)
    # ---- 6 the secret room seed
    secret_seed = rng.next()
    # ---- 7 shop
    crown_stage = stage == 10 and ((st == 0 and has_trinket(p, T_WICKED_CROWN))
                                   or (st == 1 and has_trinket(p, T_HOLY_CROWN)))
    if stage < 7 or (stage < 9 and has_trinket(p, T_SILVER_DOLLAR)) or crown_stage:
        if not excluded(ROOM_SHOP) and ctx.victory_laps < 3:
            sub, wicked, holy = _shop_subtype(level, rng)
            cfg = _pick(level, rng.next(), ROOM_SHOP, sub)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            level.shop_index = len(level.rooms)
            desc = place_room(level, gr, cfg, rng.seed)
            if desc is not None and _crown_icon(level, wicked, holy):
                desc.display_flags |= DISPLAY_ICON
    # ---- 8 treasure room(s)
    if stage < 7 or (stage < 9 and has_trinket(p, T_BLOODY_CROWN)) or crown_stage:
        treasure_ok = not excluded(ROOM_TREASURE)
    else:
        treasure_ok = False
    if treasure_ok:
        saved = rng.seed
        for i in range(2 if curses & CURSE_LABYRINTH else 1):
            sub, seed = _treasure_subtype(level, rng)
            cfg = get_random_room2(level, seed, True, sid, 0, ROOM_TREASURE, SHAPE_ANY, 1, 10, 0, sub)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            desc = place_room(level, gr, cfg, rng.seed)
            wicked = stage == 10 and st == 0 and has_trinket(p, T_WICKED_CROWN)
            holy = stage == 10 and st == 1 and has_trinket(p, T_HOLY_CROWN)
            if desc is not None and _crown_icon(level, wicked, holy):
                desc.display_flags |= DISPLAY_ICON
            if i == 0:
                saved = rng.seed
        rng.set_seed(saved, LEVEL_SHIFT)
    elif params.field_80 == 2 and level.has_mirror_dimension():     # VA 0x73A383
        cfg = get_random_room2(level, rng.next(), True, sid, 0, ROOM_TREASURE, SHAPE_ANY, 1, 10, 0, 0)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if gr is None:
            return False
        place_room(level, gr, cfg, rng.seed)
    # ---- 9 planetarium
    chance = planetarium_chance(level)
    cfg = _pick(level, rng.next(), ROOM_PLANETARIUM)
    gr = gen.get_new_end_room(cfg.shape, cfg.doors)
    if excluded(ROOM_PLANETARIUM) or excluded(ROOM_TREASURE):
        back(gr)
    elif gr is not None:
        if chance > rng.random_float() and _unlocked(ctx, A_PLANETARIUM):
            place_room(level, gr, cfg, rng.seed)
        else:
            back(gr)
    # ---- 10 stage < 11
    if stage < 11:
        # a dice room / sacrifice room
        dice = rng.next() % 50 == 0 or (rng.next() % 5 == 0 and p.keys > 1)
        rtype = ROOM_DICE if dice and not excluded(ROOM_DICE) else ROOM_SACRIFICE
        cfg = _pick(level, rng.next(), rtype)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if excluded(ROOM_SACRIFICE):
            back(gr)
        elif rng.next() % 7 == 0 or ((rng.next() & 3) == 0 and (
                full_health(p) or p.player_type in (P_LAZARUS, P_LAZARUS2))):
            if gr is not None:
                place_room(level, gr, cfg, rng.seed)
        else:
            back(gr)
        # b library
        lib = 0
        if _unlocked(ctx, A_STORE_1):
            lib = 1
        if _unlocked(ctx, A_STORE_2):
            lib = 2
        if _unlocked(ctx, A_STORE_3):
            lib = 3
        if _unlocked(ctx, A_STORE_4):
            lib = 4
        sub = rng.random_int(lib + 1)
        cfg = _pick(level, rng.next(), ROOM_LIBRARY, sub, reduce_weight=False)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if excluded(ROOM_LIBRARY):
            back(gr)
        elif rng.next() % 20 == 0 or ((rng.next() & 3) == 0 and _flag(ctx, F_BOOK_PICKED_UP)):
            if gr is not None:
                place_room(level, gr, cfg, rng.seed)
        else:
            back(gr)
        # c curse room(s)
        loc = RNG(rng.next(), CURSE_SHIFT)
        voodoo = has_collectible(p, C_VOODOO_HEAD)
        curse_cfg = curse_gr = None
        for _ in range(2 if voodoo else 1):
            cfg = _pick(level, loc.next(), ROOM_CURSE, 1) if voodoo else None
            if cfg is None:
                cfg = _pick(level, loc.next(), ROOM_CURSE, 0)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            curse_cfg, curse_gr = cfg, gr
            if excluded(ROOM_CURSE):
                back(gr)
            else:
                if loc.next() & 1:
                    loc.next()
                if gr is not None:
                    place_room(level, gr, cfg, loc.seed)
        # d miniboss room
        mb_cfg, mb_gr, sin = curse_cfg, curse_gr, -1
        difficulty = 1 if stage < 7 else 10
        denom = 0
        if sp >= 3:
            denom = 30 if _unlocked(ctx, A_THE_WOMB) else 80
            if _unlocked(ctx, A_EVERYTHING_IS_TERRIBLE):
                denom = 10
            if ctx.super_sin_counter:
                denom = 5
        if sp >= 3 and rng.next() % 10 == 0 and not _flag(ctx, F_ULTRAPRIDE_SPAWNED):
            rng.next()
            sin = MB_ULTRA_PRIDE
        else:
            super_ = bool(denom) and rng.random_int(denom) == 0
            sins = [sup if super_ else normal for flag, normal, sup in SIN_FLAGS if not _flag(ctx, flag)]
            if sins:
                sin = sins[rng.random_int(len(sins))]
        if sin >= 0:
            mb_cfg = _pick(level, rng.next(), ROOM_MINIBOSS, sin, min_difficulty=difficulty,
                           max_difficulty=difficulty)
            if mb_cfg is None:
                mb_cfg = _pick(level, rng.next(), ROOM_MINIBOSS, sin)
            mb_gr = gen.get_new_end_room(mb_cfg.shape, mb_cfg.doors) if mb_cfg is not None else None
        if excluded(ROOM_MINIBOSS):
            back(mb_gr)
        elif (rng.next() & 3) == 0 or (rng.next() % 3 == 0 and sp == 1):
            if mb_gr is not None:
                place_room(level, mb_gr, mb_cfg, rng.seed)
                miniboss_missing = False
                for flag, normal, sup in SIN_FLAGS:
                    if sin in (normal, sup):
                        _set_flag(ctx, flag, True)
                if sin == MB_ULTRA_PRIDE:
                    _set_flag(ctx, F_ULTRAPRIDE_SPAWNED, True)
        else:
            back(mb_gr)
        # e challenge room
        cfg = _pick(level, rng.next(), ROOM_CHALLENGE, int(stage in (2, 4, 6, 8)))
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if excluded(ROOM_CHALLENGE):
            back(gr)
        elif (rng.next() & 1 and sp < 3) or not full_health(p) or sp < 2:
            back(gr)
        elif gr is not None:
            place_room(level, gr, cfg, rng.seed)
        # f arcade / vault
        cain_br = p.player_type == P_CAIN and has_collectible(p, C_BIRTHRIGHT)
        vault = rng.next() % 10 == 0 or (rng.next() % 3 == 0 and p.keys > 1)
        cfg = None
        if vault and not excluded(ROOM_CHEST) and not cain_br:
            cfg = _pick(level, rng.next(), ROOM_CHEST)
        else:
            if cain_br:
                cfg = _pick(level, rng.next(), ROOM_ARCADE, 1)
            if cfg is None:
                cfg = _pick(level, rng.next(), ROOM_ARCADE, 0)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        ok = ((cfg.type == ROOM_ARCADE and (p.coins >= 5 or cain_br)) or (cfg.type == ROOM_CHEST and p.keys > 1))
        if excluded(cfg.type) or not ok or not (sp in (2, 4, 6, 8) or (cain_br and sp <= 10)):
            back(gr)
        elif gr is not None:
            place_room(level, gr, cfg, rng.seed)
    # ---- 11 bedroom
    if stage < 7:
        ex_isaacs = excluded(ROOM_ISAACS)
        barren = not excluded(ROOM_BARREN) and barren_bedroom_allowed(level)
        rtype = ROOM_ISAACS if ((rng.next() & 1) == 0 or not barren) and not ex_isaacs else ROOM_BARREN
        cfg = _pick(level, rng.next(), rtype)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if rng.next() % 50 == 0 or (rng.next() % 5 == 0 and low_health(p)):
            if gr is not None:
                place_room(level, gr, cfg, rng.seed)
        else:
            back(gr)
    # ---- 12 mirror world / abandoned mineshaft / strange door: extra rooms
    extra = 0
    if level.has_mirror_dimension():
        cfg = _pick(level, rng.next(), ROOM_DEFAULT, 0x22, stage=sid, min_difficulty=0, max_difficulty=0)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if gr is None:
            return False
        place_room(level, gr, cfg, rng.seed)
        extra = 1
    elif level.has_abandoned_mineshaft():
        if has_collectible(p, C_KNIFE_PIECE_1):
            s = RNG(rng.seed, MINESHAFT_SHIFT).next()
            cfg = get_random_room2(level, s, True, sid, STAGE_ID_MINES, ROOM_DEFAULT, SHAPE_ANY, 0, 0, 0, 10)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            place_room(level, gr, cfg, s)
            extra = 2
    elif strange_door_floor(level) and ctx.strange_door:
        # Manager+0x2C7 = the byte of achievement 635 (or the unlock-all override): the same
        # RepGameContext.strange_door that generate_dungeon's (6,5) block reads (VA 0x7418D4)
        extra = 1
    # ---- 13 secret room(s)
    if not excluded(ROOM_SECRET):
        fc = trinket_multiplier(p, T_FRAGMENTED_CARD)
        n = (2 if fc > 0 else 1) + has_collectible(p, C_LUNA)
        s = secret_seed
        for i in range(n):
            cfg = get_random_room2(level, s, True, sid, 0, ROOM_SECRET, SHAPE_ANY, 1, 10, 0, -1)
            gr = gen.get_new_secret_room(secret_room_blacklist(level))
            if gr is not None:
                level.secret_index = len(level.rooms)
                desc = place_room(level, gr, cfg, s)
                if desc is not None and ((fc == 2 and i == 0) or (fc > 2 and i < 2)):
                    desc.display_flags |= DISPLAY_ICON
            s = RNG(s, SECRET_SHIFT).next()
    # ---- 14 Dark Room: the tomb
    if stage == 11 and st == 0:
        s = rng.next()
        if _unlocked(ctx, A_THE_FORGOTTEN):
            cfg = _pick(level, s, ROOM_DEFAULT, min_variant=3, max_variant=9)
        else:
            cfg = level.rc.get_room(0, ROOM_DEFAULT, 3)
        gr = gen.get_new_end_room(cfg.shape, cfg.doors)
        if gr is None:
            return False
        place_room(level, gr, cfg, s)
    # ---- 15 ultra secret room
    if not excluded(ROOM_ULTRASECRET):
        s = RNG(secret_seed, ULTRASECRET_SHIFT).next()
        cfg = get_random_room2(level, s, True, sid, 0, ROOM_ULTRASECRET, SHAPE_ANY, 1, 10, 0, -1)
        if cfg is not None:
            gr = gen.get_new_ultra_secret_room(ultra_secret_blacklist(level))
            if gr is not None:
                desc = place_room(level, gr, cfg, s)
                if desc is not None:
                    desc.field_b4 = 99
    # ---- 16 every remaining room
    remaining = gen.get_remaining_rooms()
    pairs = []
    for gr in remaining:
        md = min_difficulty
        if md > 1 and (rng.seed & 7) == 0:
            md = 1
        if index(gr.x, gr.y) == level.start_index and gr.depth == 0:
            if stage == 11 and st == 1:
                cfg = level.rc.get_room(0x11, ROOM_DEFAULT, 0)
            elif stage == 11 and st == 0:
                cfg = level.rc.get_room(0x10, ROOM_DEFAULT, 0)
            else:
                cfg = level.rc.get_room(0, ROOM_DEFAULT, 2)
        else:
            cfg = _pick(level, rng.next(), ROOM_DEFAULT, 0, stage=sid, min_variant=int(stage == 11),
                        min_difficulty=md, max_difficulty=max_difficulty, shape=gr.shape, doors=gr.doors)
        if cfg is None:        # "[warn] LevelGenerator could not find a required room (type=ROOM_DEFAULT ..."
            return False
        pairs.append((place_room(level, gr, cfg, rng.seed), gr))
    # ---- 17 one draw per generator room outside that list
    for _ in range(len(gen.rooms) - len(remaining)):
        rng.next()
    # ---- 18 extra rooms: subtype 1 layouts
    if extra > 0:
        loc = RNG(rng.seed, EXTRA_ROOM_SHIFT)
        i = len(pairs) - 1
        while i > 0:
            j = loc.random_int(i + 1)
            pairs[i], pairs[j] = pairs[j], pairs[i]
            i -= 1
        family = EXTRA_ROOM_FAMILY.get(sid, sid)
        blacklist = end_room_blacklist(level)
        limit = len(pairs) + extra
        k = 0
        while k < limit and extra > 0:
            desc = None
            if k < len(pairs):
                desc, gr = pairs[k]
                if desc.grid_index == level.start_index:
                    k += 1
                    continue
            else:
                gr = gen.create_random_end_room(1, blacklist)
                if gr is None:
                    return False
            md = min_difficulty
            if md > 1 and (loc.seed & 7) == 0:
                md = 1
            cfg = get_random_room2(level, loc.next(), True, sid, family, ROOM_DEFAULT, gr.shape, md,
                                   max_difficulty, gr.doors, 1)
            if cfg is None:
                cfg = get_random_room2(level, loc.next(), True, sid, family, ROOM_DEFAULT, gr.shape, 1, 10,
                                       gr.doors, 1)
            if cfg is not None:
                extra -= 1
                if desc is None:
                    place_room(level, gr, cfg, loc.next())
                else:
                    desc.config, desc.doors = cfg, cfg.doors
            k += 1
        if extra > 0:
            return False
    # ---- 19 surprise miniboss (Greed) in the secret room / shop
    allow = miniboss_missing and not (_flag(ctx, F_GREED_SPAWNED) and _flag(ctx, F_SUPERGREED_SPAWNED))
    base = GREED_RICH if p.coins >= 0x14 else ONE
    if _flag(ctx, F_DONATION_SLOT_BLOWN):
        _set_flag(ctx, F_DONATION_SLOT_BLOWN, False)
        base = F32(base + GREED_FLAG30)
    if _flag(ctx, F_SHOPKEEPER_KILLED):
        _set_flag(ctx, F_SHOPKEEPER_KILLED, False)
        base = F32(base + GREED_FLAG31)
    counter, ctx.greed_counter = ctx.greed_counter, 0
    thr = F32(F32(base - F32(F32(min(counter, 5)) * GREED_STEP)) * GREED_SCALE)
    if thr > rng.random_float() and allow and level.secret_index >= 0 and stage > 4:
        if RNG(rng.seed, DESC_SHIFT).random_int(1000) != 0:
            allow = False
        level.rooms[level.secret_index].flags |= FLAG_SURPRISE_MINIBOSS
    if thr > rng.random_float() and allow and level.shop_index >= 0 and stage > 3:
        RNG(rng.seed, DESC_SHIFT).random_int(1000)
        level.rooms[level.shop_index].flags |= FLAG_SURPRISE_MINIBOSS
    # ---- 20 off-grid rooms
    for idx, rtype in ((ROOM_ERROR_IDX, ROOM_ERROR), (ROOM_BLACK_MARKET_IDX, ROOM_BLACK_MARKET),
                       (ROOM_DUNGEON_IDX, ROOM_DUNGEON)):
        desc = _offgrid(level, idx, rng)
        _set_offgrid_config(desc, _pick(level, rng.next(), rtype))
    desc = _offgrid(level, ROOM_BOSSRUSH_IDX, rng)
    s = rng.next()
    if has_collectible(p, C_BROKEN_SHOVEL_1):
        _set_offgrid_config(desc, level.rc.get_room(0, ROOM_BOSSRUSH, 0))
    else:
        _set_offgrid_config(desc, _pick(level, s, ROOM_BOSSRUSH))
    desc = _offgrid(level, ROOM_MEGA_SATAN_IDX, rng)
    _set_offgrid_config(desc, _pick(level, rng.next(), ROOM_BOSS, BOSS_MEGA_SATAN))
    desc = _offgrid(level, ROOM_THE_VOID_IDX if sid == STAGE_ID_BLUE_WOMB else ROOM_BLUE_WOOM_IDX, rng)
    _set_offgrid_config(desc, _pick(level, rng.next(), ROOM_DEFAULT, 1, stage=STAGE_ID_BLUE_WOMB,
                                    min_difficulty=0, max_difficulty=0))
    loc = RNG(rng.next(), SHOP_SHIFT)
    shop = loc.random_int(1 + _unlocked(ctx, A_STORE_3))
    shop += loc.random_int((2 if _unlocked(ctx, A_STORE_2) else 1) + _unlocked(ctx, A_STORE_4))
    desc = _offgrid(level, ROOM_SECRET_SHOP_IDX, loc)
    _set_offgrid_config(desc, _pick(level, loc.next(), ROOM_SHOP, shop))
    loc = RNG(rng.next(), ANGEL_SHOP_SHIFT)
    desc = _offgrid(level, ROOM_ANGEL_SHOP_IDX, loc)
    _set_offgrid_config(desc, _pick(level, loc.next(), ROOM_ANGEL, 1))
    if st in (4, 5) and (stage == 8 or (stage == 7 and curses & CURSE_LABYRINTH)):
        rng.next()
        desc = _offgrid(level, ROOM_SECRET_EXIT_IDX, rng)
        _set_offgrid_config(desc, _pick(level, rng.next(), ROOM_BOSS, BOSS_MOTHER, stage=sid))
    return True
