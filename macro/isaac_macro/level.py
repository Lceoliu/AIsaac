"""Level generation of Afterbirth+ v1.06 (normal floors), translated from named symbols.

Level::Init (curses)                    0x33E570
Level::generate_dungeon                 0x33ABF0
Level::place_rooms                      0x337740
Level::choose_boss / choose_double_trouble 0x330DA0 / 0x331A40
Level::place_room                       0x331B10
Level::build_secret_room_index_blacklist 0x337210

Everything that reads game state goes through GameContext, so a caller can reproduce a save
(achievements), a character (hearts, keys, coins, trinkets) and run flags. Unsupported paths raise
instead of guessing: Greed mode, Blue Womb (generate_blue_womb), challenge overrides, daily runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .levelgen import GRID, LevelGenerator, PLACEMENT, index
from .rng import RNG
from .roomconfig import SHAPE_ANY, Room, RoomConfig, stage_id

F32 = np.float32
LABYRINTH_FACTOR = 1.8                               # 0x96C328 (double)
CHOOSE_BOSS_ALT_THRESHOLD = F32(0.33000001311302185)  # 0x957E90
CHOOSE_BOSS_THRESHOLD = F32(0.10000000149011612)      # 0x9502A0
GREED_BASE, GREED_BASE_POOR = F32(1.0499999523162842), F32(1.0)   # 0x960808 / 0x94DCF4
GREED_FLAG30, GREED_FLAG31 = F32(0.10000000149011612), F32(0.019999999552965164)
GREED_SUB, GREED_SUB_STEP = F32(0.25), F32(0.05000000074505806)   # 0x9589C8 / 0x9579D0
GREED_SCALE = F32(0.3333333432674408)                              # 0x95D234
VOID_EXCLUDED_BOSSES = (60, 43, 50, 23, 67, 69, 61, 57, 44, 64, 56, 65, 47, 45, 43, 50, 52, 9, 10, 18,
                        32, 36, 27, 13, 14, 20, 17, 28, 2, 1, 21, 3, 4, 34, 37, 29, 26)  # 0x96C160

CURSE_DARKNESS, CURSE_LABYRINTH, CURSE_LOST, CURSE_UNKNOWN = 1, 2, 4, 8
CURSE_MAZE, CURSE_BLIND = 0x20, 0x40

ROOM_DEFAULT, ROOM_SHOP, ROOM_ERROR, ROOM_TREASURE, ROOM_BOSS, ROOM_MINIBOSS = 1, 2, 3, 4, 5, 6
ROOM_SECRET, ROOM_SUPERSECRET, ROOM_ARCADE, ROOM_CURSE, ROOM_CHALLENGE = 7, 8, 9, 10, 11
ROOM_LIBRARY, ROOM_SACRIFICE, ROOM_DUNGEON, ROOM_BOSSRUSH = 12, 13, 16, 17
ROOM_ISAACS, ROOM_BARREN, ROOM_CHEST, ROOM_DICE, ROOM_BLACK_MARKET = 18, 19, 20, 21, 22


class GenerationFailed(Exception):
    pass


@dataclass
class Player:
    player_type: int = 0          # +0x2618 (0 = Isaac)
    hearts: int = 6               # +0x25B8 red hearts, in half hearts
    max_hearts: int = 6           # +0x25B4 heart containers, in half hearts
    soul_hearts: int = 0          # +0x25C0
    bone_hearts: int = 0          # +0x38F8
    keys: int = 0                 # +0x25D0
    coins: int = 0                # +0x25DC
    active_item: int = 0          # +0x27FC
    trinkets: frozenset = frozenset()
    collectibles: frozenset = frozenset()

    def effective_max_hearts(self) -> int:
        """Entity_Player::GetEffectiveMaxHearts (0x267D50)."""
        if self.player_type in (4, 10, 17):
            return self.max_hearts
        return self.max_hearts + self.bone_hearts * 2

    def trinket_multiplier(self) -> int:
        """Entity_Player::GetTrinketMultiplier (0x26B340): Mom's Box doubles trinkets."""
        return 2 if self.active_item == 439 else 1


@dataclass
class GameContext:
    """The engine state that level generation reads. Defaults: a normal-mode run as Isaac."""
    achievements: frozenset | None = None   # None = every achievement unlocked
    difficulty: int = 0                     # Game+0x296878: 0 normal, 1 hard, 2/3 greed
    challenge: int = 0                      # Game+0x215EA0
    special_2160cc: int = 0                 # Game+0x2160CC (compared with 0x0D/0x12/0x13/0x26)
    state_flags: int = 0                    # Game+0x215E68, run flags (bosses/sins seen, ...)
    victory_laps: int = 0                   # Game+0x216210
    excluded_room_types: frozenset = frozenset()   # Game+0x2160D0 (challenge room filter)
    beat_it_all: bool = False               # Manager+0x1F0 != 0: 1/5 curse chance
    greed_counter: int = 0                  # Game+0x215E88
    player: Player = field(default_factory=Player)

    def unlocked(self, achievement: int) -> bool:
        """PersistentGameData::Unlocked (0x41F7F0) for a normal (non-daily) run."""
        return self.achievements is None or achievement in self.achievements

    def flag(self, bit: int) -> bool:
        return bool((self.state_flags >> bit) & 1)

    def set_flag(self, bit: int, value: bool) -> None:
        """Game::SetStateFlag (0x2C3170)."""
        if value:
            self.state_flags |= 1 << bit
        else:
            self.state_flags &= ~(1 << bit)

    @property
    def hard(self) -> bool:
        return self.difficulty == 1

    @property
    def greed(self) -> bool:
        return self.difficulty in (2, 3)


@dataclass
class RoomDesc:
    """RoomDescriptor fields written by Level::place_room."""
    list_index: int
    grid_index: int
    safe_grid_index: int
    config: Room
    doors: int
    decoration_seed: int   # desc+0x64, Room::GetDecorationSeed
    spawn_seed: int        # desc+0x68, Room::GetSpawnSeed (logged as "SpawnRNG seed")
    award_seed: int        # desc+0x6C, Room::GetAwardSeed
    shape: int
    flags: int = 0

    @property
    def type(self) -> int:
        return self.config.type

    @property
    def variant(self) -> int:
        return self.config.variant


@dataclass
class LevelResult:
    stage: int
    stage_type: int
    stage_seed: int
    curses: int
    rooms: list[RoomDesc]
    grid: list[int]                 # Level grid (+0x9708): room list index per cell, -1 empty
    attempts: int
    boss_list_index: int
    generator: LevelGenerator = field(repr=False, default=None)

    def room_at(self, cell: int) -> RoomDesc | None:
        i = self.grid[cell]
        return self.rooms[i] if i >= 0 else None


class Level:
    def __init__(self, room_config: RoomConfig, ctx: GameContext, stage: int, stage_type: int):
        if ctx.greed:
            raise NotImplementedError('Greed mode (generate_greed_dungeon) is not translated')
        if stage == 9:
            raise NotImplementedError('Blue Womb (generate_blue_womb) is not translated')
        if ctx.challenge:
            raise NotImplementedError('challenge overrides are not translated')
        self.rc, self.ctx = room_config, ctx
        self.stage, self.stage_type = stage, stage_type
        self.rng = RNG()
        self.curses = 0
        self.rooms: list[RoomDesc] = []
        self.grid = [-1] * (GRID * GRID)
        self.start_index = 0x54
        self.boss_list_index = -1
        self.attempts = 0
        self.log: list | None = None    # engine log.txt events, when a list is supplied

    # ------------------------------------------------------------------ Level::Init
    def init(self, stage_seed: int) -> LevelResult:
        ctx = self.ctx
        self.rng = RNG(stage_seed, 0x23)
        self.rng.next()                     # reset_room_list
        self.rng.random_int(1)
        second = not ctx.hard
        if ctx.unlocked(0x21):
            denom = 10 if second else 3
        elif ctx.unlocked(4):
            denom = 30 if second else 5
        else:
            denom = 80 if second else 10
        if ctx.beat_it_all:
            denom = 5 if second else 2
        gen_rng = RNG(self.rng.seed, 0x23)
        curses = 0
        if 0x104 not in ctx.player.collectibles:
            if ctx.special_2160cc != 0x12:
                if gen_rng.random_int(denom) == 0:
                    pick = gen_rng.random_int(6)
                    if pick == 0:
                        if self.stage % 2 == 1 and self.stage < 8:
                            curses |= CURSE_LABYRINTH
                    elif pick == 1:
                        curses |= CURSE_LOST
                    elif pick == 2:
                        curses |= CURSE_DARKNESS
                    elif pick == 3:
                        curses |= CURSE_UNKNOWN
                    elif pick == 4:
                        curses |= CURSE_MAZE
                    else:
                        curses |= CURSE_BLIND
                if ctx.special_2160cc == 0x26:
                    curses |= 0x4C
        elif ctx.special_2160cc == 0x26:
            curses |= 0x4C
        if self.stage == 9 or ctx.victory_laps > 2:
            curses &= ~CURSE_DARKNESS
        self.curses = curses
        if self.stage == 12:
            for sid in [s for s in range(1, 18) if s != 13]:
                self.rc.reset_room_weights(sid)
        else:
            self.rc.reset_room_weights(stage_id(self.stage, self.stage_type))
        self.rc.reset_room_weights(0)
        gen = self.generate_dungeon(gen_rng)
        return LevelResult(self.stage, self.stage_type, stage_seed, self.curses, self.rooms, self.grid,
                           self.attempts, self.boss_list_index, gen)

    # ------------------------------------------------------------------ generate_dungeon
    def generate_dungeon(self, gen_rng: RNG) -> LevelGenerator:
        ctx, rng, stage = self.ctx, self.rng, self.stage
        n = rng.random_int(2) + 5 + (stage * 10) // 3
        if n > 19:
            n = 20
        if not self.curses & CURSE_LABYRINTH:
            rooms = n + 4 if self.curses & CURSE_LOST else n
        else:
            rooms = int(n * LABYRINTH_FACTOR)
            if rooms > 44:
                rooms = 45
        if stage == 12:
            rooms = rng.random_int(5) + 50
        extra = gen_rng.random_int(2)
        if ctx.hard:
            rooms += 2 + extra
        min_dead_ends = (stage != 1) + 6 - (not self.curses & CURSE_LABYRINTH)
        if stage == 12:
            min_dead_ends += 2
        gen = LevelGenerator(rng.next())
        gen.log = self.log
        if ctx.hard:
            max_d, min_d = (15, 10) if stage < 9 and stage % 2 == 0 else (5, 5)
        else:
            max_d, min_d = (10, 5) if stage < 9 and stage % 2 == 0 else (5, 1)
        if stage == 12:
            normal = []
            for sid in [s for s in range(1, 18) if s != 13]:
                normal += self.rc.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 5, 15, 0, -1)
            min_d, max_d = 5, 15
        else:
            normal = self.rc.get_rooms(stage_id(stage, self.stage_type), ROOM_DEFAULT, SHAPE_ANY, 0,
                                       0xFFFFFFFF, min_d, max_d, 0, -1)
        if len(normal) < 20:
            if stage == 12:
                normal = []
                for sid in [s for s in range(1, 18) if s != 13]:
                    normal += self.rc.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 15, 0, -1)
            else:
                normal = self.rc.get_rooms(stage_id(stage, self.stage_type), ROOM_DEFAULT, SHAPE_ANY, 0,
                                           0xFFFFFFFF, 1, 15, 0, -1)
            min_d, max_d = 1, 15
        shapes = 0
        for r in normal:
            shapes |= 1 << r.shape
        labyrinth = bool(self.curses & CURSE_LABYRINTH)
        while True:
            self.attempts += 1
            if self.attempts > 10000:
                raise GenerationFailed('level generation did not converge')
            if self.log is not None:
                self.log.append(('generate',))
            gen.generate(rooms, stage == 11, labyrinth, stage == 12, shapes)
            if not gen.usable:
                if self.log is not None:
                    self.log.append(('unusable',))
                continue
            if len(gen.dead_ends) < min_dead_ends:
                if self.log is not None:
                    self.log.append(('dead_ends', len(gen.dead_ends), min_dead_ends))
                continue
            if self.log is not None:
                self.log.append(('placing',))
            if self.place_rooms(gen, min_d, max_d):
                return gen

    # ------------------------------------------------------------------ Level::place_room
    def place_room(self, gr, config: Room, seed: int) -> bool:
        if config is None:
            raise AssertionError('Level::place_room data is NULL!')
        if len(self.rooms) >= 0x80:
            return False
        r = RNG(seed, 0xC)
        deco, spawn, award = r.next(), r.next(), r.next()
        gi = index(gr.x, gr.y)
        desc = RoomDesc(len(self.rooms), gi, gi + (config.shape == 9), config, gr.doors, deco, spawn,
                        award, config.shape)
        for ox, oy in PLACEMENT[config.shape]:
            cell = index(gr.x + ox, gr.y + oy)
            if cell < 0:
                raise AssertionError('[Level] place_room: invalid room pos/size')
            self.grid[cell] = desc.list_index
        self.rooms.append(desc)
        return True

    # ------------------------------------------------------------------ choose_double_trouble
    def choose_double_trouble(self, stage: int) -> int:
        if stage in (3, 4):
            if self.rng.random_int(50) == 0:
                return 0xE74
            return 0
        if stage == 5:
            r, v = self.rng.random_int(25), 0xEA6
        elif stage == 7:
            r, v = self.rng.random_int(40), 0xED8
        else:
            return 0
        return v if r == 0 else 0

    # ------------------------------------------------------------------ choose_boss
    def choose_boss(self, stage: int) -> int:
        """Level::choose_boss(eLevelStage): returns an eBossId (the boss room subtype)."""
        ctx, rng, st = self.ctx, self.rng, self.stage_type
        local = RNG(rng.next(), 1)
        if stage == 6:
            return 6
        if stage == 8:
            return 0x19 if ctx.unlocked(0x22) else 8
        if stage == 11:
            if st == 1:
                return 0x28
            if st == 0:
                return 0x36
        elif stage == 10:
            if st == 1:
                return 0x27
            if st == 0:
                return 0x18
        elif stage == 12:
            return 0x46
        if ctx.special_2160cc == 0xD and stage in (1, 2, 3, 4, 5, 7):
            return {1: 0x3C, 2: 0x2B, 3: 0x32, 4: 0x17, 5: 0x33, 7: 0x0F}[stage]
        if ctx.unlocked(0x15B):
            if stage == 7:
                f = rng.random_float()
                if f < (CHOOSE_BOSS_ALT_THRESHOLD if st == 2 else CHOOSE_BOSS_THRESHOLD):
                    return 0x48
            if rng.random_int(5) == 0:
                if stage == 3:
                    ctx.set_flag(0x2A, rng.random_int(2) == 1)
                clear35 = not ctx.flag(35)
                if stage == 4:
                    return 0x43 if clear35 else 0x45
                if stage == 3:
                    return 0x45 if clear35 else 0x43
                if 5 <= stage < 8 and not ctx.flag(43):
                    ctx.set_flag(0x2B, True)
                    return 0x44
        result = self._choose_boss_afterbirth_plus(stage, local)
        return result

    def _choose_boss_afterbirth_plus(self, stage: int, local: RNG) -> int:
        ctx, rng, st = self.ctx, self.rng, self.stage_type
        if ctx.unlocked(0x15A) and rng.random_int(4) == 0:
            if stage < 9 and stage % 2 == 1:
                ctx.set_flag(0x23, rng.random_int(2) == 1)
            flags = ctx.state_flags
            v = 0x42 if rng.random_int(2) == 0 else 0x3B
            b = bool(flags & (1 << 35))
            if stage == 1:
                return 0x3C if b else 0x3D
            if stage == 2:
                return 0x3D if b else 0x3C
            if stage == 3:
                return 0x39 if b else v
            if stage == 4:
                return v if b else 0x39
            if stage in (5, 6) and not ctx.flag(36):
                ctx.set_flag(0x24, True)
                return 0x3A
        return self._choose_boss_afterbirth(stage, local)

    def _choose_boss_afterbirth(self, stage: int, local: RNG) -> int:
        ctx, rng, st = self.ctx, self.rng, self.stage_type
        if rng.random_int(6) == 0:
            if stage < 9 and stage % 2 == 1:
                ctx.set_flag(0x14, rng.random_int(2) == 1)
            flags = ctx.state_flags
            a = rng.random_int(2)
            c = rng.random_int(2)
            un = ctx.unlocked(0x15A)
            v = (0x38 if c == 0 else 0x41) if un else 0x38
            w = (0x2C if a == 0 else 0x40) if un else 0x2C
            b = not flags & (1 << 20)
            if st != 0:
                if stage in (1, 2):
                    if not ctx.flag(21):
                        ctx.set_flag(0x15, True)
                        return 0x2B
                elif stage == 3:
                    return 0x34 if b else 0x32
                elif stage == 4:
                    return 0x32 if b else 0x34
                elif stage in (5, 6):
                    if not ctx.flag(22):
                        ctx.set_flag(0x16, True)
                        return 0x33
                elif stage in (7, 8):
                    if not ctx.flag(24):
                        ctx.set_flag(0x18, True)
                        return 0x31
            else:
                if stage == 1:
                    return v if b else w
                if stage == 2:
                    return w if b else v
                if stage == 3:
                    return 0x2D if b else 0x2F
                if stage == 4:
                    return 0x2F if b else 0x2D
                if stage == 5:
                    return 0x2E if b else 0x30
                if stage == 6:
                    return 0x30 if b else 0x2E
                if stage in (7, 8) and not ctx.flag(23):
                    ctx.set_flag(0x17, True)
                    return 0x35
        return self._choose_boss_base(stage, local)

    def _choose_boss_base(self, stage: int, local: RNG) -> int:
        ctx, rng, st = self.ctx, self.rng, self.stage_type
        conquest = rng.random_int(2) == 0 and ctx.unlocked(0x42)
        if rng.random_int(5) == 0 and ctx.unlocked(5) and stage in (1, 3, 5, 7):
            if local.random_int(10) == 0 and not ctx.flag(28):
                ctx.set_flag(0x1C, True)
                return 0x16
            if stage == 3 and not ctx.flag(1):
                ctx.set_flag(1, True)
                return 10
            if stage == 1 and not ctx.flag(0):
                ctx.set_flag(0, True)
                return 9
            if stage == 5 and not ctx.flag(2):
                ctx.set_flag(2, True)
                return 11
            if stage == 7 and not ctx.flag(3):
                ctx.set_flag(3, True)
                return 0x26 if conquest else 0x0C
        u44 = rng.random_int(2) == 0 and ctx.unlocked(0x44)
        if rng.random_int(3) == 0 and stage == 7:
            return 0x2A if u44 else 0x29
        if rng.random_int(10) == 0 and ctx.flag(6) and not ctx.flag(27):
            ctx.set_flag(0x1B, True)
            return 0x17
        if rng.random_int(3) == 0 and stage == 7:
            return 0x1E if rng.random_int(2) == 0 else 0x21
        if rng.random_int(4) == 0 and st == 1:
            if stage == 4:
                return 0x24 if rng.random_int(2) == 0 else 0x1B
            if stage == 2:
                return 0x12 if rng.random_int(2) == 0 else 0x20
        if rng.random_int(5) == 0 and stage in (1, 3, 5, 7):
            return {3: 0x0E, 1: 0x0D, 5: 0x0F, 7: 0x10}[stage]
        u10 = rng.random_int(2) != 0 and ctx.unlocked(0x10)
        if rng.random_int(4) == 0:
            if stage == 1:
                if st == 0:
                    return 0x14 if u10 else 0x11
            elif stage == 3:
                return 0x1C
        u11 = rng.random_int(2) != 0 and ctx.unlocked(0x11)
        u12 = rng.random_int(2) != 0 and ctx.unlocked(0x12)
        u22 = ctx.unlocked(0x22)
        tab0 = [2, 1, 0x15 if u11 else 3, 4, 0x13 if u12 else 5, 6, 7, 0x19 if u22 else 8]
        r = rng.random_int(2)
        tab1 = [0x22, 0x25, 0x1D, 0x1A, 0x1E if r == 0 else 0x23, 0x1E, 0x1F, 0x19 if u22 else 8]
        if stage in (1, 3):
            if rng.random_int(2) == 1:
                ctx.set_flag(4, True)
                idx = stage
            else:
                ctx.set_flag(4, False)
                idx = stage - 1
        elif stage in (2, 4):
            idx = stage - 2 if ctx.flag(4) else stage - 1
        else:
            if stage == 9:
                return 0x3F
            idx = stage - 1
        if idx > 7:
            raise AssertionError('choose_boss: Index out of bounds')
        return tab1[idx] if st == 1 else tab0[idx]

    # ------------------------------------------------------------------ helpers
    def excluded(self, room_type: int) -> bool:
        return room_type in self.ctx.excluded_room_types

    def _pick_boss_room(self, rng: RNG, boss_id: int, dt: int) -> Room | None:
        """The 51-try loop shared by the boss slots; raises when the engine would fail."""
        ctx = self.ctx
        tries = 0x33
        while True:
            if dt == 0:
                room = self.rc.get_random_room(rng.next(), True, 0, ROOM_BOSS, SHAPE_ANY, 0, 0xFFFFFFFF, 1,
                                               10, 0, boss_id)
            else:
                room = self.rc.get_random_room(rng.next(), True, 0, ROOM_BOSS, SHAPE_ANY, dt, dt + 0x31, 1,
                                               10, 0, -1)
            tries -= 1
            if tries == 0:
                return room
            if room is None:
                return None
            if dt == 0:
                if not 0xE74 <= room.variant < 0xE74 + 0x96:
                    return room
            else:
                s = room.subtype
                locked = ((not ctx.unlocked(0x10) and s == 0x14) or (not ctx.unlocked(0x11) and s == 0x15)
                          or (not ctx.unlocked(0x12) and s == 0x13) or (not ctx.unlocked(0x44) and s == 0x2A)
                          or (not ctx.unlocked(0x15A) and 0x38 < s < 0x43)
                          or (not ctx.unlocked(0x15B) and 0x42 < s < 0x49))
                if not locked:
                    return room

    def _treasure_subtype(self, second: bool = False) -> tuple[int, int]:
        """The treasure-room subtype roll; returns (seed, subtype). Consumes RNG like the engine.
        The second (labyrinth) treasure room skips the special_2160cc == 0x13 check."""
        ctx, rng, p = self.ctx, self.rng, self.ctx.player
        if rng.random_int(100) == 0:
            has112 = 0x70 in p.trinkets
            return rng.next(), 3 if has112 else 1
        u = rng.random_int(100 // p.trinket_multiplier())
        if (u < 0xF and 0x52 in p.trinkets) or (not second and ctx.special_2160cc == 0x13):
            has112 = 0x70 in p.trinkets
            return rng.next(), 3 if has112 else 1
        has112 = 0x70 in p.trinkets
        return rng.next(), 2 if has112 else 0

    def _shop_level(self) -> int:
        ctx = self.ctx
        lv = 2 if ctx.unlocked(0x98) else int(ctx.unlocked(0x97))
        if ctx.unlocked(0x99):
            lv = 3
        if ctx.unlocked(0x9A):
            lv = 4
        return lv

    def blacklist(self) -> set[int]:
        """Level::build_secret_room_index_blacklist (0x337210)."""
        out: set[int] = set()
        if self.stage == 0xB:
            for off in (1, -1, 13, -13):
                c = self.start_index + off
                if 0 <= c < GRID * GRID:
                    out.add(c)
        for desc in self.rooms:
            cfg = desc.config
            if cfg is None or desc.grid_index < 0:
                continue
            y, x = divmod(desc.grid_index, GRID)
            wx, hy = cfg.width // 13, cfg.height // 7
            cells = [(x - 1, y), (x, y - 1), (x + wx, y), (x, y + hy), (x - 1, y + 1), (x + 1, y - 1),
                     (x + wx, y + 1), (x + 1, y + hy)]
            for slot in range(8):
                shape = cfg.shape
                if shape == 4 and slot in (7, 5):
                    continue
                if shape == 6 and slot in (6, 4):
                    continue
                if shape == 1 and 4 <= slot <= 7:
                    continue
                ci = index(*cells[slot])
                if ((not (cfg.doors >> slot) & 1) or cfg.type in (5, 8, 7)) and ci >= 0:
                    out.add(ci)
        return out

    # ------------------------------------------------------------------ Level::place_rooms
    def place_rooms(self, gen: LevelGenerator, min_d: int, max_d: int) -> bool:
        ctx, rng, stage, rc, p = self.ctx, self.rng, self.stage, self.rc, self.ctx.player
        sid = stage_id(stage, self.stage_type)
        self.grid = [-1] * (GRID * GRID)
        self.rooms = []
        # ---- boss room(s)
        boss_id = self.choose_boss(stage)
        dt = self.choose_double_trouble(stage)
        local = RNG(rng.next(), 0x27)
        room = self._pick_boss_room(local, boss_id, dt)
        if room is None:
            return False
        gr = gen.get_new_boss_room(room.shape, room.doors, False)
        if gr is None:
            return False
        self.boss_list_index = len(self.rooms)
        self.place_room(gr, room, rng.seed)
        if stage == 12:
            if not self._place_void_bosses(gen):
                return False
        if self.curses & CURSE_LABYRINTH:
            saved = rng.seed
            boss2 = self.choose_boss(stage + 1)
            dt2 = self.choose_double_trouble(stage + 1)
            if dt2 == 0:
                room2 = self._pick_boss_room(rng, boss2, 0)
            else:  # a single draw here: no retry and no unlock check
                room2 = rc.get_random_room(rng.next(), True, 0, ROOM_BOSS, SHAPE_ANY, dt2, dt2 + 0x31, 1, 10,
                                           0, -1)
            if room2 is None:
                return False
            gr2 = gen.get_new_boss_room(room2.shape, room2.doors, False)
            self.boss_list_index = len(self.rooms)
            if gr2 is None:
                return False
            self.place_room(gr2, room2, rng.seed)
            rng.set_seed(saved, 0x23)
        # ---- super secret room
        if not self.excluded(ROOM_SUPERSECRET):
            cfg = rc.get_random_room(rng.next(), True, 0, ROOM_SUPERSECRET, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10,
                                     0, -1)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            self.place_room(gr, cfg, rng.seed)
        secret_seed = rng.next()
        shop_index = -1
        # ---- shop and treasure room
        def place_shop() -> bool:
            nonlocal shop_index
            if self.excluded(ROOM_SHOP) or ctx.victory_laps > 2:
                return True
            lv = self._shop_level()
            r = rng.random_int(0x100)
            if r == 0:
                lv = 11
            elif lv > 1 and r == 1:
                lv = 10
            cfg = rc.get_random_room(rng.next(), True, 0, ROOM_SHOP, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, lv)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            shop_index = len(self.rooms)
            self.place_room(gr, cfg, rng.seed)
            return True

        def place_treasure() -> bool:
            if self.excluded(ROOM_TREASURE):
                return True
            seed, sub = self._treasure_subtype()
            cfg = rc.get_random_room(seed, True, 0, ROOM_TREASURE, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, sub)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            self.place_room(gr, cfg, rng.seed)
            if self.curses & CURSE_LABYRINTH:
                saved = rng.seed
                seed, sub = self._treasure_subtype(second=True)
                cfg = rc.get_random_room(seed, True, 0, ROOM_TREASURE, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, sub)
                gr = gen.get_new_end_room(cfg.shape, cfg.doors)
                if gr is None:
                    return False
                self.place_room(gr, cfg, rng.seed)
                rng.set_seed(saved, 0x23)
            return True

        if stage < 7:
            if not place_shop():
                return False
            if not place_treasure():
                return False
        elif stage < 9:
            if 0x6E in p.trinkets and not place_shop():      # Silver Dollar
                return False
            if 0x6F in p.trinkets and not place_treasure():  # Bloody Crown
                return False
        miniboss_missing = True
        if stage < 11:
            # ---- dice room / sacrifice room
            if rng.random_int(0x32) == 0 or (rng.random_int(5) == 0 and p.keys > 1):
                rtype = ROOM_SACRIFICE if self.excluded(ROOM_DICE) else ROOM_DICE
            else:
                rtype = ROOM_SACRIFICE
            cfg = rc.get_random_room(rng.next(), True, 0, rtype, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if self.excluded(ROOM_SACRIFICE) or (
                    rng.random_int(7) != 0 and (rng.random_int(4) != 0
                                                or p.hearts + p.soul_hearts < p.max_hearts)):
                if gr is not None:
                    gen.add_end_room(gr)
            elif gr is not None:
                self.place_room(gr, cfg, rng.seed)
            # ---- library
            u97, u98 = ctx.unlocked(0x97), ctx.unlocked(0x98)
            lv = 2 if u98 else int(u97)
            if ctx.unlocked(0x99):
                lv = 3
            span = 5 if ctx.unlocked(0x9A) else lv + 1
            sub = rng.random_int(span)
            cfg = rc.get_random_room(rng.next(), False, 0, ROOM_LIBRARY, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, sub)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if self.excluded(ROOM_LIBRARY) or (
                    rng.random_int(0x14) != 0 and (rng.random_int(4) != 0 or not ctx.flag(8))):
                if gr is not None:
                    gen.add_end_room(gr)
            elif gr is not None:
                self.place_room(gr, cfg, rng.seed)
            # ---- curse room
            cfg = rc.get_random_room(rng.next(), True, 0, ROOM_CURSE, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
            curse_gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if self.excluded(ROOM_CURSE) or (
                    rng.random_int(2) != 0 and (rng.random_int(4) != 0 or not ctx.flag(6))):
                if curse_gr is not None:
                    gen.add_end_room(curse_gr)
            elif curse_gr is not None:
                self.place_room(curse_gr, cfg, rng.seed)
            # ---- miniboss room
            mb_cfg, mb_gr, lo = cfg, curse_gr, -1
            denom = 30 if ctx.unlocked(4) else 80
            if ctx.unlocked(0x21):
                denom = 10
            if any(ctx.unlocked(a) for a in (0x31, 0x32, 0x33, 0x34, 0x35, 0x37)):
                denom = 5
            if stage < 3 or rng.random_int(10) != 0 or ctx.flag(32):
                high = rng.random_int(denom) != 0
                sins = []
                for bit, base in ((9, 0x834), (10, 0x83E), (11, 0x848), (12, 0x852), (13, 0x866),
                                  (14, 0x870)):
                    if not ctx.flag(bit):
                        sins.append(base - 100 if high else base)
                if sins:
                    lo = sins[rng.random_int(len(sins))]
                    hi = lo + 9
                    mb_cfg = rc.get_random_room(rng.next(), True, 0, ROOM_MINIBOSS, SHAPE_ANY, lo, hi, 1, 10,
                                                0, -1)
                    mb_gr = gen.get_new_end_room(mb_cfg.shape, mb_cfg.doors) if mb_cfg is not None else None
            else:
                rng.next()
                lo = 0x8D4
                mb_cfg = rc.get_random_room(rng.next(), True, 0, ROOM_MINIBOSS, SHAPE_ANY, lo, 0x8DD, 1, 10,
                                            0, -1)
                mb_gr = gen.get_new_end_room(mb_cfg.shape, mb_cfg.doors) if mb_cfg is not None else None
            if self.excluded(ROOM_MINIBOSS) or (
                    rng.random_int(4) != 0 and (rng.random_int(3) != 0 or stage != 1)):
                if mb_gr is not None:
                    gen.add_end_room(mb_gr)
                miniboss_missing = True
            else:
                miniboss_missing = True
                if mb_gr is not None:
                    self.place_room(mb_gr, mb_cfg, rng.seed)
                    for bit, a, b in ((9, 0x834, 2000), (10, 0x83E, 0x7DA), (11, 0x848, 0x7E4),
                                      (12, 0x852, 0x7EE), (13, 0x866, 0x802), (14, 0x870, 0x80C)):
                        if lo in (a, b):
                            ctx.set_flag(bit, True)
                    miniboss_missing = False
                    if lo == 0x8D4:
                        ctx.set_flag(0x20, True)
            # ---- challenge room
            cfg = rc.get_random_room(rng.next(), True, 0, ROOM_CHALLENGE, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if self.excluded(ROOM_CHALLENGE) or (rng.random_int(2) != 0 and stage < 3) or (
                    p.hearts + p.soul_hearts < p.max_hearts or stage < 2):
                if gr is not None:
                    gen.add_end_room(gr)
            elif gr is not None:
                self.place_room(gr, cfg, rng.seed)
            # ---- arcade / vault
            if rng.random_int(10) == 0 or (rng.random_int(3) == 0 and p.keys > 1):
                rtype = ROOM_ARCADE if self.excluded(ROOM_CHEST) else ROOM_CHEST
            else:
                rtype = ROOM_ARCADE
            cfg = rc.get_random_room(rng.next(), True, 0, rtype, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if self.excluded(ROOM_ARCADE) or p.coins < 5 or stage not in (2, 4, 6, 8):
                if gr is not None:
                    gen.add_end_room(gr)
            elif gr is not None:
                self.place_room(gr, cfg, rng.seed)
            # ---- bedroom
            if stage < 7:
                ex18, ex19 = self.excluded(ROOM_ISAACS), self.excluded(ROOM_BARREN)
                if (rng.random_int(2) == 0 or ex19) and not ex18:
                    rtype = ROOM_ISAACS
                else:
                    rtype = ROOM_BARREN
                cfg = rc.get_random_room(rng.next(), True, 0, rtype, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
                gr = gen.get_new_end_room(cfg.shape, cfg.doors)
                if p.soul_hearts > 0 or p.hearts > 1:
                    low = p.soul_hearts < 3 and p.effective_max_hearts() < 1
                else:
                    low = True
                if rng.random_int(0x32) == 0 or (rng.random_int(5) == 0 and low):
                    if gr is not None:
                        self.place_room(gr, cfg, rng.seed)
                elif gr is not None:
                    gen.add_end_room(gr)
        # ---- secret room(s)
        secret_index = -1
        if not self.excluded(ROOM_SECRET):
            cfg = rc.get_random_room(secret_seed, True, 0, ROOM_SECRET, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
            gr = gen.get_new_secret_room(self.blacklist())
            if gr is not None:
                secret_index = len(self.rooms)
                self.place_room(gr, cfg, secret_seed)
        if 0x66 in p.trinkets:
            local = RNG(secret_seed, 1)
            if not self.excluded(ROOM_SECRET):
                cfg = rc.get_random_room(local.next(), True, 0, ROOM_SECRET, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10,
                                         0, -1)
                gr = gen.get_new_secret_room(self.blacklist())
                if gr is not None:
                    secret_index = len(self.rooms)
                    self.place_room(gr, cfg, secret_seed)
        # ---- Dark Room: the tomb
        if stage == 0xB and self.stage_type == 0:
            seed = rng.next()
            if not ctx.unlocked(0x186):
                cfg = rc.get_room(0, ROOM_DEFAULT, 3)
            else:
                cfg = rc.get_random_room(seed, True, 0, ROOM_DEFAULT, SHAPE_ANY, 3, 9, 1, 10, 0, -1)
            gr = gen.get_new_end_room(cfg.shape, cfg.doors)
            if gr is None:
                return False
            self.place_room(gr, cfg, seed)
        # ---- every remaining room
        for gr in gen.get_remaining_rooms():
            gi = index(gr.x, gr.y)
            if gi == self.start_index and gr.depth == 0:
                if stage == 0xB and self.stage_type == 1:
                    cfg = rc.get_room(0x11, ROOM_DEFAULT, 0)
                elif stage == 0xB and self.stage_type == 0:
                    cfg = rc.get_room(0x10, ROOM_DEFAULT, 0)
                else:
                    cfg = rc.get_room(0, ROOM_DEFAULT, 2)
            else:
                cfg = rc.get_random_room(rng.next(), True, sid, ROOM_DEFAULT, gr.shape, int(stage == 0xB),
                                         0xFFFFFFFF, min_d, max_d, gr.doors, -1, void_level=stage == 12)
            if cfg is None:
                return False
            self.place_room(gr, cfg, rng.seed)
        # ---- Greed in the secret room / shop (content only; the RNG draws keep later rooms aligned)
        allow_greed = miniboss_missing and (ctx.state_flags & 0x18000) != 0x18000
        base = GREED_BASE if p.coins >= 0x14 else GREED_BASE_POOR
        if ctx.state_flags & (1 << 30):
            ctx.state_flags &= ~(1 << 30)
            base = F32(base + GREED_FLAG30)
        if ctx.state_flags & (1 << 31):
            base = F32(base + GREED_FLAG31)
            ctx.state_flags &= ~(1 << 31)
        sub = GREED_SUB if ctx.greed_counter >= 5 else F32(F32(ctx.greed_counter) * GREED_SUB_STEP)
        ctx.greed_counter = 0
        thr = F32(F32(base - sub) * GREED_SCALE)
        if rng.random_float() < thr and allow_greed and secret_index >= 0 and stage > 4:
            roll = RNG(rng.seed, 0xC).random_int(1000)
            self.rooms[secret_index].flags |= 0x10
            allow_greed = roll == 0
        if rng.random_float() < thr and allow_greed and shop_index >= 0 and stage > 3:
            RNG(rng.seed, 0xC).random_int(1000)
            self.rooms[shop_index].flags |= 0x10
        # ---- off-grid rooms (seeds and layouts); only the RNG consumption matters for the grid
        for _slot, rtype, sub, stage_of in ((-2, ROOM_ERROR, -1, 0), (-6, ROOM_BLACK_MARKET, -1, 0),
                                             (-4, ROOM_DUNGEON, -1, 0)):
            rng.next(), rng.next(), rng.next()
            rc.get_random_room(rng.next(), True, stage_of, rtype, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, sub)
        rng.next(), rng.next(), rng.next()
        if 0x226 not in p.collectibles:
            rc.get_random_room(rng.next(), True, 0, ROOM_BOSSRUSH, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, -1)
        rng.next(), rng.next(), rng.next()
        rc.get_random_room(rng.next(), True, 0, ROOM_BOSS, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, 0x37)
        rng.next(), rng.next(), rng.next()
        rc.get_random_room(rng.next(), True, 0xD, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 10, 0, 1)
        return True

    def _place_void_bosses(self, gen: LevelGenerator) -> bool:
        rng = self.rng
        extra = rng.random_int(4)
        pool = [b for b in range(0x46) if b not in (0x3F, 0x37, 0x3E, 6) and b not in VOID_EXCLUDED_BOSSES]
        i = len(pool) - 1
        while i > 0:
            j = rng.random_int(i + 1)
            pool[i], pool[j] = pool[j], pool[i]
            i -= 1
        for k in range(extra + 5):
            boss = pool[k]
            room = self._pick_boss_room(rng, boss, 0)
            if room is None:
                return False
            gr = gen.get_new_boss_room(room.shape, room.doors, True)
            if gr is None:
                if k < 5:
                    return False
                break
            self.boss_list_index = len(self.rooms)
            self.place_room(gr, room, rng.seed)
        return True


def generate_floor(room_config: RoomConfig, ctx: GameContext, stage: int, stage_type: int,
                   stage_seed: int) -> LevelResult:
    return Level(room_config, ctx, stage, stage_type).init(stage_seed)
