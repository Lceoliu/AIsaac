"""J460 Level: curses, room count, the generation retry loop, and dispatch.

Level::Init                        RVA 0x344940  (stage seeds, curses, weight resets, dispatch)
Level::generate_dungeon            RVA 0x340E10  (room count, dead ends, difficulty, retry loop)
Level::CanStageHaveCurseOfLabyrinth RVA 0x3385C0
Level::HasMirrorDimension          RVA 0x34EF70
Level::HasAbandonedMineshaft       RVA 0x34EFD0
Level::place_rooms                 RVA 0x339370  (rep/place_rooms.py)

Differences from AB+ that this file implements (all static evidence, not run-verified):
- stage types 4/5 (Downpour, Mines, Mausoleum, Corpse and their second variants) take the seed of
  stage + 1 (Seeds has 14 stage seeds);
- curse odds: 1/80 normal and 1/40 hard; Unlocked(4) -> 30 / 20, Unlocked(0x21) -> 10 / 6,
  "beat it all" (Manager+0x2FC) or a Game+0x26630 run -> 5 / 3; Basement I gets no curse unless
  Unlocked(0x21); Home and the Ascent floors get none; Lost and Maze are never rolled in Greed,
  Blind never in a Game+0x26630 run;
- room count: -3 on a floor with the mirror world or the abandoned mineshaft; hard mode +2..3
  as before; the minimum dead ends +1 with collectible 599 and +1 with mirror/mineshaft;
- difficulty ranges: odd stages in hard mode 5-10 (AB+ 5-5); Labyrinth floors and stages 9+ use
  1-10 (hard 5-15);
- from the 10th attempt on, every attempt number divisible by 5 that lacks dead ends adds a room
  (while below 64);
- Depths II (and Depths I with the Labyrinth) reserves cell (6, 5) above the start room when the
  strange door can appear (Unlocked-style check on Manager+0x2C7, see _strange_door_possible);
- after place_rooms, Downpour/Dross II builds the mirror world and Mines/Ashpit II the mineshaft
  (RVA 0x34F0C0 / 0x34F360, not ported);
- bosses come from the run's BossPool (Game+0x1AF70, rep/bosspool.py, passed as `bosspool`): every
  attempt starts by clearing its level blacklist (VA 0x74176C) and a successful one commits it
  (VA 0x741C22).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..level import (CURSE_BLIND, CURSE_DARKNESS, CURSE_LABYRINTH, CURSE_LOST, CURSE_MAZE, CURSE_UNKNOWN,
                     GenerationFailed, LevelResult, Player, ROOM_DEFAULT, RoomDesc)
from ..levelgen import GRID
from ..rng import RNG
from ..roomconfig import SHAPE_ANY
from .roomconfig import CURSE_OF_THE_GIANT, RepRoomConfig, VOID_STAGE_FILES, stage_id

LABYRINTH_FACTOR = 1.8
COLLECTIBLE_BLACK_CANDLE = 0x104
COLLECTIBLE_599 = 599                  # PlayerManager::FirstCollectibleOwner(599) adds a dead end
CHALLENGE_NO_CURSES = 0x12
CHALLENGE_CURSED = 0x25                # adds Darkness, Lost and Blind (0x4C)
CHALLENGE_BIG_FLOORS = 0x15            # room count 40 + Random(5)
CHALLENGE_RED_REDEMPTION = 0x2C


@dataclass
class RepGameContext:
    """Game state read by J460 floor generation. Defaults: a normal single-player run as Isaac with
    every achievement unlocked (what a debug start gives: Game+0x26589 makes Unlocked() true)."""
    achievements: frozenset | None = None   # None = all unlocked (PersistentGameData+0x38 bytes)
    difficulty: int = 0                     # Game+0x269C8: 0 normal, 1 hard, 2/3 Greed
    challenge: int = 0                      # Game+0x26584
    special_run: bool = False               # Game+0x26630 != 0 (daily-style challenge params)
    beat_it_all: bool = False               # Manager+0x2FC != 0
    ascent: bool = False                    # Game+0x2654C bit 16 (the backwards path)
    victory_laps: int = 0                   # Game+0x26774
    extra_curses: int = 0                   # Game+0x26550, OR-ed into Level::GetCurses
    strange_door: bool = True               # Manager+0x2C7 (see _strange_door_possible)
    state_flags: int = 0
    player: Player = field(default_factory=Player)
    # read by Level::place_rooms (rep/place_rooms.py)
    special_daily_id: int = 0               # Game+0x2663C DailyChallenge._specialDailyId (19: treasure
                                            # "options"; 13: fixed bosses in BossPool::GetBossId)
    greed_counter: int = 0                  # Game+0x2656C, zeroed by place_rooms (surprise miniboss odds)
    treasure_rooms_visited: int | None = None   # Game+0x26558 (planetarium odds); None = one treasure
                                            # room entered per earlier floor (stage' - 1)
    planetarium_visits: int = 0             # Game+0x2655C (non-zero: skipped treasure rooms add nothing)
    super_sin_counter: int = 0              # Manager+0x330 + Manager+0x334 (non-zero: super sins 1 in 5;
                                            # the two save counters are not identified)

    def unlocked(self, achievement: int) -> bool:
        """PersistentGameData::Unlocked (RVA 0x529AA0)."""
        if achievement == -2 or not 0 <= achievement < 0x282:
            return achievement != -2
        return achievement == 0 or self.achievements is None or achievement in self.achievements

    def flag(self, bit: int) -> bool:
        """Game::GetStateFlag (RVA 0x217A0): Game+0x26548 (uint64), bit clamped to [0, 52]."""
        return bool((self.state_flags >> min(max(bit, 0), 52)) & 1)

    def set_flag(self, bit: int, value: bool) -> None:
        """The std::bitset set place_rooms uses on Game+0x26548 (RVA 0xD72B0 or inline or/bts)."""
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


class RepLevel:
    def __init__(self, room_config: RepRoomConfig, ctx: RepGameContext, stage: int, stage_type: int,
                 bosspool=None):
        """`bosspool`: the run's rep.bosspool.BossPool (Game+0x1AF70), created once per run with
        BossPool.from_room_config(room_config, start_seed=seeds.start_seed) and shared by its floors."""
        if ctx.greed:
            raise NotImplementedError('Greed mode (RVA 0x343550)')
        if stage == 13:
            raise NotImplementedError('Home (RVA 0x34DE30)')
        if stage == 9:
            raise NotImplementedError('Blue Womb (RVA 0x34A500)')
        if ctx.ascent and stage < 7:
            raise NotImplementedError('Ascent floors (RVA 0x342EF0)')
        if ctx.challenge:
            raise NotImplementedError('challenges')
        self.rc, self.ctx = room_config, ctx
        self.bosspool = bosspool
        self.stage, self.stage_type = stage, stage_type
        self.rng = RNG()
        self.rng2 = RNG()                   # Level+0x182E4, shift triple 2, seeded like the main RNG
        self.curses = 0
        self.rooms: list[RoomDesc] = []
        self.grid = [-1] * (GRID * GRID)
        self.start_index = 0x54
        self.boss_list_index = -1
        self.attempts = 0
        self.log: list | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def alt_path(self) -> bool:
        return self.stage_type in (4, 5)

    def get_curses(self) -> int:
        """Level::GetCurses (RVA 0x348490) without special seeds."""
        return self.curses | self.ctx.extra_curses

    def can_have_labyrinth(self) -> bool:
        return self.stage % 2 == 1 and self.stage < 8 and not self.ctx.greed

    def has_mirror_dimension(self) -> bool:
        if self.ctx.greed or (1 <= self.stage <= 6 and self.ctx.ascent) or not self.alt_path:
            return False
        return self.stage == 2 or (self.stage == 1 and bool(self.get_curses() & CURSE_LABYRINTH))

    def has_abandoned_mineshaft(self) -> bool:
        if self.ctx.greed or (1 <= self.stage <= 6 and self.ctx.ascent) or not self.alt_path:
            return False
        return self.stage == 4 or (self.stage == 3 and bool(self.get_curses() & CURSE_LABYRINTH))

    def _strange_door_possible(self) -> bool:
        """generate_dungeon: Depths II (or Depths I XL), not alt path, no challenge, and the
        Manager+0x2C7 flag (or an Unlocked-style override) -> reserve the cell above the start."""
        if self.ctx.greed or self.alt_path or (1 <= self.stage <= 6 and self.ctx.ascent):
            return False
        if not (self.stage == 6 or (self.stage == 5 and self.get_curses() & CURSE_LABYRINTH)):
            return False
        return self.ctx.challenge == 0 and self.ctx.strange_door

    def curse_denominator(self) -> int:
        ctx = self.ctx
        denom = 40 if ctx.hard else 80
        if ctx.unlocked(4):
            denom = 20 if ctx.hard else 30
        if ctx.unlocked(0x21):
            denom = 6 if ctx.hard else 10
        if ctx.beat_it_all or ctx.special_run:
            denom = 3 if ctx.hard else 5
        if self.stage == 1 and not ctx.unlocked(0x21):
            denom = 0
        return denom

    # ------------------------------------------------------------------ Level::Init
    def init(self, stage_seed: int) -> LevelResult:
        """`stage_seed` is Seeds::GetStageSeed(stage + 1 on stage types 4/5, clamped to 13)."""
        ctx = self.ctx
        self.rng = RNG(stage_seed, 0x23)
        self.rng2 = RNG(stage_seed, 2)
        self.rng.next()
        # reset_room_list (RVA 0x338650) no longer draws from the level RNG
        self.rng.next()
        copy = RNG(self.rng.seed, 0x23)
        curses = 0
        if self.stage != 13 and not (1 <= self.stage <= 6 and ctx.ascent):
            denom = self.curse_denominator()
            if (denom and COLLECTIBLE_BLACK_CANDLE not in ctx.player.collectibles
                    and ctx.challenge != CHALLENGE_NO_CURSES):
                if copy.random_int(denom) == 0:
                    pick = copy.next() % 6
                    if pick == 0:
                        if self.can_have_labyrinth():
                            curses |= CURSE_LABYRINTH
                    elif pick == 1:
                        if not ctx.greed and ctx.challenge != CHALLENGE_RED_REDEMPTION:
                            curses |= CURSE_LOST
                    elif pick == 2:
                        curses |= CURSE_DARKNESS
                    elif pick == 3:
                        curses |= CURSE_UNKNOWN
                    elif pick == 4:
                        if not ctx.greed:
                            curses |= CURSE_MAZE
                    elif not ctx.special_run:
                        curses |= CURSE_BLIND
        if self.stage == 9 or ctx.victory_laps > 2:
            curses &= ~CURSE_DARKNESS
        self.curses = curses
        if self.stage == 12:
            for sid in VOID_STAGE_FILES:
                self.rc.reset_room_weights(sid)
        else:
            self.rc.reset_room_weights(stage_id(self.stage, self.stage_type))
        if self.has_abandoned_mineshaft():
            self.rc.reset_room_weights(0x1D)
        self.rc.reset_room_weights(0)
        gen = self.generate_dungeon(copy)
        return LevelResult(self.stage, self.stage_type, stage_seed, self.curses, self.rooms, self.grid,
                           self.attempts, self.boss_list_index, gen)

    # ------------------------------------------------------------------ Level::generate_dungeon
    def room_count(self, gen_rng: RNG) -> int:
        ctx, rng, stage = self.ctx, self.rng, self.stage
        n = min((stage * 10) // 3 + 5 + (rng.next() & 1), 20)
        if self.alt_path and (self.has_abandoned_mineshaft() or self.has_mirror_dimension()):
            n -= 3
        curses = self.get_curses()
        if curses & CURSE_LABYRINTH:
            n = min(int(n * LABYRINTH_FACTOR), 45)
        elif curses & CURSE_LOST:
            n += 4
        if ctx.challenge == CHALLENGE_BIG_FLOORS:
            n = rng.random_int(5) + 40
        if stage == 12:
            n = rng.random_int(5) + 50
        extra = gen_rng.next() & 1
        if ctx.hard:
            n += 2 + extra
        if curses & CURSE_OF_THE_GIANT:
            n = max((n * 3) // 5, 4)
        return n

    def min_dead_ends(self) -> int:
        labyrinth = bool(self.get_curses() & CURSE_LABYRINTH)
        m = (self.stage != 1) + (6 if labyrinth else 5)
        if self.stage == 12:
            m += 2
        if COLLECTIBLE_599 in self.ctx.player.collectibles:
            m += 1
        if self.has_abandoned_mineshaft() or self.has_mirror_dimension():
            m += 1
        return m

    def difficulty_range(self) -> tuple[int, int]:
        hard = self.ctx.hard
        if not self.get_curses() & CURSE_LABYRINTH and self.stage < 9:
            if hard:
                return (10, 15) if self.stage % 2 == 0 else (5, 10)
            return (5, 10) if self.stage % 2 == 0 else (1, 5)
        return (5, 15) if hard else (1, 10)

    def generate_dungeon(self, gen_rng: RNG):
        from .levelgen import LevelGenerator   # being ported (rep/levelgen.py)
        rooms = self.room_count(gen_rng)
        dead_ends = self.min_dead_ends()
        gen = LevelGenerator(self.rng.next())
        gen.log = self.log
        min_d, max_d = self.difficulty_range()
        sid = stage_id(self.stage, self.stage_type)
        if self.stage == 12:
            min_d, max_d = 5, 15
            normal = []
            for s in VOID_STAGE_FILES:
                normal += self.rc.get_rooms(s, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 5, 15, 0, -1)
        else:
            normal = self.rc.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, min_d, max_d, 0, -1)
        if len(normal) < 20:
            min_d, max_d = 1, 15
            if self.stage == 12:
                normal = []
                for s in VOID_STAGE_FILES:
                    normal += self.rc.get_rooms(s, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 15, 0, -1)
            else:
                normal = self.rc.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 15, 0, -1)
        shapes = 0
        for r in normal:
            shapes |= 1 << r.shape
        if self.get_curses() & CURSE_OF_THE_GIANT:
            shapes &= ~0xE
        labyrinth = bool(self.get_curses() & CURSE_LABYRINTH) or self.ctx.challenge == CHALLENGE_BIG_FLOORS
        bosspool = self.bosspool
        if bosspool is None:
            raise ValueError('J460 floors need the run\'s BossPool (Game+0x1AF70): pass '
                             'bosspool=BossPool.from_room_config(room_config, start_seed=seeds.start_seed)')
        while True:
            bosspool.clear_level_blacklist()           # 0x3382C0 at the loop head, VA 0x74176C
            self.attempts += 1
            if self.attempts > 10000:
                raise GenerationFailed('level generation did not converge')
            if self.log is not None:
                self.log.append(('generate',))
            # Every attempt clears the generator's 169 blocked cells ([ebp-0x140] = generator
            # [ebp-0x3F8] + 0x2B8, memset at VA 0x7417AE), blocks (6,5) above the start room when the
            # strange door can appear (BlockPosition, VA 0x74191C), then calls Generate (VA 0x741A20),
            # whose 7th argument is the start room (the default (6,6) shape 1 room here).
            gen.blocked = [False] * (GRID * GRID)
            if self._strange_door_possible():
                gen.block_position(6, 5)
            gen.generate(rooms, self.stage == 11, labyrinth, self.stage == 12, shapes, dead_ends)
            if not gen.usable:
                if self.log is not None:
                    self.log.append(('unusable',))
                continue
            if len(gen.dead_ends) < dead_ends:
                if self.log is not None:
                    self.log.append(('dead_ends', len(gen.dead_ends), dead_ends))
                if self.attempts > 9 and self.attempts % 5 == 0 and rooms < 0x40:
                    rooms += 1
                continue
            if self.log is not None:
                self.log.append(('placing',))
            self.grid = [-1] * (GRID * GRID)
            if self.place_rooms(gen, min_d, max_d):
                if self.has_mirror_dimension() or self.has_abandoned_mineshaft():
                    raise NotImplementedError('mirror world / abandoned mineshaft (RVA 0x34F0C0 / 0x34F360)')
                # the commit (0x21B50, VA 0x741C22) follows the mirror world, the mineshaft and the Lil
                # Portal room init (VA 0x741C11, which uses an RNG seeded from the start seed)
                bosspool.commit_level_blacklist()
                return gen
            self.rooms = []

    def place_rooms(self, gen, min_d: int, max_d: int) -> bool:
        """Level::place_rooms (RVA 0x339370), see rep/place_rooms.py."""
        from .place_rooms import place_rooms
        return place_rooms(self, gen, min_d, max_d)
