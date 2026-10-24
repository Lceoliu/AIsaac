"""J460 boss selection: BossPool (resources/bosspools.xml) and the boss-room layout picker.

Repentance+ v1.9.7.17 (J460, Windows x86). Addresses are RVAs (VA = RVA + 0x400000). Evidence:
capstone disassembly of rl/runs/human/20260923-session01/worker/runtime/isaac-ng.exe (read only;
Ghidra loses the float argument of PickBoss, the RNG copy in GetBossId and every path after a log
call), the re-decompiled corpus analysis/j460/exports/j460-fixed/{generator,ranges}, REPENTOGON's
BossPool*.zhl / MTRNG.zhl / EntityConfig_Boss.zhl / GameStateBossPool.zhl, and the game's own
resources (bosspools.xml, bossportraits.xml, achievements.xml, challenges.xml, scripts/enums.lua).
Static translation only: nothing here has been checked against the running game.

Functions (RVA -> Python)
  MTRNG::Init(uint)                  0x4FD3C0  MT19937.init: standard init_genrand
  MTRNG::Next()                      0x4FD410  MT19937.next: standard genrand_int32; an unseeded
                                               MTRNG (mti == 625, the value the inlined constructor
                                               stores) seeds itself with 5489. No floats are derived.
  BossPool::Init(uint)               0x218E0   BossPool.init
  BossPool bosspools.xml loader      0x21BF0   parse_boss_pools (name ours). Returns at once when its
                                               2nd argument (a mod) is non-zero, so the per-mod loop
                                               0x4F4890 (ecx = Manager+0x2A6C0) that Init runs
                                               afterwards never changes the pools.
  BossPool::PickBoss(pool) + xmm2    0x22620   BossPool.pick_boss (the float arrives in XMM2)
  BossPool::GetBossId(st, type, RNG*) 0x22830  BossPool.get_boss_id; the RNG* argument is never read
                                               (the console command "testbosspool" passes NULL,
                                               place_rooms passes &BossPool). Replaces AB+
                                               Level::choose_boss and folds in choose_double_trouble.
  BossPool::WasBossRemoved(id)       0x21720   BossPool.was_boss_removed (removed | level blacklist)
  level-blacklist clear              0x3382C0  BossPool.clear_level_blacklist (name ours)
  level-blacklist commit             0x21B50   BossPool.commit_level_blacklist (name ours)
  GameState store / restore          0x22F80 / 0x22EA0  BossPool.store_state / restore_state (names ours)
  boss room layout picker            0x339080  pick_boss_room; __fastcall(ecx = boss id, edx = seed)
  Level current stage id             0x338470  only read by the picker for boss 89; caller supplies it
  Game::GetStateFlag(n)              0x217A0   BossContext.state_flag (n clamped to [0, 52])
  PersistentGameData::Unlocked(id)   0x529AA0  BossContext.unlocked
  RoomConfig::GetStageID(st, t, -1)  0x42D030  rep.roomconfig.stage_id (mode -1: Greed iff difficulty 2/3)
  RoomConfig::GetRandomRoom          0x42C7D0  (+ the stage-fallback wrapper 0x42CDD0) via RepRoomConfig
  bossportraits.xml loader           0x29C420  parse_boss_achievements: EntityConfig_Boss.achievement
                                               (+0x54 of a 0x64-byte record, table at Manager+0x2A690),
                                               -1 when the attribute is absent, ids 1..103 stored

Who calls BossPool::Init, with which seed: always Seeds._gameStartSeed (Game+0x1BB88; Seeds is
Game+0x1BB84). Game::Start 0x2F5320 (call at VA 0x6F5715, after ItemPool::Init), Game::StartDebug
0x2F7750 (VA 0x6F7970), the "[Net]"/"[Daily]" start 0x2F5850 (VA 0x6F5F61), the "[Daily]" start
0x2F6DD0 (VA 0x6F7204), the "[Rerun]" start 0x2F72F0 (VA 0x6F765A), the victory-lap start 0x2F7AC0
(VA 0x6F7F67) and Game::RestoreState 0x2F8140 (VA 0x6F8579, then restore_state at VA 0x6F875E). The
console command "testbosspool" (in 0x28CDC0, VA 0x691C45) is the only caller with another seed: it
histograms 10000 x (Init(random seed), GetBossId(current stage, stage type, NULL)), an in-engine
oracle for this module's distribution.

What Init does with the start seed: a local xorshift RNG (start seed, s_Shifts[26] = (3, 13, 7)) is
stepped once per pool and pool i gets RNG(step i+1, s_Shifts[17] = (2, 21, 9)); both boss bitsets
are sized to 0x68 (104) bits and cleared; bosspools.xml is loaded (every pool it names is emptied and
refilled in file order, total weight re-summed in float32); then every non-empty pool is shuffled
with MT19937(its RNG seed): for i = n-1 .. 1, j = MT() % (i+1), swap entries i and j when j != i.

Where GetBossId and the picker are called (the Level port reproduces these; listed for reference):
  place_rooms 0x339370: GetBossId(level stage, stage type) at VA 0x739475 -> picker at 0x7394C3;
    Curse of the Labyrinth second boss GetBossId(stage + 1, stage type) at 0x739926 -> picker at
    0x73997D; the Void's extra bosses (a shuffled id list, no GetBossId) -> picker at 0x73975A. The
    picker seed is always the level RNG (Level+0x182D4) after one Next.
  place_rooms_backwards 0x341E60 (VA 0x741EC5 / 0x741F24), Level::TryInitializeExtraBossRoom
  0x34D780 (VA 0x74D835 / 0x74D848; its own RNG), the Red Redemption layout 0x34FBE0 (VA 0x750058 /
  0x7500B1).
  Level::generate_dungeon 0x340E10 calls clear_level_blacklist at the head of every attempt
  (VA 0x74176C) and commit_level_blacklist after a successful place_rooms (VA 0x741C22). These are
  the only direct call sites (E8 rel32 scan of .text): the backwards / extra-room / Red Redemption
  callers never commit, their blacklist bits die at the next generate_dungeon attempt.

State kept across floors (all in BossPool, Game+0x1AF70):
  - pools[i].rng: advanced exactly 7 steps by every GetBossId call that gets past the fixed-boss
    checks (only the pool of that call's stage id). The per-attempt draws use a copy that is not
    written back.
  - entry order: fixed by Init's shuffle for the whole run.
  - removed (_removedBosses): |= level blacklist on commit; the bits of a whole pool are cleared
    (in both sets) when PickBoss fails for it; also set directly during play by the horseman
    conversion 0x3481D0 (called from the item-use function 0x1B39D0 at VA 0x5B5797), not ported.
  - level_blacklist (_levelBlacklist): bits set by the harbinger/Fallen shortcuts and by PickBoss.
  - weights: _initialWeight and _weight both hold the XML weight; the BossPool functions never
    change them (no decay like RoomConfig weights).
  - saves: StoreState keeps the 37 pool seeds and the removed bits (GameStateBossPool); RestoreState
    runs after Init(start seed), so a continued run keeps the start-seed shuffle.

Game state read (BossContext; Manager = *(VA 0xC7169C), Game = *(VA 0xC71678)):
  achievements      PersistentGameData+0x38 bool[641] = Manager+0x4C (Manager+0x14 is the PGD); via
                    Unlocked and inline copies (GetBossId: Manager+0x51/0x6E/0x8E = ids 5/34/66;
                    picker: +0x5C/+0x5D/+0x5E/+0x90/+0x1A6/+0x1A7 = ids 16/17/18/68/346/347)
  manager_state     Manager+0x8 (2 = in game; part of the "everything unlocked" override)
  daily_id          Game+0x26630 DailyChallenge._id (non-zero: everything unlocked)
  is_debug          Game+0x26589 Game._isDebug (non-zero: everything unlocked)
  special_daily_id  Game+0x2663C DailyChallenge._specialDailyId (13: fixed bosses)
  challenge         Game+0x26584 (32 Aprils Fool, 34 Ultra Hard, 36 Scat Man)
  difficulty        Game+0x269C8 (only through GetStageID / GetRooms mode -1: 2/3 = Greed)
  state_flags       Game+0x26548, uint64 (flag 6 DEVILROOM_VISITED via GetStateFlag; flag 47
                    BACKWARDS_PATH_INIT tested raw as Game+0x2654C & 0x8000 by the picker)
  The Game field names come from REPENTOGON's Rep 1.7.9b layout shifted by +0xA88, which lands on
  every offset the J460 code uses (_gameStateFlags 0x25AC0, _challenge 0x25AFC, _isDebug 0x25B01,
  _dailyChallenge 0x25BA8, _difficulty 0x25F40). The player/character is not read by any of these
  functions, so there is no character field.

Not settled
  - Pools that bosspools.xml never names (stage ids 0, 18-25, 34 'mortis', 35, 36) keep whatever
    Game's allocation left in _totalWeight/_doubleTroubleRoomVariantStart (the pool constructor
    0x21810, run for all 37 by the Game constructor 0x2F1020, sets neither); they are modelled as
    empty with double trouble 0 (GetBossId then falls back to Monstro). Normal floors never use them
    except Mortis (stage type 5 on stages 7-8, cut content); Greed ids 24/25 and Home 35 would.
  - BossPool_Entry._weight is separate from _initialWeight (PickBoss reads both), but no writer other
    than the loader was found; code outside the functions listed above was not searched for one.
  - Only the whitespace/comment subset of rapidxml is modelled: text content would become data
    nodes that the engine iterates as pools/bosses; the parser raises instead. A boss element
    without a weight reuses the previous boss's weight (uninitialised stack for the very first one:
    the parser raises).
  - Greed: get_boss_id follows GetStageID's Greed ids, but pick_boss_room raises because
    RepRoomConfig has no Greed room sets (GetRooms mode 1). Curse of the Giant is RepRoomConfig's
    NotImplementedError.
  - Mods: bosspools.xml replaced through the resource system is not modelled (mod XMLs passed to the
    loader are ignored by the engine itself).
  - Indirect calls were not scanned: a caller of these functions through a pointer would be missed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


from ..rng import MASK32, RANDOM_FLOAT, RNG
from .roomconfig import SHAPE_ANY, stage_id

from ..f32 import F32  # noqa: E402

POOL_COUNT = 37                 # BossPool_Pool _pool[37], stride 0x3C (destructor 0x21880: 0x3C x 0x25)
BOSS_BITS = 0x68                # both vector<bool> resized to 104 by Init (FUN_00423090(0x68))
BOSS_TABLE_SIZE = 0x68          # bossportraits table size ("BossID out of bounds" bound, ids 0..103)
POOL_SEED_SHIFT = 26            # (3, 13, 7) at VA 0xB1F600: steps the start seed once per pool
POOL_RNG_SHIFT = 17             # (2, 21, 9) at VA 0xB1F594: every pool RNG
ROOM_PICK_SHIFT = 39            # (5, 15, 17) at VA 0xB1F69C: the picker's local RNG
PICK_BOSS_SCANS = 10            # PickBoss repick counter 9 .. 0
GET_BOSS_ATTEMPTS = 10          # GetBossId PickBoss attempts before "defaulting to Monstro"
ROOM_PICK_TRIES = 50            # picker: 50 retries = at most 51 GetRandomRoom calls
VARIANT_SPAN = 0x31             # a variant start v selects variants [v, v + 49]
DT_VARIANT_MIN, DT_VARIANT_COUNT = 0xE74, 0x96   # normal boss picks re-draw variants 3700..3849
MT_DEFAULT_SEED = 5489

ROOM_BOSS, STAGE_SPECIAL_ROOMS, STAGE_MAUSOLEUM = 5, 0, 0x1F

# boss ids (bossportraits.xml names)
MONSTRO, MOM, MOMS_HEART, FAMINE, PESTILENCE, WAR, DEATH = 1, 6, 8, 9, 10, 11, 12
HEADLESS_HORSEMAN, THE_FALLEN, SATAN, IT_LIVES, THE_BLOAT = 22, 23, 24, 25, 30
CONQUEST, ISAAC, BLUE_BABY, THE_LAMB, DELIRIUM = 38, 39, 40, 54, 70
REAP_CREEP, MOTHER, MOM_MAUSOLEUM = 74, 88, 89

# challenges (resources/scripts/enums.lua, challenges.xml)
CHALLENGE_APRILS_FOOL, CHALLENGE_ULTRA_HARD, CHALLENGE_SCAT_MAN = 32, 34, 36
# GameStateFlag (resources/scripts/enums.lua)
STATE_DEVILROOM_VISITED, STATE_BACKWARDS_PATH_INIT = 6, 47
NUM_STATE_FLAGS = 53
# achievements (achievements.xml)
ACH_HARBINGERS, ACH_STEVEN, ACH_CHAD, ACH_GISH = 5, 16, 17, 18
ACH_IT_LIVES, ACH_CONQUEST, ACH_TRIACHNID = 34, 66, 68
ACH_AFTERBIRTH_BOSSES, ACH_AFTERBIRTH_PLUS_BOSSES = 346, 347   # "Something wicked this way comes(+)!"
ACHIEVEMENT_LIMIT = 0x282       # Unlocked(): ids >= 642 are locked

# the loader's static std::map<string, int> (VA 0x421C6E..0x42205F): pool name -> stage id
STAGE_NAME_IDS = {
    'basement': 1, 'cellar': 2, 'caves': 4, 'catacombs': 5, 'depths': 7, 'necropolis': 8,
    'womb': 10, 'utero': 11, 'sheol': 14, 'cathedral': 15, 'dark room': 16, 'chest': 17,
    'burning basement': 3, 'flooded caves': 6, 'dank depths': 9, 'scarred womb': 12,
    'blue womb': 13, 'void': 26, 'downpour': 27, 'mines': 29, 'mausoleum': 31, 'corpse': 33,
    'dross': 28, 'ashpit': 30, 'gehenna': 32, 'mortis': 34,
}

# GetBossId: special daily 13 (switch table VA 0x422E24, stage 6 falls through)
SPECIAL_DAILY_13_BOSSES = {1: 60, 2: 43, 3: 50, 4: 23, 5: 51, 7: 15}
# double-trouble moduli (Random(50) / Random(25) / Random(40) of AB+ choose_double_trouble)
DOUBLE_TROUBLE_MODULUS = {3: 50, 4: 50, 5: 25, 7: 40}
HORSEMEN = {1: FAMINE, 3: PESTILENCE, 5: WAR, 7: DEATH}


# ---------------------------------------------------------------------------------------- MTRNG
class MT19937:
    """MTRNG (0x9C4 bytes: _seeds[624] +0x0, _currentIndex +0x9C0), standard MT19937."""
    N, M = 624, 397
    __slots__ = ('mt', 'mti')

    def __init__(self, seed: int | None = None):
        self.mt = [0] * self.N
        self.mti = self.N + 1               # the inlined constructor (BossPool::Init VA 0x421A64)
        if seed is not None:
            self.init(seed)

    def init(self, seed: int) -> None:
        """MTRNG::Init (0x4FD3C0): init_genrand."""
        mt = self.mt
        x = mt[0] = seed & MASK32
        for i in range(1, self.N):
            x = mt[i] = (1812433253 * (x ^ (x >> 30)) + i) & MASK32
        self.mti = self.N

    def _twist(self) -> None:
        mt, n, m = self.mt, self.N, self.M
        for kk in range(n):
            y = (mt[kk] & 0x80000000) | (mt[(kk + 1) % n] & 0x7FFFFFFF)
            mt[kk] = mt[(kk + m) % n] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)   # mag01 at VA 0xB67F8C
        self.mti = 0

    def next(self) -> int:
        """MTRNG::Next (0x4FD410): genrand_int32."""
        if self.mti >= self.N:
            if self.mti == self.N + 1:
                self.init(MT_DEFAULT_SEED)
            self._twist()
        y = self.mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ (y >> 18)


# ---------------------------------------------------------------------------------------- context
@dataclass
class BossContext:
    """The engine state these functions read. Defaults: every achievement unlocked, normal mode,
    no challenge, not a daily or debug run, no state flags."""
    achievements: frozenset | None = None   # PersistentGameData+0x38 (Manager+0x4C); None = all unlocked
    manager_state: int = 2                  # Manager+0x8
    daily_id: int = 0                       # Game+0x26630 DailyChallenge._id
    is_debug: bool = False                  # Game+0x26589
    special_daily_id: int = 0               # Game+0x2663C DailyChallenge._specialDailyId
    challenge: int = 0                      # Game+0x26584
    difficulty: int = 0                     # Game+0x269C8 (0 normal, 1 hard, 2/3 greed)
    state_flags: int = 0                    # Game+0x26548 (uint64)

    def unlocked(self, achievement: int) -> bool:
        """PersistentGameData::Unlocked (0x529AA0). The inline copies in GetBossId and the picker
        are the same test for their ids (all in 1..641)."""
        if achievement == -2:
            return False
        if achievement < 0:
            return True
        if achievement >= ACHIEVEMENT_LIMIT:
            return False
        if achievement == 0 or self.achievements is None or achievement in self.achievements:
            return True
        return self.manager_state == 2 and (self.daily_id != 0 or bool(self.is_debug))

    def state_flag(self, flag: int) -> bool:
        """Game::GetStateFlag (0x217A0)."""
        flag = min(max(flag, 0), NUM_STATE_FLAGS - 1)
        return bool((self.state_flags >> flag) & 1)

    @property
    def greed(self) -> bool:
        return self.difficulty in (2, 3)


# ---------------------------------------------------------------------------------------- data
@dataclass(slots=True, eq=False)
class BossEntry:
    """BossPool_Entry (0x14 bytes)."""
    id: int                          # +0x0
    initial_weight: float            # +0x4, cumulative scan
    weight: float                    # +0x8, eligibility (> 0 and prev + weight > r)
    achievement: int                 # +0xC, EntityConfig_Boss.achievement of the id (-1 = none)
    room_variant_start: int          # +0x10, XML "room": GetBossId returns -value when non-zero


@dataclass(eq=False)
class Pool:
    """BossPool_Pool (0x3C bytes); `index` is the RoomConfig stage id."""
    index: int
    name: str = ''                                      # +0x0
    entries: list = field(default_factory=list)         # +0x18
    total_weight: float = F32(0)                        # +0x24
    rng: RNG = field(default_factory=RNG)               # +0x28 (constructor 0x21810: default RNG)
    double_trouble: int = 0                             # +0x38 _doubleTroubleRoomVariantStart


@dataclass(frozen=True)
class PoolSpec:
    """One <pool> of bosspools.xml as the loader stores it (entries in file order)."""
    index: int
    name: str
    double_trouble: int
    bosses: tuple                    # (id, weight float32, room)


# ---------------------------------------------------------------------------------------- XML
@dataclass
class XmlNode:
    name: str
    attrs: list
    children: list


_ENTITIES = {'lt': '<', 'gt': '>', 'amp': '&', 'apos': "'", 'quot': '"'}
_WS = ' \t\n\r'


def _translate(value: str) -> str:
    if '&' not in value:
        return value
    out, i = [], 0
    while i < len(value):
        if value[i] == '&':
            end = value.find(';', i)
            key = value[i + 1:end] if end > 0 else ''
            if key in _ENTITIES:
                out.append(_ENTITIES[key])
                i = end + 1
                continue
            if key.startswith('#x') or key.startswith('#X'):
                out.append(chr(int(key[2:], 16)))
                i = end + 1
                continue
            if key.startswith('#') and key[1:].isdigit():
                out.append(chr(int(key[1:])))
                i = end + 1
                continue
        out.append(value[i])
        i += 1
    return ''.join(out)


def parse_xml(text: str) -> list:
    """The part of rapidxml (default flags) that the engine's loaders rely on: comments, <?...?>
    and whitespace between elements produce no nodes, and closing tags are not checked against the
    open element (J460's bosspools.xml closes <bosspools> with </pool>). Returns the top-level
    element nodes."""
    doc = XmlNode('', [], [])
    stack = [doc]
    pos, n = 0, len(text)
    while True:
        lt = text.find('<', pos)
        if text[pos:lt if lt >= 0 else n].strip(_WS):
            raise ValueError(f'XML text content at offset {pos}: rapidxml data nodes are not modelled')
        if lt < 0:
            break
        if text.startswith('<!--', lt):
            end = text.find('-->', lt + 4)
            if end < 0:
                raise ValueError('unterminated XML comment')
            pos = end + 3
        elif text.startswith('<?', lt):
            end = text.find('?>', lt + 2)
            if end < 0:
                raise ValueError('unterminated XML declaration')
            pos = end + 2
        elif text.startswith('<!', lt):
            raise ValueError(f'unsupported XML markup at offset {lt}')
        elif text.startswith('</', lt):
            end = text.find('>', lt)
            if end < 0 or len(stack) == 1:
                raise ValueError(f'unbalanced closing tag at offset {lt}')
            stack.pop()
            pos = end + 1
        else:
            i = lt + 1
            while i < n and text[i] not in _WS + '/>?':
                i += 1
            node = XmlNode(text[lt + 1:i], [], [])
            if not node.name:
                raise ValueError(f'empty element name at offset {lt}')
            while True:
                while i < n and text[i] in _WS:
                    i += 1
                if i >= n:
                    raise ValueError('unexpected end of XML data')
                if text.startswith('/>', i):
                    stack[-1].children.append(node)
                    pos = i + 2
                    break
                if text[i] == '>':
                    stack[-1].children.append(node)
                    stack.append(node)
                    pos = i + 1
                    break
                j = i
                while j < n and text[j] not in _WS + '=/>?':
                    j += 1
                key = text[i:j]
                while j < n and text[j] in _WS:
                    j += 1
                if not key or j >= n or text[j] != '=':
                    raise ValueError(f'malformed attribute at offset {i}')
                j += 1
                while j < n and text[j] in _WS:
                    j += 1
                if j >= n or text[j] not in '"\'':
                    raise ValueError(f'expected a quoted attribute value at offset {j}')
                end = text.find(text[j], j + 1)
                if end < 0:
                    raise ValueError('unterminated attribute value')
                node.attrs.append((key, _translate(text[j + 1:end])))
                i = end + 1
    if len(stack) != 1:
        raise ValueError('unexpected end of XML data (unclosed element)')
    return doc.children


_ATOI = re.compile(r'[ \t\n\v\f\r]*([+-]?[0-9]+)')
_ATOF = re.compile(r'[ \t\n\v\f\r]*([+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)')


def _atoi(s: str) -> int:
    """MSVC atoi: leading whitespace, sign, digits; 0 when there are none; saturates."""
    m = _ATOI.match(s)
    return max(-2 ** 31, min(2 ** 31 - 1, int(m.group(1)))) if m else 0


def _atof(s: str) -> float:
    """atof for plain decimal forms (the only ones in the game's XML files); 0.0 when none."""
    m = _ATOF.match(s)
    return float(m.group(1)) if m else 0.0


def parse_boss_achievements(xml: bytes) -> dict:
    """bossportraits.xml loader (0x29C420): EntityConfig_Boss.achievement per boss id. The
    attribute defaults to -1 per <boss>; ids outside 1..103 are not stored."""
    roots = [e for e in parse_xml(xml.decode('utf-8-sig')) if e.name == 'bosses']
    if not roots:
        raise ValueError("bossportraits.xml: no root node 'bosses'")
    out = {}
    for node in roots[0].children:
        bid, ach = 0, -1
        for key, value in node.attrs:
            if key == 'id':
                bid = _atoi(value)
            elif key == 'achievement':
                ach = _atoi(value)
        if 1 <= bid <= BOSS_TABLE_SIZE - 1:
            out[bid] = ach
    return out


def parse_boss_pools(xml: bytes) -> list:
    """The bosspools.xml loader (0x21BF0): every child of <bosspools> is a pool (attributes
    "name", "doubletrouble"), every child of a pool is a boss ("id", "weight" via atof -> float32,
    "room"). Unknown pool names are logged and skipped ("Boss pool: Unknown stage '%s'"). An id
    outside the 104-entry boss table raises here; the engine logs "BossID out of bounds" and reads
    past the table."""
    roots = [e for e in parse_xml(xml.decode('utf-8-sig')) if e.name == 'bosspools']
    if not roots:
        return []
    specs, weight = [], None
    for pnode in roots[0].children:
        name, dt = '', 0
        for key, value in pnode.attrs:
            if key == 'name':
                name = value
            elif key == 'doubletrouble':
                dt = _atoi(value)
        index = STAGE_NAME_IDS.get(name)
        if index is None:
            continue
        bosses = []
        for bnode in pnode.children:
            bid, room = 0, 0              # weight is not reset per boss (VA 0x4222E5)
            for key, value in bnode.attrs:
                if key == 'id':
                    bid = _atoi(value)
                elif key == 'weight':
                    weight = F32(_atof(value))
                elif key == 'room':
                    room = _atoi(value)
            if weight is None:
                raise ValueError('bosspools.xml: first boss has no weight (uninitialised in the engine)')
            if not 0 <= bid < BOSS_TABLE_SIZE:
                raise ValueError(f'bosspools.xml: BossID out of bounds ({bid})')
            bosses.append((bid, weight, room))
        specs.append(PoolSpec(index, name, dt, tuple(bosses)))
    return specs


# ---------------------------------------------------------------------------------------- BossPool
class BossPool:
    """BossPool (Game+0x1AF70): pools +0x0, _removedBosses +0x8AC, _levelBlacklist +0x8BC."""

    def __init__(self, specs: list, boss_achievements: dict, ctx: BossContext | None = None):
        self.specs = list(specs)
        self.boss_achievements = dict(boss_achievements)
        self.ctx = ctx if ctx is not None else BossContext()
        self.pools = [Pool(i) for i in range(POOL_COUNT)]
        self.removed: set = set()
        self.level_blacklist: set = set()

    @classmethod
    def from_room_config(cls, room_config, start_seed: int | None = None,
                         ctx: BossContext | None = None) -> 'BossPool':
        """Pools from resources/bosspools.xml and achievements from resources/bossportraits.xml of
        the RoomConfig's archives; runs init(start_seed) when a seed is given."""
        archives = room_config.archives
        bp = cls(parse_boss_pools(archives.read('resources/bosspools.xml')),
                 parse_boss_achievements(archives.read('resources/bossportraits.xml')), ctx)
        if start_seed is not None:
            bp.init(start_seed)
        return bp

    def copy(self) -> 'BossPool':
        bp = BossPool.__new__(BossPool)
        bp.specs, bp.boss_achievements, bp.ctx = self.specs, self.boss_achievements, self.ctx
        bp.pools = [Pool(p.index, p.name, list(p.entries), p.total_weight, p.rng.copy(), p.double_trouble)
                    for p in self.pools]
        bp.removed, bp.level_blacklist = set(self.removed), set(self.level_blacklist)
        return bp

    # ------------------------------------------------------------------ BossPool::Init
    def init(self, start_seed: int) -> None:
        start_seed &= MASK32
        if not start_seed:
            raise ValueError('RNG Seed is zero! (BossPool::Init breaks into the debugger)')
        seeder = RNG(start_seed, POOL_SEED_SHIFT)
        for pool in self.pools:
            pool.rng = RNG(seeder.next(), POOL_RNG_SHIFT)
        self.removed.clear()
        self.level_blacklist.clear()
        self._load(self.specs)
        for pool in self.pools:
            entries = pool.entries
            if not entries:
                continue
            mt = MT19937(pool.rng.seed)
            for i in range(len(entries) - 1, 0, -1):
                j = mt.next() % (i + 1)
                if j != i:
                    entries[i], entries[j] = entries[j], entries[i]

    def _load(self, specs: list) -> None:
        for spec in specs:
            pool = self.pools[spec.index]
            pool.name, pool.double_trouble = spec.name, spec.double_trouble
            pool.entries, total = [], F32(0)
            for bid, weight, room in spec.bosses:
                pool.entries.append(BossEntry(bid, weight, weight, self.boss_achievements.get(bid, -1), room))
                total = F32(weight + total)
            pool.total_weight = total

    # ------------------------------------------------------------------ bitsets
    def was_boss_removed(self, boss_id: int) -> bool:
        """BossPool::WasBossRemoved (0x21720)."""
        return boss_id in self.removed or boss_id in self.level_blacklist

    def clear_level_blacklist(self) -> None:
        """0x3382C0: start of every Level::generate_dungeon attempt."""
        self.level_blacklist.clear()

    def commit_level_blacklist(self) -> None:
        """0x21B50: removed |= level blacklist, blacklist cleared (generate_dungeon success)."""
        self.removed |= {b for b in self.level_blacklist if 0 <= b < BOSS_BITS}
        self.level_blacklist.clear()

    def store_state(self) -> tuple:
        """0x22F80 -> GameStateBossPool: (_poolSeeds[37], _removedBosses)."""
        return [p.rng.seed for p in self.pools], frozenset(b for b in self.removed if 0 <= b < BOSS_BITS)

    def restore_state(self, pool_seeds, removed) -> None:
        """0x22EA0; Game::RestoreState calls init(start seed) first."""
        for pool, seed in zip(self.pools, pool_seeds):
            pool.rng = RNG(seed, POOL_RNG_SHIFT)
        self.removed = {b for b in removed if 0 <= b < BOSS_BITS}
        self.level_blacklist = set()

    # ------------------------------------------------------------------ queries
    def pool_for(self, stage: int, stage_type: int, ctx: BossContext | None = None) -> Pool:
        ctx = ctx if ctx is not None else self.ctx
        return self.pools[stage_id(stage, stage_type, greed=ctx.greed)]

    def _eligible(self, entry: BossEntry, ctx: BossContext) -> bool:
        if entry.id == REAP_CREEP and ctx.challenge == CHALLENGE_SCAT_MAN:
            return False
        if entry.achievement >= 0 and not ctx.unlocked(entry.achievement):
            return False
        return not self.was_boss_removed(entry.id)

    # ------------------------------------------------------------------ BossPool::PickBoss
    def pick_boss(self, pool: Pool, r, ctx: BossContext | None = None) -> BossEntry | None:
        """Weighted pick for r in [0, total weight), float32 throughout. A rejected candidate maps r
        to ((cum - r) / initial_weight) * total and rescans (10 scans); after the last one the
        entries after the rejected one are probed cyclically ("boss pool ran out of repicks"). The
        pick is added to the level blacklist."""
        ctx = ctx if ctx is not None else self.ctx
        entries = pool.entries
        if not entries:
            return None
        total, zero = pool.total_weight, F32(0)
        if not zero < total:
            return None
        r = F32(r)
        chosen = None
        repicks = PICK_BOSS_SCANS - 1
        while True:
            cum = zero
            for idx, entry in enumerate(entries):
                prev = cum
                cum = F32(cum + entry.initial_weight)
                if r < cum:
                    if zero < entry.weight and r < F32(entry.weight + prev) and self._eligible(entry, ctx):
                        chosen = entry
                    elif repicks == 0:
                        return self._probe(entries, idx, ctx)
                    else:
                        r = F32(F32(F32(cum - r) / entry.initial_weight) * total)
                    break
            repicks -= 1
            if chosen is not None or repicks < 0:
                break
        if chosen is None:
            return None                       # "failed to pick random boss from pool"
        self.level_blacklist.add(chosen.id)
        return chosen

    def _probe(self, entries: list, start: int, ctx: BossContext) -> BossEntry | None:
        n = len(entries)
        i = (start + 1) % n
        while i != start:
            entry = entries[i]
            if F32(0) < entry.weight and self._eligible(entry, ctx):
                self.level_blacklist.add(entry.id)
                return entry
            i = (i + 1) % n
        return None

    # ------------------------------------------------------------------ BossPool::GetBossId
    def get_boss_id(self, stage: int, stage_type: int, rng=None, ctx: BossContext | None = None) -> int:
        """Boss room subtype for (stage, stage type), or -v: pick a boss room among variants
        [v, v + 49] (double trouble, or an entry's "room" attribute). `rng` is accepted for the
        engine's signature and ignored, as in J460."""
        ctx = ctx if ctx is not None else self.ctx
        pool = self.pool_for(stage, stage_type, ctx)
        if ctx.challenge == CHALLENGE_APRILS_FOOL:
            return THE_BLOAT
        if ctx.special_daily_id == 13 and stage in SPECIAL_DAILY_13_BOSSES:
            return SPECIAL_DAILY_13_BOSSES[stage]
        if stage_type in (4, 5):
            fixed = {6: MOM_MAUSOLEUM, 8: MOTHER, 10: BLUE_BABY, 11: BLUE_BABY, 12: DELIRIUM}.get(stage)
            if fixed is not None:
                return fixed
        else:
            if stage == 6:
                return MOM
            if stage == 8:
                return IT_LIVES if ctx.unlocked(ACH_IT_LIVES) else MOMS_HEART
            if stage == 10:
                return ISAAC if stage_type == 1 else SATAN
            if stage == 11:
                return BLUE_BABY if stage_type == 1 else THE_LAMB
            if stage == 12:
                return DELIRIUM
        r = [pool.rng.next() for _ in range(7)]            # r[5], r[6] are drawn and discarded
        local = pool.rng.copy()
        conquest = (r[0] & 1) == 0 and ctx.unlocked(ACH_CONQUEST)
        if (r[1] % 10 == 0 and ctx.unlocked(ACH_HARBINGERS) and stage in HORSEMEN
                and stage_type not in (4, 5)):
            if r[2] % 10 == 0 and not self.was_boss_removed(HEADLESS_HORSEMAN):
                self.level_blacklist.add(HEADLESS_HORSEMAN)
                return HEADLESS_HORSEMAN
            horseman = HORSEMEN[stage]
            if not self.was_boss_removed(horseman):     # Conquest itself is not checked
                self.level_blacklist.add(horseman)
                if horseman == DEATH and conquest:
                    self.level_blacklist.add(CONQUEST)
                    return CONQUEST
                return horseman
        if pool.double_trouble:
            modulus = 1 if ctx.challenge == CHALLENGE_ULTRA_HARD else DOUBLE_TROUBLE_MODULUS.get(stage, 0)
            if modulus and r[3] % modulus == 0:
                return -pool.double_trouble
        if (r[4] % 10 == 0 and stage_type not in (4, 5) and ctx.state_flag(STATE_DEVILROOM_VISITED)
                and not self.was_boss_removed(THE_FALLEN)):
            self.level_blacklist.add(THE_FALLEN)
            return THE_FALLEN
        for _ in range(GET_BOSS_ATTEMPTS):
            local.next()
            f = F32(F32(F32(local.seed) * RANDOM_FLOAT) * pool.total_weight)
            entry = self.pick_boss(pool, f, ctx)
            if entry is None:
                for e in pool.entries:
                    self.removed.discard(e.id)
                    self.level_blacklist.discard(e.id)
                continue
            if entry.id == REAP_CREEP and ctx.challenge == CHALLENGE_SCAT_MAN:
                continue
            if entry.achievement >= 0 and not ctx.unlocked(entry.achievement):
                continue
            return -entry.room_variant_start if entry.room_variant_start else entry.id
        return MONSTRO                        # "Failed to pick a boss room variant, defaulting to Monstro"


# ---------------------------------------------------------------------------------------- 0x339080
def _boss_room_locked(subtype: int, ctx: BossContext) -> bool:
    return ((not ctx.unlocked(ACH_STEVEN) and subtype == 20)
            or (not ctx.unlocked(ACH_CHAD) and subtype == 21)
            or (not ctx.unlocked(ACH_GISH) and subtype == 19)
            or (not ctx.unlocked(ACH_TRIACHNID) and subtype == 42)
            or (not ctx.unlocked(ACH_AFTERBIRTH_BOSSES) and 57 <= subtype <= 66)
            or (not ctx.unlocked(ACH_AFTERBIRTH_PLUS_BOSSES) and 67 <= subtype <= 72))


def pick_boss_room(room_config, boss_id: int, seed: int, ctx: BossContext | None = None, *,
                   level_stage_id: int | None = None, curses: int = 0, void_level: bool = False):
    """The boss-room layout picker (0x339080). `seed` is what the caller passes in EDX (place_rooms:
    the level RNG after one Next); the picker steps its own RNG(seed, s_Shifts[39]) before every
    draw. All draws: GetRandomRoom(stage file 0, type 5, any shape, difficulty 1..10, no doors,
    reduce weight).

      boss_id >= 0: subtype = boss_id, re-drawn while the room is a double-trouble variant
                    (3700..3849), at most 51 draws. Boss 89 with state flag 47
                    (BACKWARDS_PATH_INIT, Dad's Note) instead makes one draw in `level_stage_id`
                    (0x338470, the current level's stage id) with stage 31 as fallback, no filter.
      boss_id < 0:  variants [-boss_id, -boss_id + 49], any subtype, re-drawn while the room's boss
                    is locked (Steven 16, C.H.A.D. 17, Gish 18, Triachnid 68, subtypes 57-66 346,
                    67-72 347), at most 51 draws.
    Returns the RoomConfig room, or None ("could not find matching boss room")."""
    ctx = ctx if ctx is not None else BossContext()
    if ctx.greed:
        raise NotImplementedError('Greed room sets (GetRooms mode 1) are not modelled by RepRoomConfig')
    if not seed & MASK32:
        raise ValueError('RNG Seed is zero! (the picker breaks into the debugger)')
    rng = RNG(seed & MASK32, ROOM_PICK_SHIFT)

    def draw(s, stage, min_variant, max_variant, subtype):
        return room_config.get_random_room(s, True, stage, ROOM_BOSS, SHAPE_ANY, min_variant, max_variant,
                                           1, 10, 0, subtype, void_level=void_level, curses=curses)

    tries = ROOM_PICK_TRIES
    if boss_id >= 0:
        if boss_id == MOM_MAUSOLEUM and (ctx.state_flags >> STATE_BACKWARDS_PATH_INIT) & 1:
            if level_stage_id is None:
                raise ValueError('boss 89 with state flag 47 needs level_stage_id (Level 0x338470)')
            s = rng.next()
            room = draw(s, level_stage_id, 0, 0xFFFFFFFF, MOM_MAUSOLEUM)
            if room is None:
                room = draw(s, STAGE_MAUSOLEUM, 0, 0xFFFFFFFF, MOM_MAUSOLEUM)
            return room
        while True:
            room = draw(rng.next(), STAGE_SPECIAL_ROOMS, 0, 0xFFFFFFFF, boss_id)
            left, tries = tries, tries - 1
            if left <= 0 or room is None:
                return room
            if room.type != ROOM_BOSS or not DT_VARIANT_MIN <= room.variant < DT_VARIANT_MIN + DT_VARIANT_COUNT:
                return room
    start = -boss_id
    while True:
        room = draw(rng.next(), STAGE_SPECIAL_ROOMS, start, start + VARIANT_SPAN, -1)
        left, tries = tries, tries - 1
        if left <= 0 or room is None:
            return room
        if not _boss_room_locked(room.subtype, ctx):
            return room
