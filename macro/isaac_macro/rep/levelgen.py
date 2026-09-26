"""LevelGenerator: the floor layout (room grid) of Repentance+ v1.9.7.17 (J460, Windows x86).

Ported function by function from the J460 decompile (analysis/j460/exports/j460-fixed/generator),
reading the raw instructions of isaac-ng.exe wherever the decompile was unclear, by diffing each
function against the engine-validated AB+ translation isaac_macro.levelgen. Addresses are J460 RVAs
(VA = RVA + 0x400000) unless marked VA. RNG calls happen in the same order as in the binary. Static
translation only: nothing here has been checked against the running J460 engine.

Grid, door slots and directions are as in AB+: 13x13 cells, index = x + 13*y; slots 0 LEFT0, 1 UP0,
2 RIGHT0, 3 DOWN0, 4 LEFT1, 5 UP1, 6 RIGHT1, 7 DOWN1; directions (travel) 0 left, 1 up, 2 right, 3 down.

Behavioural differences from AB+ (all implemented):
- Generate (0x5ADBB0) takes two more arguments. The make_new_dead_end loop runs while there are fewer
  dead ends than `min_dead_ends` (still at most 5 times; AB+: fewer than 5), and the start room comes
  from the caller. sort_dead_ends is inlined at 0x5ADCB8-0x5ADD22 (same selection sort).
- make_rooms (0x5B04D0) places a copy of the caller's start room instead of building (6,6) shape 1.
  generate_dungeon builds (6,6) shape 1 (0x341921-0x3419A5, Generate call 0x341A20), and so does the
  Ascent generator (0x342EF0, min_dead_ends 2, call 0x343375); the Dark Closet generator (0x351650)
  passes (2,6) shape 2 with min_dead_ends 0 (call 0x351BB6); the Red Redemption builder (0x34FBE0)
  passes a copy of an existing room, whose depth, doors and dead-end flag carry over.
- Blocked cells: 169 bytes at +0x2B8, zeroed by the constructor (0x5ADAF0) but not by Generate
  (generate_dungeon clears them itself before every attempt, memset at 0x3417AE). is_pos_free
  (0x5B0180) treats a blocked cell as taken, so blocked cells also constrain get_neighbor_candidates,
  make_rooms, make_new_dead_end, the Labyrinth boss room and CreateRandomEndRoom; GetNewSecretRoom
  (0x5AE170) skips them, before the score draw, like blacklisted cells. try_resize_endroom ignores
  them. Four new helpers set them: 0x5B1860, 0x5B1910, 0x5B1950, 0x5B1A00 (block_* below).
- is_pos_free with no rooms reserves nothing (AB+ read rooms[0] unconditionally).
- CreateRandomEndRoom (0x5ADDF0) only makes 1x1 rooms whatever its first argument (never read; the
  Greed caller at 0x3436F4 pushes a dead register) and takes a cell blacklist (set<int>*, may be
  NULL) whose cells are skipped. place_rooms fills it with 0x352460 (Level side, not ported here):
  the outside cell (door target, ignore_narrow = true) of every slot of every placed room that has no
  door there in its layout or that is not a default room (type 1, subtype 0).
- New: GetNewUltraSecretRoom (0x5AE640), see get_new_ultra_secret_room.
- The Room record is 0x48 bytes; the flag AB+ keeps at +0x74 is at +0x44 (GenRoom.flag44).
- get_door_source_position (0x5AF3F0) no longer reads its ignore_narrow argument (has_shape_slot is
  called with 0); its only caller, is_placement_valid, passed false in AB+ too.
- An off-grid index reads grid[-1], which on x86 is the 32-bit shape mask at +0x10 (AB+ x64: the zero
  upper half of the bitset): count_neighbors of an off-grid cell and mark_dead_ends' parent lookup.
  Only reachable with caller-made rooms (CreateRoom) whose entry and direction disagree.

Unchanged from AB+ (checked against the decompile, and the instructions where the decompile was
unclear): index, has_shape_slot, get_door_target_position, get_room_placement_offsets and the
size/travel tables, is_placement_valid, count_neighbors, mark_dead_ends, both calc_required_doors,
place_room, get_neighbor_candidates, make_new_dead_end, try_resize_endroom, determine_boss_room,
GetNewBossRoom, GetNewEndRoom, AddEndRoom, GetRemainingRooms, GetNewSecretRoom's scoring, both
RandomShuffle instantiations, CreateRoom (AB+ 0x344C50, not used by the AB+ port).

Not settled:
- try_resize_endroom option 13 (shape 7 behind a horizontal door, 5 behind a vertical one) takes its
  door slot from the stack security cookie in J460 (AB+: an uninitialised stack slot); it is raised,
  not emulated. Whether any Rep+ special-room layout has shape 5 or 7 was not checked.
- The names of 0x5AE640, 0x5B1860, 0x5B1910, 0x5B1950, 0x5B1A00 and 0x352460 are guesses from their
  callers. The second stack argument of 0x5B1860, 0x5B1910 and 0x5B1950 (all ret 8) is never read.
- Engine undefined behaviour is raised instead of emulated: a parent outside the room list in
  mark_dead_ends (the engine logs "invalid connecting neighbor" and writes out of bounds), a direction
  outside 0..3 in try_resize_endroom / determine_boss_room (the engine reads the bytes before the
  travel table), an off-grid cell in place_room or try_resize_endroom (the engine logs and overwrites
  the shape mask).
"""
from __future__ import annotations

from .rng import RNG

GRID = 13

# get_room_size, inlined into a static table at VA 0xC5D378: (w, h) per shape 0..12
ROOM_SIZE = [(0, 0), (1, 1), (1, 1), (1, 1), (1, 2), (1, 2), (2, 1), (2, 1), (2, 2), (2, 2), (2, 2),
             (2, 2), (2, 2)]
# travel, inlined into a static table at VA 0xC5D3E0
TRAVEL = [(-1, 0), (0, -1), (1, 0), (0, 1)]
# get_room_placement_offsets (0x5AFDB0): guarded static lists at VA 0xC7F50C-0xC7F5FB, per-shape
# pointers at VA 0xC5C4A0
_L1 = [(0, 0)]
_L2 = [(0, 0), (0, 1)]
_L3 = [(0, 0), (1, 0)]
_L4 = [(0, 0), (1, 0), (0, 1), (1, 1)]
PLACEMENT = [[], _L1, _L1, _L1, _L2, _L2, _L3, _L3, _L4,
             [(1, 0), (0, 1), (1, 1)],     # 9  LTL (missing top-left)
             [(0, 0), (0, 1), (1, 1)],     # 10 LTR (missing top-right)
             [(0, 0), (1, 0), (1, 1)],     # 11 LBL (missing bottom-left)
             [(0, 0), (1, 0), (0, 1)]]     # 12 LBR (missing bottom-right)

# get_neighbor_candidates (0x5B0B00): Random() range per candidate slot
CANDIDATE_WEIGHTS = [0x30, 0x18, 0x18, 0x18, 0x18, 0x48, 0x48, 0x24, 0x24, 0x24, 0x24, 0x18, 0x18]

# the cell itself and its four neighbours, VA 0xC35E9C (blocked-cell helpers)
BLOCK_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
# the eight cells at Manhattan distance 2, copied by 0x5AE640 from VA 0xBACC90, 0xBACC70, 0xBAB120,
# 0xBAB1C0
RING2 = [(-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0), (1, 1), (0, 2), (-1, 1)]


def index(x: int, y: int) -> int:
    """LevelGenerator::index (0x5AFD70); x and y are compared unsigned."""
    if 0 <= x < GRID and 0 <= y < GRID:
        return x + y * GRID
    return -1


def has_shape_slot(shape: int, slot: int, ignore_narrow: bool) -> bool:
    """LevelGenerator::has_shape_slot (0x5AF370, __fastcall shape/slot)."""
    w, h = ROOM_SIZE[shape]
    if h == 1 and slot in (4, 6):
        return False
    if w == 1 and slot in (5, 7):
        return False
    if shape in (3, 5) and not ignore_narrow and slot in (0, 2, 4, 6):
        return False
    if shape in (2, 7) and not ignore_narrow and slot in (1, 3, 5, 7):
        return False
    return True


def door_source(x: int, y: int, shape: int, slot: int, ignore_narrow: bool = False):
    """get_door_source_position (0x5AF3F0): the room's own cell that holds the door (None = invalid).
    J460 never reads `ignore_narrow` (has_shape_slot(..., 0) at 0x5AF400); kept for the AB+ signature."""
    if not has_shape_slot(shape, slot, False):
        return None
    if shape in (1, 2, 3):
        return x, y
    if shape in (4, 5):
        return (x, y + 1) if slot in (3, 4, 6) else (x, y)
    if shape in (6, 7):
        return (x + 1, y) if slot in (2, 5, 7) else (x, y)
    if shape == 8:
        if slot in (2, 5):
            return x + 1, y
        if slot in (3, 4):
            return x, y + 1
        if slot in (6, 7):
            return x + 1, y + 1
        return x, y
    if shape == 9:
        if slot in (0, 2, 5):
            return x + 1, y
        if slot in (1, 3, 4):
            return x, y + 1
        return x + 1, y + 1
    if shape == 10:
        if slot > 2:
            return (x, y + 1) if slot in (3, 4) else (x + 1, y + 1)
        return x, y
    if shape == 11:
        if slot > 1 and slot != 3:
            return (x + 1, y) if slot in (2, 5) else (x + 1, y + 1)
        return x, y
    if shape == 12:
        if slot > 1:
            return (x + 1, y) if slot in (2, 5, 7) else (x, y + 1)
        return x, y
    return None


def door_target(x: int, y: int, shape: int, slot: int, ignore_narrow: bool = False):
    """get_door_target_position (0x5AF620): the outside cell the door leads to (None = invalid)."""
    if not has_shape_slot(shape, slot, ignore_narrow):
        return None
    if shape == 9:
        table = ((x, y), (x, y), (x + 2, y), (x, y + 2), (x - 1, y + 1), (x + 1, y - 1), (x + 2, y + 1),
                 (x + 1, y + 2))
    elif shape == 10:
        table = ((x - 1, y), (x, y - 1), (x + 1, y), (x, y + 2), (x - 1, y + 1), (x + 1, y),
                 (x + 2, y + 1), (x + 1, y + 2))
    elif shape == 11:
        table = ((x - 1, y), (x, y - 1), (x + 2, y), (x, y + 1), (x, y + 1), (x + 1, y - 1),
                 (x + 2, y + 1), (x + 1, y + 2))
    elif shape == 12:
        table = ((x - 1, y), (x, y - 1), (x + 2, y), (x, y + 2), (x - 1, y + 1), (x + 1, y - 1),
                 (x + 1, y + 1), (x + 1, y + 1))
    else:
        w, h = ROOM_SIZE[shape]
        table = ((x - 1, y), (x, y - 1), (x + w, y), (x, y + h), (x - 1, y + 1), (x + 1, y - 1),
                 (x + w, y + 1), (x + 1, y + h))
    return table[slot]


class GenRoom:
    """LevelGenerator::Room (0x48 bytes; constructor LevelGenerator_Room 0x3381B0). Width and height
    (+0x10/+0x14) are ROOM_SIZE[shape]."""
    __slots__ = ('index', 'x', 'y', 'shape', 'doors', 'dir', 'parent_slot', 'entry', 'neighbors',
                 'neighbor_count', 'dead_end', 'depth', 'flag44')

    def __init__(self, x: int, y: int, shape: int, dir_: int = -1, parent_slot: int = -1, entry=None,
                 depth: int = 0):
        self.index = -1                # +0x04
        self.x, self.y, self.shape = x, y, shape   # +0x08, +0x0C, +0x18
        self.doors = 0                 # +0x1C
        self.dir = dir_                # +0x20 direction from the parent
        self.parent_slot = parent_slot # +0x24 parent's door slot
        self.entry = entry             # +0x28 cell adjacent to the parent's door (None = XY::Invalid)
        self.neighbors: set[int] = set()  # +0x30
        self.neighbor_count = 0        # +0x38
        self.dead_end = False          # +0x3C
        self.depth = depth             # +0x40
        self.flag44 = False            # +0x44 secret and ultra secret rooms (AB+ +0x74)

    @property
    def flag74(self) -> bool:
        """The AB+ name of flag44."""
        return self.flag44

    @flag74.setter
    def flag74(self, value: bool) -> None:
        self.flag44 = value

    def copy_candidate(self) -> 'GenRoom':
        r = GenRoom(self.x, self.y, self.shape, self.dir, self.parent_slot, self.entry, self.depth)
        r.doors, r.neighbors = self.doors, set(self.neighbors)
        r.neighbor_count, r.dead_end, r.flag44 = self.neighbor_count, self.dead_end, self.flag44
        return r

    def __repr__(self):
        return (f'GenRoom(#{self.index} ({self.x},{self.y}) shape={self.shape} depth={self.depth} '
                f'dead_end={self.dead_end} doors={self.doors:08b})')


def random_shuffle(items: list, rng: RNG) -> None:
    """Global::RandomShuffle<It, RNG> (0x5B1B20 for rooms, 0x1CC070 for uint): Fisher-Yates from the
    back, j = Random(i+1)."""
    i = len(items) - 1
    while i > 0:
        j = rng.random_int(i + 1)
        items[i], items[j] = items[j], items[i]
        i -= 1


class LevelGenerator:
    def __init__(self, seed: int):
        """LevelGenerator::LevelGenerator (0x5ADAF0): RNG (seed, triple 35), all shapes, empty grid, no
        blocked cell. The boss fields and flags below are only set by Generate in the engine."""
        self.rng = RNG(seed, 0x23)
        self.shapes = 0x1FFF             # +0x10 (32-bit; grid[-1] aliases it)
        self.grid = [-1] * (GRID * GRID) # +0x14
        self.blocked = [False] * (GRID * GRID)  # +0x2B8
        self.rooms: list[GenRoom] = []   # +0x364
        self.dead_ends: list[int] = []   # +0x370
        self.others: list[int] = []      # +0x37C
        self.boss_idx = -1               # +0x388
        self.boss_count = 0              # +0x38C
        self.labyrinth = False           # +0x390
        self.dark_room = False           # +0x391
        self.void = False                # +0x392
        self.usable = True               # +0x393
        self.log: list | None = None     # engine log.txt events, when a list is supplied

    # ---------------------------------------------------------------- geometry against the grid
    def cell(self, pos) -> int:
        return -1 if pos is None else self.grid[index(*pos)] if index(*pos) >= 0 else -1

    def _grid_at(self, i: int) -> int:
        return self.grid[i] if i >= 0 else self.shapes

    def is_pos_free(self, x: int, y: int, shape: int) -> bool:
        """0x5B0180. On the Dark Room / Chest the cell above the start room is reserved (nothing while
        there are no rooms); blocked cells are not free."""
        reserved = index(self.rooms[0].x, self.rooms[0].y - 1) if self.rooms else -1
        for ox, oy in PLACEMENT[shape]:
            i = index(x + ox, y + oy)
            if (self.dark_room and i == reserved) or i == -1 or self.grid[i] >= 0 or self.blocked[i]:
                return False
        return True

    def is_placement_valid(self, x: int, y: int, shape: int) -> bool:
        """0x5B0240: every occupied neighbour must have a door leading back into this room."""
        for slot in range(8):
            src = door_source(x, y, shape, slot)
            tgt = door_target(x, y, shape, slot, True)
            if tgt is None:
                continue
            ti = index(*tgt)
            if ti < 0 or self.grid[ti] < 0:
                continue
            if src is None:
                return False
            nb = self.rooms[self.grid[ti]]
            if not any(door_target(nb.x, nb.y, nb.shape, s, False) == src for s in range(8)):
                return False
        return True

    def count_neighbors(self, pos) -> int:
        """0x5B0440: occupied orthogonal cells that belong to a different room than `pos` (an off-grid
        `pos` compares against grid[-1], the shape mask)."""
        if pos is None:
            return 0
        x, y = pos
        own_room = self._grid_at(index(x, y))
        n = 0
        for dx, dy in TRAVEL:
            i = index(x + dx, y + dy)
            if i >= 0 and self.grid[i] >= 0 and self.grid[i] != own_room:
                n += 1
        return n

    # ---------------------------------------------------------------- rooms
    def place_room(self, cand: GenRoom) -> GenRoom:
        """0x5B0330: append a copy and stamp its cells into the grid."""
        room = cand.copy_candidate()
        room.index = len(self.rooms)
        self.rooms.append(room)
        if self.log is not None and room.shape != 1:
            self.log.append(('place_room', room.shape))
        for ox, oy in PLACEMENT[room.shape]:
            i = index(room.x + ox, room.y + oy)
            if i < 0:
                raise AssertionError('place_room: invalid room')
            self.grid[i] = room.index
        return room

    def create_room(self, x: int, y: int, shape: int, entry_x: int = -1, entry_y: int = -1,
                    dir_: int = -1) -> GenRoom:
        """CreateRoom (0x5ADD30): place a room at a fixed spot, give it an entry cell and a direction
        from its parent, then recompute every room's doors and dead ends. The fixed layouts (Blue Womb,
        Greed, Home, the mineshaft ...) are built with it; no RNG."""
        room = self.place_room(GenRoom(x, y, shape))
        room.dir = dir_
        room.entry = None if (entry_x, entry_y) == (-1, -1) else (entry_x, entry_y)
        self.calc_required_doors()
        return room

    def calc_required_doors_room(self, room: GenRoom) -> None:
        """calc_required_doors(Room&) 0x5AF980."""
        room.doors = 0
        room.neighbors = set()
        for slot in range(8):
            tgt = door_target(room.x, room.y, room.shape, slot, False)
            if tgt is None:
                continue
            ti = index(*tgt)
            if ti >= 0 and self.grid[ti] >= 0:
                room.doors |= 1 << slot
                room.neighbors.add(self.grid[ti])
        room.neighbor_count = len(room.neighbors)

    def calc_required_doors(self) -> None:
        """calc_required_doors() 0x5AFC70."""
        self.dead_ends = []
        self.others = []
        self.mark_dead_ends()
        for i, room in enumerate(self.rooms):
            self.calc_required_doors_room(room)
            (self.dead_ends if room.dead_end else self.others).append(i)

    def mark_dead_ends(self) -> None:
        """0x5AFB10. The flag is only ever set here or cleared for parents, never recomputed. An off-grid
        parent cell reads grid[-1], the shape mask."""
        for room in self.rooms[1:]:
            if self.count_neighbors(room.entry) < 2:
                room.dead_end = True
        for room in self.rooms[1:]:
            if room.entry is not None:
                dx, dy = TRAVEL[(room.dir + 2) % 4]
                parent = self._grid_at(index(room.entry[0] + dx, room.entry[1] + dy))
                if not 0 <= parent < len(self.rooms):
                    raise AssertionError('invalid connecting neighbor')
                self.rooms[parent].dead_end = False

    def get_neighbor_candidates(self, room_index: int, all_shapes: bool) -> list[GenRoom]:
        """0x5B0B00: 1x1 plus up to 13 larger placements behind each open door slot."""
        room = self.rooms[room_index]
        out: list[GenRoom] = []
        for slot in range(8):
            tgt = door_target(room.x, room.y, room.shape, slot, False)
            if tgt is None or index(*tgt) < 0:
                continue
            tx, ty = tgt
            d = slot & 3
            ax, ay = (tx + TRAVEL[d][0], ty + TRAVEL[d][1]) if d in (0, 1) else (tx, ty)
            horizontal = d in (0, 2)
            if horizontal:
                c, dd = (tx, ty - 1), (ax, ay - 1)
            else:
                c, dd = (tx - 1, ty), (ax - 1, ay)
            a, b = (ax, ay), (tx, ty)
            positions = [a, b, c, a, dd, a, dd, a, dd, a, dd, b, a]
            if d == 2:
                s5, s6, s7, s9 = 11, 9, 10, 12
            elif d == 3:
                s5, s6, s7, s9 = 10, 9, 11, 12
            else:
                s5, s6, s7, s9 = 12, 10 + (d == 1), 10 + (d != 1), 9
            shapes = [6 if horizontal else 4, 4 if horizontal else 6, 4 if horizontal else 6, 8, 8,
                      s5, s6, s7, s7, s9, s9, 2 if horizontal else 3, 7 if horizontal else 5]
            if not self.is_pos_free(tx, ty, 1):
                continue
            depth = room.depth + 1
            if self.shapes & 2:
                out.append(GenRoom(tx, ty, 1, d, slot, (tx, ty), depth))
            for i in range(13):
                r = self.rng.random_int(CANDIDATE_WEIGHTS[i])
                ok = True
                if r != 0 and self.shapes & 2:
                    ok = all_shapes
                shape = shapes[i]
                if (self.shapes >> shape) & 1 and ok:
                    px, py = positions[i]
                    if self.is_pos_free(px, py, shape):
                        out.append(GenRoom(px, py, shape, d, slot, (tx, ty), depth))
        return out

    def make_rooms(self, num_rooms: int, start_room: GenRoom | None = None) -> None:
        """0x5B04D0. Starts from a copy of `start_room`, the caller's room since J460; None stands for
        the (6,6) shape 1 room generate_dungeon builds."""
        start = self.place_room(start_room if start_room is not None else GenRoom(6, 6, 1))
        queue = [start.index]            # deque; index 0 is the front
        placed = 0
        loops = 1                        # the log prints the pass that placed the last room
        while queue:
            if placed >= num_rooms:
                loops -= 1
                break
            cur = queue.pop(0)
            cands = self.get_neighbor_candidates(cur, False)
            random_shuffle(cands, self.rng)
            if cur == start.index:
                limit = self.rng.random_int(2) + 2 if num_rooms < 16 else len(cands)
            else:
                s = (self.rng.random_int(2) + self.rng.random_int(2) + self.rng.random_int(2)
                     + self.rng.random_int(2))
                limit = min(s, len(cands))
            made = 0
            for cand in cands:
                if not self.is_pos_free(cand.x, cand.y, cand.shape):
                    continue
                if self.count_neighbors(cand.entry) >= 2:
                    if not ((self.labyrinth or self.void) and self.rng.random_int(10) == 0):
                        continue
                if not self.is_placement_valid(cand.x, cand.y, cand.shape):
                    continue
                new = self.place_room(cand)
                if self.rng.random_int(3) == 0:
                    queue.append(new.index)
                else:
                    queue.insert(0, new.index)
                made += 1
                placed += 1
                if placed >= num_rooms or made >= limit:
                    break
            if placed >= num_rooms:
                break
            loops += 1
        if self.log is not None:
            if placed < num_rooms:
                self.log.append(('fail', placed, num_rooms, loops - 1))
            else:
                self.log.append(('rooms', placed, loops))

    def make_new_dead_end(self) -> bool:
        """0x5AF1E0: first free candidate with < 2 neighbours; no shuffle, no placement check."""
        i = 0
        while i < len(self.rooms):
            for cand in self.get_neighbor_candidates(self.rooms[i].index, True):
                if self.is_pos_free(cand.x, cand.y, cand.shape) and self.count_neighbors(cand.entry) < 2:
                    self.place_room(cand)
                    return True
            i += 1
        return False

    def sort_dead_ends(self) -> None:
        """Inlined into Generate at 0x5ADCB8-0x5ADD22 (AB+ 0x340E70): selection sort by depth, deepest
        first; ties keep their order."""
        de = self.dead_ends
        for p in range(len(de) - 1):
            first, best, best_pos = de[p], de[p], p
            for q in range(p + 1, len(de)):
                if self.rooms[best].depth < self.rooms[de[q]].depth:
                    best, best_pos = de[q], q
            de[p], de[best_pos] = best, first

    def generate(self, num_rooms: int, dark_room: bool, labyrinth: bool, void: bool, shapes: int,
                 min_dead_ends: int = 5, start_room: GenRoom | None = None) -> None:
        """LevelGenerator::Generate (0x5ADBB0). `min_dead_ends`: new dead ends are made (at most 5
        tries) while there are fewer; `start_room`: see make_rooms. The defaults are the AB+ behaviour.
        The blocked cells are kept."""
        self.dark_room, self.labyrinth, self.void = dark_room, labyrinth, void
        self.shapes = shapes
        self.grid = [-1] * (GRID * GRID)
        self.boss_idx, self.boss_count, self.usable = -1, 0, True
        self.dead_ends, self.others, self.rooms = [], [], []
        self.make_rooms(num_rooms, start_room)
        self.calc_required_doors()
        tries = 5
        while len(self.dead_ends) < min_dead_ends and tries > 0:
            if self.log is not None:
                self.log.append(('new_dead_end',))
            self.make_new_dead_end()
            self.calc_required_doors()
            tries -= 1
        self.sort_dead_ends()

    # ---------------------------------------------------------------- blocked cells (J460 only)
    def block_room_area(self, room: GenRoom) -> None:
        """0x5B1860 (name guessed): block every cell of `room` and its four neighbours. place_rooms
        calls it on the (last) boss room right after placing it (0x339A0D)."""
        for ox, oy in PLACEMENT[room.shape]:
            for dx, dy in BLOCK_OFFSETS:
                i = index(room.x + ox + dx, room.y + oy + dy)
                if i >= 0:
                    self.blocked[i] = True

    def block_position(self, x: int, y: int) -> None:
        """0x5B1910 (BlockPosition, name guessed): block one cell; off the grid nothing happens.
        generate_dungeon blocks (6,5) on some Depths II floors (0x34191C); the Dark Closet generator
        blocks (1,6), (2,5) and (2,7) around its start."""
        i = index(x, y)
        if i >= 0:
            self.blocked[i] = True

    def block_occupied_area(self, other: 'LevelGenerator') -> None:
        """0x5B1950 (name guessed): block every cell that is occupied in `other`'s grid or next to such
        a cell (Red Redemption builder 0x34FBE0)."""
        for y in range(GRID):
            for x in range(GRID):
                i = x + GRID * y
                for dx, dy in BLOCK_OFFSETS:
                    if self.blocked[i]:
                        break
                    j = index(x + dx, y + dy)
                    if j >= 0 and other.grid[j] >= 0:
                        self.blocked[i] = True

    def block_unused_door_targets(self, room: GenRoom, doors: int) -> None:
        """0x5B1A00 (name guessed): block the outside cell (ignore_narrow = true) of every slot of `room`
        missing from the door mask `doors` (place_rooms_backwards 0x341E60)."""
        for slot in range(8):
            if (doors >> slot) & 1:
                continue
            tgt = door_target(room.x, room.y, room.shape, slot, True)
            if tgt is not None and index(*tgt) >= 0:
                self.blocked[index(*tgt)] = True

    # ---------------------------------------------------------------- special room slots
    def try_resize_endroom(self, room: GenRoom, shape: int, doors: int) -> bool:
        """0x5B1220: reshape a dead end to `shape` in one of 14 placements, if the layout's doors fit.
        A room without an entry cell works from XY::Invalid, where every placement is off the grid."""
        d = room.dir
        if d not in (0, 1, 2, 3):
            raise NotImplementedError('try_resize_endroom: direction outside 0..3 reads outside the '
                                      'travel table')
        opp = (d + 2) % 4
        ex, ey = room.entry if room.entry is not None else (-1, -1)
        ax, ay = (ex + TRAVEL[d][0], ey + TRAVEL[d][1]) if d in (0, 1) else (ex, ey)
        horizontal = d in (0, 2)
        c = (ex, ey - 1) if horizontal else (ex - 1, ey)
        dd = (ax, ay - 1) if horizontal else (ax - 1, ay)
        e, a = (ex, ey), (ax, ay)
        positions = [e, a, e, c, a, dd, a, dd, a, dd, a, dd, e, a]
        if d == 1:
            s6, s7, s8, s10 = 12, 11, 10, 9
        elif d == 0:
            s6, s7, s8, s10 = 12, 10, 11, 9
        elif d == 2:
            s6, s7, s8, s10 = 11, 9, 10, 12
        else:
            s6, s7, s8, s10 = 10, 9, 11, 12
        shapes = [1, 6 if horizontal else 4, 4 if horizontal else 6, 4 if horizontal else 6, 8, 8,
                  s6, s7, s8, s8, s10, s10, 2 if horizontal else 3, 7 if horizontal else 5]
        slots = [opp, opp, opp + 4, opp, opp + 4, opp, opp + 4, opp, opp + 4, opp, opp + 4, opp, opp,
                 None]  # [13] is the stack security cookie; no special layout is known to use shapes 5/7
        own = self._grid_at(index(ex, ey))
        for i in range(14):
            if shapes[i] != shape:
                continue
            if slots[i] is None:
                raise NotImplementedError('try_resize_endroom option 13 reads the stack cookie')
            if not (doors >> slots[i]) & 1:
                continue
            px, py = positions[i]
            ok = True
            for ox, oy in PLACEMENT[shape]:
                ci = index(px + ox, py + oy)
                if ci < 0:
                    ok = False
                elif self.grid[ci] >= 0 and self.grid[ci] != own:
                    ok = False
            normal_links = 0
            for slot in range(8):
                tgt = door_target(px, py, shape, slot, False)
                if tgt is None:
                    continue
                ti = index(*tgt)
                if ti >= 0 and self.grid[ti] >= 0 and self.grid[ti] != own:
                    normal_links += not self.rooms[self.grid[ti]].flag44
                    if not (doors >> slot) & 1:
                        ok = False
            if normal_links < 2 and ok:
                for ox, oy in PLACEMENT[room.shape]:
                    ci = index(room.x + ox, room.y + oy)
                    if ci < 0:
                        raise AssertionError('try_resize_endroom: invalid room')
                    self.grid[ci] = -1
                for ox, oy in PLACEMENT[shape]:
                    self.grid[index(px + ox, py + oy)] = own
                room.x, room.y, room.shape = px, py, shape
                for nb in sorted(room.neighbors):
                    self.calc_required_doors_room(self.rooms[nb])
                self.calc_required_doors_room(room)
                return True
        return False

    def determine_boss_room(self, shape: int, doors: int) -> None:
        """0x5AED50."""
        for pos, idx in enumerate(self.dead_ends):
            room = self.rooms[idx]
            if room.depth <= 1:
                continue
            if self.labyrinth:
                if room.dir not in (0, 1, 2, 3):
                    raise NotImplementedError('determine_boss_room: direction outside 0..3')
                ok = True
                x, y = room.entry if room.entry is not None else (-1, -1)
                for _ in range(3):
                    x, y = x + TRAVEL[room.dir][0], y + TRAVEL[room.dir][1]
                    i = index(x, y)
                    if i >= 0:
                        if self.grid[i] != idx and self.grid[i] >= 0:
                            ok = False
                    else:
                        ok = False
                    n = 0
                    for dx, dy in TRAVEL:
                        j = index(x + dx, y + dy)
                        if j >= 0 and self.grid[j] >= 0:
                            n += idx != self.grid[j]
                    if n:
                        ok = False
                if not ok:
                    continue
            if self.try_resize_endroom(room, shape, doors):
                self.boss_idx = idx
                del self.dead_ends[pos]
                return

    def get_new_boss_room(self, shape: int, doors: int, force: bool) -> GenRoom | None:
        """0x5AEF10. On Curse of the Labyrinth the second boss room is grown next to the first."""
        if not self.labyrinth or self.boss_count < 1 or force:
            self.determine_boss_room(shape, doors)
        if self.boss_idx < 0:
            return None
        if not self.try_resize_endroom(self.rooms[self.boss_idx], shape, doors):
            return None
        idx = self.boss_idx
        self.boss_idx = -1
        self.boss_count += 1
        if self.labyrinth and self.boss_count < 2:
            cands = self.get_neighbor_candidates(idx, False)
            random_shuffle(cands, self.rng)
            for cand in cands:
                if (self.is_pos_free(cand.x, cand.y, cand.shape) and self.count_neighbors(cand.entry) < 2
                        and (doors >> cand.parent_slot) & 1):
                    new = self.place_room(cand)
                    self.boss_idx = new.index
                    new.dead_end = True
                    self.calc_required_doors_room(new)
                    self.rooms[idx].doors |= 1 << cand.parent_slot
                    break
        return self.rooms[idx]

    def get_new_end_room(self, shape: int, doors: int) -> GenRoom | None:
        """0x5AE0E0: first dead end (deepest first) that can take the layout."""
        for pos, idx in enumerate(self.dead_ends):
            room = self.rooms[idx]
            if self.try_resize_endroom(room, shape, doors):
                del self.dead_ends[pos]
                return room
        return None

    def add_end_room(self, room: GenRoom) -> None:
        """AddEndRoom (AB+ 0x342190), inlined into place_rooms at 0x33AEDB, 0x33BA6A and 0x33C3B7: give
        an unused dead end back (appended at the end)."""
        self.dead_ends.append(room.index)

    def create_random_end_room(self, shape: int = 1, blacklist: set[int] | None = None) -> GenRoom | None:
        """0x5ADDF0: a new 1x1 dead end next to a random room. `shape` is never read (J460 tests 1);
        `blacklist` (may be None) lists cells the new room must not take."""
        order = list(range(len(self.rooms)))
        random_shuffle(order, self.rng)
        for i in range(len(self.rooms)):
            for cand in self.get_neighbor_candidates(self.rooms[order[i]].index, True):
                if (cand.shape == 1 and self.is_pos_free(cand.x, cand.y, 1)
                        and self.count_neighbors(cand.entry) < 2
                        and (blacklist is None or index(cand.x, cand.y) not in blacklist)):
                    room = self.place_room(cand)
                    self.calc_required_doors()
                    return room
        return None

    def get_remaining_rooms(self) -> list[GenRoom]:
        """0x5AF0E0: non-dead-ends first, then the dead ends still unassigned."""
        return [self.rooms[i] for i in self.others] + [self.rooms[i] for i in self.dead_ends]

    def get_new_secret_room(self, blacklist: set[int]) -> GenRoom | None:
        """0x5AE170: score = Random(5)+10, -100 when a neighbour has no facing door, -6/-3 for 1/2
        neighbours; ties broken by Random(count). Blocked cells are skipped with the blacklisted ones."""
        best_score, best = 0, []
        for i in range(GRID * GRID):
            if self.grid[i] >= 0 or i in blacklist or self.blocked[i]:
                continue
            x, y = i % GRID, i // GRID
            score = self.rng.random_int(5) + 10
            count = 0
            for d, (dx, dy) in enumerate(TRAVEL):
                j = index(x + dx, y + dy)
                if j < 0 or self.grid[j] < 0:
                    continue
                nb = self.rooms[self.grid[j]]
                if door_target(nb.x, nb.y, nb.shape, (d + 2) & 3, False) is None:
                    score -= 100
                else:
                    count += 1
            if score < 0 or count == 0:
                continue
            if count < 3:
                score -= 6 if count == 1 else 3
            if best_score < score:
                best_score, best = score, [i]
            elif score == best_score:
                best.append(i)
        if not best:
            return None
        i = best[self.rng.random_int(len(best))]
        secret = GenRoom(i % GRID, i // GRID, 1)
        secret.flag44 = True
        room = self.place_room(secret)
        self.calc_required_doors_room(room)
        for dx, dy in TRAVEL:
            j = index(room.x + dx, room.y + dy)
            if j >= 0 and self.grid[j] >= 0:
                self.calc_required_doors_room(self.rooms[self.grid[j]])
        return room

    def get_new_ultra_secret_room(self, blacklist: set[int] | None = None) -> GenRoom | None:
        """0x5AE640 (name guessed: place_rooms puts a type 29 room on the result, 0x33CB05). A free,
        unblocked, unlisted cell draws score = Random(5)+10 and drops out when an orthogonal neighbour
        is occupied or blocked. Each occupied cell at Manhattan distance 2 (RING2) counts once, and its
        room's door on every side facing the cell (slot opposite to each step d with d . offset > 0)
        must lead to an in-grid, unlisted cell, else -100; -6/-3 for 1/2 such cells; ties broken by
        Random(count). The rooms at distance 2 then get every door whose outside cell is empty and next
        to the new room."""
        best_score, best = 0, []
        for i in range(GRID * GRID):
            if self.grid[i] >= 0 or self.blocked[i] or (blacklist is not None and i in blacklist):
                continue
            x, y = i % GRID, i // GRID
            score = self.rng.random_int(5) + 10
            touching = False
            for dx, dy in TRAVEL:
                j = index(x + dx, y + dy)
                if j >= 0 and (self.grid[j] >= 0 or self.blocked[j]):
                    touching = True
                    break
            if touching:
                continue
            count = 0
            for ox, oy in RING2:
                if score < 0:
                    break
                j = index(x + ox, y + oy)
                if j < 0 or self.grid[j] < 0:
                    continue
                nb = self.rooms[self.grid[j]]
                for d, (dx, dy) in enumerate(TRAVEL):
                    if dx * ox + dy * oy <= 0:
                        continue
                    tgt = door_target(nb.x, nb.y, nb.shape, (d + 2) & 3, False)
                    t = -1 if tgt is None else index(*tgt)
                    if t < 0 or (blacklist is not None and t in blacklist):
                        score -= 100
                        break
                count += 1
            if score < 0 or count == 0:
                continue
            if count < 3:
                score -= 6 if count == 1 else 3
            if best_score < score:
                best_score, best = score, [i]
            elif score == best_score:
                best.append(i)
        if not best:
            return None
        i = best[self.rng.random_int(len(best))]
        ultra = GenRoom(i % GRID, i // GRID, 1)
        ultra.flag44 = True
        room = self.place_room(ultra)
        self.calc_required_doors_room(room)
        for ox, oy in RING2:
            j = index(room.x + ox, room.y + oy)
            if j < 0 or self.grid[j] < 0:
                continue
            nb = self.rooms[self.grid[j]]
            for slot in range(8):
                tgt = door_target(nb.x, nb.y, nb.shape, slot, False)
                if tgt is None or index(*tgt) < 0 or self.grid[index(*tgt)] >= 0:
                    continue
                if abs(tgt[0] - room.x) + abs(tgt[1] - room.y) == 1:
                    nb.doors |= 1 << slot
        return room
