"""LevelGenerator: the floor layout (room grid) of Afterbirth+ v1.06, translated from named symbols.

Every function below mirrors one LevelGenerator method (RVA in the docstring). RNG calls happen in
the same order as in the binary; the order of candidates, dead ends and shuffles is kept exactly,
since any difference changes every later random number.

Grid: 13x13 cells, index = x + 13*y. Door slots: 0 LEFT0, 1 UP0, 2 RIGHT0, 3 DOWN0, 4 LEFT1, 5 UP1,
6 RIGHT1, 7 DOWN1. Directions (travel): 0 left, 1 up, 2 right, 3 down.
"""
from __future__ import annotations

from .rng import RNG

GRID = 13

# get_room_size (0x340F10): (w, h) per shape 0..12
ROOM_SIZE = [(0, 0), (1, 1), (1, 1), (1, 1), (1, 2), (1, 2), (2, 1), (2, 1), (2, 2), (2, 2), (2, 2),
             (2, 2), (2, 2)]
# travel (0x341060)
TRAVEL = [(-1, 0), (0, -1), (1, 0), (0, 1)]
# get_room_placement_offsets (0x3417A0) with the shape table at .data 0xD68FE0
_L1 = [(0, 0)]
_L2 = [(0, 0), (0, 1)]
_L3 = [(0, 0), (1, 0)]
_L4 = [(0, 0), (1, 0), (0, 1), (1, 1)]
PLACEMENT = [[], _L1, _L1, _L1, _L2, _L2, _L3, _L3, _L4,
             [(1, 0), (0, 1), (1, 1)],     # 9  LTL (missing top-left)
             [(0, 0), (0, 1), (1, 1)],     # 10 LTR (missing top-right)
             [(0, 0), (1, 0), (1, 1)],     # 11 LBL (missing bottom-left)
             [(0, 0), (1, 0), (0, 1)]]     # 12 LBR (missing bottom-right)

# get_neighbor_candidates (0x343090): Random() range per candidate slot
CANDIDATE_WEIGHTS = [0x30, 0x18, 0x18, 0x18, 0x18, 0x48, 0x48, 0x24, 0x24, 0x24, 0x24, 0x18, 0x18]


def index(x: int, y: int) -> int:
    """LevelGenerator::index (0x341770); x is compared unsigned."""
    if 0 <= x < GRID and 0 <= y < GRID:
        return x + y * GRID
    return -1


def has_shape_slot(shape: int, slot: int, ignore_narrow: bool) -> bool:
    """LevelGenerator::has_shape_slot (0x341110)."""
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
    """get_door_source_position (0x341200): the room's own cell that holds the door (None = invalid)."""
    if not has_shape_slot(shape, slot, ignore_narrow):
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
    """get_door_target_position (0x3413F0): the outside cell the door leads to (None = invalid)."""
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
    """LevelGenerator::Room (0x78 bytes)."""
    __slots__ = ('index', 'x', 'y', 'shape', 'doors', 'dir', 'parent_slot', 'entry', 'neighbors',
                 'neighbor_count', 'dead_end', 'depth', 'flag74')

    def __init__(self, x: int, y: int, shape: int, dir_: int = -1, parent_slot: int = -1, entry=None,
                 depth: int = 0):
        self.index = -1
        self.x, self.y, self.shape = x, y, shape
        self.doors = 0                 # +0x20
        self.dir = dir_                # +0x28 direction from the parent
        self.parent_slot = parent_slot # +0x2C parent's door slot
        self.entry = entry             # +0x30 cell adjacent to the parent's door (None = XY::Invalid)
        self.neighbors: set[int] = set()  # +0x38
        self.neighbor_count = 0        # +0x68
        self.dead_end = False          # +0x6C
        self.depth = depth             # +0x70
        self.flag74 = False            # +0x74 (never set by the translated code)

    def copy_candidate(self) -> 'GenRoom':
        r = GenRoom(self.x, self.y, self.shape, self.dir, self.parent_slot, self.entry, self.depth)
        r.doors, r.neighbors = self.doors, set(self.neighbors)
        r.neighbor_count, r.dead_end, r.flag74 = self.neighbor_count, self.dead_end, self.flag74
        return r

    def __repr__(self):
        return (f'GenRoom(#{self.index} ({self.x},{self.y}) shape={self.shape} depth={self.depth} '
                f'dead_end={self.dead_end} doors={self.doors:08b})')


def random_shuffle(items: list, rng: RNG) -> None:
    """Global::RandomShuffle<It, RNG> (0x3457F0 / 0x33F960): Fisher-Yates from the back."""
    i = len(items) - 1
    while i > 0:
        j = rng.random_int(i + 1)
        items[i], items[j] = items[j], items[i]
        i -= 1


class LevelGenerator:
    def __init__(self, seed: int):
        """LevelGenerator::LevelGenerator (0x3420A0)."""
        self.rng = RNG(seed, 0x23)
        self.shapes = 0x1FFF
        self.grid = [-1] * (GRID * GRID)
        self.rooms: list[GenRoom] = []
        self.dead_ends: list[int] = []   # +0x2D8
        self.others: list[int] = []      # +0x2F0
        self.boss_idx = -1               # +0x308
        self.boss_count = 0              # +0x30C
        self.labyrinth = False           # +0x310
        self.dark_room = False           # +0x311
        self.void = False                # +0x312
        self.usable = True               # +0x313
        self.log: list | None = None     # engine log.txt events, when a list is supplied

    # ---------------------------------------------------------------- geometry against the grid
    def cell(self, pos) -> int:
        return -1 if pos is None else self.grid[index(*pos)] if index(*pos) >= 0 else -1

    def is_pos_free(self, x: int, y: int, shape: int) -> bool:
        """0x341C00. On the Dark Room / Chest the cell above the start room is reserved."""
        start = self.rooms[0]
        reserved = index(start.x, start.y - 1)
        for ox, oy in PLACEMENT[shape]:
            i = index(x + ox, y + oy)
            if (self.dark_room and i == reserved) or i == -1 or self.grid[i] >= 0:
                return False
        return True

    def is_placement_valid(self, x: int, y: int, shape: int) -> bool:
        """0x341CD0: every occupied neighbour must have a door leading back into this room."""
        for slot in range(8):
            src = door_source(x, y, shape, slot, False)
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
        """0x341E20: occupied orthogonal cells that belong to a different room than `pos`."""
        if pos is None:
            return 0
        x, y = pos
        own = index(x, y)
        own_room = self.grid[own] if own >= 0 else None
        n = 0
        for dx, dy in TRAVEL:
            i = index(x + dx, y + dy)
            if i >= 0 and self.grid[i] >= 0 and self.grid[i] != own_room:
                n += 1
        return n

    # ---------------------------------------------------------------- rooms
    def place_room(self, cand: GenRoom) -> GenRoom:
        """0x343A20: append a copy and stamp its cells into the grid."""
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

    def calc_required_doors_room(self, room: GenRoom) -> None:
        """calc_required_doors(Room&) 0x342340."""
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
        """calc_required_doors() 0x342D40."""
        self.dead_ends = []
        self.others = []
        self.mark_dead_ends()
        for i, room in enumerate(self.rooms):
            self.calc_required_doors_room(room)
            (self.dead_ends if room.dead_end else self.others).append(i)

    def mark_dead_ends(self) -> None:
        """0x341EC0. The flag is only ever set here or cleared for parents, never recomputed."""
        for room in self.rooms[1:]:
            if self.count_neighbors(room.entry) < 2:
                room.dead_end = True
        for room in self.rooms[1:]:
            if room.entry is not None:
                dx, dy = TRAVEL[(room.dir + 2) % 4]
                i = index(room.entry[0] + dx, room.entry[1] + dy)
                # An off-grid index reads the upper half of the shape bitset (0) in the engine.
                parent = self.grid[i] if i >= 0 else 0
                if parent < 0:
                    raise AssertionError('invalid connecting neighbor')
                self.rooms[parent].dead_end = False

    def get_neighbor_candidates(self, room_index: int, all_shapes: bool) -> list[GenRoom]:
        """0x343090: 1x1 plus up to 13 larger placements behind each open door slot."""
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

    def make_rooms(self, num_rooms: int) -> None:
        """0x344040."""
        start = self.place_room(GenRoom(6, 6, 1))
        queue = [start.index]            # deque; index 0 is the front
        placed = 0
        loops = 1                        # local_130; the log prints the pass that placed the last room
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
        """0x343C20: first free candidate with < 2 neighbours; no shuffle, no placement check."""
        i = 0
        while i < len(self.rooms):
            for cand in self.get_neighbor_candidates(i, True):
                if self.is_pos_free(cand.x, cand.y, cand.shape) and self.count_neighbors(cand.entry) < 2:
                    self.place_room(cand)
                    return True
            i += 1
        return False

    def sort_dead_ends(self) -> None:
        """0x340E70: selection sort by depth, deepest first; ties keep their order."""
        de = self.dead_ends
        for p in range(len(de) - 1):
            first, best, best_pos = de[p], de[p], p
            for q in range(p + 1, len(de)):
                if self.rooms[best].depth < self.rooms[de[q]].depth:
                    best, best_pos = de[q], q
            de[p], de[best_pos] = best, first

    def generate(self, num_rooms: int, dark_room: bool, labyrinth: bool, void: bool,
                 shapes: int) -> None:
        """LevelGenerator::Generate (0x344840)."""
        self.shapes = shapes
        self.labyrinth, self.void, self.dark_room = labyrinth, void, dark_room
        self.grid = [-1] * (GRID * GRID)
        self.boss_idx, self.boss_count, self.usable = -1, 0, True
        self.dead_ends, self.others, self.rooms = [], [], []
        self.make_rooms(num_rooms)
        self.calc_required_doors()
        if len(self.dead_ends) < 5:
            for _ in range(5):
                if self.log is not None:
                    self.log.append(('new_dead_end',))
                self.make_new_dead_end()
                self.calc_required_doors()
                if len(self.dead_ends) >= 5:
                    break
        self.sort_dead_ends()

    # ---------------------------------------------------------------- special room slots
    def try_resize_endroom(self, room: GenRoom, shape: int, doors: int) -> bool:
        """0x342430: reshape a dead end to `shape` in one of 14 placements, if the layout's doors fit."""
        d = room.dir
        opp = (d + 2) % 4
        ex, ey = room.entry
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
                 None]  # [13] is read from an uninitialised stack slot; no special layout uses shapes 5/7
        own = self.grid[index(ex, ey)]
        for i in range(14):
            if shapes[i] != shape:
                continue
            if slots[i] is None:
                raise NotImplementedError('try_resize_endroom option 13 reads uninitialised memory')
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
                    normal_links += not self.rooms[self.grid[ti]].flag74
                    if not (doors >> slot) & 1:
                        ok = False
            if normal_links < 2 and ok:
                for ox, oy in PLACEMENT[room.shape]:
                    self.grid[index(room.x + ox, room.y + oy)] = -1
                for ox, oy in PLACEMENT[shape]:
                    self.grid[index(px + ox, py + oy)] = own
                room.x, room.y, room.shape = px, py, shape
                for nb in sorted(room.neighbors):
                    self.calc_required_doors_room(self.rooms[nb])
                self.calc_required_doors_room(room)
                return True
        return False

    def determine_boss_room(self, shape: int, doors: int) -> None:
        """0x342AC0."""
        for pos, idx in enumerate(self.dead_ends):
            room = self.rooms[idx]
            if room.depth <= 1:
                continue
            if self.labyrinth:
                ok = True
                x, y = room.entry
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
        """0x343DB0. On Curse of the Labyrinth the second boss room is grown next to the first."""
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
        """0x342CA0: first dead end (deepest first) that can take the layout."""
        for pos, idx in enumerate(self.dead_ends):
            room = self.rooms[idx]
            if self.try_resize_endroom(room, shape, doors):
                del self.dead_ends[pos]
                return room
        return None

    def add_end_room(self, room: GenRoom) -> None:
        """0x342190: give an unused dead end back (appended at the end)."""
        self.dead_ends.append(room.index)

    def create_random_end_room(self, shape: int) -> GenRoom | None:
        """0x344970."""
        order = list(range(len(self.rooms)))
        random_shuffle(order, self.rng)
        for i in range(len(self.rooms)):
            for cand in self.get_neighbor_candidates(order[i], True):
                if (cand.shape == shape and self.is_pos_free(cand.x, cand.y, shape)
                        and self.count_neighbors(cand.entry) < 2):
                    room = self.place_room(cand)
                    self.calc_required_doors()
                    return room
        return None

    def get_remaining_rooms(self) -> list[GenRoom]:
        """0x3421E0: non-dead-ends first, then the dead ends still unassigned."""
        return [self.rooms[i] for i in self.others] + [self.rooms[i] for i in self.dead_ends]

    def get_new_secret_room(self, blacklist: set[int]) -> GenRoom | None:
        """0x344D90: score = Random(5)+10, -100 when a neighbour has no facing door, -6/-3 for 1/2
        neighbours; ties broken by Random(count)."""
        best_score, best = 0, []
        for i in range(GRID * GRID):
            if self.grid[i] >= 0 or i in blacklist:
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
        secret.flag74 = True  # +0x74 = 1: not counted as a normal link by try_resize_endroom
        room = self.place_room(secret)
        self.calc_required_doors_room(room)
        for dx, dy in TRAVEL:
            j = index(room.x + dx, room.y + dy)
            if j >= 0 and self.grid[j] >= 0:
                self.calc_required_doors_room(self.rooms[self.grid[j]])
        return room
