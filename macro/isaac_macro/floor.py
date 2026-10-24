"""Abstract floor: the room graph of a generated level, and what a player can see of it.

A Floor is built from a LevelResult (the offline generator) or from an engine dump. Doors are
recomputed from the final grid with calc_required_doors semantics, because the descriptor copy in
Level::place_room is taken before the secret room is attached to its neighbours. Every pair of
adjacent rooms is connected; the super secret room touches only its parent room.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

from .level import LevelResult, ROOM_DEFAULT, ROOM_SECRET, ROOM_SUPERSECRET
from .levelgen import GRID, PLACEMENT, door_target, index

START_INDEX = 0x54   # (6, 6)
HIDDEN_TYPES = (ROOM_SECRET, ROOM_SUPERSECRET)

ROOM_TYPE_NAMES = {1: 'default', 2: 'shop', 3: 'error', 4: 'treasure', 5: 'boss', 6: 'miniboss',
                   7: 'secret', 8: 'supersecret', 9: 'arcade', 10: 'curse', 11: 'challenge',
                   12: 'library', 13: 'sacrifice', 14: 'devil', 15: 'angel', 16: 'dungeon',
                   17: 'bossrush', 18: 'isaacs', 19: 'barren', 20: 'chest', 21: 'dice',
                   22: 'blackmarket', 23: 'greedexit'}
SHAPE_NAMES = ['', '1x1', 'IH', 'IV', '1x2', 'IIV', '2x1', 'IIH', '2x2', 'LTL', 'LTR', 'LBL', 'LBR']


@dataclass
class FloorRoom:
    index: int                 # position in the level's room list
    x: int                     # top-left of the bounding box
    y: int
    shape: int
    type: int
    variant: int = 0
    subtype: int = 0
    layout_doors: int = 0xFF   # door slots of the room layout (STB); 0xFF = unknown / all
    doors: int = 0             # slots that lead to another room
    depth: int = -1            # generator depth (not observable in game)
    difficulty: int = 0
    name: str = ''
    cells: tuple = field(default=())
    file: int = -1             # room file (stage id) of the layout; -1 = unknown

    def __post_init__(self):
        if not self.cells:
            self.cells = tuple(index(self.x + ox, self.y + oy) for ox, oy in PLACEMENT[self.shape])

    @property
    def grid_index(self) -> int:
        return index(self.x, self.y)

    @property
    def safe_grid_index(self) -> int:
        return self.grid_index + (self.shape == 9)

    def slot_target(self, slot: int) -> int:
        tgt = door_target(self.x, self.y, self.shape, slot, False)
        return -1 if tgt is None else index(*tgt)


class Floor:
    def __init__(self, rooms: list[FloorRoom], stage: int = 1, stage_type: int = 0, curses: int = 0,
                 seed: int = 0, start: int = START_INDEX):
        self.rooms = [copy.copy(r) for r in rooms]   # doors are recomputed per floor
        rooms = self.rooms
        self.stage, self.stage_type, self.curses, self.seed, self.start = stage, stage_type, curses, seed, start
        self.grid = [-1] * (GRID * GRID)
        for r in rooms:
            for c in r.cells:
                if c < 0:
                    raise ValueError(f'room {r.index} lies outside the 13x13 grid')
                self.grid[c] = r.index
        self._pos = {r.index: k for k, r in enumerate(rooms)}
        self.recompute_doors()

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_level(cls, level: LevelResult) -> 'Floor':
        gen = level.generator
        rooms = []
        for d in level.rooms:
            # J460 descriptors can lie off the grid (negative index) or in another dimension
            if d.grid_index < 0 or d.config is None or getattr(d, 'dimension', 0) != 0:
                continue
            x, y = d.grid_index % GRID, d.grid_index // GRID
            depth = -1
            if gen is not None and gen.grid[d.safe_grid_index] >= 0:
                depth = gen.rooms[gen.grid[d.safe_grid_index]].depth
            cfg = d.config
            rooms.append(FloorRoom(d.list_index, x, y, d.shape, cfg.type, cfg.variant, cfg.subtype,
                                   cfg.doors, 0, depth, cfg.difficulty, cfg.name, file=cfg.stage))
        return cls(rooms, level.stage, level.stage_type, level.curses, level.stage_seed)

    @classmethod
    def from_engine_dump(cls, rec: dict, room_config=None) -> 'Floor':
        """A floor read from the engine (rl/bridge/python/abplus_probe_floors.py record).

        GridIndex is the top-left cell of the room's bounding box, as in the generator. With
        `room_config`, each room's layout door slots are looked up (special rooms and the start
        room in the stage 0 file, normal rooms in the floor's own file); otherwise 0xFF."""
        from .roomconfig import stage_id
        sid = stage_id(rec['stage'], rec['stage_type']) if rec['stage'] not in (9, 12) else None
        rooms = []
        for r in rec['rooms']:
            layout = 0xFF
            if room_config is not None:
                cfg = None
                if r['type'] == ROOM_DEFAULT and sid is not None:
                    cfg = room_config.get_room(sid, r['type'], r['variant'])
                if cfg is None:
                    cfg = room_config.get_room(0, r['type'], r['variant'])
                if cfg is not None and cfg.shape == r['shape']:
                    layout = cfg.doors
            rooms.append(FloorRoom(r['list'], r['grid'] % GRID, r['grid'] // GRID, r['shape'], r['type'],
                                   r['variant'], r['subtype'], layout))
        return cls(rooms, rec['stage'], rec['stage_type'], rec['curses'], rec.get('stage_seed', 0),
                   rec.get('start', START_INDEX))

    def recompute_doors(self) -> None:
        for r in self.rooms:
            r.doors = 0
            for slot in range(8):
                t = r.slot_target(slot)
                if t >= 0 and self.grid[t] >= 0 and self.grid[t] != r.index:
                    r.doors |= 1 << slot

    # ---------------------------------------------------------------- queries
    def room(self, list_index: int) -> FloorRoom:
        return self.rooms[self._pos[list_index]]

    def room_at(self, cell: int) -> FloorRoom | None:
        i = self.grid[cell] if 0 <= cell < GRID * GRID else -1
        return self.room(i) if i >= 0 else None

    def of_type(self, room_type: int) -> list[FloorRoom]:
        return [r for r in self.rooms if r.type == room_type]

    def neighbors(self, room: FloorRoom) -> list[tuple[int, FloorRoom]]:
        """(slot, room) for every door of `room`."""
        out = []
        for slot in range(8):
            if room.doors >> slot & 1:
                out.append((slot, self.room_at(room.slot_target(slot))))
        return out

    def distances(self, source: int | None = None, exclude=()) -> dict[int, int]:
        """BFS room distances (by list index) from the room at cell `source` (default: start)."""
        src = self.room_at(self.start if source is None else source)
        dist = {src.index: 0}
        queue = deque([src])
        while queue:
            r = queue.popleft()
            for _, nb in self.neighbors(r):
                if nb.index not in dist and nb.type not in exclude:
                    dist[nb.index] = dist[r.index] + 1
                    queue.append(nb)
        return dist

    def visible(self, hidden_types=HIDDEN_TYPES) -> 'Floor':
        """The map after exploring every reachable room without finding hidden rooms: the same
        rooms minus secret and super secret rooms; generator depths are dropped."""
        rooms = [FloorRoom(r.index, r.x, r.y, r.shape, r.type, r.variant, r.subtype, r.layout_doors, 0, -1,
                           r.difficulty, r.name, file=r.file)
                 for r in self.rooms if r.type not in hidden_types]
        return Floor(rooms, self.stage, self.stage_type, self.curses, 0, self.start)

    def to_dict(self, with_hidden_info: bool = True) -> dict:
        out = dict(stage=self.stage, stage_type=self.stage_type, curses=self.curses, start=self.start,
                   rooms=[dict(index=r.index, x=r.x, y=r.y, shape=r.shape, type=r.type,
                               variant=r.variant, subtype=r.subtype, doors=r.doors,
                               layout_doors=r.layout_doors, difficulty=r.difficulty, name=r.name,
                               **({'depth': r.depth} if with_hidden_info else {}))
                          for r in self.rooms])
        if with_hidden_info:
            out['seed'] = self.seed
        return out

    def ascii(self) -> str:
        """Text map: S start, B boss, $ shop, T treasure, s secret, X super secret, # other."""
        sym = {1: '.', 2: '$', 4: 'T', 5: 'B', 6: 'm', 7: 's', 8: 'X', 9: 'a', 10: 'c', 11: 'C', 12: 'L',
               13: 'x', 18: 'b', 19: 'b', 20: 'v', 21: 'd'}
        rows = []
        for y in range(GRID):
            row = ''
            for x in range(GRID):
                c = index(x, y)
                r = self.room_at(c)
                row += ' ' if r is None else 'S' if c == self.start else sym.get(r.type, '#')
            rows.append(row.rstrip())
        while rows and not rows[-1]:
            rows.pop()
        while rows and not rows[0]:
            rows.pop(0)
        return '\n'.join(rows)

