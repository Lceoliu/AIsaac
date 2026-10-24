"""Rule planner v0 on the abstract floor: explore, bomb for hidden rooms, then the boss.

Visibility model: a room is known once the player has stood in it or in a room with a door to it
(the minimap shows neighbours); secret and super secret rooms stay hidden until their wall is
bombed. Combat is not modelled: entering a room clears it. Locks are parameters, not engine facts
(`locked_types`: room types that cost a key; the default is empty until the rules are checked in
the decompile).

Policy (MACRO_PLANNING.md §4.1/§4.2):
1. Repeatedly walk to the nearest known unvisited room (BFS over visited/known rooms; ties by grid
   index), never entering the boss room and skipping `avoid_types`.
2. Once nothing is left, spend bombs while bombs > reserve: bomb the most likely secret cell
   (secret.secret_posterior on the explored map), then, when the secret room is found or the
   budget allows, the most likely super secret cell.
3. Walk to the boss room.
Every door crossing counts as one transition.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .floor import Floor, FloorRoom
from .level import ROOM_BOSS, ROOM_SECRET, ROOM_SUPERSECRET
from .levelgen import GRID, TRAVEL, index
from .secret import secret_posterior, super_secret_posterior


@dataclass
class PlanResult:
    transitions: int = 0
    rooms_visited: int = 0
    rooms_total: int = 0
    bombs_used: int = 0
    found_secret: bool = False
    found_super_secret: bool = False
    has_secret: bool = False
    has_super_secret: bool = False
    reached_boss: bool = False
    trace: list = field(default_factory=list)     # ('move', cell) / ('bomb', cell, hit)


class Explorer:
    def __init__(self, floor: Floor, bombs: int = 1, keys: int = 0, bomb_reserve: int = 0,
                 locked_types=(), avoid_types=(), search_super_secret: bool = True):
        self.floor = floor
        self.bombs, self.keys, self.reserve = bombs, keys, bomb_reserve
        self.locked, self.avoid = set(locked_types), set(avoid_types)
        self.search_ss = search_super_secret
        self.hidden = {r.index for r in floor.rooms if r.type in (ROOM_SECRET, ROOM_SUPERSECRET)}
        self.revealed: set[int] = set()        # hidden rooms opened by bombs
        self.visited: set[int] = set()
        self.known: set[int] = set()
        self.cur = floor.room_at(floor.start)
        self.res = PlanResult(rooms_total=len(floor.rooms) - len(self.hidden),
                              has_secret=any(r.type == ROOM_SECRET for r in floor.rooms),
                              has_super_secret=any(r.type == ROOM_SUPERSECRET for r in floor.rooms))
        self._enter(self.cur)

    # ------------------------------------------------------------------ movement
    def _passable(self, room: FloorRoom) -> bool:
        return room.index not in self.hidden or room.index in self.revealed

    def _enter(self, room: FloorRoom) -> None:
        self.cur = room
        if room.index not in self.visited:
            self.visited.add(room.index)
            self.res.rooms_visited += room.index not in self.hidden
        self.known.add(room.index)
        for _, nb in self.floor.neighbors(room):
            if self._passable(nb):
                self.known.add(nb.index)

    def _path(self, goal) -> list[FloorRoom] | None:
        """Shortest path (rooms after the current one) to the first room satisfying `goal`,
        moving only through visited rooms; ties by grid index."""
        prev = {self.cur.index: None}
        queue = deque([self.cur])
        while queue:
            r = queue.popleft()
            if r is not self.cur and goal(r):
                path = []
                while r is not self.cur:
                    path.append(r)
                    r = self.floor.room(prev[r.index])
                return path[::-1]
            if r is not self.cur and r.index not in self.visited:
                continue
            for _, nb in sorted(self.floor.neighbors(r), key=lambda sn: sn[1].grid_index):
                if nb.index in prev or not self._passable(nb) or nb.index not in self.known:
                    continue
                if nb.type in self.locked and nb.index not in self.visited and self.keys <= 0:
                    continue
                prev[nb.index] = r.index
                queue.append(nb)
        return None

    def _walk(self, path: list[FloorRoom]) -> None:
        for room in path:
            if room.type in self.locked and room.index not in self.visited:
                self.keys -= 1
            self.res.transitions += 1
            self.res.trace.append(('move', room.safe_grid_index))
            self._enter(room)

    # ------------------------------------------------------------------ phases
    def explore(self) -> None:
        while True:
            path = self._path(lambda r: r.index not in self.visited and r.type != ROOM_BOSS
                              and r.type not in self.avoid)
            if path is None:
                return
            self._walk(path)

    def explored_map(self) -> Floor:
        """What the player knows: every known room, hidden ones only once revealed."""
        rooms = [r for r in self.floor.rooms if r.index in self.known]
        return Floor(rooms, self.floor.stage, self.floor.stage_type, self.floor.curses, 0, self.floor.start)

    def _bomb(self, cell: int) -> bool:
        """Walk next to `cell`, bomb the wall; True when a hidden room is behind it."""
        adjacent = {self.floor.grid[j] for dx, dy in TRAVEL
                    if (j := index(cell % GRID + dx, cell // GRID + dy)) >= 0 and self.floor.grid[j] >= 0}
        adjacent = {i for i in adjacent if i in self.visited}
        path = self._path(lambda r: r.index in adjacent) if self.cur.index not in adjacent else []
        if path is None:
            return False
        self._walk(path)
        self.bombs -= 1
        self.res.bombs_used += 1
        target = self.floor.room_at(cell)
        hit = target is not None and target.index in self.hidden and target.index not in self.revealed
        self.res.trace.append(('bomb', cell, hit))
        if hit:
            self.revealed.add(target.index)
            self.known.add(target.index)
            self.res.found_secret |= target.type == ROOM_SECRET
            self.res.found_super_secret |= target.type == ROOM_SUPERSECRET
            self._walk([target])
        return hit

    def search_hidden(self) -> None:
        tested: set[int] = set()
        while self.bombs > self.reserve:
            known = self.explored_map()
            found_s = [r.safe_grid_index for r in known.rooms if r.type == ROOM_SECRET]
            found_ss = [r.safe_grid_index for r in known.rooms if r.type == ROOM_SUPERSECRET]
            if not found_s:
                post = secret_posterior(known, found_ss[0] if found_ss else None)
            elif self.search_ss and not found_ss:
                post = super_secret_posterior(known, found_s[0])
            else:
                return
            post = {c: p for c, p in post.items() if c not in tested and p > 0}
            if not post:
                return
            cell = max(post, key=lambda c: (post[c], -c))
            tested.add(cell)
            self._bomb(cell)

    def go_to_boss(self) -> None:
        path = self._path(lambda r: r.type == ROOM_BOSS)
        if path is not None:
            self._walk(path)
            self.res.reached_boss = True

    def run(self) -> PlanResult:
        self.explore()
        self.search_hidden()
        self.go_to_boss()
        return self.res
