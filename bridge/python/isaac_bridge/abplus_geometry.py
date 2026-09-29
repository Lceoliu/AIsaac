"""Firing geometry of combat-v5 (user spec 2026-09-26): how far the player is from a position where
its tear would hit a target, which way to fire when it is there, and which moves close the gap.

Targets: the alive roster-lineage NPCs (bridge abp-0.2.3 flag 'lineage'); when none is visible,
every doors-blocking NPC ('blocking'). A target at (ex, ey) with collision radius r (Entity.Size)
has tolerance tau = r + TEAR_RADIUS. Its horizontal firing band is |y - ey| <= tau, x in
[x_lo, x_hi], at distance

    d_h = sqrt(dist(px, [x_lo, x_hi])^2 + max(0, |py - ey| - tau)^2)

from the player; the vertical band is symmetric (d_v, y in [y_lo, y_hi], |x - ex| <= tau). d_fire =
min over the targets of min(d_h, d_v), capped at D_FIRE_MAX. The band reaches the tear range R
(the player's 'range') from the target, and since 2026-09-26 (round 2, user decision) it stops at
the first grid entity that stops tears for good on the target's row or column (rocks that are
not rubble, metal blocks, locks, statues): a tear fired from beyond it hits the rock. Poop, TNT
and pits do not cut the band (tears destroy the first two and fly over pits). Line of sight away
from the target's row/column and invulnerable phases are ignored: combat-v5 uses d_fire as a
potential, so an imperfect estimate only bends the guidance.

fire_geometry() also gives the auxiliary head's labels: 'aim', the shoot action (1 up, 2 right,
3 down, 4 left, as the bridge's SHOOT table) that hits a target the player is already aligned
with, 0 when there is none; and 'approach', for each of the 9 moves (the bridge's MOVE table) 1 if
a MOVE_STEP step that way shortens d_fire. With no target visible (burrowed, or the room between
waves) d_fire holds its previous value and both labels are empty.

walk=True (the walking-distance potential, user decision 2026-09-26) makes d_fire the distance
the player has to walk to a firing position instead of the straight line: a multi-source
shortest path over the room's walkable cells (8 neighbours, no corner cutting past a blocked
cell) from every walkable cell a firing band passes through, each starting at the straight
distance from its centre to the band. The player's distance is the best of its own and the
adjacent cells (straight to their centre plus theirs), or the straight distance to the band from
inside a cell the band passes through. Along the shortest path every step shortens it, so it has
no local minimum in front of rocks (the straight line has: EXPERIMENTS.md C15). The approach label
then marks the moves onto walkable cells that shorten the walking distance.

trap_depth() (tier 6 of the aiming arena, C28) picks starts on which heading straight for the target
runs into a dead end: the depth is how much farther from the target one has to go back from where
the straight descent stops before a firing band can be reached.

blocked_moves() (tier 8, C30) marks the moves the terrain stops dead (into a wall, or diagonally into
an inner corner); with --block-moves they are masked in training and evaluation alike.
"""
import heapq
import math

TEAR_RADIUS = 8.16      # base Isaac tear collision radius (measured on AB+ tears, scale 1.02)
D_FIRE_MAX = 600.0      # px (15 cells)
CELL = 40.0             # px per grid cell
MOVE_STEP = 10.0        # px, probe displacement of the approach label
# abp_bridge.lua MOVE: 0 stop, 1 up, 2 up-right, 3 right, 4 down-right, 5 down, 6 down-left, 7 left, 8 up-left
MOVES = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
SHOOT_UP, SHOOT_RIGHT, SHOOT_DOWN, SHOOT_LEFT = 1, 2, 3, 4
# AB+ GridEntityType: 2 rock, 3 metal block, 4 tinted, 5 bomb rock, 6 alt rock, 11 lock, 21 statue,
# 22 super-secret rock; rocks in state 2 are rubble.
TEAR_STOPS = frozenset((2, 3, 4, 5, 6, 11, 21, 22))
ROCKS = frozenset((2, 4, 5, 6, 22))
RUBBLE = 2


def targets(obs):
    """(x, y, radius) of the alive lineage NPCs, else of the doors-blocking ones."""
    entities = obs['entities']
    chosen = [e for e in entities if e.get('lineage')]
    if not chosen:
        chosen = [e for e in entities if e.get('blocking')]
    return [(e['pos'][0], e['pos'][1], e['size']) for e in chosen]


def tear_stops(obs):
    """(cells, origin): the (col, row) cells whose grid entity stops tears, and cell (0, 0)'s centre."""
    terrain = obs.get('terrain') or {}
    cells = terrain.get('cells')
    if not cells:
        return frozenset(), (0.0, 0.0)
    width = obs.get('room', {}).get('gw') or terrain.get('width') or 15
    stops = set()
    for g in obs.get('grid') or ():
        index, gtype, state = g[0], g[1], g[3]
        if gtype in TEAR_STOPS and not (gtype in ROCKS and state == RUBBLE):
            stops.add((index % width, index // width))
    return frozenset(stops), (cells[0][1], cells[0][2])


def target_limits(target, reach, stops=frozenset(), origin=(0.0, 0.0)):
    """(x_lo, x_hi, y_lo, y_hi): the stretch of the target's row and column a tear fired from still
    crosses to reach it (the tear range, cut at the first tear-stopping cell on each side)."""
    ex, ey, _ = target
    x_lo, x_hi, y_lo, y_hi = ex - reach, ex + reach, ey - reach, ey + reach
    if stops:
        x0, y0 = origin
        col, row = round((ex - x0) / CELL), round((ey - y0) / CELL)
        span = int(reach // CELL) + 2
        edge = CELL / 2 + TEAR_RADIUS            # a tear must start clear of the stopping cell
        for step in range(1, span):
            if (col - step, row) in stops:
                x_lo = max(x_lo, x0 + CELL * (col - step) + edge)
                break
        for step in range(1, span):
            if (col + step, row) in stops:
                x_hi = min(x_hi, x0 + CELL * (col + step) - edge)
                break
        for step in range(1, span):
            if (col, row - step) in stops:
                y_lo = max(y_lo, y0 + CELL * (row - step) + edge)
                break
        for step in range(1, span):
            if (col, row + step) in stops:
                y_hi = min(y_hi, y0 + CELL * (row + step) - edge)
                break
    return x_lo, x_hi, y_lo, y_hi


def band_distances(px, py, target, limits):
    """(d_h, d_v): distances from the player to the target's horizontal and vertical firing bands."""
    ex, ey, radius = target
    x_lo, x_hi, y_lo, y_hi = limits
    tau = radius + TEAR_RADIUS
    return (math.hypot(max(0.0, x_lo - px, px - x_hi), max(0.0, abs(py - ey) - tau)),
            math.hypot(max(0.0, y_lo - py, py - y_hi), max(0.0, abs(px - ex) - tau)))


def fire_distance(px, py, found, limits):
    if not found:
        return None
    return min(D_FIRE_MAX, min(min(band_distances(px, py, t, lim)) for t, lim in zip(found, limits)))


def aim_label(px, py, found, limits):
    """Shoot action that hits an aligned target (in range, clear line), the nearest one; 0 when none."""
    best, label = None, 0
    for t, lim in zip(found, limits):
        d_h, d_v = band_distances(px, py, t, lim)
        if d_h > 0 and d_v > 0:
            continue
        ex, ey, _ = t
        dx, dy = ex - px, ey - py
        # Inside both bands (point blank) the larger offset decides the axis.
        horizontal = d_h == 0 and (d_v > 0 or abs(dx) >= abs(dy))
        direction = (SHOOT_RIGHT if dx > 0 else SHOOT_LEFT) if horizontal else (SHOOT_DOWN if dy > 0 else SHOOT_UP)
        distance = math.hypot(dx, dy)
        if best is None or distance < best:
            best, label = distance, direction
    return label


def approach_mask(px, py, found, limits, d_fire):
    mask = [0] * len(MOVES)
    if not found:
        return mask
    for m, (mx, my) in enumerate(MOVES):
        if m == 0:
            continue
        norm = math.hypot(mx, my)
        d = fire_distance(px + MOVE_STEP * mx / norm, py + MOVE_STEP * my / norm, found, limits)
        mask[m] = int(d < d_fire - 1e-6)
    return mask


SQRT2 = math.sqrt(2.0)
_WALK_CACHE = {}   # terrain key -> (origin, walkable cells, neighbour lists); a room's walls rarely change
# The last walking geometries: the observation (VisibleHistory) and the reward ask for the same step, in a duel for both
# sides' views (abplus_duel), so the last four are kept.
_WALK_MEMO = {}
WALK_MEMO_SIZE = 4


def walk_grid(obs):
    """(origin, walkable {(col, row)}, neighbours {(col, row): [(cell, cost px)]}) of the room, or None."""
    terrain = obs.get('terrain') or {}
    cells = terrain.get('cells')
    if not cells:
        return None
    width = terrain.get('width') or obs.get('room', {}).get('gw') or 15
    key = (terrain.get('version'), width, tuple((c[0], bool(c[4] and c[5])) for c in cells))
    grid = _WALK_CACHE.get(key)
    if grid is None:
        walkable = {(c[0] % width, c[0] // width) for c in cells if c[4] and c[5]}
        neighbours = {}
        for col, row in walkable:
            out = []
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    n = (col + dc, row + dr)
                    if (dc or dr) and n in walkable:
                        if dc and dr and ((col + dc, row) not in walkable or (col, row + dr) not in walkable):
                            continue      # no corner cutting past a blocked cell
                        out.append((n, CELL * (SQRT2 if dc and dr else 1.0)))
            neighbours[(col, row)] = out
        grid = ((cells[0][1], cells[0][2]), walkable, neighbours)
        if len(_WALK_CACHE) > 64:
            _WALK_CACHE.clear()
        _WALK_CACHE[key] = grid
    return grid


def walk_field(grid, found, limits):
    """Walking distance (px) from each walkable cell centre to the nearest firing position."""
    (x0, y0), walkable, neighbours = grid
    dist = {}
    for t, lim in zip(found, limits):
        ex, ey, radius = t
        near = radius + TEAR_RADIUS + CELL / 2      # rows / columns a band of this target can cross
        for col, row in walkable:
            cx, cy = x0 + CELL * col, y0 + CELL * row
            if abs(cy - ey) > near and abs(cx - ex) > near:
                continue
            entry = min(band_distances(cx, cy, t, lim))
            if entry <= CELL / 2 and entry < dist.get((col, row), math.inf):
                dist[(col, row)] = entry            # a firing band passes through this cell
    heap = [(d, cell) for cell, d in dist.items()]
    heapq.heapify(heap)
    while heap:
        d, cell = heapq.heappop(heap)
        if d > dist[cell]:
            continue
        for n, cost in neighbours[cell]:
            nd = d + cost
            if nd < dist.get(n, math.inf):
                dist[n] = nd
                heapq.heappush(heap, (nd, n))
    return dist


def walk_distance(px, py, grid, field, found, limits):
    """Walking distance (px) from the player to the nearest firing position, at most D_FIRE_MAX."""
    (x0, y0), walkable, neighbours = grid
    here = (round((px - x0) / CELL), round((py - y0) / CELL))
    best = math.inf
    if here in field and field[here] <= CELL / 2:
        best = fire_distance(px, py, found, limits)        # the band crosses this cell: go straight
    if here in walkable:
        candidates = [here] + [n for n, _ in neighbours[here]]
    else:   # the position rounds onto a blocked cell (at its edge): any adjacent cell
        candidates = [(here[0] + dc, here[1] + dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1)]
    for cell in candidates:
        if cell in field:
            best = min(best, math.hypot(px - (x0 + CELL * cell[0]), py - (y0 + CELL * cell[1])) + field[cell])
    return min(D_FIRE_MAX, best)


BLOCK_EPS = 1.0   # px beyond the collision circle where blocked_moves looks for a wall


def blocked_moves(obs):
    """[bool] * 9 in MOVES order: the moves the terrain stops dead (tier 8, EXPERIMENTS.md C30). An orthogonal
    move is blocked when the points BLOCK_EPS px beyond the player's collision circle (radius players[0].size) in
    its direction, at the middle and +-0.7 radius across, all lie on cells that are not walkable; a diagonal
    move when both its orthogonal parts are (in an inner corner; along a straight wall it slides). Stay never
    is. Against the measured progress along the move on 63k replay steps: precision 0.99, recall ~0.79 (C29)."""
    out = [False] * len(MOVES)
    grid = walk_grid(obs)
    if grid is None:
        return out
    (x0, y0), walkable, _ = grid
    p = obs['players'][0]
    px, py = p['pos']
    r = float(p.get('size', 10.0))

    def wall(sx, sy):
        for side in (0.0, -0.7, 0.7):
            x = px + sx * (r + BLOCK_EPS) + (0.0 if sx else side * r)
            y = py + sy * (r + BLOCK_EPS) + (0.0 if sy else side * r)
            if (round((x - x0) / CELL), round((y - y0) / CELL)) in walkable:
                return False
        return True

    ortho = {d: wall(*d) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    for m, (dx, dy) in enumerate(MOVES):
        if m:
            out[m] = (ortho[(dx, 0)] and ortho[(0, dy)]) if dx and dy else ortho[(dx, dy)]
    return out


def straight_descent(grid, start, value):
    """The cell where heading straight downhill in value (a function of a cell, px) over the walkable cells
    from cell start stops: each step goes to the neighbour of least value while that is lower."""
    _, _, neighbours = grid
    cell = start
    while True:
        options = [(value(n), n) for n, _ in neighbours.get(cell, ())]
        if not options:
            return cell
        v, n = min(options)
        if v >= value(cell) - 1e-6:
            return cell
        cell = n


def escape_rise(grid, field, start, value):
    """The least rise of value above value(start) (px) that a walking path from cell start to a cell a
    firing band crosses (field <= CELL / 2) must make, by a bottleneck search; inf when none is reachable."""
    _, _, neighbours = grid
    goal = {c for c, d in field.items() if d <= CELL / 2}
    base = value(start)
    best = {start: base}
    heap = [(base, start)]
    while heap:
        b, cell = heapq.heappop(heap)
        if b > best[cell]:
            continue
        if cell in goal:
            return b - base
        for n, _ in neighbours.get(cell, ()):
            nb = max(b, value(n))
            if nb < best.get(n, math.inf):
                best[n] = nb
                heapq.heappush(heap, (nb, n))
    return math.inf


def trap_depth(grid, field, start, target):
    """How deep the dead end is that heading straight for the target (x, y, radius) from cell start runs
    into (tier 6, EXPERIMENTS.md C28): 0 when walking downhill in the straight distance to the target ends
    on a cell a firing band crosses, else how much farther from the target the player must first go back
    to reach such a cell."""
    (x0, y0), _, _ = grid
    ex, ey = target[0], target[1]

    def distance(cell):
        return math.hypot(x0 + CELL * cell[0] - ex, y0 + CELL * cell[1] - ey)

    stop = straight_descent(grid, start, distance)
    if field.get(stop, math.inf) <= CELL / 2:
        return 0.0
    return escape_rise(grid, field, stop, distance)


def fire_geometry(obs, previous=None, walk=False):
    """dict(d_fire, aim, approach, has_target) of an observation; previous: the last d_fire;
    walk: d_fire is the walking distance to a firing position (see the module docstring)."""
    p = obs['players'][0]
    px, py = p['pos']
    reach = float(p['range'])
    found = targets(obs)
    if not found:
        held = D_FIRE_MAX if previous is None else previous
        return dict(d_fire=held, aim=0, approach=[0] * len(MOVES), has_target=False)
    stops, origin = tear_stops(obs)
    limits = [target_limits(t, reach, stops, origin) for t in found]
    grid = walk_grid(obs) if walk else None
    if grid is None:
        d_fire = fire_distance(px, py, found, limits)
        return dict(d_fire=d_fire, aim=aim_label(px, py, found, limits),
                    approach=approach_mask(px, py, found, limits, d_fire), has_target=True)
    key = (obs.get('logic_frames'), px, py, tuple(found), id(grid))
    memo = _WALK_MEMO.get(key)
    if memo is not None:
        return dict(memo, approach=list(memo['approach']))
    field = walk_field(grid, found, limits)
    d_fire = walk_distance(px, py, grid, field, found, limits)
    (x0, y0), walkable, _ = grid
    approach = [0] * len(MOVES)
    for m, (mx, my) in enumerate(MOVES):
        if m == 0:
            continue
        norm = math.hypot(mx, my)
        qx, qy = px + MOVE_STEP * mx / norm, py + MOVE_STEP * my / norm
        if (round((qx - x0) / CELL), round((qy - y0) / CELL)) in walkable:
            approach[m] = int(walk_distance(qx, qy, grid, field, found, limits) < d_fire - 1e-6)
    result = dict(d_fire=d_fire, aim=aim_label(px, py, found, limits), approach=approach, has_target=True)
    if len(_WALK_MEMO) >= WALK_MEMO_SIZE:
        _WALK_MEMO.pop(next(iter(_WALK_MEMO)))       # the oldest (insertion order)
    _WALK_MEMO[key] = dict(result, approach=list(approach))
    return result
