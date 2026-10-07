"""Navigation geometry of the goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md, M0): the walking distance to a goal
(d_geo) and GOTO goal sampling on a room's grid.

goal_field(obs, target) is a single-source walking-distance field from the target's cell over abplus_geometry.walk_grid (8
neighbours, no corner cutting past a blocked cell), with no cap, spikes and hazard cells priced HAZARD_COST px extra to
enter, and fire places (entity type 33, which block the player) as blocked cells. For a door goal the target is the walkable
cell just inside the door (a door cell itself is never walkable). goal_distance(px, py, nav) is the distance from a position:
the nearest reachable cell among the position's cell and its neighbours plus the straight line to that cell's centre, or None
when none is reachable (the caller keeps its last value; an exception would restart the worker's game process).

The field depends on the terrain and the fire places: goal_field is recomputed whenever they change (a bomb, a tear through
poop or a fire), and the reward prices a step with the old state's potential on the old map and the new state's on the new
one (user decision 2026-09-30).

sample_goal draws a GOTO_POSITION target among the cell centres of the player's connected walkable component that are no
hazard and at least MIN_GOAL_CELLS walking cells away, stratified (user plan): 'straight' (the walk is the straight line),
'detour' (the walk is at least one cell longer than the straight line) or 'dead_end' (going straight at the target first
enters a dead end at least DEAD_END_PX deep); a stratum with no candidate falls back to any candidate. With weights (a
group's goal_strata, C43: training only) the stratum is drawn by weight among the strata that have a candidate.
"""
import heapq
import math

import numpy as np

from .abplus_geometry import CELL, walk_grid

HAZARD_COST = 160.0    # px added for entering a spike / hazard cell (the walk goes around unless it saves 4 cells)
FIRE_TYPE = 33         # EntityType.ENTITY_FIREPLACE
PICKUP_TYPE, PEDESTAL_VARIANT = 5, 100   # EntityType.ENTITY_PICKUP, PickupVariant.PICKUP_COLLECTIBLE
MIN_GOAL_CELLS = 2
DEAD_END_PX = 30.0
STRATA = ('straight', 'detour', 'dead_end')


def _cell(x, y, origin):
    return round((x - origin[0]) / CELL), round((y - origin[1]) / CELL)


def blocked_cells(obs, origin):
    """Cells a fire place or an item pedestal stands on (both block the player; the grid does not know them; a pedestal
    is a pickup of variant PEDESTAL_VARIANT, solid even once its item is taken: the floor runner's boss and treasure
    rooms)."""
    out = set()
    for e in obs.get('entities', ()):
        if e.get('type') == FIRE_TYPE or (e.get('type') == PICKUP_TYPE and e.get('variant') == PEDESTAL_VARIANT):
            out.add(_cell(e['pos'][0], e['pos'][1], origin))
    return out


def hazard_cells(obs):
    """(col, row) of the terrain cells marked hazard (spikes, damaging effects)."""
    terrain = obs['terrain']
    width = terrain['width']
    return {(c[0] % width, c[0] // width) for c in terrain['cells'] if c[7]}


def nav_key(obs):
    """What the field depends on besides the target: the walkable cells, the hazards and the fire places."""
    terrain = obs['terrain']
    cells = tuple((c[0], bool(c[4] and c[5]), bool(c[7])) for c in terrain['cells'])
    fires = tuple(sorted((round(e['pos'][0]), round(e['pos'][1])) for e in obs.get('entities', ())
                         if e.get('type') == FIRE_TYPE or (e.get('type') == PICKUP_TYPE
                                                           and e.get('variant') == PEDESTAL_VARIANT)))
    return cells, fires


def goal_field(obs, target):
    """dict(origin, walkable, neighbours, field, key) for the target (x, y) px, or None without a grid."""
    grid = walk_grid(obs)
    if grid is None:
        return None
    origin, walkable, neighbours = grid
    blocked = blocked_cells(obs, origin)
    hazards = hazard_cells(obs)
    source = _cell(target[0], target[1], origin)
    field = {}
    if source in walkable and source not in blocked:
        field[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, cell = heapq.heappop(heap)
            if d > field[cell]:
                continue
            for n, cost in neighbours[cell]:
                if n in blocked:
                    continue
                nd = d + cost + (HAZARD_COST if n in hazards else 0.0)
                if nd < field.get(n, math.inf):
                    field[n] = nd
                    heapq.heappush(heap, (nd, n))
    return dict(origin=origin, walkable=walkable, neighbours=neighbours, field=field, key=nav_key(obs),
                target=tuple(float(v) for v in target), blocked=blocked, hazards=hazards)


def goal_distance(px, py, nav):
    """Walking distance (px) from (px, py) to the goal of nav (goal_field), or None when no nearby cell reaches it."""
    origin, field, walkable, neighbours = nav['origin'], nav['field'], nav['walkable'], nav['neighbours']
    here = _cell(px, py, origin)
    if here in walkable:
        candidates = [here] + [n for n, _ in neighbours[here]]
    else:
        candidates = [(here[0] + dc, here[1] + dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1)]
    best = None
    for cell in candidates:
        if cell in field:
            d = math.hypot(px - (origin[0] + CELL * cell[0]), py - (origin[1] + CELL * cell[1])) + field[cell]
            if best is None or d < best:
                best = d
    return best


def door_target(obs, slot):
    """(x, y) of the walkable cell centre just inside door `slot` (the door goal's target), or None."""
    door = next((d for d in obs['doors'] if d['slot'] == slot), None)
    grid = walk_grid(obs)
    if door is None or grid is None:
        return None
    origin, walkable, _ = grid
    dcol, drow = _cell(door['pos'][0], door['pos'][1], origin)
    best = None
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            cell = (dcol + dc, drow + dr)
            if cell in walkable and (dc == 0 or dr == 0) and (dc or dr):
                best = cell
    if best is None:
        return None
    return origin[0] + CELL * best[0], origin[1] + CELL * best[1]


def _straight_dead_end(nav, start, target):
    """How much heading straight from start (x, y) at target makes the walk worse before the line is blocked: the largest
    rise of the walking distance to the goal (goal_distance) over its lowest value so far along the line (10 px steps).
    A wall to walk back from gives a large rise; a rock to step round gives almost none."""
    (sx, sy), (tx, ty) = start, target
    length = math.hypot(tx - sx, ty - sy)
    low, worst = math.inf, 0.0
    for i in range(1, int(length // 10) + 1):
        x, y = sx + (tx - sx) * i * 10 / length, sy + (ty - sy) * i * 10 / length
        cell = _cell(x, y, nav['origin'])
        if cell not in nav['walkable'] or cell in nav['blocked']:
            break
        d = goal_distance(x, y, nav)
        if d is None:
            continue
        low = min(low, d)
        worst = max(worst, d - low)
    return worst


def sample_goal(obs, rng, stratum=None, weights=None):
    """(target (x, y), stratum, walking distance px) for a GOTO_POSITION goal from the player's position, or None when the
    component has no candidate. weights: {stratum: weight} (see the module docstring); None keeps the uniform draw with
    its fallback, and its random draws, unchanged."""
    grid = walk_grid(obs)
    if grid is None:
        return None
    origin, walkable, _ = grid
    px, py = obs['players'][0]['pos']
    here_nav = goal_field(obs, (px, py))   # distances from the player's cell = to each candidate (symmetric costs)
    if here_nav is None or not here_nav['field']:
        return None
    hazards = here_nav['hazards']
    candidates = []
    for cell, d in here_nav['field'].items():
        if cell in hazards or d < MIN_GOAL_CELLS * CELL:
            continue
        x, y = origin[0] + CELL * cell[0], origin[1] + CELL * cell[1]
        candidates.append((cell, (x, y), d))
    if not candidates:
        return None
    typed = []
    for cell, xy, d in candidates:
        straight = math.hypot(xy[0] - px, xy[1] - py)
        if d - straight < CELL:
            kind = 'straight'
        else:
            kind = 'dead_end' if _straight_dead_end(goal_field(obs, xy), (px, py), xy) >= DEAD_END_PX else 'detour'
        typed.append((xy, kind, d))
    if stratum is None and weights:
        present = [s for s in STRATA if weights.get(s, 0) > 0 and any(c[1] == s for c in typed)]
        if present:
            p = np.array([float(weights[s]) for s in present])
            stratum = present[min(int(np.searchsorted(np.cumsum(p / p.sum()), rng.random(), side='right')), len(present) - 1)]
    stratum = stratum or STRATA[int(rng.integers(len(STRATA)))]
    pool = [c for c in typed if c[1] == stratum] or typed
    return pool[int(rng.integers(len(pool)))]


DOOR_ALIGN_PX = 8.0   # a scripted door approach corrects a larger lateral offset from the door's axis diagonally
MOVE_CODES = (3, 4, 5, 6, 7, 8, 1, 2)   # bridge move code by 45-degree sector from +x, clockwise (screen y down)


def move_code(dx, dy):
    """The bridge's 8-way move code nearest the direction (dx, dy); 0 for no direction."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0
    return MOVE_CODES[round(math.degrees(math.atan2(dy, dx)) / 45) % 8]


def _door_push(px, py, nav):
    """A door goal's last stretch, or None away from it: once the player is at the inside cell's centre or past it towards
    the door, always push outwards (never back into the room), correcting a lateral offset of more than DOOR_ALIGN_PX with a
    diagonal step (forward progress continues; a pure sideways step overshoots and oscillates, as measured)."""
    ex, ey = nav['exit']
    tx, ty = nav['target']
    nx, ny = (ex > tx) - (ex < tx), (ey > ty) - (ey < ty)   # outward unit along the door's axis
    along = (px - tx) * nx + (py - ty) * ny
    off = (py - ey) if nx else (px - ex)
    if along < -CELL / 4 and math.hypot(px - tx, py - ty) > CELL / 4:
        return None
    lx, ly = ((0, -(off > 0) + (off < 0)) if nx else (-(off > 0) + (off < 0), 0)) if abs(off) > DOOR_ALIGN_PX else (0, 0)
    return move_code(nx + lx, ny + ly)


def descent_move(px, py, nav):
    """Scripted navigation (the GOTO reference and the scripted chain's navigator): head for the goal itself from its cell,
    else for the centre of the reachable neighbouring cell (or the player's own cell) that minimises the straight line to
    it plus its walking distance; returns the move code (0 when nothing reachable is near)."""
    origin, field, walkable, neighbours = nav['origin'], nav['field'], nav['walkable'], nav['neighbours']
    here = _cell(px, py, origin)
    if nav.get('exit') is not None:
        push = _door_push(px, py, nav)
        if push is not None:
            return push
    if field.get(here) == 0.0:
        tx, ty = nav['target']
        return move_code(tx - px, ty - py)
    candidates = ([here] if here in field else []) + [n for n, _ in neighbours.get(here, ()) if n in field]
    if not candidates:
        candidates = [(here[0] + dc, here[1] + dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1)
                      if (here[0] + dc, here[1] + dr) in field]
    if not candidates:
        return 0
    def cost(c):
        cx, cy = origin[0] + CELL * c[0], origin[1] + CELL * c[1]
        return math.hypot(cx - px, cy - py) + field[c]
    best = min(candidates, key=cost)
    if best == here:   # at the cell centre's side already on the best path: go for the cheapest neighbour
        others = [n for n, _ in neighbours.get(here, ()) if n in field]
        if others:
            best = min(others, key=lambda c: field[c])
    cx, cy = origin[0] + CELL * best[0], origin[1] + CELL * best[1]
    return move_code(cx - px, cy - py)
