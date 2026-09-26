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
"""
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


def fire_geometry(obs, previous=None):
    """dict(d_fire, aim, approach, has_target) of an observation; previous: the last d_fire."""
    p = obs['players'][0]
    px, py = p['pos']
    reach = float(p['range'])
    found = targets(obs)
    if not found:
        held = D_FIRE_MAX if previous is None else previous
        return dict(d_fire=held, aim=0, approach=[0] * len(MOVES), has_target=False)
    stops, origin = tear_stops(obs)
    limits = [target_limits(t, reach, stops, origin) for t in found]
    d_fire = fire_distance(px, py, found, limits)
    return dict(d_fire=d_fire, aim=aim_label(px, py, found, limits),
                approach=approach_mask(px, py, found, limits, d_fire), has_target=True)
