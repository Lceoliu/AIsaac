"""Where are the secret and super secret rooms? Exact posteriors from the placement rules.

Secret room (LevelGenerator::GetNewSecretRoom 0x344D90, Level::build_secret_room_index_blacklist
0x337210; translated in levelgen.py / level.py):
- the blacklist is built from the rooms placed *before* the secret room, i.e. the boss room(s), the
  super secret room and the other special rooms. Normal rooms get their layouts afterwards and
  always have the door, so only special-room layouts matter;
- every empty, non-blacklisted cell draws Random(5) + 10; a neighbour whose shape has no door on
  that side (narrow rooms) removes the cell, cells with no neighbour are skipped, and 1 or 2
  neighbour cells lose 6 or 3 points;
- the highest score wins, ties are broken uniformly.
The random scores are the only unknown once the map, the special-room layouts and the super secret
room are known, so the posterior below is exact under the model "scores are i.i.d. uniform"
(the engine draws them from the generator's xorshift stream).

Super secret room: the deepest dead end left after the boss room, reshaped to a 1x1 layout. It
always touches exactly one room, a normal room. Its position is not determined by the visible map,
so we weight the candidate cells with a small table fitted on generated floors (see
tools/fit_super_secret.py) and marginalise the secret-room posterior over it.
"""
from __future__ import annotations

from math import comb

from .floor import Floor, FloorRoom
from .level import ROOM_BOSS, ROOM_DEFAULT, ROOM_SECRET, ROOM_SUPERSECRET
from .levelgen import GRID, ROOM_SIZE, TRAVEL, door_target, index

SCORE_BASE, SCORE_SPAN = 10, 5
OFFSETS = {1: -6, 2: -3}          # neighbour count -> score offset (3 or 4 neighbours: 0)
SPECIAL_BLOCKING_TYPES = (ROOM_BOSS, ROOM_SUPERSECRET, ROOM_SECRET)
ALL_DOORS = 0xFF


# ------------------------------------------------------------------------------------------ rules
def blacklist(rooms, stage: int, start: int = 0x54) -> set[int]:
    """Level::build_secret_room_index_blacklist over `rooms` (the rooms placed before the secret
    room). Uses the bounding-box cells of the engine, including its diagonal cells for narrow and
    L-shaped rooms."""
    out: set[int] = set()
    if stage == 11:
        for off in (1, -1, GRID, -GRID):
            if 0 <= start + off < GRID * GRID:
                out.add(start + off)
    for r in rooms:
        x, y = r.x, r.y
        wx, hy = ROOM_SIZE[r.shape]
        cells = [(x - 1, y), (x, y - 1), (x + wx, y), (x, y + hy), (x - 1, y + 1), (x + 1, y - 1),
                 (x + wx, y + 1), (x + 1, y + hy)]
        for slot in range(8):
            if r.shape == 4 and slot in (7, 5):
                continue
            if r.shape == 6 and slot in (6, 4):
                continue
            if r.shape == 1 and 4 <= slot <= 7:
                continue
            ci = index(*cells[slot])
            if ci >= 0 and (not (r.layout_doors >> slot) & 1 or r.type in SPECIAL_BLOCKING_TYPES):
                out.add(ci)
    return out


def candidate_offsets(floor: Floor, banned: set[int]) -> dict[int, int]:
    """Cells that can hold the secret room, with their score offset (0, -3 or -6)."""
    out = {}
    for i in range(GRID * GRID):
        if floor.grid[i] >= 0 or i in banned:
            continue
        x, y = i % GRID, i // GRID
        count, valid = 0, True
        for d, (dx, dy) in enumerate(TRAVEL):
            j = index(x + dx, y + dy)
            if j < 0 or floor.grid[j] < 0:
                continue
            nb = floor.room(floor.grid[j])
            if door_target(nb.x, nb.y, nb.shape, (d + 2) & 3, False) is None:
                valid = False
                break
            count += 1
        if valid and count:
            out[i] = OFFSETS.get(count, 0)
    return out


def win_probabilities(offsets: dict[int, int]) -> dict[int, float]:
    """P(cell is chosen) when each cell scores U + offset, U ~ uniform{10..14}, ties uniform.

    For a cell scoring s: P(win) = sum over tie counts k of P(k others tie at s, none higher)/(k+1)
    = integral_0^1 prod_j (P(S_j < s) + P(S_j = s) z) dz. Cells with equal offsets share a value.
    """
    if not offsets:
        return {}
    groups: dict[int, int] = {}
    for o in offsets.values():
        groups[o] = groups.get(o, 0) + 1

    def lt(o, s):
        return min(max(s - (SCORE_BASE + o), 0), SCORE_SPAN) / SCORE_SPAN

    def eq(o, s):
        return 1.0 / SCORE_SPAN if SCORE_BASE + o <= s < SCORE_BASE + o + SCORE_SPAN else 0.0

    value = {}
    for h in groups:
        total = 0.0
        for u in range(SCORE_SPAN):
            s = SCORE_BASE + h + u
            poly = [1.0]
            for o, n in groups.items():
                n -= o == h
                if n <= 0:
                    continue
                a, b = lt(o, s), eq(o, s)
                if b == 0.0:
                    if a == 0.0:
                        poly = [0.0]
                        break
                    poly = [c * a ** n for c in poly]
                    continue
                term = [comb(n, k) * a ** (n - k) * b ** k for k in range(n + 1)]
                new = [0.0] * (len(poly) + n)
                for i, c in enumerate(poly):
                    if c:
                        for k, t in enumerate(term):
                            new[i + k] += c * t
                poly = new
            total += sum(c / (k + 1) for k, c in enumerate(poly)) / SCORE_SPAN
        value[h] = total
    return {cell: value[o] for cell, o in offsets.items()}


# ------------------------------------------------------------------------------ super secret
def super_secret_candidates(floor: Floor) -> dict[int, dict]:
    """Empty cells that could hold the super secret room, with the features used to weight them.

    A candidate touches exactly one room cell, that room is a normal room whose shape (and layout,
    when known) has a door towards the cell. `delta` = (BFS depth of the cell) - (deepest visible
    dead end other than boss rooms and the start room).
    """
    dist = floor.distances()
    dead = [floor.room(i) for i in dist
            if floor.room(i).type != ROOM_BOSS and floor.room(i).grid_index != floor.start
            and len(floor.neighbors(floor.room(i))) == 1]
    deepest = max((dist[r.index] for r in dead), default=0)
    out = {}
    for i in range(GRID * GRID):
        if floor.grid[i] >= 0:
            continue
        x, y = i % GRID, i // GRID
        occupied = [floor.grid[j] for dx, dy in TRAVEL
                    if (j := index(x + dx, y + dy)) >= 0 and floor.grid[j] >= 0]
        if len(occupied) != 1:
            continue
        parent = floor.room(occupied[0])
        if parent.type != ROOM_DEFAULT or parent.index not in dist:
            continue
        slots = [s for s in range(8) if parent.slot_target(s) == i]
        if not any(parent.layout_doors >> s & 1 for s in slots):
            continue
        out[i] = dict(parent=parent.index, depth=dist[parent.index] + 1,
                      delta=dist[parent.index] + 1 - deepest,
                      parent_is_start=parent.grid_index == floor.start)
    return out


# Relative weights per delta, fitted by tools/fit_super_secret.py (conditional-logit MLE on 11,896
# floors of 1,500 generated runs, stages 1-8, seed 1). Keys are clipped to [-2, 3].
SUPER_SECRET_DELTA_WEIGHTS = {-2: 0.0082, -1: 0.0489, 0: 1.0, 1: 1.504, 2: 1.4456, 3: 0.9732}


def super_secret_prior(floor: Floor, weights: dict[int, float] | None = None) -> dict[int, float]:
    weights = weights or SUPER_SECRET_DELTA_WEIGHTS
    cands = super_secret_candidates(floor)
    raw = {c: weights[max(-2, min(3, f['delta']))] for c, f in cands.items()}
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()} if total > 0 else {}


# ------------------------------------------------------------------------------ posteriors
def _with_room(floor: Floor, room: FloorRoom) -> Floor:
    return Floor(floor.rooms + [room], floor.stage, floor.stage_type, floor.curses, floor.seed, floor.start)


def secret_given_super_secret(floor: Floor, super_secret: int | None) -> dict[int, float]:
    """Exact P(secret room cell) given the visible floor and the super secret cell (None = the
    floor has none, or it is already part of `floor`)."""
    if super_secret is not None and floor.grid[super_secret] < 0:
        ss = FloorRoom(max(r.index for r in floor.rooms) + 1, super_secret % GRID, super_secret // GRID, 1,
                       ROOM_SUPERSECRET, layout_doors=ALL_DOORS)
        floor = _with_room(floor, ss)
    placed_before = [r for r in floor.rooms if r.type not in (ROOM_DEFAULT, ROOM_SECRET)]
    offsets = candidate_offsets(floor, blacklist(placed_before, floor.stage, floor.start))
    return win_probabilities(offsets)


def secret_posterior(floor: Floor, super_secret: int | dict[int, float] | None = None) -> dict[int, float]:
    """P(secret room cell | visible floor). `super_secret`: its cell if found, a distribution over
    cells, or None to use super_secret_prior(floor). Special-room layouts come from
    FloorRoom.layout_doors (0xFF when the layout is unknown: no slot is assumed missing)."""
    if any(r.type == ROOM_SUPERSECRET for r in floor.rooms):
        return secret_given_super_secret(floor, None)
    if isinstance(super_secret, int):
        return secret_given_super_secret(floor, super_secret)
    prior = super_secret if super_secret is not None else super_secret_prior(floor)
    if not prior:
        return secret_given_super_secret(floor, None)
    out: dict[int, float] = {}
    for cell, w in prior.items():
        for c, p in secret_given_super_secret(floor, cell).items():
            out[c] = out.get(c, 0.0) + w * p
    total = sum(out.values())
    return {c: p / total for c, p in out.items()} if total > 0 else out


def super_secret_posterior(floor: Floor, secret: int | None = None) -> dict[int, float]:
    """P(super secret cell | visible floor [, secret room cell if found]). The secret room is never
    next to the super secret room (its neighbours are blacklisted); when the secret room is
    unknown, each super secret hypothesis is weighted by the prior only."""
    prior = super_secret_prior(floor)
    if secret is not None:
        prior = {c: w for c, w in prior.items()
                 if not any(index(c % GRID + dx, c // GRID + dy) == secret for dx, dy in TRAVEL)}
        total = sum(prior.values())
        prior = {c: w / total for c, w in prior.items()} if total > 0 else {}
    return prior


def bomb_order(posterior: dict[int, float]) -> list[tuple[int, float]]:
    """Cells to test, most likely first (ties by cell index)."""
    return sorted(posterior.items(), key=lambda kv: (-kv[1], kv[0]))


def _rank(posterior: dict[int, float], truth: int) -> tuple[int, int]:
    p_truth = posterior[truth]
    higher = sum(1 for p in posterior.values() if p > p_truth + 1e-12)
    ties = sum(1 for p in posterior.values() if abs(p - p_truth) <= 1e-12)
    return higher, ties


def hit_within(posterior: dict[int, float], truth: int, bombs: int) -> float:
    """P(the truth is among the first `bombs` cells tested), tied cells in random order."""
    if truth not in posterior:
        return 0.0
    higher, ties = _rank(posterior, truth)
    return min(max((bombs - higher) / ties, 0.0), 1.0)


def expected_bombs(posterior: dict[int, float], truth: int) -> float:
    """Bombs used when testing cells in posterior order until `truth`; tied cells are tested in
    random order, so the result is the mean position inside the tie group."""
    if truth not in posterior:
        return float('nan')
    higher, ties = _rank(posterior, truth)
    return higher + (ties + 1) / 2
