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

Layout evidence (LayoutEvidence / hidden_posterior): normal rooms get their layouts after both
hidden rooms are placed, from RoomConfig::GetRandomRoom with the room's connections as required
doors, and that call prefers layouts whose door slots equal the required doors exactly (factor
9.99). A normal room whose layout has a door slot on a wall with no visible door is therefore
evidence of a hidden room behind that wall, and a layout without the slot rules the cell out. In
AB+ no two normal layouts share their visible content but differ in door slots, so a player who
recognises the room knows its slots.
"""
from __future__ import annotations

from collections import defaultdict
from math import comb

from .floor import Floor, FloorRoom
from .level import ROOM_BOSS, ROOM_DEFAULT, ROOM_SECRET, ROOM_SUPERSECRET
from .levelgen import GRID, ROOM_SIZE, TRAVEL, door_target, index
from .roomconfig import EXACT_DOORS_FACTOR, SHAPE_ANY, stage_id

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
    boss_depth = max((dist[r.index] for r in floor.rooms if r.type == ROOM_BOSS and r.index in dist), default=0)
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
                      boss_delta=dist[parent.index] + 1 - boss_depth,
                      parent_is_start=parent.grid_index == floor.start)
    return out


# Relative weights per delta, fitted by tools/fit_super_secret.py (conditional-logit MLE on 11,896
# floors of 1,500 generated runs, stages 1-8, seed 1). Keys are clipped to [-2, 3].
SUPER_SECRET_DELTA_WEIGHTS = {-2: 0.0126, -1: 0.0466, 0: 1.0, 1: 1.6891, 2: 1.1624, 3: 0.4776}
# Depth relative to the boss room, clipped to [-3, 1] (the boss takes the deepest dead end first).
SUPER_SECRET_BOSS_WEIGHTS = {-3: 0.2968, -2: 0.9324, -1: 1.4996, 0: 1.0, 1: 0.1865}


def super_secret_prior(floor: Floor, weights: dict[int, float] | None = None,
                       boss_weights: dict[int, float] | None = None) -> dict[int, float]:
    weights = weights or SUPER_SECRET_DELTA_WEIGHTS
    boss_weights = boss_weights or SUPER_SECRET_BOSS_WEIGHTS
    cands = super_secret_candidates(floor)
    raw = {c: weights[max(-2, min(3, f['delta']))] * boss_weights[max(-3, min(1, f['boss_delta']))]
           for c, f in cands.items()}
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


# ------------------------------------------------------------------------------ layout evidence
class LayoutEvidence:
    """P(layout | required doors) for the normal rooms of one floor kind, from GetRandomRoom's rule
    (float32 details and the in-floor weight decay are ignored) and the stage's room pool, filtered
    like Level::generate_dungeon (normal-mode difficulty ranges, 1-15 fallback)."""

    def __init__(self, room_config, stage: int, stage_type: int, hard: bool = False):
        sid = stage_id(stage, stage_type)
        if hard:
            max_d, min_d = (15, 10) if stage < 9 and stage % 2 == 0 else (5, 5)
        else:
            max_d, min_d = (10, 5) if stage < 9 and stage % 2 == 0 else (5, 1)
        pool = room_config.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, min_d, max_d, 0, -1)
        if len(pool) < 20:
            pool = room_config.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, 1, 15, 0, -1)
        min_variant = int(stage == 11)
        self.by_shape: dict[int, list] = defaultdict(list)
        self.layout: dict[tuple, tuple] = {}
        for r in pool:
            if r.variant >= min_variant:
                self.by_shape[r.shape].append((r.doors, float(r.initial_weight)))
                self.layout[(r.shape, r.variant)] = (r.doors, float(r.initial_weight))
        self._totals: dict = {}
        self.factor = float(EXACT_DOORS_FACTOR)

    def known(self, room: FloorRoom) -> bool:
        return (room.shape, room.variant) in self.layout

    def prob(self, room: FloorRoom, required: int) -> float:
        doors, weight = self.layout[(room.shape, room.variant)]
        if required & doors != required:
            return 0.0
        key = (room.shape, required)
        if key not in self._totals:
            total = exact = 0.0
            for d, w in self.by_shape[room.shape]:
                if required & d == required:
                    total += w
                    if d == required:
                        exact += w
            self._totals[key] = (total, exact)
        total, exact = self._totals[key]
        p_exact = min(1.0, self.factor * exact / total) if exact > 0 else 0.0
        p = (1.0 - p_exact) * weight / total
        if doors == required:
            p += p_exact * weight / exact
        return p


def _facing_slots(floor: Floor) -> dict[int, list]:
    """cell -> [(normal room, slot bits)] for every empty cell that a normal room (not the start
    room, whose layout is fixed) has a door target in."""
    out: dict[int, dict] = defaultdict(dict)
    for r in floor.rooms:
        if r.type != ROOM_DEFAULT or r.grid_index == floor.start:
            continue
        for slot in range(8):
            t = r.slot_target(slot)
            if t >= 0 and floor.grid[t] < 0:
                out[t][r.index] = out[t].get(r.index, 0) | (1 << slot)
    return {c: [(floor.room(i), bits) for i, bits in m.items()] for c, m in out.items()}


def hidden_posterior(floor: Floor, evidence: LayoutEvidence | None = None,
                     ss_prior: dict[int, float] | None = None,
                     secret_cell: int | None = None) -> tuple[dict[int, float], dict[int, float]]:
    """Joint posterior of the secret room cell c and the super secret room cell h given the visible
    floor, returned as the two marginals (P(secret at c), P(super secret at h)).

    P(c, h) is proportional to prior(h) * P_rules(c | h) * the product over normal rooms R next to
    c or h of P(layout_R | visible doors_R + slots to c and h) / P(layout_R | visible doors_R).
    Without `evidence` this is secret_posterior / super_secret_prior. `secret_cell` conditions on a
    secret room already found there (it must not be part of `floor`).
    """
    prior = ss_prior if ss_prior is not None else super_secret_prior(floor)
    if not prior:
        prior = {None: 1.0}
    facing = _facing_slots(floor) if evidence is not None else {}
    base: dict[int, float] = {}

    def layout_factor(cells) -> float:
        extra: dict[int, list] = {}
        for cell in cells:
            for room, bits in facing.get(cell, ()):
                if evidence.known(room):
                    extra.setdefault(room.index, [room, 0])[1] |= bits
        f = 1.0
        for room, bits in extra.values():
            if room.index not in base:
                base[room.index] = evidence.prob(room, room.doors)
            if base[room.index] > 0:
                f *= evidence.prob(room, room.doors | bits) / base[room.index]
            if f == 0.0:
                break
        return f

    joint: dict[tuple, float] = {}
    for h, ph in prior.items():
        rule = secret_given_super_secret(floor, h)
        for c, pc in rule.items():
            if secret_cell is not None and c != secret_cell:
                continue
            w = ph * pc
            if w > 0 and evidence is not None:
                w *= layout_factor([c] if h is None else [c, h])
            if w > 0:
                joint[(c, h)] = w
    total = sum(joint.values())
    if total <= 0:
        return {}, {}
    ps: dict[int, float] = defaultdict(float)
    pss: dict[int, float] = defaultdict(float)
    for (c, h), w in joint.items():
        ps[c] += w / total
        if h is not None:
            pss[h] += w / total
    return dict(ps), dict(pss)


def wall_slot_heuristic(floor: Floor) -> dict[int, float]:
    """Player rule of thumb: cells behind walls where a recognised normal-room layout has a door
    slot but shows no door come first (more such walls first), then cells with more neighbours."""
    facing = _facing_slots(floor)
    out = {}
    for i in range(GRID * GRID):
        if floor.grid[i] >= 0:
            continue
        x, y = i % GRID, i // GRID
        nbs = sum(1 for dx, dy in TRAVEL if (j := index(x + dx, y + dy)) >= 0 and floor.grid[j] >= 0)
        if not nbs:
            continue
        slots = sum(1 for room, bits in facing.get(i, ()) if room.layout_doors & bits)
        out[i] = slots * 10 + nbs
    return out
