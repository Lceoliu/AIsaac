"""Hidden rooms on J460 (Repentance+) floors: the secret, super secret and ultra secret rooms.

The secret and super secret rooms follow the AB+ rules of isaac_macro.secret, plus J460's blocked
cells: right after the (last) boss room, place_rooms blocks its cells and their four neighbours
(LevelGenerator 0x5B1860, VA 0x739A0D), and on Depths II (or Depths I with the Labyrinth)
generate_dungeon has already blocked (6, 5) above the start room (VA 0x74191C). GetNewSecretRoom
(0x5AE170) skips blocked cells like blacklisted ones. The blacklist already covers every side of a
boss room, so this only removes the notch cell of an L-shaped boss room and (6, 5).

The ultra secret room comes after the secret room (place_rooms step 15, GetNewUltraSecretRoom
0x5AE640; see rep/levelgen.py). A candidate cell is free, not blocked, not blacklisted (door targets
of the special rooms placed so far: every side of boss, super secret, secret and curse rooms, the
doorless sides of the others) and touches no room or blocked cell. It scores Random(5) + 10 with the
secret room's -6 / -3 offsets for 1 / 2 occupied cells at Manhattan distance 2 (RING2), and drops
out when one of those rooms has no door towards it, or that door leads off the grid or into the
blacklist; the best score wins, ties uniformly. The RING2 rooms then get a door on every side that
faces an empty cell next to the ultra secret room (where the Red Key opens red rooms). Normal rooms
take their layouts after that (step 16), with those slots as required doors, so a doorless layout
slot can point at the secret room, the super secret room or a red-room cell next to the ultra secret
room, and the posterior below is joint over all three.

Normal rooms get subtype 0 layouts, and their minimum difficulty drops to 1 when the level RNG state
is divisible by 8 (VA 0x73CC08): RepLayoutEvidence is a 7/8 : 1/8 mixture of the two pools.
Static translation: nothing here has been checked against the running J460 engine.
"""
from __future__ import annotations

from collections import defaultdict

from ..floor import Floor, FloorRoom
from ..level import CURSE_LABYRINTH, ROOM_BOSS, ROOM_CURSE, ROOM_DEFAULT, ROOM_SECRET, ROOM_SUPERSECRET
from ..levelgen import GRID, TRAVEL, door_target, index
from ..roomconfig import SHAPE_ANY
from ..secret import OFFSETS, LayoutEvidence, LayoutFactor, secret_given_super_secret, super_secret_prior, \
    win_probabilities
from .levelgen import BLOCK_OFFSETS, RING2
from .place_rooms import ROOM_ULTRASECRET
from .roomconfig import stage_id

HIDDEN_TYPES = (ROOM_SECRET, ROOM_SUPERSECRET, ROOM_ULTRASECRET)
MIN_DIFFICULTY_DROP = 1 / 8
STRANGE_DOOR_CELL = index(6, 5)
ULTRA_BLACKLIST_ALL_SIDES = (ROOM_BOSS, ROOM_SUPERSECRET, ROOM_SECRET, ROOM_CURSE)

# Super secret room weights refitted on J460 port floors by tools/fit_super_secret.py 1500 1 --rep
# (conditional logit, 11,896 floors of 1,500 runs, stages 1-8; the features and the AB+ values are in
# isaac_macro.secret). The placement is the same, and so are the weights, within noise.
SUPER_SECRET_DELTA_WEIGHTS = {-2: 0.0126, -1: 0.0558, 0: 1.0, 1: 1.6805, 2: 1.1183, 3: 0.4896}
SUPER_SECRET_BOSS_WEIGHTS = {-3: 0.3486, -2: 0.9901, -1: 1.575, 0: 1.0, 1: 0.1377}


class RepLayoutEvidence(LayoutEvidence):
    """LayoutEvidence for J460 normal rooms: subtype 0, variant >= 1 on stage 11, the difficulty range
    of RepLevel.difficulty_range (Labyrinth floors and stages 9+ use 1-10, hard 5-15; fewer than 20
    rooms -> 1-15), and one pick in eight with the minimum lowered to 1."""

    def __init__(self, room_config, stage: int, stage_type: int, labyrinth: bool = False,
                 hard: bool = False):
        sid = stage_id(stage, stage_type)
        if not labyrinth and stage < 9:
            if hard:
                min_d, max_d = (10, 15) if stage % 2 == 0 else (5, 10)
            else:
                min_d, max_d = (5, 10) if stage % 2 == 0 else (1, 5)
        else:
            min_d, max_d = (5, 15) if hard else (1, 10)
        if len(room_config.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, 0, 0xFFFFFFFF, min_d, max_d, 0, -1)) < 20:
            min_d, max_d = 1, 15
        min_variant = int(stage == 11)

        def pool(lo: int) -> list:
            return room_config.get_rooms(sid, ROOM_DEFAULT, SHAPE_ANY, min_variant, 0xFFFFFFFF, lo, max_d, 0, 0)

        if min_d > 1:
            self._build(sid, [(1 - MIN_DIFFICULTY_DROP, pool(min_d)), (MIN_DIFFICULTY_DROP, pool(1))])
        else:
            self._build(sid, [(1.0, pool(min_d))])


def strange_door_floor(stage: int, stage_type: int, curses: int) -> bool:
    """generate_dungeon blocks (6, 5) on Depths II, or Depths I with the Labyrinth, off the alternate
    path (RepLevel._strange_door_possible with the strange door unlocked, no challenge)."""
    return stage_type not in (4, 5) and (stage == 6 or (stage == 5 and bool(curses & CURSE_LABYRINTH)))


def last_boss_room(floor: Floor) -> FloorRoom | None:
    """The boss room placed last. On a Labyrinth floor the second boss room is grown next to the
    first (LevelGenerator::GetNewBossRoom 0x5AEF10) and normally touches only it."""
    bosses = floor.of_type(ROOM_BOSS)
    if len(bosses) <= 1:
        return bosses[0] if bosses else None
    leaves = [b for b in bosses if all(nb is not None and nb.type == ROOM_BOSS for _, nb in floor.neighbors(b))]
    return leaves[0] if len(leaves) == 1 else max(bosses, key=lambda r: r.index)


def blocked_cells(floor: Floor, strange_door: bool = False) -> set[int]:
    """The generator's blocked cells when the secret and ultra secret rooms are placed."""
    out: set[int] = set()
    boss = last_boss_room(floor)
    if boss is not None:
        for c in boss.cells:
            for dx, dy in BLOCK_OFFSETS:
                j = index(c % GRID + dx, c // GRID + dy)
                if j >= 0:
                    out.add(j)
    if strange_door:
        out.add(STRANGE_DOOR_CELL)
    return out


def ultra_secret_blacklist(rooms) -> set[int]:
    """The blacklist of place_rooms step 15 (VA 0x73C9E2) over the rooms placed before the ultra
    secret room: door targets (ignore_narrow) of their doorless slots, all slots of boss, super
    secret, secret and curse rooms."""
    out: set[int] = set()
    for r in rooms:
        for slot in range(8):
            if r.layout_doors >> slot & 1 and r.type not in ULTRA_BLACKLIST_ALL_SIDES:
                continue
            tgt = door_target(r.x, r.y, r.shape, slot, True)
            if tgt is not None and 0 <= tgt[0] < GRID and 0 <= tgt[1] < GRID:
                out.add(tgt[0] + GRID * tgt[1])
    return out


def _near(a: int, b: int, dist: int) -> bool:
    return abs(a % GRID - b % GRID) + abs(a // GRID - b // GRID) <= dist


class UltraSecretRule:
    """GetNewUltraSecretRoom on a visible floor (hidden rooms removed), for hypotheses about the
    secret and super secret cells. Normal rooms get their layouts later but occupy their cells
    already; the placed-before rooms are the special rooms and the Dark Room's tomb (a default room
    from room file 0 other than the start room)."""

    def __init__(self, floor: Floor, blocked: set[int]):
        self.floor = floor
        occupied = [g >= 0 for g in floor.grid]
        placed = [r for r in floor.rooms
                  if r.type != ROOM_DEFAULT or (r.file == 0 and r.grid_index != floor.start)]
        self.blacklist = ultra_secret_blacklist(placed)
        self.cands: dict[int, tuple[int, frozenset]] = {}   # cell -> (RING2 count, facing door targets)
        for i in range(GRID * GRID):
            if occupied[i] or i in blocked or i in self.blacklist:
                continue
            x, y = i % GRID, i // GRID
            if any((j := index(x + dx, y + dy)) >= 0 and (occupied[j] or j in blocked) for dx, dy in TRAVEL):
                continue
            count, ok, targets = 0, True, set()
            for ox, oy in RING2:
                j = index(x + ox, y + oy)
                if j < 0 or not occupied[j]:
                    continue
                nb = floor.room_at(j)
                for d, (dx, dy) in enumerate(TRAVEL):
                    if dx * ox + dy * oy <= 0:
                        continue
                    tgt = door_target(nb.x, nb.y, nb.shape, (d + 2) & 3, False)
                    t = -1 if tgt is None else index(*tgt)
                    if t < 0 or t in self.blacklist:
                        ok = False
                        break
                    targets.add(t)
                if not ok:
                    break
                count += 1
            if ok and count:
                self.cands[i] = (count, frozenset(targets))
        self._cache: dict[frozenset, dict] = {}

    def given(self, *hidden: int | None) -> dict:
        """P(ultra secret cell | the secret / super secret rooms at `hidden`), {None: 1.0} if no cell
        qualifies. A 1x1 hidden room kills every candidate within distance 2 (it touches it, or its
        blacklisted side faces it) and every candidate whose facing door targets include one of its
        sides; it never adds to a count."""
        cells = [h for h in hidden if h is not None]
        sides = set()
        for h in cells:
            for dx, dy in TRAVEL:
                j = index(h % GRID + dx, h // GRID + dy)
                if j >= 0:
                    sides.add(j)
        alive = frozenset(u for u, (_, targets) in self.cands.items()
                          if not any(_near(u, h, 2) for h in cells) and not (targets & sides))
        out = self._cache.get(alive)
        if out is None:
            out = win_probabilities({u: OFFSETS.get(self.cands[u][0], 0) for u in alive}) or {None: 1.0}
            self._cache[alive] = out
        return out

    def red_slots(self, u: int) -> dict[int, int]:
        """{room index: slot bits} the RING2 rooms get towards the empty cells next to `u`."""
        x, y = u % GRID, u // GRID
        grid = self.floor.grid
        out: dict[int, int] = {}
        for ox, oy in RING2:
            j = index(x + ox, y + oy)
            if j < 0 or grid[j] < 0:
                continue
            nb = self.floor.room(grid[j])
            for slot in range(8):
                tgt = door_target(nb.x, nb.y, nb.shape, slot, False)
                if tgt is None or index(*tgt) < 0 or grid[index(*tgt)] >= 0:
                    continue
                if abs(tgt[0] - x) + abs(tgt[1] - y) == 1:
                    out[nb.index] = out.get(nb.index, 0) | 1 << slot
        return out


def rep_super_secret_prior(floor: Floor) -> dict[int, float]:
    return super_secret_prior(floor, SUPER_SECRET_DELTA_WEIGHTS, SUPER_SECRET_BOSS_WEIGHTS)


def hidden_posterior(floor: Floor, evidence: LayoutEvidence | None = None,
                     ss_prior: dict[int, float] | None = None, secret_cell: int | None = None,
                     strange_door: bool = False) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Joint posterior of the secret room cell c, the super secret cell h and the ultra secret cell u
    given the visible floor (all three hidden), returned as the marginals (P(c), P(h), P(u)).

    P(c, h, u) is proportional to prior(h) * P_rules(c | h) * P_rules(u | c, h) * the product over
    recognised normal-room layouts of P(layout | visible doors + slots to c, h and u's red-room
    cells) / P(layout | visible doors). `strange_door`: (6, 5) is blocked (strange_door_floor);
    `secret_cell` conditions on a secret room found there (not part of `floor`)."""
    prior = ss_prior if ss_prior is not None else rep_super_secret_prior(floor)
    if not prior:
        prior = {None: 1.0}
    blocked = blocked_cells(floor, strange_door)
    ultra = UltraSecretRule(floor, blocked)
    factor = LayoutFactor(floor, evidence) if evidence is not None else None
    red: dict[int, dict] = {}
    if factor is not None:
        for u in ultra.cands:
            slots = {}
            for i, bits in ultra.red_slots(u).items():
                room = floor.room(i)
                if room.type == ROOM_DEFAULT and room.grid_index != floor.start and evidence.known(room):
                    slots[i] = bits
            red[u] = slots

    joint: dict[tuple, float] = {}
    for h, ph in prior.items():
        rule = secret_given_super_secret(floor, h, blocked)
        for c, pc in rule.items():
            if secret_cell is not None and c != secret_cell:
                continue
            w = ph * pc
            if w <= 0:
                continue
            base = factor.cell_slots([c] if h is None else [c, h]) if factor is not None else None
            for u, pu in ultra.given(c, h).items():
                wu = w * pu
                if factor is not None and wu > 0:
                    extra = base
                    if u is not None and red[u]:
                        extra = {i: [room, bits] for i, (room, bits) in base.items()}
                        for i, bits in red[u].items():
                            extra.setdefault(i, [floor.room(i), 0])[1] |= bits
                    wu *= factor(extra)
                if wu > 0:
                    joint[(c, h, u)] = wu
    total = sum(joint.values())
    if total <= 0:
        return {}, {}, {}
    ps: dict[int, float] = defaultdict(float)
    pss: dict[int, float] = defaultdict(float)
    pus: dict[int, float] = defaultdict(float)
    for (c, h, u), w in joint.items():
        ps[c] += w / total
        if h is not None:
            pss[h] += w / total
        if u is not None:
            pus[u] += w / total
    return dict(ps), dict(pss), dict(pus)
