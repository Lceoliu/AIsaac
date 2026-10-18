"""B34 (2026-10-16): the counterfactual branch teacher (rl/docs/COUNTERFACTUAL_TEACHER.md; train_tok --cf 1, floor /
run modes, needs --teacher-thread 1; off by default, then nothing here runs). The generalisation of teacher v2
(tok_teacher2.py: lanes, Context / play_branch / taken_score, replication, records_of, the V2 ring, the minibatch merge
in train_tok): it reuses that machinery and adds choice points, the target intent, the cap's value_global bootstrap and
the lanes' temporal state. --teacher-v2 keeps working beside it (its hurt / death / random points, its own flags);
--cf does not need it.

Choice points (the worker's episode loop, tok_floor; facts of the observation only, never "what to do"), at a record t
of the episode that is not the first record of its room (the restore point is the room's entry, and a branch never
starts on it; triggers visible on entry show at the room's second record):
  pickups   the first record of the room in which a pickup (ROW ent_id type 5) of a variant class is visible: heart (10),
            coin (20), key (30), bomb (40), chest (50-54, 60, 360), pill (70), card (300), trinket (350); target: the
            nearest one of that class;
  pedestal  the first record of the room with a collectible pedestal (variant 100) holding an item (ent_item > 0 or
            subtype != 0); target: the nearest such pedestal;
  locked    the first record of the room with a locked door (ROW doors column 3) while the player holds a key (player
            column 9 > 0); target: the nearest locked door;
  shop / treasure  the first record of the room showing a door whose target room type (doors column 4 x 30) is a shop
            (2) / a treasure room (4); target: that door;
  cleared   the record with EV_CLEARED (the first record after the room's clear) when the room has >= 2 doors; target:
            one of its open doors (all doors when none is open), drawn from the worker's cf rng;
  wall      B40 (2026-10-18, cfg.cf_wall; off by default, then never detected): a cleared room (player column 23) while
            the player holds >= 1 bomb (ROW 'bombs') with a candidate secret-room wall: a wall of one of the room's
            minimap cells (map channel 3) whose neighbour cell is inside the 13 x 13 grid, not shown on the minimap
            (channel 0) and not next to a boss room (channel 4), with no door of the room on it (each door is assigned
            to the nearest wall midpoint; hidden doors are walls in the bridge's door list, abp_bridge.lua visible_door)
            and with door slots on that side (narrow rooms have none on their long sides). Not in boss / secret /
            super secret rooms. The candidate taken: the highest p_vis (wall_candidates: AB+'s secret-room placement
            score, U{10..14} + offset by the neighbour cell's shown neighbours: 1 -> -6, 2 -> -3, 3+ -> 0, as in
            rl/macro/isaac_macro/secret.py, over the unknown cells next to the shown map), ties the nearest. target:
            ('wall', info) with the bomb spot (the wall's middle, one tile in; inner-area tile coordinates), the side,
            the neighbour cell, p_vis, the room key. Once per room per episode: detect does not mark it seen, the
            caller does (consume_wall) when the record offers it (a record whose point went to another trigger tries
            again at the next record); a cleared room without a candidate is marked seen at once.
            The secret-room inferencer itself (rl/macro secret_posterior) is not used: it needs the whole visible floor
            with room shapes, types and layout door slots, which the ROW map (merged cells, no shapes, part of the
            floor) does not carry; p_vis keeps its dominant term (the neighbour-count offset) and its exact tie rule.
  random    with probability cf_random per game hour per record (worker's rng), in the same room as the record before;
            no target.
  One point per record at most (the first trigger of a type new to the episode, else the first in TRIGGERS order).
  Quota (Quota): a per-worker token bucket that fills at cfg.cf_points per game hour of the worker's play (capacity
  CF_BURST tokens); a trigger takes a token when one is there. Without a token, a trigger of a type the episode has
  not shown yet replaces the oldest of the episode's pending points whose type had shown before ("unseen trigger types
  first"), else it is skipped (cf_skipped). A point's state is snapshotted at once (temporal models: the actor copies
  the worker's temporal slot into the point's bank row before the record's own step; cf_snap block); the episode's
  points are queued at its end (their taken score needs its continuation); when the queue is full the oldest point
  goes (cf_dropped; the death teacher's searches are a queue of their own and never compete).
Branches (cf_search; K = cfg.cf_branches counts taken): taken = the episode's own continuation (taken_score over the
  worker's log, bootstrapped with the actor's value_global of the window's last record, its value without a global
  token), target (TargetIntent: the move toward the trigger's target, tok_pickup.move_toward, every other head the
  actor's, until it is reached: a pickup of that class happened / the target is gone / the room changed / a door
  target: the room changed, or cfg.cf_target_seconds; then the policy), and sticky high-temperature policy branches
  for the rest (tok_teacher2.Intent 'sticky' with eps = cfg.cf_sticky_eps: at each re-pick the move is uniform over the
  nine with that probability, kept for a geometric number of decisions of mean cfg.teacher_v2_sticky, for
  cfg.cf_target_seconds; one seed each). A wall point's target branch is BombIntent ('bomb_at', B40): the move toward
  the bomb spot until the player is within BOMB_ARRIVE tiles of it (or BOMB_STUCK decisions without getting closer
  within BOMB_NEAR tiles), then one decision with move 0 and the bomb head 1 (only with a bomb held), then the policy;
  no bomb when the room changes or cfg.cf_target_seconds pass first. Its records are distilled through the bomb
  decision (at least cfg.cf_distill_steps, at most cfg.cf_wall_distill: the walk and the bomb are the lesson), and the
  search's result carries 'wall' (bombed at, opened: after the bomb the neighbour cell showed on the minimap or a door
  appeared on that wall, its score - taken, p_vis, side) for the cf_wall_* counters.
  B40 deferral (cfg.cf_defer_target, A30's A2; off = as before): a door target's TargetIntent (locked / shop /
  treasure / cleared) and every BombIntent wait for the room's clear (Deferral: the policy's whole action until the
  record's room is clear or EV_CLEARED, then the intent with its cap h counted from there; the room changing first: it
  never starts). A30 measured the undeferred door branch: the door triggers fire on the room's second record, in 69% of
  the cases (81% on floor 2+) while the room is still being fought, and "push into a closed door for 20 s" scored -5.6
  vs the policy. A deferred branch's records are kept and distilled from its first own decision (play_branch
  keep_from), so the imitation records teach the walk after the clear, not the policy's own fight. Every branch: the same restored state and the game's own random numbers
  (the episode's MT state at the point: common random numbers), cfg.cf_seconds or the floor's end (run mode: EV_EXIT
  ends a branch, bootstrapped), score = discounted trainer rewards (tok_branch.step_reward + the heal / resource
  terms when the trainer has them, tok_teacher2.gain_reward; gamma cfg.cf_gamma) + gamma^T x V_global at the cut.
  The best branch is replayed once with another MT reseed (the noise floor): the point improves only if best - taken
  > cfg.cf_margin and > |best - replicate|.
Records: the improving branch's first cfg.cf_distill_steps records into the V2 ring (records_of, kind KIND_CF, weight
  min(best - taken, cfg.cf_cap), meta 'trig' = trigger index + 1); train_tok trains them with teacher v2's loss path
  (--teacher-v2-buffer / -batch / -reuse / -coef / -vf / -weight), --cf-advantage 1 replaces the imitation term of the
  first record (j = 0) by a clipped policy-gradient term with the constant advantage best - taken.
Lanes of temporal models: the lane's temporal slot starts as a copy of the point's snapshot (LanePolicy.tbegin, the
  actor's TemporalState.copy_slot_), every answer hands back (o_prev, has, o) of the lane's step (lane_t), and the
  imitation records keep them (v2_t): the learner reads them as stored outputs (tok_policy temporal 'tail').
"""
import numpy as np

from .tok_obs import CELL, EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM
from .tok_pickup import move_toward
from .tok_teacher2 import INTENT_BOMB, INTENT_TARGET, V2_RING, Intent, play_branch, restore, taken_score

# (B40: 'wall' before 'random', which stays last; the indices are run-time only: the V2 ring's 'trig' meta is not saved)
TRIGGERS = ('heart', 'coin', 'key', 'bomb', 'chest', 'pill', 'card', 'trinket', 'pedestal', 'locked', 'shop',
            'treasure', 'cleared', 'wall', 'random')
TI = {k: i for i, k in enumerate(TRIGGERS)}
PICKUP_CLASS = {10: 'heart', 20: 'coin', 30: 'key', 40: 'bomb', 50: 'chest', 51: 'chest', 52: 'chest', 53: 'chest',
                54: 'chest', 60: 'chest', 360: 'chest', 70: 'pill', 300: 'card', 350: 'trinket'}
CLASS_VARIANTS = {}
for _v, _c in PICKUP_CLASS.items():
    CLASS_VARIANTS.setdefault(_c, set()).add(_v)
PEDESTAL_VARIANT = 100
ROOM_SHOP, ROOM_TREASURE = 2, 4
CF_BURST = 2.0       # the quota's bucket: tokens it holds at most
CF_BANK_EXTRA = 6    # bank rows per worker beyond cfg.cf_queue (the point being searched, the episode's pending ones)
LANE_T_PAD = 2       # lane_t / v2_t columns beyond 2 D: has, valid (1: written by a lane step)
# B40 (the wall trigger and BombIntent; geometry in tiles of the room's inner area: the bounding box of the terrain's
# 'inside' cells (ROW grid channel 0) = Room TopLeft .. BottomRight, origin its top-left corner, y down)
SIDES = ('up', 'down', 'left', 'right')
SIDE_STEP = ((-1, 0), (1, 0), (0, -1), (0, 1))   # (dy, dx) on the minimap per side
MAP_N = 13
ROOM_BOSS, ROOM_SECRET, ROOM_SUPERSECRET = 5, 7, 8
NARROW_TILES = (10.0, 5.0)   # a cell narrower than this (tiles across, along) has no door slots on its long sides
SECRET_OFFSETS = (0, -6, -3)   # AB+ secret-room score offset by shown neighbours 1, 2 (3+: 0); index = count, 0 unused
SCORE_BASE, SCORE_SPAN = 10, 5
BOMB_ARRIVE = 0.6    # tiles from the bomb spot: arrived
BOMB_NEAR = 2.0      # tiles: close enough to bomb when stuck
BOMB_STUCK = 8       # decisions without getting closer (by 0.05 tiles) that count as stuck


def _el(r):
    """A ROW element from an element or a length-1 view."""
    return r[0] if getattr(r, 'shape', ()) == (1,) else r


def pickup_mask(r, cls):
    """bool [n_ent]: the entities of record r (a ROW element) of pickup class cls ('pedestal': collectible pedestals
    holding an item)."""
    n = int(r['n_ent'])
    ids = r['ent_id'][:n]
    if cls == 'pedestal':
        return (ids[:, 0] == 5) & (ids[:, 1] == PEDESTAL_VARIANT) & ((r['ent_item'][:n, 0] > 0) | (ids[:, 2] != 0))
    return (ids[:, 0] == 5) & np.isin(ids[:, 1], list(CLASS_VARIANTS[cls]))


def nearest(r, mask):
    """Index of the nearest entity under mask (ent column 2 = distance / 300), or -1."""
    sel = np.flatnonzero(mask)
    if not len(sel):
        return -1
    return int(sel[np.argmin(r['ent'][sel, 2])])


def door_types(r):
    """int [n_doors]: each door's target room type (doors column 4 x 30, rounded)."""
    k = int(r['n_doors'])
    return np.rint(r['doors'][:k, 4] * 30).astype(np.int64)


def inner_area(r):
    """(W, H) of record r's inner area in tiles (the bounding box of ROW grid channel 0, 'inside'), or None."""
    inside = r['grid'][0]
    rows = np.flatnonzero(inside.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(inside.any(0))
    return float(cols[-1] - cols[0] + 1), float(rows[-1] - rows[0] + 1)


def player_tiles(r, area):
    """The player's position in inner-area tiles (player columns 0, 1: (x - TopLeft) / (BottomRight - TopLeft))."""
    return float(r['player'][0]) * area[0], float(r['player'][1]) * area[1]


def room_walls(r, area=None):
    """The perimeter walls of record r's room: a list of dicts (cell (my, mx) on the minimap, side index (SIDES), nb the
    neighbour cell (my, mx), wall: its midpoint and target: the bomb spot (the midpoint one tile in) in inner-area
    tiles, slots: False on a narrow room's long sides, door: a door of the room is on it), or [] without the player's
    room on the minimap or without a terrain. Each of the record's doors is put on the nearest wall midpoint."""
    here = r['map'][3]
    ys, xs = np.nonzero(here)
    area = area or inner_area(r)
    if not len(ys) or area is None:
        return []
    cells = set(zip(ys.tolist(), xs.tolist()))
    my0, mx0 = int(ys.min()), int(xs.min())
    bh, bw = int(ys.max()) - my0 + 1, int(xs.max()) - mx0 + 1
    cw, ch = area[0] / bw, area[1] / bh   # tiles per minimap cell
    out = []
    for y, x in sorted(cells):
        rr, cc = y - my0, x - mx0
        for s, (dy, dx) in enumerate(SIDE_STEP):
            nb = (y + dy, x + dx)
            if nb in cells:
                continue
            if s < 2:   # up / down
                wx, wy = (cc + 0.5) * cw, (rr + (s == 1)) * ch
                t = (wx, wy + (0.5 if s == 0 else -0.5))
                slots = ch >= NARROW_TILES[1]
            else:
                wx, wy = (cc + (s == 3)) * cw, (rr + 0.5) * ch
                t = (wx + (0.5 if s == 2 else -0.5), wy)
                slots = cw >= NARROW_TILES[0]
            out.append(dict(cell=(y, x), side=s, nb=nb, wall=(wx, wy), target=t, slots=bool(slots), door=False))
    k = int(r['n_doors'])
    if k and out:
        px, py = player_tiles(r, area)
        mids = np.array([w['wall'] for w in out])
        for j in range(k):
            dxy = r['doors'][j, :2] * 200.0 / CELL   # the door relative to the player, in tiles
            i = int(np.argmin(np.hypot(mids[:, 0] - (px + dxy[0]), mids[:, 1] - (py + dxy[1]))))
            out[i]['door'] = True
    return out


def secret_scores(shown, boss):
    """p_vis [13, 13]: AB+'s secret-room placement rule on the visible minimap. Candidates: cells not shown, not next to
    a boss cell, with >= 1 shown neighbour; each scores U{10..14} + offset (1 shown neighbour -6, 2 -3, 3+ 0;
    LevelGenerator::GetNewSecretRoom, rl/macro/isaac_macro/secret.py candidate_offsets), the highest wins, ties
    uniform: p = P(the cell wins) (win_probabilities there). An approximation: unseen rooms, layout door slots and the
    blacklist of special rooms other than the boss are not known from the ROW."""
    s = np.pad(shown.astype(np.int8), 1)
    n = s[:-2, 1:-1] + s[2:, 1:-1] + s[1:-1, :-2] + s[1:-1, 2:]
    b = np.pad(boss.astype(np.int8), 1)
    nb = b[:-2, 1:-1] + b[2:, 1:-1] + b[1:-1, :-2] + b[1:-1, 2:]
    cand = (~shown) & (n > 0) & (nb == 0)
    out = np.zeros(shown.shape, np.float64)
    if not cand.any():
        return out
    offs = np.where(n >= 3, 0, np.where(n == 2, -3, -6))
    groups = {o: int(((offs == o) & cand).sum()) for o in (0, -3, -6)}
    groups = {o: c for o, c in groups.items() if c}
    value = win_values(groups)
    for o, v in value.items():
        out[(offs == o) & cand] = v
    return out


def win_values(groups):
    """{offset: P(one cell of that offset wins)} for groups {offset: cells}: each cell scores U{10..14} + offset, the
    highest wins, ties uniform (the exact integral of rl/macro/isaac_macro/secret.py win_probabilities, ported)."""
    from math import comb

    def lt(o, s_):
        return min(max(s_ - (SCORE_BASE + o), 0), SCORE_SPAN) / SCORE_SPAN

    def eq(o, s_):
        return 1.0 / SCORE_SPAN if SCORE_BASE + o <= s_ < SCORE_BASE + o + SCORE_SPAN else 0.0

    value = {}
    for h in groups:
        total = 0.0
        for u in range(SCORE_SPAN):
            s_ = SCORE_BASE + h + u
            poly = [1.0]
            for o, n in groups.items():
                n -= o == h
                if n <= 0:
                    continue
                a, b = lt(o, s_), eq(o, s_)
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
    return value


def room_key(r):
    """(stage, the first minimap cell of the player's room): the room within an episode (a run's floors differ)."""
    idx = np.flatnonzero(r['map'][3].ravel())
    return int(r['stage']), int(idx[0]) if len(idx) else -1


def wall_candidates(r):
    """The candidate secret-room walls of record r (module docstring 'wall'), best first: dicts of room_walls plus
    p (p_vis of the neighbour cell); [] when there are none (or the room is a boss / secret / super secret room)."""
    m = r['map']
    here = m[3] > 0
    if not here.any() or (m[4][here] > 0).any() or int(r['room_type']) in (ROOM_BOSS, ROOM_SECRET, ROOM_SUPERSECRET):
        return []
    area = inner_area(r)
    walls = room_walls(r, area)
    if not walls:
        return []
    shown = m[0] > 0
    p = None
    out = []
    for w in walls:
        y, x = w['nb']
        if w['door'] or not w['slots'] or not (0 <= y < MAP_N and 0 <= x < MAP_N) or shown[y, x]:
            continue
        if p is None:
            p = secret_scores(shown, m[4] > 0)
        if p[y, x] <= 0:   # (next to a boss room)
            continue
        out.append(dict(w, p=float(p[y, x])))
    if out:
        px, py = player_tiles(r, area)
        out.sort(key=lambda w: (-w['p'], np.hypot(w['target'][0] - px, w['target'][1] - py), w['side']))
    return out


def consume_wall(seen, walled, target):
    """The wall trigger of a room is used up (the record offered it to the quota): marked seen for the room and its
    room key kept for the episode (walled)."""
    seen.add('wall')
    if walled is not None and target is not None:
        walled.add(target[1]['room'])


def detect(row, seen, rng=None, wall=False, walled=None):
    """The choice-point triggers of record row (a ROW element or a length-1 view) whose type is not in seen (the
    room's triggers so far; the found ones are added). Returns [(trigger index, target)] in TRIGGERS order; a target is
    ('ent', class) or ('door', door index) or ('wall', info) or None. rng (cleared: the door drawn; None: the first open
    one). wall (B40, cfg.cf_wall): the wall trigger too; it is not added to seen (consume_wall: the caller, when the
    record offers it), walled: the episode's room keys whose wall trigger was used (None: no episode memory)."""
    r = _el(row)
    out = []
    n = int(r['n_ent'])
    if n:
        ids = r['ent_id'][:n]
        pick = ids[:, 0] == 5
        if pick.any():
            for cls in ('heart', 'coin', 'key', 'bomb', 'chest', 'pill', 'card', 'trinket'):
                if cls not in seen and pickup_mask(r, cls).any():
                    seen.add(cls)
                    out.append((TI[cls], ('ent', cls)))
            if 'pedestal' not in seen and pickup_mask(r, 'pedestal').any():
                seen.add('pedestal')
                out.append((TI['pedestal'], ('ent', 'pedestal')))
    k = int(r['n_doors'])
    if k:
        doors = r['doors'][:k]
        if 'locked' not in seen and float(r['player'][9]) > 0:
            locked = np.flatnonzero(doors[:, 3] > 0.5)
            if len(locked):
                seen.add('locked')
                j = int(locked[np.argmin(np.hypot(doors[locked, 0], doors[locked, 1]))])
                out.append((TI['locked'], ('door', j)))
        types = door_types(r)
        for name, rt in (('shop', ROOM_SHOP), ('treasure', ROOM_TREASURE)):
            if name not in seen:
                hit = np.flatnonzero(types == rt)
                if len(hit):
                    seen.add(name)
                    out.append((TI[name], ('door', int(hit[0]))))
        if 'cleared' not in seen and int(r['events']) & EV_CLEARED and k >= 2:
            seen.add('cleared')
            opened = np.flatnonzero(doors[:, 2] > 0.5)
            pool = opened if len(opened) else np.arange(k)
            j = int(pool[int(rng.integers(len(pool)))]) if rng is not None else int(pool[0])
            out.append((TI['cleared'], ('door', j)))
    if wall and 'wall' not in seen and float(r['player'][23]) > 0.5 and int(r['bombs']) > 0:   # B40
        key = room_key(r)
        cands = [] if (walled is not None and key in walled) else wall_candidates(r)
        if not cands:
            seen.add('wall')   # (nothing to bomb here: not looked at again in this room)
        else:
            w = cands[0]
            out.append((TI['wall'], ('wall', dict(x=w['target'][0], y=w['target'][1], side=w['side'],
                                                  cell=w['nb'][0] * MAP_N + w['nb'][1], p=w['p'], n=len(cands),
                                                  room=key, wall=w['wall']))))
    out.sort(key=lambda x: x[0])
    return out


class Quota:
    """The per-worker choice-point quota (module docstring): a token bucket filling at points_per_hour per game hour
    of play (tick(n): n decisions of fpd logic frames), capacity burst; and the episode's pending points with the
    "unseen trigger types first" replacement. offer() decides one trigger."""

    def __init__(self, points_per_hour, fpd, burst=CF_BURST):
        self.rate = float(points_per_hour) * fpd / 30.0 / 3600.0
        self.burst = max(1.0, float(burst))
        self.budget = 0.0
        self.types = set()     # trigger types the running episode has shown (accepted or not)
        self.pending = []      # the running episode's accepted points: dicts with 'trig', 'fresh' (and the caller's)
        self.skipped = self.replaced = 0

    def tick(self, n=1):
        self.budget = min(self.burst, self.budget + self.rate * n)

    def begin_episode(self):
        self.types = set()
        self.pending = []

    def choose(self, found):
        """The trigger of a record (detect's list, non-empty) the record offers: the first of a type the episode has
        not shown yet, else the first."""
        for tr in found:
            if TRIGGERS[tr[0]] not in self.types:
                return tr
        return found[0]

    def offer(self, trig):
        """(accepted, the pending point it replaces or None) for a trigger index; the episode's types are updated.
        The caller appends the accepted point (a dict with 'trig' and 'fresh') to self.pending."""
        name = TRIGGERS[trig]
        fresh = name not in self.types
        self.types.add(name)
        if self.budget >= 1.0:
            self.budget -= 1.0
            return True, None, fresh
        if fresh:
            stale = [p for p in self.pending if not p['fresh']]
            if stale:
                victim = stale[0]
                self.pending.remove(victim)
                self.replaced += 1
                return True, victim, fresh
        self.skipped += 1
        return False, None, fresh


def room_cleared(r):
    """B40: record r's room is clear (player column 23, Room:IsClear) or was cleared during the step (EV_CLEARED)."""
    return float(r['player'][23]) > 0.5 or bool(int(r['events']) & EV_CLEARED)


class Deferral:
    """B40 (A30's A2, cfg.cf_defer_target): an intent that waits for the room's clear. While the record's room is not
    clear the policy's whole action is played (j0 None); at the first clear record j0 = j and the intent's own
    decisions begin (its cap h counts from there). The room changing first: the intent never starts (gone). keep_from()
    is play_branch's first kept record (None: nothing kept yet). Without defer, j0 = 0 at once (as before B40)."""

    def defer_reset(self):
        self.j0 = None if self.defer else 0
        self.gone, self.wait_map3 = False, None

    def keep_from(self):
        return self.j0

    def waiting(self, j, r):
        """True while the intent must still wait at record j (r: the record, a ROW element)."""
        if self.j0 is not None:
            return False
        if self.gone:
            return True
        m3 = r['map'][3]
        if self.wait_map3 is None:
            self.wait_map3 = m3.copy()
        elif (j > 0 and int(r['events']) & (EV_NEW_ROOM | EV_EXIT)) or ((m3 != self.wait_map3).any() and m3.any()):
            self.gone = True
            return True
        if not room_cleared(r):
            return True
        self.j0 = j
        return False


class TargetIntent(Deferral):
    """The target branch's movement (module docstring): the move toward the target until it is reached or h decisions
    passed, then the policy's whole action; every other head is always the policy's. B40: defer (Deferral) waits for
    the room's clear first (door targets with cfg.cf_defer_target)."""
    kind = 'target'

    def __init__(self, target, h, defer=False):
        self.target, self.h, self.defer = target, int(h), bool(defer)
        self.reset()

    @property
    def code(self):
        return INTENT_TARGET

    def reset(self):
        self.arrived, self.map3, self.count, self.steps = False, None, None, 0
        self.defer_reset()

    def move(self, r):
        """The move toward the target in record r (a ROW element), or 0 when it is reached / gone."""
        kind, what = self.target
        map3 = r['map'][3]
        if self.map3 is None:
            self.map3 = map3.copy()
        elif (int(r['events']) & (EV_NEW_ROOM | EV_EXIT)) or ((map3 != self.map3).any() and map3.any()):
            return 0   # the room changed: reached (a door) or left behind
        if kind == 'door':
            j = int(what)
            if j >= int(r['n_doors']):
                return 0
            d = r['doors'][j]
            return move_toward(d[0], d[1])
        mask = pickup_mask(r, what)
        c = int(mask.sum())
        if self.count is None:
            self.count = c
        if c == 0 or c < self.count or (what == 'pedestal' and int(r['events']) & EV_ITEM):
            return 0   # picked up (one fewer of the class) / gone
        self.count = c
        j = nearest(r, mask)
        return move_toward(r['ent'][j, 0], r['ent'][j, 1])

    def compose(self, j, a, row=None):
        if self.arrived or row is None:
            return a
        if self.waiting(j, _el(row)) or j - self.j0 >= self.h:
            return a
        m = self.move(_el(row))
        if not m:
            self.arrived = True
            return a
        self.steps += 1
        return (int(m),) + tuple(a[1:])


class BombIntent(Deferral):
    """B40: the wall point's branch (module docstring): the move toward the bomb spot (('wall', info): info x, y in
    inner-area tiles), then one decision with move 0 and bomb 1, then the policy's whole action. Every other head is
    always the policy's. It watches every record it is shown (observe) for the counters: bombed (the branch step j of
    the bomb decision, -1: none), opened (after the bomb: the neighbour cell info['cell'] shown on the minimap, or more
    doors on the bombed wall than before), bombs0 / bombs_min (ROW 'bombs'). defer (Deferral, cfg.cf_defer_target): it
    waits for the room's clear (a wall point's room is clear already: no change then)."""
    kind = 'bomb_at'

    def __init__(self, target, h, defer=False):
        self.target, self.h, self.defer = target, int(h), bool(defer)
        self.info = target[1]
        self.reset()

    @property
    def code(self):
        return INTENT_BOMB

    def reset(self):
        self.done, self.bombed, self.opened = False, -1, False
        self.map3, self.best, self.since = None, None, 0
        self.bombs0 = self.bombs_min = None
        self.doors0 = None
        self.steps = 0
        self.defer_reset()

    def wall_doors(self, r):
        """Doors of record r on the bombed wall (room_walls' assignment)."""
        if self.map3 is None or (r['map'][3] != self.map3).any():
            return 0
        cell = (int(self.info['cell']) // MAP_N, int(self.info['cell']) % MAP_N)
        return sum(1 for w in room_walls(r) if w['door'] and w['nb'] == cell)

    def observe(self, r):
        b = int(r['bombs'])
        if self.bombs0 is None:
            self.bombs0 = self.bombs_min = b
        self.bombs_min = min(self.bombs_min, b)
        if self.map3 is None:
            self.map3 = r['map'][3].copy()
            self.doors0 = self.wall_doors(r)
        if self.bombed >= 0 and not self.opened:
            c = int(self.info['cell'])
            if r['map'][0].ravel()[c] > 0 or self.wall_doors(r) > self.doors0:
                self.opened = True

    def compose(self, j, a, row=None):
        if row is None:
            return a
        r = _el(row)
        self.observe(r)
        if self.done:
            return a
        if self.waiting(j, r):
            if self.gone:
                self.done = True
            return a
        if j - self.j0 >= self.h or (j > 0 and int(r['events']) & (EV_NEW_ROOM | EV_EXIT)) or \
                ((r['map'][3] != self.map3).any() and r['map'][3].any()):
            self.done = True   # the cap, or the room changed: no bomb
            return a
        area = inner_area(r)
        if area is None:
            self.done = True
            return a
        px, py = player_tiles(r, area)
        dx, dy = float(self.info['x']) - px, float(self.info['y']) - py
        dist = float(np.hypot(dx, dy))
        if self.best is None or dist < self.best - 0.05:
            self.best, self.since = dist, 0
        else:
            self.since += 1
        if dist <= BOMB_ARRIVE or (self.since >= BOMB_STUCK and dist <= BOMB_NEAR):
            self.done = True
            if int(r['bombs']) <= 0:
                return a
            self.bombed = j
            return (0, a[1], 1) + tuple(a[3:])
        self.steps += 1
        return (int(move_toward(dx, dy)),) + tuple(a[1:])

    def summary(self):
        return dict(bombed=int(self.bombed), opened=bool(self.opened), steps=int(self.steps),
                    bombs0=self.bombs0, bombs_min=self.bombs_min, j0=self.j0)


def intents(pt, cfg, rng, h):
    """The branch intents of a point (taken is not among them): the target intent when the point has a target (B40: a
    wall target's BombIntent), then sticky high-temperature intents up to cfg.cf_branches - 1 in all."""
    out = []
    defer = bool(getattr(cfg, 'cf_defer_target', False))   # B40: door / wall targets wait for the room's clear
    if pt.target is not None and pt.target[0] == 'wall':
        out.append(BombIntent(pt.target, h, defer=defer))
    elif pt.target is not None:
        out.append(TargetIntent(pt.target, h, defer=defer and pt.target[0] == 'door'))
    while len(out) < max(1, int(cfg.cf_branches) - 1):
        out.append(Intent('sticky', 0, h, max(float(cfg.teacher_v2_sticky), 1.0), int(rng.integers(1, 2 ** 31 - 1)),
                          eps=cfg.cf_sticky_eps))
    return out


def judge(taken, scores, rep, margin):
    """The improvement test: (best index, gain = best - taken, spread = |best - replicate| or None, improving)."""
    b = int(np.argmax(scores))
    gain = float(scores[b]) - float(taken)
    spread = None if rep is None else abs(float(scores[b]) - float(rep))
    return b, gain, spread, rep is not None and gain > margin and gain > spread


def cf_search(pt, ctx, policy, rng, branch=play_branch, restore_fn=restore):
    """One choice point: restore (the room's entry forked with the episode's reseed, its actions replayed to the
    point's record d = pt.at), taken's score, the intents' branches, the best one's replicate. Returns a dict like
    tok_teacher2.search_point's (depths 1, gain0 = gain, d, k 0, taken, taken_steps, best (play_branch's dict +
    'intent'), gain, rep, spread, improving, scores, ends) or None (d is the room's entry record). branch / restore_fn:
    stand-ins in tests. B40: 'distill' = the records of the best branch to distill (cfg.cf_distill_steps; a BombIntent
    that bombed: through its bomb decision, at most cfg.cf_wall_distill) and, for a wall point, 'wall' = the BombIntent
    branch's summary (BombIntent.summary(), its score, gain = score - taken, p_vis, side, best: it was the best and
    the point improves) taken before any replicate; with cfg.cf_defer_target 'defer' = the deferred intent's start
    (started: the room was cleared within the branch, wait: decisions until then). A deferred branch's records are
    kept from its start (play_branch keep_from): 'distill' counts from there."""
    cfg = ctx.cfg
    d = int(pt.at)
    if d < pt.offset + 1:
        return None
    h = max(1, int(round(cfg.cf_target_seconds * 30 / cfg.frames_per_decision)))
    S, hp0 = restore_fn(pt, d, ctx)
    try:
        vals = pt.log.vg if (ctx.boot_vg and len(getattr(pt.log, 'vg', ()) or ())) else None
        extra = pt.log.gain_r if (getattr(ctx, 'gains', False) and len(getattr(pt.log, 'gain_r', ()) or ())) else None
        taken, steps = taken_score(pt.log, d, ctx.cap, ctx.gamma, ctx.rw, ctx.run, vals, ctx.exit_end, extra)
        runs = []
        k_dist = int(cfg.cf_distill_steps)
        k_wall = max(k_dist, int(getattr(cfg, 'cf_wall_distill', k_dist)))
        for intent in intents(pt, cfg, rng, h):
            bomb = intent.kind == 'bomb_at'
            b = branch(S, ctx, pt, d, hp0, intent, policy, keep=k_wall if bomb else k_dist)
            b['intent'] = intent
            if bomb:   # (before a replicate resets the intent)
                b['wall'] = intent.summary()
            runs.append(b)
        scores = [b_['score'] for b_ in runs]
        bi = int(np.argmax(scores))
        rep = None
        if scores[bi] - taken > cfg.cf_margin:   # the noise floor: the best intent again from another reseed
            r = branch(S, ctx, pt, d, hp0, runs[bi]['intent'], policy, reseed=int(rng.integers(1, 2 ** 31 - 1)),
                       keep=False)
            ctx.stats['replicates'] += 1
            rep = r['score']
        bi, gain, spread, improving = judge(taken, scores, rep, cfg.cf_margin)
    finally:
        try:
            S.close()
        except OSError:
            pass
    best = runs[bi]
    distill = k_dist
    if 'wall' in best and best['wall']['bombed'] >= 0:   # B40: the walk and the bomb decision
        distill = min(max(k_dist, best['wall']['bombed'] - (best['wall'].get('j0') or 0) + 1), k_wall)
    out = dict(depths=1, gain0=gain, d=d, k=0, taken=taken, taken_steps=steps, best=best, gain=gain, rep=rep,
               spread=spread, improving=improving, scores=scores, ends=[b_['end'] for b_ in runs], distill=distill)
    for b_ in runs:
        if 'wall' in b_:
            info = pt.target[1]
            out['wall'] = dict(b_['wall'], score=float(b_['score']), gain=float(b_['score']) - float(taken),
                               p=float(info.get('p', 0.0)), side=int(info.get('side', -1)),
                               best=bool(improving and b_ is best))
    if runs and getattr(runs[0]['intent'], 'defer', False):   # B40: the deferred target / bomb_at branch's wait
        j0 = runs[0].get('from', 0)
        out['defer'] = dict(started=j0 is not None, wait=int(j0 or 0))
    return out


def cf_sizes(workers, t_dim):
    """Shared-memory block sizes (bytes) of attach_cf's arrays, in its order (t_dim: the temporal width D, 0 without
    a temporal model)."""
    out = [8 * workers, 8 * workers]
    if t_dim:
        w = 2 * t_dim + LANE_T_PAD
        out += [4 * workers * w, 4 * workers * V2_RING * w]
    return tuple(out)


def attach_cf(names, workers, t_dim):
    """(blocks, main_vg float64 [workers]: the actor's value_global of each worker's last answered ordinary record
    (written before its reply, like main_value), cf_snap int64 [workers]: the worker's snapshot request for the record
    it sends (bank row + 1; the actor copies the worker's temporal slot there before that record's step and writes 0;
    0: none), lane_t float32 [workers, 2 D + 2] (temporal: each lane answer's o_prev, has, o; else None), v2_t float32
    [workers, V2_RING, 2 D + 2] (each imitation record's; else None))."""
    from multiprocessing import shared_memory
    k = 4 if t_dim else 2
    blocks = [shared_memory.SharedMemory(name=n) for n in names[:k]]
    main_vg = np.ndarray((workers,), np.float64, buffer=blocks[0].buf)
    cf_snap = np.ndarray((workers,), np.int64, buffer=blocks[1].buf)
    lane_t = v2_t = None
    if t_dim:
        w = 2 * t_dim + LANE_T_PAD
        lane_t = np.ndarray((workers, w), np.float32, buffer=blocks[2].buf)
        v2_t = np.ndarray((workers, V2_RING, w), np.float32, buffer=blocks[3].buf)
    return blocks, main_vg, cf_snap, lane_t, v2_t


def cf_names_offset(cfg):
    """Where attach_cf's names start in the lane names list (after teacher v2's 6 blocks and the memory's 3)."""
    return 6 + (3 if getattr(cfg, 'memory_dim', 0) else 0)


def bank_rows(cfg):
    """Bank rows (temporal snapshots) per worker."""
    return int(cfg.cf_queue) + CF_BANK_EXTRA


def t_split(t, dim):
    """(o_prev [.., D], has [..], o [.., D]) of lane_t / v2_t vectors."""
    t = np.asarray(t)
    return t[..., :dim], t[..., dim] > 0.5, t[..., dim + 1:2 * dim + 1]


def t_pack(o_prev, has, o):
    """lane_t vectors from (o_prev [n, D], has [n], o [n, D]) (the actor's side; the last column 1 = valid)."""
    o_prev = np.asarray(o_prev, np.float32)
    n, dim = o_prev.shape
    out = np.zeros((n, 2 * dim + LANE_T_PAD), np.float32)
    out[:, :dim] = o_prev
    out[:, dim] = np.asarray(has, np.float32)
    out[:, dim + 1:2 * dim + 1] = o
    out[:, -1] = 1.0   # (valid: written by a lane step)
    return out
