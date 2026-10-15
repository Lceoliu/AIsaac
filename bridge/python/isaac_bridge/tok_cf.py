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
  cfg.cf_target_seconds; one seed each). Every branch: the same restored state and the game's own random numbers
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

from .tok_obs import EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM
from .tok_pickup import move_toward
from .tok_teacher2 import INTENT_TARGET, V2_RING, Intent, play_branch, restore, taken_score

TRIGGERS = ('heart', 'coin', 'key', 'bomb', 'chest', 'pill', 'card', 'trinket', 'pedestal', 'locked', 'shop',
            'treasure', 'cleared', 'random')
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


def detect(row, seen, rng=None):
    """The choice-point triggers of record row (a ROW element or a length-1 view) whose type is not in seen (the
    room's triggers so far; the found ones are added). Returns [(trigger index, target)] in TRIGGERS order; a target is
    ('ent', class) or ('door', door index) or None. rng (cleared: the door drawn; None: the first open one)."""
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


class TargetIntent:
    """The target branch's movement (module docstring): the move toward the target until it is reached or h decisions
    passed, then the policy's whole action; every other head is always the policy's."""
    kind = 'target'

    def __init__(self, target, h):
        self.target, self.h = target, int(h)
        self.reset()

    @property
    def code(self):
        return INTENT_TARGET

    def reset(self):
        self.arrived, self.map3, self.count, self.steps = False, None, None, 0

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
        if self.arrived or j >= self.h or row is None:
            return a
        m = self.move(_el(row))
        if not m:
            self.arrived = True
            return a
        self.steps += 1
        return (int(m),) + tuple(a[1:])


def intents(pt, cfg, rng, h):
    """The branch intents of a point (taken is not among them): the target intent when the point has a target, then
    sticky high-temperature intents up to cfg.cf_branches - 1 in all."""
    out = []
    if pt.target is not None:
        out.append(TargetIntent(pt.target, h))
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
    stand-ins in tests."""
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
        for intent in intents(pt, cfg, rng, h):
            b = branch(S, ctx, pt, d, hp0, intent, policy, keep=int(cfg.cf_distill_steps))
            b['intent'] = intent
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
    return dict(depths=1, gain0=gain, d=d, k=0, taken=taken, taken_steps=steps, best=runs[bi], gain=gain, rep=rep,
                spread=spread, improving=improving, scores=scores, ends=[b_['end'] for b_ in runs])


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
