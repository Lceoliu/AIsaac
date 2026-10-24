"""Counterfactual branches at decision points (2026-10-07): the infrastructure of the user's 10-06 design, parts (a)-(b).

Pure reward is too sparse for long-horizon choices (an item's value shows minutes later). At a decision point the
engine is forked, each option is rolled forward with the CURRENT policy for minutes of game time with common random
numbers, and the outcome difference becomes a training target for that decision (the learning side, the bandit's
uncertainty term and the archive use of branch end states come later; here: points, branches, outcomes, records).

Branch points (detected by the worker in its own record stream, retrospectively at record t; no item knowledge):
  item       the inventory gained a collectible X at record t (EV_ITEM; the ids whose count went up) and a record
             t_touch of the same room showed a pedestal (pickup 5.100) holding X within ITEM_REACH px of the player
             (PedestalTracker: the last such record; the pedestal empties at the touch, the inventory shows the item a
             few decisions later). The decision is d = t_touch - cfg.branch_item_back; the choice record's ROW is
             record t_touch. Options: 0 take (the episode's own continuation), 1 skip.
  item_left  the room changed at record t (not a floor change) while record t-1 still shows a pedestal holding an item
             (the player walks out leaving it). Options: 0 leave (the episode's own), 1 take.
  door       the room changed at record t (not a floor change, no pedestal left) and record t-1 shows at least one open,
             unlocked door to an unvisited room other than the door the player went through (the nearest one).
             Options: 0 the door taken, then the alternatives (in random order).
  exit       the floor changed (run mode's EV_EXIT): only counted (branches later).
Every branch starts from the same restored state: the room's parked entry (the hindsight teacher's restore point:
template or archive entry with the episode's reseed, or the room-entry snapshot) forked and `play`ed with the
episode's own actions up to decision d (exact: A19, the teacher probes), parked as S. Each branch is a fork of S that
plays a short scripted PREFIX and then becomes a BRANCH EPISODE:
  - the episode's own option (option 0): the episode's own actions d .. t-1 replayed (play): the branch's first record
    is the episode's record t;
  - skip (item): the same replay after `pickup_block` (abp_bridge.lua): the engine's MC_PRE_PICKUP_COLLISION answers
    false for the player's contacts with that one pedestal (by InitSeed and room / position) for the rest of the
    branch; in AB+ v1.06 the player then bumps into it as into a shop item it cannot afford, nothing is taken, nothing
    else changes (checked in-engine 2026-10-07: nil takes the item; false keeps it on its pedestal; true lets the
    player pass and the pedestal vanishes). The pedestal stays visible to the policy, which may push against it;
  - another door / take (item_left): a scripted walk (walk_to: breadth-first path over the room's walkable cells,
    moves toward the next cell, straight at the target from its cell) until the room changes / the item is gained,
    at most cfg.branch_walk_seconds; a walk that does not get there marks the branch invalid. This is the only
    scripted control, and only for the branch's first seconds.
  After its prefix a branch reseeds the game's global MT with the point's reseed R (the same for all branches of the
  point, ABP_RESEED in place = what fork(reseed=R) does): common random numbers from the branch episode's first record
  on (prefixes run on the episode's own MT state, so the replayed option-0 prefix equals the episode). A branch
  episode is an ordinary episode for the trainer (records in the worker's slots, actions from the actor), tagged with
  ROW 'point' and 'branch' (its number from 1), and ends at death, the floor's exit or a cleared boss room (run mode),
  cfg.branch_seconds of game time, or the episode's own stall / time rules (done 3). The branch's own ends give the
  last record done = DONE_BRANCH (a truncation: train_tok bootstraps there).
Outcome of a branch: the components COMP (raw counts over prefix + branch episode) and their weighted sum with
  cfg.branch_weights (default the floor reward's terms: the branch's undiscounted return). Progressive widening: the
  first two options; while the largest |outcome - outcome of option 0| is below cfg.branch_close and options remain,
  one more, up to cfg.branch_max; with probability cfg.branch_noise one replicate of option 0 (same prefix, same R):
  its difference to option 0 is the noise floor of a label (policy sampling + the game's divergence after it).
  (Phase B2 replaces the outcome by the score and the replicate by pairs: see the second docstring below.)
Choice record (one per processed point, ring CHOICE_RING per worker, like the teacher's): the ROW of record t-1 (doors,
  item_left: what the policy saw just before the step that left the room) or t_touch (item) and CHOICE_META float64
  values: CHOICE_HEAD
  header, then per branch COMP_F components (see choice_meta).
Scheduling: per worker a queue (cfg.branch_queue points, each holding its room's parked entry) ordered by priority =
  novelty + cfg.branch_uncert x uncertainty(point); novelty = 1 / sqrt(1 + points of the same key processed by this
  worker), key = ('item', collectible id) or ('door', room type behind the door taken, alternatives); uncertainty (Phase
  B2) is the choice heads' spread at the point's decision record. Points run between episodes while the worker's
  branch time is below cfg.branch_share of its wall time.
"""
"""
Phase B2 (2026-10-07, the same day): the paired counterfactual made informative, and the learning side.
  Common random numbers for the policy: a branch PAIR (the episode's own option and an alternative) shares the engine's
    reseed R and a policy-noise key K = crn_key(point, pair) carried in ROW 'crn' of every record of its branch episodes;
    the actor's sampling uniforms of such a record are tok_policy.crn_uniforms(K, ROW 't', slot) (GraphActor), so two
    branches of a pair draw the same noise at the same decision index, and two branches whose states coincide play the
    same actions (a twin, cfg.branch_twin: same option, same R, same K: it must play byte-identically).
  Score with a value bootstrap: a branch's score is the prefix's reward (undiscounted, a few decisions) plus gamma^pre x
    (the branch episode's discounted return over its records + gamma^T x V(its last record) when it ended truncated:
    cap, boss, floor exit in run mode). V is the actor's value of that record (the learner's weights of the last
    update), read by the worker from the aux block the actor writes. Rewards are the trainer's (cfg.branch_reward),
    gamma the trainer's: the score estimates Q^gamma(S, option) of the current policy, in PPO's advantage units. The raw
    components and the undiscounted weighted outcome stay in the record. cfg.branch_seconds 60 (was 180).
  Pairs: pair 1 (options 0 and 1, then for doors the widening over more options, all with pair 1's R and K); with
    probability cfg.branch_noise one unconditional replicate pair (options 0 and 1 with a new R and K: the noise floor of
    a label, var(D1 - D2) / 2); then while |mean D| < 2 sigma / sqrt(pairs) (sigma: the trainer's pooled noise estimate
    of the kind, aux block; cfg.branch_noise_prior before there is one) up to cfg.branch_replicate_max conditional pairs.
    COMP 'replicate': 0 pair 1, 1 the unconditional replicate pair, 2 a conditional pair, 3 the twin.
  Decision record (the choice record's ROW, the choice head's input and the advantage correction's sample): the record
    whose decided action is applied in the step where the options part: item: record d (its action is the touch,
    one decision of lag); item_left / door: record t - 2 (its action is the step that leaves the room). The worker
    keeps copies of the last records and the actor's log-probability, choice prediction and spread for each record.
  item_left takes skip a pedestal whose pickup has a price > 0 (Entity_Pickup::GetPrice, +0xC30, Lua .Price: a shop
    item; devil deals have negative prices and stay) and one with no path from the player (Walker); another pedestal
    of the room qualifies instead.
"""
import math
import zlib
from collections import deque

import numpy as np

KIND_ITEM, KIND_DOOR, KIND_ITEM_LEFT, KIND_EXIT = 1, 2, 3, 4
KIND_NAMES = {KIND_ITEM: 'item', KIND_DOOR: 'door', KIND_ITEM_LEFT: 'item_left', KIND_EXIT: 'exit'}
KIND_IDS = {v: k for k, v in KIND_NAMES.items()}
ITEM_REACH = 100.0          # px: a gained collectible's pedestal must be this close to the player at record t-1
CHOICE_RING = 64            # choice records per worker the trainer may fall behind by
BRANCH_MAX = 10             # branches in one choice record at most (pairs, widening, the twin)
# per-branch components (raw; the prefix's are included), then (Phase B2) the score's parts and the pair bookkeeping
COMP = ('exit', 'boss', 'room', 'explore', 'damage', 'hurt', 'death', 'timeout', 'items', 'decisions',
        'prefix_decisions', 'end', 'took', 'option', 'replicate', 'valid',
        'pre_r', 'ret', 'v_end', 'boot', 'gamma_t', 'score', 'pair', 'opt', 'crn', 'reseed',
        'tw_steps', 'tw_same', 'tw_act_diff', 'version', 'archived', 'snap', 'snap_miss', 'walk_left',
        'walk_dist', 'score10', 'score20', 'score30')
# the score bootstrapped at shorter horizons too (s of the branch episode: its return up to that record + gamma^h x the
# actor's value of that record; a branch that ended earlier keeps its score): the horizon / noise trade-off
SCORE_HORIZONS = ((10, 'score10'), (20, 'score20'), (30, 'score30'))
COMP_F = len(COMP)
CI = {k: i for i, k in enumerate(COMP)}
REP_FIRST, REP_NOISE, REP_COND, REP_TWIN = 0, 1, 2, 3
# end codes (COMP 'end'): how a branch ended
END_DEATH, END_EXIT, END_BOSS, END_CAP, END_STALL, END_PREFIX_DEATH, END_INVALID, END_STOP, END_ERROR = range(1, 10)
# header of a choice record (Phase B2 appended: the decision record and the actor's outputs there)
HEAD = ('point', 'kind', 'branches', 'options', 'key', 't', 'reseed', 'version0', 'version1', 'seed', 'episode',
        'stage', 'room', 'worker', 'prep_s', 'wall_s',
        'rec', 'rec_fallback', 'pend0', 'pend1', 'pend2', 'pend3', 'pend4', 'act0', 'act1', 'act2', 'act3', 'act4',
        'logp_b', 'pred', 'uncert', 'noise_var', 'pairs', 'shop_skip', 'unreach_skip')
CHOICE_HEAD = len(HEAD)
HI = {k: i for i, k in enumerate(HEAD)}
CHOICE_META = CHOICE_HEAD + BRANCH_MAX * COMP_F
DEFAULT_WEIGHTS = 'exit=3,boss=2,room=1,explore=0.5,damage=1,hurt=0.75,death=2,timeout=1,time=0.002'
# the aux block (Phase B2): float64 [workers + 1, AUX_F]. Row w: what the actor answered for worker w's last record
# (value, log-probability of the sampled action, choice prediction mean and spread), the weight snapshot worker w asks
# for its branch records (AUX_REQ, written by the worker) and the one the actor used (AUX_USED); row `workers`: the
# trainer's pooled noise variance of a label per kind id (columns 1 item, 2 door, 3 item_left; 0 = none yet), the
# current weight snapshot (AUX_SNAP) and whether there are snapshots (AUX_SNAP_ON).
# Weight snapshots (train_tok --branch-snap-every): the actor keeps two frozen copies of the weights, refreshed in turn
# every that many updates; a point pins the newest one at its start and every record of its branch episodes is
# answered with it, so all branches of a point (a pair, its twin) are played by the same policy although PPO updates
# the weights every few seconds. A branch with an answer from another snapshot (its own was replaced: the point ran
# longer than a period) counts COMP 'snap_miss' and is no label.
AUX_F = 8
AUX_VALUE, AUX_LOGP, AUX_PRED, AUX_UNCERT, AUX_REQ, AUX_USED = 0, 1, 2, 3, 4, 5
AUX_SNAP, AUX_SNAP_ON = 6, 7
PRICE_LUA = ("local s = '' for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 5 and e.Variant == 100 and "
             "e.SubType > 0 then local pk = e:ToPickup() s = s .. e.SubType .. ',' .. e.Position.X .. ',' .. "
             "e.Position.Y .. ',' .. tostring(pk.Price) .. ';' end end return s")


def crn_key(pid, pair):
    """The policy-noise key of a point's branch pair (never 0)."""
    from .tok_policy import _mix64
    z = int(_mix64(np.array([(int(pid) * 64 + int(pair)) & 0x7FFFFFFFFFFFFFFF], np.uint64))[0])
    return (z & 0x7FFFFFFFFFFFFFFF) or 1


def parse_reward(text):
    """The trainer's reward terms ('name=weight,...') as a dict with every floor term."""
    w = {k: 0.0 for k in ('exit', 'boss', 'room', 'explore', 'damage', 'hurt', 'death', 'timeout', 'time')}
    for kv in (text or DEFAULT_WEIGHTS).split(','):
        if kv.strip():
            k, v = kv.split('=')
            w[k.strip()] = float(v)
    return w


def step_reward(damage, hurt, done, events, w, run):
    """The trainer's reward of the decision before a record (train_tok Episodes.arrive, floor / run modes)."""
    r = w['damage'] * damage - w['hurt'] * hurt - w['time']
    if done == 1:
        r += w['exit']
    elif done == 2:
        r -= w['death']
    elif done == 3:
        r -= w['timeout']
    r += w['room'] * bool(events & 1) + w['explore'] * bool(events & 2) + w['boss'] * bool(events & 4)
    if run:
        r += w['exit'] * bool(events & 8)
    return r


def label_valid(c):
    """A branch whose score is a label: its prefix did what the option is (or the player died in it) and its episode
    ran to one of its own ends."""
    end = int(c[CI['end']])
    return end in (END_DEATH, END_EXIT, END_BOSS, END_CAP, END_STALL, END_PREFIX_DEATH) and c[CI['snap_miss']] == 0


def pair_diffs(comps, a=1, b=0, field='score'):
    """Per pair (in order): score(option index a) - score(option index b) over the branches of comps (COMP_F vectors)
    that are not twins, both label-valid; [(pair, replicate code, diff)]. field: the COMP score column."""
    by = {}
    for c in comps:
        if int(c[CI['replicate']]) == REP_TWIN:
            continue
        by.setdefault(int(c[CI['pair']]), {})[int(c[CI['opt']])] = c
    out = []
    for p in sorted(by):
        x = by[p]
        if a in x and b in x and label_valid(x[a]) and label_valid(x[b]):
            out.append((p, int(x[a][CI['replicate']]), float(x[a][CI[field]] - x[b][CI[field]])))
    return out
# the bridge's move codes: 0 none, then clockwise from up
MOVES = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
CELL = 40.0


def parse_weights(text):
    w = {k: 0.0 for k in ('exit', 'boss', 'room', 'explore', 'damage', 'hurt', 'death', 'timeout', 'time')}
    for kv in (text or DEFAULT_WEIGHTS).split(','):
        if kv.strip():
            k, v = kv.split('=')
            w[k.strip()] = float(v)
    return w


def outcome(comp, w):
    """The weighted outcome of one branch's components (a COMP_F vector)."""
    c = comp
    return (w['exit'] * c[CI['exit']] + w['boss'] * c[CI['boss']] + w['room'] * c[CI['room']]
            + w['explore'] * c[CI['explore']] + w['damage'] * c[CI['damage']] - w['hurt'] * c[CI['hurt']]
            - w['death'] * c[CI['death']] - w['timeout'] * c[CI['timeout']]
            - w['time'] * (c[CI['decisions']] + c[CI['prefix_decisions']]))


def parse_kinds(text):
    return {KIND_IDS[k.strip()] for k in (text or '').split(',') if k.strip() in KIND_IDS}


class Point:
    """A queued decision point: what is needed to restore it and to run its branches."""
    __slots__ = ('kind', 'pid', 'base', 'reseed', 'offset', 'applied', 'd', 't', 'row', 'key', 'item', 'visited',
                 'stage', 'seed', 'episode', 'room', 'n_alt', 'order',
                 # Phase B2: the decision record (index, copy in row, fallback flag), the action under way there and
                 # the action decided there (5 each), the actor's log-probability of it, prediction and spread there
                 'rec', 'rec_fallback', 'pend', 'act', 'logp_b', 'pred', 'uncert')

    def __init__(self, **kw):
        self.rec, self.rec_fallback, self.pend, self.act = -1, 0, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)
        self.logp_b = self.pred = self.uncert = 0.0
        for k, v in kw.items():
            setattr(self, k, v)


class PointQueue:
    """The worker's candidate points, at most `cap`; the lowest priority one makes room (its parked entry released)."""

    def __init__(self, cap, uncert_coef=0.0):
        self.cap, self.uncert_coef = max(int(cap), 1), float(uncert_coef)
        self.items = []
        self.counts = {}       # key -> points of that key processed by this worker

    def __len__(self):
        return len(self.items)

    def uncertainty(self, point):
        """The choice heads' spread at the point's decision record (Phase B2: the actor's output there, read by the
        worker from the aux block; 0 without choice heads or without a pedestal in that record)."""
        return float(point.uncert or 0.0)

    def priority(self, point):
        return 1.0 / math.sqrt(1.0 + self.counts.get(point.key, 0)) + self.uncert_coef * self.uncertainty(point)

    def push(self, point):
        """Queue a point; returns the points dropped (to be released)."""
        self.items.append(point)
        dropped = []
        while len(self.items) > self.cap:
            j = min(range(len(self.items)), key=lambda i: (self.priority(self.items[i]), self.items[i].order))
            dropped.append(self.items.pop(j))
        return dropped

    def pop(self):
        j = max(range(len(self.items)), key=lambda i: (self.priority(self.items[i]), self.items[i].order))
        p = self.items.pop(j)
        self.counts[p.key] = self.counts.get(p.key, 0) + 1
        return p

    def clear(self):
        out, self.items = self.items, []
        return out


def nearest_door(row):
    """Index of the door nearest to the player in a ROW record (doors' relative positions), -1 without doors."""
    k = int(row['n_doors'][0])
    if k <= 0:
        return -1
    d = row['doors'][0][:k]
    return int(np.argmin(d[:, 0] ** 2 + d[:, 1] ** 2))


def door_alternatives(row, a):
    """Doors of a record other than `a` that are open, unlocked and lead to an unvisited room."""
    k = int(row['n_doors'][0])
    d = row['doors'][0][:k]
    return [j for j in range(k) if j != a and d[j, 2] > 0.5 and d[j, 3] < 0.5 and d[j, 5] < 0.5]


def pedestals(row):
    """(entity index, collectible id, distance px) of the pedestals with an item in a ROW record."""
    n = int(row['n_ent'][0])
    ids = row['ent_id'][0][:n]
    it = row['ent_item'][0][:n, 0]
    sel = np.flatnonzero((ids[:, 0] == 5) & (ids[:, 1] == 100) & (it > 0))
    return [(int(j), int(it[j]), float(row['ent'][0][j, 2]) * 300.0) for j in sel]


class PedestalTracker:
    """Per episode (items): the last record at which each collectible id stood on a pedestal within ITEM_REACH px of
    the player, and a copy of that record. In AB+ the pedestal is emptied at the touch and the inventory block shows the
    item a few decisions later (seen 2026-10-07: the pedestal read 0 two decisions before EV_ITEM), so an item point's
    decision is the last record that still showed the item next to the player. Reset at room changes."""

    def __init__(self):
        self.seen = {}

    def reset(self):
        if self.seen:
            self.seen = {}

    def update(self, row, t):
        n = int(row['n_ent'][0])
        it = row['ent_item'][0][:n, 0]
        if not it.any():
            return
        for j, item, dist in pedestals(row):
            if dist < ITEM_REACH:
                self.seen[item] = (t, row[0].copy())

    def touch(self, item):
        """(record, its copy) of the last record with `item` next to the player, or None."""
        return self.seen.get(item)


def gained_items(before, after):
    """Collectible ids whose count went up between two inventory dicts."""
    before = before or {}
    return [i for i, c in (after or {}).items() if c > before.get(i, 0)]


def inv_counts(o):
    if o.inv is None:
        return {}
    return {int(i): int(c) for i, c in zip(o.inv['id'], o.inv['count'])}


def move_to(px, py, x, y, tol=6.0):
    sx = 0 if abs(x - px) < tol else (1 if x > px else -1)
    sy = 0 if abs(y - py) < tol else (1 if y > py else -1)
    return MOVES[(sx, sy)]


class Walker:
    """A scripted walk to a point of the room (walk_to): breadth-first over the walkable cells of the lean terrain."""

    def __init__(self, terrain):
        self.w, self.h = int(terrain['width']), int(terrain['height'])
        self.free = np.zeros(self.w * self.h, bool)
        self.pos = {}
        x0 = y0 = None
        for c in terrain['cells']:
            index, x, y, collision, inside, walkable, pit, hazard = c[:8]
            self.pos[int(index)] = (float(x), float(y))
            if int(index) == 0:
                x0, y0 = float(x), float(y)
            self.free[int(index)] = bool(inside) and bool(walkable) and not pit and not hazard
        self.origin = (x0 if x0 is not None else 0.0, y0 if y0 is not None else 0.0)

    def cell(self, x, y):
        col = int(round((x - self.origin[0]) / CELL))
        r = int(round((y - self.origin[1]) / CELL))
        return min(max(r, 0), self.h - 1) * self.w + min(max(col, 0), self.w - 1)

    def nearest_free(self, x, y):
        idx = np.flatnonzero(self.free)
        if not len(idx):
            return self.cell(x, y)
        r, c = np.divmod(idx, self.w)
        cx = self.origin[0] + c * CELL
        cy = self.origin[1] + r * CELL
        return int(idx[np.argmin((cx - x) ** 2 + (cy - y) ** 2)])

    def path(self, a, b):
        """Cells from a to b (8-connected, no corner cutting), [] when b cannot be reached."""
        if a == b:
            return [a]
        prev = {a: None}
        q = deque([a])
        w, h = self.w, self.h
        while q:
            u = q.popleft()
            ur, uc = divmod(u, w)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                r, c = ur + dr, uc + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                v = r * w + c
                if v in prev or not self.free[v]:
                    continue
                if dr and dc and not (self.free[ur * w + c] and self.free[r * w + uc]):
                    continue
                prev[v] = u
                if v == b:
                    out = [v]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                q.append(v)
        return []

    def move(self, px, py, tx, ty):
        """The move toward (tx, ty) from (px, py)."""
        here = self.cell(px, py)
        if not self.free[here]:
            here = self.nearest_free(px, py)
        goal = self.nearest_free(tx, ty)
        p = self.path(here, goal)
        if len(p) <= 1:
            return move_to(px, py, tx, ty)
        nx, ny = self.pos.get(p[1], (tx, ty))
        return move_to(px, py, nx, ny, tol=4.0)


def walk_to(clone, decoder, o, target, until, cap, fpd, lean_step):
    """Scripted walk of a lean clone toward target (x, y) until until(o) or `cap` decisions. Returns (o, decisions,
    reached). The path is recomputed every decision on the current terrain."""
    walker, key = None, None
    for k in range(cap):
        if o.dead:
            return o, k, False
        if until(o):
            return o, k, True
        if walker is None or key != o.terrain_key:
            walker, key = Walker(o.terrain), o.terrain_key
        p = o.players[0]
        m = walker.move(float(p['x']), float(p['y']), target[0], target[1])
        o = lean_step(clone, decoder, (m, 0, 0, 0, 0), fpd)
    return o, cap, bool(until(o))


def restore(point, fpd, alarm):
    """The point's decision state: a lean fork of its parked room entry (reseeded as the episode was when the entry is
    its start), the episode's actions offset .. d-1 played. Returns (S, its decoder, the observation at decision d)."""
    from .abplus_lean import LeanDecoder, read_lean
    from .tok_sampler import play_actions
    S = point.base.clone.fork(reseed=point.reseed, lean=True, alarm=alarm)
    try:
        dec = LeanDecoder()
        if point.d > point.offset:
            o = play_actions(S, dec, point.applied[point.offset:point.d], fpd)
        else:
            S._send({"cmd": "obs"})
            o = read_lean(S, dec)
    except BaseException:
        try:
            S.close()
        except OSError:
            pass
        raise
    return S, dec, o


def pedestal_prices(S):
    """[(collectible id, x, y, price)] of the room's pedestals holding an item, read from the engine (Lua .Price)."""
    out = []
    for part in str(S.lua(PRICE_LUA) or '').split(';'):
        f = part.split(',')
        if len(f) == 4:
            try:
                out.append((int(float(f[0])), float(f[1]), float(f[2]), int(float(f[3]))))
            except ValueError:
                pass
    return out


def options_of(point, oS, rng, S=None, notes=None):
    """(how, target (x, y), option id) per option of a point in its restored state oS, the episode's own first:
    replay / block (item), replay / walk_take (item_left), replay / walk_door per other door to an unvisited room in
    random order (doors; option id = door slot). [] when the pedestal or the doors are not found.
    item_left with the restored clone S given (Phase B2): pedestals with a price > 0 (shop items) and pedestals without
    a path from the player are not taken; the point's own collectible if it qualifies, else the nearest one that does
    (point.item is set to it); notes (a dict) counts 'shop_skip' and 'unreach_skip'."""
    p0 = oS.players[0]
    px, py = float(p0['x']), float(p0['y'])
    ents = oS.entities
    if point.kind == KIND_ITEM_LEFT and S is not None:
        notes = notes if notes is not None else {}
        prices = pedestal_prices(S)
        walker = Walker(oS.terrain) if oS.terrain is not None else None
        here = None
        if walker is not None:
            here = walker.cell(px, py)
            if not walker.free[here]:
                here = walker.nearest_free(px, py)
        ok = []
        for item, x, y, price in prices:
            if price > 0:
                notes['shop_skip'] = notes.get('shop_skip', 0) + 1
                continue
            if walker is not None and not walker.path(here, walker.nearest_free(x, y)):
                notes['unreach_skip'] = notes.get('unreach_skip', 0) + 1
                continue
            ok.append((item != point.item, (x - px) ** 2 + (y - py) ** 2, item, x, y))
        if not ok:
            return []
        _, _, item, x, y = min(ok)
        point.item = item
        return [('replay', (x, y), 0), ('walk_take', (x, y), 1)]
    if point.kind in (KIND_ITEM, KIND_ITEM_LEFT):
        sel = np.flatnonzero((ents['type'] == 5) & (ents['variant'] == 100) & (ents['subtype'] == point.item))
        if not len(sel):
            return []
        j = sel[np.argmin((ents['x'][sel] - px) ** 2 + (ents['y'][sel] - py) ** 2)]
        where = (float(ents['x'][j]), float(ents['y'][j]))
        if point.kind == KIND_ITEM:
            return [('replay', where, 0), ('block', where, 1)]
        return [('replay', where, 0), ('walk_take', where, 1)]
    doors = oS.doors
    if not len(doors):
        return []
    a = int(np.argmin((doors['x'] - px) ** 2 + (doors['y'] - py) ** 2))
    out = [('replay', (float(doors['x'][a]), float(doors['y'][a])), int(doors['slot'][a]))]
    alts = [j for j in range(len(doors)) if j != a and doors['open'][j] and not doors['locked'][j]
            and not (doors['seen'][j] & 1)]
    for j in rng.permutation(len(alts)):
        k = alts[int(j)]
        out.append(('walk_door', (float(doors['x'][k]), float(doors['y'][k])), int(doors['slot'][k])))
    return out


def prefix(S, oS, point, option, reseed, fpd, alarm, walk_cap):
    """One branch's start: a lean fork B of the restored state S (observation oS), the option's prefix, then (when the
    prefix did what the option is) the reseed and B's first observation. Returns (B, decoder, first observation or
    None, comp (COMP_F, the prefix's part), status, notes) with status 'ok', 'prefix_death' or 'invalid' and notes a
    dict (walk_fail, skip_took, block_found, o_prefix: the observation at the prefix's end). The caller closes B."""
    from .abplus_lean import LeanDecoder, read_lean
    from .tok_sampler import lean_step, play_actions
    comp = np.zeros(COMP_F, np.float64)
    comp[CI['option']] = option[2]
    notes = dict(walk_fail=False, skip_took=False, block_found=None)
    B = S.fork(lean=True, alarm=alarm)
    try:
        dec = LeanDecoder()
        how, target = option[0], option[1]
        inv0 = inv_counts(oS)
        room0, stage0 = oS.room[4], oS.room[5]
        reached = True
        if how in ('replay', 'block'):
            if how == 'block':
                found = B.pickup_block(subtype=point.item, x=target[0], y=target[1])
                reached = bool(found.get('found'))
                notes['block_found'] = reached
            o = play_actions(B, dec, point.applied[point.d:point.t], fpd)
            pre = point.t - point.d
        else:
            B._send({"cmd": "obs"})
            o = read_lean(B, dec)
            if how == 'walk_door':
                def until(x):
                    return x.room[4] != room0 or x.room[5] != stage0
            else:   # walk_take
                def until(x):
                    return inv_counts(x).get(point.item, 0) > inv0.get(point.item, 0)
            o, pre, reached = walk_to(B, dec, o, target, until, walk_cap, fpd, lean_step)
            notes['walk_fail'] = not reached
            # Phase B2 diagnostics: the walk ended in another room / this far from its target (px)
            comp[CI['walk_left']] = o.room[4] != room0 or o.room[5] != stage0
            p_ = o.players[0]
            comp[CI['walk_dist']] = math.hypot(float(p_['x']) - target[0], float(p_['y']) - target[1])
        notes['o_prefix'] = o
        took = point.kind != KIND_DOOR and inv_counts(o).get(point.item, 0) > inv0.get(point.item, 0)
        changed = o.room[4] != room0 or o.room[5] != stage0
        comp[CI['prefix_decisions']] = pre
        comp[CI['took']] = took
        comp[CI['hurt']] += float(o.damage_taken - oS.damage_taken)
        if changed and o.room[5] == stage0 and o.room[4] not in point.visited:
            comp[CI['explore']] += 1
        if point.kind == KIND_ITEM:
            valid = reached and took == (how == 'replay')
            notes['skip_took'] = how == 'block' and took
        elif point.kind == KIND_ITEM_LEFT:
            valid = (took and how == 'walk_take') or (how == 'replay' and changed and not took)
        else:
            valid = changed and reached
        if o.dead:
            comp[CI['death']] = 1
            comp[CI['end']] = END_PREFIX_DEATH
            return B, dec, None, comp, 'prefix_death', notes
        if not valid:
            comp[CI['end']] = END_INVALID
            return B, dec, None, comp, 'invalid', notes
        comp[CI['valid']] = 1
        notes['changed'] = changed
        B.reseed(reseed)   # common random numbers from the branch episode's first record on
        B._send({"cmd": "obs"})
        return B, dec, read_lean(B, dec), comp, 'ok', notes
    except BaseException:
        try:
            B.close()
        except OSError:
            pass
        raise


def choice_meta(point, worker, comps, reseed, v0, v1, prep_s, wall_s, n_options, extra=None):
    """The CHOICE_META float64 vector of a processed point (comps: per branch COMP_F vectors, in run order); extra:
    more head fields (Phase B2: noise_var, pairs, shop_skip, unreach_skip)."""
    m = np.zeros(CHOICE_META, np.float64)
    head = dict(point=point.pid, kind=point.kind, branches=len(comps), options=n_options, key=point.item, t=point.t,
                reseed=reseed, version0=v0, version1=v1, seed=point.seed, episode=point.episode, stage=point.stage,
                room=point.room, worker=worker, prep_s=prep_s, wall_s=wall_s, rec=point.rec,
                rec_fallback=point.rec_fallback, logp_b=point.logp_b, pred=point.pred, uncert=point.uncert)
    for j in range(5):
        head[f'pend{j}'] = point.pend[j] if j < len(point.pend) else 0
        head[f'act{j}'] = point.act[j] if j < len(point.act) else 0
    head.update(extra or {})
    for k, v in head.items():
        m[HI[k]] = float(v)
    for j, c in enumerate(comps[:BRANCH_MAX]):
        m[CHOICE_HEAD + j * COMP_F:CHOICE_HEAD + (j + 1) * COMP_F] = c
    return m


def meta_comps(m):
    """The COMP_F vectors of a choice record's branches."""
    n = min(int(m[HI['branches']]), BRANCH_MAX)
    return [m[CHOICE_HEAD + j * COMP_F:CHOICE_HEAD + (j + 1) * COMP_F] for j in range(n)]


def take_labels(m, field='score'):
    """The choice heads' labels of a choice record (Phase B2): [(pair, replicate code, D)] with D = score(take) -
    score(skip) for the item kinds (item: option index 0 took; item_left: option index 1 takes) and score(alternative)
    - score(own) for doors; the sign s with which D turns into own - alternative (+1 item, -1 item_left and doors)."""
    kind = int(m[HI['kind']])
    comps = meta_comps(m)
    if kind == KIND_ITEM:
        return pair_diffs(comps, 0, 1, field), 1.0
    return pair_diffs(comps, 1, 0, field), -1.0


def decode_meta(m, weights=None):
    """A choice record's meta as a dict (head fields, branches: list of {component: value}, outcomes)."""
    w = parse_weights(weights) if isinstance(weights, str) or weights is None else weights
    out = {k: float(m[HI[k]]) for k in HEAD}
    out['kind_name'] = KIND_NAMES.get(int(out['kind']), '?')
    br = []
    for j in range(min(int(out['branches']), BRANCH_MAX)):
        c = m[CHOICE_HEAD + j * COMP_F:CHOICE_HEAD + (j + 1) * COMP_F]
        b = {k: float(c[i]) for k, i in CI.items()}
        b['outcome'] = float(outcome(c, w))
        br.append(b)
    out['branches_detail'] = br
    return out
