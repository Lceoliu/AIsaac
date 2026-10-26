"""Retention of teacher / v2 (CF) records' sources (approved policy, 2026-10-24).

1. A branch record references its ancestors' history (source keys by owner,
   RawHistory.origin); it owns only the frames its branch produced. A pickled
   prefix copy is replaced by references to an existing holder's arrays.
2. After the update in which a record's loss was first used, its own source
   graphs are released (the horizon's settle-then-detach path, frontier = inf);
   its values stay readable; later samples read it without source gradient.
   Ring overwrite or episode end releases everything (unchanged).
3. Record-owned raw frames + source values have a byte budget; above it the
   oldest v2 records are evicted whole (teacher records own one value frame).
4. Which records train, branch reads, PPO/teacher semantics: unchanged.
"""
import numpy as np

from .rebuild_memory import share_prefix


class RecordLedger:
    def __init__(self, teach_buffer, v2_buffer, budget_bytes, row_bytes, block):
        self.budget, self.row_bytes, self.block = budget_bytes, row_bytes, block
        self.teach_cons = np.zeros(teach_buffer, bool); self.teach_now = np.zeros(teach_buffer, bool)
        self.v2_cons = np.zeros(v2_buffer, bool); self.v2_now = np.zeros(v2_buffer, bool)
        self.v2_valid = np.zeros(v2_buffer, bool); self.v2_stamp = np.zeros(v2_buffer, np.int64)
        self.stamp = 0
        # (accounting) the update each record was admitted, and the waits of those consumed this update
        self.updates = 0
        self.teach_admit = np.zeros(teach_buffer, np.int64); self.v2_admit = np.zeros(v2_buffer, np.int64)
        self.waits = []
        self.donors = {}   # uid -> the longest known history of that episode (references only)
        self.shared_frames = self.evicted = 0
        self.value_bytes_frame = 0.
        self.last = {}

    # ---- rule 1: references, not copies -----------------------------------
    def on_rollout(self, histories):
        for h in histories:
            if h is not None and (h.uid not in self.donors or h.length > self.donors[h.uid].length):
                self.donors[h.uid] = h

    def share(self, contexts):
        out = []
        for h in contexts:
            if h is None or not h.origin and h.uid not in self.donors:
                out.append(h); continue
            best, gain = h, 0
            for uid in [h.uid] + [a for a, _ in h.origin]:
                donor = self.donors.get(uid)
                if donor is None or donor is h: continue
                shared, n = share_prefix(h, donor, self.block)
                if n > gain: best, gain = shared, n
            self.shared_frames += gain
            if best.uid not in self.donors or best.length > self.donors[best.uid].length:
                self.donors[best.uid] = best
            out.append(best)
        return out

    def prune(self, live_histories):
        # Only histories that are themselves live stay donors. Records already
        # reference their ancestors' arrays; keeping an ended ancestor's whole
        # history as a donor would hold frames no record shares (measured:
        # learner raw 1.23 GiB instead of 0.25 at update 18, 4-card run).
        keep = {h.uid for h in live_histories if h is not None}
        self.donors = {u: h for u, h in self.donors.items() if u in keep}

    # ---- rule 2: consumption -----------------------------------------------
    def admit_teacher(self, slots):
        self.teach_cons[slots] = self.teach_now[slots] = False
        self.teach_admit[slots] = self.updates

    def admit_v2(self, slots):
        self.v2_cons[slots] = self.v2_now[slots] = False
        self.v2_valid[slots] = True
        self.v2_admit[slots] = self.updates
        self.v2_stamp[slots] = self.stamp + np.arange(len(slots)); self.stamp += len(slots)

    def picked(self, teach=None, v2=None):
        if teach is not None: self.teach_now[teach] = True
        if v2 is not None: self.v2_now[v2] = True

    def frozen(self, teach=None, v2=None):
        """Indices (teacher picks first, then v2 picks) of records consumed in an
        earlier update: read without source gradient."""
        n = 0 if teach is None else len(teach)
        out = [] if teach is None else list(np.flatnonzero(self.teach_cons[teach]))
        if v2 is not None: out += [n + int(i) for i in np.flatnonzero(self.v2_cons[v2])]
        return out

    def end_update(self):
        new_t = self.teach_now & ~self.teach_cons; new_v = self.v2_now & ~self.v2_cons
        self.waits = ([('teach', int(w)) for w in self.updates - self.teach_admit[new_t]] +
                      [('v2', int(w)) for w in self.updates - self.v2_admit[new_v]])
        self.teach_cons |= self.teach_now; self.teach_now[:] = False
        self.v2_cons |= self.v2_now; self.v2_now[:] = False
        self.updates += 1

    def wait_stats(self, teach_valid=None):
        """(accounting) waits (updates from admission to first use) of the records first used in the last
        update, by kind, and the ages of the valid records still unconsumed."""
        out = {}
        for kind in ('teach', 'v2'):
            w = [x for k, x in self.waits if k == kind]
            out[f'record_{kind}_wait_median'] = float(np.median(w)) if w else float('nan')
            out[f'record_{kind}_wait_p90'] = float(np.percentile(w, 90)) if w else float('nan')
            out[f'record_{kind}_consumed_new'] = len(w)
        age = self.updates - self.v2_admit[self.v2_valid & ~self.v2_cons]
        out['record_v2_unconsumed'] = int(len(age))
        out['record_v2_unconsumed_age_max'] = int(age.max()) if len(age) else 0
        return out

    def frontier(self, frontier, teach_contexts, v2_contexts, horizon=0):
        """Frames each episode's pending losses can still reach, as half-open frame intervals
        {uid: [(lo, hi), ...]} (2026-10-25, the union of every consumer's reachable range; before: one
        earliest decision per episode, which kept everything after it).

        frontier: {uid: the rollout's first pending decision F} -> [(F - H, inf)] (H = horizon; 0: [(0, inf)]).
        An unconsumed record deciding at L = h.length reads, with gradient, frames [L - H, L] of its history
        and its current observation (frame L: the query, encoded with the record, possibly a chunk of its own):
        its own frames from its fork on, [max(own start, L - H), L + 1), and of each ancestor only the frames it
        shares, [L - H, shared count) (empty when the fork lies more than H before L). A consumed record adds
        nothing; an episode only consumed records refer to gets no interval (values only)."""
        inf = float('inf')
        out = {uid: [(f - horizon if horizon else 0, inf)] if f != inf else [] for uid, f in frontier.items()}
        consumed = set()
        for contexts, cons in ((teach_contexts, self.teach_cons), (v2_contexts, self.v2_cons)):
            for slot, h in enumerate(contexts or ()):
                if h is None: continue
                if cons[slot]:
                    consumed.add(h.uid); consumed.update(a for a, _ in h.origin); continue
                L = h.length; lo = L - horizon if horizon else 0
                own = (max(h.own_start(), lo), L + 1)   # (frame L: the record's own query, 2026-10-26)
                if own[0] < own[1]: out.setdefault(h.uid, []).append(own)
                else: out.setdefault(h.uid, [])
                for a, shared in h.origin:
                    span = (lo, min(shared, L))
                    if span[0] < span[1]: out.setdefault(a, []).append(span)
                    else: out.setdefault(a, [])
        for uid in consumed:
            out.setdefault(uid, [])
        return out

    # ---- rule 3: byte budget --------------------------------------------------
    def owned(self, v2_contexts, teach_n):
        frames = {}
        for slot in np.flatnonzero(self.v2_valid):
            h = v2_contexts[slot]
            if h is None: continue
            frames[h.uid] = max(frames.get(h.uid, 0), h.length + 1 - h.own_start())
        own = sum(frames.values())
        raw = own * self.row_bytes
        values = (own + teach_n) * self.value_bytes_frame
        return raw, values, own, len(frames)

    def enforce(self, v2_contexts, teach_n, value_bytes_frame):
        if value_bytes_frame: self.value_bytes_frame = value_bytes_frame
        raw, values, own, uids = self.owned(v2_contexts, teach_n)
        evicted = []
        while raw + values > self.budget and self.v2_valid.any():
            valid = np.flatnonzero(self.v2_valid)
            # oldest 1% (at least one) per pass: whole records, oldest first
            take = valid[np.argsort(self.v2_stamp[valid])[:max(1, len(valid) // 100)]]
            for slot in take:
                v2_contexts[slot] = None
            self.v2_valid[take] = False; evicted.extend(int(s) for s in take)
            raw, values, own, uids = self.owned(v2_contexts, teach_n)
        self.evicted += len(evicted)
        self.last = dict(record_bytes=raw + values, record_raw_bytes=raw, record_value_bytes=values,
                         record_own_frames=own, record_v2_episodes=uids, record_v2_live=int(self.v2_valid.sum()),
                         record_evicted=self.evicted, record_shared_frames=self.shared_frames,
                         record_v2_consumed=int((self.v2_cons & self.v2_valid).sum()),
                         record_teach_consumed=int(self.teach_cons.sum()))
        return evicted

    def sample_v2(self, rng, held, count):
        """The same draw as before while no record was evicted."""
        valid = np.flatnonzero(self.v2_valid[:held])
        if len(valid) == held: return rng.integers(0, held, count)
        return valid[rng.integers(0, len(valid), count)]

    def v2_held(self, held):
        return int(self.v2_valid[:held].sum())


# ---- rule 5: raw bytes of encoded, consumed frames ----------------------------
def release_raw(limit, block, slot_lists=(), context_lists=(), context_dicts=()):
    """Replace released blocks everywhere a process holds raw histories.

    slot_lists: lists of RawHistory (ActorMemory.raw); context_lists: lists of
    lists of prefixes (Rollout.raw_contexts); context_dicts: {key: RawHistory}.
    limit(uid): frames of that owner whose bytes nothing will read again."""
    from .rebuild_memory import released
    memo = {}
    for raw in slot_lists:
        if raw is None: continue
        for i, h in enumerate(raw): raw[i] = released(h, limit, block, memo)
    for rows in context_lists:
        if rows is None: continue
        for row in rows:
            for k, h in enumerate(row):
                if h is not None: row[k] = released(h, limit, block, memo)
    for table in context_dicts:
        if table is None: continue
        for key, h in list(table.items()):
            if h is not None: table[key] = released(h, limit, block, memo)


def lineage_intervals(slot_lists=(), context_dicts=()):
    """(ancestor uid, frames it shares, the branch's current length) of branches in flight (lanes, the
    choice-point bank, contexts awaiting publication): what in_flight_frontier needs."""
    # One triple per (ancestor, shared): the shortest branch, i.e. the widest [length - H, shared); the union
    # over longer branches' ranges is the same interval (2026-10-26: the per-context triples were ~10^5 a hand-off)
    low = {}
    for raw in slot_lists:
        for h in raw or ():
            if h is not None and h.length:
                for a, n in h.origin: low[a, n] = min(low.get((a, n), h.length), h.length)
    for table in context_dicts:
        for h in (table or {}).values():
            if h is not None:
                for a, n in h.origin: low[a, n] = min(low.get((a, n), h.length), h.length)
    return {(a, n, l) for (a, n), l in low.items()}


def in_flight_frontier(frontier, lineage, horizon):
    """(2026-10-26) The frames an in-flight branch's future records can still read of each ancestor: they
    decide at or after the branch's current length l, so at most [l - H, shared) (no horizon: [0, shared)).
    Added to frontier's intervals ({uid: [(lo, hi), ...]}); an ancestor with nothing reachable gets no
    interval (values only), as before."""
    widest = {}   # (several actors may report the same ancestor and shared count: the lowest lo)
    for a, shared, length in lineage:
        lo = length - horizon if horizon else 0
        if lo < shared: widest[a, shared] = min(widest.get((a, shared), lo), lo)
        elif a not in frontier: frontier[a] = []
    for (a, shared), lo in widest.items():
        reach = frontier.setdefault(a, [])
        if not isinstance(reach, list): reach = frontier[a] = []   # (a stale number: lists only here)
        reach.append((lo, shared))
    return frontier


def grace_keep(recent, uids, update, grace):
    """(rule 5, D2a) episodes seen in the last `grace` updates (in a rollout or in an actor's lineage report):
    recent {uid: last update seen} is refreshed with uids and pruned; returns the uids still kept. The learner
    keeps their source values; the frames a branch can still read come from its lineage report
    (in_flight_frontier), not from this."""
    for u in uids: recent[u] = update
    for u in [u for u, t in recent.items() if update - t > grace]: del recent[u]
    return set(recent)


def learner_frontier(requests, ledger, teach_contexts, v2_contexts, reach, values_only, horizon):
    """The learner's frontiers for SourcePipeline.retain_histories (2026-10-26, from train_tok's
    horizon_frontiers): each requested history's first decision in this rollout; the stored records'
    reachable ranges (RecordLedger.frontier); the in-flight branches' ancestors' ranges (in_flight_frontier,
    reach = (ancestor, shared, length) triples); an empty interval list (values only) for the other uids in
    values_only (in-flight ancestors without reachable frames, ended episodes in their grace period).
    requests: RawHistory objects. Returns (frontiers, the rollout's own {uid: first decision})."""
    frontier = {}
    for h in requests:
        frontier[h.uid] = min(frontier.get(h.uid, h.length), h.length)
    rollout = dict(frontier)
    frontier = ledger.frontier(frontier, teach_contexts, v2_contexts, horizon)
    in_flight_frontier(frontier, reach, horizon)
    for uid in values_only: frontier.setdefault(uid, [])
    return frontier, rollout


def lineage(slot_lists=(), context_dicts=()):
    """Ancestor episodes of branches in flight (lanes, the choice-point bank,
    contexts awaiting publication): the learner keeps their source values."""
    out = set()
    for raw in slot_lists:
        for h in raw or ():
            if h is not None and h.length: out.update(a for a, _ in h.origin)
    for table in context_dicts:
        for h in (table or {}).values():
            if h is not None: out.update(a for a, _ in h.origin)
    return out
