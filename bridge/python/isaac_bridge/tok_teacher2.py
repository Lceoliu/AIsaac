"""Teacher v2 (2026-10-10, rl/docs/SCALING_THESIS.md S5b): search findings enter the policy update as whole trajectories.

Why: the hindsight teacher (tok_sampler.hindsight) keeps finding moves the policy does not know (98% of hurts
"avoidable"), but the policy's prior on them does not rise (progress.csv safe_mass_new flat at 0.61 over a game year):
only ONE decision per search is distilled, while the search validated a route of several steps that a memoryless
policy cannot execute from that one label; the candidates are nine hand-written held moves, and the value head never
sees what the search found.

Decision points (tok_floor queues them per episode, like the teacher's hurts): a hurt, the fatal step of a death (the
death teacher's `fatal` items) and, with cfg.teacher_v2_random > 0, random records (that many per game hour of play).
Restore: the room's parked entry (the teacher's restore point) forked with the episode's reseed, its actions replayed
to decision d (exact: A19); for a hurt d = hurt record - 1 - k over the depths (cfg.teacher_depths, the fatal step
cfg.teacher_death_depths when set, cfg.teacher_v2_depths overrides the hurts'); the first depth with an improving
candidate ends the point. A random point has the one depth d = its record.
Candidates at d (Intent): the nine held moves, each for h decisions (h drawn from cfg.teacher_v2_hold), and
cfg.teacher_v2_sticky_n "sticky policy" intents (for max(hold) decisions the policy's own sampled move is kept for a
geometric number of decisions of mean cfg.teacher_v2_sticky). An intent constrains ONLY the movement: shooting, bombs,
the active item and pills are the policy's answer at every step, from the branch's own state (the policy samples at
temperature 1; no high-temperature sampling, user 2026-10-10). After the intent the policy plays on. A branch runs
cfg.teacher_v2_seconds of game time or to its episode's end (death, the floor's exit, the stall / time rules).
The policy: in training the actor itself, through a second channel per worker (the "lane": one record slot, the action
under way and the answer in shared memory, a pipe of its own; the search thread of the worker asks, train_tok answers
with the actor's weights and never stores the records in the rollout). Probes use a stand-in with the same interface
(request(row, under_way); answer() -> (action 5-tuple, value, log-probability)).
Score: the trainer's reward terms (tok_branch.step_reward with cfg.branch_reward; heal / resource terms are not in it)
discounted with cfg.branch_gamma, + gamma^T V(last record) when the branch is cut by the time cap (V: the actor's value
of that record). The episode's own continuation ("taken") is scored over the same window from the episode's records
(the worker keeps each record's damage, hurt, done code, events and the actor's value). The best candidate is replayed
once with another reseed of the game's MT (the noise floor: spread = |best - replicate|); the point improves only if
best - taken > cfg.teacher_v2_margin and > spread.
Records: for an improving point EVERY step of the best branch is an imitation record (V2 ring, like the teacher's):
the record's ROW (as the policy saw it), the action under way and the applied action (move, shoot, bomb, item, pill),
weight min(best - taken, cfg.teacher_v2_cap) and the branch's discounted return-to-go G at that step (+ the meta below).
train_tok (--teacher-v2 1): -w * log pi(a | obs) summed over the heads + --teacher-v2-vf * (V(obs) - G)^2, merged into
the PPO minibatch steps (teacher_v2_loss).
Records match the episode: the branch's EpisodeState starts at the episode's decision index (t = d: the time features),
with the room's visited set and floor of the restore point, the room's starting monster HP (the damage share's
denominator, from the restore point's first observation) and the episode's stall counter at d (logged per record).
abplus_probe_teacher_v2.py checks this (a branch replaying the episode's own actions = the episode's records and score).
"""
import os
import time

import numpy as np

from .abplus_lean import LeanDecoder, lean_logic_frames, read_lean, read_lean_raw
from .tok_branch import step_reward
from .tok_obs import RC_HP0, RC_PROGRESS, ROW, EpisodeState, FastRow, encode_row

V2_RING = 384        # imitation records per worker the trainer may fall behind by (a point writes <= cap of them)
# per imitation record (float64): the action under way and the applied one (5 each, items / pill 0 without items),
# the weight, the return-to-go G, the point's gain (best - taken), spread (|best - replicate|), taken and best scores,
# the step j in the branch and its length n, the depth k, the hold h, the intent (0-8 a held move, 9 sticky), the
# point's kind, its id, the worker, the actor's weights when the point started (control[3]), the actor's value and
# log-probability at that record, the replicate's score
V2_FIELDS = ('pend0', 'pend1', 'pend2', 'pend3', 'pend4', 'act0', 'act1', 'act2', 'act3', 'act4', 'weight', 'ret',
             'gain', 'spread', 'taken', 'best', 'j', 'n', 'depth', 'hold', 'intent', 'kind', 'point', 'worker',
             'version', 'value', 'logp', 'rep',
             'trig')   # B34: the counterfactual teacher's trigger type (tok_cf.TRIGGERS index + 1; 0: teacher v2's)
V2_META = len(V2_FIELDS)
VI = {k: i for i, k in enumerate(V2_FIELDS)}
KIND_HURT, KIND_FATAL, KIND_RANDOM = 1, 2, 3
KIND_CF = 4          # B34 (2026-10-16): a counterfactual teacher's choice point (tok_cf.py)
INTENT_STICKY = 9
INTENT_TARGET = 10   # B34: the move toward the choice point's target (tok_cf.TargetIntent)
INTENT_BOMB = 11     # B40 (2026-10-18): the walk to a candidate secret-room wall and one bomb (tok_cf.BombIntent)
END_DONE, END_DEATH, END_CAP, END_STOP = 1, 2, 3, 4
END_EXIT = 5         # B34 (ctx.exit_end, run mode): the floor's exit reached (the score bootstraps from that record)
# lane int32 per worker: the action under way (5, written by the worker), the answer (5, the actor); B34: [10] the
# request's sequence number (the worker's; the actor echoes it in LANE_OUT [4]: an answer to an older request, e.g.
# of a search child killed while it waited, is recognised and skipped), [11] the temporal begin (written by the worker
# before a branch's first request: the bank row + 1 of the snapshot to start from, -1: a reset state; read and cleared
# by the actor)
LANE_IO = 12
# lane float64 per worker: the actor's value and log-probability of the answer, its weights, value_global (B34; the
# value where the model has no global token), the request's sequence number (B34),
# completed-search epoch (worker writes last, after publishing imitation records).
LANE_OUT = 6


def v2_sizes(workers):
    """Shared-memory block sizes (bytes) of attach_v2's arrays, in its order."""
    return (ROW.itemsize * workers, 4 * workers * LANE_IO, 8 * workers * LANE_OUT, 8 * workers,
            ROW.itemsize * workers * V2_RING, 8 * workers * V2_RING * V2_META)


def attach_v2(names, workers):
    """(blocks, lane rows [workers] ROW, lane io int32 [workers, LANE_IO], lane out float64 [workers, LANE_OUT], the
    actor's value of each worker's last answered ordinary record float64 [workers], v2 rows [workers, V2_RING] ROW,
    v2 meta float64 [workers, V2_RING, V2_META])."""
    from multiprocessing import shared_memory
    blocks = [shared_memory.SharedMemory(name=n) for n in names[:6]]   # (more names: attach_v2_mem's blocks)
    lane_rows = np.ndarray((workers,), ROW, buffer=blocks[0].buf)
    lane_io = np.ndarray((workers, LANE_IO), np.int32, buffer=blocks[1].buf)
    lane_out = np.ndarray((workers, LANE_OUT), np.float64, buffer=blocks[2].buf)
    main_value = np.ndarray((workers,), np.float64, buffer=blocks[3].buf)
    rows = np.ndarray((workers, V2_RING), ROW, buffer=blocks[4].buf)
    meta = np.ndarray((workers, V2_RING, V2_META), np.float64, buffer=blocks[5].buf)
    return blocks, lane_rows, lane_io, lane_out, main_value, rows, meta


def v2_mem_sizes(workers, dim):
    """2026-10-11 (memory, train_tok --memory gru): shared-memory block sizes (bytes) of attach_v2_mem's arrays."""
    return 4 * workers * dim, 4 * workers * (dim + 1), 4 * workers * V2_RING * dim


def attach_v2_mem(names, workers, dim):
    """2026-10-11 (memory): (blocks, main_h float32 [workers, dim]: the GRU input state of each worker's last answered
    ordinary record (written by the actor before its reply, like main_value), lane_h float32 [workers, dim + 1]: the
    lane's state (the worker writes a branch's start state and sets column dim = 1, LanePolicy.begin; the actor then
    writes the input state of each answer there), v2_h float32 [workers, V2_RING, dim]: each imitation record's input
    state)."""
    from multiprocessing import shared_memory
    blocks = [shared_memory.SharedMemory(name=n) for n in names[:3]]
    main_h = np.ndarray((workers, dim), np.float32, buffer=blocks[0].buf)
    lane_h = np.ndarray((workers, dim + 1), np.float32, buffer=blocks[1].buf)
    v2_h = np.ndarray((workers, V2_RING, dim), np.float32, buffer=blocks[2].buf)
    return blocks, main_h, lane_h, v2_h


def gains_of(row):
    """(health units, coins + bombs + keys) of a record from its player token (train_tok Episodes.cur_gains)."""
    pl = row['player'][0] if getattr(row, 'shape', ()) == (1,) else row['player']
    return (float(np.rint(pl[7] * 12.0)),
            float(np.rint(pl[8] * 10.0) + np.rint(pl[9] * 10.0) + np.rint(pl[10] * 50.0)))


def gain_reward(g_prev, g, rw):
    """B34: the trainer's heal / resource reward between two records' gains (gains only; 0 without the terms)."""
    return rw.get('heal', 0.0) * max(g[0] - g_prev[0], 0.0) + rw.get('resource', 0.0) * max(g[1] - g_prev[1], 0.0)


def parse_ints(text, default=()):
    """'2,4,8' -> (2, 4, 8); '' -> default."""
    out = tuple(int(v) for v in str(text or '').split(',') if v.strip())
    return out or tuple(default)


def five(a):
    """An action as a 5-tuple of ints (move, shoot, bomb, item, pill)."""
    a = tuple(int(v) for v in a)
    return a[:5] + (0,) * (5 - len(a[:5]))


class V2Stop(Exception):
    """The worker is stopping: a lane answer will not come."""


class V2Log:
    """The episode's per-record values the taken score needs: damage share, half hearts lost, done code, events, the
    stall counter after the record (EpisodeState.progress_t) and the actor's value of the record (answered ones); h
    (2026-10-11, memory): the GRU input state the actor used for the record (answered ones; empty without memory);
    vg (B34): the actor's value_global of the record (answered ones; with the counterfactual teacher only); gain_r
    (B34, add's rw given): the trainer's heal / resource reward of the step before the record (gain_reward; 0 for the
    episode's first record)."""
    __slots__ = ('damage', 'hurt', 'done', 'events', 'progress', 'value', 'h', 'vg', 'gain_r', '_g')

    def __init__(self):
        self.damage, self.hurt, self.done, self.events, self.progress, self.value = [], [], [], [], [], []
        self.h = []
        self.vg = []
        self.gain_r, self._g = [], None

    def add(self, row, done, progress, rw=None):
        self.damage.append(float(row['damage'][0]))
        self.hurt.append(float(row['hurt'][0]))
        self.done.append(int(done))
        self.events.append(int(row['events'][0]))
        self.progress.append(int(progress))
        if rw is not None:   # B34: the heal / resource term (train_tok Episodes: gains since the record before)
            g = gains_of(row)
            self.gain_r.append(gain_reward(self._g, g, rw) if self._g is not None else 0.0)
            self._g = g


class V2Point:
    """A queued decision point of teacher v2 (B34: and of the counterfactual teacher, kind KIND_CF: trig its trigger
    type, target the trigger's target (tok_cf.detect), bank its temporal snapshot's bank row or None)."""
    __slots__ = ('base', 'reseed', 'applied', 'log', 'at', 'offset', 'kind', 'seed', 'episode', 'visited', 'stage0',
                 'pid', 'version', 'trig', 'target', 'bank')

    def __init__(self, **kw):
        self.pid, self.version = 0, 0
        self.trig, self.target, self.bank = 0, None, None
        for k, v in kw.items():
            setattr(self, k, v)


class Intent:
    """A candidate's movement constraint for its first h decisions (kind 'hold': move m every step; 'sticky': the
    policy's sampled move kept for a geometric number of decisions of mean `mean`, from its own rng); the rest of the
    action is always the policy's, and after h decisions the policy's whole action."""

    def __init__(self, kind, move=0, h=4, mean=4.0, seed=0, eps=0.0):
        self.kind, self.move, self.h, self.mean, self.seed = kind, int(move), int(h), float(mean), int(seed)
        self.eps = float(eps)   # B34 (sticky, the counterfactual teacher's "high temperature"): at each re-pick the
        #                         move is uniform over the nine with this probability, else the policy's (0: as before)
        self.reset()

    @property
    def code(self):
        return self.move if self.kind == 'hold' else INTENT_STICKY

    def reset(self):
        self.rng = np.random.default_rng([self.seed, 0x51C])
        self.left, self.cur = 0, 0

    def compose(self, j, a, row=None):
        """The applied action of branch step j given the policy's answer a (5-tuple) (row: the record answered; B34's
        intents read it, these do not)."""
        if j >= self.h:
            return a
        if self.kind == 'hold':
            return (self.move,) + a[1:]
        if self.left <= 0:
            self.cur = a[0]
            if self.eps > 0 and self.rng.random() < self.eps:
                self.cur = int(self.rng.integers(9))
            self.left = int(self.rng.geometric(1.0 / max(self.mean, 1.0)))
        self.left -= 1
        return (self.cur,) + a[1:]


def candidates(cfg, rng):
    """The intents at one depth: the nine held moves (h drawn per move from cfg.teacher_v2_hold) and
    cfg.teacher_v2_sticky_n sticky-policy intents (for max(hold) decisions) when cfg.teacher_v2_sticky > 0."""
    holds = tuple(cfg.teacher_v2_hold) or (4,)
    out = [Intent('hold', m, int(holds[int(rng.integers(len(holds)))])) for m in range(9)]
    if cfg.teacher_v2_sticky > 0:
        for _ in range(int(cfg.teacher_v2_sticky_n)):
            out.append(Intent('sticky', 0, max(holds), cfg.teacher_v2_sticky, int(rng.integers(1, 2 ** 31 - 1))))
    return out


class Context:
    """What the branches of one worker share: settings, the reward terms, the encoder, the record slot.
    B34 (2026-10-16, the counterfactual teacher reuses it): seconds (None: cfg.teacher_v2_seconds) the branches' cap,
    gamma (None: cfg.branch_gamma) the discount, boot_vg: the cap bootstraps from the actor's value_global (LanePolicy.vg)
    instead of its value, exit_end: a record with EV_EXIT (run mode: the next floor) ends a branch (and the taken
    window) like a cap: its reward counted, then gamma^T x the bootstrap of that record; gains: the trainer's heal /
    resource terms are part of the score (branches: gain_reward between records; taken: the log's gain_r)."""

    def __init__(self, cfg, limit, stall, run, fast_fn, group, row, rw, seconds=None, gamma=None, boot_vg=False,
                 exit_end=False, gains=False):
        from .tok_branch import parse_reward
        self.cfg, self.limit, self.stall, self.run, self.fast_fn, self.group = cfg, limit, stall, run, fast_fn, group
        self.row = row                                   # a length-1 ROW view (the lane slot, or a probe's buffer)
        self.address = row.ctypes.data
        self.rw = rw if isinstance(rw, dict) else parse_reward(rw)
        self.gamma = float(cfg.branch_gamma if gamma is None else gamma)
        self.fpd, self.items, self.alarm = cfg.frames_per_decision, bool(cfg.items), cfg.clone_alarm
        sec = cfg.teacher_v2_seconds if seconds is None else seconds
        self.cap = max(1, int(round(sec * 30 / cfg.frames_per_decision)))
        self.boot_vg, self.exit_end = bool(boot_vg), bool(exit_end)
        self.gains = bool(gains) and (self.rw.get('heal', 0.0) != 0 or self.rw.get('resource', 0.0) != 0)
        self.stats = dict(branches=0, decisions=0, frames=0, replicates=0)


def taken_score(log, d, cap, gamma, rw, run, values=None, exit_end=False, extra=None):
    """The episode's own continuation from record d over at most cap steps: (discounted return + gamma^m V(record
    d + m) when that record is not an end, steps m). The value of a record that has none (the log broke off) is 0.
    values (B34): the bootstrap values per record (None: log.value); exit_end (B34): a record with EV_EXIT ends the
    window like the cap (its reward counted, then its value); extra (B34): a reward per record added to the step's
    (the log's gain_r: heal / resource)."""
    from .tok_obs import EV_EXIT
    n = len(log.done)
    vals = log.value if values is None else values
    g, disc, m = 0.0, 1.0, 0
    for i in range(1, cap + 1):
        t = d + i
        if t >= n:
            break
        g += disc * (step_reward(log.damage[t], log.hurt[t], log.done[t], log.events[t], rw, run)
                     + (extra[t] if extra is not None and t < len(extra) else 0.0))
        disc *= gamma
        m = i
        if log.done[t]:
            return g, m
        if exit_end and log.events[t] & EV_EXIT:
            break
    t = d + m
    v = vals[t] if t < len(vals) else 0.0
    return g + disc * v, m


def returns_to_go(rews, gamma, v_end):
    """G [n + 1] float64 of a branch's rewards (rews[i]: the reward of the step after record i): G[n] = v_end (the
    bootstrap at a cut, 0 at an end), G[i] = rews[i] + gamma G[i + 1]; the branch's score is G[0]."""
    n = len(rews)
    G = np.zeros(n + 1, np.float64)
    G[n] = v_end
    for i in range(n - 1, -1, -1):
        G[i] = rews[i] + gamma * G[i + 1]
    return G


def play_branch(S, ctx, pt, d, hp0, intent, policy, reseed=None, keep=True, trace=None):
    """One branch from the restored clone S (in the state of record d; S itself is not changed: the branch is a fork):
    the intent composed with the policy's answers, cut at ctx.cap steps. Returns a dict: score (= G[0]), steps n, end
    code, boot (cut by the cap), v_end, and with keep the records 0 .. n-1 (rows: ROW copies, pend, act: 5-tuples,
    ret: return-to-go, value, logp per record). trace (a list): each record's ROW copy and the reward (probes).
    B34: keep may be an int k (True: all): only the first k records are kept (the counterfactual teacher distils
    those); the intent's compose sees the record it acts on; pt.bank (when set) names the lane's temporal snapshot
    (policy.tbegin); with ctx.boot_vg the cap's bootstrap is the actor's value_global; with ctx.exit_end a record with
    EV_EXIT ends the branch (END_EXIT, bootstrapped); 't' (temporal lanes): each kept record's (o_prev, has, o).
    B40: an intent with keep_from() (tok_cf.Deferral) has its records kept from its first own decision j0 on (none
    while it returns None); 'from' = j0 (None: it never started), 'ret' the return-to-go of the kept records."""
    from .tok_obs import EV_EXIT
    from .tok_sampler import applied_action, lean_step, step_action
    B = S.fork(lean=True, alarm=ctx.alarm)
    intent.reset()
    out = dict(score=0.0, n=0, end=END_STOP, boot=False, v_end=0.0, frames=0)
    rows, pends, acts, vals, lps, rews, hs, ts = [], [], [], [], [], [], [], []
    n_keep = (1 << 62) if keep is True else int(keep or 0)
    keep_from = getattr(intent, 'keep_from', None)   # B40: a deferred intent's first own decision (None: not yet)
    kidx = []
    begin = getattr(policy, 'begin', None)
    if begin is not None:   # 2026-10-11 (memory): the branch's GRU state starts as a copy of the episode's at record d
        log_h = getattr(pt.log, 'h', None) or []
        begin(log_h[d] if d < len(log_h) else None)
    tbegin = getattr(policy, 'tbegin', None)
    if tbegin is not None:   # B34 (temporal): the lane's state from the snapshot of record d (None: a reset state)
        tbegin(getattr(pt, 'bank', None))
    try:
        if reseed is not None:
            B.reseed(reseed)
        dec = LeanDecoder()
        B._send({"cmd": "obs"})
        ob = read_lean(B, dec)
        lf0 = lf = ob.logic_frames
        st = EpisodeState(ctx.limit, floor=True, stall=ctx.stall, run=ctx.run, items=ctx.items,
                          stall_pickups=getattr(ctx.cfg, 'stall_progress', 'events') == 'pickups')
        st.visited = set(pt.visited)
        st.stage0 = int(pt.stage0) if pt.stage0 is not None else int(ob.room[5])
        st.progress_t = pt.log.progress[d - 1] if d >= 1 else d   # the stall counter before record d
        enc = FastRow(ctx.fast_fn, st, dec, pt.episode, pt.seed, ctx.group) if ctx.fast_fn is not None else None
        row = ctx.row
        item, t, j = ob, d, 0
        under = five(pt.applied[d])   # record d's action under way (the episode's own: decided before d)
        while True:
            if enc is not None:
                done = enc.encode(item, row, ctx.address, t)
            else:
                done = encode_row(item, st, row, t)
                row['episode'], row['seed'], row['group'], row['first'] = pt.episode, pt.seed, ctx.group, 0
            if j == 0:   # the episode's room state at d: the room's starting monster HP, the stall counter after d
                st.hp0 = hp0
                st.progress_t = pt.log.progress[d]
                if enc is not None:
                    enc.ctx[RC_HP0], enc.ctx[RC_PROGRESS] = hp0, st.progress_t
                g_prev = gains_of(row) if ctx.gains else None
            else:
                rews.append(step_reward(float(row['damage'][0]), float(row['hurt'][0]), done,
                                        int(row['events'][0]), ctx.rw, ctx.run))
                if g_prev is not None:   # B34: the heal / resource terms
                    g_now = gains_of(row)
                    rews[-1] += gain_reward(g_prev, g_now, ctx.rw)
                    g_prev = g_now
            if trace is not None:
                trace.append((row[0].copy(), rews[-1] if j else 0.0))
            if done:
                out['end'] = END_DEATH if done == 2 else END_DONE
                break
            policy.request(row, under)
            exit_now = ctx.exit_end and j > 0 and bool(int(row['events'][0]) & EV_EXIT)
            if j >= ctx.cap or exit_now:   # cut: only this record's value (the bootstrap)
                _, v, _ = policy.answer()
                v = getattr(policy, 'vg', v) if ctx.boot_vg else v
                out['end'], out['boot'], out['v_end'] = END_EXIT if exit_now else END_CAP, True, float(v)
                break
            if enc is not None:   # the step of the action under way, while the policy answers
                step_action(B, under, ctx.items, ctx.fpd)
                item = read_lean_raw(B)
                lf = lean_logic_frames(item)
            else:
                item = lean_step(B, dec, applied_action(under, ctx.items), ctx.fpd)
                lf = item.logic_frames
            a, v, lp = policy.answer()
            act = intent.compose(j, five(a), row)
            k0 = keep_from() if keep_from is not None else 0
            if k0 is not None and j - k0 < n_keep:
                kidx.append(j)
                rows.append(row[0].copy())
                pends.append(under)
                acts.append(act)
                vals.append(float(v))
                lps.append(float(lp))
                hs.append(getattr(policy, 'h_used', None))   # (memory: the answer's GRU input state; else None)
                ts.append(getattr(policy, 't_used', None))   # (B34, temporal: the lane step's outputs; else None)
            under = act
            t += 1
            j += 1
    finally:
        try:
            B.close()
        except OSError:
            pass
    n = len(rews)
    G = returns_to_go(rews, ctx.gamma, out['v_end'] if out['boot'] else 0.0)
    out.update(score=float(G[0]), n=n, frames=lf - lf0, rews=rews,
               **{'from': keep_from() if keep_from is not None else 0})
    if keep:
        k = min(n, len(rows))
        out.update(rows=rows[:k], pend=pends[:k], act=acts[:k], ret=G[kidx[:k]].copy(), value=vals[:k], logp=lps[:k],
                   h=hs[:k] if hs and hs[0] is not None else None, t=ts[:k] if ts and ts[0] is not None else None)
    ctx.stats['branches'] += 1
    ctx.stats['decisions'] += n
    ctx.stats['frames'] += lf - lf0
    return out


def restore(pt, d, ctx):
    """The point's state at record d: (S, the room's starting monster HP). S is a lean fork of the parked room entry
    (pt.base: a tok_floor.Parked, forked under its lock), reseeded as the episode was, the episode's actions
    pt.offset .. d-1 played. The caller closes S. B37 (2026-10-16): S lives as long as the point's branches
    (ctx.alarm = cfg.clone_alarm real seconds, not 120): the counterfactual teacher forks every branch and the
    replicate from S one after the other, and under a slow actor (C100: 16 ms calls, 44 s lane wait per point) a
    point's 4 x 180 game s + replicate outlived the 120-s clone watchdog, the next fork found S gone ("connection
    closed by game": 231 of 5,190 points lost at u721; C99 at 10 s per point: 0)."""
    from .tok_sampler import play_actions
    S = pt.base.fork(reseed=pt.reseed, lean=True, alarm=(max(int(ctx.alarm), 120) if int(ctx.alarm) > 0 else 0))
    try:
        dec = LeanDecoder()
        S._send({"cmd": "obs"})
        o0 = read_lean(S, dec)   # the restore point = the room's first record: its monsters' HP is the room's start
        hp0 = max(float(o0.monsters_hp), 1.0)
        if d > pt.offset:
            play_actions(S, dec, pt.applied[pt.offset:d], ctx.fpd)
    except BaseException:
        try:
            S.close()
        except OSError:
            pass
        raise
    return S, hp0


def point_depths(pt, cfg):
    """The depths searched for a point, in order."""
    if pt.kind == KIND_RANDOM:
        return (0,)
    if pt.kind == KIND_FATAL and cfg.teacher_death_depths:
        return tuple(cfg.teacher_death_depths)
    return tuple(cfg.teacher_v2_depths) or tuple(cfg.teacher_depths)


def search_point(pt, ctx, policy, rng, on_depth=None):
    """Teacher v2 at one point: per depth the candidates and the taken score; the first depth whose best candidate
    beats the taken continuation by more than the margin and its replicate's spread ends the search. Returns a dict:
    depths tried, gain0 (best - taken at the first depth), and for the last depth tried: d, k, taken, best (the best
    branch's play_branch dict + 'intent'), gain, rep (its replicate's score or None), spread, improving. None when no
    depth was possible (d before the restore point). on_depth (probes): called with (k, d, taken, branches, S, hp0)."""
    cfg = ctx.cfg
    res = None
    for k in point_depths(pt, cfg):
        d = pt.at - 1 - k
        if d < pt.offset + 1:
            break
        S, hp0 = restore(pt, d, ctx)
        try:
            taken, steps = taken_score(pt.log, d, ctx.cap, ctx.gamma, ctx.rw, ctx.run)
            best, scores = None, []
            for intent in candidates(cfg, rng):
                b = play_branch(S, ctx, pt, d, hp0, intent, policy)
                b['intent'] = intent
                scores.append(b['score'])
                if best is None or b['score'] > best['score']:
                    best = b
            gain = best['score'] - taken
            rep, spread = None, None
            if gain > cfg.teacher_v2_margin:   # the noise floor: the best intent again from another reseed
                r = play_branch(S, ctx, pt, d, hp0, best['intent'], policy,
                                reseed=int(rng.integers(1, 2 ** 31 - 1)), keep=False)
                ctx.stats['replicates'] += 1
                rep, spread = r['score'], abs(best['score'] - r['score'])
            improving = rep is not None and gain > spread
            depths = (res['depths'] if res else 0) + 1
            res = dict(depths=depths, gain0=res['gain0'] if res else gain, d=d, k=k, taken=taken, taken_steps=steps,
                       best=best, gain=gain, rep=rep, spread=spread, improving=improving, scores=scores)
            if on_depth is not None:
                on_depth(k, d, taken, scores, S, hp0)
        finally:
            try:
                S.close()
            except OSError:
                pass
        if res['improving']:
            break
    return res


def records_of(res, ctx, pt, worker, cap_w, steps=None):
    """The imitation records of an improving point: (rows [n] ROW, meta [n, V2_META] float64). steps (B34): the first
    that many of the best branch's records only (None: all it kept); meta 'n' stays the branch's length."""
    b = res['best']
    n = min(b['n'], len(b['rows']))
    if steps is not None:
        n = min(n, int(steps))
    rows = np.zeros(n, ROW)
    meta = np.zeros((n, V2_META), np.float64)
    w = min(res['gain'], cap_w)
    for j in range(n):
        rows[j] = b['rows'][j]
        m = meta[j]
        m[VI['pend0']:VI['pend0'] + 5] = b['pend'][j]
        m[VI['act0']:VI['act0'] + 5] = b['act'][j]
        m[VI['ret']], m[VI['value']], m[VI['logp']] = b['ret'][j], b['value'][j], b['logp'][j]
        m[VI['j']] = j
    meta[:, VI['weight']] = w
    meta[:, VI['n']] = b['n']
    for k, v in (('gain', res['gain']), ('spread', res['spread']), ('taken', res['taken']), ('best', b['score']),
                 ('depth', res['k']), ('hold', b['intent'].h), ('intent', b['intent'].code), ('kind', pt.kind),
                 ('point', pt.pid), ('worker', worker), ('version', pt.version), ('rep', res['rep']),
                 ('trig', int(getattr(pt, 'trig', 0) or 0) + 1 if pt.kind == KIND_CF else 0)):
        meta[:, VI[k]] = v
    return rows, meta


class LanePolicy:
    """The actor through a worker's lane (training): request() writes the action under way and wakes the actor (the
    record is already in the lane slot), answer() waits for its answer (stop(): give up, V2Stop).
    lane_h (2026-10-11, memory; attach_v2_mem's, None without): begin(h0) hands the actor a branch's start state (a copy
    of the episode's at the fork; None: zeros), so the branch never touches the worker's own state; after each answer
    h_used is the GRU input state the actor used for it.
    B34 (2026-10-16): every request carries a sequence number (LANE_IO [10], random per request) that the actor echoes
    (LANE_OUT [4]); an answer with another number belongs to an older request (a search child killed while it waited,
    a V2Stop) and is skipped (a stand-in actor that echoes nothing leaves 0 there: accepted). After each answer vg is
    the actor's value_global of the record (its value without a global token). Temporal models: tbegin(row) before a
    branch's first request names the snapshot the lane's temporal state starts from (bank row; None: a reset state);
    lane_t (cf_attach's [workers, 2 D + 2] float32, None without): after each answer t_used is (o_prev [D], has, o [D])
    of the lane's temporal step (the stored outputs of an imitation record)."""

    def __init__(self, conn, lane_io, lane_out, index, stop, lane_h=None, lane_t=None):
        self.conn, self.io, self.out, self.i, self.stop = conn, lane_io, lane_out, index, stop
        self.wait_s = 0.0
        self.calls = 0
        self.lane_h, self.h_used = lane_h, None
        self.lane_t, self.t_used = lane_t, None
        self.vg = 0.0
        self.seq = 0
        if lane_h is None:   # (without memory: play_branch finds no begin)
            self.begin = None

    def begin(self, h0):
        dim = self.lane_h.shape[1] - 1
        self.lane_h[self.i, :dim] = 0.0 if h0 is None else h0
        self.lane_h[self.i, dim] = 1.0   # (read and cleared by the actor at the branch's first request)

    def tbegin(self, bank_row):
        """B34: the lane's temporal state at the branch's first request: a copy of the snapshot in bank row bank_row
        (the parent episode's slot before the point's record), or a reset state (None)."""
        if self.io.shape[1] > 11:
            self.io[self.i, 11] = -1 if bank_row is None or bank_row < 0 else int(bank_row) + 1

    def request(self, row, under_way):
        self.io[self.i, :5] = under_way
        if self.io.shape[1] > 10:
            self.seq = int.from_bytes(os.urandom(4), 'little') & 0x7FFFFFFF or 1
            self.io[self.i, 10] = self.seq
        self.conn.send_bytes(b'0')

    def answer(self):
        t0 = time.perf_counter()
        while True:
            while not self.conn.poll(0.05):
                if self.stop():
                    raise V2Stop()
            self.conn.recv_bytes()
            echo = int(self.out[self.i, 4]) if self.out.shape[1] > 4 else 0
            if echo == 0 or echo == self.seq:
                break
        self.wait_s += time.perf_counter() - t0
        self.calls += 1
        if self.lane_h is not None:   # (memory)
            self.h_used = self.lane_h[self.i, :self.lane_h.shape[1] - 1].copy()
        if self.lane_t is not None:   # (B34, temporal)
            self.t_used = self.lane_t[self.i].copy()
        self.vg = float(self.out[self.i, 3]) if self.out.shape[1] > 3 else float(self.out[self.i, 0])
        return tuple(int(v) for v in self.io[self.i, 5:10]), float(self.out[self.i, 0]), float(self.out[self.i, 1])


def teacher_v2_loss(logp, value, weight, ret, share, coef, vf):
    """train_tok's teacher-v2 term on b imitation records (torch tensors [b]): logp = sum over the heads of
    log pi(a_head | obs) of the recorded actions, value V(obs), weight w, ret G. Returns (share x (coef x mean(-w logp)
    + vf x mean((V - G)^2)), stats [3]: mean(-w logp), mean((V - G)^2), mean(logp), detached). share: the records'
    weight in the minibatch (train_tok: --teacher-v2-weight x b / minibatch decisions)."""
    import torch
    pi = -(weight * logp).mean()
    v = ((value.float() - ret) ** 2).mean()
    return share * (coef * pi + vf * v), torch.stack([pi, v, logp.mean()]).detach()
