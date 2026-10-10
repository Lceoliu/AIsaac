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
             'version', 'value', 'logp', 'rep')
V2_META = len(V2_FIELDS)
VI = {k: i for i, k in enumerate(V2_FIELDS)}
KIND_HURT, KIND_FATAL, KIND_RANDOM = 1, 2, 3
INTENT_STICKY = 9
END_DONE, END_DEATH, END_CAP, END_STOP = 1, 2, 3, 4
LANE_IO = 10         # lane int32 per worker: the action under way (5, written by the worker), the answer (5, the actor)
LANE_OUT = 4         # lane float64 per worker: the actor's value and log-probability of the answer, its weights, 0


def v2_sizes(workers):
    """Shared-memory block sizes (bytes) of attach_v2's arrays, in its order."""
    return (ROW.itemsize * workers, 4 * workers * LANE_IO, 8 * workers * LANE_OUT, 8 * workers,
            ROW.itemsize * workers * V2_RING, 8 * workers * V2_RING * V2_META)


def attach_v2(names, workers):
    """(blocks, lane rows [workers] ROW, lane io int32 [workers, LANE_IO], lane out float64 [workers, LANE_OUT], the
    actor's value of each worker's last answered ordinary record float64 [workers], v2 rows [workers, V2_RING] ROW,
    v2 meta float64 [workers, V2_RING, V2_META])."""
    from multiprocessing import shared_memory
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    lane_rows = np.ndarray((workers,), ROW, buffer=blocks[0].buf)
    lane_io = np.ndarray((workers, LANE_IO), np.int32, buffer=blocks[1].buf)
    lane_out = np.ndarray((workers, LANE_OUT), np.float64, buffer=blocks[2].buf)
    main_value = np.ndarray((workers,), np.float64, buffer=blocks[3].buf)
    rows = np.ndarray((workers, V2_RING), ROW, buffer=blocks[4].buf)
    meta = np.ndarray((workers, V2_RING, V2_META), np.float64, buffer=blocks[5].buf)
    return blocks, lane_rows, lane_io, lane_out, main_value, rows, meta


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
    stall counter after the record (EpisodeState.progress_t) and the actor's value of the record (answered ones)."""
    __slots__ = ('damage', 'hurt', 'done', 'events', 'progress', 'value')

    def __init__(self):
        self.damage, self.hurt, self.done, self.events, self.progress, self.value = [], [], [], [], [], []

    def add(self, row, done, progress):
        self.damage.append(float(row['damage'][0]))
        self.hurt.append(float(row['hurt'][0]))
        self.done.append(int(done))
        self.events.append(int(row['events'][0]))
        self.progress.append(int(progress))


class V2Point:
    """A queued decision point of teacher v2."""
    __slots__ = ('base', 'reseed', 'applied', 'log', 'at', 'offset', 'kind', 'seed', 'episode', 'visited', 'stage0',
                 'pid', 'version')

    def __init__(self, **kw):
        self.pid, self.version = 0, 0
        for k, v in kw.items():
            setattr(self, k, v)


class Intent:
    """A candidate's movement constraint for its first h decisions (kind 'hold': move m every step; 'sticky': the
    policy's sampled move kept for a geometric number of decisions of mean `mean`, from its own rng); the rest of the
    action is always the policy's, and after h decisions the policy's whole action."""

    def __init__(self, kind, move=0, h=4, mean=4.0, seed=0):
        self.kind, self.move, self.h, self.mean, self.seed = kind, int(move), int(h), float(mean), int(seed)
        self.reset()

    @property
    def code(self):
        return self.move if self.kind == 'hold' else INTENT_STICKY

    def reset(self):
        self.rng = np.random.default_rng([self.seed, 0x51C])
        self.left, self.cur = 0, 0

    def compose(self, j, a):
        """The applied action of branch step j given the policy's answer a (5-tuple)."""
        if j >= self.h:
            return a
        if self.kind == 'hold':
            return (self.move,) + a[1:]
        if self.left <= 0:
            self.cur = a[0]
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
    """What the branches of one worker share: settings, the reward terms, the encoder, the record slot."""

    def __init__(self, cfg, limit, stall, run, fast_fn, group, row, rw):
        from .tok_branch import parse_reward
        self.cfg, self.limit, self.stall, self.run, self.fast_fn, self.group = cfg, limit, stall, run, fast_fn, group
        self.row = row                                   # a length-1 ROW view (the lane slot, or a probe's buffer)
        self.address = row.ctypes.data
        self.rw = rw if isinstance(rw, dict) else parse_reward(rw)
        self.gamma = float(cfg.branch_gamma)
        self.fpd, self.items, self.alarm = cfg.frames_per_decision, bool(cfg.items), cfg.clone_alarm
        self.cap = max(1, int(round(cfg.teacher_v2_seconds * 30 / cfg.frames_per_decision)))
        self.stats = dict(branches=0, decisions=0, frames=0, replicates=0)


def taken_score(log, d, cap, gamma, rw, run):
    """The episode's own continuation from record d over at most cap steps: (discounted return + gamma^m V(record
    d + m) when that record is not an end, steps m). The value of a record that has none (the log broke off) is 0."""
    n = len(log.done)
    g, disc, m = 0.0, 1.0, 0
    for i in range(1, cap + 1):
        t = d + i
        if t >= n:
            break
        g += disc * step_reward(log.damage[t], log.hurt[t], log.done[t], log.events[t], rw, run)
        disc *= gamma
        m = i
        if log.done[t]:
            return g, m
    t = d + m
    v = log.value[t] if t < len(log.value) else 0.0
    return g + disc * v, m


def play_branch(S, ctx, pt, d, hp0, intent, policy, reseed=None, keep=True, trace=None):
    """One branch from the restored clone S (in the state of record d; S itself is not changed: the branch is a fork):
    the intent composed with the policy's answers, cut at ctx.cap steps. Returns a dict: score (= G[0]), steps n, end
    code, boot (cut by the cap), v_end, and with keep the records 0 .. n-1 (rows: ROW copies, pend, act: 5-tuples,
    ret: return-to-go, value, logp per record). trace (a list): each record's ROW copy and the reward (probes)."""
    from .tok_sampler import applied_action, lean_step, step_action
    B = S.fork(lean=True, alarm=ctx.alarm)
    intent.reset()
    out = dict(score=0.0, n=0, end=END_STOP, boot=False, v_end=0.0, frames=0)
    rows, pends, acts, vals, lps, rews = [], [], [], [], [], []
    try:
        if reseed is not None:
            B.reseed(reseed)
        dec = LeanDecoder()
        B._send({"cmd": "obs"})
        ob = read_lean(B, dec)
        lf0 = lf = ob.logic_frames
        st = EpisodeState(ctx.limit, floor=True, stall=ctx.stall, run=ctx.run, items=ctx.items)
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
            else:
                rews.append(step_reward(float(row['damage'][0]), float(row['hurt'][0]), done,
                                        int(row['events'][0]), ctx.rw, ctx.run))
            if trace is not None:
                trace.append((row[0].copy(), rews[-1] if j else 0.0))
            if done:
                out['end'] = END_DEATH if done == 2 else END_DONE
                break
            policy.request(row, under)
            if j >= ctx.cap:   # cut: only this record's value (the bootstrap)
                _, v, _ = policy.answer()
                out['end'], out['boot'], out['v_end'] = END_CAP, True, float(v)
                break
            if enc is not None:   # the step of the action under way, while the policy answers
                step_action(B, under, ctx.items, ctx.fpd)
                item = read_lean_raw(B)
                lf = lean_logic_frames(item)
            else:
                item = lean_step(B, dec, applied_action(under, ctx.items), ctx.fpd)
                lf = item.logic_frames
            a, v, lp = policy.answer()
            act = intent.compose(j, five(a))
            if keep:
                rows.append(row[0].copy())
                pends.append(under)
                acts.append(act)
                vals.append(float(v))
                lps.append(float(lp))
            under = act
            t += 1
            j += 1
    finally:
        try:
            B.close()
        except OSError:
            pass
    n = len(rews)
    G = np.zeros(n + 1, np.float64)
    G[n] = out['v_end'] if out['boot'] else 0.0
    for i in range(n - 1, -1, -1):
        G[i] = rews[i] + ctx.gamma * G[i + 1]
    out.update(score=float(G[0]), n=n, frames=lf - lf0, rews=rews)
    if keep:
        out.update(rows=rows[:n], pend=pends[:n], act=acts[:n], ret=G[:n].copy(), value=vals[:n], logp=lps[:n])
    ctx.stats['branches'] += 1
    ctx.stats['decisions'] += n
    ctx.stats['frames'] += lf - lf0
    return out


def restore(pt, d, ctx):
    """The point's state at record d: (S, the room's starting monster HP). S is a lean fork of the parked room entry
    (pt.base: a tok_floor.Parked, forked under its lock), reseeded as the episode was, the episode's actions
    pt.offset .. d-1 played. The caller closes S."""
    from .tok_sampler import play_actions
    S = pt.base.fork(reseed=pt.reseed, lean=True, alarm=120)
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


def records_of(res, ctx, pt, worker, cap_w):
    """The imitation records of an improving point: (rows [n] ROW, meta [n, V2_META] float64)."""
    b = res['best']
    n = b['n']
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
    meta[:, VI['n']] = n
    for k, v in (('gain', res['gain']), ('spread', res['spread']), ('taken', res['taken']), ('best', b['score']),
                 ('depth', res['k']), ('hold', b['intent'].h), ('intent', b['intent'].code), ('kind', pt.kind),
                 ('point', pt.pid), ('worker', worker), ('version', pt.version), ('rep', res['rep'])):
        meta[:, VI[k]] = v
    return rows, meta


class LanePolicy:
    """The actor through a worker's lane (training): request() writes the action under way and wakes the actor (the
    record is already in the lane slot), answer() waits for its answer (stop(): give up, V2Stop)."""

    def __init__(self, conn, lane_io, lane_out, index, stop):
        self.conn, self.io, self.out, self.i, self.stop = conn, lane_io, lane_out, index, stop
        self.wait_s = 0.0
        self.calls = 0

    def request(self, row, under_way):
        self.io[self.i, :5] = under_way
        self.conn.send_bytes(b'0')

    def answer(self):
        t0 = time.perf_counter()
        while not self.conn.poll(0.05):
            if self.stop():
                raise V2Stop()
        self.conn.recv_bytes()
        self.wait_s += time.perf_counter() - t0
        self.calls += 1
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
