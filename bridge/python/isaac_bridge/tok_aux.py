"""Predictive auxiliary loss on the temporal transformer's output o_t (train_tok --aux-predict 1; rl/docs/
PREDICTIVE_MEMORY_DESIGN.md section 3). The learner's side only: the actor's graphs, the model's state_dict and the
weights' hand-over never see it (the head is a module of its own, not a part of TokPolicy).

Time convention. Decision t of a worker has the record rows[t] it was taken on, and is closed by the next record of the
worker: that record's hurt / events / done describe the step t -> t + 1 and give reward[t]. An episode's last record
(done != 0) is never stored as a decision, so a stored row alone cannot show the death, the hurt that caused it or the
last step's events: Rollout(close=True) keeps what closed each decision (close_hurt, close_done, close_ev; Episodes.arrive
writes them) and the tables below read those.

Targets of decision t at scale k (a horizon of k decisions; rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 3.2):
  - state: the explicit fields of rows[t + k]. Valid only when that row exists in this rollout and belongs to the same
    episode;
  - events: 'did it happen in the steps t -> t + 1, ..., t + k - 1 -> t + k', one bit per EVENTS entry;
  - return: reward[t] + ... + reward[t + k - 1].
    Events and return are valid when the whole window is in this rollout, or when the episode really ended inside it: after
    an end nothing more can happen, the window is complete as it stands. (Masking those would remove every death and
    every last hurt from the targets: a death can only be the last step of its episode.) A window that runs into the
    rollout's end, or into a boundary whose end is unknown (a worker error: CLOSE_BROKE; a branch's truncation: boot), is
    masked. progress.csv reports the masked shares (aux_ms_k, aux_me_k).

Normalisation: state fields by their running mean and standard deviation over every stored record seen so far (AuxNorm,
saved in the checkpoint), clipped to +-CLIP; a return by the running standard deviation of its scale. The room type is a
class, not a number: N_ROOM_TYPE-way cross-entropy in place of that column's MSE (M1 review, 2026-11-03), counting as one
column of the state's mean (reg_k = (sum of the other columns' MSE + CE) / len(STATE); progress.csv aux_room_ce_k holds
the CE alone). Loss: the state as above, BCE on the events, Huber on the return, each mean over its valid rows, equal
weight per scale: L_aux = mean_k (reg_k + ev_k + ret_k); the trainer adds --aux-coef x L_aux to the part's loss. The gradient reaches the
head, the temporal transformer and the spatial encoder through o_t as PPO's does (--temporal-bptt's chunks), and never
passes through an advantage.
"""
import numpy as np

from .tok_obs import EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM, KIND_COLUMN, N_ROOM_TYPE

# Rollout close_done of a decision closed by the next episode's first record (a worker error): the end is not a real one
CLOSE_BROKE = 255
EVENTS = ('hurt', 'clear', 'room', 'new_room', 'pickup', 'floor', 'death')
# (red and soul half hearts are ROW player columns 4, 5; phearts holds black, eternal, bone)
STATE = ('x', 'y', 'hearts', 'soul', 'black', 'eternal', 'bone', 'coins', 'bombs', 'keys', 'enemies', 'room_type',
         'stage')
ROOM_COL = STATE.index('room_type')
REG_COLS = tuple(i for i in range(len(STATE)) if i != ROOM_COL)   # the regressed columns
_PLAYER_COLS = (0, 1, 4, 5)
NPC_COLUMN = KIND_COLUMN + 6
CLIP = 10.0


def parse_horizons(text):
    """'1,8,64,256' -> (1, 8, 64, 256): positive, strictly increasing."""
    out = tuple(int(v) for v in str(text).split(',') if v.strip())
    if not out or min(out) < 1 or any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f'--aux-horizons: positive and increasing expected, got {text!r}')
    return out


def state_rows(rows, npc_column=NPC_COLUMN):
    """float32 [n, len(STATE)]: the explicit state of records `rows` (a ROW array, no copy of its entity tables)."""
    n = len(rows)
    pl = rows['player']
    ph = rows['phearts']
    # the visible NPCs among the record's real entity tokens (columns past n_ent hold older values)
    npc = (rows['ent'][:, :, npc_column] > 0) & (np.arange(rows['ent'].shape[1])[None] < rows['n_ent'][:, None])
    out = np.empty((n, len(STATE)), np.float32)
    out[:, :4] = pl[:, _PLAYER_COLS]
    out[:, 4:7] = ph
    out[:, 7], out[:, 8], out[:, 9] = pl[:, 10], pl[:, 8], pl[:, 9]
    out[:, 10] = npc.sum(1)
    out[:, 11], out[:, 12] = rows['room_type'], rows['stage']
    return out


def _gains(rows):
    """[n, 2]: health units and coins + bombs + keys of records (train_tok Episodes.cur_gains' columns)."""
    pl = rows['player']
    return np.stack([np.rint(pl[:, 7] * 12.0),
                     np.rint(pl[:, 8] * 10.0) + np.rint(pl[:, 9] * 10.0) + np.rint(pl[:, 10] * 50.0)], 1)


def event_rows(rows, n_closed, hurt, done, ev, boundary):
    """bool [n_closed, len(EVENTS)]: what happened in the step closing each decision. rows: the worker's stored records;
    hurt, done, ev: its close_* arrays; boundary[i]: decision i's step left the episode (rows[i + 1], if any, is another
    episode's)."""
    n = n_closed
    out = np.zeros((n, len(EVENTS)), bool)
    if not n:
        return out
    ev = ev[:n].astype(np.int64)
    done = done[:n]
    out[:, 0] = hurt[:n] > 0
    out[:, 1] = ((ev & EV_CLEARED) > 0) | (done == 1)
    out[:, 3] = (ev & EV_NEW_ROOM) > 0
    out[:, 5] = (ev & EV_EXIT) > 0
    out[:, 6] = done == 2
    out[:, 4] = (ev & EV_ITEM) > 0
    # another room: TemporalAttention.step's rule (a record without a current room neither moves nor replaces the last)
    out[:, 2] = (ev & (EV_NEW_ROOM | EV_EXIT)) > 0
    m = len(rows)
    if m > 1:
        map3 = rows['map'][:, 3].reshape(m, -1)
        shown = map3.any(1)
        first = rows['first'] != 0
        at = np.arange(m)
        last = np.maximum.accumulate(np.where(shown, at, -1))
        ep0 = np.maximum.accumulate(np.where(first, at, 0))
        k = min(n, m - 1)
        live = ~boundary[:k]
        prev = last[:k]
        cmp = live & shown[1:k + 1] & (prev >= ep0[:k]) & (prev >= 0)
        moved = np.zeros(k, bool)
        j = np.flatnonzero(cmp)
        if len(j):
            moved[j] = (map3[j + 1] != map3[prev[j]]).any(1)
        out[:k, 2] |= moved
        g = _gains(rows)
        out[:k, 4] |= live & (g[1:k + 1] > g[:k]).any(1)
    return out


def worker_tables(length, open_k, terminal, boot, first, reward, close_done, events, state, horizons):
    """One worker's targets (the module docstring). length: its stored decisions; open_k: the open one (-1: none);
    terminal, boot, first, reward, close_done: its arrays; events: event_rows; state: state_rows. Returns dict of
    arrays [length, K, ...]: state, m_state, ev, ret, m_ev."""
    L, K = int(length), len(horizons)
    U = L - 1 if (L and open_k == L - 1) else L   # the decisions with their step closed
    S, E = state.shape[1], events.shape[1]
    out = dict(state=np.zeros((L, K, S), np.float32), m_state=np.zeros((L, K), bool), ev=np.zeros((L, K, E), bool),
               ret=np.zeros((L, K), np.float64), m_ev=np.zeros((L, K), bool))
    if U <= 0:
        return out
    next_first = np.zeros(U, bool)
    k_f = min(U, L - 1)
    next_first[:k_f] = first[1:k_f + 1] != 0
    bnd = terminal[:U] | boot[:U] | next_first
    hard = boot[:U] | (close_done[:U] == CLOSE_BROKE) | (next_first & ~terminal[:U])   # an end of unknown outcome
    BIG = 1 << 60
    end = np.minimum.accumulate(np.where(bnd, np.arange(U), BIG)[::-1])[::-1]
    t = np.arange(U)
    cum_e = np.concatenate([np.zeros((1, E), np.int64), np.cumsum(events[:U], 0, dtype=np.int64)])
    cum_r = np.concatenate([[0.0], np.cumsum(reward[:U], dtype=np.float64)])
    usable = ~boot[:U]
    for ki, h in enumerate(horizons):
        e = t + h - 1          # the window's last step
        reach = end <= e       # the episode's end is inside the window
        hard_end = np.zeros(U, bool)
        hard_end[reach] = hard[end[reach]]
        valid = ((e < U) | reach) & ~hard_end & usable
        stop = np.minimum(e, end)
        stop = np.minimum(stop, U - 1)
        out['m_ev'][:U, ki] = valid
        out['ev'][:U, ki] = ((cum_e[stop + 1] - cum_e[t]) > 0) & valid[:, None]
        out['ret'][:U, ki] = np.where(valid, cum_r[stop + 1] - cum_r[t], 0.0)
        ok = (t + h < L) & ~reach & usable
        idx = np.flatnonzero(ok)
        out['m_state'][idx, ki] = True
        out['state'][idx, ki] = state[idx + h]
    return out


def boundaries(length, terminal, boot, first):
    """bool [length]: decision i's step left the episode (terminal, a branch's end, or the next stored record starts
    another)."""
    L = int(length)
    out = terminal[:L] | boot[:L]
    if L > 1:
        out = out.copy()
        out[:L - 1] |= first[1:L] != 0
    return out


def rollout_tables(roll, horizons):
    """The targets of a whole rollout (Rollout with close arrays, or a view of one): dict of arrays [workers, m, K, ...]
    (m = the longest worker), plus state_raw: [records, len(STATE)] of every stored record (AuxNorm's input)."""
    K = len(horizons)
    m = int(roll.length.max()) if roll.n else 0
    S, E = len(STATE), len(EVENTS)
    out = dict(state=np.zeros((roll.n, m, K, S), np.float32), m_state=np.zeros((roll.n, m, K), bool),
               ev=np.zeros((roll.n, m, K, E), bool), ret=np.zeros((roll.n, m, K), np.float64),
               m_ev=np.zeros((roll.n, m, K), bool))
    raws = []
    for w in range(roll.n):
        L = int(roll.length[w])
        if not L:
            continue
        rows = roll.rows[w, :L]
        U = L - 1 if roll.open[w] == L - 1 else L
        first = rows['first']
        bnd = boundaries(L, roll.terminal[w], roll.boot[w], first)
        st = state_rows(rows)
        raws.append(st)
        evs = event_rows(rows, U, roll.close_hurt[w], roll.close_done[w], roll.close_ev[w], bnd)
        t = worker_tables(L, roll.open[w], roll.terminal[w], roll.boot[w], first, roll.reward[w], roll.close_done[w],
                          evs, st, horizons)
        for k_, v_ in t.items():
            out[k_][w, :L] = v_
    out['state_raw'] = np.concatenate(raws) if raws else np.zeros((0, S), np.float32)
    return out


class AuxNorm:
    """Running mean / variance of the state fields (every stored record) and running variance of each scale's return (its
    valid rows). Plain numpy: it lives in the learner and in the checkpoint (state / load)."""

    def __init__(self, n_state, n_scales):
        self.n = 0.0
        self.mean = np.zeros(n_state, np.float64)
        self.m2 = np.zeros(n_state, np.float64)
        self.rn = np.zeros(n_scales, np.float64)
        self.rm2 = np.zeros(n_scales, np.float64)   # (about 0: a return is not centred, its second moment scales it)

    def update(self, state_raw, ret, m_ev):
        """state_raw [records, S]; ret, m_ev [..., K]. Batched Welford."""
        if len(state_raw):
            b = float(len(state_raw))
            mu = state_raw.astype(np.float64).mean(0)
            var = state_raw.astype(np.float64).var(0)
            d = mu - self.mean
            tot = self.n + b
            self.m2 += b * var + d * d * self.n * b / tot
            self.mean += d * b / tot
            self.n = tot
        for k in range(ret.shape[-1]):
            v = ret[..., k][m_ev[..., k]]
            self.rn[k] += v.size
            self.rm2[k] += float((v * v).sum())

    def state_std(self):
        return np.sqrt(np.maximum(self.m2 / max(self.n, 1.0), 1e-4))

    def ret_std(self):
        return np.sqrt(np.maximum(self.rm2 / np.maximum(self.rn, 1.0), 1e-4))

    def state(self):
        # (plain lists: train_tok reads a checkpoint with torch.load's default, weights_only, which refuses numpy arrays)
        return dict(n=self.n, mean=self.mean.tolist(), m2=self.m2.tolist(), rn=self.rn.tolist(), rm2=self.rm2.tolist())

    def load(self, d):
        self.n = float(d['n'])
        self.mean, self.m2 = np.asarray(d['mean'], np.float64), np.asarray(d['m2'], np.float64)
        self.rn, self.rm2 = np.asarray(d['rn'], np.float64), np.asarray(d['rm2'], np.float64)


def base_bce(p):
    """The BCE of always predicting the base rate p (per column), the events' floor of no skill."""
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def build_targets(roll, flat, weight, horizons, norm, device):
    """The update's targets at the rows of `flat` (the learner's flat positions, weight: 1 = trained): the norm updated
    first (the first rollout normalises by its own statistics), everything normalised, on the device in one copy per
    array. Returns (tensors dict for part_loss, log dict of plain numbers)."""
    import torch
    tb = rollout_tables(roll, horizons)
    norm.update(tb['state_raw'][:, REG_COLS], tb['ret'], tb['m_ev'])
    w_, t_ = flat // roll.cap, flat % roll.cap
    m_state, m_ev = tb['m_state'][w_, t_], tb['m_ev'][w_, t_]
    ev = tb['ev'][w_, t_]
    raw = tb['state'][w_, t_]
    room = raw[..., ROOM_COL].astype(np.int64)
    state = np.clip((raw[..., REG_COLS] - norm.mean.astype(np.float32)) / norm.state_std().astype(np.float32),
                    -CLIP, CLIP) * m_state[..., None]
    ret = (tb['ret'][w_, t_] / norm.ret_std()).astype(np.float32) * m_ev
    log = {}
    tr = weight > 0
    n_tr = max(int(tr.sum()), 1)
    for ki, h in enumerate(horizons):
        log[f'aux_ms_{h}'] = 1.0 - float(m_state[tr, ki].sum()) / n_tr    # masked shares of the trained rows
        log[f'aux_me_{h}'] = 1.0 - float(m_ev[tr, ki].sum()) / n_tr
        sel = m_ev[tr, ki]
        rate = ev[tr, ki][sel].mean(0) if sel.any() else np.zeros(len(EVENTS))
        log[f'aux_ev0_{h}'] = float(base_bce(rate).mean())
        for j, name in enumerate(EVENTS):
            log[f'aux_rate_{name}_{h}'] = float(rate[j])

    def to(a, dt=None):
        x = torch.from_numpy(np.ascontiguousarray(a))
        return (x if dt is None else x.to(dt)).to(device)
    return dict(state=to(state), room=to(room), m_state=to(m_state, torch.float32), ev=to(ev, torch.float32), ret=to(ret),
                m_ev=to(m_ev, torch.float32)), log


def make_head(d_in, n_scales, hidden=256):
    """The prediction head: o_t -> [K, R + N_ROOM_TYPE + E + 1] (regressed state, room-type logits, event logits, return
    per scale)."""
    from torch import nn

    class AuxHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.K, self.R, self.E = int(n_scales), len(REG_COLS), len(EVENTS)
            self.out = self.R + N_ROOM_TYPE + self.E + 1
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, self.K * self.out))

        def forward(self, o):
            return self.net(o.float()).view(o.shape[0], self.K, self.out)
    return AuxHead()


def minibatch_den(tg, sel, w):
    """The denominators of a minibatch's means, per scale: its valid trained rows (sel: its rows, w: the rows' weight)."""
    ws = w.index_select(0, sel)[:, None]
    return dict(state=(tg['m_state'].index_select(0, sel) * ws).sum(0).clamp(min=1.0),
                ev=(tg['m_ev'].index_select(0, sel) * ws).sum(0).clamp(min=1.0))


def part_loss(head, o, tg, part, w, den):
    """A part's share of the minibatch's L_aux: o [n, D] (with gradient), part [n] its rows in tg, w [n] the rows' weight,
    den: minibatch_den. Returns (L_aux's share, detached [4, K] reg / ev / ret / room CE shares, detached [K, E] per-event
    BCE shares). Shares of the minibatch's parts add up to its means."""
    from torch.nn import functional as F
    p = head(o)
    R, E = head.R, head.E
    Rc = R + N_ROOM_TYPE
    ms = tg['m_state'].index_select(0, part) * w[:, None]
    me = tg['m_ev'].index_select(0, part) * w[:, None]
    room = tg['room'].index_select(0, part)
    ce = F.cross_entropy(p[..., R:Rc].reshape(-1, N_ROOM_TYPE), room.reshape(-1), reduction='none').view(room.shape)
    reg = ((p[..., :R] - tg['state'].index_select(0, part)).pow(2).sum(-1) + ce) / (R + 1)   # (the CE: one column)
    bce = F.binary_cross_entropy_with_logits(p[..., Rc:Rc + E], tg['ev'].index_select(0, part), reduction='none')
    ret = F.huber_loss(p[..., Rc + E], tg['ret'].index_select(0, part), reduction='none', delta=1.0)
    l_reg = (reg * ms).sum(0) / den['state']
    l_room = (ce * ms).sum(0) / den['state']
    l_ev = (bce.mean(-1) * me).sum(0) / den['ev']
    l_ret = (ret * me).sum(0) / den['ev']
    per_ev = (bce * me[..., None]).sum(0) / den['ev'][:, None]
    import torch
    return (l_reg + l_ev + l_ret).mean(), torch.stack([l_reg, l_ev, l_ret, l_room]).detach(), per_ev.detach()


def log_row(acc, acc_ev, steps, horizons):
    """progress.csv's columns of an update from the parts' summed shares (acc [4, K], acc_ev [K, E], numpy)."""
    row = {}
    for ki, h in enumerate(horizons):
        row[f'aux_reg_{h}'], row[f'aux_ev_{h}'], row[f'aux_ret_{h}'], row[f'aux_room_ce_{h}'] = (
            float(acc[j, ki]) / steps for j in range(4))
        for j, name in enumerate(EVENTS):
            row[f'aux_bce_{name}_{h}'] = float(acc_ev[ki, j]) / steps
    row['aux_loss'] = float(acc[:3].sum(0).mean()) / steps
    return row
