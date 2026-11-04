"""Offline world-model evaluation from the decision stream log (rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 5): how well can
events and explicit fields k decisions ahead be predicted on real play? Runs off the training host (the 3090); the log format is
in rl/docs/STREAM_LOG_FORMAT.md.

  prep    <stream dir> [<stream dir> ...] --out DATA [--update-min N] [--update-max N]
          decode the zstd parts into memory-mapped columns and a table of contiguous segments. z_t is only comparable within a
          narrow range of weights, so a long run's log is cut to its late part with --update-min.
  train   --data DATA --arm wm|latent|base --out DIR [--steps N]
          wm:     causal transformer (256 wide, 4 layers) over [z_t, g_t, explicit fields, a_t], teacher forcing, multi-horizon heads
          latent: the same with Dreamer-v3 style categorical latents (32 x 32) between the observation and the transformer
          base:   the current frame only (the same input at t, no history): the verdict's reference
          One training reports both held-out sets: 'episodes' (other episodes of the training seeds, or of unique seeds when
          every episode has its own) and 'seeds' (every episode of floor seeds never trained on).
  verdict --wm DIR --base DIR [--latent DIR]
          the design's line: AUC at k=32 of hurt / clear / room below 0.75, or Brier at k=64 not better than the current frame's,
          excludes imagination training and latent planning. Prints every number the line reads, per event and per held-out set.

Targets of decision t at scale k, as in isaac_bridge.tok_aux. Events: 'did it happen in the steps t -> t + 1, ..., t + k - 1 ->
t + k'; a window running past a segment's last decision is valid only if the episode really ended there (clear, death, timeout).
Fields: the record at t + k, when it is stored in the same segment. The stored records of a segment are consecutive decisions; a log
gap (a dropped batch) or a run's end starts a new segment with an unknown end.
"""
import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from isaac_bridge.tok_aux import EVENTS, STATE
from isaac_bridge.tok_obs import EV_CLEARED, EV_EXIT, EV_ITEM, EV_NEW_ROOM
from isaac_bridge.tok_stream import iter_records

HORIZONS = (1, 4, 16, 32, 64, 128, 256)
E_COL = {name: i for i, name in enumerate(EVENTS)}
S_COL = {name: i for i, name in enumerate(STATE)}
REAL_ENDS = (1, 2, 3)   # close_done: clear, death, timeout (not a branch's cut or a worker's error)
VERDICT_EVENTS = ('hurt', 'clear', 'room')
NO_ROOM = 255


# ---- events and segments ----

def _gains(state):
    """[n, 2]: health units and coins + bombs + keys from the explicit fields (tok_aux._gains' columns, as far as the stream has
    them; the ROW cross-check is in tests/test_world_model_smoke.py)."""
    s = state
    return np.stack([np.rint(s[:, S_COL['hearts']] * 12.0) + np.rint(s[:, S_COL['soul']] * 12.0),
                     np.rint(s[:, S_COL['coins']] * 50.0) + np.rint(s[:, S_COL['bombs']] * 10.0) + np.rint(s[:, S_COL['keys']] * 10.0)], 1)


def stream_events(rec):
    """bool [n, len(EVENTS)] of a segment's records: tok_aux.event_rows' definitions on the stream's fields. The step of the last
    record has no next record: only its close_* fields count."""
    n = len(rec)
    out = np.zeros((n, len(EVENTS)), bool)
    if not n:
        return out
    ev = rec['close_ev'].astype(np.int64)
    done = rec['close_done']
    out[:, E_COL['hurt']] = rec['close_hurt'] > 0
    out[:, E_COL['clear']] = ((ev & EV_CLEARED) > 0) | (done == 1)
    out[:, E_COL['new_room']] = (ev & EV_NEW_ROOM) > 0
    out[:, E_COL['floor']] = (ev & EV_EXIT) > 0
    out[:, E_COL['death']] = done == 2
    out[:, E_COL['pickup']] = (ev & EV_ITEM) > 0
    out[:, E_COL['room']] = (ev & (EV_NEW_ROOM | EV_EXIT)) > 0
    if n > 1:
        # another room: the last shown room against the next shown one (a record without a map neither moves nor replaces it)
        shown = rec['room'] != NO_ROOM
        at = np.arange(n)
        last = np.maximum.accumulate(np.where(shown, at, -1))
        prev = last[:-1]
        j = np.flatnonzero(shown[1:] & (prev >= 0))
        out[j, E_COL['room']] |= rec['room'][j + 1] != rec['room'][prev[j]]
        g = _gains(rec['state'])
        out[:-1, E_COL['pickup']] |= (g[1:] > g[:-1]).any(1)
    return out


def split_segments(rec):
    """[(a, b)] of a worker's records in log order: runs of consecutive decisions of one episode."""
    n = len(rec)
    if not n:
        return []
    cut = (rec['episode'][1:] != rec['episode'][:-1]) | (rec['t'][1:] != rec['t'][:-1] + 1) | (rec['first'][1:] != 0)
    at = np.flatnonzero(cut) + 1
    return list(zip(np.r_[0, at].tolist(), np.r_[at, n].tolist()))


def prep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    files = {k: open(out / f'{k}.bin', 'wb') for k in ('z', 'g', 'state', 'action', 'events')}
    segs = []
    n = 0
    width = heads = None
    action_max = None
    for ri, root in enumerate(args.roots):
        fmt = json.loads((Path(root) / 'format.json').read_text())
        width, heads = fmt['width'], fmt['heads']
        by_worker = {}
        for w, rec in iter_records(root, 'stream'):
            by_worker.setdefault(w, []).append(rec)
        for w, parts in sorted(by_worker.items()):
            rec = np.concatenate(parts)
            keep = (rec['update'] >= args.update_min) & (rec['update'] <= args.update_max)
            rec = rec[keep]
            for a, b in split_segments(rec):
                s = rec[a:b]
                if len(s) < args.min_length:
                    continue
                end = bool(s['terminal'][-1]) and int(s['close_done'][-1]) in REAL_ENDS
                s['z'].astype('<i2').tofile(files['z'])
                s['g'].astype('<i2').tofile(files['g'])
                s['state'].astype('<f4').tofile(files['state'])
                s['action'].astype('i1').tofile(files['action'])
                np.packbits(stream_events(s), axis=1, bitorder='little').tofile(files['events'])
                segs.append((n, len(s), int(s['seed'][0]), ri, w, int(s['episode'][0]), int(end), int(s['update'][0])))
                n += len(s)
                top = s['action'].max(0)
                action_max = top if action_max is None else np.maximum(action_max, top)
        print(f'{root}: {n} decisions, {len(segs)} segments', flush=True)
    for f in files.values():
        f.close()
    np.save(out / 'segments.npy', np.array(segs, np.int64))
    (out / 'meta.json').write_text(json.dumps(dict(
        n=n, width=width, heads=heads, action_card=(action_max + 1).tolist() if n else [], events=list(EVENTS), state=list(STATE),
        roots=[str(r) for r in args.roots], update_min=args.update_min, update_max=args.update_max)))
    print('segments', len(segs), 'decisions', n, 'real ends', sum(s[6] for s in segs), 'distinct seeds', len({s[2] for s in segs}))


class Data:
    def __init__(self, root):
        self.root = Path(root)
        m = self.meta = json.loads((self.root / 'meta.json').read_text())
        n, w, h = m['n'], m['width'], m['heads']
        E = len(EVENTS)
        self.nbytes_ev = (E + 7) // 8
        self.z = np.memmap(self.root / 'z.bin', '<i2', 'r', shape=(n, w))
        self.g = np.memmap(self.root / 'g.bin', '<i2', 'r', shape=(n, w))
        self.state = np.memmap(self.root / 'state.bin', '<f4', 'r', shape=(n, len(STATE)))
        self.action = np.memmap(self.root / 'action.bin', 'i1', 'r', shape=(n, h))
        self.events = np.memmap(self.root / 'events.bin', 'u1', 'r', shape=(n, self.nbytes_ev))
        self.segs = np.load(self.root / 'segments.npy')

    def split(self, seed=1, seed_share=0.1, episode_share=0.1):
        """Index arrays of segments: train, 'episodes', 'seeds'. An episode cut into segments stays on one side."""
        rng = np.random.default_rng(seed)
        seeds = np.unique(self.segs[:, 2])
        held = rng.permutation(seeds)[:max(1, int(round(len(seeds) * seed_share)))]
        is_seed = np.isin(self.segs[:, 2], held)
        key = self.segs[:, 3] * (1 << 40) + self.segs[:, 4] * (1 << 28) + self.segs[:, 5]
        uniq = np.unique(key[~is_seed])
        out_ep = rng.permutation(uniq)[:max(1, int(round(len(uniq) * episode_share)))]
        is_ep = np.isin(key, out_ep) & ~is_seed
        return np.flatnonzero(~is_seed & ~is_ep), np.flatnonzero(is_ep), np.flatnonzero(is_seed)

    def window(self, seg, start, L, horizons):
        """The inputs and targets of decisions start .. start + L - 1 of a segment, padded to L. mask_e, mask_s [L, K]."""
        a, n, real_end = int(self.segs[seg, 0]), int(self.segs[seg, 1]), bool(self.segs[seg, 6])
        K, E, S = len(horizons), len(EVENTS), len(STATE)
        m = min(L, n - start)
        H = horizons[-1]
        stop = min(n, start + L + H)
        ev = np.unpackbits(np.asarray(self.events[a + start:a + stop]), axis=1, count=E, bitorder='little').astype(np.int32)
        cum = np.concatenate([np.zeros((1, E), np.int32), np.cumsum(ev, 0)])
        st = np.asarray(self.state[a + start:a + stop])
        have = stop - start   # the stored decisions from start on
        t_ev = np.zeros((L, K, E), bool)
        m_ev = np.zeros((L, K), bool)
        t_st = np.zeros((L, K, S), np.float32)
        m_st = np.zeros((L, K), bool)
        at = np.arange(m)
        last = n - 1 - start   # the segment's last decision, relative to start
        for ki, k in enumerate(horizons):
            e = at + k - 1
            ok = (e <= last) | real_end
            e_c = np.minimum(e, have - 1)
            t_ev[:m, ki] = (cum[e_c + 1] - cum[at]) > 0
            m_ev[:m, ki] = ok
            j = at[at + k <= last]
            t_st[j, ki] = st[j + k]
            m_st[j, ki] = True
        def pad(x):
            x = np.asarray(x)
            if m == L:
                return x.copy()
            out = np.zeros((L,) + x.shape[1:], x.dtype)
            out[:m] = x
            return out
        return dict(z=pad(self.z[a + start:a + start + m]), g=pad(self.g[a + start:a + start + m]), state=pad(st[:m]),
                    action=pad(self.action[a + start:a + start + m]), live=pad(np.ones(m, bool)), t_ev=t_ev, m_ev=m_ev, t_st=t_st, m_st=m_st)


def stack_batch(items):
    return {k: np.stack([it[k] for it in items]) for k in items[0]}


class Loader:
    """Random windows, weighted by the decisions they can start at; a thread prepares the batches."""

    def __init__(self, data, segs, L, horizons, batch, seed, depth=4):
        self.data, self.segs, self.L, self.h, self.batch = data, segs, L, horizons, batch
        self.rng = np.random.default_rng(seed)
        self.p = data.segs[segs, 1].astype(np.float64)
        self.p /= self.p.sum()
        self.q = queue.Queue(depth)
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop:
            picks = self.rng.choice(self.segs, self.batch, p=self.p)
            items = []
            for s in picks:
                n = int(self.data.segs[s, 1])
                start = int(self.rng.integers(0, max(1, n - self.L // 2)))
                items.append(self.data.window(s, start, self.L, self.h))
            self.q.put(stack_batch(items))

    def next(self):
        return self.q.get()

    def close(self):
        self.stop = True
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass


def eval_windows(data, segs, L, max_windows, seed=7):
    """(segment, start) of fixed windows covering decisions with half a window of context (a segment's first half window: what
    context there is); a random subset of max_windows."""
    out = []
    for s in segs:
        n = int(data.segs[s, 1])
        out.append((s, 0))
        out.extend((s, st) for st in range(L // 2, max(1, n - L // 2), L // 2))
    if len(out) > max_windows:
        rng = np.random.default_rng(seed)
        out = [out[i] for i in rng.choice(len(out), max_windows, replace=False)]
    return out


# ---- metrics ----

def auc(score, y):
    """Mann-Whitney with average ranks."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    s = np.asarray(score)[order]
    rank = np.empty(len(s), np.float64)
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        rank[a:b] = (a + b - 1) / 2.0
    full = np.empty(len(s), np.float64)
    full[order] = rank
    return float((full[y].sum() - n1 * (n1 - 1) / 2.0) / (n1 * n0))


def verdict(wm, base):
    """The design's line (section 5.2), on one held-out set's results: {'pass': bool, 'failed': [...], 'rows': [...]}."""
    rows, failed = [], []
    for e in VERDICT_EVENTS:
        a = wm['32']['events'][e]['auc']
        rows.append(dict(event=e, k=32, metric='auc', value=a, line=0.75, ok=bool(a >= 0.75)))
        if not a >= 0.75:
            failed.append(f'{e} AUC k=32 {a:.3f} < 0.75')
        bw, bb = wm['64']['events'][e]['brier'], base['64']['events'][e]['brier']
        rows.append(dict(event=e, k=64, metric='brier', value=bw, line=bb, ok=bool(bw < bb)))
        if not bw < bb:
            failed.append(f'{e} Brier k=64 {bw:.4f} not better than the current frame {bb:.4f}')
    return dict(ok=not failed, failed=failed, rows=rows)


# ---- models ----

class Inputs(nn.Module):
    def __init__(self, meta, norm):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(c + 1, 16) for c in meta['action_card'])
        self.register_buffer('mu', torch.tensor(norm['mu'], dtype=torch.float32))
        self.register_buffer('sd', torch.tensor(norm['sd'], dtype=torch.float32))
        self.n_obs = 2 * meta['width'] + len(STATE)
        self.n_act = 16 * len(self.emb)

    def obs(self, b):
        z = b['z'].view(torch.bfloat16).float()
        g = b['g'].view(torch.bfloat16).float()
        s = ((b['state'] - self.mu) / self.sd).clamp(-10.0, 10.0)
        return torch.cat([z, g, s], -1)

    def act(self, b):
        a = b['action'].long().clamp(min=-1) + 1   # (-1: a head without a sample)
        return torch.cat([m(a[..., i]) for i, m in enumerate(self.emb)], -1)


class CausalCore(nn.Module):
    def __init__(self, width, layers, heads, window):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(window, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.window = window

    def forward(self, x):
        L = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device, dtype=x.dtype)
        return self.norm(self.tr(x + self.pos[:L], mask=mask, is_causal=True))


class Outputs(nn.Module):
    def __init__(self, width, K):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, K * (len(EVENTS) + len(STATE))))

    def forward(self, h):
        y = self.net(h)
        y = y.view(*h.shape[:-1], self.K, len(EVENTS) + len(STATE))
        return y[..., :len(EVENTS)], y[..., len(EVENTS):]


class WM(nn.Module):
    """[z_t, g_t, fields, a_t] -> causal transformer -> heads."""

    def __init__(self, meta, norm, width=256, layers=4, heads=8, window=256, K=len(HORIZONS)):
        super().__init__()
        self.inp = Inputs(meta, norm)
        self.proj = nn.Linear(self.inp.n_obs + self.inp.n_act, width)
        self.core = CausalCore(width, layers, heads, window)
        self.out = Outputs(width, K)

    def forward(self, b):
        h = self.core(self.proj(torch.cat([self.inp.obs(b), self.inp.act(b)], -1)))
        return self.out(h), {}


class Base(nn.Module):
    """The current frame only: the same input and heads, residual MLP of the same depth and width, no history."""

    def __init__(self, meta, norm, width=256, layers=4, K=len(HORIZONS), **_):
        super().__init__()
        self.inp = Inputs(meta, norm)
        self.proj = nn.Linear(self.inp.n_obs + self.inp.n_act, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.out = Outputs(width, K)

    def forward(self, b):
        h = self.proj(torch.cat([self.inp.obs(b), self.inp.act(b)], -1))
        for f in self.blocks:
            h = h + f(h)
        return self.out(self.norm(h)), {}


class Latent(nn.Module):
    """Dreamer-v3 style categorical latents, 32 x 32: the posterior of s_t reads frame t; the transformer reads [s_<=t, a_<=t]; its
    h_t gives the prior of s_{t+1} and the heads. Teacher forcing: the heads never roll the prior out. Decoder: s_t -> the frame."""

    def __init__(self, meta, norm, width=256, layers=4, heads=8, window=256, K=len(HORIZONS), cats=32, classes=32, **_):
        super().__init__()
        self.cats, self.classes = cats, classes
        self.inp = Inputs(meta, norm)
        n = cats * classes
        self.post = nn.Sequential(nn.Linear(self.inp.n_obs, 2 * width), nn.GELU(), nn.Linear(2 * width, n))
        self.proj = nn.Linear(n + self.inp.n_act, width)
        self.core = CausalCore(width, layers, heads, window)
        self.prior = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n))
        self.dec = nn.Sequential(nn.Linear(n, 2 * width), nn.GELU(), nn.Linear(2 * width, self.inp.n_obs))
        self.out = Outputs(width, K)

    def _probs(self, logits):
        p = logits.view(*logits.shape[:-1], self.cats, self.classes).softmax(-1)
        return 0.99 * p + 0.01 / self.classes   # (1% uniform mix)

    def forward(self, b):
        obs = self.inp.obs(b)
        q = self._probs(self.post(obs))
        idx = torch.distributions.Categorical(probs=q).sample()
        hard = nn.functional.one_hot(idx, self.classes).to(q.dtype)
        s = (hard + q - q.detach()).flatten(-2)
        h = self.core(self.proj(torch.cat([s, self.inp.act(b)], -1)))
        p = self._probs(self.prior(h))
        # KL(q_{t+1} || p_t): dynamics 0.5 / representation 0.1, free bits 1 nat, as Dreamer v3
        qn, pn = q[:, 1:], p[:, :-1]
        def kl(a, c):
            return (a * (a.log() - c.log())).sum(-1).sum(-1)
        live = b['live'][:, 1:]
        den = live.sum().clamp(min=1)
        dyn = (kl(qn.detach(), pn).clamp(min=1.0) * live).sum() / den
        rep = (kl(qn, pn.detach()).clamp(min=1.0) * live).sum() / den
        rec = (((self.dec(s) - obs) ** 2).mean(-1) * b['live']).sum() / b['live'].sum().clamp(min=1)
        return self.out(h), dict(kl_dyn=dyn, kl_rep=rep, recon=rec, extra=0.5 * dyn + 0.1 * rep + rec)


ARMS = dict(wm=WM, latent=Latent, base=Base)


def to_device(b, device):
    out = {}
    for k, v in b.items():
        t = torch.from_numpy(v).to(device, non_blocking=True)
        out[k] = t
    out['live'] = out['live'].float()
    return out


def field_stats(data, segs, rows=2000000, seed=3):
    rng = np.random.default_rng(seed)
    pick = rng.choice(segs, min(len(segs), 4000), replace=False)
    parts = []
    for s in pick:
        a, n = int(data.segs[s, 0]), int(data.segs[s, 1])
        parts.append(np.asarray(data.state[a:a + n]))
        if sum(len(p) for p in parts) > rows:
            break
    x = np.concatenate(parts)
    return dict(mu=x.mean(0).tolist(), sd=(x.std(0) + 1e-6).tolist())


def event_rates(data, segs):
    tot = np.zeros(len(EVENTS), np.int64)
    n = 0
    for s in segs:
        a, m = int(data.segs[s, 0]), int(data.segs[s, 1])
        ev = np.unpackbits(np.asarray(data.events[a:a + m]), axis=1, count=len(EVENTS), bitorder='little')
        tot += ev.sum(0, dtype=np.int64)
        n += m
    return (tot / max(n, 1)).tolist()


@torch.no_grad()
def evaluate(model, data, windows, L, horizons, device, batch, norm):
    """Results per horizon: for every event auc / brier / base-rate brier; for every field the standardised MSE (mean, per field)."""
    model.eval()
    K, E, S = len(horizons), len(EVENTS), len(STATE)
    mu = torch.tensor(norm['mu'], device=device, dtype=torch.float32)
    sd = torch.tensor(norm['sd'], device=device, dtype=torch.float32)
    sc, ye, se_sum, se_n = [[[] for _ in range(E)] for _ in range(K)], [[[] for _ in range(E)] for _ in range(K)], np.zeros((K, S)), np.zeros(K)
    for i in range(0, len(windows), batch):
        items = []
        keep_from = []
        for s, st in windows[i:i + batch]:
            items.append(data.window(s, st, L, horizons))
            keep_from.append(0 if st == 0 else L // 2)
        b = to_device(stack_batch(items), device)
        (ev, st_p), _ = model(b)
        keep = torch.zeros(ev.shape[:2], dtype=torch.bool, device=device)
        for j, kf in enumerate(keep_from):
            keep[j, kf:] = True
        keep &= b['live'] > 0
        pr = ev.sigmoid()
        tgt_s = ((b['t_st'] - mu) / sd).clamp(-10.0, 10.0)
        err = (st_p - tgt_s) ** 2
        for ki in range(K):
            me = keep & b['m_ev'][:, :, ki]
            ms = keep & b['m_st'][:, :, ki]
            p_k, t_k = pr[:, :, ki][me].float().cpu().numpy(), b['t_ev'][:, :, ki][me].cpu().numpy()
            for e in range(E):
                sc[ki][e].append(p_k[:, e])
                ye[ki][e].append(t_k[:, e])
            se_sum[ki] += err[:, :, ki][ms].sum(0).cpu().numpy()
            se_n[ki] += int(ms.sum())
    out = {}
    for ki, k in enumerate(horizons):
        evs = {}
        for e, name in enumerate(EVENTS):
            p, y = np.concatenate(sc[ki][e]), np.concatenate(ye[ki][e])
            rate = float(y.mean()) if len(y) else float('nan')
            evs[name] = dict(n=int(len(y)), pos=int(y.sum()), rate=rate, auc=auc(p, y), brier=float(((p - y) ** 2).mean()) if len(y) else float('nan'),
                             brier_rate=rate * (1.0 - rate))
        mse = (se_sum[ki] / max(se_n[ki], 1)).tolist()
        out[str(k)] = dict(events=evs, fields=dict(zip(STATE, mse)), mse=float(np.mean(mse)), n_fields=int(se_n[ki]))
    model.train()
    return out


def train(args):
    data = Data(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = args.device
    torch.manual_seed(args.seed)
    tr, held_ep, held_seed = data.split(args.seed)
    horizons = tuple(int(v) for v in args.horizons.split(','))
    L = args.window   # (base: the same windows, its layers just never look along them)
    norm = field_stats(data, tr)
    model = ARMS[args.arm](data.meta, norm, width=args.width, layers=args.layers, heads=args.heads, window=L, K=len(horizons)).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 500.0) * (0.5 + 0.5 * math.cos(math.pi * min(i, args.steps) / args.steps)))
    loader = Loader(data, tr, L, horizons, args.batch, args.seed + 1)
    meta = dict(arm=args.arm, args=vars(args), params=params, decisions=int(data.meta['n']), segments=dict(train=len(tr), episodes=len(held_ep), seeds=len(held_seed)),
                decisions_split=dict(train=int(data.segs[tr, 1].sum()), episodes=int(data.segs[held_ep, 1].sum()), seeds=int(data.segs[held_seed, 1].sum())),
                distinct_seeds=int(len(np.unique(data.segs[:, 2]))), train_event_rates=dict(zip(EVENTS, event_rates(data, tr))), horizons=horizons)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ('arm', 'params', 'decisions', 'segments', 'decisions_split', 'distinct_seeds', 'train_event_rates')}), flush=True)
    win = dict(episodes=eval_windows(data, held_ep, L, args.eval_windows), seeds=eval_windows(data, held_seed, L, args.eval_windows))
    mu = torch.tensor(norm['mu'], device=device, dtype=torch.float32)
    sd = torch.tensor(norm['sd'], device=device, dtype=torch.float32)
    log = open(out / 'progress.csv', 'w', buffering=1)
    log.write('step,seconds,ev,fields,extra,lr\n')
    t0 = time.time()
    results = {}
    def run_eval(step):
        res = {k: evaluate(model, data, w, L, horizons, device, args.eval_batch, norm) for k, w in win.items()}
        results[str(step)] = res
        (out / 'results.json').write_text(json.dumps(dict(meta=meta, final=res, curve=results), indent=1, default=str))
        line = {k: {str(h): round(res[k][str(h)]['events']['hurt']['auc'], 3) for h in horizons} for k in res}
        print(f'step {step} {time.time() - t0:.0f}s hurt auc {line}', flush=True)
    for step in range(args.steps):
        b = to_device(loader.next(), device)
        (ev, st), aux = model(b)
        bce = nn.functional.binary_cross_entropy_with_logits(ev, b['t_ev'].float(), reduction='none').mean(-1)
        m_e = b['m_ev'].float()
        le = (bce * m_e).sum() / m_e.sum().clamp(min=1)
        tgt = ((b['t_st'] - mu) / sd).clamp(-10.0, 10.0)
        m_s = b['m_st'].float()
        ls = (((st - tgt) ** 2).mean(-1) * m_s).sum() / m_s.sum().clamp(min=1)
        loss = le + ls + aux.get('extra', 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            log.write(f'{step},{time.time() - t0:.1f},{le.item():.5f},{ls.item():.5f},{float(aux.get("extra", 0.0)):.5f},{sched.get_last_lr()[0]:.6f}\n')
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            run_eval(step + 1)
    loader.close()
    torch.save(dict(model=model.state_dict(), norm=norm, args=vars(args)), out / 'final.pt')


def cmd_verdict(args):
    wm = json.loads((Path(args.wm) / 'results.json').read_text())['final']
    base = json.loads((Path(args.base) / 'results.json').read_text())['final']
    lat = json.loads((Path(args.latent) / 'results.json').read_text())['final'] if args.latent else None
    out = {}
    for split in wm:
        out[split] = dict(wm=verdict(wm[split], base[split]))
        if lat:
            out[split]['latent'] = verdict(lat[split], base[split])
        for name, res in [('wm', wm[split]), ('base', base[split])] + ([('latent', lat[split])] if lat else []):
            print(split, name)
            for k in res:
                ev = res[k]['events']
                print(f"  k={k:>4} " + ' '.join(f"{e} auc {ev[e]['auc']:.3f} brier {ev[e]['brier']:.4f}" for e in EVENTS) + f" field mse {res[k]['mse']:.3f}")
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def main():
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('roots', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--update-min', type=int, default=0)
    p.add_argument('--update-max', type=int, default=1 << 31)
    p.add_argument('--min-length', type=int, default=2)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--arm', choices=sorted(ARMS), required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--steps', type=int, default=30000)
    t.add_argument('--batch', type=int, default=64)
    t.add_argument('--window', type=int, default=256)
    t.add_argument('--width', type=int, default=256)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--heads', type=int, default=8)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--horizons', default=','.join(map(str, HORIZONS)))
    t.add_argument('--eval-every', type=int, default=5000)
    t.add_argument('--eval-windows', type=int, default=4000)
    t.add_argument('--eval-batch', type=int, default=32)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    v = sub.add_parser('verdict')
    v.add_argument('--wm', required=True)
    v.add_argument('--base', required=True)
    v.add_argument('--latent', default='')
    v.add_argument('--out', default='')
    args = a.parse_args()
    if args.cmd == 'prep':
        prep(args)
    elif args.cmd == 'train':
        train(args)
    else:
        cmd_verdict(args)


if __name__ == '__main__':
    main()
