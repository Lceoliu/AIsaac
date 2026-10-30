"""Truncated memory training (train_tok --memory-train truncated, 2026-10-30).

Inference is unchanged: the actor's Reader reads the whole episode (the most recent W source values u and every
closed Writer block). Training: a loss reaches only the frames and Writer blocks of the rollout being trained, per
worker. Older values are the actor's own numbers, constants (stop-gradient):

  - the source values u of the W decisions before a worker's rollout: the rollout's carried context rows (P = W rows
    per worker, Rollout.carry), t_s holding exactly the u the actor wrote;
  - the Writer tokens of the blocks the episode closed before the rollout: LearnerCarry, built from the rollouts
    themselves (t_m: the tokens of the block a decision closes, as the actor computed them).

A worker's rollout is its decisions 0 .. length - 1; its open last decision (reward still to come) is decision 0 of
the next rollout, so the context of the next rollout ends right before it (LearnerCarry.advance).
"""
import numpy as np
import torch


class LearnerCarry:
    """Per worker the current episode's Writer tokens of the blocks closed before the next rollout to train: the
    actor's values (float32 [blocks * tokens, D] on the learner's device), the episode serial (Rollout.t_ep) they
    belong to, and how many blocks. Audit counters: mismatches (a context that could not be rebuilt exactly; must stay
    0), constants with a gradient (must stay 0)."""

    def __init__(self, workers, block, tokens, width, device):
        self.block, self.tokens, self.D = int(block), int(tokens), int(width)
        self.device = torch.device(device)
        self.ep = np.full(workers, -10, np.int64)
        self.blocks = np.zeros(workers, np.int64)
        self.mem = [None] * workers
        self.mismatch = 0

    def _empty(self):
        return torch.zeros((0, self.D), dtype=torch.float32, device=self.device)

    def memory(self, w, ep, t0):
        """The constant Writer tokens [t0 // block * tokens, D] of worker w's episode ep before its decision at episode
        index t0 (None counted as a mismatch when the carry does not hold exactly those blocks)."""
        need = t0 // self.block
        if need == 0:
            return self._empty()
        if self.ep[w] != ep or self.blocks[w] < need:
            self.mismatch += 1
            return None
        return self.mem[w][:need * self.tokens]

    def advance(self, rl):
        """After training on rollout rl: every worker's blocks closed by its decisions before the open one (or all of
        them when none is open) appended (their episode's tokens replace another episode's)."""
        P, B = rl.tP, self.block
        rows_all, ws = [], []
        for w in range(rl.n):
            n = int(rl.open[w]) if rl.open[w] >= 0 else int(rl.length[w])
            if n <= 0:
                continue
            ep = rl.t_ep[w, P:P + n]
            t = rl.t_step[w, P:P + n] - 1
            # Only the last episode in the rollout goes on: the carry continues it or starts it afresh (an episode the
            # carry never saw from its start is a mismatch only when one of its blocks is needed: memory())
            at = int(np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])[-1])
            if ep[at] != self.ep[w]:
                self.ep[w], self.blocks[w], self.mem[w] = ep[at], 0, None
                if t[at] != 0:
                    self.blocks[w] = -1   # (started mid-episode: its earlier blocks are unknown)
            close = at + np.flatnonzero((t[at:] + 1) % B == 0)
            if not len(close):
                continue
            if self.blocks[w] < 0 or self.blocks[w] != (t[close[0]] + 1) // B - 1:
                self.blocks[w] = -1   # (a gap: never served)
                continue
            rows_all.append(rl.t_row[w, P + close])
            ws.append((w, len(close)))
            self.blocks[w] += len(close)
        if not rows_all:
            return
        rows = torch.from_numpy(np.concatenate(rows_all)).to(rl.t_m.device)
        tokens = rl.t_m.index_select(0, rows).to(self.device, torch.float32)
        at = 0
        for w, k in ws:
            new = tokens[at:at + k].reshape(k * self.tokens, self.D)
            at += k
            self.mem[w] = new if self.mem[w] is None else torch.cat([self.mem[w], new])


def worker_context(rl, w, window):
    """Worker w's carried context of rollout rl: the episode serial and index of its decision 0 and the constant
    source values before it (float32 [m, D], m = min(t0, window), episode indices t0 - m .. t0 - 1), from the carried
    rows. Returns (ep0, t0, values, ok): ok False when the carried rows do not hold exactly those decisions."""
    P = rl.tP
    ep0, t0 = int(rl.t_ep[w, P]), int(rl.t_step[w, P]) - 1
    m = min(t0, window)
    if m == 0:
        return ep0, t0, rl.t_s.new_zeros((0, rl.t_s.shape[1])), True
    ep, t, rows = rl.t_ep[w, P - m:P], rl.t_step[w, P - m:P] - 1, rl.t_row[w, P - m:P]
    ok = P >= m and bool((ep == ep0).all()) and bool((t == np.arange(t0 - m, t0)).all()) and bool((rows >= 0).all())
    if not ok:
        return ep0, t0, None, False
    return ep0, t0, rl.t_s.index_select(0, torch.from_numpy(rows).to(rl.t_s.device)).float(), True


def chunk_meta(rl, w, n, window, block, carry):
    """Host plan of worker w's rollout chunk (its decisions 0 .. n - 1): each decision's episode index t, the episode
    segments, the first segment's constants when its episode began before the rollout (the carried source values,
    episode indices t0 - m .. t0 - 1, and Writer tokens), and the blocks its decisions close. Frames are numbered in
    the chunk's local source space [carried values (m) | the chunk's decisions (n)]; ok False: a context that could
    not be rebuilt exactly (built without it; counted)."""
    P = rl.tP
    ep = rl.t_ep[w, P:P + n]
    t = (rl.t_step[w, P:P + n] - 1).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    segs = [(int(a), int(b)) for a, b in zip(starts, np.r_[starts[1:], n])]
    meta = dict(w=w, n=n, t=t, segs=segs, m=0, pre_t=np.zeros(0, np.int64), pre=None, mem=None, ok=True)
    for a, b in segs:   # (one decision per episode step)
        if not (t[a:b] == t[a] + np.arange(b - a)).all() or (a > 0 and t[a] != 0):
            meta['ok'] = False
    if t[0] > 0:   # (the first segment's episode began before the rollout)
        _, t0, values, ok = worker_context(rl, w, window)
        mem = carry.memory(w, int(ep[0]), int(t[0])) if carry is not None else None
        if not ok or mem is None:
            meta['ok'] = False
        else:
            meta.update(m=len(values), pre_t=np.arange(t0 - len(values), t0), pre=values, mem=mem)
    blocks = []
    for a, b in segs:
        m0 = meta['m'] if a == 0 else 0
        tb = t[a] - m0   # (episode index of the segment's first local frame)
        for k in a + np.flatnonzero((t[a:b] + 1) % block == 0):
            first = int(t[k]) - block + 1
            if first < tb:   # (a carried episode whose earlier frames are missing: counted above)
                meta['ok'] = False
                continue
            local = first - tb + np.arange(block)
            blocks.append((a, int(t[k]), np.where(local < m0, local, meta['m'] + a + (local - m0))))
    meta['blocks'] = blocks
    return meta


class TruncatedBatch:
    """One update's truncated-memory plan: per worker with decisions a chunk of its whole rollout (decisions 0 ..
    length - 1, in worker order: the rows of train_tok's all_flat), its host plan and its constants (carried source
    values and Writer tokens, float32 on the device). part(workers) builds a part's layout; TruncatedPart.forward is
    the policy's forward for that part's rows. Build it before LearnerCarry.advance(rl)."""

    def __init__(self, model, rl, carry, device):
        rd = model.reader
        self.model, self.device = model, torch.device(device)
        self.W, self.B, self.T = rd.W, rd.block, rd.tokens
        self.workers = np.flatnonzero(rl.length > 0)
        self.chunk_of = {int(w): i for i, w in enumerate(self.workers)}
        self.meta = [chunk_meta(rl, int(w), int(rl.length[w]), rd.W, rd.block, carry) for w in self.workers]
        self.mismatch = sum(not m['ok'] for m in self.meta)
        pre = [m['pre'] for m in self.meta if m['m']]
        mem = [m['mem'] for m in self.meta if m['mem'] is not None and len(m['mem'])]
        D = rd.D
        self.const_u = torch.cat(pre).to(self.device, torch.float32) if pre else torch.zeros((0, D), device=self.device)
        self.const_m = torch.cat(mem).to(self.device, torch.float32) if mem else torch.zeros((0, D), device=self.device)
        self.pre_off = np.cumsum([0] + [m['m'] for m in self.meta])
        self.mem_off = np.cumsum([0] + [0 if m['mem'] is None else len(m['mem']) for m in self.meta])
        self.debug = None   # (tests: a dict that receives each part's in-graph tensors)

    def constants(self):
        """The constant tensors (audit memory_constant_grad_count: none may require a gradient)."""
        return [self.const_u, self.const_m]

    def bounds(self, micro):
        """Parts of whole chunks over all workers' decisions (the rows of all_flat): [(first row, end row, workers)],
        each at least one chunk, at most `micro` rows otherwise."""
        out, lo, cnt, ws = [], 0, 0, []
        for w, m in zip(self.workers, self.meta):
            if ws and cnt + m['n'] > micro:
                out.append((lo, lo + cnt, ws))
                lo, cnt, ws = lo + cnt, 0, []
            ws.append(int(w)); cnt += m['n']
        if ws:
            out.append((lo, lo + cnt, ws))
        return out

    def window(self):
        """Each decision's gradient window (rows of all_flat): its steps from the rollout's first decision or, in an
        episode begun later in the rollout, from that episode's first decision."""
        return np.concatenate([np.concatenate([np.arange(b - a) for a, b in m['segs']]) for m in self.meta]) \
            if self.meta else np.zeros(0, np.int64)

    def part(self, workers, n_teach=0, records=None):
        return TruncatedPart(self, [self.chunk_of[int(w)] for w in workers], n_teach, records)


class TruncatedPart:
    """The Reader layout of a part of whole chunks (rows chunk after chunk), plus rows after them: n_teach
    context-free teacher records (bank: the zero entry only, as TokPolicy.forward without context) and the records of
    `records` (record_constants: teacher v2 / cf imitation records with constant contexts)."""

    def __init__(self, batch, cids, n_teach, records):
        self.batch, self.cids, self.n_teach, self.records = batch, list(cids), int(n_teach), records
        B, T = batch.B, batch.T
        metas = [batch.meta[c] for c in self.cids]
        self.n_main = int(sum(m['n'] for m in metas))
        cat = lambda xs: np.concatenate(xs).astype(np.int64) if xs else np.zeros(0, np.int64)   # noqa: E731
        pre_idx = cat([np.arange(batch.pre_off[c], batch.pre_off[c + 1]) for c in self.cids])
        mem_idx = cat([np.arange(batch.mem_off[c], batch.mem_off[c + 1]) for c in self.cids])
        Mp, Mc = len(pre_idx), len(mem_idx)
        u_t = cat([m['pre_t'] for m in metas] + [m['t'] for m in metas])
        mem_t = cat([(np.arange(batch.mem_off[c + 1] - batch.mem_off[c]) // T + 1) * B - 1 for c in self.cids])
        block_idx, block_t, groups = [], [], []
        pre_at = cm_at = row_at = 0
        for m in metas:
            def glob(local, m=m, pre_at=pre_at, row_at=row_at):   # chunk-local -> the part's source index
                return np.where(local < m['m'], pre_at + local, Mp + row_at + (local - m['m']))
            seg_blocks = {}
            for a, end_t, idx in m['blocks']:
                seg_blocks.setdefault(a, []).append(len(block_t))
                block_idx.append(glob(idx)); block_t.append(end_t)
            for a, b in m['segs']:
                carried = a == 0 and m['m'] > 0
                recent = np.r_[pre_at + np.arange(m['m']) if carried else np.zeros(0, np.int64),
                               Mp + row_at + np.arange(a, b)].astype(np.int64)
                rt = np.r_[m['pre_t'] if carried else np.zeros(0, np.int64), m['t'][a:b]].astype(np.int64)
                ncm = len(m['mem']) if carried and m['mem'] is not None else 0
                nm = np.asarray(seg_blocks.get(a, []), np.int64)
                groups.append(dict(rows=row_at + np.arange(a, b), qt=m['t'][a:b], recent=recent, rt=rt,
                                   cmem=cm_at + np.arange(ncm), cmt=(np.arange(ncm) // T + 1) * B - 1,
                                   nmem=nm, nmt=np.asarray([block_t[i] for i in nm], np.int64)))
            pre_at += m['m']
            cm_at += 0 if m['mem'] is None else len(m['mem'])
            row_at += m['n']
        self.Mp, self.Mc, self.nb = Mp, Mc, len(block_t)
        U = Mp + self.n_main
        # bank columns: 0 the zero entry; 1 + i: source value i of U = [carried values | the part's decisions];
        # 1 + U + j: Writer token j of [carried tokens | the part's new blocks' tokens]
        G = len(groups)
        count = max([len(g['rows']) for g in groups] + [1])
        S = max([1 + len(g['recent']) + len(g['cmem']) + T * len(g['nmem']) for g in groups] + [1])
        banks = np.zeros((G, S), np.int64)
        kind = np.full((G, S), -1, np.int64)
        bt = np.zeros((G, S), np.int64)
        qt = np.zeros((G, count), np.int64)
        pos = np.zeros(self.n_main, np.int64)
        for i, g in enumerate(groups):
            mi = np.r_[1 + U + g['cmem'], 1 + U + Mc + (g['nmem'][:, None] * T + np.arange(T)[None]).reshape(-1)]
            mt = np.r_[g['cmt'], np.repeat(g['nmt'], T)]
            col = np.r_[0, 1 + g['recent'], mi].astype(np.int64)
            r = len(g['recent'])
            banks[i, :len(col)] = col
            kind[i, 0] = 0; kind[i, 1:1 + r] = 1; kind[i, 1 + r:len(col)] = 2
            bt[i, 1:1 + r] = g['rt']; bt[i, 1 + r:len(col)] = mt
            k = len(g['rows'])
            qt[i, :k] = g['qt']
            pos[g['rows']] = i * count + np.arange(k)
        self.shape = (G, count)
        self.host = dict(pre_idx=pre_idx, mem_idx=mem_idx, u_t=u_t, mem_t=np.r_[mem_t, np.repeat(np.asarray(
            block_t, np.int64), T)].astype(np.int64), block_idx=np.asarray(block_idx, np.int64).reshape(-1, B),
            banks=banks, kind=kind, bt=bt, qt=qt, pos=pos,
            main_t=cat([m['t'] for m in metas]))

    def forward(self, model, b, need_h=False, need_vg=False):
        """The policy's outputs for the part's rows [main decisions, teacher records, records]: Spatial with gradient
        (b: the rows' batch; a compact learner plan needs b['_rec'], the host plan's [counts, starts]), the Writer on
        the blocks closed in the rollout with gradient (a block begun before the rollout: its carried frames constant,
        its frames in the rollout with gradient), the Reader over constants and in-rollout values."""
        rd = model.reader
        dev = next(model.parameters()).device
        B, W = self.batch.B, self.batch.W
        y, pad, u = model.encode_rebuild(b)
        if pad is None:
            y, pad = pad_compact(y, b['_rec'])
        h = {k: torch.as_tensor(v, device=dev) for k, v in self.host.items()}
        n = self.n_main
        uf = u.float()
        outs_o = []
        if n:
            U = torch.cat([self.batch.const_u.index_select(0, h['pre_idx']), uf[:n]])
            if self.nb:
                new = rd.write(U.index_select(0, h['block_idx'].reshape(-1)).reshape(-1, B, rd.D)).flatten(0, 1).float()
            else:
                new = U.new_zeros((0, rd.D))
            M = torch.cat([self.batch.const_m.index_select(0, h['mem_idx']), new])
            src = torch.cat([U.new_zeros((1, rd.D)), U + rd.position(h['u_t'], U.dtype),
                             M + rd.position(h['mem_t'], M.dtype)])
            kvs = [layer.kv(src).float() for layer in rd.layers]
            G, count = self.shape
            qt = h['qt'][:, :, None]
            kind, bt = h['kind'][:, None, :], h['bt'][:, None, :]
            valid = (kind == 0) | ((kind == 1) & (bt < qt) & (bt >= qt - W)) | ((kind == 2) & (bt < qt // B * B))
            layout = dict(shape=(G, count), positions=h['pos'], indices=h['banks'], valid=valid)
            q = uf[:n] + rd.position(h['main_t'], uf.dtype)
            outs_o.append(rd.read_bank(q, kvs, None, None, layout))
            if self.batch.debug is not None:
                self.batch.debug.update(u=uf, U=U, new=new, M=M)
        if len(u) > n:
            outs_o.append(self._extra(model, uf[n:], dev))
        o = torch.cat(outs_o)
        c = rd.fuse(o, y, pad)
        return model._heads(y[:, 0] + rd.read(o) + c, need_h,
                            yg=y[:, -1] + rd.read_g(o) + rd.fusion_g(c), need_vg=need_vg)

    def _extra(self, model, ue, dev):
        rd = model.reader
        nt = self.n_teach
        parts = []
        if nt:
            zero = ue.new_zeros((1, rd.D))
            kvs = [layer.kv(zero).float() for layer in rd.layers]
            ix = torch.zeros((nt, 1), dtype=torch.long, device=dev)
            parts.append(rd.read_bank(ue[:nt] + rd.position(torch.zeros(nt, device=dev), ue.dtype), kvs, ix,
                                      torch.ones_like(ix, dtype=torch.bool)))
        if len(ue) > nt:
            r = self.records
            src = torch.cat([ue.new_zeros((1, rd.D)), r['u'] + rd.position(r['ut'], r['u'].dtype),
                             r['mem'] + rd.position(r['mt'], r['mem'].dtype)])
            kvs = [layer.kv(src).float() for layer in rd.layers]
            q = ue[nt:] + rd.position(r['qt'], ue.dtype)
            parts.append(rd.read_bank(q, kvs, r['indices'], r['valid']))
        return torch.cat(parts)


@torch.no_grad()
def record_constants(model, requests, micro):
    """(M1, an approximation) the constant contexts of imitation records (teacher v2 / cf: (raw history, row,
    pending) requests) re-encoded without gradient with the current weights from their raw histories (the source
    path's SourceBatch, values only): dict for TruncatedPart (u [m, D], ut, mem [k, D], mt, qt [r], indices [r, S],
    valid [r, S]); bank of each: the zero entry, its W most recent frames, its closed blocks' tokens. M2 replaces this
    with the values the lanes computed."""
    from .rebuild_memory import SourceBatch
    rd = model.reader
    sb = SourceBatch(model, requests, micro=micro, cache=False)
    u, mem = sb.u.float(), sb.mem.float()
    banks = []
    for h, _, _ in requests:
        L = h.length
        frames = sb.history_frames[h.uid][max(0, L - rd.W):L]
        memory = sb.history_memory[h.uid][:(L // rd.block) * rd.tokens]
        banks.append(np.r_[0, 1 + frames, 1 + len(u) + memory].astype(np.int64))
    S = max(len(x) for x in banks)
    idx = np.zeros((len(banks), S), np.int64)
    valid = np.zeros((len(banks), S), bool)
    for i, x in enumerate(banks):
        idx[i, :len(x)] = x; valid[i, :len(x)] = True
    dev = u.device
    return dict(u=u, ut=sb.frame_times.to(dev), mem=mem, mt=sb.memory_times.to(dev), qt=sb.query_times.to(dev),
                indices=torch.as_tensor(idx, device=dev), valid=torch.as_tensor(valid, device=dev))


def pad_compact(y, rec):
    """A compact encode's tokens (each record's real tokens contiguous; rec [n, 2]: counts, starts) in the padded
    [n, width, d] layout the Reader's fusion reads, each record's last real token (the global one) in the last column,
    pad True on padding (as SourcePipeline._gather_queries; differentiable)."""
    counts, starts = rec[:, 0].astype(np.int64), rec[:, 1].astype(np.int64)
    n, width = len(counts), int(counts.max())
    index = np.repeat(np.arange(n), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    dev = y.device
    target = torch.as_tensor(index * width + np.where(within == counts[index] - 1, width - 1, within), device=dev)
    source = torch.as_tensor(starts[index] + within, device=dev)
    out = y.new_zeros((n * width, y.shape[-1])).index_copy(0, target, y.index_select(0, source))
    pad = torch.ones(n * width, dtype=torch.bool, device=dev)
    pad[target] = False
    return out.reshape(n, width, -1), pad.reshape(n, width)
