"""B44 packed learner: one execution per learner part instead of one per exact chunk length.

forward_chunks_paged (the B42 path) ran once per exact chunk length (forward_bucketed's groups):
~27 calls per 8192-row part, each repeating the host work (decode upload, page gathers, bank
metadata, new pinned buffers) and a Spatial/Writer/Reader pass on ~45 rows per chunk.
forward_packed runs the same math once per part:

* host, once per part: every chunk's snapshot parts; the historical selection (same generator
  calls in the same order as forward_bucketed + forward_chunks_paged); host decodes (DECODE_CACHE /
  the store's block-source cache); the row-packed Spatial plans; the page gathers (window K/V,
  open-tail latents, block values) and every index/metadata table of the Writer and the Reader.
  All of it is written into one persistent pinned staging slab (Staging; two alternating
  buffers, grown rarely, never pinned per call) and uploaded with one copy.
* Spatial: the main rows, the historical blocks' sources and the open tails' sources in one
  batch, one compiled call per row-total bucket (structured_input.T_BUCKETS) of at most
  SPATIAL_ROWS rows, each a non-reentrant checkpoint with gradients (see SPATIAL_CKPT).
* Writer: the historical blocks and every chunk's newly completed blocks in one call.
* Reader: one call per chunk-length bucket T in T_LEARN (chunks padded to T; FlexAttention for
  T > 1, see read_flex). Time boundaries
  are data, not calls: each chunk is its own batch row (its own prefix K/V, bank and steps), so
  segments of one bucket never see each other; a padded position has a later step than every
  real one of its row, so the causal step mask already hides it from real queries, and it writes
  no block; its output rows are dropped. The bank side uses each row's block-end table: a bank
  token is visible to a query only after its block's end step (long_mask), so no decision reads
  memory written after it.

The model definitions are unchanged; values equal forward_chunks_paged's up to summation order.
ISAAC_PACKED_LEARNER=0 restores the per-length path (the equivalence reference).
"""
import os
import time

import numpy as np
import torch

from . import block_paged as bp

T_LEARN = (1, 32, 64, 128)
PACKED = os.environ.get('ISAAC_PACKED_LEARNER', '1') == '1'
# Spatial with gradients: one compiled call per row-total bucket, each split into calls of at most
# SPATIAL_ROWS rows, each a non-reentrant checkpoint (SPATIAL_CKPT=1). Without checkpoint the main +
# historical rows of one 8192-row part (~16k rows, ~0.7M tokens) needed > 23.6 GB on the 4090 (B44
# fixed-input benchmark: OOM in the first training part); 4096-row checkpoints peak at 7.7 GB.
SPATIAL_CKPT = int(os.environ.get('ISAAC_LEARN_SPATIAL_CKPT', '1'))
SPATIAL_ROWS = int(os.environ.get('ISAAC_LEARN_SPATIAL_ROWS', '4096'))


PREP_LOCK = __import__('threading').Lock()   # (B48: one host preparation at a time: page store caches, staging)
PREFETCH_ON = os.environ.get('ISAAC_LEARN_PREFETCH', '1') == '1'


class Prefetch:
    """B48: the next learner part's host preparation (_prepare_packed) on a background thread while the
    current part runs on the GPU. submit(model, chunks, **kwargs) starts it (grad mode: True, the PPO
    parts); forward_packed takes it when called with the same chunks (by identity) and arguments, else
    it is dropped (its staging buffer freed). The thread does host work only (numpy decodes and index
    arithmetic, the pinned slab fill); the parts' draws stay in the parts' order (one preparation at a
    time: PREP_LOCK)."""
    def __init__(self):
        self.job = None
        self.hits = self.misses = 0

    @staticmethod
    def _key(chunks, kw, grad):
        return (tuple(id(c) for c in chunks), tuple(sorted((k, v) for k, v in kw.items() if k != 'generator')),
                id(kw.get('generator')), bool(grad))

    def submit(self, model, chunks, **kw):
        import threading
        self.drop()
        if not PREFETCH_ON or not chunks:
            return
        full = dict(generator=None, train_history=True, version=0, spatial_micro=None, require_next_state=False,
                    deduplicate_sources=True, history_mode='random', max_rows=None)
        full.update(kw)
        box = {}

        def work():
            with PREP_LOCK:
                t0 = time.perf_counter()
                try:
                    box['P'] = _prepare_packed(model, chunks, True, **full)
                    bp.ucount('learn_prefetch_host_s', time.perf_counter()-t0)
                except BaseException as e:   # (re-raised by the caller that takes it)
                    box['error'] = e
        th = threading.Thread(target=work, name='learner-prefetch', daemon=True)
        th.start()
        self.job = (self._key(chunks, full, True), th, box, list(chunks))

    def take(self, chunks, kw, grad):
        """The prepared part for exactly these chunks / arguments (waited for), else None: another call
        (e.g. a no-grad forward between two PPO parts) leaves the prepared part for its own call."""
        if self.job is None:
            return None
        key, th, box, _ = self.job
        if key != self._key(chunks, kw, grad):
            bp.ucount('learn_prefetch_other_calls', 1)
            return None
        self.job = None
        t0 = time.perf_counter()
        th.join()
        bp.ucount('learn_prefetch_wait_s', time.perf_counter()-t0)
        if 'error' in box:
            raise box['error']
        self.hits += 1
        bp.ucount('learn_prefetch_hits', 1)
        return box['P']

    @staticmethod
    def _free(box):
        P = box.get('P')
        if P is not None and P['slab'].cuda:
            STAGING.drop(P['slab'].k)

    def drop(self):
        """The prepared part, unused, given up (its staging buffer freed)."""
        if self.job is not None:
            _, th, box, _ = self.job
            self.job = None
            th.join()
            self._free(box)
            self.misses += 1
            bp.ucount('learn_prefetch_dropped', 1)


PREFETCH = Prefetch()


# Reader attention of the T > 1 buckets: 'flex' (FlexAttention, masks and the learnable relative /
# age biases evaluated inside the kernel: no [B,H,Q,K] bias tensors or their gradients) or 'sdpa'
# (BlockMemory.read itself: the equivalence reference).
READER = os.environ.get('ISAAC_PACKED_READER', 'flex')


_FLEX = {}


def _flex():
    if 'f' not in _FLEX:
        from torch.nn.attention.flex_attention import flex_attention
        from .block_compile import configure, _backend
        configure()
        # compiles counted in block_compile.STATS (after warm-up: steady compiles)
        _FLEX['f'] = torch.compile(flex_attention, backend=_backend, dynamic=True)
    return _FLEX['f']


def warm_reader(learner, bf16=True):
    """Compile the Flex Reader kernels before training (read_flex on synthetic chunks): each
    T > 1 bucket, two batch / prefix / bank sizes (symbolic afterwards), the learner's three
    modes (training forward+backward, no_grad, inference_mode). Returns seconds."""
    t0 = time.perf_counter()
    memory = learner.block_memory
    c = memory.config
    dev = next(learner.parameters()).device
    if READER != 'flex' or dev.type != 'cuda' or c.width//c.heads < 16:
        return 0.0
    dh = c.width//c.heads
    kv_dtype = torch.bfloat16 if bf16 else torch.float32
    for T in [t for t in T_LEARN if t > 1]:
        # key lengths: multiples of 128; one-tile and several-tile prefix / bank tables (a table
        # dimension of 1 compiles its own Flex kernel: an all-empty-prefix bucket, a short bank)
        for b, pd, nb in ((2, c.window, 48), (3, 32, 16), (2, 0, 16), (3, 0, 48)):
            g = torch.Generator().manual_seed(T+b)
            steps = torch.full((b,), 4000, dtype=torch.long)
            position = (steps[:, None]+torch.arange(T)).to(dev)
            kp = (steps[:, None]-pd+torch.arange(pd)).repeat_interleave(c.latents, 1).to(dev)
            m = nb*c.memory_tokens
            ends = (steps[:, None]-1-c.block_steps*torch.arange(nb)).flip(1).repeat_interleave(c.memory_tokens, 1).to(dev)
            for mode in ('grad', 'no_grad', 'inference'):
                ctx = torch.inference_mode() if mode == 'inference' else torch.set_grad_enabled(mode == 'grad')
                with ctx, torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    z = torch.randn(b, T, c.latents, c.latent_width, generator=g).to(dev)
                    bank = torch.randn(b, m, c.latent_width, generator=g).to(dev)
                    prefixes = tuple(tuple(torch.randn(b, c.heads, pd*c.latents, dh, generator=g).to(dev, kv_dtype)
                                           for _ in (0, 1)) for _ in range(c.layers))
                    if mode == 'grad':
                        z.requires_grad_(True)
                    h = read_flex(memory, z, position, prefixes, bank, ends, kp)
                if mode == 'grad':
                    h.float().sum().backward()
    learner.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.perf_counter()-t0


def _kernel_options(dtype):
    """Triton tile sizes that fit consumer Ada/Ampere shared memory (101 KB on the 4090): torch
    2.7.1's defaults for head dim 64 (A100 configs, 3 stages) need ~115 KB and fail to compile."""
    cap = torch.cuda.get_device_capability()
    if cap[0] != 8 or cap[1] == 0:
        return None
    if dtype == torch.float32:
        return dict(fwd_BLOCK_M=32, fwd_BLOCK_N=32, fwd_num_warps=4, fwd_num_stages=2)
    return dict(fwd_BLOCK_M=64, fwd_BLOCK_N=64, fwd_num_warps=4, fwd_num_stages=2,
                bwd_BLOCK_M1=32, bwd_BLOCK_N1=64, bwd_BLOCK_M2=64, bwd_BLOCK_N2=32, bwd_num_warps=4,
                bwd_num_stages=2)


def _block_mask(allowed, mask_mod, block=128):
    """A FlexAttention BlockMask from a dense [B, Q, K] mask (Q, K multiples of `block`): per
    128x128 tile, full (all visible: the kernel skips mask_mod), partial (mask_mod applied in the
    kernel) or empty (skipped). Built with plain tensor ops: torch 2.7.1's compiled
    create_block_mask returned uninitialised block counts after a shape change (B44)."""
    from torch.nn.attention.flex_attention import BlockMask
    B, Q, K = allowed.shape
    if Q % block or K % block:
        raise ValueError(f'Flex Reader lengths must be multiples of {block}: Q={Q}, K={K}')
    m = allowed.view(B, Q//block, block, K//block, block)
    some = m.any(4).any(2)
    full = m.all(4).all(2)

    def table(blk):
        num = blk.sum(-1, dtype=torch.int32)[:, None].contiguous()
        idx = torch.argsort((~blk).to(torch.int8), dim=-1, stable=True).to(torch.int32)[:, None].contiguous()
        return num, idx
    kn, ki = table(some & ~full)
    fn, fi = table(full)
    return BlockMask.from_kv_blocks(kn, ki, fn, fi, BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(Q, K))


def read_flex(memory, z, position, prefixes, bank, ends, kp):
    """BlockMemory.read (same parameters, same function) with FlexAttention for both attentions:
    the short window (prefix + current keys, mask 0 <= q_step - k_step < window, score + the
    learnable short_relative[h, distance]) and the long bank (a bank token visible after its
    block's end step and within memory_blocks blocks, score + memory_age[h, age]). Each batch row is
    one chunk (its own keys, bank and steps); masks and biases are evaluated per (row, query, key)
    inside the kernel. Returns hidden [b, t, width]."""
    from torch.nn import functional as F
    c = memory.config
    b, t = z.shape[:2]
    qt = position.repeat_interleave(c.latents, 1)                  # [b, Q]
    keys = torch.cat((kp, qt), 1)                                  # [b, P+Q]
    qn, kn, mn = qt.shape[1], keys.shape[1], ends.shape[1]
    window, bs, blocks = c.window, c.block_steps, c.memory_blocks
    short_rel, age_w = memory.short_relative, memory.memory_age

    # Flex evaluates these at the padded positions of a partial 128-block too: every index into a
    # captured tensor is clamped, and positions past the lengths are masked (block_attention.plan).
    def short_mask(bb, h, q, k):
        qs, ks = qt[bb, q.clamp(max=qn-1)], keys[bb, k.clamp(max=kn-1)]
        d = qs-ks
        return (q < qn) & (k < kn) & (ks >= 0) & (d >= 0) & (d < window)

    def short_score(score, bb, h, q, k):
        d = qt[bb, q.clamp(max=qn-1)]-keys[bb, k.clamp(max=kn-1)]
        return score+short_rel[h, d.clamp(0, window-1)]

    def bank_mask(bb, h, q, k):
        e = ends[bb, k.clamp(max=mn-1)]
        d = qt[bb, q.clamp(max=qn-1)]-e
        return (q < qn) & (k < mn) & (e >= 0) & (d > 0) & (torch.div(d-1, bs, rounding_mode='floor') < blocks)

    def bank_score(score, bb, h, q, k):
        d = qt[bb, q.clamp(max=qn-1)]-ends[bb, k.clamp(max=mn-1)]
        return score+age_w[h, torch.div(d-1, bs, rounding_mode='floor').clamp(0, blocks-1)]
    flex = _flex()
    # dense masks (int32 steps) -> block masks; a query that sees no bank token reads zeros
    # (CrossAttention's allowed.any rule)
    q32, k32, e32 = qt.int(), keys.int(), ends.int()
    ds = q32[:, :, None]-k32[:, None, :]
    sm = _block_mask((k32[:, None, :] >= 0) & (ds >= 0) & (ds < window), short_mask)
    de = q32[:, :, None]-e32[:, None, :]
    bank_allowed = (e32[:, None, :] >= 0) & (de > 0) & (de <= blocks*bs)
    seen = bank_allowed.any(-1, keepdim=True)
    bm = _block_mask(bank_allowed, bank_mask)
    del ds, de, bank_allowed
    x = (memory.input(z)+memory.slot_embedding[None, None]).flatten(1, 2)
    for layer, prefix in zip(memory.layers, prefixes):
        nb, n, d = x.shape
        q, k, v = layer.qkv(layer.n1(x)).reshape(nb, n, 3, layer.heads, layer.dh).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        pk, pv = (a.to(k.dtype) for a in prefix)
        y = flex(q.contiguous(), torch.cat((pk, k), 2).contiguous(), torch.cat((pv, v), 2).contiguous(),
                 score_mod=short_score, block_mask=sm, kernel_options=_kernel_options(q.dtype))
        x = x+layer.out(y.transpose(1, 2).reshape(nb, n, d))
        if layer.memory is not None:
            ca = layer.memory
            xq = layer.n_memory(x)
            qq = ca.q(xq).reshape(nb, n, ca.heads, ca.dh).transpose(1, 2)
            kk, vv = ca.keys(bank)
            yy = flex(qq.contiguous(), kk.to(qq.dtype).contiguous(), vv.to(qq.dtype).contiguous(),
                      score_mod=bank_score, block_mask=bm, kernel_options=_kernel_options(qq.dtype))
            yy = ca.out(yy.transpose(1, 2).reshape(nb, n, d))
            x = x+torch.where(seen, yy, torch.zeros_like(yy))
        x = x+layer.ff(layer.n2(x))
    tokens = memory.norm(x).reshape(b, t, c.latents, c.width)
    return memory.pool(tokens.flatten(0, 1)).reshape(b, t, c.width)


def chunk_bucket(n):
    """The Reader layout length of a chunk of n decisions (past T_LEARN: doubling)."""
    for t in T_LEARN:
        if n <= t:
            return t
    t = T_LEARN[-1]
    while t < n:
        t *= 2
    return t


class Staging:
    """Persistent pinned host slabs for the learner's per-part upload. A buffer is taken by acquire and
    given back by release (after the upload: its copy's event) or drop (never uploaded); it is rewritten
    only once free and its last copy has finished. B48: a pool, not two alternating buffers (a part
    prepared ahead by the prefetch thread holds its buffer until it runs); a buffer is reallocated only
    when a part needs more bytes (x1.25 headroom)."""
    ALIGN = 256

    def __init__(self):
        import threading
        self.bufs = []   # [buffer, event of its last copy, busy]
        self.lock = threading.Lock()
        self.allocs = 0

    def acquire(self, n):
        with self.lock:
            free = [i for i, b in enumerate(self.bufs) if not b[2]]
            fit = [i for i in free if self.bufs[i][0].numel() >= n]
            done = [i for i in fit if self.bufs[i][1] is None or self.bufs[i][1].query()]
            i = (done or fit or free or [None])[0]
            if i is None:
                self.bufs.append([None, None, False])
                i = len(self.bufs)-1
            self.bufs[i][2] = True
            ev = self.bufs[i][1]
        if ev is not None:
            ev.synchronize()
        b = self.bufs[i]
        b[1] = None
        if b[0] is None or b[0].numel() < n:
            b[0] = None
            b[0] = torch.empty(int(n*1.25)+self.ALIGN, dtype=torch.uint8, pin_memory=True)
            self.allocs += 1
        return i, b[0]

    def release(self, i):
        ev = torch.cuda.Event()
        ev.record()
        with self.lock:
            self.bufs[i][1] = ev
            self.bufs[i][2] = False

    def drop(self, i):
        """A buffer whose slab was never uploaded (a prefetched part not used)."""
        with self.lock:
            self.bufs[i][2] = False


STAGING = Staging()


def _nbytes(shape, dtype):
    return int(np.prod(shape, dtype=np.int64))*torch.empty((), dtype=dtype).element_size()


class Slab:
    """One part's host tensors in one staging buffer: .host[name] are writable CPU views; upload()
    returns the device views after a single host->device copy (CPU devices: the host tensors)."""

    def __init__(self, specs, device):
        self.device = torch.device(device)
        self.layout, at = {}, 0
        for name, shape, dtype in specs:
            at = (at+Staging.ALIGN-1)//Staging.ALIGN*Staging.ALIGN
            n = _nbytes(shape, dtype)
            self.layout[name] = (at, n, tuple(int(x) for x in shape), dtype)
            at += n
        self.total = max(at, 1)
        self.cuda = self.device.type == 'cuda'
        if self.cuda:
            self.k, self.base = STAGING.acquire(self.total)
        else:
            self.base = torch.empty(self.total, dtype=torch.uint8)
        self.host = {k: self._view(self.base, *v) for k, v in self.layout.items()}

    @staticmethod
    def _view(base, at, n, shape, dtype):
        return base.narrow(0, at, n).view(dtype).view(shape)

    def upload(self):
        if not self.cuda:
            return self.host
        flat = self.base.narrow(0, 0, self.total).to(self.device, non_blocking=True)
        STAGING.release(self.k)
        bp.ucount('learn_staged_mb', self.total/2**20)
        return {k: self._view(flat, *v) for k, v in self.layout.items()}


def _np_dtype(t):
    return torch.from_numpy(np.empty(0, dtype=t)).dtype


def _compact(rows, pending, packed):
    """Compact decode of records (or a (records, frame arena) pair) with their actions under way."""
    from .structured_input import decode_frames_compact
    d = decode_frames_compact([rows], packed=packed)[0]
    d['pending'] = np.asarray(pending, dtype=np.int64)
    return d


def _host_decode(chunks, packed):
    """Main rows' compact host decodes, once per update (B42 fix 3: DECODE_CACHE / the chunk's
    own); this call's fresh chunks in one joint decode."""
    from .structured_input import decode_frames_compact
    tag = 'compact1' if packed else 'compact0'
    hosts = [None]*len(chunks)
    fresh, keys = [], []
    bp.ucount('learn_main_decode_calls', 1)
    for i, c in enumerate(chunks):
        h = c.__dict__.get('_host')
        if h is not None and h[0] == tag:
            hosts[i] = h[1]
            continue
        # (the per-length path caches padded decodes under rows_key(..., packed): a distinct key)
        key = bp.rows_key(c.rows, c.pending, packed)+(tag,) if len(c.rows) > 1 else None
        d = bp.DECODE_CACHE.get(key) if key is not None else None
        if d is None:
            fresh.append(i); keys.append(key)
        else:
            hosts[i] = d
            object.__setattr__(c, '_host', (tag, d))
    for i in range(len(chunks)):
        if hosts[i] is not None:
            bp.ucount('learn_main_decode_rows_cached', len(chunks[i].rows))
    if fresh:
        t0 = time.perf_counter()
        # (B45: compact rollout records are read from their frame arena directly)
        ds = decode_frames_compact([(chunks[i].rows, chunks[i].frames) if chunks[i].frames is not None
                                    else chunks[i].rows for i in fresh], packed)
        for i, key, d in zip(fresh, keys, ds):
            c = chunks[i]
            bp.ucount('learn_main_decode_rows_fresh', len(c.rows))
            d['pending'] = np.asarray(c.pending, dtype=np.int64)
            hosts[i] = d
            if key is not None:
                bp.DECODE_CACHE[key] = d
                object.__setattr__(c, '_host', (tag, d))
        bp.ucount('learn_main_decode_s', time.perf_counter()-t0)
    return hosts


def _block_decode(store, bids, packed, block_steps):
    """Historical blocks' compact source decodes: cached in the page store across rollouts (pruned
    when a block is freed, PagedStore.collect); this call's fresh blocks in one source_rows and one
    joint decode."""
    from .structured_input import decode_frames_compact
    tag = 'compact1' if packed else 'compact0'
    bids = np.asarray(bids, np.int64).tolist()
    hosts = [store.decoded.get((bid, tag)) for bid in bids]
    fresh = [i for i, h in enumerate(hosts) if h is None]
    bp.ucount('learn_hist_decode_rows_cached', (len(bids)-len(fresh))*block_steps)
    if fresh:
        t0 = time.perf_counter()
        bp.ucount('learn_hist_decode_rows_fresh', len(fresh)*block_steps)
        # (B45: the raw frames joined into one byte array, decoded in place)
        rows_b, buf_b, pend_b = store.source_arena(np.concatenate([store.b_src[bids[i]] for i in fresh]))
        parts = [(rows_b[j*block_steps:(j+1)*block_steps], buf_b) for j in range(len(fresh))]
        for j, (i, d) in enumerate(zip(fresh, decode_frames_compact(parts, packed))):
            d['pending'] = pend_b[j*block_steps:(j+1)*block_steps]
            hosts[i] = d
            store.cache_decoded((bids[i], tag), d)
        bp.ucount('learn_hist_decode_s', time.perf_counter()-t0)
    return hosts


ROW_KEYS = ('v_room', 'v_map', 'pending')


def _merge_into(hosts, slab_specs):
    """The merged layout of compact host decodes: per-row arrays concatenated, each compact list
    concatenated (real entries only) plus one zero entry. Registers the staging specs; returns
    (shapes, counts [n,5], starts {name: (first entry per row, zero-entry index)})."""
    from .structured_input import RAGGED
    counts = np.concatenate([h['counts'] for h in hosts])
    n = len(counts)
    shapes, starts = {}, {}
    for k in ROW_KEYS:
        x0 = hosts[0][k]
        shapes[k] = (n, *x0.shape[1:])
        slab_specs.append(('h:'+k, shapes[k], _np_dtype(x0.dtype)))
    for i, name in enumerate(RAGGED):
        c = counts[:, i]
        K = int(c.sum())
        starts[name] = (np.concatenate(([0], np.cumsum(c)[:-1])).astype(np.int64), K)
        shapes['c_'+name] = (K+1, hosts[0]['c_'+name].shape[1])
        slab_specs.append(('h:c_'+name, shapes['c_'+name], torch.float32))
    return shapes, counts, starts


def _fill_merged(hosts, shapes, host_views):
    for k, shape in shapes.items():
        out = host_views['h:'+k].numpy()
        at = 0
        compact = k.startswith('c_')
        for h in hosts:
            x = h[k]
            out[at:at+len(x)] = x
            at += len(x)
        if compact:
            out[at:] = 0


def forward_packed(model, chunks, generator=None, train_history=True, version=0, spatial_micro=None,
                   require_next_state=False, deduplicate_sources=True, history_mode='random', max_rows=None):
    """forward_bucketed(model, chunks) for paged chunks in one execution (module docstring).
    Returns (outputs in the chunks' decision order, ctx with o_seq, z_out, recompute_stats, replay).
    B48: the host preparation (_prepare_packed) is the one PREFETCH made ahead for these chunks and
    arguments when there is one (the same work, done earlier on the prefetch thread), else done here."""
    grad = torch.is_grad_enabled()
    kw = dict(generator=generator, train_history=train_history, version=version, spatial_micro=spatial_micro,
              require_next_state=require_next_state, deduplicate_sources=deduplicate_sources,
              history_mode=history_mode, max_rows=max_rows)
    P = PREFETCH.take(chunks, kw, grad)
    if P is None:
        with PREP_LOCK:
            P = _prepare_packed(model, chunks, grad, **kw)
    return _run_packed(model, P)


def _prepare_packed(model, chunks, grad, generator=None, train_history=True, version=0, spatial_micro=None,
                    require_next_state=False, deduplicate_sources=True, history_mode='random', max_rows=None):
    """B48: forward_packed's host part (history draw, decodes, plans, index arrays, the filled staging
    slab); touches no parameter and no CUDA stream (runs on the prefetch thread). grad: the run's
    autograd mode (thread-local, so passed in)."""
    from .structured_input import packed_rows_plan, t_bucket
    if history_mode not in bp.HISTORY_MODES:
        raise ValueError(f'history_mode {history_mode!r}: one of {bp.HISTORY_MODES}')
    if not chunks:
        raise ValueError('forward_packed requires a nonempty batch')
    t0 = time.perf_counter()
    device = next(model.parameters()).device
    if spatial_micro is None:
        spatial_micro = bp.SPATIAL_MICRO_TRAIN if grad else bp.SPATIAL_MICRO_EVAL
    memory = model.block_memory
    cfg = memory.config
    bs, mt, lat, lw = cfg.block_steps, cfg.memory_tokens, cfg.latents, cfg.latent_width
    store = chunks[0].snapshot.store
    B = len(chunks)
    lens = np.array([len(c.rows) for c in chunks], np.int64)
    bp.ucount('learn_packed_calls', 1)
    bp.count('learn.calls', 1); bp.count('learn.chunks', B); bp.count('learn.rows', int(lens.sum()))
    parts = []
    for chunk in chunks:
        snap = chunk.snapshot
        if snap.store is not store:
            raise ValueError('a paged learner batch must use one page store')
        if chunk.rows['first'][0]:
            parts.append(dict(step=0, kv_steps=np.empty(0, np.int64), kvs=np.empty(0, np.int64),
                              pdec=np.empty(0, np.int64), pkvs=np.empty(0, np.int64),
                              bid=np.empty(0, np.int64), ends=np.empty(0, np.int64)))
        else:
            parts.append(store.snapshot_parts(snap))
    steps = np.array([p['step'] for p in parts], np.int64)
    # ---- historical selection: forward_bucketed's order (length groups by first appearance)
    seen = {}
    for n in lens.tolist():
        seen.setdefault(n, len(seen))
    selected = [np.empty(0, np.int64)]*B
    if train_history and history_mode == 'random' and cfg.history_recompute_blocks:
        for i in sorted(range(B), key=lambda i: (seen[int(lens[i])], i)):
            p = parts[i]
            eligible = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
            if not len(eligible):
                continue
            pick = torch.randperm(len(eligible), generator=generator)[:cfg.history_recompute_blocks].numpy()
            selected[i] = eligible[pick]
            if grad:
                bp.HIST['age'] += ((p['step']-p['ends'][eligible[pick]])//bs).tolist()
    unique = {}
    for p, sel in zip(parts, selected):
        for k in sel:
            unique.setdefault(int(p['bid'][k]), len(unique))
    ub = np.fromiter(unique.keys(), np.int64, len(unique))
    # ---- host decodes: main rows | historical block sources | open-tail sources
    packed = device.type == 'cuda'
    hosts = _host_decode(chunks, packed)
    n_main = int(lens.sum())
    hist_hosts = _block_decode(store, ub, packed, bs) if len(ub) else []
    n_hist = len(ub)*bs
    enc = [i for i, p in enumerate(parts) if len(p['pdec']) and lens[i]+len(p['pdec']) > bs]
    tail_hosts, n_tail = [], 0
    if enc:
        rows_t, buf_t, pend_t = store.source_arena(np.concatenate([parts[i]['pdec'] for i in enc]))
        tail_hosts, n_tail = [_compact((rows_t, buf_t), pend_t, packed)], len(rows_t)
        bp.ucount('learn_tail_decode_rows', n_tail)
    all_hosts = hosts+hist_hosts+tail_hosts
    n_rows = n_main+n_hist+n_tail
    specs = []
    shapes, counts, starts = _merge_into(all_hosts, specs)
    # ---- Spatial plans: rows sorted by real tokens, one compiled call per row-total bucket
    tot = 30+counts.sum(1)
    order_h = np.argsort(tot, kind='stable')
    tb = np.asarray([t_bucket(int(x)) for x in tot[order_h]], np.int64)
    cuts = [0]+(np.flatnonzero(np.diff(tb))+1).tolist()+[n_rows]
    groups = []
    for a, b_ in zip(cuts[:-1], cuts[1:]):
        step_ = SPATIAL_ROWS if SPATIAL_ROWS > 0 else b_-a
        groups += [(x, min(x+step_, b_)) for x in range(a, b_, step_)]
    plans = []
    ints = {}
    for g, (a, b_) in enumerate(groups):
        idx, T = packed_rows_plan(None, order_h[a:b_], counts[order_h[a:b_]], starts)
        names = list(idx)
        for k in names:
            ints[f'sp{g}:{k}'] = idx[k]
        plans.append((g, names, T, b_-a))
        bp.ucount('learn_spatial_calls', 1)
        bp.ucount('learn_spatial_tokens_real', int(tot[order_h[a:b_]].sum()))
        bp.ucount('learn_spatial_tokens_padded', max(2, b_-a)*T)
    inverse = np.empty_like(order_h)
    inverse[order_h] = np.arange(n_rows)
    ints['sp:inverse'] = inverse
    # ---- page rows: window K/V, open-tail latents of chunks without re-encoded tails, block values
    enc_set = set(enc)
    rest = [i for i in range(B) if i not in enc_set]
    pk_all = np.concatenate([parts[i]['pkvs'] for i in rest]) if rest else np.empty(0, np.int64)
    all_kvs = np.concatenate([p['kvs'] for p in parts])
    ukv, kinv = np.unique(all_kvs, return_inverse=True)
    n_old = np.array([len(p['bid']) for p in parts], np.int64)
    all_bid = np.concatenate([p['bid'] for p in parts])
    ubid, binv = np.unique(all_bid, return_inverse=True)
    specs += [('kv', (len(ukv)+1, *store.kv_shape), store.kv_dtype),
              ('zpage', (len(pk_all), *store.z_shape), torch.float32),
              ('blocks', (len(ubid), *store.b_shape), torch.float32)]
    # ---- Writer: [historical blocks | new blocks]; source rows in pool = [Spatial rows | page tails]
    tail_len = np.array([len(p['pdec']) for p in parts], np.int64)
    tail_src = [None]*B
    at_t, at_p = n_main+n_hist, n_rows
    for i in range(B):
        if i in enc_set:
            tail_src[i] = at_t+np.arange(tail_len[i]); at_t += tail_len[i]
        else:
            tail_src[i] = at_p+np.arange(len(parts[i]['pkvs'])); at_p += len(parts[i]['pkvs'])
    row_start = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    complete = (tail_len+lens)//bs
    new_idx, new_ends, new_start = [], [], np.zeros(B, np.int64)
    for i in range(B):
        new_start[i] = len(new_ends)
        if not complete[i]:
            continue
        seq = np.concatenate((tail_src[i], row_start[i]+np.arange(lens[i])))
        first_end = int(steps[i])-int(tail_len[i])+bs-1
        for j in range(int(complete[i])):
            new_idx.append(seq[j*bs:(j+1)*bs]); new_ends.append(first_end+j*bs)
    n_new = len(new_ends)
    if n_new:
        ints['w:new'] = np.stack(new_idx).reshape(-1)
    # ---- old block values: page values, historical replacements (random) / gradient leaves (gradreplay)
    replay_host = None
    if n_old.sum():
        source = binv.copy()
        if history_mode == 'gradreplay' and train_history and grad:
            groups_r, at = [], 0
            for p in parts:
                nb = len(p['bid'])
                elig = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
                groups_r.append((binv[at+elig], (p['step']-p['ends'][elig])//bs))
                source[at+elig] += len(ubid)
                at += nb
            replay_host = groups_r
        if len(ub):
            at = 0
            for p, sel in zip(parts, selected):
                for k in sel:
                    source[at+k] = len(ubid)+(len(ubid) if replay_host is not None else 0)+unique[int(p['bid'][k])]
                at += len(p['bid'])
        ints['old:source'] = source
    old_start = np.concatenate(([0], np.cumsum(n_old)[:-1])).astype(np.int64)
    n_old_total = int(n_old.sum())
    # ---- Reader buckets
    tbk = np.array([chunk_bucket(int(n)) for n in lens], np.int64)
    flex_cap = None
    rbuckets = []
    for T in sorted(set(tbk.tolist())):
        ids = np.flatnonzero(tbk == T)
        # (T > 1: at least 2 rows, a fully padded one if needed: the compiled Flex kernels keep a
        # symbolic batch, a 1-row bucket would specialise and compile again)
        Bb = max(len(ids), 2) if T > 1 else len(ids)
        flex_on = cfg.attention == 'flex' and device.type == 'cuda' and T == cfg.window and T > 1
        m_blocks = n_old[ids]+complete[ids]
        max_m = max(1, int(m_blocks.max())*mt)
        Pd = int(max(len(parts[i]['kvs']) for i in ids))
        if T > 1:
            # whole 128-token Flex blocks: bank and prefix keys rounded up (masked padding), so
            # the kernels see a few sizes only
            max_m = (max_m+127)//128*128
            Pd = (Pd*lat+127)//128*128//lat
        if flex_on:
            max_m = ((cfg.memory_blocks+(T+bs-1)//bs+1)*mt+127)//128*128
        zidx = np.full((Bb, T), n_main, np.int64)
        real = []
        kvidx = np.full((Bb, max(Pd, 0)), len(ukv), np.int64)
        kp = np.full((Bb, Pd*lat), -1, np.int64)
        ends = np.full((Bb, max_m), -1, np.int64)
        dst, src = [], []
        at_k = np.concatenate(([0], np.cumsum([len(p['kvs']) for p in parts])[:-1])).astype(np.int64)
        for r, i in enumerate(ids.tolist()):
            n = int(lens[i])
            zidx[r, :n] = row_start[i]+np.arange(n)
            real.append(r*T+np.arange(n))
            nk = len(parts[i]['kvs'])
            kvidx[r, :nk] = kinv[at_k[i]:at_k[i]+nk]
            kp[r, :nk*lat] = np.repeat(parts[i]['kv_steps'], lat)
            vals = np.concatenate((old_start[i]+np.arange(n_old[i]), n_old_total+new_start[i]+np.arange(complete[i])))
            e = np.concatenate((parts[i]['ends'], np.asarray(new_ends[new_start[i]:new_start[i]+complete[i]], np.int64)))
            if len(vals):
                tok = (vals[:, None]*mt+np.arange(mt)).reshape(-1)
                ends[r, :len(tok)] = np.repeat(e, mt)
                dst.append(r*max_m+np.arange(len(tok))); src.append(tok)
        key = f'r{T}:'
        ints[key+'z'] = zidx.reshape(-1); ints[key+'real'] = np.concatenate(real)
        ints[key+'kv'] = kvidx.reshape(-1); ints[key+'kp'] = kp.reshape(-1); ints[key+'ends'] = ends.reshape(-1)
        ints[key+'steps'] = np.concatenate((steps[ids], np.zeros(Bb-len(ids), np.int64)))
        if dst:
            ints[key+'dst'] = np.concatenate(dst); ints[key+'src'] = np.concatenate(src)
        rbuckets.append((T, ids, Bb, max_m, Pd, bool(dst)))
        bp.ucount('learn_reader_calls', 1)
        bp.ucount('learn_reader_rows_real', int(lens[ids].sum())); bp.ucount('learn_reader_rows_padded', Bb*T)

    out_order = np.concatenate([np.concatenate([row_start[i]+np.arange(lens[i]) for i in rb[1].tolist()])
                                for rb in rbuckets])
    out_inv = np.empty_like(out_order)
    out_inv[out_order] = np.arange(n_main)
    ints['out:inv'] = out_inv
    names_i = list(ints)
    sizes_i = [len(ints[k]) for k in names_i]
    specs.append(('ints', (int(sum(sizes_i)),), torch.int64))
    # ---- one staging slab, one upload
    slab = Slab(specs, device)
    _fill_merged(all_hosts, shapes, slab.host)
    if len(ukv):
        store.kv(ukv, out=slab.host['kv'][:len(ukv)])
    slab.host['kv'][len(ukv):] = 0
    if len(pk_all):
        store.z(pk_all, out=slab.host['zpage'])
    if len(ubid):
        store.blocks(ubid, out=slab.host['blocks'])
    if sizes_i:
        np.concatenate([np.asarray(ints[k], np.int64) for k in names_i], out=slab.host['ints'].numpy())
    bp.ucount('learn_packed_host_s', time.perf_counter()-t0)
    return dict(bs=bs, cfg=cfg, device=device, grad=grad, lat=lat, lw=lw, memory=memory, n_hist=n_hist,
                n_main=n_main, n_new=n_new, n_old_total=n_old_total, n_tail=n_tail, names_i=names_i, packed=packed,
                pk_all=pk_all, plans=plans, rbuckets=rbuckets, replay_host=replay_host, selected=selected,
                sizes_i=sizes_i, slab=slab, spatial_micro=spatial_micro, store=store, ub=ub, ubid=ubid,
                unique=unique, version=version)


def _run_packed(model, P):
    """B48: forward_packed's device part on a prepared part P (_prepare_packed): one upload, Spatial,
    Writer, Reader and heads."""
    from torch.utils.checkpoint import checkpoint
    from .structured_input import packed_rows_apply, RAGGED, canvas_decoder
    (bs, cfg, device, grad, lat, lw, memory, n_hist, n_main, n_new, n_old_total, n_tail, names_i, packed, pk_all,
     plans, rbuckets, replay_host, selected, sizes_i, slab, spatial_micro, store, ub, ubid, unique, version) = (
        P[k] for k in ('bs', 'cfg', 'device', 'grad', 'lat', 'lw', 'memory', 'n_hist', 'n_main', 'n_new',
                       'n_old_total', 'n_tail', 'names_i', 'packed', 'pk_all', 'plans', 'rbuckets', 'replay_host',
                       'selected', 'sizes_i', 'slab', 'spatial_micro', 'store', 'ub', 'ubid', 'unique', 'version'))
    if grad != torch.is_grad_enabled():
        raise RuntimeError('a learner part prepared for another autograd mode')
    with torch.inference_mode(False), torch.no_grad():
        dev = slab.upload()
        di = dict(zip(names_i, dev['ints'].split(sizes_i)))
        batch = {k: dev['h:'+k] for k in ROW_KEYS}
        batch.update(('v_'+name, dev['h:c_'+name]) for name in RAGGED)   # compact lists [K+1, width]
        if packed:   # (off the GPU the host decoded the canvases already: decode_frame_compact packed=False)
            with torch.autocast(device.type, enabled=False):
                batch['v_room'], local, batch['v_map'] = canvas_decoder()(
                    batch['v_room'], batch['v_local'][:, None], batch['v_map'])
                batch['v_local'] = local[:, 0]
    # ---- Spatial
    zs = []
    for g, names, T, m in plans:
        part = packed_rows_apply(batch, {k: di[f'sp{g}:{k}'] for k in names}, T)
        if grad and SPATIAL_CKPT and m > 0:
            zs.append(checkpoint(model.encode_block, part, use_reentrant=False)[:m])
        else:
            zs.append(model.encode_block(part)[:m])
    z_all = (torch.cat(zs) if len(zs) > 1 else zs[0]).index_select(0, di['sp:inverse'])
    encoded = z_all[:n_main]
    zdt = z_all.dtype
    U = dev['kv']
    stats = dict(selected_blocks=int(sum(map(len, selected))), unique_objects=len(unique),
                 encoded_blocks=len(unique), historical_rows=int(n_hist), tail_rows=int(n_tail))
    if grad and len(ub):
        bp.HIST['rows'] += n_hist; bp.HIST['blocks'] += len(ub)
    bp.count('learn.hist_rows', n_hist); bp.count('learn.tail_rows', n_tail)
    replay = []

    def call(z):
        # Writer: historical blocks and every chunk's new blocks, one call
        pool = torch.cat((z_all, dev['zpage'].to(zdt))) if len(pk_all) else z_all
        w_in = []
        if n_hist:
            w_in.append(z_all[n_main:n_main+n_hist].reshape(len(ub), bs, lat, lw))
        if n_new:
            w_in.append(pool.index_select(0, di['w:new']).view(n_new, bs, lat, lw))
        W = None
        if w_in:
            bp.ucount('learn_writer_calls', 1)
            W = memory.writer(torch.cat(w_in) if len(w_in) > 1 else w_in[0])
        hist_written = W[:len(ub)] if n_hist else None
        new_written = W[len(ub):] if n_new else None
        vparts = []
        if n_old_total:
            pool_b = dev['blocks']
            if replay_host is not None:
                leaf = pool_b.detach()
                if bp._LEAF_HOOK[0] is not None:
                    leaf = bp._LEAF_HOOK[0](ubid, leaf)
                leaf = leaf.detach().requires_grad_(True)
                replay.append(dict(pool=leaf, ubid=ubid, store=store, chunks=replay_host, spatial_micro=spatial_micro))
                pool_b = torch.cat((pool_b.detach(), leaf))
            if hist_written is not None:
                pool_b = torch.cat((pool_b.to(hist_written.dtype), hist_written))
            vparts.append(pool_b.index_select(0, di['old:source']).to(zdt))
        if new_written is not None:
            vparts.append(new_written.to(zdt))
        vt = None
        if vparts:
            vt = ((torch.cat(vparts) if len(vparts) > 1 else vparts[0])+memory.memory_slot_embedding).reshape(-1, lw)
        zpad = torch.cat((z, z.new_zeros((1, lat, lw))))
        outs = []
        dh = cfg.width//cfg.heads
        for T, ids, Bb, max_m, Pd, has_bank in rbuckets:
            key = f'r{T}:'
            zb = zpad.index_select(0, di[key+'z']).view(Bb, T, lat, lw)
            if has_bank:
                bank = vt.new_zeros((Bb*max_m, lw)).index_copy(0, di[key+'dst'], vt.index_select(0, di[key+'src']))
                bank = bank.view(Bb, max_m, lw)
            else:
                bank = (vt if vt is not None else z).new_zeros((Bb, max_m, lw))
            if Pd:
                G = U.index_select(0, di[key+'kv']).view(Bb, Pd, *store.kv_shape)
                prefixes = tuple(tuple(G[:, :, l, j].permute(0, 2, 1, 3, 4).reshape(Bb, cfg.heads, Pd*lat, dh)
                                       for j in (0, 1)) for l in range(cfg.layers))
            else:
                prefixes = tuple(tuple(torch.zeros((Bb, cfg.heads, 0, dh), device=device, dtype=store.kv_dtype)
                                       for _ in (0, 1)) for _ in range(cfg.layers))
            kp = di[key+'kp'].view(Bb, Pd*lat)
            ends = di[key+'ends'].view(Bb, max_m)
            position = di[key+'steps'][:, None]+torch.arange(T, device=device)
            if READER == 'flex' and T > 1 and device.type == 'cuda' and cfg.width//cfg.heads >= 16:
                # (FlexAttention needs a head width >= 16; smaller test models read with SDPA)
                hidden = read_flex(memory, zb, position, prefixes, bank, ends, kp)
            else:
                hidden = memory.read(zb, position, prefixes, bank, ends, kp)[0]
            outs.append(hidden.reshape(Bb*T, -1).index_select(0, di[key+'real']))
        H = (torch.cat(outs) if len(outs) > 1 else outs[0]).index_select(0, di['out:inv'])
        return dict(hidden=H[None], states=[], written_count=n_new)
    ctx = dict(memory_call=call, length=n_main, version=int(version), reset_done=True, require_next_state=False,
               encoded=encoded)
    outputs = model(batch, need_vg=True, temporal=ctx)
    return outputs, dict(o_seq=ctx['o_seq'], z_out=ctx['z_out'], recompute_stats=stats, replay=replay)
