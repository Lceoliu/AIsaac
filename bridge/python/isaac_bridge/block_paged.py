"""A1: GPU-resident slot execution + paged snapshots for block memory.

Production replacement of the object path (EpisodeMemory / RawHistory / Snapshot):

* SlotBlockActor keeps every live slot in a BlockSlotState (dense GPU rows). A step
  is one batched gather -> Reader (CUDA graph when enabled) -> commit; no
  EpisodeMemory, MemoryBlock tuple, RawHistory or LZ4 packing per decision.
* Every decision's detached K/V delta and latent go once into PagedStore pages
  (host memory, fixed-size pages, never reallocated); every complete block's value
  goes once into block pages; every raw model input (visible frame + in-flight
  action) goes once into the source table keyed by a never-reused decision id.
* A snapshot is a PagedSnapshot = (EpisodeLog, step, pending, version): O(1) to
  take, fork or hand to the learner (no pickling). The log is the slot lineage's
  append-only list of decision ids/page slots and block ids/ends; restoring a
  snapshot (teacher branch) materialises one arena row from pages.
* Leases: PagedStore.collect() marks what live snapshots (weakly tracked) and live
  slots can still read -- window K/V, open-tail latents, the last memory_blocks
  blocks, and the raw sources of those blocks/tails -- and frees the rest. Nothing
  referenced by a rollout, a teacher buffer, a search point or a lane is released.
* The learner (forward_chunks_paged) builds dense batch tensors from pages: unique
  page rows are gathered once, moved once, then indexed on the device; historical
  Writer recomputation selects blocks with the same generator calls as
  historical_blocks, and re-encodes their raw sources from the source table.

Single-process only (learner thread in the actor process): pages are process-local.
"""
import os
import threading
import weakref

import numpy as np
import torch

from .block_slots import BlockSlotState
from .structured_input import from_rows


# ----------------------------------------------------------------------------- timing (ISAAC_BLOCK_TIMING=1)

TIMING_ON = os.environ.get('ISAAC_BLOCK_TIMING', '0') == '1'
TIMING = {}


class _Clock:
    """Phase timer; with ISAAC_BLOCK_TIMING=1 it synchronises the GPU at each mark (measurement only)."""
    def __init__(self, prefix, device):
        self.prefix, self.cuda = prefix, torch.device(device).type == 'cuda'
        self.t = self._now()

    def _now(self):
        import time
        if TIMING_ON and self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def mark(self, name):
        if not TIMING_ON:
            return
        t = self._now()
        key = self.prefix+name
        TIMING[key] = TIMING.get(key, 0.0)+t-self.t
        self.t = t


def count(name, value):
    if TIMING_ON:
        TIMING[name] = TIMING.get(name, 0)+value


DUMP_DIR = os.environ.get('ISAAC_BLOCK_DUMP', '')
_DUMPED = set()


def dump_inputs(name, rows, pending):
    """Measurement only (ISAAC_BLOCK_DUMP=dir): save one real input batch (ROW records + actions
    under way) once per name, for the standalone kernel-count profile (bridge/ops/b42_kernels.py)."""
    if not DUMP_DIR or name in _DUMPED:
        return
    _DUMPED.add(name)
    os.makedirs(DUMP_DIR, exist_ok=True)
    np.save(os.path.join(DUMP_DIR, name+'_rows.npy'), np.asarray(rows))
    np.save(os.path.join(DUMP_DIR, name+'_pending.npy'), np.asarray(pending, np.int64))


# B42 spike localisation: per-update counters (ISAAC_BLOCK_TIMING=1), drained into progress.csv by
# the learner after each update (update_report); both threads add to them.
UPD = {}
UPD_KEYS = ('learn_fwd_part_wall_s', 'learn_fwd_part_gpu_s', 'learn_bwd_wall_s', 'learn_bwd_gpu_s', 'learn_bwd_calls',
            'learn_zero_wall_s', 'learn_clip_wall_s', 'learn_clip_gpu_s', 'learn_opt_wall_s', 'learn_opt_gpu_s',
            'learn_admission_wait_s', 'learn_yield_wait_s', 'learn_yields_given',
            'actor_admission_wait_s', 'actor_admission_misses',
            'actor_captures_step', 'actor_capture_step_s', 'actor_captures_other', 'actor_capture_other_s',
            'actor_graph_calls', 'actor_graph_s', 'actor_graph_overlap_s',
            'actor_eager_calls', 'actor_eager_merged_calls', 'actor_eager_s', 'actor_eager_overlap_s',
            'learn_main_decode_calls', 'learn_main_decode_rows_fresh', 'learn_main_decode_rows_cached',
            'learn_hist_decode_rows_fresh', 'learn_hist_decode_rows_cached', 'learn_tail_decode_rows',
            'learn_nvcsw', 'learn_nivcsw', 'learn_cpu_s', 'gc_collect_s', 'gc_collect_calls', 'update_t0', 'update_t1',
            'gpu_reserved_now_gb',
            # B44 packed learner: calls per update, Spatial tokens (real / padded layout), Reader rows
            'learn_paged_calls', 'learn_packed_calls', 'learn_reader_calls', 'learn_reader_rows_real',
            'learn_reader_rows_padded', 'learn_spatial_calls', 'learn_spatial_tokens_real',
            'learn_spatial_tokens_padded', 'learn_writer_calls', 'learn_packed_host_s', 'learn_staged_mb',
            'gpu_peak_allocated_gb')
LEARNER_ACTIVE = [False]


def _time_gc():
    """Measurement only (ISAAC_BLOCK_TIMING=1): time every gc.collect() (torch.cuda.graph's __enter__
    runs one before each capture, in a process holding the page store's Python objects)."""
    import gc
    import time
    collect = gc.collect

    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return collect(*a, **k)
        finally:
            ucount('gc_collect_s', time.perf_counter()-t); ucount('gc_collect_calls', 1)
    gc.collect = timed


if TIMING_ON:
    _time_gc()


def ucount(name, value):
    if TIMING_ON:
        UPD[name] = UPD.get(name, 0)+value


class GpuSpan:
    """Wall time and CUDA-event time of a region (events read at drain(): no synchronisation)."""
    PENDING = []

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if TIMING_ON:
            import time
            self.t = time.perf_counter()
            if torch.cuda.is_available():
                self.e0 = torch.cuda.Event(enable_timing=True); self.e0.record()
        return self

    def __exit__(self, *exc):
        if TIMING_ON:
            import time
            ucount(self.name+'_wall_s', time.perf_counter()-self.t)
            if torch.cuda.is_available():
                e1 = torch.cuda.Event(enable_timing=True); e1.record()
                GpuSpan.PENDING.append((self.name+'_gpu_s', self.e0, e1))
        return False

    @staticmethod
    def drain():
        for name, a, b in GpuSpan.PENDING:
            b.synchronize()
            ucount(name, a.elapsed_time(b)/1e3)
        GpuSpan.PENDING.clear()


# B42 fix 3: host-decoded rollout chunks of one update, keyed by the rows' memory (pointer, shape,
# strides) and the actions under way. The learner clears it at the start and end of each update
# (the rollout buffers are reused by later rollouts).
DECODE_CACHE = {}


def rows_key(rows, pending, packed):
    def k(a):
        a = np.asarray(a)
        return (a.__array_interface__['data'][0], a.shape, a.strides, a.dtype.str)
    return k(rows)+k(pending)+(bool(packed),)


def clear_decode_cache():
    DECODE_CACHE.clear()


# B43 history-ablation diagnostics, every update in block mode (progress.csv hist_* columns).
HIST = dict(age=[], g_max=[], g_rand=[], rows=0, blocks=0, aux=0.0, spans=[])


class HistSpan:
    """Seconds of the historical recompute (CUDA events when on the GPU: no synchronisation)."""
    def __init__(self, device):
        self.cuda = torch.device(device).type == 'cuda'

    def __enter__(self):
        import time
        self.t = time.perf_counter()
        if self.cuda:
            self.e0 = torch.cuda.Event(enable_timing=True); self.e0.record()
        return self

    def __exit__(self, *exc):
        import time
        if self.cuda:
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            HIST['spans'].append((self.e0, e1))
        else:
            HIST['spans'].append(time.perf_counter()-self.t)
        return False


def hist_report():
    """This update's history diagnostics (fixed keys; drained)."""
    secs = 0.0
    for s in HIST['spans']:
        if isinstance(s, tuple):
            s[1].synchronize(); secs += s[0].elapsed_time(s[1])/1e3
        else:
            secs += s

    def q(v, f):
        return float(f(v)) if len(v) else 0.0
    out = dict(hist_rows=HIST['rows'], hist_blocks=HIST['blocks'], hist_s=secs, hist_aux_loss=HIST['aux'],
               hist_age_median=q(HIST['age'], np.median), hist_age_max=q(HIST['age'], np.max),
               hist_g_max_max=q(HIST['g_max'], np.max), hist_g_max_median=q(HIST['g_max'], np.median),
               hist_g_rand_median=q(HIST['g_rand'], np.median))
    HIST.update(age=[], g_max=[], g_rand=[], rows=0, blocks=0, aux=0.0, spans=[])
    return out


def grad_replay_aux(model, replays, coef=1.0, autocast=None):
    """B43 C arm, after the PPO backward of the batches whose forward_chunks_paged contexts gave
    `replays` (their 'replay' dicts): per chunk choose 1 window-external block of largest
    ||g_j|| (g_j = dL/dM_j, the gradient the Reader sent to its cached value) and
    history_recompute_blocks-1 others at random (torch's RNG); re-encode the chosen blocks' raw
    observations with the current Spatial + Writer (f_j) and backpropagate
    coef * sum_j sg(g_j) . f_j (g summed over the batches that read the block): the first-order
    gradient the Writer/Spatial would have received had the Reader read f_j, without a second
    Reader pass. autocast: a context factory for the re-encoding forward.
    Returns diagnostics: aux value, rows, ||g|| of the max and random picks, picks' ages (blocks)."""
    import contextlib
    stats = dict(aux=0.0, rows=0, blocks=0, g_max=[], g_rand=[], age=[])
    if not replays:
        return stats
    model_cfg = model.block_memory.config
    k = int(model_cfg.history_recompute_blocks)
    if k <= 0:
        return stats
    chosen = {}                                                    # bid -> summed g
    store = device = micro = None
    for rp in replays:
        g = rp['pool'].grad
        if g is None:
            continue
        store, device, micro = rp['store'], g.device, rp['spatial_micro']
        norms = g.detach().float().flatten(1).norm(dim=1).cpu().numpy()
        picks = set()
        for uidx, ages in rp['chunks']:
            if not len(uidx):
                continue
            first = int(np.argmax(norms[uidx]))
            pick = [first]
            rest = np.delete(np.arange(len(uidx)), first)
            if k > 1 and len(rest):
                pick += rest[torch.randperm(len(rest))[:k-1].numpy()].tolist()
            stats['g_max'].append(float(norms[uidx[first]]))
            stats['g_rand'] += [float(norms[uidx[j]]) for j in pick[1:]]
            stats['age'] += [int(ages[j]) for j in pick]
            picks.update(int(uidx[j]) for j in pick)
        for j in sorted(picks):
            bid = int(rp['ubid'][j])
            chosen[bid] = chosen[bid]+g[j] if bid in chosen else g[j].clone()
    if not chosen:
        return stats
    bids = np.fromiter(chosen.keys(), np.int64, len(chosen))
    G = torch.stack([chosen[int(b)] for b in bids]).detach().float()
    with HistSpan(device):
        with (autocast() if autocast is not None else contextlib.nullcontext()):
            f, rows = encode_blocks(model, store, bids, device, micro)
        aux = coef*(G*f.float()).sum()
        aux.backward()
    stats.update(aux=float(aux.detach()), rows=rows, blocks=len(bids))
    HIST['age'] += stats['age']; HIST['g_max'] += stats['g_max']; HIST['g_rand'] += stats['g_rand']
    HIST['rows'] += rows; HIST['blocks'] += len(bids); HIST['aux'] += stats['aux']
    return stats


def update_report():
    """The per-update counters as progress.csv columns (all present on every row)."""
    GpuSpan.drain()
    out = {'b42_'+k: UPD.get(k, 0) for k in UPD_KEYS}
    UPD.clear()
    return out


def timing_report(reset=True):
    out = dict(TIMING)
    if reset:
        TIMING.clear()
    return out


# ----------------------------------------------------------------------------- logs / snapshots

class EpisodeLog:
    """Append-only lineage of one slot: decision ids, their K/V page slots, blocks."""
    __slots__ = ('episode', 'offset', 'dec', 'kvs', 'n', 'blk', 'blk_end', 'nb', '__weakref__')

    def __init__(self, episode=None, offset=0, dec=None, kvs=None, blk=None, blk_end=None):
        self.episode, self.offset = episode, int(offset)
        dec = np.empty(0, np.int64) if dec is None else np.asarray(dec, np.int64)
        kvs = np.empty(0, np.int64) if kvs is None else np.asarray(kvs, np.int64)
        blk = np.empty(0, np.int64) if blk is None else np.asarray(blk, np.int64)
        blk_end = np.empty(0, np.int64) if blk_end is None else np.asarray(blk_end, np.int64)
        self.n, self.nb = len(dec), len(blk)
        self.dec = np.empty(max(64, 2*self.n), np.int64); self.dec[:self.n] = dec
        self.kvs = np.empty(len(self.dec), np.int64); self.kvs[:self.n] = kvs
        self.blk = np.empty(max(16, 2*self.nb), np.int64); self.blk[:self.nb] = blk
        self.blk_end = np.empty(len(self.blk), np.int64); self.blk_end[:self.nb] = blk_end

    @property
    def end(self):
        return self.offset + self.n

    def push(self, dec, kvs):
        k = len(dec)
        if self.n + k > len(self.dec):
            size = max(2*len(self.dec), self.n + k)
            for name in ('dec', 'kvs'):
                grown = np.empty(size, np.int64); grown[:self.n] = getattr(self, name)[:self.n]
                setattr(self, name, grown)
        self.dec[self.n:self.n+k] = dec; self.kvs[self.n:self.n+k] = kvs; self.n += k

    def push_block(self, bid, end):
        if self.nb == len(self.blk):
            for name in ('blk', 'blk_end'):
                grown = np.empty(2*len(self.blk), np.int64); grown[:self.nb] = getattr(self, name)[:self.nb]
                setattr(self, name, grown)
        self.blk[self.nb] = bid; self.blk_end[self.nb] = end; self.nb += 1

    def decisions(self, lo, hi):
        if lo < self.offset or hi > self.end:
            raise ValueError('snapshot reads outside its retained log prefix')
        return self.dec[lo-self.offset:hi-self.offset], self.kvs[lo-self.offset:hi-self.offset]

    def blocks_before(self, step, limit):
        """Complete blocks visible to a snapshot at `step` (end < step), the newest `limit`."""
        at = int(np.searchsorted(self.blk_end[:self.nb], step, 'left'))
        lo = max(0, at-limit)
        return self.blk[lo:at], self.blk_end[lo:at]

    def branch(self, step, config):
        """Self-contained child log for a branch restored at `step` (copies only what can be read)."""
        c = config
        lo = max(self.offset, step-max(c.window, c.block_steps))
        dec, kvs = self.decisions(lo, step)
        blk, end = self.blocks_before(step, c.memory_blocks)
        return EpisodeLog(self.episode, lo, dec.copy(), kvs.copy(), blk.copy(), end.copy())


class PagedSnapshot:
    """Before-observation memory reference: (log, step) plus the in-flight action."""
    __slots__ = ('store', 'log', 'step', 'pending', 'policy_version', '__weakref__')

    def __init__(self, store, log, step, pending, policy_version):
        self.store, self.log, self.step = store, log, int(step)
        self.pending, self.policy_version = pending, int(policy_version)
        store.track(self)

    def fork(self):
        return PagedSnapshot(self.store, self.log, self.step, self.pending.copy(), self.policy_version)

    def to(self, device):
        return self

    @property
    def episode(self):
        return self.log.episode


class PagedPayload(list):
    """In-process snapshot hand-off (rollout / teacher records): no serialisation."""


# ----------------------------------------------------------------------------- store

# B44: decoded historical block sources kept across rollouts (~0.5 MB each compact): at most this many.
DECODED_CAP = int(os.environ.get('ISAAC_BLOCK_DECODED_CAP', '6000'))


class PagedStore:
    def __init__(self, config, nh, kv_dtype=torch.float32, page=4096):
        c = self.config = config
        self.nh, self.page, self.kv_dtype = int(nh), int(page), kv_dtype
        self.kv_shape = (c.layers, 2, c.heads, c.latents, c.width//c.heads)
        self.z_shape = (c.latents, c.latent_width)
        self.b_shape = (c.memory_tokens, c.latent_width)
        self.kv_pages, self.z_pages, self.b_pages = [], [], []
        self.d_meta = np.zeros((0, 2), np.int64)        # per K/V slot: step, version
        self.d_used = np.zeros(0, bool)
        self.d_free = np.empty(0, np.int64)
        self.b_meta = np.zeros((0, 2), np.int64)        # per block: end, writer version
        self.b_src = np.zeros((0, c.block_steps), np.int64)
        self.b_srcver = np.zeros((0, c.block_steps), np.int64)
        self.b_used = np.zeros(0, bool)
        self.b_free = np.empty(0, np.int64)
        self.raw = {}                                   # decision id -> (first, schema, frame bytes, pending)
        self.decoded = {}                               # learner cache: (block id, packed) -> host frames
        self.next_decision = 0
        self.snaps = weakref.WeakSet()
        self.roots = []                                 # weak methods -> iterable of (log, step) still live
        self.lock = threading.Lock()
        self.collections = 0
        self.freed = dict(kv=0, blocks=0, raw=0)

    def track(self, snap):
        self.snaps.add(snap)

    # ---- allocation
    def _grow_d(self):
        P = self.page
        self.kv_pages.append(torch.zeros((P, *self.kv_shape), dtype=self.kv_dtype))
        self.z_pages.append(torch.zeros((P, *self.z_shape), dtype=torch.float32))
        base = len(self.d_used)
        self.d_meta = np.concatenate((self.d_meta, np.full((P, 2), -1, np.int64)))
        self.d_used = np.concatenate((self.d_used, np.zeros(P, bool)))
        self.d_free = np.concatenate((np.arange(base+P-1, base-1, -1, dtype=np.int64), self.d_free))

    def _grow_b(self):
        P = self.page
        self.b_pages.append(torch.zeros((P, *self.b_shape), dtype=torch.float32))
        base = len(self.b_used)
        bs = self.config.block_steps
        self.b_meta = np.concatenate((self.b_meta, np.full((P, 2), -1, np.int64)))
        self.b_src = np.concatenate((self.b_src, np.full((P, bs), -1, np.int64)))
        self.b_srcver = np.concatenate((self.b_srcver, np.full((P, bs), -1, np.int64)))
        self.b_used = np.concatenate((self.b_used, np.zeros(P, bool)))
        self.b_free = np.concatenate((np.arange(base+P-1, base-1, -1, dtype=np.int64), self.b_free))

    def _alloc(self, n, kind):
        free = 'd_free' if kind == 'd' else 'b_free'
        while len(getattr(self, free)) < n:
            (self._grow_d if kind == 'd' else self._grow_b)()
        pool = getattr(self, free)
        ids = pool[len(pool)-n:][::-1].copy()
        setattr(self, free, pool[:len(pool)-n])
        (self.d_used if kind == 'd' else self.b_used)[ids] = True
        return ids

    def _scatter(self, pages, ids, values):
        P = self.page
        pg, off = ids//P, ids % P
        for p in np.unique(pg):
            m = np.flatnonzero(pg == p)
            pages[p][torch.from_numpy(off[m])] = values[torch.from_numpy(m)].to(pages[p].dtype)

    def _gather(self, pages, ids, shape, dtype, out=None):
        ids = np.asarray(ids, np.int64)
        out = torch.empty((len(ids), *shape), dtype=dtype) if out is None else out
        if not len(ids):
            return out
        P = self.page
        pg, off = ids//P, ids % P
        for p in np.unique(pg):
            m = np.flatnonzero(pg == p)
            out[torch.from_numpy(m)] = pages[p].index_select(0, torch.from_numpy(off[m]))
        return out

    # ---- writes (actor thread)
    def write_decisions(self, rows, pending, kv, z, steps, version):
        """kv [n,L,2,H,lat,dh] cpu, z [n,lat,lw] cpu. Returns (decision ids, K/V slots)."""
        n = len(steps)
        with self.lock:
            kvs = self._alloc(n, 'd')
            dec = np.arange(self.next_decision, self.next_decision+n, dtype=np.int64)
            self.next_decision += n
        self._scatter(self.kv_pages, kvs, kv)
        self._scatter(self.z_pages, kvs, z)
        self.d_meta[kvs, 0] = steps; self.d_meta[kvs, 1] = int(version)
        first, schema, nbytes = rows['first'], rows['visible_schema'], rows['visible_bytes']
        frames = rows['visible_frame']
        for i in range(n):
            self.raw[int(dec[i])] = (int(first[i]), int(schema[i]), frames[i, :int(nbytes[i])].tobytes(),
                                     np.array(pending[i], np.int64))
        return dec, kvs

    def write_blocks(self, values, ends, version, src, srcver):
        with self.lock:
            bid = self._alloc(len(ends), 'b')
        self._scatter(self.b_pages, bid, values)
        self.b_meta[bid, 0] = ends; self.b_meta[bid, 1] = int(version)
        self.b_src[bid] = src; self.b_srcver[bid] = srcver
        return bid

    # ---- reads
    def kv(self, kvs, out=None):
        return self._gather(self.kv_pages, kvs, self.kv_shape, self.kv_dtype, out)

    def z(self, kvs, out=None):
        return self._gather(self.z_pages, kvs, self.z_shape, torch.float32, out)

    def blocks(self, bid, out=None):
        return self._gather(self.b_pages, bid, self.b_shape, torch.float32, out)

    def source_rows(self, decisions):
        """(rows, pending) for raw decision ids, assembled as one input-only source batch."""
        from .block_training import source_dtype
        entries = [self.raw[int(d)] for d in decisions]
        width = max([4]+[len(e[2]) for e in entries])
        rows = np.zeros(len(entries), dtype=source_dtype(width, True))
        pending = np.zeros((len(entries), self.nh), np.int64)
        for i, (first, schema, frame, pend) in enumerate(entries):
            rows['first'][i] = first; rows['visible_schema'][i] = schema; rows['visible_bytes'][i] = len(frame)
            rows['visible_frame'][i, :len(frame)] = np.frombuffer(frame, np.uint8)
            pending[i] = pend
        return rows, pending

    # ---- leases
    def live_pairs(self):
        for _ in range(8):
            try:
                pairs = [(s.log, s.step) for s in list(self.snaps)]
                break
            except RuntimeError:   # concurrently mutated weak set: retry
                continue
        else:
            raise RuntimeError('could not enumerate live snapshots')
        alive = []
        for root in self.roots:
            method = root()
            if method is not None:
                pairs.extend(method()); alive.append(root)
        self.roots[:] = alive
        return pairs

    def collect(self):
        """Free every page row / source that no live snapshot or slot can read."""
        c = self.config
        W = max(c.window, c.block_steps)
        groups = {}
        for log, step in self.live_pairs():
            groups.setdefault(id(log), (log, []))[1].append(step)
        live_kv, live_raw, live_blk = [], [], []
        for log, steps in groups.values():
            steps = np.unique(np.asarray(steps, np.int64))
            if log.n:
                hi = steps-log.offset
                lo = np.maximum(steps-W, log.offset)-log.offset
                diff = np.zeros(log.n+1, np.int64)
                np.add.at(diff, np.clip(lo, 0, log.n), 1); np.add.at(diff, np.clip(hi, 0, log.n), -1)
                mask = np.cumsum(diff)[:-1] > 0
                live_kv.append(log.kvs[:log.n][mask]); live_raw.append(log.dec[:log.n][mask])
            if log.nb:
                at = np.searchsorted(log.blk_end[:log.nb], steps, 'left')
                diff = np.zeros(log.nb+1, np.int64)
                np.add.at(diff, np.maximum(at-c.memory_blocks, 0), 1); np.add.at(diff, at, -1)
                mask = np.cumsum(diff)[:-1] > 0
                live_blk.append(log.blk[:log.nb][mask])
        live_kv = np.unique(np.concatenate(live_kv)) if live_kv else np.empty(0, np.int64)
        live_blk = np.unique(np.concatenate(live_blk)) if live_blk else np.empty(0, np.int64)
        raw = [self.b_src[live_blk].reshape(-1)] + live_raw
        live_raw = np.unique(np.concatenate(raw)) if raw else np.empty(0, np.int64)
        with self.lock:
            keep = np.zeros(len(self.d_used), bool); keep[live_kv] = True
            dead = np.flatnonzero(self.d_used & ~keep)
            self.d_used[dead] = False; self.d_meta[dead] = -1
            self.d_free = np.concatenate((dead[::-1], self.d_free))
            keep = np.zeros(len(self.b_used), bool); keep[live_blk] = True
            dead_b = np.flatnonzero(self.b_used & ~keep)
            self.b_used[dead_b] = False; self.b_meta[dead_b] = -1
            self.b_free = np.concatenate((dead_b[::-1], self.b_free))
        keys = np.fromiter(self.raw.keys(), np.int64, len(self.raw))
        dead_r = keys[~np.isin(keys, live_raw)]
        for k in dead_r.tolist():
            del self.raw[k]
        # (B44) decoded block sources stay cached while their block lives (a freed id is reused)
        dead_set = set(dead_b.tolist())
        self.decoded = {k: v for k, v in list(self.decoded.items()) if k[0] not in dead_set}
        self.collections += 1
        self.freed['kv'] += len(dead); self.freed['blocks'] += len(dead_b); self.freed['raw'] += len(dead_r)
        return dict(kv=int(self.d_used.sum()), blocks=int(self.b_used.sum()), raw=len(self.raw),
                    freed_kv=len(dead), freed_blocks=len(dead_b), freed_raw=len(dead_r))

    def cache_decoded(self, key, value):
        """Learner cache of decoded block sources (key (block id, tag)); oldest entries dropped past
        DECODED_CAP blocks (host memory bound)."""
        self.decoded[key] = value
        while len(self.decoded) > DECODED_CAP:
            del self.decoded[next(iter(self.decoded))]

    def stats(self):
        return dict(kv_rows=int(self.d_used.sum()), kv_capacity=len(self.d_used), blocks=int(self.b_used.sum()),
                    block_capacity=len(self.b_used), raw=len(self.raw),
                    raw_mb=sum(len(v[2]) for v in list(self.raw.values()))/2**20)

    # ---- materialisation of one snapshot (restore) / many (learner)
    def snapshot_parts(self, snap):
        c = self.config
        s, log = snap.step, snap.log
        lo = max(log.offset, s-c.window)
        dec, kvs = log.decisions(lo, s)
        p = s % c.block_steps
        pdec, pkvs = log.decisions(s-p, s)
        bid, ends = log.blocks_before(s, c.memory_blocks)
        return dict(step=s, kv_steps=np.arange(lo, s, dtype=np.int64), kvs=kvs, pdec=pdec, pkvs=pkvs,
                    bid=bid, ends=ends)


# ----------------------------------------------------------------------------- actor

# B42 fix 2: start-up pre-capture of the actor graphs (ISAAC_BLOCK_PRECAPTURE=0: off).
PRECAPTURE = os.environ.get('ISAAC_BLOCK_PRECAPTURE', '1') == '1'
PRECAPTURE_CAPS = (1, 2, 4, 8, 16)            # rows of a main-only / lane-only call (16 workers)
PRECAPTURE_MERGED_CAPS = (2, 4, 8, 16, 32)    # rows of a merged main + teacher-lane call
PRECAPTURE_ENTITIES = (16, 32, 64, 128)       # structured_input.BUCKETS['entity']
PRECAPTURED = [False]


def late_capture(kind, key):
    """After the start-up pre-capture, every new graph key is printed (ISAAC_BLOCK_TIMING=1)."""
    if TIMING_ON and PRECAPTURED[0]:
        print('B42 late capture', kind, key, flush=True)


def synthetic_rows(n, entities, body=1, inventory=0, slots=0):
    """n ROW records holding one all-zero complete visible-v2 frame with these counts."""
    from .tok_obs import ROW
    from .structured_input import pack_frame, write_frame, PF, EF, NF, SF, RC
    payload = pack_frame(np.zeros((body, PF), np.float32), np.zeros((RC, 16, 28), np.float32),
                         np.zeros((body, RC, 9, 9), np.float32), np.zeros((8, 13, 13), np.float32),
                         np.zeros((entities, EF), np.float32), np.zeros((inventory, NF), np.float32),
                         np.zeros((slots, SF), np.float32))
    rows = np.zeros(n, ROW)
    for i in range(n):
        write_frame(rows[i:i+1], payload)
    return rows


@torch.inference_mode()
def precapture_actor(actor, merged=True):
    """B42 fix 2: capture an actor's common graph keys before they are first needed: its whole-step
    graph for (rows {1..16}, entity bucket), and with merged=True (the main actor) the Spatial
    encoder and Reader graphs a merged main + teacher-lane call replays (rows {2..32}). One body,
    fixed inventory/slot buckets (the prof-f2/f3 actor batches); other keys stay lazy and are
    printed as late captures. Runs in the actor's autocast context. Returns (graphs, seconds)."""
    import time as time_
    t = time_.perf_counter()
    made = 0
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=actor.bf16):
        if actor.step_graph is not None:
            for cap in PRECAPTURE_CAPS:
                for e in PRECAPTURE_ENTITIES:
                    made += int(actor.step_graph.precapture(cap, e))
        if merged and actor.encoder is not None:
            memory = actor.model.block_memory
            scratch = actor.state.slots-1
            nh = len(actor.model.heads)
            for cap in PRECAPTURE_MERGED_CAPS:
                for e in PRECAPTURE_ENTITIES:
                    before = actor.encoder.captures
                    _, z = actor.encoder(synthetic_rows(cap, e), np.zeros((cap, nh), np.int64))
                    made += actor.encoder.captures-before
                if actor.reader is not None:
                    ids = torch.full((cap,), scratch, dtype=torch.long, device=z.device)
                    steps, prefixes, bank, ends, kp = actor.state.read_inputs(memory, ids, z.dtype)
                    before = actor.reader.captures
                    actor.reader(z[:, None], steps[:, None], prefixes, bank, ends, kp)
                    made += actor.reader.captures-before
    torch.cuda.synchronize()
    return made, time_.perf_counter()-t


# B42 glue 5: a whole single-arena actor step as one CUDA graph (ISAAC_BLOCK_STEP_GRAPH=0: off).
STEP_GRAPH = os.environ.get('ISAAC_BLOCK_STEP_GRAPH', '1') == '1'
_STEP_KEYS = ('v_room', 'v_map', 'v_body', 'v_body_mask', 'v_local', 'v_local_mask', 'v_entity', 'v_entity_mask',
              'v_inventory', 'v_inventory_mask', 'v_slots', 'v_slots_mask', 'pending')


class ActorStepGraph:
    """One CUDA graph per (row capacity, ragged buckets, autocast, greedy) replaying a whole
    actor step of one arena: bitplane canvas expansion, compiled Spatial + resampler, slot
    gather, Reader, the step's K/V/latent/step-counter writes (commit_device), heads and
    Gumbel-max sampling with the replay mask. The inputs (host-decoded frames, actions under
    way, arena rows, uniforms, replay mask, policy version) go in one pinned slab -> one H2D
    copy into the graph's static slab; the outputs come back in three D2H copies and one
    synchronisation. Host-decided work stays outside: episode resets, the Writer on completed
    blocks, pages. Padded rows address the arena's scratch row (never a live slot). LRU of
    128 graphs."""
    def __init__(self, actor, limit=128):
        from collections import OrderedDict
        self.actor, self.limit = actor, int(limit)
        self.graphs = OrderedDict()
        self.captures = self.replays = self.busy_misses = 0
        self.device = next(actor.model.parameters()).device
        self.capture = torch.cuda.Stream(device=self.device)

    def _layout(self, host, cap, heads):
        shapes = {}
        for k in _STEP_KEYS:
            v = host[k]
            shapes[k] = ((cap,)+tuple(v.shape[1:]), np.dtype(v.dtype))
        shapes['ids'] = ((cap,), np.dtype(np.int64))
        shapes['u'] = ((cap, heads), np.dtype(np.float32))
        shapes['replay'] = ((cap,), np.dtype(bool))
        shapes['version'] = ((1,), np.dtype(np.int64))
        layout, at = {}, 0
        for k, (shape, dt) in shapes.items():
            at = (at+15)//16*16
            layout[k] = (at, shape, dt)
            at += int(np.prod(shape))*dt.itemsize
        return layout, at

    @staticmethod
    def _views(flat, layout, torch_side):
        out = {}
        for k, (at, shape, dt) in layout.items():
            nbytes = int(np.prod(shape))*dt.itemsize
            if torch_side:
                tdt = torch.from_numpy(np.empty((), dt)).dtype
                out[k] = flat.narrow(0, at, nbytes).view(tdt).view(shape)
            else:
                out[k] = flat[at:at+nbytes].view(dt).reshape(shape)
        return out

    def _body(self, st, greedy):
        from .structured_input import canvas_decoder
        actor = self.actor
        model, memory, state = actor.model, actor.model.block_memory, actor.state
        with torch.inference_mode(False), torch.no_grad(), torch.autocast('cuda', enabled=False):
            room, local, minimap = canvas_decoder()(st['v_room'], st['v_local'], st['v_map'])
        b = {k: st[k] for k in _STEP_KEYS}
        b['v_room'], b['v_local'], b['v_map'] = room, local, minimap
        ids, version = st['ids'], st['version'][0]
        keep = {}

        def call(z):
            steps, prefixes, bank, ends, kp = state.read_inputs(memory, ids, z.dtype)
            hidden, tokens, current, allowed, mask = memory.read(z[:, None], steps[:, None], prefixes, bank, ends, kp)
            current = torch.stack([torch.stack(pair) for pair in current])
            state.commit_device(ids, steps, z, current, version)
            keep['current'], keep['z'] = current, z
            return dict(hidden=hidden, tokens=tokens, local_mask=allowed, memory_mask=mask, positions=steps[:, None])
        ctx = {'memory_call': call, 'reset_done': True}
        result = model.act(b, u=None if greedy else st['u'], greedy=greedy, need_vg=True, temporal=ctx)
        replay = st['replay']
        actions = torch.where(replay[:, None], b['pending'], result[0])
        logp = result[1].masked_fill(replay, 0)
        out = torch.cat((actions.float(), logp[:, None].float(), result[2][:, None].float(),
                         result[3][:, None].float()), 1)
        return out, keep['current'], keep['z']

    def _capture(self, key, layout, size, greedy):
        from .block_training import CUDA_CAPTURE_ADMISSION, capture_admission
        if not capture_admission((id(self), key)):
            self.busy_misses += 1
            return None
        import time as time_
        t_cap = time_.perf_counter()
        try:
            if len(self.graphs) >= self.limit:
                self.graphs.popitem(last=False)
            dev = self.device
            # Normal (not inference-mode) tensors, as from_host makes: the compiled decoder and
            # Spatial guard on the dispatch keys (an inference tensor recompiled them).
            with torch.inference_mode(False):
                flat = torch.zeros(size, dtype=torch.uint8, device=dev)
                st = self._views(flat, layout, True)
                st['ids'].fill_(self.actor.state.slots-1)       # the arena's scratch row
                st['u'].fill_(.5)
            host = torch.zeros(size, dtype=torch.uint8).pin_memory()
            hv = self._views(host.numpy(), layout, False)
            for _ in range(3):
                self._body(st, greedy)
            torch.cuda.synchronize(dev)
            # Autocast's weight-cast cache: casts cached by the warm-up live outside the graph's
            # pool and are freed when the caller's autocast exits while the graph would still
            # read them (NaN). Cleared here, the casts run (and are recorded) inside the capture.
            # (cache_enabled=False instead changed dynamo's global state: recompiles.)
            torch.clear_autocast_cache()
            graph = torch.cuda.CUDAGraph()
            from .block_training import graph_capture
            with graph_capture(graph, self.capture):   # B42: no full gc.collect / empty_cache per capture
                out, current, z = self._body(st, greedy)
            back = (torch.empty(out.shape, dtype=out.dtype).pin_memory(),
                    torch.empty(current.shape, dtype=current.dtype).pin_memory(),
                    torch.empty(z.shape, dtype=z.dtype).pin_memory())
            self.graphs[key] = (flat, host, hv, graph, out, current, z, back)
            self.captures += 1
            ucount('actor_captures_step', 1); ucount('actor_capture_step_s', time_.perf_counter()-t_cap)
            late_capture('step', key)
            return self.graphs[key]
        finally:
            CUDA_CAPTURE_ADMISSION.release()

    def precapture(self, n, entities, greedy=False):
        """B42 fix 2: capture the graph of an n-row step whose frames have `entities` entities
        (one body, no items/slots: inventory/slot widths are their fixed buckets) without
        replaying it, in the caller's inference/autocast context. True when captured now."""
        from .structured_input import decode_frame_host
        host = decode_frame_host(synthetic_rows(n, entities), packed=True)
        host['pending'] = np.zeros((n, len(self.actor.model.heads)), np.int64)
        cap = 1 << (n-1).bit_length()
        auto = torch.is_autocast_enabled('cuda')
        key = (cap, tuple(host[k].shape[1] for k in ('v_body', 'v_entity', 'v_inventory', 'v_slots')),
               auto, torch.get_autocast_dtype('cuda') if auto else None, bool(greedy))
        if key in self.graphs:
            return False
        layout, size = self._layout(host, cap, sum(self.actor.model.heads))
        return self._capture(key, layout, size, greedy) is not None

    def __call__(self, host, rows_ids, u, replay, version, greedy):
        """host: decode_frame_host(rows, packed=True) plus 'pending'; rows_ids: arena rows [n];
        u: uniforms [n, heads] (numpy or tensor; None when greedy); replay: bool [n].
        Returns (out [n, heads+3] numpy, current K/V [L,2,n,H,lat,dh] cpu, z [n,lat,w] cpu,
        z dtype), or None when not captured (capture admission busy / over 1024 tokens)."""
        n = len(rows_ids)
        cap = 1 << (n-1).bit_length()
        heads = sum(self.actor.model.heads)
        widths = tuple(host[k].shape[1] for k in ('v_body', 'v_entity', 'v_inventory', 'v_slots'))
        if 30+2*widths[0]+sum(widths[1:]) > 1024:
            return None
        auto = torch.is_autocast_enabled('cuda')
        adt = torch.get_autocast_dtype('cuda') if auto else None
        key = (cap, widths, auto, adt, bool(greedy))
        entry = self.graphs.get(key)
        if entry is None:
            layout, size = self._layout(host, cap, heads)
            entry = self._capture(key, layout, size, greedy)
            if entry is None:
                return None
        else:
            self.graphs.move_to_end(key)
        flat, hbuf, hv, graph, out, current, z, back = entry
        hbuf.numpy()[:] = 0
        for k in _STEP_KEYS:
            hv[k][:n] = host[k]
        hv['ids'][:] = self.actor.state.slots-1
        hv['ids'][:n] = rows_ids
        hv['u'][:] = .5
        if u is not None:
            hv['u'][:n] = u.float().cpu().numpy() if isinstance(u, torch.Tensor) else u
        hv['replay'][:n] = replay
        hv['version'][0] = int(version)
        flat.copy_(hbuf, non_blocking=True)
        graph.replay()
        self.replays += 1
        bo, bc, bz = back
        bo.copy_(out, non_blocking=True)
        bc.copy_(current, non_blocking=True)
        bz.copy_(z, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        return bo.numpy()[:n], bc[:, :, :n], bz[:n], z.dtype


class SlotBlockActor:
    """BlockActor's interface on dense GPU slots and paged snapshots."""
    def __init__(self, model, slots, graph_entities=0, bf16=False, store=None):
        if model.block_memory is None:
            raise ValueError('SlotBlockActor requires a block-memory policy')
        from .block_training import SpatialEncoderGraph, ReaderStepGraph
        self.model = model
        self.graph_entities, self.bf16 = int(graph_entities), bool(bf16)
        device = next(model.parameters()).device
        self.encoder = SpatialEncoderGraph(model, slots, graph_entities, bf16) if graph_entities else None
        self.reader = ReaderStepGraph(model, bf16) if graph_entities else None
        cfg = model.block_memory.config
        kv_dtype = torch.bfloat16 if self.bf16 else torch.float32
        self.store = PagedStore(cfg, len(model.heads), kv_dtype) if store is None else store
        graph_step = bool(graph_entities) and STEP_GRAPH and device.type == 'cuda'
        # B42 glue 5: one extra arena row (the scratch row padded graph rows address).
        self.state = BlockSlotState(cfg, slots+int(graph_step), device=device, kv_dtype=kv_dtype)
        self.step_graph = ActorStepGraph(self) if graph_step else None
        self.logs = [EpisodeLog(None) for _ in range(slots)]
        self.store.roots.append(weakref.WeakMethod(self._live))
        self.version = 0
        self.last_contexts = []
        self.last_result = None
        self.pending = [np.zeros((1, len(model.heads)), dtype=np.int64) for _ in range(slots)]

    def _live(self):
        return [(log, int(step)) for log, step in zip(self.logs, self.state.step_host)]

    def lane_actor(self, model, slots):
        """A teacher-generation actor sharing this actor's page store (merged batches allowed)."""
        lane = SlotBlockActor(model, slots, graph_entities=self.graph_entities, bf16=self.bf16, store=self.store)
        if PRECAPTURE and lane.step_graph is not None:
            # B42 fix 2: its lane-only step graphs now (a capture refused while the learner holds
            # the admission stays lazy)
            n, sec = precapture_actor(lane, merged=False)
            if TIMING_ON:
                print(f'B42 lane pre-capture: {n} graphs in {sec:.1f} s', flush=True)
        return lane

    def snapshot(self, slot):
        return PagedSnapshot(self.store, self.logs[slot], self.state.step_host[slot], self.pending[slot].copy(),
                             self.version)

    def restore(self, slot, snapshot):
        if not isinstance(snapshot, PagedSnapshot) or snapshot.store is not self.store:
            raise ValueError('slot actors restore paged snapshots of their own store')
        st, parts = self.store, self.store.snapshot_parts(snapshot)
        meta = st.d_meta
        self.state.load(slot, parts['step'], st.kv(parts['kvs']), parts['kv_steps'], meta[parts['kvs'], 1],
                        st.z(parts['pkvs']), meta[parts['pkvs'], 1], parts['pdec'],
                        st.blocks(parts['bid']), parts['ends'], st.b_meta[parts['bid'], 1],
                        st.b_src[parts['bid']], st.b_srcver[parts['bid']])
        self.logs[slot] = snapshot.log.branch(snapshot.step, self.model.block_memory.config)
        self.pending[slot] = snapshot.pending.copy()

    @torch.no_grad()
    def publish(self, weights, version):
        self.model.load_state_dict(weights)
        self.version = int(version)

    @torch.inference_mode()
    def step(self, rows, pending, slots, ukeys=None, greedy=False, evaluate=False, owners=None, uniform=None):
        from .tok_policy import crn_uniforms
        slots = list(map(int, slots))
        owners = [self]*len(slots) if owners is None else owners
        if any(owner.version != self.version for owner in owners):
            raise ValueError('a combined inference batch must use one policy version')
        if any(owner.store is not self.store for owner in owners):
            raise ValueError('a combined inference batch must share one page store')
        device = next(self.model.parameters()).device
        memory = self.model.block_memory
        for i, (owner, slot) in enumerate(zip(owners, slots)):
            if rows['first'][i]:
                owner.state.reset([slot])
                owner.logs[slot] = EpisodeLog((os.getpid(), slot, int(rows['episode'][i])))
            owner.pending[slot] = pending[i:i+1].copy()
        contexts = [owner.snapshot(slot) for owner, slot in zip(owners, slots)]
        clock = _Clock('actor.', device)
        groups = {}
        for i, owner in enumerate(owners):
            groups.setdefault(id(owner), (owner, []))[1].append(i)
        groups = list(groups.values())
        reader = self.reader
        version = self.version
        steps_host = np.array([owner.state.step_host[slot] for owner, slot in zip(owners, slots)], np.int64)
        record = {}
        import time as time_
        t_call, active = time_.perf_counter(), LEARNER_ACTIVE[0]
        if self.step_graph is not None and all(owner is self for owner in owners):
            done = self._graph_step(rows, pending, slots, ukeys, greedy, evaluate, uniform, steps_host, record, clock)
            if done is not None:
                result, host, kv_host, z_host = done
                self._pages(rows, pending, slots, owners, groups, record, host, kv_host, z_host, steps_host, contexts)
                clock.mark('pages')
                if TIMING_ON:
                    dt = time_.perf_counter()-t_call
                    ucount('actor_graph_calls', 1); ucount('actor_graph_s', dt)
                    ucount('actor_graph_overlap_s', dt if active or LEARNER_ACTIVE[0] else 0.0)
                return result
        if TIMING_ON:
            ucount('actor_eager_calls', 1); ucount('actor_eager_merged_calls', int(len(groups) > 1))

        def slot_call(z):
            clock.mark('decode_spatial')
            perm, parts, hosts = [], [], []
            for owner, positions in groups:
                host, ids = owner.state._ids([slots[i] for i in positions])
                parts.append((owner, host, ids, owner.state.read_inputs(memory, ids, z.dtype)))
                perm.extend(positions)
            if len(parts) == 1:
                steps, prefixes, bank, ends, kp = parts[0][3]
            else:
                steps = torch.cat([p[3][0] for p in parts])
                prefixes = tuple(tuple(torch.cat([p[3][1][l][j] for p in parts]) for j in (0, 1))
                                 for l in range(memory.config.layers))
                bank, ends, kp = (torch.cat([p[3][k] for p in parts]) for k in (2, 3, 4))
            identity = perm == list(range(len(perm)))
            zp = z if identity else z[torch.as_tensor(perm, device=z.device)]
            hidden, tokens, current, allowed, mask = (memory.read if reader is None else reader)(
                zp[:, None], steps[:, None], prefixes, bank, ends, kp)
            current = torch.stack([torch.stack(pair) for pair in current])
            at = 0
            for owner, host, ids, inp in parts:
                n = len(host)
                completed, written = owner.state.commit(memory, host, ids, inp[0], zp[at:at+n],
                                                        current[:, :, at:at+n], version)
                record[id(owner)] = (at, completed, written)
                at += n
            record['current'], record['z'] = current, zp
            inv = None
            if not identity:
                inv = torch.empty(len(perm), dtype=torch.long)
                inv[torch.as_tensor(perm)] = torch.arange(len(perm))
                inv = inv.to(z.device)
                hidden, tokens, allowed, mask = (x.index_select(0, inv) for x in (hidden, tokens, allowed, mask))
            record['perm'] = perm
            positions = steps[:, None] if inv is None else steps.index_select(0, inv)[:, None]
            clock.mark('reader_slot')
            count('actor.writer_rows', sum(len(record[id(o)][1]) for o, _ in groups))
            return dict(hidden=hidden, tokens=tokens, local_mask=allowed, memory_mask=mask, positions=positions)

        ctx = {'memory_call': slot_call, 'version': version, 'reset_done': True}
        clock.mark('prepare')
        count('actor.calls', 1); count('actor.rows', len(rows))
        if DUMP_DIR and len(rows) >= 16:
            if 'actor' not in _DUMPED:
                os.makedirs(DUMP_DIR, exist_ok=True)
                torch.save(dict(model=self.model.state_dict(), config=self.model.config),
                           os.path.join(DUMP_DIR, 'model.pt'))
            dump_inputs('actor', rows, pending)
        if self.encoder is None:
            b = from_rows(rows, device, pending)
        else:
            b, ctx['encoded'] = self.encoder(rows, pending)
        replay = np.broadcast_to(np.asarray(evaluate, dtype=bool), (len(rows),))
        all_replay = bool(replay.all())
        if ukeys is not None and not all_replay:
            uniform = torch.as_tensor(crn_uniforms(ukeys[0], ukeys[1], sum(self.model.heads)), device=device)
        if all_replay:
            output = self.model(b, need_vg=True, temporal=ctx)
            result = (b['pending'], torch.zeros(len(rows), device=device), output[1], output[-1])
        else:
            if replay.any() and uniform is None and not greedy:
                uniform = torch.full((len(rows), sum(self.model.heads)), .5, device=device)
                uniform[np.flatnonzero(~replay).tolist()] = torch.rand((int((~replay).sum()), sum(self.model.heads)), device=device)
            result = self.model.act(b, u=uniform, greedy=greedy, need_vg=True, temporal=ctx)
            if replay.any():
                mask = torch.from_numpy(replay.copy())
                if device.type == 'cuda':
                    mask = mask.pin_memory()
                mask = mask.to(device, non_blocking=device.type == 'cuda')
                result = (torch.where(mask[:, None], b['pending'], result[0]),
                          result[1].masked_fill(mask, 0), *result[2:])
        self.last_result = ctx.get('block_result')
        clock.mark('heads_sample')
        # One host transfer for actions plus this step's pages (same stream: ordered).
        host = torch.cat((result[0].float(), *(x[:, None].float() for x in result[1:])), 1).cpu().numpy()
        clock.mark('host_sync')
        perm = record['perm']
        cur, zc = record['current'], record['z']
        kv_host = cur.permute(2, 0, 1, 3, 4, 5).to('cpu')           # [n(perm order),L,2,H,lat,dh]
        z_host = zc.float().to('cpu')
        self._pages(rows, pending, slots, owners, groups, record, host, kv_host, z_host, steps_host, contexts)
        clock.mark('pages')
        if TIMING_ON:
            dt = time_.perf_counter()-t_call
            ucount('actor_eager_s', dt); ucount('actor_eager_overlap_s', dt if active or LEARNER_ACTIVE[0] else 0.0)
        return result


    def _graph_step(self, rows, pending, slots, ukeys, greedy, evaluate, uniform, steps_host, record, clock):
        """B42 glue 5: the step through ActorStepGraph (None: not captured; the eager path runs)."""
        from .structured_input import decode_frame_host
        from .tok_policy import crn_uniforms
        n = len(slots)
        heads = sum(self.model.heads)
        host = decode_frame_host(rows, packed=True)
        host['pending'] = np.asarray(pending, np.int64)
        replay = np.broadcast_to(np.asarray(evaluate, dtype=bool), (n,)).copy()
        u = None
        if not greedy:
            if uniform is not None:
                u = uniform
            elif ukeys is not None:
                u = np.asarray(crn_uniforms(ukeys[0], ukeys[1], heads), np.float32)
            else:
                # The eager path's main-only call samples with torch.multinomial; Gumbel-max on
                # torch uniforms is the same categorical distribution (the graph needs uniforms).
                u = torch.rand((n, heads)).numpy()
            u = np.where(replay[:, None], np.float32(.5), u.float().cpu().numpy() if isinstance(u, torch.Tensor) else u)
        clock.mark('decode_host')
        out = self.step_graph(host, np.asarray(slots, np.int64), u, replay, self.version, greedy)
        if out is None:
            return None
        out, cur, zc, zdtype = out
        clock.mark('graph')
        count('actor.graph_calls', 1)
        memory = self.model.block_memory
        bs = memory.config.block_steps
        ids = steps = None
        if np.any((steps_host+1) % bs == 0):
            ids = self.state._ids(slots)[1]
            steps = torch.from_numpy(steps_host)
            if ids.device.type == 'cuda':
                steps = steps.pin_memory()
            steps = steps.to(ids.device, non_blocking=True)
        completed, written = self.state.commit_blocks(memory, np.asarray(slots, np.int64), ids, steps, zdtype,
                                                      self.version)
        clock.mark('writer')
        count('actor.writer_rows', len(completed))
        record[id(self)] = (0, completed, written)
        dev = next(self.model.parameters()).device
        nh = len(self.model.heads)
        res = torch.as_tensor(out).to(dev)
        result = (res[:, :nh].long(), res[:, nh], res[:, nh+1], res[:, nh+2])
        self.last_result = None
        return result, out.copy(), cur.permute(2, 0, 1, 3, 4, 5), zc.float()

    def _pages(self, rows, pending, slots, owners, groups, record, host, kv_host, z_host, steps_host, contexts):
        """Pages of this step (decisions, completed blocks), lineage logs and host outputs."""
        memory = self.model.block_memory
        version = self.version
        for owner, positions in groups:
            at, completed, written = record[id(owner)]
            k = len(positions)
            sub = np.asarray(positions)
            dec, kvs = self.store.write_decisions(rows[sub], pending[sub], kv_host[at:at+k], z_host[at:at+k],
                                                  steps_host[sub], version)
            if len(completed):
                values = written.float().cpu()
                srcs, srcver, ends, logs = [], [], [], []
            for j, i in enumerate(positions):
                log = owner.logs[slots[i]]
                log.push(dec[j:j+1], kvs[j:j+1])
            for q, j in enumerate(completed.tolist()):
                i = positions[j]
                log = owner.logs[slots[i]]
                bs = memory.config.block_steps
                end = int(steps_host[i])
                d_ids, k_ids = log.decisions(end-bs+1, end+1)
                srcs.append(d_ids); srcver.append(self.store.d_meta[k_ids, 1]); ends.append(end); logs.append(log)
            if len(completed):
                bid = self.store.write_blocks(values, np.asarray(ends), version, np.stack(srcs), np.stack(srcver))
                for log, b_, e_ in zip(logs, bid, ends):
                    log.push_block(int(b_), e_)
        for owner in dict.fromkeys(owners):
            positions = [i for i, o in enumerate(owners) if o is owner]
            owner.host_output = host[positions]
            owner.last_contexts = [contexts[i] for i in positions]
        for i, (owner, slot) in enumerate(zip(owners, slots)):
            owner.pending[slot] = host[i:i+1, :len(self.model.heads)].astype(np.int64)

# ----------------------------------------------------------------------------- learner

def dense_memory_forward(memory, z, steps, prefixes, kp, old_vals, n_old, old_ends, tails, version=0, reader=None):
    """BlockMemory.forward(require_next_state=False) on dense batch tensors.

    z [B,T,lat,lw]; steps host [B]; prefixes per layer (k,v) [B,H,P,dh] zero padded with
    kp [B,P] (-1 padded); old_vals [sum n_old, tokens, lw] bank values in chronological
    order per sequence (replacements already applied); n_old host [B]; old_ends host
    [sum n_old]; tails: per-sequence open-tail latents [len_i, lat, lw]."""
    c = memory.config
    b, t = z.shape[:2]
    dev = z.device
    bs, mt = c.block_steps, c.memory_tokens
    tail_len = np.array([len(x) for x in tails], np.int64)
    complete = (tail_len+t)//bs
    pool_parts = [x.to(z.dtype) for x in tails if len(x)] + [z.flatten(0, 1)]
    pool = torch.cat(pool_parts) if len(pool_parts) > 1 else pool_parts[0]
    tail_start = np.concatenate(([0], np.cumsum(tail_len)[:-1]))
    z_base = int(tail_len.sum())
    block_index, new_rows, new_ends = [], [], []
    for i in range(b):
        if not complete[i]:
            continue
        seq = np.concatenate((tail_start[i]+np.arange(tail_len[i]), z_base+i*t+np.arange(t)))
        first_end = int(steps[i])-int(tail_len[i])+bs-1
        for j in range(int(complete[i])):
            block_index.append(seq[j*bs:(j+1)*bs]); new_rows.append(i); new_ends.append(first_end+j*bs)
    written = None
    ucount('learn_reader_calls', 1); ucount('learn_reader_rows_real', b*t); ucount('learn_reader_rows_padded', b*t)
    if block_index:
        idx = torch.from_numpy(np.stack(block_index)).to(dev)
        ucount('learn_writer_calls', 1)
        written = memory.writer(pool[idx])
    counts = n_old+complete
    max_m = max(1, int(counts.max())*mt)
    flex_on = c.attention == 'flex' and z.is_cuda and t == c.window and t > 1
    if flex_on:
        max_m = ((c.memory_blocks+(t+bs-1)//bs+1)*mt+127)//128*128
    # destinations, chronological per sequence: old blocks then new blocks
    old_row = np.repeat(np.arange(b), n_old)
    old_pos = np.concatenate([np.arange(n) for n in n_old]) if len(old_row) else np.empty(0, np.int64)
    new_row = np.asarray(new_rows, np.int64)
    new_pos = n_old[new_row]+np.concatenate([np.arange(k) for k in complete if k]) if len(new_row) else np.empty(0, np.int64)
    parts = []
    if len(old_row):
        parts.append(old_vals.to(z.dtype))
    if written is not None:
        parts.append(written.to(z.dtype))
    ends_host = np.full((b, max_m), -1, np.int64)
    rows_all = np.concatenate((old_row, new_row)).astype(np.int64)
    pos_all = np.concatenate((old_pos, new_pos)).astype(np.int64)
    ends_all = np.concatenate((np.asarray(old_ends, np.int64), np.asarray(new_ends, np.int64)))
    tok_rows = np.repeat(rows_all, mt)
    tok_pos = (pos_all[:, None]*mt+np.arange(mt)).reshape(-1)
    ends_host[tok_rows, tok_pos] = np.repeat(ends_all, mt)
    if parts:
        vals = (torch.cat(parts) if len(parts) > 1 else parts[0])+memory.memory_slot_embedding
        bank = vals.new_zeros((b, max_m, c.latent_width))
        bank = bank.index_put((torch.from_numpy(tok_rows).to(dev), torch.from_numpy(tok_pos).to(dev)),
                              vals.reshape(-1, c.latent_width))
    else:
        bank = z.new_zeros((b, max_m, c.latent_width))
    meta = torch.from_numpy(np.concatenate((np.asarray(steps, np.int64)[:, None], ends_host), 1))
    if dev.type == 'cuda':
        meta = meta.pin_memory()
    meta = meta.to(dev, non_blocking=True)
    position = meta[:, :1]+torch.arange(t, device=dev)
    ends = meta[:, 1:]
    hidden, tokens, current_kv, allowed, bm = (memory.read if reader is None else reader)(
        z, position, prefixes, bank, ends, kp)
    return dict(hidden=hidden, tokens=tokens, states=[], local_mask=allowed, memory_mask=bm,
                positions=position, current_kv=current_kv, written_count=len(block_index))


RAGGED = ('v_body', 'v_local', 'v_entity', 'v_inventory', 'v_slots')


def encode_sorted(model, batch, micro, counts=None):
    """Spatial over a decoded batch in token-sorted microbatches with tight padding.

    Rows are ordered by their real token count and each microbatch keeps only the fixed
    token bucket (structured_input.BUCKETS) of its longest row (masks are left-aligned), so
    padding attention is mostly not computed. Identical math to one padded call; with
    gradients enabled each microbatch is a non-reentrant checkpoint (bounded activations).
    counts (B42): the host real counts [n,5] (host_counts of the host-decoded dict): no
    device->host synchronisation. None: counted on the device (one sync)."""
    from torch.utils.checkpoint import checkpoint
    from .structured_input import bucket_width, packed_rows_batch
    n = len(batch['v_body'])
    dev = batch['v_body'].device
    if counts is None:
        counts = torch.stack([batch[k+'_mask'].sum(1) for k in RAGGED], 1).cpu().numpy()
    counts = np.asarray(counts, np.int64)
    order_h = np.argsort(counts.sum(1), kind='stable')
    counts_h = counts[order_h]
    order = torch.from_numpy(order_h)
    if dev.type == 'cuda':
        order = order.pin_memory()
    order = order.to(dev, non_blocking=True)
    grad = torch.is_grad_enabled() and n > micro
    out = []
    for begin in range(0, n, micro):
        if SPATIAL_PACKED:
            # B42 glue 4: row-packed microbatch, one compiled graph per row-total bucket.
            m = len(order_h[begin:begin+micro])
            part = packed_rows_batch(batch, order_h[begin:begin+micro], counts_h[begin:begin+micro], dev)
            ucount('learn_spatial_calls', 1)
            ucount('learn_spatial_tokens_real', int(counts_h[begin:begin+micro].sum())+30*m)
            ucount('learn_spatial_tokens_padded', int(part['rows_valid'].numel()))
            zp = checkpoint(model.encode_block, part, use_reentrant=False) if grad else model.encode_block(part)
            out.append(zp[:m])
            continue
        idx = order[begin:begin+micro]
        width = counts_h[begin:begin+micro].max(0)
        part = {}
        for k, v in batch.items():
            v = v.index_select(0, idx)
            base = k[:-5] if k.endswith('_mask') else k
            if base in RAGGED:
                v = v[:, :min(v.shape[1], bucket_width(base[2:], int(width[RAGGED.index(base)])))]
            part[k] = v
        out.append(checkpoint(model.encode_block, part, use_reentrant=False) if grad else model.encode_block(part))
    z = torch.cat(out)
    inverse = np.empty_like(order_h)
    inverse[order_h] = np.arange(n)
    inverse = torch.from_numpy(inverse)
    if dev.type == 'cuda':
        inverse = inverse.pin_memory()
    return z.index_select(0, inverse.to(dev, non_blocking=True))


def _to(x, device):
    if torch.device(device).type == 'cuda' and x.device.type == 'cpu':
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)


# Spatial microbatch of the paged learner. No-grad passes run up to 8192 rows at once.
# With gradients 2048 rows OOMed the shared 24 GB 4090 in checkpoint recompute (B42 prof-d,
# u5), so training stays at 512 (~1.4k ops per call: launch-bound; see B42). Same math at any size.
SPATIAL_MICRO_TRAIN, SPATIAL_MICRO_EVAL = 512, 8192
# B42 glue 4: the learner's Spatial on row-packed microbatches (ISAAC_SPATIAL_PACKED=0: off).
SPATIAL_PACKED = os.environ.get('ISAAC_SPATIAL_PACKED', '1') == '1'


HISTORY_MODES = ('random', 'detach', 'gradreplay')
_LEAF_HOOK = [None]


def encode_blocks(model, store, bids, device, spatial_micro):
    """Current Spatial + Writer on the raw sources of complete blocks `bids` (host int64): their
    block values [len(bids), memory_tokens, latent_width], differentiable. Host decodes are
    cached in the page store. Returns (values, rows encoded)."""
    from .structured_input import decode_frame_host, merge_host, from_host, host_counts
    cfg = model.block_memory.config
    packed = torch.device(device).type == 'cuda'
    hosts = []
    for bid in np.asarray(bids, np.int64).tolist():
        d = store.decoded.get((bid, packed))
        ucount('learn_hist_decode_rows_fresh' if d is None else 'learn_hist_decode_rows_cached', cfg.block_steps)
        if d is None:
            rows_b, pend_b = store.source_rows(store.b_src[bid])
            d = decode_frame_host(rows_b, packed=packed)
            d['pending'] = pend_b; d['first'] = np.asarray(rows_b['first'])
            store.cache_decoded((bid, packed), d)
        hosts.append(d)
    merged = merge_host(hosts)
    zh = encode_sorted(model, from_host(merged, device), spatial_micro, host_counts(merged))
    ucount('learn_writer_calls', 1)
    values = model.block_memory.writer(zh.reshape(len(hosts), cfg.block_steps, cfg.latents, cfg.latent_width))
    return values, len(hosts)*cfg.block_steps


def forward_chunks_paged(model, chunks, generator=None, train_history=True, version=0, spatial_micro=None,
                         require_next_state=True, deduplicate_sources=True, history_mode='random'):
    """forward_chunks for chunks whose snapshots are PagedSnapshot (learner thread).

    history_mode (B43): 'random' -- history_recompute_blocks random window-external blocks per
    chunk re-encoded and used differentiably by the Reader (B and D arms); 'detach' -- none (A arm,
    = history_recompute_blocks 0); 'gradreplay' -- none in this forward: every window-external
    block value is a requires_grad leaf so the PPO backward yields g_j = dL/dM_j, and
    grad_replay_aux (after that backward) re-encodes a few blocks with an auxiliary loss (C arm)."""
    if history_mode not in HISTORY_MODES:
        raise ValueError(f'history_mode {history_mode!r}: one of {HISTORY_MODES}')
    from .block_training import source_rows
    if not chunks:
        raise ValueError('forward_chunks requires a nonempty batch')
    if require_next_state:
        # Continuation states are only needed by the object reference path.
        require_next_state = False
    length = len(chunks[0].rows)
    if any(len(c.rows) != length for c in chunks):
        raise ValueError('bucket chunks by decision length before forward_chunks')
    device = next(model.parameters()).device
    if spatial_micro is None:
        spatial_micro = SPATIAL_MICRO_TRAIN if torch.is_grad_enabled() else SPATIAL_MICRO_EVAL
    clock = _Clock('learn.', device)
    memory = model.block_memory
    cfg = memory.config
    store = chunks[0].snapshot.store
    B = len(chunks)
    count('learn.calls', 1); count('learn.chunks', B); count('learn.rows', B*length)
    ucount('learn_paged_calls', 1)
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
    # ---- historical selection (same generator calls as historical_blocks)
    selected = []
    for p in parts:
        if not train_history or history_mode != 'random':
            selected.append(np.empty(0, np.int64)); continue
        eligible = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
        if not len(eligible) or cfg.history_recompute_blocks == 0:
            selected.append(np.empty(0, np.int64)); continue
        pick = torch.randperm(len(eligible), generator=generator)[:cfg.history_recompute_blocks].numpy()
        selected.append(eligible[pick])
        if torch.is_grad_enabled():
            HIST['age'] += ((p['step']-p['ends'][eligible[pick]])//cfg.block_steps).tolist()
    unique = {}
    for p, sel in zip(parts, selected):
        for k in sel:
            unique.setdefault(int(p['bid'][k]), len(unique))
    def encode_sources(rows, pending):
        from .structured_input import decode_frame_host, from_host, host_counts
        d = decode_frame_host(rows, packed=device.type == 'cuda')
        ucount('learn_tail_decode_rows', len(rows))
        d['pending'] = np.asarray(pending, np.int64); d['first'] = np.asarray(rows['first'])
        return encode_sorted(model, from_host(d, device), spatial_micro, host_counts(d))
    hist_written = None
    hist_rows = 0
    if unique:
        clock.mark('select')
        ub = np.fromiter(unique.keys(), np.int64, len(unique))
        if torch.is_grad_enabled():
            with HistSpan(device):
                hist_written, hist_rows = encode_blocks(model, store, ub, device, spatial_micro)
            HIST['rows'] += hist_rows; HIST['blocks'] += len(ub)
        else:
            hist_written, hist_rows = encode_blocks(model, store, ub, device, spatial_micro)
        clock.mark('hist_spatial_writer')
        count('learn.hist_rows', hist_rows)
    # ---- open tails: re-encoded raw sources when a new block becomes visible / continuation needed
    tails = [None]*B
    tail_rows = 0
    enc = [i for i, p in enumerate(parts) if len(p['pdec']) and length+len(p['pdec']) > cfg.block_steps]
    if enc:
        dec_all = np.concatenate([parts[i]['pdec'] for i in enc])
        rows_t, pend_t = store.source_rows(dec_all)
        tail_rows = len(rows_t)
        zt = encode_sources(rows_t, pend_t)
        at = 0
        for i in enc:
            n = len(parts[i]['pdec']); tails[i] = zt[at:at+n]; at += n
    clock.mark('tails')
    count('learn.tail_rows', tail_rows)
    rest = [i for i in range(B) if tails[i] is None]
    pk_all = np.concatenate([parts[i]['pkvs'] for i in rest]) if rest else np.empty(0, np.int64)
    if len(pk_all):
        zp = _to(store.z(pk_all), device)
    at = 0
    for i in rest:
        n = len(parts[i]['pkvs'])
        tails[i] = zp[at:at+n] if n else torch.zeros((0, cfg.latents, cfg.latent_width), device=device)
        at += n
    # ---- window K/V: unique page rows once, then a device gather into the padded prefix
    nk = np.array([len(p['kvs']) for p in parts], np.int64)
    Pd = int(nk.max()) if B else 0
    dh = cfg.width//cfg.heads
    if Pd:
        all_kvs = np.concatenate([p['kvs'] for p in parts])
        ukv, inverse = np.unique(all_kvs, return_inverse=True)
        U = _to(store.kv(ukv), device)
        U = torch.cat((U, U.new_zeros((1, *U.shape[1:]))))
        index = np.full((B, Pd), len(ukv), np.int64)
        kp_host = np.full((B, Pd*cfg.latents), -1, np.int64)
        at = 0
        for i, p in enumerate(parts):
            n = len(p['kvs'])
            index[i, :n] = inverse[at:at+n]; at += n
            kp_host[i, :n*cfg.latents] = np.repeat(p['kv_steps'], cfg.latents)
        G = U[torch.from_numpy(index).to(device)]                       # [B,Pd,L,2,H,lat,dh]
        prefixes = tuple(tuple(G[:, :, l, j].permute(0, 2, 1, 3, 4).reshape(B, cfg.heads, Pd*cfg.latents, dh)
                               for j in (0, 1)) for l in range(cfg.layers))
        kp = _to(torch.from_numpy(kp_host), device)
    else:
        kdt = store.kv_dtype
        prefixes = tuple(tuple(torch.zeros((B, cfg.heads, 0, dh), device=device, dtype=kdt) for _ in (0, 1))
                         for _ in range(cfg.layers))
        kp = torch.zeros((B, 0), device=device, dtype=torch.long)
    # ---- old bank values (detached pages) with differentiable historical replacements
    n_old = np.array([len(p['bid']) for p in parts], np.int64)
    old_vals = None
    replay = None
    old_ends = np.concatenate([p['ends'] for p in parts]) if B else np.empty(0, np.int64)
    if n_old.sum():
        all_bid = np.concatenate([p['bid'] for p in parts])
        ubid, binv = np.unique(all_bid, return_inverse=True)
        pool = _to(store.blocks(ubid), device)
        source = binv.copy()
        if history_mode == 'gradreplay' and train_history and torch.is_grad_enabled():
            # B43 C arm: every window-external block value of this batch is a gradient leaf (a
            # fresh device copy of the pages: nothing else is kept alive); the PPO backward
            # leaves g_j = dL/dM_j in pool.grad for grad_replay_aux.
            # Only window-external reads (the blocks the random arms may re-encode) go through the
            # leaf, so g_j has the D arm's meaning; reads of a block still inside a chunk's K/V
            # window use the detached copy.
            leaf = pool.detach()
            if _LEAF_HOOK[0] is not None:   # (tests: finite differences on the leaf values only)
                leaf = _LEAF_HOOK[0](ubid, leaf)
            leaf = leaf.detach().requires_grad_(True)
            groups, at = [], 0
            for p in parts:
                nb = len(p['bid'])
                elig = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
                groups.append((binv[at+elig], (p['step']-p['ends'][elig])//cfg.block_steps))
                source[at+elig] += len(ubid)
                at += nb
            pool = torch.cat((pool.detach(), leaf))
            replay = dict(pool=leaf, ubid=ubid, store=store, chunks=groups, spatial_micro=spatial_micro)
        if hist_written is not None:
            pool = torch.cat((pool.to(hist_written.dtype), hist_written))
            at = 0
            for p, sel in zip(parts, selected):
                for k in sel:
                    source[at+k] = len(ubid)+unique[int(p['bid'][k])]
                at += len(p['bid'])
        old_vals = pool[torch.from_numpy(source).to(device)]
    clock.mark('pages_prefix_bank')
    # Host-decoded frames are cached on each chunk: the same rollout chunk is read by
    # the start-of-update pass and every epoch (decode once, upload per use).
    from .structured_input import decode_frame_host, merge_host, from_host
    packed = device.type == 'cuda'
    hosts = []
    ucount('learn_main_decode_calls', 1)
    for c in chunks:
        h = c.__dict__.get('_host')
        if h is None or h[0] != packed:
            # B42 fix 3: rollout chunks of the start-of-update pass and of the PPO epochs are
            # different Chunk objects over the same rollout rows: decode once per update.
            key = rows_key(c.rows, c.pending, packed) if len(c.rows) > 1 else None
            d = DECODE_CACHE.get(key) if key is not None else None
            if d is None:
                ucount('learn_main_decode_rows_fresh', len(c.rows))
                d = decode_frame_host(c.rows, packed=packed)
                d['pending'] = np.asarray(c.pending, dtype=np.int64); d['first'] = np.asarray(c.rows['first'])
                if key is not None:
                    DECODE_CACHE[key] = d
            else:
                ucount('learn_main_decode_rows_cached', len(c.rows))
            h = (packed, d)
            if len(c.rows) > 1:
                object.__setattr__(c, '_host', h)
        else:
            ucount('learn_main_decode_rows_cached', len(c.rows))
        hosts.append(h[1])
    rows = chunks[0].rows if len(chunks) == 1 else None
    if DUMP_DIR and torch.is_grad_enabled() and B*length >= 512:
        dump_inputs('learner', np.concatenate([np.asarray(c.rows) for c in chunks])[:SPATIAL_MICRO_TRAIN],
                    np.concatenate([np.asarray(c.pending, np.int64) for c in chunks])[:SPATIAL_MICRO_TRAIN])
    from .structured_input import host_counts
    merged = merge_host(hosts)
    batch = from_host(merged, device)
    clock.mark('main_decode')
    if TIMING_ON:
        keys = ('v_body_mask', 'v_local_mask', 'v_entity_mask', 'v_inventory_mask', 'v_slots_mask')
        count('learn.tokens_real', int(host_counts(merged).sum())+30*B*length)
        count('learn.tokens_padded', B*length*(30+sum(batch[k].shape[1] for k in keys)))
    # Same bounded-activation, token-sorted Spatial as the historical sources
    # (identical math; required at --micro 8192 on a 24 GB GPU).
    encoded = encode_sorted(model, batch, spatial_micro, host_counts(merged))

    def call(z):
        return dense_memory_forward(memory, z.reshape(B, length, *z.shape[1:]), steps, prefixes, kp,
                                    old_vals, n_old, old_ends, tails, version)
    ctx = dict(memory_call=call, length=length, version=int(version), reset_done=True,
               require_next_state=False,
               recompute_stats=dict(selected_blocks=int(sum(map(len, selected))), unique_objects=len(unique),
                                    encoded_blocks=len(unique), historical_rows=int(hist_rows),
                                    tail_rows=int(tail_rows)))
    if encoded is not None:
        ctx['encoded'] = encoded
        clock.mark('main_spatial')
    if replay is not None:
        ctx['replay'] = replay
    outputs = model(batch, need_vg=True, temporal=ctx)
    clock.mark('reader_heads')
    return outputs, batch, ctx, selected
