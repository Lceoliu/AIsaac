"""B29 (2026-10-14): latency of the actor's network call (GraphActor, train_tok's inference) without the engine.

On synthetic ROW records (the real layout; entity counts drawn from --ent, inventories like a run's) and a checkpoint's
weights, every call is split into: host packing (records into the pinned staging buffer), the host's enqueue time (of
which the graph launch call), the wait for the GPU, the host's post-processing (result()); and on the GPU (CUDA events
on the call's stream): the H2D copy, the ops before the replay (uniforms, the GRU states' gather), the replay, the ops
after it (state write-back, D2H). Medians over --iters calls.

Sections (--parts, comma list):
  sizes   : graphs of rows_max = each of --sizes, full batches, the 32-entity graph; plus the 64-entity graph at 64
  partial : the production shape (rows_max 64) with 8 / 16 / 32 / 64 records per call
  profile : torch.profiler over replays of the 64-row graph: kernels per replay and their summed GPU time
  variants: --variants (base, bf16, compile:<mode>, bf16+compile:<mode>) at --variant-sizes, plus numerics against base
            on the same records with common random numbers (the same uniforms): max |d logp|, |d value|, |d value_global|,
            action agreement; and for bf16 the eager forward's max |d logit| against float32
  temporal: a stand-in of the B27 temporal module (random weights; --temporal-* sizes) inside the graph, base and
            compiled, at --variant-sizes

usage (bridge python dir, PYTHONPATH=.): python abplus_bench_actor.py --checkpoint <ck.pt> [--parts sizes,partial,...]
"""
import argparse
import json
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaac_bridge.tok_obs import DOOR_CAP, ENT_CAP, N_CHAR, ROW
from isaac_bridge.tok_policy import ROW_BYTES, GraphActor, TokPolicy, decode_rows, heads_of


def synth_rows(rng, n, ent_lo, ent_hi):
    """Records within the policy's input ranges, entity counts uniform in [ent_lo, ent_hi], inventories like a run's
    (0-3 held collectibles, an active item half the time, a trinket 30 %, a pill 40 %)."""
    r = np.zeros(n, ROW)
    r['player'] = rng.normal(size=r['player'].shape)
    r['player'][:, 22] = rng.integers(0, 2, n)
    r['n_ent'] = rng.integers(ent_lo, ent_hi + 1, n)
    r['n_doors'] = rng.integers(1, 5, n)
    r['ent'] = rng.normal(size=r['ent'].shape)
    r['ent_id'][..., 0] = rng.integers(0, 1024, r['ent_id'].shape[:2])
    r['ent_id'][..., 1] = rng.integers(0, 1024, r['ent_id'].shape[:2])
    r['ent_id'][..., 2] = rng.integers(0, 256, r['ent_id'].shape[:2])
    r['ent_item'][..., 0] = rng.integers(0, 700, r['ent_item'].shape[:2])
    r['ent_item'][..., 1] = rng.integers(0, 128, r['ent_item'].shape[:2])
    r['doors'] = rng.normal(size=r['doors'].shape)
    for k in ('patch', 'grid', 'map'):
        r[k] = rng.integers(0, 2, r[k].shape)
    r['bombs'] = rng.integers(0, 3, n)
    k = rng.integers(0, 4, n)
    for i in range(n):
        r['inv'][i, :k[i], 0] = rng.integers(1, 700, k[i])
        r['inv'][i, :k[i], 1] = 1
    hi = np.array([700, 128, 128, 16, 32, 32])
    r['pitem'] = rng.integers(0, hi[None], (n, 6))
    r['pitem'][:, 0] *= rng.random(n) < 0.5
    r['pitem'][:, 1] *= rng.random(n) < 0.3
    r['pitem'][:, 2] = 0
    r['pitem'][:, 3] *= rng.random(n) < 0.4
    r['pitem'][:, 5] = 0
    r['pinv'] = rng.random(r['pinv'].shape)
    r['pcharge'] = rng.random(r['pcharge'].shape)
    r['pchar'] = rng.integers(0, N_CHAR, n)
    r['t'] = rng.integers(0, 2000, n)
    r['first'] = rng.random(n) < 0.01
    return r


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ck['config'])
    kw = {k: cfg.pop(k) for k in list(cfg) if k not in ('width', 'layers', 'heads')}
    model = TokPolicy(cfg['width'], cfg['layers'], cfg['heads'], **kw).to(device).eval()
    model.load_state_dict(ck['model'])
    return model, ck['config']


class Timer:
    """The instrumented GraphActor.submit + result (the same operations, CUDA events between them)."""

    def __init__(self, ga):
        self.ga = ga
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def call(self, rows, pending, widx, reset):
        if self.ga.fold:
            return self.call_fold(rows, pending, widx, reset)
        ga, E = self.ga, self.ev
        t0 = time.perf_counter()
        n = len(rows)
        if ga.mem:
            ga.widx_np[:n] = widx
            ga.keep_np[:n] = (~np.asarray(reset, bool)).astype(np.float32)
        most = int(rows['n_ent'].max())
        g = int(np.searchsorted(ga.sizes, most))
        self.g = g
        ga.stage_np[:n, :ROW_BYTES] = np.ascontiguousarray(rows).view(np.uint8).reshape(n, ROW_BYTES)
        ga.stage_np[:n, ROW_BYTES:] = np.ascontiguousarray(pending, np.int64).view(np.uint8).reshape(n, 8 * ga.nh)
        t1 = time.perf_counter()
        with torch.cuda.stream(ga.stream):
            E[0].record()
            ga.raw[:n].copy_(ga.stage[:n], non_blocking=True)
            E[1].record()
            ga.u.uniform_()
            if ga.mem:
                ga.widx[:n].copy_(ga.widx_stage[:n], non_blocking=True)
                ga.keep[:n].copy_(ga.keep_stage[:n], non_blocking=True)
                ga.h_in[:n] = ga.table.index_select(0, ga.widx[:n]) * ga.keep[:n, None]
            E[2].record()
            tr = time.perf_counter()
            ga.graphs[g].replay()
            tr = time.perf_counter() - tr
            E[3].record()
            if ga.mem:
                ga.table.index_copy_(0, ga.widx[:n], ga.h_outs[g][:n])
            ga.out_host[:n].copy_(ga.outs[g][:n], non_blocking=True)
            ga.done.record()
            E[4].record()
        t2 = time.perf_counter()
        E[4].synchronize()
        t3 = time.perf_counter()
        ga._n = n
        ga.result()
        t4 = time.perf_counter()
        gpu = [E[i].elapsed_time(E[i + 1]) for i in range(4)]
        return dict(total=1e3 * (t4 - t0), pack=1e3 * (t1 - t0), enqueue=1e3 * (t2 - t1), launch=1e3 * tr,
                    wait=1e3 * (t3 - t2), post=1e3 * (t4 - t3), h2d=gpu[0], pre=gpu[1], graph=gpu[2], after=gpu[3],
                    gpu=sum(gpu), slack=1e3 * (t3 - t1) - sum(gpu))


    def call_fold(self, rows, pending, widx, reset):
        """GraphActor._submit_fold + result, instrumented (pre = the control block's H2D copy)."""
        ga, E = self.ga, self.ev
        t0 = time.perf_counter()
        n = len(rows)
        g = self.g = ga._pack_fold(rows, pending, widx, reset)
        r = ga.rsizes[int(np.searchsorted(ga.rsizes, n))]
        g += ga.rsizes.index(r) * len(ga.sizes)
        t1 = time.perf_counter()
        with torch.cuda.stream(ga.stream):
            E[0].record()
            ga.raw[:n].copy_(ga.stage[:n], non_blocking=True)
            E[1].record()
            ga.ctrl[:r].copy_(ga.ctrl_stage[:r], non_blocking=True)
            E[2].record()
            tr = time.perf_counter()
            ga.graphs[g].replay()
            tr = time.perf_counter() - tr
            E[3].record()
            ga.out_host[:n].copy_(ga.outs[g][:n], non_blocking=True)
            ga.done.record()
            E[4].record()
        t2 = time.perf_counter()
        E[4].synchronize()
        t3 = time.perf_counter()
        ga._n = n
        ga.result()
        t4 = time.perf_counter()
        gpu = [E[i].elapsed_time(E[i + 1]) for i in range(4)]
        return dict(total=1e3 * (t4 - t0), pack=1e3 * (t1 - t0), enqueue=1e3 * (t2 - t1), launch=1e3 * tr,
                    wait=1e3 * (t3 - t2), post=1e3 * (t4 - t3), h2d=gpu[0], pre=gpu[1], graph=gpu[2], after=gpu[3],
                    gpu=sum(gpu), slack=1e3 * (t3 - t1) - sum(gpu))


def med(rs, keys=None):
    keys = keys or list(rs[0])
    return {k: round(statistics.median(r[k] for r in rs), 4) for k in keys}


class Bench:
    def __init__(self, args, model):
        self.args, self.model = args, model
        self.rng = np.random.default_rng(args.seed)
        self.heads = heads_of(model)
        lo, hi = (int(x) for x in args.ent.split(':'))
        self.rows32 = synth_rows(self.rng, 4096, lo, min(hi, args.graph_entities))
        self.rows64 = synth_rows(self.rng, 4096, max(lo, args.graph_entities + 1), max(hi, args.graph_entities + 1))
        self.pend = np.stack([self.rng.integers(0, k, 4096) for k in self.heads], 1).astype(np.int64)
        if args.ent_real:   # (empirical entity counts of a saved rollout: one value per record)
            emp = np.load(args.ent_real)
            self.rows_real = synth_rows(self.rng, 4096, 0, 1)
            self.rows_real['n_ent'] = self.rng.choice(emp, 4096)
        self.workers = max(256, max(args.sizes))
        self.table0 = torch.randn((self.workers, model.memory), device='cuda') * 0.5 if model.memory else None

    def actor(self, rows_max, ents=None, **kw):
        table = self.table0.clone() if self.table0 is not None else None
        ga = GraphActor(self.model, rows_max, ents or self.args.graph_entities, stream=torch.cuda.Stream(priority=-5),
                        crn=True, memory=table, **kw)
        return ga

    def batch(self, n, big=False, crn=False):
        src = self.rows64 if big else self.rows32
        if big == 'real':
            src, big = self.rows_real, False
        sel = self.rng.choice(len(src), n, replace=False)
        rows = src[sel].copy()
        if big:   # one record over the small graph's entities (the rest as usual): the 64-entity graph
            rows[1:] = self.rows32[self.rng.choice(len(self.rows32), n - 1, replace=False)]
        if crn:
            rows['crn'] = self.rng.integers(1, 2 ** 62, n)
        widx = self.rng.choice(self.workers, n, replace=False)
        reset = rows['first'] != 0
        return rows, self.pend[sel], widx, reset

    def timed(self, ga, n, big=False, iters=None):
        iters = iters or self.args.iters
        tm = Timer(ga)
        nb = 16 if big != 'real' else 256
        batches = [self.batch(n, big) for _ in range(nb)]
        for i in range(20):
            tm.call(*batches[i % nb])
        rs, gs = [], []
        for i in range(iters):
            rs.append(tm.call(*batches[i % nb]))
            gs.append(tm.g)
        # plain submit + result (no events): the end-to-end wall time
        torch.cuda.synchronize()
        plain = []
        for i in range(iters):
            b = batches[i % nb]
            t = time.perf_counter()
            ga.act(b[0], b[1], b[2], b[3])
            plain.append(1e3 * (time.perf_counter() - t))
        out = med(rs)
        if big == 'real':   # (mixed graphs: means, and how often each graph ran)
            out = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}
            out['graph_share'] = [round(float(np.mean(np.array(gs) == g)), 3) for g in range(len(ga.sizes))]
            out['plain_mean'] = round(float(np.mean(plain)), 4)
        out['plain'] = round(statistics.median(plain), 4)
        out['plain_p90'] = round(float(np.percentile(plain, 90)), 4)
        return out

    def outputs(self, ga, batch):
        """result() of one call on a batch, the table reset first (the same input states for every actor)."""
        ga.table.copy_(self.table0)
        r = ga.act(*batch)
        return r[0], r[1], r[2], ga.vg


def part_sizes(B, log):
    for s in B.args.sizes:
        ga = B.actor(s)
        log('sizes', rows_max=s, n=s, ents=32, **B.timed(ga, s))
        if s == 64:
            log('sizes', rows_max=s, n=s, ents=64, **B.timed(ga, s, big=True))
        del ga
        torch.cuda.empty_cache()


def part_partial(B, log):
    ga = B.actor(64)
    for n in (8, 16, 32, 64):
        log('partial', rows_max=64, n=n, ents=32, **B.timed(ga, n))


def part_profile(B, log, variant='base'):
    from torch.profiler import ProfilerActivity, profile
    set_model_flags(B.model, variant)
    for big in (False, True):
        ga = B.actor(64, **variant_kw(variant))
        tm = Timer(ga)
        bt = B.batch(64, big)
        for _ in range(10):
            tm.call(*bt)
        torch.cuda.synchronize()
        reps = 20
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                tm.call(*bt)
            torch.cuda.synchronize()
        kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        tot = sum(e.device_time for e in kern) / 1e3 / reps
        by = {}
        for e in kern:
            k = by.setdefault(e.name, [0, 0.0])
            k[0] += 1
            k[1] += e.device_time / 1e3
        top = sorted(by.items(), key=lambda kv: -kv[1][1])[:25]
        log('profile', variant=variant, ents=64 if big else 32, kernels_per_call=round(len(kern) / reps, 1),
            kernel_ms_per_call=round(tot, 4),
            top=[(name[:90], round(c / reps, 1), round(ms / reps * 1e3, 1)) for name, (c, ms) in top])


def variant_kw(v):
    kw = {}
    for part in v.split('+'):
        if part in ('math', 'nhwc'):   # (model flags: set_model_flags, before the actor is built)
            continue
        if part == 'bf16':
            kw['bf16'] = True
        elif part == 'fold':
            kw['fold'] = True
        elif part == 'sdpa':
            kw['sdpa_math'] = True
        elif part.startswith('ents='):   # e.g. ents=16/32: entity-token sizes of the graphs
            kw['ents'] = [int(x) for x in part[5:].split('/')]
        elif part.startswith('rows='):   # e.g. rows=16/32: more batch sizes of the graphs (fold)
            kw['row_sizes'] = [int(x) for x in part[5:].split('/')]
        elif part.startswith('compile'):
            kw['compile_mode'] = part.split(':', 1)[1] if ':' in part else 'default'
    return kw


def set_model_flags(model, v):
    """math: the blocks' attention as _MathAttention (strict float32 products); nhwc: channels-last convolutions
    (TokPolicy.set_math_attention, as the learner's fast path); off for every other variant."""
    parts = v.split('+')
    model.set_math_attention('math' in parts, nhwc='nhwc' in parts, ln=False)


def part_variants(B, log):
    ref = {}
    for v in ['base'] + [x for x in B.args.variants if x != 'base']:
        for s in B.args.variant_sizes:
            t = time.perf_counter()
            set_model_flags(B.model, v)
            ga = B.actor(s, **variant_kw(v))
            build = time.perf_counter() - t
            res = dict(build_s=round(build, 1), **B.timed(ga, s))
            res64 = B.timed(ga, s, big=True, iters=max(50, B.args.iters // 4))
            res['graph64'] = res64['graph']
            res['plain64'] = res64['plain']
            if B.args.ent_real:   # (records with the saved rollout's entity counts: mean over mixed graphs)
                rr = B.timed(ga, B.args.real_n or s, big='real')
                res.update(real_plain_mean=rr['plain_mean'], real_graph_mean=rr['graph'], real_share=rr['graph_share'])
            # numerics: the same records with common random numbers (the same uniforms in every actor)
            rs = np.random.default_rng(B.args.seed + s)
            saved, B.rng = B.rng, rs
            bt = B.batch(s, crn=True)
            bt64 = B.batch(s, big=True, crn=True)
            B.rng = saved
            o = [B.outputs(ga, bt), B.outputs(ga, bt64)]
            if v == 'base':
                ref[s] = o
            else:
                d = {}
                for j, tag in ((0, ''), (1, '64')):
                    a, b_ = ref[s][j], o[j]
                    d['act_agree' + tag] = float((a[0] == b_[0]).all(1).mean())
                    d['dlogp' + tag] = float(np.abs(a[1] - b_[1]).max())
                    d['dvalue' + tag] = float(np.abs(a[2] - b_[2]).max())
                    if a[3] is not None:
                        d['dvg' + tag] = float(np.abs(a[3] - b_[3]).max())
                res.update({k: float(f'{x:.3g}') for k, x in d.items()})
            if 'bf16' in v:
                res['dlogit_eager'] = logit_delta(B, bt)
            log('variants', variant=v, rows_max=s, **res)
            if B.args.profile_each:
                part_profile(B, log, v)
            set_model_flags(B.model, 'base')
            del ga
            torch.cuda.empty_cache()


def logit_delta(B, bt):
    """max |logit(bf16 autocast) - logit(float32)| of the eager forward on the same records and input states."""
    rows, pend, widx, reset = bt
    raw = torch.zeros((len(rows), ROW_BYTES + 8 * len(B.heads)), dtype=torch.uint8)
    raw[:, :ROW_BYTES] = torch.from_numpy(np.ascontiguousarray(rows).view(np.uint8).reshape(len(rows), ROW_BYTES))
    raw[:, ROW_BYTES:] = torch.from_numpy(pend.view(np.uint8).reshape(len(rows), -1))
    raw = raw.cuda()
    b = decode_rows(raw, entities=ENT_CAP, n_heads=len(B.heads))
    h = B.table0.index_select(0, torch.from_numpy(widx).cuda()) if B.table0 is not None else None
    with torch.inference_mode():
        kw = dict(mem=h) if h is not None else {}
        l32 = B.model(b, ENT_CAP, **kw)[0].float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l16 = B.model(b, ENT_CAP, **kw)[0].float()
    return float((l32 - l16).abs().max())


def part_wcheck(B, log):
    """Weights changed in place after the capture (as train_tok's take_weights does) reach a variant's graphs: the
    variant against base on the same records (common random numbers), before and after a perturbation."""
    v = B.args.wcheck_variant
    set_model_flags(B.model, 'base')
    ga_b = B.actor(64)
    set_model_flags(B.model, v)
    ga_v = B.actor(64, **variant_kw(v))
    rs = np.random.default_rng(B.args.seed + 99)
    saved_rng, B.rng = B.rng, rs
    bt = B.batch(64, crn=True)
    B.rng = saved_rng

    def d(a, b):
        return dict(act=float((a[0] == b[0]).all(1).mean()), logp=float(np.abs(a[1] - b[1]).max()),
                    value=float(np.abs(a[2] - b[2]).max()))
    o0b, o0v = B.outputs(ga_b, bt), B.outputs(ga_v, bt)
    saved = [p.detach().clone() for p in B.model.parameters()]
    g = torch.Generator(device='cuda').manual_seed(5)
    for p in B.model.parameters():
        p.add_(torch.randn(p.shape, device=p.device, generator=g) * (0.1 * p.abs().mean() + 1e-3))
    o1b, o1v = B.outputs(ga_b, bt), B.outputs(ga_v, bt)
    for p, q in zip(B.model.parameters(), saved):
        p.copy_(q)
    set_model_flags(B.model, 'base')
    log('wcheck', variant=v, before=d(o0b, o0v), after=d(o1b, o1v), base_change=d(o0b, o1b))


class TemporalStandIn(nn.Module):
    """B27 stand-in (random weights): one query token per record, `layers` layers of width d; each attends to its
    worker's ring of `ring` past keys / values (per layer, on the device, gathered by the record's worker slot) and to
    `room` room tokens (projected per layer from a [n, room, width_in] tensor); the step's key / value written back
    into the ring at the worker's position (which then advances)."""

    def __init__(self, workers, rows_max, width_in, d=256, layers=4, heads=4, ring=256, room=128, kv_dtype=torch.float32):
        super().__init__()
        self.h, self.ring, self.layers = heads, ring, layers
        self.inp = nn.Linear(width_in, d)
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.q = nn.ModuleList(nn.Linear(d, 3 * d) for _ in range(layers))   # q, and the step's own k, v
        self.kv_room = nn.ModuleList(nn.Linear(width_in, 2 * d) for _ in range(layers))
        self.o = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(layers))
        self.register_buffer('K', torch.randn(layers, workers, ring, d).to(kv_dtype))
        self.register_buffer('V', torch.randn(layers, workers, ring, d).to(kv_dtype))
        self.register_buffer('pos', torch.zeros(workers, dtype=torch.int64))
        self.register_buffer('room_x', torch.randn(rows_max, room, width_in))
        self.register_buffer('room_n', torch.randint(20, room + 1, (rows_max,)))
        self.register_buffer('widx', torch.arange(rows_max) % workers)

    def forward(self, x):
        n, d = x.shape[0], self.inp.out_features
        widx = self.widx[:n]
        p = self.pos.index_select(0, widx) % self.ring
        room = self.room_x[:n]
        T = room.shape[1]
        valid = torch.cat([torch.ones((n, self.ring), dtype=torch.bool, device=x.device),
                           torch.arange(T, device=x.device)[None] < self.room_n[:n, None]], 1)
        bias = torch.zeros((n, 1, 1, self.ring + T), device=x.device).masked_fill(~valid[:, None, None], -1e9)
        h = self.inp(x)
        dh = d // self.h
        for l in range(self.layers):
            q, k_new, v_new = self.q[l](self.n1[l](h)).chunk(3, -1)
            kr, vr = self.kv_room[l](room).chunk(2, -1)
            K = torch.cat([self.K[l].index_select(0, widx).to(kr.dtype), kr], 1)
            V = torch.cat([self.V[l].index_select(0, widx).to(vr.dtype), vr], 1)
            a = F.scaled_dot_product_attention(q.reshape(n, self.h, 1, dh), K.view(n, -1, self.h, dh).transpose(1, 2),
                                               V.view(n, -1, self.h, dh).transpose(1, 2), attn_mask=bias.to(q.dtype))
            h = h + self.o[l](a.reshape(n, d))
            h = h + self.ff[l](self.n2[l](h))
            self.K[l].index_put_((widx, p), k_new.to(self.K.dtype))
            self.V[l].index_put_((widx, p), v_new.to(self.V.dtype))
        self.pos.index_add_(0, widx, torch.ones_like(widx))
        return h


def part_temporal(B, log):
    a = B.args
    for kv in a.temporal_kv.split(','):
        kv_dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[kv]
        for v in a.temporal_variants:
            for s in a.variant_sizes:
                tm = TemporalStandIn(B.workers, s, B.model.config['width'], a.temporal_width, a.temporal_layers, 4,
                                     a.temporal_ring, a.temporal_room, kv_dtype).cuda().eval()
                ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'bf16' in v.split('+') else None

                def extra(b, outs, h_in, tm=tm, ctx=ctx):
                    x = outs[0].float().new_zeros((outs[0].shape[0], B.model.config['width']))
                    x = x + outs[1].float()[:, None]   # (a dependency on the network's outputs)
                    if ctx is None:
                        return tm(x)
                    with ctx:
                        return tm(x)
                t = time.perf_counter()
                ga = B.actor(s, extra=extra, **variant_kw(v))
                build = time.perf_counter() - t
                log('temporal', kv=kv, variant=v, rows_max=s, build_s=round(build, 1), **B.timed(ga, s))
                del ga, tm
                torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--parts', default='sizes,partial,profile,variants,temporal')
    p.add_argument('--sizes', default='8,16,32,64,128,256')
    p.add_argument('--variant-sizes', default='64,128')
    p.add_argument('--variants', default='base,bf16,compile:default,compile:max-autotune-no-cudagraphs,'
                                         'bf16+compile:max-autotune-no-cudagraphs')
    p.add_argument('--profile-variants', default='base')
    p.add_argument('--profile-each', type=int, default=0, help='1: a kernel profile after every variant')
    p.add_argument('--wcheck-variant', default='compile:max-autotune-no-cudagraphs+fold+sdpa+ents=16/32/40')
    p.add_argument('--temporal-variants', default='base,compile:max-autotune-no-cudagraphs')
    p.add_argument('--temporal-kv', default='fp32,bf16')
    p.add_argument('--temporal-width', type=int, default=256)
    p.add_argument('--temporal-layers', type=int, default=4)
    p.add_argument('--temporal-ring', type=int, default=256)
    p.add_argument('--temporal-room', type=int, default=128)
    p.add_argument('--ent', default='12:40', help='entity counts lo:hi (the 32-entity graph uses lo..min(hi, 32))')
    p.add_argument('--graph-entities', type=int, default=32)
    p.add_argument('--ent-real', default='', help='a .npy of entity counts (one per record) for the mixed-graph timing')
    p.add_argument('--real-n', type=int, default=0, help='records per call of the mixed-graph timing (0: rows_max)')
    p.add_argument('--iters', type=int, default=300)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tf32', type=int, default=1)
    p.add_argument('--out', default='')
    args = p.parse_args()
    args.sizes = [int(x) for x in args.sizes.split(',')]
    args.variant_sizes = [int(x) for x in args.variant_sizes.split(',')]
    args.variants = args.variants.split(',')
    args.temporal_variants = args.temporal_variants.split(',')
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.manual_seed(args.seed)
    model, cfg = load_model(args.checkpoint, 'cuda')
    print('model', cfg, flush=True)
    B = Bench(args, model)
    out = open(args.out, 'a') if args.out else None

    def log(part, **kw):
        line = json.dumps(dict(part=part, **kw))
        print(line, flush=True)
        if out:
            out.write(line + '\n')
            out.flush()
    torch.cuda.set_stream(torch.cuda.Stream(priority=-5))
    with torch.inference_mode():
        for part in args.parts.split(','):
            if part == 'sizes':
                part_sizes(B, log)
            elif part == 'partial':
                part_partial(B, log)
            elif part == 'profile':
                for v in args.profile_variants.split(','):
                    part_profile(B, log, v)
            elif part == 'variants':
                part_variants(B, log)
            elif part == 'temporal':
                part_temporal(B, log)
            elif part == 'wcheck':
                part_wcheck(B, log)


if __name__ == '__main__':
    main()
