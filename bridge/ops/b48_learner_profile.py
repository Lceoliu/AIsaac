"""B48 A: where one real update's learner time goes, on its dumped inputs (learner_dump / b44_learner_bench).

  python b48_learner_profile.py <learner_inputs.pt> [--reps 2] [--no-history] [--ckpt 0|1] [--json out.json]

Replays the update's calls as b44_learner_bench.replay (forward, surrogate loss, backward, clip, Adam; every
segment GPU-synchronised) and adds, from the last rep:
  - module GPU time by CUDA events: Spatial (structured_spatial + resampler; forward, and its re-run inside
    backward when checkpointed), Writer, Reader (read_flex / BlockMemory.read), heads (pi, value, danger, gtok);
    backward: Writer and heads by module backward hooks, Reader between markers on its output and its z input,
    Spatial as the rest of the backward;
  - rows by type (main / historical / open-tail) and the Spatial plans: per call T bucket, real rows, real and
    padded tokens.
--no-history: the grad calls without historical re-encoding (train_history=False): the historical rows' cost.
--ckpt: block_learner.SPATIAL_CKPT for the replay (memory peak / time with and without checkpoint).
Measurement only; the model and its arithmetic are unchanged."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('ISAAC_BLOCK_TIMING', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
sys.path.insert(0, HERE)
import numpy as np
import torch

from isaac_bridge import block_learner, block_paged as bp, learner_dump, structured_input as si
from isaac_bridge.block_training import forward_bucketed
from b44_learner_bench import surrogate, sync, fresh, load_model


class Spans:
    """CUDA event pairs per (region, phase); elapsed summed after a synchronisation."""
    def __init__(self):
        self.on = False
        self.pairs = {}
        self.open = {}

    def start(self, key):
        if self.on:
            e = torch.cuda.Event(enable_timing=True); e.record()
            self.open.setdefault(key, []).append(e)

    def end(self, key):
        if self.on and self.open.get(key):
            e = torch.cuda.Event(enable_timing=True); e.record()
            self.pairs.setdefault(key, []).append((self.open[key].pop(), e))

    def totals(self):
        torch.cuda.synchronize()
        return {f'{r}:{p}': round(sum(a.elapsed_time(b) for a, b in v)/1e3, 3) for (r, p), v in self.pairs.items()}


SP = Spans()


class Marker(torch.autograd.Function):
    """Identity; its backward records an event (the Reader's backward span)."""
    @staticmethod
    def forward(ctx, x, key):
        ctx.key = key
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        (SP.start if ctx.key[1] == 'out' else SP.end)((ctx.key[0], 'bwd'))
        return g, None


def instrument(model):
    def fwd_hooks(module, region):
        def pre(m, a):
            SP.start((region, 'fwd' if torch.is_grad_enabled() or not m.training else 'fwd_nograd')
                     if not getattr(torch.utils.checkpoint, '_b48_recompute', False) else (region, 'recompute'))
        def post(m, a, o):
            SP.end((region, 'fwd' if torch.is_grad_enabled() or not m.training else 'fwd_nograd')
                   if not getattr(torch.utils.checkpoint, '_b48_recompute', False) else (region, 'recompute'))
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def bwd_hooks(module, region):
        module.register_full_backward_pre_hook(lambda m, go: SP.start((region, 'bwd')))
        module.register_full_backward_hook(lambda m, gi, go: SP.end((region, 'bwd')))

    mem = model.block_memory
    for name, mod in (('spatial', model.structured_spatial), ('spatial', mem.resampler)):
        fwd_hooks(mod, name)
    fwd_hooks(mem.writer, 'writer'); bwd_hooks(mem.writer, 'writer')
    for name in ('pi', 'value', 'danger', 'gtok'):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Module):
            fwd_hooks(mod, 'heads'); bwd_hooks(mod, 'heads')

    def wrap_reader(fn):
        def run(memory, z, *a, **k):
            phase = 'fwd' if torch.is_grad_enabled() else 'fwd_nograd'
            if torch.is_grad_enabled() and z.requires_grad:
                z = Marker.apply(z, ('reader', 'in'))
            SP.start(('reader', phase))
            out = fn(memory, z, *a, **k)
            SP.end(('reader', phase))
            if torch.is_grad_enabled():
                if isinstance(out, tuple):
                    out = (Marker.apply(out[0], ('reader', 'out')),)+tuple(out[1:])
                else:
                    out = Marker.apply(out, ('reader', 'out'))
            return out
        return run
    block_learner.read_flex = wrap_reader(block_learner.read_flex)
    plain = type(mem).read
    type(mem).read = lambda self, z, *a, **k: wrap_reader(lambda m, z_, *a_, **k_: plain(m, z_, *a_, **k_))(self, z, *a, **k)

    # Spatial's re-run inside the backward (non-reentrant checkpoint): flagged so its hooks count as recompute
    # (forward_packed imports torch.utils.checkpoint.checkpoint at each call: patched there)
    orig_ckpt = torch.utils.checkpoint.checkpoint

    def ckpt(fn, *a, **k):
        return orig_ckpt(_RecomputeFlag(fn), *a, **k)
    torch.utils.checkpoint.checkpoint = ckpt


class _RecomputeFlag:
    """checkpoint's function: the second call (the re-run during backward) is flagged as recompute."""
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        torch.utils.checkpoint._b48_recompute = self.calls > 1
        try:
            return self.fn(*a, **k)
        finally:
            torch.utils.checkpoint._b48_recompute = False


PLANS = []


def record_plans():
    orig = si.packed_rows_plan

    def plan(widths, rows, counts, starts=None):
        idx, T = orig(widths, rows, counts, starts)
        if SP.on:
            PLANS.append((int(T), int(len(rows)), int((30+np.asarray(counts).sum(1)).sum())))
        return idx, T
    si.packed_rows_plan = plan


def replay(model, opt, calls, no_history, seed):
    fresh(calls)
    bp.UPD.clear()
    torch.manual_seed(seed)
    cache = {}
    t = dict(fwd_grad=0.0, fwd_nograd=0.0, bwd=0.0, opt=0.0)
    torch.cuda.reset_peak_memory_stats()
    t_all = sync()
    for c in calls:
        kw = dict(c['kwargs'])
        if c['grad']:
            if no_history:
                kw['train_history'] = False
            model.train()
            t0 = sync()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            t1 = sync(); t['fwd_grad'] += t1-t0
            loss = surrogate(out, cache)
            SP.start(('backward', 'all'))
            loss.backward()
            SP.end(('backward', 'all'))
            t2 = sync(); t['bwd'] += t2-t1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); opt.zero_grad(set_to_none=True)
            t['opt'] += sync()-t2
            del out, ctx, loss
        else:
            t0 = sync()
            mode = torch.inference_mode() if c['inference'] else torch.no_grad()
            with mode, torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            t['fwd_nograd'] += sync()-t0
            del out, ctx
    t['total'] = sync()-t_all
    upd = dict(bp.UPD)
    bp.clear_decode_cache()
    r = {k: round(v, 3) for k, v in t.items()}
    for k in ('learn_main_decode_rows_fresh', 'learn_main_decode_rows_cached', 'learn_hist_decode_rows_fresh',
              'learn_tail_decode_rows', 'learn_spatial_calls', 'learn_spatial_tokens_real', 'learn_spatial_tokens_padded',
              'learn_reader_calls', 'learn_reader_rows_real', 'learn_reader_rows_padded', 'learn_writer_calls',
              'learn_packed_host_s', 'learn_main_decode_s', 'learn_hist_decode_s'):
        r[k] = round(upd.get(k, 0), 3)
    r['peak_allocated_gb'] = round(torch.cuda.max_memory_allocated()/2**30, 2)
    r['peak_reserved_gb'] = round(torch.cuda.max_memory_reserved()/2**30, 2)
    return r


def simulate_pool(model, calls, reps=3):
    """Historical blocks per grad call: selections, distinct blocks with the current per-chunk draw, with the
    B48 per-call pool (K=8, 16) and with one pool per segment for the whole update (offline only); chunks per
    segment per call."""
    cfg = model.block_memory.config
    out = []
    gen = torch.Generator().manual_seed(0)
    update_union = {}
    per_call = []
    for c in calls:
        if not c['grad']:
            continue
        chunks = c['chunks']
        parts = [None if ch.rows['first'][0] else ch.snapshot.store.snapshot_parts(ch.snapshot) for ch in chunks]
        idx = [i for i, p in enumerate(parts) if p is not None and np.any(p['ends'] <= p['step']-cfg.window)]
        per_call.append((chunks, parts, idx))
        for i in idx:
            p = parts[i]
            e = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
            update_union.setdefault(chunks[i].snapshot.log.episode, set()).update(p['bid'][e].tolist())
    tot = dict(calls=len(per_call), chunks=0, chunks_with_history=0, selections=0, distinct_now=0.0,
               distinct_pool8=0.0, distinct_pool16=0.0, distinct_update_pool8=0.0, distinct_update_pool16=0.0,
               segments=0, chunks_per_segment_per_call=[])
    for chunks, parts, idx in per_call:
        tot['chunks'] += len(chunks); tot['chunks_with_history'] += len(idx)
        segs = {}
        for i in idx:
            segs.setdefault(chunks[i].snapshot.log.episode, []).append(i)
        tot['segments'] += len(segs)
        tot['chunks_per_segment_per_call'] += [len(v) for v in segs.values()]
        n_take = cfg.history_recompute_blocks
        for r in range(reps):
            now = set()
            for i in idx:
                p = parts[i]
                e = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
                pick = torch.randperm(len(e), generator=gen)[:n_take].numpy()
                now.update(p['bid'][e[pick]].tolist())
                if r == 0:
                    tot['selections'] += min(n_take, len(e))
            tot['distinct_now'] += len(now)/reps
            for K in (8, 16):
                sel = block_learner.pool_selection(chunks, [p if p is not None else dict(ends=np.empty(0), step=0,
                                                   bid=np.empty(0, np.int64)) for p in parts], idx, n_take,
                                                   cfg.window, K, gen)
                tot[f'distinct_pool{K}'] += len({int(parts[i]['bid'][k]) for i in idx for k in sel[i]})/reps
                # one pool per segment for the whole update: K blocks of the segment's update-wide union
                pools = {seg: set(np.random.default_rng(r*7+K).permutation(sorted(u))[:K].tolist())
                         for seg, u in update_union.items()}
                used = set()
                for i in idx:
                    p = parts[i]
                    e = np.flatnonzero(p['ends'] <= p['step']-cfg.window)
                    bids = p['bid'][e]
                    take = min(n_take, len(e))
                    inside = [b for b in bids.tolist() if b in pools[chunks[i].snapshot.log.episode]]
                    rng = np.random.default_rng(r*131+i)
                    if len(inside) >= take:
                        used.update(rng.permutation(inside)[:take].tolist())
                    else:
                        rest = [b for b in bids.tolist() if b not in set(inside)]
                        used.update(inside); used.update(rng.permutation(rest)[:take-len(inside)].tolist())
                tot[f'distinct_update_pool{K}'] += len(used)/reps
    cps = np.asarray(tot.pop('chunks_per_segment_per_call'))
    tot['chunks_per_segment_per_call'] = dict(mean=round(float(cps.mean()), 2) if len(cps) else 0,
                                              p50=float(np.median(cps)) if len(cps) else 0, max=int(cps.max()) if len(cps) else 0)
    tot['segments_in_update'] = len(update_union)
    bs = cfg.block_steps
    for k in [k for k in tot if k.startswith('distinct')]:
        tot[k] = round(tot[k], 1)
        tot[k.replace('distinct', 'hist_rows')] = int(tot[k]*bs)
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--no-history', action='store_true')
    p.add_argument('--ckpt', type=int, default=None)
    p.add_argument('--json', default='')
    p.add_argument('--pool-sim', action='store_true', help='historical block counts with the B48 pools (no replay)')
    a = p.parse_args()
    dev = torch.device('cuda')
    from isaac_bridge import block_compile
    block_compile.configure()
    block_learner.PACKED = True
    if a.ckpt is not None:
        block_learner.SPATIAL_CKPT = a.ckpt
    ck, store, chunks, calls, extra = learner_dump.load(a.path)
    model, opt = load_model(ck, dev)
    if a.pool_sim:
        if not hasattr(block_learner, 'pool_selection'):   # (the B48 pool code is kept on refs/b48/b48b-with-pool)
            raise SystemExit('--pool-sim needs the B48 history pool code (refs/b48/b48b-with-pool)')
        r = simulate_pool(model, calls)
        print(json.dumps(r), flush=True)
        if a.json:
            with open(a.json, 'w') as f:
                json.dump(r, f, indent=1)
        return
    instrument(model)
    record_plans()
    out = dict(calls=len(calls), grad_calls=sum(c['grad'] for c in calls), chunks=len(chunks),
               no_history=a.no_history, ckpt=block_learner.SPATIAL_CKPT)
    for i in range(a.reps):
        last = i == a.reps-1
        SP.on = last
        SP.pairs, SP.open = {}, {}
        PLANS.clear()
        r = replay(model, opt, calls, a.no_history, seed=i)
        if last:
            r['gpu_s'] = SP.totals()
            pl = np.asarray(PLANS, np.int64).reshape(-1, 3)
            buckets = {}
            for T, m, tok in pl.tolist():
                b = buckets.setdefault(T, [0, 0, 0, 0])
                b[0] += 1; b[1] += m; b[2] += tok; b[3] += max(2, m)*T
            r['spatial_buckets'] = {int(T): dict(calls=v[0], rows=v[1], tokens_real=v[2], tokens_padded=v[3],
                                                 real_over_padded=round(v[2]/max(v[3], 1), 3))
                                    for T, v in sorted(buckets.items())}
        print(json.dumps(dict(rep=i, **r)), flush=True)
        out[f'rep{i}'] = r
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
