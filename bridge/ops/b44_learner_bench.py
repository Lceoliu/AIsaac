"""B44 fixed-input learner benchmark: one real update's learner work, replayed without games.

Input: <dir>/learner_inputs.pt from a short run with ISAAC_LEARNER_DUMP=<dir> (isaac_bridge/learner_dump.py):
every forward_bucketed call of that update (its chunks, grad mode, keyword arguments), the referenced
page store and the weights the update started from.

  python b44_learner_bench.py <learner_inputs.pt> --mode old|packed [--reps 2] [--json out.json]
  python b44_learner_bench.py <learner_inputs.pt> --equiv [--calls 3]
  B60: [--pipeline] (host preparations queued one call ahead, as the learner) [--busy N] (N pure-Python busy
  threads during the reps: the GIL exposure) [--opcount] (one more rep counting Python-level torch calls by phase);
  ISAAC_BLOCK_NATIVE_PACK=0 / 1 selects the Python / C host packing.

Replay (each rep): the calls in their recorded order; a no-grad / inference call is a forward
(bf16 autocast, as the learner); a grad call is forward + a fixed surrogate loss over every output
(the same for both paths: fixed random weights on logits, value, danger, value_global) + backward
+ clip_grad_norm_(0.5) + Adam step (the run's optimizer: lr 1e-4, eps 1e-5). Every segment is
GPU-synchronised (wall = GPU-finished time). The first rep compiles; reported: the last rep.
--mode old: the per-length path (ISAAC_PACKED_LEARNER=0); packed: one execution per call.
--equiv: the first grad calls on both paths from the same weights (same history draws): max |diff|
of every output, the loss and every parameter gradient.
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault('ISAAC_BLOCK_TIMING', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch

from isaac_bridge import block_learner, block_paged as bp, learner_dump
from isaac_bridge.block_training import forward_bucketed
from isaac_bridge.tok_policy import TokPolicy


def surrogate(out, gen_cache):
    loss = 0.0
    for i, o in enumerate(out):
        o = o.float()
        key = (i, o.shape[1:] if o.dim() > 1 else ())
        if key not in gen_cache:
            g = torch.Generator(device='cpu').manual_seed(100+i)
            gen_cache[key] = torch.randn(o.shape[1:] if o.dim() > 1 else (1,), generator=g).to(o.device)
        w = gen_cache[key]
        loss = loss+((o*w).mean() if o.dim() > 1 else (o*w[0]).mean()+o.square().mean()*1e-3)
    return loss


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def fresh(calls):
    """A new update's state: no host decode cached on the chunks or in the page store (a real
    update decodes its new rollout once; DECODE_CACHE is cleared at its start)."""
    for c in calls:
        for ch in c['chunks']:
            ch.__dict__.pop('_host', None)
            ch.snapshot.store.decoded = {}
    bp.clear_decode_cache()


class OpCount(torch.overrides.TorchFunctionMode):
    """B60: Python-level torch calls (functions, methods, tensor operators) by phase; each is at least one GIL
    reacquisition when the call drops the GIL into C++."""
    def __init__(self):
        super().__init__()
        self.phase = 'other'
        self.n = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n[self.phase] = self.n.get(self.phase, 0)+1
        return func(*args, **(kwargs or {}))


def busy_threads(n):
    """B60 diagnosis: n pure-Python busy threads (a GIL-holding consumer); returns done() -> spin count."""
    import threading
    stop = threading.Event()
    spins = [0]*n

    def spin(i):
        x = 0
        while not stop.is_set():
            for _ in range(1000):
                x += 1
            spins[i] += 1
    th = [threading.Thread(target=spin, args=(i,), daemon=True) for i in range(n)]
    for t_ in th:
        t_.start()

    def done():
        stop.set()
        for t_ in th:
            t_.join()
        return sum(spins)
    return done


def replay(model, opt, calls, packed, seed=0, pipeline=False, ops=None):
    """pipeline (B60): every call's host preparation queued on block_learner.PREFETCH right after the previous
    call's forward (the first one before the replay), as the learner does; ops: an OpCount (phases fwd_grad / bwd /
    opt / fwd_nograd)."""
    block_learner.PACKED = packed
    fresh(calls)
    bp.UPD.clear()
    torch.manual_seed(seed)
    cache = {}
    t = dict(fwd_grad=0.0, fwd_nograd=0.0, bwd=0.0, opt=0.0)
    n_grad = 0
    torch.cuda.reset_peak_memory_stats()

    def queue(i):
        if pipeline and i < len(calls):
            c_ = calls[i]
            kw_ = dict(c_['kwargs'])
            kw_.setdefault('require_next_state', False)
            block_learner.PREFETCH.submit(model, c_['chunks'], grad=bool(c_['grad']), **kw_)

    def phase(name):
        if ops is not None:
            ops.phase = name
    block_learner.CRIT_LOG = []
    t_all = sync()
    cpu0 = time.thread_time()
    queue(0)
    for i, c in enumerate(calls):
        kw = dict(c['kwargs'])
        if c['grad']:
            n_grad += 1
            model.train()
            t0 = sync()
            phase('fwd_grad')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            queue(i+1)
            t1 = sync(); t['fwd_grad'] += t1-t0
            phase('bwd')
            loss = surrogate(out, cache)
            loss.backward()
            t2 = sync(); t['bwd'] += t2-t1
            phase('opt')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); opt.zero_grad(set_to_none=True)
            t['opt'] += sync()-t2
            del out, ctx, loss
        else:
            t0 = sync()
            phase('fwd_nograd')
            mode = torch.inference_mode() if c['inference'] else torch.no_grad()
            with mode, torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            queue(i+1)
            t['fwd_nograd'] += sync()-t0
            del out, ctx
    phase('other')
    t['total'] = sync()-t_all
    t['learner_thread_cpu'] = time.thread_time()-cpu0
    block_learner.PREFETCH.drop()
    crit = block_learner.CRIT_LOG
    block_learner.CRIT_LOG = None
    # B60: host packing on the learner's critical path, per forward call (prefetched part's remaining wait, or a
    # part prepared on the learner thread)
    t['pack_crit_s'] = sum(w for _, w, _ in crit)
    t['pack_crit_grad_s'] = sum(w for g, w, _ in crit if g)
    t['pack_crit_nograd_s'] = sum(w for g, w, _ in crit if not g)
    t['pack_crit_max_s'] = max((w for _, w, _ in crit), default=0.0)
    t['pack_crit_prefetched'] = sum(1 for _, _, h in crit if h)
    t['pack_crit_per_call'] = [round(w, 4) for _, w, _ in crit]
    t['staging_buffers'] = len(block_learner.STAGING.bufs)
    t['staging_pinned_mb'] = sum(b[0].numel() for b in block_learner.STAGING.bufs if b[0] is not None)/2**20
    bp.GpuSpan.drain()
    upd = dict(bp.UPD)
    bp.clear_decode_cache()
    keys = ('learn_paged_calls', 'learn_packed_calls', 'learn_reader_calls', 'learn_reader_rows_real',
            'learn_reader_rows_padded', 'learn_spatial_calls', 'learn_spatial_tokens_real', 'learn_spatial_tokens_padded',
            'learn_writer_calls', 'learn_packed_host_s', 'learn_staged_mb', 'learn_main_decode_rows_fresh',
            'learn_hist_decode_rows_fresh', 'learn_tail_decode_rows', 'learn_main_decode_s', 'learn_hist_decode_s',
            'learn_prefetch_host_s', 'learn_prefetch_cpu_s', 'learn_prefetch_wait_s', 'learn_prefetch_hits',
            'learn_prefetch_dropped', 'learn_prefetch_other_calls', 'learn_prefetch_submitted',
            'learn_prefetch_wait_grad_s', 'learn_prefetch_wait_nograd_s', 'learn_prefetch_skip_wait_s',
            'learn_pack_crit_s')
    r = {k: round(v, 3) for k, v in t.items()}
    r.update({k: upd.get(k, 0) for k in keys})
    r['grad_calls'] = n_grad
    r['calls'] = len(calls)
    r['peak_allocated_gb'] = round(torch.cuda.max_memory_allocated()/2**30, 2)
    r['peak_reserved_gb'] = round(torch.cuda.max_memory_reserved()/2**30, 2)
    return r


def load_model(ck, dev):
    model = TokPolicy(**ck['config']).to(dev)
    model.load_state_dict(ck['model'])
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
    return model, opt


def equiv(ck, calls, dev, n, fp32=False):
    grads = [c for c in calls if c['grad']][:n]
    rep = []
    for k, c in enumerate(grads):
        res = []
        for packed in (False, True):
            model, _ = load_model(ck, dev)
            model.train()
            block_learner.PACKED = packed
            fresh([c])
            torch.manual_seed(1234+k)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
                out, ctx = forward_bucketed(model, c['chunks'], **c['kwargs'])
            loss = surrogate(out, {})
            loss.backward()
            res.append(([o.detach().float() for o in out], float(loss), {n_: p.grad.float() for n_, p in
                                                                       model.named_parameters() if p.grad is not None},
                        ctx['recompute_stats']))
        (oa, la, ga, sa), (ob, lb, gb, sb) = res
        names = ('logits', 'value', 'danger', 'value_global')
        r = dict(call=k, rows=int(sum(len(x.rows) for x in c['chunks'])), chunks=len(c['chunks']),
                 loss_old=la, loss_packed=lb, loss_abs_diff=abs(la-lb),
                 out_max_abs_diff={names[i] if i < len(names) else str(i): float((x-y).abs().max())
                                   for i, (x, y) in enumerate(zip(oa, ob))},
                 out_max_abs={names[i] if i < len(names) else str(i): float(x.abs().max()) for i, x in enumerate(oa)},
                 grad_max_abs_diff=max(float((ga[n_]-gb[n_]).abs().max()) for n_ in ga),
                 grad_max_rel_diff=max(float((ga[n_]-gb[n_]).norm()/(ga[n_].norm()+1e-12)) for n_ in ga),
                 grad_keys_equal=ga.keys() == gb.keys(), stats_old=sa, stats_packed=sb, fp32=fp32,
                 grad_total_rel_diff=float(torch.sqrt(sum(((ga[n_]-gb[n_])**2).sum() for n_ in ga)) /
                                          torch.sqrt(sum((ga[n_]**2).sum() for n_ in ga))),
                 grad_worst=sorted(((float((ga[n_]-gb[n_]).norm()/(ga[n_].norm()+1e-12)), n_, float(ga[n_].norm()))
                                    for n_ in ga), reverse=True)[:6])
        print(json.dumps(r), flush=True)
        rep.append(r)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--mode', choices=('old', 'packed'), default='packed')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--json', default='')
    p.add_argument('--equiv', action='store_true')
    p.add_argument('--calls', type=int, default=3)
    p.add_argument('--fp32', action='store_true', help='--equiv without autocast (FP32 everywhere)')
    p.add_argument('--cprofile', action='store_true', help='cProfile of the last rep (host time by function)')
    p.add_argument('--pipeline', action='store_true', help='B60: host preparations queued one call ahead (learner)')
    p.add_argument('--busy', type=int, default=0, help='B60: pure-Python busy threads during the reps')
    p.add_argument('--opcount', action='store_true', help='B60: one extra rep counting Python-level torch calls')
    p.add_argument('--variants', default='',
                   help='B60: n<0|1>p<0|1>b<N>[i<us>],... (native, pipeline, busy threads, switch interval)')
    a = p.parse_args()
    sys.setswitchinterval(5e-4)   # (B60: train_tok's --switch-interval default)
    dev = torch.device('cuda')
    from isaac_bridge import block_compile
    block_compile.configure()
    t = time.perf_counter()
    ck, store, chunks, calls, extra = learner_dump.load(a.path)
    lens = [len(c.rows) for c in chunks]
    info = dict(calls=len(calls), grad_calls=sum(c['grad'] for c in calls), chunks=len(chunks),
                rows_per_call=[int(sum(len(x.rows) for x in c['chunks'])) for c in calls],
                chunk_len_hist=np.bincount(lens).tolist(), load_s=round(time.perf_counter()-t, 1),
                torch=torch.__version__)
    print(json.dumps(info), flush=True)
    if a.equiv:
        out = dict(info=info, equiv=equiv(ck, calls, dev, a.calls, a.fp32))
    else:
        model, opt = load_model(ck, dev)
        reps = []
        # B60 --variants n<0|1>p<0|1>b<N>,...: host packing in C (n), pipelined preparations (p), N busy threads (b),
        # each a.reps reps in one process (the first rep of the first variant compiles)
        # (an optional i<microseconds> suffix: sys.setswitchinterval for that variant; default 500 us, train_tok's
        # --switch-interval default)
        def parse(v):
            b_, _, i_ = v[5:].partition('i')
            return int(v[1]), int(v[3]), int(b_), int(i_ or 500)
        variants = [parse(v) for v in a.variants.split(',')] if a.variants else \
            [(int(block_learner.NATIVE_PACK), int(a.pipeline), a.busy, 500)]
        for native_v, pipe_v, busy_v, sw_v in variants:
            block_learner.NATIVE_PACK = bool(native_v)
            sys.setswitchinterval(sw_v*1e-6)
            tag = f'n{native_v}p{pipe_v}b{busy_v}i{sw_v}'
            for i in range(a.reps):
                prof = None
                if a.cprofile and i == a.reps-1:
                    import cProfile
                    prof = cProfile.Profile(); prof.enable()
                done = busy_threads(busy_v) if busy_v else None
                r = replay(model, opt, calls, a.mode == 'packed', seed=i, pipeline=bool(pipe_v))
                if done is not None:
                    r['busy_spins'] = done()
                if prof is not None:
                    import pstats, io
                    prof.disable()
                    for key in ('tottime', 'cumulative'):
                        buf = io.StringIO()
                        pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(35)
                        print(buf.getvalue(), flush=True)
                r['variant'] = tag
                print(json.dumps(dict(rep=i, mode=a.mode, **r)), flush=True)
                reps.append(r)
        if a.opcount:
            ops = OpCount()
            with ops:
                r = replay(model, opt, calls, a.mode == 'packed', seed=a.reps, pipeline=a.pipeline, ops=ops)
            r['torch_calls'] = ops.n
            r['torch_calls_per_grad_call'] = {k: round(v/max(r['grad_calls'], 1), 1) for k, v in ops.n.items()}
            print(json.dumps(dict(rep='opcount', mode=a.mode, **r)), flush=True)
            reps.append(dict(r, opcount=True))
        from isaac_bridge import learn_native
        lib = learn_native.load()
        out = dict(info=info, mode=a.mode, reps=reps, pipeline=a.pipeline, busy=a.busy,
                   native=bool(lib is not None and block_learner.NATIVE_PACK),
                   env={k: v for k, v in os.environ.items() if k.startswith('ISAAC_')})
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
