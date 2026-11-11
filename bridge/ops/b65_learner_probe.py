"""BM1 learner probes (offline, fixed input; production paths untouched): a dumped update's PPO parts (the grad calls
of <dir>/learner_inputs.pt, isaac_bridge/learner_dump.py) on the learner GPU(s).

  python b65_learner_probe.py <learner_inputs.pt> [--parts N] [--reps 3] [--json out.json]
  visible GPUs: cuda:0 the learner card; cuda:1 (optional) a second card for P2.

Per grad part, its host preparation (history draws, decodes, the packed slab) is made once (block_learner.
_prepare_packed) and reused, so every variant runs the same tensors; the loss is b44_learner_bench's fixed surrogate.
  eager    the part's forward (block_learner._run_packed under bf16 autocast) and backward, host time and the learner
           stream's span (events at the boundaries; one synchronisation after the part), as learn_segments measures.
  P1       the same forward + backward captured into one CUDA graph per part (thread_local capture mode, one memory pool
           for every part's graph, .grad preallocated so the backward accumulates into fixed tensors) and replayed:
           replay time; the gradients against eager's (max relative difference over all parameters: max|a-b|/max|a|);
           the extra GPU memory the graphs hold; the part's shapes (are they the same across parts?). A capture that
           fails is reported with its error (the blocker), not fixed.
  P2       every grad part forward + backward, all on cuda:0 (one thread) against split over cuda:0 and cuda:1 (two
           threads, a model copy each, parts alternating): wall seconds; eager, and replay when P1's capture worked on
           both cards. No optimizer step anywhere (the probes measure forward / backward only).
Prints one JSON line per measurement and a summary."""
import argparse
import inspect
import json
import os
import sys
import threading
import time
import traceback

os.environ.setdefault('ISAAC_BLOCK_TIMING', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch

from isaac_bridge import block_learner, block_paged as bp, learner_dump
from isaac_bridge.tok_policy import TokPolicy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b44_learner_bench import surrogate, fresh   # noqa: E402

PREP_KEYS = [k for k in inspect.signature(block_learner._prepare_packed).parameters if k not in ('model', 'chunks', 'grad')]


def emit(**kw):
    print(json.dumps(kw, default=str), flush=True)


def load_model(ck, dev):
    m = TokPolicy(**ck['config']).to(dev)
    m.load_state_dict(ck['model'])
    m.train()
    return m


def prep(model, call):
    kw = {k: v for k, v in call['kwargs'].items() if k in PREP_KEYS}
    with block_learner.PREP_LOCK:
        return block_learner._prepare_packed(model, call['chunks'], True, **kw)


def params(model):
    return [p for p in model.parameters() if p.requires_grad]


def zero_grads(model):
    for p in params(model):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        else:
            p.grad.zero_()


def fwd_bwd(model, P, cache):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, ctx = block_learner._run_packed(model, P)
    loss = surrogate(out, cache)
    loss.backward()
    return loss


def shapes(P):
    sl = P.get('slab')
    return dict(slab=int(getattr(sl, 'total', 0) or 0), **{k: (list(v.shape) if hasattr(v, 'shape') else v)
                                                          for k, v in P.items() if k in ('n_main', 'n_hist', 'n_new', 'n_tail', 'n_old_total')})


def eager_part(model, P, cache):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    t0 = time.perf_counter(); ev[0].record()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, ctx = block_learner._run_packed(model, P)
    loss = surrogate(out, cache)
    t1 = time.perf_counter(); ev[1].record()
    loss.backward()
    t2 = time.perf_counter(); ev[2].record()
    torch.cuda.synchronize()
    return dict(fwd_host=t1-t0, fwd_span=ev[0].elapsed_time(ev[1])/1e3, bwd_host=t2-t1, bwd_span=ev[1].elapsed_time(ev[2])/1e3)


def grads(model):
    return [p.grad.detach().float().clone() for p in params(model)]


def rel_diff(a, b):
    num = max(float((x-y).abs().max()) for x, y in zip(a, b))
    den = max(float(x.abs().max()) for x in a)
    return num/max(den, 1e-30)


def capture(model, P, cache, pool):
    """The part's forward + backward as one CUDA graph (thread_local), warm-up on a side stream first."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            zero_grads(model)
            fwd_bwd(model, P, cache)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    zero_grads(model)
    torch.clear_autocast_cache()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool, stream=s, capture_error_mode='thread_local'):
        fwd_bwd(model, P, cache)
    torch.cuda.synchronize()
    return g


def main():
    a = argparse.ArgumentParser()
    a.add_argument('path')
    a.add_argument('--parts', type=int, default=0, help='first N grad parts (0: all)')
    a.add_argument('--reps', type=int, default=3)
    a.add_argument('--json', default='')
    args = a.parse_args()
    from isaac_bridge import block_compile
    block_compile.configure()
    ck, store, chunks, calls, extra = learner_dump.load(args.path)
    gcalls = [c for c in calls if c['grad']]
    if args.parts:
        gcalls = gcalls[:args.parts]
    ndev = torch.cuda.device_count()
    emit(stage='load', calls=len(calls), grad_parts=len(gcalls), devices=ndev, torch=torch.__version__)
    fresh(calls)
    torch.manual_seed(0)
    dev0 = torch.device('cuda', 0)
    torch.cuda.set_device(0)
    model = load_model(ck, dev0)
    cache = {}
    Ps = [prep(model, c) for c in gcalls]
    emit(stage='shapes', parts=[shapes(P) for P in Ps])
    # eager: warm-up (compiles) then reps; per part the median over reps
    res = {'eager': [], 'p1': [], 'p2': {}}
    for r in range(args.reps+1):
        rows = []
        for P in Ps:
            zero_grads(model)
            rows.append(eager_part(model, P, cache))
        if r > 0:
            res['eager'].append(rows)
    tot = lambda rows, k: sum(x[k] for x in rows)   # noqa: E731
    last = res['eager'][-1]
    eager_sum = {k: round(tot(last, k), 3) for k in ('fwd_host', 'fwd_span', 'bwd_host', 'bwd_span')}
    emit(stage='eager', per_update=eager_sum, parts=len(Ps))
    # eager reference gradients per part
    ref = []
    for P in Ps:
        zero_grads(model)
        fwd_bwd(model, P, cache)
        torch.cuda.synchronize()
        ref.append(grads(model))
    # P1: capture every part (one pool), replay
    pool = torch.cuda.graph_pool_handle()
    m0 = torch.cuda.memory_reserved(0)
    graphs, blockers = [], []
    for i, P in enumerate(Ps):
        try:
            graphs.append(capture(model, P, cache, pool))
        except BaseException as e:
            tb = traceback.format_exc().strip().splitlines()
            blockers.append(dict(part=i, error=f'{type(e).__name__}: {e}'[:500], where=[l for l in tb if 'File' in l][-4:]))
            graphs.append(None)
            torch.cuda.synchronize()
    m1 = torch.cuda.memory_reserved(0)
    emit(stage='p1_capture', captured=sum(g is not None for g in graphs), parts=len(Ps), extra_reserved_gb=round((m1-m0)/2**30, 2),
         blockers=blockers[:5])
    if all(g is not None for g in graphs):
        diffs = []
        for g, rg in zip(graphs, ref):
            zero_grads(model)
            g.replay()
            torch.cuda.synchronize()
            diffs.append(rel_diff(rg, grads(model)))
        times = []
        for r in range(args.reps):
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); ev0.record()
            for g in graphs:
                g.replay()
            t1 = time.perf_counter(); ev1.record(); torch.cuda.synchronize()
            times.append((t1-t0, ev0.elapsed_time(ev1)/1e3))
        emit(stage='p1_replay', per_update_host=round(times[-1][0], 3), per_update_span=round(times[-1][1], 3),
             grad_max_rel_diff=max(diffs), grad_rel_diff_per_part=[float(f'{d:.3g}') for d in diffs])
        res['p1'] = dict(host=times[-1][0], span=times[-1][1], diff=max(diffs))
    # P2: one card vs two cards (two threads)
    if ndev >= 2:
        torch.cuda.set_device(1)
        model1 = load_model(ck, torch.device('cuda', 1))
        cache1 = {}
        Ps1 = [prep(model1, c) for c in gcalls]
        for P in Ps1:   # (warm-up: compiles on the second card)
            zero_grads(model1); fwd_bwd(model1, P, cache1)
        torch.cuda.synchronize(1)
        torch.cuda.set_device(0)

        def run(dev, mdl, plist, cch):
            torch.cuda.set_device(dev)
            for P in plist:
                fwd_bwd(mdl, P, cch)
            torch.cuda.synchronize(dev)
        zero_grads(model)
        t0 = time.perf_counter(); run(0, model, Ps, cache); one = time.perf_counter()-t0
        zero_grads(model); zero_grads(model1)
        th = [threading.Thread(target=run, args=(0, model, Ps[0::2], cache)),
              threading.Thread(target=run, args=(1, model1, Ps1[1::2], cache1))]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        two = time.perf_counter()-t0
        emit(stage='p2_eager', one_card_s=round(one, 3), two_cards_s=round(two, 3))
        res['p2']['eager'] = (one, two)
        if res['p1']:
            pool1 = torch.cuda.graph_pool_handle(device=1) if 'device' in inspect.signature(torch.cuda.graph_pool_handle).parameters else None
            torch.cuda.set_device(1)
            try:
                g1 = [capture(model1, P, cache1, pool1) for P in Ps1[1::2]]
                torch.cuda.set_device(0)

                def rep(dev, gl):
                    torch.cuda.set_device(dev)
                    for g in gl:
                        g.replay()
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter(); rep(0, graphs); one_g = time.perf_counter()-t0
                th = [threading.Thread(target=rep, args=(0, graphs[0::2])), threading.Thread(target=rep, args=(1, g1))]
                t0 = time.perf_counter()
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                two_g = time.perf_counter()-t0
                emit(stage='p2_replay', one_card_s=round(one_g, 3), two_cards_s=round(two_g, 3))
                res['p2']['replay'] = (one_g, two_g)
            except BaseException as e:
                emit(stage='p2_replay', error=f'{type(e).__name__}: {e}'[:400])
    emit(stage='summary', eager=eager_sum, p1=res['p1'], p2=res['p2'], blockers=len(blockers))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, default=str)


if __name__ == '__main__':
    main()
