"""B52 condition 6: the learner's long kernels, on one real update's dumped inputs, under the torch profiler.

  python b52_learner_kernels.py <learner_inputs.pt> <out dir> [--warm 1]

In train_tok the torch profiler runs on the learner thread (ISAAC_RL_B38_TORCHPROF); on the 4090 (torch 2.7.1+cu118)
kineto refused that ("External init callback must run in same thread as registerClient") and the run hung (prof-b52-kern,
2026-11-10). Here the update is replayed in the main thread as b44_learner_bench.replay (the recorded forward_bucketed
calls in order, compiled Spatial, packed learner, bf16 autocast; a grad call = forward + fixed surrogate loss + backward
+ clip + Adam; every segment GPU-synchronised, so no actor kernel shares the GPU): --warm reps first (compile, caches),
then one rep under torch.profiler with record_function regions fwd_grad / bwd / opt / fwd_nograd. Writes trace.json and
ops.txt into <out dir>; bridge/ops/b52_kernels.py lists the kernels over a threshold by region and name.
Measurement only; the model and its arithmetic are unchanged."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('ISAAC_BLOCK_TIMING', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
sys.path.insert(0, HERE)
import torch

from isaac_bridge import block_learner, block_paged as bp, learner_dump
from isaac_bridge.block_training import forward_bucketed
from b44_learner_bench import surrogate, sync, fresh, load_model, replay


def replay_marked(model, opt, calls):
    from torch.profiler import record_function
    block_learner.PACKED = True
    fresh(calls)
    bp.UPD.clear()
    torch.manual_seed(0)
    cache = {}
    for c in calls:
        kw = dict(c['kwargs'])
        if c['grad']:
            model.train()
            with record_function('fwd_grad'), torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            sync()
            with record_function('bwd'):
                loss = surrogate(out, cache)
                loss.backward()
            sync()
            with record_function('opt'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step(); opt.zero_grad(set_to_none=True)
            sync()
            del out, ctx, loss
        else:
            mode = torch.inference_mode() if c['inference'] else torch.no_grad()
            with record_function('fwd_nograd'), mode, torch.autocast('cuda', dtype=torch.bfloat16):
                out, ctx = forward_bucketed(model, c['chunks'], **kw)
            sync()
            del out, ctx
    bp.GpuSpan.drain()
    bp.clear_decode_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('out')
    p.add_argument('--warm', type=int, default=1)
    a = p.parse_args()
    dev = torch.device('cuda')
    from isaac_bridge import block_compile
    block_compile.configure()
    block_learner.PACKED = True
    ck, store, chunks, calls, extra = learner_dump.load(a.path)
    model, opt = load_model(ck, dev)
    for i in range(a.warm):
        r = replay(model, opt, calls, True, seed=100+i)
        print('warm', i, {k: r[k] for k in ('fwd_grad', 'bwd', 'opt', 'fwd_nograd', 'total')}, flush=True)
    from torch.profiler import ProfilerActivity, profile
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        replay_marked(model, opt, calls)
    print(f'profiled rep: {sync()-t0:.2f} s wall (profiler on), {len(calls)} calls', flush=True)
    ka = prof.key_averages()
    with open(out / 'ops.txt', 'w') as f:
        f.write(ka.table(sort_by='self_cuda_time_total', row_limit=50))
    prof.export_chrome_trace(str(out / 'trace.json'))
    print('written', out, flush=True)


if __name__ == '__main__':
    main()
