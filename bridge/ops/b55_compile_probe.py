"""B55: where the ~610 s compiled-Spatial warm-up goes, and whether the FX graph / AOTAutograd caches help.

  python b55_compile_probe.py <checkpoint> <out.json> --fx-cache on|off [--steady]

Builds the actor and learner models of a block-memory checkpoint (as eval_tok / train_tok do), runs the production
warm-up (block_compile.warm_spatial, then block_learner.warm_reader) and records:
  - wall time of each warm-up,
  - torch._dynamo.utils.compilation_time_metrics (per timed phase: total s, calls),
  - torch._dynamo.utils.counters (fxgraph_cache_hit/miss/bypass, autograd_cache_*), block_compile.STATS.
--fx-cache on enables Inductor's FX graph cache and the AOTAutograd cache for this process only (block_compile's
default, off, is not changed); the cache directory is the caller's TORCHINDUCTOR_CACHE_DIR.
--steady (after mark_steady): the compiled Spatial / decoder graphs on batch sizes the warm-up did not use
(actor (entity, body) buckets with active and frozen parameters, learner row-packed T buckets in the three modes,
canvas decoders); block_compile.STATS['steady_compiles'] must stay 0 (the B42 failure: a cached graph carrying
the warm-up's batch-range guards recompiled for other sizes). Measurement only; no training."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np  # noqa: E402
import torch  # noqa: E402


def metrics():
    from torch._dynamo.utils import compilation_time_metrics, counters
    t = {k: dict(s=round(float(sum(v)), 2), n=len(v)) for k, v in compilation_time_metrics.items()}
    c = {k: dict(v) for k, v in counters.items() if v}
    return dict(time=dict(sorted(t.items(), key=lambda kv: -kv[1]['s'])), counters=c)


def steady(learner, model, bf16=True):
    from isaac_bridge import block_compile
    from isaac_bridge.block_paged import SPATIAL_PACKED
    from isaac_bridge.structured_input import BUCKETS, T_BUCKETS, canvas_decoder, packed_rows_batch
    dev = next(learner.parameters()).device
    out = []

    def note(what):
        out.append((what, block_compile.STATS['steady_compiles']))
    with torch.inference_mode(False), torch.no_grad(), torch.autocast('cuda', enabled=False):
        for p in BUCKETS['body']:
            for n in (5, 16, 40):
                canvas_decoder()(torch.zeros((n, 1792), device=dev), torch.zeros((n, p, 324), device=dev),
                                 torch.zeros((n, 169), device=dev))
    note('canvas decoders n 5/16/40')
    params = list(model.parameters())
    flags = [q.requires_grad for q in params]
    for frozen in (False, True):
        for q, f in zip(params, flags):
            q.requires_grad_(f and not frozen)
        for p in BUCKETS['body']:
            for e in BUCKETS['entity']:
                for n in (5, 16, 40):
                    b = block_compile.synthetic_visible(n, p, e, dev)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                        model.structured_spatial(b)
        note(f'actor spatial frozen={frozen} n 5/16/40')
    for q, f in zip(params, flags):
        q.requires_grad_(f)
    learner.train()
    if SPATIAL_PACKED:
        for T in T_BUCKETS:
            inv = max(2, T-30-2-2-2-128) if T > 164 else 2
            ent = min(128, T-30-2-2-inv)
            for n in (5, 37, max(8, min(700, 120000//T))):
                b = block_compile.synthetic_visible(n, 1, 128, dev, inventory=max(32, inv), slots=16)
                counts = np.array([[1, 1, ent, inv, 2]]*n)
                for mode in ('grad', 'no_grad', 'inference'):
                    ctx = torch.inference_mode() if mode == 'inference' else torch.set_grad_enabled(mode == 'grad')
                    with ctx, torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                        part = packed_rows_batch(b, np.arange(n), counts, dev)
                        z = learner.encode_block(part)
                    if mode == 'grad':
                        z.float().sum().backward()
            note(f'learner rows T={T}')
    learner.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    ap.add_argument('--fx-cache', choices=('on', 'off'), required=True)
    ap.add_argument('--steady', action='store_true')
    a = ap.parse_args()
    from isaac_bridge import block_compile, block_learner
    block_compile.configure()
    if a.fx_cache == 'on':
        import torch._functorch.config as functorch_config
        import torch._inductor.config as inductor_config
        inductor_config.fx_graph_cache = True
        functorch_config.enable_autograd_cache = True
    dev = torch.device('cuda')
    from eval_tok import build_eval_model
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model, _ = build_eval_model(ck, True, dev)
    learner, _ = build_eval_model(ck, True, dev)
    learner.train()
    rec = dict(checkpoint=a.checkpoint, fx_cache=a.fx_cache, torch=torch.__version__,
               inductor_cache_dir=os.environ.get('TORCHINDUCTOR_CACHE_DIR'), triton_cache_dir=os.environ.get('TRITON_CACHE_DIR'))
    t0 = time.perf_counter()
    rec['spatial_graphs'] = block_compile.warm_spatial(learner, (model,), bf16=True)
    rec['spatial_s'] = round(time.perf_counter()-t0, 1)
    print(f"B55 spatial warm-up: {rec['spatial_graphs']} graphs in {rec['spatial_s']} s", flush=True)
    rec['spatial'] = metrics()
    c0 = block_compile.STATS['compiles']
    rec['reader_s'] = round(block_learner.warm_reader(learner, bf16=True), 1)
    rec['reader_graphs'] = block_compile.STATS['compiles']-c0
    print(f"B55 Flex Reader warm-up: {rec['reader_graphs']} graphs in {rec['reader_s']} s", flush=True)
    rec['after_reader'] = metrics()
    block_compile.mark_steady()
    if a.steady:
        t1 = time.perf_counter()
        rec['steady'] = steady(learner, model)
        rec['steady_s'] = round(time.perf_counter()-t1, 1)
        print('B55 steady:', rec['steady'], flush=True)
    rec['stats'] = dict(block_compile.STATS)
    with open(a.out, 'w') as f:
        json.dump(rec, f, indent=1)
    print('B55 STATS', rec['stats'], flush=True)


if __name__ == '__main__':
    main()
