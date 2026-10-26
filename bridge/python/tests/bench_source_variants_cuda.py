"""Source-side variants, measured (2026-10-26; measurement only, no decision) — CUDA.

The same weights and records through one source pipeline per variant (compiled sources, BF16 source precision
as in the long runs unless noted), one interval each (encode, Reader step, VJP, Adam):
  full        varlen flash attention, full graphs (the reference of the checkpoint variants)
  ckpt_all    --source-layer-checkpoint 1 (every Spatial block recomputed in the VJP)
  ckpt_part   --source-checkpoint-blocks (default 2,3: two of the blocks; all blocks save the same shapes)
  noflash     --source-flash 0: bucketed attention, full graphs
  fp32        --source-flash 0, fp32 sources: the numerical reference of flash / noflash
Per variant: saved-tensor bytes per frame (census of the encodes), bytes per frame left allocated after the
encode, encode seconds, VJP seconds (second interval: compilation excluded), source-gradient relative error
against `full` and against `fp32`.
  python tests/bench_source_variants_cuda.py [--device cuda:0] [--frames 1024] [--width 192] [--layers 4]
"""
import argparse, contextlib, copy, io, json, os, sys, time
from pathlib import Path
os.environ.update(ISAAC_SOURCE_POINTWISE_RECOMPUTE='1', ISAAC_SOURCE_POINTWISE_COMPONENTS='1',
                  ISAAC_SOURCE_COMPONENT_STATIC='1', ISAAC_SOURCE_SAVED_PROFILE='1')
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda:0')
p.add_argument('--frames', type=int, default=1024)
p.add_argument('--width', type=int, default=192)
p.add_argument('--layers', type=int, default=4)
p.add_argument('--blocks', default='2,3')
a = p.parse_args()
torch.manual_seed(79)
cfg = dict(width=a.width, layers=a.layers, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=64,
           temporal_layers=2, temporal_heads=4, temporal_window=8, rebuild_memory=True, memory_block=8, memory_tokens=2,
           source_cache=True)
base = TokPolicy(**cfg).to(a.device).eval()
rng = np.random.default_rng(79)
W = 8; T = a.frames // W


def requests():
    seq = TS.make_seq(rng, nw=W, T=T + 1, changes={(w, t) for w in range(W) for t in range(5, T, 11)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
    hs = [RawHistory.empty() for _ in range(W)]; req = []
    for t in range(T):
        for w in range(W):
            req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 8)
    return req


sets = [requests(), requests()]   # (interval 1: compilation; interval 2: measured)
VARIANTS = dict(full=dict(flash_source=True), ckpt_all=dict(flash_source=True, layer_checkpoint=True),
                ckpt_part=dict(flash_source=True, layer_checkpoint=True, checkpoint_blocks=[int(b) for b in a.blocks.split(',')]),
                noflash=dict(flash_source=False), fp32=dict(flash_source=False, amp=False))
results = {}
for name, kw in VARIANTS.items():
    model = copy.deepcopy(base)
    pipe = SourcePipeline(model, [a.device], period=0, micro=256, update_rollouts=1, compile_source=1,
                          **dict(dict(amp=True), **kw))
    opt = torch.optim.SGD(model.main_parameters(), lr=0.0)   # (lr 0: both intervals see the same weights)
    for k, req in enumerate(sets):
        torch.cuda.synchronize(a.device); before = torch.cuda.memory_allocated(a.device); t0 = time.perf_counter()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            torch.cuda.synchronize(a.device)
        encode_s = time.perf_counter() - t0
        retained = torch.cuda.memory_allocated(a.device) - before
        census = [json.loads(l.split(' ', 1)[1]) for l in log.getvalue().splitlines() if l.startswith('SOURCE_SAVED ')]
        view = SourceView(bank, np.arange(len(req)))
        out = view.forward(np.arange(len(req)), need_vg=True)
        (out[0].float().square().mean() + out[1].float().square().mean()).backward()
        view.backward(); pipe.fast_step(opt, 1e9)
        torch.cuda.synchronize(a.device); t0 = time.perf_counter()
        pipe.finish_rollout(opt, 1e9, overlap=False)
        torch.cuda.synchronize(a.device); vjp_s = time.perf_counter() - t0
        grads = {n: q.grad.detach().double().clone() for n, q in model.named_parameters() if source_parameter(n) and q.grad is not None}
        pipe.retain_histories(set())   # (the next interval's frames only)
        del bank, view, out
    results[name] = dict(grads=grads, saved=sum(c['bytes'] for c in census) / max(1, sum(c['rows'] for c in census)),
                         retained=retained / len(sets[1]), encode_s=encode_s, vjp_s=vjp_s)
    pipe.close()


def rel(x, y):
    common = sorted(set(x['grads']) & set(y['grads']))
    gx = torch.cat([x['grads'][n].flatten() for n in common]); gy = torch.cat([y['grads'][n].flatten() for n in common])
    return float((gx - gy).norm() / gy.norm())


table = {name: dict(saved_KiB_per_frame=r['saved'] / 1024, retained_KiB_per_frame=r['retained'] / 1024,
                    encode_s=r['encode_s'], vjp_s=r['vjp_s'], rel_err_vs_full=rel(r, results['full']),
                    rel_err_vs_fp32=rel(r, results['fp32'])) for name, r in results.items()}
print('SOURCE_VARIANTS ' + json.dumps(dict(frames=len(sets[1]), width=a.width, layers=a.layers, blocks=a.blocks, variants=table)))
for name, r in table.items():
    print(f'{name:10s} saved {r["saved_KiB_per_frame"]:7.1f} KiB/f  retained {r["retained_KiB_per_frame"]:7.1f} KiB/f  '
          f'encode {r["encode_s"]:.3f} s  vjp {r["vjp_s"]:.3f} s  err/full {r["rel_err_vs_full"]:.2e}  err/fp32 {r["rel_err_vs_fp32"]:.2e}')
