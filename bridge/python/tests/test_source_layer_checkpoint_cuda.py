"""--source-layer-checkpoint (CUDA): the Spatial blocks under activation checkpointing give the full graph's
source gradients, and save less.

Two source pipelines on the same weights and records (compiled sources, varlen flash attention, BF16 source
precision as in the long runs): one with the blocks' full graphs, one with layer checkpointing. One interval
each (encode, Reader steps, VJP, Adam). Gradient relative error, parameter difference, the saved bytes per
frame (saved-tensor hooks around one encode) and the VJP seconds are printed; the gradients must agree to
floating point (relative error < 1e-2 under BF16 recomputation, observed values in the output).
  python tests/test_source_layer_checkpoint_cuda.py [--device cuda:0] [--frames 512]
"""
import argparse, contextlib, copy, io, json, os, sys, time
from pathlib import Path
# the long runs' source settings (read when a pipeline is built); the saved-tensor census of every encode
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
p.add_argument('--frames', type=int, default=256)
a = p.parse_args()
torch.manual_seed(77)
cfg = dict(width=64, layers=2, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=64, temporal_layers=2,
           temporal_heads=4, temporal_window=8, rebuild_memory=True, memory_block=8, memory_tokens=2, source_cache=True)
base = TokPolicy(**cfg).to(a.device).eval()
rng = np.random.default_rng(77)
W = 4; T = a.frames // W
seq = TS.make_seq(rng, nw=W, T=T + 1, changes={(w, t) for w in range(W) for t in range(5, T, 11)}, resets=set(), exits=set())
pend = [M.synth_pending(rng, W, base.heads) for _ in seq]
hs = [RawHistory.empty() for _ in range(W)]; req = []
for t in range(T):
    for w in range(W):
        req.append((hs[w], seq[t][w], pend[t][w])); hs[w] = hs[w].append(seq[t][w], pend[t][w], 8)

results = {}
for name, ckpt in (('full', False), ('checkpoint', True)):
    model = copy.deepcopy(base)
    pipe = SourcePipeline(model, [a.device], period=0, micro=64, update_rollouts=1, amp=True, compile_source=1,
                          flash_source=True, layer_checkpoint=ckpt)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
    torch.cuda.synchronize(a.device); before = torch.cuda.memory_allocated(a.device)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):   # (the encoder threads' SOURCE_SAVED lines)
        with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
        torch.cuda.synchronize(a.device)
    retained = torch.cuda.memory_allocated(a.device) - before
    census = [json.loads(l.split(' ', 1)[1]) for l in log.getvalue().splitlines() if l.startswith('SOURCE_SAVED ')]
    saved_bytes = sum(c_['bytes'] for c_ in census); saved_rows = sum(c_['rows'] for c_ in census)
    view = SourceView(bank, np.arange(len(req)))
    out = view.forward(np.arange(len(req)), need_vg=True)
    (out[0].float().square().mean() + out[1].float().square().mean()).backward()
    view.backward(); pipe.fast_step(opt, 1e9)
    torch.cuda.synchronize(a.device); t0 = time.perf_counter()
    pipe.finish_rollout(opt, 1e9, overlap=False)        # the VJP and the source Adam step
    torch.cuda.synchronize(a.device); vjp_s = time.perf_counter() - t0
    grads = {n: q.grad.detach().double().clone() for n, q in model.named_parameters() if source_parameter(n) and q.grad is not None}
    results[name] = dict(params={n: q.detach().double().clone() for n, q in model.named_parameters()}, grads=grads,
                         saved_bytes=saved_bytes, saved_rows=saved_rows, retained=retained, vjp_s=vjp_s, frames=len(req))
    pipe.close()

f, c = results['full'], results['checkpoint']
common = sorted(set(f['grads']) & set(c['grads']))
assert common and set(f['grads']) == set(c['grads']), (sorted(f['grads']), sorted(c['grads']))
gf = torch.cat([f['grads'][n].flatten() for n in common]); gc = torch.cat([c['grads'][n].flatten() for n in common])
rel = float((gf - gc).norm() / gf.norm())
worst = max(float(((f['grads'][n] - c['grads'][n]).norm() / f['grads'][n].norm().clamp_min(1e-30))) for n in common)
dparam = max(float((f['params'][n] - c['params'][n]).abs().max()) for n in f['params'])
out = dict(frames=f['frames'], gradient_relative_error=rel, worst_tensor_relative_error=worst, max_param_difference=dparam,
           saved_bytes_per_frame=dict(full=f['saved_bytes'] / max(1, f['saved_rows']), checkpoint=c['saved_bytes'] / max(1, c['saved_rows'])),
           retained_bytes_per_frame=dict(full=f['retained'] / f['frames'], checkpoint=c['retained'] / c['frames']),
           vjp_seconds=dict(full=f['vjp_s'], checkpoint=c['vjp_s']))
assert rel < 1e-2, out
assert c['saved_bytes'] < f['saved_bytes'], out
print('PASS layer checkpoint: source gradients as the full graph', json.dumps(out))
