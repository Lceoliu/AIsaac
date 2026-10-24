"""Source-only BF16: independent dense reference, FP32 drift, old-graph credit."""
import argparse
import copy
import json
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter

p = argparse.ArgumentParser()
p.add_argument('--learner', default='cpu')
p.add_argument('--sources', default='cpu,cpu')
p.add_argument('--queries', type=int, default=8)
p.add_argument('--compile-source',nargs='?',const=1,type=int,default=0,choices=(0,1,2))
p.add_argument('--flash-source',action='store_true')
p.add_argument('--saved-gpu-gb', type=float, default=0)
p.add_argument('--component-graph-cache',action='store_true')
p.add_argument('--saved-8bit',type=int,default=0)
p.add_argument('--identity-hooks',action='store_true')
a = p.parse_args()
if a.identity_hooks:
    from isaac_bridge.saved_activation import SavedActivation8bit
    SavedActivation8bit.pack=lambda self,t:t.detach()
    SavedActivation8bit.unpack=lambda self,t:t
if a.component_graph_cache:os.environ['ISAAC_SOURCE_COMPONENT_GRAPH_CACHE']='1'
torch.set_num_threads(2); torch.manual_seed(71)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
           rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).to(a.learner).eval()
fp32 = copy.deepcopy(model); mixed = copy.deepcopy(model)
pipe = SourcePipeline(model, a.sources.split(','), period=1, micro=16, amp=True, saved_gpu_gb=a.saved_gpu_gb,
                      compile_source=a.compile_source,flash_source=a.flash_source,saved_8bit=a.saved_8bit)
opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
# Independent dense reference: only these two source operations autocast.
# Casting at the transport boundary remains differentiable in the reference.
encode, write = mixed.encode_rebuild, mixed.reader.write
def source_encode(b):
    b['_source_bf16'] = True
    with torch.autocast(torch.device(a.learner).type, dtype=torch.bfloat16):
        y, pad, u = encode(b)
    return y.float(), pad, u.float()
def source_write(x):
    with torch.autocast(torch.device(a.learner).type, dtype=torch.bfloat16):
        m = write(x.to(torch.bfloat16))
    return m.float()
mixed.encode_rebuild, mixed.reader.write = source_encode, source_write
rng = np.random.default_rng(71)
seq = TS.make_seq(rng, nw=2, T=70, changes={(0,8),(0,32),(1,8),(1,32)}, resets=set(), exits=set())
if a.component_graph_cache:
    for rows in seq[1:]:
        for field in ('grid','map','patch'):rows[field]=seq[0][field]
pending = [M.synth_pending(rng, 2, model.heads) for _ in seq]
hs = [RawHistory.empty(), RawHistory.empty()]
for t in range(63):
    for w in range(2): hs[w] = hs[w].append(seq[t][w], pending[t][w], 4)
req = [(hs[w], seq[63][w], pending[63][w]) for w in range(2)]
req = [req[i % 2] for i in range(a.queries)]
outputs, grads = {}, {}
early = None
for label, m in [('fp32', fp32), ('mixed_dense', mixed), ('pipeline', model)]:
    bank = SourceBatch(m, req, micro=16, cache=False, pipeline=pipe if label == 'pipeline' else None)
    if label == 'pipeline':
        early = pipe.chunks[0]['future'].result()['u']; early.retain_grad()
        assert early.dtype == torch.bfloat16
        assert all(v.dtype == torch.float32 for v in (bank.u, bank.y, bank.mem, *bank.kvs))
    out = bank.forward(np.arange(len(req)), need_vg=True)
    outputs[label] = torch.cat([out[0].flatten(),out[1].flatten(),out[-1].flatten()]).detach()
    loss = out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()
    loss.backward()
    if label == 'pipeline':
        bank.backward()
        for chunk in pipe.chunks+pipe.writes:
            assert all(v.dtype == torch.float32 for v in chunk['future'].result()['grads'].values())
        pipe.fast_step(opt, 1e9); pipe.finish(opt, 1e9)
    grads[label] = {n:v.grad.detach().clone() for n,v in m.named_parameters()
                    if source_parameter(n) and v.grad is not None}
metrics = {}
missing={label:sorted(set(grads[label])-set(grads['pipeline'])) for label in ('mixed_dense','fp32')}
assert not any(missing.values()), missing
for label in ('mixed_dense','fp32'):
    d = outputs['pipeline']-outputs[label]
    ga = torch.cat([grads['pipeline'][n].flatten() for n in grads[label]]).double()
    gb = torch.cat([grads[label][n].flatten() for n in grads[label]]).double()
    metrics[label] = dict(output_max_abs=float(d.abs().max()),
                         output_relative_l2=float(d.norm()/outputs[label].norm()),
                         source_grad_max_abs=float((ga-gb).abs().max()),
                         source_grad_relative_l2=float((ga-gb).norm()/gb.norm()),
                         source_grad_cosine=float(torch.nn.functional.cosine_similarity(ga,gb,dim=0)))
    # Declared BF16 acceptance limits, not FP32 exact-equivalence tolerances.
    assert metrics[label]['output_relative_l2'] < .03, metrics
    assert metrics[label]['source_grad_relative_l2'] < .10, metrics
if a.saved_8bit:
    assert metrics['mixed_dense']['source_grad_relative_l2'] < .02, metrics
    assert metrics['mixed_dense']['source_grad_cosine'] > .999, metrics
assert early.grad[0].abs().sum() > 0
initial_components={}
if a.component_graph_cache:
    for replicas in pipe.versions.values():
        for replica in replicas:
            if hasattr(replica,'_components'):
                for name,(hit,total) in replica._components.calls.items():
                    initial_components[name]=initial_components.get(name,0)+total-hit
    assert 0<initial_components['grid']<=4, initial_components
    assert 0<initial_components['map']<=4, initial_components
encoded = [pipe.encoded_rows]
for t in range(64,70):
    for w in range(2): hs[w] = hs[w].append(seq[t-1][w],pending[t-1][w],4)
    req = [(hs[w],seq[t][w],pending[t][w]) for w in range(2)]
    req = [req[i%2] for i in range(a.queries)]
    model.zero_grad(); early.grad = None
    with torch.no_grad(): resident = SourceBatch(model,req,cache=False,pipeline=pipe)
    bank = SourceView(resident,np.arange(len(req)))
    out = bank.forward(np.arange(len(req)),need_vg=True)
    (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward();bank.backward()
    pipe.fast_step(opt,1e9);pipe.finish(opt,1e9)
    assert early.grad[0].abs().sum() > 0, 'old room lost gradient across source cycles'
    before = pipe.encoded_rows
    with torch.no_grad(): SourceBatch(model,req,cache=False,pipeline=pipe)
    assert pipe.encoded_rows == before
    encoded.append(before)
assert encoded == [128,130,132,134,136,138,140], encoded
assert all(v.dtype == torch.float32 for v in model.parameters())
for state in opt.state.values():
    assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
print(json.dumps(dict(result='PASS',source_precision='bf16',reader_precision='fp32',optimizer='fp32',
      saved_8bit_block=a.saved_8bit,
      queries=a.queries,metrics=metrics,encoded_rows_by_cycle=encoded,versions=pipe.version,
      old_room_gradient=float(early.grad[0].abs().sum()),encoder_replay=0,
      initial_unique_components=initial_components,
      offloaded_encode_calls=sum(e.get('saved_on_cpu',False) for e in pipe.events)),indent=2))
if a.component_graph_cache:
    pipe.retain_histories(set())
    assert not pipe.frames
    assert not any(getattr(replica,'_components',None) and replica._components.values
                   for replicas in pipe.versions.values() for replica in replicas)
pipe.close()
