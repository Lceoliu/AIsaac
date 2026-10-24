"""Two-rate optimizer parity, cross-room credit and real multi-device execution."""
import argparse
import copy
import json
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView, JoinedSources
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter

p = argparse.ArgumentParser()
p.add_argument('--learner', default='cpu')
p.add_argument('--sources', default='cpu,cpu')
p.add_argument('--queries', type=int, default=5)
p.add_argument('--length', type=int, default=32)
p.add_argument('--source-micro',type=int,default=8)
p.add_argument('--reader-micro',type=int,default=0)
p.add_argument('--trace')
p.add_argument('--sparse', action='store_true')
p.add_argument('--resident', action='store_true')
p.add_argument('--rollout-clock', action='store_true')
p.add_argument('--overlap-rollout', action='store_true')
p.add_argument('--joined', action='store_true')
a = p.parse_args()
torch.set_num_threads(2); torch.manual_seed(71)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
           rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).to(a.learner).eval(); reference = copy.deepcopy(model)
source_reference = copy.deepcopy(model)
pipeline = SourcePipeline(model, a.sources.split(','), period=0 if a.rollout_clock else 2, micro=a.source_micro)
opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
fast = [p for n, p in reference.named_parameters() if not source_parameter(n)]
slow = [p for n, p in reference.named_parameters() if source_parameter(n)]
ref_fast = torch.optim.Adam(fast, lr=1e-4, eps=1e-5)
ref_slow = torch.optim.Adam(slow, lr=1e-4, eps=1e-5)
saved = {n: p.detach().clone() for n, p in model.named_parameters()}
rng = np.random.default_rng(71)
seq = TS.make_seq(rng, nw=2, T=a.length, changes={(0,8),(0,16),(1,8),(1,16)}, resets=set(), exits=set())
if a.sparse:
    for rows in seq:
        rows['n_ent'] = 0
        rows['n_doors'] = 0
pending = [M.synth_pending(rng, 2, model.heads) for _ in seq]
histories = [RawHistory.empty(), RawHistory.empty()]
requests = [[], []]
for t, rows in enumerate(seq):
    for w in range(2):
        if t in (a.length//2-5,a.length//2-1): requests[0].append((histories[w],rows[w],pending[t][w]))
        if t in (a.length-5,a.length-1): requests[1].append((histories[w],rows[w],pending[t][w]))
        histories[w] = histories[w].append(rows[w],pending[t][w],4)
requests = [[r[i % len(r)] for i in range(a.queries)] for r in requests]
if a.resident:
    with torch.no_grad():resident=SourceBatch(model,requests[0]+requests[1],cache=False,pipeline=pipeline)
    assert len(resident.u)==len(resident.u_frames)
    assert len(resident.u)<len(resident.frame_keys), 'unused direct-read frames were transferred'
slow_grads = {}; logits_error = 0.
prof = None
if a.trace:
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA])
    prof.__enter__()
for step, req in enumerate(requests):
    model.zero_grad(); reference.zero_grad()
    bank = (SourceView(resident,np.arange(len(req))+step*len(req)) if a.resident else
            SourceBatch(model, req, cache=False, pipeline=pipeline))
    if a.joined:
        cut=len(req)//2
        bank=JoinedSources(SourceView(resident,np.arange(cut)+step*len(req)),
                           SourceBatch(model,req[cut:],cache=False,pipeline=pipeline))
        assert not bank.cached
    if step == 0:
        early = pipeline.chunks[0]['future'].result()['u']; early.retain_grad()
        rows_before = pipeline.encoded_rows
        pipeline.prefetch(requests[1])
    chunks=[]
    for lo in range(0,len(req),a.reader_micro or len(req)):
        hi=min(lo+(a.reader_micro or len(req)),len(req))
        out = bank.forward(np.arange(lo,hi), need_vg=True)
        loss = out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()
        (loss*((hi-lo)/len(req))).backward(retain_graph=hi<len(req))
        chunks.append(tuple(value.detach() for value in out))
    out=tuple(torch.cat(values) for values in zip(*chunks))
    bank.backward()
    pipeline.fast_step(opt, 1e9)
    full = SourceBatch(reference, req, micro=8, cache=False)
    expected = full.forward(np.arange(len(req)), need_vg=True)
    logits_error = max(logits_error, float((out[0]-expected[0]).detach().abs().max()))
    torch.testing.assert_close(out[0], expected[0], rtol=5e-4, atol=3e-5)
    (expected[0].square().mean()+expected[1].square().mean()+expected[-1].square().mean()).backward()
    for name, param in reference.named_parameters():
        if source_parameter(name) and param.grad is not None:
            if name not in slow_grads: slow_grads[name] = param.grad.detach().clone()
            else: slow_grads[name].add_(param.grad)
    ref_fast.step()
    for name, param in model.named_parameters():
        if source_parameter(name): assert torch.equal(param, saved[name]), name
    assert pipeline.version == 0
assert pipeline.encoded_rows == 2*a.length, pipeline.encoded_rows
assert pipeline.reused_rows > 0
if a.overlap_rollout:
    assert a.rollout_clock
    pipeline.start_finish(opt,1e9)
    assert pipeline.version==0 and pipeline.pending is not None and pipeline.steps==0
    # Actor publication may use the old source while its packet is in flight.
    for name,param in model.named_parameters():
        if source_parameter(name):torch.testing.assert_close(param,saved[name],rtol=0,atol=0)
    # This is also the checkpoint fence: it must commit, not discard, the packet.
    pipeline.finish(opt,1e9)
    assert pipeline.pending is None
else:
    pipeline.finish(opt, 1e9)
for name, param in reference.named_parameters():
    if source_parameter(name): param.grad = slow_grads[name]/2 if name in slow_grads else None
ref_slow.step()
grad_error = weight_error = 0.
for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
    if source_parameter(name) and name in slow_grads:
        grad_error = max(grad_error, float((actual.grad-expected.grad).abs().max()))
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-3, atol=5e-5, msg=name)
    weight_error = max(weight_error, float((actual-expected).detach().abs().max()))
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=5e-5, msg=name)
assert early.grad[0].abs().sum() > 0, 'earliest room source lost its gradient'
assert opt.state[model.pi.weight]['step'] == 2
assert opt.state[model.reader.inp.weight]['step'] == 1
assert pipeline.version == 1 and pipeline.steps == 0
assert all(c['uid'] == next(key[0] for key, (cid, _) in pipeline.frames.items() if pipeline.chunks[cid] is c)
           for c in pipeline.chunks if c is not None)
try:
    bank.forward([0])
    raise AssertionError('old source version accepted after optimizer boundary')
except RuntimeError as error:
    assert 'source version 0 consumed after switch to 1' in str(error)
for replica in pipeline.replicas:
    for name, param in replica.named_parameters():
        if source_parameter(name): torch.testing.assert_close(param.cpu(), dict(model.named_parameters())[name].cpu())
with torch.no_grad():
    encoded_before = pipeline.encoded_rows
    refreshed = (SourceView(resident,np.arange(len(requests[0]))) if a.resident else
                 SourceBatch(model, requests[0], cache=False, pipeline=pipeline))
    expected = SourceBatch(source_reference, requests[0], micro=8, cache=False)
    expected.model = model
    expected.kvs = [layer.kv(expected.sources()).float() for layer in model.reader.layers]
    torch.testing.assert_close(refreshed.forward([0])[0], expected.forward([0])[0], rtol=5e-4, atol=3e-5)
    assert pipeline.encoded_rows == encoded_before, 'source update re-encoded cached history'
    before_branch = refreshed.forward([0])[0].clone()
    fork = histories[0].fork()
    changed = seq[-1][0].copy(); changed['player'] += .2
    branch_req = [(fork, changed, pending[-1][0])]
    encoded_branch = pipeline.encoded_rows
    branch = SourceBatch(model, branch_req, cache=False, pipeline=pipeline)
    # A branch references its parent's frames (keys, values, graphs): only its
    # own decision frame is encoded (record retention rule 1).
    assert pipeline.encoded_rows == encoded_branch + 1, (pipeline.encoded_rows, encoded_branch)
    assert all(fork.key(t) == (histories[0].uid, t) for t in range(fork.length))
    shared = [i for i, k in enumerate(branch.frame_keys) if k[0] == histories[0].uid]
    bu = {f: r for r, f in enumerate(branch.u_frames)}
    compared = 0
    for i in shared:   # the parent's cached values, unchanged (the parent's encoding version)
        if i in bu:
            cid, offset = pipeline.frames[branch.frame_keys[i]]
            cached = pipeline.chunks[cid]['future'].result()['u'][offset].float()
            assert torch.equal(branch.u[bu[i]].to(cached.device), cached); compared += 1
    assert compared, 'no shared parent frame compared'
    # Its own frame is encoded with the current source weights.
    branch_ref = SourceBatch(model, branch_req, micro=8, cache=False)
    torch.testing.assert_close(branch.y[0][~branch.pad[0]], branch_ref.y[0][~branch_ref.pad[0]], rtol=5e-4, atol=3e-5)
    torch.testing.assert_close(refreshed.forward([0])[0], before_branch, rtol=0, atol=0)
assert fork.uid != histories[0].uid
checkpoint = io.BytesIO()
torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), source_version=pipeline.version), checkpoint)
checkpoint.seek(0)
loaded = torch.load(checkpoint, weights_only=True, map_location=a.learner)
resumed = TokPolicy(**cfg).to(a.learner).eval(); resumed.load_state_dict(loaded['model'])
resumed_opt = torch.optim.Adam(resumed.main_parameters(), lr=1e-4, eps=1e-5)
resumed_opt.load_state_dict(loaded['optimizer'])
assert resumed_opt.state[resumed.pi.weight]['step'] == 2
assert resumed_opt.state[resumed.reader.inp.weight]['step'] == 1
resumed_pipeline = SourcePipeline(resumed, a.sources.split(','), period=2, micro=8,
                                  version=loaded['source_version'])
with torch.no_grad():
    resumed_bank = SourceBatch(resumed, requests[0], cache=False, pipeline=resumed_pipeline)
    # Checkpoints contain parameters/version, not live autograd graphs. A cold
    # resume starts an empty cache under those parameters; no old graph is mixed.
    cold_reference = SourceBatch(resumed, requests[0], cache=False)
    torch.testing.assert_close(resumed_bank.forward([0])[0], cold_reference.forward([0])[0], rtol=5e-4, atol=3e-5)
assert resumed_bank.source_version == 1
resumed_pipeline.close()
if prof:
    prof.__exit__(None,None,None)
    prof.export_chrome_trace(a.trace)
print(json.dumps(dict(result='PASS', learner=a.learner, sources=a.sources, queries=a.queries, reader_micro=a.reader_micro,
      logits_max_abs=logits_error, source_gradient_max_abs=grad_error, weights_max_abs=weight_error,
      early_room_gradient=float(early.grad[0].abs().sum()), fast_steps=2, source_steps=1,
      version=pipeline.version, first_cycle_unique_source_rows=2*a.length, encoder_replay=0,
      encoded_rows=pipeline.encoded_rows, reused_rows=pipeline.reused_rows,
      events=pipeline.events), indent=2))
pipeline.close()
