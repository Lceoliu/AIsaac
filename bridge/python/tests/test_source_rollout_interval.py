"""Delayed source Adam: exact accumulated credit, retired rooms and save fence."""
import argparse
import copy
import json
import sys
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
p.add_argument('--interval', type=int, choices=(1, 2, 4), default=4)
p.add_argument('--source-bf16', action='store_true')
p.add_argument('--source-flash', action='store_true')
p.add_argument('--source-compile', action='store_true')
p.add_argument('--prefetch', action='store_true')
a = p.parse_args()
torch.set_num_threads(2); torch.manual_seed(93)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
           rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).to(a.learner).eval()
reference = copy.deepcopy(model)
pipe = SourcePipeline(model, a.sources.split(','), period=0, micro=8,
                      update_rollouts=a.interval, amp=a.source_bf16,
                      flash_source=a.source_flash, compile_source=a.source_compile)
if a.source_bf16:
    # The established precision gate's independent dense mixed-precision path.
    encode, write = reference.encode_rebuild, reference.reader.write
    def source_encode(b):
        b['_source_bf16'] = True
        with torch.autocast(torch.device(a.learner).type, dtype=torch.bfloat16):
            y, pad, u = encode(b)
        return y.float(), pad, u.float()
    def source_write(x):
        with torch.autocast(torch.device(a.learner).type, dtype=torch.bfloat16):
            m = write(x.to(torch.bfloat16))
        return m.float()
    reference.encode_rebuild, reference.reader.write = source_encode, source_write
opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
fast = [v for n, v in reference.named_parameters() if not source_parameter(n)]
slow = [v for n, v in reference.named_parameters() if source_parameter(n)]
ref_fast = torch.optim.Adam(fast, lr=1e-4, eps=1e-5)
ref_slow = torch.optim.Adam(slow, lr=1e-4, eps=1e-5)
saved = {n: v.detach().clone() for n, v in model.named_parameters() if source_parameter(n)}
rng = np.random.default_rng(93)
hs = [RawHistory.empty(), RawHistory.empty()]
for t, rows in enumerate(TS.make_seq(rng, nw=2, T=20, changes={(0, 8), (1, 8)}, resets=set(), exits=set())):
    rows['n_ent'] = 0; rows['n_doors'] = 0
    pending = M.synth_pending(rng, 2, model.heads)
    if t == 19: base = [(hs[w], rows[w], pending[w]) for w in range(2)]
    for w in range(2): hs[w] = hs[w].append(rows[w], pending[w], 4)
dead, live = hs[0].uid, hs[1].uid
grads = {}; total_steps = 2*a.interval
for rollout in range(a.interval):
    pipe.retain_histories({dead, live} if rollout == 0 else {live})
    if rollout:
        assert (dead, 0) not in pipe.frames, 'retired graph was kept until the slow boundary'
        assert early.grad[0].abs().sum() > 0, 'retired-room credit was discarded before collection'
    req = [base[i % 2] if rollout == 0 else base[1] for i in range(a.queries)]
    if a.prefetch:
        pipe.prefetch(req,write_ahead=True)
        assert pipe.writes, 'streamed prefix did not enqueue Writer work'
        count = pipe.encoded_rows
    with torch.no_grad(): resident = SourceBatch(model, req, cache=False, pipeline=pipe)
    if a.prefetch: assert pipe.encoded_rows == count, 'handoff re-encoded a prefetched row'
    if rollout == 0:
        early = pipe.chunks[pipe.frames[(dead, 0)][0]]['future'].result()['u']
        early.retain_grad(); encoded = pipe.encoded_rows
    else: assert pipe.encoded_rows == encoded, 'unchanged source was replayed'
    for _ in range(2):
        model.zero_grad(); reference.zero_grad()
        bank = SourceView(resident, np.arange(a.queries))
        out = bank.forward(np.arange(a.queries), need_vg=True)
        (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
        bank.backward(); pipe.fast_step(opt, 1e9)
        full = SourceBatch(reference, req, micro=8, cache=False)
        expected = full.forward(np.arange(a.queries), need_vg=True)
        if a.source_bf16:
            assert float(((out[0]-expected[0]).norm()/expected[0].norm()).detach()) < .03
        else: torch.testing.assert_close(out[0], expected[0], rtol=5e-4, atol=3e-5)
        (expected[0].square().mean()+expected[1].square().mean()+expected[-1].square().mean()).backward()
        for name, param in reference.named_parameters():
            if source_parameter(name) and param.grad is not None:
                if name not in grads: grads[name] = param.grad.detach().clone()
                else: grads[name].add_(param.grad)
        ref_fast.step()
    pipe.finish_rollout(opt, 1e9, overlap=True)
    assert not pipe.collected and resident.u.grad is None and resident.y.grad is None and resident.mem.grad is None
    assert pipe.version == 0
    for name, param in model.named_parameters():
        if name in saved: torch.testing.assert_close(param, saved[name], rtol=0, atol=0)
    if rollout+1 < a.interval:
        assert pipe.pending is None and pipe.steps == 2*(rollout+1)
        if rollout == 0: assert early.grad is None, 'source VJP ran before retirement or interval boundary'
    else: assert pipe.pending is not None and pipe.steps == 0
pipe.finish(opt, 1e9)
for name, param in reference.named_parameters():
    if source_parameter(name): param.grad = grads[name]/total_steps if name in grads else None
ref_slow.step()
gradient_error = weight_error = 0.
for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
    if name in grads:
        gradient_error = max(gradient_error, float((actual.grad-expected.grad).abs().max()))
        if not a.source_bf16:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-3, atol=5e-5, msg=name)
    weight_error = max(weight_error, float((actual-expected).detach().abs().max()))
    if not a.source_bf16: torch.testing.assert_close(actual, expected, rtol=2e-3, atol=5e-5, msg=name)
ga = torch.cat([dict(model.named_parameters())[n].grad.flatten() for n in grads]).double()
gb = torch.cat([dict(reference.named_parameters())[n].grad.flatten() for n in grads]).double()
gradient_relative = float((ga-gb).norm()/gb.norm())
assert gradient_relative < (.10 if a.source_bf16 else .001)
if a.source_bf16: assert weight_error < 3e-4
early_gradient = float(early.grad[0].abs().sum())
assert early_gradient > 0, 'retired first-room gradient was lost'
assert pipe.version == 1 and opt.state[model.reader.inp.weight]['step'] == 1
assert opt.state[model.pi.weight]['step'] == total_steps
pipe.retain_histories({live})
assert (dead, 0) not in pipe.frames and (live, 0) in pipe.frames
try:
    bank.forward([0])
    raise AssertionError('stale source version accepted')
except RuntimeError as error:
    assert 'source version 0 consumed after switch to 1' in str(error)
# Old live source stays byte-identical across the commit; Reader uses fresh KV.
old_u = pipe.chunks[pipe.frames[(live, 0)][0]]['future'].result()['u'].detach().clone()
bank = SourceBatch(model, [base[1]]*a.queries, cache=False, pipeline=pipe)
assert pipe.encoded_rows == encoded and bank.source_version == 1
torch.testing.assert_close(pipe.chunks[pipe.frames[(live, 0)][0]]['future'].result()['u'], old_u, rtol=0, atol=0)
# A checkpoint flushes a partial slow interval exactly once.
model.zero_grad()
out = bank.forward(np.arange(a.queries), need_vg=True)
out[0].square().mean().backward(); bank.backward(); pipe.fast_step(opt, 1e9)
pipe.finish_rollout(opt, 1e9, overlap=True)
pipe.finish(opt, 1e9); pipe.finish(opt, 1e9)
assert pipe.version == 2 and opt.state[model.reader.inp.weight]['step'] == 2
assert pipe.pending is None and pipe.steps == 0 and pipe.rollouts_since_update == 0
pipe.close()
print(json.dumps(dict(result='PASS', interval=a.interval, queries=a.queries,
    source_precision='bf16' if a.source_bf16 else 'fp32', source_gradient_relative_l2=gradient_relative,
    fast_steps=total_steps, source_steps=1, source_gradient_max_abs=gradient_error,
    weight_max_abs=weight_error, retired_early_room_gradient=early_gradient,
    encoder_replay=0, partial_checkpoint_source_steps=1, reader_banks_released=True)))
