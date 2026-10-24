"""Old sources keep their forward weights and receive later-room gradients.

The independent reference encodes dense observations under explicit immutable
weight snapshots. It does not use the pipeline's packed outputs or grouped Reader.
"""
import argparse
import copy
import json
import numpy as np
import torch
import test_memory_smoke as M
import test_temporal_smoke as TS
from isaac_bridge.tok_policy import TokPolicy, to_batch
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter


def reference_bank(pipeline, read_model, requests):
    bank = SourceBatch(read_model, requests, micro=8, cache=False)
    snapshots = {v: copy.deepcopy(replicas[0])._apply(
                    lambda t: t if t.is_meta else t.to(next(read_model.parameters()).device))
                 for v, replicas in pipeline.versions.items()}
    groups = {}
    for i, key in enumerate(bank.frame_keys):
        cid, _ = pipeline.frames[key]
        groups.setdefault(pipeline.chunks[cid]['version'], []).append(i)
    u = y = pad = None
    for version, rows in groups.items():
        b = to_batch(bank.rows[rows], bank.pending[rows], next(read_model.parameters()).device, entities=64)
        yy, pp, uu = snapshots[version].encode_rebuild(b)
        if u is None:
            u = uu.new_zeros((len(bank.rows), uu.shape[-1]))
            y = yy.new_zeros((len(bank.rows),) + yy.shape[1:])
            pad = pp.new_ones((len(bank.rows), pp.shape[1]))
        ix = torch.as_tensor(rows, device=u.device)
        u = u.index_copy(0, ix, uu); y = y.index_copy(0, ix, yy); pad[ix] = pp
    mem = u.new_zeros((len(bank.block_keys), read_model.reader.tokens, read_model.reader.D))
    groups = {}
    for i, key in enumerate(bank.block_keys):
        wid, _ = pipeline.blocks[key]
        groups.setdefault(pipeline.writes[wid]['version'], []).append(i)
    for version, rows in groups.items():
        ix = torch.as_tensor(rows, device=u.device)
        values = snapshots[version].reader.write(u[bank.block_indices[ix]])
        mem = mem.index_copy(0, ix, values)
    bank.u, bank.y, bank.pad, bank.mem = u, y[bank.unique_query_d], pad[bank.unique_query_d], mem.flatten(0, 1)
    bank.kvs = [layer.kv(bank.sources()).float() for layer in read_model.reader.layers]
    return bank, snapshots


p = argparse.ArgumentParser()
p.add_argument('--learner', default='cpu')
p.add_argument('--sources', default='cpu,cpu')
p.add_argument('--queries', type=int, default=7)
p.add_argument('--length', type=int, default=16)
a = p.parse_args()
torch.set_num_threads(2); torch.manual_seed(73)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
           rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).to(a.learner).eval()
pipeline = SourcePipeline(model, a.sources.split(','), period=4, micro=8)
opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
original = {n: x.detach().clone() for n, x in model.named_parameters() if source_parameter(n)}
rng = np.random.default_rng(73)
seq = TS.make_seq(rng, nw=2, T=a.length+5, changes={(0,8),(1,8),(0,16),(1,16)}, resets=set(), exits=set())
pending = [M.synth_pending(rng, 2, model.heads) for _ in seq]
histories = [RawHistory.empty(), RawHistory.empty()]
requests = []
for t, rows in enumerate(seq):
    requests.append([(histories[w], rows[w], pending[t][w]) for w in range(2)])
    histories = [histories[w].append(rows[w], pending[t][w], 4) for w in range(2)]
results = []
for cycle, length in enumerate([a.length, a.length, a.length+2, a.length+4, a.length+5]):
    req = [requests[length-1][i % 2] for i in range(a.queries)]
    model.zero_grad()
    with torch.inference_mode():
        value_bank = SourceBatch(model, req, cache=False, pipeline=pipeline)
        value_logits = value_bank.forward(np.arange(len(req)))[0].clone()
    encoded = pipeline.encoded_rows
    bank = SourceBatch(model, req, cache=False, pipeline=pipeline)
    assert pipeline.encoded_rows == encoded, 'training re-encoded value-pass sources'
    early = pipeline.chunks[0]['future'].result()['u']; early.retain_grad(); early.grad = None
    ref_model = copy.deepcopy(model)
    ref, snapshots = reference_bank(pipeline, ref_model, req)
    output = bank.forward(np.arange(len(req)), need_vg=True)
    torch.testing.assert_close(output[0], value_logits, rtol=5e-4, atol=3e-5)
    expected = ref.forward(np.arange(len(req)), need_vg=True)
    torch.testing.assert_close(output[0], expected[0], rtol=5e-4, atol=3e-5)
    for values in [output, expected]:
        (values[0].square().mean()+values[1].square().mean()+values[-1].square().mean()).backward()
    bank.backward(); pipeline.fast_step(opt, 1e9); pipeline.finish(opt, 1e9)
    error = 0.
    for name, param in model.named_parameters():
        if not source_parameter(name): continue
        gradients = [dict(x.named_parameters())[name].grad for x in [ref_model] + list(snapshots.values())]
        gradients = [g for g in gradients if g is not None]
        if gradients:
            expected_grad = sum(gradients)
            error = max(error, float((param.grad-expected_grad).abs().max()))
            torch.testing.assert_close(param.grad, expected_grad, rtol=2e-3, atol=5e-5, msg=name)
    assert early.grad[0].abs().sum() > 0, 'late loss lost its gradient to the cached earliest room'
    assert pipeline.encoded_rows == 2*length, 'source update replayed old observations'
    for name, param in pipeline.versions[0][0].named_parameters():
        if source_parameter(name): torch.testing.assert_close(param.to(a.learner), original[name], rtol=0, atol=0)
    results.append(dict(cycle=cycle, version=pipeline.version, encoded_rows=pipeline.encoded_rows,
                        gradient_max_abs=error, earliest_room_gradient=float(early.grad[0].abs().sum())))
uid = histories[0].uid
block = tuple((uid, t) for t in range(a.length, a.length+4))
assert {pipeline.chunks[pipeline.frames[k][0]]['version'] for k in block} == {2, 3}
assert pipeline.writes[pipeline.blocks[block][0]]['version'] == 3
pipeline.retain_histories({uid})
assert all(key[0] == uid for key in pipeline.frames)
print(json.dumps(dict(result='PASS', micro_queries=a.queries, cycles=results,
                     mixed_source_writer_versions=True, encoder_replay=0), indent=2))
pipeline.close()
