"""Same inputs, old-version cross-room gradients, no persistent activation graph.

--checkout lets the identical probe run against the pristine or restored tree.
CPU is a correctness/smoke test; its timings are not GPU training throughput.
"""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--checkout', type=Path, default=Path(__file__).resolve().parents[3])
p.add_argument('--mode', choices=('auto', 'compare'), default='compare')
p.add_argument('--micro', type=int, default=8192)
p.add_argument('--horizon', type=int, default=0)
p.add_argument('--bf16', action='store_true')
p.add_argument('--process', action='store_true')
p.add_argument('--retire-before-update', action='store_true')
# (2026-10-27, CUDA gate) the learner and source devices, the long runs' source settings, and the gradient threshold
p.add_argument('--device', default='cpu', help='learner device; sources on --sources')
p.add_argument('--sources', default='cpu')
p.add_argument('--flash', action='store_true')
p.add_argument('--compile', action='store_true')
p.add_argument('--grad-threshold', type=float, default=None, help='relative L2 of the source gradients (replay vs stored)')
a = p.parse_args()
root = a.checkout / 'bridge' / 'python'
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'tests'))
import numpy as np
import torch
import test_memory_smoke as M
import test_temporal_smoke as TS
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter


def run(base, seq, pend, replay):
    model = copy.deepcopy(base)
    options = dict(period=0, micro=a.micro, update_rollouts=1, amp=a.bf16,
                   horizon=a.horizon, processes=a.process, flash_source=a.flash, compile_source=int(a.compile))
    if replay: options['replay'] = True
    pipe = SourcePipeline(model, a.sources.split(','), **options)
    opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
    histories = [RawHistory.empty() for _ in range(6)]
    outputs, gradients, versions = [], [], []
    graphs = 0
    late = None
    start = time.perf_counter()
    for rollout in range(4):
        requests = []
        for t in range(rollout*8, (rollout+1)*8):
            for w in range(6):
                if t % 4 == 3: requests.append((histories[w], seq[t][w], pend[t][w]))
                if t == 7 and w == 0:
                    late = (histories[w].fork(), seq[t][w].copy(), pend[t][w].copy())
                histories[w] = histories[w].append(seq[t][w], pend[t][w], 4)
        # A delayed branch reads its ancestor under the original source version.
        if rollout == 3: requests.append(late)
        bank = SourceBatch(model, requests, cache=False, pipeline=pipe)
        # Reusing an unchanged request must not re-encode its history.
        before = pipe.encoded_rows
        pipe.prefetch(requests)
        assert pipe.encoded_rows == before
        for step in range(2):
            bank = SourceBatch(model, requests, cache=False, pipeline=pipe)
            opt.zero_grad(set_to_none=True)
            out = bank.forward(np.arange(len(requests)), need_vg=True)
            outputs.append(torch.cat([x.detach().flatten().float() for x in out]))
            loss = out[0].square().mean() + out[1].square().mean() + out[-1].square().mean()
            loss.backward()
            bank.backward()
            pipe.fast_step(opt, 1e9)
        for chunk in pipe.chunks + pipe.writes:
            if chunk is None: continue
            result = chunk['future'].result()
            graphs += sum(torch.is_tensor(result.get(key)) and result[key].grad_fn is not None
                          for key in ('u', 'y', 'mem'))
        if a.retire_before_update and rollout == 3:
            pipe._flush_collected()
            alive = [h.uid for h in histories] + [late[0].uid]
            pipe.retain_histories(alive, frontiers={uid: float('inf') for uid in alive})
            for future in pipe.control_futures: future.result()
        pipe.finish_rollout(opt, 1e9)
        pipe.finish(opt, 1e9)
        gradients.append({n: v.grad.detach().clone() for n, v in model.named_parameters()
                          if source_parameter(n) and v.grad is not None})
        versions.append(pipe.version)
        pipe.retain_histories([h.uid for h in histories] + [late[0].uid],
                              frontiers={h.uid: [(0, h.length+1)] for h in histories})
    audit = float(pipe.horizon_dropped) if pipe.horizon_dropped is not None else 0.
    assert audit == 0., audit
    assert all(g and any(float(v.abs().max()) > 0 for v in g.values()) for g in gradients)
    if replay: assert graphs == 0, graphs
    state = {n: v.detach().clone() for n, v in model.named_parameters()}
    replay_events = [e for e in pipe.events if e.get('replay')]
    if replay: assert all(e['cached_graph_outputs'] == 0 for e in replay_events)
    report = dict(mode='replay' if replay else 'stored', micro=a.micro, horizon=a.horizon,
                  bf16=a.bf16, processes=a.process, updates=versions, audit=audit,
                  cached_graph_outputs=graphs, encoded_rows=pipe.encoded_rows,
                  value_encoder_batches=sum(e['kind']=='encode' for e in pipe.events),
                  replayed_rows=sum(e['replayed_rows'] for e in replay_events),
                  replay_encoder_batches=sum(e['encoder_batches'] for e in replay_events),
                  replay_writer_batches=sum(e['writer_batches'] for e in replay_events),
                  seconds=round(time.perf_counter()-start, 4))
    pipe.close()
    return report, state, outputs, gradients


def main():
    torch.set_num_threads(2); torch.manual_seed(71)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
               temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
               rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
    model = TokPolicy(**cfg).to(a.device).eval()
    rng = np.random.default_rng(71)
    seq = TS.make_seq(rng, nw=6, T=32, changes={(0,6),(1,9),(2,13),(4,3)}, resets=set(), exits=set())
    # Ragged packed-token offsets must survive cross-episode regrouping.
    for t, rows in enumerate(seq):
        rows['n_ent'][np.arange(6) % 3 == t % 3] = 0
        rows['n_doors'][np.arange(6) % 2 == t % 2] = 0
    pend = [M.synth_pending(rng, 6, model.heads) for _ in seq]
    available = 'replay' in inspect.signature(SourcePipeline).parameters
    if a.mode == 'auto':
        report, _, _, _ = run(model, seq, pend, available)
        print(json.dumps(dict(result='PASS', **report)))
        return
    assert available
    old = run(model, seq, pend, False)
    new = run(model, seq, pend, True)
    error = max(float((x-y).abs().max()) for x, y in zip(old[2], new[2]))
    grad_abs = max(float((x[n]-y[n]).abs().max()) for x, y in zip(old[3], new[3]) for n in x)
    grad_rel = max(float(torch.linalg.vector_norm(torch.cat([(x[n]-y[n]).flatten() for n in x])) /
                         torch.linalg.vector_norm(torch.cat([x[n].flatten() for n in x])).clamp_min(1e-12))
                   for x, y in zip(old[3], new[3]))
    weight = max(float((old[1][n]-new[1][n]).abs().max()) for n in old[1])
    print(json.dumps(dict(baseline=old[0], modified=new[0], logits_max_abs=error,
                         gradients_max_abs=grad_abs, gradients_relative_l2=grad_rel, weights_max_abs=weight)))
    # BF16 re-batching changes reduction rounding, unlike the FP32 reference.
    assert grad_rel < (a.grad_threshold if a.grad_threshold is not None else (0.02 if a.bf16 else 1e-5)), grad_rel
    assert error < (0.02 if a.bf16 else 1e-5), error
    assert weight < (0.002 if a.bf16 else 2e-5), weight
    print('PASS: cross-room/versioned VJP parity; value-only cache; unchanged reuse; micro=8192')


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main()
