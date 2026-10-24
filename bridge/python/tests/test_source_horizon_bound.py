"""Retained source graphs stay bounded with a horizon and grow without one.

One long synthetic episode pair (2 x 320 decisions) trained rollout by rollout
(32 decisions per worker each, update_rollouts=2) through the full pipeline
loop: retain/detach -> resident bank -> Reader -> credit -> slow VJP/Adam.
The progress.csv statistic source_graph_frames_uid_max must stop growing at
H + one rollout + chunk/Writer-job granularity, while H=0 keeps every frame.
The audited dropped cotangent must be exactly zero.
"""
import argparse, copy, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.source_pipeline import SourcePipeline

H, T, ROLL, BLOCK, MICRO = 24, 320, 32, 4, 8
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True,
           temporal_width=32, temporal_layers=2, temporal_heads=4, temporal_window=4,
           rebuild_memory=True, memory_block=BLOCK, memory_tokens=2, source_cache=True)


def run():
    if sys.platform == 'win32':   # source workers' periodic malloc_trim needs glibc (Linux only)
        os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    p = argparse.ArgumentParser(); p.add_argument('--process', action='store_true'); a = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(41)
    base = TokPolicy(**CFG).eval()
    rng = np.random.default_rng(41)
    seq = TS.make_seq(rng, nw=2, T=T, changes={(w, t) for w in range(2) for t in range(10 + 7 * w, T, 23)},
                      resets=set(), exits=set())
    hs = [RawHistory.empty(), RawHistory.empty()]
    rollouts = [[] for _ in range(T // ROLL)]
    for t, rows in enumerate(seq):
        pending = M.synth_pending(rng, 2, base.heads)
        for w in range(2):
            rollouts[t // ROLL].append((hs[w], rows[w], pending[w]))
            hs[w] = hs[w].append(rows[w], pending[w], BLOCK)
    trace = {}
    for horizon in (H, 0):
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=MICRO, update_rollouts=2, horizon=horizon,
                              processes=a.process)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, eps=1e-5)
        rows = []
        for req in rollouts:
            pipe.poll_finish(opt, 1e9, wait=True)
            front = {}
            for h, _, _ in req: front[h.uid] = min(front.get(h.uid, h.length), h.length)
            pipe.retain_histories({h.uid for h, _, _ in req}, frontiers=front)
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            stats = pipe.horizon_stats()
            if a.process:   # what the worker processes actually still hold as graphs
                for worker, executor in enumerate(pipe.workers):
                    pipe.remote_memory[worker] = executor.submit(lambda w=worker: pipe.remotes[w].request('retire', (
                        [i for i, c in enumerate(pipe.chunks) if c is not None and c['worker'] == w],
                        [i for i, c in enumerate(pipe.writes) if c is not None and c['worker'] == w],
                        list(pipe.versions)))[2]).result()
                assert pipe.horizon_stats()['source_worker_graph_frames'] == stats['source_graph_frames'], (pipe.remote_memory, stats)
            view = SourceView(bank, np.arange(len(req)))
            out = view.forward(np.arange(len(req)), need_vg=True)
            (out[0].square().mean() + out[1].square().mean() + out[-1].square().mean()).backward()
            view.backward(); pipe.fast_step(opt, 1e9); pipe.finish_rollout(opt, 1e9)
            rows.append(dict(first=min(front.values()), graph_uid_max=stats['source_graph_frames_uid_max'],
                             graph=stats['source_graph_frames'], detached=stats['source_detached_frames']))
            del view, out, bank
        pipe.finish(opt, 1e9)
        rows[-1]['dropped'] = pipe.horizon_stats()['source_horizon_dropped_grad']
        trace[horizon] = rows
        pipe.close()
    bounded, unbounded = trace[H], trace[0]
    # Reachable frames after the bank is built: [first-H, first+ROLL); plus at
    # most one chunk (MICRO rows) below it and one pinned Writer job (2 blocks
    # per worker group at MICRO//BLOCK) -- a constant, not a function of T.
    bound = H + ROLL + MICRO + 2 * MICRO
    assert max(r['graph_uid_max'] for r in bounded) <= bound, (bounded, bound)
    assert unbounded[-1]['graph_uid_max'] == T, unbounded[-1]
    assert bounded[-1]['dropped'] == 0. and unbounded[-1]['dropped'] == 0.
    late = [r['graph_uid_max'] for r in bounded[3:]]
    assert max(late) - min(late) <= 2 * MICRO, late
    print(json.dumps(dict(result='PASS', process=a.process, horizon=H, rollout=ROLL, bound=bound,
                          bounded_graph_uid_max=[r['graph_uid_max'] for r in bounded],
                          unbounded_graph_uid_max=[r['graph_uid_max'] for r in unbounded],
                          bounded_detached=[r['detached'] for r in bounded], dropped=bounded[-1]['dropped'])))


if __name__ == '__main__':
    run()
