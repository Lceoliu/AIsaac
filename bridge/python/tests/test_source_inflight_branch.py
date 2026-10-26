"""A branch in flight while its parent advances (2026-10-26) — CPU, threads and source processes.

As test_source_reach_intervals, but the branch is forked in rollout 1 and runs in an actor (in flight) until its
record arrives at the learner in rollout ARRIVE; meanwhile P's rollout frontier passes the fork point plus H.
The learner learns of the branch only through the actors' lineage report (ancestor, shared frames, the branch's
current length). The audit must stay exactly zero and the parameters equal full retention's.
(Built from test_source_reach_intervals: the record admitted late, lineage reported while in flight.)

Retention by reachable frame intervals (2026-10-25, option d) — CPU, threads and source processes.

An episode P runs 8 rollouts of 8 frames (horizon H = 6). A branch record B forks P at frame 10, has
its own frames 10..13 and decides at L = 14; it is admitted after rollout 1 and consumed only in rollout 6
(late). Pipeline A retains with RecordLedger.frontier's intervals: P keeps [F - H, inf) for the rollout
and [L - H, 10) = [8, 10) for B, nothing in between. Pipeline R keeps every graph (no frontiers).
1. Same Reader outputs (bitwise) and the same source parameters after every interval, up to float
   summation order (A settles released chunks' credit earlier): the late record's loss still reaches P's
   encoder and Writer graphs in [8, 10) exactly as with full retention.
2. A releases P's chunks between the two ranges while B waits, and B's range once B is consumed.
3. Control V (the ancestor range left out, values only): different parameters from the consumption on,
   i.e. the late record's gradient into P's encoder / Writer is real and A keeps it.
  python tests/test_source_reach_intervals.py [--process]
"""
import copy, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch, SourceView
from isaac_bridge.record_retention import RecordLedger, lineage_intervals, in_flight_frontier
from isaac_bridge.source_pipeline import SourcePipeline

H, B_, ROLL, N, FORK, OWN, CONSUME = 6, 4, 8, 8, 10, 4, 6
ARRIVE = 4


def lineage_of(h):
    return lineage_intervals(slot_lists=[[h]])


def in_flight(front, a, shared, length):
    return in_flight_frontier(front, [(a, shared, length)], H)


def main(processes):
    torch.set_num_threads(2); torch.manual_seed(61)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=B_, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).eval()
    rng = np.random.default_rng(61)
    seq = TS.make_seq(rng, nw=2, T=ROLL * N + 1, changes={(0, 5), (0, 17), (1, 3)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 2, base.heads) for _ in seq]
    runs = {}
    for name in ('A', 'R', 'V'):   # V: a control, the record's ancestor range left out (values only, option a)
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, processes=processes, horizon=H)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        ledger = RecordLedger(1, 4, float('inf'), 1, B_)
        P = RawHistory.empty(); prefixes = []
        rec = None; outs, params, held = [], [], []
        for r in range(N):
            if rec is not None and name in ('A', 'V'):
                arrived = r > ARRIVE
                front = ledger.frontier({P.uid: P.length}, None, [rec[0]] if arrived else [], H)
                if not arrived:   # in flight: the learner knows only the actors' lineage report
                    for a_, shared, length in lineage_of(rec[0]):
                        front = in_flight(front, a_, shared, length)
                if name == 'V': front[P.uid] = [(P.length - H, float('inf'))]
                pipe.retain_histories({P.uid, rec[0].uid}, frontiers=front)
                for f in pipe.control_futures: f.result()
            graphs = sorted(t for (u, t) in pipe.frames
                            if u == P.uid and not pipe.chunks[pipe.frames[(u, t)][0]].get('detached'))
            held.append(graphs)
            req = []
            for t in range(r * ROLL, (r + 1) * ROLL):
                req.append((P, seq[t][0], pend[t][0])); prefixes.append(P); P = P.append(seq[t][0], pend[t][0], B_)
            if r == 1:   # the branch: P's prefix up to FORK, then its own frames
                b = prefixes[FORK].fork()
                for t in range(FORK, FORK + OWN): b = b.append(seq[t][1], pend[t][1], B_)
                rec = (b, seq[FORK + OWN][1], pend[FORK + OWN][1])
            if r == ARRIVE:
                ledger.admit_v2(np.array([0]))
                with torch.no_grad(): SourceBatch(model, [rec], cache=False, pipeline=pipe)   # (the record's arrival)
            consume = r == CONSUME
            if consume: req.append(rec); ledger.picked(v2=np.array([0]))
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            view = SourceView(bank, np.arange(len(req)))
            out = view.forward(np.arange(len(req)), need_vg=True)
            (out[0].square().mean() + out[1].square().mean()).backward()
            outs.append(out[0].detach().clone())
            view.backward(); pipe.fast_step(opt, 1e9)
            pipe.finish_rollout(opt, 1e9)   # the interval's VJP and source Adam step
            ledger.end_update()
            params.append([p.detach().clone() for p in model.parameters()])
        dropped = pipe.horizon_stats()['source_horizon_dropped_grad']
        assert name != 'A' or dropped == 0, ('the audit dropped gradient', dropped)
        pipe.close()
        runs[name] = (outs, params, held)
    (oA, pA, hA), (oR, pR, hR), (_, pV, _) = runs['A'], runs['R'], runs['V']
    # the control: without the ancestor range the late record's loss no longer reaches P's [8, 10)
    assert any(not torch.equal(x, y) for x, y in zip(pV[CONSUME], pR[CONSUME])), 'the record did not train P frames'
    for r in range(N):
        assert torch.equal(oA[r], oR[r]), f'Reader outputs differ in rollout {r}'
        for x, y in zip(pA[r], pR[r]):
            assert torch.allclose(x, y, rtol=1e-5, atol=1e-7), (r, float((x - y).abs().max()))
    diff = [max(float((x - y).abs().max()) for x, y in zip(pA[r], pR[r])) for r in range(N)]
    print(json.dumps(dict(result='PASS', processes=processes, max_param_diff=diff,
                          held_frames_A=[len(h) for h in hA], held_frames_R=[len(h) for h in hR])))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
