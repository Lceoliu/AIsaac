"""A waiting record's own query frame (2026-10-26) — CPU, threads and source processes.

As test_source_reach_intervals, but the record's current observation (its decision frame L) is encoded on
its own when the record arrives (as the learner's v2 / teacher arrival batch does), then the record waits
for rollouts with retention running before its loss is used. The audit must stay exactly zero
(source_horizon_dropped_grad) and the parameters equal full retention's.
(The test below is test_source_reach_intervals' with the arrival encode added.)

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
from isaac_bridge.record_retention import RecordLedger
from isaac_bridge.source_pipeline import SourcePipeline

H, B_, ROLL, N, FORK, OWN, CONSUME = 6, 4, 8, 8, 10, 5, 6   # (records decide at 15 and 16: 16 opens a block, no Writer holds it)


def main(processes):
    torch.set_num_threads(2); torch.manual_seed(61)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=B_, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).eval()
    rng = np.random.default_rng(61)
    seq = TS.make_seq(rng, nw=2, T=ROLL * N + 1, changes={(0, 5), (0, 17), (1, 3)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 2, base.heads) for _ in seq]
    runs = {}
    for name in ('A', 'R'):   # (the values-only control is test_source_reach_intervals'; here the ancestor's reachable
        #                    frame 9 lies outside the recent window and in a block out of horizon: no gradient either way)
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, processes=processes, horizon=H)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        ledger = RecordLedger(1, 4, float('inf'), 1, B_)
        first = None
        P = RawHistory.empty(); prefixes = []
        rec = None; outs, params, held = [], [], []
        for r in range(N):
            if rec is not None and name in ('A', 'V'):
                front = ledger.frontier({P.uid: P.length}, None, [first[0], rec[0]], H)
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
                first = (b, seq[FORK + OWN][1], pend[FORK + OWN][1])   # a record deciding at L = 15
                b2 = b.append(seq[FORK + OWN][1], pend[FORK + OWN][1], B_)
                rec = (b2, seq[FORK + OWN + 1][1], pend[FORK + OWN + 1][1])   # the branch's next decision, L = 16
                ledger.admit_v2(np.array([0, 1]))
                # (arrivals, one batch each as the learner's: the second encodes only its query frame 16)
                with torch.no_grad():
                    SourceBatch(model, [first], cache=False, pipeline=pipe); SourceBatch(model, [rec], cache=False, pipeline=pipe)
            consume = r == CONSUME
            if consume: req += [first, rec]; ledger.picked(v2=np.array([0, 1]))
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
    (oA, pA, hA), (oR, pR, hR) = runs['A'], runs['R']
    for r in range(N):
        assert torch.equal(oA[r], oR[r]), f'Reader outputs differ in rollout {r}'
        for x, y in zip(pA[r], pR[r]):
            assert torch.allclose(x, y, rtol=1e-5, atol=1e-7), (r, float((x - y).abs().max()))
    diff = [max(float((x - y).abs().max()) for x, y in zip(pA[r], pR[r])) for r in range(N)]
    # while B waits (rollouts 2..6): P keeps the chunk holding [8, 10) (frames 8..15: one rollout's encode)
    # and the chunks its rollout decisions reach; the chunks between them are released
    for r in range(3, CONSUME + 1):
        gap = [t for t in hA[r] if 2 * ROLL <= t < (r - 1) * ROLL]
        assert any(8 <= t < FORK for t in hA[r]) and not gap, (r, hA[r])
    assert any(2 * ROLL <= t < (CONSUME - 1) * ROLL for t in hR[CONSUME])   # (R keeps the gap)
    # after B's loss was consumed (rollout 6), its range goes too
    assert not any(t < FORK for t in hA[CONSUME + 1]), hA[CONSUME + 1]
    assert len(hR[N - 1]) == ROLL * (N - 1)   # the reference kept everything
    print(json.dumps(dict(result='PASS', processes=processes, max_param_diff=diff,
                          held_frames_A=[len(h) for h in hA], held_frames_R=[len(h) for h in hR])))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
