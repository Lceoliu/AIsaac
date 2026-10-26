"""An ended episode in its grace period, read with gradient by a late record (2026-10-26) — CPU, threads and processes.

Episode P runs rollouts 0..2 (24 frames) and ends; episode Q takes its worker from rollout 3 on. A branch B forks P
at frame 10 in rollout 1, runs in an actor (in flight) and its record (L = 14) reaches the learner only in rollout
ARRIVE = 4, after P ended; the record is consumed in rollout CONSUME = 6. Grace = 2 updates: in rollouts 3 and 4
P has no rollout decision and is kept for its grace period (rule 5, D2a: values only by itself).
The frontiers come from the production code path (record_retention.grace_keep + learner_frontier, as train_tok).
  A: production: the actors report B's lineage while it is in flight; the learner learns of the record at ARRIVE.
     The audit must be exactly zero and the parameters equal to full retention's (R).
  R: full retention (no frontiers).
  V: control: the same without the lineage report (grace only). P's frames in [L - H, 10) are values only when
     the record reads them: the audit is non-zero and the parameters differ from R. The grace period by itself
     cannot keep a late record's gradient; the in-flight report does (merge_jobs keeps both actors' reports).
  python tests/test_source_grace_late.py [--process]
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
from isaac_bridge.record_retention import RecordLedger, lineage_intervals, grace_keep, learner_frontier
from isaac_bridge.source_pipeline import SourcePipeline

H, B_, ROLL, N, FORK, OWN = 6, 4, 8, 8, 10, 4
P_END, ARRIVE, CONSUME, GRACE = 3, 4, 6, 2   # P's last rollout is P_END - 1


def main(processes):
    torch.set_num_threads(2); torch.manual_seed(67)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=B_, memory_tokens=2, source_cache=True)
    base = TokPolicy(**cfg).eval()
    rng = np.random.default_rng(67)
    seq = TS.make_seq(rng, nw=2, T=ROLL * N + 1, changes={(0, 5), (0, 17), (1, 3)}, resets=set(), exits=set())
    pend = [M.synth_pending(rng, 2, base.heads) for _ in seq]
    runs = {}
    for name in ('A', 'R', 'V'):
        model = copy.deepcopy(base)
        pipe = SourcePipeline(model, ['cpu', 'cpu'], period=0, micro=8, update_rollouts=1, processes=processes, horizon=H)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-3, eps=1e-5)
        ledger = RecordLedger(1, 4, float('inf'), 1, B_)
        P, Q = RawHistory.empty(), RawHistory.empty(); prefixes = []
        rec = None; v2 = [None]; recent = {}; outs, params, held = [], [], []
        last = []   # the previous rollout's requested histories
        for r in range(N):
            if r and name in ('A', 'V'):
                arrived = rec is not None and r > ARRIVE
                reach = lineage_intervals(slot_lists=[[rec[0]]]) if rec is not None and not arrived and name == 'A' else set()
                in_flight = {a for a, _, _ in reach}
                in_flight |= grace_keep(recent, [h.uid for h in last] + sorted(in_flight), r, GRACE)
                live = {h.uid for h in last} | in_flight | {h.uid for h in v2 if h is not None}
                if rec is not None: live.add(rec[0].uid)   # (the record's own frames: the learner holds them once arrived)
                front, _ = learner_frontier(last, ledger, None, v2, reach, in_flight, H)
                pipe.retain_histories(live, frontiers=front)
                for f in pipe.control_futures: f.result()
            graphs = sorted(t for (u, t) in pipe.frames
                            if u == P.uid and not pipe.chunks[pipe.frames[(u, t)][0]].get('detached'))
            held.append(graphs)
            req = []
            for t in range(r * ROLL, (r + 1) * ROLL):
                if r < P_END:
                    req.append((P, seq[t][0], pend[t][0])); prefixes.append(P); P = P.append(seq[t][0], pend[t][0], B_)
                else:   # P ended: Q in its worker
                    req.append((Q, seq[t][0], pend[t][0])); Q = Q.append(seq[t][0], pend[t][0], B_)
            last = [h for h, _, _ in req]
            if r == 1:   # the branch: P's prefix up to FORK, then its own frames (in an actor until ARRIVE)
                b = prefixes[FORK].fork()
                for t in range(FORK, FORK + OWN): b = b.append(seq[t][1], pend[t][1], B_)
                rec = (b, seq[FORK + OWN][1], pend[FORK + OWN][1])
            if r == ARRIVE:
                v2[0] = rec[0]; ledger.admit_v2(np.array([0]))
                with torch.no_grad(): SourceBatch(model, [rec], cache=False, pipeline=pipe)   # (the record's arrival)
            consume = r == CONSUME
            if consume: req.append(rec); ledger.picked(v2=np.array([0]))
            with torch.no_grad(): bank = SourceBatch(model, req, cache=False, pipeline=pipe)
            view = SourceView(bank, np.arange(len(req)))
            out = view.forward(np.arange(len(req)), need_vg=True)
            (out[0].square().mean() + out[1].square().mean()).backward()
            outs.append(out[0].detach().clone())
            view.backward(); pipe.fast_step(opt, 1e9)
            pipe.finish_rollout(opt, 1e9)
            ledger.end_update()
            params.append([p.detach().clone() for p in model.parameters()])
        dropped = float(pipe.horizon_stats()['source_horizon_dropped_grad'])
        pipe.close()
        runs[name] = (outs, params, held, dropped)
    (oA, pA, hA, dA), (oR, pR, hR, _), (_, pV, _, dV) = runs['A'], runs['R'], runs['V']
    assert dA == 0, ('A: the audit dropped gradient', dA)
    assert dV > 0 and any(not torch.equal(x, y) for x, y in zip(pV[CONSUME], pR[CONSUME])), \
        ('the control should lose the late record\'s gradient into P', dV)
    for r in range(N):
        assert torch.equal(oA[r], oR[r]), f'Reader outputs differ in rollout {r}'
        for x, y in zip(pA[r], pR[r]):
            assert torch.allclose(x, y, rtol=1e-5, atol=1e-7), (r, float((x - y).abs().max()))
    diff = [max(float((x - y).abs().max()) for x, y in zip(pA[r], pR[r])) for r in range(N)]
    print(json.dumps(dict(result='PASS', processes=processes, dropped_A=dA, dropped_V=dV, max_param_diff=diff,
                          held_P_frames_A=[len(h) for h in hA], held_P_frames_R=[len(h) for h in hR])))


if __name__ == '__main__':
    if sys.platform == 'win32': os.environ.setdefault('ISAAC_HOST_TRIM_INTERVAL', '0')
    main('--process' in sys.argv)
