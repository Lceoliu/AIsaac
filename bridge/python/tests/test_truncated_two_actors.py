"""--memory-train truncated with two actor processes (CPU; train_tok --actor-processes 2's layout): tests 4 and 1 on
the learner's merged rollout.

Two eager rebuild actors of 3 workers each (their own records, episode starts at step 0 and at RESETS) store into
their own rollouts over the shared host arrays of all 6 workers (full_from_local / slice_into), handed over at
HANDOVERS (each actor's open decisions carried). At each hand-over the finished buffer is merged as the learner process
merges it (merge_rollouts: both actors' pools, t_s and t_m included, in one mirror, the second actor's rows moved), and
also mirrored as the one-actor learner process does (mirror_temporal) for the first actor alone. On the merged view:
  test 4: every worker's carried source values and LearnerCarry's Writer tokens == its own actor's state, bit for bit;
  test 1: the learner's batched outputs over all 6 workers' rollouts == each actor's step outputs (max abs <= 1e-5);
  no context mismatch. On the mirrored view: the same as on the actor's own rollout, bit for bit.
  python tests/test_truncated_two_actors.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
import train_tok as TT
from isaac_bridge import tok_policy as TP
from isaac_bridge.truncated_memory import LearnerCarry, TruncatedBatch, worker_context

W, BLOCK, TOKENS = 4, 4, 2
RESETS = {(1, 12), (2, 30), (0, 25)}
HANDOVERS = (10, 22, 31)
NW, T = 3, 40


def main():
    torch.set_num_threads(2); torch.manual_seed(13)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=W, rebuild_memory=True, memory_block=BLOCK, memory_tokens=TOKENS,
               source_cache=True)
    model = TP.TokPolicy(**cfg).eval()
    ta = model.tatt
    tcfg = dict(P=W, L=ta.L, D=ta.D, R=ta.R, kv='fp32', device='cpu', truncated=True, tokens=TOKENS)
    seqs, pends = [], []
    for a in range(2):
        rng = np.random.default_rng(130 + a)
        seqs.append(TS.make_seq(rng, nw=NW, T=T, changes={(0, 5), (1, 9)}, resets=RESETS if a == 0 else {(2, 17)},
                                exits=set()))
        pends.append([M.synth_pending(rng, NW, model.heads) for _ in range(T)])
    rolls = [[TT.Rollout(NW, 64, len(model.heads), 0, True, dict(tcfg, rows=None)),
              TT.Rollout(NW, 64, len(model.heads), 0, True, dict(tcfg, rows=7))] for _ in range(2)]
    fulls = [TT.full_from_local(r, 2 * NW, 0, np.zeros) for r in rolls[0]]
    for bid, r in enumerate(rolls[1]):
        TT.slice_into(r, fulls[bid], NW)
    states = [TP.TemporalState(model, NW, 'cpu', torch.float32) for _ in range(2)]
    carry = LearnerCarry(2 * NW, BLOCK, TOKENS, ta.D, 'cpu')
    carry1 = LearnerCarry(NW, BLOCK, TOKENS, ta.D, 'cpu')   # (the one-actor mirror of actor 0)
    logs = [dict(logits=[], value=[], vg=[]) for _ in range(2)]
    bid, start, snaps = 0, 0, None
    stats = dict(ctx=0, err=0.0, merges=0)
    for t in range(T + 1):
        if t + 1 in HANDOVERS or t + 1 == T:
            snaps = [[(st.rebuild.recent[w].clone() if st.rebuild.recent[w] is not None else None,
                       st.rebuild.mem[w].clone() if st.rebuild.mem[w] is not None else None) for w in range(NW)]
                     for st in states]
        if t in HANDOVERS or t == T:
            for a in range(2):
                rolls[a][1 - bid].carry(rolls[a][bid])
            parts = [(a * NW, TT.rollout_scalars(rolls[a][bid]), TT.rollout_tdev(rolls[a][bid])) for a in range(2)]
            view, _ = TT.merge_rollouts(fulls[bid], parts, {}, 'cpu')
            mirrored, _ = TT.mirror_temporal(rolls[0][bid], {}, 'cpu')
            for v_, c_ in ((view, carry), (mirrored, carry1)):
                tb = TruncatedBatch(model, v_, c_, 'cpu')
                assert tb.mismatch == 0, tb.mismatch
                c_.advance(v_)
                ws = list(tb.workers)
                rows = np.concatenate([v_.rows[w, :v_.length[w]] for w in ws])
                pend = np.concatenate([v_.pending[w, :v_.length[w]] for w in ws])
                with torch.no_grad():
                    out = tb.part(ws).forward(model, TP.to_batch(rows, pend, 'cpu'), need_vg=True)
                at = 0
                for w in ws:
                    n = int(v_.length[w])
                    a, ww = divmod(w, NW)
                    steps = start + np.arange(n)
                    for j, key in ((0, 'logits'), (1, 'value'), (-1, 'vg')):
                        exp = torch.stack([logs[a][key][s][ww] for s in steps])
                        stats['err'] = max(stats['err'], float((out[j][at:at + n] - exp).abs().max()))
                    at += n
                stats['merges'] += 1
            if t < T:   # test 4 on the merged next buffer (its carried context) against each actor's snapshot
                parts = [(a * NW, TT.rollout_scalars(rolls[a][1 - bid]), TT.rollout_tdev(rolls[a][1 - bid]))
                         for a in range(2)]
                nxt, _ = TT.merge_rollouts(fulls[1 - bid], parts, {}, 'cpu')
                for w in range(2 * NW):
                    a, ww = divmod(w, NW)
                    ep0, t0, values, ok = worker_context(nxt, w, W)
                    assert ok, (t, w)
                    recent, mem = snaps[a][ww]
                    if t0:
                        assert torch.equal(values, recent[-min(t0, W):].float()), (t, w, 'source values')
                    got = carry.memory(w, ep0, t0)
                    need = t0 // BLOCK * TOKENS
                    assert got is not None and got.shape[0] == need, (t, w)
                    if need:
                        assert torch.equal(got, mem[:need].float()), (t, w, 'writer tokens')
                    stats['ctx'] += 1
            bid, start = 1 - bid, t - 1
        if t == T:
            break
        for a in range(2):
            rows = seqs[a][t]
            temporal = states[a].step_inputs(model, rows, pends[a][t], np.arange(NW))
            with torch.no_grad():
                out = model(TP.to_batch(rows, pends[a][t], 'cpu'), temporal=temporal, need_vg=True)
            for key, j in (('logits', 0), ('value', 1), ('vg', -1)):
                logs[a][key].append(out[j])
            TT.store(rolls[a][bid], np.arange(NW), rows, pends[a][t], np.zeros((NW, len(model.heads)), np.int64),
                     np.zeros(NW, np.float32), out[1].numpy(), 0, None, out[-1].numpy(),
                     TP.temporal_out(temporal['out']))
    assert stats['err'] <= 1e-5, stats
    assert carry.mismatch == 0 and carry1.mismatch == 0
    print(f"PASS truncated two actors: {stats['ctx']} merged worker contexts == their actor's state bit for bit; "
          f"{stats['merges']} merged / mirrored rollouts: actor = learner max_abs {stats['err']:.2e}; mismatches 0")


if __name__ == '__main__':
    main()
