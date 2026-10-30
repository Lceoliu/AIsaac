"""--memory-train truncated on CUDA: the graph actor's values reach the learner (tests 4 and 1).

RebuildGraphActor (the production actor: CUDA graphs, TensorActorMemory) steps each actor's 3 workers through 40
decisions (episode starts at step 0 and at RESETS) and writes every decision's source value and closed blocks' Writer
tokens into train_tok rollouts on its GPU (tdest; control row 21), handed over at HANDOVERS. The learner on its own
GPU gets the finished buffer as train_tok's learner process does: one actor -> mirror_temporal (hand_off), two
actors -> merge_rollouts (both actors' pools, cross-device). Checks:
  test 4: every worker's carried source values (t_s) and LearnerCarry's Writer tokens (t_m) == the graph actor's
          TensorActorMemory (ring values, closed blocks) before the rollout's first decision, bit for bit;
  test 1: the learner's batched outputs over each finished rollout == the actor's outputs of the same decisions
          (logits, value, value_global): FP32 with TF32 off: max abs <= --threshold (default 1e-6); --bf16 (actor and
          learner under BF16 autocast): max abs <= 2 x the BF16 noise floor (the BF16 actor against the FP32 learner
          on the same constants) + 1e-6;
  no context mismatch, no constant with a gradient.
  python tests/test_truncated_actor_cuda.py [--bf16] [--actors cuda:0[,cuda:1]] [--learner cuda:2] [--threshold 1e-6]
"""
import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
import train_tok as TT
from isaac_bridge import tok_policy as TP
from isaac_bridge.rebuild_actor import RebuildGraphActor
from isaac_bridge.truncated_memory import LearnerCarry, TruncatedBatch, worker_context

W, BLOCK, TOKENS = 4, 4, 2
NW, T = 3, 40
HANDOVERS = (10, 22, 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--actors', default='cuda:0')
    ap.add_argument('--learner', default='cuda:0')
    ap.add_argument('--threshold', type=float, default=1e-6)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=W, rebuild_memory=True, memory_block=BLOCK, memory_tokens=TOKENS,
               source_cache=True)
    base = TP.TokPolicy(**cfg).eval()
    devs = [torch.device(d) for d in args.actors.split(',')]
    ldev = torch.device(args.learner)
    A = len(devs)
    models = [TP.TokPolicy(**cfg).to(d).eval() for d in devs]
    for m in models:
        m.load_state_dict(base.state_dict())
    learner = TP.TokPolicy(**cfg).to(ldev).eval()
    learner.load_state_dict(base.state_dict())
    ta = base.tatt
    seqs, pends = [], []
    for a in range(A):
        rng = np.random.default_rng(170 + a)
        seqs.append(TS.make_seq(rng, nw=NW, T=T, changes={(0, 5), (1, 9)},
                                resets={(1, 12), (2, 30)} if a == 0 else {(0, 17)}, exits=set()))
        pends.append([M.synth_pending(rng, NW, base.heads) for _ in range(T)])
    rolls, states, actors = [], [], []
    for a, d in enumerate(devs):
        tcfg = dict(P=W, L=ta.L, D=ta.D, R=ta.R, kv='fp32', device=str(d), truncated=True, tokens=TOKENS)
        rolls.append([TT.Rollout(NW, 64, len(base.heads), 0, True, dict(tcfg, rows=None)),
                      TT.Rollout(NW, 64, len(base.heads), 0, True, dict(tcfg, rows=7))])
        with torch.cuda.device(d):
            states.append(TP.TemporalState(models[a], NW, d, torch.float32))
            actors.append(RebuildGraphActor(models[a], NW, entities=[8, 16], temporal=states[a], row_sizes=[NW],
                                            bf16=args.bf16))
    fulls = None
    if A > 1:
        fulls = [TT.full_from_local(r, A * NW, 0, np.zeros) for r in rolls[0]]
        for a in range(1, A):
            for bid, r in enumerate(rolls[a]):
                TT.slice_into(r, fulls[bid], a * NW)
    carry = LearnerCarry(A * NW, BLOCK, TOKENS, ta.D, ldev)
    logs = [dict(logits=[], value=[], vg=[]) for _ in range(A)]
    mirror = {}
    bid, start, snaps = 0, 0, None
    st = dict(ctx=0, err=0.0, floor=0.0, rollouts=0)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if args.bf16 else nullcontext()

    def view_of(b):
        if A > 1:
            parts = [(a * NW, TT.rollout_scalars(rolls[a][b]), TT.rollout_tdev(rolls[a][b])) for a in range(A)]
            with torch.cuda.device(ldev):
                return TT.merge_rollouts(fulls[b], parts, {}, ldev)[0]
        return TT.mirror_temporal(rolls[0][b], mirror, ldev)[0]

    for t in range(T + 1):
        if t + 1 in HANDOVERS or t + 1 == T:   # the graph actors' state before the decision carried open
            snaps = []
            for a in range(A):
                torch.cuda.synchronize(devs[a])
                fast = states[a].rebuild.fast
                snaps.append([(fast.u[s].clone(), fast.mem[s].clone(), int(fast.length[s])) for s in range(NW)])
        if t in HANDOVERS or t == T:
            for a in range(A):
                torch.cuda.synchronize(devs[a])
                rolls[a][1 - bid].carry(rolls[a][bid])
            v_ = view_of(bid)
            torch.cuda.synchronize(ldev)
            tb = TruncatedBatch(learner, v_, carry, ldev)
            assert tb.mismatch == 0, tb.mismatch
            carry.advance(v_)
            ws = list(tb.workers)
            rows = np.concatenate([v_.rows[w, :v_.length[w]] for w in ws])
            pend = np.concatenate([v_.pending[w, :v_.length[w]] for w in ws])
            with torch.no_grad():
                with amp:
                    out = tb.part(ws).forward(learner, TP.to_batch(rows, pend, ldev), need_vg=True)
                ref32 = tb.part(ws).forward(learner, TP.to_batch(rows, pend, ldev), need_vg=True) if args.bf16 else None
            assert all(not c.requires_grad for c in tb.constants())
            at = 0
            for w in ws:
                n = int(v_.length[w])
                a, ww = divmod(w, NW)
                steps = start + np.arange(n)
                for j, key in ((0, 'logits'), (1, 'value'), (-1, 'vg')):
                    exp = torch.stack([logs[a][key][s][ww] for s in steps]).to(ldev).float()
                    st['err'] = max(st['err'], float((out[j][at:at + n].float() - exp).abs().max()))
                    if ref32 is not None:
                        st['floor'] = max(st['floor'], float((ref32[j][at:at + n].float() - exp).abs().max()))
                at += n
            st['rollouts'] += 1
            if t < T:   # test 4 on the next buffer's carried context
                nxt = view_of(1 - bid)
                torch.cuda.synchronize(ldev)
                for w in range(A * NW):
                    a, ww = divmod(w, NW)
                    ep0, t0, values, ok = worker_context(nxt, w, W)
                    assert ok, (t, w)
                    u_ring, mem, length = snaps[a][ww]
                    assert length == t0, (t, w, length, t0)
                    m = min(t0, W)
                    if m:
                        ring = u_ring[torch.arange(t0 - m, t0, device=u_ring.device) % W].to(ldev)
                        assert torch.equal(values, ring), (t, w, 'source values')
                    got = carry.memory(w, ep0, t0)
                    need = t0 // BLOCK * TOKENS
                    assert got is not None and got.shape[0] == need, (t, w)
                    if need:
                        assert torch.equal(got, mem[:need].to(ldev)), (t, w, 'writer tokens')
                    st['ctx'] += 1
            bid, start = 1 - bid, t - 1
        if t == T:
            break
        for a in range(A):
            rows = seqs[a][t]
            roll = rolls[a][bid]
            with torch.cuda.device(devs[a]):
                tdest = roll.t_dest(np.arange(NW))
                assert actors[a].submit(rows, pends[a][t], widx=np.arange(NW), tdest=tdest)
                act, lp, val = actors[a].result()
            logs[a]['logits'].append(actors[a].last_logits.clone())
            logs[a]['value'].append(torch.as_tensor(val, device=devs[a]))
            logs[a]['vg'].append(torch.as_tensor(actors[a].vg, device=devs[a]))
            TT.store(roll, np.arange(NW), rows, pends[a][t], act, lp, val, 0, None, actors[a].vg, actors[a].t_out)
    limit = 2 * st['floor'] + 1e-6 if args.bf16 else args.threshold
    print(f"truncated CUDA ({'BF16' if args.bf16 else 'FP32, TF32 off'}; actors {args.actors}, learner {args.learner}): "
          f"{st['ctx']} worker contexts == the graph actor's state bit for bit; {st['rollouts']} rollouts: actor = "
          f"learner max_abs {st['err']:.3e}" + (f" (BF16 floor {st['floor']:.3e}, limit {limit:.3e})" if args.bf16 else
                                                 f" (limit {limit:.1e})") + f"; mismatches {carry.mismatch}", flush=True)
    assert carry.mismatch == 0
    assert st['err'] <= limit, (st, limit)
    print('PASS')


if __name__ == '__main__':
    main()
