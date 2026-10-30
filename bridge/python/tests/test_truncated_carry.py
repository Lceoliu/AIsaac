"""--memory-train truncated, test 4 (CPU, the eager actor): the context the learner rebuilds for a rollout equals the
actor's own state.

An eager rebuild actor (TemporalState + ActorMemory) steps 3 workers through 40 decisions (episode starts at step 0
and at RESETS) and stores them into train_tok rollouts A / B, handed over (B.carry(A), the open decision carried) at
HANDOVERS. After each hand-over:
  1. the carried source values before the new rollout's decision 0 (worker_context: the P = W carried rows of t_s)
     equal the actor's recent ring at that decision, bit for bit;
  2. the Writer tokens LearnerCarry holds after advancing over the previous rollout equal the actor's closed blocks
     at that decision, bit for bit (t_m: the tokens the actor wrote where a decision closed a block);
  3. no mismatch is counted.
  python tests/test_truncated_carry.py
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
from isaac_bridge.truncated_memory import LearnerCarry, worker_context

W, BLOCK, TOKENS = 4, 4, 2
RESETS = {(1, 12), (2, 30)}
HANDOVERS = (10, 22, 31)   # (the decision at step h - 1 open, carried as decision 0 of the next rollout)


def main():
    torch.set_num_threads(2); torch.manual_seed(7)
    cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
               temporal_heads=4, temporal_window=W, rebuild_memory=True, memory_block=BLOCK, memory_tokens=TOKENS,
               source_cache=True)
    model = TP.TokPolicy(**cfg).eval()
    rng = np.random.default_rng(7)
    nw, T = 3, 40
    seq = TS.make_seq(rng, nw=nw, T=T, changes={(0, 5), (1, 9), (2, 20)}, resets=RESETS, exits=set())
    pend = [M.synth_pending(rng, nw, model.heads) for _ in range(T)]
    ta = model.tatt
    tcfg = dict(P=W, L=ta.L, D=ta.D, R=ta.R, kv='fp32', device='cpu', truncated=True, tokens=TOKENS)
    rolls = [TT.Rollout(nw, 64, len(model.heads), 0, True, dict(tcfg, rows=None)),
             TT.Rollout(nw, 64, len(model.heads), 0, True, dict(tcfg, rows=7))]   # (B's pool has to grow)
    st = TP.TemporalState(model, nw, 'cpu', torch.float32)
    carry = LearnerCarry(nw, BLOCK, TOKENS, ta.D, 'cpu')
    roll, checks, snap = rolls[0], 0, None
    for t in range(T):
        if t + 1 in HANDOVERS:   # the actor's state before the decision that will be open at the hand-over
            snap = [(st.rebuild.recent[w].clone() if st.rebuild.recent[w] is not None else None,
                     st.rebuild.mem[w].clone() if st.rebuild.mem[w] is not None else None) for w in range(nw)]
        if t in HANDOVERS:
            other = rolls[1] if roll is rolls[0] else rolls[0]
            other.carry(roll)
            carry.advance(roll)
            roll = other
            for w in range(nw):
                ep0, t0, values, ok = worker_context(roll, w, W)
                assert ok, (t, w)
                recent, mem = snap[w]
                exp_r = recent[-min(t0, W):] if t0 else recent.new_zeros((0, ta.D)) if recent is not None else None
                if t0 == 0:
                    assert values.shape[0] == 0, (t, w, values.shape)
                else:
                    assert torch.equal(values, exp_r.float()), (t, w, 'source values')
                got = carry.memory(w, ep0, t0)
                assert got is not None, (t, w, 'carry mismatch')
                need = t0 // BLOCK * TOKENS
                assert got.shape[0] == need, (t, w, got.shape, need)
                if need:
                    assert torch.equal(got, mem[:need].float()), (t, w, 'writer tokens')
                checks += 1
        rows = seq[t]
        temporal = st.step_inputs(model, rows, pend[t], np.arange(nw))
        with torch.no_grad():
            out = model(TP.to_batch(rows, pend[t], 'cpu'), temporal=temporal, need_vg=True)
        TT.store(roll, np.arange(nw), rows, pend[t], np.zeros((nw, len(model.heads)), np.int64),
                 np.zeros(nw, np.float32), out[1].numpy(), 0, None, out[-1].numpy(),
                 TP.temporal_out(temporal['out']))
    assert carry.mismatch == 0, carry.mismatch
    print(f'PASS truncated carry: {checks} worker contexts at {len(HANDOVERS)} hand-overs equal the actor state '
          f'(source values and Writer tokens bit for bit), mismatches {carry.mismatch}')


if __name__ == '__main__':
    main()
