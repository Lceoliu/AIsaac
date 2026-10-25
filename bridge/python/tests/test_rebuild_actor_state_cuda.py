"""RebuildGraphActor writes the temporal state's per-slot outputs inside its core graph (CUDA).

After every call: prev_o of each record's slot is the o the call put into the rollout (tdest),
steps is its decision index + 1, prev_stage its record's stage; other slots unchanged; the
rollout's step K / V rows stay zero.
  python tests/test_rebuild_actor_state_cuda.py [--compile]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy, TemporalState, GraphActor

torch.manual_seed(5)
compile_mode = 'default' if '--compile' in sys.argv else None
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
           temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).cuda().eval()
st = TemporalState(model, 6, 'cuda', torch.float32)
actor = GraphActor(model, 4, [8, 16], temporal=st, row_sizes=[2, 4], fold=True, compile_mode=compile_mode,
                   sdpa_math=compile_mode is not None)
rng = np.random.default_rng(5)
seq = TS.make_seq(rng, nw=4, T=40, changes={(0, 4), (0, 9)}, resets=set(), exits=set())
D = st.D
td = dict(kv=torch.zeros((64, st.L, 2, D), device='cuda'), o=torch.zeros((64, D), device='cuda'),
          s=torch.zeros((64, D), device='cuda'), obase=0)
row = 0
with torch.no_grad():
    for t, rows in enumerate(seq[:30]):
        n = [4, 3, 2, 4][t % 4]; rows = rows[:n].copy()
        slots = rng.permutation(4)[:n]
        pending = M.synth_pending(rng, n, model.heads)
        before = (st.prev_o.clone(), st.steps.clone(), st.prev_stage.clone())
        td['rows'] = np.arange(row, row + n) % 64; row += n
        actor.submit(rows, pending, widx=slots, tdest=td, ukeys=(np.arange(1, n + 1, dtype=np.uint64), np.full(n, t, dtype=np.uint64)))
        actor.result()
        s_ = torch.as_tensor(slots, device='cuda'); r_ = torch.as_tensor(td['rows'], device='cuda')
        assert torch.equal(st.prev_o[s_], td['o'][r_]), t
        assert np.array_equal(st.steps[s_].cpu().numpy(), actor.t_out['steps']), t
        assert np.array_equal(st.prev_stage[s_].cpu().numpy(), rows['stage'].astype(np.int64)), t
        other = np.setdiff1d(np.arange(st.scratch), slots)
        o_ = torch.as_tensor(other, device='cuda')
        for a, b in zip(before, (st.prev_o, st.steps, st.prev_stage)):
            assert torch.equal(a[o_], b[o_]), ('another slot changed', t)
        assert not td['kv'].any(), 'the rollout step K / V were written'
print('PASS RebuildGraphActor temporal state written in the graph: prev_o, steps, prev_stage; other slots and K / V untouched')
