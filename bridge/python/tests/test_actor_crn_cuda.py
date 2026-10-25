"""RebuildGraphActor's sampling uniforms (CUDA): common random numbers only for branch records.

Records with ROW crn != 0 get crn_uniforms(crn, t); records with crn == 0 get fresh uniforms (two
of them at the same decision index differ, and neither equals crn_uniforms(0, t)), as GraphActor.
  python tests/test_actor_crn_cuda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy, TemporalState, GraphActor, crn_uniforms

torch.manual_seed(9)
cfg = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
           temporal_heads=4, temporal_window=4, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)
model = TokPolicy(**cfg).cuda().eval()
st = TemporalState(model, 6, 'cuda', torch.float32)
actor = GraphActor(model, 4, [8, 16], temporal=st, row_sizes=[2, 4], fold=True, crn=True)
rng = np.random.default_rng(9)
seq = TS.make_seq(rng, nw=4, T=6, changes=set(), resets=set(), exits=set())
H = sum(model.heads)
with torch.no_grad():
    for t, rows in enumerate(seq):
        rows = rows.copy(); rows['t'] = 7   # every record at the same decision index
        rows['crn'] = [0, 0, 1234567 + t, 0]
        actor.submit(rows, M.synth_pending(rng, 4, model.heads), widx=np.arange(4)); actor.result()
        u = actor.uniform[:4].cpu().numpy()
        assert np.array_equal(u[2], crn_uniforms(rows['crn'][2:3], rows['t'][2:3], H)[0]), t
        same = crn_uniforms(np.zeros(1, np.int64), np.full(1, 7), H)[0]
        assert not np.array_equal(u[0], same) and not np.array_equal(u[1], same), 'a plain record used crn_uniforms(0, t)'
        assert not np.array_equal(u[0], u[1]), 'two plain records at the same t share their noise'
print('PASS branch records: crn_uniforms(crn, t); plain records: fresh uniforms (not shared across workers)')
