"""ISAAC_ACTOR_TORCH_PROFILE (actor_profile.py) on both actors (CUDA).

A profiled actor and an identical unprofiled twin get the same calls: every
call's actions and logits are equal (the window and the eager attribution pass
change nothing the policy reads), and the profile files hold the window.
  python tests/test_actor_profile_cuda.py [--compile] [--kinds rebuild,graph]
"""
import copy, json, os, sys, tempfile
from pathlib import Path
OUT = tempfile.mkdtemp(prefix='actorprof-')
os.environ.update(ISAAC_ACTOR_TORCH_PROFILE=OUT, ISAAC_ACTOR_TORCH_PROFILE_AT='5', ISAAC_ACTOR_TORCH_PROFILE_COUNT='10',
                  ISAAC_ACTOR_TORCH_PROFILE_EAGER='3')
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy, TemporalState, GraphActor

compile_mode = 'default' if '--compile' in sys.argv else None
base = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2,
            temporal_heads=4, temporal_window=4)
kinds = sys.argv[sys.argv.index('--kinds') + 1].split(',') if '--kinds' in sys.argv else ['rebuild', 'graph']
for kind, cfg in (('rebuild', dict(base, rebuild_memory=True, memory_block=4, memory_tokens=2, source_cache=True)),
                  ('graph', base)):
    if kind not in kinds: continue
    import isaac_bridge.actor_profile as AP   # (one window per process: a fresh one for each kind)
    AP.PROCESS.update(owner=None, main=0, window_main0=0, others={})
    torch.manual_seed(23)
    model = TokPolicy(**cfg).cuda().eval(); twin = copy.deepcopy(model)
    actors = []
    for m in (model, twin):
        st = TemporalState(m, 6, 'cuda', torch.float32)
        actors.append(GraphActor(m, 4, [8, 16], temporal=st, row_sizes=[2, 4], fold=True, compile_mode=compile_mode,
                                 sdpa_math=compile_mode is not None))
    actors[0].prof_role = 'main'
    rng = np.random.default_rng(23)
    seq = TS.make_seq(rng, nw=4, T=40, changes={(0, 4), (0, 8)}, resets=set(), exits=set())
    with torch.no_grad():
        for t, rows in enumerate(seq[:24]):
            n = [4, 3, 2, 4][t % 4]; rows = rows[:n].copy()
            pending = M.synth_pending(rng, n, model.heads)
            keys = (np.arange(1, n + 1, dtype=np.uint64), np.full(n, t, dtype=np.uint64))
            got = []
            for a in actors:
                a.submit(rows, pending, widx=np.arange(n), ukeys=keys); got.append(a.result())
            np.testing.assert_array_equal(got[0][0], got[1][0])
            np.testing.assert_array_equal(got[0][1], got[1][1])
            np.testing.assert_array_equal(got[0][2], got[1][2])
    files = sorted(p.name for p in Path(OUT).glob(f'actor-{kind}-*'))
    assert any(f.endswith('.json') and not f.endswith('eager.json') for f in files) and any(f.endswith('.eager.json') for f in files) \
        and any(f.endswith('.trace.json.gz') for f in files), files
    s = json.loads(next(Path(OUT).glob(f'actor-{kind}-*[0-9].json')).read_text())
    e = json.loads(next(Path(OUT).glob(f'actor-{kind}-*.eager.json')).read_text())
    assert s['calls'] == 10 and s['per_call']['latency_ms'] > 0 and 'h.core_replay' in s['host_sections_ms_per_call'], s.keys()
    assert e['eager_calls'] == 3 and 'p.core_eager' in e['parts_ms_per_call'], e['parts_ms_per_call'].keys()
    print('PASS', kind, json.dumps(dict(per_call=s['per_call'], sections=sorted(s['host_sections_ms_per_call']),
                                        kernel_classes=s['kernel_classes_ms_per_call'],
                                        parts={k: round(v['device_ms_per_call'], 4) for k, v in e['parts_ms_per_call'].items()})))
