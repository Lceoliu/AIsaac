"""The host seconds of tok_aux's tables on a rollout of C99's size (64 workers, 32,768 stored records of full ROW size),
without the game. usage: python -X utf8 tests/aux_table_bench.py"""
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402,F401  (the path set-up, the gymnasium stand-in)

import numpy as np   # noqa: E402

from isaac_bridge import tok_aux as A   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402

W, L, CAP = 64, 512, 576
rng = np.random.default_rng(1)
r = types.SimpleNamespace(n=W, cap=CAP, rows=np.zeros((W, CAP), ROW), length=np.full(W, L, np.int64),
                          open=np.full(W, L - 1, np.int64), terminal=np.zeros((W, CAP), bool), boot=np.zeros((W, CAP), bool),
                          reward=rng.normal(size=(W, CAP)).astype(np.float32), close_hurt=np.zeros((W, CAP), np.float32),
                          close_done=np.zeros((W, CAP), np.uint8), close_ev=np.zeros((W, CAP), np.uint8))
r.rows['n_ent'] = rng.integers(0, 40, (W, CAP))
r.rows['first'][:, 0] = 1
r.rows['ent'][:, :, :, A.NPC_COLUMN] = rng.random((W, CAP, 64)) < 0.2
r.rows['map'][:, :, 3, 6, 6] = 1
r.terminal[rng.random((W, CAP)) < 0.002] = True
print(f'rollout: {W} workers x {L} records, ROW {ROW.itemsize} bytes')
H = (1, 8, 64, 256)
norm = A.AuxNorm(len(A.STATE), len(H))
flat = (np.arange(W)[:, None] * CAP + np.arange(L)[None]).reshape(-1)
weight = np.ones(len(flat), np.float32)
for i in range(4):
    t0 = time.perf_counter()
    tg, log = A.build_targets(r, flat, weight, H, norm, 'cpu')
    print(f'build_targets: {1000 * (time.perf_counter() - t0):.0f} ms; masked events {[round(log[f"aux_me_{h}"], 3) for h in H]},'
          f' state {[round(log[f"aux_ms_{h}"], 3) for h in H]}')
