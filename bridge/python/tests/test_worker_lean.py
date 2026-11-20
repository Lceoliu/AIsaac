"""(2026-11-20, the memory ledger) The sampler's workers without torch. A worker loaded torch (about 220 MB private per
process) only for names it took from structured_input, a torch module that re-exports them; they now come from their
own torch-free modules (the same objects) and splitmix64's finaliser lives in tok_obs. Every worker reports whether it
holds torch (ring confirmation, stats torch_held); ISAAC_RL_WORKER_LEAN=1 (default off) enforces it. Checked here:
  (a) the regression guard: a fresh process doing a worker's own work (tests/lean_probe_child.py: the shared-memory
      handshake, v2 visible frames, v3 records through encode_row and, given a libabp_turbo.so, FastRow), then train_tok
      imported as spawn imports it (__mp_main__): no torch -- whoever adds a torch import on a worker path fails here;
      the same records byte for byte as a process that loaded structured_input first (the worker before), and the
      private memory that saves (Linux);
  (b) the names are the same objects (structured_input's re-exports and their sources; tok_policy._mix64);
  (c) the confirmation and the loop check: with the switch on the trainer refuses a worker holding torch or with the
      switch off, and starts with clean ones; with it off it starts and logs the count (a real TokSampler, stub workers
      spawned: Linux).
This module imports no torch at its top level: the spawned stub workers import it."""
import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge import tok_sampler as TS   # noqa: E402


def turbo_so():
    so = os.environ.get('ISAAC_RL_TEST_TURBO_SO') or os.path.join(os.environ.get('ABP_HOME', ''), 'tools',
                                                                  'libabp_turbo.so')
    return so if so and os.path.exists(so) else ''


def probe_inputs(path):
    """v2 observations (test_visible_review's packer world, entities varied) and 30 v3 lean payloads
    (test_fast_row_v9's floor episode), pickled for the children."""
    pytest.importorskip('torch')   # (the test helpers below import it; the children never do)
    import test_fast_row_v9 as F
    from isaac_bridge import visible_producer as VP
    from isaac_bridge.abplus_lean import ENTITY
    from isaac_bridge.schema_v2 import P_FIELDS, SCHEMA
    obs = []
    for ne in (0, 3, 40, 170):
        players = [dict.fromkeys(P_FIELDS, 0) for _ in range(2)]
        for p in players:
            p.update(hearts=6, max_hearts=8, soul=2, black=1, bone=1, eternal=1, golden=1)
        players[1].update(x=20)
        ents = np.zeros(ne, dtype=ENTITY)
        ents['type'][:] = 10
        ents['kind'][:] = 6
        ents['x'] = np.arange(ne)[::-1]
        obs.append(types.SimpleNamespace(
            players=players, entities=ents, doors=[dict(x=0, y=0, open=1, locked=0, target_type=1, seen=3)],
            grid=[(0, 17, 0, 0, 0, 0, 0)], room=[0, 0, 0, 0, 0, 7, 0, 0, 0, 0], map=[],
            terrain=dict(width=1, height=1, cells=[(0, 0, 0, 0, 1, 1, 0, 0, 0, 0)]), visible_v3_frame=None,
            visible=dict(schema=SCHEMA, stage=7, curses=(1 << 31) | 8 | 64, capabilities=1, inventory=[], slots=[],
                         bodies=[dict(controller=i, controller_known=1, animation=0, animation_known=1)
                                 for i in range(2)])))
    old = VP.VISIBLE_V3_MODE
    VP.VISIBLE_V3_MODE = 2
    try:
        frame = VP.observation_frame_v3(F._raw())
        sc = F.scenario('off', False)
        pls = [F.payload(k, v3=True, frame=frame, **sc) for k in range(F.N)]
    finally:
        VP.VISIBLE_V3_MODE = old
    with open(path, 'wb') as f:
        pickle.dump(dict(v2=obs, v3=pls), f)


def run_probe(inputs, lean, so, *flags):
    env = dict(os.environ, ISAAC_RL_WORKER_LEAN='1' if lean else '0')
    out = subprocess.run([sys.executable, str(HERE / 'lean_probe_child.py'), str(inputs), so, *flags], env=env,
                         capture_output=True, text=True, timeout=600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith('LEAN_PROBE ')]
    assert out.returncode == 0 and lines, (out.returncode, out.stdout[-2000:], out.stderr[-4000:])
    return json.loads(lines[-1][len('LEAN_PROBE '):])


def test_a_a_worker_holds_no_torch_and_writes_the_same_records(tmp_path):
    inputs = tmp_path / 'inputs.pkl'
    probe_inputs(inputs)
    so = turbo_so()
    new = run_probe(inputs, False, so, 'mp_main')
    old = run_probe(inputs, False, so, 'old')
    print(f'worker probe: now     {new}', f'worker probe: before  {old}', sep='\n')
    assert not new['torch_after_attach'] and not new['torch'] and not new['torch_after_mp_main']
    assert old['torch']   # (the comparison: a worker as before, with structured_input's torch)
    for key in ('v2', 'v2_bytes', 'encode_row', 'fast_row', 'fast_row_fast'):
        assert new.get(key) == old.get(key), key
    assert len(set(new['v2_bytes'])) > 1 and min(new['v2_bytes']) > 0   # (real v2 frames, of different lengths)
    if so:
        assert new['fast_row_fast'] > 0   # (the native encoder took records)
    if new['private_mb'] is not None:
        assert old['private_mb'] - new['private_mb'] > 150, (old['private_mb'], new['private_mb'])


def test_b_the_same_objects():
    """The names a worker now takes from schema_v2 / visible_producer are structured_input's (it re-exports them);
    tok_policy's splitmix64 is tok_obs.mix64 and gives the numbers it gave."""
    pytest.importorskip('torch')
    from isaac_bridge import schema_v2 as S2
    from isaac_bridge import structured_input as SI
    from isaac_bridge import tok_branch as TB
    from isaac_bridge import tok_policy as TP
    from isaac_bridge import visible_native as VN
    from isaac_bridge import visible_producer as VP
    for name in ('VERSION', 'FRAME_BYTES', 'SCHEMA'):
        assert getattr(SI, name) is getattr(S2, name) and VN._si(name)[0] is getattr(SI, name), name
    for name in ('write_frame', 'observation_frame'):
        assert getattr(SI, name) is getattr(VP, name), name
    assert VN._si('observation_frame')[0] is SI.observation_frame
    assert TP._mix64 is TO.mix64

    def old(z):   # (tok_policy._mix64 as it was)
        with np.errstate(over='ignore'):
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            return z ^ (z >> np.uint64(31))
    z = np.random.default_rng(0).integers(0, 2 ** 63, 1000, dtype=np.int64).astype(np.uint64)
    assert np.array_equal(TO.mix64(z), old(z))
    for pid, pair in ((0, 0), (5, 3), (123456, 63), (2 ** 40, 7)):
        want = int(old(np.array([(int(pid) * 64 + int(pair)) & 0x7FFFFFFFFFFFFFFF], np.uint64))[0])
        got = TB.crn_key(pid, pair)
        assert got == TB.crn_key(pid, pair) and isinstance(got, (int, np.integer))
        assert int(TO.mix64(np.array([(pid * 64 + pair) & 0x7FFFFFFFFFFFFFFF], np.uint64))[0]) == want


def test_c_confirmation_values():
    on, off = TS.ring_ack_value(False, True), TS.ring_ack_value(False, False)
    assert TS.ring_ack_ok(TS.ring_ack_value(False, True, False), on)
    assert not TS.ring_ack_ok(TS.ring_ack_value(False, True, True), on)        # (torch held: refused)
    assert not TS.ring_ack_ok(TS.ring_ack_value(False, False, False), on)      # (the switch off in the worker)
    assert not TS.ring_ack_ok(TS.ring_ack_value(False, True, False), off)      # (the switch on in the worker only)
    assert TS.ring_ack_ok(TS.ring_ack_value(False, False, True), off)          # (switch off: torch as before)
    assert not TS.ring_ack_ok(TS.ring_ack_value(True, False, True), off)       # (the ring mode, as before)
    assert 'torch loaded' in TS.ring_ack_text(TS.ring_ack_value(False, True, True))


def test_c_loop_check(monkeypatch, capfd):
    """Torch loaded in a running worker: counted (stats torch_held) and warned once with the worker's index, the
    switch on or off -- never the run's end (the coordinator)."""
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(TO, '_TORCH_WARNED', [])
    for lean in (False, True):
        monkeypatch.setattr(TO, 'WORKER_LEAN', lean)
        assert TO.check_worker_torch(3) == 1
    err = capfd.readouterr().err
    assert err.count('tok worker 3: WARNING torch was loaded') == 1
    monkeypatch.delitem(sys.modules, 'torch')
    assert TO.check_worker_torch(3) == 0
    assert 'torch_held' in TS.ST and TS.STATS[-2:] == ('np_runs', 'np_removed')


def _wait_closed(conn):
    try:
        conn.recv_bytes()
    except (EOFError, OSError):
        pass


def stub_worker_lean(index, cfg, names, conn, teacher_names=None, *args, **kw):
    """A worker of this tree up to its games (attach, confirm, the mode check); ISAAC_RL_TEST_TORCH_WORKER=<i>: worker
    i imports torch first (a worker-side torch import the switch must catch)."""
    if str(index) == os.environ.get('ISAAC_RL_TEST_TORCH_WORKER', ''):
        import torch   # noqa: F401
    from isaac_bridge import tok_sampler as S_
    blocks, rows, actions, stats, control = S_.attach(names, cfg.workers)
    gb, tg, vg, ack = S_.attach_ring_gen(cfg.ring_gen_block, cfg.workers)
    S_.ring_confirm(ack, index)
    del tg, vg, ack, rows, actions, stats, control
    S_.check_ring_mode(cfg)
    _wait_closed(conn)


def lean_sampler(monkeypatch, tmp_path, worker_lean_env, trainer_lean=True):
    import select
    if not hasattr(select, 'poll'):
        pytest.skip('TokSampler: select.poll (Linux)')
    monkeypatch.setenv('ISAAC_RL_FATAL_DIR', str(tmp_path))
    monkeypatch.setenv('ISAAC_RL_PNG_SHARED', '0')
    monkeypatch.setenv('ISAAC_RL_RING_CONFIRM_S', '60')
    monkeypatch.setenv('ISAAC_RL_WORKER_LEAN', worker_lean_env)   # (the workers' own environment)
    monkeypatch.setattr(TO, 'WORKER_LEAN', trainer_lean)           # (the trainer's)
    monkeypatch.setattr(TS, 'worker_main', stub_worker_lean)
    cfg = TS.TokSamplerConfig(workers=2, teacher=not trainer_lean, name='leantest', port=os.getpid() % 30000 + 20000)
    return TS.TokSampler(cfg)


def test_c_a_worker_holding_torch_ends_the_run(monkeypatch, tmp_path):
    """The switch on, no rings (no teacher): the trainer still waits for the confirmations; worker 1 imported torch:
    the run ends with it named -- not the worker dropped, not the run going on heavier."""
    from isaac_bridge.visible_producer import VisibleV3Fatal
    monkeypatch.setenv('ISAAC_RL_TEST_TORCH_WORKER', '1')
    with pytest.raises(VisibleV3Fatal, match=r'worker 1: .*torch loaded'):
        lean_sampler(monkeypatch, tmp_path, '1')


def test_c_a_worker_without_the_switch_ends_the_run(monkeypatch, tmp_path):
    from isaac_bridge.visible_producer import VisibleV3Fatal
    monkeypatch.delenv('ISAAC_RL_TEST_TORCH_WORKER', raising=False)
    with pytest.raises(VisibleV3Fatal, match=r'ISAAC_RL_WORKER_LEAN 0'):
        lean_sampler(monkeypatch, tmp_path, '0')


def test_c_clean_workers_start(monkeypatch, tmp_path, capfd):
    monkeypatch.delenv('ISAAC_RL_TEST_TORCH_WORKER', raising=False)
    s = lean_sampler(monkeypatch, tmp_path, '1')
    try:
        want = TS.ring_ack_value(TO.RING_COMPACT, True, False)
        assert [int(a) for a in s.ring_ack] == [want] * 2
        assert 'ISAAC_RL_WORKER_LEAN 1, torch not loaded' in capfd.readouterr().err
    finally:
        s.close(timeout=2.0)


def test_c_switch_off_counts_and_starts(monkeypatch, tmp_path, capfd):
    """The switch off (the default), the teacher's ring on: a worker holding torch is not refused -- the run starts
    and the log says how many workers hold it (the memory ledger's watch; the switch decides the refusal)."""
    monkeypatch.setenv('ISAAC_RL_TEST_TORCH_WORKER', '1')
    s = lean_sampler(monkeypatch, tmp_path, '0', trainer_lean=False)
    try:
        assert int(s.ring_ack[1]) & TS.ACK_TORCH and not int(s.ring_ack[0]) & TS.ACK_TORCH
        assert 'workers holding torch: 1/2' in capfd.readouterr().err
    finally:
        s.close(timeout=2.0)
