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
    """ISAAC_RL_TEST_TURBO_SO only (as tests/test_fast_row_v9.py): a libabp_turbo.so built from this tree. Never a
    game home's library found by itself -- the gate's ABP_HOME may hold an older layout's (refused, rightly, by FastRow)."""
    so = os.environ.get('ISAAC_RL_TEST_TURBO_SO', '')
    return so if so and os.path.exists(so) else ''


def probe_inputs(path):
    """Lean payloads, pickled for the children: 8 with a v2 visible block (flag 32; test_obs_v2_smoke's world,
    entities varied, the terrain block on the first) and 30 v3 records (test_fast_row_v9's floor episode)."""
    pytest.importorskip('torch')   # (the test helpers below import it; the children never do)
    import struct
    import test_fast_row_v9 as F
    import test_obs_v2_smoke as V
    from isaac_bridge import abplus_lean as AL
    from isaac_bridge import visible_producer as VP
    from isaac_bridge.schema_v2 import SCHEMA
    obs = []
    for k in range(8):
        p, d, e, room = V.world(0)   # (its first record's world; the entities varied)
        e = np.resize(e, (0, 3, 9, 40, 1, 25, 64, 2)[k])
        vis = json.dumps(dict(schema=SCHEMA, stage=3 + k, curses=8 if k % 2 else 0, capabilities=1, inventory=[],
                              slots=[], bodies=[dict(controller=0, controller_known=1, animation=k % 3,
                                                     animation_known=1) for _ in range(len(p))])).encode()
        flags = 32 | (2 if k == 0 else 0)
        out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
        out += struct.pack('<iiiiiiiiffff', *room)
        out += struct.pack('<dddd', 0.0, 30.0, 30.0, 1.0)
        out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
        out += np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)[:, :148].tobytes()
        if k == 0:
            tj, gj = V.terrain_json()
            out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
        out += struct.pack('<I', len(vis)) + vis
        obs.append(out)
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


@pytest.fixture(scope='module')
def probe_pkl(tmp_path_factory):
    """The children's inputs, built once per module (the helpers that build them import torch and may start its
    pools: before the tests' own thread check, conftest.nothing_left_running)."""
    path = tmp_path_factory.mktemp('lean') / 'inputs.pkl'
    probe_inputs(path)
    return path


@pytest.fixture(scope='module')
def probes(probe_pkl):
    """The child run now (then train_tok imported as __mp_main__) and as a worker before (structured_input first)."""
    so = turbo_so()
    new = run_probe(probe_pkl, False, so, 'mp_main')
    old = run_probe(probe_pkl, False, so, 'old')
    print(f'worker probe: now     {new}', f'worker probe: before  {old}', sep='\n')
    return so, new, old


def test_a_a_worker_holds_no_torch_and_writes_the_same_records(probes):
    so, new, old = probes
    assert not new['torch_after_attach'] and not new['torch'] and not new['torch_after_mp_main']
    assert old['torch']   # (the comparison: a worker as before, with structured_input's torch)
    for key in ('v2', 'v2_bytes', 'v2_native', 'encode_row', 'fast_row', 'fast_row_fast'):
        assert new.get(key) == old.get(key), key
    assert len(set(new['v2_bytes'])) > 1 and min(new['v2_bytes']) > 0   # (real v2 frames, of different lengths)
    if new['private_mb'] is not None:
        assert old['private_mb'] - new['private_mb'] > 150, (old['private_mb'], new['private_mb'])


def test_a_the_native_encoders_compared(probes):
    """(Codex round 29: no silent pass) FastRow (libabp_turbo.so) and the native v2 encoder took records in both runs
    and wrote the same: skipped -- visibly -- where the library is not built (Windows)."""
    so, new, old = probes
    if not so:
        pytest.skip('ISAAC_RL_TEST_TURBO_SO not set (a libabp_turbo.so of this tree): FastRow and native v2 not compared')
    assert new['fast_row_fast'] > 0 and new['fast_row'] == old['fast_row'], (new.get('fast_row_fast'),)
    assert new['v2_native'] and old['v2_native'], 'the native v2 encoder did not load'


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
    for pid, pair in ((0, 0), (5, 3), (123456, 63), (2 ** 40, 7), (987654321, 1)):   # (crn_key as it was in 48f53a4)
        want = int(old(np.array([(int(pid) * 64 + int(pair)) & 0x7FFFFFFFFFFFFFFF], np.uint64))[0])
        assert TB.crn_key(pid, pair) == ((want & 0x7FFFFFFFFFFFFFFF) or 1), (pid, pair)


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


# ---------------------------------------------------------------- (d) the shared memory's layout (Codex round 29)
ABI_COLUMNS = {   # SHARED_ABI -> (stats columns, crc32 of their names in order); a STATS change needs a new row here
    3: (184, 0xfbc81f98),
}


def test_d_the_layout_version_is_bound_to_the_columns():
    """Any change to STATS bumps SHARED_ABI (tok_sampler's rule): the current version's columns are the ones recorded
    for it -- a column added, removed or moved without the bump fails here."""
    import zlib
    assert TS.SHARED_ABI in ABI_COLUMNS, f'SHARED_ABI {TS.SHARED_ABI}: add its row (len(STATS), crc32) to ABI_COLUMNS'
    assert (len(TS.STATS), zlib.crc32('\n'.join(TS.STATS).encode())) == ABI_COLUMNS[TS.SHARED_ABI], \
        'STATS changed without a new SHARED_ABI'
    assert TS.RING_PROTOCOL == 2


def old_layout_blocks(workers):
    """A trainer of the trees before (25c630c's: 182 stats columns, control[3] = (visible schema << 32) | ROW size)."""
    from multiprocessing import shared_memory
    from isaac_bridge.schema_v2 import VERSION
    sizes = (TO.ROW.itemsize * 2 * workers, 4 * TS.ACTION_F * workers, 8 * 182 * workers, 8 * 4)
    blocks = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
    control = np.ndarray((4,), np.int64, buffer=blocks[3].buf)
    control[3] = (VERSION << 32) | TO.ROW.itemsize
    del control
    return blocks


def test_d_a_worker_of_this_tree_refuses_an_older_trainers_memory():
    """A new worker on segments an old trainer made: refused by the layout check, before any view (not by a NumPy
    buffer error that a larger block would not raise)."""
    blocks = old_layout_blocks(2)
    try:
        with pytest.raises(ValueError, match='sampler shared memory: layout'):
            TS.attach([b.name for b in blocks], 2)
    finally:
        for b in blocks:
            b.close()
            b.unlink()


def stub_worker_mixed(index, cfg, names, conn, teacher_names=None, *args, **kw):
    """Worker ISAAC_RL_TEST_OLD_WORKER=<i> runs 25c630c's attach (its layout check, its 182-column stats, no ring
    confirmation) and writes decisions / frames at its stride; the others are this tree's (stub_worker_lean)."""
    if str(index) != os.environ.get('ISAAC_RL_TEST_OLD_WORKER', ''):
        return stub_worker_lean(index, cfg, names, conn, teacher_names)
    from multiprocessing import shared_memory
    from isaac_bridge.schema_v2 import VERSION
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    stats = np.ndarray((cfg.workers, 182), np.float64, buffer=blocks[2].buf)
    control = np.ndarray((4,), np.int64, buffer=blocks[3].buf)
    if int(control[3]) != ((VERSION << 32) | TO.ROW.itemsize):
        raise ValueError('sampler shared-memory visible schema/ROW size mismatch')   # (the old check, as it was)
    stats[index, 0], stats[index, 1] = 17, 68   # (decisions, frames at the old stride)
    del stats, control
    _wait_closed(conn)


def test_d_an_older_worker_ends_the_run_with_the_switch_off(monkeypatch, tmp_path):
    """Two workers, one of the trees before; no ring (no teacher), ISAAC_RL_WORKER_LEAN off: the trainer still waits
    for every confirmation and stops -- before, it started and the old worker wrote its counts into the wrong columns
    (worker 0's np_removed, worker 1's decisions: Codex round 29)."""
    import select
    from isaac_bridge.visible_producer import VisibleV3Fatal
    if not hasattr(select, 'poll'):
        pytest.skip('TokSampler: select.poll (Linux)')
    monkeypatch.setenv('ISAAC_RL_TEST_OLD_WORKER', '1')
    monkeypatch.delenv('ISAAC_RL_TEST_TORCH_WORKER', raising=False)
    monkeypatch.setattr(TS, 'worker_main', stub_worker_mixed)
    monkeypatch.setenv('ISAAC_RL_FATAL_DIR', str(tmp_path))
    monkeypatch.setenv('ISAAC_RL_PNG_SHARED', '0')
    monkeypatch.setenv('ISAAC_RL_RING_CONFIRM_S', '60')
    monkeypatch.setenv('ISAAC_RL_WORKER_LEAN', '0')
    monkeypatch.setattr(TO, 'WORKER_LEAN', False)
    cfg = TS.TokSamplerConfig(workers=2, teacher=False, name='abitest', port=os.getpid() % 30000 + 21000)
    with pytest.raises(VisibleV3Fatal, match=r'worker\(s\) 1 \(exit 1\) ended before confirming'):
        TS.TokSampler(cfg)


# ---------------------------------------------------------------- (e) the actor's weights version (the coordinator)
def stub_worker_version(index, cfg, names, conn, teacher_names=None, *args, **kw):
    """Attach, confirm, then until the pipe closes: the actor's weights version as a worker reads it for its records
    (tok_sampler.actor_version -- what tok_floor writes into branch comps, lab results, CF / V2 points), reported in
    its stats column 'decisions'."""
    import select as select_
    from isaac_bridge import tok_sampler as S_
    blocks, rows, actions, stats, control = S_.attach(names, cfg.workers)
    gb, tg, vg, ack = S_.attach_ring_gen(cfg.ring_gen_block, cfg.workers)
    S_.ring_confirm(ack, index)
    while True:
        stats[index, S_.ST['decisions']] = S_.actor_version(control)
        if select_.select([conn], [], [], 0.01)[0]:
            break
    del tg, vg, ack, rows, actions, stats, control
    _wait_closed(conn)


def test_e_the_workers_record_the_trainers_weights_version(monkeypatch, tmp_path):
    """The trainer publishes the actor's weights version (control[CTRL_VERSION], train_tok's take_weights); a worker
    reads it for its records -- each change seen, equal to the trainer's, and the layout slot [3] untouched (before,
    the version was written into [3] and the records showed the handshake number until the first update)."""
    import select
    import time
    if not hasattr(select, 'poll'):
        pytest.skip('TokSampler: select.poll (Linux)')
    monkeypatch.setenv('ISAAC_RL_FATAL_DIR', str(tmp_path))
    monkeypatch.setenv('ISAAC_RL_PNG_SHARED', '0')
    monkeypatch.setenv('ISAAC_RL_RING_CONFIRM_S', '60')
    monkeypatch.setattr(TS, 'worker_main', stub_worker_version)
    cfg = TS.TokSamplerConfig(workers=2, teacher=False, name='vertest', port=os.getpid() % 30000 + 22000)
    s = TS.TokSampler(cfg)
    try:
        abi = int(s.control[3])
        assert abi == TS.shared_abi()
        for v in (0, 7, 8):
            s.control[TS.CTRL_VERSION] = v   # (as take_weights does)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10 and any(int(x) != v for x in s.stats[:, TS.ST['decisions']]):
                time.sleep(0.01)
            assert [int(x) for x in s.stats[:, TS.ST['decisions']]] == [v, v], v
        assert int(s.control[3]) == abi
    finally:
        s.close(timeout=2.0)


def test_e_no_reader_of_the_old_slot():
    """tok_floor reads the version through actor_version only; train_tok writes it to CTRL_VERSION only."""
    src = (HERE.parent / 'isaac_bridge' / 'tok_floor.py').read_text(encoding='utf-8')
    assert 'control[3]' not in src and src.count('actor_version(control)') == 7
    tt = (HERE.parent / 'train_tok.py').read_text(encoding='utf-8')
    assert 'control[3]' not in tt and tt.count('sampler.control[CTRL_VERSION] = version') == 2
