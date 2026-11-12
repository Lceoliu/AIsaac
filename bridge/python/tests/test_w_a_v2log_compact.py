"""W-a (2026-11-12): V2Log keeps a record's ROW compactly (header + the used part of the visible frame) under
ISAAC_RL_V2LOG_COMPACT=1; a replay (replay_value -> row_into) puts into the row buffer exactly the bytes the actor reads
(the header and visible_bytes of the frame), equal to the whole-copy path's; B: the worker CPU helpers (wait_child's
child CPU, name_thread) count only, and the stats keep A28's columns last."""
import os
import sys

import numpy as np
import pytest

from isaac_bridge import tok_teacher2 as t2
from isaac_bridge.tok_obs import ROW, ROW_FRAME_AT


def records(rng, n):
    rows = np.zeros(n, ROW)
    b = rows.view(np.uint8).reshape(n, ROW.itemsize)
    b[:] = rng.integers(0, 256, b.shape, dtype=np.uint8)   # (stale bytes everywhere, the frame tail too)
    sizes = rng.integers(0, 131072, n)
    sizes[0], sizes[1] = 0, 131072
    rows['visible_bytes'] = sizes
    return rows


@pytest.mark.parametrize('compact', [False, True])
def test_replayed_rows_hold_the_actors_bytes(monkeypatch, compact):
    monkeypatch.setattr(t2, 'V2LOG_COMPACT', compact)
    rng = np.random.default_rng(5)
    rows = records(rng, 12)
    log = t2.V2Log()
    slot = np.zeros(1, ROW)
    for i in range(len(rows)):
        slot[:] = rows[i:i+1]   # (the worker's slot, rewritten every record)
        log.add_row(slot)
    if compact:
        assert all(isinstance(r, bytes) for r in log.rows)
        assert len(log.rows[0]) == ROW_FRAME_AT and len(log.rows[1]) == ROW_FRAME_AT+131072
    buf = np.zeros(3, ROW)
    view = buf[1:2]   # (the lane slot: a length-1 contiguous view)
    view.view(np.uint8)[:] = 0xAB
    for i in rng.permutation(len(rows)):
        log.row_into(int(i), view)
        n = int(rows['visible_bytes'][i])
        for name in ROW.names:   # (every field the actor reads; the frame up to visible_bytes. Padding bytes between
            #                       fields are not data: a structured copy does not carry them either)
            if name == 'visible_frame':
                assert view['visible_frame'][0][:n].tobytes() == rows['visible_frame'][i][:n].tobytes(), i
            else:   # (bytes: random float bits include NaNs)
                assert np.asarray(view[name][0]).tobytes() == np.asarray(rows[name][i]).tobytes(), (i, name)
    assert (buf[0:1].view(np.uint8) == 0).all() and (buf[2:3].view(np.uint8) == 0).all()   # (the neighbours untouched)


def test_row_into_refuses_a_copy(monkeypatch):
    monkeypatch.setattr(t2, 'V2LOG_COMPACT', True)
    log = t2.V2Log()
    log.add_row(np.zeros(1, ROW))
    with pytest.raises(ValueError):
        log.row_into(0, np.zeros(4, ROW)[::2])


def test_stats_keep_a28_last_and_cpu_columns():
    from isaac_bridge.tok_sampler import STATS
    assert STATS[-2:] == ('np_runs', 'np_removed')
    for k in ('enc_cpu', 'book_cpu', 'wait_cpu', 'step_cpu', 'teach_thr_cpu', 'cf_thr_cpu', 'search_child_cpu', 'cf_child_cpu'):
        assert k in STATS


@pytest.mark.skipif(not hasattr(os, 'fork') or not hasattr(os, 'wait4'), reason='POSIX fork / wait4')
def test_wait_child_counts_the_childs_cpu(monkeypatch):
    from isaac_bridge import tok_floor
    for on in (False, True):
        monkeypatch.setattr(tok_floor, 'WORKER_CPU', on)
        pid = os.fork()
        if pid == 0:
            x = 0
            for i in range(3_000_000):
                x += i
            os._exit(0)
        wpid, status, cpu = tok_floor.wait_child(pid, 0)
        assert wpid == pid and status == 0
        assert (cpu > 0.01) if on else cpu == 0.0
    if sys.platform.startswith('linux'):
        monkeypatch.setattr(tok_floor, 'WORKER_CPU', True)
        import threading
        names = []

        def run():
            tok_floor.name_thread('w7-search')
            with open(f'/proc/self/task/{threading.get_native_id()}/comm') as f:
                names.append(f.read().strip())
        th = threading.Thread(target=run)
        th.start(); th.join()
        assert names == ['w7-search']
