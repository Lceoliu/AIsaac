"""B (2026-11-12, ISAAC_RL_WORKER_CPU=1): the worker CPU helpers (wait_child's child CPU, name_thread) count only, and
the stats keep A28's columns last. (W-a's V2Log tests left with W-a: the mainline's 645760f keeps the records compactly
always -- V2Log.keep_row / put_row, tests/test_v2log_compact.py.)"""
import os
import sys

import pytest

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
