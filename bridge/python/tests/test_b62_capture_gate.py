"""B62: the learner prefetch worker's CUDA calls (block_learner.Staging.acquire: event query / synchronisation, the
pinned allocation and the old buffer's free) and the actor's graph captures exclude each other (block_training.
SIDE_CUDA). The worker runs without the capture admission; capture_yield hands the admission to an actor capture
between two learner parts while the worker prepares the next one, so without the gate one such call during a
global-mode capture invalidates the capture ('operation failed due to a previous error during capture').

The real capture runs in a subprocess (gated: the worker's call waits for the capture's end, the graph replays right;
control group, gate off: must fail), so a broken capture cannot leak into the other tests."""
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

from isaac_bridge.block_training import SideCuda

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_side_calls_wait_for_a_capture_and_a_capture_for_the_calls():
    g = SideCuda()
    entered = threading.Event()

    def side():
        with g.call():
            entered.set()
    with g.capture():
        t = threading.Thread(target=side, daemon=True)
        t.start()
        assert not entered.wait(0.3)   # (blocked while the capture runs)
    t.join(5)
    assert entered.is_set() and g.waits == 1
    # a capture waits for a call under way
    inside, release, captured = threading.Event(), threading.Event(), threading.Event()

    def long_call():
        with g.call():
            inside.set()
            release.wait(5)
    t = threading.Thread(target=long_call, daemon=True)
    t.start()
    inside.wait(5)

    def cap():
        with g.capture():
            captured.set()
    c = threading.Thread(target=cap, daemon=True)
    c.start()
    assert not captured.wait(0.3)
    release.set()
    t.join(5); c.join(5)
    assert captured.is_set() and g.active == 0 and g.capturing == 0


def test_disabled_gate_does_not_block():
    g = SideCuda()
    g.ENABLED = False
    entered = threading.Event()

    def side():
        with g.call():
            entered.set()
    with g.capture():
        t = threading.Thread(target=side, daemon=True)
        t.start()
        assert entered.wait(5)
    t.join(5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_staging_acquire_waits_while_a_capture_is_under_way():
    from isaac_bridge.block_learner import Staging
    from isaac_bridge.block_training import SIDE_CUDA
    st = Staging()
    k, _ = st.acquire(1000)
    st.release(k)
    done = threading.Event()

    def worker():
        i, _ = st.acquire(1 << 20)
        st.drop(i)
        done.set()
    with SIDE_CUDA.capture():
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        assert not done.wait(0.3)
    t.join(10)
    assert done.is_set()


SCRIPT = r'''
import sys, threading, time, torch
from isaac_bridge import block_training
from isaac_bridge.block_learner import Staging
from isaac_bridge.block_training import graph_capture, SIDE_CUDA
block_training.SideCuda.ENABLED = sys.argv[1] == 'gated'
st = Staging()
k, _ = st.acquire(1000)          # a small buffer with an event of its last copy: the next acquire queries /
st.release(k)                    # synchronises that event, frees the buffer and pins a larger one
torch.cuda.synchronize()
go, done, err = threading.Event(), threading.Event(), []
def worker():
    go.wait()
    try:
        i, _ = st.acquire(16 << 20)
        st.drop(i)
    except BaseException as e:
        err.append(repr(e))
    done.set()
t = threading.Thread(target=worker, daemon=True)
t.start()
x = torch.ones(4096, device='cuda')
stream = torch.cuda.Stream()
graph = torch.cuda.CUDAGraph()
during = None
try:
    with graph_capture(graph, stream):
        y = x*2
        go.set()
        during = done.wait(1.0)   # (the worker's acquire finished while the capture ran?)
        for _ in range(64):
            y = y+1
    graph.replay()
    torch.cuda.synchronize()
    ok = bool((y == 66).all())
    print('CAPTURE ok' if ok else 'CAPTURE wrong values')
except BaseException as e:
    print('CAPTURE error', repr(e)[:300])
t.join(30)
print('WORKER', 'error '+err[0][:300] if err else 'ok', 'during' if during else 'after', 'waits', SIDE_CUDA.waits)
'''


def _run(mode):
    r = subprocess.run([sys.executable, '-c', SCRIPT, mode], cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout+r.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_worker_staging_during_a_real_capture_gated_and_control():
    rc, out = _run('gated')
    assert rc == 0 and 'CAPTURE ok' in out and 'WORKER ok after waits 1' in out, out
    rc, out = _run('control')   # (gate off: the worker's calls run inside the capture, which must not survive them)
    assert 'WORKER ok during' in out or 'WORKER error' in out, out
    assert 'CAPTURE ok' not in out or 'WORKER error' in out, out
