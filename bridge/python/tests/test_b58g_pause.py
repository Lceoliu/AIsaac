"""The driver's blocking waits end when the driver thread is blocked on Python (A1's finding, d1).

In d1 the driver thread runs the native lane service; a lane job the graphs cannot take (kind 5) or a lane graph not
captured yet (kind 1) makes the driver thread wait in lane_wait_python for Python's answer to a record that sits in
the queue the actor's main thread consumes. A main thread that calls pause() before consuming it waited forever (and
stop() joined a thread that waited forever). Now: pause() raises in that state (bounded), block_driver.pause_serving
(request_pause, the records consumed meanwhile) returns with the driver paused between steps after the lane job is
done and replied, and stop() ends the wait. Setup: test_b58d_lanes' world with lane pipes, every lane job forced to
Python (set_lane_force_python(-2): kind 5), a branch begin sent through a lane pipe. Skipped without the extension."""
import threading
import time

import numpy as np
import pytest

from test_b58d_lanes import world, capture
from test_b52_cpp_actor import EXT
from test_visible_input import synth_rows, synth_pending
from isaac_bridge import block_driver as bdrv

pytestmark = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')

MSG = b'\x00\x00\x00\x010'


def blocked_driver():
    """the driver thread waiting in lane_wait_python: its kind 5 record queued, nobody consuming it"""
    w = world(True, lane_pipes=True)
    capture(w, 0, 0, 0)
    w.d.set_lane_force_python(-2)
    rng = np.random.default_rng(97)
    rows = synth_rows(rng, 1); rows['first'] = 0; rows['episode'] = 7
    pend = synth_pending(rng, 1, w.m.heads)
    w.s.lane_rows[0] = rows[0]; w.s.lane_io[0, :5] = pend[0]; w.s.lane_io[0, 10] = 800; w.s.lane_io[0, 11] = 1
    w.d.start()
    w.pipes[0][1].sendall(MSG)
    t0 = time.perf_counter()
    while w.d.queue_size() == 0:   # (the kind 5 record pushed: the driver thread now waits for its answer)
        assert time.perf_counter()-t0 < 30, (w.d.last_error(), w.d.lane_error())
        time.sleep(0.005)
    time.sleep(0.05)
    return w


def in_thread(fn, t_max):
    out = []
    th = threading.Thread(target=lambda: out.append(_call(fn)), daemon=True)
    t0 = time.perf_counter()
    th.start(); th.join(t_max)
    return (out[0] if out else 'hung'), time.perf_counter()-t0


def _call(fn):
    try:
        return fn()
    except Exception as e:   # noqa: BLE001
        return e


def test_pause_does_not_wait_for_its_own_lane_record():
    w = blocked_driver()
    try:
        out, dt = in_thread(w.d.pause, 5.0)
        assert isinstance(out, RuntimeError) and 'lane record' in str(out), (out, dt)
        assert dt < 1.0 and not w.d.is_paused()
        # pause_serving: the kind 5 record answered (the job stepped in Python), the round replied, then paused
        out, dt = in_thread(lambda: bdrv.pause_serving(w.d, w.svc.on_native_lane_record), 30.0)
        assert out is None and w.d.is_paused(), (out, dt)
        got = b''
        while len(got) < 5:
            got += w.pipes[0][1].recv(5-len(got))
        assert got == b'\x00\x00\x00\x01a' and w.s.lane_out[0, 4] == 800
        assert w.d.queue_size() == 0 and w.d.lane_counters()['python_jobs'] == 1
    finally:
        w.d.resume(); w.d.pause(); w.d.stop()
    assert w.d.last_error() == '' and w.d.lane_error() == ''


def test_stop_ends_a_lane_wait():
    w = blocked_driver()
    out, dt = in_thread(w.d.stop, 5.0)
    assert out is None and dt < 2.0, (out, dt)
    assert not w.d.running


def blocked_lane_thread():
    """(d2) the lane thread waiting in lane_wait_python: its kind 5 record queued, nobody consuming it"""
    w = world(True, lane_pipes=True)
    capture(w, 0, 0, 0)
    w.d.set_lane_force_python(-2)
    rng = np.random.default_rng(98)
    rows = synth_rows(rng, 1); rows['first'] = 0; rows['episode'] = 7
    pend = synth_pending(rng, 1, w.m.heads)
    w.s.lane_rows[0] = rows[0]; w.s.lane_io[0, :5] = pend[0]; w.s.lane_io[0, 10] = 900; w.s.lane_io[0, 11] = 1
    w.d.lanes_start()
    w.pipes[0][1].sendall(MSG)
    t0 = time.perf_counter()
    while w.d.queue_size() == 0:
        assert time.perf_counter()-t0 < 30, w.d.lane_error()
        time.sleep(0.005)
    time.sleep(0.05)
    return w


def test_lanes_pause_does_not_wait_for_its_own_lane_record():
    """lanes_pause() used to return while the lane thread waited for Python: once the record was answered the lane thread
    went on with GPU work although the caller believed it paused. Now it raises (bounded); lanes_pause_serving returns
    with the lane thread parked between passes, the job answered and replied, and nothing runs until lanes_resume."""
    w = blocked_lane_thread()
    try:
        out, dt = in_thread(w.d.lanes_pause, 5.0)
        assert isinstance(out, RuntimeError) and 'lane record' in str(out), (out, dt)
        assert dt < 1.0 and not w.d.lanes_paused
        out, dt = in_thread(lambda: bdrv.lanes_pause_serving(w.d, w.svc.on_native_lane_record), 30.0)
        assert out is None and w.d.lanes_paused, (out, dt)
        got = b''
        while len(got) < 5:
            got += w.pipes[0][1].recv(5-len(got))
        assert got == b'\x00\x00\x00\x01a' and w.s.lane_out[0, 4] == 900
        c0 = w.d.lane_counters()
        w.pipes[0][1].sendall(MSG)   # (a new request while parked: not served until lanes_resume)
        time.sleep(0.3)
        c1 = w.d.lane_counters()
        assert c1['rounds'] == c0['rounds'] and c1['jobs'] == c0['jobs'] and w.d.queue_size() == 0, (c0, c1)
        w.d.lanes_resume()
        t0 = time.perf_counter()
        while w.d.lane_counters()['rounds'] == c0['rounds']:   # (served after the resume: a kind 5 record again)
            rec = w.d.next_record(0.01)
            if rec is not None:
                w.svc.on_native_lane_record(rec)
            assert time.perf_counter()-t0 < 30, w.d.lane_error()
        assert w.d.lane_counters()['python_jobs'] >= 1
    finally:
        bdrv.lanes_pause_serving(w.d, w.svc.on_native_lane_record)
        w.d.lanes_stop()
    assert w.d.lane_error() == ''
