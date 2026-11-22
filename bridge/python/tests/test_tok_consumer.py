"""P package (Codex round 18-2, 2026-11-19): a fatal the C++ driver found reaches the run, whenever it was found --
during the drain's boundary or at the end (DriverCollect.close, after the driver's stop and close_serve). Stubs only:
no extension, no GPU."""
import numpy as np
import pytest

from isaac_bridge import tok_consumer
from isaac_bridge.visible_producer import VisibleV3Fatal


class StubD:
    def __init__(self, code=86, open_w=0):
        self.code, self.open_w, self.stopped = code, open_w, False

    def stop(self):
        self.stopped = True

    def close_serve(self, timeout):
        return 0, self.open_w

    def take(self, n, timeout):
        z = np.zeros(0, np.int64)
        return dict(seq=z, kind=z, worker=z, call=z, n_call=z)

    def fatal(self):
        return (self.code, 1, b'a bad frame' if self.code else b'')

    def pause(self):
        pass

    def drain_state(self):
        return True, 0, 0, self.code


class StubSampler:
    def __init__(self):
        self.control = np.zeros(4, np.int64)
        self.conns = [0, 1]

    def _drop(self, c):
        pass


def collect_stub(d):
    dc = tok_consumer.DriverCollect.__new__(tok_consumer.DriverCollect)
    dc.started, dc.d, dc.sampler, dc.stat = True, d, StubSampler(), {'take_s': 0.0}
    dc.partial, dc.next_seq = {}, 0
    return dc


def test_close_raises_a_fatal_found_at_the_end():
    dc = collect_stub(StubD(86))
    with pytest.raises(VisibleV3Fatal, match='a bad frame'):
        dc.close(timeout=1.0)
    assert dc.d.stopped and dc.sampler.control[0] == 1   # (the cleanup ran first)


def test_close_raises_when_workers_stay_open():
    dc = collect_stub(StubD(0, open_w=2))
    with pytest.raises(RuntimeError, match='2 workers still open'):
        dc.close(timeout=1.0)


def test_close_is_quiet_when_all_is_well():
    dc = collect_stub(StubD(0))
    dc.close(timeout=1.0)
    assert dc.d.stopped


def test_the_drain_boundary_carries_the_fatal():
    dc = collect_stub(StubD(86))
    dc.step = lambda *a, **k: 0   # (nothing queued: the fatal comes with the drained state)
    with pytest.raises(VisibleV3Fatal):
        dc.drain(None, None, None, {}, None)
