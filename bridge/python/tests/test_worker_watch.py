"""WorkerWatch (block_driver): the collect in driver mode ends with a RuntimeError, not a silent wait, when its workers
ended (2026-11-12, acceptg-c-78ad1e8: every worker failed at its first decode, the trainer waited out its 75 min
timeout -- the driver owns the workers' pipes, so the sampler's live list never shrank), and progress workers_live
shows a partial loss.

Without the extension (fakes for the driver, the rollout, the worker processes):
  1. every worker ended (the driver's count): batch_wait's idle return raises, naming the count;
  2. too few left to fill the collect (the ended ones' usable decisions + a full capacity per live one < the target):
     raises; just enough left: no error;
  3. one ended, the rest can fill: no error, workers_live one lower (train_tok.workers_live, the progress column);
  4. the second guard: no worker process alive while the driver still counts pipes open: raises;
  5. normal stops never raise: stopped() (STOP, the game-hours end, a SIGINT's shutdown) with every worker gone; all
     alive over many checks; the throttle; the Python path's count (sampler.live).
With the extension (the gpu57 gate; skipped elsewhere): the driver's own count -- a worker's pipe closed by its end
(EOF) leaves workers_live / dead_workers at once while the others keep stepping; all closed: batch_wait raises within
the idle timeout; a driver stopped with its workers alive counts them all live."""
import time
import types

import numpy as np
import pytest

from isaac_bridge import block_driver as bdrv


class FakeDriver:
    def __init__(self, n):
        self.n_workers, self.dead = n, []
        self.running = True

    def workers_live(self):
        return self.n_workers - len(self.dead)

    def dead_workers(self):
        return list(self.dead)

    def next_record(self, timeout):
        time.sleep(min(timeout, 0.01))
        return None

    def last_error(self):
        return ''

    def is_paused(self):
        return False

    def queue_size(self):
        return 0


class Proc:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


def make(n=4, cap=10, target=30):
    d = FakeDriver(n)
    s = types.SimpleNamespace(cfg=types.SimpleNamespace(workers=n), procs=[Proc() for _ in range(n)])
    roll = types.SimpleNamespace(length=np.zeros(n, np.int64), open=np.full(n, -1, np.int64), cap=cap)
    return d, s, roll, bdrv.WorkerWatch(d, s, roll, target, stopped=lambda: False, every=0.0)


def wait(d, w):
    return bdrv.batch_wait(d, lambda *a: None, lambda r: None, 0.01, watch=w)


def test_all_workers_ended_raises():
    d, s, roll, w = make()
    assert wait(d, w) == 'idle'
    d.dead = [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match='all 4 workers ended'):
        wait(d, w)


def test_too_few_left_to_fill_raises():
    d, s, roll, w = make(n=4, cap=10, target=30)
    roll.length[:] = [5, 5, 5, 5]
    roll.open[:] = [4, -1, -1, -1]       # (worker 0's last decision never closes: not usable)
    d.dead = [0, 1]                      # usable 4 + 5, room 2 x 10: 29 < 30
    with pytest.raises(RuntimeError, match=r'2 of 4 workers ended \(workers \[0, 1\]\), 2 left: at most 9 \+ 20'):
        wait(d, w)
    roll.open[0] = -1                    # usable 10 + room 20 = 30: it can still fill
    assert wait(d, w) == 'idle'


def test_one_ended_still_fills_and_the_column_drops():
    import train_tok as T
    d, s, roll, w = make(n=4, cap=10, target=30)
    assert T.workers_live(d, s) == 4 and w.live() == 4
    d.dead = [2]
    assert wait(d, w) == 'idle'
    w.check(force=True)
    assert T.workers_live(d, s) == 3 and w.live() == 3


def test_no_worker_process_alive_raises():
    d, s, roll, w = make()
    for p in s.procs:
        p.alive = False
    with pytest.raises(RuntimeError, match='0 worker processes alive'):
        w.check(force=True)


def test_normal_stops_never_raise():
    d, s, roll, w = make()
    for _ in range(50):                  # all alive: never
        w.check(force=True)
    stopping = [True]
    w2 = bdrv.WorkerWatch(d, s, roll, 30, stopped=lambda: stopping[0], every=0.0)
    d.dead = [0, 1, 2, 3]
    for p in s.procs:
        p.alive = False
    w2.check(force=True)                 # (STOP / the game hours / a SIGINT's shutdown: the workers closed by the run)
    assert wait(d, w2) == 'idle'
    stopping[0] = False
    with pytest.raises(RuntimeError):
        w2.check(force=True)
    w3 = bdrv.WorkerWatch(d, s, roll, 30, stopped=lambda: False, every=60.0)
    w3.t_last = time.perf_counter()
    w3.check()                           # (throttled: no look within `every`)


def test_python_path_count():
    import train_tok as T
    conns = [object() for _ in range(3)]
    s = types.SimpleNamespace(cfg=types.SimpleNamespace(workers=3), conns=conns, live=[conns[0], conns[2]], procs=[])
    roll = types.SimpleNamespace(length=np.zeros(3, np.int64), open=np.full(3, -1, np.int64), cap=10)
    w = bdrv.WorkerWatch(None, s, roll, 10, every=0.0)
    assert w.dead() == [1] and w.live() == 2 and T.workers_live(None, s) == 2
    w.check(force=True)                  # (2 x 10 >= 10; procs unknown: no second guard)


# ------------------------------------------------------------------------------------------- with the extension
try:
    from test_b52_cpp_actor import EXT
except Exception:   # noqa: BLE001
    EXT = None
ext = pytest.mark.skipif(EXT is None, reason='cpp actor extension unavailable')


def until(fn, s=3.0):
    t = time.perf_counter() + s
    while time.perf_counter() < t:
        if fn():
            return True
        time.sleep(0.01)
    return False


@ext
def test_driver_counts_closed_pipes():
    """(no stepping here: the stepping with a worker gone is the fakes' case 3 -- the driver's count is what this adds)"""
    import torch
    from test_block_memory import make_policy
    import test_b58e_batch as E
    torch.manual_seed(48)
    m = make_policy('cuda').eval()
    W = 3
    actor, s, d, book, rolls = E.setup(m, W)
    E.KEEP[:] = [actor, s, book, rolls]
    d.set_batch(True, 256)
    assert d.workers_live() == W and list(d.dead_workers()) == []
    d.start()
    s._pairs[1][1].close()               # worker 1 ended: its pipe's EOF
    assert until(lambda: d.workers_live() == W - 1), d.workers_live()
    assert list(d.dead_workers()) == [1]
    roll = types.SimpleNamespace(length=np.zeros(W, np.int64), open=np.full(W, -1, np.int64), cap=E.CAP)
    w = bdrv.WorkerWatch(d, types.SimpleNamespace(cfg=types.SimpleNamespace(workers=W), procs=[]), roll, 2 * E.CAP,
                         stopped=lambda: False, every=0.0)
    assert bdrv.batch_wait(d, lambda *a: None, lambda r: None, 0.2, watch=w) == 'idle'   # (two left hold 2 x CAP)
    s._pairs[0][1].close(); s._pairs[2][1].close()
    assert until(lambda: d.workers_live() == 0), d.workers_live()
    assert sorted(d.dead_workers()) == [0, 1, 2]
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match='all 3 workers ended'):
        for _ in range(100):
            bdrv.batch_wait(d, lambda *a: None, lambda r: None, 0.5, watch=w)
    assert time.perf_counter() - t0 < 1.5
    d.pause(); d.stop()


@ext
def test_driver_stopped_with_live_workers_counts_them_live():
    import torch
    from test_block_memory import make_policy
    import test_b58e_batch as E
    torch.manual_seed(49)
    m = make_policy('cuda').eval()
    actor, s, d, book, rolls = E.setup(m, 2)
    E.KEEP[:] = [actor, s, book, rolls]
    d.set_batch(True, 256)
    d.start()
    time.sleep(0.2)
    d.stop()
    assert d.workers_live() == 2 and list(d.dead_workers()) == []
