"""B62: every test leaves nothing running behind it (always on).

A background thread left running by one test can make a CUDA call while a later test captures a graph in global mode
and invalidate that capture ('operation failed due to a previous error during capture'); the capture then never ends
cleanly and every later capture / CUDA random draw in the session fails with it (the b62d / b62g gates: 26 failures
from test_b54_driver_lanes on, intermittent). This check turns that into a deterministic failure of the test that
left the thread. At each test's end:
  * the learner prefetch queue (block_learner.PREFETCH) must be empty; its idle worker is then stopped (a process-wide
    singleton by design, restarted by the next submit);
  * every C++ actor driver made during the test (block_driver.make_driver) must be stopped (`running`, `lanes_running`
    false): its loop and lane threads are native, invisible to Python. The driver runs the native consumer and the
    native lane service on those threads;
  * every Python thread started during the test, and (Linux) every native thread of the process started during it,
    must have ended within LINGER_S seconds. Native threads named like the process-lifetime pools torch and the CUDA
    driver start lazily (POOL_PREFIXES) are not counted.
A failing check is reported as an error of that test's teardown, with the threads' names; leftover drivers are then
stopped and the prefetch worker closed so that the next tests start clean."""
import os
import threading
import time
import weakref

import pytest

LINGER_S = 2.0
POOL_PREFIXES = ('pt_', 'cuda')   # (pt_autograd_N, pt_thread_pool; cuda-EvtHandlr, cuda000...: process-lifetime)
_DRIVERS = []   # weak references to the drivers made since the current test started


def _install_driver_registry():
    try:
        from isaac_bridge import block_driver
    except Exception:   # (no torch / no extension: nothing to track)
        return
    if getattr(block_driver.make_driver, '_b62_tracked', False):
        return
    make = block_driver.make_driver

    def tracked(*a, **k):
        d = make(*a, **k)
        if d is not None:
            try:
                _DRIVERS.append(weakref.ref(d))
            except TypeError:   # (not weak-referenceable: kept until the test's end)
                _DRIVERS.append(lambda d=d: d)
        return d
    tracked._b62_tracked = True
    block_driver.make_driver = tracked


def _tasks():
    """{native thread id: name} of this process (Linux), None elsewhere."""
    try:
        ids = os.listdir('/proc/self/task')
    except OSError:
        return None
    out = {}
    for t in ids:
        try:
            with open(f'/proc/self/task/{t}/comm') as f:
                out[int(t)] = f.read().strip()
        except OSError:
            pass
    return out


def _stop(d):
    for name in ('lanes_stop', 'pause', 'stop'):
        try:
            getattr(d, name)()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def nothing_left_running(request):
    _install_driver_registry()
    del _DRIVERS[:]
    py0 = {t.ident for t in threading.enumerate()}
    nat0 = _tasks()
    yield
    problems = []
    # the learner prefetch: nothing queued, the worker stopped
    try:
        from isaac_bridge import block_learner
    except Exception:
        block_learner = None
    if block_learner is not None:
        pf = block_learner.PREFETCH
        if pf.jobs:
            problems.append(f'learner prefetch: {len(pf.jobs)} part(s) left queued')
        close = getattr(pf, 'close', None)   # (absent before B62's fix: trees checked with this file added)
        if close is not None and not close():
            problems.append('learner prefetch: the worker did not stop')
    # the C++ drivers made during the test
    for r in list(_DRIVERS):
        d = r()
        if d is None:
            continue
        running, lanes = bool(getattr(d, 'running', False)), bool(getattr(d, 'lanes_running', False))
        if running or lanes:
            problems.append(f'C++ actor driver left running (loop {running}, lanes {lanes})')
            _stop(d)
        del d
    del _DRIVERS[:]
    # threads started during the test, given LINGER_S to end
    deadline = time.monotonic()+LINGER_S
    while True:
        py = [t for t in threading.enumerate() if t.ident not in py0 and t.is_alive()]
        py_native = {t.native_id for t in threading.enumerate()}
        nat = _tasks()
        new_nat = {} if nat is None or nat0 is None else {
            k: v for k, v in nat.items() if k not in nat0 and k not in py_native and not v.startswith(POOL_PREFIXES)}
        if (not py and not new_nat) or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    if py:
        problems.append('Python threads still alive: '+', '.join(f'{t.name} (daemon {t.daemon})' for t in py))
    if new_nat:
        problems.append('native threads still alive: '+', '.join(f'{v} [{k}]' for k, v in sorted(new_nat.items())))
    if problems:
        pytest.fail(f'{request.node.nodeid} left work running: '+'; '.join(problems), pytrace=False)
