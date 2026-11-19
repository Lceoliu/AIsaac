"""(2026-11-18, Codex round 3) A visible-v3 protocol error (VisibleV3Fatal: another schema version, a malformed frame
header, a bridge in another mode, a record without the v3 frame a v3 worker needs) is not a game fault: a sampler
worker stops with V3_FATAL_EXIT and TokSampler raises it in the parent (poll / poll_ready / check_fatal / close), so
train_tok (check_actors) and eval_tok fail instead of going on with the other workers or skipping seeds."""
import multiprocessing as mp
import sys
import time
import types

import pytest

from isaac_bridge import visible_producer as VP
from isaac_bridge.env import BridgeError
from isaac_bridge.tok_sampler import TokSampler


def test_the_fatal_class_is_no_game_fault():
    """None of the retry paths' exception types catches it (they catch BridgeError, OSError, RuntimeError,
    ValueError)."""
    assert not issubclass(VP.VisibleV3Fatal, (BridgeError, OSError, RuntimeError, ValueError))
    assert VP.V3_FATAL_EXIT not in (0, 1, 2) and 0 < VP.V3_FATAL_EXIT < 128


class _Proc:
    def __init__(self, code, pid=None):
        self.exitcode, self.pid = code, pid

    def join(self, timeout=None):
        pass


def _sampler(codes):
    s = object.__new__(TokSampler)
    s.procs = [_Proc(c) for c in codes]
    s.fatal, s._closing, s._fatal_reason = None, False, {}
    s.cfg = types.SimpleNamespace(fatal_dir='')
    return s


@pytest.fixture(autouse=True)
def v3_worker(monkeypatch):
    """These tests are of a worker configured for v3 (the fatal machinery is off in mode 0: fatal_active)."""
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 2)


def test_check_fatal_by_exit_code():
    _sampler([None, 0, 1, -15]).check_fatal()   # (running, ended, an ordinary failure, killed: not this)
    with pytest.raises(VP.VisibleV3Fatal, match='tok worker 2 stopped on a visible-v3 protocol error'):
        _sampler([0, None, VP.V3_FATAL_EXIT]).check_fatal()


def _worker(conn, code):
    conn.send_bytes(b'x')   # (the workers' last message, as in worker_main's finally)
    conn.close()
    sys.exit(code)


@pytest.mark.parametrize('code, fatal', [(0, False), (VP.V3_FATAL_EXIT, True)])
def test_poll_raises_once_a_worker_ended_fatally(code, fatal):
    """Two real worker processes on pipes: one ends normally, the other with `code`; TokSampler.poll drops both
    (their 'x') and raises VisibleV3Fatal for V3_FATAL_EXIT -- before, the parent went on with the live workers."""
    ctx = mp.get_context('spawn')
    s = object.__new__(TokSampler)
    s.conns, s.procs = [], []
    for c in (0, code):
        ours, theirs = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(theirs, c), daemon=True)
        p.start()
        theirs.close()
        s.conns.append(ours)
        s.procs.append(p)
    s.index = {c: i for i, c in enumerate(s.conns)}
    s.live = list(s.conns)
    s.fatal, s._closing, s._fatal_reason = None, False, {}
    s.cfg = types.SimpleNamespace(fatal_dir='')
    s._poller = None

    class _Poller:   # (_drop unregisters from poll_ready's poller)
        def unregister(self, fd):
            pass
    s._poller = _Poller()
    end = time.time() + 60
    raised = False
    try:
        while s.live and time.time() < end:
            s.poll(want=2, max_wait=0.5, idle=0.5)
    except VP.VisibleV3Fatal:
        raised = True
    for p in s.procs:
        p.join(10)
    assert raised == fatal
    if not fatal:   # (and check_fatal after the join: nothing)
        s.check_fatal()
    else:
        with pytest.raises(VP.VisibleV3Fatal):
            s.check_fatal()


def test_eval_counts_missing_seeds():
    """eval_tok.seeds_missing: requested (group, seed) pairs of the workers' lists, played ones, and the missing count
    (less the empty start states). eval_tok exits non-zero (after its summary) when it is > 0."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'eval_tok.py'
    spec = importlib.util.spec_from_file_location('eval_tok_for_test', path)
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    names, assign, seeds = ['normal', 'boss'], [0, 0, 1], [[10, 12], [11, 13], [10, 11, 12, 13]]
    every = [dict(group=names[assign[w]], seed=s_) for w, l_ in enumerate(seeds) for s_ in l_]
    assert et.seeds_missing(names, assign, seeds, every, 0) == (8, 8, 0)
    assert et.seeds_missing(names, assign, seeds, every[1:], 1) == (8, 7, 0)     # one start state was empty
    assert et.seeds_missing(names, assign, seeds, every[2:], 1) == (8, 6, 1)     # one more seed never played
    assert et.seeds_missing(names, assign, seeds, every + [dict(group='normal', seed=99)], 0)[2] == 0
    src = path.read_text(encoding='utf-8')
    assert 'if missing > 0:\n            raise SystemExit(' in src.replace('\r\n', '\n')


def test_eval_main_fails_on_missing_seeds(tmp_path, monkeypatch):
    """eval_tok.main run to its end (2026-11-18, round 4: the exit message used a name main does not have -- a
    NameError, not the SystemExit) with a sampler that plays nothing: every requested seed missing -> the summary
    written, then SystemExit naming the count; every start state empty instead -> no exit, 0 missing."""
    import importlib.util
    import json
    from pathlib import Path
    import numpy as np
    from isaac_bridge.tok_sampler import ST, STATS
    py = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('eval_tok_main_test', py / 'eval_tok.py')
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    for empty, want_exit in ((0, True), (1, False)):
        class Sampler:   # (no worker: no record, the stats as the workers would leave them)
            def __init__(self, cfg):
                self.live = []
                self.stats = np.zeros((cfg.workers, len(STATS)))
                self.stats[:, ST['empty']] = empty * np.array([len(s_) for s_ in cfg.eval_seeds])

            def close(self):
                pass
        monkeypatch.setattr(et, 'TokSampler', Sampler)
        out = tmp_path / f'out{empty}'
        monkeypatch.setattr(sys, 'argv', ['eval_tok.py', '--random', '--groups', 'normal:2', '--seeds', '2147490000:4',
                                          '--groups-file', str(py.parent / 'abplus' / 'catalog' / 'scaling2_groups.json'),
                                          '--out', str(out)])
        if want_exit:
            with pytest.raises(SystemExit, match='4 of 4 requested seeds have no episode'):
                et.main()
        else:
            et.main()
        s = json.loads((out / 'summary.json').read_text())
        assert (s['seeds_requested'], s['seeds_played'], s['seeds_missing']) == (4, 0, 4 if want_exit else 0)


def _fatal_worker(conn, how, d):
    """A worker that meets a VisibleV3Fatal, reported as worker_main does (report_fatal, before cleanup) with its
    sampler's fatal directory d; then the exit. 'pipe': no fatal directory (the pipe alone carries the reason), exit 0
    as if its cleanup had failed; 'file': the pipe message lost (the C++ actor reads worker pipes itself), only the
    file, exit 1; 'nothing': exit 86, no report; 'clean': no fatal at all, exit 0."""
    from isaac_bridge.tok_sampler import report_fatal
    exc = VP.VisibleV3Fatal('visible-v3: the bridge runs mode 1, this worker is configured for mode 0')
    if how == 'pipe':
        report_fatal(1, conn, exc, fatal_dir='')
        sys.exit(0)
    if how == 'file':
        class _Gone:
            def send_bytes(self, b):
                raise OSError('pipe consumed elsewhere')
        report_fatal(1, _Gone(), exc, fatal_dir=d)
        conn.send_bytes(b'x')
        sys.exit(1)
    conn.send_bytes(b'x')
    sys.exit(VP.V3_FATAL_EXIT if how == 'nothing' else 0)


def _run_one(tmp_path, how, use_poll_ready=False, stale=False):
    """One real worker process (_fatal_worker) under a hand-made TokSampler (index 1, its fatal directory a fresh
    run directory); records polled by poll() or poll_ready() until it is gone, then check_fatal. The VisibleV3Fatal
    text, or None. stale: files of a former run with this worker's index and pid lie in the base directory and in
    another run's directory (the same output directory reused, the pid repeated in a new container)."""
    import os
    from isaac_bridge.tok_sampler import fatal_file
    ctx = mp.get_context('spawn')
    run_dir = tmp_path / 'fatal-run-now'
    ours, theirs = ctx.Pipe()
    p = ctx.Process(target=_fatal_worker, args=(theirs, how, str(run_dir)), daemon=True)
    p.start()
    theirs.close()
    if stale:
        for d in (tmp_path, tmp_path / 'fatal-run-before'):
            fatal_file(d, 1, p.pid).parent.mkdir(parents=True, exist_ok=True)
            fatal_file(d, 1, p.pid).write_text('tok worker 1: a former run, its reason')
    s = object.__new__(TokSampler)
    s.conns, s.procs, s.index, s.live = [None, ours], [_Proc(0), p], {ours: 1}, [ours]
    s.fatal, s._closing, s._fatal_reason = None, False, {}
    s.cfg = types.SimpleNamespace(fatal_dir=str(run_dir))
    if use_poll_ready:
        import select
        s._poller, s._by_fd, s._lanes = select.poll(), {ours.fileno(): ours}, {}
        s._poller.register(ours.fileno(), select.POLLIN)
    else:
        class _Poller:
            def unregister(self, fd):
                pass
        s._poller = _Poller()
    end, raised = time.time() + 60, None
    while s.live and time.time() < end and raised is None:
        try:
            if use_poll_ready:
                s.poll_ready(idle=0.5)
            else:
                s.poll(want=1, max_wait=0.5, idle=0.5)
        except VP.VisibleV3Fatal as exc:
            raised = str(exc)
    p.join(10)
    if raised is None:
        try:
            s.check_fatal()
        except VP.VisibleV3Fatal as exc:
            raised = str(exc)
    return raised, p


HAVE_POLL = hasattr(__import__('select'), 'poll')


@pytest.mark.parametrize('reader', ['poll', pytest.param('poll_ready', marks=pytest.mark.skipif(
    not HAVE_POLL, reason='select.poll (Linux)'))])
def test_the_reason_travels_on_the_pipe_alone(tmp_path, reader):
    """(2026-11-18, Codex round 5: the pipe case without a file to fall back on, poll and poll_ready apart) The
    worker sends 'F' + its reason and exits 0 (its cleanup failed): the parent's VisibleV3Fatal carries the reason."""
    raised, _ = _run_one(tmp_path, 'pipe', use_poll_ready=reader == 'poll_ready')
    assert raised and 'tok worker 1 stopped on a visible-v3 protocol error (exit' in raised
    assert 'configured for mode 0' in raised and 'stderr' not in raised and not list(tmp_path.rglob('fatal-worker-*'))


def test_the_reason_from_the_file_when_the_pipe_lost_it(tmp_path):
    raised, p = _run_one(tmp_path, 'file')
    assert raised and 'configured for mode 0' in raised
    assert (tmp_path / 'fatal-run-now' / f'fatal-worker-1-{p.pid}.txt').read_text().strip().endswith('mode 0')


def test_no_reason_is_said_plainly(tmp_path):
    raised, _ = _run_one(tmp_path, 'nothing')
    assert raised and 'the reason did not reach the parent' in raised and '(exit 86)' in raised


def test_a_former_runs_file_is_not_this_runs_fatal(tmp_path):
    """(2026-11-18, Codex round 5) A file of a former run with this worker's index and pid (the output directory
    reused, the pid repeated in a new container) does not make a clean exit fatal: each sampler reads only its own
    run directory (fatal_run_dir: new for every sampler)."""
    from isaac_bridge.tok_sampler import fatal_run_dir
    raised, _ = _run_one(tmp_path, 'clean', stale=True)
    assert raised is None
    assert fatal_run_dir() != fatal_run_dir()


def test_a_connection_closed_during_the_mode_query_is_retried_not_fatal(monkeypatch):
    """(2026-11-18, A1's mode-0 smoke on c80c9f2: a worker exited 86 at the end of training, its reason lost) The
    path, deterministically: a StartBuilder's build connects a relaunched instance and the game goes away during the
    visible_v3 query (EOF). The builder's result is a plain BridgeError -- the worker's recover() relaunches, as
    before 05f1622 -- not VisibleV3Fatal (which the worker re-raises at builder.take(): exit 86). With c80c9f2's
    abplus.check_visible_v3 the result is VisibleV3Fatal in mode 0 too."""
    import socket
    import types
    from isaac_bridge.abplus import check_visible_v3
    from isaac_bridge.env import IsaacBridgeEnv
    from isaac_bridge.tok_sampler import StartBuilder

    class Inst:   # (Instance.reset's connect, reduced to the mode query)
        digest_ready = False

        def reset(self, seed):
            ours, theirs = socket.socketpair()
            c = object.__new__(IsaacBridgeEnv)
            c._sock, c._buf = ours, b''
            theirs.shutdown(socket.SHUT_WR)   # (the game gone: EOF on the answer)
            self.keep = theirs
            check_visible_v3(c, 2)
    cfg = types.SimpleNamespace(build_timeout=30, reset_timeout=30)
    b = StartBuilder(15, Inst(), cfg, next_seed=lambda: 1)
    b.start()
    assert b.ready.wait(10)
    seed, exc = b.take()
    assert seed == 1 and exc is not None and not isinstance(exc, VP.VisibleV3Fatal)
    assert isinstance(exc, BridgeError) and 'connection closed by game' in str(exc)


@pytest.mark.parametrize('module, func', [('tok_floor', 'floor_worker_main'), ('tok_sampler', 'worker_main')])
def test_the_exit_86_flag_is_set_only_by_a_visible_v3_fatal(module, func):
    """(2026-11-18, A1's mode-0 smoke on 253f759: a floor worker exited 86 with no reason anywhere) The worker
    function's 'sys.exit(V3_FATAL_EXIT)' is guarded by a variable that only its None start and the 'except
    VisibleV3Fatal' handler assign. Before: the flag was named 'fatal', a name the floor worker's death-teaching code
    already assigns (fatal = death_teaching and done == 2 ...: False / True, never None) -- every worker past that
    line ended with exit 86."""
    import ast
    import inspect
    import importlib
    mod = importlib.import_module(f'isaac_bridge.{module}')
    tree = ast.parse(inspect.getsource(getattr(mod, func)))
    guard = None
    for node in ast.walk(tree):   # if <name> is not None: sys.exit(V3_FATAL_EXIT)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                isinstance(node.test.ops[0], ast.IsNot) and isinstance(node.test.left, ast.Name) and \
                'V3_FATAL_EXIT' in ast.unparse(node.body[0]):
            guard = node.test.left.id
    assert guard
    in_handler, other = 0, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None and 'VisibleV3Fatal' in ast.unparse(node.type):
            in_handler += sum(isinstance(t, ast.Name) and t.id == guard for n in ast.walk(node)
                              if isinstance(n, ast.Assign) for t in n.targets)
    for node in ast.walk(tree):
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []):
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and n.id == guard:
                    other.append(ast.unparse(node)[:80])
    assert in_handler == 1 and len(other) == 2 and any(o.endswith('= None') for o in other), (guard, other)


def test_mode_0_has_no_fatal_machinery(monkeypatch, tmp_path):
    """(2026-11-18, the coordinator: mode 0 as 05f1622) A worker not configured for v3: the parent reads no exit 86
    and no reason as fatal (check_fatal, close), the connect sends no mode query, and the worker functions take a
    VisibleV3Fatal as an ordinary error (no exit 86; the forked children 1)."""
    import inspect
    from isaac_bridge import tok_floor, tok_sampler
    monkeypatch.setattr(VP, 'VISIBLE_V3_MODE', 0)
    assert not VP.fatal_active()
    s = _sampler([VP.V3_FATAL_EXIT, 0])
    s._fatal_reason[0] = 'tok worker 0: a reason'
    s.check_fatal()   # (no raise)
    assert s.fatal is None
    for src in (inspect.getsource(tok_sampler.worker_main), inspect.getsource(tok_floor.floor_worker_main)):
        i = src.rindex('except VisibleV3Fatal as exc:')   # (the worker's own: the last one)
        assert 'fatal_active()' in src[i:i + 200] and 'traceback.print_exc()' in src[i:i + 500]
    src = inspect.getsource(tok_floor)
    assert src.count('code = V3_FATAL_EXIT if fatal_active() else 1') == 2


@pytest.mark.skipif(__import__('os').name == 'nt', reason='a POSIX pipe (os.read on its fd)')
def test_the_reason_after_the_cpp_actor_took_the_f(tmp_path):
    """(2026-11-18, Codex round 5: the C++ actor reads 5 bytes of a worker message) The worker sends 'F' alone (5
    bytes) and then 'R' + the reason; after a reader took exactly the 'F' (as the C++ actor does), poll() reads the 'R'
    message whole -- no length read from the middle of a message -- and the reason reaches the VisibleV3Fatal."""
    import os
    ctx = mp.get_context('spawn')
    ours, theirs = ctx.Pipe()
    p = ctx.Process(target=_fatal_worker, args=(theirs, 'pipe', ''), daemon=True)
    p.start()
    theirs.close()
    assert ours.poll(30)
    head = os.read(ours.fileno(), 5)   # (the C++ actor's read)
    assert head == bytes([0, 0, 0, 1]) + b'F'
    s = object.__new__(TokSampler)
    s.conns, s.procs, s.index, s.live = [None, ours], [_Proc(0), p], {ours: 1}, [ours]
    s.fatal, s._closing, s._fatal_reason = None, False, {}
    s.cfg = types.SimpleNamespace(fatal_dir='')

    class _Poller:
        def unregister(self, fd):
            pass
    s._poller = _Poller()
    with pytest.raises(VP.VisibleV3Fatal, match='configured for mode 0'):
        end = time.time() + 30
        while s.live and time.time() < end:
            s.poll(want=1, max_wait=0.5, idle=0.5)
    p.join(10)
