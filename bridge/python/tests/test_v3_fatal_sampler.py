"""(2026-11-18, Codex round 3) A visible-v3 protocol error (VisibleV3Fatal: another schema version, a malformed frame
header, a bridge in another mode, a record without the v3 frame a v3 worker needs) is not a game fault: a sampler
worker stops with V3_FATAL_EXIT and TokSampler raises it in the parent (poll / poll_ready / check_fatal / close), so
train_tok (check_actors) and eval_tok fail instead of going on with the other workers or skipping seeds."""
import multiprocessing as mp
import sys
import time

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
    return s


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
    """A worker that meets a VisibleV3Fatal: its reason reported as worker_main does (report_fatal, before cleanup),
    then the exit -- 86, or another code as if its cleanup had failed; 'file only': the pipe message lost (the C++
    actor reads worker pipes itself), only the file left; 'nothing': 86 without any report."""
    import os
    os.environ['ISAAC_RL_FATAL_DIR'] = d
    from isaac_bridge.tok_sampler import report_fatal
    exc = VP.VisibleV3Fatal('visible-v3: the bridge runs mode 1, this worker is configured for mode 0')
    if how == 'pipe, exit 0':
        report_fatal(1, conn, exc)
        sys.exit(0)
    if how == 'file only, exit 1':
        class _Gone:
            def send_bytes(self, b):
                raise OSError('pipe consumed elsewhere')
        report_fatal(1, _Gone(), exc)
        conn.send_bytes(b'x')
        sys.exit(1)
    conn.send_bytes(b'x')
    sys.exit(VP.V3_FATAL_EXIT)


@pytest.mark.parametrize('how', ['pipe, exit 0', 'file only, exit 1', 'nothing, exit 86'])
def test_the_reason_reaches_the_parent(tmp_path, how):
    """(2026-11-18, the coordinator: 'the reason is in its stderr' was all the parent could say, and the stderr was
    empty) The worker's reason travels on its pipe ('F' + text) and in fatal-worker-<i>-<pid>.txt, both before its
    cleanup; TokSampler's VisibleV3Fatal carries it, from the pipe or else the file; a worker that left neither (exit
    86) is reported as such, plainly."""
    ctx = mp.get_context('spawn')
    s = object.__new__(TokSampler)
    ours, theirs = ctx.Pipe()
    p = ctx.Process(target=_fatal_worker, args=(theirs, how, str(tmp_path)), daemon=True)
    p.start()
    theirs.close()
    s.conns, s.procs, s.index, s.live = [None, ours], [_Proc(0), p], {ours: 1}, [ours]
    s.fatal, s._closing, s._fatal_reason = None, False, {}

    class _Poller:
        def unregister(self, fd):
            pass
    s._poller = _Poller()
    import os
    os.environ['ISAAC_RL_FATAL_DIR'] = str(tmp_path)
    try:
        end, raised = time.time() + 60, None
        while s.live and time.time() < end and raised is None:
            try:
                s.poll(want=1, max_wait=0.5, idle=0.5)
            except VP.VisibleV3Fatal as exc:
                raised = str(exc)
        p.join(10)
        if raised is None:
            with pytest.raises(VP.VisibleV3Fatal) as got:
                s.check_fatal()
            raised = str(got.value)
    finally:
        os.environ.pop('ISAAC_RL_FATAL_DIR', None)
    assert 'tok worker 1 stopped on a visible-v3 protocol error' in raised and 'stderr' not in raised
    if how == 'nothing, exit 86':
        assert 'the reason did not reach the parent' in raised
    else:
        assert 'configured for mode 0' in raised
        assert (tmp_path / f'fatal-worker-1-{p.pid}.txt').read_text().strip().endswith('configured for mode 0')


def test_a_connection_closed_during_the_mode_query_is_retried_not_fatal():
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
            check_visible_v3(c, 0)
    cfg = types.SimpleNamespace(build_timeout=30, reset_timeout=30)
    b = StartBuilder(15, Inst(), cfg, next_seed=lambda: 1)
    b.start()
    assert b.ready.wait(10)
    seed, exc = b.take()
    assert seed == 1 and exc is not None and not isinstance(exc, VP.VisibleV3Fatal)
    assert isinstance(exc, BridgeError) and 'connection closed by game' in str(exc)
