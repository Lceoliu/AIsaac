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
    def __init__(self, code):
        self.exitcode = code

    def join(self, timeout=None):
        pass


def _sampler(codes):
    s = object.__new__(TokSampler)
    s.procs = [_Proc(c) for c in codes]
    s.fatal, s._closing = None, False
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
    s.fatal, s._closing = None, False
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
