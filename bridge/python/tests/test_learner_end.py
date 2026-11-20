"""(2026-11-20, Codex round 19) the trainer's end of the learner process (train_tok.end_learner_process): a run
succeeds only with the learner's 'done' message and exit code 0. Codex's stub: a learner process dead with exit code
-9 and no 'done' (killed after its last fresh weights, before last.pt) -- before the fix the trainer printed
'finished' and returned 0; now it is a failure. Also: an 'error' message, a nonzero exit after 'done', a learner that
does not end (SIGTERM ignored: killed)."""
import queue

import train_tok as T


class Proc:
    def __init__(self, exitcode, alive=False, ignore_term=False):
        self.exitcode, self.alive, self.ignore_term = exitcode, alive, ignore_term
        self.killed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        if not self.ignore_term:
            self.alive, self.exitcode = False, -15

    def kill(self):
        self.alive, self.exitcode, self.killed = False, -9, True


def _q(*msgs):
    q = queue.Queue()
    for m in msgs:
        q.put(m)
    return q


DONE = ('done', dict(update=12, decisions=34, seconds=5.0))


def test_killed_before_done_is_a_failure():
    failure, L = [], {}
    got = T.end_learner_process(Proc(-9), _q(), failure, L, wait_s=2, join_s=0, kill_s=0)
    assert not got and len(failure) == 1 and 'without its done message (exit code -9)' in str(failure[0])


def test_done_and_exit_0_is_success():
    failure, L = [], {}
    assert T.end_learner_process(Proc(0), _q(DONE), failure, L, wait_s=2, join_s=0)
    assert failure == [] and L['update'] == 12


def test_done_then_nonzero_exit_is_a_failure():
    failure = []
    T.end_learner_process(Proc(1), _q(DONE), failure, {}, wait_s=2, join_s=0)
    assert len(failure) == 1 and 'exit code 1 after its done' in str(failure[0])


def test_error_message_is_a_failure():
    failure = []
    T.end_learner_process(Proc(1), _q(('error', 'Traceback ...')), failure, {}, wait_s=2, join_s=0)
    assert any('learner process:' in str(f) for f in failure)


def test_a_learner_that_ignores_sigterm_is_killed_and_a_failure():
    p, failure = Proc(None, alive=True, ignore_term=True), []
    T.end_learner_process(p, _q(DONE), failure, {}, wait_s=2, join_s=0, kill_s=0)
    assert p.killed and not p.is_alive() and len(failure) == 1 and 'killed' in str(failure[0])
