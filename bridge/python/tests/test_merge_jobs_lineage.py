"""In-flight lineage of both actor processes survives merge_jobs (2026-10-26) — CPU, no game, no GPU.

With --actor-processes 2 the learner merges the actors' halves of a hand-off (merge_jobs). Each half reports the
ancestors of its branches in flight (job['lineage']: (ancestor uid, frames shared, branch length)). Before this fix
merge_jobs kept halves[0]'s 'lineage' only: the second actor's ancestors got values only, and a record it published
later read released graphs (the t7d run's first dropped gradient: an episode of actor 1, reach [] when released).
Checks: the merged lineage is the union of both halves; in_flight_frontier then gives the second actor's ancestor
its reachable interval [length - H, shared).
  python tests/test_merge_jobs_lineage.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402,F401  (the test package's import-time stubs)

import numpy as np   # noqa: E402
import train_tok as TT   # noqa: E402
from isaac_bridge.record_retention import in_flight_frontier   # noqa: E402


def half(aid, lineage):
    return dict(aid=aid, seq=7, bid=0, finished=[], branch_finished=[], collect_s=1.0, t_start=0.0, stats=np.zeros((1, 2)),
                lineage=lineage, lineage_src={'lanes': lineage})


def main():
    a0 = [((100, 1), 50, 60)]                          # actor 0: a branch of episode (100, 1) forked at 50, now 60
    a1 = [((200, 7), 182, 189), ((200, 9), 182, 184)]  # actor 1: a grandchild chain forked at 182
    j = TT.merge_jobs([half(0, a0), half(1, a1)])
    got = set(map(tuple, j.get('lineage') or ()))
    assert got == set(a0) | set(a1), ('merged lineage', sorted(got))
    assert set(map(tuple, j['lineage_src']['lanes'])) == set(a0) | set(a1), j['lineage_src']
    H = 1024
    front = in_flight_frontier({}, got, H)
    assert front.get((200, 7)) == [(189 - H, 182)], front
    assert front.get((100, 1)) == [(60 - H, 50)], front
    print('{"result": "PASS", "lineage": %d}' % len(got))


if __name__ == '__main__':
    main()
