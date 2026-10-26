"""The compact in-flight lineage report keeps the same reachable frames (2026-10-26) — CPU, no model.

lineage_intervals now reports one (ancestor, shared, length) triple per (ancestor, shared): the shortest branch.
For random branch trees (forks of forks, several contexts per branch, two actors' reports merged) the frames
in_flight_frontier marks reachable must equal those of the per-context report (every branch's own triple),
for H = 0 and several horizons.
  python tests/test_lineage_compact.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from isaac_bridge.rebuild_memory import RawHistory
from isaac_bridge.record_retention import lineage_intervals, in_flight_frontier


def frames(front):
    out = set()
    for a, reach in front.items():
        for lo, hi in reach: out.update((a, t) for t in range(max(0, int(lo)), int(hi)))
    return out, set(front)


def per_context(slot_lists, context_dicts):   # (the report before 2026-10-26: one triple per history)
    out = set()
    for raw in slot_lists:
        for h in raw:
            if h is not None and h.length: out.update((a, n, h.length) for a, n in h.origin)
    for table in context_dicts:
        for h in table.values(): out.update((a, n, h.length) for a, n in h.origin)
    return out


def grow(h, n, rng):
    for _ in range(n):
        h = h.append(np.zeros(1, [('x', 'u1')])[0], np.zeros(2, np.int64), 4)
    return h


def main():
    rng = np.random.default_rng(5)
    for trial in range(40):
        roots = [grow(RawHistory.empty(), int(rng.integers(5, 60)), rng) for _ in range(3)]
        hist = list(roots)
        for _ in range(int(rng.integers(3, 15))):   # forks of forks at random points, then their own frames
            p = hist[int(rng.integers(len(hist)))]
            b = p.fork()
            hist.append(grow(b, int(rng.integers(0, 40)), rng))
        lanes = [h for h in hist[3:] if rng.random() < 0.5]
        ctx = {i: h for i, h in enumerate(hist[3:]) if rng.random() < 0.6}
        halves = [lineage_intervals(slot_lists=[lanes[::2]], context_dicts=[ctx]),
                  lineage_intervals(slot_lists=[lanes[1::2]])]
        merged = set().union(*halves)
        old = per_context([lanes], [ctx])
        for H in (0, 3, 10, 50):
            a = frames(in_flight_frontier({}, merged, H))
            b = frames(in_flight_frontier({}, old, H))
            assert a == b, (trial, H)
        assert len(merged) <= len(old)
    print('{"result": "PASS", "trials": 40}')


if __name__ == '__main__':
    main()
