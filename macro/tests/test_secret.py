import itertools
from fractions import Fraction

import pytest

from isaac_macro.dataset import iter_floors
from isaac_macro.secret import (SCORE_BASE, SCORE_SPAN, secret_posterior, super_secret_candidates,
                                win_probabilities)


def brute_force(offsets):
    cells = list(offsets)
    out = {c: Fraction(0) for c in cells}
    for us in itertools.product(range(SCORE_SPAN), repeat=len(cells)):
        scores = [SCORE_BASE + u + offsets[c] for u, c in zip(us, cells)]
        best = max(scores)
        winners = [c for c, s in zip(cells, scores) if s == best]
        for c in winners:
            out[c] += Fraction(1, len(winners) * SCORE_SPAN ** len(cells))
    return out


@pytest.mark.parametrize('offsets', [
    {1: 0}, {1: 0, 2: 0}, {1: -6, 2: 0}, {1: -3, 2: 0, 3: 0}, {1: -6, 2: -3, 3: -3, 4: 0},
    {1: -6, 2: -6, 3: -3, 4: -3, 5: 0},
])
def test_win_probabilities_exact(offsets):
    exact = brute_force(offsets)
    ours = win_probabilities(offsets)
    for c in offsets:
        assert abs(ours[c] - float(exact[c])) < 1e-12
    assert abs(sum(ours.values()) - 1) < 1e-12


def test_posterior_supports_generated_truth(rc):
    for rec, floor in iter_floors(rc, 40, seed=3):
        vis = floor.visible()
        if rec['secret']:
            truth = rec['secret'][0]
            ss = rec['super_secret'][0] if rec['super_secret'] else None
            assert secret_posterior(vis, ss).get(truth, 0) > 0
            assert secret_posterior(vis).get(truth, 0) > 0
        if rec['super_secret']:
            assert rec['super_secret'][0] in super_secret_candidates(vis)
