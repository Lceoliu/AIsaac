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


def test_layout_evidence_is_a_distribution(rc):
    from isaac_macro.floor import FloorRoom
    from isaac_macro.secret import LayoutEvidence
    ev = LayoutEvidence(rc, 2, 0)
    for shape, layouts in list(ev.by_shape.items())[:6]:
        variants = [v for (s, v) in ev.layout if s == shape]
        for required in {doors for doors, _ in layouts[:15]}:
            total = sum(ev.prob(FloorRoom(0, 0, 0, shape, 1, v), required) for v in variants)
            assert abs(total - 1.0) < 1e-9, (shape, required, total)


def test_joint_posterior_supports_truth(rc):
    from isaac_macro.secret import LayoutEvidence, hidden_posterior
    for rec, floor in iter_floors(rc, 25, seed=11):
        if len(rec['secret']) != 1 or len(rec['super_secret']) != 1:
            continue
        ps, pss = hidden_posterior(floor.visible(), LayoutEvidence(rc, floor.stage, floor.stage_type))
        assert ps.get(rec['secret'][0], 0) > 0
        assert pss.get(rec['super_secret'][0], 0) > 0
        assert abs(sum(ps.values()) - 1) < 1e-9


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


def test_condition_joint_is_bayes_by_elimination():
    from isaac_macro.secret import condition_joint, joint_marginals
    joint = {(1, 5, 9): 0.4, (1, 6, None): 0.2, (2, 5, 9): 0.3, (3, None, 8): 0.1}
    ps, pss, pus = joint_marginals(joint, 3)
    assert abs(ps[1] - 0.6) < 1e-12 and abs(pss[5] - 0.7) < 1e-12 and abs(pus[9] - 0.7) < 1e-12
    empty = condition_joint(joint, empty=[1])        # bombed cell 1: nothing there
    assert set(empty) == {(2, 5, 9), (3, None, 8)} and abs(empty[(2, 5, 9)] - 0.75) < 1e-12
    found = condition_joint(joint, {2: 9})           # the third room found at cell 9
    assert abs(joint_marginals(found, 3)[0][1] - 4 / 7) < 1e-12
    assert condition_joint(joint, {0: 1}, empty=[5, 6]) == {}   # contradiction


def test_hidden_posterior_conditioning_matches_the_joint(rc):
    from isaac_macro.secret import LayoutEvidence, condition_joint, hidden_joint, hidden_posterior, joint_marginals
    for rec, floor in iter_floors(rc, 6, seed=17):
        if len(rec['secret']) != 1:
            continue
        vis = floor.visible()
        ev = LayoutEvidence(rc, floor.stage, floor.stage_type)
        joint = hidden_joint(vis, ev)
        assert abs(sum(joint.values()) - 1) < 1e-9
        _, pss = hidden_posterior(vis, ev, secret_cell=rec['secret'][0])
        _, expected = joint_marginals(condition_joint(joint, {0: rec['secret'][0]}), 2)
        assert pss.keys() == expected.keys() and all(abs(pss[k] - expected[k]) < 1e-12 for k in pss)


def test_superset_layouts_never_impossible(rc):
    """Where the exact-door layouts would always win with initial weights, a layout with extra door
    slots keeps a small probability (GetRandomRoom's in-floor weight decay is not modelled)."""
    from isaac_macro.floor import FloorRoom
    from isaac_macro.secret import NON_EXACT_FLOOR, LayoutEvidence
    ev = LayoutEvidence(rc, 3, 0)
    checked = 0
    for (shape, variant), (doors, _) in ev.layout.items():
        for required in {d for d, _ in ev.by_shape[shape] if d & doors == d and d != doors}:
            share, total, exact = ev._stats(shape, required)[0]
            if exact and ev.factor * exact / total >= 1:
                p = ev.prob(FloorRoom(0, 0, 0, shape, 1, variant), required)
                assert 0 < p <= NON_EXACT_FLOOR
                checked += 1
    assert checked > 0
