"""train_tok.lr_at: the learning-rate schedule over the run's progress (--lr-schedule linear | cosine, --lr-warmup);
the progress itself is the whole run's with ISAAC_GAME_HOURS_TOTAL (prog0 0), so a resume continues the curve."""
import math

import numpy as np

from train_tok import lr_at


def test_linear_no_warmup_is_the_old_schedule():
    for f in np.linspace(0, 1, 101):
        assert lr_at(f, 3e-4, 1e-4) == 3e-4 + (1e-4 - 3e-4) * f


def test_cosine_with_warmup():
    lr0, lr1, w = 3e-4, 3e-5, 0.01
    assert lr_at(0.0, lr0, lr1, 'cosine', w) == 0.0
    assert math.isclose(lr_at(w / 2, lr0, lr1, 'cosine', w), lr0 / 2)
    assert math.isclose(lr_at(w, lr0, lr1, 'cosine', w), lr0)
    mid = w + (1 - w) / 2
    assert math.isclose(lr_at(mid, lr0, lr1, 'cosine', w), (lr0 + lr1) / 2)
    assert math.isclose(lr_at(1.0, lr0, lr1, 'cosine', w), lr1)
    xs = np.linspace(w, 1, 500)
    ys = [lr_at(x, lr0, lr1, 'cosine', w) for x in xs]
    assert all(a >= b for a, b in zip(ys, ys[1:]))   # (non-increasing after the warmup)


def test_resume_continues_the_curve():
    """A run stopped at update 1000 of 4125 and resumed: the progress is cumulative (prog0 0), so the rates of the
    updates just before and after the resume are neighbours on the one curve, not a restart at the warmup."""
    per, total = 32768, 4125 * 32768
    before = lr_at(1000 * per / total, 3e-4, 3e-5, 'cosine', 0.01)
    after = lr_at(1001 * per / total, 3e-4, 3e-5, 'cosine', 0.01)
    assert 0 < before - after < 1e-6 and after > 1e-4
