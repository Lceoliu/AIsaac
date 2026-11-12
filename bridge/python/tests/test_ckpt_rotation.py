"""The checkpoint rotation's choice (isaac_bridge.ckpt_rotation): kept = the newest N + every milestone, exactly."""
from isaac_bridge.ckpt_rotation import rotation_drop, update_of


def _names(ups):
    return [f'update-{u:06d}.pt' for u in ups]


def _kept(ups, keep, mile):
    names = _names(ups) + ['last.pt', 'last-rank1.pt', 'update-12.pt', 'tmp_update-000300.pt']
    drop = set(rotation_drop(names, keep, mile))
    return {update_of(n) for n in names if update_of(n) is not None and n not in drop}, drop


def test_newest_n_plus_milestones_outside_them():
    ups = list(range(25, 1001, 25))   # (the training configuration: a checkpoint every 25 updates)
    kept, drop = _kept(ups, 5, 250)
    assert kept == {900, 925, 950, 975, 1000} | {250, 500, 750}   # (1000 both newest and a milestone)
    assert all(n.startswith('update-') and len(n) == 16 for n in drop)   # (never last*.pt, a tmp_ file, an odd name)
    assert len(drop) == len(ups) - len(kept)


def test_milestone_not_among_the_newest():
    kept, _ = _kept([3, 6, 9, 12, 15], 2, 6)
    assert kept == {12, 15, 6}   # (6: a milestone outside the newest two; 12 both)


def test_unsorted_input_and_fewer_than_keep():
    assert rotation_drop(_names([9, 3, 6]), 5, 0) == []
    assert rotation_drop(_names([9, 3, 6]), 1, 0) == _names([3, 6])
    assert rotation_drop(_names([9, 3, 6]), 0, 0) == []   # (keep 0: the rotation off)
