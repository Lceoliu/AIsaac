import os

import pytest

from isaac_macro.items import HARMFUL, WEAPON_CHANGERS, ItemTable, default_rep_metadata_path
from isaac_macro.roomconfig import default_archive_path


@pytest.fixture(scope='module')
def table():
    if not os.path.exists(default_archive_path()):
        pytest.skip('AB+ archive not found')
    return ItemTable()


def test_counts(table):
    cols = table.collectibles()
    assert len(cols) == 549 and max(c.id for c in cols) == 552
    assert sum(1 for (k, _) in table.items if k == 'trinket') == 127
    assert len(table.pools) == 26


def test_hand_lists_match_names(table):
    for d in (WEAPON_CHANGERS, HARMFUL):
        for i, name in d.items():
            assert table.collectible(i).name == name


def test_rep_quality(table):
    if not os.path.exists(default_rep_metadata_path()):
        pytest.skip('Rep+ metadata not found')
    assert sum(1 for c in table.collectibles() if c.quality is not None) >= 540
    assert {c.id for c in table.collectibles() if c.quest} >= {238, 239, 327, 328}
    s = table.pool_quality('treasure')
    assert abs(sum(s['dist']) - 1) < 1e-9
