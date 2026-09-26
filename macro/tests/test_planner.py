from isaac_macro.dataset import iter_floors
from isaac_macro.planner import Explorer


def test_planner_explores_and_reaches_boss(rc):
    for rec, floor in iter_floors(rc, 15, seed=8):
        res = Explorer(floor, bombs=0).run()
        assert res.reached_boss and res.bombs_used == 0
        assert res.rooms_visited >= res.rooms_total - 2       # only rooms behind a boss room are skipped
        assert not res.found_secret and not res.found_super_secret
        res = Explorer(floor, bombs=3).run()
        assert res.bombs_used <= 3
        hits = [t for t in res.trace if t[0] == 'bomb' and t[2]]
        assert len(hits) == res.found_secret + res.found_super_secret
        if res.found_secret:
            assert any(t[1] in rec['secret'] for t in hits)
