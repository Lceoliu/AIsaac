"""Hidden-room inference on J460 floors (isaac_macro.rep.secret), checked against the J460 port's own
floors: the rules must never rule out the generated rooms, and the required doors the model derives
for normal rooms must be the generator's. Static port only, not engine agreement."""
from types import SimpleNamespace

from isaac_macro.floor import Floor, FloorRoom
from isaac_macro.levelgen import GRID
from isaac_macro.rep.dataset import iter_floors
from isaac_macro.rep.roomconfig import stage_id
from isaac_macro.rep.secret import (HIDDEN_TYPES, RepLayoutEvidence, UltraSecretRule, blocked_cells,
                                    hidden_posterior, strange_door_floor)


def _floors(rep_rc, runs, seed):
    for rec, floor in iter_floors(rep_rc, runs, seed, last_stage=11, debug_start=True):
        yield rec, floor, floor.visible(HIDDEN_TYPES)


def test_rep_layout_evidence_is_a_distribution(rep_rc):
    for stage, labyrinth in ((2, False), (1, True), (6, False)):
        ev = RepLayoutEvidence(rep_rc, stage, 0, labyrinth=labyrinth)
        assert len(ev.components) == (2 if stage % 2 == 0 and not labyrinth else 1)
        for shape, layouts in list(ev.by_shape.items())[:5]:
            variants = [v for (s, v) in ev.layout if s == shape]
            for required in {doors for doors, _ in layouts[:12]}:
                total = sum(ev.prob(FloorRoom(0, 0, 0, shape, 1, v), required) for v in variants)
                assert abs(total - 1.0) < 1e-9, (stage, shape, required, total)


def test_floor_from_level_skips_off_grid_rooms():
    cfg = SimpleNamespace(type=1, variant=2, subtype=0, doors=0xF, difficulty=1, name='start', stage=0)
    on = SimpleNamespace(list_index=0, grid_index=0x54, safe_grid_index=0x54, shape=1, config=cfg, dimension=0)
    off = SimpleNamespace(list_index=0x1F8, grid_index=-2, safe_grid_index=-2, shape=1, config=None, dimension=0)
    other = SimpleNamespace(list_index=1, grid_index=0x55, safe_grid_index=0x55, shape=1, config=cfg, dimension=1)
    level = SimpleNamespace(rooms=[on, off, other], generator=None, stage=1, stage_type=0, curses=0, stage_seed=1)
    floor = Floor.from_level(level)
    assert [r.index for r in floor.rooms] == [0] and floor.rooms[0].file == 0


def test_required_doors_equal_generator_doors(rep_rc):
    """Normal rooms take their layouts with the generator's doors as required doors; the model's
    version is the visible doors, the slots towards the secret and super secret rooms, and the ultra
    secret room's red-room slots. They must agree room by room."""
    from isaac_macro.rep.level import RepGameContext
    from isaac_macro.rep.rng import Seeds
    from isaac_macro.rep.run import iter_run
    checked = 0
    for seed in (5, 77, 1234, 99991):
        for level in iter_run(rep_rc, Seeds(seed), RepGameContext(), last_stage=11, debug_start=True):
            floor = Floor.from_level(level)
            vis = floor.visible(HIDDEN_TYPES)
            gen = level.generator
            hidden = [r.safe_grid_index for r in floor.rooms if r.type in (7, 8)]
            ultra = [r.safe_grid_index for r in floor.rooms if r.type == 29]
            rule = UltraSecretRule(vis, blocked_cells(vis, strange_door_floor(level.stage, level.stage_type, level.curses)))
            red = rule.red_slots(ultra[0]) if ultra else {}
            sid = stage_id(level.stage, level.stage_type)
            for r in vis.rooms:
                if r.type != 1 or r.grid_index == vis.start or r.file != sid or r.subtype != 0:
                    continue
                expected = r.doors | red.get(r.index, 0)
                for slot in range(8):
                    if r.slot_target(slot) in hidden:
                        expected |= 1 << slot
                assert expected == gen.rooms[gen.grid[r.safe_grid_index]].doors, (seed, level.stage, r.index)
                checked += 1
    assert checked > 100


def test_rules_support_the_generated_rooms(rep_rc):
    ev_cache = {}
    for rec, floor, vis in _floors(rep_rc, 10, 3):
        truth = {k: rec[k][0] for k in ('secret', 'super_secret', 'ultra_secret') if len(rec[k]) == 1}
        blocked = blocked_cells(vis, strange_door_floor(floor.stage, floor.stage_type, floor.curses))
        assert truth.get('secret') not in blocked
        if 'ultra_secret' in truth:
            rule = UltraSecretRule(vis, blocked)
            assert rule.given(truth.get('secret'), truth.get('super_secret')).get(truth['ultra_secret'], 0) > 0
        key = (floor.stage, floor.stage_type, bool(floor.curses & 2))
        if key not in ev_cache:
            ev_cache[key] = RepLayoutEvidence(rep_rc, floor.stage, floor.stage_type, labyrinth=key[2])
        sd = strange_door_floor(floor.stage, floor.stage_type, floor.curses)
        ps, pss, pus = hidden_posterior(vis, ev_cache[key], strange_door=sd)
        assert abs(sum(ps.values()) - 1) < 1e-9
        for name, post in (('secret', ps), ('super_secret', pss), ('ultra_secret', pus)):
            if name in truth:
                assert post.get(truth[name], 0) > 0, (rec['run_seed'], floor.stage, name)
        assert all(0 <= c < GRID * GRID for c in pus)
