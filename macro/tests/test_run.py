import random

from isaac_macro.level import CURSE_LABYRINTH, GameContext
from isaac_macro.rng import Seeds
from isaac_macro.run import generate_run, next_stage


def test_next_stage_types_follow_seed_and_unlocks():
    rnd = random.Random(1)
    locked = GameContext(achievements=frozenset())
    unlocked = GameContext()
    for _ in range(300):
        seeds = Seeds(rnd.getrandbits(32) or 1)
        for stage in range(0, 8):
            nxt, st = next_stage(locked, seeds, stage, 0, 0)
            seed = seeds.stage_seed(nxt)
            assert nxt == stage + 1
            assert st == (1 if nxt in (7, 8) and seed % 2 == 0 else 0)
            nxt, st = next_stage(unlocked, seeds, stage, 0, 0)
            assert st == (2 if seed % 3 == 0 else 1 if seed % 2 == 0 else 0)


def test_labyrinth_skips_a_floor():
    ctx, seeds = GameContext(), Seeds(77)
    assert next_stage(ctx, seeds, 1, 0, CURSE_LABYRINTH)[0] == 3
    assert next_stage(ctx, seeds, 2, 0, CURSE_LABYRINTH)[0] == 3
    assert next_stage(ctx, seeds, 7, 0, CURSE_LABYRINTH)[0] == 10
    assert next_stage(ctx, seeds, 8, 0, 0) == (10, 0)
    assert next_stage(ctx, seeds, 8, 0, 0, cathedral=True) == (10, 1)
    assert next_stage(ctx, seeds, 10, 1, 0) == (11, 1)


def test_generate_run(rc):
    floors = generate_run(rc, 'SXG6 R8XA', last_stage=11)
    stages = [f.stage for f in floors]
    assert stages[0] == 1 and stages[-1] == 11
    assert stages == sorted(stages)
    floors = generate_run(rc, 12345, debug_start=True, last_stage=2)
    assert (floors[0].stage, floors[0].stage_type) == (1, 0)
