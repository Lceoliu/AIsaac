"""Teacher-v2 records per minibatch step (2026-11-13, tok_teacher2.v2_plan; train_tok L['v2_acc']). main1 never trained
on its CF records: v2_step = int(min(batch, reuse x rate / planned)) was 3-4 (rate 8-16 a update, 15 steps) and a step
draws only at >= 8. Now: int(target) >= 8 the old count every step (bit for bit); 0 < target < 8 an accumulator, a fixed
8 on the steps it reaches 8, carried over updates. Here: the schedule, main1's configuration over many updates, and that
train_tok's prefetch (first_part) and its step draw from the same plan."""
import inspect
import math
import random

from isaac_bridge.tok_teacher2 import V2_MIN, v2_plan


def old_step(batch, reuse, rate, planned):
    return int(min(batch, reuse * rate / planned))


def test_old_rule_at_or_above_eight():
    rnd = random.Random(5)
    for _ in range(2000):
        batch, reuse, planned = rnd.choice((8, 64, 256)), rnd.choice((1.0, 2.0, 4.0)), rnd.randint(1, 40)
        rate = rnd.uniform(0, 3000)
        target = min(batch, reuse * rate / planned)
        if int(target) < V2_MIN:
            continue
        acc0 = rnd.uniform(0, 7.9)
        counts, acc = v2_plan(target, planned, acc0)
        assert counts == [old_step(batch, reuse, rate, planned)] * planned and acc == acc0


def test_below_eight_fixed_batches():
    for target in (0.01, 0.5, 1.0, 3.2, 3.73, 4.0, 7.99):
        acc, total, steps = 0.0, 0, 0
        for update in range(200):
            counts, acc = v2_plan(target, 15, acc)
            assert set(counts) <= {0, V2_MIN}, counts
            assert 0 <= acc < V2_MIN
            total += sum(counts)
            steps += 15
        # the records used: target per step on average; off by less than one batch (the accumulator's rest)
        assert abs(total - target * steps) < V2_MIN, (target, total, target * steps)


def test_nothing_without_a_target():
    assert v2_plan(0.0, 15, 0.0) == ([0] * 15, 0.0)
    assert v2_plan(0.0, 15, 5.5) == ([0] * 15, 5.5)   # (no rate this update: the rest kept, nothing drawn)
    assert v2_plan(3.0, 0, 2.0) == ([], 2.0)


def test_main1_configuration_trains():
    """main1: --teacher-v2-reuse 4 (batch 256), 3 epochs x 5 minibatches, v2_rate 8-16 records an update: the old rule
    drew nothing (v2_step 2-4); now every update has steps with 8 records, ~reuse x rate records an update."""
    acc, used, rate_sum = 0.0, 0, 0.0
    for update in range(100):
        rate = 8 + 8 * (0.5 + 0.5 * math.sin(update / 7))
        target = min(256, 4.0 * rate / 15)
        assert old_step(256, 4.0, rate, 15) < V2_MIN
        counts, acc = v2_plan(target, 15, acc)
        assert sum(c > 0 for c in counts) >= 1, (update, rate, counts)
        used += sum(counts)
        rate_sum += 4.0 * rate
    assert abs(used - rate_sum) < V2_MIN


def test_train_tok_draws_from_the_plan():
    import train_tok
    src = inspect.getsource(train_tok)
    assert "v2_step >= 8 and v2_on" not in src   # (the old gate: neither the prefetch nor the step uses it)
    assert src.count('v2_count(') == 3   # (its definition, first_part's prefetch, the minibatch step)
    assert "vn2 = v2_count(ep2, at2)" in src and "v2_rng.integers(0, L['v2_n'], vn2)" in src
    assert "v2_n_at = v2_count(ep_, at)" in src and "v2_rng.integers(0, L['v2_n'], v2_n_at)" in src
    assert "v_share = args.teacher_v2_weight * v2_n_at / n_sel" in src
    assert "v2_plan(v2_target, args.epochs * len(mb_ats0), L.get('v2_acc', 0.0))" in src
    assert "v2_step = int(v2_target)" in src   # (the progress row's v2_step as before)
