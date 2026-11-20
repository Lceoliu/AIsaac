"""(Codex round 22, items 1 and 2) isaac_bridge.cpu_plan: the trainer's CPU plan refused before any worker starts, and
the actor main thread's mask read back. Pure functions: no affinity is set (runs anywhere). The cases: Codex's two CPU
stubs (the pin requested without driver CPUs -- before: silently off; two CPUs requested, one allowed -- before: set,
the log printing two), a worker mask holding the actor's core, two groups on one physical core's two threads."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from isaac_bridge.cpu_plan import parse_cpus, plan_problems, readback_problem, siblings_of   # noqa: E402

LANE_B = parse_cpus('12-19,36-43')
SIB = {c: {c, c + 24} if c < 24 else {c, c - 24} for c in LANE_B}   # (184: CPU n's sibling is n + 24)


def test_the_lanes_own_plans_pass():
    for learner, driver, actor in (('17-18,41-42', '19,43', ''), ('17,41', '19,43', ''), ('17,41', '19,43', '18,42')):
        assert plan_problems(LANE_B, parse_cpus(learner), parse_cpus(driver), parse_cpus(actor), pin=True,
                             siblings=SIB) == []


def test_pin_without_driver_cpus_is_refused():   # (Codex stub 1: before, ISAAC_TOK_PIN=1 left the mask as it was)
    bad = plan_problems(LANE_B, parse_cpus('17,41'), set(), pin=True, siblings=SIB)
    assert len(bad) == 1 and 'no ISAAC_DRIVER_CPUS' in bad[0]


def test_a_group_partly_outside_the_lane_is_refused():   # (Codex stub 2: two requested, one allowed)
    lane = LANE_B - {43}
    bad = plan_problems(lane, parse_cpus('17,41'), parse_cpus('19,43'), pin=True, siblings=SIB)
    assert any('driver CPUs [43] are not in the lane' in b for b in bad), bad


def test_readback_must_equal_the_request():   # (the same stub after the set: the kernel kept one of two)
    assert readback_problem('actor', {19, 43}, {19, 43}) is None
    msg = readback_problem('actor', {19, 43}, {19})
    assert msg and 'requested CPUs [19, 43], read back [19]' in msg


def test_a_worker_mask_holding_the_actors_core_is_refused():
    bad = plan_problems(LANE_B, parse_cpus('17,41'), parse_cpus('19,43'), pin=True, siblings=SIB,
                        worker_masks={'ISAAC_RL_WORKER_CPUS': parse_cpus('12-16,19,36-40')})
    assert len(bad) == 1 and 'ISAAC_RL_WORKER_CPUS' in bad[0] and 'outside: [19]' in bad[0]
    assert plan_problems(LANE_B, parse_cpus('17,41'), parse_cpus('19,43'), pin=True, siblings=SIB,
                         worker_masks={'ISAAC_RL_SEARCH_CPUS': parse_cpus('12-16,18,36-40,42')}) == []


def test_two_groups_on_one_physical_core_are_refused():
    # the actor on 18, the driver on 42: CPU 18's sibling (one physical core's two threads)
    bad = plan_problems(LANE_B, parse_cpus('17,41'), parse_cpus('42,19,43'), parse_cpus('18'), pin=True, siblings=SIB)
    assert any('shares its physical core' in b for b in bad), bad
    # a group taking one thread of a core whose other thread the workers keep
    bad = plan_problems(LANE_B, parse_cpus('17'), parse_cpus('19,43'), pin=True, siblings=SIB)
    assert any('learner CPU 17 shares its physical core with lane CPUs [41]' in b for b in bad), bad
    # the learner shared with the workers (ISAAC_LEARNER_CPUS_SHARED=1): not an exclusive group
    assert plan_problems(LANE_B, parse_cpus('17'), parse_cpus('19,43'), pin=True, learner_shared=True,
                         siblings=SIB) == []


def test_overlapping_groups_and_actor_without_pin():
    bad = plan_problems(LANE_B, parse_cpus('17,41'), parse_cpus('17,19,41,43'), pin=True, siblings=SIB)
    assert any('learner and driver CPUs overlap' in b for b in bad), bad
    bad = plan_problems(LANE_B, parse_cpus('17,41'), parse_cpus('19,43'), parse_cpus('18,42'), pin=False,
                        siblings=SIB)
    assert any('ISAAC_ACTOR_CPUS needs' in b for b in bad), bad


def test_siblings_from_sysfs(tmp_path):
    for c, text in ((3, '3,27'), (27, '3,27')):
        d = tmp_path / f'cpu{c}' / 'topology'
        d.mkdir(parents=True)
        (d / 'thread_siblings_list').write_text(text + '\n')
    assert siblings_of({3, 27, 5}, root=str(tmp_path)) == {3: {3, 27}, 27: {3, 27}, 5: {5}}
