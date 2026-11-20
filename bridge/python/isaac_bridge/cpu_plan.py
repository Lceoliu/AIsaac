"""(Codex round 22, items 1 and 2; 2026-11-20) The trainer's CPU plan checked before any worker starts, and the actor
main thread's mask read back after it is set.

A lane (the trainer's affinity when it starts: the container's cpuset, or taskset's set in direct mode) is split into
groups: the learner's (ISAAC_LEARNER_CPUS / --learner-cpus; not exclusive with ISAAC_LEARNER_CPUS_SHARED=1), the
driver's (ISAAC_DRIVER_CPUS: the C++ driver thread, and with ISAAC_TOK_PIN=1 the actor's main thread), the actor's own
(ISAAC_ACTOR_CPUS) and the workers' (the rest). plan_problems() lists what is wrong with a plan:
  - ISAAC_TOK_PIN=1 without a non-empty ISAAC_DRIVER_CPUS (before: the pin silently off);
  - a group with CPUs outside the lane (before: a partly outside mask was set, the log printing the request);
  - two exclusive groups sharing a logical CPU, or a physical core (SMT siblings, /sys .../thread_siblings_list): an
    exclusive group's CPUs' siblings in the lane must all be in that group;
  - an explicit worker / search mask (ISAAC_RL_WORKER_CPUS / ISAAC_RL_SEARCH_CPUS: tok_sampler sets them itself, past
    the trainer's subtraction) not inside the workers' CPUs.
readback_problem() compares a thread's mask read back after setting it with the request (exactly equal)."""
import os


def parse_cpus(spec):
    """'12-16,36-40' -> {12, ..., 16, 36, ..., 40} (empty / None -> set())."""
    out = set()
    for part in filter(None, (spec or '').replace(' ', '').split(',')):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def siblings_of(cpus, root='/sys/devices/system/cpu', warn=True):
    """{cpu: its SMT siblings (itself included)} from sysfs; a CPU whose file is not readable is its own only -- and
    then (A1's review) a line says so: the physical-core check cannot see that CPU's siblings."""
    out, unread = {}, []
    for c in cpus:
        try:
            with open(f'{root}/cpu{c}/topology/thread_siblings_list') as f:
                out[c] = parse_cpus(f.read().strip()) | {c}
        except (OSError, ValueError):
            out[c] = {c}
            unread.append(c)
    if unread and warn:
        import sys
        print(f'CPU plan: WARNING the SMT siblings of CPUs {sorted(unread)} are not readable ({root}): the '
              f'physical-core check does not cover them', file=sys.stderr, flush=True)
    return out


def plan_problems(lane, learner=(), driver=(), actor=(), pin=False, learner_shared=False, worker_masks=None,
                  siblings=None):
    """The problems of a CPU plan (empty: fine). lane: the trainer's CPUs at its start; learner / driver / actor: the
    groups' CPUs; pin: ISAAC_TOK_PIN=1; worker_masks: {name: CPUs} of explicit worker / search masks; siblings:
    {cpu: siblings} (default: sysfs)."""
    lane, learner, driver, actor = set(lane), set(learner), set(driver), set(actor)
    bad = []
    if pin and not driver:
        bad.append('ISAAC_TOK_PIN=1 but no ISAAC_DRIVER_CPUS: the actor main thread would not be pinned')
    if actor and not (pin and driver):
        bad.append('ISAAC_ACTOR_CPUS needs ISAAC_TOK_PIN=1 and ISAAC_DRIVER_CPUS')
    groups = [('learner', learner, not learner_shared), ('driver', driver, True), ('actor', actor, True)]
    for name, g, _ in groups:
        if g - lane:
            bad.append(f'{name} CPUs {sorted(g - lane)} are not in the lane {sorted(lane)}')
    excl = [(n, g) for n, g, e in groups if g and e]
    for i, (n1, g1) in enumerate(excl):
        for n2, g2 in excl[i + 1:]:
            if g1 & g2:
                bad.append(f'{n1} and {n2} CPUs overlap: {sorted(g1 & g2)}')
    if siblings is None:
        siblings = siblings_of(lane)
    for n, g in excl:   # an exclusive group's physical cores: no sibling of it in the lane outside the group
        for c in sorted(g & lane):
            other = (siblings.get(c, {c}) & lane) - g
            if other:
                bad.append(f'{n} CPU {c} shares its physical core with lane CPUs {sorted(other)} outside the {n} group')
    reserved = set().union(*(g for _, g in excl)) if excl else set()
    workers = lane - reserved
    for name, m in (worker_masks or {}).items():
        m = set(m)
        if m and not m <= workers:
            bad.append(f'{name} {sorted(m)} is not inside the workers\' CPUs {sorted(workers)} '
                       f'(outside: {sorted(m - workers)})')
    return bad


def readback_problem(what, requested, got):
    """None when a mask read back equals the request, else the message."""
    if set(got) != set(requested):
        return f'{what}: requested CPUs {sorted(requested)}, read back {sorted(got)}'
    return None


def env_worker_masks(env=None):
    """The explicit masks tok_sampler applies itself (ISAAC_RL_WORKER_CPUS, ISAAC_RL_SEARCH_CPUS), parsed."""
    env = os.environ if env is None else env
    return {k: parse_cpus(env.get(k, '')) for k in ('ISAAC_RL_WORKER_CPUS', 'ISAAC_RL_SEARCH_CPUS') if env.get(k)}
