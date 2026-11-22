"""P package (Codex round 18-4): the gate tool's bitwise comparison takes a bf16 tensor's finiteness from its values
before comparing its bits -- the same NaN / inf bit pattern on both sides is not 'finite and equal'."""
import os
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import p_chain  # noqa: E402


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_bf16_nonfinite_is_not_equal(bad):
    t = torch.tensor([[1.0, bad], [2.0, 3.0]], dtype=torch.bfloat16)
    r = p_chain.bitdiff(p_chain.tensor_bits(t), p_chain.tensor_bits(t.clone()))
    assert r['n'] == 2 and not r['finite'] and r['equal'] == 1, r


def test_bf16_equal_rows_are_equal():
    t = torch.tensor([[1.0, -0.0], [2.0, 3.0]], dtype=torch.bfloat16)
    r = p_chain.bitdiff(p_chain.tensor_bits(t), p_chain.tensor_bits(t.clone()))
    assert r['finite'] and r['equal'] == 2, r
    u = t.clone()
    u[0, 1] = 0.0   # (+0 against -0: different bits)
    assert p_chain.bitdiff(p_chain.tensor_bits(t), p_chain.tensor_bits(u))['equal'] == 1


def test_p_cpuacct_capacity_is_the_lane_not_the_pinned_mask(monkeypatch):
    """(Codex round 22) p_cpuacct's capacity: the trainer's container's effective cpuset, not the trainer's own mask --
    ISAAC_TOK_PIN narrows that to the actor's CPUs (before: a lane of 8 logical CPUs read as 2, idle -4)."""
    import importlib
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
    acct = importlib.import_module('p_cpuacct')
    files = {'/proc/7/cgroup': b'0::/docker/abc\n',
             '/sys/fs/cgroup/docker/abc/cpuset.cpus.effective': b'12-15,36-39\n',
             '/proc/7/status': b'Name:\tpython\nCpus_allowed_list:\t19,43\n'}
    monkeypatch.setattr(acct, 'rd', lambda path: files.get(path))
    assert acct.lane_of(7) == '12-15,36-39' and len(acct.cpu_set(acct.lane_of(7))) == 8
    assert acct.allowed(7) == '19,43'   # (the pinned mask: listed under 'masks', not the capacity)
    files.pop('/sys/fs/cgroup/docker/abc/cpuset.cpus.effective')
    assert acct.lane_of(7) is None
