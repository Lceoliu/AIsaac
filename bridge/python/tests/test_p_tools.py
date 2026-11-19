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
