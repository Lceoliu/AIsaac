"""BM1 P2: a part split in two (block_learner.prepare_split + run_prepared on two model copies, the history drawn once
for the whole part) gives the whole part's gradient (forward_packed) up to float32 summation order: the halves'
gradients add up, the re-encoded historical blocks shared by both halves and the teachers' records in the second half
included. CPU, float32, Spatial checkpoints on and off."""
import copy

import pytest
import torch

from test_block_packed_learner import _rollout_and_teacher
from isaac_bridge import block_learner


def _loss(outs):
    g = torch.Generator().manual_seed(3)
    return sum((o.double()*torch.randn(o.shape, generator=g, dtype=torch.float64)).sum() for o in outs)


def _grads(m):
    return {k: p.grad.detach().double().clone() for k, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize('ckpt', [1, 0])
def test_split_halves_add_up_to_the_whole_part(ckpt, monkeypatch):
    monkeypatch.setattr(block_learner, 'SPATIAL_CKPT', ckpt)
    m, chunks, teach = _rollout_and_teacher('cpu', seed=21)
    m.train()
    ch = chunks+teach
    m.zero_grad(set_to_none=True)
    outs, _ = block_learner.forward_packed(m, ch, generator=torch.Generator().manual_seed(5))
    _loss(outs).backward()
    gw = _grads(m)
    m2 = copy.deepcopy(m)
    m.zero_grad(set_to_none=True)
    m2.zero_grad(set_to_none=True)
    cut = len(ch)//2
    with block_learner.PREP_LOCK:
        P0, P1 = block_learner.prepare_split((m, m2), ch, cut, True, generator=torch.Generator().manual_seed(5))
    o0, _ = block_learner.run_prepared(m, P0)
    o1, _ = block_learner.run_prepared(m2, P1)
    assert len(o0[0])+len(o1[0]) == len(outs[0])
    _loss([torch.cat((x, y)) for x, y in zip(o0, o1)]).backward()
    g0, g1 = _grads(m), _grads(m2)
    keys = sorted(gw)
    assert set(g0) | set(g1) == set(gw)
    a = torch.cat([gw[k].reshape(-1) for k in keys])
    b = torch.cat([((g0[k] if k in g0 else 0)+(g1[k] if k in g1 else 0)).reshape(-1) for k in keys])
    rel = float((a-b).norm()/a.norm())
    assert rel <= 1e-5, rel
