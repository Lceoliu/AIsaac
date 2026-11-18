"""seq-v1 step 2: TokPolicy(frame='v3') -- FrameEncoderV3 on the decoded v3 frames, T1's temporal side and heads.
CPU only, a small model (width 64) on the 19 golden frames (tests/fixtures/visible_v3) as actor steps.

- forward and backward: logits of the first arm's five heads (20), finite values and gradients;
- the drop head: the policy layer holds 22 logits (schema_v3.HEAD_SIZES), the forward gives 20, drop's two rows get no
  gradient, drop's "drop" bias is schema_v3.DROP_BIAS, the first 20 biases as T1's;
- no ROW observation field enters: the ROW fields of the batch randomised (all but those the temporal step and the
  gate read), the outputs bit for bit the same;
- the configuration: frame=v3 needs T1's (temporal, feedback 0, global token, items, no item tokens, no block memory,
  no GRU); TokPolicy(**config) rebuilds the same model; a model without frame has no frame_v3 and the old modules."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402

from isaac_bridge import schema_v3 as V3   # noqa: E402
from isaac_bridge import tok_policy as TP   # noqa: E402
from isaac_bridge import v3_input as VI   # noqa: E402
from isaac_bridge.frame_v3 import v3_tensors   # noqa: E402

FIX = HERE / 'fixtures' / 'visible_v3'
NAMES = [f['name'] for f in json.loads((FIX / 'manifest.json').read_text())['frames']]
CFG = dict(items=True, width=64, layers=2, heads=4, global_token=True, temporal=True, temporal_width=32,
           temporal_layers=2, temporal_heads=4, temporal_window=8, temporal_rooms=0, temporal_feedback=0, frame='v3')


def build(seed=3):
    torch.manual_seed(seed)
    return TP.TokPolicy(**CFG).eval()


def batch(names, seed=0):
    """A to_batch of synthetic ROW records (the temporal / gate fields) plus the v3 frames of `names`."""
    rng = np.random.default_rng(seed)
    rows = TS.make_seq(rng, nw=len(names), T=1)[0]
    m = build()
    pend = np.stack([rng.integers(0, k, len(names)) for k in m.heads], 1).astype(np.int64)
    b = TP.to_batch(rows, pend, 'cpu')
    frames = [VI.decode_frame((FIX / f'{n}.v3.bin').read_bytes(), V3.VERSION) for n in names]
    b['v3'] = v3_tensors(VI.pad_batch(frames), 'cpu')
    return b


def step(m, b, st=None):
    st = st or TP.TemporalState(m, len(b['pending']), 'cpu', torch.float32)
    td = dict(mode='step', state=st, slot=torch.arange(len(b['pending'])))
    with torch.no_grad():
        return m(b, need_vg=True, temporal=td)


def test_forward_backward_and_the_drop_head():
    m = build()
    b = batch(NAMES[:6])
    with torch.no_grad():   # (the actor's step: in place on the temporal state, no gradient)
        logits, value, danger, vg = step(m, b)
    assert logits.shape == (6, sum(m.heads)) == (6, 20) and value.shape == (6,) and vg.shape == (6,)
    for t in (logits, value, danger, vg):
        assert torch.isfinite(t).all()
    m.train()   # (the gradient through the frame encoder and the heads: forward without a temporal context)
    logits, value, danger, vg = m(b, need_vg=True)
    assert logits.shape == (6, 20)
    (logits.square().sum() + value.sum() + vg.sum()).backward()
    assert m.pi.weight.shape[0] == sum(V3.HEAD_SIZES) == 22
    assert m.pi.weight.grad[20:].abs().max() == 0 and m.pi.bias.grad[20:].abs().max() == 0
    assert m.pi.weight.grad[:20].abs().max() > 0
    fe = [p for n, p in m.named_parameters() if n.startswith('frame_v3.') and p.grad is not None]
    assert fe and all(torch.isfinite(p.grad).all() for p in fe)
    ref = TP.TokPolicy(**dict(CFG, frame=None, temporal_feedback=0))
    assert torch.equal(m.pi.bias.detach()[:20], ref.pi.bias.detach())
    assert float(m.pi.bias[21]) == V3.DROP_BIAS and float(m.pi.bias[20]) == 0.0


def test_no_row_observation_field_enters():
    m = build()
    TS.perturb_all(m, 7)
    b = batch(NAMES[6:12])
    ref = step(m, b)
    g = torch.Generator().manual_seed(9)
    keep = {'v3', 'pending', 'first', 'map', 'events', 'stage'}   # (the temporal step's record fields)
    c = dict(b)
    for k, v in b.items():
        if k not in keep and torch.is_tensor(v) and v.is_floating_point():
            c[k] = torch.randn(v.shape, generator=g).to(v.dtype)
        elif k not in keep and torch.is_tensor(v) and v.dtype in (torch.int64, torch.int32, torch.uint8):
            c[k] = (v + 1) % 3
    got = step(m, c)
    for a, b_ in zip(ref, got):
        assert torch.equal(a, b_)


def test_the_action_under_way_reaches_the_controlled_body():
    m = build()
    TS.perturb_all(m, 8)
    b = batch(NAMES[:4])
    ref = step(m, b)[0]
    c = dict(b, pending=(b['pending'] + 1) % torch.tensor(m.heads))
    assert not torch.equal(ref, step(m, c)[0])


def test_configuration():
    m = build()
    assert m.config['frame'] == 'v3' and m.frame_v3 is not None
    r = TP.TokPolicy(**m.config)
    assert {k: v.shape for k, v in r.state_dict().items()} == {k: v.shape for k, v in m.state_dict().items()}
    for name in ('player', 'entity', 'room', 'blocks', 'type_emb', 'item_emb'):
        assert not hasattr(m, name)
    old = TP.TokPolicy(**dict(CFG, frame=None))
    assert old.frame_v3 is None and 'frame' not in old.config and not any(k.startswith('frame_v3') for k in old.state_dict())
    assert old.pi.out_features == 20 and type(old.pi) is torch.nn.Linear
    for bad in (dict(temporal_feedback=1), dict(global_token=False), dict(item_tokens=True, item_shared=True),
                dict(temporal=False)):
        with pytest.raises(ValueError):
            TP.TokPolicy(**dict(CFG, **bad))
    n = sum(p.numel() for p in m.parameters())
    assert n > sum(p.numel() for p in m.frame_v3.parameters()) > 0
