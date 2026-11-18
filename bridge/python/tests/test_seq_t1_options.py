"""seq-v1 step 0 (2026-11-18): C114 T1's temporal side as committed options -- train_tok --temporal-rooms 0 (no room
token is ever made), --temporal-feedback 0 (no feedback / history token), --obs-visible 0 (the workers write no
visible frame; C110 Q@maa's ISAAC_NO_VISIBLE). CPU only, small synthetic models (test_temporal_smoke's sequences).

- rooms 0: TemporalAttention keeps one (masked) room slot and its room modules, so a rooms-0 model has the parameters
  of a rooms-1 model (the same random numbers for one seed); over a sequence with room changes and stage exits no room
  token is made (step's create always False, nroom 0, the cache's room slot never written), and the outputs equal a
  rooms-1 model whose room creation is switched off (the C114 nvt tree's ISAAC_T_NOROOMS) bit for bit.
- feedback 0: _temporal_in gives has False for every record; the outputs do not depend on the previous step's o (the
  state's prev_o randomised: same outputs; with feedback 1 they change); packed_index / learner_plan with every flag
  False have no history token.
- configs: temporal_feedback is a config key only when 0; rooms 0 is kept; TokPolicy(**config) rebuilds the model.
- obs-visible 0: tok_obs.encode_row writes visible_schema 0 even when the observation carries a visible frame."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402  (its path set-up, synthetic sequences, perturb_all)

from isaac_bridge import tok_policy as TP   # noqa: E402

CFG = dict(TS.BASE, **TS.SMALL, **TS.FUSED, temporal=True, temporal_width=32, temporal_layers=2, temporal_heads=4,
           temporal_window=8)


def build(seed=5, **kw):
    torch.manual_seed(seed)
    m = TP.TokPolicy(**dict(CFG, **kw)).eval()
    return m


def steps(m, seq, pend, st, keep=None):
    """The actor's eager steps over seq (every worker each step): logits, value per step; the step infos."""
    nw = len(seq[0])
    out, infos = [], []
    with torch.no_grad():
        for t, rows in enumerate(seq):
            b = TP.to_batch(rows, pend[t], 'cpu')
            td = dict(mode='step', state=st, slot=torch.arange(nw))
            if keep is not None:
                keep(t, st)
            o = m(b, need_vg=True, temporal=td)
            out.append(torch.cat([o[0].reshape(nw, -1), o[1].reshape(nw, -1)], 1))
            infos.append(td.get('out'))
    return torch.stack(out), infos


def pend_of(seq, m, seed=3):
    g = np.random.default_rng(seed)
    return [np.stack([g.integers(0, k, len(r)) for k in m.heads], 1).astype(np.int64) for r in seq]


def test_rooms0_parameters_equal_rooms1():
    a, b = build(temporal_rooms=0), build(temporal_rooms=1)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]), k
    assert a.tatt.R == 1 and not a.tatt.rooms_on and b.tatt.rooms_on
    assert a.config['temporal_rooms'] == 0 and b.config['temporal_rooms'] == 1


def test_rooms0_makes_no_room_and_equals_rooms1_with_creation_off(monkeypatch):
    rng = np.random.default_rng(11)
    seq = TS.make_seq(rng)
    a = build(temporal_rooms=0)
    TS.perturb_all(a, 21)
    b = build(temporal_rooms=1)
    b.load_state_dict(a.state_dict(), strict=True)
    pend = pend_of(seq, a)
    st_a = TP.TemporalState(a, len(seq[0]), 'cpu', torch.float32)
    log_a, _ = steps(a, seq, pend, st_a)
    assert int(st_a.nroom.max()) == 0
    assert float(st_a.kv[:, :, :, :, :1].abs().max()) == 0.0   # (the one room slot never written)
    # the reference: rooms 1 with creation switched off (what the C114 nvt tree's ISAAC_T_NOROOMS did)
    b.tatt.rooms_on = False
    st_b = TP.TemporalState(b, len(seq[0]), 'cpu', torch.float32)
    log_b, _ = steps(b, seq, pend, st_b)
    assert torch.equal(log_a, log_b)
    # and with creation on the sequence's room changes do make rooms and change the outputs
    b.tatt.rooms_on = True
    st_c = TP.TemporalState(b, len(seq[0]), 'cpu', torch.float32)
    log_c, _ = steps(b, seq, pend, st_c)
    assert int(st_c.nroom.max()) >= 1 and not torch.equal(log_a, log_c)


def test_feedback0_ignores_the_previous_output():
    rng = np.random.default_rng(12)
    seq = TS.make_seq(rng)
    for fb in (0, 1):
        m = build(temporal_rooms=0, temporal_feedback=fb)
        TS.perturb_all(m, 22)
        pend = pend_of(seq, m)
        st = TP.TemporalState(m, len(seq[0]), 'cpu', torch.float32)
        ref, _ = steps(m, seq, pend, st)
        g = torch.Generator().manual_seed(7)

        def scramble(t, s):
            if t > 0:
                s.prev_o.copy_(torch.randn(s.prev_o.shape, generator=g))
        st2 = TP.TemporalState(m, len(seq[0]), 'cpu', torch.float32)
        got, _ = steps(m, seq, pend, st2, keep=scramble)
        if fb == 0:
            assert torch.equal(ref, got)
        else:
            assert not torch.equal(ref, got)
    m0 = build(temporal_rooms=0, temporal_feedback=0)
    b = TP.to_batch(seq[1], pend_of(seq, m0)[1], 'cpu')
    st = TP.TemporalState(m0, len(seq[0]), 'cpu', torch.float32)
    st.steps.fill_(3)
    _, hist = m0._temporal_in(b, dict(mode='step', state=st, slot=torch.arange(len(seq[0]))))
    assert not bool(hist['has'].any())


def test_plans_without_history_tokens():
    ne, nd = np.array([3, 0, 5]), np.array([1, 2, 0])
    off = np.zeros(3, bool)
    keep0, starts0 = TP.packed_index(ne, nd, 8, n_item=np.array([1, 0, 2]), item_tok=4, glob=1, hist=off)
    keep1, starts1 = TP.packed_index(ne, nd, 8, n_item=np.array([1, 0, 2]), item_tok=4, glob=1, hist=~off)
    assert len(keep1) - len(keep0) == 3   # (one history token per record where the flag is set, none when off)
    p0 = TP.learner_plan(ne, nd, 8, n_item=np.array([1, 0, 2]), item_tok=4, glob=1, hist=off)
    assert int(p0['rec'][:, 8].sum()) == 0


def test_config_keys():
    m0 = build(temporal_rooms=0, temporal_feedback=0)
    m1 = build(temporal_rooms=128, temporal_feedback=1)
    assert m0.config['temporal_feedback'] == 0 and 'temporal_feedback' not in m1.config
    r = TP.TokPolicy(**m0.config)
    assert not r.tatt.feedback and not r.tatt.rooms_on and r.tatt.R == 1
    assert {k: v.shape for k, v in r.state_dict().items()} == {k: v.shape for k, v in m0.state_dict().items()}
    with pytest.raises(ValueError):
        TP.TemporalAttention(32, True, D=32, layers=1, heads=4, window=8, rooms=-1)


def test_obs_visible_0_writes_no_frame(monkeypatch):
    from isaac_bridge import tok_obs
    from isaac_bridge.tok_obs import ROW

    class Obs:   # (a visible frame present; encode_row stops at players, after the visible branch)
        visible = object()
    row = np.zeros(1, ROW)
    row['visible_schema'] = 7
    monkeypatch.setenv('ISAAC_NO_VISIBLE', '1')
    with pytest.raises(AttributeError):
        tok_obs.encode_row(Obs(), None, row, 0)
    assert int(row['visible_schema'][0]) == 0
