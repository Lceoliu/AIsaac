"""seq-v1 step 2: TokPolicy(frame='v3') -- FrameEncoderV3 on the decoded v3 frames, T1's temporal side and heads.
CPU only, a small model (width 64) on the 19 golden frames (tests/fixtures/visible_v3) as actor steps.

- forward and backward: logits of the first arm's five heads (20), finite values and gradients;
- the drop head: the policy layer holds 22 logits (schema_v3.HEAD_SIZES), the forward gives 20, drop's two rows get no
  gradient, drop's "drop" bias is schema_v3.DROP_BIAS, the first 20 biases as T1's;
- no ROW observation field enters: batches of only v3, pending and the temporal metadata equal full batches with every
  other field randomised, over two steps; rooms 0 is required and the room fields change nothing; padding contents
  (junk, NaN, ids past the tables) never reach the real tokens; the gate is T1's;
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
           temporal_layers=2, temporal_heads=4, temporal_window=8, temporal_rooms=0, temporal_feedback=0, frame='v3',
           v3_inputs=VI.INPUT_REV)


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
    VI.put_batch(b, v3_tensors(VI.pad_batch(frames), 'cpu'))
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


TEMPORAL_FIELDS = ('first', 'map', 'events', 'stage')   # (the T1 metadata the temporal step reads from the records)


def _two_steps(m, batches):
    """Two actor steps on one fresh TemporalState: the outputs of each."""
    st = TP.TemporalState(m, len(batches[0]['pending']), 'cpu', torch.float32)
    return [step(m, b, st) for b in batches], st


def test_no_row_observation_field_enters():
    """(Codex, step 2's review: the randomisation missed int16 fields, kept map / events / stage, ran one step.) The
    model run on batches holding ONLY v3, pending and the temporal metadata -- any other field read would be a KeyError
    -- equals the run on the full batches with every other ROW field randomised (every dtype), over two steps."""
    m = build()
    TS.perturb_all(m, 7)
    full = [batch(NAMES[6:12], seed=s_) for s_ in (0, 1)]
    for b in full:
        b['first'][:] = 0
    full[0]['first'][:] = 1
    keep = lambda k: k.startswith(VI.V3_PREFIX) or k in ('pending',) + TEMPORAL_FIELDS   # noqa: E731
    minimal = [{k: v for k, v in b.items() if keep(k)} for b in full]
    g = torch.Generator().manual_seed(9)
    noisy = []
    for b in full:
        c = {}
        for k, v in b.items():
            if keep(k) or not torch.is_tensor(v):
                c[k] = v
            elif v.is_floating_point():
                c[k] = torch.randn(v.shape, generator=g).to(v.dtype)
            else:
                c[k] = torch.randint(0, 7, v.shape, generator=g).to(v.dtype)
        noisy.append(c)
    assert {str(v.dtype) for k, v in full[0].items() if torch.is_tensor(v) and not keep(k)} >= {'torch.int16'}
    ref, _ = _two_steps(m, minimal)
    got, _ = _two_steps(m, noisy)
    for r_, g_ in zip(ref, got):
        for a, b_ in zip(r_, g_):
            assert torch.equal(a, b_)


def test_rooms_0_makes_no_room_and_ignores_the_room_fields():
    """(Codex: rooms must be 0 -- the default 128 brought the room tokens back.) Two steps with a room change in ROW's
    map (current-room bit) and EV_NEW_ROOM / stage events: no room token is made and the outputs equal the run without
    the change."""
    m = build()
    TS.perturb_all(m, 11)
    full = [batch(NAMES[:4], seed=s_) for s_ in (0, 1)]
    full[0]['first'][:] = 1
    full[1]['first'][:] = 0
    base = [{k: v for k, v in b.items() if k.startswith(VI.V3_PREFIX) or k in ('pending',) + TEMPORAL_FIELDS}
            for b in full]
    moved = [dict(b) for b in base]
    mp = base[1]['map'].clone()
    mp[:, 3] = torch.roll(mp[:, 3], 1, dims=-1)   # (the current room moves)
    moved[1] = dict(base[1], map=mp, events=base[1]['events'] | 0xFF, stage=base[1]['stage'] + 1)
    ref, st = _two_steps(m, base)
    got, st2 = _two_steps(m, moved)
    assert int(st.nroom.max()) == 0 and int(st2.nroom.max()) == 0
    for r_, g_ in zip(ref, got):
        for a, b_ in zip(r_, g_):
            assert torch.equal(a, b_)


def test_padding_contents_never_reach_the_real_tokens():
    """(Codex: padding was masked only in attention.) Padded entity / inventory / slot / body rows holding finite junk,
    NaN, and ids past every table: the outputs are the clean ones bit for bit, and finite."""
    m = build()
    TS.perturb_all(m, 12)
    b = batch(NAMES[:3])
    ref = step(m, b)
    for junk in ('finite', 'nan', 'ids'):
        v = {k: t.clone() for k, t in VI.get_batch(b).items()}
        for g_ in ('P', 'E', 'N', 'S'):
            pad = ~v[g_ + '_mask']
            for k in [k for k in v if k.startswith(g_ + '_') and k != g_ + '_mask']:
                t = v[k]
                if t.dtype == torch.bool:
                    continue
                if junk == 'ids' and not t.is_floating_point():
                    t[pad] = 70000
                elif junk == 'nan' and t.is_floating_point():
                    t[pad] = float('nan')
                elif junk == 'finite' and t.is_floating_point():
                    t[pad] = 3.25
        pl = ~v['P_mask']
        v['local'][pl] = float('nan') if junk == 'nan' else 2.5
        got = step(m, VI.put_batch(dict(b), v))
        for a, b_ in zip(ref, got):
            assert torch.isfinite(b_).all() and torch.equal(a, b_), junk


def test_the_action_gate_is_t1s():
    """The gate is T1's, from the records' bombs / ready / pocket fields; a v3 batch carries no structured-input gate."""
    rows = _rows_with_frames(NAMES[:3])
    rows['bombs'] = [0, 2, 0]
    b = TP.to_batch(rows, np.zeros((3, 5), np.int64), 'cpu')
    assert 'v_gate' not in b and 'v_body' not in b
    g = TP.TokPolicy.gate(b)
    assert g[:, 0].tolist() == [False, True, False]


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
                dict(temporal=False), dict(temporal_rooms=1), dict(temporal_rooms=128)):
        with pytest.raises(ValueError):
            TP.TokPolicy(**dict(CFG, **bad))
    with pytest.raises(ValueError, match='temporal_rooms 0'):   # (left out: the default 128)
        TP.TokPolicy(**{k: v for k, v in CFG.items() if k != 'temporal_rooms'})
    n = sum(p.numel() for p in m.parameters())
    assert n > sum(p.numel() for p in m.frame_v3.parameters()) > 0


def _rows_with_frames(names, seed=0):
    from isaac_bridge import visible_producer as VP
    rng = np.random.default_rng(seed)
    rows = np.ascontiguousarray(TS.make_seq(rng, nw=len(names), T=1)[0])
    for i, n in enumerate(names):
        VP.write_frame_v3(rows[i:i + 1], (FIX / f'{n}.v3.bin').read_bytes())
    return rows


def test_records_carry_their_frames_to_the_model():
    """to_batch (host records) and decode_rows (records as bytes on the device: the learner) both give b['v3'], equal
    to pad_batch of the host decodes in the layout's caps; the model reads it."""
    names = NAMES[:5]
    rows = _rows_with_frames(names)
    pend = np.zeros((len(names), 5), np.int64)
    b = TP.to_batch(rows, pend, 'cpu')
    raw = torch.from_numpy(rows.view(np.uint8).reshape(len(rows), -1).copy())
    d = TP.decode_rows(raw, pending=torch.from_numpy(pend))
    frames = [VI.decode_frame((FIX / f'{n}.v3.bin').read_bytes(), V3.VERSION) for n in names]
    eb = VI.bucket(max(len(f['E_num']) for f in frames), VI.E_BUCKETS)
    ref = VI.pad_batch(frames, e_bucket=eb, n_max=VI.N_MAX_V3, s_max=VI.S_MAX_V3)
    assert set(VI.get_batch(d)) == {'raw', 'counts'}   # (decode_rows: lazy, decoded by the forward that reads it)
    got = VI.get_batch(b)
    for k in ref:
        assert torch.equal(got[k], torch.as_tensor(ref[k])), k
    lz = VI.materialize(VI.get_batch(d))   # (the learner's: N / S in their buckets -- the same rows, less padding)
    for k in ref:
        r_ = torch.as_tensor(ref[k])
        if k[:2] in ('N_', 'S_'):
            r_ = r_[:, :lz[k].shape[1]]
            assert not torch.as_tensor(ref[k])[:, lz[k].shape[1]:].any() or k.endswith('_pad'), k
        assert torch.equal(lz[k], r_), k
    m = build()
    with torch.no_grad():
        out = m(b, need_vg=True)
    assert out[0].shape == (len(names), 20) and torch.isfinite(out[0]).all()


def test_a_record_without_a_v3_frame_is_refused_in_a_v3_batch():
    from isaac_bridge.visible_producer import VisibleV3Fatal
    rows = _rows_with_frames(NAMES[:3])
    rows['visible_schema'][1] = 0
    rows['visible_bytes'][1] = 0
    with pytest.raises(VisibleV3Fatal):
        TP.to_batch(rows, np.zeros((3, 5), np.int64), 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA: GraphActor')
@pytest.mark.parametrize('fold', [False, True])
def test_graph_actor_v3_equals_the_eager_step(fold):
    """GraphActor of a v3 model: each call's records checked on the host, decoded on the device inside the graph of
    their entity bucket; the values, value_global and the temporal step's counters equal the eager step's (to_batch,
    the same rows) over a short sequence; v3_input.STATS counts the calls and buckets."""
    cuda = torch.device('cuda')
    m = build().to(cuda)
    TS.perturb_all(m, 31)
    st = TP.TemporalState(m, 6, cuda, torch.float32)
    actor = TP.GraphActor(m, 6, 32, stream=torch.cuda.Stream(), crn=False, temporal=st, fold=fold)
    st_e = TP.TemporalState(m, 6, cuda, torch.float32)
    before = dict(VI.STATS)
    widx = np.array([5, 0, 3, 1])
    rng = np.random.default_rng(4)
    for t in range(4):
        rows = _rows_with_frames(NAMES[4 * t:4 * t + 4], seed=t)
        if t == 0:
            rows['first'] = 1
        pend = np.stack([rng.integers(0, k, 4) for k in m.heads], 1).astype(np.int64)
        acts, lp, val = actor.act(rows, pend, widx, None, None)
        torch.cuda.synchronize()
        td = dict(mode='step', state=st_e, slot=torch.from_numpy(widx).to(cuda))
        b_ref = TP.to_batch(rows, pend, cuda)
        with torch.no_grad():
            lg_ref, v_ref, _, vg_ref = m(b_ref, need_vg=True, temporal=td)
            # (Codex round 10: the actor's log-probability of the actions it chose, against the eager logits)
            lp_ref, _ent = m.log_prob(lg_ref, m.gate(b_ref), torch.as_tensor(np.asarray(acts), device=cuda).long())
        assert np.allclose(lp, lp_ref.cpu().numpy(), atol=1e-4), (t, lp, lp_ref)
        assert np.allclose(val, v_ref.cpu().numpy(), atol=1e-4), (t, val, v_ref)
        assert np.allclose(actor.vg, vg_ref.cpu().numpy(), atol=1e-4)
        assert np.array_equal(actor.t_out['steps'], td['out']['steps'].cpu().numpy())
    assert VI.STATS['calls'] - before['calls'] >= 4
    assert sum(VI.STATS[f'bucket{b}'] - before[f'bucket{b}'] for b in VI.E_BUCKETS) >= 4


def test_take_selects_the_v3_rows_too():
    """(184, 2026-11-18: the learner's minibatch selection, tok_policy.take, met a nested b['v3'] -- KeyError at the
    first update.) The v3 tensors are top-level batch keys: take() picks their rows like any field's, and the model
    on the taken batch equals the model on a batch built from those records alone."""
    m = build()
    b = batch(NAMES[:6])
    idx = torch.tensor([4, 1, 3])
    t = TP.take(b, idx)
    v, vt = VI.get_batch(b), VI.get_batch(t)
    assert set(v) == set(vt) and v
    for k in v:
        assert torch.equal(vt[k], v[k][idx]), k
    with torch.no_grad():
        a = m(t, need_vg=True)
        c = m(TP.take(b, torch.arange(6)), need_vg=True)
    for x, y in zip(a, c):
        if torch.is_tensor(x):
            assert torch.allclose(x, y[idx], atol=1e-5), 'row-selected batch differs'
    # the learner's batches (decode_rows: the frames' bytes and counts, decoded per forward) select rows the same way
    rows = _rows_with_frames(NAMES[:6])
    pend = np.zeros((6, 5), np.int64)
    raw = torch.from_numpy(rows.view(np.uint8).reshape(6, -1).copy())
    d = TP.decode_rows(raw, pending=torch.from_numpy(pend))
    e = TP.to_batch(rows, pend, 'cpu')
    dt, et = VI.materialize(VI.get_batch(TP.take(d, idx))), VI.get_batch(TP.take(e, idx))
    c_ = VI.get_batch(d)['counts'][idx].max(0).values.tolist()   # (E, N, S each in its bucket: materialize)
    eb, nb, sb = (VI.bucket(c_[1], VI.E_BUCKETS), VI.bucket(min(c_[2], VI.N_MAX_V3), VI.N_BUCKETS),
                  VI.bucket(min(c_[3], VI.S_MAX_V3), VI.S_BUCKETS))
    assert (nb, sb) != (VI.N_MAX_V3, VI.S_MAX_V3)   # (the golden frames' few rows: the narrow buckets are exercised)
    ref = VI.pad_batch([VI.decode_frame((FIX / f'{NAMES[i]}.v3.bin').read_bytes(), V3.VERSION) for i in idx.tolist()],
                       e_bucket=eb, n_max=nb, s_max=sb)
    for k in ref:
        assert torch.equal(dt[k], torch.as_tensor(ref[k])), k
    with torch.no_grad():
        for x, y in zip(m(TP.take(d, idx), need_vg=True), m(TP.take(e, idx), need_vg=True)):
            if torch.is_tensor(x):
                assert torch.allclose(x, y, atol=1e-5)
    with pytest.raises(ValueError, match='without v3 tensors'):
        m({k: v_ for k, v_ in b.items() if not k.startswith(VI.V3_PREFIX)}, need_vg=True)


def test_cat_batch_joins_v3_parts_of_different_widths():
    """(184, 2026-11-18: the learner's part_prep joins two parts with cat_batch; their lazy frames had different byte
    widths -- RuntimeError.) A part widened by zero bytes past its frames joins another; the joined batch's outputs
    equal each part's."""
    m = build()
    rows = _rows_with_frames(NAMES[:6])
    raw = torch.from_numpy(rows.view(np.uint8).reshape(6, -1).copy())
    d = TP.decode_rows(raw, pending=torch.from_numpy(np.zeros((6, 5), np.int64)))
    a_, b_ = TP.take(d, torch.tensor([0, 1, 2])), TP.take(d, torch.tensor([3, 4, 5]))
    a_ = dict(a_, **{'v3.raw': torch.nn.functional.pad(a_['v3.raw'], (0, 40))})   # (wider by 40 zero bytes)
    w1, w2 = a_['v3.raw'].shape[1], b_['v3.raw'].shape[1]
    assert w1 != w2
    j = TP.cat_batch(a_, b_)
    assert j['v3.raw'].shape[1] == max(w1, w2)
    with torch.no_grad():
        oj, oa, ob = m(j, need_vg=True), m(a_, need_vg=True), m(b_, need_vg=True)
    for x, ya, yb in zip(oj, oa, ob):
        if torch.is_tensor(x):
            assert torch.allclose(x, torch.cat([ya, yb]), atol=1e-5)


@pytest.mark.parametrize('flag', ['body_role', 'position_known'])
def test_batched_entries_refuse_a_p0_without_the_contract(flag):
    """(Codex round 10: only the single-frame entry checked P row 0.) Every batched entry -- the actor's / eval's
    to_batch (rows_counts), the learner's decode_rows (lazy_from_raw) and batch_from_raw -- refuses a record whose P row
    0 is not the controlled body (body_role 1) with a position (position_known 1): VisibleV3Fatal, never a decode
    of relative quantities against another body."""
    from isaac_bridge.visible_producer import VisibleV3Fatal
    rows = _rows_with_frames(NAMES[:3])
    f = rows['visible_frame'].view('<f4')
    f[1, V3.FIXED + V3.col(V3.P_COLS, flag)] = 0.0
    pend = np.zeros((3, 5), np.int64)
    with pytest.raises(VisibleV3Fatal, match='P row 0'):
        TP.to_batch(rows, pend, 'cpu')
    raw = torch.from_numpy(rows.view(np.uint8).reshape(3, -1).copy())
    with pytest.raises(VisibleV3Fatal, match='P row 0'):
        TP.decode_rows(raw, pending=torch.from_numpy(pend))
    fa, sa, ba = (TP.ROW.fields[k][1] for k in ('visible_frame', 'visible_schema', 'visible_bytes'))
    with pytest.raises(VisibleV3Fatal, match='P row 0'):
        VI.batch_from_raw(raw, fa, sa, ba)
    f[1, V3.FIXED + V3.col(V3.P_COLS, flag)] = 1.0   # (restored: accepted)
    TP.to_batch(rows, pend, 'cpu')


def test_rows_to_device_copies_the_frame_prefix_only():
    """(seq-v1: v3 frames fill ~15 KB of the 128-KiB field.) rows_to_device sends each record up to the longest frame
    of the set -- the same batch as before (to_batch), fewer bytes -- and refuses a declared length past the field."""
    from isaac_bridge.visible_producer import VisibleV3Fatal
    rows = _rows_with_frames(NAMES[:5])
    pend = np.zeros((5, 5), np.int64)
    idx = np.array([4, 0, 2], np.int64)
    out, raw, _p = TP.rows_to_device(rows, idx, pend[idx], 'cpu', with_raw=True)
    assert raw.shape[1] == TP.ROW.fields['visible_frame'][1] + int(rows['visible_bytes'][idx].max())
    ref = TP.to_batch(rows[idx], pend[idx], 'cpu')
    got, want = VI.materialize(VI.get_batch(out)), VI.get_batch(ref)

    def same(g_, w_):   # (the learner's decode: N / S in their buckets -- the eager one's first columns)
        for k in w_:
            assert torch.equal(g_[k], w_[k][:, :g_[k].shape[1]] if k[:2] in ('N_', 'S_') else w_[k]), k
    same(got, want)
    stage = torch.empty((8, TP.ROW_BYTES), dtype=torch.uint8)   # (the learner's pinned stage, reinterpreted)
    out2 = TP.rows_to_device(rows, idx, pend[idx], 'cpu', stage=stage)
    same(VI.materialize(VI.get_batch(out2)), want)
    bad = rows.copy()
    bad['visible_bytes'][0] = 1 << 30
    with pytest.raises(VisibleV3Fatal, match='visible_bytes'):
        TP.rows_to_device(bad, idx, pend[idx], 'cpu')


def test_input_revision_2():
    """(2026-11-19) The model maps v3_input.WIDEN's columns back -- E age (minutes) to log2(1 + seconds) / 2, S charges
    (120ths) linear in 12ths up to 1 and 1 + log2 past it -- and a configuration without the revision is refused."""
    import math
    from isaac_bridge.frame_v3 import FrameEncoderV3
    enc = FrameEncoderV3(64, 1, 4)
    ja = VI.PLAN['E']['num'].index(V3.col(V3.E_COLS, 'age'))
    x = torch.zeros(3, len(VI.PLAN['E']['num']))
    x[:, ja] = torch.tensor([0.0, 8 / 60, 8.0])   # (0 s, 8 game seconds, the clamp: 8 game minutes)
    y = enc._log_inputs(x, 'E')
    assert torch.allclose(y[:, ja], torch.tensor([0.0, math.log2(9) / 2, math.log2(481) / 2]))
    assert torch.equal(torch.cat([y[:, :ja], y[:, ja + 1:]], 1), torch.cat([x[:, :ja], x[:, ja + 1:]], 1))
    for name in ('charge', 'max_charge'):
        jc = VI.PLAN['S']['num'].index(V3.col(V3.S_COLS, name))
        s = torch.zeros(3, len(VI.PLAN['S']['num']))
        s[:, jc] = torch.tensor([0.05, 0.1, 0.8])   # (6 / 120, 12 / 120, 96 / 120)
        assert torch.allclose(enc._log_inputs(s, 'S')[:, jc], torch.tensor([0.5, 1.0, 1 + math.log2(8)])), name
    # the L token's last Linear x5 and the G token's parameter std 0.15 at init (independent expectations)
    torch.manual_seed(0)
    ref = torch.nn.Linear(enc.l_cnn[-1].in_features, enc.l_cnn[-1].out_features)
    assert enc.l_cnn[-1].weight.abs().max() > 4 * ref.weight.abs().max()
    m = build()
    assert 0.1 < float(m.gtok.tok.std()) < 0.2
    assert 'trigger_once' not in m.config and TP.TokPolicy(**CFG, trigger_once=0).config['trigger_once'] == 0
    with pytest.raises(ValueError, match='revision'):
        TP.TokPolicy(**{k: v for k, v in CFG.items() if k != 'v3_inputs'})
    assert build().config['v3_inputs'] == VI.INPUT_REV
