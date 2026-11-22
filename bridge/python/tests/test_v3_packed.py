"""CPU checks for packing only the real v3 tokens through the spatial blocks."""
import os
import pickle
import statistics
import time

import numpy as np
import pytest
import torch

import test_v3_policy as VP
from isaac_bridge import schema_v3 as V3
from isaac_bridge import tok_policy as TP
from isaac_bridge import v3_input as VI
from isaac_bridge.frame_v3 import FrameEncoderV3
from isaac_bridge.visible_producer import write_frame_v3


def _row_batch(counts):
    """Resize the tables of a golden frame, with an unlocated secondary body when p > 1."""
    f = np.frombuffer((VP.FIX / 'devilprice.v3.bin').read_bytes(), '<f4')
    wire = VI.split_frame(f, V3.VERSION)
    rows = VP.TS.make_seq(np.random.default_rng(9), nw=len(counts), T=1)[0]
    for i, c in enumerate(counts):
        p = c[0]
        parts = [np.resize(wire[t], (n, wire[t].shape[1])) for t, n in zip('PENS', c)]
        parts[0][1:, V3.col(V3.P_COLS, 'body_role')] = V3.BODY_ROLE['twin']
        parts[0][1:, V3.col(V3.P_COLS, 'position_known')] = 1
        if p > 1:
            parts[0][1, V3.col(V3.P_COLS, 'position_known')] = 0
        parts.insert(1, np.resize(wire['local'], (p, *wire['local'].shape[1:])))
        f_new = np.concatenate([f[:V3.FIXED]] + [part.ravel() for part in parts])
        f_new[V3.H_P:V3.H_S + 1] = c
        f_new[V3.H_SIZE] = len(f_new)
        write_frame_v3(rows[i:i + 1], f_new.tobytes())
    return rows


def _batch(rows, lazy):
    rng = np.random.default_rng(178)
    pending = np.stack([rng.integers(h, size=len(rows)) for h in V3.HEAD_SIZES[:5]], 1).astype(np.int64)
    if not lazy:
        return TP.to_batch(rows, pending, 'cpu')
    raw = torch.from_numpy(rows.view(np.uint8).reshape(len(rows), -1))
    batch = TP.decode_rows(raw, pending=torch.from_numpy(pending))
    assert set(VI.get_batch(batch)) == {'raw', 'counts'}
    return batch


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    return float((actual - expected).detach().abs().max())


def _outputs(model, v, pending, buckets):
    enc, gtok = model.frame_v3, model.gtok.tok
    ref, pad, g = enc(v, pending, gtok)
    got, got_pad, got_g = enc.forward_packed(v, pending, gtok, buckets=buckets)
    assert got_g == g and torch.equal(got_pad, pad)
    assert not got[pad].any()
    err = max(_close(got[~pad], ref[~pad]), _close(got[:, 0], ref[:, 0]), _close(got[:, g], ref[:, g]))
    return got, ref, g, err


@pytest.mark.parametrize('lazy', (False, True))
def test_goldens_and_temporal_policy(lazy):
    b = _batch(VP._rows_with_frames(VP.NAMES), lazy)
    m = VP.build()
    with torch.no_grad():
        _, _, _, enc_err = _outputs(m, VI.materialize(VI.get_batch(b)), b['pending'], 3)
        results = []
        for packed in (False, True):
            st = TP.TemporalState(m, len(VP.NAMES), 'cpu', torch.float32)
            ctx = dict(mode='step', state=st, slot=torch.arange(len(VP.NAMES)))
            results.append(m(b, packed=packed, temporal=ctx, need_vg=True))
        err = max(_close(a, b_) for a, b_ in zip(*results) if torch.is_tensor(a))
    print(f'goldens {len(VP.NAMES)} lazy={lazy}: encoder max abs {enc_err:.8g}, policy max abs {err:.8g}')


@pytest.mark.parametrize('limits,widths', [
    ((16, 8, 8), (16, 8, 8)), ((17, 9, 9), (32, 16, 16)), ((32, 16, 16), (32, 16, 16)),
    ((33, 17, 17), (64, 32, 32)), ((64, 32, 32), (64, 32, 32)),
    ((65, 33, 33), (128, 64, 32)), ((128, 65, 39), (128, 64, 32)),
])
def test_wire_bucket_boundaries(limits, widths):
    e, n, s = limits
    counts = [(1, 0, 0, 0), (4, e, n, s), (1, 0, n, 0), (1, e, 0, 0),
              (4, 0, 0, s), (1, e - 1, n - 1, s - 1), (4, e, n, s)]
    b = _batch(_row_batch(counts), True)
    v = VI.materialize(VI.get_batch(b))
    assert tuple(v[t + '_mask'].shape[1] for t in 'ENS') == widths
    assert [int(v[t + '_mask'][1].sum()) for t in 'ENS'] == [e, min(n, 64), min(s, 32)]
    assert v['P_mask'][1].sum().item() == 4 and v['local_pad'][1].tolist() == [False, True, False, False]
    m = VP.build()
    with torch.no_grad():
        _, pad, _ = m.frame_v3.tokens(v, b['pending'], m.gtok.tok)
        assert (~pad).sum(1).tolist() == [p + (p if p == 1 else p - 1) + en + min(in_, 64) + min(sl, 32) + 30
                                         for p, en, in_, sl in counts]
        _, _, _, err = _outputs(m, v, b['pending'], 3)
    print(f'boundary {limits}: max abs {err:.8g}')


def test_single_record_empty_tables():
    b = _batch(_row_batch([(1, 0, 0, 0)]), True)
    with torch.no_grad():
        _outputs(VP.build(), VI.materialize(VI.get_batch(b)), b['pending'], 1)


def test_lazy_reordered_micros():
    b = _batch(_row_batch([(1, e, e // 2, e // 4) for e in (17, 0, 1, 5, 120, 9, 2)]), True)
    m = VP.build()
    with torch.no_grad():
        ref = m(b, packed=False, need_vg=True)
        first = TP.take(b, torch.tensor([3, 0, 6, 4]))
        second = TP.take(b, torch.tensor([5, 1, 2]))
        assert VI.materialize(VI.get_batch(first))['E_mask'].shape[1] == 128
        assert VI.materialize(VI.get_batch(second))['E_mask'].shape[1] == 16
        out = [m(part, packed=True, need_vg=True) for part in (first, second)]
        idx = torch.tensor([3, 0, 6, 4, 5, 1, 2])
        err = max(_close(torch.cat((a, b_)), r[idx]) for a, b_, r in zip(*out, ref) if torch.is_tensor(r))
    print(f'lazy reordered micros: max abs {err:.8g}')


@pytest.mark.parametrize('buckets', (1, 2, 3))
def test_blocks_receive_only_real_tokens_and_use_the_buckets(buckets, monkeypatch):
    # Deliberately interleave small / medium / large frames (rather than grouping by count in input order).
    b = _batch(_row_batch(([(1, 120, 0, 0), (1, 0, 0, 0), (1, 60, 0, 0)] * 22)[:64]), True)
    enc_calls = []
    original = TP.Block.forward_packed

    def track(self, x, bias, keep, shape):
        enc_calls.append((x.shape[0], len(shape['buckets'])))
        return original(self, x, bias, keep, shape)

    monkeypatch.setattr(TP.Block, 'forward_packed', track)
    m = VP.build()
    with torch.no_grad():
        _, _, _, err = _outputs(m, VI.materialize(VI.get_batch(b)), b['pending'], buckets)
    assert enc_calls == [(sum(([152, 32, 92] * 22)[:64]), buckets)] * len(m.frame_v3.blocks)
    print(f'64 records with {buckets} buckets: max abs {err:.8g}')


@pytest.mark.parametrize('math', (False, True))
def test_all_encoder_parameter_gradients(math):
    counts = [(1, 120, 3, 3), (4, 0, 0, 0), (1, 16, 1, 5), (4, 24, 17, 0), (1, 0, 0, 12),
              (4, 2, 78, 33), (1, 64, 0, 0)]
    b = _batch(_row_batch(counts), True)
    # Four blocks here, like seq1; exercise both SDPA and the optional B18 math/LayerNorm path.
    torch.manual_seed(97)
    m = TP.TokPolicy(**dict(VP.CFG, layers=4))
    m.set_math_attention(math)
    got, ref, g, err = _outputs(m, VI.materialize(VI.get_batch(b)), b['pending'], 3)
    proj = torch.randn((len(counts), 2, got.shape[-1]), generator=torch.Generator().manual_seed(507))
    params = dict(m.frame_v3.named_parameters())
    params['gtok'] = m.gtok.tok
    grads = [torch.autograd.grad((x[:, (0, g)] * proj).sum(), tuple(params.values())) for x in (got, ref)]
    worst = (0., '')
    for name, actual, expected in zip(params, *grads):
        actual, expected = actual.double(), expected.double()
        norm = expected.norm().item()
        diff = (actual - expected).norm().item()
        assert np.isfinite(norm) and np.isfinite(diff), name
        if norm:
            worst = max(worst, (diff / norm, name))
        else:
            assert diff == 0., name
    print(f'gradient math={math}: max abs output {err:.8g}, max relative norm {worst[0]:.8g} ({worst[1]})')
    assert worst[0] <= 1e-5, worst


def test_padded_and_default_are_bitwise_the_original_policy(monkeypatch):
    rows = VP._rows_with_frames(VP.NAMES[:7])
    b = _batch(rows, False)
    m = VP.build()

    def unexpected(*args, **kwargs):
        pytest.fail('default/packed=False ran the packed encoder')

    monkeypatch.setattr(FrameEncoderV3, 'forward_packed', unexpected)
    with torch.no_grad():
        st = TP.TemporalState(m, len(rows), 'cpu', torch.float32)
        ctx = dict(mode='step', state=st, slot=torch.arange(len(rows)))
        # The 4229d8f v3 forward, before dispatching on packed.
        tctx, _ = m._temporal_in(b, ctx)
        v = VI.materialize(VI.get_batch(b))
        x, pad, g = m.frame_v3.tokens(v, b['pending'], m.gtok.tok)
        bias = torch.zeros((x.shape[0], 1, 1, x.shape[1]), device=x.device, dtype=x.dtype).masked_fill(
            pad[:, None, None], -1e9)
        for block in m.frame_v3.blocks:
            x = block(x, bias)
        expected = m._heads(x[:, 0], False, None, x[:, g], True, tctx)
        expected_state = st
        for kwargs in ({}, {'packed': False}):
            st = TP.TemporalState(m, len(rows), 'cpu', torch.float32)
            ctx = dict(mode='step', state=st, slot=torch.arange(len(rows)))
            actual = m(b, need_vg=True, temporal=ctx, **kwargs)
            assert len(actual) == len(expected)
            for a, e in zip(actual, expected):
                if torch.is_tensor(e):
                    assert a.shape == e.shape and a.dtype == e.dtype
                    assert a.numpy().tobytes() == e.numpy().tobytes()
                else:
                    assert a == e
            for field in ('kv', 'steps', 'nroom', 'prev_o', 'prev_stage', 'prev_map'):
                assert getattr(st, field).numpy().tobytes() == getattr(expected_state, field).numpy().tobytes(), field


def _original_learner_plan(n_ent, n_doors, entities, buckets=3, n_item=None, item_tok=0, glob=0, hist=None):
    """Frozen 4229d8f plan (including its tie-breaks and exact numpy arithmetic)."""
    ne = np.minimum(np.asarray(n_ent, np.int64), entities)
    nd = np.minimum(np.asarray(n_doors, np.int64), TP.DOOR_CAP)
    cnt = 3 + nd + ne
    n = len(cnt)
    hs = int(hist is not None)
    fused = bool(item_tok or glob or hs)
    if fused:
        ni = np.minimum(np.asarray(n_item, np.int64), item_tok) if item_tok else np.zeros(n, np.int64)
        nh = np.asarray(hist, bool).astype(np.int64) if hs else None
        cnt = cnt + ni + glob + (nh if hs else 0)
    rec = np.empty((n, (9 if hs else 8) if fused else 6), np.int64)
    rec[:, 0], rec[:, 2], rec[:, 3] = cnt, nd, ne
    rec[:, 1] = np.cumsum(cnt) - cnt
    rec[:, 4] = np.cumsum(nd) - nd
    rec[:, 5] = np.cumsum(ne) - ne
    if fused:
        rec[:, 6], rec[:, 7] = ni, np.cumsum(ni) - ni
    if hs:
        rec[:, 8] = nh
    top = int(cnt.max())
    upto = np.cumsum(np.bincount(cnt, minlength=top + 1))
    w = np.arange(top + 1, dtype=np.float64)
    a, b = np.meshgrid(w, w, indexing='ij')
    na, nb = upto[a.astype(np.int64)], upto[b.astype(np.int64)]
    n1, n2, n3 = na, nb - na, n - nb
    cost = n1 * a ** 2 + n2 * b ** 2 + n3 * float(top) ** 2 + 2.0e5 * ((n1 > 0) + (n2 > 0) + (n3 > 0))
    cost[~((a <= b) & (b <= top))] = np.inf
    if buckets < 3:
        cost[(n1 > 0) & (n2 > 0) & (n3 > 0)] = np.inf
    if buckets < 2:
        cost[(n1 > 0) | (n2 > 0)] = np.inf
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    group = np.where(cnt <= i, 0, np.where(cnt <= j, 1, 2))
    base = np.zeros(n, np.int64)
    out = dict(rec=rec, base=base, m=int(cnt.sum()), m_doors=int(nd.sum()), m_ent=int(ne.sum()),
               t=3 + TP.DOOR_CAP + entities + item_tok + hs + glob, entities=entities, buckets=[])
    if fused:
        out.update(m_item=int(ni.sum()), item_tok=int(item_tok), glob=int(glob), hist=hs)
    at = 0
    for k in range(3):
        r = np.flatnonzero(group == k)
        if not len(r):
            continue
        width = int(cnt[r].max())
        base[r] = at + np.arange(len(r)) * width
        out['buckets'].append((cnt[r], width))
        at += len(r) * width
    return out


def test_learner_plan_byte_for_byte():
    rng = np.random.default_rng(4229)
    for case in range(120):
        n = (1, 7, 64, 512)[case % 4]
        ne, nd, ni = (rng.integers(0, cap, size=n) for cap in (130, 11, 24))
        args = (ne, nd, (16, 32, 64, 128)[(case // 4) % 4], 1 + case % 3)
        kw = dict(n_item=ni, item_tok=16 if case & 1 else 0, glob=int(bool(case & 2)),
                  hist=rng.random(n) > .5 if case & 4 else None)
        assert pickle.dumps(TP.learner_plan(*args, **kw)) == pickle.dumps(_original_learner_plan(*args, **kw)), case


@pytest.mark.skipif(os.getenv('ISAAC_TEST_V3_PACKED_BENCH') != '1', reason='opt-in CPU benchmark')
def test_cpu_benchmark():
    rng = np.random.default_rng(619)
    n = 512
    counts = np.column_stack((np.ones(n, int), rng.poisson(7, n), rng.integers(0, 5, n), np.full(n, 3)))
    rows = _row_batch(counts)
    b = _batch(rows, True)
    raw = VI.get_batch(b)
    v = VI.decode_frames_device(raw['raw'].contiguous().view(torch.float32), raw['counts'], 32, 8, 8, check=False)
    torch.manual_seed(96)
    enc, gtok = FrameEncoderV3(), torch.nn.Parameter(torch.randn(192))
    pending = b['pending']
    with torch.no_grad():
        _, pad, _ = enc.tokens(v, pending, gtok)
    count = 32 + counts[:, 1:].sum(1)
    np.testing.assert_array_equal((~pad).sum(1).numpy(), count)
    tokens = int(count.sum())
    _, layout = TP._attention_buckets(count, 3)
    dense_attn = n * pad.shape[1] ** 2
    packed_attn = sum(len(c) * width ** 2 for c, width in layout)
    print(f'tokens / QKV+FF work: {tokens}/{pad.numel()} ({tokens / pad.numel():.2%}); '
          f'attention: {packed_attn}/{dense_attn} ({packed_attn / dense_attn:.2%}); '
          f'layout: {[(len(c), w) for c, w in layout]}', flush=True)
    with torch.no_grad():
        ref, _, _ = enc(v, pending, gtok)
        got, _, _ = enc.forward_packed(v, pending, gtok)
        print(f'benchmark batch: max abs {_close(got[~pad], ref[~pad]):.8g}', flush=True)

    def run(fn):
        enc.zero_grad(set_to_none=True)
        gtok.grad = None
        start = time.perf_counter()
        x, _, g = fn(v, pending, gtok)
        x[:, (0, g)].square().sum().backward()
        return time.perf_counter() - start

    fns = (enc.forward, enc.forward_packed)
    for fn in fns:
        run(fn)
    times = [[], []]
    for i in range(5):
        for j in (i % 2, 1 - i % 2):
            times[j].append(run(fns[j]))
    padded, packed = map(statistics.median, times)
    print(f'B={n} fp32 CPU threads={torch.get_num_threads()}: padded {padded * 1e3:.2f} ms, '
          f'packed {packed * 1e3:.2f} ms, saved {1 - packed / padded:.2%}')
    assert tokens < pad.numel() and packed_attn < dense_attn
