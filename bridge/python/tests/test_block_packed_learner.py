"""B44 packed learner gates: forward_packed (one execution per part) against the per-length path
(forward_chunks_paged per exact chunk length), document isolation inside a padded bucket, no read
of memory written after a decision, and the B43 history modes."""
import numpy as np
import pytest
import torch

from test_block_slot_integration import pair, frame, _collect_rollout, _losses, _loss, DEV
from test_visible_input import synth_pending
from isaac_bridge import block_learner
from isaac_bridge.block_training import Chunk, forward_bucketed, source_rows
from isaac_bridge.structured_input import from_rows, FIXED
from train_tok import gae

TOL = dict(rtol=1e-5, atol=2e-6)


def _run(model, chunks, packed, monkeypatch, seed=5, **kw):
    monkeypatch.setattr(block_learner, 'PACKED', packed)
    model.zero_grad(set_to_none=True)
    return forward_bucketed(model, chunks, generator=torch.Generator().manual_seed(seed), **kw)


def _grads(model):
    return {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


def _rollout_and_teacher(device, seed=21, block_config=None):
    """A rollout's chunks (assorted lengths, an episode reset) plus one-row teacher records
    (snapshots at assorted phases, some completing a block) from the same page store."""
    if block_config is None:
        m, ref, new, old = pair(device, slots=3, seed=seed)
    else:   # (a Reader head width FlexAttention's kernels accept)
        from dataclasses import asdict
        from isaac_bridge.tok_policy import TokPolicy
        from isaac_bridge.block_paged import SlotBlockActor
        torch.manual_seed(seed)
        m = TokPolicy(width=32, layers=1, heads=4, items=True, item_tokens=True, global_token=True,
                      block_memory=True, block_config=asdict(block_config)).to(device).eval()
        new = SlotBlockActor(m, 3)
    rng = np.random.default_rng(seed)
    from train_tok import Rollout, store
    roll = Rollout(3, 48, 5, value_g=True, block_steps=7)   # (chunks of up to 7 decisions: odd starts)
    episodes = np.array([1, 1, 1])
    teach = []
    for t in range(37):
        first = np.array([int(t == 0)]*3)
        if t in (11, 23):
            first[t % 3] = 1; episodes[t % 3] += 1
        rows = frame(rng, 3, t, episodes, first)
        pending = synth_pending(rng, 3, m.heads)
        if t >= 6 and t % 3 == 0:
            for i in range(3):
                teach.append(Chunk(rows[i:i+1].copy(), pending[i:i+1].copy(), new.snapshot(i)))
        result = new.step(rows, pending, [0, 1, 2], ukeys=(np.full(3, t), np.arange(3)))
        action, lp, v, vg = [x.float().cpu().numpy() for x in result]
        store(roll, np.arange(3), rows, pending, action.astype(np.int64), lp, v, 0, vg=vg,
              block_context=new.last_contexts)
        roll.reward[:, t] = .1
    _, _, mask = gae(roll, .99, .95)
    plan = roll.block.plan(roll, mask)
    return m, plan['chunks'], teach


def test_chunk_buckets():
    assert [block_learner.chunk_bucket(n) for n in (1, 2, 31, 32, 33, 64, 65, 128, 129, 300)] == \
        [1, 32, 32, 32, 64, 64, 128, 128, 256, 512]


@pytest.mark.parametrize('device', DEV)
def test_packed_matches_per_length_path_outputs_loss_and_every_gradient(device, monkeypatch):
    """Rollout chunks of assorted lengths + one-row teacher records in one part, random history
    (the same generator draws): outputs, the PPO loss and every parameter gradient equal the
    per-length path's; the part runs as one Spatial pass per row bucket and <= 4 Reader calls."""
    m, chunks, teach = _rollout_and_teacher(device)
    batch_chunks = chunks+teach
    assert len({len(c.rows) for c in batch_chunks}) >= 4
    m.train()
    from isaac_bridge import block_paged as bp
    res = []
    for packed in (False, True):
        bp.UPD.clear()
        on = bp.TIMING_ON
        bp.TIMING_ON = True
        try:
            out, ctx = _run(m, batch_chunks, packed, monkeypatch)
        finally:
            bp.TIMING_ON = on
        calls = dict(bp.UPD)
        rows = source_rows([c.framed() for c in batch_chunks]); pend = np.concatenate([c.pending for c in batch_chunks])
        loss = _losses(m, out, from_rows(rows, device, pend), 3)
        loss.backward()
        res.append((loss.detach(), [x.detach() for x in out], ctx['recompute_stats'], _grads(m), calls))
    (la, oa, sa, ga, ca), (lb, ob, sb, gb, cb) = res
    for key in ('selected_blocks', 'tail_rows'):
        assert sa[key] == sb[key] > 0, key
    # a block selected by chunks of two length groups is re-encoded once (the old path: once per call)
    assert sb['encoded_blocks'] <= sa['encoded_blocks']
    torch.testing.assert_close(lb, la, **TOL)
    for x, y in zip(ob, oa):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=5e-6)
    assert ga.keys() == gb.keys()
    for k in ga:
        torch.testing.assert_close(gb[k], ga[k], rtol=1e-4, atol=5e-6, msg=k)
    assert gb['block_memory.writer.queries'].abs().sum() > 0
    # one execution: the per-length path made one call per length group
    assert cb['learn_packed_calls'] == 1 and cb.get('learn_paged_calls', 0) == 0
    assert ca['learn_paged_calls'] >= 4
    assert cb['learn_reader_calls'] <= len(block_learner.T_LEARN) < ca['learn_reader_calls']
    assert cb['learn_writer_calls'] == 1


def test_bucket_rows_are_isolated_segments(monkeypatch):
    """Inside one padded bucket every chunk is its own segment: a chunk's outputs and its
    gradient contribution do not depend on which other chunks share the bucket."""
    m, chunks, teach = _rollout_and_teacher('cpu', seed=33)
    m.train()
    mono = [c for c in chunks if len(c.rows) > 1]
    a = mono[:2]
    out_a, _ = _run(m, a, True, monkeypatch, train_history=False)
    out_ab, _ = _run(m, a+mono[2:]+teach, True, monkeypatch, train_history=False)
    n = sum(len(c.rows) for c in a)
    assert len({block_learner.chunk_bucket(len(c.rows)) for c in mono}) == 1   # one shared bucket
    for x, y in zip(out_a, out_ab):
        torch.testing.assert_close(y[:n].detach(), x.detach(), rtol=1e-5, atol=2e-6)
    # a bucket of one chunk runs with a fully padded second row (no 1-row kernels): same values
    out_1, _ = _run(m, a[:1], True, monkeypatch, train_history=False)
    n1 = len(a[0].rows)
    for x, y in zip(out_1, out_ab):
        torch.testing.assert_close(y[:n1].detach(), x.detach(), rtol=1e-5, atol=2e-6)


def test_no_decision_reads_later_inputs_or_later_memory(monkeypatch):
    """Causality in the padded layout: changing the observations of a chunk's decisions >= k
    (which also changes the blocks they complete inside the chunk) leaves every output of its
    decisions < k unchanged; outputs from k on do change."""
    m, chunks, _ = _rollout_and_teacher('cpu', seed=41)
    m.eval()
    c = max(chunks, key=lambda c: len(c.rows))
    n = len(c.rows)
    bs = m.block_memory.config.block_steps
    assert n > 2*bs
    k = bs+1   # a block completes after decision k (inside the chunk), written partly from changed rows
    rows = np.array(c.framed(), copy=True)   # (B45: compact rollout records: their frames inline here)
    f = rows['visible_frame'].view('<f4')
    f[k:, FIXED:FIXED+2] += 0.75   # the player's position (the first body fields)
    c2 = Chunk(rows, c.pending, c.snapshot)
    with torch.no_grad():
        o1, _ = _run(m, [c], True, monkeypatch, train_history=False)
        o2, _ = _run(m, [c2], True, monkeypatch, train_history=False)
    for x, y in zip(o1, o2):
        torch.testing.assert_close(y[:k], x[:k], rtol=0, atol=0)
    assert (o1[0][k:]-o2[0][k:]).abs().max() > 1e-6   # (a small initialised model: small but nonzero)


def test_packed_detach_and_gradreplay_equal_per_length(monkeypatch):
    """B43 modes on the packed path: detach (A) gives the per-length loss and gradients; for
    same-length chunks (one per-length call) gradreplay (C) gives the same block-value gradients
    g_j and, after grad_replay_aux, the same parameter gradients."""
    from test_block_slot_integration import _replay_setup
    from isaac_bridge.block_paged import grad_replay_aux
    m, chunks = _replay_setup()
    m.train()
    for mode in ('detach', 'gradreplay'):
        res = []
        for packed in (False, True):
            out, ctx = _run(m, chunks, packed, monkeypatch, seed=8, history_mode=mode)
            _loss(out).backward()
            g = [r['pool'].grad.clone() for r in ctx['replay']]
            if mode == 'gradreplay':
                torch.manual_seed(4)
                st = grad_replay_aux(m, ctx['replay'], coef=1.0)
                assert st['blocks'] > 0
            res.append((g, _grads(m)))
        (ga, pa), (gb, pb) = res
        assert len(ga) == len(gb) == (1 if mode == 'gradreplay' else 0)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-7)
        assert pa.keys() == pb.keys()
        for k in pa:
            torch.testing.assert_close(pb[k], pa[k], rtol=1e-4, atol=5e-6, msg=f'{mode} {k}')


def test_staging_slab_views_and_reuse():
    """The CPU slab: typed views at aligned offsets; (CUDA: two persistent pinned buffers)."""
    s = block_learner.Slab([('a', (3, 2), torch.float32), ('b', (5,), torch.int64), ('c', (0, 4), torch.bool)], 'cpu')
    s.host['a'][...] = 1.5; s.host['b'][...] = torch.arange(5)
    d = s.upload()
    assert d['a'].sum() == 9 and d['b'].tolist() == [0, 1, 2, 3, 4] and d['c'].shape == (0, 4)
    offs = [v[0] for v in s.layout.values()]
    assert all(o % block_learner.Staging.ALIGN == 0 for o in offs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_staging_reuses_pinned_buffers_on_cuda():
    st = block_learner.STAGING
    before = st.allocs
    for _ in range(6):
        s = block_learner.Slab([('x', (1000,), torch.float32)], 'cuda')
        s.host['x'][...] = 2.0
        assert float(s.upload()['x'].sum()) == 2000.0
    assert st.allocs-before <= 2
    assert all(b[0] is None or b[0].is_pinned() for b in st.bufs)
    assert not any(b[2] for b in st.bufs)   # (B48: every buffer given back after its upload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA (FlexAttention)')
def test_flex_reader_matches_sdpa_reader(monkeypatch):
    """The packed path's FlexAttention Reader (masks and learnable biases inside the kernel) gives
    BlockMemory.read's (SDPA) outputs and every gradient (FP32), including memory_age and
    short_relative, over buckets with prefixes, banks and padded rows."""
    from test_block_memory import small
    m, chunks, teach = _rollout_and_teacher('cuda', seed=51, block_config=small(width=32, latent_width=32))
    with torch.no_grad():
        m.block_memory.memory_age.normal_(0, .5); m.block_memory.short_relative.normal_(0, .5)
    m.train()
    res = []
    for reader in ('sdpa', 'flex'):
        monkeypatch.setattr(block_learner, 'READER', reader)
        out, _ = _run(m, chunks+teach, True, monkeypatch)
        _loss(out).backward()
        res.append(([o.detach() for o in out], _grads(m)))
    (oa, ga), (ob, gb) = res
    for x, y in zip(oa, ob):
        torch.testing.assert_close(y, x, rtol=1e-4, atol=2e-5)
    assert ga.keys() == gb.keys()
    for k in ga:
        torch.testing.assert_close(gb[k], ga[k], rtol=1e-3, atol=2e-5, msg=k)
    assert ga['block_memory.memory_age'].abs().sum() > 0 and ga['block_memory.short_relative'].abs().sum() > 0


def test_dense_block_mask_tables():
    """_block_mask: per 128x128 tile, full / partial / empty from the dense mask (the kernel's
    sparse tables; torch's compiled create_block_mask is not used)."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand((3, 256, 384), generator=g) > 0.5
    a[:, :128, :128] = True
    a[:, 128:, 128:256] = False
    bm = block_learner._block_mask(a, None)
    t = a.view(3, 2, 128, 3, 128)
    some, full = t.any(4).any(2), t.all(4).all(2)
    for b in range(3):
        for qb in range(2):
            n, f = int(bm.kv_num_blocks[b, 0, qb]), int(bm.full_kv_num_blocks[b, 0, qb])
            assert sorted(bm.kv_indices[b, 0, qb, :n].tolist()) == torch.nonzero(some[b, qb] & ~full[b, qb]).flatten().tolist()
            assert sorted(bm.full_kv_indices[b, 0, qb, :f].tolist()) == torch.nonzero(full[b, qb]).flatten().tolist()
    assert bm.seq_lengths == (256, 384)


@pytest.mark.parametrize('packed', [False, True])
def test_compact_decode_equals_padded_decode(packed):
    """decode_frame_compact: the real entries of decode_frame_host's padded lists, in row order;
    room/map identical; a joint decode of several arrays equals separate ones."""
    from test_visible_input import synth_rows
    from isaac_bridge.structured_input import decode_frame_host, decode_frame_compact, RAGGED, host_counts
    rows = synth_rows(np.random.default_rng(3), 9)
    a, b = decode_frame_host(rows, packed=packed), decode_frame_compact(rows, packed=packed)
    np.testing.assert_array_equal(np.asarray(a['v_room']), np.asarray(b['v_room']))
    np.testing.assert_array_equal(np.asarray(a['v_map']), np.asarray(b['v_map']))
    np.testing.assert_array_equal(host_counts(a), b['counts'])
    for name in RAGGED:
        v, mask = a['v_'+name], a['v_'+name+'_mask']
        real = v[mask].reshape(-1, b['c_'+name].shape[1])
        np.testing.assert_array_equal(b['c_'+name], real)
    # several arrays decoded jointly == each alone
    from isaac_bridge.structured_input import decode_frames_compact
    parts = [rows[:4], synth_rows(np.random.default_rng(4), 6), rows[4:]]
    for d, r in zip(decode_frames_compact(parts, packed), parts):
        e = decode_frame_compact(r, packed)
        for k in e:
            np.testing.assert_array_equal(np.asarray(d[k]), np.asarray(e[k]), err_msg=k)


def test_decoded_block_sources_survive_collect_until_freed(monkeypatch):
    """B44: decoded historical sources stay cached across PagedStore.collect while their block is
    live and are dropped when it is freed (a freed block id is reused for a new block)."""
    import gc
    m, chunks, teach = _rollout_and_teacher('cpu', seed=61)
    m.train()
    store = chunks[0].snapshot.store
    _run(m, chunks+teach, True, monkeypatch)
    assert store.decoded
    store.collect()
    kept = set(store.decoded)
    assert kept and all(store.b_used[k[0]] for k in kept)   # live blocks: kept
    del chunks, teach
    gc.collect()
    store.collect()
    assert len(store.decoded) < len(kept)                    # their snapshots are gone: freed, dropped
    assert all(store.b_used[k[0]] for k in store.decoded)
