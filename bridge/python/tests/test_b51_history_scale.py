"""B51: history_scale 'unbiased' (the random arm's default): a chunk's re-encoded historical blocks pass the gradient
back x N_i / |selected_i| straight through. Forward outputs and loss bit-identical to 'none'; the history path's
gradient exactly N/k times larger; per-occurrence scales (a block shared by chunks with different N_i); all three
learner paths (packed, per-length paged, object) agree."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from test_block_memory import small
from test_block_packed_learner import _rollout_and_teacher
from isaac_bridge import block_learner, block_memory, block_paged
from isaac_bridge.block_training import forward_bucketed

CFG = small(memory_blocks=8, history_recompute_blocks=2)   # (banks of up to 8 blocks: N_i > k happens)


def _model_and_chunks(seed=51):
    m, chunks, teach = _rollout_and_teacher('cpu', seed=seed, block_config=CFG)
    return m, chunks, teach


def _set_scale(m, mode):
    m.block_memory.config = replace(m.block_memory.config, history_scale=mode)


def _sum_loss(out):
    """A loss that is a sum over rows (a batch's gradient = the sum of its chunks' gradients)."""
    g = torch.Generator().manual_seed(3)
    return sum((o.float()*torch.randn(o.shape[1:] if o.dim() > 1 else (1,), generator=g)).sum() for o in out)


UPSTREAM = []   # (the gradient arriving at the re-encoded blocks' Writer output, per run)


def _capture_upstream(m, packed, n_hist):
    """Hooks recording dL/dW of the re-encoded historical blocks' Writer output (packed: the first n_hist rows of
    the call's single Writer output; per-length: block_paged.encode_blocks' values)."""
    handles = []
    if packed:
        def fwd(mod, a, out):
            if not UPSTREAM_ARMED[0]:
                return
            UPSTREAM_ARMED[0] = False
            out.register_hook(lambda g: UPSTREAM.append(g[:n_hist].detach().clone()))
        handles.append(m.block_memory.writer.register_forward_hook(fwd))
        return handles, None
    orig = block_paged.encode_blocks

    def enc(*a, **k):
        v, rows = orig(*a, **k)
        if v.requires_grad:
            v.register_hook(lambda g: UPSTREAM.append(g.detach().clone()))
        return v, rows
    return handles, (orig, enc)


UPSTREAM_ARMED = [False]


def _run(m, chunks, mode, seed=5, factor=None, monkeypatch=None, packed=True, n_hist=0):
    _set_scale(m, mode)
    block_learner.PACKED = packed
    if factor is not None:   # (a constant scale for every selected block: 0 removes the history path's gradient)
        monkeypatch.setattr(block_memory, 'history_scale_factor', lambda n, k, c: factor)
    handles, swap = _capture_upstream(m, packed, n_hist) if n_hist else ([], None)
    if swap is not None:
        block_paged.encode_blocks = swap[1]
    UPSTREAM_ARMED[0] = True
    m.zero_grad(set_to_none=True)
    try:
        out, ctx = forward_bucketed(m, chunks, generator=torch.Generator().manual_seed(seed))
        loss = _sum_loss(out)
        loss.backward()
    finally:
        for h in handles:
            h.remove()
        if swap is not None:
            block_paged.encode_blocks = swap[0]
    grads = {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}
    if factor is not None:
        monkeypatch.undo()
    return loss.detach(), [o.detach() for o in out], grads, ctx['recompute_stats']


def _hist_params(m):
    names = [n for n, _ in m.named_parameters() if n.startswith(('structured_spatial.', 'block_memory.resampler.',
                                                                  'block_memory.writer.'))]
    return names


def _eligible(chunk, cfg):
    if chunk.rows['first'][0]:
        return 0
    p = chunk.snapshot.store.snapshot_parts(chunk.snapshot)
    return int(np.count_nonzero(p['ends'] <= p['step']-cfg.window))


@pytest.mark.parametrize('packed', [True, False])
def test_unbiased_forward_bit_identical_and_history_gradient_times_n_over_k(packed, monkeypatch):
    m, chunks, teach = _model_and_chunks()
    m.train()
    cfg = m.block_memory.config
    one = [c for c in chunks+teach if _eligible(c, cfg) > cfg.history_recompute_blocks]
    assert one, 'no chunk with more window-external blocks than k'
    c = max(one, key=lambda c: _eligible(c, cfg))
    N, k = _eligible(c, cfg), cfg.history_recompute_blocks
    UPSTREAM.clear()
    l_none, o_none, g_none, s_none = _run(m, [c], 'none', packed=packed, n_hist=k)
    l_unb, o_unb, g_unb, s_unb = _run(m, [c], 'unbiased', packed=packed, n_hist=k)
    # the gradient arriving at the re-encoded blocks' Writer output: exactly N/k times larger (one multiply)
    assert len(UPSTREAM) == 2 and UPSTREAM[0].abs().sum() > 0
    torch.testing.assert_close(UPSTREAM[1], (N/k)*UPSTREAM[0], rtol=2e-7, atol=0)
    l_0, _, g_0, _ = _run(m, [c], 'unbiased', factor=0.0, monkeypatch=monkeypatch, packed=packed)
    # (a) the forward is unchanged, bit for bit
    assert s_none['selected_blocks'] == s_unb['selected_blocks'] == k
    torch.testing.assert_close(l_unb, l_none, rtol=0, atol=0)
    torch.testing.assert_close(l_0, l_none, rtol=0, atol=0)
    for x, y in zip(o_unb, o_none):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    # (b) the history path's gradient (minus the common part: scale 0) is N/k times the unscaled one
    names = _hist_params(m)
    h_none = torch.cat([(g_none[n]-g_0[n]).reshape(-1) for n in names if n in g_none])
    h_unb = torch.cat([(g_unb[n]-g_0[n]).reshape(-1) for n in names if n in g_unb])
    assert h_none.norm() > 0
    # (on the parameters the common part -- main rows, new blocks -- is ~30x the history part in this model:
    # the difference of two FP32 gradients carries that cancellation, ~1e-6 relative; the exact N/k scaling is
    # the upstream check above)
    rel = float((h_unb-(N/k)*h_none).norm()/((N/k)*h_none).norm())
    assert rel <= 1e-5, rel
    # every other parameter (Reader, heads): the same gradient (the scale acts on the block values only)
    for n in g_none:
        if n not in names:
            torch.testing.assert_close(g_unb[n], g_none[n], rtol=0, atol=0, msg=n)


def test_history_scales_per_occurrence():
    """history_scales: N_i/|sel_i| on each chunk's own selected entries (a block selected by two chunks gets
    each chunk's own scale), 1 elsewhere; None when nothing is scaled ('none', or N_i <= |sel_i|)."""
    cfg = replace(CFG, window=3, block_steps=2)
    parts = [dict(bid=np.array([10, 11, 12, 13, 14]), ends=np.array([1, 3, 5, 7, 9]), step=13),   # N=5 (ends<=10)
             dict(bid=np.array([10, 11, 12]), ends=np.array([1, 3, 5]), step=9),                  # N=3
             dict(bid=np.array([10, 11]), ends=np.array([1, 3]), step=6)]                         # N=2
    selected = [np.array([0, 2]), np.array([0, 1]), np.array([0, 1])]   # block 10 selected by all three
    s = block_paged.history_scales(parts, selected, cfg)
    np.testing.assert_array_equal(s, [2.5, 1, 2.5, 1, 1, 1.5, 1.5, 1, 1, 1])
    assert block_paged.history_scales(parts, selected, replace(cfg, history_scale='none')) is None
    assert block_paged.history_scales(parts[2:], selected[2:], cfg) is None


def test_shared_block_each_chunk_scaled_by_its_own_factor(monkeypatch):
    """Two chunks of one episode with different N_i selecting a common block (re-encoded once, shared): the batch's
    history gradient equals the sum of each chunk's own (alone, the same draws) history gradient x N_i/k."""
    m, chunks, teach = _model_and_chunks(seed=52)
    m.train()
    cfg = m.block_memory.config
    k = cfg.history_recompute_blocks
    cands = [c for c in teach if _eligible(c, cfg) > k]
    pair = None
    for a in range(len(cands)):
        for b in range(a+1, len(cands)):
            A, Bc = cands[a], cands[b]
            if A.snapshot.log.episode != Bc.snapshot.log.episode or _eligible(A, cfg) == _eligible(Bc, cfg):
                continue
            g = torch.Generator().manual_seed(5)
            pa = A.snapshot.store.snapshot_parts(A.snapshot); pb = Bc.snapshot.store.snapshot_parts(Bc.snapshot)
            ea = np.flatnonzero(pa['ends'] <= pa['step']-cfg.window); eb = np.flatnonzero(pb['ends'] <= pb['step']-cfg.window)
            sa = set(pa['bid'][ea[torch.randperm(len(ea), generator=g)[:k].numpy()]].tolist())
            sb = set(pb['bid'][eb[torch.randperm(len(eb), generator=g)[:k].numpy()]].tolist())
            if sa & sb:
                pair = (A, Bc, len(ea), len(eb))
                break
        if pair:
            break
    if pair is None:
        pytest.skip('no pair of chunks sharing a selected block with these draws')
    A, Bc, Na, Nb = pair

    def hist(chunks_, seed_skip=0):
        # (the second chunk alone: its draw comes after the first chunk's in the batch)
        def run(mode, factor=None):
            gen = torch.Generator().manual_seed(5)
            if seed_skip:
                torch.randperm(seed_skip, generator=gen)
            _set_scale(m, mode)
            if factor is not None:
                monkeypatch.setattr(block_memory, 'history_scale_factor', lambda n, kk, c: factor)
            m.zero_grad(set_to_none=True)
            block_learner.PACKED = True
            out, ctx = forward_bucketed(m, chunks_, generator=gen)
            _sum_loss(out).backward()
            if factor is not None:
                monkeypatch.undo()
            return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}, ctx
        g_u, ctx = run('unbiased')
        g_0, _ = run('unbiased', 0.0)
        return {n: g_u[n]-g_0[n] for n in g_u if n in g_0}, ctx
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # (several CPU threads: index-accumulating backward not bit-reproducible run to run)
    try:
        both, ctx = hist([A, Bc])
        assert ctx['recompute_stats']['encoded_blocks'] < ctx['recompute_stats']['selected_blocks']   # (shared)
        ha, _ = hist([A])
        hb, _ = hist([Bc], seed_skip=Na)
    finally:
        torch.set_num_threads(threads)
    # (history parts are differences of FP32 parameter gradients whose common part dominates: compared as one
    # vector, relative error <= 1e-5, as the single-chunk test)
    names = [n for n in _hist_params(m) if n in both]
    x = torch.cat([both[n].reshape(-1) for n in names])
    y = torch.cat([(ha[n]+hb[n]).reshape(-1) for n in names])
    assert y.norm() > 0
    rel = float((x-y).norm()/y.norm())
    assert rel <= 1e-5, rel


def test_causality_and_packed_equivalence_under_unbiased(monkeypatch):
    from test_block_packed_learner import (test_no_decision_reads_later_inputs_or_later_memory as causal,
                                           test_packed_matches_per_length_path_outputs_loss_and_every_gradient as eq)
    causal(monkeypatch)
    eq('cpu', monkeypatch)


def test_hscale_in_slab_bit_identical(monkeypatch):
    """BM1 B step 1 (ISAAC_LEARN_HSCALE_SLAB=1): the gradient scales staged in the part's slab instead of their own
    host->device copy: loss, outputs and every gradient bit-identical to the default path (except the parameters
    whose gradient already differs between two runs of the default path: block_memory.memory_age / short_relative's
    accumulation order, ~4e-7 relative; on those, the same size of difference)."""
    m, chunks, teach = _model_and_chunks()
    m.train()
    seen = []
    orig = block_paged.history_scales

    def spy(*a, **k):
        r = orig(*a, **k)
        seen.append(r is not None)
        return r
    monkeypatch.setattr(block_paged, 'history_scales', spy)
    runs = []
    for on in (False, False, True):
        monkeypatch.setattr(block_learner, 'HSCALE_SLAB', on)
        runs.append(_run(m, chunks+teach, 'unbiased', packed=True))
    assert any(seen), 'no part with scaled history entries'
    (l0, o0, g0, _), (_, _, g0b, _), (l1, o1, g1, _) = runs
    floor = {n for n in g0 if not torch.equal(g0[n], g0b[n])}
    assert len(floor) <= 2, floor
    assert torch.equal(l0, l1)
    assert all(torch.equal(x, y) for x, y in zip(o0, o1))
    assert g0.keys() == g1.keys()
    for n in g0:
        if n in floor:
            assert float((g0[n]-g1[n]).norm()/g0[n].norm()) <= 1e-5, n
        else:
            assert torch.equal(g0[n], g1[n]), n
