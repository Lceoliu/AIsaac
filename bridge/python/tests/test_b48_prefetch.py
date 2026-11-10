"""B48: the learner part's host preparation done ahead on the prefetch thread (block_learner.PREFETCH) gives
bit-identical outputs, losses and every gradient to the serial path, over consecutive parts with an optimizer
step between them (the preparation reads nothing the step changes); other calls in between leave the prepared
part for its own call; a part prepared for nothing is dropped and its staging buffer freed."""
import numpy as np
import pytest
import torch

from test_block_packed_learner import _rollout_and_teacher
from test_block_slot_integration import _losses, DEV
from isaac_bridge import block_learner
from isaac_bridge.block_training import forward_bucketed, source_rows
from isaac_bridge.structured_input import from_rows


def _parts(chunks, k=3):
    idx = np.array_split(np.arange(len(chunks)), k)
    return [[chunks[i] for i in ix] for ix in idx]


@pytest.mark.parametrize('device', DEV)
@pytest.mark.parametrize('other_call', [False, True])
def test_prefetched_parts_equal_serial_bit_for_bit(device, other_call, monkeypatch):
    # (one intra-op thread: the serial CPU path's index-accumulating backward -- memory_age, short_relative --
    # is itself not bit-reproducible run to run with several threads)
    # (CUDA: deterministic kernels, CUBLAS_WORKSPACE_CONFIG set by the runner: atomic accumulation in the index /
    # attention backward is otherwise not bit-reproducible run to run, serial vs serial included)
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        _compare(device, other_call, monkeypatch)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det)


def _compare(device, other_call, monkeypatch):
    """Serial, (CUDA: serial again,) prefetched: the uploaded bytes, outputs and losses bit-identical; every
    gradient bit-identical on the CPU; on CUDA, where atomically accumulated gradients differ between two
    serial runs, the prefetched run's gradients differ from the serial one by no more than that."""
    serial = None
    runs = (False, False, True) if device == 'cuda' else (False, True)
    spread = None
    for prefetch in runs:
        m_, ch_, te_ = _rollout_and_teacher(device, seed=37)
        parts = _parts(ch_+te_)
        hits0 = block_learner.PREFETCH.hits

        def run():
            m_.train()
            torch.manual_seed(0)
            opt = torch.optim.Adam(m_.main_parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(9)
            kw = dict(generator=gen, version=3, history_mode='random', max_rows=4096)
            res = []
            for k, part in enumerate(parts):
                out, ctx = forward_bucketed(m_, part, **kw)
                if prefetch and k+1 < len(parts):
                    block_learner.PREFETCH.submit(m_, parts[k+1], **kw)
                if other_call:
                    with torch.no_grad():
                        forward_bucketed(m_, parts[0], train_history=False, version=3, max_rows=4096)
                rows = source_rows([c.framed() for c in part]); pend = np.concatenate([c.pending for c in part])
                loss = _losses(m_, out, from_rows(rows, device, pend), 3)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grads = {n: p.grad.detach().clone() for n, p in m_.named_parameters() if p.grad is not None}
                res.append((loss.detach().clone(), [o.detach().clone() for o in out], grads,
                            dict(ctx['recompute_stats'])))
                opt.step()
            return res
        monkeypatch.setattr(block_learner, 'PACKED', True)
        uploads = []
        upload = block_learner.Slab.upload

        def recorded(slab):   # (every part's upload, the no-grad calls in between too: the same sequence; each
            #                    tensor's bytes, not the alignment gaps between them)
            raw = slab.base.numpy()
            uploads.append({k: bytes(raw[at:at+n]) for k, (at, n, _, _) in slab.layout.items()})
            return upload(slab)
        monkeypatch.setattr(block_learner.Slab, 'upload', recorded)
        got = run()
        monkeypatch.setattr(block_learner.Slab, 'upload', upload)
        got = (got, uploads)
        if prefetch:
            assert block_learner.PREFETCH.hits-hits0 == len(parts)-1
            assert block_learner.PREFETCH.job is None
        if serial is None:
            serial = got
            continue
        (serial_res, serial_up), (got_res, got_up) = serial, got
        # the prepared parts are the same bytes (the slab each part uploads: decodes, index arrays, pages)
        assert len(serial_up) == len(got_up) == len(parts)*(2 if other_call else 1) and serial_up == got_up
        diff = 0.0
        for (la, oa, ga, sa), (lb, ob, gb, sb) in zip(serial_res, got_res):
            assert sa == sb and sa['selected_blocks'] > 0
            assert ga.keys() == gb.keys()
            pairs = [(la, lb)]+list(zip(oa, ob))+[(ga[k], gb[k]) for k in ga]
            for x, y in pairs:
                if device == 'cpu':   # (bit-identical)
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                diff = max(diff, float((x.float()-y.float()).abs().max()))
        if device == 'cuda':
            if not prefetch:   # (the serial run-to-run spread)
                spread = diff
                continue
            print(f'serial vs serial max |diff| (loss, outputs, grads) {spread:.3e}, serial vs prefetched {diff:.3e}')
            # (floor: four FP32 ulps at the compared values' magnitude (~2: 2.4e-7 per ulp). B60: on the 4090 the
            # prefetched run twice differed by exactly one ulp (2.38e-7) while the serial pair happened to agree to
            # 0 / 1.5e-8, the uploaded bytes asserted equal above: one serial pair underestimates the run-to-run
            # spread of the atomically accumulated gradients)
            assert diff <= max(4*spread, 1e-6)


def test_unused_prefetch_is_dropped_and_frees_its_buffer(monkeypatch):
    m, chunks, teach = _rollout_and_teacher('cpu', seed=38)
    m.train()
    monkeypatch.setattr(block_learner, 'PACKED', True)
    block_learner.PREFETCH.submit(m, chunks, version=1)
    # a call with other chunks: the prepared part stays for its own call
    with torch.no_grad():
        forward_bucketed(m, teach, train_history=False, version=1)
    assert block_learner.PREFETCH.job is not None
    misses = block_learner.PREFETCH.misses
    block_learner.PREFETCH.drop()
    assert block_learner.PREFETCH.job is None and block_learner.PREFETCH.misses == misses+1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA (pinned staging pool)')
def test_staging_pool_keeps_a_prepared_slab_until_it_runs():
    st = block_learner.STAGING
    held = block_learner.Slab([('x', (1000,), torch.float32)], 'cuda')
    held.host['x'][...] = 7.0
    for _ in range(4):   # other parts in between: never the held buffer
        s = block_learner.Slab([('x', (1000,), torch.float32)], 'cuda')
        assert s.k != held.k
        s.host['x'][...] = 1.0
        s.upload()
    assert float(held.upload()['x'].sum()) == 7000.0
    assert not any(b[2] for b in st.bufs)
