"""D6: the block-memory learner gives every trained row the advantage, return, old value, old log-probability and
action of ITS OWN (worker, t).

train_tok.learn()'s block branch cuts each worker's stored decisions into chunks (BlockRollout.plan), gathers the
per-row PPO inputs at the plan's flat positions, draws a permutation of chunks per epoch, cuts each minibatch into
parts of whole chunks (temporal_bounds) and runs forward_bucketed on each part's chunk list. Nothing checked that the
row order of the gathered tensors, the forward's row order and the rollout's (worker, t) agree. This test follows
that path from the rollout to the loss tensors and recovers each row's (worker, t) twice: from the plan's chunk
table (tch: offset, length, worker, begin) and from its flat position (worker * capacity + t).

Production functions are called where they exist (gae, BlockRollout.plan, temporal_bounds, rows_to_device, take,
block_learner.EpochPerms, forward_bucketed, normalise_advantages, TokPolicy.gate / log_prob). The indexing that is
inline code in learn() is replicated verbatim (modulo `.to(device)` and the pinned `dev()` copy, CPU here); each
replicated statement cites its train_tok.py line (tree maa1e98a).

Parametrised over the packed / per-length learner (block_learner.PACKED, which forward_bucketed reads at call time;
ISAAC_PACKED_LEARNER is only read once, at block_learner's import, so the env var is set for the record and the
module attribute is what switches the path; no importlib.reload: it would replace the module's PREFETCH singleton
and caches other modules hold), two seeds of the epoch permutations (torch's default CPU generator, as
EpochPerms / torch.randperm draw them in learn()), and two minibatch configurations (several minibatches with
several parts each; and the production defaults --minibatch 2048 --micro 0: one minibatch, one part)."""
import functools
from collections import Counter

import numpy as np
import pytest
import torch

from test_block_slot_integration import _collect_rollout
from isaac_bridge import block_learner
from isaac_bridge import block_paged
from isaac_bridge import block_training
from isaac_bridge.block_training import forward_bucketed
from isaac_bridge.tok_pickup import normalise_advantages
from isaac_bridge.tok_policy import rows_to_device, take
from train_tok import gae, temporal_bounds

GAMMA, GAMMA_G, LAM = 0.995, 0.999, 0.95   # (--gamma, --gamma-global, --lam defaults: train_tok.py:1644, 2118, 1645)
T_T = 128   # (block mode sets args.temporal_bptt = 128: train_tok.py:2353; T_t = int(args.temporal_bptt): 2357)
EPOCHS = 2  # (--epochs default: train_tok.py:1632)
CONFIGS = {'mb512_micro6': dict(minibatch=512, micro=6),      # mb_units = 4 chunks, parts of <= 6 rows
           'defaults': dict(minibatch=2048, micro=0)}         # mb_units = 16 chunks: one minibatch, one part
FWD_TOL = dict(rtol=1e-4, atol=1e-5)   # (recomputed vs the actor's values: summation order only; a misaligned row
#                                        differs by O(0.1))


@functools.lru_cache(maxsize=1)
def _rollout():
    m, ref, new, old, rolls = _collect_rollout('cpu')
    return m, rolls[0], new   # (the actor kept alive: its page store holds the snapshots' pages)


def _wt_text(wt, i):
    return f'(worker {int(wt[i, 0])}, t {int(wt[i, 1])})'


def _check(where, field, wt, got, want, tol=None):
    """got / want per row [k] or [k, heads]; exact unless tol. On failure: the first mismatching (worker, t)."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f'{where}: {field} shapes {got.shape} vs {want.shape}'
    bad = ~np.isclose(got, want, **tol) if tol is not None else got != want
    if bad.ndim > 1:
        bad = bad.reshape(len(bad), -1).any(1)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(f'{where}: first mismatch at {_wt_text(wt, i)}, field {field}: expected {want[i]!r}, '
                             f'gathered {got[i]!r} ({int(bad.sum())} of {len(bad)} rows differ)')


def _np(x):
    return x.detach().float().cpu().numpy() if x.dtype.is_floating_point else x.detach().cpu().numpy()


def _recompute(m, rl, micro, minibatch):
    """The update-start pass of the block branch (train_tok.py:3394-3447, `recompute or args.decoupled`): every stored
    decision's value, value_global and the log-probability of its stored action, scattered back to the rollout
    layout. Replicated verbatim apart from autocast / the B60 prefetch (preparation only) / the priv critic / slog."""
    n, capacity, nh = rl.n, rl.cap, rl.nh
    lengths = rl.length
    device = 'cpu'
    all_flat = (np.arange(n)[:, None] * capacity + np.arange(capacity)[None, :])[
        np.arange(capacity)[None, :] < lengths[:, None]]                                        # train_tok.py:3394
    all_mask = np.arange(capacity)[None, :] < lengths[:, None]                                  # :3407
    rp_b = rl.block.plan(rl, all_mask)                                                          # :3408
    assert np.array_equal(rp_b['pos'], all_flat)                                                # :3409
    every = rows_to_device(rl.rows, all_flat, rl.pending.reshape(-1, nh)[all_flat], device, head=True)   # :3411
    value_now = rl.value.copy()                                                                 # :3414
    lp_parts, values_b, globals_b = [], [], []
    with torch.inference_mode():
        rgroups_b = list(temporal_bounds(rp_b['tch'], np.arange(len(rp_b['chunks'])), micro or minibatch))  # :3421
        for g_b, (lo_b, hi_b, ids_b) in enumerate(rgroups_b):
            out_b, ctx_b = forward_bucketed(m, [rp_b['chunks'][i] for i in ids_b],
                                            train_history=False, version=0)                    # :3432
            bb = take(every, torch.arange(lo_b, hi_b, device=device))                           # :3434
            act_b = torch.from_numpy(rl.actions.reshape(-1, nh)[all_flat[lo_b:hi_b]]).to(device)   # :3435
            lp_parts.append(m.log_prob(out_b[0], m.gate(bb), act_b)[0])                         # :3436
            vb_, gb_ = out_b[1], out_b[-1]                                                      # :3437
            values_b.append(vb_.float()); globals_b.append(gb_.float())
        value_now.reshape(-1)[all_flat] = torch.cat(values_b).cpu().numpy()                     # :3445
        prox_all = torch.cat(lp_parts).clone()                                                  # :3446
        vg_now = np.zeros_like(rl.value)                                                        # :3535
        vg_now.reshape(-1)[all_flat] = torch.cat(globals_b).cpu().numpy()                       # :3551
    return all_flat, value_now, vg_now, prox_all


@pytest.mark.parametrize('config', list(CONFIGS))
@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('packed', [1, 0])
def test_trained_rows_get_their_own_adv_ret_value_logp_action(packed, seed, config, monkeypatch):
    monkeypatch.setenv('ISAAC_PACKED_LEARNER', str(packed))           # (for the record: read at import only)
    monkeypatch.setattr(block_learner, 'PACKED', bool(packed))        # (what forward_bucketed reads: block_training:862)
    minibatch, micro_arg = CONFIGS[config]['minibatch'], CONFIGS[config]['micro']
    m, rl, _actor = _rollout()
    m.eval()
    block_paged.clear_decode_cache()   # (learn() does this at each update's start: train_tok.py:3310)
    n, capacity, nh = rl.n, rl.cap, rl.nh
    device = 'cpu'
    tag = f'packed={packed} seed={seed} {config}'
    # which learner path forward_bucketed took (counted, behaviour unchanged): forward_packed (block_training:864) or
    # the per-length forward_chunks (block_training:876)
    calls = dict(packed=0, per_length=0)

    def counted(fn, key):
        def wrapper(*a, **k):
            calls[key] += 1
            return fn(*a, **k)
        return wrapper
    monkeypatch.setattr(block_learner, 'forward_packed', counted(block_learner.forward_packed, 'packed'))
    monkeypatch.setattr(block_training, 'forward_chunks', counted(block_training.forward_chunks, 'per_length'))

    # ---- the update-start pass (with --recompute-values on / auto+overlap, or --decoupled) -------------------------
    all_flat, value_re, vg_re, prox_all = _recompute(m, rl, micro_arg, minibatch)
    live = np.arange(capacity)[None, :] < rl.length[:, None]
    wt_live = np.argwhere(live)
    # its scatter: each decision's recomputed value / value_global / log-probability is that decision's own (the same
    # weights as the actor's: equal up to summation order)
    _check(f'{tag} recompute', 'value (recomputed vs stored)', wt_live, value_re[live], rl.value[live], FWD_TOL)
    _check(f'{tag} recompute', 'value_global (recomputed vs stored)', wt_live, vg_re[live], rl.value_g[live], FWD_TOL)
    _check(f'{tag} recompute', 'prox logp (recomputed vs stored logp)', wt_live,
           _np(prox_all), rl.logp.reshape(-1)[all_flat], FWD_TOL)

    # ---- the epoch permutations (EpochPerms / torch.randperm on the default CPU generator: train_tok.py:4490, 4565-4568)
    # (gap: in learn() epoch e+1's permutation is drawn after epoch e's parts drew their history blocks from the same
    #  generator; here all are drawn first. The draw's value differs, the permutation -> rows indexing does not.)
    forward_cache, summary = {}, []
    for value_source in ('stored', 'recomputed'):
        where0 = f'{tag} values={value_source}'
        value_now = None if value_source == 'stored' else value_re
        vg_now = rl.value_g if value_source == 'stored' else vg_re          # :3531 (B24 stored) / :3551 (recompute)
        _, ret_g, _ = gae(rl, GAMMA_G, LAM, vg_now)                          # :3532 / :3552
        adv, ret, mask = gae(rl, GAMMA, LAM, value_now)                      # :3555
        # ---- plan -> flat rows (the block learner is `chunked`: train_tok.py:2359) ------------------------------------
        mplan = rl.block.plan(rl, mask)                                      # :3565
        flat, lw = mplan['pos'], mplan['weight']                             # :3567
        count = int(lw.sum())                                                # :3568
        pos = torch.from_numpy(np.searchsorted(all_flat, flat)).to(device)   # :3596 (the `every is not None` branch)
        prox = prox_all[pos].clone()                                         # :3600 (args.decoupled)
        batch = rows_to_device(rl.rows, flat, rl.pending.reshape(-1, nh)[flat], device, head=True)   # :3609 (block_on)
        b_act = torch.from_numpy(rl.actions.reshape(-1, nh)[flat]).to(device)                     # :3613
        b_logp = torch.from_numpy(rl.logp.reshape(-1)[flat]).to(device)                           # :3614
        b_adv = torch.from_numpy(adv.reshape(-1)[flat]).to(device)                                 # :3615
        b_ret = torch.from_numpy(ret.reshape(-1)[flat]).to(device)                                 # :3616
        used_value = rl.value if value_now is None else value_now                                  # :3617
        b_val = torch.from_numpy(used_value.reshape(-1)[flat]).to(device)                          # :3618
        b_ret_g = torch.from_numpy(ret_g.reshape(-1)[flat]).to(device)                             # :3621
        b_vg = torch.from_numpy(vg_now.reshape(-1)[flat]).to(device)                               # :3622
        pf_np = rl.rows['pforce'].reshape(-1)[flat] > 0                                            # :3628 (pf_on)
        b_pf = torch.from_numpy(pf_np).to(device)                                                  # :3629
        b_w = torch.from_numpy(lw).to(device)                                                      # :3639
        tch = mplan['tch']                                                                         # :3646

        # (worker, t) of every flat row, twice: the chunk table (offset, length, worker, begin) and the flat position
        assert tch.shape[1] == 4 and len(tch) == len(mplan['chunks'])
        wt = np.full((len(flat), 2), -1, np.int64)
        for c, (off_, len_, w_, b_) in enumerate(tch):
            assert len(mplan['chunks'][c].rows) == len_, f'{where0}: chunk {c} has {len(mplan["chunks"][c].rows)} rows, tch says {len_}'
            wt[off_:off_ + len_, 0] = w_
            wt[off_:off_ + len_, 1] = b_ + np.arange(len_)
        assert (wt >= 0).all() and int(tch[-1, 0] + tch[-1, 1]) == len(flat), f'{where0}: tch does not tile flat'
        _check(where0, 'flat position vs chunk table (worker, t)', wt, np.stack(np.divmod(flat, capacity), 1), wt)
        assert len(np.unique(flat)) == len(flat), f'{where0}: a flat position appears twice in the plan'
        # every chunk's records are the rollout's at its rows (byte for byte; the records are distinct, so this pins
        # the chunk's (worker, t), independent of the plan's own bookkeeping)
        rec = rl.rows.reshape(-1).view(np.uint8).reshape(n * capacity, -1)
        assert len({rec[p].tobytes() for p in flat}) == len(flat), 'the synthetic records are not distinct'
        chunk_bytes = np.concatenate([c.rows.view(np.uint8).reshape(len(c.rows), -1) for c in mplan['chunks']])
        _check(where0, 'chunk record bytes', wt, chunk_bytes, rec[wt[:, 0] * capacity + wt[:, 1]])
        chunk_pend = np.concatenate([c.pending for c in mplan['chunks']])
        _check(where0, 'chunk pending', wt, chunk_pend, rl.pending[wt[:, 0], wt[:, 1]])

        # the main PPO row weights: lw = the plan's weight = mask at the row (1: trained; 0: a row of a trained chunk
        # that is only stepped through: an open decision / a branch end). pmean and the advantage normalisation use it
        # as pm (0 / 1); the part's share w = its trained rows / the minibatch's. Every trained row's weight is 1.
        _check(where0, 'row weight lw (= mask)', wt, lw, mask[wt[:, 0], wt[:, 1]].astype(np.float32))
        assert set(np.unique(lw).tolist()) <= {0.0, 1.0}
        zero_wt = [tuple(x) for x in wt[lw == 0].tolist()]
        trained_expected = {tuple(x) for x in np.argwhere(mask).tolist()}

        n_units, mb_units = len(tch), max(1, minibatch // T_T)               # :4042
        perms = block_learner.EpochPerms(n_units, device)                    # :4490 (ahead_: PREFETCH_EPOCH, no memory)
        torch.manual_seed(seed)
        perm_list = [perms.get(e) for e in range(EPOCHS)]
        skipped_chunks = 0
        for ep_ in range(EPOCHS):
            perm, perm_h = perm_list[ep_]                                    # :4565
            seen = []
            for at in range(0, n_units, mb_units):                           # :4574
                sel_h = perm_h[at:at + mb_units]                             # :4575
                if len(sel_h) < mb_units // 4:                               # :4576
                    skipped_chunks += len(sel_h)
                    continue
                sel = perm[at:at + mb_units]                                 # :4578
                tc_h = sel_h                                                 # :4581
                sel_h = np.concatenate([np.arange(o_, o_ + c_) for o_, c_ in tch[tc_h, :2]])   # :4582
                sel = torch.from_numpy(np.ascontiguousarray(sel_h, np.int64))   # :4588 (dev(): pinned copy, CPU here)
                n_sel = max(float(lw[sel_h].sum()), 1.0)                     # :4589
                # advantage normalisation, both branches: without forced decisions (inline, :4593-4596) and with them
                # (pf_on: tok_pickup.normalise_advantages, :4591); with no forced row they give the same values
                a_all, w_all = b_adv[sel], b_w[sel]                          # :4593
                mu_ = (a_all * w_all).sum() / n_sel                          # :4594
                adv_std = (((a_all - mu_) ** 2 * w_all).sum() / max(n_sel - 1.0, 1.0)).sqrt()   # :4595
                a_all = (a_all - mu_) / (adv_std + 1e-8) * w_all             # :4596
                a_pf, _ = normalise_advantages(b_adv[sel], b_pf[sel], b_w[sel])                  # :4591
                micro = micro_arg or len(sel)                                # :4616
                bounds = temporal_bounds(tch, tc_h, micro)                   # :4666
                where_mb = f'{where0} epoch {ep_} minibatch at {at}'
                # the advantage normalisation recomputed from the rollout at each row's own (worker, t)
                wt_mb = wt[sel_h]
                m_rows = mask[wt_mb[:, 0], wt_mb[:, 1]]
                raw = adv[wt_mb[:, 0], wt_mb[:, 1]].astype(np.float64)
                k_ = max(float(m_rows.sum()), 1.0)
                mu_e = (raw * m_rows).sum() / k_
                sd_e = np.sqrt(((raw - mu_e) ** 2 * m_rows).sum() / max(k_ - 1.0, 1.0))
                want_a = (raw - mu_e) / (sd_e + 1e-8) * m_rows
                _check(where_mb, 'normalised advantage a', wt_mb, _np(a_all), want_a, dict(rtol=1e-4, atol=1e-5))
                f_rows = rl.rows['pforce'][wt_mb[:, 0], wt_mb[:, 1]] > 0
                _check(where_mb, 'forced flag', wt_mb, _np(b_pf[sel]), f_rows)
                keep = m_rows & ~f_rows
                k2 = max(float(keep.sum()), 1.0)
                mu2 = (raw * keep).sum() / k2
                sd2 = np.sqrt(((raw - mu2) ** 2 * keep).sum() / max(k2 - 1.0, 1.0))
                _check(where_mb, 'normalised advantage a (pf_on)', wt_mb, _np(a_pf),
                       (raw - mu2) / (sd2 + 1e-8) * keep, dict(rtol=1e-4, atol=1e-5))
                assert bounds[0][0] == 0 and bounds[-1][1] == len(sel_h)
                w_sum = 0.0
                for lo_, hi_, cids_ in bounds:
                    part = sel[lo_:hi_]                                      # :4154
                    part_h = sel_h[lo_:hi_]                                  # :4155
                    pn_ = float(lw[part_h].sum())                            # :4157
                    w, pm_, pn_ = pn_ / n_sel, b_w[part], max(pn_, 1.0)      # :4158
                    w_sum += w
                    mb = {k: v[part] for k, v in batch.items()}              # :4179 (non-fast path)
                    R_ = dict(act=b_act[part], logp=b_logp[part], ret=b_ret[part], a=a_all[lo_:hi_])   # :4243
                    R_['pf'] = b_pf[part]                                    # :4246
                    R_['prox'] = prox[part]                                  # :4248
                    R_['ret_g'] = b_ret_g[part]                              # :4250
                    chunks_b = [mplan['chunks'][i] for i in cids_]           # :4692
                    where = f'{where_mb} part rows {lo_}:{hi_} chunks {list(cids_)}'
                    pw = wt[part_h]
                    W, T = pw[:, 0], pw[:, 1]
                    # the part's rows are exactly its chunks' rows, in the forward's row order
                    fwd_wt = np.concatenate([np.stack([np.full(tch[c, 1], tch[c, 2]), tch[c, 3] + np.arange(tch[c, 1])], 1)
                                             for c in cids_])
                    _check(where, 'forward row (worker, t) vs gathered row (worker, t)', pw, fwd_wt, pw)
                    fwd_bytes = np.concatenate([c.rows.view(np.uint8).reshape(len(c.rows), -1) for c in chunks_b])
                    _check(where, 'forward input record bytes', pw, fwd_bytes, rec[W * capacity + T])
                    # the gathered per-row PPO inputs: the rollout's arrays at the row's own (worker, t), exactly
                    _check(where, 'adv (b_adv[sel])', pw, _np(b_adv[sel][lo_:hi_]), adv[W, T])
                    _check(where, 'ret', pw, _np(R_['ret']), ret[W, T])
                    _check(where, 'old value (b_val)', pw, _np(b_val[part]), used_value[W, T])
                    _check(where, 'old logp', pw, _np(R_['logp']), rl.logp[W, T])
                    _check(where, 'action', pw, _np(R_['act']), rl.actions[W, T])
                    _check(where, 'ret_g', pw, _np(R_['ret_g']), ret_g[W, T])
                    _check(where, 'old value_global (b_vg)', pw, _np(b_vg[part]), vg_now[W, T])
                    _check(where, 'forced flag', pw, _np(R_['pf']), rl.rows['pforce'][W, T] > 0)
                    _check(where, 'prox logp (decoupled)', pw, _np(R_['prox']), rl.logp[W, T], FWD_TOL)
                    _check(where, 'normalised advantage a (part slice)', pw, _np(R_['a']),
                           want_a[lo_:hi_], dict(rtol=1e-4, atol=1e-5))
                    _check(where, 'row weight pm', pw, _np(pm_), mask[W, T].astype(np.float32))
                    _check(where, 'gate batch pending', pw, _np(mb['pending']), rl.pending[W, T])
                    _check(where, 'gate batch n_ent', pw, _np(mb['n_ent']), rl.rows['n_ent'][W, T])
                    # the forward over the part's chunks (forward_bucketed as at :4698; no history recompute, no grad:
                    # the outputs do not depend on which history blocks are re-encoded) against the gathered rows
                    key = tuple(int(c) for c in cids_)
                    if key not in forward_cache:
                        with torch.no_grad():
                            outs_, _ = forward_bucketed(m, chunks_b, max_rows=micro_arg or minibatch,
                                                        train_history=False, version=0)
                        forward_cache[key] = [o.detach() for o in outs_]
                    outs_ = forward_cache[key]
                    assert len(outs_[0]) == hi_ - lo_, f'{where}: forward gave {len(outs_[0])} rows for {hi_ - lo_}'
                    with torch.no_grad():
                        logp_f = m.log_prob(outs_[0], m.gate(mb), R_['act'])[0]
                    _check(where, 'forward value vs old value', pw, _np(outs_[1]), _np(b_val[part]), FWD_TOL)
                    _check(where, 'forward value_global vs old value_global', pw, _np(outs_[-1]), _np(b_vg[part]),
                           FWD_TOL)
                    _check(where, 'forward logp(stored action) vs old logp', pw, _np(logp_f), _np(R_['logp']),
                           FWD_TOL)
                    seen.extend(tuple(x) for x in pw.tolist())
                    summary.append((value_source, ep_, at, lo_, hi_, len(cids_), int(lw[part_h].sum()), round(w, 4)))
                assert abs(w_sum - float(lw[sel_h].sum()) / n_sel) < 1e-9 and (lw[sel_h].sum() == 0 or abs(w_sum - 1) < 1e-9), \
                    f'{where_mb}: the parts\' shares w sum to {w_sum}'
            # coverage: each epoch steps through every planned row once and trains every mask-True (worker, t) once
            cnt = Counter(seen)
            dup = [k for k, v in cnt.items() if v > 1]
            assert not dup, f'{where0} epoch {ep_}: (worker, t) gathered twice: {dup[:5]}'
            if skipped_chunks == 0:
                assert set(cnt) == {tuple(x) for x in wt.tolist()}, f'{where0} epoch {ep_}: planned rows not covered'
                trained = {k for k in cnt if mask[k]}
                missing = sorted(trained_expected - trained)
                extra = sorted(trained - trained_expected)
                assert not missing and not extra, (f'{where0} epoch {ep_}: trained (worker, t) != mask: missing '
                                                   f'{missing[:5]}, extra {extra[:5]}')
        print(f'D6 {where0}: {int(live.sum())} stored decisions, {int(mask.sum())} with mask, {len(flat)} planned '
              f'rows in {len(tch)} chunks ({count} weight 1, {len(zero_wt)} weight 0 at {zero_wt}); '
              f'{n_units} units, mb_units {mb_units}, skipped tail chunks {skipped_chunks}; '
              f'forced rows {int(pf_np.sum())}; parts per epoch '
              f'{[sum(1 for s in summary if s[0] == value_source and s[1] == e) for e in range(EPOCHS)]}')
    print(f'D6 {tag}: parts (values, epoch, minibatch at, lo, hi, chunks, trained rows, share w): {summary}')
    print(f'D6 {tag}: forward calls {calls}')
    # the path under test was the one taken
    if packed:
        assert calls['packed'] > 0 and calls['per_length'] == 0, f'{tag}: forward calls {calls}'
    else:
        assert calls['packed'] == 0 and calls['per_length'] > 0, f'{tag}: forward calls {calls}'
    # negative control: a one-row shift of a gathered column is caught, with its (worker, t) in the message
    W, T = wt[:, 0], wt[:, 1]
    with pytest.raises(AssertionError, match=r'first mismatch at \(worker \d+, t \d+\), field adv'):
        _check(f'{tag} negative control', 'adv', wt, np.roll(_np(b_adv), 1), adv[W, T])
