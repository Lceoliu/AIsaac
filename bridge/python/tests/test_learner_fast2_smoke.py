"""Smoke test of B38 step 4 (2026-10-17, train_tok --learner-fast 2): the learner's start-of-update pass on the B18 fast
path's host plans, and the CNNs' distinct grids padded to a multiple of cnn_bucket.

Plain script (CPU; the same checks on a GPU where there is one):
  cd rl/bridge/python && python tests/test_learner_fast2_smoke.py
Checks (a temporal model with memory, item tokens and the global token, test_temporal_smoke's rollouts A and B):
  (a) the start-of-update pass as --learner-fast 2 runs it (train_tok.rows_to_device with the records' bytes,
      fast_plan_counts, fast_plan_rows: bucketed attention, real-token plans, the CNNs on the distinct grids, the
      readout from the stored temporal outputs) == as --learner-fast 1 runs it (take() of the gathered fields, packed
      without host plans): logits, value, value_global per record within float tolerance (same function, another
      summation order), the log-probabilities of the stored actions too;
  (b) the PPO batch from the records' bytes (decode_rows of the gathered bytes, as --learner-fast 2 builds it after
      the pass) == take() of the fields, every field bit for bit;
  (c) cnn_bucket (the distinct grids padded with copies of the first, never gathered back): forward outputs and every
      parameter's gradient of a fast-path forward / backward == without padding, within float tolerance (the same
      convolutions on more rows), the padded batch sizes multiples of the bucket;
  (d) row_fields + item_count == item_count of the whole records.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402
import test_memory_smoke as M   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_policy as TP   # noqa: E402
import train_tok as TT   # noqa: E402

FAIL = []
FPARTS = {'math', 'ekeep', 'buckets', 'nhwc', 'ln', 'dedup', 'logp', 'raw'}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def run(dev):
    rng = np.random.default_rng(41)
    cfg = dict(TS.SMALL, memory=TS.DM, **TS.BASE, **TS.FUSED, **TS.TK)
    torch.manual_seed(5)
    m = TP.TokPolicy(**cfg).eval()
    TS.perturb_all(m, 6)
    seq = TS.make_seq(rng)
    pend = [M.synth_pending(rng, TS.NW, m.heads) for _ in seq]
    ra, rb = TS.new_rolls(m)
    TS.run_actor(m, seq, pend, TS.DM, TP.TemporalState(m, TS.NW, 'cpu', torch.float32), (ra, rb))
    m = m.to(dev)
    NI_f, G_f = m.extra_tokens()
    tag = dev.type
    worst = dict(logits=0.0, value=0.0, vg=0.0, logp=0.0)
    same_b, ok_items = True, True
    for roll in (ra, rb):
        cap, nh = roll.cap, len(m.heads)
        all_flat = (np.arange(roll.n)[:, None] * cap + np.arange(cap)[None, :])[
            np.arange(cap)[None, :] < roll.length[:, None]]
        pend_all = roll.pending.reshape(-1, nh)[all_flat]
        every = TP.rows_to_device(roll.rows, all_flat, pend_all, dev)
        every2, raw_e, pend_e = TP.rows_to_device(roll.rows, all_flat, pend_all, dev, with_raw=True)
        acts = torch.from_numpy(roll.actions.reshape(-1, nh)[all_flat]).to(dev)
        h_all = torch.from_numpy(roll.h.reshape(-1, roll.h.shape[-1])[all_flat]).to(dev)
        t_all = TT.temporal_rows(roll, all_flat, cap, dev)
        fr = TT.fast_plan_counts(roll, all_flat, every2, FPARTS, TP.INV_CAP if not hasattr(m, 'item_slots') else
                                 m.item_slots, NI_f, True)
        ref_items = TP.item_count(roll.rows.reshape(-1)[all_flat], fr_slots(m))
        ok_items &= np.array_equal(TP.item_count(TT.row_fields(roll, all_flat, ('inv', 'pitem')), fr_slots(m)),
                                   ref_items)
        chunk = 7
        m.set_math_attention(True, nhwc=dev.type == 'cuda', ln=True)
        with torch.no_grad():
            for at in range(0, len(all_flat), chunk):
                hi = min(at + chunk, len(all_flat))
                part = torch.arange(at, hi, device=dev)
                tk = dict(temporal=dict(mode='stored', o=t_all['o'][part], o_prev=t_all['o_prev'][part],
                                        has=t_all['has'][part]))
                b1 = TP.take(every, part)
                o1 = m(b1, packed=True, mem=h_all[part], need_vg=True, **tk)
                b2 = TT.fast_plan_rows(fr, raw_e, pend_e, at, hi, FPARTS, 3, NI_f, G_f, True)
                o2 = m(b2, packed=True, mem=h_all[part], need_vg=True, **tk)
                lp1 = m.log_prob(o1[0], m.gate(b1), acts[part])[0]
                lp2 = m.log_prob(o2[0], m.gate(b2), acts[part])[0]
                worst['logits'] = max(worst['logits'], float((o1[0] - o2[0]).abs().max()))
                worst['value'] = max(worst['value'], float((o1[1] - o2[1]).abs().max()))
                worst['vg'] = max(worst['vg'], float((o1[-1] - o2[-1]).abs().max()))
                worst['logp'] = max(worst['logp'], float((lp1 - lp2).abs().max()))
        m.set_math_attention(False)
        pos = torch.arange(0, len(all_flat), 2, device=dev)   # (as the PPO batch: some of the records)
        bt = TP.take(every, pos)
        bd = TP.decode_rows(raw_e.index_select(0, pos), pend_e.index_select(0, pos))
        for k, v in bt.items():
            w = bd[k]
            if k in TP.ENTITY_FIELDS:
                n_ = min(v.shape[1], w.shape[1])
                v, w = v[:, :n_], w[:, :n_]
            same_b &= v.shape == w.shape and torch.equal(v, w)
    tol = 1e-4 if dev.type == 'cuda' else 1e-5
    check(f'(a) [{tag}] the start-of-update pass on the fast plans == as before (logits, value, value_global, log-prob)',
          max(worst.values()) < tol, ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    check(f'(b) [{tag}] the PPO batch from the records\' bytes == take() of the fields, bit for bit', same_b)
    check(f'(d) [{tag}] row_fields + item_count == item_count of the whole records', ok_items)
    # (c) cnn_bucket: forward / backward of a fast-path batch with and without padding
    roll = rb
    cap, nh = roll.cap, len(m.heads)
    flat = (np.arange(roll.n)[:, None] * cap + np.arange(cap)[None, :])[
        np.arange(cap)[None, :] < roll.length[:, None]]
    every2, raw_e, pend_e = TP.rows_to_device(roll.rows, flat, roll.pending.reshape(-1, nh)[flat], dev, with_raw=True)
    fr = TT.fast_plan_counts(roll, flat, every2, FPARTS, fr_slots(m), NI_f, True)
    b2 = TT.fast_plan_rows(fr, raw_e, pend_e, 0, len(flat), FPARTS, 3, NI_f, G_f, True)
    h_all = torch.from_numpy(roll.h.reshape(-1, roll.h.shape[-1])[flat]).to(dev)
    t_all = TT.temporal_rows(roll, flat, cap, dev)
    tk = dict(temporal=dict(mode='stored', o=t_all['o'], o_prev=t_all['o_prev'], has=t_all['has']))
    outs, grads = [], []
    m.train()
    m.set_math_attention(True, nhwc=dev.type == 'cuda', ln=True)
    sizes = []
    for bucket in (0, 16):
        m.cnn_bucket = bucket
        m.zero_grad(set_to_none=True)
        o = m(b2, packed=True, mem=h_all, need_vg=True, **tk)
        (o[0].float().pow(2).mean() + o[1].float().pow(2).mean() + o[-1].float().pow(2).mean()).backward()
        outs.append([x.detach().clone() for x in (o[0], o[1], o[-1])])
        grads.append([None if q.grad is None else q.grad.clone() for q in m.parameters()])
        if bucket:
            sizes = [int(b2[k + '_u'].shape[0]) for k in TP.DEDUP if k + '_u' in b2]
    m.cnn_bucket = 0
    m.set_math_attention(False)
    m.eval()
    d_out = max(float((a - b).abs().max()) for a, b in zip(*outs))
    d_g = max((float((a - b).abs().max()) / max(float(a.abs().max()), 1e-12)) for a, b in zip(*grads)
              if a is not None and b is not None)
    none_same = all((a is None) == (b is None) for a, b in zip(*grads))
    check(f'(c) [{tag}] cnn_bucket 16: outputs and gradients == without padding (distinct grids {sizes}, padded to '
          'multiples of 16 inside)', d_out < tol and d_g < 1e-3 and none_same,
          f'outputs {d_out:.1e}, gradients (relative) {d_g:.1e}')


def fr_slots(m):
    return getattr(m, 'item_slots', TP.INV_CAP)


def main():
    run(torch.device('cpu'))
    if torch.cuda.is_available():
        run(torch.device('cuda', 0))
    print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
