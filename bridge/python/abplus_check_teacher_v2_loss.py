"""Teacher v2 loss check (2026-10-10, isaac_bridge/tok_teacher2.teacher_v2_loss as train_tok uses it): on a hand-made
batch of ROW records (random policy inputs, random gates) and a small TokPolicy on the CPU:

1. the loss from the model's log_prob (and log_prob_fast, the B18 path) equals the value computed by hand in float64
   from the same logits and values: per head log-softmax with the gated "use" entries at -1e9, the recorded actions'
   log-probabilities summed over the heads, share x (coef x mean(-w logp) + vf x mean((V - G)^2));
2. train_tok's merged layout (the records after PPO rows and teacher rows in one forward, sliced at k + teacher rows)
   gives the same loss and the same gradient as a forward of the imitation records alone;
3. one Adam step on the loss alone raises the recorded actions' log-probability (the direction).

usage (bridge python dir, PYTHONPATH=.; CPU only): python abplus_check_teacher_v2_loss.py [--items]
"""
import argparse

import numpy as np
import torch

from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_policy import TokPolicy, cat_batch, to_batch
from isaac_bridge.tok_teacher2 import teacher_v2_loss


def hand_rows(rng, n, items):
    rows = np.zeros(n, ROW)
    rows['player'] = rng.normal(size=rows['player'].shape).astype(np.float32) * 0.5
    rows['n_ent'] = rng.integers(0, 6, n)
    rows['n_doors'] = rng.integers(0, 4, n)
    for i in range(n):
        k = int(rows['n_ent'][i])
        rows['ent'][i, :k] = rng.normal(size=(k, rows['ent'].shape[2])).astype(np.float32) * 0.5
        rows['ent_id'][i, :k, 0] = rng.integers(1, 300, k)
        rows['ent_id'][i, :k, 1] = rng.integers(0, 20, k)
        d = int(rows['n_doors'][i])
        rows['doors'][i, :d] = rng.normal(size=(d, rows['doors'].shape[2])).astype(np.float32) * 0.5
    rows['grid'] = rng.integers(0, 2, rows['grid'].shape)
    rows['patch'] = rng.integers(0, 2, rows['patch'].shape)
    rows['map'] = rng.integers(0, 2, rows['map'].shape)
    rows['bombs'] = rng.integers(0, 2, n)            # the bomb head's gate: about half the records without a bomb
    if items:
        rows['player'][:, 22] = rng.integers(0, 2, n)   # the active item ready (the item head's gate)
        rows['pinv'][:, 2] = rng.integers(0, 2, n)      # a pill held (the pill head's gate)
    return rows


def by_hand(logits, value, actions, gate, heads, w, G, share, coef, vf):
    """The expected loss in float64 (numpy): see the module docstring."""
    lg = logits.astype(np.float64)
    logp = np.zeros(len(lg))
    at = 0
    for i, h in enumerate(heads):
        x = lg[:, at:at + h].copy()
        if i >= 2:   # bomb, item, pill: "use" only through the gate
            x[:, 1] = np.where(gate[:, i - 2], x[:, 1], -1e9)
        x = x - x.max(1, keepdims=True)
        ls = x - np.log(np.exp(x).sum(1, keepdims=True))
        logp += ls[np.arange(len(lg)), actions[:, i]]
        at += h
    pi = np.mean(-w * logp)
    v = np.mean((value.astype(np.float64) - G) ** 2)
    return share * (coef * pi + vf * v), logp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', action='store_true')
    p.add_argument('--n', type=int, default=24)
    args = p.parse_args()
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    model = TokPolicy(64, 2, 4, items=args.items, charge=True, ent_ext=True).double().float()
    model.eval()
    heads = list(model.heads)
    n = args.n
    rows = hand_rows(rng, n, args.items)
    pend = np.stack([rng.integers(0, h, n) for h in heads], 1).astype(np.int64)
    acts = np.stack([rng.integers(0, h, n) for h in heads], 1).astype(np.int64)
    b = to_batch(rows, pend, 'cpu')
    gate = model.gate(b).numpy()
    for i in range(2, len(heads)):   # a recorded "use" only where the gate allows it (a worker's action always does)
        acts[:, i] = np.where(gate[:, i - 2], acts[:, i], 0)
    w = rng.uniform(0.3, 2.0, n).astype(np.float32)
    G = rng.normal(size=n).astype(np.float32)
    share, coef, vf = 0.37, 1.0, 0.25
    out = {}
    with torch.no_grad():
        logits, value, _ = model(b)
    exp, logp_np = by_hand(logits.numpy(), value.numpy(), acts, gate, heads, w.astype(np.float64),
                           G.astype(np.float64), share, coef, vf)
    a_t, w_t, G_t = torch.from_numpy(acts), torch.from_numpy(w), torch.from_numpy(G)
    for name in ('log_prob', 'log_prob_fast'):
        lp, _ = getattr(model, name)(logits, model.gate(b), a_t)
        loss, stats = teacher_v2_loss(lp, value, w_t, G_t, share, coef, vf)
        out[name] = dict(loss=float(loss), expected=float(exp), abs_diff=abs(float(loss) - exp),
                         logp_max_abs_diff=float(np.abs(lp.double().numpy() - logp_np).max()),
                         stats=[round(float(s), 6) for s in stats])
    # 2: the merged layout (8 "PPO" rows, 5 "teacher" rows, then the imitation records) against the records alone
    other = hand_rows(rng, 13, args.items)
    pend_o = np.stack([rng.integers(0, h, 13) for h in heads], 1).astype(np.int64)
    bo = to_batch(other, pend_o, 'cpu')
    merged = cat_batch(cat_batch({k: v[:8] for k, v in bo.items()}, {k: v[8:] for k, v in bo.items()}), b)
    grads = []
    losses = []
    for kind in ('alone', 'merged'):
        model.zero_grad(set_to_none=True)
        if kind == 'alone':
            lg_, v_, _ = model(b)
            lp_, _ = model.log_prob(lg_, model.gate(b), a_t)
            loss_, _ = teacher_v2_loss(lp_, v_, w_t, G_t, share, coef, vf)
        else:
            lg_, v_, _ = model(merged)
            o_ = 8 + 5
            lp_, _ = model.log_prob(lg_[o_:], model.gate(merged)[o_:], a_t)
            loss_, _ = teacher_v2_loss(lp_, v_[o_:], w_t, G_t, share, coef, vf)
        loss_.backward()
        losses.append(float(loss_))
        grads.append(torch.cat([q.grad.reshape(-1) for q in model.parameters() if q.grad is not None]))
    out['merged'] = dict(loss_alone=losses[0], loss_merged=losses[1], loss_abs_diff=abs(losses[0] - losses[1]),
                         grad_rel_diff=float((grads[0] - grads[1]).norm() / grads[0].norm()))
    # 3: one step on the loss alone (coef 1, vf 0) raises the recorded actions' log-probability
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad(set_to_none=True)
    lg_, v_, _ = model(b)
    lp0, _ = model.log_prob(lg_, model.gate(b), a_t)
    teacher_v2_loss(lp0, v_, w_t, G_t, 1.0, 1.0, 0.0)[0].backward()
    opt.step()
    with torch.no_grad():
        lg_, v_, _ = model(b)
        lp1, _ = model.log_prob(lg_, model.gate(b), a_t)
    out['step'] = dict(logp_before=float(lp0.mean()), logp_after=float(lp1.mean()),
                       raised=bool(float(lp1.mean()) > float(lp0.mean())))
    ok = all(v['abs_diff'] < 1e-5 * max(1.0, abs(v['expected'])) for k, v in out.items() if k.startswith('log_prob')) \
        and out['merged']['loss_abs_diff'] < 1e-5 and out['merged']['grad_rel_diff'] < 1e-4 and out['step']['raised']
    out['ok'] = ok
    import json
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
