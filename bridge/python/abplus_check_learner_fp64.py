"""B18 (2026-10-10): is the fast learner path (train_tok --learner-fast 1) as close to exact arithmetic as the old one?

On one minibatch of a job saved by abplus_bench_learner.py (its records, the learner's weights before it), the gradient
of a PPO-shaped loss (log-probability of the stored actions x a fixed random advantage, value error, entropy) is
computed three ways: the old path (TokPolicy packed, scaled_dot_product_attention), the fast path (learner_plan's
entity / door packing and attention buckets, float32 matrix-product attention, the layer norms' column sums,
channels-last convolutions, the CNNs on the distinct grids) and the old path in float64 (the reference: the model and
the float inputs in double precision). Printed per parameter tensor and overall: |g - g64| / |g64| for old and fast,
for each part of the fast path alone, and for the old path with every weight moved by about one float32 rounding step
(the gradient's own sensitivity). Both float32 paths in strict float32 (TF32 off; --tf32 1: TF32 products, as in
training). A fast path as close to float64 as the old one computes the same function; the differences between the two
float32 paths are then the conditioning of the gradient, not a change of mathematics.

usage: python abplus_check_learner_fp64.py --job runs/b18/job-w48.pkl --checkpoint runs/c67-last.pt [--rows 4096]
       [--seed 0] [--tf32 0]
"""
import argparse
import pickle

import numpy as np
import torch

from isaac_bridge.tok_policy import (TokPolicy, learner_plan, packed_index, plan_tokens, rows_to_device,
                                     used_entities)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--checkpoint', required=True, help='for the model config (the weights come from the job)')
    p.add_argument('--rows', type=int, default=4096)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tf32', type=int, default=0, help='1: the float32 paths with TF32 products (training default)')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    dev = torch.device('cuda')
    with open(a.job, 'rb') as f:
        job = pickle.load(f)
    cfg = torch.load(a.checkpoint, map_location='cpu')['config']
    model = TokPolicy(cfg['width'], cfg['layers'], cfg['heads'], items=cfg.get('items', False),
                      charge=cfg.get('charge', False), choice=cfg.get('choice', 0), ent_ext=cfg.get('ent_ext', False),
                      pchar=cfg.get('pchar', False)).to(dev)
    model.load_state_dict(job['learner']['model'])
    r = job['roll']
    nh = r['nh']
    flat = np.flatnonzero((np.arange(r['rows'].shape[1])[None] < r['length'][:, None]).reshape(-1))
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(flat, a.rows, replace=False))
    rows = r['rows'].reshape(-1)
    b = rows_to_device(rows, idx, r['pending'].reshape(-1, nh)[idx], dev)
    act = torch.from_numpy(r['actions'].reshape(-1, nh)[idx]).to(dev)
    adv = torch.from_numpy(rng.standard_normal(a.rows).astype(np.float32)).to(dev)
    ret = torch.from_numpy(rng.standard_normal(a.rows).astype(np.float32)).to(dev)
    ne, nd = rows['n_ent'][idx].astype(np.int64), rows['n_doors'][idx].astype(np.int64)
    used = min(b['ent'].shape[1], used_entities(ne))
    mb = {k: (v[:, :used] if k in ('ent', 'ent_id', 'ent_item') else v).contiguous() for k, v in b.items()}
    keep, starts = packed_index(ne, nd, used)
    mb['keep'], mb['starts'] = torch.from_numpy(keep).to(dev), torch.from_numpy(starts).to(dev)
    print('rows', a.rows, 'entity tokens', used, 'real tokens', len(keep), flush=True)

    def grads(model_, batch, math, extra):
        model_.zero_grad(set_to_none=True)
        model_.set_math_attention(math)
        bb = dict(batch, **extra)
        logits, value, _ = model_(bb, packed=True)
        logp, ent = model_.log_prob(logits, model_.gate(bb), act)
        loss = -(logp * adv.to(logp.dtype)).mean() + 0.5 * ((value - ret.to(value.dtype)) ** 2).mean() - \
            0.01 * ent.sum(-1).mean()
        loss.backward()
        return float(loss), {n: q.grad.detach().double().clone() for n, q in model_.named_parameters()
                             if q.grad is not None}

    l_old, g_old = grads(model, mb, False, {})
    plan = learner_plan(ne, nd, used)
    tok = plan_tokens(plan, torch.from_numpy(plan['rec']).to(dev), torch.from_numpy(plan['base']).to(dev))
    assert (tok['keep'].cpu().numpy() == keep).all() and (tok['starts'].cpu().numpy() == starts).all()
    ekeep_np = np.flatnonzero(np.arange(used)[None] < ne[:, None])   # (the plain definitions, for the check)
    dkeep_np = np.flatnonzero(np.arange(8)[None] < np.minimum(nd, 8)[:, None])
    assert (tok['ekeep'].cpu().numpy() == ekeep_np).all() and (tok['dkeep'].cpu().numpy() == dkeep_np).all()
    extra = {k: tok[k] for k in ('ekeep', 'gpos', 'dkeep', 'xperm')}
    extra['buckets'] = [(torch.from_numpy(c).to(dev), w) for c, w in plan['buckets']]
    dedup = {}
    for k in ('grid', 'map', 'patch'):   # the distinct grids, exactly (numpy on the bytes)
        _, u, i = np.unique(np.ascontiguousarray(rows[k][idx]).reshape(len(idx), -1), axis=0, return_index=True,
                            return_inverse=True)
        dedup[k + '_u'], dedup[k + '_i'] = torch.from_numpy(u).to(dev), torch.from_numpy(i.reshape(-1)).to(dev)
        print(k, 'distinct', len(u), flush=True)
    extra.update(dedup)
    print('buckets (records, width):', [(len(c), w) for c, w in plan['buckets']], flush=True)
    l_new, g_new = grads(model, mb, True, extra)
    # the gradient's own sensitivity: the old path with every weight moved by about one float32 rounding step
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        for q in model.parameters():
            q.mul_(1 + (torch.randint(0, 2, q.shape, generator=gen, device=dev) * 2 - 1) * 2.0 ** -24)
    g_ulp = grads(model, mb, False, {})[1]
    model.load_state_dict(saved)
    parts = {'old, weights moved by 1 ulp': g_ulp,
             'math+ln+nhwc only': grads(model, mb, True, {})[1],
             'buckets only': grads(model, mb, False, {k: extra[k] for k in ('buckets', 'gpos')})[1],
             'ekeep only': grads(model, mb, False, {'ekeep': extra['ekeep']})[1],
             'dedup only': grads(model, mb, False, dedup)[1]}
    # float64 reference: the old path with the model and every float input in double (the forward's .float()
    # casts become .double() for this call)
    m64 = TokPolicy(cfg['width'], cfg['layers'], cfg['heads'], items=cfg.get('items', False),
                    charge=cfg.get('charge', False), choice=cfg.get('choice', 0), ent_ext=cfg.get('ent_ext', False),
                    pchar=cfg.get('pchar', False)).to(dev)
    m64.load_state_dict(job['learner']['model'])
    m64.double()
    mb64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in mb.items()}
    float_ = torch.Tensor.float
    torch.Tensor.float = lambda self: self.double()
    try:
        l_64, g_64 = grads(m64, mb64, False, {})
    finally:
        torch.Tensor.float = float_
    print(f'loss old {l_old:.9f} fast {l_new:.9f} float64 {l_64:.12f}', flush=True)
    rows_ = []
    for n_, g in g_64.items():
        den = float(g.norm()) or 1e-30
        rows_.append((n_, float((g_old[n_] - g).norm()) / den, float((g_new[n_] - g).norm()) / den,
                      float((g_new[n_] - g_old[n_]).norm()) / den, float(g.norm())))
    cat = {k: torch.cat([d[n_].reshape(-1) for n_ in g_64]) for k, d in (('old', g_old), ('new', g_new),
                                                                          ('64', g_64))}
    tot = float(cat['64'].norm())
    print(f'overall |g - g64| / |g64|: old {float((cat["old"] - cat["64"]).norm()) / tot:.3e} fast '
          f'{float((cat["new"] - cat["64"]).norm()) / tot:.3e} | |fast - old| / |g64| '
          f'{float((cat["new"] - cat["old"]).norm()) / tot:.3e} | parts: ' + ', '.join(
              f'{k} {float((torch.cat([d[n_].reshape(-1) for n_ in g_64]) - cat["64"]).norm()) / tot:.3e}'
              for k, d in parts.items()), flush=True)
    print('per tensor (relative to the float64 gradient): name, old, fast, fast - old, |g64|; the 12 worst for old:')
    for row in sorted(rows_, key=lambda x: -x[1])[:12]:
        print('  %-28s %.2e %.2e %.2e %.2e' % row)
    worse = [x for x in rows_ if x[2] > 2 * x[1] + 1e-9]
    print('tensors where fast is more than 2x further from float64 than old:', len(worse), 'of', len(rows_))
    for row in sorted(worse, key=lambda x: -x[2])[:12]:
        print('  %-28s %.2e %.2e %.2e %.2e' % row)


if __name__ == '__main__':
    main()
