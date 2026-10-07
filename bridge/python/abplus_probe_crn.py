"""Probe of the policy's common random numbers (Phase B2, 2026-10-07; tok_policy.crn_uniforms, GraphActor's per-row
override for records whose ROW 'crn' is not 0, TokPolicy.act(u=...)). No game is run: the records are real ones (a
run's choices-last.npz, mapped field by field into the current ROW) and the weights a checkpoint's.

  uniforms     crn_uniforms is a deterministic function of (key, t, slot): the same call twice is equal; for 2e6
               values: mean, variance, a 1000-bin chi-square and Kolmogorov-Smirnov distance against U(0, 1); the
               correlation of neighbouring slots, of t and t + 1 and of two keys.
  off          with no record carrying a key, the new GraphActor's actions, log-probabilities and values are
               bit-identical to the reference GraphActor's (--old: the module of the version before Phase B2) under
               the same torch seed, call after call; also with --crn 0 on records that carry keys.
  mixed        half of the records carry keys: the other half's outputs are still bit-identical to the reference's
               (u.uniform_ draws what it drew); the keyed records' actions are the same in two calls under different
               torch seeds (they depend on (key, t) only) and change with the key.
  eager        TokPolicy.act(u=crn uniforms) gives the GraphActor's actions for the keyed records (share equal).
  logp         the stored log-probability is the policy's log-probability of the sampled action: GraphActor's logp
               against TokPolicy.evaluate on the same actions (max abs difference).
  distribution for --draws keys per record, the empirical frequency of each head's actions against the policy's
               probabilities (total variation, and a chi-square per head summed over records), next to the same with
               fresh uniforms (the old sampler): the keyed sampler draws from the same distribution.

usage (PYTHONPATH=. from the bridge's python dir, host with CUDA):
  python abplus_probe_crn.py --init <checkpoint.pt> --records <run>/choices-last.npz --old <old tok_policy.py> --items
"""
import argparse
import importlib.util
import json
import math
import sys

import numpy as np
import torch

from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_policy import GraphActor, TokPolicy, crn_uniforms, load_compatible, to_batch


def load_old(path):
    spec = importlib.util.spec_from_file_location('isaac_bridge.tok_policy_ref', path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'isaac_bridge'
    sys.modules['isaac_bridge.tok_policy_ref'] = mod
    spec.loader.exec_module(mod)
    return mod


def records(path, n):
    z = np.load(path)
    old = z['rows']
    rows = np.zeros(len(old), ROW)
    for name in old.dtype.names:
        if name in ROW.names:
            rows[name] = old[name]
    rows['point'], rows['branch'], rows['crn'] = 0, 0, 0
    reps = -(-n // len(rows))
    rows = np.concatenate([rows] * reps)[:n].copy()
    rows['t'] = np.arange(n) % 500
    return rows


def uniform_tests():
    out = {}
    keys = np.arange(1, 2001, dtype=np.int64) * 7919 + 3
    t = np.arange(2000) % 450
    u = crn_uniforms(keys, t, 1000).astype(np.float64)   # 2e6 values
    out['deterministic'] = bool(np.array_equal(crn_uniforms(keys, t, 1000), crn_uniforms(keys, t, 1000)))
    x = u.reshape(-1)
    out['n'] = len(x)
    out['min'], out['max'] = float(x.min()), float(x.max())
    out['mean'], out['var'] = float(x.mean()), float(x.var())   # 0.5, 1/12 = 0.08333
    counts = np.bincount(np.minimum((x * 1000).astype(int), 999), minlength=1000)
    e = len(x) / 1000
    out['chi2_1000bins'] = float(((counts - e) ** 2 / e).sum())   # ~ 999 +- 45
    xs = np.sort(x)
    out['ks_d'] = float(np.max(np.abs(xs - (np.arange(1, len(xs) + 1) - 0.5) / len(xs))))   # ~ 1/sqrt(n) scale
    out['ks_d_crit_1pct'] = 1.63 / math.sqrt(len(xs))
    out['corr_slot_neighbours'] = float(np.corrcoef(u[:, :-1].reshape(-1), u[:, 1:].reshape(-1))[0, 1])
    a = crn_uniforms(np.full(2000, 12345), np.arange(2000), 16).astype(np.float64)
    out['corr_t_neighbours'] = float(np.corrcoef(a[:-1].reshape(-1), a[1:].reshape(-1))[0, 1])
    b = crn_uniforms(np.full(2000, 12346), np.arange(2000), 16).astype(np.float64)
    out['corr_keys_k_k+1'] = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
    out['corr_crit_1pct'] = 2.58 / math.sqrt(a.size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', required=True)
    p.add_argument('--records', required=True)
    p.add_argument('--old', required=True, help='tok_policy.py of the version before Phase B2')
    p.add_argument('--items', action='store_true')
    p.add_argument('--charge', type=int, default=1)
    p.add_argument('--choice', type=int, default=4)
    p.add_argument('--rows', type=int, default=16)
    p.add_argument('--calls', type=int, default=200)
    p.add_argument('--draws', type=int, default=4000)
    p.add_argument('--out', default='')
    args = p.parse_args()
    res = dict(uniforms=uniform_tests())
    print(json.dumps(res['uniforms']), flush=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    state = torch.load(args.init, map_location=dev)['model']
    old_mod = load_old(args.old)
    ref_model = old_mod.TokPolicy(192, 4, 4, items=args.items, charge=bool(args.charge)).to(dev).eval()
    load_compatible(ref_model, state)
    model = TokPolicy(192, 4, 4, items=args.items, charge=bool(args.charge), choice=args.choice).to(dev).eval()
    load_compatible(model, state)
    with torch.no_grad():   # the same weights (the choice heads aside)
        for (na, a), (nb, b) in zip(model.named_parameters(), ref_model.named_parameters()):
            assert na == nb
            a.copy_(b)
    nh = len(model.heads)
    rows = records(args.records, args.rows)
    pend = np.zeros((len(rows), nh), np.int64)
    ref = old_mod.GraphActor(ref_model, args.rows, 32)
    new = GraphActor(model, args.rows, 32)
    new_off = GraphActor(model, args.rows, 32, crn=False)

    def calls(actor, rows_, seed, k):
        torch.manual_seed(seed)
        outs = []
        for _ in range(k):
            a, lp, v = actor.act(rows_, pend)
            outs.append((a, lp, v, None if actor.__class__ is not GraphActor or actor.extra is None
                         else actor.extra.copy()))
        return outs

    def same(x, y):
        return all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
                   for a, b in zip(x, y))
    # off: no keys
    r_ref = calls(ref, rows, 5, args.calls)
    r_new = calls(new, rows, 5, args.calls)
    res['off_bit_identical'] = same(r_ref, r_new)
    keyed = rows.copy()
    keyed['crn'] = np.arange(1, len(rows) + 1) * 1000003
    res['off_crn0_on_keyed_bit_identical'] = same(r_ref, calls(new_off, keyed, 5, args.calls))
    # mixed: half the records keyed
    mixed = rows.copy()
    half = np.arange(len(rows)) % 2 == 1
    mixed['crn'][half] = keyed['crn'][half]
    r_mix = calls(new, mixed, 5, args.calls)
    res['mixed_unkeyed_bit_identical'] = all(
        np.array_equal(a[0][~half], b[0][~half]) and np.array_equal(a[1][~half], b[1][~half])
        for a, b in zip(r_ref, r_mix))
    r_mix2 = calls(new, mixed, 99, args.calls)
    res['mixed_keyed_same_other_seed'] = all(np.array_equal(a[0][half], b[0][half]) for a, b in zip(r_mix, r_mix2))
    mixed2 = mixed.copy()
    mixed2['crn'][half] += 1
    r_mix3 = calls(new, mixed2, 5, args.calls)
    res['mixed_keyed_other_key_share_equal'] = float(np.mean([np.mean(a[0][half] == b[0][half])
                                                             for a, b in zip(r_mix, r_mix3)]))
    # eager with the same uniforms
    b = to_batch(keyed, pend, dev)
    u = torch.from_numpy(crn_uniforms(keyed['crn'], keyed['t'], sum(model.heads))).to(dev)
    with torch.no_grad():
        a_e, lp_e, v_e = model.act(b, u=u)
    a_g, lp_g, v_g = new.act(keyed, pend)
    res['eager_vs_graph_actions_equal_share'] = float((a_e.cpu().numpy() == a_g).all(1).mean())
    res['eager_vs_graph_logp_max_abs'] = float(np.abs(lp_e.cpu().numpy() - lp_g).max())
    # logp correctness
    with torch.no_grad():
        lp_ev, _, v_ev, _ = model.evaluate(b, torch.from_numpy(a_g).to(dev))
    res['logp_vs_evaluate_max_abs'] = float(np.abs(lp_ev.cpu().numpy() - lp_g).max())
    res['value_vs_evaluate_max_abs'] = float(np.abs(v_ev.cpu().numpy() - v_g).max())
    # distribution: per record, many keys vs fresh uniforms
    with torch.no_grad():
        logits = model(b)[0]
        probs = [torch.softmax(h, -1).cpu().numpy() for h in model.split(logits.float(), model.gate(b))]
    cnt_k = [np.zeros_like(pr) for pr in probs]
    cnt_f = [np.zeros_like(pr) for pr in probs]
    torch.manual_seed(1)
    for d in range(args.draws):
        kk = rows.copy()
        kk['crn'] = (np.arange(len(rows)) + 1) * 7777777 + d * 131 + 1
        a_k, _, _ = new.act(kk, pend)
        a_f, _, _ = new.act(rows, pend)
        for h in range(nh):
            np.add.at(cnt_k[h], (np.arange(len(rows)), a_k[:, h]), 1)
            np.add.at(cnt_f[h], (np.arange(len(rows)), a_f[:, h]), 1)
    dist = {}
    for name, cnt in (('keyed', cnt_k), ('fresh', cnt_f)):
        tv, chi2, dof = [], 0.0, 0
        for h in range(nh):
            f = cnt[h] / args.draws
            tv.append(float(0.5 * np.abs(f - probs[h]).sum(1).mean()))
            e = probs[h] * args.draws
            m = e > 5
            chi2 += float(((cnt[h][m] - e[m]) ** 2 / e[m]).sum())
            dof += int(m.sum() - m.any(1).sum())
        dist[name] = dict(tv_per_head=tv, chi2=round(chi2, 1), dof=dof)
    res['distribution'] = dist
    res['crn_rows_counted'] = new.crn_rows
    print(json.dumps(res, indent=1), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
