"""B49 (measurement only): how concentrated the history gradient is, and how close history-sampling rules get to
the full first-order history gradient, on one real update's learner inputs (learner_dump, see b44_learner_bench).

  python b49_grad_stats.py <learner_inputs.pt> [--chunks 40] [--draws 8] [--json out.json] [--bf16]

Loss: the inputs carry no advantages / old log-probabilities, so the loss is b44_learner_bench.surrogate (fixed
random weights on every output: logits, value, danger, value_global), the same for every rule compared.

Part 1 (every grad call of the update, history_mode 'gradreplay'): forward with every window-external block's value
a gradient leaf (page values), surrogate backward; q_j = dL/dM_j (the leaf's .grad: summed over the call's chunks
that read block j). Per chunk: the ||q_j|| of its window-external blocks, the top-1 / 2 / 4 share of their sum
(and of their squared sum), mean ||q_j|| by block age; over the update: the share of the total held by the top
10% of (chunk, block) entries.

Part 2 (reference gradients; chunks with 16-32 window-external blocks): per chunk alone, 'gradreplay' forward +
surrogate backward gives g_base (the Spatial + Writer parameter gradient with every historical block a page value:
main rows, new blocks) and q_j. Per window-external block j, g_j = d(q_j . f_j)/d(theta) with f_j the block re-encoded
by the current Spatial + Writer. First-order proxy (q_j at the page value, f_j with the current parameters, no
second Reader pass): the full history gradient is G_full = sum_j g_j, a rule's is the sum over its chosen set
(Random4 x N/4: scaled). Compared on the Spatial + Writer parameters: cos(g_base + G_rule, g_base + G_full),
cos(G_rule, G_full), ||G_rule|| / ||G_full||; random rules over --draws draws; rows re-encoded per rule.
Rules: Detach (none), Random4, Random4xN/4, Top1+Random3 (the gradreplay arm), Top2+Random2, Top4, All.
Parameters are not updated (no optimizer step)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
sys.path.insert(0, HERE)
import numpy as np
import torch

from isaac_bridge import block_learner, block_paged as bp, learner_dump
from isaac_bridge.block_training import forward_bucketed
from b44_learner_bench import surrogate, fresh, load_model


def sw_params(model):
    mods = (model.structured_spatial, model.block_memory.resampler, model.block_memory.writer)
    return [p for m in mods for p in m.parameters() if p.requires_grad]


def flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(grads, params)])


def autocast_ctx(on):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=on)


def part1(model, calls, bf16):
    cache = {}
    per_chunk = []
    entries = []
    age_sum, age_n = {}, {}
    edges = [4, 8, 16, 32, 64, 128]
    for c in calls:
        if not c['grad']:
            continue
        kw = dict(c['kwargs']); kw['history_mode'] = 'gradreplay'; kw['train_history'] = True
        model.zero_grad(set_to_none=True)
        with autocast_ctx(bf16):
            out, ctx = forward_bucketed(model, c['chunks'], **kw)
        loss = surrogate(out, cache)
        loss.backward()
        for rp in ctx['replay']:
            g = rp['pool'].grad
            if g is None:
                continue
            norms = g.detach().float().flatten(1).norm(dim=1).cpu().numpy()
            for uidx, ages in rp['chunks']:
                if not len(uidx):
                    continue
                q = norms[uidx]
                per_chunk.append(q)
                entries += q.tolist()
                b = np.searchsorted(edges, ages, side='right')
                for bi, v in zip(b.tolist(), q.tolist()):
                    age_sum[bi] = age_sum.get(bi, 0.0)+v; age_n[bi] = age_n.get(bi, 0)+1
        del out, ctx, loss
    def share(qs, k, sq):
        v = np.sort(qs**2 if sq else qs)[::-1]
        return float(v[:k].sum()/max(v.sum(), 1e-30))
    res = dict(chunks_with_history=len(per_chunk))
    sizes = np.array([len(q) for q in per_chunk])
    for k in (1, 2, 4):
        res[f'top{k}_share_of_sum_norms'] = dict(mean=float(np.mean([share(q, k, False) for q in per_chunk])),
                                                 mean_if_eligible_gt_4=float(np.mean([share(q, k, False) for q in per_chunk if len(q) > 4])))
        res[f'top{k}_share_of_sum_sq'] = dict(mean=float(np.mean([share(q, k, True) for q in per_chunk])),
                                              mean_if_eligible_gt_4=float(np.mean([share(q, k, True) for q in per_chunk if len(q) > 4])))
    res['eligible_per_chunk_percentiles'] = {p: float(np.percentile(sizes, p)) for p in (25, 50, 75, 90, 100)}
    cv = [float(q.std()/max(q.mean(), 1e-30)) for q in per_chunk if len(q) > 4]
    res['within_chunk_cv_median'] = float(np.median(cv)) if cv else None
    names = ['4-7', '8-15', '16-31', '32-63', '64-127', '128+']
    res['mean_norm_by_age_blocks'] = {names[b-1]: dict(n=age_n[b], mean=age_sum[b]/age_n[b]) for b in sorted(age_n) if b >= 1}
    e = np.sort(np.asarray(entries))[::-1]
    top = max(1, len(e)//10)
    res['update_top10pct_share'] = dict(sum_norms=float(e[:top].sum()/e.sum()), sum_sq=float((e[:top]**2).sum()/(e**2).sum()),
                                        entries=len(e))
    return res


def part2(model, calls, n_chunks, draws, bf16, seed=0):
    rng = np.random.default_rng(seed)
    seen, pool = set(), []
    for c in calls:
        if not c['grad']:
            continue
        for ch in c['chunks']:
            if id(ch) in seen or ch.rows['first'][0]:
                continue
            seen.add(id(ch))
            p = ch.snapshot.store.snapshot_parts(ch.snapshot)
            e = int((p['ends'] <= p['step']-model.block_memory.config.window).sum())
            if 16 <= e <= 32:
                pool.append((ch, c['kwargs']))
    pick = rng.permutation(len(pool))[:n_chunks]
    params = sw_params(model)
    cache = {}
    bs = model.block_memory.config.block_steps
    rules = ('Detach', 'Random4', 'Random4xN/4', 'Top1+Random3', 'Top2+Random2', 'Top4', 'All')
    out = {r: dict(cos_total=[], cos_hist=[], norm_ratio=[], rows=[]) for r in rules}
    per_chunk_info = []
    for n_done, i in enumerate(pick.tolist()):
        ch, kw0 = pool[i]
        kw = dict(kw0); kw['history_mode'] = 'gradreplay'; kw['train_history'] = True
        model.zero_grad(set_to_none=True)
        with autocast_ctx(bf16):
            o, ctx = forward_bucketed(model, [ch], **kw)
        loss = surrogate(o, cache)
        loss.backward()
        g_base = flat([p.grad for p in params], params)
        rp = ctx['replay'][0]
        uidx, ages = rp['chunks'][0]
        q = rp['pool'].grad.detach()
        bids = rp['ubid'][uidx]
        qn = q[uidx].float().flatten(1).norm(dim=1).cpu().numpy()
        G = []
        for j, bid in enumerate(bids.tolist()):
            with autocast_ctx(bf16):
                f, _ = bp.encode_blocks(model, rp['store'], np.array([bid]), q.device, rp['spatial_micro'])
            s = (q[uidx[j]].float()*f[0].float()).sum()
            G.append(flat(torch.autograd.grad(s, params, allow_unused=True), params))
        G = torch.stack(G)   # [E, D]
        E = len(G)
        full_h = G.sum(0)
        full_t = g_base+full_h
        order = np.argsort(-qn)

        def score(rule, sel, scale=1.0):
            h = G[sel].sum(0)*scale if len(sel) else torch.zeros_like(full_h)
            t = g_base+h
            cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            out[rule]['cos_total'].append(cos(t, full_t))
            out[rule]['cos_hist'].append(cos(h, full_h) if len(sel) else 0.0)
            out[rule]['norm_ratio'].append(float(h.norm()/max(full_h.norm(), 1e-30)))
            out[rule]['rows'].append(len(sel)*bs)
        score('Detach', [])
        score('All', list(range(E)))
        score('Top4', order[:4].tolist())
        for d in range(draws):
            r4 = rng.permutation(E)[:4].tolist()
            score('Random4', r4)
            score('Random4xN/4', r4, E/4)
            rest = [j for j in rng.permutation(E).tolist() if j != order[0]][:3]
            score('Top1+Random3', [int(order[0])]+rest)
            rest2 = [j for j in rng.permutation(E).tolist() if j not in (order[0], order[1])][:2]
            score('Top2+Random2', [int(order[0]), int(order[1])]+rest2)
        per_chunk_info.append(dict(eligible=E, base_over_hist_norm=float(g_base.norm()/max(full_h.norm(), 1e-30)),
                                   top1_share_sq=float((np.sort(qn)[::-1][:1]**2).sum()/(qn**2).sum())))
        del o, ctx, loss, G
        print(json.dumps(dict(chunk=n_done, eligible=E, base_over_hist=per_chunk_info[-1]['base_over_hist_norm'])),
              flush=True)
    summary = {}
    for r, v in out.items():
        summary[r] = {k: (dict(mean=float(np.mean(x)), p10=float(np.percentile(x, 10)), p50=float(np.median(x)),
                              p90=float(np.percentile(x, 90))) if k != 'rows' else float(np.mean(x)))
                      for k, x in v.items()}
    # the random rules' draw-to-draw spread (per chunk std over the draws, averaged)
    for r in ('Random4', 'Random4xN/4', 'Top1+Random3', 'Top2+Random2'):
        x = np.asarray(out[r]['cos_hist']).reshape(len(pick), draws)
        summary[r]['cos_hist_draw_std_mean'] = float(x.std(1).mean())
        y = np.asarray(out[r]['cos_total']).reshape(len(pick), draws)
        summary[r]['cos_total_draw_std_mean'] = float(y.std(1).mean())
    eli = np.array([c['eligible'] for c in per_chunk_info])
    return dict(chunks=len(pick), candidates=len(pool), eligible_mean=float(eli.mean()),
                base_over_hist_norm_median=float(np.median([c['base_over_hist_norm'] for c in per_chunk_info])),
                rules=summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--chunks', type=int, default=40)
    p.add_argument('--draws', type=int, default=8)
    p.add_argument('--bf16', action='store_true', help='autocast bf16 as the learner (default FP32: cleaner cosines)')
    p.add_argument('--json', default='')
    a = p.parse_args()
    from isaac_bridge import block_compile
    block_compile.configure()
    block_learner.PACKED = True
    ck, store, chunks, calls, extra = learner_dump.load(a.path)
    model, _ = load_model(ck, torch.device('cuda'))
    model.train()
    fresh(calls)
    torch.manual_seed(0)
    r1 = part1(model, calls, a.bf16)
    print(json.dumps(dict(part1=r1)), flush=True)
    r2 = part2(model, calls, a.chunks, a.draws, a.bf16)
    print(json.dumps(dict(part2=r2)), flush=True)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(part1=r1, part2=r2, bf16=a.bf16, loss='b44 surrogate'), f, indent=1)


if __name__ == '__main__':
    main()
