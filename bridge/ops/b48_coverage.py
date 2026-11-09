"""B48 (read only, CPU): history-gradient coverage of one real update's learner inputs (learner_dump).

  python b48_coverage.py <learner_inputs.pt> [--draws 20] [--json out.json]

Over the update's grad calls (the PPO parts), per chunk: length bucket, blocks newly completed inside the chunk,
bank blocks visible (n_old), window-external (eligible: end <= step - window), selected for re-encoding in
'random' mode (min(history_recompute_blocks, eligible), a uniform draw). Coverage: share of the bank entries
the forward carries a gradient for (new blocks + selected historical ones); selection probability by block age
(blocks between the block's end and the chunk's start); blocks re-encoded at least once in the update (Monte
Carlo over the draws, and the exact expectation) out of the blocks the update's chunks reference; how often one
block is re-encoded in the update (by distinct chunks)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
import numpy as np

from isaac_bridge import learner_dump
from isaac_bridge.block_learner import chunk_bucket


def hist(values, bins):
    v = np.asarray(values)
    return {str(b): int((v == b).sum()) for b in bins} | ({f'>{bins[-1]}': int((v > bins[-1]).sum())} if len(v) else {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--draws', type=int, default=20)
    p.add_argument('--json', default='')
    a = p.parse_args()
    ck, store, chunks, calls, extra = learner_dump.load(a.path)
    cfg = store.config
    bs, window = cfg.block_steps, cfg.window
    take = int(getattr(cfg, 'history_recompute_blocks', 4))
    grad_calls = [c for c in calls if c['grad']]
    rows = []   # per (call, chunk): bucket, new, n_old, eligible, selected, step, bids, ends
    for ci, c in enumerate(grad_calls):
        train_history = c['kwargs'].get('train_history', True)
        for ch in c['chunks']:
            n = len(ch.rows)
            if ch.rows['first'][0]:
                part = dict(step=0, pdec=np.empty(0), bid=np.empty(0, np.int64), ends=np.empty(0, np.int64))
            else:
                part = store.snapshot_parts(ch.snapshot)
            tail = len(part['pdec'])
            new = (tail+n)//bs
            e = np.flatnonzero(part['ends'] <= part['step']-window)
            sel = min(take, len(e)) if train_history else 0
            rows.append(dict(call=ci, bucket=chunk_bucket(n), n=n, new=int(new), n_old=len(part['bid']),
                             eligible=len(e), selected=sel, step=int(part['step']), bids=part['bid'], ends=part['ends'],
                             elig=e))
    out = dict(grad_calls=len(grad_calls), chunks=len(rows), history_recompute_blocks=take, window=window,
               block_steps=bs)
    out['chunk_length_bucket'] = hist([r['bucket'] for r in rows], [1, 32, 64, 128])
    out['new_blocks_per_chunk'] = hist([r['new'] for r in rows], [0, 1, 2, 3, 4])
    q = lambda k: {s: float(np.percentile([r[k] for r in rows], s)) for s in (0, 25, 50, 75, 90, 100)}
    out['n_old_percentiles'] = q('n_old')
    out['eligible_percentiles'] = q('eligible')
    out['eligible_zero_chunks'] = int(sum(r['eligible'] == 0 for r in rows))
    out['selected_per_chunk'] = hist([r['selected'] for r in rows], [0, 1, 2, 3, 4])
    tot_old = sum(r['n_old'] for r in rows); tot_new = sum(r['new'] for r in rows)
    tot_sel = sum(r['selected'] for r in rows); tot_elig = sum(r['eligible'] for r in rows)
    out['bank_entries'] = dict(old=tot_old, new=tot_new, eligible=tot_elig, selected=tot_sel,
                               inside_window_from_pages=tot_old-tot_elig)
    out['share_with_gradient'] = round((tot_sel+tot_new)/max(tot_old+tot_new, 1), 4)
    out['share_of_old_with_gradient'] = round(tot_sel/max(tot_old, 1), 4)
    # selection probability by age (blocks from the block's end to the chunk's start)
    edges = [0, 4, 8, 16, 32, 64, 128, 256]
    num = np.zeros(len(edges)); den = np.zeros(len(edges))
    for r in rows:
        if not len(r['ends']):
            continue
        age = (r['step']-r['ends'])//bs
        p_sel = np.zeros(len(age))
        if r['eligible']:
            p_sel[r['elig']] = r['selected']/r['eligible']
        b = np.searchsorted(edges, age, side='right')-1
        np.add.at(num, b, p_sel); np.add.at(den, b, 1)
    out['selection_probability_by_age_blocks'] = {f'{edges[i]}-{edges[i+1]-1 if i+1 < len(edges) else "+"}':
                                                  dict(blocks=int(den[i]), p=round(num[i]/den[i], 4) if den[i] else None)
                                                  for i in range(len(edges))}
    # blocks referenced by the update's chunks; covered at least once (expectation and draws)
    ref = set()
    miss = {}   # bid -> prod(1 - p) over the chunks able to select it
    for r in rows:
        ref.update(r['bids'].tolist())
        if r['eligible'] and r['selected']:
            p_ = r['selected']/r['eligible']
            for b in r['bids'][r['elig']].tolist():
                miss[b] = miss.get(b, 1.0)*(1-p_)
    expect = sum(1-v for v in miss.values())
    rng = np.random.default_rng(0)
    covered, times = [], {}
    for d in range(a.draws):
        seen = {}
        for r in rows:
            if r['eligible'] and r['selected']:
                pick = rng.permutation(r['eligible'])[:r['selected']]
                for b in r['bids'][r['elig'][pick]].tolist():
                    seen[b] = seen.get(b, 0)+1
        covered.append(len(seen))
        if d == 0:
            times = seen
    out['blocks_referenced'] = len(ref)
    out['blocks_eligible_somewhere'] = len(miss)
    out['blocks_covered_expected'] = round(expect, 1)
    out['blocks_covered_draws'] = dict(mean=float(np.mean(covered)), min=int(min(covered)), max=int(max(covered)))
    out['covered_share_of_referenced'] = round(float(np.mean(covered))/max(len(ref), 1), 4)
    out['covered_share_of_eligible'] = round(float(np.mean(covered))/max(len(miss), 1), 4)
    t = np.asarray(list(times.values()))
    out['reencodes_per_covered_block'] = hist(t, [1, 2, 3, 4, 5]) if len(t) else {}
    out['reencodes_total'] = int(t.sum()) if len(t) else 0
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
