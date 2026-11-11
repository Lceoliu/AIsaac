"""BM1 B (static-shape learner parts): how much padding every dimension to its maximum would waste, counted offline on a
dumped update (ISAAC_LEARNER_DUMP's learner_inputs.pt: its grad parts), CPU only (no GPU work).

  python b66_static_waste.py <learner_inputs.pt> [--draws 3]

Each grad part is prepared as the learner prepares it (block_learner._prepare_packed, host packing; the history draw is
random: --draws repeats per part with different seeds). Per dimension, the static cap is the maximum over every part and
draw; the waste is (cap - used) summed over the parts, weighted by its cost proxy:
  rows       main rows (+ the teachers' records: in the dump's chunks) of the part
  spatial    rows x token bucket T per Spatial group (T_BUCKETS): cap = per T, the most rows any part had in it
  hist       historical blocks re-encoded (Spatial + Writer, block_steps rows each)
  writer     Writer inputs: historical + new blocks
  reader     Reader rows Bb x T per chunk-length bucket (T_LEARN), and the bank / prefix widths max_m / Pd per bucket
             (attention cost ~ Bb x T x (T + max_m + Pd x latents))
Prints one JSON line per part and a summary (used, cap, waste share per dimension)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get('TREE_PY') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch

from isaac_bridge import block_learner, learner_dump
from isaac_bridge.tok_policy import TokPolicy


def shapes(P):
    cfg = P['cfg']
    out = dict(rows=int(P['n_main']), hist=int(len(P['ub'])), new=int(P['n_new']), tail=int(P['n_tail']),
               old=int(P['n_old_total']), spatial={}, reader={})
    for g, names, T, m in P['plans']:
        out['spatial'][int(T)] = out['spatial'].get(int(T), 0)+int(m)
    for T, ids, Bb, max_m, Pd, _ in P['rbuckets']:
        out['reader'][int(T)] = dict(Bb=int(Bb), real=int(len(ids)), max_m=int(max_m), Pd=int(Pd),
                                     cost=int(Bb)*int(T)*(int(T)+int(max_m)+int(Pd)*cfg.latents))
    out['writer'] = out['hist']+out['new']
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('path')
    a.add_argument('--draws', type=int, default=3)
    args = a.parse_args()
    ck, store, chunks, calls, extra = learner_dump.load(args.path)
    gcalls = [c for c in calls if c['grad']]
    model = TokPolicy(**ck['config'])
    model.load_state_dict(ck['model'])
    keys = [k for k in block_learner._prepare_packed.__code__.co_varnames
            if k in ('generator', 'train_history', 'version', 'spatial_micro', 'require_next_state',
                     'deduplicate_sources', 'history_mode', 'max_rows')]
    S = []
    for i, c in enumerate(gcalls):
        kw = {k: v for k, v in c['kwargs'].items() if k in keys}
        for d in range(args.draws):
            torch.manual_seed(1000*i+d)
            with block_learner.PREP_LOCK:
                P = block_learner._prepare_packed(model, c['chunks'], True, packed=True, **kw)
            s = shapes(P)
            S.append(s)
            print(json.dumps(dict(part=i, draw=d, **s)), flush=True)
            if P['slab'].cuda:
                block_learner.STAGING.drop(P['slab'].k)
    # caps: per scalar dimension the max; per Spatial T the max rows; per Reader T the max Bb, max_m, Pd
    out = {}
    for k in ('rows', 'hist', 'writer', 'old', 'tail'):
        used = [s[k] for s in S]
        cap = max(used)
        out[k] = dict(mean=round(float(np.mean(used)), 1), cap=cap,
                      waste=round(1-float(np.sum(used))/(cap*len(used)), 3) if cap else 0.0)
    Ts = sorted({T for s in S for T in s['spatial']})
    capT = {T: max(s['spatial'].get(T, 0) for s in S) for T in Ts}
    used_sp = sum(T*s['spatial'].get(T, 0) for s in S for T in Ts)
    cap_sp = len(S)*sum(T*capT[T] for T in Ts)
    out['spatial'] = dict(per_T={T: dict(cap_rows=capT[T], mean_rows=round(float(np.mean([s['spatial'].get(T, 0)
                                                                                           for s in S])), 1))
                                 for T in Ts},
                          waste=round(1-used_sp/max(cap_sp, 1), 3))
    Rs = sorted({T for s in S for T in s['reader']})
    cfg = model.block_memory.config
    capR = {T: dict(Bb=max(s['reader'].get(T, {}).get('Bb', 0) for s in S),
                    max_m=max(s['reader'].get(T, {}).get('max_m', 0) for s in S),
                    Pd=max(s['reader'].get(T, {}).get('Pd', 0) for s in S)) for T in Rs}
    used_r = sum(s['reader'][T]['cost'] for s in S for T in s['reader'])
    cap_r = len(S)*sum(capR[T]['Bb']*T*(T+capR[T]['max_m']+capR[T]['Pd']*cfg.latents) for T in Rs)
    used_rr = sum(s['reader'][T]['real']*T for s in S for T in s['reader'])
    out['reader'] = dict(per_T={T: dict(cap=capR[T], mean_Bb=round(float(np.mean([s['reader'].get(T, {}).get('Bb', 0)
                                                                                    for s in S])), 1))
                                for T in Rs},
                         waste_cost=round(1-used_r/max(cap_r, 1), 3),
                         waste_rows=round(1-used_rr/max(len(S)*sum(capR[T]['Bb']*T for T in Rs), 1), 3))
    print(json.dumps(dict(summary=out, parts=len(gcalls), draws=args.draws)), flush=True)


if __name__ == '__main__':
    main()
