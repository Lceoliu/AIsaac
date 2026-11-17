"""C108 D1: are two checkpoints' weights bit-identical? ckpt_same.py <a.pt> <b.pt> ... (pairs). Every tensor of every
state_dict-like entry (a dict of tensors: model, the critic, the aux head, ...) compared with torch.equal; optimiser
states are listed apart (Adam's step counters and moments do change with lr 0)."""
import sys

import torch


def tensors(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if torch.is_tensor(v):
                out[prefix + str(k)] = v
            elif isinstance(v, dict):
                out.update(tensors(v, prefix + str(k) + '.'))
    return out


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    A = torch.load(a, map_location='cpu', weights_only=False)
    B = torch.load(b, map_location='cpu', weights_only=False)
    print(f'{a} vs {b}: keys {sorted(A)}; update {A.get("update")} -> {B.get("update")}')
    for top in sorted(set(A) & set(B)):
        ta, tb = tensors(A[top]), tensors(B[top])
        if not ta:
            continue
        opt = 'opt' in top.lower()
        diff = [k for k in ta if k in tb and (ta[k].shape != tb[k].shape or not torch.equal(ta[k], tb[k]))]
        miss = sorted(set(ta) ^ set(tb))
        n = sum(int(v.numel()) for v in ta.values())
        print(f'  {top}{" (optimiser)" if opt else ""}: {len(ta)} tensors, {n} values, differing {len(diff)}'
              f'{" e.g. " + ", ".join(diff[:4]) if diff else ""}{"; only in one: " + str(miss[:4]) if miss else ""}')
