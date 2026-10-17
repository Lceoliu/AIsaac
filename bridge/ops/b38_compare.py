"""B38 step 3a: the update one replayed job makes (abplus_b38_replay.py, hpc_b38.sh replay) compared between
configurations: per pair the relative distance ||delta_a - delta_b|| / ||delta_a|| and the cosine of the two
updates (delta = the weights after the update - the starting checkpoint's), next to the same numbers for two runs of
one configuration (the GPU's own run-to-run variation). usage: python b38_compare.py <runs dir> <start checkpoint>"""
import sys

import torch

R, CK = sys.argv[1], sys.argv[2]
base = torch.load(CK, map_location='cpu', weights_only=False)['model']
keys = [k for k, v in base.items() if v.is_floating_point()]


def delta(tag):
    m = torch.load(f'{R}/b38-eq-{tag}/checkpoints/last.pt', map_location='cpu', weights_only=False)['model']
    return torch.cat([(m[k].double() - base[k].double()).reshape(-1) for k in keys])


d = {t: delta(t) for t in ('1a', '1b', '2', '1c')}


def cmp(a, b):
    rel = float((d[a] - d[b]).norm() / d[a].norm())
    cos = float(d[a] @ d[b] / (d[a].norm() * d[b].norm()))
    return f'rel {rel:.3g}, cos {cos:.5f}'


print(f"update norm {float(d['1a'].norm()):.4g}; 1 rank micro 2048 vs again: {cmp('1a', '1b')}; 2 ranks (micro 4096 "
      f"= 2 parts of 2048) vs 1 rank micro 2048: {cmp('2', '1a')}; 2 ranks vs 1 rank micro 4096: {cmp('2', '1c')}; "
      f"1 rank micro 2048 vs 4096: {cmp('1a', '1c')}")
