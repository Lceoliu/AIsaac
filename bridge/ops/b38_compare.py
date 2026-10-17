"""B38 step 3a: the weights after one replayed update (abplus_b38_replay.py, hpc_b38.sh replay) compared between
configurations. usage: python b38_compare.py <runs dir> <the replays' starting checkpoint>"""
import sys

import torch

R, CK = sys.argv[1], sys.argv[2]
ck = {t: torch.load(f'{R}/b38-eq-{t}/checkpoints/last.pt', map_location='cpu', weights_only=False)['model']
      for t in ('1a', '1b', '2', '1c')}
base = torch.load(CK, map_location='cpu', weights_only=False)['model']


def d(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a if a[k].is_floating_point())


print(f"one update from the same job and weights: |1 rank micro 2048 - again| {d(ck['1a'], ck['1b']):.3g}, "
      f"|2 ranks - 1 rank micro 2048| {d(ck['2'], ck['1a']):.3g}, |2 ranks - 1 rank micro 4096| "
      f"{d(ck['2'], ck['1c']):.3g}, |1 rank micro 2048 - micro 4096| {d(ck['1a'], ck['1c']):.3g}; the update moved "
      f"the weights by up to {d(ck['1a'], base):.3g}")
