"""load_compatible with the charge inputs (2026-10-07): a checkpoint without them, loaded into TokPolicy(charge=True)
(and items + charge) with tok_policy.load_compatible, must give the same move / shoot / bomb logits, value and danger
logits as the checkpoint's own model, whatever ROW 'pcharge' holds (its new input columns start at zero). CPU only.

usage (from the bridge's python dir, PYTHONPATH=.): python abplus_probe_charge_compat.py <checkpoint.pt> [n]
"""
import json
import sys

import numpy as np
import torch

from isaac_bridge.tok_obs import ENT_CAP, ROW
from isaac_bridge.tok_policy import HEADS, TokPolicy, load_compatible, to_batch


def random_rows(n, rng):
    rows = np.zeros(n, ROW)
    rows['player'] = rng.uniform(-1, 1, rows['player'].shape)
    rows['ent'] = rng.uniform(-1, 1, rows['ent'].shape)
    rows['ent_id'] = rng.integers(0, 256, rows['ent_id'].shape)
    rows['doors'] = rng.uniform(-1, 1, rows['doors'].shape)
    rows['patch'] = rng.integers(0, 2, rows['patch'].shape)
    rows['grid'] = rng.integers(0, 2, rows['grid'].shape)
    rows['map'] = rng.integers(0, 2, rows['map'].shape)
    rows['n_ent'] = rng.integers(1, ENT_CAP, n)
    rows['n_doors'] = rng.integers(0, 8, n)
    rows['pcharge'] = rng.uniform(0, 1, rows['pcharge'].shape)
    return rows


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    torch.manual_seed(0)
    ck = torch.load(path, map_location='cpu')
    config = dict(ck['config'])
    ref = TokPolicy(**config).eval()
    ref.load_state_dict(ck['model'])
    rng = np.random.default_rng(0)
    rows = random_rows(n, rng)
    pending = np.stack([rng.integers(0, k, n) for k in HEADS], 1)
    out = {}
    with torch.no_grad():
        b = to_batch(rows, pending, 'cpu')
        r_logits, r_value, r_danger = ref(b)
        for name, extra in (('charge', dict(charge=True)), ('items', dict(items=True)),
                            ('items+charge', dict(items=True, charge=True))):
            model = TokPolicy(**dict(config, **extra)).eval()
            changed = load_compatible(model, ck['model'])
            pend = pending if not extra.get('items') else np.concatenate([pending, np.zeros((n, 2), np.int64)], 1)
            bb = to_batch(rows, pend, 'cpu')
            logits, value, danger = model(bb)
            k = sum(HEADS)
            out[name] = dict(changed=changed, config=model.config,
                             logits_max_abs=float((logits[:, :k] - r_logits[:, :k]).abs().max()),
                             value_max_abs=float((value - r_value).abs().max()),
                             danger_max_abs=float((danger - r_danger).abs().max()),
                             logits_equal=bool(torch.equal(logits[:, :k], r_logits[:, :k])),
                             value_equal=bool(torch.equal(value, r_value)))
            # the pcharge columns are really read: zero them in the batch, outputs must stay the same; set the new
            # columns' weights nonzero, outputs must move
            if not extra.get('charge'):
                continue
            w = model.player[0].weight
            with torch.no_grad():
                w[:, -12:] = 0.01
            moved = float((model(bb)[0][:, :k] - logits[:, :k]).abs().max())
            out[name]['moves_when_trained'] = moved
    print(json.dumps(dict(checkpoint=path, records=n, config=config, results=out), indent=1))


if __name__ == '__main__':
    main()
