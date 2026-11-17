"""C110 gate arm (2026-11-15): use actions in dumped rollouts (train_tok --dump-rollouts). c110_usecount.py <rollouts dir>
Prints per file the arrays' shapes and, for 'actions' ([..., 5]: move, shoot, bomb, item, pill), the decisions whose
bomb / item / pill action is not 0 (with ISAAC_BLOCK_GATE=closed all must be 0), forced ones (rows pforce > 0) apart."""
import glob
import sys

import numpy as np

tot = np.zeros(3, np.int64); n = 0
for f in sorted(glob.glob(sys.argv[1] + '/*.npz')):
    z = np.load(f, allow_pickle=False)
    print(f, {k: z[k].shape for k in z.files if k != 'rows'})
    a = z['actions'].reshape(-1, z['actions'].shape[-1])
    c = (a[:, 2:5] != 0).sum(0)
    if 'rows' in z.files and 'pforce' in z['rows'].dtype.names:
        pf = z['rows']['pforce'].reshape(-1)[:len(a)] > 0
        print('  forced decisions', int(pf.sum()), 'use actions among them', (a[pf, 2:5] != 0).sum(0).tolist())
    print('  decisions', len(a), 'use actions (bomb, item, pill)', c.tolist())
    tot += c; n += len(a)
print('TOTAL decisions', n, 'use actions (bomb, item, pill)', tot.tolist())
