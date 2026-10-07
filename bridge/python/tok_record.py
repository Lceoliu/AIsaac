"""Episode recorder of eval_tok.py --record: every ROW record (isaac_bridge/tok_obs.py) of every episode the server
receives, and the action the policy decided at each of them, saved as one compressed npz per episode.

The records are the policy's observation as the worker wrote it (the lag-1 protocol: eval_tok copies the record out
of its shared slot before replying, so none is overwritten before it is saved). The action decided at record t is
applied during the step t+1 -> t+2 (the one-decision action lag); the step t -> t+1 runs the one decided at t-1.

Per episode file ep<k>_<seed>.npz:
  player (T, 31) float32, the ROW's player features; t, hurt, damage, done, first, bombs, events (T,)
  n_ent (T,) and ent (N, 33) float32 / ent_id (N, 3) int16: the first n_ent entity rows of every record, concatenated
  n_doors (T,), doors (T, 8, 7) float32
  grids (G, 7, 16, 28) uint8 with grid_idx (T,): the room grids, each distinct one once
  maps (M, 8, 13, 13) uint8 with map_idx (T,): the minimaps, each distinct one once
  act (T, 3) int16: move, shoot, bomb decided at that record (-1: none, the episode's last record); with items
      (2026-10-06) (T, 5): + item, pill, and the item fields ent_item (N, 2), inv (T, 16, 2), pitem (T, 6), pinv
      (T, 12); stage (T,) with items or in run mode
  meta: JSON (the episode's outcome as eval_tok reports it, the run's settings)
index.json lists the files with their outcomes. An episode whose records stop without an end (a worker error) is
saved with done -1 ('aborted').
"""
import json
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, directory, workers, run, n_heads=3):
        self.nh = n_heads
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.buf = [[] for _ in range(workers)]
        self.acts = [[] for _ in range(workers)]
        self.index = []
        (self.dir / 'run.json').write_text(json.dumps(run, indent=1, default=str))

    def add(self, i, r):
        if r['first'] and self.buf[i]:
            self._save(i, None)
        self.buf[i].append(np.array(r, copy=True))
        self.acts[i].append((-1,) * self.nh)

    def act(self, i, action):
        if self.acts[i]:
            self.acts[i][-1] = tuple(int(v) for v in action[:self.nh])

    def finish(self, i, episode):
        self._save(i, episode)

    def close(self):
        for i in range(len(self.buf)):
            if self.buf[i]:
                self._save(i, None)
        (self.dir / 'index.json').write_text(json.dumps(self.index, indent=1))

    def _save(self, i, episode):
        rows = np.stack(self.buf[i])
        acts = np.array(self.acts[i], np.int16)
        self.buf[i], self.acts[i] = [], []
        seed = int(rows['seed'][0])
        if episode is None:
            episode = dict(seed=seed, done=-1, decisions=int(rows['t'][-1]), hurt=float(rows['hurt'].sum()),
                           aborted=True)
        n = rows['n_ent'].astype(np.int64)
        ent = np.concatenate([rows['ent'][k, :n[k]] for k in range(len(rows))]) if n.sum() else \
            np.zeros((0, rows['ent'].shape[-1]), np.float32)
        ent_id = np.concatenate([rows['ent_id'][k, :n[k]] for k in range(len(rows))]) if n.sum() else \
            np.zeros((0, 3), np.int16)

        def dedup(a):
            keys, uniq, idx = {}, [], np.zeros(len(a), np.int16)
            for k in range(len(a)):
                b = a[k].tobytes()
                if b not in keys:
                    keys[b] = len(uniq)
                    uniq.append(a[k])
                idx[k] = keys[b]
            return np.stack(uniq), idx

        more = {}
        if self.nh > 3:   # items
            more = dict(ent_item=np.concatenate([rows['ent_item'][k, :n[k]] for k in range(len(rows))]) if n.sum()
                        else np.zeros((0, 2), np.int16), inv=rows['inv'], pitem=rows['pitem'], pinv=rows['pinv'])
        if self.nh > 3 or self.run.get('mode') == 'run':
            more['stage'] = rows['stage']
        grids, grid_idx = dedup(rows['grid'])
        maps, map_idx = dedup(rows['map'])
        name = f'ep{len(self.index):03d}_{seed}.npz'
        meta = dict(episode, worker=i, records=len(rows))
        np.savez_compressed(self.dir / name, player=rows['player'], t=rows['t'], hurt=rows['hurt'],
                            damage=rows['damage'], done=rows['done'], first=rows['first'], bombs=rows['bombs'],
                            events=rows['events'], n_ent=n.astype(np.int16), ent=ent, ent_id=ent_id,
                            n_doors=rows['n_doors'].astype(np.int8), doors=rows['doors'], grids=grids,
                            grid_idx=grid_idx, maps=maps, map_idx=map_idx, act=acts,
                            meta=np.array(json.dumps(meta)), **more)
        self.index.append(dict(meta, file=name))
