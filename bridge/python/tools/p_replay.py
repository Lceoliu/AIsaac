"""P package: the parity replay (2026-11-19, the coordinator). Replays the records of a train_tok --dump-rollouts
directory (D1's: real records and the model) through an actor backend under a FORCED call schedule, and compares two
runs record by record: the outputs (actions, logp, value, value_global, the temporal counters), each record's K / V, s
and o (the rows a rollout keeps), and the TemporalState after the last call.

Schedules (every worker's records in t order, every record exactly once; per t the workers that have record t are
split into calls):
  all             one call per t (all those workers)
  k:<k>           calls of k workers (a seeded shuffle per t)
  poisson:<m>     call sizes 1 + Poisson(m - 1) (seq1: m = 7.8), seeded
  <spec>@<seed>   the shuffle / sizes' seed (default 1)
Sampling uses counter-based uniforms (GraphActor ukeys: key = worker + 1, t = the record's index), so a record's
draw never depends on the call it is in: two schedules differ only by numerics.

Backends: 'py' (tok_policy.GraphActor; --fold, --fp32, --row-buckets); the C++ driver's test entry joins as another
backend with the same call() / state() (P package phase 1).

   usage: python tools/p_replay.py <rollouts dir> [--steps 300] [--pairs 'poisson:7.8=poisson:7.8,k:16=k:4,...']
          [--fold 0|1] [--fp32] [--out f.json]
A pair 'X=Y' runs X and Y on fresh backends and reports per field: records compared, bit-equal records, max |diff|;
'X=X' is the determinism check (expected: every field bit-equal). Prints one JSON object (and --out <file>)."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

STATE = ('kv', 'steps', 'nroom', 'prev_o', 'prev_stage', 'prev_map')
OUTS = ('actions', 'logp', 'value', 'vg', 'create', 'nroom', 'steps', 'stage_prev', 'e_bucket')


def schedule(length, steps, spec):
    """[(workers int64 [k], t)] for records t < min(length, steps) of every worker."""
    spec, _, seed = spec.partition('@')
    rng = np.random.default_rng(int(seed or 1))
    kind, _, arg = spec.partition(':')
    out = []
    for t in range(int(min(length.max(), steps))):
        have = np.flatnonzero(length > t)
        if not len(have):
            break
        if kind == 'all':
            out.append((have, t))
            continue
        have = rng.permutation(have)
        at = 0
        while at < len(have):
            if kind == 'k':
                k = int(arg)
            elif kind == 'poisson':
                k = 1 + int(rng.poisson(float(arg) - 1))
            else:
                raise ValueError(f'schedule {spec!r}')
            out.append((np.sort(have[at:at + k]), t))
            at += k
    return out


class PyBackend:
    """tok_policy.GraphActor (the training loop's call) on its own TemporalState and rollout-shaped K / V, s, o."""

    def __init__(self, model, workers, rows_kept, fold=False, bf16=True, row_sizes=None):
        import torch
        from isaac_bridge import tok_policy as TP
        dev = next(model.parameters()).device
        self.torch = torch
        ta = model.tatt
        self.ts = TP.TemporalState(model, workers, dev, torch.bfloat16 if bf16 else torch.float32)
        self.td = dict(kv=torch.zeros((rows_kept, ta.L, 2, ta.D), dtype=self.ts.dtype, device=dev),
                       s=torch.zeros((rows_kept, ta.D), dtype=self.ts.dtype, device=dev),
                       o=torch.zeros((rows_kept, ta.D), device=dev), obase=0)
        mem = int(getattr(model, 'memory', 0) or 0)
        self.table = torch.zeros((workers + 1, mem), device=dev) if mem else None
        self.g = TP.GraphActor(model, workers, 32, stream=torch.cuda.Stream(), crn=False, memory=self.table,
                               temporal=self.ts, bf16=bf16, fold=fold, row_sizes=row_sizes)
        self.ts.reset_()   # (as after the capture: every slot at the start)
        if self.table is not None:
            self.table.zero_()
        torch.cuda.synchronize()

    def call(self, rows, pending, widx, first, dest_rows, ukeys):
        g = self.g
        if not g.submit(rows, pending, widx, first, tdest=dict(self.td, rows=dest_rows), ukeys=ukeys):
            raise RuntimeError('a call did not fit the graphs')
        actions, logp, value = g.result()
        t = g.t_out
        return dict(actions=actions, logp=logp, value=value, vg=g.vg if g.vg is not None else np.zeros(len(rows)),
                    create=t['create'], nroom=t['nroom'], steps=t['steps'], stage_prev=t['stage_prev'])

    def state(self):
        """The TemporalState's tensors of the worker slots, slot axis first (the scratch slot left out: the graphs'
        padded rows write there, from whatever the staging rows held -- not a record's state)."""
        self.torch.cuda.synchronize()
        out = {}
        for k in STATE:
            v = getattr(self.ts, k).float().cpu().numpy()
            v = np.moveaxis(v, 2, 0) if k == 'kv' else v
            out[k] = v[:self.ts.scratch]
        return out

    def kept(self):
        self.torch.cuda.synchronize()
        return {k: self.td[k].float().cpu().numpy() for k in ('kv', 's', 'o')}


class CppBackend(PyBackend):
    """(P package phase 1) The same GraphActor(fold=True)'s graphs and buffers, each call's middle through the C++
    driver (isaac_bridge.tok_driver.Middle: the check, the staging, the copies, the replay, the ring in C++)."""

    def __init__(self, model, workers, rows_kept, fold=True, bf16=True, row_sizes=None):
        super().__init__(model, workers, rows_kept, fold=True, bf16=bf16, row_sizes=row_sizes)
        from isaac_bridge.tok_driver import Middle
        self.mid = Middle(self.g, ring=self.td)

    def call(self, rows, pending, widx, first, dest_rows, ukeys):
        actions, logp, value, vg, t = self.mid.call(rows, pending, widx, dest_rows, ukeys)
        return dict(actions=actions, logp=logp, value=value, vg=vg if vg is not None else np.zeros(len(rows)),
                    create=t['create'], nroom=t['nroom'], steps=t['steps'], stage_prev=t['stage_prev'])


def replay(backend, z, steps, spec):
    """Every record of the schedule through the backend: (outputs per record [W, T, ...], done mask [W, T])."""
    rows, pend, length = z['rows'], z['pending'].astype(np.int64), z['length']
    W = len(length)
    T = int(min(length.max(), steps))
    out = {}
    done = np.zeros((W, T), bool)
    for w, t in schedule(length, steps, spec):
        sub = np.ascontiguousarray(rows[w, t])
        res = backend.call(sub, np.ascontiguousarray(pend[w, t]), w.astype(np.int64), sub['first'] != 0,
                           (w * T + t).astype(np.int64), (w.astype(np.int64) + 1, np.full(len(w), t, np.int64)))
        from isaac_bridge import v3_input as VI
        res['e_bucket'] = np.full(len(w), VI.rows_counts(sub, stats=False)[1], np.int64)   # (the call's E bucket)
        for k, v in res.items():
            v = np.asarray(v)
            if k not in out:
                out[k] = np.zeros((W, T) + v.shape[1:], v.dtype)
            out[k][w, t] = v
        done[w, t] = True
    return out, done


def diff(a, b, mask=None):
    """(records compared, bit-equal records, max |a - b|) of two arrays [N, ...] (rows where mask)."""
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    a2, b2 = a.reshape(len(a), -1), b.reshape(len(b), -1)
    eq = np.all((a2 == b2) | (np.isnan(a2.astype(np.float64)) & np.isnan(b2.astype(np.float64))), 1)
    d = np.abs(a2.astype(np.float64) - b2.astype(np.float64))
    return dict(n=int(len(a)), equal=int(eq.sum()), max_abs=float(np.nanmax(d)) if d.size else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--steps', type=int, default=300, help='records per worker at most (from the rollout start)')
    ap.add_argument('--pairs', default='poisson:7.8=poisson:7.8,poisson:7.8=poisson:7.8@2,all=k:1,k:16=k:4')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--row-buckets', default='')
    ap.add_argument('--fp32', action='store_true', help='graphs in float32 (default bf16, as --actor-bf16 1)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    import torch
    from isaac_bridge import tok_policy as TP
    dev = torch.device('cuda')
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    ck = torch.load(pts[0], map_location='cpu', weights_only=False)
    model = TP.TokPolicy(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    z = np.load(paths[0], allow_pickle=False)
    z = dict(rows=z['rows'], pending=z['pending'], length=z['length'])
    W = len(z['length'])
    T = int(min(z['length'].max(), a.steps))
    rs = [int(x) for x in a.row_buckets.split(',') if x.strip()] or None
    def mk(kind):   # ('py' / 'cpp' before a schedule spec: 'cpp:poisson:7.8'; default py)
        cls = CppBackend if kind == 'cpp' else PyBackend
        return cls(model, W, W * T, fold=bool(a.fold), bf16=not a.fp32, row_sizes=rs)
    res = dict(rollout=os.path.basename(paths[0]), workers=W, steps=T, fold=a.fold, bf16=not a.fp32,
               row_buckets=rs, gpu=torch.cuda.get_device_name(0), pairs={})
    for pair in a.pairs.split(','):
        x, _, y = pair.partition('=')
        runs = []
        for spec in (x, y):
            kind = 'py'
            if spec.split(':', 1)[0] in ('py', 'cpp'):
                kind, spec = spec.split(':', 1)
            be = mk(kind)
            out, done = replay(be, z, a.steps, spec)
            runs.append((out, done, be.kept(), be.state(), len(schedule(z['length'], a.steps, spec))))
            del be
            torch.cuda.empty_cache()
        (oa, da, ka, sa, na), (ob, db, kb, sb, nb) = runs
        assert (da == db).all()
        rec = {k: diff(oa[k], ob[k], da) for k in OUTS if k in oa}
        rows_kept = da.reshape(-1)
        rec.update({f'kept_{k}': diff(ka[k], kb[k], rows_kept) for k in ka})
        rec.update({f'state_{k}': diff(sa[k], sb[k]) for k in sa})
        same = da & (oa['e_bucket'] == ob['e_bucket'])   # (records whose calls had the same E bucket in both runs)
        rec['same_bucket_records'] = int(same.sum())
        rec.update({f'same_bucket_{k}': diff(oa[k], ob[k], same) for k in ('actions', 'logp', 'value')})
        rec.update({f'same_bucket_kept_{k}': diff(ka[k], kb[k], same.reshape(-1)) for k in ka})
        rec['calls'] = [na, nb]
        rec['all_bit_equal'] = all(v['equal'] == v['n'] for k, v in rec.items() if isinstance(v, dict))
        res['pairs'][pair] = rec
        print(pair, 'all bit-equal' if rec['all_bit_equal'] else
              {k: v for k, v in rec.items() if isinstance(v, dict) and v['equal'] != v['n']}, flush=True)
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    main()
