"""P package phase 1: the gates on the GPU through the whole driver chain (2026-11-19, P-driver-design 11.5).

Fake workers on socketpairs replay the records of a train_tok --dump-rollouts directory (D1's: real v3 frames and the
model) into shared ROW slots; the real C++ driver (abp_tok_thread.h TokDriver over TokMiddle and the captured fold
graphs) answers them, and tok_consumer.DriverCollect -- the training loop's consumer -- commits every call: the go mask
and pending checked against a book, each decision's K / V, s and o copied from the driver's output ring into a
rollout's t_kv / t_s / t_o, store. Sampling uses B36's per-record uniforms (TokDriver.test_ukeys: key worker + 1, t the
ROW 't'), so a Python replay draws the same numbers whatever the calls.

Gate 1 (same batch, bitwise): the calls the driver made, replayed in commit order through tok_policy.GraphActor
  (fold=1, p_replay.PyBackend) with the pending each call used, the same weights per call; compared bit for bit:
  actions / logp / value / vg / the four temporal counters per record, the rollout's t_kv / t_s / t_o rows, and every
  worker slot's TemporalState at the end.
Gate 2 (variable batch, numeric; --alt): the same records through Python in other call splits -- 'single' (one record
  per call), 'wide' (each worker's n-th record together) -- against the driver's (and Python's same split against the
  alternative: the reference line). Per alternative: action flips, and on the records whose actions agree the logp /
  value / vg differences: mean (bias) and its standard error, share of |dlogp| > 0.1, max; the counters must agree.
Gate 3 (asynchronous faults): small credits (--blocks, --ring), a slow consumer (--lag), workers at random paces
  (--pace), weight swaps during the collect (--swaps: drain, new weights, version, resume -- the training loop's
  _W), a worker's EOF right after a record (--kill w:i, with the queue full when --blocks is small), the consumer's
  death (--die-after: store raises after that many calls; the driver must stop promptly); --flat: one episode per
  worker (first / done cleared after record 0, 't' = the record index) so the temporal step passes 256. Every run
  checks: each worker's records committed once each, in order, none lost (the book's arrivals against what was sent);
  each record's version the weights' version then.

   usage: python tools/p_chain.py <rollouts dir> [--steps 300] [--workers 16] [--fp32] [--blocks 64] [--ring 8192]
          [--lag 0] [--pace 0] [--swaps 0] [--kill w:i] [--die-after 0] [--flat] [--ends n] [--alt single,wide] [--out f.json]
Prints one JSON object (and --out)."""
import argparse
import glob
import json
import os
import socket
import struct
import sys
import threading
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p_replay import STATE, PyBackend, diff   # noqa: E402

REPLY = b'\x00\x00\x00\x01a'
OUTS = ('actions', 'logp', 'value', 'vg', 'create', 'nroom', 'steps', 'stage_prev')
TC = ('create', 'nroom', 'steps', 'stage_prev')


class Sampler:
    """What DriverCollect reads of tok_sampler's: the ROW slots [2 W], actions, values, the driver's pipe ends, the
    connections and _drop (a worker's end / EOF)."""

    def __init__(self, W, nh):
        from isaac_bridge.tok_obs import ROW
        self.rows = np.zeros(2 * W, ROW)
        self.actions = np.zeros((W, nh), np.int32)
        self.main_value = np.zeros(W)
        self.main_vg = np.zeros(W)
        pairs = [socket.socketpair() for _ in range(W)]
        self.drv, self.wk = [p[0] for p in pairs], [p[1] for p in pairs]
        self._fds = [s.fileno() for s in self.drv]
        self.conns = list(range(W))
        self.dropped = []

    def _drop(self, c):
        self.dropped.append(c)

    @property
    def live(self):
        return len(set(self.dropped)) < len(self.conns)

    def close(self):
        for s in self.drv + self.wk:
            try:
                s.close()
            except OSError:
                pass


class Book:
    """Episodes.arrive's go rule and pending resets (cap_t = trunc = 0: a done record needs an action only as a
    branch's end, DONE_BRANCH); every arrival logged per worker."""

    def __init__(self, W):
        self.arrived = defaultdict(list)

    def arrive(self, roll, rows, idx, pending):
        from isaac_bridge.tok_obs import DONE_BRANCH
        done = rows['done'].astype(np.int64)
        first = rows['first'] != 0
        go = (done == 0) | (done == DONE_BRANCH)
        pending[idx[(done != 0) & ~go]] = 0
        pending[idx[first & go]] = 0
        for w, r in zip(idx.tolist(), rows):
            self.arrived[w].append((int(r['t']), int(r['first']), int(r['done'])))
        return go


class Roll:
    """The rollout's temporal rows: worker w's i-th decision at row w * T + i."""

    def __init__(self, W, T, ta, dtype, dev):
        import torch
        self.T = T
        self.t_kv = torch.zeros((W * T, ta.L, 2, ta.D), dtype=dtype, device=dev)
        self.t_s = torch.zeros((W * T, ta.D), dtype=dtype, device=dev)
        self.t_o = torch.zeros((W * T, ta.D), device=dev)
        self.t_obase = 0
        self.cnt = np.zeros(W, np.int64)
        self.last = None

    def usable(self):
        return 0   # (the collect runs until every worker has ended)

    def t_assign(self, a_idx):
        r = a_idx * self.T + self.cnt[a_idx]
        self.cnt[a_idx] += 1
        self.last = r
        return r


def state_of(ts):
    """p_replay.PyBackend.state of a TemporalState: the worker slots' tensors, slot axis first."""
    import torch
    torch.cuda.synchronize()
    out = {}
    for k in STATE:
        v = getattr(ts, k).float().cpu().numpy()
        v = np.moveaxis(v, 2, 0) if k == 'kv' else v
        out[k] = v[:ts.scratch]
    return out


def numstats(a, b):
    """Gate 2: b against a, records aligned. Flips; on the records whose actions agree, the differences."""
    eq = np.all(a['actions'] == b['actions'], 1)
    res = dict(records=int(len(eq)), flips=int((~eq).sum()),
               counters_equal=bool(all(np.array_equal(a[k], b[k]) for k in TC)))
    for k in ('logp', 'value', 'vg'):
        d = (b[k].astype(np.float64) - a[k].astype(np.float64))[eq]
        n = max(len(d), 1)
        res[k] = dict(bias=float(d.mean()) if len(d) else 0.0, se=float(d.std() / np.sqrt(n)) if len(d) else 0.0,
                      max_abs=float(np.abs(d).max()) if len(d) else 0.0,
                      gt_0_1=float((np.abs(d) > 0.1).mean()) if len(d) else 0.0)
    res['pass'] = bool(res['counters_equal'] and res['logp']['gt_0_1'] < 1e-4
                       and abs(res['logp']['bias']) <= 3 * res['logp']['se'] + 1e-6)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rollouts')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--blocks', type=int, default=64)
    ap.add_argument('--ring', type=int, default=8192)
    ap.add_argument('--lag', type=float, default=0.0, help='seconds the consumer sleeps per committed call')
    ap.add_argument('--pace', type=float, default=0.0, help='a worker sleeps U(0, pace) s between records')
    ap.add_argument('--swaps', type=int, default=0, help='weight swaps during the collect')
    ap.add_argument('--kill', default='', help="w:i -- worker w's socket closed right after sending its i-th record")
    ap.add_argument('--die-after', type=int, default=0, help='store raises at this call (the consumer dies)')
    ap.add_argument('--flat', action='store_true')
    ap.add_argument('--ends', type=int, default=0, help="every n-th record an episode's end (done 3, or DONE_BRANCH "
                    "every other time: a branch's end, still answered), the next a new episode's first")
    ap.add_argument('--alt', default='', help='gate 2: single,wide')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=1500)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    def watchdog():
        print(json.dumps(dict(error=f'p_chain: no end within {a.timeout} s (a hang)')), flush=True)
        os._exit(3)
    wd = threading.Timer(a.timeout, watchdog)
    wd.daemon = True
    wd.start()

    import torch
    from isaac_bridge import tok_consumer, tok_driver
    from isaac_bridge import tok_policy as TP
    dev = torch.device('cuda')
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    pts = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.pt')))
    ck = torch.load(pts[0], map_location='cpu', weights_only=False)
    model = TP.TokPolicy(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    assert not int(getattr(model, 'memory', 0) or 0), 'p_chain: a GRU model (phase 1 refuses it)'
    assert getattr(model, 'tatt', None) is not None, 'p_chain: not a temporal model'
    zf = np.load(paths[0], allow_pickle=False)
    length = zf['length']
    W = min(a.workers, len(length))
    T = int(min(length[:W].max(), a.steps))
    rows = np.ascontiguousarray(zf['rows'][:W, :T])
    length = np.minimum(length[:W], T)
    del zf
    bf16 = not a.fp32
    mod = tok_driver.load(verbose=True)
    assert mod is not None, f'p_chain: the extension: {tok_driver.REASON}'

    # ---- the weights: version v's parameters (v = 0: the checkpoint's; v > 0: seeded small perturbations)
    params = [p for p in model.parameters()]
    w0 = [p.detach().clone() for p in params]
    rng_w = np.random.default_rng(a.seed + 100)
    wsets = [w0]
    for v in range(1, a.swaps + 1):
        g = torch.Generator(device='cpu').manual_seed(int(rng_w.integers(1 << 30)))
        wsets.append([p + (0.01 * max(float(p.float().std()) if p.numel() > 1 else 0.0, 1e-3)
                           * torch.randn(p.shape, generator=g)).to(p.device, p.dtype) for p in w0])

    def load_w(v):
        with torch.no_grad():
            for p, q in zip(params, wsets[v]):
                p.copy_(q)
        torch.cuda.synchronize()

    # ---- the driver chain
    stream = torch.cuda.Stream()
    ts = TP.TemporalState(model, W, dev, torch.bfloat16 if bf16 else torch.float32)
    sampler = Sampler(W, 8)
    gkw = dict(bf16=bf16)
    dc = tok_consumer.DriverCollect(mod, model, sampler, W, 32, gkw, ts, None, stream, 0, 0, 0,
                                    blocks=a.blocks, ring=a.ring)
    nh = dc.nh
    ts.reset_()   # (as p_replay's: every slot at the start after the capture)
    torch.cuda.synchronize()
    dc.d.test_ukeys = True
    roll = Roll(W, T, model.tatt, ts.dtype, dev)
    book = Book(W)
    pending = np.zeros((W, nh), np.int64)
    log = []          # ('call', index into calls) / ('swap', v) in commit order
    calls = []
    sent = defaultdict(list)
    errors = []
    kill_w, kill_i = (int(x) for x in a.kill.split(':')) if a.kill else (-1, -1)
    state = dict(version=0, calls=0)

    def store(roll_, a_idx, sub, under_way, actions, logp, value, ver, h, vg, t, priv):
        state['calls'] += 1
        if a.die_after and state['calls'] == a.die_after:
            raise RuntimeError('p_chain: the consumer dies (--die-after)')
        if ver != state['version']:
            errors.append(f'call {state["calls"]}: version {ver}, the weights {state["version"]}')
        calls.append(dict(a_idx=a_idx.copy(), rows=sub.copy(), pend=under_way.copy(), dest=roll_.last.copy(),
                                 actions=np.asarray(actions).copy(), logp=np.asarray(logp, np.float32).copy(),
                                 value=np.asarray(value, np.float32).copy(), vg=np.asarray(vg, np.float32).copy(),
                                 **{k: np.asarray(t[k]).astype(np.int64) for k in TC}))
        log.append(('call', len(calls) - 1))
        if a.lag:
            time.sleep(a.lag)

    def worker(w):
        rng = np.random.default_rng(a.seed * 1000 + w)
        s = sampler.wk[w]
        try:
            for i in range(int(length[w])):
                slot = 2 * w + (i & 1)
                sampler.rows[slot] = rows[w, i]
                if a.flat:
                    sampler.rows['first'][slot] = 1 if i == 0 else 0
                    sampler.rows['done'][slot] = 0
                    sampler.rows['t'][slot] = i
                if a.ends:   # (an end record -- no action unless a branch's -- then a new episode at once)
                    from isaac_bridge.tok_obs import DONE_BRANCH
                    e_ = i // a.ends
                    sampler.rows['first'][slot] = 1 if i % a.ends == 0 else 0
                    sampler.rows['done'][slot] = (DONE_BRANCH if e_ % 2 else 3) if i % a.ends == a.ends - 1 else 0
                    sampler.rows['t'][slot] = i
                r = sampler.rows[slot]
                sent[w].append((int(r['t']), int(r['first']), int(r['done'])))
                s.sendall(struct.pack('!i', 1) + (b'0' if slot % 2 == 0 else b'1'))
                if w == kill_w and i == kill_i:
                    s.close()
                    return
                got = b''
                while len(got) < 5:
                    c = s.recv(5 - len(got))
                    if not c:
                        return
                    got += c
                if got != REPLY:
                    errors.append(f'worker {w} record {i}: reply {got!r}')
                if a.pace:
                    time.sleep(float(rng.uniform(0, a.pace)))
            s.sendall(struct.pack('!i', 1) + b'x')
        except OSError as e:
            if not a.die_after:
                errors.append(f'worker {w}: {e}')

    swap_at = sorted(np.random.default_rng(a.seed + 7).choice(np.arange(5, max(int(length.sum()) // 8, 6)),
                                                             size=a.swaps, replace=False).tolist()) if a.swaps else []

    class Weights:
        def ready(self):
            return bool(swap_at) and state['calls'] >= swap_at[0]

        def __call__(self):
            swap_at.pop(0)
            state['version'] += 1
            load_w(state['version'])
            log.append(('swap', state['version']))
            return state['version']

    prof = defaultdict(float)
    load_w(0)
    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(W)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    res = dict(rollout=os.path.basename(paths[0]), workers=W, steps=T, bf16=bf16, flat=a.flat, ends=a.ends,
               blocks=a.blocks,
               ring=a.ring, lag=a.lag, pace=a.pace, swaps=a.swaps, kill=a.kill, die_after=a.die_after,
               gpu=torch.cuda.get_device_name(0))
    died = None
    try:
        dc.collect(roll, book, pending, 1, prof, store, weights=Weights() if a.swaps else None, version=0)
    except Exception as e:   # noqa: BLE001
        died = f'{type(e).__name__}: {e}'
    t_collect = time.perf_counter() - t0
    tc0 = time.perf_counter()
    closer = threading.Thread(target=dc.close, daemon=True)
    closer.start()
    closer.join(20)
    res['close_s'] = time.perf_counter() - tc0
    res['closed'] = not closer.is_alive()
    sampler.close()
    for th in threads:
        th.join(5)
    res['collect_s'] = t_collect
    res['counters'] = {k: v for k, v in dc.d.counters().items() if isinstance(v, (int, float))}
    res['consumer'] = dict(dc.stat)
    if a.die_after:
        res['died'] = died
        res['pass'] = bool(died and 'consumer dies' in died and res['closed'])
        print(json.dumps(res, indent=1))
        if a.out:
            open(a.out, 'w').write(json.dumps(res, indent=1))
        os._exit(0 if res['pass'] else 1)
    if died:
        errors.append(f'the collect raised: {died}')

    # ---- gate 3: every record committed once, in order (the book's arrivals against what the workers sent)
    lost = {w: (len(sent[w]), len(book.arrived[w])) for w in range(W) if sent[w] != book.arrived[w]}
    recs = sum(len(c['a_idx']) for c in calls)
    res['gate3'] = dict(records_sent=int(sum(len(v) for v in sent.values())),
                        records_arrived=int(sum(len(v) for v in book.arrived.values())), decisions=int(recs),
                        calls=len(calls), swaps_done=sum(1 for k, _ in log if k == 'swap'),
                        ends=int(sum(1 for v in sent.values() for r in v if r[2] != 0)),
                        firsts=int(sum(1 for v in sent.values() for r in v if r[1] != 0)),
                        gone=dc.stat['gone'], workers_mismatch=lost, errors=errors[:20],
                        max_steps=int(max((c['steps'].max() for c in calls), default=0)),
                        rows_per_call=float(recs / max(len(calls), 1)))
    res['gate3']['pass'] = not lost and not errors and res['closed'] and (
        a.swaps == 0 or res['gate3']['swaps_done'] == a.swaps)

    # ---- the driver's side, kept before its tensors go
    drv = {k: np.concatenate([c[k] for c in calls]) for k in OUTS}
    dest_all = np.concatenate([c['dest'] for c in calls])
    torch.cuda.synchronize()
    kept_d = dict(kv=roll.t_kv.float().cpu().numpy(), s=roll.t_s.float().cpu().numpy(),
                  o=roll.t_o.float().cpu().numpy())
    state_d = state_of(ts)
    del dc, roll
    torch.cuda.empty_cache()

    def py_run(split):
        """The same records through Python's GraphActor(fold=1), the calls of `split`; outputs in the driver's order."""
        load_w(0)
        be = PyBackend(model, W, W * T, fold=True, bf16=bf16)
        out = {k: np.zeros_like(drv[k]) for k in OUTS}
        base = np.cumsum([0] + [len(c['a_idx']) for c in calls])
        if split == 'same':
            groups = []
            for k, x in log:
                groups.append(('swap', x) if k == 'swap' else ('call', [(x, j) for j in range(len(calls[x]['a_idx']))]))
        else:
            assert a.swaps == 0, 'gate 2 splits need --swaps 0'
            items = [(ci, j) for ci, c in enumerate(calls) for j in range(len(c['a_idx']))]
            if split == 'single':
                groups = [('call', [it]) for it in items]
            elif split == 'wide':
                nth, lv = defaultdict(int), defaultdict(list)
                for ci, j in items:
                    w = int(calls[ci]['a_idx'][j])
                    lv[nth[w]].append((ci, j))
                    nth[w] += 1
                groups = [('call', lv[i]) for i in sorted(lv)]
            else:
                raise ValueError(split)
        for kind, g in groups:
            if kind == 'swap':
                load_w(g)
                continue
            c_rows = np.concatenate([calls[ci]['rows'][j:j + 1] for ci, j in g])
            widx = np.array([calls[ci]['a_idx'][j] for ci, j in g], np.int64)
            pend = np.stack([calls[ci]['pend'][j] for ci, j in g])
            dest = np.array([calls[ci]['dest'][j] for ci, j in g], np.int64)
            r = be.call(c_rows, pend, widx, c_rows['first'] != 0, dest,
                        (widx + 1, c_rows['t'].astype(np.int64)))
            at = np.array([base[ci] + j for ci, j in g])
            for k in OUTS:
                out[k][at] = np.asarray(r[k]).astype(out[k].dtype).reshape(out[k][at].shape)
        kept, st = be.kept(), be.state()
        del be
        torch.cuda.empty_cache()
        return out, kept, st

    # ---- gate 1: the same calls through Python, bit for bit
    po, pk, ps = py_run('same')
    g1 = {k: diff(drv[k], po[k]) for k in OUTS}
    g1.update({f'rollout_{k}': diff(kept_d[k][dest_all], pk[k][dest_all]) for k in kept_d})
    g1.update({f'state_{k}': diff(state_d[k], ps[k]) for k in ps})
    g1['all_bit_equal'] = all(v['equal'] == v['n'] for v in g1.values() if isinstance(v, dict))
    res['gate1'] = g1

    # ---- gate 2: other splits
    if a.alt:
        res['gate2'] = {}
        for split in a.alt.split(','):
            ao, _, _ = py_run(split)
            res['gate2'][split] = dict(driver=numstats(drv, ao), python_same_split=numstats(po, ao))
    res['pass'] = bool(res['gate3']['pass'] and g1['all_bit_equal']
                       and all(v['driver']['pass'] for v in res.get('gate2', {}).values()))
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)
    os._exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
