"""P package phase 1: the gates on the GPU through the whole driver chain (2026-11-19, P-driver-design 11.5; rewritten
after Codex round 17).

Fake workers on socketpairs replay the records of a train_tok --dump-rollouts directory (D1's: real v3 frames and the
model) into shared ROW slots; the real C++ driver (abp_tok_thread.h TokDriver over TokMiddle and the captured fold
graphs) answers them, and tok_consumer.DriverCollect -- the training loop's consumer -- commits every call (the go mask
and pending checked against a book, each decision's K / V, s and o copied from the driver's output ring into the
rollout's t_kv / t_s / t_o, store). Sampling uses B36's per-record uniforms (TokDriver.test_ukeys: key worker + 1, t the
ROW 't'), so a Python replay draws the same numbers whatever the calls.

Every expectation comes from the SENDING side, never from what the tested chain handed on: each worker logs the rows it
wrote, its snapshot requests, whether it got its reply, and the actions / values it read back. The reference
(p_replay.PyBackend: tok_policy.GraphActor fold=1, its own TemporalState, its own snapshot bank) walks the calls the
driver made -- only the partition: which workers' next records formed each call, in commit order (the book's log) --
and derives everything else itself: each worker's next record, go (Episodes.arrive's rule with the cap / stall
truncations), the pending resets and the actions under way, each decision's rollout row, the snapshot copies.

Comparisons are bitwise: finiteness first (a NaN / inf anywhere fails), then the bytes of the original dtype (+0 / -0
and NaN payloads differ).

Gate 1 (same calls): per decision -- actions, logp, value, vg, the four temporal counters, the pending used, the version
  -- against the reference; the rows the consumer stored against the sender's (the prefix the driver copies: fixed
  fields and the frame); the rollout's t_kv / t_s / t_o rows; the actions / value / vg each worker read back; at the end
  every worker slot's TemporalState and the snapshot bank; each snapshot request cleared after a go record's step, left
  (to the worker's cancel) on an end.
Gate 2 (other calls; --alt): each record's log-probability of the SAME action -- the driver's -- recomputed in other
  splits ('single': one record per call; 'wide': each worker's n-th record together) by forcing the Gumbel-max uniforms
  (1 - 1e-7 at the action, 1e-10 elsewhere: tok_policy.gumbel_sample's clamp), and value / vg. Coverage: every record
  must come out with its forced action (and the forced same split must give the driver's numbers bit for bit). The
  error line: Python against Python (the first alternative against the last). Fails on a non-finite value, counters that
  differ, |dlogp| > 0.1 in 1e-4 or more of the records, a logp / value / vg difference (max, or mean beyond 3 standard
  errors) more than twice the line's.
Gate 3 (asynchronous faults): small credits (--blocks, --ring), a slow consumer (--lag), random paces (--pace), weight
  swaps during the collect (--swaps: drain, new weights, version, resume), a worker's EOF right after a record (--kill
  w:i), episode ends (--ends n: done 1 / 2 / 3 below the cap (the stall's) / 3 at the cap / DONE_BRANCH in turn, then a
  new episode; --cap-t / --trunc: the run's truncation rule), snapshot requests (--cf p: a bank row of the worker's own,
  reused in turn), the consumer's death (--die-after: store raises at that call; then the training loop's close: every
  worker must get its reply and end). Four-way reconciliation per worker and record: sent, replied (as the worker saw
  it), booked (arrive), stored (the reference's decisions = the stored ones) -- none lost, doubled or reordered; each
  version the weights' version then.

   usage: python tools/p_chain.py <rollouts dir> [--steps 300] [--workers 16] [--fp32] [--blocks 64] [--ring 8192]
          [--lag 0] [--pace 0] [--swaps 0] [--kill w:i] [--die-after 0] [--flat] [--ends n] [--cap-t 0] [--trunc 0]
          [--cf 0] [--alt single,wide] [--out f.json]
Prints one JSON object (and --out); exit 0 only when every gate it ran passed."""
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

from p_replay import STATE, PyBackend   # noqa: E402

REPLY = b'\x00\x00\x00\x01a'
OUTS = ('actions', 'logp', 'value', 'vg', 'create', 'nroom', 'steps', 'stage_prev')
TC = ('create', 'nroom', 'steps', 'stage_prev')
BANK_R = 4   # bank rows per worker (reused in turn)


def bitdiff(a, b):
    """(Codex 17-4) a against b, bit for bit: dict(n, equal, finite, max_abs). Rows [N, ...]; finiteness first (a NaN /
    inf on either side fails the row), then the bytes of the original dtypes (+0 / -0 differ). a / b may be (bits, finite
    per row) pairs (tensor_bits: a bf16 tensor's bits are int16, its finiteness taken before -- Codex 18-4)."""
    fa_ = fb_ = None
    if isinstance(a, tuple):
        a, fa_ = a
    if isinstance(b, tuple):
        b, fb_ = b
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(n=int(len(a)), equal=0, finite=False, max_abs=float('inf'),
                    why=f'{a.shape}/{a.dtype} vs {b.shape}/{b.dtype}')
    n = len(a)
    if n == 0:
        return dict(n=0, equal=0, finite=True, max_abs=0.0)
    a2, b2 = a.reshape(n, -1), b.reshape(n, -1)
    fin = (np.isfinite(a2).all(1) & np.isfinite(b2).all(1)) if a.dtype.kind == 'f' else np.ones(n, bool)
    if fa_ is not None:
        fin &= np.asarray(fa_, bool).reshape(n)
    if fb_ is not None:
        fin &= np.asarray(fb_, bool).reshape(n)
    eq = (a2.view(np.uint8).reshape(n, -1) == b2.view(np.uint8).reshape(n, -1)).all(1) & fin
    with np.errstate(invalid='ignore'):
        d = np.abs(a2.astype(np.float64) - b2.astype(np.float64))
    return dict(n=int(n), equal=int(eq.sum()), finite=bool(fin.all()),
                max_abs=float(np.nanmax(d)) if np.isfinite(d).any() else float('nan'))


def all_equal(rec):
    return all(v['equal'] == v['n'] and v.get('finite', True) for v in rec.values()
               if isinstance(v, dict) and 'equal' in v)


def go_rule(rows, cap_t, trunc, done_branch):
    """Episodes.arrive's go (with is_trunc): a record needs an action unless it ends its episode -- a branch's end and a
    truncation (cap / stall, as TRUNC says) are still answered."""
    done = rows['done'].astype(np.int64)
    t3 = done == 3
    at_cap = rows['t'].astype(np.int64) >= cap_t
    trunc_ = (t3 & ((at_cap & bool(trunc & 1)) | (~at_cap & bool(trunc & 2)))) if (cap_t > 0 and trunc) \
        else np.zeros(len(rows), bool)
    return (done == 0) | (done == done_branch) | trunc_


class Sampler:
    """What DriverCollect reads of tok_sampler's: ROW slots [2 W], actions, values, cf_snap, control, the driver's pipe
    ends, the connections and _drop (a worker's end / EOF)."""

    def __init__(self, W, cf):
        from isaac_bridge.tok_obs import ROW
        self.rows = np.zeros(2 * W, ROW)
        self.actions = np.zeros((W, 8), np.int32)
        self.main_value = np.zeros(W)
        self.main_vg = np.zeros(W)
        self.cf_snap = np.zeros(W, np.int64) if cf else None
        self.control = np.zeros(4, np.int64)
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
    """Logs each arrive (the call: its workers in order) and returns the go mask by Episodes.arrive's rule with its
    pending resets."""

    def __init__(self, cap_t, trunc):
        from isaac_bridge.tok_obs import DONE_BRANCH
        self.cap_t, self.trunc, self.db = cap_t, trunc, DONE_BRANCH
        self.calls = []                      # workers int64 [k] per arrive, in commit order
        self.arrived = defaultdict(list)     # worker -> [(t, first, done)]
        self.origin_rec = None

    def arrive(self, roll, rows, idx, pending):
        done = rows['done'].astype(np.int64)
        first = rows['first'] != 0
        go = go_rule(rows, self.cap_t, self.trunc, self.db)
        pending[idx[(done != 0) & ~go]] = 0
        pending[idx[first & go]] = 0
        self.calls.append(idx.copy())
        for w, r in zip(idx.tolist(), rows):
            self.arrived[w].append((int(r['t']), int(r['first']), int(r['done'])))
        return go


class Roll:
    """The rollout's temporal rows: worker w's d-th decision at row w * T + d; capacity T decisions per worker."""

    def __init__(self, W, T, ta, dtype, dev):
        import torch
        self.T, self.cap = T, T
        self.t_kv = torch.zeros((W * T, ta.L, 2, ta.D), dtype=dtype, device=dev)
        self.t_s = torch.zeros((W * T, ta.D), dtype=dtype, device=dev)
        self.t_o = torch.zeros((W * T, ta.D), device=dev)
        self.t_obase = 0
        self.length = np.zeros(W, np.int64)
        self.last = None

    def usable(self):
        return 0   # (the collect runs until every worker has ended)

    def t_assign(self, a_idx):
        r = a_idx * self.T + self.length[a_idx]
        self.last = r
        return r


def tensor_bits(t, axis=0):
    """(bits, finite): the tensor with `axis` first, a bf16 one as its int16 bits; finite per row along that axis --
    taken from the values BEFORE the bits (Codex 18-4: a NaN / inf bf16 is otherwise just an int16)."""
    import torch
    t = t.detach()
    fin = torch.isfinite(t.float()) if t.is_floating_point() else torch.ones_like(t, dtype=torch.bool)
    fin = np.moveaxis(fin.cpu().numpy(), axis, 0)
    v = np.moveaxis((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).cpu().numpy(), axis, 0)
    return np.ascontiguousarray(v), fin.reshape(fin.shape[0], -1).all(1)


def sel(x, i):
    return x[0][i], x[1][i]


def state_of(ts, upto):
    """The slots' tensors as tensor_bits pairs, slot axis first; the slots below `upto` (the scratch slot left out)."""
    import torch
    torch.cuda.synchronize()
    return {k: sel(tensor_bits(getattr(ts, k), 2 if k == 'kv' else 0), slice(0, upto)) for k in STATE}


def slot_bits(ts, s):
    """(Codex 18-5) one slot's state as a lane would read it: per tensor (bits, finite)."""
    return {k: sel(tensor_bits(getattr(ts, k), 2 if k == 'kv' else 0), slice(s, s + 1)) for k in STATE}


def numstats(a, b, line=None):
    """Gate 2: b against a over aligned records (the same forced actions). With `line` (Python against Python): pass /
    fail by Codex 17-4's criteria."""
    res = dict(records=int(len(a['logp'])),
               finite=bool(all(np.isfinite(x[k].astype(np.float64)).all() for x in (a, b) for k in ('logp', 'value', 'vg'))),
               counters_equal=bool(all(np.array_equal(a[k], b[k]) for k in TC)))
    for k in ('logp', 'value', 'vg'):
        d = b[k].astype(np.float64) - a[k].astype(np.float64)
        n = max(len(d), 1)
        res[k] = dict(bias=float(d.mean()) if len(d) else 0.0, se=float(d.std() / np.sqrt(n)) if len(d) else 0.0,
                      max_abs=float(np.abs(d).max()) if len(d) else 0.0,
                      gt_0_1=float((np.abs(d) > 0.1).mean()) if len(d) else 0.0)
    if line is not None:
        why = []
        if not res['finite']:
            why.append('non-finite')
        if not res['counters_equal']:
            why.append('counters')
        if res['logp']['gt_0_1'] >= 1e-4:
            why.append('|dlogp| > 0.1')
        for k in ('logp', 'value', 'vg'):
            if res[k]['max_abs'] > 2 * line[k]['max_abs'] + 1e-6:
                why.append(f'{k} max beyond the line')
            if abs(res[k]['bias']) > 2 * abs(line[k]['bias']) + 3 * res[k]['se'] + 1e-7:
                why.append(f'{k} bias beyond the line')
        res['fail'] = why
        res['pass'] = not why
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
    ap.add_argument('--flat', action='store_true', help="one episode per worker, 't' = the record index")
    ap.add_argument('--ends', type=int, default=0, help='every n-th record an end (done 1 / 2 / 3 / 3 at the cap / '
                    'DONE_BRANCH in turn), the next a new episode')
    ap.add_argument('--cap-t', type=int, default=0, help="the run's CAP_T ('t' at or past it: the cap's end)")
    ap.add_argument('--trunc', type=int, default=0, help="the run's TRUNC (1 the cap's ends, 2 the stall's, 3 both)")
    ap.add_argument('--cf', type=float, default=0.0, help='snapshot requests per record (probability)')
    ap.add_argument('--alt', default='', help='gate 2: single,wide')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--require', default='', help="gates that must have run (Codex 18-4): '1,2,3' -- gate 2 with "
                    "--alt naming two different splits, every gate over a non-zero number of records")
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
    from isaac_bridge.tok_obs import DONE_BRANCH, ROW
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
    src_rows = np.ascontiguousarray(zf['rows'][:W, :T])
    length = np.minimum(length[:W], T)
    del zf
    bf16 = not a.fp32
    kdt = torch.bfloat16 if bf16 else torch.float32
    mod = tok_driver.load(verbose=True)
    assert mod is not None, f'p_chain: the extension: {tok_driver.REASON}'
    # the uniforms' columns per head as gumbel_sample walks them (TokPolicy.split: heads 2..4 two-wide)
    widths = [2 if j in (2, 3, 4) else int(h) for j, h in enumerate(model.heads)]

    # ---- the weights: version v's parameters (0: the checkpoint's; v > 0: seeded small perturbations)
    params = list(model.parameters())
    w0 = [p.detach().clone() for p in params]
    rng_w = np.random.default_rng(a.seed + 100)
    wsets = [w0]
    for _ in range(a.swaps):
        g = torch.Generator(device='cpu').manual_seed(int(rng_w.integers(1 << 30)))
        wsets.append([p + (0.01 * max(float(p.float().std()) if p.numel() > 1 else 0.0, 1e-3)
                           * torch.randn(p.shape, generator=g)).to(p.device, p.dtype) for p in w0])

    def load_w(v):
        with torch.no_grad():
            for p, q in zip(params, wsets[v]):
                p.copy_(q)
        torch.cuda.synchronize()

    # ---- the records each worker will send (the source rows; ends / firsts / 't' as the options say)
    plan = []
    kinds = [1, 2, 3, 3, DONE_BRANCH]
    for w in range(W):
        rs = src_rows[w, :int(length[w])].copy()
        idx = np.arange(len(rs))
        if a.flat or a.ends:
            rs['t'] = idx
            rs['first'] = (idx == 0).astype(rs['first'].dtype)
            rs['done'] = 0
        if a.ends:
            rs['first'] = (idx % a.ends == 0).astype(rs['first'].dtype)
            for j in np.flatnonzero(idx % a.ends == a.ends - 1):
                c = (j // a.ends + w) % len(kinds)
                rs['done'][j] = kinds[c]
                if kinds[c] == 3:   # (c == 2: below the cap -- the stall's end; c == 3: at / past it -- the cap's)
                    rs['t'][j] = (max(a.cap_t, 1) + j) if c == 3 else min(j, max(a.cap_t - 1, 0))
        plan.append(rs)
    rng_cf = np.random.default_rng(a.seed + 33)
    cf_plan = [(rng_cf.random(len(plan[w])) < a.cf) if a.cf else np.zeros(len(plan[w]), bool) for w in range(W)]

    # ---- the driver chain
    load_w(0)
    stream = torch.cuda.Stream()
    ts = TP.TemporalState(model, W, dev, kdt)
    bank = TP.TemporalState(model, W * BANK_R, dev, kdt) if a.cf else None
    sampler = Sampler(W, bool(a.cf))
    dc = tok_consumer.DriverCollect(mod, model, sampler, W, 32, dict(bf16=bf16), ts, bank, stream, 0, a.cap_t, a.trunc,
                                    blocks=a.blocks, ring=a.ring)
    nh = dc.nh
    ts.reset_()
    if bank is not None:
        bank.reset_()
    torch.cuda.synchronize()
    dc.d.test_ukeys = True
    roll = Roll(W, T, model.tatt, kdt, dev)
    book = Book(a.cap_t, a.trunc)
    pending = np.zeros((W, nh), np.int64)
    swaps_log = []    # (version, the number of calls booked before it)
    stores = []
    sent = defaultdict(list)    # worker -> [dict(i, row [1], cf, replied, actions, value, vg, cf_after)]
    errors = []
    kill_w, kill_i = (int(x) for x in a.kill.split(':')) if a.kill else (-1, -1)
    state = dict(version=0, calls=0)

    def store(roll_, a_idx, sub, under_way, actions, logp, value, ver, h, vg, t, priv):
        state['calls'] += 1
        if a.die_after and state['calls'] == a.die_after:
            raise RuntimeError('p_chain: the consumer dies (--die-after)')
        if ver != state['version']:
            errors.append(f'call {state["calls"]}: version {ver}, the weights {state["version"]}')
        stores.append(dict(a_idx=a_idx.copy(), rows=sub.copy(), pend=np.asarray(under_way).copy(), ver=int(ver),
                           dest=np.asarray(roll_.last).copy(), actions=np.asarray(actions).copy(),
                           logp=np.asarray(logp).copy(), value=np.asarray(value).copy(), vg=np.asarray(vg).copy(),
                           **{k: np.asarray(t[k]).astype(np.int64) for k in TC}))
        roll_.length[a_idx] += 1
        if a.lag:
            time.sleep(a.lag)

    gens_d = defaultdict(list)   # bank row -> the generations read back after each write (Codex 18-5)

    def worker(w):
        rng = np.random.default_rng(a.seed * 1000 + w)
        s = sampler.wk[w]
        nb = 0
        try:
            for i in range(len(plan[w])):
                if sampler.control[0]:   # (the run is closing: the worker ends)
                    break
                slot = 2 * w + (i & 1)
                sampler.rows[slot] = plan[w][i]
                cf = 0
                if cf_plan[w][i]:
                    cf = w * BANK_R + nb % BANK_R + 1
                    nb += 1
                    sampler.cf_snap[w] = cf
                ent = dict(i=i, row=sampler.rows[slot:slot + 1].copy(), cf=cf, replied=False,
                           raw=sampler.rows[slot:slot + 1].view(np.uint8).reshape(-1).copy())   # (the bytes: a
                #  structured copy() does not keep the padding between fields, the driver's memcpy does)
                sent[w].append(ent)
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
                if cf and bank is not None and go_rule(sampler.rows[slot:slot + 1], a.cap_t, a.trunc,
                                                       DONE_BRANCH)[0]:
                    gens_d[cf - 1].append(slot_bits(bank, cf - 1))   # (the generation a lane would read now)
                ent.update(replied=True, actions=sampler.actions[w, :nh].astype(np.int64),
                           value=float(sampler.main_value[w]), vg=float(sampler.main_vg[w]),
                           cf_after=int(sampler.cf_snap[w]) if sampler.cf_snap is not None else 0)
                if sampler.cf_snap is not None and sampler.cf_snap[w]:   # (an end's request: cancelled, as tok_floor)
                    sampler.cf_snap[w] = 0
                if a.pace:
                    time.sleep(float(rng.uniform(0, a.pace)))
            s.sendall(struct.pack('!i', 1) + b'x')
        except OSError as e:
            if not a.die_after:
                errors.append(f'worker {w}: {e}')

    n_rec = int(sum(len(p) for p in plan))
    swap_at = sorted(np.random.default_rng(a.seed + 7).choice(np.arange(5, max(n_rec // 8, 6)), size=a.swaps,
                                                             replace=False).tolist()) if a.swaps else []

    class Weights:
        def ready(self):
            return bool(swap_at) and state['calls'] >= swap_at[0]

        def __call__(self):
            swap_at.pop(0)
            state['version'] += 1
            load_w(state['version'])
            swaps_log.append((state['version'], len(book.calls)))
            return state['version']

    prof = defaultdict(float)
    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(W)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    res = dict(rollout=os.path.basename(paths[0]), workers=W, steps=T, bf16=bf16, flat=a.flat, ends=a.ends,
               cap_t=a.cap_t, trunc=a.trunc, cf=a.cf, blocks=a.blocks, ring=a.ring, lag=a.lag, pace=a.pace,
               swaps=a.swaps, kill=a.kill, die_after=a.die_after, gpu=torch.cuda.get_device_name(0))
    died = None
    try:
        dc.collect(roll, book, pending, 1, prof, store, weights=Weights() if a.swaps else None, version=0)
    except Exception as e:   # noqa: BLE001
        died = f'{type(e).__name__}: {e}'
    res['collect_s'] = time.perf_counter() - t0
    tc0 = time.perf_counter()
    close_res = {}

    def do_close():   # (the training loop's close; its return / exception kept -- Codex 18-4)
        try:
            dc.close(timeout=20.0)
            close_res['ok'] = True
        except Exception as e:   # noqa: BLE001
            close_res['error'] = f'{type(e).__name__}: {e}'
    closer = threading.Thread(target=do_close, daemon=True)
    closer.start()
    closer.join(40)
    res['close_s'] = time.perf_counter() - tc0
    res['close_error'] = close_res.get('error')
    res['closed'] = not closer.is_alive() and bool(close_res.get('ok'))
    for th in threads:
        th.join(10)
    res['workers_hung'] = [w for w, th in enumerate(threads) if th.is_alive()]
    sampler.close()
    res['counters'] = {k: v for k, v in dc.d.counters().items() if isinstance(v, (int, float))}
    res['consumer'] = dict(dc.stat)

    if a.die_after:   # (the consumer died: the exception out, the close prompt and clean, every worker answered)
        res['died'] = died
        res['unreplied'] = {w: [e['i'] for e in sent[w] if not e['replied']] for w in sent
                            if any(not e['replied'] for e in sent[w])}
        res['hit_died_at'] = state['calls'] == a.die_after
        res['pass'] = bool(died and 'consumer dies' in died and res['hit_died_at'] and res['closed']
                           and not res['workers_hung'] and not res['unreplied'])
        _out(res, a)
        return
    if died:
        errors.append(f'the collect raised: {died}')

    # ---- gate 3: per worker and record, sent / replied / booked (stored: below, against the reference's decisions)
    recon = {}
    for w in range(W):
        s_ = [(int(e['row']['t'][0]), int(e['row']['first'][0]), int(e['row']['done'][0])) for e in sent[w]]
        if s_ != book.arrived[w]:
            recon[w] = f'sent {len(s_)} / booked {len(book.arrived[w])}: differ'
        unreplied = [e['i'] for e in sent[w] if not e['replied']]
        if unreplied and not (w == kill_w and unreplied == [kill_i]):
            recon.setdefault(w, f'records without a reply: {unreplied[:5]}')
    res['gate3'] = dict(records_sent=int(sum(len(v) for v in sent.values())),
                        records_booked=int(sum(len(v) for v in book.arrived.values())),
                        decisions_stored=int(sum(len(s_['a_idx']) for s_ in stores)), calls=len(book.calls),
                        swaps_done=len(swaps_log),
                        ends=int(sum(1 for v in sent.values() for e in v if e['row']['done'][0] != 0)),
                        boots=int(sum(1 for v in sent.values() for e in v if e['row']['done'][0] != 0
                                      and go_rule(e['row'], a.cap_t, a.trunc, DONE_BRANCH)[0])),
                        cf_requests=int(sum(1 for v in sent.values() for e in v if e['cf'])),
                        gone=dc.stat['gone'], mismatch=recon, errors=errors[:20], workers_hung=res['workers_hung'])
    # (Codex 18-4) each fault injected must have happened
    hits = {}
    if a.kill:
        hits['kill'] = bool(len(sent[kill_w]) == kill_i + 1 and dc.stat['gone'] >= 1)
    if a.blocks <= 4 or a.ring <= 64:
        hits['short_of_credits'] = bool(res['counters'].get('credit_waits', 0) > 0)
    if a.swaps:
        hits['swaps'] = len(swaps_log) == a.swaps
    if a.ends:
        allr = [e['row'] for v in sent.values() for e in v]
        dn = np.array([int(r['done'][0]) for r in allr])
        tt = np.array([int(r['t'][0]) for r in allr])
        kinds_ = dict(win=int((dn == 1).sum()), death=int((dn == 2).sum()), branch=int((dn == DONE_BRANCH).sum()),
                      stall_end=int(((dn == 3) & (tt < a.cap_t)).sum()), cap_end=int(((dn == 3) & (tt >= a.cap_t)).sum()))
        res['gate3']['end_kinds'] = kinds_
        hits['ends'] = all(v > 0 for k, v in kinds_.items() if k != 'cap_end' or a.cap_t > 0)
    if a.cf:
        hits['cf'] = bool(res['gate3']['cf_requests'] > 0)
        hits['bank_reuse'] = bool(gens_d) and all(len(g) >= 2 for g in gens_d.values())
        res['gate3']['bank_rows_written'] = len(gens_d)
        res['gate3']['bank_writes'] = int(sum(len(g) for g in gens_d.values()))
    if a.lag:
        hits['lag'] = bool(res['counters'].get('credit_waits', 0) > 0 or a.blocks > 4)
    res['gate3']['hits'] = hits

    # ---- the tested chain's results, kept before its tensors go
    torch.cuda.synchronize()
    kept_d = dict(kv=tensor_bits(roll.t_kv), s=tensor_bits(roll.t_s), o=tensor_bits(roll.t_o))
    state_d = state_of(ts, W)
    bank_d = state_of(bank, W * BANK_R) if bank is not None else None
    del dc, roll
    torch.cuda.empty_cache()

    def reference(split, forced=None):
        """Python's GraphActor over the senders' records. split 'same': the driver's calls (the book's partition, the
        swaps where they came); 'single' / 'wide' (gate 2, no swaps). forced: actions per (worker, decision) -- the
        uniforms forced to them. Returns per decision (worker, ordinal) the outputs, the pending used, the version, the
        row; the kept rows, the state, the bank."""
        load_w(0)
        be = PyBackend(model, W, W * T, fold=True, bf16=bf16)
        bank_r = TP.TemporalState(model, W * BANK_R, dev, kdt) if a.cf else None
        if bank_r is not None:
            bank_r.reset_()
        torch.cuda.synchronize()
        pend = np.zeros((W, nh), np.int64)
        nxt, ndec = np.zeros(W, np.int64), np.zeros(W, np.int64)
        gens_r = defaultdict(list)
        out = {}
        ver = 0
        if split == 'same':
            seq, si = [], 0
            for ci, ws in enumerate(book.calls):
                while si < len(swaps_log) and swaps_log[si][1] == ci:
                    seq.append(('swap', swaps_log[si][0]))
                    si += 1
                seq.append(('call', ws))
        elif split == 'single':
            seq = [('call', np.array([w], np.int64)) for ws in book.calls for w in ws]
        elif split == 'wide':
            lv, cnt = defaultdict(list), defaultdict(int)
            for ws in book.calls:
                for w in ws:
                    lv[cnt[int(w)]].append(int(w))
                    cnt[int(w)] += 1
            seq = [('call', np.array(lv[i], np.int64)) for i in sorted(lv)]
        else:
            raise ValueError(split)
        orig_u = TP.crn_uniforms
        if forced is not None:
            def forced_u(keys, t, columns):
                u = np.full((len(keys), columns), 1e-10, np.float32)
                for j, (k_, t_) in enumerate(zip(np.asarray(keys).tolist(), np.asarray(t).tolist())):
                    act = forced[(int(k_) - 1, int(t_))]
                    at = 0
                    for h, hs in enumerate(widths):
                        u[j, at + int(act[h])] = 1 - 1e-7
                        at += hs
                return u
            TP.crn_uniforms = forced_u
        try:
            for kind, x in seq:
                if kind == 'swap':
                    load_w(x)
                    ver = x
                    continue
                ws = x
                rows = np.empty(len(ws), plan[0].dtype)   # (np.concatenate would repack the padded record dtype)
                for j_, w_ in enumerate(ws.tolist()):
                    rows[j_] = plan[w_][int(nxt[w_])]
                cfs = np.array([sent[int(w)][int(nxt[w])]['cf'] for w in ws], np.int64)
                srows = [sent[int(w)][int(nxt[w])]['raw'] for w in ws]   # (the slot's bytes as the worker wrote them)
                nxt[ws] += 1
                go = go_rule(rows, a.cap_t, a.trunc, DONE_BRANCH)
                done = rows['done'] != 0
                first = rows['first'] != 0
                pend[ws[done & ~go]] = 0
                pend[ws[first & go]] = 0
                g = np.flatnonzero(go)
                if not len(g):
                    continue
                gw, gr = ws[g], np.ascontiguousarray(rows[g])
                m = cfs[g] > 0
                if bank_r is not None and m.any():   # (the slot before this record's step, into the named row)
                    bank_r.copy_slot_(gw[m], cfs[g][m] - 1, other=be.ts)
                    for row_ in (cfs[g][m] - 1).tolist():
                        gens_r[row_].append(slot_bits(bank_r, row_))
                dec = ndec[gw].copy()
                ndec[gw] += 1
                keys_t = (gw + 1, gr['t'].astype(np.int64)) if forced is None else (gw + 1, dec)
                r = be.call(gr, pend[gw].copy(), gw.astype(np.int64), gr['first'] != 0,
                            (gw * T + dec).astype(np.int64), keys_t)
                for j, w in enumerate(gw.tolist()):
                    out[(w, int(dec[j]))] = dict(pend=pend[w].copy(), ver=ver, row=gr[j:j + 1].copy(),
                                                 sent_row=srows[int(g[j])],
                                                 **{k: np.asarray(r[k])[j] for k in OUTS})
                pend[gw] = np.asarray(r['actions'], np.int64)
        finally:
            TP.crn_uniforms = orig_u
        torch.cuda.synchronize()
        kept = dict(kv=tensor_bits(be.td['kv']), s=tensor_bits(be.td['s']), o=tensor_bits(be.td['o']))   # (pairs)
        st = state_of(be.ts, W)
        bk = state_of(bank_r, W * BANK_R) if bank_r is not None else None
        del be
        torch.cuda.empty_cache()
        reference.gens = gens_r
        return out, kept, st, bk

    # the tested side per decision (worker, ordinal), in the stores' order
    drv = {}
    cnt = defaultdict(int)
    for s_ in stores:
        for j, w in enumerate(s_['a_idx'].tolist()):
            drv[(w, cnt[w])] = dict(pend=s_['pend'][j], ver=s_['ver'], row=s_['rows'][j:j + 1],
                                    dest=int(s_['dest'][j]), **{k: np.asarray(s_[k])[j] for k in OUTS})
            cnt[w] += 1
    ref, kept_r, state_r, bank_rr = reference('same')
    keys = sorted(ref)
    g1 = dict(decisions_ref=len(ref), decisions_driver=len(drv), same_keys=sorted(drv) == keys)
    if g1['same_keys']:
        def col(src, k, dt):
            return np.stack([np.asarray(src[x][k]) for x in keys]).astype(dt)
        g1['actions'] = bitdiff(col(drv, 'actions', np.int64), col(ref, 'actions', np.int64))
        for k in ('logp', 'value', 'vg'):
            g1[k] = bitdiff(col(drv, k, np.float32), col(ref, k, np.float32))
        for k in TC:
            g1[k] = bitdiff(col(drv, k, np.int64), col(ref, k, np.int64))
        g1['pending'] = bitdiff(col(drv, 'pend', np.int64), col(ref, 'pend', np.int64))
        g1['version'] = bitdiff(np.array([drv[x]['ver'] for x in keys], np.int64),
                                np.array([ref[x]['ver'] for x in keys], np.int64))
        dest = np.array([w * T + d for w, d in keys], np.int64)
        g1['dest'] = bitdiff(np.array([drv[x]['dest'] for x in keys], np.int64), dest)
        # the rows stored against the slot's bytes as the worker wrote them (the prefix the driver copies: the fixed
        # fields and the padding between them, then the frame's bytes)
        fa = ROW.fields['visible_frame'][1]
        same_rows, row_diff, row_ex, pad_rows = 0, {}, [], 0
        for x in keys:
            sr = drv[x]['row'].view(np.uint8).reshape(-1)
            rr = ref[x]['sent_row']
            nb_ = int(ref[x]['row']['visible_bytes'][0])
            e = min(fa + ((max(nb_, 0) + 3) & ~3), ROW.itemsize)
            # every named field and the frame's bytes; the padding between fields is not data (the consumer's
            # rows[g] -- numpy fancy indexing of the padded record dtype -- copies field by field into fresh memory,
            # its padding left uninitialised; store writes the rollout field by field): counted apart
            m_ = np.zeros(e, bool)
            for f in ROW.names:
                o_ = ROW.fields[f][1]
                if o_ < fa:
                    m_[o_:o_ + ROW.fields[f][0].itemsize] = True
            m_[fa:e] = True
            ok_ = bool(np.array_equal(sr[:e][m_], rr[:e][m_]))
            same_rows += ok_
            pad_rows += not np.array_equal(sr[:e][~m_], rr[:e][~m_])
            if not ok_:   # (where they differ: the fields, the frame)
                d_ = np.flatnonzero((sr[:e] != rr[:e]) & m_)
                for o_ in d_[:64]:
                    f_ = next((f for f in ROW.names if ROW.fields[f][1] <= o_ < ROW.fields[f][1]
                               + ROW.fields[f][0].itemsize), 'padding')
                    row_diff[f_] = row_diff.get(f_, 0) + 1
                if len(row_ex) < 3:
                    row_ex.append(dict(key=x, n=int(len(d_)), first=int(d_[0]), e=int(e), nb=nb_,
                                       drv=sr[d_[0]:d_[0] + 8].tolist(), sent=rr[d_[0]:d_[0] + 8].tolist()))
        g1['rows_prefix'] = dict(n=len(keys), equal=same_rows, where=row_diff, examples=row_ex,
                                 padding_differs=int(pad_rows))
        for k in kept_d:
            g1[f'rollout_{k}'] = bitdiff(sel(kept_d[k], dest), sel(kept_r[k], dest))
        # what each worker read back against the reference's answer
        rb_a, rf_a, rb_v, rf_v, rb_g, rf_g = [], [], [], [], [], []
        for w in range(W):
            d_ = 0
            for e in sent[w]:
                if not go_rule(e['row'], a.cap_t, a.trunc, DONE_BRANCH)[0]:
                    continue
                r_ = ref.get((w, d_))
                d_ += 1
                if not e['replied'] or r_ is None:
                    continue
                rb_a.append(e['actions'])
                rf_a.append(np.asarray(r_['actions'], np.int64))
                rb_v.append(e['value'])
                rf_v.append(float(np.float32(r_['value'])))
                rb_g.append(e['vg'])
                rf_g.append(float(np.float32(r_['vg'])))
        g1['worker_actions'] = bitdiff(np.array(rb_a, np.int64), np.array(rf_a, np.int64))
        g1['worker_value'] = bitdiff(np.array(rb_v), np.array(rf_v))
        g1['worker_vg'] = bitdiff(np.array(rb_g), np.array(rf_g))
        for k in state_d:
            g1[f'state_{k}'] = bitdiff(state_d[k], state_r[k])
        if bank_d is not None:
            for k in bank_d:
                g1[f'bank_{k}'] = bitdiff(bank_d[k], bank_rr[k])
            reqs = [e for v in sent.values() for e in v if e['cf'] and e['replied']]
            ok_cf = sum(1 for e in reqs if e['cf_after'] == (0 if go_rule(e['row'], a.cap_t, a.trunc, DONE_BRANCH)[0]
                                                            else e['cf']))
            g1['cf_cleared'] = dict(n=len(reqs), equal=int(ok_cf))
            # (Codex 18-5) every generation of every bank row, as read back after its write, against the reference's
            gr_ = reference.gens
            n_g = eq_g = 0
            for row_ in sorted(set(gens_d) | set(gr_)):
                a_, b_ = gens_d.get(row_, []), gr_.get(row_, [])
                n_g += max(len(a_), len(b_))
                if len(a_) == len(b_):
                    eq_g += sum(all_equal({k: bitdiff(x[k], y[k]) for k in STATE}) for x, y in zip(a_, b_))
            g1['bank_generations'] = dict(n=int(n_g), equal=int(eq_g))
        g1['all_bit_equal'] = all_equal(g1)
    else:
        g1['all_bit_equal'] = False
        g1['missing'] = [k for k in keys if k not in drv][:10]
        g1['extra'] = [k for k in drv if k not in ref][:10]
    res['gate1'] = g1
    res['gate3']['decisions_reference'] = len(ref)
    res['gate3']['pass'] = bool(not recon and not errors and res['closed'] and not res['workers_hung']
                                and res['gate3']['swaps_done'] == a.swaps and g1['same_keys'] and all(hits.values()))

    # ---- gate 2: the driver's actions forced in other splits
    if a.alt:
        assert a.swaps == 0, 'gate 2 splits need --swaps 0'
        forced = {x: np.asarray(drv[x]['actions'], np.int64) for x in keys}
        alts = a.alt.split(',')
        runs = {s_: reference(s_, forced)[0] for s_ in ['same'] + alts}
        cov = {s_: int(sum(np.array_equal(np.asarray(runs[s_][x]['actions'], np.int64), forced[x]) for x in keys))
               for s_ in runs}

        def arr(o_):
            return {k: np.stack([np.asarray(o_[x][k]) for x in keys]) for k in OUTS}
        D_ = arr(drv)
        S_ = arr(runs['same'])
        g2 = dict(records=len(keys), coverage=cov,
                  same_forced_vs_driver={k: bitdiff(D_[k].astype(np.float32), S_[k].astype(np.float32))
                                         for k in ('logp', 'value', 'vg')})
        line = numstats(arr(runs[alts[0]]), arr(runs[alts[-1]])) if len(alts) > 1 else numstats(S_, arr(runs[alts[0]]))
        g2['line_python'] = line
        for s_ in alts:
            g2[s_] = numstats(D_, arr(runs[s_]), line)
        g2['pass'] = bool(all(c == len(keys) for c in cov.values()) and all_equal(g2['same_forced_vs_driver'])
                          and all(g2[s_]['pass'] for s_ in alts))
        res['gate2'] = g2
    res['pass'] = bool(res['gate3']['pass'] and g1['all_bit_equal'] and res.get('gate2', {}).get('pass', True))
    req = [x for x in a.require.split(',') if x]   # (Codex 18-4: a gate not run is not a gate passed)
    miss = []
    if '1' in req and not g1.get('decisions_ref'):
        miss.append('gate 1: no decisions')
    if '2' in req and (len(set(a.alt.split(',')) - {''}) < 2 or not res.get('gate2', {}).get('records')):
        miss.append('gate 2: --alt with two different splits over a non-zero number of records')
    if '3' in req and not res['gate3'].get('records_sent'):
        miss.append('gate 3: no records')
    res['required'] = dict(gates=req, missing=miss)
    res['pass'] = res['pass'] and not miss
    _out(res, a)


def _out(res, a):
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)
    os._exit(0 if res.get('pass') else 1)


if __name__ == '__main__':
    main()
