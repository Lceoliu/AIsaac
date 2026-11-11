"""BM1 P2: one learner on two cards (train_tok --learner-devices MAIN,SECOND). Data parallel inside a minibatch step:
each PPO part is cut in two halves of whole chunks (rows balanced, the teachers' records with the second half), the
first runs on the main card (the learner thread, the learner's own modules), the second on the second card (a thread of
its own, replicas of the trained modules). Every part's loss is already a share of its minibatch's means (its rows'
sums over the minibatch's denominators), so the two halves' gradients add up to the part's: the same algorithm, batch,
losses and budgets, the sums in another order. The halves' history blocks are drawn once for the whole part
(block_learner.prepare_split): the same draws as on one card.

Per minibatch step: the replicas' gradients zeroed with the main optimisers' (zero_grad); the halves run; at the top of
train_tok's step_params (the one parameter-update entry) the second card's gradients are added into the main card's
(reduce: one flat peer copy per dtype); the optimisers step on the main card only (Adam state lives there alone); then
the new parameters are copied to the replicas (broadcast: one flat peer copy per dtype). The replicas also get the
parameters at each update's start (the learner's weights may change between updates: restore, choice heads untouched).

Replicas hold no state a forward changes (the check, EXPERIMENTS BM1 P2): the trained modules (TokPolicy, the
privileged critic, the aux head) have three buffers (structured_input coordinates, transformer_policy frequency /
room_shifts), all constants written at construction; no dropout (p = 0 everywhere), no batch norm, no running
statistics, no counters or caches on the modules written in forward. What forward does write lives outside the modules:
block_paged's counters (UPD, TIMING, HIST: under one lock, block_paged._UPD_LOCK) and the compiled kernels' caches
(Dynamo: compiled per device at the warm-up, its compile lock for any later compile). ISAAC_P2_CHECK=1 (debug, off by
default) hashes the replicas' parameters after each broadcast against the main card's (a synchronisation per step).

Streams: the learner thread's current stream on the second card is this object's stream (bind(), once, in the learner
thread), so every copy it makes there (inputs, the reduce, the broadcast) is ordered with the second card's work by
PyTorch's cross-device copy (a two-way barrier between the source's and the destination's current streams); the
second card's thread runs on the same stream.

A failure on the second card's thread is re-raised on the learner thread at the part's join (the run fails through the
learner's failure path, the card and the part in the message); there is no fallback to one card."""
import os
import queue
import threading
import time

import torch

CHECK = os.environ.get('ISAAC_P2_CHECK', '0') == '1'


class TwoCard:
    def __init__(self, main_dev, dev2, pairs):
        """pairs: [(main module, its replica on dev2)] (the replicas built like the main modules, the same parameter
        order). The main parameters synchronised: each pair's parameters()."""
        self.main_dev, self.dev2 = torch.device(main_dev), torch.device(dev2)
        self.pairs = list(pairs)
        self.mp = [p for m, _ in self.pairs for p in m.parameters()]
        self.rp = [p for _, r in self.pairs for p in r.parameters()]
        if len(self.mp) != len(self.rp) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(self.mp, self.rp)):
            raise RuntimeError('P2: the replicas do not match the learner modules parameter for parameter')
        for _, r in self.pairs:
            r.train()
        self.stream = torch.cuda.Stream(device=self.dev2)
        self.bound = None
        self.q = queue.Queue()
        self.res = queue.Queue()
        self.thread = None
        self.busy = False
        self.stats = dict(reduce_s=0.0, bcast_s=0.0, join_wait_s=0.0, checks=0)

    # ---- the learner thread
    def bind(self):
        """In the learner thread (once): its current stream on the second card is this object's."""
        me = threading.get_ident()
        if self.bound != me:
            cur = torch.cuda.current_device()
            torch.cuda.set_stream(self.stream)   # (it also makes the stream's device current: set back below)
            torch.cuda.set_device(cur)
            self.bound = me
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._worker, name='learner-p2', daemon=True)
            self.thread.start()

    def to2(self, x):
        """x (tensors in dicts / lists / tuples) on the second card (non-blocking, from the learner thread)."""
        if torch.is_tensor(x):
            return x.to(self.dev2, non_blocking=True)
        if isinstance(x, dict):
            return {k: self.to2(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self.to2(v) for v in x)
        return x

    def to_main(self, x):
        if torch.is_tensor(x):
            return x.to(self.main_dev, non_blocking=True)
        return x

    @staticmethod
    def _groups(ts):
        g = {}
        for i, t in enumerate(ts):
            g.setdefault(t.dtype, []).append(i)
        return g

    @torch.no_grad()
    def broadcast(self):
        """The main card's parameters onto the replicas: per dtype one flat buffer, one peer copy."""
        t0 = time.perf_counter()
        for _, idx in self._groups(self.mp).items():
            flat = torch.cat([self.mp[i].detach().reshape(-1) for i in idx]).to(self.dev2, non_blocking=True)
            views = [v.view_as(self.rp[i]) for v, i in zip(torch.split(flat, [self.rp[i].numel() for i in idx]), idx)]
            torch._foreach_copy_([self.rp[i].data for i in idx], views)
        self.stats['bcast_s'] += time.perf_counter()-t0
        if CHECK:
            self.check()

    @torch.no_grad()
    def reduce(self):
        """The second card's gradients added into the main card's (main.grad += second's; a main parameter without a
        gradient takes the second card's as it is): per dtype one flat buffer, one peer copy."""
        t0 = time.perf_counter()
        have = [i for i, p in enumerate(self.rp) if p.grad is not None]
        for _, idx_ in self._groups([self.rp[i] for i in have]).items():
            idx = [have[j] for j in idx_]
            flat = torch.cat([self.rp[i].grad.reshape(-1) for i in idx]).to(self.main_dev, non_blocking=True)
            views = [v.view_as(self.mp[i]) for v, i in zip(torch.split(flat, [self.rp[i].numel() for i in idx]), idx)]
            add_to, add_from = [], []
            for v, i in zip(views, idx):
                p = self.mp[i]
                if p.grad is None:
                    p.grad = v.clone()
                else:
                    add_to.append(p.grad)
                    add_from.append(v)
            if add_to:
                torch._foreach_add_(add_to, add_from)
        self.stats['reduce_s'] += time.perf_counter()-t0

    def zero_grad(self):
        for p in self.rp:
            p.grad = None

    @torch.no_grad()
    def check(self):
        """(ISAAC_P2_CHECK=1) the replicas' parameters against the main card's: an integer hash of each side's bytes
        (int32 words times their position's weight, summed in int64: order-free, exact); a mismatch raises."""
        def h(ts, dev):
            out = torch.zeros((), dtype=torch.int64, device=dev)
            for t in ts:
                iv = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()]
                w = t.detach().reshape(-1).view(iv).to(torch.int64)
                k = torch.arange(w.numel(), device=dev, dtype=torch.int64) % 1000003 + 1
                out += (w * k).sum()
            return int(out)
        a, b = h(self.mp, self.main_dev), h(self.rp, self.dev2)
        self.stats['checks'] += 1
        if a != b:
            raise RuntimeError(f'P2 check: the replicas on {self.dev2} differ from the main card after the broadcast '
                               f'(hash {a} vs {b})')

    def run(self, fn, label):
        """fn() on the second card's thread (asynchronous); join() waits for it."""
        if self.busy:
            raise RuntimeError('P2: a second-card job is already running')
        self.busy = True
        self.q.put((fn, label))

    def join(self):
        """The running job's result; its exception re-raised here (the card and the job's label in the message)."""
        t0 = time.perf_counter()
        ok, val, label = self.res.get()
        self.stats['join_wait_s'] += time.perf_counter()-t0
        self.busy = False
        if not ok:
            raise RuntimeError(f'P2 second learner card {self.dev2} failed ({label}): {val!r}') from val
        return val

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.q.put(None)
            self.thread.join(10.0)

    # ---- the second card's thread
    def _worker(self):
        try:
            import ctypes
            ctypes.CDLL(None).prctl(15, b'learner-p2', 0, 0, 0)
        except Exception:
            pass
        torch.cuda.set_device(self.dev2)
        torch.cuda.set_stream(self.stream)
        while True:
            item = self.q.get()
            if item is None:
                return
            fn, label = item
            del item
            try:
                out = (True, fn(), label)
            except BaseException as e:   # (re-raised by join on the learner thread)
                out = (False, e, label)
            del fn   # (the job's tensors and graph released before the learner thread goes on)
            self.res.put(out)
            del out
