"""BM1 C: K training shards as one training run (torch.distributed). Each shard (rank) collects its own rollouts and
computes its own gradients; at every optimiser step the three optimisers' gradients are averaged over the ranks
(weighted by each rank's minibatch size: the global minibatch's mean), so every rank takes the same step and keeps
bit-identical parameters and Adam states. Off (None) unless ISAAC_SHARD_WORLD > 1: a run without it is unchanged.

Environment (set per trainer by ops/bm1/shardk.sh):
  ISAAC_SHARD_WORLD   K (the number of shards); ISAAC_SHARD_RANK this trainer's rank 0..K-1
  MASTER_ADDR / MASTER_PORT   the rendezvous (127.0.0.1 and a lane port: the K trainers share one container)
  ISAAC_SHARD_DIR     a directory every rank sees (pid files: a rank that ends is noticed by the others)
  ISAAC_SHARD_TIMEOUT seconds a collective may wait (default 600)
  ISAAC_SHARD_CHECK   'update' (default: the state hash compared after every update) or 'step' (after every step)
  ISAAC_SHARD_FLIP    test hook 'rank:update:step': that rank flips one bit of one parameter after that step

Two process groups: the learner thread's (NCCL for CUDA tensors, gloo on the CPU: tests) for the gradients, the
learner's agreements and the hashes; the main thread's control group (gloo) for the collect loop's agreement to go
on (the two threads never share a group: a group's collectives must come in one order).
Every rank must make the same collectives in the same order: the step count of an update is agreed (a rank with fewer
minibatches takes zero-weight steps), the learning rate comes from the global progress, the stop is agreed."""
import os
import threading
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def _int_view(t):
    iv = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()]
    return t.detach().reshape(-1).view(iv).to(torch.int64)


@torch.no_grad()
def state_hash(tensors, device):
    """An integer hash of the tensors' bytes (words times their position's weight, summed in int64, per tensor its own
    multiplier): exact, a one-bit change always changes it."""
    out = torch.zeros((), dtype=torch.int64, device=device)
    for j, t in enumerate(tensors):
        if t is None or t.numel() == 0:
            continue
        w = _int_view(t.to(device))
        k = torch.arange(w.numel(), device=device, dtype=torch.int64) % 1000003 + 1
        out += (w * k).sum() * (2 * j + 1)
    return out


class ShardSync:
    def __init__(self, rank, world, device, timeout=600.0, check='update', flip=None, shard_dir=None):
        self.rank, self.world = int(rank), int(world)
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.timeout = float(timeout)
        self.check_mode = check
        self.flip = flip   # (rank, update, step) or None
        self.dir = shard_dir
        self.stats = dict(reduce_s=0.0, wait_s=0.0, steps=0, pad_steps=0, rows=0.0, checks=0, mismatches=0, restores=0)
        self.peer_lost = None
        self.pg = self.ctrl = None
        self._watch = None

    # ---- set-up
    @classmethod
    def from_env(cls, device):
        world = int(os.environ.get('ISAAC_SHARD_WORLD', '1') or 1)
        if world <= 1:
            return None
        flip = os.environ.get('ISAAC_SHARD_FLIP', '')
        return cls(int(os.environ['ISAAC_SHARD_RANK']), world, device,
                   timeout=float(os.environ.get('ISAAC_SHARD_TIMEOUT', '600')),
                   check=os.environ.get('ISAAC_SHARD_CHECK', 'update'),
                   flip=tuple(int(v) for v in flip.split(':')) if flip else None,
                   shard_dir=os.environ.get('ISAAC_SHARD_DIR') or None)

    def init(self):
        """The process groups (env:// rendezvous), the pid file and the watchdog of the others'."""
        to = timedelta(seconds=self.timeout)
        if self.cuda:
            torch.cuda.set_device(self.device)
        dist.init_process_group('nccl' if self.cuda else 'gloo', init_method='env://', rank=self.rank,
                                world_size=self.world, timeout=to, **(dict(device_id=self.device) if self.cuda else {}))
        self.pg = dist.group.WORLD
        self.ctrl = dist.new_group(backend='gloo', timeout=to)
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, f'rank{self.rank}.pid'), 'w') as f:
                f.write(str(os.getpid()))
            self.barrier_ctrl()   # (every pid file written before anyone watches)
            self._watch = threading.Thread(target=self._watchdog, name='shard-watch', daemon=True)
            self._watch.start()
        return self

    def _peer_pids(self):
        out = {}
        for r in range(self.world):
            if r == self.rank:
                continue
            try:
                out[r] = int(open(os.path.join(self.dir, f'rank{r}.pid')).read())
            except (OSError, ValueError):
                pass
        return out

    def _watchdog(self):
        """Every second: a peer whose process is gone aborts this rank's collectives (a learner blocked in one
        raises instead of hanging; the run then ends through its failure path)."""
        pids = self._peer_pids()
        while self.peer_lost is None:
            time.sleep(1.0)
            for r, pid in pids.items():
                try:   # (gone, or a zombie its parent has not reaped yet: ended either way)
                    st = open(f'/proc/{pid}/stat').read()
                    dead = st[st.rfind(')') + 2] in 'ZX'
                except OSError:
                    dead = True
                if dead:
                    self.peer_lost = (r, pid, time.time())
            if self.peer_lost is not None:
                print(f'shard {self.rank}: rank {self.peer_lost[0]} (pid {self.peer_lost[1]}) ended: this rank\'s '
                      'collectives aborted', flush=True)
                self.abort()

    def abort(self):
        for g in (self.pg, self.ctrl):
            if g is None:
                continue
            try:
                dist.distributed_c10d._abort_process_group(g)
                continue
            except Exception:
                pass
            for dev in (torch.device('cuda'), torch.device('cpu')):
                try:
                    g._get_backend(dev).abort()
                except Exception:
                    pass

    def close(self):
        if self.peer_lost is None and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    # ---- the learner thread's collectives
    def _t(self, values, dtype=torch.float64):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def sum(self, values):
        t = self._t(values)
        dist.all_reduce(t, group=self.pg)
        return t.tolist()

    def max_int(self, value):
        t = self._t([int(value)], torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.pg)
        return int(t.item())

    @torch.no_grad()
    def reduce_grads(self, params, weight):
        """Every rank's gradients of `params` (one list: the optimisers' parameters in a fixed order; a missing
        gradient counts as zero) replaced by sum_r(weight_r * g_r) / sum_r(weight_r): per dtype one flat buffer, the
        weights' sum in the float32 one's last element. The time waiting for the slowest rank (a one-element all_reduce
        first) and the reduce's own time are counted apart."""
        t0 = time.perf_counter()
        if self.cuda:
            torch.cuda.current_stream().synchronize()
        tw = self._t([1.0], torch.float32)
        dist.all_reduce(tw, group=self.pg)   # (the barrier: every rank's backward done)
        if self.cuda:
            torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        self.stats['wait_s'] += t1 - t0
        groups = {}
        for i, p in enumerate(params):
            groups.setdefault(p.dtype, []).append(i)
        if torch.float32 not in groups:
            groups[torch.float32] = []
        wsum = None
        for dt, idx in groups.items():
            parts = [(params[i].grad if params[i].grad is not None else torch.zeros_like(params[i])).reshape(-1)
                     for i in idx]
            if dt == torch.float32:
                parts = parts + [self._t([1.0], torch.float32)]
            flat = torch.cat(parts) if parts else self._t([1.0], torch.float32)
            flat.mul_(float(weight))
            if dt == torch.float32:
                flat[-1] = float(weight)
            dist.all_reduce(flat, group=self.pg)
            if dt == torch.float32:
                wsum = flat[-1].clone()
                flat = flat[:-1]
            groups[dt] = (idx, flat)
        for dt, (idx, flat) in groups.items():
            if not idx:
                continue
            flat = flat / wsum.clamp(min=1e-30).to(flat.dtype)
            for v, i in zip(torch.split(flat, [params[i].numel() for i in idx]), idx):
                p = params[i]
                if p.grad is None:
                    p.grad = v.view_as(p).clone()
                else:
                    p.grad.copy_(v.view_as(p))
        self.stats['rows'] += float(wsum.item())   # (a synchronisation: the reduce's end, the global rows)
        self.stats['reduce_s'] += time.perf_counter() - t1
        self.stats['steps'] += 1

    # ---- consistency
    def _state(self, models, opts, extra_arrays):
        ts = [p for m in models if m is not None for p in m.parameters()]
        for o in opts:
            if o is None:
                continue
            for group in o.param_groups:
                for p in group['params']:
                    st = o.state.get(p, {})
                    for k in sorted(st):
                        if torch.is_tensor(st[k]):
                            ts.append(st[k])
        for a in extra_arrays:
            ts.append(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)))
        return ts

    @torch.no_grad()
    def check(self, models, opts, extra_arrays=(), restore=True):
        """The ranks' state hashes compared (all_gather); a mismatch: rank 0's state broadcast to every rank (the
        parameters, the Adam states, the extra arrays written back by the caller from the returned list) and counted.
        Returns (mismatch, the extra arrays as rank 0 has them)."""
        h = state_hash(self._state(models, opts, extra_arrays), self.device)
        hs = [torch.zeros_like(h) for _ in range(self.world)]
        dist.all_gather(hs, h, group=self.pg)
        self.stats['checks'] += 1
        vals = [int(x.item()) for x in hs]
        if len(set(vals)) == 1:
            return False, list(extra_arrays)
        self.stats['mismatches'] += 1
        print(f'shard {self.rank}: state hashes differ {vals}: rank 0\'s broadcast', flush=True)
        if not restore:
            return True, list(extra_arrays)
        for t in self._state(models, opts, ()):
            buf = t if t.device == self.device else t.to(self.device)
            dist.broadcast(buf, 0, group=self.pg)
            if buf is not t:
                t.copy_(buf)
        out = []
        for a in extra_arrays:
            b = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(self.device)
            dist.broadcast(b, 0, group=self.pg)
            out.append(b.cpu().numpy())
        self.stats['restores'] += 1
        return True, out

    @torch.no_grad()
    def broadcast_state(self, models, opts):
        """Rank 0's parameters and optimiser states to every rank (the start: the ranks' seeds differ)."""
        for t in self._state(models, opts, ()):
            buf = t if t.device == self.device else t.to(self.device)
            dist.broadcast(buf, 0, group=self.pg)
            if buf is not t:
                t.copy_(buf)

    def maybe_flip(self, update, step, params):
        """(test hook ISAAC_SHARD_FLIP) one bit of the first parameter's first element flipped on one rank."""
        if self.flip and self.flip[0] == self.rank and self.flip[1] == update and self.flip[2] == step:
            p = params[0]
            v = p.data.reshape(-1)[:1].view({4: torch.int32, 2: torch.int16, 8: torch.int64}[p.element_size()])
            v ^= 1
            print(f'shard {self.rank}: ISAAC_SHARD_FLIP: one bit of a parameter flipped (update {update}, step {step})',
                  flush=True)

    # ---- the main thread's control group
    def stop_requested(self):
        """(ops/bm1/shardk.sh) a file 'stop' in ISAAC_SHARD_DIR: the ranks end together at the next collect (a SIGINT
        to every trainer could land on one rank inside the agreement and leave the others waiting in it)."""
        return bool(self.dir) and os.path.exists(os.path.join(self.dir, 'stop'))

    def all_true(self, flag):
        t = torch.tensor([1 if flag else 0], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.ctrl)
        return bool(t.item())

    def barrier_ctrl(self):
        t = torch.zeros(1)
        dist.all_reduce(t, group=self.ctrl)


def auxnorm_moments(state_raw, ret, m_ev):
    """AuxNorm.update's batch as sums: [b, sum x (S), sum x^2 (S), per scale count (K), per scale sum v^2 (K)]."""
    x = state_raw.astype(np.float64)
    K = ret.shape[-1]
    cnt = np.zeros(K)
    sq = np.zeros(K)
    for k in range(K):
        v = ret[..., k][m_ev[..., k]].astype(np.float64)
        cnt[k] = v.size
        sq[k] = float((v * v).sum())
    return np.concatenate(([float(len(x))], x.sum(0) if len(x) else np.zeros(x.shape[1]),
                           (x * x).sum(0) if len(x) else np.zeros(x.shape[1]), cnt, sq))


def auxnorm_apply(norm, m):
    """The global batch's sums applied as AuxNorm.update would apply the concatenated batch."""
    S = len(norm.mean)
    K = len(norm.rn)
    b = m[0]
    if b > 0:
        mu = m[1:1 + S] / b
        var = np.maximum(m[1 + S:1 + 2 * S] / b - mu * mu, 0.0)
        d = mu - norm.mean
        tot = norm.n + b
        norm.m2 += b * var + d * d * norm.n * b / tot
        norm.mean += d * b / tot
        norm.n = tot
    norm.rn += m[1 + 2 * S:1 + 2 * S + K]
    norm.rm2 += m[1 + 2 * S + K:1 + 2 * S + 2 * K]
