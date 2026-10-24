"""Non-recurrent memory sources and versioned source-gradient transport.

The only persistent history is raw observation data. Spatial never consumes H;
reader outputs are never written into a future source. SourcePipeline decouples
source versions from Reader steps; direct and replay modes remain as references.
This module is opt-in; the legacy policy is unchanged.
"""
from dataclasses import dataclass
from contextlib import nullcontext
from collections import OrderedDict
import itertools
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_episodes = itertools.count()
# Opt-in synchronized attribution; disabled in production throughput measurements.
PROFILE_MEMORY = os.environ.get('ISAAC_MEMORY_PROFILE') == '1'
MEMORY_TIMES = {}
MEMORY_COUNTS = {}
PROFILE_ACTOR = os.environ.get('ISAAC_ACTOR_PROFILE')
ACTOR_TIMES = {}


def actor_part(name, start=0):
    if not PROFILE_ACTOR:
        return 0.0
    torch.cuda.current_stream().synchronize()
    now = time.perf_counter()
    if start:
        ACTOR_TIMES[name] = ACTOR_TIMES.get(name, 0.0) + now-start
    return now


def memory_tick():
    if PROFILE_MEMORY:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        return time.perf_counter()
    return 0.0


def memory_part(name, start):
    end = memory_tick()
    if PROFILE_MEMORY:
        MEMORY_TIMES[name] = MEMORY_TIMES.get(name, 0.0) + end - start
    return end



@dataclass(frozen=True)
class RawHistory:
    uid: tuple
    blocks: tuple = ()
    tail: tuple = ()
    length: int = 0
    # Lineage of a fork, nearest first: ((parent uid, frames shared), ...).
    # Frame t < shared belongs to that ancestor: sources key it by the
    # ancestor (one encoding, one graph), never by the branch.
    origin: tuple = ()

    @classmethod
    def empty(cls):
        return cls((os.getpid(), next(_episodes)))

    def key(self, t):
        """The source key of frame t: (owning episode, t)."""
        for uid, shared in reversed(self.origin):   # farthest ancestor shares the fewest frames
            if t < shared: return uid, t
        return self.uid, t

    def own_start(self):
        """First frame this episode owns (the frames before it are an ancestor's)."""
        return self.origin[0][1] if self.origin else 0

    def append(self, row, pending, block):
        # Keep one row's bytes, not a view that lets a long-lived episode pin
        # every other episode's observations from the same Actor batch.
        r = np.frombuffer(row.tobytes(),dtype=row.dtype)
        p = np.array(pending, copy=True).reshape(1, -1)
        r.flags.writeable = p.flags.writeable = False
        return self.append_owned(r,p,block)

    def append_owned(self,r,p,block):
        """Append views of an immutable batch owned by the collector."""
        tail = self.tail + ((r, p),)
        blocks = self.blocks
        if len(tail) == block:
            rr = np.concatenate([v[0].view(np.uint8).reshape(len(v[0]),-1) for v in tail]).view(r.dtype).reshape(-1)
            pp = np.concatenate([v[1] for v in tail])
            rr.flags.writeable = pp.flags.writeable = False
            blocks, tail = blocks + ((rr, pp),), ()
        return RawHistory(self.uid, blocks, tail, self.length + 1, self.origin)

    def fork(self):
        return RawHistory((os.getpid(), next(_episodes)), self.blocks, self.tail, self.length,
                          ((self.uid, self.length),) + self.origin)

    def arrays(self):
        parts = self.blocks + self.tail
        rows=np.concatenate([r.view(np.uint8).reshape(len(r),-1) for r,p in parts]).view(parts[0][0].dtype).reshape(-1)
        return rows,np.concatenate([p for r,p in parts])


def share_prefix(h, donor, block):
    """h with the frames it shares with `donor` (a history holding the same
    owners' frames) referencing donor's arrays instead of its own copies.

    A branch record crossing a process boundary arrives with a pickled copy of
    its whole prefix. Rows are byte-identical, so only storage changes: whole
    shared blocks and the shared rows of a still-open tail block. A block that
    mixes ancestor and own rows stays the record's (at most block-1 rows)."""
    if donor is None: return h, 0
    # Leading frames with the same source key are the same owner's frames
    # (an ancestor's, or this episode's own frames held by an earlier record).
    limit = min(h.length, donor.length)
    bounds = sorted({0, limit} | {n for _, n in h.origin + donor.origin if n < limit})
    shared = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if h.key(lo) != donor.key(lo) or h.key(hi - 1) != donor.key(hi - 1): break
        shared = hi
    if shared == 0: return h, 0
    whole = min(shared // block, len(h.blocks), len(donor.blocks))
    blocks = donor.blocks[:whole] + h.blocks[whole:]
    tail = h.tail
    referenced = whole * block
    if whole == len(h.blocks) and shared > whole * block:   # the record's open block holds shared rows
        n = min(shared - whole * block, len(tail))
        if whole < len(donor.blocks):
            rr, pp = donor.blocks[whole]
            rows = tuple((rr[i:i + 1], pp[i:i + 1]) for i in range(n))
        else:
            rows = donor.tail[:n] if len(donor.tail) >= n else None
        if rows is not None:
            tail = rows + tail[n:]; referenced += n
    return RawHistory(h.uid, blocks, tail, h.length, h.origin), referenced


def profile_history_storage(role):
    """Opt-in retained raw storage / allocator audit after observed host OOM."""
    import gc,json,ctypes
    from pathlib import Path
    before=Path('/proc/self/status').read_text().split('VmRSS:')[1].splitlines()[0].strip()
    storage={};histories=0
    for h in gc.get_objects():
        if type(h) is not RawHistory:continue
        histories+=1
        for r,p in h.blocks+h.tail:
            for a in (r,p):
                while isinstance(a,np.ndarray) and a.base is not None:a=a.base
                storage[id(a)]=a.nbytes if isinstance(a,np.ndarray) else len(a)
    unreachable=gc.collect()
    after_gc=Path('/proc/self/status').read_text().split('VmRSS:')[1].splitlines()[0].strip()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    after=Path('/proc/self/status').read_text().split('VmRSS:')[1].splitlines()[0].strip()
    print('HISTORY_STORAGE '+json.dumps(dict(pid=os.getpid(),role=role,histories=histories,
          raw_storage_bytes=sum(storage.values()),allocations=len(storage),rss_before=before,
          rss_after_gc=after_gc,unreachable=unreachable,rss_after=after)),flush=True)


def memory_account(role, step, groups, extra=None):
    """Opt-in (ISAAC_MEMORY_ACCOUNT=1) host-memory audit: one MEMORY_ACCOUNT line.

    Walks the named containers (dicts/lists/tuples/objects, bounded depth) and
    sums distinct numpy bases, CPU tensor storages (pinned separately) and the
    raw-history arrays of every RawHistory; frames = sum over episodes of the
    longest prefix seen. Shared arrays are counted once across all groups.
    No tracemalloc, no gc scan; RSS comes from /proc/self/status."""
    import json
    seen_obj, seen_base = set(), set()
    out = {}

    def base(a):
        while isinstance(a, np.ndarray) and a.base is not None: a = a.base
        return a

    def walk(x, acc, depth):
        if id(x) in seen_obj or depth > 6: return
        seen_obj.add(id(x))
        if isinstance(x, RawHistory):
            acc['episodes'][x.uid] = max(acc['episodes'].get(x.uid, 0), x.length)
            for r, p in x.blocks + x.tail:
                for arr in (r, p):
                    b = base(arr)
                    if id(b) not in seen_base:
                        seen_base.add(id(b)); acc['raw_bytes'] += b.nbytes if isinstance(b, np.ndarray) else len(b)
        elif isinstance(x, np.ndarray):
            b = base(x)
            if id(b) not in seen_base:
                seen_base.add(id(b))
                acc['numpy_bytes'] += b.nbytes if isinstance(b, np.ndarray) else len(b)
                if x.dtype == object:
                    for v in x.reshape(-1): walk(v, acc, depth + 1)
        elif torch.is_tensor(x):
            if x.device.type == 'cpu':
                s = x.untyped_storage()
                if s.data_ptr() not in seen_base:
                    seen_base.add(s.data_ptr())
                    acc['pinned_bytes' if x.is_pinned() else 'cpu_tensor_bytes'] += s.nbytes()
        elif isinstance(x, dict):
            acc['container_bytes'] += sys.getsizeof(x)
            for k, v in x.items(): walk(k, acc, depth + 1); walk(v, acc, depth + 1)
        elif isinstance(x, (list, tuple, set, frozenset, collections_deque)):
            acc['container_bytes'] += sys.getsizeof(x)
            for v in x: walk(v, acc, depth + 1)
        elif isinstance(x, (bytes, bytearray, memoryview)):
            acc['bytes_objects'] += len(x)
        elif hasattr(x, '__dict__') and not isinstance(x, (type, torch.nn.Module)) and not callable(x):
            walk(vars(x), acc, depth + 1)

    import sys
    from collections import deque as collections_deque
    for name, value in groups.items():
        acc = dict(raw_bytes=0, numpy_bytes=0, cpu_tensor_bytes=0, pinned_bytes=0, container_bytes=0,
                   bytes_objects=0, episodes={})
        walk(value, acc, 0)
        acc['frames'] = sum(acc['episodes'].values()); acc['episodes'] = len(acc['episodes'])
        out[name] = acc
    status = {}
    try:
        for line in open('/proc/self/status'):
            key = line.split(':')[0]
            if key in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem'): status[key] = int(line.split()[1]) * 1024
    except OSError:
        pass
    host = {}
    if torch.cuda.is_available() and hasattr(torch.cuda, 'host_memory_stats'):
        try:
            stats = torch.cuda.host_memory_stats()
            host = {k: v for k, v in stats.items() if 'allocated_bytes' in k or 'reserved_bytes' in k}
        except Exception as error:   # (an older runtime without host allocator statistics)
            host = dict(error=repr(error))
    print('MEMORY_ACCOUNT ' + json.dumps(dict(role=role, pid=os.getpid(), step=step, time=time.time(), proc=status,
                                               torch_host=host, groups=out, **(extra or {}))), flush=True)


def trim_host_heap(role):
    """Return already-freed glibc arenas; no live history or activation is dropped."""
    import ctypes
    start=time.perf_counter();ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(f'HOST_HEAP_TRIM role={role} seconds={time.perf_counter()-start:.6f}',flush=True)


def pack_histories(contexts, *, previous=None, shared=False):
    """IPC: one longest immutable prefix per episode plus (episode, length) refs.

    Older prefixes' partial blocks otherwise serialize a second copy of almost
    every ROW alongside the later completed blocks. No observation is discarded.
    """
    latest, slots, locations = {}, {}, {}
    refs = np.full((len(contexts), len(contexts[0]), 2), -1, np.int32)
    for w, row in enumerate(contexts):
        for t, h in enumerate(row):
            if h is None:
                continue
            if h.uid not in slots:
                slots[h.uid] = len(slots)
            if h.uid not in latest or h.length > latest[h.uid].length:
                latest[h.uid] = h
            refs[w,t] = slots[h.uid], h.length
            if shared:locations.setdefault(h.uid,{})[h.length]=(w,t)
    if previous is None:return list(latest.values()), refs
    deltas=[];lengths={}
    for uid,h in latest.items():
        start=previous.get(uid,0);parts=[];at=0
        for rr,pp in h.blocks+h.tail:
            stop=at+len(rr)
            if stop>start:
                begin=max(start,at)
                if shared:
                    # Shared rollout rows remain immutable until the learner's
                    # acknowledgement. Only older/fork-only rows need payloads.
                    index=locations[uid]
                    lo=begin
                    while lo<stop:
                        available=lo in index;hi=lo+1
                        while hi<stop and (hi in index)==available:hi+=1
                        if available:parts.append(('shared',np.asarray([index[t] for t in range(lo,hi)],np.int32)))
                        else:parts.append((rr[lo-at:hi-at],pp[lo-at:hi-at]))
                        lo=hi
                else:parts.append((rr[begin-at:],pp[begin-at:]))
            at=stop
        end=max(start,h.length)
        deltas.append((uid,start,end,tuple(parts)));lengths[uid]=end
    previous.clear();previous.update(lengths)
    return 'delta',deltas,refs


def unpack_histories(packed, block, *, previous=None, rows=None, pending=None, latest_only=False):
    if len(packed)==3:
        _,deltas,refs=packed
        histories=[];live={}
        for uid,start,end,parts in deltas:
            prior=previous[uid] if start else RawHistory(uid,(),(),0)
            if parts:
                joined=prior.tail+tuple(parts)
                dtype=rows.dtype if any(isinstance(r,str) for r,p in joined) else joined[0][0].dtype
                count=sum(len(p) for r,p in joined)
                rr=np.empty(count,dtype=dtype)
                pending_template=pending if isinstance(joined[0][0],str) else joined[0][1]
                pp=np.empty((count,pending_template.shape[-1]),dtype=pending_template.dtype)
                dest=rr.view(np.uint8).reshape(count,-1)
                import itertools
                at=0
                for shared,group in itertools.groupby(joined,key=lambda x:isinstance(x[0],str)):
                    group=list(group)
                    if shared:
                        w,t=np.concatenate([p for r,p in group]).T
                        ix=w.astype(np.int64)*rows.shape[1]+t
                        copy_end=at+len(ix)
                        # One owned copy, not advanced indexing followed by a
                        # second rollout-sized concatenation. Coordinates are
                        # produced internally by pack_histories and in bounds.
                        np.take(rows.view(np.uint8).reshape(-1,dest.shape[1]),ix,axis=0,out=dest[at:copy_end],mode='clip')
                        np.take(pending.reshape(-1,pending.shape[-1]),ix,axis=0,out=pp[at:copy_end],mode='clip')
                        at=copy_end
                    else:
                        for r,p in group:
                            copy_end=at+len(r);dest[at:copy_end]=r.view(np.uint8).reshape(len(r),-1);pp[at:copy_end]=p;at=copy_end
                rr.flags.writeable=pp.flags.writeable=False
                full=len(rr)//block*block
                h=RawHistory(uid,prior.blocks+tuple((rr[i:i+block],pp[i:i+block]) for i in range(0,full,block)),
                             tuple((rr[i:i+1],pp[i:i+1]) for i in range(full,len(rr))),end,prior.origin)
            else:h=prior
            histories.append(h);live[uid]=h
        previous.clear();previous.update(live)
        packed=histories,refs
    histories, refs = packed
    if latest_only:
        # Source prefetch needs one latest observation per episode, not tens of
        # thousands of per-decision RawHistory objects for a Reader rollout.
        flat=refs.reshape(-1,2)
        valid=np.flatnonzero(flat[:,0]>=0)
        lengths=np.full(len(histories),-1,np.int64)
        np.maximum.at(lengths,flat[valid,0],flat[valid,1])
        at=valid[flat[valid,1]==lengths[flat[valid,0]]]
        _,first=np.unique(flat[at,0],return_index=True)
        result=[]
        for loc in at[first]:
            index,length=map(int,flat[loc]);h=histories[index]
            whole,tail=divmod(length,block)
            if whole < len(h.blocks):
                rr,pp=h.blocks[whole]
                partial=tuple((rr[i:i+1],pp[i:i+1]) for i in range(tail))
            else:partial=h.tail[:tail]
            result.append((RawHistory(h.uid,h.blocks[:whole],partial,length,h.origin),*divmod(int(loc),refs.shape[1])))
        return result
    templates = []
    for h in histories:
        tails = []
        for rr, pp in h.blocks:
            rr.flags.writeable = pp.flags.writeable = False
            tails.append(tuple((rr[i:i+1], pp[i:i+1]) for i in range(len(rr))))
        for rr, pp in h.tail:
            rr.flags.writeable = pp.flags.writeable = False
        templates.append(tails + [h.tail])
    result = []
    prefixes = {}
    for row in refs:
        out = []
        for index, length in row:
            if index < 0:
                out.append(None)
                continue
            key = int(index), int(length)
            if key not in prefixes:
                h = histories[index]
                whole, tail = divmod(int(length), block)
                prefixes[key] = RawHistory(h.uid, h.blocks[:whole], templates[index][whole][:tail], int(length), h.origin)
            out.append(prefixes[key])
        result.append(out)
    return result


class HorizonAttention(torch.autograd.Function):
    """Grouped Reader attention whose source cotangents obey a per-pair mask.

    Forward is the unmasked layer's exact op sequence (same values, same SDPA).
    Backward recomputes P (no saved score matrix, no source replay). Query and
    Reader K/V weights receive every visible pair's gradient; the source table
    receives only pairs with grad_valid (decision-to-source age <= horizon).
    kv must be the K/V projection of the detached source table; src/weight let
    the masked K/V cotangent reach the sources through that projection.
    """
    @staticmethod
    def forward(ctx, q, kv, src, weight, banks, valid, grad_valid, heads):
        groups, d = banks.shape[0], q.shape[1]*q.shape[3]
        bank = F.embedding(banks, kv).to(q.dtype).reshape(groups, banks.shape[1], 2, heads, d // heads)
        k, v = bank[:, :, 0].transpose(1, 2), bank[:, :, 1].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None])
        ctx.save_for_backward(q, kv, src, weight, banks, valid, grad_valid, a)
        ctx.heads = heads
        return a

    @staticmethod
    def backward(ctx, da):
        q, kv, src, weight, banks, valid, grad_valid, a = ctx.saved_tensors
        heads = ctx.heads
        groups, _, count, dh = q.shape
        n = banks.shape[1]
        scale = dh ** -.5
        dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        dbank = torch.empty((groups, n, 2, heads, dh), dtype=torch.float32, device=q.device)
        dmask = torch.empty_like(dbank)
        # Bound the transient score matrices; groups are independent banks.
        step = max(1, 2**26 // max(1, heads*count*n))
        for lo in range(0, groups, step):
            hi = min(groups, lo+step)
            bank = F.embedding(banks[lo:hi], kv).float().reshape(hi-lo, n, 2, heads, dh)
            k, v = bank[:, :, 0].transpose(1, 2), bank[:, :, 1].transpose(1, 2)
            qs, do, o = q[lo:hi].float(), da[lo:hi].float(), a[lo:hi].float()
            s = (qs @ k.transpose(-1, -2)).mul_(scale).masked_fill_(~valid[lo:hi, None], float('-inf'))
            p = torch.softmax(s, -1); del s
            ds = (do @ v.transpose(-1, -2)).sub_((do*o).sum(-1, keepdim=True)).mul_(p)
            dq[lo:hi] = (ds @ k).mul_(scale)
            dbank[lo:hi, :, 0] = (ds.transpose(-1, -2) @ qs).mul_(scale).transpose(1, 2)
            dbank[lo:hi, :, 1] = (p.transpose(-1, -2) @ do).transpose(1, 2)
            g = grad_valid[lo:hi, None].to(p.dtype)
            dmask[lo:hi, :, 0] = (ds.mul_(g).transpose(-1, -2) @ qs).mul_(scale).transpose(1, 2)
            dmask[lo:hi, :, 1] = (p.mul_(g).transpose(-1, -2) @ do).transpose(1, 2)
        flat = banks.reshape(-1)
        dkv = dsrc = None
        if ctx.needs_input_grad[1]:
            dkv = torch.ops.aten.embedding_dense_backward(dbank.reshape(-1, 2*heads*dh), flat, kv.shape[0], -1, False)
        if ctx.needs_input_grad[2]:
            dmkv = torch.ops.aten.embedding_dense_backward(dmask.reshape(-1, 2*heads*dh), flat, kv.shape[0], -1, False)
            dsrc = (dmkv @ weight.detach().float()).to(src.dtype)
        return dq.to(q.dtype), dkv, dsrc, None, None, None, None, None


class ReaderLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, query, kv, indices, valid, layout=None):
        n, d = query.shape
        projected = self.q(self.norm(query))
        # Embedding's segmented reduction avoids hot-index atomic contention.
        # No padding_idx / frequency scaling: every consumer keeps its gradient.
        if layout is None:
            q = projected.reshape(n, self.heads, 1, d // self.heads)
            bank = F.embedding(indices, kv).to(q.dtype).reshape(n, indices.shape[1], 2, self.heads, d // self.heads)
            k, v = bank[:, :, 0].transpose(1, 2), bank[:, :, 1].transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :]).reshape(n, d)
        else:
            groups, count = layout['shape']
            pos, banks = layout['positions'], layout['indices']
            q = projected.new_zeros((groups*count, d)).index_copy(0, pos, projected)
            q = q.reshape(groups, count, self.heads, d//self.heads).transpose(1, 2)
            if layout.get('src') is not None and torch.is_grad_enabled():
                a = HorizonAttention.apply(q, kv, layout['src'], self.kv.weight, banks,
                                           layout['valid'], layout['grad_valid'], self.heads)
            else:
                bank = F.embedding(banks, kv).to(q.dtype).reshape(groups, banks.shape[1], 2, self.heads, d//self.heads)
                k, v = bank[:, :, 0].transpose(1, 2), bank[:, :, 1].transpose(1, 2)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=layout['valid'][:, None])
            a = a.transpose(1, 2).reshape(groups*count, d).index_select(0, pos)
        x = query + self.out(a)
        return x + self.ff(self.ffnorm(x))


class MemoryReader(nn.Module):
    def __init__(self, spatial, width=256, layers=4, heads=4, window=256, block=64, tokens=4):
        super().__init__()
        self.D, self.L, self.H, self.W = width, layers, heads, window
        self.block, self.tokens = block, tokens
        self.inp = nn.Linear(2 * spatial, width)
        self.layers = nn.ModuleList(ReaderLayer(width, heads) for _ in range(layers))
        self.queries = nn.Parameter(torch.randn(tokens, width) * .02)
        self.writer = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.writer_norm = nn.LayerNorm(width)
        self.norm = nn.LayerNorm(width)
        self.fusion_q = nn.Linear(width, spatial)
        self.fusion = nn.MultiheadAttention(spatial, heads, dropout=0., batch_first=True)
        self.read = nn.Linear(width, spatial)
        self.read_g = nn.Linear(width, spatial)
        self.fusion_g = nn.Linear(spatial, spatial)

    def write(self, u):
        # [blocks, C, D]; position is added after source encoding, never cached
        # by an observation's content or used to reveal future observations.
        pos = torch.arange(self.block, device=u.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, self.D, 2, device=u.device).float() * (-np.log(10000.) / self.D))
        a = pos * freq
        pe = torch.stack((a.sin(), a.cos()), -1).flatten(1).to(u.dtype)
        x = u + pe[None]
        q = self.queries[None].expand(len(u), -1, -1).to(u.dtype)
        y, _ = self.writer(q, x, x, need_weights=False)
        return self.writer_norm(q + y)

    def position(self, positions, dtype):
        freq = torch.exp(torch.arange(0, self.D, 2, device=positions.device).float() * (-np.log(10000.) / self.D))
        a = positions.float()[:, None] * freq
        return torch.stack((a.sin(), a.cos()), -1).flatten(1).to(dtype)

    def read_bank(self, u, kvs, indices, valid, layout=None):
        q = u
        for layer, kv in zip(self.layers, kvs):
            q = layer(q, kv, indices, valid, layout)
        return self.norm(q)

    def fuse(self, o, y, pad):
        q = self.fusion_q(o)[:, None]
        c, _ = self.fusion(q, y, y, key_padding_mask=pad, need_weights=False)
        return c[:, 0]


class ComponentCache:
    """Exact raw-payload keys, never fingerprints or post-attention embeddings.

    Learner: deduplicate within each encoder microbatch with differentiable
    index_select (all repeated consumers add gradients). Persistent source graphs
    belong to immutable parameter snapshots, never to in-place updated weights.
    Actor: reuse pre-Spatial outputs within a bounded weight-version interval.
    The bounded LRU is a compute cache, not a limit on the raw memory history.
    """
    fields = dict(player=('player','patch','inv','inv_all','pitem','pinv','pcharge','phearts','pchar'),
                  items=('inv','pitem','pinv'), grid=('grid',), map=('map',), patch=('patch',))
    layouts = {}

    @classmethod
    def payload_keys(cls,rows,pending,name):
        # Copy fields in bulk, not Python's rows × fields scalar loop. The key
        # remains the identical literal payload, with no probabilistic hash.
        if name not in cls.layouts:
            fields=[(f,rows.dtype.fields[f][0]) for f in cls.fields[name]]
            if name=='player':fields.append(('_pending',pending.dtype,(pending.shape[1],)))
            cls.layouts[name]=np.dtype(fields)
        packed=np.empty(len(rows),dtype=cls.layouts[name])
        for f in cls.fields[name]:packed[f]=rows[f]
        if name=='player':packed['_pending']=pending
        return [r.tobytes() for r in packed]

    def __init__(self):
        self.generation = -1
        self.values = {}
        self.calls = {}

    def encode(self, model, name, b, fn):
        rows, pending = b['_raw_rows'], b['_raw_pending']
        keys = self.payload_keys(rows,pending,name)
        unique = {}; positions = []; inverse = []
        for i,key in enumerate(keys):
            if key not in unique:
                unique[key] = len(positions); positions.append(i)
            inverse.append(unique[key])
        source_uid=b.get('_source_component_uid')
        source_graph=source_uid is not None and name in ('grid','map','patch')
        persistent = source_graph or (not torch.is_grad_enabled() and b.get('_actor_components', False))
        if self.generation != model.source_generation:
            self.values.clear(); self.generation = model.source_generation
        table_key=(source_uid,name) if source_graph else name
        table = self.values.setdefault(table_key, OrderedDict()) if persistent else {}
        missing = [i for key,i in unique.items() if key not in table]
        dev = b['player'].device
        if missing:
            selected = [positions[i] for i in missing]
            select = torch.as_tensor(selected,device=dev)
            # Component inputs are observation rows; packed token indices are
            # not row arrays (an empty room can have an empty dkeep / ekeep).
            fields = self.fields[name] + (('pending',) if name == 'player' else ())
            subset = {k: b[k].index_select(0,select) for k in fields}
            if name == 'player':
                # Player changes need not imply a changed terrain patch. Keep
                # provenance for the nested patch CNN's independent exact cache.
                subset['_raw_rows'] = (rows if len(selected) == len(rows) else
                    rows.view(np.uint8).reshape(len(rows),-1)[selected].view(rows.dtype).reshape(-1))
                subset['_raw_pending'] = pending if len(selected) == len(rows) else pending[selected]
                subset['_actor_components'] = b.get('_actor_components',False)
                if source_uid is not None:subset['_source_component_uid']=source_uid
            output = fn(subset)
            output = output if isinstance(output,tuple) else (output,)
            if not persistent:
                # Unique outputs are already in the required order. Splitting
                # them into one tensor per row and stacking again creates a
                # large, entirely redundant autograd graph.
                ix = torch.as_tensor(inverse, device=dev)
                outputs = tuple(v.index_select(0, ix) for v in output)
                hit, total = self.calls.get(name, (0, 0))
                self.calls[name] = (hit + len(rows) - len(missing), total + len(rows))
                return outputs[0] if len(outputs) == 1 else outputs
            if source_graph:
                # Consumers accumulate in FP32 before the one BF16 source VJP.
                output=tuple(v.float() if v.is_floating_point() else v for v in output)
            missing_keys = list(unique)
            for j,i in enumerate(missing):
                table[missing_keys[i]] = tuple(v[j] for v in output)
        ordered = list(unique)
        outputs = tuple(torch.stack([table[key][j] for key in ordered])
                        for j in range(len(table[ordered[0]])))
        ix = torch.as_tensor(inverse,device=dev)
        outputs = tuple(v.index_select(0,ix) for v in outputs)
        hit,total = self.calls.get(name,(0,0))
        self.calls[name] = (hit + len(rows) - len(missing),total + len(rows))
        if persistent:
            for key in ordered: table.move_to_end(key)
            while len(table) > 2048: table.popitem(last=False)
        return outputs[0] if len(outputs)==1 else outputs

    def retain_source_uids(self,uids):
        self.values={key:value for key,value in self.values.items()
                     if not isinstance(key,tuple) or key[0] in uids}


class StepComponents:
    """One Adam generation's unique component leaves and accumulated VJP.

    Only raw-source addressing survives across encoder microbatches. The first
    encounter encodes a component; all later frames gather that same leaf. The
    final replay visits each unique source once, after Spatial gradients arrive.
    """
    def __init__(self, model):
        self.model=model
        self.tables={}
        self.chunks=[]
        self.plans={}

    def encode(self,name,b,fn):
        rows,pending=b['_raw_rows'],b['_raw_pending']
        plan_key=(name,b['_source_range'])
        plan=self.plans.get(plan_key)
        if plan is None:
            table=self.tables.setdefault(name,{})
            keys=ComponentCache.payload_keys(rows,pending,name)
            missing={}
            for i,key in enumerate(keys):
                if key not in table:missing.setdefault(key,i)
            if missing:
                positions=list(missing.values())
                ix=torch.as_tensor(positions,device=b['player'].device)
                fields=ComponentCache.fields[name]+(('pending',) if name=='player' else ())
                subset={k:b[k].index_select(0,ix) for k in fields}
                values=fn(subset)
                values=values if isinstance(values,tuple) else (values,)
                leaves=tuple(v.detach().requires_grad_(v.is_floating_point()) for v in values)
                chunk=len(self.chunks)
                self.chunks.append((rows,pending,np.asarray(positions,np.int64),fn,leaves))
                for j,key in enumerate(missing):table[key]=(chunk,j)
            groups={}
            for at,key in enumerate(keys):
                c,j=table[key];groups.setdefault(c,[]).append((at,j))
            positions=np.concatenate([np.asarray(g)[:,0] for g in groups.values()])
            gather=[(c,torch.as_tensor(np.asarray(g)[:,1],device=b['player'].device)) for c,g in groups.items()]
            order=torch.as_tensor(np.argsort(positions),device=b['player'].device)
            plan=(gather,order)
            self.plans[plan_key]=plan
        gather,order=plan
        result=tuple(torch.cat([self.chunks[c][4][k].index_select(0,ix) for c,ix in gather]).index_select(0,order)
                     for k in range(len(self.chunks[gather[0][0]][4])))
        return result[0] if len(result)==1 else result

    def backward(self):
        from .tok_policy import to_batch
        dev=next(self.model.parameters()).device
        for rows,pending,positions,fn,leaves in self.chunks:
            if not any(v.grad is not None for v in leaves):continue
            b=to_batch(rows[positions],pending[positions],dev,entities=64)
            values=fn(b)
            values=values if isinstance(values,tuple) else (values,)
            pairs=[(v,leaf.grad) for v,leaf in zip(values,leaves) if leaf.grad is not None]
            torch.autograd.backward([v for v,g in pairs],[g for v,g in pairs])


def recent_frame_union(query, lengths, window, n_rows):
    """Union contiguous episode ranges without expanding every query window."""
    changes = np.zeros(n_rows + 1, np.int64)
    np.add.at(changes, query - np.minimum(lengths, window), 1)
    np.add.at(changes, query, -1)
    return np.flatnonzero(np.cumsum(changes[:-1]) > 0)


def reader_indices(arrays, device):
    """One owned pinned transfer; do not synchronize the previous Reader step."""
    arrays=[np.asarray(a,np.int64) for a in arrays]
    packed=torch.from_numpy(np.concatenate([a.reshape(-1) for a in arrays]))
    if device.type=='cuda':packed=packed.pin_memory()
    packed=packed.to(device,non_blocking=True)
    return tuple(t.view(a.shape) for t,a in zip(packed.split([a.size for a in arrays]),arrays))


class SourceBatch:
    """One Reader batch's source view, shared by all decision microbatches.

    Requests are (raw prefix, current ROW, current pending). All references for
    an episode share the longest raw prefix; earlier queries are causally masked.
    cache=True stores leaf activations, accumulates every consumer gradient, then
    replays KV projection -> Writer -> Spatial before Adam. cache=False is the
    direct source graph used by speed-first training, without a second forward.
    With a pipeline, original source graphs stay on source devices for a complete
    slow update cycle; this batch transports values and accumulated gradients.
    """
    def __init__(self, model, requests, micro=512, cache=True, pipeline=None, stateless=(), frozen=()):
        from .tok_policy import to_batch
        stamp = memory_tick()
        self.model, self.micro, self.cached = model, micro, cache
        self.pipeline = pipeline
        self.source_version = pipeline.version if pipeline is not None else None
        self.requests = requests
        self.components = StepComponents(model) if cache and model.source_cache else None
        dev = next(model.parameters()).device
        reader = model.reader
        latest = {}
        for request in requests:
            history = request[0]
            if history.uid not in latest or history.length > latest[history.uid][0].length:
                latest[history.uid] = request
        if pipeline is not None: pipeline.prefetch(requests,stateless)
        # Append only the longest request per episode. Copy immutable source
        # arrays in bulk, not hundreds of thousands of structured numpy scalars.
        row_parts, pending_parts, frame_times, identities = [], [], [], {}
        self.frame_keys = []
        self.history_frames, self.history_memory = {}, {}
        block_ids, block_rows, memory_times = {}, [], []
        n_rows = 0
        for uid, (prefix, row, pending) in latest.items():
            history = prefix.append(row, pending, reader.block)
            index_parts, time_at = [], 0
            for rr, pp in history.blocks + history.tail:
                key = (uid, id(rr)) if pipeline is not None else id(rr)
                if key not in identities:
                    identities[key] = np.arange(n_rows, n_rows + len(rr), dtype=np.int64)
                    n_rows += len(rr)
                    if pipeline is None:
                        row_parts.append(rr); pending_parts.append(pp)
                    frame_times.extend(range(time_at, time_at + len(rr)))
                    self.frame_keys.extend(history.key(t) for t in range(time_at, time_at + len(rr)))
                index_parts.append(identities[key])
                time_at += len(rr)
            indices = np.concatenate(index_parts)
            self.history_frames[uid] = indices
            memories = []
            for at in range(0,len(indices)-reader.block+1,reader.block):
                key = tuple(indices[at:at+reader.block])
                if key not in block_ids:
                    block_ids[key] = len(block_rows)
                    block_rows.append(key); memory_times.append(at+reader.block-1)
                memories.extend(range(block_ids[key]*reader.tokens,(block_ids[key]+1)*reader.tokens))
            self.history_memory[uid] = np.asarray(memories,np.int64)
        self.rows, self.pending = (np.concatenate(row_parts), np.concatenate(pending_parts)) if pipeline is None else (None, None)
        self.frame_times = torch.as_tensor(frame_times,device=dev)
        self.memory_times = torch.as_tensor(memory_times,device=dev).repeat_interleave(reader.tokens)
        query_lengths = np.asarray([h.length for h,_,_ in requests],np.int64)
        self.query_times = torch.as_tensor(query_lengths,device=dev)
        # Records whose loss was consumed in an earlier update: read only, no
        # source gradient (their own graphs were released, RecordLedger rule 2).
        self.query_frozen = None
        if len(frozen):
            mask = np.zeros(len(requests), bool); mask[np.asarray(list(frozen), np.int64)] = True
            self.query_frozen = torch.as_tensor(mask, device=dev)
        query = np.array([self.history_frames[h.uid][h.length] for h,_,_ in requests],np.int64)
        if pipeline is not None:
            self.live_frames = recent_frame_union(query, query_lengths, reader.W, n_rows)
        else:
            recent = [self.history_frames[h.uid][max(0,h.length-reader.W):h.length] for h,_,_ in requests]
            self.live_frames = np.unique(np.concatenate(recent))
        self.live_frames_d = torch.as_tensor(self.live_frames,device=dev)
        self.block_indices = torch.as_tensor(np.asarray(block_rows,np.int64).reshape(-1,reader.block),device=dev)
        self.block_keys = [tuple(self.frame_keys[i] for i in block) for block in block_rows]
        self.unique_query, inverse = np.unique(query, return_inverse=True)
        self.query = torch.as_tensor(query, device=dev)
        self.query_inverse = torch.as_tensor(inverse, device=dev)
        self.unique_query_d = torch.as_tensor(self.unique_query, device=dev)
        if pipeline is not None:
            # Old frames outside the direct-read window still train through all
            # Writer tokens. Their raw u values need not cross to the Reader.
            self.u_frames = np.union1d(self.live_frames, self.unique_query)
            self.u_live = torch.as_tensor(np.searchsorted(self.u_frames,self.live_frames),device=dev)
            self.u_query = torch.as_tensor(np.searchsorted(self.u_frames,query),device=dev)
        else:
            self.u_live, self.u_query = self.live_frames_d, self.query
        stamp = memory_part('source_plan', stamp)
        if PROFILE_MEMORY:
            for name, count in [('source_batches',1), ('source_rows',n_rows),
                                ('queries',len(requests)), ('writer_blocks',len(block_rows))]:
                MEMORY_COUNTS[name] = MEMORY_COUNTS.get(name,0) + count
        if pipeline is not None:
            self.u, self.y, self.pad, self.mem = pipeline.materialize(self)
        else:
            encoded, ys, pads = [], [], []
            context = torch.no_grad() if cache else nullcontext()
            # no_grad casts must not poison the caller's autocast parameter cache:
            # the same weights are replayed with gradients before leaving that scope.
            with context, torch.autocast(dev.type, enabled=torch.is_autocast_enabled(dev.type),
                                         dtype=torch.get_autocast_dtype(dev.type), cache_enabled=False):
                for lo in range(0, len(self.rows), micro):
                    hi = min(lo + micro, len(self.rows))
                    b = to_batch(self.rows[lo:hi], self.pending[lo:hi], dev, entities=64)
                    b['_raw_rows'], b['_raw_pending'] = self.rows[lo:hi], self.pending[lo:hi]
                    b['_source_packed'] = True
                    b['_step_components'], b['_source_range'] = self.components, (lo, hi)
                    y, pad, u = model.encode_rebuild(b)
                    encoded.append(u)
                    take = self.unique_query[(self.unique_query >= lo) & (self.unique_query < hi)] - lo
                    if len(take):
                        ix = torch.as_tensor(take, device=dev)
                        ys.append(y.index_select(0, ix)); pads.append(pad.index_select(0, ix))
                u = torch.cat(encoded)
                y = torch.cat(ys)
            stamp = memory_part('source_encode', stamp)
            self.u = u.detach().requires_grad_() if cache else u
            self.y = y.detach().requires_grad_() if cache else y
            self.pad = torch.cat(pads)
            # Writer receives every frame, independently per fixed block. No room-exit
            # importance rule, no recursive summary and no 128-visit cutoff.
            with torch.no_grad() if cache else nullcontext(), torch.autocast(
                    dev.type, enabled=torch.is_autocast_enabled(dev.type),
                    dtype=torch.get_autocast_dtype(dev.type), cache_enabled=False):
                mem = reader.write(self.u[self.block_indices]).flatten(0, 1) if len(self.block_indices) else self.u.new_empty((0, reader.D))
            self.mem = mem.detach().requires_grad_() if cache else mem
        src = self.sources()
        horizon = (pipeline.horizon or self.query_frozen is not None) if pipeline is not None else 0
        # With a gradient horizon, K/V weights see every read; the masked source
        # cotangent returns through HorizonAttention (identical K/V values).
        self.kv_source = src if horizon else None
        with torch.no_grad() if cache else nullcontext(), torch.autocast(
                dev.type, enabled=torch.is_autocast_enabled(dev.type),
                dtype=torch.get_autocast_dtype(dev.type), cache_enabled=False):
            # Quantized forward values, FP32 consumer-gradient accumulation. A
            # BF16 leaf loses small contributions when hundreds of reads collide.
            kvs = [layer.kv(src.detach() if horizon else src).float() for layer in reader.layers]
        self.kvs = [v.detach().requires_grad_() for v in kvs] if cache else kvs
        stamp = memory_part('source_writer_kv', stamp)
        if pipeline is not None:
            self.bank_times = np.concatenate([[-1], np.asarray(frame_times)[self.live_frames],
                                               np.repeat(memory_times, reader.tokens)])
            self.uid_group={uid:i for i,uid in enumerate(self.history_frames)}
            self.query_groups=np.asarray([self.uid_group[h.uid] for h,_,_ in requests],np.int64)
            unions=[]
            for uid,frames in self.history_frames.items():
                # Pipeline frame indices for one UID form a contiguous range.
                # Search its endpoints, not the entire live array per query batch.
                lo,hi=np.searchsorted(self.live_frames,[frames[0],frames[-1]+1])
                unions.append(np.concatenate([[0],1+np.arange(lo,hi),
                              1+len(self.live_frames)+self.history_memory[uid]]))
            banks=np.zeros((len(unions),max(map(len,unions))),np.int64)
            present=np.zeros_like(banks,bool)
            for i,union in enumerate(unions):
                banks[i,:len(union)]=union;present[i,:len(union)]=True
            self.group_banks=torch.as_tensor(banks,device=dev)
            self.group_present=torch.as_tensor(present,device=dev)
            self.group_times=torch.as_tensor(self.bank_times[banks],device=dev)
            return
        banks = []
        for h, _, _ in requests:
            frames = self.history_frames[h.uid][max(0,h.length-reader.W):h.length]
            recent = np.searchsorted(self.live_frames,frames) + 1
            memory = self.history_memory[h.uid][:(h.length//reader.block)*reader.tokens] + len(self.live_frames) + 1
            banks.append(np.concatenate([np.zeros(1, np.int64), recent, memory]))
        self.host_banks = banks
        size = max(map(len, banks))
        idx = np.zeros((len(banks), size), np.int64)
        valid = np.zeros_like(idx, bool)
        for i, bank in enumerate(banks):
            idx[i, :len(bank)] = bank; valid[i, :len(bank)] = True
        self.indices = torch.as_tensor(idx, device=dev)
        self.valid = torch.as_tensor(valid, device=dev)
        memory_part('source_banks', stamp)

    def sources(self):
        rd = self.model.reader
        return torch.cat([self.u.new_zeros((1, rd.D)),
                          self.u[self.u_live] + rd.position(self.frame_times[self.live_frames_d], self.u.dtype),
                          self.mem + rd.position(self.memory_times, self.mem.dtype)])

    def grouped_layout(self, indices):
        """Share an episode bank across its queries instead of duplicating KV
        8192 times. The per-query mask is exactly the original bank membership."""
        ids_h,inverse,counts=np.unique(self.query_groups[indices],return_inverse=True,return_counts=True)
        count=int(counts.max());ng=len(counts)
        order=np.argsort(inverse,kind='stable')
        ranks=np.arange(len(indices))-np.repeat(np.cumsum(counts)-counts,counts)
        positions = np.empty(len(indices), np.int64)
        positions[order]=np.repeat(np.arange(ng),counts)*count+ranks
        dev = self.u.device
        pos,ids,ix=reader_indices((positions,ids_h,indices),dev)
        banks=self.group_banks.index_select(0,ids)
        bt=self.group_times.index_select(0,ids)[:,None]
        times=self.query_times.index_select(0,ix)
        times=times.new_zeros(ng*count).index_copy(0,pos,times).reshape(ng,count,1)
        recent=((banks>0)&(banks<=len(self.live_frames)))[:,None]
        memory=(banks>len(self.live_frames))[:,None]
        mask=self.group_present.index_select(0,ids)[:,None] & (
            (banks==0)[:,None] | (recent & (bt<times) & (bt>=times-self.model.reader.W)) |
            (memory & (bt<(times//self.model.reader.block)*self.model.reader.block)))
        layout=dict(shape=(ng,count),positions=pos,indices=banks,valid=mask)
        horizon=self.pipeline.horizon
        if self.kv_source is not None:
            grad=mask
            if horizon:
                # Source age per (decision, entry): a frame's own time, a Writer
                # block's first frame. Reads keep mask; gradients need age <= horizon.
                start=torch.where(memory,bt-(self.model.reader.block-1),bt)
                grad=grad&((banks==0)[:,None]|(times-start<=horizon))
            if self.query_frozen is not None:
                frozen=self.query_frozen.index_select(0,ix)
                frozen=frozen.new_zeros(ng*count).index_copy(0,pos,frozen).reshape(ng,count,1)
                grad=grad&~frozen
            layout.update(src=self.kv_source,grad_valid=grad)
        return layout,ix

    def forward(self, indices, need_h=False, need_vg=False):
        if self.pipeline is not None: self.pipeline.check_version(self.source_version)
        stamp = memory_tick()
        m, rd = self.model, self.model.reader
        host = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices, np.int64)
        if self.pipeline is not None:
            layout,ix = self.grouped_layout(host)
        else:
            layout=None
            ix=torch.as_tensor(indices,dtype=torch.long,device=self.u.device)
        yi = self.query_inverse[ix]
        y, pad = self.y[yi], self.pad[yi]
        q = self.u[self.u_query[ix]] + rd.position(self.query_times[ix], self.u.dtype)
        if self.query_frozen is not None:   # same values; no cotangent to a released source
            frozen = self.query_frozen[ix]
            q = torch.where(frozen[:, None], q.detach(), q)
            y = torch.where(frozen[:, None, None], y.detach(), y)
        o = rd.read_bank(q, self.kvs, None if layout is not None else self.indices[ix],
                        None if layout is not None else self.valid[ix], layout)
        c = rd.fuse(o, y, pad)
        result = m._heads(y[:, 0] + rd.read(o) + c, need_h,
                        yg=y[:, -1] + rd.read_g(o) + rd.fusion_g(c), need_vg=need_vg)
        memory_part('reader_forward', stamp)
        return result

    def backward(self):
        if self.pipeline is not None:
            self.pipeline.collect(self)
            return
        if not self.cached:
            return
        from .tok_policy import to_batch
        stamp = memory_tick()
        m, rd, dev = self.model, self.model.reader, self.u.device
        # Projections share the source leaves, not their short-lived graphs.
        for layer, cached in zip(rd.layers, self.kvs):
            if cached.grad is not None:
                layer.kv(self.sources()).backward(cached.grad)
        stamp = memory_part('vjp_kv', stamp)
        if len(self.block_indices) and self.mem.grad is not None:
            # Keep the Writer graph bounded without truncating a block or dropping
            # any consumer gradient. Indexing backward adds repeated source uses.
            step = max(1, self.micro // rd.block)
            for lo in range(0, len(self.block_indices), step):
                hi = min(lo + step, len(self.block_indices))
                rd.write(self.u[self.block_indices[lo:hi]]).flatten(0, 1).backward(self.mem.grad[lo * rd.tokens:hi * rd.tokens])
        stamp = memory_part('vjp_writer', stamp)
        for lo in range(0, len(self.rows), self.micro):
            hi = min(lo + self.micro, len(self.rows))
            b = to_batch(self.rows[lo:hi], self.pending[lo:hi], dev, entities=64)
            b['_raw_rows'], b['_raw_pending'] = self.rows[lo:hi], self.pending[lo:hi]
            b['_source_packed'] = True
            b['_step_components'], b['_source_range'] = self.components, (lo, hi)
            y, _, u = m.encode_rebuild(b)
            outputs, gradients = [], []
            if self.u.grad is not None:
                outputs.append(u); gradients.append(self.u.grad[lo:hi])
            pos = np.flatnonzero((self.unique_query >= lo) & (self.unique_query < hi))
            if len(pos) and self.y.grad is not None:
                local = torch.as_tensor(self.unique_query[pos] - lo, device=dev)
                outputs.append(y.index_select(0, local))
                gradients.append(self.y.grad[torch.as_tensor(pos, device=dev)])
            if outputs:
                torch.autograd.backward(outputs, gradients)
        stamp = memory_part('vjp_spatial', stamp)
        if self.components is not None:
            self.components.backward()
        memory_part('vjp_components', stamp)


class ActorMemory:
    """Raw prefixes and incremental caches with bounded parameter-version lag."""
    def __init__(self, slots):
        self.raw = [RawHistory.empty() for _ in range(slots)]
        self.recent = [None] * slots
        self.tail = [[] for _ in range(slots)]
        self.mem = [None] * slots
        self.generation = [-1] * slots
        self.recent_kv = [None] * slots
        self.memory_kv = [None] * slots
        self.fast = None

    def reset(self, slot):
        self.raw[slot] = RawHistory.empty()
        self.recent[slot] = self.mem[slot] = None
        self.tail[slot] = []
        self.generation[slot] = -1
        self.recent_kv[slot] = self.memory_kv[slot] = None
        if self.fast is not None:self.fast.length[slot]=0

    def copy(self, source, destination, other):
        self.raw[destination] = other.raw[source].fork()
        self.recent[destination] = other.recent[source]
        self.mem[destination] = other.mem[source]
        self.tail[destination] = list(other.tail[source])
        self.generation[destination] = other.generation[source]
        self.recent_kv[destination] = other.recent_kv[source]
        self.memory_kv[destination] = other.memory_kv[source]
        if other.fast is not None:
            if self.fast is None:
                self.fast=other.fast.new(len(self.raw))
            self.fast.copy(source,destination,other.fast)
        elif self.fast is not None:self.fast.load(self,[destination])

    @torch.no_grad()
    def refresh(self, model, slots):
        from .tok_policy import to_batch
        rd, dev = model.reader, next(model.parameters()).device
        stale = [int(s) for s in slots if self.generation[int(s)] != model.source_generation]
        for s in stale:
            h = self.raw[s]
            if h.length:
                rows, pending = h.arrays()
                values=[]
                for at in range(0,len(rows),512):
                    b=to_batch(rows[at:at+512],pending[at:at+512],dev)
                    b['_raw_rows'], b['_raw_pending'] = rows[at:at+512], pending[at:at+512]
                    values.append(model.encode_rebuild(b)[2])
                u=torch.cat(values)
                full=h.length//rd.block*rd.block
                self.mem[s]=rd.write(u[:full].reshape(-1,rd.block,rd.D)).flatten(0,1) if full else u.new_empty((0,rd.D))
                self.recent[s]=u[-rd.W:]
                self.tail[s]=list(u[full:].unbind(0))
                rt=torch.arange(max(0,h.length-rd.W),h.length,device=dev)
                mt=(torch.arange(h.length//rd.block,device=dev)+1).repeat_interleave(rd.tokens)*rd.block-1
                self.recent_kv[s]=torch.stack([layer.kv(self.recent[s]+rd.position(rt,u.dtype)) for layer in rd.layers],1)
                self.memory_kv[s]=torch.stack([layer.kv(self.mem[s]+rd.position(mt,u.dtype)) for layer in rd.layers],1)
            self.generation[s]=model.source_generation

    @torch.no_grad()
    def forward(self, model, b, temporal, need_h, need_vg):
        stamp=actor_part('start')
        rd=model.reader
        slots=np.asarray(temporal['host_slots'],np.int64)
        rows,pending=temporal['raw_rows'],temporal['raw_pending']
        for s,first in zip(slots,rows['first']):
            if first:self.reset(int(s))
        if self.fast is not None:self.fast.export(self,slots)
        self.refresh(model,slots)
        stamp=actor_part('refresh',stamp)
        b['_raw_rows'], b['_raw_pending'], b['_actor_components'] = rows, pending, True
        y,pad,u=model.encode_rebuild(b)
        stamp=actor_part('encode',stamp)
        prefixes=[self.raw[int(s)] for s in slots]
        zero=torch.stack([layer.kv(u.new_zeros((1,rd.D))) for layer in rd.layers],1)
        banks=[]
        for s in slots:
            parts=[zero]
            if self.recent_kv[s] is not None:parts.append(self.recent_kv[s])
            if self.memory_kv[s] is not None:parts.append(self.memory_kv[s])
            banks.append(torch.cat(parts))
        bank=nn.utils.rnn.pad_sequence(banks,batch_first=True)
        n,size=bank.shape[:2]
        valid=torch.arange(size,device=u.device)[None]<torch.as_tensor([len(v) for v in banks],device=u.device)[:,None]
        indices=torch.arange(n*size,device=u.device).reshape(n,size)
        kvs=list(bank.flatten(0,1).unbind(1))
        qt=torch.as_tensor([h.length for h in prefixes],device=u.device)
        stamp=actor_part('bank',stamp)
        o=rd.read_bank(u+rd.position(qt,u.dtype),kvs,indices,valid)
        c=rd.fuse(o,y,pad)
        outs=model._heads(y[:,0]+rd.read(o)+c,need_h,
                         yg=y[:,-1]+rd.read_g(o)+rd.fusion_g(c),need_vg=need_vg)
        stamp=actor_part('reader_heads',stamp)
        frame_kv=torch.stack([layer.kv(u+rd.position(qt,u.dtype)) for layer in rd.layers],1)
        closed=[]
        for i,s in enumerate(slots):
            self.raw[s]=self.raw[s].append(rows[i],pending[i],rd.block)
            self.recent[s]=u[i:i+1] if self.recent[s] is None else torch.cat([self.recent[s],u[i:i+1]])[-rd.W:]
            self.recent_kv[s]=frame_kv[i:i+1] if self.recent_kv[s] is None else torch.cat([self.recent_kv[s],frame_kv[i:i+1]])[-rd.W:]
            self.tail[s].append(u[i])
            if len(self.tail[s])==rd.block:closed.append(int(s))
        if closed:
            memories=rd.write(torch.stack([torch.stack(self.tail[s]) for s in closed]))
            times=torch.as_tensor([self.raw[s].length-1 for s in closed],device=u.device).repeat_interleave(rd.tokens)
            mkv=torch.stack([layer.kv(memories.flatten(0,1)+rd.position(times,u.dtype)) for layer in rd.layers],1)
            mkv=mkv.reshape(len(closed),rd.tokens,len(rd.layers),2*rd.D)
            for i,s in enumerate(closed):
                self.mem[s]=memories[i] if self.mem[s] is None else torch.cat([self.mem[s],memories[i]])
                self.memory_kv[s]=mkv[i] if self.memory_kv[s] is None else torch.cat([self.memory_kv[s],mkv[i]])
                self.tail[s]=[]
        stamp=actor_part('append_write',stamp)
        state=temporal['state'];sd=temporal['slot']
        state.prev_o.index_copy_(0,sd,o.float())
        state.steps.index_copy_(0,sd,qt+1)
        stage=torch.as_tensor(rows['stage'].astype(np.int64),device=u.device)
        state.prev_stage.index_copy_(0,sd,stage)
        temporal['out']=dict(create=torch.zeros(n,dtype=torch.bool,device=u.device),
                            nroom=torch.zeros(n,dtype=torch.long,device=u.device),steps=qt+1,
                            stage_prev=stage,kv=u.new_zeros((n,state.L,2,state.D)),s=u,o=o,
                            raw_contexts=prefixes)
        if self.fast is not None:self.fast.load(self,slots)
        actor_part('temporal_out',stamp)
        return outs


class RebuildActor:
    """The production asynchronous actor interface; source history is host-owned.

    This initial reader path uses batched PyTorch kernels, not the legacy CUDA
    graph whose persistent H/decision-KV state has different semantics. Its real
    cost is included in the full-pipeline benchmark.
    """
    def __init__(self, model, rows_max, entities=32, stream=None, crn=True, memory=None,
                 bf16=False, compile_mode=None, extra=None, fold=False, sdpa_math=False,
                 row_sizes=None, temporal=None):
        self.model,self.n,self.stream=model,rows_max,stream
        self.crn,self.bf16,self.extra_fn=crn,bf16,extra
        self.tstate=temporal
        self.nh=len(model.heads)
        self.vg_on=model.gtok is not None
        self.t_on=True;self.mem=0;self.fold=False
        self.done=torch.cuda.Event()
        self.extra=self.extra_out=self.h_used=self.vg=self.t_out=None
        self.calls=0
        if PROFILE_ACTOR:
            import cProfile
            self.profile=cProfile.Profile()

    @torch.no_grad()
    def submit(self,rows,pending,widx=None,reset=None,tdest=None,ukeys=None):
        from .tok_policy import to_batch,crn_uniforms,gumbel_sample,temporal_out
        n=len(rows)
        if n>self.n:return False
        self._n=n
        self.calls+=1
        if PROFILE_ACTOR and self.calls==129:self.profile.enable()
        stream=self.stream if self.stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(stream),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
            stamp=actor_part('start')
            b=to_batch(rows,pending,next(self.model.parameters()).device)
            td=self.tstate.step_inputs(self.model,rows,pending,widx)
            actor_part('input',stamp)
            outputs=self.model(b,temporal=td,need_h=True,need_vg=self.vg_on)
            stamp=actor_part('start')
            if ukeys is not None:
                uniforms=torch.as_tensor(crn_uniforms(ukeys[0],ukeys[1],sum(self.model.heads)),device=b['player'].device)
            elif self.crn and 'crn' in rows.dtype.names:
                uniforms=torch.as_tensor(crn_uniforms(rows['crn'],rows['t'],sum(self.model.heads)),device=b['player'].device)
            else:
                uniforms=torch.rand((n,sum(self.model.heads)),device=b['player'].device)
            actions,lp=gumbel_sample(self.model.split(outputs[0].float(),self.model.gate(b)),uniforms)
            self.extra_out=self.extra_fn(b,outputs,None) if self.extra_fn is not None else None
            fields=[actions.float(),lp[:,None],outputs[1].float()[:,None]]
            self.n_extra=0 if self.extra_out is None else self.extra_out.shape[1]
            if self.extra_out is not None:fields.append(self.extra_out.float())
            if self.vg_on:fields.append(outputs[-1].float()[:,None])
            out=torch.cat(fields,1)
            self._cpu=torch.empty(out.shape,dtype=out.dtype,device='cpu',pin_memory=True)
            self._cpu.copy_(out,non_blocking=True)
            # t_out stores raw provenance for training, never a substitute hidden state.
            info=td['out']
            self.t_out=dict(create=np.zeros(n,bool),nroom=np.zeros(n,np.int64),
                            steps=np.array([h.length+1 for h in info['raw_contexts']],np.int64),
                            stage_prev=rows['stage'].astype(np.int64),raw_contexts=info['raw_contexts'])
            if tdest is None:
                self.t_out.update(kv=info['kv'],s=info['s'],o=info['o'])
            else:
                idx=torch.as_tensor(tdest['rows'],device=out.device)
                tdest['kv'].index_copy_(0,idx,info['kv'].to(tdest['kv'].dtype))
                tdest['o'].index_copy_(0,idx+tdest['obase'],info['o'].float())
                if tdest.get('s') is not None:tdest['s'].index_copy_(0,idx,info['s'].to(tdest['s'].dtype))
            self.done.record(stream)
            actor_part('sampling_transfer',stamp)
        if PROFILE_ACTOR and self.calls%256==0:
            import json
            from pathlib import Path
            base=Path(PROFILE_ACTOR)/f'actor-{os.getpid()}-{id(self)}'
            base.with_suffix('.json').write_text(json.dumps(dict(calls=self.calls,seconds=ACTOR_TIMES)))
            if self.calls==256:
                self.profile.disable()
                self.profile.dump_stats(str(base.with_suffix('.pstats')))
        return True

    def result(self):
        self.done.synchronize()
        out=self._cpu.numpy().copy();n=self.nh
        self.extra=out[:,n+2:n+2+self.n_extra] if self.n_extra else None
        self.vg=out[:,-1] if self.vg_on else None
        return out[:,:n].astype(np.int64),out[:,n],out[:,n+1]

    def act(self,*args,**kwargs):
        return self.result() if self.submit(*args,**kwargs) else None


def rollout_requests(rollout, flat):
    """The raw prefix BEFORE each selected decision, across rollout boundaries."""
    rows=rollout.rows.reshape(-1);pending=rollout.pending.reshape(-1,rollout.nh)
    return [(rollout.raw_contexts[int(i)//rollout.cap][int(i)%rollout.cap],
             rows[int(i)],pending[int(i)]) for i in flat]


class SourceView(SourceBatch):
    """One minibatch's gradient leaves over a rollout-resident numerical bank.

    No encoder replay and no repeated host/source-device materialization. Reader
    K/V is rebuilt with current fast weights, and cotangents go to the original
    source graphs. Only selected Spatial queries allocate a Y gradient buffer.
    """
    def __init__(self, parent, indices):
        self.__dict__=parent.__dict__.copy()
        self.source_version=self.pipeline.version
        self.requests=[parent.requests[i] for i in indices]
        self.query_groups=parent.query_groups[indices]
        query=np.asarray([parent.history_frames[h.uid][h.length] for h,_,_ in self.requests])
        self.unique_query,inverse=np.unique(query,return_inverse=True)
        ix,self.query_inverse,yi=reader_indices((indices,inverse,np.searchsorted(parent.unique_query,self.unique_query)),parent.u.device)
        self.query=parent.query.index_select(0,ix)
        self.u_query=parent.u_query.index_select(0,ix)
        self.query_times=parent.query_times.index_select(0,ix)
        self.query_frozen=None if parent.query_frozen is None else parent.query_frozen.index_select(0,ix)
        if self.pipeline.period == 0:
            self.parent=parent
            self.y=parent.y.index_select(0,yi)
        else:
            self.y=parent.y.index_select(0,yi).detach().requires_grad_()
        self.pad=parent.pad.index_select(0,yi)
        self.u=parent.u if self.pipeline.period == 0 else parent.u.detach().requires_grad_()
        self.mem=parent.mem if self.pipeline.period == 0 else parent.mem.detach().requires_grad_()
        refs,_,mrefs=parent.pipeline_refs
        self.pipeline_refs=refs,[refs[i] for i in self.unique_query],mrefs
        src=self.sources()
        horizon=self.pipeline.horizon or self.query_frozen is not None
        self.kv_source=src if horizon else None
        self.kvs=[layer.kv(src.detach() if horizon else src).float() for layer in self.model.reader.layers]


class JoinedSources:
    """Keep small teacher/branch additions separate from the resident rollout."""
    def __init__(self, main, extra):
        self.main,self.extra=main,extra
        self.size=len(main.requests)
        self.cached=main.cached and extra.cached

    def forward(self,indices,need_h=False,need_vg=False):
        indices=indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
        left=np.flatnonzero(indices<self.size);right=np.flatnonzero(indices>=self.size)
        if not len(right):return self.main.forward(indices,need_h,need_vg)
        if not len(left):return self.extra.forward(indices-self.size,need_h,need_vg)
        a=self.main.forward(indices[left],need_h,need_vg)
        b=self.extra.forward(indices[right]-self.size,need_h,need_vg)
        order=torch.as_tensor(np.argsort(np.concatenate([left,right])),device=self.main.u.device)
        return tuple(torch.cat([x,y]).index_select(0,order) for x,y in zip(a,b))

    def backward(self):
        self.main.backward();self.extra.backward()
