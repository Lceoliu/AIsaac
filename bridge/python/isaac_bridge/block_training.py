"""Source-bearing block-memory actor/learner and teacher adapters.

These adapters reuse TokPolicy's action masks, the existing PPO/decoupled loss and
the existing CF loss. Sources are immutable and travel with a snapshot, rather
than process-local integer ids. No graph is retained between optimiser updates.
"""
from dataclasses import dataclass, replace
import io
import os
import threading
import time
import hashlib
from functools import cached_property, lru_cache
from contextlib import contextmanager

import numpy as np
import torch
import lz4.frame
from torch.nn import functional as F

from .block_memory import historical_blocks
from .tok_policy import to_batch, crn_uniforms, decode_rows, ROW_BYTES
from .tok_obs import ROW
from .tok_pickup import ppo_part_terms
from .tok_teacher2 import teacher_v2_loss
from .structured_input import from_rows, FRAME_BYTES, VERSION
from .block_paged import PagedSnapshot, PagedPayload, forward_chunks_paged
from .block_nvtx import span as nv_span

CUDA_CAPTURE_ADMISSION = threading.RLock()
CAPTURE_WAITING = threading.Event()
_CAPTURE_RETRY = {}


@contextmanager
def learner_cuda_activity():
    """No CUDA capture may overlap submissions from the learner thread."""
    from .block_paged import ucount, LEARNER_ACTIVE
    t=time.perf_counter()
    with CUDA_CAPTURE_ADMISSION:
        ucount('learn_admission_wait_s',time.perf_counter()-t)
        LEARNER_ACTIVE[0]=True
        try:
            yield
        finally:
            LEARNER_ACTIVE[0]=False


def capture_yield():
    """Learner thread, between forward groups (no CUDA call in progress): if an actor
    graph is waiting to be captured, hand it the admission and take it back after.
    Without this a learner busy for the whole update kept every graph key first seen
    during learning (all teacher-generation actors) on the eager path."""
    if CAPTURE_WAITING.is_set() and CUDA_CAPTURE_ADMISSION._is_owned():
        from .block_paged import ucount
        t=time.perf_counter()
        state=CUDA_CAPTURE_ADMISSION._release_save()
        time.sleep(0.002)
        CUDA_CAPTURE_ADMISSION._acquire_restore(state)
        ucount('learn_yield_wait_s',time.perf_counter()-t);ucount('learn_yields_given',1)


@contextmanager
def graph_capture(graph,stream,pool=None):
    """torch.cuda.graph(graph, stream=stream) without its gc.collect() and empty_cache().

    B42 prof-f3: the full collection over this process's Python objects (page store, logs,
    rollouts) took ~5.7 s per capture while the learner, which had yielded the capture
    admission, waited. The admission protocol is unchanged: no other CUDA work is submitted
    while this capture runs. The caching allocator keeps its blocks (no empty_cache). pool (B46):
    a graph_pool_handle shared with other graphs that never replay concurrently."""
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool,capture_error_mode='global')
        try:
            yield graph
        finally:
            graph.capture_end()


def capture_admission(key):
    """Actor thread: admission to capture a new graph key (bounded wait, spaced retries)."""
    if CUDA_CAPTURE_ADMISSION.acquire(blocking=False):
        return True
    from .block_paged import ucount
    now=time.monotonic()
    if now<_CAPTURE_RETRY.get(key,0.0):
        ucount('actor_admission_misses',1)
        return False
    CAPTURE_WAITING.set()
    try:
        got=CUDA_CAPTURE_ADMISSION.acquire(timeout=0.5)
    finally:
        CAPTURE_WAITING.clear()
    ucount('actor_admission_wait_s',time.monotonic()-now)
    if not got:
        _CAPTURE_RETRY[key]=now+10.0
        ucount('actor_admission_misses',1)
    return got


@dataclass(frozen=True)
class PackedRows:
    """Lossless model-input history, not a copy of the legacy transport envelope."""
    data: bytes
    count: int
    visible_width: int = FRAME_BYTES
    input_only: bool = False  # absent in old pickles, whose data still uses ROW

    @classmethod
    def pack(cls, rows):
        rows=source_rows([rows])
        return cls(lz4.frame.compress(rows.tobytes(),compression_level=0),len(rows),
                   rows['visible_frame'].shape[1],True)

    def compact_array(self):
        return np.frombuffer(lz4.frame.decompress(self.data),dtype=source_dtype(self.visible_width,self.input_only))

    def tobytes(self):
        return np.asarray(self).tobytes()

    def __len__(self):
        return self.count

    def __array__(self,dtype=None,copy=None):
        rows=np.asarray(self.compact_array(),dtype=dtype)
        return rows.copy() if copy else rows


@lru_cache(maxsize=128)
def source_dtype(width,input_only=False):
    if input_only:
        return np.dtype([('first','u1'),('visible_schema','<i4'),('visible_bytes','<i4'),
                         ('visible_frame','u1',(width,))],align=True)
    if width==FRAME_BYTES:return ROW
    return np.dtype(dict(names=ROW.names,
        formats=[np.dtype(('u1',(width,))) if name=='visible_frame' else ROW.fields[name][0]
                 for name in ROW.names],offsets=[ROW.fields[name][1] for name in ROW.names],
        itemsize=ROW.itemsize-FRAME_BYTES+width))


def source_rows(sources):
    """Assemble one Spatial microbatch, not a full padded historical ROW pool."""
    arrays=[s.compact_array() if isinstance(s,PackedRows) else s for s in sources]
    width=max(max(4,int(a['visible_bytes'].max())) for a in arrays)
    dtype=source_dtype(width,True)
    rows=np.zeros(sum(map(len,arrays)),dtype=dtype)
    at=0
    for array in arrays:
        for name in ('first','visible_schema','visible_bytes'):rows[name][at:at+len(array)]=array[name]
        for i,n in enumerate(array['visible_bytes']):
            rows['visible_frame'][at+i,:n]=array['visible_frame'][i,:n]
        at+=len(array)
    return rows


def frame_rows(rows, frames=None):
    """rows as they are when they hold visible_frame; compact ROWC rows (B45) as source_dtype records (first,
    visible_schema, visible_bytes, visible_frame) read from the frame bytes `frames`."""
    if 'visible_frame' in rows.dtype.names:
        return rows
    if frames is None:
        raise ValueError('compact records need their frame arena')
    size = np.asarray(rows['visible_bytes'], np.int64)
    out = np.zeros(len(rows), dtype=source_dtype(max(4, int(size.max()) if len(size) else 4), True))
    for name in ('first', 'visible_schema', 'visible_bytes'):
        out[name] = rows[name]
    for i, (o, n) in enumerate(zip(np.asarray(rows['visible_off'], np.int64), size)):
        out['visible_frame'][i, :n] = frames[o:o+n]
    return out


@dataclass(frozen=True)
class RawBlock:
    end_step: int
    rows: np.ndarray
    pending: np.ndarray
    versions: tuple

    def __post_init__(self):
        if isinstance(self.rows,np.ndarray):
            object.__setattr__(self,'rows',PackedRows.pack(self.rows))

    def __setstate__(self,state):
        # Previously saved new-input jobs carry ndarrays in this same field.
        self.__dict__.update(state)
        self.__post_init__()

    @cached_property
    def encoding_key(self):
        # Equal independent Writer inputs can share one differentiable forward
        # within a minibatch, even after separate IPC messages. This digest replaces
        # expensive duplicate Spatial/Writer work; it is not an integrity check.
        key=hashlib.blake2b(self.rows.data,digest_size=16)
        key.update(self.pending.tobytes())
        return key.digest()


@dataclass(frozen=True)
class RawHistory:
    episode: tuple
    step: int = 0
    blocks: tuple = ()
    tail: tuple = ()

    def __post_init__(self):
        if any(isinstance(row,np.ndarray) for row,_,_ in self.tail):
            object.__setattr__(self,'tail',tuple(
                (PackedRows.pack(row) if isinstance(row,np.ndarray) else row,pending,version)
                for row,pending,version in self.tail))

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.__post_init__()

    def append(self, row, pending, version, config):
        row, pending = PackedRows.pack(row), pending.copy()
        pending.flags.writeable = False
        tail = self.tail + ((row, pending, int(version)),)
        blocks = self.blocks
        if len(tail) == config.block_steps:
            rows = source_rows([x[0] for x in tail])
            actions = np.concatenate([x[1] for x in tail])
            rows.flags.writeable = actions.flags.writeable = False
            block = RawBlock(self.step, rows, actions, tuple(x[2] for x in tail))
            blocks = (blocks + (block,))[-config.memory_blocks:]
            tail = ()
        return RawHistory(self.episode, self.step + 1, blocks, tail)


@dataclass(frozen=True)
class Snapshot:
    memory: object
    history: RawHistory
    pending: np.ndarray
    policy_version: int
    kv_rows: object = None

    def fork(self):
        # MemoryBlock and RawBlock values are immutable; only writable cache/tail
        # tensors and the in-flight action need independent storage.
        return replace(self, memory=self.memory.fork(), pending=self.pending.copy())

    def to(self, device):
        return move_snapshots([self],device)[0]


class SpatialEncoderGraph:
    """GraphActor's raw staging/capture pattern, restricted to stateless Spatial.

    Each service owns its graphs and capture stream; neither Reader state nor
    sampling is captured. Lazy row buckets keep two pinned teacher generations cheap.
    """
    def __init__(self,model,slots,entities,bf16):
        self.model,self.slots,self.entities,self.bf16=model,slots,int(entities),bool(bf16)
        self.device=next(model.parameters()).device
        self.graphs={}
        self.busy_misses=0
        self.captures=0
        self.replays=0
        self.capture=torch.cuda.Stream(device=self.device)

    @torch.inference_mode()
    def __call__(self,rows,pending):
        # Decode complete ragged frames outside capture. Shape key includes every
        # P/E/N/S bucket, never a legacy n_ent-only capacity or truncated inventory.
        source=from_rows(rows,self.device,pending)
        tokens=30+sum(source['v_'+k].shape[1] for k in ('body','local','entity','inventory','slots'))
        if tokens>1024:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
                return source,self.model.encode_block(source,packed=False)
        # Bucket allocations upward, never drop entities/items/slots. Exact
        # batch/count keys churned the 32-entry cache and repeatedly synchronized
        # the GPU for capture as main and search batch sizes varied.
        n=len(rows)
        capacity=1<<(n-1).bit_length()
        shapes={}
        ragged=tuple('v_'+name for name in ('body','local','entity','inventory','slots'))
        for k,v in source.items():
            shape=list(v.shape);shape[0]=capacity
            if k in ragged or k.removesuffix('_mask') in ragged:
                shape[1]=1<<(shape[1]-1).bit_length()
                # B42: entity bucket {16,32,64,...,entities}, not always the 128 cap: a typical
                # row has ~10 real entities, and E=128 tripled the actor's Spatial tokens.
                minimum={'v_entity':min(16,self.entities),'v_inventory':32,'v_slots':16}.get(k.removesuffix('_mask'),1)
                shape[1]=max(shape[1],minimum)
                if k.removesuffix('_mask')=='v_entity':shape[1]=min(shape[1],max(self.entities,v.shape[1]))
            shapes[k]=tuple(shape)
        padded_tokens=30+sum(shapes[k][1] for k in ragged)
        if padded_tokens>1024:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
                return source,self.model.encode_block(source,packed=False)
        key=tuple((k,shapes[k],v.dtype) for k,v in source.items())
        if key not in self.graphs:
            if not capture_admission((id(self),key)):
                self.busy_misses+=1
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
                    return source,self.model.encode_block(source,packed=False)
            try:
                if len(self.graphs)>=128:self.graphs.pop(next(iter(self.graphs)))
                with torch.inference_mode(False):   # B42: normal tensors (compiled Spatial guards dispatch keys)
                    static={k:v.new_zeros(shapes[k]) for k,v in source.items()}
                    for k,v in source.items():
                        static[k][tuple(slice(0,s) for s in v.shape)].copy_(v)
                # B42: the autocast cache is cleared before capture (casts recorded inside the
                # graph) rather than disabled: a disabled cache is a different dynamo global state
                # (the compiled Spatial recompiled for this context).
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
                    for _ in range(3):self.model.encode_block(static,packed=False)
                    torch.cuda.synchronize(self.device)
                    torch.clear_autocast_cache()
                    graph=torch.cuda.CUDAGraph()
                    t_cap=time.perf_counter()
                    with graph_capture(graph,self.capture):
                        z=self.model.encode_block(static,packed=False)
                self.graphs[key]=(static,graph,z)
                self.captures+=1
                from .block_paged import ucount
                ucount('actor_captures_other',1);ucount('actor_capture_other_s',time.perf_counter()-t_cap)
                from .block_paged import late_capture
                late_capture('spatial',(n,)+tuple((k,s_) for k,s_,_ in key if k in ('v_body','v_entity')))
            finally:
                CUDA_CAPTURE_ADMISSION.release()
        static,graph,z=self.graphs[key]
        for k,v in source.items():
            if k.endswith('_mask'):static[k].zero_()
            static[k][tuple(slice(0,s) for s in v.shape)].copy_(v,non_blocking=True)
        graph.replay()
        self.replays+=1
        return source,z[:n]


class ReaderStepGraph:
    """Capture the pure tensor Reader, retaining all state in explicit inputs.

    PyTorch CUDA graphs eliminate per-layer Python/kernel launch overhead. Cache
    slots are bucketed upward; masks retain exact short/long causal visibility.
    Writer, immutable history, generation ownership and sampling remain outside.
    """
    def __init__(self,model,bf16):
        self.memory=model.block_memory
        self.bf16=bool(bf16)
        self.graphs={}
        self.captures=self.replays=0
        self.capture=torch.cuda.Stream(device=next(model.parameters()).device)

    @torch.inference_mode()
    def __call__(self,z,position,prefixes,bank,ends,kp):
        n=len(z);cfg=self.memory.config
        batch=1<<(n-1).bit_length()
        bank_size=max(256,1<<(bank.shape[1]-1).bit_length())
        prefix_size=cfg.window*cfg.latents
        prefix_dtype=torch.bfloat16 if self.bf16 else prefixes[0][0].dtype
        key=(batch,bank_size,z.dtype,bank.dtype,prefix_dtype)
        if key not in self.graphs:
            if not capture_admission((id(self),key)):
                self.busy_misses=getattr(self,'busy_misses',0)+1
                return self.memory.read(z,position,prefixes,bank,ends,kp)
            try:
                sz=z.new_zeros((batch,*z.shape[1:]))
                sp=position.new_zeros((batch,1))
                sb=bank.new_zeros((batch,bank_size,bank.shape[-1]))
                se=ends.new_full((batch,bank_size),-1)
                sk=kp.new_full((batch,prefix_size),-1)
                spr=tuple(tuple(x.new_zeros((batch,x.shape[1],prefix_size,x.shape[-1]),dtype=prefix_dtype)
                                for x in layer) for layer in prefixes)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16,cache_enabled=False):
                    for _ in range(3):self.memory.read(sz,sp,spr,sb,se,sk)
                    torch.cuda.synchronize(z.device)
                    graph=torch.cuda.CUDAGraph()
                    t_cap=time.perf_counter()
                    with graph_capture(graph,self.capture):
                        result=self.memory.read(sz,sp,spr,sb,se,sk)
                if len(self.graphs)>=32:self.graphs.pop(next(iter(self.graphs)))
                self.graphs[key]=(sz,sp,spr,sb,se,sk,graph,result)
                self.captures+=1
                from .block_paged import ucount
                ucount('actor_captures_other',1);ucount('actor_capture_other_s',time.perf_counter()-t_cap)
                from .block_paged import late_capture
                late_capture('reader',key)
            finally:
                CUDA_CAPTURE_ADMISSION.release()
        sz,sp,spr,sb,se,sk,graph,result=self.graphs[key]
        sz[:n].copy_(z);sp[:n].copy_(position)
        sb[:n,:bank.shape[1]].copy_(bank)
        se.fill_(-1);se[:n,:ends.shape[1]].copy_(ends)
        sk.fill_(-1);sk[:n,:kp.shape[1]].copy_(kp)
        for dest,source in zip(spr,prefixes):
            for d,s in zip(dest,source):d[:n,:,:s.shape[2]].copy_(s)
        graph.replay();self.replays+=1
        hidden,tokens,kv,allowed,mask=result
        # With window=1 the canonical next cache would otherwise alias reusable
        # graph output. Longer windows already own their concatenated cache.
        kv=tuple(tuple(x[:n].clone() if cfg.window==1 else x[:n] for x in layer) for layer in kv)
        return hidden[:n],tokens[:n],kv,allowed[:n],mask[:n]


class BlockActor:
    """Batched normal/search policy service, with explicit snapshot ownership."""
    def __init__(self, model, slots, graph_entities=0, bf16=False):
        if model.block_memory is None:
            raise ValueError("BlockActor requires a block-memory policy")
        self.model = model
        self.graph_entities,self.bf16=int(graph_entities),bool(bf16)
        self.encoder=SpatialEncoderGraph(model,slots,graph_entities,bf16) if graph_entities else None
        self.reader=ReaderStepGraph(model,bf16) if graph_entities else None
        self.memory = [model.block_memory.empty() for _ in range(slots)]
        self.history = [RawHistory((os.getpid(), i, 0)) for i in range(slots)]
        self.version = 0
        self.kv_rows = [() for _ in range(slots)]
        self.last_contexts = []
        self.pending = [np.zeros((1, len(model.heads)), dtype=np.int64) for _ in range(slots)]

    def snapshot(self, slot):
        memory=self.memory[slot]
        # Snapshot prefixes share immutable per-decision KV, not a full 128-step
        # copy per teacher record. Only the live actor owns a dense rolling cache.
        empty_kv=tuple((k.new_empty((k.shape[0],0,k.shape[2])),v.new_empty((v.shape[0],0,v.shape[2])))
                       for k,v in memory.kv)
        return Snapshot(replace(memory,kv=empty_kv),self.history[slot],self.pending[slot].copy(),
                        self.version,self.kv_rows[slot])

    def restore(self, slot, snapshot):
        s = snapshot.to(next(self.model.parameters()).device).fork()
        self.memory[slot], self.history[slot], self.pending[slot] = s.memory, s.history, s.pending
        if s.kv_rows is not None:
            self.kv_rows[slot]=s.kv_rows
        else:
            n=self.model.block_memory.config.latents
            self.kv_rows[slot]=tuple(torch.stack([torch.stack([x[:,j:j+n] for x in layer]) for layer in s.memory.kv])
                                     for j in range(0,len(s.memory.kv_steps),n))

    @torch.no_grad()
    def publish(self, weights, version):
        # External async service must drain its in-flight call before publication.
        self.model.load_state_dict(weights)
        self.version = int(version)
        # Canonical Writer values/short KV retain their labelled versions.
        # This implementation projects long KV inside Reader on every call, so no
        # obsolete long projection can survive publication.

    @torch.inference_mode()
    def step(self, rows, pending, slots, ukeys=None, greedy=False, evaluate=False,
             owners=None, uniform=None):
        slots = list(map(int, slots))
        owners = [self] * len(slots) if owners is None else owners
        if any(owner.version != self.version for owner in owners):
            raise ValueError("a combined inference batch must use one policy version")
        device = next(self.model.parameters()).device
        for i, (owner, slot) in enumerate(zip(owners, slots)):
            if rows["first"][i]:
                episode = (os.getpid(), slot, int(rows["episode"][i]))
                owner.memory[slot] = self.model.block_memory.empty(episode=episode)
                owner.history[slot] = RawHistory(episode)
                owner.kv_rows[slot] = ()
            owner.pending[slot] = pending[i:i+1].copy()
        contexts = [owner.snapshot(slot) for owner, slot in zip(owners, slots)]
        ctx = {"states": [owner.memory[slot] for owner, slot in zip(owners, slots)],
               "version": self.version,"reset_done":True}
        if self.reader is not None:ctx['reader']=self.reader
        if self.encoder is None:
            b = from_rows(rows, device, pending)
        else:
            b,ctx['encoded']=self.encoder(rows,pending)
        replay=np.broadcast_to(np.asarray(evaluate,dtype=bool),(len(rows),))
        all_replay=bool(replay.all())
        if ukeys is not None and not all_replay:
            uniform = torch.as_tensor(crn_uniforms(ukeys[0], ukeys[1], sum(self.model.heads)), device=device)
        if all_replay:
            output=self.model(b,need_vg=True,temporal=ctx)
            result=(b['pending'],torch.zeros(len(rows),device=device),output[1],output[-1])
        else:
            if replay.any() and uniform is None and not greedy:
                uniform=torch.full((len(rows),sum(self.model.heads)),.5,device=device)
                uniform[np.flatnonzero(~replay).tolist()]=torch.rand((int((~replay).sum()),sum(self.model.heads)),device=device)
            result = self.model.act(b, u=uniform, greedy=greedy, need_vg=True, temporal=ctx)
            if replay.any():
                mask=torch.from_numpy(replay.copy())
                if device.type=='cuda':mask=mask.pin_memory()
                mask=mask.to(device,non_blocking=device.type=='cuda')
                result=(torch.where(mask[:,None],b['pending'],result[0]),
                        result[1].masked_fill(mask,0),*result[2:])
        # One result transfer per batch, not one blocking action copy per worker.
        host=torch.cat((result[0].float(),*(x[:,None].float() for x in result[1:])),1).cpu().numpy()
        for owner in dict.fromkeys(owners):
            positions=[i for i,o in enumerate(owners) if o is owner]
            owner.host_output=host[positions]
            owner.last_contexts=[contexts[i] for i in positions]
        deltas=torch.stack([torch.stack(layer) for layer in ctx['block_result']['current_kv']]).permute(2,0,1,3,4,5)
        for i, (owner, slot) in enumerate(zip(owners, slots)):
            history = owner.history[slot].append(rows[i:i+1], pending[i:i+1], self.version,
                                                self.model.block_memory.config)
            sources = {raw.end_step: raw for raw in history.blocks}
            memory = ctx["states_out"][i]
            memory = replace(memory, blocks=tuple(replace(m, source=sources[m.end_step])
                             if m.source is None else m for m in memory.blocks))
            owner.memory[slot], owner.history[slot] = memory, history
            cfg=self.model.block_memory.config
            delta=deltas[i]
            owner.kv_rows[slot]=(owner.kv_rows[slot]+(delta,))[-cfg.window:]
            # Snapshot is now before the NEXT observation. Its in-flight action
            # is this answer, not the preceding observation's pending action.
            owner.pending[slot] = host[i:i+1,:len(self.model.heads)].astype(np.int64)
        return result


@dataclass(frozen=True)
class Chunk:
    """One episode, contiguous decisions, snapshot before its first observation. rows: ROW records, or (B45) compact
    ROWC records whose frames are in `frames` (a byte array: the rollout's FrameArena buffer)."""
    rows: np.ndarray
    pending: np.ndarray
    snapshot: Snapshot
    frames: object = None

    def framed(self):
        """Records with their frames inline (ROW, or source_dtype records for compact rows)."""
        return frame_rows(self.rows, self.frames)

    def __post_init__(self):
        if len(self.rows) != len(self.pending):
            raise ValueError("rows/pending lengths differ")
        if len(self.rows) > 1 and self.rows["first"][1:].any():
            raise ValueError("a Chunk cannot cross an episode reset")

    def to_snapshot(self, device):
        return self.snapshot.to(device)


def move_snapshots(snapshots, device, materialize=True):
    # Several chunk starts share most of the long bank. Transfer each immutable
    # block once, rather than copying an entire 2048-token bank for every chunk.
    target_device=torch.device(device)
    unique_blocks={id(block):block for snap in snapshots for block in snap.memory.blocks}
    blocks=unique_blocks
    if unique_blocks and any(block.value.device!=target_device for block in unique_blocks.values()):
        packed_blocks=torch.stack([block.value for block in unique_blocks.values()])
        if packed_blocks.device.type=='cpu' and target_device.type=='cuda':
            packed_blocks=packed_blocks.pin_memory()
        packed_blocks=packed_blocks.to(device,non_blocking=target_device.type=='cuda')
        blocks={key:replace(block,value=value) for (key,block),value
                in zip(unique_blocks.items(),packed_blocks.unbind())}
    if target_device.type=='cpu' and not materialize:
        # A replay entry must not pin every other owner's tensor in its packet.
        blocks={key:replace(block,value=block.value.clone())
                if block.value.untyped_storage().nbytes()>block.value.numel()*block.value.element_size()
                else block for key,block in blocks.items()}
    unique_rows={id(row):row for snap in snapshots if snap.kv_rows is not None for row in snap.kv_rows}
    moved={}
    packed = None
    if unique_rows:
        keys=list(unique_rows)
        rows=list(unique_rows.values())
        if materialize:
            # Time is contiguous after selecting refs: one gather for every
            # snapshot, instead of stack/permute/copy for each chunk start.
            packed_rows=torch.stack(rows,dim=3)
            if packed_rows.device.type=='cpu' and target_device.type=='cuda':
                packed_rows=packed_rows.pin_memory()
            packed_rows=packed_rows.to(device,non_blocking=target_device.type=='cuda')
            moved=dict(zip(keys,packed_rows.unbind(3)))
            lookup={key:i for i,key in enumerate(keys)}
            indices=torch.tensor([lookup[id(row)] for snap in snapshots
                                  if snap.kv_rows is not None for row in snap.kv_rows],
                                 device=device,dtype=torch.long)
            packed=packed_rows.index_select(3,indices).flatten(3,4)
        elif all(row.device==torch.device(device) for row in rows):
            moved=unique_rows
        else:
            moved=dict(zip(keys,torch.stack(rows).to(device).unbind()))
        if target_device.type=='cpu' and not materialize:
            moved={key:row.clone() if row.untyped_storage().nbytes()>row.numel()*row.element_size()
                   else row for key,row in moved.items()}
    # These three fields used to issue three blocking transfers per snapshot.
    # Pack their bytes once; preserve each field's dtype/shape and CPU ownership.
    metadata=[x for snap in snapshots for x in
              (snap.memory.kv_steps,snap.memory.kv_versions,snap.memory.partial)]
    if metadata and metadata[0].device!=target_device:
        sizes=[x.numel()*x.element_size() for x in metadata]
        slab=torch.cat([x.contiguous().view(torch.uint8).reshape(-1) for x in metadata])
        if slab.device.type=='cpu' and target_device.type=='cuda':slab=slab.pin_memory()
        slab=slab.to(device,non_blocking=target_device.type=='cuda')
        metadata=[part.view(original.dtype).reshape(original.shape)
                  for part,original in zip(slab.split(sizes),metadata)]
    if target_device.type=='cpu' and not materialize:
        metadata=[x.clone() if x.untyped_storage().nbytes()>x.numel()*x.element_size() else x
                  for x in metadata]
    meta_iter=iter(metadata)
    host = []
    cursor = 0
    for snap in snapshots:
        memory = snap.memory
        refs=None if snap.kv_rows is None else tuple(moved[id(row)] for row in snap.kv_rows)
        if refs is not None and materialize and refs:
            length=len(refs)*refs[0].shape[-2]
            kv=tuple((layer[0,: ,cursor:cursor+length],layer[1,: ,cursor:cursor+length])
                     for layer in packed)
            cursor+=length
        else:
            kv=tuple((k.to(device),v.to(device)) for k,v in memory.kv)
        memory = replace(memory,
                         kv=kv,
                         kv_steps=next(meta_iter), kv_versions=next(meta_iter), partial=next(meta_iter),
                         blocks=tuple(blocks[id(b)] for b in memory.blocks))
        host.append(replace(snap, memory=memory,kv_rows=refs))
    return host


def pack_snapshots(snapshots):
    """Trusted internal IPC/checkpoint payload; references are deduplicated by pickle."""
    if snapshots and all(isinstance(s,PagedSnapshot) for s in snapshots):
        return PagedPayload(snapshots)   # in-process hand-off: descriptors, no serialisation
    import lz4.frame
    stream = io.BytesIO()
    from .structured_input import SCHEMA
    # Protocol 2 converts NumPy's large ROW byte arrays through Latin-1 strings.
    # Protocol 5 keeps them binary; LZ4 removes transport padding losslessly before
    # queue IPC. No raw historical observation or state/gradient budget is dropped.
    torch.save(dict(input_schema=SCHEMA,snapshots=move_snapshots(snapshots,"cpu",materialize=False)),
               stream,pickle_protocol=5)
    return lz4.frame.compress(stream.getvalue(),compression_level=0)


def unpack_snapshots(payload, device):
    # Only accept payloads emitted by this training job, not arbitrary user files.
    if isinstance(payload,PagedPayload):
        return list(payload)
    from .structured_input import SCHEMA
    if payload.startswith(b'\x04\x22\x4d\x18'):
        import lz4.frame
        payload=lz4.frame.decompress(payload)
    envelope = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(envelope,dict) or envelope.get('input_schema') != SCHEMA:
        raise ValueError('old snapshot input schema rejected')
    snapshots = envelope['snapshots']
    return move_snapshots(snapshots,device,materialize=False)


class BlockRollout:
    """Chunk-start snapshots plus one open decision per worker, not one KV per row.

    The original Rollout retains observations, rewards, behaviour probabilities,
    forced actions and episode accounting. This sidecar only owns memory inputs.
    """
    def __init__(self, chunk_steps=128):
        self.chunk_steps = int(chunk_steps)
        self.starts = {}
        self.open = {}

    def record(self, worker, index, row, snapshot):
        worker, index = int(worker), int(index)
        starts = self.starts.setdefault(worker, {})
        if not starts or row["first"] or index - max(starts) >= self.chunk_steps:
            starts[index] = snapshot
        self.open[worker] = (index, snapshot)

    def reset(self):
        self.starts.clear()
        self.open.clear()

    def carry(self, other, workers, indices):
        self.reset()
        for worker, index in zip(workers, indices):
            previous, snapshot = other.open[int(worker)]
            if previous != index:
                raise ValueError("open decision and memory snapshot are out of sync")
            self.starts[int(worker)] = {0: snapshot}
            self.open[int(worker)] = (0, snapshot)

    def payload(self):
        keys, snapshots = [], []
        for worker, starts in self.starts.items():
            for index, snapshot in starts.items():
                keys.append((worker,index))
                snapshots.append(snapshot)
        return self.chunk_steps, keys, pack_snapshots(snapshots)

    @classmethod
    def from_payloads(cls, parts, device="cpu"):
        """parts = [(worker_offset, payload), ...], in the existing actor order."""
        out = cls(parts[0][1][0])
        for offset, (size, keys, packed) in parts:
            if size != out.chunk_steps:
                raise ValueError("actors must agree on chunk length")
            for (worker,index), snapshot in zip(keys,unpack_snapshots(packed,device)):
                out.starts.setdefault(worker+int(offset), {})[index] = snapshot
        return out

    def plan(self, rollout, mask):
        """Whole, same-episode chunks. Positions retain the native Rollout layout."""
        chunks, positions, weights, table = [], [], [], []
        offset = 0
        for worker in range(rollout.n):
            starts = self.starts.get(worker, {})
            indices = sorted(starts)
            for j, begin in enumerate(indices):
                end = indices[j+1] if j+1 < len(indices) else int(rollout.length[worker])
                if end <= begin or not mask[worker,begin:end].any():
                    continue
                if end-begin > self.chunk_steps:
                    raise ValueError("missing chunk-start snapshot in rollout")
                arena=getattr(rollout,'frames',None)   # (B45: compact records' frames)
                chunks.append(Chunk(rollout.rows[worker,begin:end],rollout.pending[worker,begin:end],starts[begin],
                                    arena.buf if arena is not None else None))
                positions.extend(range(worker*rollout.cap+begin,worker*rollout.cap+end))
                weights.extend(mask[worker,begin:end])
                table.append((offset,end-begin,worker,begin))
                offset += end-begin
        return dict(chunks=chunks,pos=np.asarray(positions,np.int64),weight=np.asarray(weights,np.float32),
                    tch=np.asarray(table,np.int64).reshape(-1,4))


def forward_bucketed(model, chunks, max_rows=None, **kwargs):
    """Parallelise equal-length chunks; restore the caller's exact decision order."""
    # This interface returns representations/loss inputs, never continuation state.
    kwargs.setdefault('require_next_state',False)
    if chunks and isinstance(chunks[0].snapshot,PagedSnapshot):
        from . import learner_dump
        learner_dump.record(chunks,dict(kwargs,max_rows=max_rows))   # (B44 fixed-input benchmark; measurement only)
        from . import block_learner
        if block_learner.PACKED:   # B44: one execution per part (ISAAC_PACKED_LEARNER=0: per exact length)
            with nv_span('learner.forward_packed'):
                return block_learner.forward_packed(model,chunks,max_rows=max_rows,**kwargs)
    by_length = {}
    for index, chunk in enumerate(chunks):
        by_length.setdefault(len(chunk.rows), []).append(index)
    results, contexts = {}, {}
    replays = []
    stats=dict.fromkeys(('selected_blocks','unique_objects','encoded_blocks','historical_rows','tail_rows'),0)
    for length, group in by_length.items():
        size = len(group) if max_rows is None else max(1,int(max_rows)//length)
        for at in range(0,len(group),size):
            capture_yield()
            indices = group[at:at+size]
            output, _, ctx, _ = forward_chunks(model,[chunks[i] for i in indices],**kwargs)
            for key,value in ctx['recompute_stats'].items():stats[key]+=value
            if ctx.get('replay') is not None:replays.append(ctx['replay'])
            for j,index in enumerate(indices):
                results[index] = tuple(x[j*length:(j+1)*length] for x in output)
                contexts[index] = dict(o=ctx["o_seq"][j*length:(j+1)*length],
                                       z=tuple(x[j*length:(j+1)*length] for x in ctx["z_out"]))
    output = tuple(torch.cat([results[i][k] for i in range(len(chunks))]) for k in range(len(results[0])))
    ctx = dict(o_seq=torch.cat([contexts[i]["o"] for i in range(len(chunks))]),
               z_out=tuple(torch.cat([contexts[i]["z"][k] for i in range(len(chunks))]) for k in range(2)),
               recompute_stats=stats,replay=replays)
    return output,ctx


def forward_chunks(model, chunks, generator=None, train_history=True, version=0, spatial_micro=None,
                   require_next_state=True,deduplicate_sources=True,history_mode='random'):
    """Same-length chunks are one batch; caller buckets variable lengths.

    A configured budget of window-external blocks is differentiable. The raw cross-boundary
    Writer tail is separately re-encoded, and does not consume that budget.
    """
    if not chunks:
        raise ValueError("forward_chunks requires a nonempty batch")
    if isinstance(chunks[0].snapshot,PagedSnapshot):
        return forward_chunks_paged(model,chunks,generator=generator,train_history=train_history,version=version,
                                    spatial_micro=spatial_micro,require_next_state=require_next_state,
                                    deduplicate_sources=deduplicate_sources,history_mode=history_mode)
    if history_mode!='random':
        if history_mode!='detach':
            raise ValueError('history_mode gradreplay needs the paged slot path (--block-slots 1)')
        train_history=False   # A arm: no historical recompute (= history_recompute_blocks 0)
    spatial_micro = 512 if spatial_micro is None else spatial_micro
    length = len(chunks[0].rows)
    if any(len(c.rows) != length for c in chunks):
        raise ValueError("bucket chunks by decision length before forward_chunks")
    device = next(model.parameters()).device
    snaps = move_snapshots([c.snapshot for c in chunks],device)
    states = [s.memory for s in snaps]
    for i,chunk in enumerate(chunks):
        if chunk.rows['first'][0]:
            states[i]=model.block_memory.empty(device=device,episode=states[i].episode)
    selected = [historical_blocks(s,generator,config=model.block_memory.config) if train_history else () for s in states]
    unique = {}
    source_keys={}
    for blocks in selected:
        for block in blocks:
            if block.source is None:
                raise ValueError("historical gradient requires retained raw observations")
            key=block.source.encoding_key if deduplicate_sources else id(block.source)
            source_keys[id(block.source)]=key
            unique[key] = block.source
    def encode_sources(sources,pending):
        # Teacher microbatches can reference 64 historical rows per example.
        # Bound Spatial activations by a GPU-sized microbatch, using
        # PyTorch's non-reentrant checkpoint rather than retaining thousands of
        # historical Spatial activations until the merged loss backward.
        from torch.utils.checkpoint import checkpoint
        encoded=[]
        total=sum(map(len,sources));source_index=source_offset=0
        for begin in range(0,total,spatial_micro):
            left=min(spatial_micro,total-begin);parts=[]
            while left:
                source=sources[source_index]
                take=min(left,len(source)-source_offset)
                if source_offset==0 and take==len(source):parts.append(source)
                else:
                    array=source.compact_array() if isinstance(source,PackedRows) else source
                    parts.append(array[source_offset:source_offset+take])
                source_offset+=take;left-=take
                if source_offset==len(source):source_index+=1;source_offset=0
            batch=from_rows(source_rows(parts),device,pending[begin:begin+spatial_micro])
            encoded.append(checkpoint(model.encode_block,batch,use_reentrant=False)
                if torch.is_grad_enabled() and total>spatial_micro else model.encode_block(batch))
        return torch.cat(encoded)
    encoded_blocks = {}
    if unique:
        keys = list(unique)
        sources = [unique[k].rows for k in keys]
        pending = np.concatenate([unique[k].pending for k in keys])
        z = encode_sources(sources,pending)
        cfg = model.block_memory.config
        written = model.block_memory.writer(z.reshape(len(keys), cfg.block_steps, cfg.latents, cfg.latent_width))
        encoded_blocks = dict(zip(keys, written.unbind()))
    replacements = [{b.end_step: encoded_blocks[source_keys[id(b.source)]] for b in blocks} for blocks in selected]
    # A block ending on the final query is still causally invisible. Loss-only
    # single-step teachers therefore need no tail re-encoding. Preserve the two
    # selected old blocks and use the detached tail for the invisible write.
    tails={id(s.history.tail):s.history.tail for s,state in zip(snaps,states)
           if s.history.tail and (require_next_state or
                                  length+len(state.partial)>model.block_memory.config.block_steps)}
    encoded_tails={}
    if tails:
        tail_keys=list(tails)
        sources=[item[0] for key in tail_keys for item in tails[key]]
        pending=np.concatenate([item[1] for key in tail_keys for item in tails[key]])
        tail_z=encode_sources(sources,pending)
        offset=0
        for key in tail_keys:
            length_tail=len(tails[key])
            encoded_tails[key]=tail_z[offset:offset+length_tail]
            offset+=length_tail
    prefixes = []
    for snap in snaps:
        if id(snap.history.tail) in encoded_tails:
            prefixes.append(encoded_tails[id(snap.history.tail)])
        else:
            prefixes.append(snap.memory.partial.detach())
    rows = source_rows([c.framed() for c in chunks])
    pending = np.concatenate([c.pending for c in chunks])
    batch = from_rows(rows, device, pending)
    ctx = dict(states=states, length=length, replacements=replacements,
               source_prefixes=prefixes, version=int(version),reset_done=True,
               require_next_state=require_next_state,
               recompute_stats=dict(selected_blocks=sum(map(len,selected)),unique_objects=len(source_keys),
                   encoded_blocks=len(unique),historical_rows=sum(len(s.rows) for s in unique.values()),
                   tail_rows=sum(map(len,tails.values()))))
    outputs = model(batch, need_vg=True, temporal=ctx)
    return outputs, batch, ctx, selected


def ppo_loss(model, outputs, batch, actions, behavior_logp, advantage, returns,
             global_returns=None, clip=0.2, vf_coef=0.5, ent_coef=0.01,
             vf_global_coef=0.5, forced=None, pickup_imitation=0.0,
             proximal_logp=None, max_weight=2.0, sample_weight=None):
    """Use the existing forced-action and decoupled-PPO implementation verbatim."""
    logits, value = outputs[:2]
    lp, entropy = model.log_prob(logits, model.gate(batch), actions)
    forced = torch.zeros_like(lp, dtype=torch.bool) if forced is None else forced.bool()
    pn = None if sample_weight is None else sample_weight.sum()
    terms = ppo_part_terms(logits, lp, entropy, behavior_logp, advantage, value, returns,
                           actions, forced, clip, sample_weight, pn, proximal_logp, max_weight)
    pg, vf, ent, imitation, ratio, kl, clipped = terms
    loss = pg + vf_coef*vf - ent_coef*ent + pickup_imitation*imitation
    if global_returns is not None:
        sq = (outputs[-1]-global_returns).square()
        gv = .5*(sq.mean() if sample_weight is None else (sq*sample_weight).sum()/pn)
        loss = loss + vf_global_coef*gv
    return loss, dict(pg=pg.detach(), vf=vf.detach(), entropy=ent.detach(),
                      kl=kl.detach(), clipped=clipped.detach(), ratio=ratio.detach())


def cf_loss(model, outputs, batch, actions, weights, returns, share, coef, vf):
    """Existing winner-action imitation and return regression, not soft labels."""
    lp, _ = model.log_prob(outputs[0], model.gate(batch), actions)
    return teacher_v2_loss(lp, outputs[1], weights, returns, share, coef, vf)


def injury_loss(logits, danger, meta, share, teach_coef, danger_coef):
    """Existing safe-action probability mass and danger BCE."""
    hurt = meta[:, 3:12].float()
    danger_term = F.binary_cross_entropy_with_logits(danger.float(), hurt)
    logp = F.log_softmax(logits.float()[:, :9], -1)
    safe = meta[:, 14] > 0
    safe_mass = torch.logsumexp(logp.masked_fill(hurt > 0, -1e9), -1)
    imitation = -(safe_mass*safe).sum()/safe.sum().clamp(min=1)
    return share*(teach_coef*imitation + danger_coef*danger_term)


@dataclass(frozen=True)
class TeacherTrajectory:
    """Accepted real or hypothetical continuation; never committed to main actor."""
    snapshot: Snapshot
    rows: np.ndarray
    pending: np.ndarray
    actions: np.ndarray
    terminal: bool = False

    @torch.inference_mode()
    def bootstrap(self, teacher):
        if self.terminal:
            return 0.0
        snap = self.snapshot.to(next(teacher.parameters()).device)
        # Force recorded observations/pending actions. Do not sample a replacement
        # trajectory when reconstructing the taken continuation's final value.
        chunk = Chunk(self.rows, self.pending, snap)
        outputs, _, _, _ = forward_chunks(teacher, [chunk], train_history=False)
        return float(outputs[-1][-1])

