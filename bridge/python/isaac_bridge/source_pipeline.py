"""Versioned source graphs on dedicated devices; no forward replay.

Reader updates consume detached transport leaves, not detached training history.
Cached graphs and immutable forward-weight snapshots survive source updates; new
observations alone use the new snapshot. Later-room cotangents return to their
original graphs. Applying those gradients to the current master is delayed-gradient
optimization, not a claim of current-weight gradient equivalence. CUDA work uses
one worker per source GPU, allowing lookahead encoding alongside the Reader.
"""
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext

import numpy as np
import torch
from torch.nn.utils.stateless import _reparametrize_module


def source_parameter(name):
    return name == 'kind' or name == 'gtok.tok' or name == 'reader.queries' or name.startswith((
        'type_emb.', 'variant_emb.', 'subtype_emb.', 'item_emb.', 'trinket_emb.',
        'pill_emb.', 'effect_emb.', 'card_emb.', 'entity.', 'patch.', 'player.',
        'room.', 'door.', 'map.', 'blocks.', 'norm.', 'char_emb.', 'itok.',
        'room_type_emb.', 'gtok.norm.', 'reader.inp.', 'reader.writer.', 'reader.writer_norm.'))


def _runs(values):
    """Sorted integers as half-open runs [lo, hi)."""
    out = []
    for v in values:
        if out and out[-1][1] == v: out[-1][1] = v + 1
        else: out.append([v, v + 1])
    return out


class ImmutableSourceCast(torch.autograd.Function):
    """One immutable BF16 storage, a separate FP32 gradient edge per forward.

    Reusing a single differentiable .to(BF16) would sum different forwards'
    gradients in BF16 before the cast backward. This alias shares only values;
    each consumer converts its own gradient to the FP32 snapshot first.
    """
    @staticmethod
    def forward(ctx, parameter, value):
        ctx.dtype=parameter.dtype
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.dtype),None


class SourcePipeline:
    def __init__(self, model, devices, *, period=4, micro=2048, amp=False, version=0, saved_gpu_gb=0, saved_cpu_gb=1, saved_8bit=0, compile_source=False, flash_source=False, update_rollouts=1, shared_devices=(), actor_reserve_gb=0, processes=False, defer_backward=False, horizon=0):
        self.model, self.period, self.micro, self.amp = model, period, micro, amp
        # Gradient horizon in decisions (0: unbounded). Reads are unchanged; a
        # decision's loss reaches only sources at most `horizon` decisions older.
        # Graphs no future in-horizon decision can reach become detached values.
        self.horizon=horizon
        self.horizon_dropped=None
        self.horizon_detached=dict(chunks=0,writes=0,frames=0,packets=0)
        self.origins={}   # branch uid -> RawHistory.origin: its ancestors' frames are keyed by them
        self.released_misses=0   # rule 5: requests for frames with neither values nor bytes (an error)
        self.update_rollouts,self.rollouts_since_update=update_rollouts,0
        self.saved_gpu_gb = saved_gpu_gb
        self.saved_cpu_gb = saved_cpu_gb
        self.saved_8bit = saved_8bit
        self.compile_source = compile_source
        self.flash_source = flash_source
        self.processes=processes
        self.ipc_async=False
        self.defer_backward=defer_backward
        self.deferred_packets=None
        if compile_source:
            import threading
            self.compile_lock=threading.RLock()
            torch._dynamo.config.cache_size_limit=max(torch._dynamo.config.cache_size_limit,256)
            from torch._functorch import config as aot_config
            # The source graph is reused by later rooms. AOT's donated-buffer
            # optimization requires retain_graph=False (CUDA regression gate).
            aot_config.donated_buffer=False
        self.version, self.steps, self.fast_steps = version, 0, 0
        self.device = next(model.parameters()).device
        self.devices = [torch.device(d) for d in devices]
        self.shared_devices = {torch.device(d) for d in shared_devices}
        self.actor_reserve_gb = actor_reserve_gb
        self.prefetch_live = {}
        self.prefetch_pressure = False
        self.names = [n for n, _ in model.named_parameters() if source_parameter(n)]
        self.touched=[set() for _ in devices]
        self.component_forwards={}
        self.versions = {version: self._replicas()}
        self.lengths = {}
        self.workers = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'source-{i}')
                        for i in range(len(devices))]
        self.backward_workers=([ThreadPoolExecutor(max_workers=1,thread_name_prefix=f'vjp-{i}')
                               for i in range(len(devices))]
                              if processes and os.environ.get('ISAAC_SOURCE_PREEMPT_VJP')=='1' else self.workers)
        self.frames, self.blocks, self.chunks, self.writes, self.owners = {}, {}, [], [], {}
        self.direct_grads = {}
        self.retired_pending = []
        self.collected = {}
        self.pending = None
        self.events = []
        self.reader_events = []
        self.reader_while_backward = 0
        self.reader_while_retired = 0
        self.control_futures=[]
        self.remote_memory=[dict(allocated=0,reserved=0) for _ in devices]
        if processes:
            from .source_worker import SourceProcess
            options=dict(period=period,micro=micro,amp=amp,version=version,saved_gpu_gb=saved_gpu_gb,saved_cpu_gb=saved_cpu_gb,saved_8bit=saved_8bit,
                         compile_source=compile_source,flash_source=flash_source,update_rollouts=update_rollouts)
            spill_limits=([min(saved_gpu_gb,max(1.,b-1.)) for b in self._memory_budgets()]
                          if saved_gpu_gb>0 else [0.]*len(self.devices))
            self.remotes=[SourceProcess(self.versions[version][0],d,
                dict(options,saved_gpu_gb=limit,shared_devices=[d] if d in self.shared_devices else []))
                for d,limit in zip(self.devices,spill_limits)]
        self.encoded_rows = self.reused_rows = self.written_blocks = 0
        self.profile_path = os.environ.get('ISAAC_SOURCE_WORKER_PROFILE')
        if self.profile_path:
            import cProfile, gc, threading
            self.profilers = [cProfile.Profile() for _ in devices]
            started = {}
            def gc_timing(phase, info):
                generation = info['generation']
                if phase == 'start': started[generation] = time.perf_counter()
                else:
                    elapsed = time.perf_counter() - started[generation]
                    if elapsed > .05:
                        print(f'SOURCE_GC generation={generation} seconds={elapsed:.6f} '
                              f'thread={threading.current_thread().name}', flush=True)
            gc.callbacks.append(gc_timing)

    def _replicas(self):
        # A new version can first be needed by value recomputation, whose caller
        # runs in inference_mode. Its source snapshot must still be trainable.
        with torch.inference_mode(False):
            # Source forwards never execute Reader attention or policy heads.
            # Keep their module shapes for the existing encoder API, without
            # cloning unused weights onto every card for every retained version.
            replicas=[]
            for device in ([torch.device('cpu')] if self.processes else self.devices):
                memo={id(p):torch.nn.Parameter(torch.empty_like(p,device='meta'),requires_grad=False)
                      for n,p in self.model.named_parameters() if not source_parameter(n)}
                replica=copy.deepcopy(self.model,memo)
                replica._apply(lambda t:t if t.is_meta else t.to(device))
                replicas.append(replica)
        for worker,replica in enumerate(replicas):
            for name, param in replica.named_parameters():
                param.requires_grad_(source_parameter(name))
                if param.requires_grad and not self.processes:
                    # PyTorch 2.7 AOT + saved-tensor hooks can bypass a leaf's
                    # Tensor hook while still accumulating its gradient (LN
                    # gamma regression). Observe the actual accumulation.
                    param.register_post_accumulate_grad_hook(lambda param,n=name,used=self.touched[worker]:used.add(n))
            replica.train()
            if self.amp and not self.processes:
                # Autocast otherwise saves another copy of the same immutable
                # Linear/Conv weights for every tiny episode encoder invocation.
                params=dict(replica.named_parameters())
                names=[f'{module_name}.{name}' if module_name else name
                       for module_name,module in replica.named_modules()
                       if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d))
                       for name,p in module.named_parameters(recurse=False)
                       if p is not None]
                with torch.inference_mode(False),torch.no_grad():
                    replica._source_cast_values={n:params[n].to(torch.bfloat16)
                        for n in names if n.startswith(('patch.','room.','map.','blocks.',
                                                       'reader.inp.','reader.writer.'))
                        and params[n].dtype==torch.float32}
            if self.compile_source and not self.processes:
                if os.environ.get('ISAAC_SOURCE_POINTWISE_RECOMPUTE')=='1':
                    for block in replica.blocks:
                        block.source_pointwise_recompute=True
                    if os.environ.get('ISAAC_SOURCE_POINTWISE_COMPONENTS')=='1':
                        from .source_rematerialize import compile_source_component
                        for name in ('entity','door','patch','room','map'):
                            component=getattr(replica,name)
                            if name not in self.component_forwards:
                                template=copy.deepcopy(getattr(self.model,name)).to('meta')
                                self.component_forwards[name]=compile_source_component(template)
                            def component_forward(x,fn=self.component_forwards[name],module=component,lock=self.compile_lock):
                                if os.environ.get('ISAAC_SOURCE_COMPONENT_STATIC')=='1':
                                    # Protocol channels/spatial axes never vary;
                                    # only the number of changed observations does.
                                    torch._dynamo.mark_static(x,list(range(1,x.ndim)))
                                with lock:return fn(x,tuple(module.parameters()))
                            component.forward=component_forward
                # Fuse the tensor-only Spatial stack; raw-record planning and
                # component identity handling stay outside Dynamo. Optional
                # selective GELU rematerialization never replays encoder GEMMs.
                method=('encode_rebuild' if self.compile_source==2 else
                        '_blocks_flash' if self.flash_source else '_blocks_bucketed')
                compiled=torch.compile(getattr(replica,method),dynamic=True)
                lock=self.compile_lock
                def source_blocks(*args,fn=compiled):
                    # Dynamo/FX tracing is process-global in this runtime. Only
                    # the host dispatch is serialized; CUDA kernels stay async.
                    with lock:return fn(*args)
                setattr(replica,method,source_blocks)
        if self.processes:
            params=[p for n,p in replicas[0].named_parameters() if source_parameter(n)]
            with torch.inference_mode(False):
                flat=torch.cat([p.detach().reshape(-1) for p in params]).share_memory_()
                for param,value in zip(params,flat.split([p.numel() for p in params])):
                    param.data=value.view_as(param)
            replicas[0]._source_snapshot_flat=flat
            return replicas*len(self.devices)
        return replicas

    def _remote(self, worker, command, payload, version=None):
        remote=self.remotes[worker]
        state=None
        if version is not None and version not in remote.installed:
            state=self.versions[version][0]._source_snapshot_flat
        result,events,memory=remote.request(command,payload,version,state)
        self.events.extend(events);self.remote_memory[worker]=memory
        return result

    @property
    def replicas(self):
        if self.version not in self.versions:
            self.versions[self.version] = self._replicas()
        return self.versions[self.version]

    def retain_histories(self, uids, *, overlap=False, frontiers=None):
        """Retire episodes no longer in the rollout/teacher working set, not
        representations merely because an optimizer version has advanced.

        frontiers: {uid: earliest decision index any future loss may use}. With
        a horizon, live graphs no such decision can reach become detached values."""
        if self.pending is not None: return
        uids = set(uids).union(*self.prefetch_live.values())
        uids.update(h.uid for batch in self.collected.values() for h,_,_ in batch.requests)
        # A live branch keeps the ancestor frames it reads (they are keyed by the ancestor).
        uids.update(a for uid in list(uids) for a, _ in self.origins.get(uid, ()))
        live_chunks = {c for k, (c, _) in self.frames.items() if k[0] in uids}
        live_writes = {c for k, (c, _) in self.blocks.items() if k[0][0] in uids}
        if self.steps:
            # Closed episodes will never receive another consumer. Settle their
            # complete credit before releasing graphs, but do NOT step Adam.
            # Live histories still use one VJP at the slow interval boundary.
            # A stateless batch survives until all its constituent UIDs retire.
            retired = [c for i,c in enumerate(self.chunks) if c is not None and i not in live_chunks]
            retired += [c for i,c in enumerate(self.writes) if c is not None and i not in live_writes]
            packets = []
            for chunk in retired:
                result = chunk['future'].result()
                if result['grads']:
                    packets.append((chunk, result['grads'])); result['grads'] = {}
            if packets:
                futures = [w.submit(self._backward,i,packets,False) for i,w in enumerate(self.workers)]
                self.retired_pending.extend(futures)
                # Process workers own these graphs and enqueue retire after
                # VJP. Reader can run now; Adam must still collect all credit.
                self._collect_retired(wait=not (overlap and self.processes))
        self.frames = {k: v for k, v in self.frames.items() if k[0] in uids}
        self.blocks = {k: v for k, v in self.blocks.items() if k[0][0] in uids}
        self.lengths = {k: v for k, v in self.lengths.items() if k in uids}
        self.origins = {k: v for k, v in self.origins.items() if k in uids}
        self.owners = {k: v for k, v in self.owners.items() if k[0] in uids}
        self.chunks = [c if i in live_chunks else None for i, c in enumerate(self.chunks)]
        self.writes = [c if i in live_writes else None for i, c in enumerate(self.writes)]
        if self.horizon and frontiers:
            self._detach_horizon(frontiers, overlap)
        elif frontiers:
            # Without a horizon, an episode with no future loss (frontier inf: a consumed
            # record, an in-flight branch's ancestor, an ended episode in its grace period)
            # needs values only (rule 2, rule 5 D2a); its graphs would hold VRAM unused.
            done = {uid: f for uid, f in frontiers.items() if f == float('inf')}
            if done: self._detach_horizon(done, overlap)
        # A detached value no longer needs its immutable weight snapshot.
        versions = {c['version'] for c in self.chunks + self.writes if c is not None and not c.get('detached')}
        self.versions = {v: r for v, r in self.versions.items() if v in versions or v == self.version}
        if not self.processes:
            for replicas in self.versions.values():
                for replica in replicas:
                    if hasattr(replica,'_components'):replica._components.retain_source_uids(uids)
        if self.processes:
            for worker,executor in enumerate(self.workers):
                keep=([i for i,c in enumerate(self.chunks) if c is not None and c['worker']==worker],
                      [i for i,c in enumerate(self.writes) if c is not None and c['worker']==worker],list(self.versions))
                self.control_futures.append(executor.submit(self._remote,worker,'retire',keep))

    def _detach_horizon(self, frontiers, overlap):
        """Turn graphs outside every future decision's horizon into values.

        A frame s is reachable by decision t iff t-s <= horizon; a Writer block
        starting at frame b iff t-b <= horizon (all its frames are then inside).
        Future decisions of an episode are >= frontiers[uid], so a chunk whose
        last frame (a write whose last block start) is < frontier-horizon can
        never receive another cotangent. Credit already accumulated by earlier
        in-horizon decisions is settled first (same Adam interval, like retired
        episodes); retain_graph=True because later graphs of the same episode
        may share its component nodes. The values stay readable."""
        limit = {uid: f - self.horizon for uid, f in frontiers.items()}
        def old(c, field):
            if c is None or c.get('detached') or not c['future'].done(): return False
            if c.get('uid') is None and c.get('uids'):   # stateless teacher frames of several records
                return all(u in limit and c[field] < limit[u] for u in c['uids'])
            return c.get('uid') in limit and c[field] < limit[c['uid']]
        writes = [i for i, c in enumerate(self.writes) if old(c, 'last_start')]
        # A Writer graph still inside the horizon also holds the encoder graphs
        # of all its frames. Such chunks stay graphs until that write detaches.
        leaving = set(writes)
        pinned = set().union(*(c['chunks'] for i, c in enumerate(self.writes)
                               if c is not None and not c.get('detached') and i not in leaving))
        chunks = [i for i, c in enumerate(self.chunks) if old(c, 'last') and i not in pinned]
        self.pinned_rows = sum(self.chunks[i]['rows'] for i in pinned if self.chunks[i] is not None and old(self.chunks[i], 'last'))
        if not chunks and not writes: return
        packets = []
        for c in [self.chunks[i] for i in chunks] + [self.writes[i] for i in writes]:
            result = c['future'].result()
            if result['grads']:
                packets.append((c, result['grads'])); result['grads'] = {}
            c['detached'] = result['_detached'] = True
        if packets:
            futures = [w.submit(self._backward, i, packets, True, 'horizon_backward') for i, w in enumerate(self.workers)]
            self.retired_pending.extend(futures)
            self._collect_retired(wait=not (overlap and self.processes))
        for worker, executor in enumerate(self.workers):
            mine = ([i for i in chunks if self.chunks[i]['worker'] == worker],
                    [i for i in writes if self.writes[i]['worker'] == worker])
            if mine[0] or mine[1]:
                self.control_futures.append(executor.submit(self._detach, worker, *mine))
        self.horizon_detached['chunks'] += len(chunks); self.horizon_detached['writes'] += len(writes)
        self.horizon_detached['frames'] += sum(self.chunks[i]['rows'] for i in chunks)
        self.horizon_detached['packets'] += len(packets)

    def _detach(self, worker, chunks, writes):
        """Drop graph references after the worker's FIFO VJP; keep values."""
        if self.processes:
            return self._remote(worker, 'detach', (chunks, writes))
        for i in chunks:
            result = self.chunks[i]['future'].result()
            # A consumed record's own decision frame is still read (frozen): keep y values.
            result['u'] = result['u'].detach(); result['y'] = result['y'].detach()
        for i in writes:
            result = self.writes[i]['future'].result()
            result['mem'] = result['mem'].detach()

    def value_bytes_per_frame(self):
        """Mean source value bytes (u and Spatial tokens y) per encoded frame, over
        the newest completed chunks that still hold both (record byte budget)."""
        total = rows = 0
        for c in reversed(self.chunks):
            if c is None or c.get('detached') or not c['future'].done(): continue
            r = c['future'].result()
            if r.get('y') is None or not torch.is_tensor(r['y']): continue
            total += r['y'].numel() * r['y'].element_size() + r['u'].numel() * r['u'].element_size(); rows += c['rows']
            if rows >= 4096: break
        return total / rows if rows else 0.

    def horizon_stats(self):
        """Learner-side accounting of retained source graphs (progress.csv)."""
        per_uid = {}
        graph = detached = 0
        for c in self.chunks:
            if c is None: continue
            if c.get('detached'): detached += c['rows']; continue
            graph += c['rows']
            if c.get('uid') is not None: per_uid[c['uid']] = per_uid.get(c['uid'], 0) + c['rows']
        dropped = 0. if self.horizon_dropped is None else float(self.horizon_dropped)
        allocated = [m['allocated'] for m in self.remote_memory] if self.processes else [
            torch.cuda.memory_allocated(d) if d.type == 'cuda' else 0 for d in self.devices]
        return dict(source_horizon=self.horizon, source_graph_frames=graph,
                    source_graph_frames_uid_max=max(per_uid.values(), default=0),
                    source_graph_writes=sum(c is not None and not c.get('detached') for c in self.writes),
                    source_pinned_rows=getattr(self, 'pinned_rows', 0),
                    source_detached_frames=detached,
                    source_detached_total=self.horizon_detached['frames'],
                    source_horizon_dropped_grad=dropped,
                    source_alloc_gib_max=max(allocated, default=0) / 2**30,
                    **({'source_worker_graph_frames': sum(m.get('graph_frames', 0) for m in self.remote_memory)}
                       if self.processes else {}))

    def _collect_retired(self, *, wait):
        pending=[]
        for future in self.retired_pending:
            if not wait and not self._backward_ready(future):
                pending.append(future)
                continue
            started=time.perf_counter()
            result=future.result()
            if wait and '_ready' in result:result['_ready'].synchronize()
            ready=time.perf_counter()
            for name,grad in self._packed_gradients(result):
                if name not in self.direct_grads:self.direct_grads[name]=grad
                else:self.direct_grads[name].add_(grad)
            self.events.extend([
                dict(kind='retired_credit_wait',device='host',start=started,end=ready),
                dict(kind='retired_credit_transfer',device='host',start=ready,end=time.perf_counter())])
        self.retired_pending=pending

    def check_version(self, version):
        if version != self.version:
            raise RuntimeError(f'source version {version} consumed after switch to {self.version}')

    def _context(self, worker):
        device = self.devices[worker]
        return torch.cuda.device(device) if device.type == 'cuda' else nullcontext()

    def _cast_context(self, replica):
        if not self.amp:return nullcontext()
        params=dict(replica.named_parameters())
        return _reparametrize_module(replica,{n:ImmutableSourceCast.apply(params[n],value)
            for n,value in replica._source_cast_values.items()})

    def _encode(self, worker, parts, version, cid=None, keys=None):
        if self.processes:
            result=self._remote(worker,'encode',(cid,parts,keys),version)
            result['grads']={}
            return result
        from .tok_policy import to_batch, learner_plan, plan_tokens, item_count
        started = time.perf_counter()
        if self.profile_path and version == 3: self.profilers[worker].enable()
        # Assemble only this encoder job on its worker. A rollout-wide ROW copy
        # formerly blocked the learner before any GPU work could be submitted.
        rows=np.concatenate([r.view(np.uint8).reshape(len(r),-1) for r,p in parts]).view(parts[0][0].dtype).reshape(-1)
        pending=np.concatenate([p for r,p in parts])
        replica, device = self.versions[version][worker], self.devices[worker]
        with self._context(worker), torch.enable_grad(), self._cast_context(replica), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=self.amp):
            b = to_batch(rows, pending, device, entities=64)
            b['_raw_rows'], b['_raw_pending'], b['_source_packed'] = rows, pending, True
            if os.environ.get('ISAAC_SOURCE_COMPONENT_GRAPH_CACHE')=='1' and keys is not None:
                uids={key[0] for key in keys}
                if len(uids)==1:b['_source_component_uid']=next(iter(uids))
            b['_source_bf16'] = self.amp
            b['_source_flash'] = self.flash_source
            # Reuse the existing token-length buckets instead of restoring every
            # sparse source row to the maximum token width in every Spatial layer.
            ni, g = replica.extra_tokens()
            plan = learner_plan(rows['n_ent'], rows['n_doors'], 64,
                                n_item=item_count(rows, replica.itok.slots) if ni else None,
                                item_tok=ni, glob=g)
            # Counts already live on the host. Reuse the same integer planner
            # there instead of launching dozens of tiny GPU indexing kernels
            # for every episode, then transfer its indices in one packed copy.
            tok=plan_tokens(plan,torch.from_numpy(plan['rec']),torch.from_numpy(plan['base']))
            if self.flash_source:
                del tok['keep'],tok['gpos']
            sizes=[t.numel() for t in tok.values()]
            indices=torch.cat([t.reshape(-1) for t in tok.values()]).to(device,non_blocking=True)
            b.update((name,value.view_as(tok[name])) for name,value in zip(tok,indices.split(sizes)))
            if self.flash_source:
                b['cu_seqlens']=torch.as_tensor(np.r_[0,np.cumsum(plan['rec'][:,0])],dtype=torch.int32,device=device)
                b['max_seqlen']=int(plan['rec'][:,0].max())
            b['_source_compact'] = True
            if not self.flash_source:
                b['buckets'] = [(torch.as_tensor(counts, device=device), width)
                                for counts, width in plan['buckets']]
            if self.compile_source==2:
                # Exact raw-key dedup stays outside Dynamo; the already grouped
                # components remain differentiable inputs to the compiled graph.
                components={}
                components['player']=replica._component('player',b,replica._player_token)
                components['grid']=replica._cnn(replica.room,b,'grid')
                components['map']=replica._cnn(replica.map,b,'map')
                if ni:components['items']=replica._component('items',b,lambda part:replica.itok(part,compact=True))
                b['_encoded_components']=components
                del b['_raw_rows'],b['_raw_pending']
            profiling = bool(os.environ.get('ISAAC_SOURCE_SAVED_PROFILE')) and version == 0 and not self.saved_8bit
            saved = {}
            def packed(t):
                storage = t.untyped_storage()
                saved.setdefault(storage.data_ptr(), (storage.nbytes(), str(t.dtype), tuple(t.shape),
                                                       type(t.grad_fn).__name__))
                return t.detach()
            spill = (self.saved_gpu_gb > 0 and device.type == 'cuda' and
                     torch.cuda.memory_allocated(device) >= self.saved_gpu_gb * 2**30)
            # Preserve the graph and exact saved values, not a replay recipe.
            # Pinned saved tensors return asynchronously only for backward.
            saving = nullcontext()
            cpu_bytes = 0
            limited_bytes = 0
            if self.saved_8bit:
                from .saved_activation import SavedActivation8bit
                saving = SavedActivation8bit(replica, self.saved_8bit)
            elif spill:
                cpu_store = torch.autograd.graph.save_on_cpu(pin_memory=True)
                # Immutable snapshot weights already remain on this GPU. CPU
                # copies save no VRAM and formerly multiplied host allocation.
                weights={p.untyped_storage().data_ptr() for p in replica.parameters() if not p.is_meta}
                retained=sum(c['future'].result().get('saved_cpu_bytes',0) for c in self.chunks
                             if c is not None and c['worker']==worker and c['future'].done())
                def pack_cpu(t):
                    nonlocal cpu_bytes, limited_bytes
                    if t.device.type!='cuda' or t.untyped_storage().data_ptr() in weights:
                        return None, None, t.detach()
                    # save_on_cpu materializes broadcast attention biases. That
                    # changes strideH to an unaligned width squared (CUDA test:
                    # 29*29=841). Store the unexpanded values and restore the
                    # original zero strides, rather than copying the broadcast.
                    small = t[tuple(slice(0,1) if s == 0 else slice(None) for s in t.stride())]
                    size=small.numel()*small.element_size()
                    if retained+cpu_bytes+size>self.saved_cpu_gb*2**30:
                        limited_bytes+=size
                        return None, None, t.detach()
                    cpu_bytes+=size
                    return t.shape, small.stride(), cpu_store.pack_hook(small)
                def unpack_cpu(record):
                    shape, stride, value = record
                    if shape is None:return value
                    restored = cpu_store.unpack_hook(value)
                    if restored.stride() != stride:
                        # CUTLASS also requires the original aligned batch
                        # stride (e.g. padded 32, not compact width 29).
                        aligned = torch.empty_strided(restored.shape, stride,
                                                      device=restored.device, dtype=restored.dtype)
                        aligned.copy_(restored); restored = aligned
                    return restored.expand(shape)
                saving = torch.autograd.graph.saved_tensors_hooks(pack_cpu, unpack_cpu)
            with saving, torch.autograd.graph.saved_tensors_hooks(packed, lambda t:t) if profiling else nullcontext():
                y, pad, u = replica.encode_rebuild(b)
            if profiling:
                import json
                print('SOURCE_SAVED '+json.dumps(dict(device=str(device), rows=len(rows),
                      bytes=sum(v[0] for v in saved.values()),
                      largest=sorted(saved.values(),reverse=True)[:12])),flush=True)
            if device.type == 'cuda' and not self.ipc_async: torch.cuda.current_stream(device).synchronize()
        self.events.append(dict(kind='encode', device=str(device), version=version,
                                rows=len(rows), saved_on_cpu=bool(cpu_bytes), saved_cpu_bytes=cpu_bytes,
                                spill_limited_bytes=limited_bytes, start=started, end=time.perf_counter(),
                                **(saving.stats() if self.saved_8bit else {})))
        if self.profile_path and version == 3:
            self.profilers[worker].disable()
            self.profilers[worker].dump_stats(f'{self.profile_path}/source-worker-{worker}.pstats')
        return dict(y=y, u=u, starts=plan['rec'][:, 1], counts=plan['rec'][:, 0], grads={},saved_cpu_bytes=cpu_bytes,
                    **(saving.stats() if self.saved_8bit else {}))

    def _memory_budgets(self):
        budgets=[]
        for i,device in enumerate(self.devices):
            if device.type != 'cuda':
                budgets.append(1.)
                continue
            free,total = torch.cuda.mem_get_info(device)
            resident=sum(m['reserved'] for d,m in zip(self.devices,self.remote_memory) if d==device)
            capacity = (free + torch.cuda.memory_reserved(device)+resident)/2**30 - 2
            if device in self.shared_devices:
                # The Actor's *future* graph pool is not visible in mem_get_info.
                # Reserve its measured peak plus headroom before assigning new
                # immutable blocks; current idle Actor usage is not its budget.
                capacity = min(capacity, total/2**30 - self.actor_reserve_gb - 2)
            budgets.append(max(1.,capacity/self.devices.count(device)))
        return budgets

    def schedule(self, keys, parts, stateless=(), *, speculative=False, owner_keys=None):
        from .tok_policy import item_count
        missing = {}
        seen = set()
        for i, key in enumerate(keys):
            if key in self.frames or key in seen:
                self.reused_rows += 1
                continue
            seen.add(key)
            # A branch's own frames in a block it shares with its ancestor go
            # to that block's worker: the Writer reads all its frames locally.
            owner_key = owner_keys[i] if owner_keys is not None else (key[0],key[1]//self.model.reader.block)
            missing.setdefault(owner_key, []).append(i)
        ni,g=self.model.extra_tokens()
        costs=np.concatenate([np.minimum(r['n_ent'].astype(np.int64),64)+r['n_doors']+3+g for r,p in parts])
        if ni:
            costs+=item_count({k:np.concatenate([r[k] for r,p in parts]) for k in ('inv','pitem')},self.model.itok.slots)
        ends=np.cumsum([len(r) for r,p in parts]);starts=np.r_[0,ends[:-1]]
        loads=[0]*len(self.devices)
        # Actor-sharing cards have less usable VRAM than dedicated source cards.
        # Assign new immutable Writer blocks, not whole growing episodes. A
        # partially encoded block keeps its card; all its future rows and Writer
        # graph therefore stay local, including across source-version updates.
        budgets=self._memory_budgets()
        completed=[0]*len(self.devices)
        for chunk in self.chunks:
            if chunk is not None:
                loads[chunk['worker']]+=chunk['cost']
                if chunk['future'].done():completed[chunk['worker']]+=chunk['cost']
        # Calibrate token cost against actual retained allocation. Dense entity
        # embeddings and saved activations need not scale with real-token count.
        allocated=[(torch.cuda.memory_allocated(d)/self.devices.count(d)+self.remote_memory[i]['allocated']+
                    self.remote_memory[i].get('saved_cpu_bytes',0))/2**30
                   if d.type=='cuda' else 0. for i,d in enumerate(self.devices)]
        measured=[max(allocated[i],1.)/completed[i]
                  for i,d in enumerate(self.devices) if d.type=='cuda' and completed[i]]
        estimate=max(measured) if measured else 1.
        # Queued work has a token cost but no allocation yet. Including it in
        # the calibration denominator made a busy card appear artificially cheap
        # and admitted more work onto the same already-promised VRAM.
        scales=np.asarray([max(allocated[i],1.)/completed[i]
                           if d.type=='cuda' and completed[i] else estimate for i,d in enumerate(self.devices)])
        for key,indices in missing.items():
            if key in self.owners:loads[self.owners[key]]+=int(costs[indices].sum())
        unassigned={}
        for key,indices in missing.items():
            if key not in self.owners:unassigned.setdefault(key[0],[]).append((key,indices))
        # Balance whole encoder-sized runs of new blocks. Round-robin every
        # block fragments a rollout into hundreds of tiny encoder invocations.
        blocks_per_batch=max(1,self.micro//self.model.reader.block)
        assigned={}
        for block_parts in unassigned.values():
            for lo in range(0,len(block_parts),blocks_per_batch):
                group=block_parts[lo:lo+blocks_per_batch]
                incoming=sum(int(costs[indices].sum()) for key,indices in group)
                worker=int(np.argmin((np.asarray(loads)+incoming)*scales/budgets))
                for key,indices in group:
                    assigned[key]=worker
                    loads[worker]+=int(costs[indices].sum())
        # Lookahead is optional; required Reader inputs are not. Do not enqueue
        # a future rollout beyond the measured live-graph capacity and then
        # discover the pressure inside a CUDA allocation. A rejected prefix is
        # still present in the Actor's normal full hand-off, not discarded.
        if speculative and any(completed) and np.any(np.asarray(loads)*scales>budgets):
            self.prefetch_pressure=True
            return False
        self.owners.update(assigned)
        # Real episode graphs keep independent lifetimes. Ordinary teacher
        # records have no temporal chain: group their single frames by device,
        # rather than launching an entire encoder once per teacher sample.
        grouped={}; single={}
        for key,indices in missing.items():
            worker=self.owners[key]
            for i in indices:
                uid=keys[i][0]   # the frame's owner, not the block's: lifetimes follow frames
                if uid in stateless: single.setdefault(worker,[]).append(i)
                else: grouped.setdefault((uid,worker),[]).append(i)
        groups=[(uid,worker,indices) for (uid,worker),indices in grouped.items()]
        groups.extend((None,worker,indices) for worker,indices in single.items())
        self.replicas
        for uid,worker,indices in groups:
            for lo in range(0, len(indices), self.micro):
                ix = indices[lo:lo+self.micro]
                part_ids=np.searchsorted(ends,ix,side='right')
                cuts=np.r_[0,np.flatnonzero(part_ids[1:]!=part_ids[:-1])+1,len(ix)]
                selected=[]
                for begin,end in zip(cuts[:-1],cuts[1:]):
                    part=int(part_ids[begin]);r,p=parts[part]
                    local=np.asarray(ix[begin:end])-starts[part]
                    if local[-1]-local[0]+1==len(local):
                        take=slice(local[0],local[-1]+1);selected.append((r[take],p[take]))
                    else:
                        selected.append((r.view(np.uint8).reshape(len(r),-1)[local].view(r.dtype).reshape(-1),p[local]))
                cid = len(self.chunks)
                future = self.workers[worker].submit(self._encode, worker, selected, self.version,cid,[keys[i] for i in ix])
                chunk = dict(worker=worker, future=future, version=self.version, uid=uid,
                             cost=int(costs[ix].sum()),remote_kind='encode',remote_id=cid,
                             rows=len(ix),last=max(keys[i][1] for i in ix))
                if uid is None: chunk['uids'] = frozenset(keys[i][0] for i in ix)
                self.chunks.append(chunk)
                for offset, i in enumerate(ix): self.frames[keys[i]] = (cid, offset)
                self.encoded_rows += len(ix)
        return True

    def prefetch(self, requests, stateless=(), *, write_ahead=False, speculative=False):
        latest = {}
        for h, row, pending in requests:
            if h.uid not in latest or h.length > latest[h.uid][0].length:
                latest[h.uid] = h, row, pending
        rr, pp, keys, owners = [], [], [], []
        block = self.model.reader.block
        for uid, (h, row, pending) in latest.items():
            if h.origin: self.origins[uid] = h.origin
            start = self.lengths.get(uid, 0)
            end = h.length + 1
            self.reused_rows += min(start, end)
            if end <= start: continue
            # Only frames without a source entry need observation bytes; an
            # ancestor's frames (a branch's prefix) are read from its values.
            needed = [t for t in range(start, end - 1) if h.key(t) not in self.frames]
            self.reused_rows += (end - 1 - start) - len(needed)
            for lo, hi in _runs(needed):
                at = 0
                for rows, pend in h.blocks + h.tail:
                    stop = at + len(rows)
                    if stop > lo and at < hi:
                        a, b = max(lo, at) - at, min(hi, stop) - at
                        try:
                            rr.append(rows[a:b]); pp.append(pend[a:b])
                        except RuntimeError as error:   # rule 5: a released frame has no values either
                            self.released_misses += 1
                            raise RuntimeError(f'source frames {h.key(lo + 0)}..{hi - 1} of {uid} have neither values '
                                               f'nor bytes ({self.released_misses} such requests)') from error
                    at = stop
            current=np.frombuffer(row.tobytes(),dtype=row.dtype)
            rr.append(current); pp.append(np.array(pending).reshape(1, -1))
            frames = needed + [end - 1]
            keys.extend(h.key(t) for t in frames)
            owners.extend((h.key(t//block*block)[0], t//block) for t in frames)
        if keys:
            if not self.schedule(keys,list(zip(rr,pp)),stateless,speculative=speculative,owner_keys=owners):return False
        self.lengths.update((uid,h.length+1) for uid,(h,_,_) in latest.items()
                            if h.length+1>self.lengths.get(uid,0))
        if write_ahead:
            self._schedule_writes([tuple(h.key(t) for t in range(at,at+block))
                for uid,(h,_,_) in latest.items() for at in range(0,h.length+2-block,block)])
        return True

    def _schedule_writes(self, block_keys):
        new = {}
        for keys in block_keys:
            if keys not in self.blocks:
                uid = keys[0][0]
                worker = self.owners[(uid,keys[0][1]//self.model.reader.block)]
                new.setdefault((uid,worker), []).append(keys)
        for (uid,worker), keys in new.items():
            self.replicas
            wid = len(self.writes)
            future = self.workers[worker].submit(self._write, worker, keys, self.version,wid)
            self.writes.append(dict(worker=worker, future=future, version=self.version, uid=uid,remote_kind='write',remote_id=wid,
                                    last_start=max(block[0][1] for block in keys),
                                    chunks={self.frames[key][0] for block in keys for key in block}))
            for offset,key in enumerate(keys): self.blocks[key] = (wid,offset)
            self.written_blocks += len(keys)

    def _write(self, worker, block_keys, version, wid=None):
        if self.processes:
            result=self._remote(worker,'write',(wid,block_keys),version)
            result['grads']={}
            return result
        started = time.perf_counter()
        replica, device = self.versions[version][worker], self.devices[worker]
        with self._context(worker), torch.enable_grad(), self._cast_context(replica), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=self.amp):
            grouped = {}
            for dst, key in enumerate(key for block in block_keys for key in block):
                cid, offset = self.frames[key]
                grouped.setdefault(cid, []).append((dst, offset))
            frames = None
            for cid, entries in grouped.items():
                value = self.chunks[cid]['future'].result()['u']
                selected = value.index_select(0, torch.as_tensor([i for _, i in entries], device=device))
                if frames is None:
                    frames = selected.new_zeros((len(block_keys)*replica.reader.block, selected.shape[-1]))
                frames = frames.index_copy(0, torch.as_tensor([i for i, _ in entries], device=device), selected)
            saving = nullcontext()
            if self.saved_8bit:
                from .saved_activation import SavedActivation8bit
                saving = SavedActivation8bit(replica, self.saved_8bit)
            with saving:
                mem = replica.reader.write(frames.reshape(len(block_keys), replica.reader.block, -1))
            if device.type == 'cuda' and not self.ipc_async: torch.cuda.current_stream(device).synchronize()
        self.events.append(dict(kind='writer', device=str(device), version=version,
                                rows=len(block_keys), start=started, end=time.perf_counter(),
                                **(saving.stats() if self.saved_8bit else {})))
        return dict(mem=mem, grads={}, **(saving.stats() if self.saved_8bit else {}))

    @torch.no_grad()
    def _gather_plan(self, plan, field, size):
        first=plan[0][2][0][0][field]
        out=first.new_empty((size,)+tuple(first.shape[1:]),device=self.device)
        for device,take,parts in plan:
            # One source->Reader copy per device, not one per episode chunk.
            # The same indices send cotangents back in the opposite direction.
            selected=torch.cat([result[field].index_select(0,put) for result,put in parts])
            out.index_copy_(0,take,selected.to(self.device,non_blocking=True))
        return out

    def _query_tokens(self, refs, width):
        grouped = {}
        for dst, (cid, row) in enumerate(refs): grouped.setdefault(cid, []).append((dst, row))
        for cid, entries in grouped.items():
            result = self.chunks[cid]['future'].result()
            dst, row = np.asarray(entries, np.int64).T
            counts = result['counts'][row]
            index = np.repeat(np.arange(len(row)), counts)
            within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts)-counts, counts)
            source = result['starts'][row[index]] + within
            target = dst[index]*width + np.where(within == counts[index]-1, width-1, within)
            yield result, source, target

    @torch.no_grad()
    def _gather_queries(self, refs):
        grouped={}
        for c,i in refs:grouped.setdefault(c,[]).append(i)
        # Detached query frames (consumed records, read frozen) keep their y values.
        assert all(self.chunks[c]['future'].result().get('y') is not None for c in grouped), 'query frame without y'
        width=max(self.chunks[c]['future'].result()['counts'][indices].max() for c,indices in grouped.items())
        first = self.chunks[refs[0][0]]['future'].result()['y']
        plan=self._credit_plan(refs,self.chunks,'y',width)
        # Padding values are masked before use, but keep them initialized for
        # exact eager/compiled comparisons and numerical diagnostics.
        y=torch.zeros((len(refs)*width,first.shape[-1]),dtype=first.dtype,device=self.device)
        pad = torch.ones(len(refs)*width, dtype=torch.bool, device=self.device)
        for device,target,parts in plan:
            selected=torch.cat([result['y'].index_select(0,source) for result,source in parts])
            y.index_copy_(0,target,selected.to(self.device,non_blocking=True));pad[target]=False
        return y.reshape(len(refs), width, -1), pad.reshape(len(refs), width),plan

    @torch.no_grad()
    def _credit_queries(self, refs, gradient):
        if gradient is None: return
        for result, source, target in self._query_tokens(refs, gradient.shape[1]):
            if result.get('_detached'):   # frozen read: exact zeros, audited
                self._drop(gradient.flatten(0, 1).index_select(0, torch.as_tensor(target, device=gradient.device)))
                continue
            value = result['y']
            if 'y' not in result['grads']: result['grads']['y'] = torch.zeros_like(value, dtype=torch.float32)
            part = gradient.flatten(0, 1).index_select(0, torch.as_tensor(target, device=gradient.device))
            result['grads']['y'].index_add_(0, torch.as_tensor(source, device=value.device),
                                          part.to(value.device, non_blocking=True))

    def materialize(self, batch):
        started = time.perf_counter()
        self._schedule_writes(batch.block_keys)
        refs = [self.frames[key] for key in batch.frame_keys]
        queries = [refs[i] for i in batch.unique_query]
        mrefs = [self.blocks[key] for key in batch.block_keys]
        batch.pipeline_refs = refs, queries, mrefs
        # Keep actual dependency waits separate from index planning and copies.
        # A Future.result inside _credit_plan used to attribute GPU waiting to
        # CPU indexing. These waits add no device synchronization to the path.
        for cid in dict.fromkeys(c for c, _ in refs):
            self.chunks[cid]['future'].result()
        encoded = time.perf_counter()
        for cid in dict.fromkeys(c for c, _ in mrefs):
            self.writes[cid]['future'].result()
        written = time.perf_counter()
        self.events.extend([
            dict(kind='ready_encode_wait', device='host', start=started, end=encoded),
            dict(kind='ready_writer_wait', device='host', start=encoded, end=written)])
        # Only source compute is mixed precision. Reader inputs and accumulated
        # cotangents remain FP32, including repeated reads of the same old source.
        urefs=[refs[i] for i in batch.u_frames]
        uplan=self._credit_plan(urefs,self.chunks,'u')
        mplan=self._credit_plan(mrefs,self.writes,'mem')
        u = self._gather_plan(uplan,'u',len(urefs)).float().requires_grad_()
        y, pad, yplan = self._gather_queries(queries)
        y = y.float().requires_grad_()
        mem = (self._gather_plan(mplan,'mem',len(mrefs)).flatten(0, 1) if mrefs else
               u.new_empty((0, self.model.reader.D))).float().requires_grad_()
        # These mappings are invariant across all Reader minibatches/epochs.
        # Build device indices once, then transfer one packed cotangent per GPU.
        batch.pipeline_plans = (uplan,mplan)
        batch.pipeline_yplan=yplan
        self.events.append(dict(kind='plan_transport', device='host', start=written, end=time.perf_counter()))
        return u, y, pad, mem

    def _credit_plan(self, refs, chunks, field, width=None):
        groups = {}
        if width is None:
            by_chunk = {}
            for src, (cid, offset) in enumerate(refs):
                by_chunk.setdefault(cid, []).append((src, offset))
            entries = ((chunks[cid]['future'].result(), *np.asarray(indices, np.int64).T)
                       for cid, indices in by_chunk.items())
        else:
            entries = ((result, target, source) for result, source, target in self._query_tokens(refs, width))
        for result, take, put in entries:
            device = result[field].device
            groups.setdefault(device, []).append((result, take, put))
        plans = []
        for device, parts in groups.items():
            take = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=self.device)
            put_all = torch.as_tensor(np.concatenate([p[2] for p in parts]), device=device)
            mapped, at = [], 0
            for result, _, put in parts:
                mapped.append((result, put_all[at:at+len(put)])); at += len(put)
            plans.append((device, take, mapped))
        return plans

    @torch.no_grad()
    def _credit_packed(self, plan, field, gradient):
        if gradient is None: return
        for device, take, parts in plan:
            packed = gradient.index_select(0, take).to(device, non_blocking=True)
            at = 0
            for result, put in parts:
                if result.get('_detached'):
                    # Outside every reader's horizon: masked to exact zeros.
                    self._drop(packed[at:at+len(put)]); at += len(put)
                    continue
                if field not in result['grads']:
                    result['grads'][field] = torch.zeros_like(result[field], dtype=torch.float32)
                result['grads'][field].index_add_(0, put, packed[at:at+len(put)])
                at += len(put)

    def _drop(self, gradient):
        # Audit, not a correction: progress.csv must report exactly zero.
        total = gradient.detach().abs().sum().to(self.device, dtype=torch.float64)
        self.horizon_dropped = total if self.horizon_dropped is None else self.horizon_dropped + total

    @torch.no_grad()
    def _credit(self, refs, chunks, field, gradient):
        if gradient is None: return
        grouped = {}
        for src, (cid, offset) in enumerate(refs): grouped.setdefault(cid, []).append((src, offset))
        for cid, entries in grouped.items():
            result = chunks[cid]['future'].result()
            take = torch.as_tensor([i for i, _ in entries], device=gradient.device)
            if result.get('_detached'):
                self._drop(gradient.index_select(0, take)); continue
            value = result[field]
            if field not in result['grads']: result['grads'][field] = torch.zeros_like(value, dtype=torch.float32)
            part = gradient.index_select(0, take).to(value.device, non_blocking=True)
            result['grads'][field].index_add_(0, torch.as_tensor([i for _, i in entries], device=value.device), part)

    def collect(self, batch):
        self.check_version(batch.source_version)
        if self.period == 0:
            # Accumulate consumer cotangents on resident FP32 leaves, and
            # transfer them once per rollout, independently of the Adam clock.
            # Rebuilding query scatter plans for every PPO minibatch is wasteful.
            parent = getattr(batch, 'parent', batch)
            self.collected[id(parent)] = parent
            return
        self._collect_batch(batch)

    def _collect_batch(self, batch):
        refs, queries, mrefs = batch.pipeline_refs
        up, mp = batch.pipeline_plans
        self._credit_packed(up, 'u', batch.u.grad)
        if batch.y.grad is not None:
            qp = batch.pipeline_yplan if self.period == 0 else self._credit_plan(queries, self.chunks, 'y', batch.y.shape[1])
            self._credit_packed(qp, 'y', batch.y.grad.flatten(0, 1))
        grad = batch.mem.grad
        self._credit_packed(mp, 'mem', None if grad is None else
                            grad.reshape(-1, self.model.reader.tokens, self.model.reader.D))

    def fast_step(self, optimizer, max_norm):
        if any(not self._backward_ready(f) for f in self.retired_pending):
            self.reader_while_retired+=1
        if self.pending is not None and not all(self._backward_ready(f) for f in self.pending[0]):
            self.reader_while_backward += 1
        # norm / gtok.norm are shared by source encoding and readout. Their
        # readout gradients also belong to the frozen, slow parameter version.
        for name, param in self.model.named_parameters():
            if source_parameter(name) and param.grad is not None:
                if name not in self.direct_grads: self.direct_grads[name] = param.grad.detach().clone()
                else: self.direct_grads[name].add_(param.grad)
                param.grad = None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        optimizer.step()
        self.steps += 1; self.fast_steps += 1

    def _backward(self, worker, packets, retain_graph=True, kind=None):
        if self.processes:
            selected=[(c['remote_kind'],c['remote_id'],g) for c,g in packets if c['worker']==worker]
            if self.devices[worker].type=='cuda':torch.cuda.current_stream(self.devices[worker]).synchronize()
            return self._remote(worker,'backward',(selected,retain_graph))
        started = time.perf_counter()
        device = self.devices[worker]
        with self._context(worker):
            # Every immutable weight version contributes to the same master
            # gradient. Accumulate directly into one buffer per parameter name,
            # rather than allocating and then adding N version-sized gradients.
            # create_graph=False keeps AccumulateGrad in-place on this stream.
            used=self.touched[worker];used.clear()
            first=next(iter(self.versions.values()))[worker]
            shared={n:torch.zeros_like(p) for n,p in first.named_parameters() if source_parameter(n)}
            for replicas in self.versions.values():
                for name,param in replicas[worker].named_parameters():
                    param.grad=shared[name] if name in shared else None
            groups={}
            for chunk, grads in packets:
                if chunk['worker'] == worker:
                    # Encoder chunks and their Writers have one episode owner.
                    # Backpropagate each independent episode into the same FP32
                    # parameter buffers, instead of keeping all episodes'
                    # transient activation gradients live in one engine call.
                    owner=chunk.get('uid') if self.ipc_async else None
                    outputs,gradients=groups.setdefault(owner,([],[]))
                    result = chunk['future'].result()
                    for name, grad in grads.items():
                        outputs.append(result[name]); gradients.append(grad)
            served=0;service_seconds=0.
            for outputs,gradients in groups.values():
                if outputs:torch.autograd.backward(outputs,gradients,retain_graph=retain_graph)
                if retain_graph and hasattr(self,'service_forward'):
                    service_start=time.perf_counter()
                    served+=self.service_forward()
                    service_seconds+=time.perf_counter()-service_start
            if device.type == 'cuda' and not self.ipc_async: torch.cuda.current_stream(device).synchronize()
            grads={n:shared[n] for n in self.names if n in used}
            for replicas in self.versions.values():replicas[worker].zero_grad(set_to_none=True)
        self.events.append(dict(kind=kind or ('source_backward' if retain_graph else 'retired_backward'), device=str(device), version=self.version,
                                gradient_buffer_bytes=sum(g.numel()*g.element_size() for g in shared.values()),
                                serviced_forward_calls=served,forward_service_seconds=service_seconds,
                                start=started, end=time.perf_counter()))
        return grads

    def start_finish(self, optimizer, max_norm, *, defer=False):
        """Queue one closed cotangent interval while Reader keeps training.

        Dictionaries are swapped, not cleared in place: subsequent reads cannot
        alter the packet being backpropagated. A second interval cannot overtake
        the first. Commits occur only at minibatch boundaries on the learner.
        """
        self.poll_finish(optimizer,max_norm,wait=True)
        self._collect_retired(wait=True)
        if not self.steps: return
        self._flush_collected()
        packets=[]
        for chunk in self.chunks+self.writes:
            # An unfinished speculative forward cannot have consumer credit:
            # materialize() waits for it before exposing any Reader leaf. Do
            # not turn closing the current interval into a future-rollout fence.
            if chunk is not None and chunk['future'].done():
                result=chunk['future'].result()
                if result['grads']:
                    packets.append((chunk,result['grads']));result['grads']={}
        if defer:
            self.deferred_packets=packets
            futures=[]
        else:futures=[w.submit(self._backward,i,packets) for i,w in enumerate(self.backward_workers)]
        self.pending=(futures,self.direct_grads,self.steps)
        self.direct_grads={};self.steps=0;self.rollouts_since_update=0

    def dispatch_finish(self):
        """Start a closed VJP only after the next Reader's inputs are ready.

        Otherwise a FIFO source worker runs historical backward ahead of the
        new observations that Reader needs, merely renaming commit_wait to
        ready_encode_wait. Snapshot weights and the Adam interval are unchanged.
        """
        if self.deferred_packets is None:return
        packets=self.deferred_packets;self.deferred_packets=None
        futures=[w.submit(self._backward,i,packets) for i,w in enumerate(self.backward_workers)]
        self.pending=(futures,self.pending[1],self.pending[2])

    def _flush_collected(self):
        for batch in self.collected.values():
            self._collect_batch(batch)
            batch.u.grad = batch.y.grad = batch.mem.grad = None
        self.collected.clear()

    def finish_rollout(self, optimizer, max_norm, *, overlap=False):
        # Transfer this rollout's complete cotangents now. Keeping its Reader
        # banks until the slow boundary would unnecessarily retain their VRAM.
        self._flush_collected()
        self.rollouts_since_update+=1
        if self.rollouts_since_update==self.update_rollouts:
            self.start_finish(optimizer,max_norm,defer=self.defer_backward and overlap)
            if not overlap:self.poll_finish(optimizer,max_norm,wait=True)

    def poll_finish(self, optimizer, max_norm, wait=False):
        self._collect_retired(wait=False)
        pending_controls=[]
        for future in self.control_futures:
            if future.done():future.result()
            else:pending_controls.append(future)
        self.control_futures=pending_controls
        if self.pending is None:return
        if self.deferred_packets is not None:
            if not wait:return
            self.dispatch_finish()
        futures,direct,steps=self.pending
        if not wait and not all(self._backward_ready(f) for f in futures):return
        started = time.perf_counter()
        results = [future.result() for future in futures]
        if wait:
            for result in results:
                if '_ready' in result:result['_ready'].synchronize()
        ready = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        params = dict(self.model.named_parameters())
        for name, grad in direct.items(): params[name].grad = grad
        for result in results:
            dst,src=[],[]
            for name,value in self._packed_gradients(result):
                if params[name].grad is None:params[name].grad=value
                else:dst.append(params[name].grad);src.append(value)
            if dst:torch._foreach_add_(dst,src)
        transferred = time.perf_counter()
        torch._foreach_div_([params[n].grad for n in self.names if params[n].grad is not None],steps)
        torch.nn.utils.clip_grad_norm_([params[n] for n in self.names], max_norm)
        optimizer.step()
        self.version += 1
        # Never mutate weights referenced by an old autograd graph. Newly
        # observed frames use this new snapshot; old frames keep theirs.
        self.model.invalidate_source_cache()
        self.pending=None
        self.events.extend([
            dict(kind='commit_wait', device='host', start=started, end=ready),
            dict(kind='commit_transfer', device='host', start=ready, end=transferred),
            dict(kind='commit_adam', device='host', start=transferred, end=time.perf_counter())])

    def _packed_gradients(self, result):
        if not result:return
        names=[n for n in result if n!='_ready']
        if not names:return
        sizes=[result[n].numel() for n in names]
        flat=torch.cat([result[n].reshape(-1) for n in names]).to(self.device,non_blocking=True)
        for name,value in zip(names,flat.split(sizes)):
            yield name,value.view_as(result[name])

    @staticmethod
    def _backward_ready(future):
        if not future.done():return False
        result=future.result()
        return '_ready' not in result or result['_ready'].query()

    def finish(self, optimizer, max_norm):
        self.start_finish(optimizer,max_norm)
        self.poll_finish(optimizer,max_norm,wait=True)

    def close(self):
        with ExitStack() as cleanup:
            if self.processes:
                for remote in self.remotes:cleanup.callback(remote.close)
            for worker in self.workers:worker.shutdown(wait=True)
            if self.backward_workers is not self.workers:
                for worker in self.backward_workers:worker.shutdown(wait=True)
            if self.processes:
                self.chunks.clear();self.writes.clear();self.frames.clear();self.blocks.clear();self.collected.clear()
                import gc
                gc.collect()
            for future in self.control_futures:future.result()
