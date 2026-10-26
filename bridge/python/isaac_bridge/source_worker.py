"""Persistent source graphs behind torch.multiprocessing CUDA IPC.

Only detached values and cotangents cross the process boundary. The producer
owns the immutable parameters, saved activations and autograd graph until the
learner retires their identities. No encoder replay or distributed autograd.
"""
import copy
import gc
import os
import traceback
import threading
from collections import deque
from multiprocessing.connection import wait
from concurrent.futures import Future

import torch
import torch.multiprocessing as mp


def ipc_export(values):
    """Export compact ordinary allocations, never an internal activation segment."""
    tensors={k:v for k,v in values.items() if torch.is_tensor(v)}
    if not tensors or next(iter(tensors.values())).device.type!='cuda':
        return {k:v.detach() if torch.is_tensor(v) else v for k,v in values.items()}
    from train_tok import ipc_safe_allocator
    with _EXPORT_LOCK:   # (the allocator setting is process-wide: a concurrent VJP thread's export must not interleave)
        return _ipc_export(values, tensors, ipc_safe_allocator)


_EXPORT_LOCK = threading.Lock()


def _ipc_export(values, tensors, ipc_safe_allocator):
    restore=ipc_safe_allocator()
    reply=dict(values)
    try:
        with torch.no_grad():
            for dtype in {v.dtype for v in tensors.values()}:
                names=[k for k,v in tensors.items() if v.dtype==dtype]
                flat=torch.cat([tensors[k].detach().reshape(-1) for k in names])
                for name,value in zip(names,flat.split([tensors[k].numel() for k in names])):
                    reply[name]=value.view_as(tensors[name])
    finally:
        if restore:torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    return reply


def _tensor_bytes(x, seen):
    """CUDA bytes of the tensors in x (dicts / lists / tuples), each storage once."""
    if torch.is_tensor(x):
        if x.device.type != 'cuda': return 0
        st = x.untyped_storage(); key = st.data_ptr()
        if key in seen: return 0
        seen.add(key); return st.nbytes()
    if isinstance(x, dict): return sum(_tensor_bytes(v, seen) for v in x.values())
    if isinstance(x, (list, tuple)): return sum(_tensor_bytes(v, seen) for v in x)
    return 0


def memory_breakdown(local, exports, device):
    """(ISAAC_SOURCE_MEMORY_BREAKDOWN=1) this card's allocation by kind, at a retire (once per update).

    graph_*: what an encode / Writer call left allocated (outputs + saved activations, measured
    around the call) for chunks / writes that are still graphs; export_*: the compact IPC copies
    the learner reads, by tensor name, split by graph / detached owner; versions: parameter
    snapshots' storages; cache: reserved - allocated."""
    out = dict(graph_chunk_bytes=0, graph_chunk_rows=0, graph_write_bytes=0, detached_chunk_rows=0)
    for c in local.chunks:
        if c is None: continue
        if c.get('detached'): out['detached_chunk_rows'] += c['rows']
        else: out['graph_chunk_bytes'] += c.get('alloc', 0); out['graph_chunk_rows'] += c['rows']
    for c in local.writes:
        if c is not None and not c.get('detached'): out['graph_write_bytes'] += c.get('alloc', 0)
    seen = set()
    for (kind, index), values in exports.items():
        owner = (local.chunks if kind == 'encode' else local.writes)[index] if index < len(
            local.chunks if kind == 'encode' else local.writes) else None
        state = 'detached' if owner is not None and owner.get('detached') else 'graph'
        for name, v in values.items():
            key = f'export_{state}_{kind}_{name}'
            out[key] = out.get(key, 0) + _tensor_bytes(v, seen)
    vs = set()
    out['versions'] = len(local.versions)
    out['version_bytes'] = sum(_tensor_bytes(list(r.parameters()) + list(r.buffers()), vs)
                               for reps in local.versions.values() for r in reps)
    if device.type == 'cuda':
        out['allocated'] = torch.cuda.memory_allocated(device); out['reserved'] = torch.cuda.memory_reserved(device)
        out['cache_bytes'] = out['reserved'] - out['allocated']
    return out


def source_worker(connection, model, device, options, vjp_connection=None):
    from .source_pipeline import SourcePipeline, source_parameter
    import faulthandler; faulthandler.enable(all_threads=True)   # (a native crash prints every thread's stack)
    if os.environ.get('ISAAC_SOURCE_MPS_PRIORITY'):
        os.environ['CUDA_MPS_CLIENT_PRIORITY']=os.environ['ISAAC_SOURCE_MPS_PRIORITY']
    torch.set_num_threads(1)
    device=torch.device(device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
    # Spawn shares the incoming CPU snapshot. Never mutate that shared storage.
    model=copy.deepcopy(model)
    del model._source_snapshot_flat
    local=SourcePipeline(model,[device],**options)
    # Fresh immutable CUDA IPC exports carry producer events. Their consumer
    # streams wait for data without making this CPU dispatcher wait per chunk.
    local.ipc_async=True
    retired=0
    gradient_reply=None
    last_cotangents=None
    exports={}
    breakdown_on=os.environ.get('ISAAC_SOURCE_MEMORY_BREAKDOWN')=='1' and device.type=='cuda'
    breakdown,vjp={},{}
    # --source-streams >= 2 (with the separate VJP connection): each VJP runs in its own thread while this one
    # keeps serving encodes and Writers. CUDA: a pool of streams, the one for new work advancing at every VJP;
    # autograd runs each backward op on its forward op's stream, so a VJP over the graphs of earlier periods
    # does not queue behind the encodes of the current one. Explicit dependencies: an encode / Writer records
    # an event after its export (the default stream waits on it before the IPC send); a Writer's stream waits
    # on the events of the chunks it reads and records its use of their u (the allocator keeps that storage
    # until the Writer's stream is past it); a VJP's streams wait on the default stream (the cotangents'
    # IPC imports) and the default stream waits on all of them before the gradients are exported. Freed VJP
    # workspace is reused by later work on the same stream only (the caching allocator is per stream).
    concurrent=int(getattr(local,'streams',0) or 0)>=2 and vjp_connection is not None
    pool=[torch.cuda.Stream(device) for _ in range(4)] if concurrent and device.type=='cuda' else None
    period=[0]   # (VJPs received: the stream of new work)
    events_lock=threading.Lock()
    vjp_thread=[None]
    local.concurrent_vjp=concurrent
    shuffle=None
    if os.environ.get('ISAAC_SOURCE_STREAM_SHUFFLE'):   # (test aid: random host and device delays at every step)
        import random, time as time_
        rng_=random.Random(int(os.environ['ISAAC_SOURCE_STREAM_SHUFFLE']) * 7919 + (device.index or 0))
        shuffle_lock=threading.Lock()
        def shuffle():
            with shuffle_lock: delay,cycles=rng_.uniform(0,.02),rng_.randrange(0,4_000_000)
            if device.type=='cuda':torch.cuda._sleep(cycles)
            time_.sleep(delay)
        local.vjp_group_hook=shuffle
    def new_work():
        from contextlib import nullcontext
        return torch.cuda.stream(pool[period[0]%len(pool)]) if pool is not None else nullcontext()
    def dispatch(connection,message):
        nonlocal retired,gradient_reply,last_cotangents,exports,breakdown
        command,payload,version,state=message
        try:
            if state is not None and version not in local.versions:
                with torch.no_grad():
                    params=[p for n,p in local.model.named_parameters() if source_parameter(n)]
                    for param,value in zip(params,state.split([p.numel() for p in params])):
                        param.copy_(value.view_as(param))
                local.version=version
                local.versions[version]=local._replicas()
            if command=='encode':
                cid,parts,keys=payload
                before=torch.cuda.memory_allocated(device) if breakdown_on else 0
                done=None
                with new_work():
                    if shuffle is not None:shuffle()
                    result=local._encode(0,parts,version,keys=keys)
                    reply=ipc_export({k:v for k,v in result.items() if k!='grads'})
                    if pool is not None:done=torch.cuda.Event();done.record()
                if done is not None:torch.cuda.current_stream(device).wait_event(done)
                alloc=torch.cuda.memory_allocated(device)-before if breakdown_on else 0
                future=Future();future.set_result(result)
                local.chunks.extend([None]*max(0,cid+1-len(local.chunks)))
                uids={key[0] for key in keys}
                local.chunks[cid]=dict(worker=0,future=future,version=version,
                                       uid=next(iter(uids)) if len(uids)==1 else None,rows=len(keys),alloc=alloc,event=done)
                for offset,key in enumerate(keys): local.frames[key]=(cid,offset)
                exports['encode',cid]=reply
                if os.environ.get('ISAAC_SOURCE_EDGE_EXPORT')=='1':
                    # Reader uses the compact IPC copy. Preserve the original
                    # autograd edge, not a second copy of every Spatial output.
                    # u stays a Tensor: future Writer blocks still consume it.
                    result['y']=torch.autograd.graph.get_gradient_edge(result['y'])
                del result,future
            elif command=='write':
                wid,keys=payload
                before=torch.cuda.memory_allocated(device) if breakdown_on else 0
                done=None
                with new_work():
                    if shuffle is not None:shuffle()
                    if pool is not None:
                        stream=torch.cuda.current_stream(device)
                        for cid in {local.frames[key][0] for block in keys for key in block}:
                            chunk=local.chunks[cid]
                            if chunk.get('event') is not None:stream.wait_event(chunk['event'])
                            u=chunk['future'].result()['u']
                            if torch.is_tensor(u):u.record_stream(stream)
                    result=local._write(0,keys,version)
                    reply=ipc_export(dict(mem=result['mem']))
                    if pool is not None:done=torch.cuda.Event();done.record()
                if done is not None:torch.cuda.current_stream(device).wait_event(done)
                alloc=torch.cuda.memory_allocated(device)-before if breakdown_on else 0
                future=Future();future.set_result(result)
                local.writes.extend([None]*max(0,wid+1-len(local.writes)))
                local.writes[wid]=dict(worker=0,future=future,version=version,uid=keys[0][0][0],alloc=alloc,event=done)
                exports['write',wid]=reply
                if os.environ.get('ISAAC_SOURCE_EDGE_EXPORT')=='1':
                    result['mem']=torch.autograd.graph.get_gradient_edge(result['mem'])
                del result,future
            elif command=='backward':
                if connection is vjp_connection:connection.send(('started',None,[],{}))
                packets,retain=payload
                # The preceding interval has committed before another VJP can
                # arrive. Keep imports until then: CUDA IPC's tensor deleter
                # synchronizes its stream, so deleting them now would undo the
                # asynchronous dispatch we just established.
                last_cotangents=packets
                decoded=[((local.chunks if kind=='encode' else local.writes)[index],grads)
                         for kind,index,grads in packets]
                if breakdown_on:
                    torch.cuda.synchronize(device);vjp_before=torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                if pool is not None:   # (the forward streams the VJP's ops run on see the imported cotangents)
                    for stream in pool:stream.wait_stream(torch.cuda.current_stream(device))
                grads=local._backward(0,decoded,retain)
                if pool is not None:   # (and the gradients are exported after all of them)
                    for stream in pool:torch.cuda.current_stream(device).wait_stream(stream)
                reply=ipc_export(grads);del grads
                if breakdown_on:   # (the VJP's own peak: synchronized, so the peak is this call's)
                    torch.cuda.synchronize(device);peak=torch.cuda.max_memory_allocated(device)
                    if peak>=vjp.get('vjp_peak',0):   # (the largest since the last retire)
                        vjp.update(vjp_peak=peak,vjp_before=vjp_before,vjp_packets=len(decoded),vjp_retain=bool(retain))
                if device.type=='cuda':
                    ready=torch.cuda.Event(interprocess=True)
                    ready.record()
                    reply['_ready']=ready
                gradient_reply=reply
                del decoded,packets
            elif command=='retire':
                chunks,writes,versions=map(set,payload)
                local.frames={k:v for k,v in local.frames.items() if v[0] in chunks}
                local.chunks=[c if i in chunks else None for i,c in enumerate(local.chunks)]
                local.writes=[c if i in writes else None for i,c in enumerate(local.writes)]
                local.versions={v:r for v,r in local.versions.items() if v in versions}
                uids={uid for uid,_ in local.frames}
                for replicas in local.versions.values():
                    for replica in replicas:
                        if hasattr(replica,'_components'):replica._components.retain_source_uids(uids)
                exports={key:value for key,value in exports.items()
                         if key[1] in (chunks if key[0]=='encode' else writes)}
                retired+=1
                if breakdown_on:
                    breakdown=dict(memory_breakdown(local,exports,device),retire=retired,**vjp);vjp.clear()
                if os.environ.get('ISAAC_MEMORY_ACCOUNT')=='1':
                    from .rebuild_memory import memory_account
                    memory_account('source-'+str(device),retired,dict(chunks=local.chunks,writes=local.writes,exports=exports),
                        extra=dict(frames=len(local.frames),saved_cpu_bytes=sum(c['future'].result().get('saved_cpu_bytes',0)
                                   for c in local.chunks if c is not None),
                                   live_chunks=sum(c is not None for c in local.chunks),
                                   graph_frames=sum(c['rows'] for c in local.chunks if c is not None and not c.get('detached')),
                                   cuda_allocated=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0))
                trim_interval=int(os.environ.get('ISAAC_HOST_TRIM_INTERVAL','4'))
                if trim_interval and retired%trim_interval==0:
                    from .rebuild_memory import trim_host_heap
                    trim_host_heap('source-'+str(device))
                reply=None
            elif command=='detach':
                # Gradient horizon: settled graphs become their exported values.
                # Later Writer blocks may still read u; nothing is backpropagated.
                chunks,writes=payload
                for cid in chunks:
                    result=local.chunks[cid]['future'].result()
                    result['u']=exports['encode',cid]['u'];result['y']=None
                    local.chunks[cid]['detached']=result['_detached']=True
                for wid in writes:
                    result=local.writes[wid]['future'].result()
                    result['mem']=exports['write',wid]['mem']
                    local.writes[wid]['detached']=result['_detached']=True
                reply=None
            elif command=='close':
                local.chunks.clear();local.writes.clear();local.frames.clear();local.versions.clear()
                gc.collect();connection.send(('ok',None,[],{}));return True
            else:
                raise ValueError(command)
            if device.type=='cuda' and device in local.shared_devices and command in ('backward','retire'):
                # Freed source workspaces otherwise remain reserved in this
                # process while Actor CUDA Graph capture OOMs on the same GPU.
                idle=torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device)
                if idle>2**30:
                    import time
                    started=time.perf_counter()
                    torch.cuda.empty_cache()
                    local.events.append(dict(kind='release_shared_cache',device=str(device),
                        start=started,end=time.perf_counter()))
            memory=(dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device))
                    if device.type=='cuda' else dict(allocated=0,reserved=0))
            memory.update(chunks=sum(c is not None for c in local.chunks),
                          saved_cpu_bytes=sum(c['future'].result()['saved_cpu_bytes'] for c in local.chunks if c is not None),
                          writes=sum(c is not None for c in local.writes),
                          frames=len(local.frames),versions=len(local.versions),
                          detached_chunks=sum(c is not None and bool(c.get('detached')) for c in local.chunks),
                          detached_writes=sum(c is not None and bool(c.get('detached')) for c in local.writes),
                          graph_frames=sum(c['rows'] for c in local.chunks if c is not None and not c.get('detached')))
            if breakdown_on and breakdown:memory['breakdown']=breakdown
            if os.environ.get('ISAAC_SOURCE_COMPONENT_GRAPH_CACHE')=='1':
                component_calls={}
                for replicas in local.versions.values():
                    for replica in replicas:
                        if hasattr(replica,'_components'):
                            for key,(hits,total) in replica._components.calls.items():
                                h,t=component_calls.get(key,(0,0))
                                component_calls[key]=(h+hits,t+total)
                memory['component_calls']=component_calls
            with events_lock:   # (two threads append; send the ones present, keep any appended meanwhile)
                sent=list(local.events);del local.events[:len(sent)]
            connection.send(('ok',reply,sent,memory))
            # Imported cotangents are released before the next request. Keep
            # reply storage alive until the consumer has received its IPC handle.
            del payload,state
        except BaseException:
            connection.send(('error',traceback.format_exc(),[],{}))
            raise
    deferred=deque()
    def service_forward():
        if not connection.poll():return 0
        served=0
        # The sender submits its next chunk only after receiving this reply.
        # A zero-time poll after one chunk left all Writer requests behind the
        # VJP. Drain that forward burst, including the short RPC turnaround.
        while connection.poll(.01 if served else 0):
            message=connection.recv()
            if message[0] not in ('encode','write'):
                deferred.append(message)
                break
            dispatch(connection,message)
            served+=1
        return served
    if vjp_connection is not None and not concurrent:local.service_forward=service_forward
    while True:
        if deferred:
            channel,message=connection,deferred.popleft()
        else:
            channel=wait([connection,vjp_connection] if vjp_connection is not None else [connection])[0]
            message=channel.recv()
        if concurrent and channel is vjp_connection and message[0]=='backward':
            if vjp_thread[0] is not None:vjp_thread[0].join()   # (one interval at a time, as the learner sends)
            period[0]+=1   # (new encodes / Writers from here on: the next stream of the pool)
            vjp_thread[0]=threading.Thread(target=dispatch,args=(channel,message),name='source-vjp',daemon=True)
            vjp_thread[0].start()
            continue
        if message[0]=='close' and vjp_thread[0] is not None:vjp_thread[0].join()
        if dispatch(channel,message):break
    local.close();connection.close()
    if vjp_connection is not None:vjp_connection.close()
    if device.type=='cuda':
        # (2026-10-26) Everything is settled and replied; leave without the interpreter's teardown. The
        # CUDA gate runs saw source processes exit with SIGSEGV (-11) after a clean close, without a Python
        # frame under faulthandler (native teardown of the CUDA context / IPC state), in 2 of 3 runs.
        import sys
        torch.cuda.synchronize(device);sys.stdout.flush();sys.stderr.flush()
        os._exit(0)


class SourceProcess:
    def __init__(self, model, device, options):
        if torch.device(device).type=='cuda':
            from train_tok import ipc_safe_allocator
            ipc_safe_allocator()
        context=mp.get_context('spawn')
        self.connection,child=context.Pipe()
        self.vjp_connection=None;vjp_child=None
        if os.environ.get('ISAAC_SOURCE_PREEMPT_VJP')=='1':self.vjp_connection,vjp_child=context.Pipe()
        self.backward_started=threading.Event()
        self.process=context.Process(target=source_worker,args=(child,model,str(device),options,vjp_child),daemon=True)
        self.process.start();child.close()
        if vjp_child is not None:vjp_child.close()
        self.installed={options['version']}

    def request(self, command, payload, version=None, state=None):
        connection=self.vjp_connection if self.vjp_connection is not None and command=='backward' and payload[1] else self.connection
        if connection is self.vjp_connection:self.backward_started.clear()
        connection.send((command,payload,version,state))
        status,result,events,memory=connection.recv()
        if status=='started':
            self.backward_started.set()
            status,result,events,memory=connection.recv()
        if status=='error': raise RuntimeError(result)
        if state is not None:self.installed.add(version)
        if command=='retire':self.installed.intersection_update(payload[2])
        return result,events,memory

    def close(self):
        try:
            if self.process.is_alive(): self.request('close',None)
        finally:
            self.process.join();self.connection.close()
            if self.vjp_connection is not None:self.vjp_connection.close()
        if self.process.exitcode: raise RuntimeError(f'source process exit {self.process.exitcode}')
