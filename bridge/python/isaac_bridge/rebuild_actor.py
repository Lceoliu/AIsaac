"""CUDA-graph inference for the non-recurrent source/Reader policy.

Uses the existing raw-byte decoder, sampling and PyTorch graph primitives.
Host histories retain provenance; device rings retain source values, not H.
Completed Writer outputs grow geometrically: no history is discarded.
"""
from collections import OrderedDict
from contextlib import nullcontext
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from .rebuild_memory import ComponentCache, PROFILE_ACTOR, ACTOR_TIMES, actor_part
from .actor_profile import ActorProfile
from .tok_policy import ROW_BYTES, decode_rows, crn_uniforms, gumbel_sample, _sdpa_math
from .tok_obs import ROW, ENT_CAP


class TensorActorMemory:
    def __init__(self, model, slots, scratch=64, projected=True):
        self.model,self.slots,self.scratch=model,slots,scratch
        slots+=scratch
        rd=model.reader;dev=next(model.parameters()).device
        self.capacity=64
        self.length=np.zeros(slots,np.int64)
        self.generations=np.full(slots,model.cache_generation,np.int64)
        self.revision=0
        self.u=torch.zeros((slots,rd.W,rd.D),device=dev)
        self.kv=None
        self.tail=torch.zeros((slots,rd.block,rd.D),device=dev)
        self.mem=torch.zeros((slots,self.capacity,rd.D),device=dev)
        self.mkv=None
        if projected:self.enable_reader()

    def enable_reader(self):
        if self.kv is not None:return
        rd=self.model.reader;slots=len(self.length);dev=self.u.device
        self.kv=torch.zeros((slots,rd.W,rd.L,2*rd.D),device=dev)
        self.mkv=torch.zeros((slots,self.capacity,rd.L,2*rd.D),device=dev)
        self.generations.fill(-1)
        torch._dynamo.mark_dynamic(self.mkv,1)

    def new(self, slots):
        return TensorActorMemory(self.model,slots,self.scratch,projected=False)

    def grow(self, required):
        if required<=self.capacity:return
        cap=self.capacity
        while cap<required:cap*=2
        self.mem=F.pad(self.mem,(0,0,0,cap-self.capacity))
        if self.mkv is not None:
            self.mkv=F.pad(self.mkv,(0,0,0,0,0,cap-self.capacity))
            torch._dynamo.mark_dynamic(self.mkv,1)
        self.capacity=cap;self.revision+=1

    def copy(self, src, dst, other):
        rd=self.model.reader
        count=int(other.length[src])//rd.block*rd.tokens
        self.grow(count)
        self.u[dst].copy_(other.u[src])
        self.tail[dst].copy_(other.tail[src])
        self.mem[dst,:count].copy_(other.mem[src,:count])
        if self.kv is not None and other.kv is not None:
            self.kv[dst].copy_(other.kv[src])
            self.mkv[dst,:count].copy_(other.mkv[src,:count])
            self.generations[dst]=other.generations[src]
        else:self.generations[dst]=-1
        self.length[dst]=other.length[src]

    def load(self, host, slots):
        rd=self.model.reader
        for s in slots:
            length=host.raw[s].length
            self.length[s]=length
            self.generations[s]=-1
            if not length:continue
            count=length//rd.block*rd.tokens;self.grow(count)
            ix=torch.arange(max(0,length-rd.W),length,device=self.u.device)%rd.W
            self.u[s,ix]=host.recent[s]
            if self.kv is not None:self.kv[s,ix]=host.recent_kv[s]
            if count:
                self.mem[s,:count]=host.mem[s]
                if self.mkv is not None:self.mkv[s,:count]=host.memory_kv[s]
            if host.tail[s]:self.tail[s,:len(host.tail[s])]=torch.stack(host.tail[s])

    def export(self, host, slots):
        self.refresh(slots);rd=self.model.reader
        for s in slots:
            length=int(self.length[s]);count=length//rd.block*rd.tokens
            if length:
                ix=torch.arange(max(0,length-rd.W),length,device=self.u.device)%rd.W
                host.recent[s]=self.u[s,ix];host.recent_kv[s]=self.kv[s,ix]
                host.mem[s]=self.mem[s,:count];host.memory_kv[s]=self.mkv[s,:count]
                host.tail[s]=list(self.tail[s,:length%rd.block].clone().unbind(0))
            host.generation[s]=self.model.source_generation

    @torch.no_grad()
    def refresh(self, slots=None):
        """Only cheap Reader projections refresh; source/Writer values stay old."""
        rd=self.model.reader
        self.enable_reader()
        selected=np.arange(len(self.length)) if slots is None else np.asarray(slots)
        stale=selected[self.generations[selected]!=self.model.cache_generation]
        active=stale[self.length[stale]>0]
        for lo in range(0,len(active),32):
            group=active[lo:lo+32]
            dev=self.u.device;s=torch.as_tensor(group,device=dev)
            t=torch.as_tensor(self.length[group],device=dev)[:,None]
            ring=torch.arange(rd.W,device=dev)[None]
            times=t-1-(t-1-ring)%rd.W
            x=self.u[s]+rd.position(times.flatten(),self.u.dtype).reshape(len(s),rd.W,rd.D)
            for i,layer in enumerate(rd.layers):self.kv[s,:,i]=layer.kv(x)
            count=int(self.length[group].max())//rd.block*rd.tokens
            if count:
                mt=(torch.arange(count,device=dev)//rd.tokens+1)*rd.block-1
                x=self.mem[s,:count]+rd.position(mt,self.mem.dtype)[None]
                for i,layer in enumerate(rd.layers):self.mkv[s,:count,i]=layer.kv(x)
        self.generations[stale]=self.model.cache_generation


class RebuildGraphActor:
    def __init__(self,model,rows_max,entities=32,stream=None,crn=True,memory=None,
                 bf16=False,compile_mode=None,extra=None,fold=False,sdpa_math=False,
                 row_sizes=None,temporal=None):
        self.model,self.n,self.stream=model,rows_max,stream
        self.crn,self.bf16,self.extra_fn=crn,bf16,extra
        self.tstate=temporal
        self.nh=len(model.heads);self.vg_on=model.gtok is not None
        self.t_on=True;self.mem=0;self.fold=True
        self.extra=self.extra_out=self.h_used=self.vg=self.t_out=None
        self.done=torch.cuda.Event()
        self.capture=torch.cuda.Stream()
        self.graph_pool=(torch.cuda.graph_pool_handle() if
                         os.environ.get('ISAAC_ACTOR_SHARED_GRAPH_POOL')=='1' else None)
        self.graphs={};self.component_graphs={};self.writer_graphs={}
        dev=next(model.parameters()).device;n=self.n
        if temporal.rebuild.fast is None:
            temporal.rebuild.fast=TensorActorMemory(model,len(temporal.rebuild.raw),self.n)
            temporal.rebuild.fast.load(temporal.rebuild,range(len(temporal.rebuild.raw)))
        self.state=temporal.rebuild.fast
        self.state.enable_reader()
        self.stage=torch.zeros((n,ROW_BYTES+8*self.nh),dtype=torch.uint8,pin_memory=True)
        self.raw=torch.zeros_like(self.stage,device=dev)
        # slot, length and each component's (missing row, destination, gather).
        self.control_h=torch.zeros((17,n),dtype=torch.long,pin_memory=True)
        self.control=torch.zeros_like(self.control_h,device=dev)
        self.uniform_h=torch.zeros((n,sum(model.heads)),pin_memory=True)
        self.uniform=torch.zeros_like(self.uniform_h,device=dev)
        # Reserve the maximum width, but each transfer uses a compact view.
        # A [:n,:actual_width] slice of a wider 2-D buffer is non-contiguous:
        # D2H then allocates a pageable temporary despite non_blocking=True.
        self.cpu=torch.zeros(n*(self.nh+2+2+int(self.vg_on)),pin_memory=True)
        self.tables={};self.keys={};self.cache_generation=-1
        self.component_names=('patch','player','grid','map','items')
        width=model.kind.shape[-1]
        for name in self.component_names:
            shapes=([(model.patch[-2].out_features,)] if name=='patch' else
                    [(width,)] if name!='items' else [(model.itok.n_tok,width),(model.itok.n_tok,)])
            self.tables[name]=tuple(torch.zeros((2048,*shape),device=dev,
                dtype=torch.bool if name=='items' and j==1 else torch.float32) for j,shape in enumerate(shapes))
            self.keys[name]=OrderedDict()
        es=[entities] if np.ndim(entities)==0 else list(entities)
        self.esizes=sorted({min(int(e),ENT_CAP) for e in es}|{ENT_CAP})
        self.rsizes=sorted({min(int(r),n) for r in (row_sizes or (16,32))}|{n})
        self.revision=self.state.revision
        self.sdpa_math=sdpa_math
        self.core_fn=self._core
        self.component_fn=self._component_run
        if compile_mode:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit=max(dynamo.config.cache_size_limit,256)
            dynamo.config.accumulated_cache_size_limit=max(dynamo.config.accumulated_cache_size_limit,4096)
            # Only the memory-bank axis is dynamic. Growing an episode still
            # recaptures addresses, but must not recompile the entire policy.
            self.core_fn=torch.compile(self._core,mode=compile_mode,dynamic=None)
            self.component_fn=torch.compile(self._component_run,mode=compile_mode,dynamic=False)
        self.calls=0
        if PROFILE_ACTOR:
            import cProfile
            self.profile=cProfile.Profile()
        self.aprof=ActorProfile(self,'rebuild')   # (ISAAC_ACTOR_TORCH_PROFILE only)

    def _capture(self,fn):
        # Independent cuBLAS workspace per actor, as in the legacy GraphActor.
        self.capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture),(_sdpa_math() if self.sdpa_math else nullcontext()):
            for _ in range(3):out=fn()
        torch.cuda.current_stream().wait_stream(self.capture)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        if self.graph_pool is None:
            with torch.cuda.graph(graph,stream=self.capture),(_sdpa_math() if self.sdpa_math else nullcontext()):out=fn()
        else:
            # Every graph owns its outputs outside the shared scratch pool.
            # No live graph-pool tensor crosses a replay boundary, so independent
            # buckets may run in any order. Pools are never shared across actors.
            out=tuple(torch.empty_like(value) for value in out)
            def capture_to_outputs():
                for dest,value in zip(out,fn()):dest.copy_(value)
            with torch.cuda.graph(graph,pool=self.graph_pool,stream=self.capture),(_sdpa_math() if self.sdpa_math else nullcontext()):
                capture_to_outputs()
        return graph,out

    def _component_run(self,j,m):
        name=self.component_names[j];base=2+3*j
        b=decode_rows(self.raw.index_select(0,self.control[base,:m]),entities=8,n_heads=self.nh)
        model=self.model
        if name=='player':
            source=self.control[4].index_select(0,self.control[base,:m])
            b['_encoded_components']={'patch':self.tables['patch'][0].index_select(0,source)}
            out=(model._player_token(b),)
        elif name=='items':out=model.itok(b,compact=True)
        else:out=(getattr(model, 'room' if name=='grid' else name)(model._conv_in(b[name])),)
        for table,value in zip(self.tables[name],out):
            table.index_copy_(0,self.control[base+1,:m],value.to(table.dtype))
        return out

    def _prepare_components(self,rows,pending):
        if self.cache_generation!=self.model.source_generation:
            for table in self.keys.values():table.clear()
            self.cache_generation=self.model.source_generation
        ctl=self.control_h.numpy();counts=[]
        for j,name in enumerate(self.component_names):
            table=self.keys[name];missing=[];dest=[];gather=[]
            for i,key in enumerate(ComponentCache.payload_keys(rows,pending,name)):
                if key not in table:
                    pos=len(table) if len(table)<2048 else table.popitem(last=False)[1]
                    table[key]=pos;missing.append(i);dest.append(pos)
                table.move_to_end(key);gather.append(table[key])
            base=2+3*j;ctl[base+2,:len(rows)]=gather
            ctl[base+2,len(rows):]=gather[0]
            m=len(missing)
            if m:
                bucket=next(r for r in self.rsizes if r>=m)
                ctl[base,:m]=missing;ctl[base,m:bucket]=missing[0]
                ctl[base+1,:m]=dest;ctl[base+1,m:bucket]=dest[0]
                counts.append((j,bucket))
        return counts

    def _core(self,r,e):
        model=self.model;rd=model.reader;st=self.state
        b=decode_rows(self.raw[:r],entities=e,n_heads=self.nh)
        comps={}
        for j,name in enumerate(self.component_names):
            vals=tuple(t.index_select(0,self.control[2+3*j+2,:r]) for t in self.tables[name])
            comps[name]=vals if name=='items' else vals[0]
        b['_encoded_components']=comps
        y,pad,u=model.encode_rebuild(b)
        slots=self.control[0,:r];times=self.control[1,:r]
        pos=rd.position(times,u.dtype)
        recent=st.kv.index_select(0,slots)
        memories=st.mkv.index_select(0,slots)
        zero=torch.stack([l.kv(u.new_zeros((1,rd.D))) for l in rd.layers],1)
        bank=torch.cat((zero[None].expand(r,-1,-1,-1),recent,memories),1)
        valid=torch.cat((torch.ones((r,1),dtype=torch.bool,device=u.device),
            torch.arange(rd.W,device=u.device)[None]<times[:,None],
            torch.arange(st.mkv.shape[1],device=u.device)[None]<(times//rd.block*rd.tokens)[:,None]),1)
        q=u+pos
        for layer,kv in zip(rd.layers,bank.unbind(2)):
            query=layer.q(layer.norm(q)).reshape(r,rd.H,1,rd.D//rd.H)
            kv=kv.reshape(r,-1,2,rd.H,rd.D//rd.H)
            attn=F.scaled_dot_product_attention(query,kv[:,:,0].transpose(1,2),kv[:,:,1].transpose(1,2),
                                               attn_mask=valid[:,None,None]).reshape(r,rd.D)
            x=q+layer.out(attn);q=x+layer.ff(layer.ffnorm(x))
        o=rd.norm(q);c=rd.fuse(o,y,pad)
        outs=model._heads(y[:,0]+rd.read(o)+c,True,yg=y[:,-1]+rd.read_g(o)+rd.fusion_g(c),need_vg=self.vg_on)
        actions,lp=gumbel_sample(model.split(outs[0].float(),model.gate(b)),self.uniform[:r])
        ex=self.extra_fn(b,outs,None) if self.extra_fn is not None else None
        fields=[actions.float(),lp[:,None],outs[1].float()[:,None]]
        if ex is not None:fields.append(ex.float())
        if self.vg_on:fields.append(outs[-1].float()[:,None])
        frame=torch.stack([l.kv(u+pos) for l in rd.layers],1)
        # Padding rows use distinct scratch slots, never duplicate live indices.
        st.u[slots,times%rd.W]=u
        st.kv[slots,times%rd.W]=frame
        st.tail[slots,times%rd.block]=u
        return torch.cat(fields,1),u,o,outs[0]

    @torch.no_grad()
    def submit(self,rows,pending,widx=None,reset=None,tdest=None,ukeys=None):
        n=len(rows)
        if n>self.n:return False
        self._n=n
        self.calls+=1
        ap=self.aprof;ap.begin(n);ap.mark('reset_prefixes')
        profiling=PROFILE_ACTOR and 257<=self.calls<=384
        if profiling:self.profile.enable()
        stream=self.stream if self.stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(stream),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.bf16):
            stamp=actor_part('start')
            st=self.state;rd=self.model.reader;host=self.tstate.rebuild
            slots=np.asarray(widx,np.int64)
            for s,first in zip(slots,rows['first']):
                if first:host.reset(int(s))
            prefixes=[host.raw[s] for s in slots]
            times=np.asarray([h.length for h in prefixes],np.int64)
            ap.mark('grow_refresh')
            ap.note(stale=int((st.generations[slots]!=self.model.cache_generation).sum()),capacity=int(st.capacity),
                    len_max=int(times.max()),len_mean=float(times.mean()))
            st.grow(int((times.max()+1)//rd.block*rd.tokens))
            st.refresh(slots)
            if self.revision!=st.revision:
                self.graphs.clear();self.revision=st.revision
            stamp=actor_part('graph_refresh',stamp)
            ap.mark('stage_pack')
            r=next(r for r in self.rsizes if r>=n)
            e=next(e for e in self.esizes if e>=int(rows['n_ent'].max()))
            sn=self.stage.numpy();sn[:n,:ROW_BYTES]=rows.view(np.uint8).reshape(n,ROW_BYTES)
            sn[:n,ROW_BYTES:]=np.ascontiguousarray(pending,np.int64).view(np.uint8).reshape(n,-1)
            sn[n:r]=sn[0]
            ap.mark('component_keys')
            counts=self._prepare_components(rows,pending)
            ap.note(r=r,e=e,component_buckets=len(counts),component_rows=sum(m for _,m in counts))
            stamp=actor_part('graph_host_pack_keys',stamp)
            ap.mark('h2d_uniform')
            ctl=self.control_h.numpy();ctl[0,:n]=slots;ctl[1,:n]=times
            ctl[0,n:r]=st.slots+np.arange(r-n);ctl[1,n:r]=0
            self.raw.copy_(self.stage,non_blocking=True);self.control.copy_(self.control_h,non_blocking=True)
            if ukeys is not None:un=crn_uniforms(ukeys[0],ukeys[1],sum(self.model.heads))
            elif self.crn and 'crn' in rows.dtype.names:un=crn_uniforms(rows['crn'],rows['t'],sum(self.model.heads))
            else:un=None
            if un is None:self.uniform.uniform_()
            else:
                self.uniform_h.numpy()[:n]=un;self.uniform.copy_(self.uniform_h,non_blocking=True)
            ap.mark('component_replay')
            for j,m in counts:
                key=(j,m)
                if key not in self.component_graphs:self.component_graphs[key]=self._capture(lambda:self.component_fn(j,m))
                self.component_graphs[key][0].replay()
            stamp=actor_part('graph_components',stamp)
            ap.mark('core_replay')
            key=(r,e)
            ap.note(core_capture=key not in self.graphs)
            if key not in self.graphs:
                # Core mutates the current ring slot only, which the read excludes
                # until the next decision. Warmup must not expose that new frame.
                saved=(st.u[slots,times%rd.W].clone(),st.kv[slots,times%rd.W].clone(),st.tail[slots,times%rd.block].clone())
                self.graphs[key]=self._capture(lambda:self.core_fn(r,e))
                st.u[slots,times%rd.W]=saved[0];st.kv[slots,times%rd.W]=saved[1];st.tail[slots,times%rd.block]=saved[2]
            graph,out=self.graphs[key];graph.replay()
            stamp=actor_part('graph_core',stamp)
            ap.mark('d2h')
            cols,u,o,logits=(v[:n] for v in out)
            self.last_logits=logits
            self.n_extra=cols.shape[1]-self.nh-2-int(self.vg_on)
            self.cpu[:cols.numel()].view_as(cols).copy_(cols,non_blocking=True)
            ap.mark('rawhistory_append')
            for i,s in enumerate(slots):
                host.raw[s]=prefixes[i].append(rows[i],pending[i],rd.block)
            ap.mark('state_tstate')
            st.length[slots]=times+1
            sd=self.control[0,:n]
            self.tstate.prev_o.index_copy_(0,sd,o.float())
            self.tstate.steps.index_copy_(0,sd,self.control[1,:n]+1)
            dt,off=ROW.fields['stage'][:2]
            self.tstate.prev_stage.index_copy_(0,sd,self.raw[:n,off].long())
            closed=np.flatnonzero((times+1)%rd.block==0)
            ap.mark('writer_close');ap.note(closed=len(closed))
            if len(closed):
                ss=torch.as_tensor(slots[closed],device=u.device)
                mt=torch.as_tensor(times[closed],device=u.device)
                values=rd.write(st.tail.index_select(0,ss))
                ix=(mt//rd.block)[:,None]*rd.tokens+torch.arange(rd.tokens,device=u.device)[None]
                st.mem[ss[:,None],ix]=values
                projected=torch.stack([l.kv(values+rd.position(mt,u.dtype)[:,None]) for l in rd.layers],2)
                st.mkv[ss[:,None],ix]=projected
            stamp=actor_part('graph_append_writer',stamp)
            ap.mark('tdest')
            self.t_out=dict(create=np.zeros(n,bool),nroom=np.zeros(n,np.int64),steps=times+1,
                            stage_prev=rows['stage'].astype(np.int64),raw_contexts=prefixes)
            dummy=u.new_zeros((n,self.tstate.L,2,self.tstate.D))
            if tdest is None:self.t_out.update(kv=dummy,s=u,o=o)
            else:
                ix=torch.as_tensor(tdest['rows'],device=u.device)
                tdest['kv'].index_copy_(0,ix,dummy.to(tdest['kv'].dtype))
                tdest['o'].index_copy_(0,ix+tdest['obase'],o.float())
                if tdest.get('s') is not None:tdest['s'].index_copy_(0,ix,u.to(tdest['s'].dtype))
            if ap.eager_wanted():   # (measurement aid: the network once more, uncompiled, outside the graphs;
                # its ring writes are put back to the replay's values, as after a capture's warm-up)
                saved=(st.u[slots,times%rd.W].clone(),st.kv[slots,times%rd.W].clone(),st.tail[slots,times%rd.block].clone())
                with (_sdpa_math() if self.sdpa_math else nullcontext()):ap.eager(lambda:self._core(r,e))
                st.u[slots,times%rd.W]=saved[0];st.kv[slots,times%rd.W]=saved[1];st.tail[slots,times%rd.block]=saved[2]
            self.done.record(stream)
            ap.submitted()
            actor_part('graph_output',stamp)
        if profiling:self.profile.disable()
        if PROFILE_ACTOR and self.calls%384==0:
            import json,os
            from pathlib import Path
            base=Path(PROFILE_ACTOR)/f'actor-{os.getpid()}-{id(self)}'
            base.with_suffix('.json').write_text(json.dumps(dict(calls=self.calls,seconds=ACTOR_TIMES,
                graphs=len(self.graphs),component_graphs=len(self.component_graphs))))
            if self.calls==384:self.profile.dump_stats(str(base.with_suffix('.pstats')))
        return True

    def result(self):
        t=time.perf_counter()
        self.done.synchronize()
        wait=time.perf_counter()-t
        width=self.nh+2+self.n_extra+int(self.vg_on)
        out=self.cpu[:self._n*width].view(self._n,width).numpy().copy();n=self.nh
        self.extra=out[:,n+2:n+2+self.n_extra] if self.n_extra else None
        self.vg=out[:,-1] if self.vg_on else None
        self.aprof.finished(wait)
        return out[:,:n].astype(np.int64),out[:,n],out[:,n+1]

    def act(self,*args,**kwargs):
        return self.result() if self.submit(*args,**kwargs) else None
