"""Dense active-slot state kernels for block memory.

This is the tensor execution core. Pool/lease ownership and the actor/rollout
integration live in block_paged (PagedStore / SlotBlockActor). No EpisodeMemory
or RawHistory object is constructed by step().

A step is split into read_inputs (gather, before the Reader) and commit (write
the four new KV positions, the latent tail and, on a completed block, one Writer
call), so several arenas (main actor, teacher generations) can share one Reader
call. Bank and tail latents are float32 (Writer/LayerNorm outputs are float32 even
under autocast); only K/V use kv_dtype, which is lossless for BF16-produced K/V.
"""
import numpy as np
import torch


class BlockSlotState:
    def __init__(self,config,slots,device='cpu',dtype=torch.float32,kv_dtype=None):
        self.config,self.device,self.dtype=config,torch.device(device),dtype
        self.kv_dtype=dtype if kv_dtype is None else kv_dtype
        c=config;self.slots=int(slots);self.step_host=np.zeros(slots,np.int64)
        self.kv=torch.zeros((c.layers,2,slots,c.heads,c.window*c.latents,c.width//c.heads),device=device,dtype=self.kv_dtype)
        self.bank=torch.zeros((slots,c.memory_blocks,c.memory_tokens,c.latent_width),device=device,dtype=dtype)
        self.partial=torch.zeros((slots,c.block_steps,c.latents,c.latent_width),device=device,dtype=dtype)
        self.steps=torch.zeros(slots,device=device,dtype=torch.long)
        self.kv_steps=torch.full((slots,c.window*c.latents),-1,device=device,dtype=torch.long)
        self.kv_versions=torch.full_like(self.kv_steps,-1)
        self.bank_ends=torch.full((slots,c.memory_blocks),-1,device=device,dtype=torch.long)
        self.bank_versions=torch.full_like(self.bank_ends,-1)
        self.partial_sources=torch.full((slots,c.block_steps),-1,device=device,dtype=torch.long)
        self.partial_versions=torch.full_like(self.partial_sources,-1)
        self.bank_sources=torch.full((slots,c.memory_blocks,c.block_steps),-1,device=device,dtype=torch.long)
        self.bank_source_versions=torch.full_like(self.bank_sources,-1)
        self.kv_axis=torch.arange(c.window*c.latents,device=device)
        self.bank_axis=torch.arange(c.memory_blocks,device=device)
        self.latent_axis=torch.arange(c.latents,device=device)
        self._rows=('bank','partial','steps','kv_steps','kv_versions','bank_ends','bank_versions',
                    'partial_sources','partial_versions','bank_sources','bank_source_versions')

    def _ids(self,slots):
        host=np.asarray(slots,dtype=np.int64).reshape(-1)
        ids=torch.from_numpy(host)
        if self.device.type=='cuda':ids=ids.pin_memory().to(self.device,non_blocking=True)
        return host,ids

    @torch.no_grad()
    def reset(self,slots):
        host,ids=self._ids(slots)
        self.step_host[host]=0;self.kv.index_fill_(2,ids,0)
        for name in self._rows:
            getattr(self,name).index_fill_(0,ids,0 if name in ('bank','partial','steps') else -1)

    @torch.no_grad()
    def fork(self,sources,destinations):
        src_host,src=self._ids(sources);dst_host,dst=self._ids(destinations)
        self.step_host[dst_host]=self.step_host[src_host].copy()
        self.kv.index_copy_(2,dst,self.kv.index_select(2,src))
        for name in self._rows:
            value=getattr(self,name);value.index_copy_(0,dst,value.index_select(0,src))

    @torch.no_grad()
    def load(self,slot,step,kv,kv_steps,kv_versions,partial,partial_versions,partial_sources,
             bank,bank_ends,bank_versions,bank_sources,bank_source_versions):
        """Materialise one slot from paged storage (restore of an arbitrary snapshot).

        kv [n,layers,2,heads,latents,dh] for decisions kv_steps (host, ascending, n<=window);
        partial [p,latents,latent_width] for the open tail; bank [m,tokens,latent_width] for
        complete blocks ending at bank_ends (host, ascending, m<=memory_blocks)."""
        c=self.config;slot=int(slot);step=int(step)
        self.reset([slot])
        dev=self.device
        def put(x,dtype=None):
            x=torch.as_tensor(x)
            if dev.type=='cuda' and x.device.type=='cpu':x=x.pin_memory()
            return x.to(dev,dtype=dtype,non_blocking=True)
        n=len(kv_steps)
        if n:
            steps=np.asarray(kv_steps,np.int64)
            columns=(steps[:,None]*c.latents+np.arange(c.latents))%(c.window*c.latents)
            col=put(columns.reshape(-1))
            values=put(kv,self.kv_dtype)                       # [n,L,2,H,lat,dh]
            values=values.permute(1,2,3,0,4,5).reshape(c.layers,2,c.heads,n*c.latents,c.width//c.heads)
            self.kv[:,:,slot].index_copy_(3,col,values)
            self.kv_steps[slot].index_copy_(0,col,put(np.repeat(steps,c.latents)))
            self.kv_versions[slot].index_copy_(0,col,put(np.repeat(np.asarray(kv_versions,np.int64),c.latents)))
        p=len(partial_versions)
        if p:
            self.partial[slot,:p]=put(partial,self.dtype)
            self.partial_versions[slot,:p]=put(np.asarray(partial_versions,np.int64))
            self.partial_sources[slot,:p]=put(np.asarray(partial_sources,np.int64))
        m=len(bank_ends)
        if m:
            ends=np.asarray(bank_ends,np.int64)
            col=put((ends//c.block_steps)%c.memory_blocks)
            self.bank[slot].index_copy_(0,col,put(bank,self.dtype))
            self.bank_ends[slot].index_copy_(0,col,put(ends))
            self.bank_versions[slot].index_copy_(0,col,put(np.asarray(bank_versions,np.int64)))
            self.bank_sources[slot].index_copy_(0,col,put(np.asarray(bank_sources,np.int64)))
            self.bank_source_versions[slot].index_copy_(0,col,put(np.asarray(bank_source_versions,np.int64)))
        self.step_host[slot]=step;self.steps[slot]=step

    def gather(self,ids,metadata=True):
        """One batched tensor view, with chronological ring order and padding."""
        c=self.config;b=len(ids);steps=self.steps.index_select(0,ids)
        ki=((steps-c.window).clamp_min(0)[:,None]*c.latents+self.kv_axis)%len(self.kv_axis)
        bi=((torch.div(steps,c.block_steps,rounding_mode='floor')-c.memory_blocks).clamp_min(0)[:,None]
            +self.bank_axis)%c.memory_blocks
        kv=self.kv.index_select(2,ids).gather(4,ki[None,None,:,None,:,None].expand(
            c.layers,2,b,c.heads,c.window*c.latents,c.width//c.heads))
        bank=self.bank.index_select(0,ids).gather(1,bi[:,:,None,None].expand(-1,-1,c.memory_tokens,c.latent_width))
        read=dict(steps=steps,kv=kv,kv_steps=self.kv_steps.index_select(0,ids).gather(1,ki),bank=bank,
                  ends=self.bank_ends.index_select(0,ids).gather(1,bi))
        if not metadata:return read
        read.update(
                    kv_versions=self.kv_versions.index_select(0,ids).gather(1,ki),
                    bank_versions=self.bank_versions.index_select(0,ids).gather(1,bi),
                    bank_sources=self.bank_sources.index_select(0,ids).gather(1,bi[:,:,None].expand(-1,-1,c.block_steps)),
                    bank_source_versions=self.bank_source_versions.index_select(0,ids).gather(1,bi[:,:,None].expand(-1,-1,c.block_steps)),
                    partial=self.partial.index_select(0,ids),partial_sources=self.partial_sources.index_select(0,ids),
                    partial_versions=self.partial_versions.index_select(0,ids))
        return read

    def read_inputs(self,memory,ids,dtype):
        """Reader inputs for these slots: (steps, prefixes, bank, ends, kv_steps)."""
        c=self.config
        previous=self.gather(ids,metadata=False)
        bank=(previous['bank'].to(dtype)+memory.memory_slot_embedding).flatten(1,2)
        # Read before commit: the current block is causally invisible, so do not
        # append it (2056 tokens would round ReaderStepGraph up to 4096).
        ends=previous['ends'].repeat_interleave(c.memory_tokens,1)
        prefixes=tuple((previous['kv'][layer,0],previous['kv'][layer,1]) for layer in range(c.layers))
        return previous['steps'],prefixes,bank,ends,previous['kv_steps']

    @torch.no_grad()
    def commit(self,memory,host,ids,steps,z,current,version=0,source_ids=None):
        """Write the new step. current: [layers,2,b,heads,latents,dh]. Returns completed-block info."""
        self.commit_device(ids,steps,z,current,version,source_ids)
        return self.commit_blocks(memory,host,ids,steps,z.dtype,version)

    @torch.no_grad()
    def commit_device(self,ids,steps,z,current,version=0,source_ids=None):
        """The per-step device writes of commit (capturable: no host reads). version: an int
        or a 0-d long device tensor (B42 glue 5: the actor step graph)."""
        c=self.config;phase=steps%c.block_steps
        columns=(steps[:,None]*c.latents+self.latent_axis)%len(self.kv_axis)
        # Only the four new positions are written; never clone a 512-token
        # destination cache merely to scatter four entries back into it.
        self.kv[:,:,ids[:,None],:,columns,:]=current.permute(2,4,0,1,3,5).to(self.kv_dtype)
        ver=(torch.full_like(columns,int(version)) if not isinstance(version,torch.Tensor)
             else version.expand_as(columns))
        for name,values in (('kv_steps',steps[:,None].expand(-1,c.latents)),('kv_versions',ver)):
            getattr(self,name)[ids[:,None],columns]=values
        self.partial[ids,phase]=z.to(self.dtype)
        self.partial_versions[ids,phase]=(torch.full_like(phase,int(version)) if not isinstance(version,torch.Tensor)
                                          else version.expand_as(phase))
        # Device-side fill values (a Python scalar here is a host->device copy, illegal in capture).
        self.partial_sources[ids,phase]=(torch.full_like(phase,-1) if source_ids is None
                                         else torch.as_tensor(source_ids,device=self.device,dtype=torch.long))
        self.steps.index_copy_(0,ids,steps+1)

    @torch.no_grad()
    def commit_blocks(self,memory,host,ids,steps,dtype,version=0):
        """The completed-block part of commit (host-decided Writer call), then the host step."""
        c=self.config
        completed=np.flatnonzero((self.step_host[host]+1)%c.block_steps==0)
        written=None
        if len(completed):
            _,positions=self._ids(completed);done=ids.index_select(0,positions)
            written=memory.writer(self.partial.index_select(0,done).to(dtype))
            end=steps.index_select(0,positions);column=torch.div(end,c.block_steps,rounding_mode='floor')%c.memory_blocks
            self.bank[done,column]=written.to(self.dtype)
            self.bank_ends[done,column]=end;self.bank_versions[done,column]=int(version)
            self.bank_sources[done,column]=self.partial_sources.index_select(0,done)
            self.bank_source_versions[done,column]=self.partial_versions.index_select(0,done)
            self.partial_sources.index_fill_(0,done,-1);self.partial_versions.index_fill_(0,done,-1)
        self.step_host[host]+=1
        return completed,written

    @torch.no_grad()
    def step(self,memory,z,slots,version=0,source_ids=None,reader=None):
        c=self.config;host,ids=self._ids(slots)
        steps,prefixes,bank,ends,kp=self.read_inputs(memory,ids,z.dtype)
        hidden,tokens,current,allowed,mask=(memory.read if reader is None else reader)(
            z[:,None],steps[:,None],prefixes,bank,ends,kp)
        current=torch.stack([torch.stack(pair) for pair in current])
        completed,written=self.commit(memory,host,ids,steps,z,current,version,source_ids)
        return dict(hidden=hidden,tokens=tokens,local_mask=allowed,memory_mask=mask,positions=steps[:,None],
                    current_kv=current,completed=completed,written=written)
