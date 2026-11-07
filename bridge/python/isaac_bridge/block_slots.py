"""Dense active-slot state kernels for block memory.

This is the tensor execution core. Pool/lease ownership and the actor/rollout
integration are separate: callers reserve destination slots before fork/copy.
No EpisodeMemory or RawHistory object is constructed by step().
"""
import numpy as np
import torch


class BlockSlotState:
    def __init__(self,config,slots,device='cpu',dtype=torch.float32):
        self.config,self.device,self.dtype=config,torch.device(device),dtype
        c=config;self.step_host=np.zeros(slots,np.int64)
        self.kv=torch.zeros((c.layers,2,slots,c.heads,c.window*c.latents,c.width//c.heads),device=device,dtype=dtype)
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
        host=np.asarray(slots,dtype=np.int64)
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
                    kv_versions=self.kv_versions.index_select(0,ids).gather(1,ki),bank=bank,
                    ends=self.bank_ends.index_select(0,ids).gather(1,bi),
                    bank_versions=self.bank_versions.index_select(0,ids).gather(1,bi),
                    bank_sources=self.bank_sources.index_select(0,ids).gather(1,bi[:,:,None].expand(-1,-1,c.block_steps)),
                    bank_source_versions=self.bank_source_versions.index_select(0,ids).gather(1,bi[:,:,None].expand(-1,-1,c.block_steps)),
                    partial=self.partial.index_select(0,ids),partial_sources=self.partial_sources.index_select(0,ids),
                    partial_versions=self.partial_versions.index_select(0,ids))
        return read

    @torch.no_grad()
    def step(self,memory,z,slots,version=0,source_ids=None,reader=None):
        c=self.config;host,ids=self._ids(slots);b=len(host)
        previous=self.gather(ids,metadata=False);steps=previous['steps'];phase=steps%c.block_steps
        bank=(previous['bank'].to(z.dtype)+memory.memory_slot_embedding).flatten(1,2)
        # Read before commit: the current block is causally invisible, so do not
        # append it (2056 tokens would round ReaderStepGraph up to 4096).
        ends=previous['ends'].repeat_interleave(c.memory_tokens,1)
        prefixes=tuple((previous['kv'][layer,0],previous['kv'][layer,1]) for layer in range(c.layers))
        hidden,tokens,current,allowed,mask=(memory.read if reader is None else reader)(
            z[:,None],steps[:,None],prefixes,bank,ends,previous['kv_steps'])
        columns=(steps[:,None]*c.latents+self.latent_axis)%len(self.kv_axis)
        current=torch.stack([torch.stack(pair) for pair in current])
        # Only the four new positions are written; never clone a 512-token
        # destination cache merely to scatter four entries back into it.
        self.kv[:,:,ids[:,None],:,columns,:]=current.permute(2,4,0,1,3,5).to(self.dtype)
        for name,values in (('kv_steps',steps[:,None].expand(-1,c.latents)),
                            ('kv_versions',torch.full_like(columns,int(version)))):
            getattr(self,name)[ids[:,None],columns]=values
        self.partial[ids,phase]=z.to(self.dtype)
        self.partial_versions[ids,phase]=int(version)
        self.partial_sources[ids,phase]=(-1 if source_ids is None else torch.as_tensor(source_ids,device=self.device,dtype=torch.long))
        completed=np.flatnonzero((self.step_host[host]+1)%c.block_steps==0)
        if len(completed):
            _,positions=self._ids(completed);done=ids.index_select(0,positions)
            written=memory.writer(self.partial.index_select(0,done).to(z.dtype))
            end=steps.index_select(0,positions);column=torch.div(end,c.block_steps,rounding_mode='floor')%c.memory_blocks
            self.bank[done,column]=written.to(self.dtype)
            self.bank_ends[done,column]=end;self.bank_versions[done,column]=int(version)
            self.bank_sources[done,column]=self.partial_sources.index_select(0,done)
            self.bank_source_versions[done,column]=self.partial_versions.index_select(0,done)
            self.partial_sources.index_fill_(0,done,-1);self.partial_versions.index_fill_(0,done,-1)
        self.step_host[host]+=1;self.steps.index_copy_(0,ids,steps+1)
        return dict(hidden=hidden,tokens=tokens,local_mask=allowed,memory_mask=mask,positions=steps[:,None])
