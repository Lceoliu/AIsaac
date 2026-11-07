"""Four-slot spatial readout and independent block memory.

This module owns the neural computation and immutable episode state. Environment
stepping, PPO and teacher selection stay in their existing modules. All attention
uses PyTorch SDPA; there is no recurrent Spatial feedback or persistent source graph.
"""
from dataclasses import dataclass, replace
from typing import Optional
import copy
import math

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BlockConfig:
    latent_width: int = 256
    latents: int = 4
    width: int = 256
    layers: int = 4
    heads: int = 4
    window: int = 128
    block_steps: int = 32
    memory_tokens: int = 8
    memory_blocks: int = 256
    writer_layers: int = 2
    writer_heads: int = 4
    attention: str = 'sdpa'

    def __post_init__(self):
        if self.attention not in ('sdpa','flex'):
            raise ValueError('attention backend must be sdpa or flex')
        if self.width % self.heads or self.latent_width % self.writer_heads:
            raise ValueError("attention width must be divisible by heads")
        if self.layers < 2 or self.layers % 2:
            raise ValueError("Reader depth must be positive and even")
        if min(self.window, self.block_steps, self.memory_blocks, self.latents, self.memory_tokens) < 1:
            raise ValueError("memory dimensions must be positive")

    @property
    def cross_layers(self):
        return (self.layers // 2 - 1, self.layers - 1)


def time_encoding(steps, width, dtype):
    freq = torch.exp(torch.arange(0, width, 2, device=steps.device).float() * (-math.log(10000.0) / width))
    a = steps.float()[..., None] * freq
    return torch.stack((a.sin(), a.cos()), -1).flatten(-2)[..., :width].to(dtype)


def short_mask(query_steps, key_steps, window):
    delta = query_steps[..., :, None] - key_steps[..., None, :]
    return (key_steps[..., None, :] >= 0) & (delta >= 0) & (delta < window)


def long_mask(query_steps, end_steps, block_steps, blocks):
    delta = query_steps[..., :, None] - end_steps[..., None, :]
    # A complete block first becomes visible on its following decision.
    newest = torch.div(query_steps, block_steps, rounding_mode="floor") - 1
    block_id = torch.div(end_steps, block_steps, rounding_mode="floor")
    age = newest[..., :, None] - block_id[..., None, :]
    return (end_steps[..., None, :] >= 0) & (delta > 0) & (age >= 0) & (age < blocks)


class CrossAttention(nn.Module):
    def __init__(self, width, context_width, heads):
        super().__init__()
        self.heads, self.dh = heads, width // heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(context_width, 2 * width)
        self.out = nn.Linear(width, width)

    def keys(self, context):
        b, n, _ = context.shape
        k, v = self.kv(context).reshape(b, n, 2, self.heads, self.dh).unbind(2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, query, context=None, allowed=None, projected=None, bias=None,flex=None):
        b, qn, d = query.shape
        q = self.q(query).reshape(b, qn, self.heads, self.dh).transpose(1, 2)
        k, v = self.keys(context) if projected is None else projected
        mask = None if allowed is None else allowed[:, None]
        if bias is not None:
            mask = bias.to(q.dtype).masked_fill(~mask, float("-inf"))
        if flex is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            from .block_attention import attend
            y=attend(q.contiguous(),k.contiguous(),v.contiguous(),flex)
        y = self.out(y.transpose(1, 2).reshape(b, qn, d))
        if allowed is not None:
            y = torch.where(allowed.any(-1, keepdim=True), y, torch.zeros_like(y))
        return y


class QueryReadout(nn.Module):
    """Fixed learned queries read the complete, unpooled spatial representation."""
    def __init__(self, context_width, width, slots, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.context_norm = nn.LayerNorm(context_width)
        self.query_norm = nn.LayerNorm(width)
        self.attn = CrossAttention(width, context_width, heads)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, padding=None):
        q = self.queries[None].expand(len(tokens), -1, -1)
        allowed = None if padding is None else (~padding[:, None]).expand(-1, q.shape[1], -1)
        return self.norm(q + self.attn(self.query_norm(q), self.context_norm(tokens), allowed))


class WriterLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.cross = CrossAttention(width, width, heads)
        self.self_attn = CrossAttention(width, width, heads)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(width) for _ in range(3))
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, source):
        x = x + self.cross(self.n1(x), source)
        n = self.n2(x)
        x = x + self.self_attn(n, n)
        return x + self.ff(self.n3(x))


class BlockWriter(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = config
        self.queries = nn.Parameter(torch.randn(c.memory_tokens, c.latent_width) * 0.02)
        self.slots = nn.Parameter(torch.randn(c.latents, c.latent_width) * 0.02)
        self.source_norm = nn.LayerNorm(c.latent_width)
        self.layers = nn.ModuleList(WriterLayer(c.latent_width, c.writer_heads) for _ in range(c.writer_layers))
        self.norm = nn.LayerNorm(c.latent_width)

    def forward(self, z):
        # [blocks, steps, slots, latent_width]; all blocks are independent.
        c = self.config
        if z.shape[1:] != (c.block_steps, c.latents, c.latent_width):
            raise ValueError("Writer requires complete blocks with the configured shape")
        pos = time_encoding(torch.arange(c.block_steps, device=z.device), c.latent_width, z.dtype)
        source = self.source_norm(z + pos[None, :, None] + self.slots[None, None]).flatten(1, 2)
        x = self.queries[None].expand(len(z), -1, -1)
        for layer in self.layers:
            x = layer(x, source)
        return self.norm(x)


@dataclass(frozen=True)
class MemoryBlock:
    end_step: int
    value: torch.Tensor
    writer_version: int
    source_versions: tuple
    source: object = None


@dataclass(frozen=True)
class EpisodeMemory:
    """Immutable snapshots: fork shares complete blocks, never mutable tails."""
    step: int
    kv: tuple
    kv_steps: torch.Tensor
    kv_versions: torch.Tensor
    blocks: tuple
    partial: torch.Tensor
    partial_versions: tuple
    episode: object = None

    def fork(self):
        return replace(self, kv=tuple((k.clone(), v.clone()) for k, v in self.kv),
                       kv_steps=self.kv_steps.clone(), kv_versions=self.kv_versions.clone(),
                       partial=self.partial.clone())

    def to(self, device):
        return replace(self, kv=tuple((k.to(device), v.to(device)) for k, v in self.kv),
                       kv_steps=self.kv_steps.to(device), kv_versions=self.kv_versions.to(device),
                       blocks=tuple(replace(b, value=b.value.to(device)) for b in self.blocks),
                       partial=self.partial.to(device))


class ReaderLayer(nn.Module):
    def __init__(self, config, with_memory):
        super().__init__()
        c = config
        self.heads, self.dh = c.heads, c.width // c.heads
        self.n1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.out = nn.Linear(c.width, c.width)
        self.memory = CrossAttention(c.width, c.latent_width, c.heads) if with_memory else None
        self.n_memory = nn.LayerNorm(c.width) if with_memory else None
        self.n2 = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(), nn.Linear(4 * c.width, c.width))

    def forward(self, x, prefix, allowed, bank, bank_allowed, bank_bias,bank_flex=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, self.dh).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        pk, pv = (a.to(k.dtype) for a in prefix)
        y = F.scaled_dot_product_attention(q, torch.cat((pk, k), 2), torch.cat((pv, v), 2),
                                          attn_mask=allowed[:, None], dropout_p=0.0)
        x = x + self.out(y.transpose(1, 2).reshape(b, n, d))
        if self.memory is not None:
            x = x + self.memory(self.n_memory(x), bank, bank_allowed, bias=bank_bias,flex=bank_flex)
        return x + self.ff(self.n2(x)), (k, v)


class BlockMemory(nn.Module):
    def __init__(self, spatial_width, config=None):
        super().__init__()
        self.config = c = config if isinstance(config, BlockConfig) else BlockConfig(**(config or {}))
        self.resampler = QueryReadout(spatial_width, c.latent_width, c.latents, c.writer_heads)
        self.writer = BlockWriter(c)
        self.input = nn.Linear(c.latent_width, c.width)
        self.slot_embedding = nn.Parameter(torch.randn(c.latents, c.width) * 0.02)
        self.memory_slot_embedding = nn.Parameter(torch.randn(c.memory_tokens, c.latent_width) * 0.02)
        self.memory_age = nn.Parameter(torch.zeros(c.heads, c.memory_blocks))
        self.layers = nn.ModuleList(ReaderLayer(c, i in c.cross_layers) for i in range(c.layers))
        self.norm = nn.LayerNorm(c.width)
        self.pool = QueryReadout(c.width, c.width, 1, c.heads)

    def empty(self, device=None, dtype=None, episode=None):
        c = self.config
        p = next(self.parameters())
        device = p.device if device is None else device
        dtype = p.dtype if dtype is None else dtype
        kv = tuple((torch.empty(c.heads, 0, c.width // c.heads, device=device, dtype=dtype),
                    torch.empty(c.heads, 0, c.width // c.heads, device=device, dtype=dtype)) for _ in self.layers)
        return EpisodeMemory(0, kv, torch.empty(0, device=device, dtype=torch.long),
                             torch.empty(0, device=device, dtype=torch.long), (),
                             torch.empty(0, c.latents, c.latent_width, device=device, dtype=dtype), (), episode)

    def read(self, z, position, prefixes, bank, ends, kp):
        """Tensor-only Reader shared by eager training and captured actor inference."""
        c=self.config
        b,t=z.shape[:2]
        qt=position.repeat_interleave(c.latents,1)
        bank=bank+time_encoding(ends.clamp_min(0),c.latent_width,bank.dtype)
        bm=long_mask(qt,ends,c.block_steps,c.memory_blocks)
        bank_flex=bank_bias=None
        if c.attention=='flex' and z.is_cuda and t==c.window and t>1:
            from .block_attention import plan
            bank_flex=plan(bm,qt,ends,self.memory_age,c.block_steps)
        else:
            age=(torch.div(qt,c.block_steps,rounding_mode='floor')[:,:,None]-1
                 -torch.div(ends,c.block_steps,rounding_mode='floor')[:,None]).clamp(0,c.memory_blocks-1)
            bank_bias=self.memory_age[:,age].permute(1,0,2,3)
        allowed=short_mask(qt,torch.cat((kp,qt),1),c.window)
        x=self.input(z)+self.slot_embedding[None,None]+time_encoding(position,c.width,z.dtype)[:,:,None]
        x=x.flatten(1,2)
        current_kv=[]
        for layer,prefix in zip(self.layers,prefixes):
            x,kv=layer(x,prefix,allowed,bank,bm,bank_bias,bank_flex)
            current_kv.append(kv)
        tokens=self.norm(x).reshape(b,t,c.latents,c.width)
        hidden=self.pool(tokens.flatten(0,1)).reshape(b,t,c.width)
        return hidden,tokens,current_kv,allowed,bm

    def forward(self, z, states, version=0, replacements=None, source_prefixes=None, reader=None,
                require_next_state=True):
        """Parallel chunk with detached per-layer pre-chunk KV.

        z: [batch, decisions, four slots, latent_width]. Each sequence stays in one
        episode. replacements maps historical end_step to differentiable Writer
        output. source_prefixes optionally supplies recomputed cross-boundary tails.
        Returned states own detached caches; the returned representation has a graph.
        """
        c = self.config
        b, t, slots, width = z.shape
        if (slots, width) != (c.latents, c.latent_width) or len(states) != b:
            raise ValueError("latent/state batch shape does not match BlockConfig")
        dev = z.device
        # Independent complete blocks from every sequence go through Writer together.
        blocks_in, owners, combined = [], [], []
        for i, state in enumerate(states):
            tail = state.partial.detach() if source_prefixes is None else source_prefixes[i]
            all_z = torch.cat((tail.to(z.dtype), z[i]), 0)
            combined.append(all_z)
            complete = len(all_z) // c.block_steps
            first_end = state.step - len(tail) + c.block_steps - 1
            for j in range(complete):
                blocks_in.append(all_z[j*c.block_steps:(j+1)*c.block_steps])
                owners.append((i, first_end + j*c.block_steps, j))
        written = self.writer(torch.stack(blocks_in)) if blocks_in else None
        new_blocks = [[] for _ in states]
        for n, (i, end, j) in enumerate(owners):
            tail_versions = states[i].partial_versions if source_prefixes is None else (int(version),) * len(source_prefixes[i])
            versions = tail_versions + (int(version),) * t
            sv = versions[j*c.block_steps:(j+1)*c.block_steps]
            new_blocks[i].append(MemoryBlock(end, written[n], int(version), sv))
        # A single memory layout is shared by the two long-reading layers.
        memories = []
        for i, s in enumerate(states):
            replace_i = {} if replacements is None else replacements[i]
            memories.append([replace(m, value=replace_i.get(m.end_step, m.value.detach()))
                             for m in s.blocks] + new_blocks[i])
        max_m = max(1, max(len(m) for m in memories) * c.memory_tokens)
        flex_on=c.attention=='flex' and z.is_cuda and t==c.window and t>1
        if flex_on:
            # Fixed bank capacity avoids a new compiler graph every 16 blocks;
            # Flex's block mask skips the not-yet-filled slots. Ragged tails and
            # single-step actors use SDPA rather than compiling arbitrary lengths.
            max_m=((c.memory_blocks+(t+c.block_steps-1)//c.block_steps+1)*c.memory_tokens+127)//128*128
        # One pinned metadata transfer for the entire request batch. Constructing
        # each bank's ends directly on CUDA synchronizes the host once per owner.
        metadata_host=torch.empty((b,1+max_m),dtype=torch.long,pin_memory=z.is_cuda)
        metadata_np=metadata_host.numpy()
        metadata_np.fill(-1)
        metadata_np[:,0]=[s.step for s in states]
        banks=[]
        for i,ms in enumerate(memories):
            if ms:
                value = torch.cat([m.value for m in ms], 0).to(z.dtype)
                value = (value.reshape(-1, c.memory_tokens, c.latent_width)
                         + self.memory_slot_embedding).flatten(0, 1)
            else:
                value = z.new_empty((0, c.latent_width))
            metadata_np[i,1:1+len(ms)*c.memory_tokens]=np.repeat([m.end_step for m in ms],c.memory_tokens)
            banks.append(F.pad(value, (0, 0, 0, max_m-len(value))) if len(value)!=max_m else value)
        metadata=metadata_host.to(dev,non_blocking=True)
        position=metadata[:,:1]+torch.arange(t,device=dev)
        qt=position.repeat_interleave(c.latents,1)
        bank,ends=torch.stack(banks),metadata[:,1:]
        max_p = max(len(s.kv_steps) for s in states)
        kp = torch.stack([F.pad(s.kv_steps, (0, max_p-len(s.kv_steps)), value=-1) for s in states])
        prefixes=[]
        for l in range(c.layers):
            prefix = tuple(torch.stack([F.pad(s.kv[l][j].detach(), (0, 0, 0, max_p-len(s.kv_steps)))
                                        if len(s.kv_steps) != max_p else s.kv[l][j].detach()
                                        for s in states]) for j in (0, 1))
            prefixes.append(prefix)
        hidden,tokens,current_kv,allowed,bm=(self.read if reader is None else reader)(
            z,position,prefixes,bank,ends,kp)
        result=dict(hidden=hidden,tokens=tokens,states=[],written=new_blocks,
                    local_mask=allowed,memory_mask=bm,positions=position,current_kv=current_kv)
        if not require_next_state:
            return result
        next_states = []
        new_versions = torch.full_like(qt, int(version))
        batch_cache=None
        if b>1 and t<c.window:
            # Rebuild all ragged rolling caches in one gather per K/V tensor,
            # rather than eight tiny cat kernels for every owner in the batch.
            lengths=[min(len(s.kv_steps)+t*c.latents,c.window*c.latents) for s in states]
            indices_host=torch.empty((b,max(lengths)),dtype=torch.long,pin_memory=z.is_cuda)
            indices_np=indices_host.numpy()
            for i,s in enumerate(states):
                keep=lengths[i]-t*c.latents
                indices_np[i,:keep]=np.arange(len(s.kv_steps)-keep,len(s.kv_steps))
                indices_np[i,keep:lengths[i]]=np.arange(max_p,max_p+t*c.latents)
                indices_np[i,lengths[i]:]=max_p+t*c.latents-1
            indices=indices_host.to(dev,non_blocking=True)
            next_times=torch.cat((kp,qt),1).gather(1,indices)
            previous_versions=torch.stack([F.pad(s.kv_versions,(0,max_p-len(s.kv_versions)),value=-1)
                if len(s.kv_versions)!=max_p else s.kv_versions for s in states])
            next_versions=torch.cat((previous_versions,new_versions),1).gather(1,indices)
            batch_cache=tuple(tuple(torch.cat((prefixes[l][j].to(current_kv[l][j].dtype),current_kv[l][j]),2)
                .gather(2,indices[:,None,:,None].expand(-1,c.heads,-1,c.width//c.heads)).detach()
                for j in (0,1)) for l in range(c.layers))
        for i, s in enumerate(states):
            if t >= c.window:
                times = qt[i, -c.window*c.latents:]
                versions = new_versions[i, -c.window*c.latents:]
                kv = tuple(tuple(current_kv[l][j][i, :, -c.window*c.latents:].detach()
                                 for j in (0, 1)) for l in range(c.layers))
            elif batch_cache is not None:
                times=next_times[i,:lengths[i]]
                versions=next_versions[i,:lengths[i]]
                kv=tuple(tuple(batch_cache[l][j][i,:,:lengths[i]] for j in (0,1)) for l in range(c.layers))
            else:
                times = torch.cat((s.kv_steps, qt[i]))[-c.window*c.latents:]
                versions = torch.cat((s.kv_versions, new_versions[i]))[-len(times):]
                kv = tuple(tuple(torch.cat((s.kv[l][j].to(current_kv[l][j].dtype), current_kv[l][j][i]), 1)
                                 [:, -c.window*c.latents:].detach() for j in (0, 1)) for l in range(c.layers))
            # Historical recomputation is a learner-local view, not a write-back
            # into the actor's canonical version-labelled memory.
            canonical = list(s.blocks) + new_blocks[i]
            ms = tuple(replace(m, value=m.value.detach()) if m.value.requires_grad else m
                       for m in canonical[-c.memory_blocks:])
            remainder = len(combined[i]) % c.block_steps
            tail = combined[i][-remainder:].detach() if remainder else z.new_empty((0, c.latents, c.latent_width))
            tail_versions = s.partial_versions if source_prefixes is None else (int(version),)*len(source_prefixes[i])
            pver = (tail_versions + (int(version),)*t)[-remainder:] if remainder else ()
            next_states.append(EpisodeMemory(s.step+t, kv, times, versions, ms, tail, pver, s.episode))
        result['states']=next_states
        return result


def historical_blocks(state, generator=None):
    """Fixed budget: newest complete source block plus one uniformly older block."""
    if not state.blocks:
        return ()
    recent = len(state.blocks) - 1
    if recent == 0:
        return (state.blocks[recent],)
    older = int(torch.randint(recent, (1,), generator=generator).item())
    return state.blocks[recent], state.blocks[older]


def recompute_history(model, states, encode_source, generator=None):
    """Source loader/encoder is shared by PPO and teacher minibatches.

    encode_source receives a MemoryBlock with a retained raw-source reference and
    returns [block_steps, slots, latent_width]. Cached context is never mutated.
    """
    selected = [historical_blocks(s, generator) for s in states]
    unique = {}
    for chosen in selected:
        for block in chosen:
            key = id(block)
            if key not in unique:
                unique[key] = (block, encode_source(block))
    if not unique:
        return [{} for _ in states], selected
    keys = list(unique)
    written = model.writer(torch.stack([unique[k][1] for k in keys]))
    by_id = dict(zip(keys, written.unbind()))
    return [{block.end_step: by_id[id(block)] for block in chosen} for chosen in selected], selected


class TeacherGenerations:
    """Two shared immutable policies, pinned for a whole comparison/recheck.

    publish returns False under backpressure; it never mutates a pinned policy.
    The main actor can continue regardless of whether publication succeeds.
    """
    def __init__(self):
        self.slots = [None, None]
        self.refs = [0, 0]
        self.versions = [-1, -1]

    def publish(self, model, version):
        if version in self.versions:
            return True
        free = [i for i in range(2) if self.refs[i] == 0]
        if not free:
            return False
        i = min(free, key=lambda n: self.versions[n])
        if self.slots[i] is None:
            self.slots[i] = copy.deepcopy(model).eval().requires_grad_(False)
        else:
            self.slots[i].load_state_dict(model.state_dict())
        self.versions[i] = int(version)
        return True

    def acquire(self, version=None):
        i = (max(range(2), key=lambda n: self.versions[n]) if version is None
             else self.versions.index(int(version)))
        if self.slots[i] is None:
            raise RuntimeError("publish a teacher generation before acquiring")
        self.refs[i] += 1
        return i, self.versions[i], self.slots[i]

    def release(self, slot):
        if self.refs[slot] <= 0:
            raise RuntimeError("teacher generation released without a matching acquire")
        self.refs[slot] -= 1


def warm_start_spatial(model, old_state):
    raise ValueError('visible-v1 requires fresh random initialization; C99/legacy warm-start is forbidden')

