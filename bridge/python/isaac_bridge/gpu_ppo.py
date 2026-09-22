"""Synchronous on-policy PPO with asynchronous, chunked CPU/CUDA collection.

Weights never change during collection. Only frame-local features are cached;
the temporal Transformer is evaluated afresh and training uses raw observations.
"""
from concurrent.futures import as_completed
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from .gpu_buffer import GpuHistoryRolloutBuffer
from .gpu_env import GpuFrameVecEnv,decode_frame,metadata


class FrameSampler:
    def __init__(self,model):
        self.model=model;self.env=model.env;self.buffer=model.rollout_buffer
        self.encoder=model.policy.features_extractor
        self.features=torch.zeros((*self.buffer.lengths.shape,self.encoder.features_dim),device=model.device)
        self.starts=torch.ones(self.env.num_envs,dtype=torch.bool,device=model.device)
        self.workers=[torch.arange(c.start,c.stop,device=model.device) for c in self.env.chunks]
        self.generation=-1;self.version=-1;self.steps=0
        self.upload_bytes=0;self.encoded_frames=0;self.prefix_rebuilds=0

    def encode(self,position,workers,raw):
        for k,v in raw.items():self.buffer.frames[k][position,workers]=v
        self.features[position,workers]=self.encoder.encode_frames({k:v[:,None] for k,v in raw.items()})[:,0]
        self.encoded_frames+=len(workers)

    def begin(self):
        b=self.buffer
        version=(self.model.policy_version,tuple(p._version for p in self.model.policy.parameters()))
        # Join all previously queued writes before compacting the shared prefix.
        stream=torch.cuda.current_stream(self.model.device)
        for c in self.env.chunks:
            if self.generation>=0:stream.wait_event(c.ready)
        if self.generation!=self.env.generation:
            b.reset();b.frame_pos=b.history-1;b.lengths.zero_();self.features.zero_();self.starts.fill_(True)
            for c,ids in zip(self.env.chunks,self.workers):
                slot=c.slots[0]
                slot.device.copy_(slot.host,non_blocking=True)
                self.encode(b.frame_pos,ids,decode_frame(slot.device,self.env.observation_space))
                b.lengths[b.frame_pos,ids]=1
                self.upload_bytes+=slot.host.numel()*slot.host.element_size()
                slot.upload_done.record(stream);slot.consumed.record(stream);slot.used=True
            self.generation=self.env.generation;self.version=version
        else:
            start=b.frame_pos-b.history+1
            self.features[:b.history].copy_(self.features[start:b.frame_pos+1].clone())
            b.retain_prefix()
            if self.version!=version:
                # Chunk prefix re-encoding limits temporary activations after an update.
                for ids in self.workers:
                    obs=b.window(b.frame_pos,ids)
                    fused=self.encoder.encode_frames(obs)
                    _,times,valid=b.indices(b.frame_pos,ids)
                    rows=ids[:,None].expand_as(times)
                    self.features[times[valid],rows[valid]]=fused[valid]
                    self.encoded_frames+=int(valid.sum().item())
                self.prefix_rebuilds+=1;self.version=version
        ready=torch.cuda.Event();ready.record(stream)
        for c in self.env.chunks:c.compute_stream.wait_event(ready)

    def latent(self,position,ids):
        b=self.buffer;ids,times,valid=b.indices(position,ids)
        fused=self.features[times,ids[:,None]].masked_fill(~valid[:,:,None],0)
        elapsed=b.frames['time'][times,ids[:,None]].masked_fill(~valid,0)
        seq=self.encoder.temporal_features(fused,elapsed,valid)
        last=valid.long().sum(-1)-1
        return seq[torch.arange(len(ids),device=ids.device),last]

    def values(self,position,ids):
        policy=self.model.policy
        return policy.value_net(policy.mlp_extractor.forward_critic(self.latent(position,ids))).flatten()

    def action(self,position,ids,deterministic=False):
        policy=self.model.policy
        masks=torch.ones((len(ids),49),dtype=torch.bool,device=ids.device)
        masks[:,46]=self.buffer.frames['player'][position,ids,9]>0;masks[:,48]=False
        pi,vf=policy.mlp_extractor(self.latent(position,ids))
        distribution=policy._get_action_dist_from_latent(pi)
        distribution.apply_masking(masks)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions,policy.value_net(vf).flatten(),distribution.log_prob(actions),masks


class GpuMaskablePPO(MaskablePPO):
    def __init__(self,*args,**kwargs):
        # TF32 cuDNN changes CNN rounding with batch shape (one frame vs H
        # frames). Keep collection and PPO likelihood evaluation in strict FP32.
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        kwargs.setdefault('rollout_buffer_class',GpuHistoryRolloutBuffer)
        kwargs.setdefault('batch_size',32)
        self.policy_version=0;self._gpu_sampler=None;self._collecting=False
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_gpu_sampler','_collecting']

    def _setup_learn(self,*args,**kwargs):
        result=super()._setup_learn(*args,**kwargs)
        if not isinstance(self.env,GpuFrameVecEnv):raise TypeError('GPU PPO requires GpuFrameVecEnv')
        # The VecEnv reset contract is preserved, but its one-time NumPy window
        # must not remain resident throughout training. State lives in raw GPU frames.
        self._last_obs={}
        return result

    def train(self):
        if self._collecting:raise RuntimeError('Weights are frozen while collecting a rollout')
        try:super().train()
        finally:self.policy_version+=1

    def collect_rollouts(self,env,callback,rollout_buffer,n_rollout_steps,use_masking=True):
        if not use_masking:raise ValueError('Isaac requires action masking')
        if n_rollout_steps!=rollout_buffer.buffer_size:raise ValueError('Collect the configured rollout length')
        self.policy.set_training_mode(False)
        self._collecting=True
        versions=tuple(p._version for p in self.policy.parameters())
        try:
            with torch.no_grad():
                if self._gpu_sampler is None:self._gpu_sampler=FrameSampler(self)
                s=self._gpu_sampler;b=rollout_buffer;s.begin()
                callback.on_rollout_start()
                dones=self._last_episode_starts
                for t in range(n_rollout_steps):
                    if versions!=tuple(p._version for p in self.policy.parameters()):
                        raise RuntimeError('Policy weights changed during frozen collection')
                    old=b.frame_pos;new=old+1;pending={}
                    for c,ids in zip(env.chunks,s.workers):
                        slot=c.slots[s.steps%2]
                        with torch.cuda.stream(c.compute_stream):
                            actions,values,log_prob,masks=s.action(old,ids)
                            starts=s.starts[ids]
                            slot.actions.copy_(actions,non_blocking=True)
                            slot.action_ready.record(c.compute_stream)
                        future=env.executor.submit(c.advance_and_upload,slot)
                        pending[future]=(c,ids,slot,actions,values,log_prob,masks,starts)
                    rewards=np.empty(env.num_envs,np.float32);dones=np.zeros(env.num_envs,bool);infos=[None]*env.num_envs
                    for future in as_completed(pending):
                        c,ids,slot,actions,values,log_prob,masks,starts=pending[future]
                        cr,cd,ci=future.result()
                        terminal=np.flatnonzero(cd)
                        rewards[c.start:c.stop]=cr;dones[c.start:c.stop]=cd;infos[c.start:c.stop]=ci
                        s.upload_bytes+=(c.n+len(terminal))*slot.host.shape[1]*slot.host.element_size()
                        with torch.cuda.stream(c.compute_stream):
                            c.compute_stream.wait_event(slot.upload_done)
                            raw=decode_frame(slot.device,env.observation_space)
                            b.lengths[new,ids]=(b.lengths[old,ids]+1).clamp_max(b.history)
                            s.encode(new,ids,raw)
                            reward=metadata(slot.device,'reward').clone()
                            if len(terminal):
                                local=torch.as_tensor(terminal,device=self.device);done_ids=ids[local]
                                terminal_obs=b.window(new,done_ids)
                                for j,i in enumerate(terminal):ci[i]['terminal_observation']={k:v[j] for k,v in terminal_obs.items()}
                                timeout=[i for i in terminal if ci[i]['TimeLimit.truncated']]
                                if timeout:
                                    ti=torch.as_tensor(timeout,device=self.device)
                                    reward[ti]+=self.gamma*s.values(new,ids[ti])
                                reset=decode_frame(slot.reset_device[:len(terminal)],env.observation_space)
                                s.encode(new,done_ids,reset);b.lengths[new,done_ids]=1
                            b.add_chunk(t,ids,actions,values,log_prob,masks,reward,starts)
                            s.starts[ids]=metadata(slot.device,'done').bool()
                            slot.consumed.record(c.compute_stream);c.ready.record(c.compute_stream)
                    for c in env.chunks:torch.cuda.current_stream(self.device).wait_event(c.ready)
                    b.frame_pos=new;b.pos=t+1;b.full=b.pos==b.buffer_size;s.steps+=1
                    self.num_timesteps+=env.num_envs;self._last_episode_starts=dones;self._last_obs={}
                    self._update_info_buffer(infos,dones)
                    actions=b.actions[t];values=b._values[t];log_probs=b.log_probs[t]
                    callback.update_locals(locals())
                    if not callback.on_step():return False
                last_values=torch.empty(env.num_envs,device=self.device)
                for c,ids in zip(env.chunks,s.workers):
                    with torch.cuda.stream(c.compute_stream):
                        last_values[ids]=s.values(b.frame_pos,ids);c.ready.record(c.compute_stream)
                    torch.cuda.current_stream(self.device).wait_event(c.ready)
                b.compute_returns_and_advantage(last_values,dones)
                if versions!=tuple(p._version for p in self.policy.parameters()):
                    raise RuntimeError('Policy weights changed during frozen collection')
                callback.on_rollout_end()
                return True
        finally:self._collecting=False
