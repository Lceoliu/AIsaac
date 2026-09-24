"""Synchronous on-policy PPO with asynchronous, chunked CPU/CUDA collection.

Weights never change during collection. Only frame-local features are cached;
the temporal Transformer is evaluated afresh and training uses raw observations.
"""
from concurrent.futures import as_completed
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import explained_variance
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
    # Collection sampler factory, called with the model; None = FrameSampler (graph_sampler.py
    # provides a CUDA-graph variant for the AB+ learner).
    sampler_class=None

    def __init__(self,*args,**kwargs):
        # TF32 cuDNN changes CNN rounding with batch shape (one frame vs H
        # frames). Keep collection and PPO likelihood evaluation in strict FP32.
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        kwargs.setdefault('rollout_buffer_class',GpuHistoryRolloutBuffer)
        kwargs.setdefault('batch_size',32)
        self.policy_version=0;self._gpu_sampler=None;self._collecting=False
        self.session=None;self.sampling_deterministic=False
        # None keeps SB3's per-window minibatches; an int enables frame-deduplicated segment
        # minibatches with gradient accumulation (see _train_segments).
        self.micro_batch_size=None;self.segment_length=32
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_gpu_sampler','_collecting','session','sampler_class']

    def _setup_learn(self,*args,**kwargs):
        result=super()._setup_learn(*args,**kwargs)
        if not isinstance(self.env,GpuFrameVecEnv):raise TypeError('GPU PPO requires GpuFrameVecEnv')
        # The VecEnv reset contract is preserved, but its one-time NumPy window
        # must not remain resident throughout training. State lives in raw GPU frames.
        self._last_obs={}
        return result

    def train(self):
        if self._collecting:raise RuntimeError('Weights are frozen while collecting a rollout')
        try:
            if self.micro_batch_size:self._train_segments()
            else:super().train()
        finally:self.policy_version+=1
        if self.session is not None:self.session.after_update(self)

    def evaluate_segments(self,workers,t0):
        """Values, log-probs and entropies for whole L-step segments of single workers.

        Every stored frame the segment needs is encoded once and shared by the windows that
        contain it. Each sample still sees exactly its own causal window (same frames, times,
        padding and last token), so per-sample outputs equal policy.evaluate_actions on
        buffer.window(); only float summation order differs.
        """
        b=self.rollout_buffer;enc=self.policy.features_extractor
        H,L,dev=b.history,self.segment_length,self.device
        rows=t0[:,None]+torch.arange(L+H-1,device=dev)             # frame rows the segment may need
        q=t0[:,None]+torch.arange(L,device=dev)+H-1                 # frame row of each sample's observation
        lengths=b.lengths[q,workers[:,None]]
        first=q[:,0]-lengths[:,0]+1                                  # windows' union is [first, q[:,-1]]
        obs={k:v[rows,workers[:,None]] for k,v in b.frames.items()}
        obs['history_mask']=(rows>=first[:,None]).to(obs['history_mask'].dtype)
        obs=preprocess_obs(obs,self.observation_space,normalize_images=self.policy.normalize_images)
        frames=enc.encode_frames(obs)
        j=torch.arange(H,device=dev)
        valid=j<lengths[...,None]
        offset=torch.where(valid,torch.arange(L,device=dev)[:,None]+H-lengths[...,None]+j,0)
        seg=torch.arange(len(workers),device=dev)[:,None,None]
        window=frames[seg,offset].masked_fill(~valid[...,None],0).flatten(0,1)
        elapsed=obs['time'][seg,offset].masked_fill(~valid,0).flatten(0,1)
        valid=valid.flatten(0,1)
        latent=enc.temporal_features(window,elapsed,valid)
        latent=latent[torch.arange(len(latent),device=dev),valid.long().sum(-1)-1]
        pi,vf=self.policy.mlp_extractor(latent)
        dist=self.policy._get_action_dist_from_latent(pi)
        steps=(q-(H-1)).flatten();owners=workers[:,None].expand(-1,L).flatten()
        dist.apply_masking(b.action_masks[steps,owners])
        return self.policy.value_net(vf).flatten(),dist.log_prob(b.actions[steps,owners]),dist.entropy(),steps,owners

    def _train_segments(self):
        """PPO over frame-deduplicated segment minibatches with gradient accumulation.

        batch_size is the effective minibatch (whole segments, advantages normalised over it);
        micro_batch_size only bounds activation memory. Loss, clipping, entropy/value weights,
        grad clipping, target_kl and log keys follow MaskablePPO.train.
        """
        b=self.rollout_buffer;policy=self.policy
        T,N,L=b.buffer_size,b.n_envs,self.segment_length
        if T%L or self.batch_size%self.micro_batch_size or self.micro_batch_size%L:
            raise ValueError('segment_length must divide n_steps and micro_batch_size; micro_batch_size must divide batch_size')
        policy.set_training_mode(True)
        self._update_learning_rate(policy.optimizer)
        clip_range=self.clip_range(self._current_progress_remaining)
        clip_range_vf=self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        per_worker=T//L;segments=N*per_worker
        seg_batch=self.batch_size//L;seg_micro=self.micro_batch_size//L
        pg_losses,value_losses,entropy_losses,clip_fractions=[],[],[],[]
        continue_training=True;steps=0;loss=0.0
        for epoch in range(self.n_epochs):
            approx_kl_divs=[]
            order=torch.randperm(segments,device=self.device)
            for s in range(0,segments,seg_batch):
                chosen=order[s:s+seg_batch]
                workers=torch.div(chosen,per_worker,rounding_mode='floor');t0=(chosen%per_worker)*L
                adv=b.advantages[t0[:,None]+torch.arange(L,device=self.device),workers[:,None]]
                mean,std,n=adv.mean(),adv.std(),adv.numel()
                policy.optimizer.zero_grad()
                sums=torch.zeros(5,device=self.device)
                for m in range(0,len(chosen),seg_micro):
                    values,log_prob,entropy,t,w=self.evaluate_segments(workers[m:m+seg_micro],t0[m:m+seg_micro])
                    a=(b.advantages[t,w]-mean)/(std+1e-8)
                    old=b.log_probs[t,w]
                    ratio=torch.exp(log_prob-old)
                    pg=-torch.min(a*ratio,a*torch.clamp(ratio,1-clip_range,1+clip_range))
                    if clip_range_vf is not None:
                        values=b._values[t,w]+torch.clamp(values-b._values[t,w],-clip_range_vf,clip_range_vf)
                    vl=(b._returns[t,w]-values)**2
                    el=-entropy
                    ((pg.sum()+self.ent_coef*el.sum()+self.vf_coef*vl.sum())/n).backward()
                    with torch.no_grad():
                        lr_=log_prob-old
                        sums+=torch.stack([pg.sum(),vl.sum(),el.sum(),((torch.exp(lr_)-1)-lr_).sum(),
                                           ((ratio-1).abs()>clip_range).float().sum()])
                pg_loss,value_loss,entropy_loss,approx_kl,clip_fraction=(sums/n).tolist()
                pg_losses.append(pg_loss);value_losses.append(value_loss);entropy_losses.append(entropy_loss)
                approx_kl_divs.append(approx_kl);clip_fractions.append(clip_fraction)
                loss=pg_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                # Same semantics as SB3: KL is measured on this minibatch before its step.
                if self.target_kl is not None and approx_kl>1.5*self.target_kl:
                    continue_training=False
                    if self.verbose>=1:print(f'Early stopping at step {epoch} due to reaching max kl: {approx_kl:.2f}')
                    break
                torch.nn.utils.clip_grad_norm_(policy.parameters(),self.max_grad_norm)
                policy.optimizer.step();steps+=1
            if not continue_training:break
        self._n_updates+=self.n_epochs
        explained_var=explained_variance(b.values.flatten(),b.returns.flatten())
        self.logger.record('train/entropy_loss',np.mean(entropy_losses))
        self.logger.record('train/policy_gradient_loss',np.mean(pg_losses))
        self.logger.record('train/value_loss',np.mean(value_losses))
        self.logger.record('train/approx_kl',np.mean(approx_kl_divs))
        self.logger.record('train/clip_fraction',np.mean(clip_fractions))
        self.logger.record('train/loss',loss)
        self.logger.record('train/explained_variance',explained_var)
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip_range)
        if clip_range_vf is not None:self.logger.record('train/clip_range_vf',clip_range_vf)
        self.logger.record('train/optimizer_steps',steps)

    def collect_rollouts(self,env,callback,rollout_buffer,n_rollout_steps,use_masking=True):
        if not use_masking:raise ValueError('Isaac requires action masking')
        if n_rollout_steps!=rollout_buffer.buffer_size:raise ValueError('Collect the configured rollout length')
        self.policy.set_training_mode(False)
        self._collecting=True
        versions=tuple(p._version for p in self.policy.parameters())
        try:
            with torch.no_grad():
                if self._gpu_sampler is None:self._gpu_sampler=(self.sampler_class or FrameSampler)(self)
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
                            actions,values,log_prob,masks=s.action(old,ids,self.sampling_deterministic)
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
