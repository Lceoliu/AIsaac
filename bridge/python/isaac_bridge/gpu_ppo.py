"""On-policy PPO with asynchronous, chunked CPU/CUDA collection.

Weights never change during collection. Only frame-local features are cached;
the temporal Transformer is evaluated afresh and training uses raw observations.

async_training (segment training only) overlaps each update with the next rollout. A copy of the
policy, the actor, collects; the learner trains on a snapshot of the previous rollout in a
background thread on its own CUDA stream. At every rollout boundary the update is joined, the
session checkpoints/evaluates, the actor copies the learner's weights in place (captured CUDA
graphs stay valid), the finished rollout is copied into the snapshot and the next update starts.
Each rollout is therefore collected by weights one update older than those its update starts
from. The update uses the decoupled PPO objective (IMPACT, Luo et al. 2019; Hilton et al. 2021):
the clipped ratio is taken against the proximal policy, the weights the update starts from
(log-probabilities recomputed without gradient before the first step), and each sample is weighted
by pi_proximal/pi_behaviour, truncated at lag_weight_max, for the one-update lag. Clipping against
the collection log-probabilities instead anchors the trust region one update back: measured on
AB+ (abp-mix-01d, 2026-09-25) the clip fraction rose from 0.03 to 0.2-0.46 and approx_kl doubled.
With no lag (synchronous training) the objective is standard PPO.

Action heads (combat-v5): any MultiDiscrete ending in (bomb, item) works; the masks allow the bomb
only with bombs and never the item (action_masks). Entropy is logged per head (train/entropy_<head>)
and head_ent_coefs, if set, weights each head's entropy instead of ent_coef. A policy with an
auxiliary geometry head (transformer_policy.GeometryPolicy) and observations carrying its labels
(aim_label, fire_distance, approach) adds aux_coef * (cross-entropy + squared error) to the loss and
logs aux/* and behavior/mode_* (argmax of the shoot head against the aim label, of the move head
against the moves that shorten d_fire). Frames with a 'hit' field record each step's hit reward;
train/hit_advantage_gap is the mean advantage of those steps minus the rest, in advantage std.

Online distillation (user plan 2026-09-28; policy distillation, Rusu et al. 2016; kickstarting, Schmitt
et al. 2018): with a teacher (a frozen policy, load_frozen_policy) every training sample adds
distill_coef * T^2 * sum over the heads of KL(teacher || student), both distributions from logits under the
sample's stored action masks and softened by distill_temperature T. The PPO policy-gradient term is
weighted by rl_coef (0: pure distillation); the value loss, the entropy term and the auxiliary aim loss are
unchanged. distill_coef moves linearly to distill_coef_end over the learn() budget (progress_remaining).
The samples are fresh rollouts of the student (on-policy distillation); for the first teacher_acts_updates
updates the teacher collects them instead (its states, the student learns their targets). The log adds
distill/kl, distill/coef and distill/agree_<head> (argmax agreement of the two policies).
"""
from concurrent.futures import as_completed
from types import SimpleNamespace
import copy
import json
import math
import sys
import threading
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import explained_variance
from .gpu_buffer import GpuHistoryRolloutBuffer
from .gpu_env import GpuFrameVecEnv,decode_frame,metadata,metadata_array

HEAD_NAMES={(45,2,2):('joint','bomb','item'),(9,5,2,2):('move','shoot','bomb','item')}


class LearningRateSchedule:
    """SB3 learning-rate schedule of progress_remaining (1 -> 0 over the learn() budget): constant, or
    linear / cosine from initial down to final (C36). A class, not a closure, so checkpoints pickle it."""
    def __init__(self,kind,initial,final):
        if kind not in ('constant','linear','cosine'):raise ValueError(f'unknown learning-rate schedule {kind}')
        self.kind,self.initial,self.final=kind,float(initial),float(final)
    def __call__(self,progress_remaining):
        done=min(1.0,max(0.0,1.0-float(progress_remaining)))
        if self.kind=='constant':return self.initial
        if self.kind=='linear':return self.initial+(self.final-self.initial)*done
        return self.final+(self.initial-self.final)*0.5*(1.0+math.cos(math.pi*done))
    def __repr__(self):
        return f'LearningRateSchedule({self.kind!r}, {self.initial:g}, {self.final:g})'


class GroupLearningRate:
    """C44 (design 4.8): a parameter group's learning rate, of progress_remaining and the completed updates: warmup_updates
    linear steps from warmup_start up to the schedule, then the schedule (LearningRateSchedule). A class, so it pickles."""
    def __init__(self,kind,initial,final,warmup_updates=0,warmup_start=None):
        self.schedule=LearningRateSchedule(kind,initial,final)
        self.warmup_updates=int(warmup_updates);self.warmup_start=float(warmup_start if warmup_start is not None else final)
    def __call__(self,progress_remaining,updates=0):
        lr=self.schedule(progress_remaining)
        if updates<self.warmup_updates:
            lr=self.warmup_start+(lr-self.warmup_start)*updates/self.warmup_updates
        return lr
    def __repr__(self):
        return f'GroupLearningRate({self.schedule!r}, warmup {self.warmup_updates} from {self.warmup_start:g})'


def full_kl(prox_logits,new_logits,nvec):
    """Per sample, the sum over the heads of KL(pi_proximal || pi_new) from their (masked) logits (C32)."""
    kl=0
    for p,q in zip(torch.split(prox_logits,nvec,-1),torch.split(new_logits,nvec,-1)):
        p,q=torch.log_softmax(p,-1),torch.log_softmax(q,-1)
        kl=kl+(p.exp()*(p-q)).sum(-1)
    return kl


def load_frozen_policy(path,device,observation_space=None,action_space=None):
    """The policy of a saved model.zip, without its algorithm (no rollout buffer), frozen for inference (a
    distillation teacher). The spaces, if given, must equal the saved ones."""
    from stable_baselines3.common.save_util import load_from_zip_file
    data,params,_=load_from_zip_file(path,device=device)
    for name,space in (('observation_space',observation_space),('action_space',action_space)):
        if space is not None and data[name]!=space:raise ValueError(f'teacher {name} differs from the environment')
    policy=data['policy_class'](data['observation_space'],data['action_space'],lambda _:0.0,**data['policy_kwargs'])
    policy.load_state_dict(params['policy']);policy.to(device);policy.set_training_mode(False)
    for p in policy.parameters():p.requires_grad_(False)
    return policy


ACTIVE_READY=19   # transformer_obs.PLAYER_FIELDS index of active_ready


def action_masks(nvec,bombs_ok,move_block=None,item_ok=None):
    """Masks of MultiDiscrete heads ending in (bomb, item): the bomb only with bombs, never the item (item_ok given: the
    item only with a charged active item, as the environment's own masks; goal line, user decision 2026-09-30);
    move_block (n, 9) bool (C30, --block-moves): the moves of the first head the terrain stops dead."""
    masks=torch.ones((len(bombs_ok),sum(nvec)),dtype=torch.bool,device=bombs_ok.device)
    bomb=sum(nvec[:-2]);item=bomb+nvec[-2]
    masks[:,bomb+1]=bombs_ok;masks[:,item+1]=item_ok if item_ok is not None else False
    if move_block is not None:masks[:,:nvec[0]]&=~move_block
    return masks


class FrameSampler:
    def __init__(self,model):
        self.model=model;self.env=model.env;self.buffer=model.rollout_buffer
        self.features=torch.zeros((*self.buffer.lengths.shape,self.encoder.features_dim),device=model.device)
        self.starts=torch.ones(self.env.num_envs,dtype=torch.bool,device=model.device)
        self.workers=[torch.arange(c.start,c.stop,device=model.device) for c in self.env.chunks]
        self.generation=-1;self.version=-1;self.steps=0
        self.upload_bytes=0;self.encoded_frames=0;self.prefix_rebuilds=0
        self.nvec=[int(n) for n in self.env.action_space.nvec]
        # C30 (--block-moves): per environment, the moves its newest frame's terrain stops dead (the frames'
        # move_block field, set by collect_rollouts); masked in every action and stored with the masks.
        self.block_moves=bool(getattr(model,'block_moves',False))
        self.move_block=torch.zeros((self.env.num_envs,self.nvec[0]),dtype=torch.bool,device=model.device)

    # The weights that collect: the actor copy while training runs asynchronously.
    @property
    def policy(self):return self.model.sampling_policy
    @property
    def encoder(self):return self.policy.features_extractor

    def encode(self,position,workers,raw):
        for k,v in raw.items():self.buffer.frames[k][position,workers]=v
        self.features[position,workers]=self.encoder.encode_frames({k:v[:,None] for k,v in raw.items()})[:,0]
        self.encoded_frames+=len(workers)

    def begin(self):
        b=self.buffer
        if self.features.shape[-1]!=self.encoder.features_dim:
            # Another sampling policy (distillation: teacher <-> student) with another width: re-encode the prefix.
            self.features=torch.zeros((*b.lengths.shape,self.encoder.features_dim),device=self.model.device);self.version=None
        version=(self.model.policy_version,tuple(p._version for p in self.policy.parameters()))
        # Join all previously queued writes before compacting the shared prefix.
        stream=torch.cuda.current_stream(self.model.device)
        for c in self.env.chunks:
            if self.generation>=0:stream.wait_event(c.ready)
        if self.generation!=self.env.generation:
            b.reset();b.frame_pos=b.history-1;b.lengths.zero_();self.features.zero_();self.starts.fill_(True)
            for c,ids in zip(self.env.chunks,self.workers):
                slot=c.slots[0]
                slot.device.copy_(slot.host,non_blocking=True)
                self.encode(b.frame_pos,ids,decode_frame(slot.device,self.env.observation_space,self.env.frame_dtype))
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
        policy=self.policy
        return policy.value_net(policy.mlp_extractor.forward_critic(self.latent(position,ids))).flatten()

    def action(self,position,ids,deterministic=False):
        policy=self.policy
        masks=action_masks(self.nvec,self.buffer.frames['player'][position,ids,9]>0,
                           self.move_block[ids] if self.block_moves else None,
                           self.buffer.frames['player'][position,ids,ACTIVE_READY]>0 if self.model.item_available else None)
        pi,vf=policy.mlp_extractor(self.latent(position,ids))
        distribution=policy._get_action_dist_from_latent(pi)
        distribution.apply_masking(masks)
        actions=distribution.get_actions(deterministic=deterministic)
        return actions,policy.value_net(vf).flatten(),distribution.log_prob(actions),masks


class GpuMaskablePPO(MaskablePPO):
    # Collection sampler factory, called with the model; None = FrameSampler (graph_sampler.py
    # provides a CUDA-graph variant for the AB+ learner).
    sampler_class=None
    # Overlap each update with the next rollout (module docstring). While both run, the
    # collection thread waits at most this long for the GIL (CPython's default is 5 ms).
    async_training=False
    async_switch_interval=0.0005
    lag_weight_max=2.0
    # C30: mask the moves the terrain stops dead (the frames' move_block field; FrameSampler.move_block).
    block_moves=False
    # C32 diagnostic (--kl-probe): per update, the samples with the largest pi_new / pi_proximal, appended to
    # kl_probe_path as one JSON line (host syncs: only for probe runs); greedy_actors only labels them.
    kl_probe=False
    kl_probe_path=None
    greedy_actors=0
    # Online distillation (module docstring): the frozen teacher policy and the loss weights.
    teacher=None
    distill_coef=1.0
    distill_coef_end=None
    distill_temperature=1.0
    rl_coef=1.0
    teacher_acts_updates=0
    # Goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md, user decisions 2026-09-30), active when the frames carry
    # 'goal'. item_available: the item head by the active item's charge (else never). Advantages are normalised per task
    # (COMBAT / GOTO). loss_tasks 'goto' (stage 1, the shared weights frozen): the PPO, value and entropy terms average
    # over the GOTO samples only. The auxiliary aim loss skips GOTO samples. preserve_teacher (a frozen copy of the
    # migrated policy, equal to C39 on COMBAT frames): preserve_coef x the mean over the minibatch's single-room samples
    # (frames' source 0) of KL(teacher || policy); with preserve_target the coefficient adapts after each update (x1.5
    # above 1.5 target, /1.5 below target / 1.5, within preserve_bounds).
    item_available=False
    loss_tasks='all'
    preserve_teacher=None
    preserve_coef=0.0
    preserve_target=None
    preserve_bounds=(0.1,30.0)
    # C44 (design 4.8): {param group name: GroupLearningRate} for an optimizer with named groups (GeometryPolicy(lr_groups=
    # True): 'shared', 'goal'); a group without an entry follows lr_schedule. None: SB3's one rate for every group.
    lr_groups=None
    # C45 (goal-hp3, user decisions 2026-10-01): imitation of the scripted A* expert on the move head. The frames' expert_move
    # (transformer_obs: GOTO frames always, COMBAT frames once stuck against the terrain) is a target distribution; the loss
    # adds expert_coef x (1 - done / expert_until, at least 0) x the mean over the minibatch's labelled samples of the
    # cross-entropy -sum target log pi_move (done: the run's completed fraction).
    expert_coef=0.0
    expert_until=0.5
    # A15 (diagnosis of C45's imitation): imitation_only keeps the expert cross-entropy alone (a supervised fit on the
    # policy's own states, every other loss term dropped); grad_probe logs, on the first micro-batch of each update, each
    # loss term's gradient norm per optimizer group (one backward pass each), and every step's total norm before clipping.
    imitation_only=False
    grad_probe=False

    def _probe_gradients(self,terms,n):
        '''grad_probe: the gradient norm of each loss term (a tensor) over each optimizer group's parameters.'''
        groups=[(g.get('name',f'group{i}'),[p for p in g['params'] if p.requires_grad])
                for i,g in enumerate(self.policy.optimizer.param_groups)]
        params=[p for _,ps in groups for p in ps]
        for name,term in terms.items():
            if not torch.is_tensor(term) or not term.requires_grad:continue
            grads=torch.autograd.grad(term/n,params,retain_graph=True,allow_unused=True)
            i=0
            for gname,ps in groups:
                sq=sum(float((g*g).sum()) for g in grads[i:i+len(ps)] if g is not None)
                self.logger.record(f'probe/grad_{name}_{gname}',sq**0.5)
                i+=len(ps)

    def __init__(self,*args,**kwargs):
        # TF32 cuDNN changes CNN rounding with batch shape (one frame vs H
        # frames). Keep collection and PPO likelihood evaluation in strict FP32.
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        kwargs.setdefault('rollout_buffer_class',GpuHistoryRolloutBuffer)
        kwargs.setdefault('batch_size',32)
        self.policy_version=0;self._gpu_sampler=None;self._collecting=False
        self.session=None;self.sampling_deterministic=False
        self.actor=None;self.train_buffer=None;self._train_stream=None
        # None keeps SB3's per-window minibatches; an int enables frame-deduplicated segment
        # minibatches with gradient accumulation (see _train_segments).
        self.micro_batch_size=None;self.segment_length=32
        # combat-v5: per-head entropy weights (None: ent_coef on the summed entropy) and the
        # auxiliary geometry loss weight (used when the policy has aux_outputs).
        self.head_ent_coefs=None;self.aux_coef=0.0
        # weight of the aux head's d_fire regression next to its aim cross-entropy (C21: 0 lets the aim learn)
        self.aux_distance_coef=1.0
        # aux_balance: aligned and unaligned samples weigh half each in the aim cross-entropy (C22: with one
        # small target ~8% of the frames are aligned and the head learns 'none' everywhere)
        self.aux_balance=False
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_gpu_sampler','_collecting','session','sampler_class',
                                                'async_training','actor','train_buffer','_train_stream','teacher',
                                                'preserve_teacher']

    @property
    def sampling_policy(self):
        if self.teacher is not None and self.policy_version<self.teacher_acts_updates:return self.teacher
        return self.actor if self.actor is not None else self.policy

    def current_distill_coef(self):
        end=self.distill_coef if self.distill_coef_end is None else self.distill_coef_end
        return end+(self.distill_coef-end)*float(self._current_progress_remaining)

    def _setup_learn(self,*args,**kwargs):
        result=super()._setup_learn(*args,**kwargs)
        if not isinstance(self.env,GpuFrameVecEnv):raise TypeError('GPU PPO requires GpuFrameVecEnv')
        # The VecEnv reset contract is preserved, but its one-time NumPy window
        # must not remain resident throughout training. State lives in raw GPU frames.
        self._last_obs={}
        return result

    def current_expert_coef(self):
        """The imitation loss's coefficient now: expert_coef, falling linearly to 0 at expert_until of the run."""
        if not self.expert_coef:
            return 0.0
        done=1.0-float(self._current_progress_remaining)
        return float(self.expert_coef)*max(0.0,1.0-done/max(1e-9,float(self.expert_until)))

    def _update_learning_rate(self,optimizers):
        if not self.lr_groups:
            return super()._update_learning_rate(optimizers)
        updates=self._n_updates//max(1,self.n_epochs)
        self.logger.record('train/learning_rate',self.lr_schedule(self._current_progress_remaining))
        for optimizer in (optimizers if isinstance(optimizers,list) else [optimizers]):
            for group in optimizer.param_groups:
                fn=self.lr_groups.get(group.get('name'))
                group['lr']=fn(self._current_progress_remaining,updates) if fn is not None else self.lr_schedule(self._current_progress_remaining)
                if group.get('name'):self.logger.record(f"train/lr_{group['name']}",group['lr'])

    def train(self):
        if self._collecting:raise RuntimeError('Weights are frozen while collecting a rollout')
        try:
            if self.micro_batch_size:self._train_segments()
            else:super().train()
        finally:self.policy_version+=1
        if self.session is not None:self.session.after_update(self)

    def learn(self,total_timesteps,callback=None,log_interval=1,tb_log_name='MaskablePPO',
              reset_num_timesteps=True,use_masking=True,progress_bar=False):
        if not self.async_training:
            return super().learn(total_timesteps,callback,log_interval,tb_log_name,reset_num_timesteps,
                                 use_masking,progress_bar)
        if not self.micro_batch_size:raise ValueError('async_training requires segment training (micro_batch_size)')
        # MaskablePPO.learn, with each train() running while the next rollout is collected.
        iteration=0
        total_timesteps,callback=self._setup_learn(total_timesteps,callback,reset_num_timesteps,tb_log_name,progress_bar)
        callback.on_training_start(locals(),globals())
        if self.actor is None:
            optimizer=self.policy.optimizer;self.policy.optimizer=None  # the actor needs no Adam state
            try:self.actor=copy.deepcopy(self.policy)
            finally:self.policy.optimizer=optimizer
            self.actor.set_training_mode(False)
            for p in self.actor.parameters():p.requires_grad_(False)
            self.train_buffer=self.rollout_buffer_class(self.n_steps,self.observation_space,self.action_space,
                device=self.device,gamma=self.gamma,gae_lambda=self.gae_lambda,n_envs=self.n_envs,
                **self.rollout_buffer_kwargs)
            self._train_stream=torch.cuda.Stream(self.device,priority=0)  # 0 is the lowest priority
        switch=sys.getswitchinterval();sys.setswitchinterval(self.async_switch_interval)
        job=None
        try:
            while self.num_timesteps<total_timesteps:
                start=time.perf_counter()
                continue_training=self.collect_rollouts(self.env,callback,self.rollout_buffer,self.n_steps,use_masking)
                if job is not None:self._finish_update(job,time.perf_counter()-start);job=None
                if not continue_training:break
                iteration+=1
                self._update_current_progress_remaining(self.num_timesteps,total_timesteps)
                if log_interval is not None and iteration%log_interval==0:self.dump_logs(iteration)
                job=self._start_update()
            if job is not None:self._finish_update(job,None);job=None
        finally:
            if job is not None:job.thread.join()
            sys.setswitchinterval(switch)
        callback.on_training_end()
        return self

    def _start_update(self):
        """Actor <- learner, snapshot <- rollout, then train on the snapshot in a thread."""
        with torch.no_grad():
            for a,p in zip(self.actor.parameters(),self.policy.parameters()):a.copy_(p)
            for a,p in zip(self.actor.buffers(),self.policy.buffers()):a.copy_(p)
            self.copy_rollout(self.rollout_buffer,self.train_buffer)
        ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(self.device))
        job=SimpleNamespace(error=None,seconds=0.0)
        def run():
            start=time.perf_counter()
            try:
                with torch.cuda.stream(self._train_stream):
                    self._train_stream.wait_event(ready)
                    try:self._train_segments(self.train_buffer,self.proximal_log_probs(self.train_buffer))
                    finally:self.policy_version+=1
                self._train_stream.synchronize()
            except BaseException as error:job.error=error
            job.seconds=time.perf_counter()-start
        job.thread=threading.Thread(target=run,name='ppo-update',daemon=True);job.thread.start()
        return job

    def _finish_update(self,job,collect_s):
        start=time.perf_counter();job.thread.join()
        if job.error is not None:raise job.error
        self.logger.record('async/update_s',job.seconds)
        self.logger.record('async/join_wait_s',time.perf_counter()-start)
        if collect_s is not None:self.logger.record('async/collect_s',collect_s)
        if self.session is not None:self.session.after_update(self)

    def proximal_log_probs(self,buffer):
        """Log-probabilities of the buffer's actions under the current weights, without gradient."""
        T,N,L=buffer.buffer_size,buffer.n_envs,self.segment_length
        per_worker=T//L;step=self.micro_batch_size//L
        out=torch.empty_like(buffer.log_probs)
        # C32: the proximal distributions themselves, for the exact KL to the updated policy (train/kl_full).
        self._prox_logits=torch.zeros((T,N,int(sum(self.action_space.nvec))),device=self.device)
        if self.kl_probe:
            self._prox_heads=torch.zeros((T,N,len(self.action_space.nvec)),device=self.device)
            self._prox_mode=torch.zeros((T,N,len(self.action_space.nvec)),dtype=torch.long,device=self.device)
        self.policy.set_training_mode(True)  # no dropout: the same function as the training passes
        with torch.no_grad():
            for s in range(0,N*per_worker,step):
                chosen=torch.arange(s,min(s+step,N*per_worker),device=self.device)
                workers=torch.div(chosen,per_worker,rounding_mode='floor');t0=(chosen%per_worker)*L
                _,log_prob,_,t,w,extra=self.evaluate_segments(workers,t0,buffer,True)
                self._prox_logits[t,w]=extra['logits']
                if self.kl_probe:
                    self._prox_heads[t,w]=extra['head_log_prob'];self._prox_mode[t,w]=extra['mode']
                out[t,w]=log_prob
        return out

    @staticmethod
    def copy_rollout(src,dst):
        """Everything segment training reads from a GpuHistoryRolloutBuffer."""
        for k,v in src.frames.items():dst.frames[k].copy_(v)
        for name in ('lengths','actions','rewards','_returns','episode_starts','_values','log_probs','advantages','action_masks','hits'):
            getattr(dst,name).copy_(getattr(src,name))
        dst.frame_pos,dst.pos,dst.full=src.frame_pos,src.pos,src.full

    def evaluate_segments(self,workers,t0,buffer=None,details=False):
        """Values, log-probs and entropies for whole L-step segments of single workers.

        Every stored frame the segment needs is encoded once and shared by the windows that
        contain it. Each sample still sees exactly its own causal window (same frames, times,
        padding and last token), so per-sample outputs equal policy.evaluate_actions on
        buffer.window(); only float summation order differs. details=True adds a dict: per-head
        entropies and argmaxes, and with an auxiliary head its outputs and the samples' labels.
        """
        b=buffer if buffer is not None else self.rollout_buffer
        x=self._segment_inputs(workers,t0,b);obs,steps,owners=x.obs,x.steps,x.owners
        latent=self._segment_latent(self.policy,x)
        pi,vf=self.policy.mlp_extractor(latent)
        dist=self.policy._get_action_dist_from_latent(pi)
        dist.apply_masking(b.action_masks[steps,owners])
        values,log_prob,entropy=self.policy.value_net(vf).flatten(),dist.log_prob(b.actions[steps,owners]),dist.entropy()
        if not details:return values,log_prob,entropy,steps,owners
        extra={'head_entropy':torch.stack([d.entropy() for d in dist.distributions],1),
               'mode':torch.stack([d.probs.argmax(-1) for d in dist.distributions],1),
               # Normalised log-probabilities of every action of every head (masked ones ~ -1e8): train/kl_full.
               'logits':torch.cat([d.logits for d in dist.distributions],-1)}
        if self.kl_probe:   # per-head log-probabilities of the taken actions (the KL outlier probe)
            taken=b.actions[steps,owners].long().unbind(-1)
            extra['head_log_prob']=torch.stack([d.log_prob(a) for d,a in zip(dist.distributions,taken)],1)
        if hasattr(self.policy,'aux_outputs') and 'aim_label' in obs:
            extra['aux']=self.policy.aux_outputs(latent)
            # Each sample's own frame is the last H-1+l row of its segment's rows.
            extra['labels']={k:obs[k][:,b.history-1:].flatten(0,1) for k in ('aim_label','fire_distance','approach')}
        extra['inputs']=x
        return values,log_prob,entropy,steps,owners,extra

    def _segment_inputs(self,workers,t0,b):
        """The raw frames of whole L-step segments of single workers and the causal window of every sample."""
        H,L,dev=b.history,self.segment_length,self.device
        rows=t0[:,None]+torch.arange(L+H-1,device=dev)             # frame rows the segment may need
        q=t0[:,None]+torch.arange(L,device=dev)+H-1                 # frame row of each sample's observation
        lengths=b.lengths[q,workers[:,None]]
        first=q[:,0]-lengths[:,0]+1                                  # windows' union is [first, q[:,-1]]
        obs={k:v[rows,workers[:,None]] for k,v in b.frames.items()}
        obs['history_mask']=(rows>=first[:,None]).to(obs['history_mask'].dtype)
        obs=preprocess_obs(obs,self.observation_space,normalize_images=self.policy.normalize_images)
        j=torch.arange(H,device=dev)
        valid=j<lengths[...,None]
        offset=torch.where(valid,torch.arange(L,device=dev)[:,None]+H-lengths[...,None]+j,0)
        seg=torch.arange(len(workers),device=dev)[:,None,None]
        return SimpleNamespace(obs=obs,valid=valid,offset=offset,seg=seg,steps=(q-(H-1)).flatten(),
                               owners=workers[:,None].expand(-1,L).flatten())

    @staticmethod
    def _segment_latent(policy,x):
        """A policy's features of every sample of _segment_inputs (each frame encoded once)."""
        enc=policy.features_extractor
        frames=enc.encode_frames(x.obs)
        window=frames[x.seg,x.offset].masked_fill(~x.valid[...,None],0).flatten(0,1)
        elapsed=x.obs['time'][x.seg,x.offset].masked_fill(~x.valid,0).flatten(0,1)
        valid=x.valid.flatten(0,1)
        latent=enc.temporal_features(window,elapsed,valid)
        return latent[torch.arange(len(latent),device=latent.device),valid.long().sum(-1)-1]

    def teacher_logits(self,x,b,teacher=None):
        """The teacher's normalised log-probabilities of every action of every head on the samples of
        _segment_inputs, under the samples' stored masks (no gradient)."""
        teacher=self.teacher if teacher is None else teacher
        with torch.no_grad():
            latent=self._segment_latent(teacher,x)
            dist=teacher._get_action_dist_from_latent(teacher.mlp_extractor.forward_actor(latent))
            dist.apply_masking(b.action_masks[x.steps,x.owners])
            return torch.cat([d.logits for d in dist.distributions],-1)

    def _train_segments(self,buffer=None,proximal=None):
        """PPO over frame-deduplicated segment minibatches with gradient accumulation.

        batch_size is the effective minibatch (whole segments, advantages normalised over it);
        micro_batch_size only bounds activation memory. Loss, clipping, entropy/value weights,
        grad clipping, target_kl and log keys follow MaskablePPO.train. proximal (asynchronous
        training, see the module docstring): log-probabilities under the weights the update starts
        from; the ratio, approx_kl and clip_fraction are taken against them and the policy loss of
        each sample is weighted by pi_proximal/pi_behaviour, truncated at lag_weight_max.
        """
        b=buffer if buffer is not None else self.rollout_buffer;policy=self.policy
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
        kl_fulls,outlier_shares=[],[]   # C32: exact KL(pi_proximal || pi_new) and the share of |log ratio| > 5
        continue_training=True;steps=0;loss=0.0
        nvec=tuple(int(n) for n in self.action_space.nvec)
        heads=HEAD_NAMES.get(nvec) or tuple(f'head{i}' for i in range(len(nvec)))
        head_coefs=(torch.as_tensor(self.head_ent_coefs,dtype=torch.float32,device=self.device)
                    if self.head_ent_coefs is not None else None)
        # sums over the samples of: head entropies (heads), aux CE, aux squared error, aux hits,
        # aligned samples, aligned aux hits, aligned mode aim hits, approach samples, approach hits
        diag=None
        grad_norms=[]   # grad_probe: each optimizer step's gradient norm before clipping
        probe=[]   # --kl-probe: the samples with the largest pi_new / pi_proximal of each micro-batch
        distill_coef=self.current_distill_coef() if self.teacher is not None else 0.0
        tau=float(self.distill_temperature)
        # sums over the first epoch's samples: KL(teacher || student), argmax agreement per head
        distill_sums=torch.zeros(1+len(nvec),device=self.device)
        # goal line: each sample's task (True: GOTO) and source (0 single room, 1 chain, 2 GOTO), from its own frame
        goal_line='goal' in b.frames
        if goal_line:
            from .transformer_policy import goal_gate
            H=b.history
            task_goto=goal_gate(b.frames['goal'][H-1:H-1+T])
            source=b.frames['source'][H-1:H-1+T].round().long()
            preserve_sums=torch.zeros(3,device=self.device)   # KL sum, single samples, first-epoch samples
        preserve=goal_line and self.preserve_teacher is not None and self.preserve_coef>0
        expert_now=self.current_expert_coef() if 'expert_move' in b.frames else 0.0
        if expert_now>0:
            expert_frames=b.frames['expert_move'][b.history-1:b.history-1+T]
            # first epoch: CE sum, labelled, GOTO labelled, GOTO samples, COMBAT labelled, COMBAT samples, agreements
            expert_sums=torch.zeros(7,device=self.device)
        for epoch in range(self.n_epochs):
            approx_kl_divs=[]
            order=torch.randperm(segments,device=self.device)
            for s in range(0,segments,seg_batch):
                chosen=order[s:s+seg_batch]
                workers=torch.div(chosen,per_worker,rounding_mode='floor');t0=(chosen%per_worker)*L
                adv=b.advantages[t0[:,None]+torch.arange(L,device=self.device),workers[:,None]]
                mean,std,n=adv.mean(),adv.std(),adv.numel()
                if goal_line:
                    rows_t=t0[:,None]+torch.arange(L,device=self.device)
                    goto_mb=task_goto[rows_t,workers[:,None]]
                    n_goto=goto_mb.sum();n_single=(source[rows_t,workers[:,None]]==0).sum()
                    if expert_now>0:n_expert=(expert_frames[rows_t,workers[:,None]].sum(-1)>0).sum()
                    def task_stats(mask):
                        k=mask.sum()
                        if k<2:return mean,std   # too few for their own statistics: the minibatch's
                        values=adv[mask];return values.mean(),values.std()
                    (mean_g,std_g),(mean_c,std_c)=task_stats(goto_mb),task_stats(~goto_mb)
                policy.optimizer.zero_grad()
                sums=torch.zeros(5,device=self.device);robust=torch.zeros(2,device=self.device)
                for m in range(0,len(chosen),seg_micro):
                    values,log_prob,entropy,t,w,extra=self.evaluate_segments(workers[m:m+seg_micro],t0[m:m+seg_micro],b,True)
                    a=(b.advantages[t,w]-mean)/(std+1e-8)
                    weight=None
                    if goal_line:
                        goto_tw=task_goto[t,w]
                        a=torch.where(goto_tw,(b.advantages[t,w]-mean_g)/(std_g+1e-8),(b.advantages[t,w]-mean_c)/(std_c+1e-8))
                        if self.loss_tasks=='goto':   # stage 1: the losses average over the GOTO samples
                            weight=goto_tw.float()*(n/max(1,int(n_goto)))
                    old=b.log_probs[t,w]
                    anchor=old if proximal is None else proximal[t,w]
                    ratio=torch.exp(log_prob-anchor)
                    pg=-torch.min(a*ratio,a*torch.clamp(ratio,1-clip_range,1+clip_range))
                    if proximal is not None:pg=pg*torch.exp(anchor-old).clamp_max(self.lag_weight_max)
                    if clip_range_vf is not None:
                        values=b._values[t,w]+torch.clamp(values-b._values[t,w],-clip_range_vf,clip_range_vf)
                    vl=(b._returns[t,w]-values)**2
                    el=-entropy
                    if weight is not None:
                        pg,vl,el=pg*weight,vl*weight,el*weight
                        ent=(-(extra['head_entropy']*head_coefs).sum(-1)*weight).sum() if head_coefs is not None else self.ent_coef*el.sum()
                    else:
                        ent=(-(extra['head_entropy']*head_coefs).sum() if head_coefs is not None else self.ent_coef*el.sum())
                    if preserve:
                        single=(source[t,w]==0).float()
                        kl_p=full_kl(self.teacher_logits(extra['inputs'],b,self.preserve_teacher),extra['logits'],nvec)
                        distill_p=self.preserve_coef*(kl_p*single).sum()*(n/max(1,int(n_single)))
                        if epoch==0:
                            with torch.no_grad():preserve_sums+=torch.stack([(kl_p*single).sum(),single.sum(),single.new_tensor(len(single))])
                    else:
                        distill_p=0.0
                    imitation=0.0
                    if expert_now>0:
                        target=expert_frames[t,w];labelled=target.sum(-1)>0
                        target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
                        logp=torch.log_softmax(extra['logits'][:,:target.shape[-1]],-1).clamp_min(-30.0)
                        ce=-(target*logp).sum(-1)*labelled.float()
                        imitation=expert_now*ce.sum()*(n/max(1,int(n_expert)))
                        if epoch==0:
                            with torch.no_grad():
                                goto_tw=task_goto[t,w];best=logp.argmax(-1)
                                agree=(target.gather(1,best[:,None]).squeeze(1)>0)&labelled
                                expert_sums+=torch.stack([ce.sum(),labelled.float().sum(),(labelled&goto_tw).float().sum(),
                                                          goto_tw.float().sum(),(labelled&~goto_tw).float().sum(),
                                                          (~goto_tw).float().sum(),agree.float().sum()])
                    distill=0.0
                    if self.teacher is not None:
                        target=self.teacher_logits(extra['inputs'],b)
                        kl=full_kl(target/tau,extra['logits']/tau,nvec)
                        distill=distill_coef*tau*tau*kl.sum()
                        if epoch==0:
                            with torch.no_grad():
                                agree=[(p.argmax(-1)==q.argmax(-1)).float().sum() for p,q in
                                       zip(torch.split(target,nvec,-1),torch.split(extra['logits'],nvec,-1))]
                                distill_sums+=torch.stack([kl.sum(),*agree])
                    aux=0.0
                    if 'aux' in extra:
                        logits,predicted=extra['aux'];labels=extra['labels']
                        ce=torch.nn.functional.cross_entropy(logits,labels['aim_label'].long(),reduction='none')
                        se=(predicted-labels['fire_distance'])**2
                        if goal_line:   # the aim labels mean nothing on a GOTO sample
                            keep=(~task_goto[t,w]).float();ce,se=ce*keep,se*keep
                        weighted=ce
                        if self.aux_balance:
                            aligned=(labels['aim_label']>0).float();share=aligned.mean().clamp(0.02,0.98)
                            weighted=ce*(aligned/(2*share)+(1-aligned)/(2*(1-share)))
                        aux=self.aux_coef*(weighted.sum()+self.aux_distance_coef*se.sum())
                    if self.grad_probe and epoch==0 and s==0 and m==0:
                        self._probe_gradients(dict(pg=self.rl_coef*pg.sum(),entropy=ent,value=self.vf_coef*vl.sum(),aux=aux,
                                                   preserve=distill_p,imitation=imitation),n)
                    if self.imitation_only:
                        if torch.is_tensor(imitation):(imitation/n).backward()
                    else:
                        ((self.rl_coef*pg.sum()+ent+self.vf_coef*vl.sum()+aux+distill+distill_p+imitation)/n).backward()
                    if epoch==0:
                        with torch.no_grad():
                            row=[extra['head_entropy'].sum(0)]
                            if 'aux' in extra:
                                aim=labels['aim_label'].long();aligned=aim>0;guess=logits.argmax(-1)
                                row.append(torch.stack([ce.sum(),se.sum(),(guess==aim).float().sum(),aligned.float().sum(),
                                                        ((guess==aim)&aligned).float().sum()]))
                                if heads[:2]==('move','shoot'):
                                    mode=extra['mode'];far=labels['fire_distance']>1
                                    closer=labels['approach'].gather(1,mode[:,:1]).squeeze(1)>0
                                    row.append(torch.stack([((mode[:,1]==aim)&aligned).float().sum(),far.float().sum(),
                                                            (closer&far).float().sum()]))
                            row=torch.cat(row)
                            diag=row if diag is None else diag+row
                    with torch.no_grad():
                        lr_=log_prob-anchor
                        sums+=torch.stack([pg.sum(),vl.sum(),el.sum(),((torch.exp(lr_)-1)-lr_).sum(),
                                           ((ratio-1).abs()>clip_range).float().sum()])
                        if self.kl_probe:probe.extend(self._probe_samples(b,lr_,log_prob,anchor,old,t,w,extra,epoch,s))
                        if proximal is not None:
                            robust+=torch.stack([full_kl(self._prox_logits[t,w],extra['logits'],nvec).sum(),
                                                 (lr_.abs()>5).float().sum()])
                pg_loss,value_loss,entropy_loss,approx_kl,clip_fraction=(sums/n).tolist()
                pg_losses.append(pg_loss);value_losses.append(value_loss);entropy_losses.append(entropy_loss)
                approx_kl_divs.append(approx_kl);clip_fractions.append(clip_fraction)
                if proximal is not None:
                    kl_full,outlier_share=(robust/n).tolist();kl_fulls.append(kl_full);outlier_shares.append(outlier_share)
                loss=pg_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                # Same semantics as SB3: KL is measured on this minibatch before its step.
                if self.target_kl is not None and approx_kl>1.5*self.target_kl:
                    continue_training=False
                    if self.verbose>=1:print(f'Early stopping at step {epoch} due to reaching max kl: {approx_kl:.2f}')
                    break
                total_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),self.max_grad_norm)
                if self.grad_probe:grad_norms.append(float(total_norm))
                policy.optimizer.step();steps+=1
            if not continue_training:break
        self._n_updates+=self.n_epochs
        self._log_diagnostics(b,diag,heads)
        if grad_norms:
            self.logger.record('probe/grad_norm',float(np.mean(grad_norms)))
            self.logger.record('probe/grad_clipped_share',float(np.mean([g>self.max_grad_norm for g in grad_norms])))
            self.logger.record('probe/max_grad_norm',float(self.max_grad_norm))
        if expert_now>0:
            ce_sum,lab,lab_goto,goto_n,lab_combat,combat_n,agree=expert_sums.tolist()
            self.logger.record('expert/coef',expert_now)
            self.logger.record('expert/ce',ce_sum/max(1.0,lab))
            self.logger.record('expert/agree',agree/max(1.0,lab))
            self.logger.record('expert/goto_labelled_share',lab_goto/max(1.0,goto_n))
            self.logger.record('expert/combat_stuck_share',lab_combat/max(1.0,combat_n))
        if goal_line:
            self.logger.record('goal/goto_share',float(task_goto.float().mean()))
            self.logger.record('goal/single_share',float((source==0).float().mean()))
            self.logger.record('goal/chain_share',float((source==1).float().mean()))
        if preserve:
            kl_sum,singles,samples=preserve_sums.tolist()
            kl_mean=kl_sum/max(1.0,singles)
            self.logger.record('preserve/kl',kl_mean);self.logger.record('preserve/coef',self.preserve_coef)
            if self.preserve_target:
                low,high=self.preserve_bounds
                if kl_mean>1.5*self.preserve_target:self.preserve_coef=min(high,self.preserve_coef*1.5)
                elif kl_mean<self.preserve_target/1.5:self.preserve_coef=max(low,self.preserve_coef/1.5)
        if self.teacher is not None:
            count=b.buffer_size*b.n_envs;values=(distill_sums/count).tolist()
            self.logger.record('distill/kl',values[0]);self.logger.record('distill/coef',distill_coef)
            for name,value in zip(heads,values[1:]):self.logger.record(f'distill/agree_{name}',value)
            self.logger.record('distill/teacher_collects',int(self.policy_version<self.teacher_acts_updates))
        if self.kl_probe and self.kl_probe_path:
            probe.sort(key=lambda r:-r['log_ratio'])
            with open(self.kl_probe_path,'a',encoding='utf8') as fh:
                fh.write(json.dumps({'update':self._n_updates//max(1,self.n_epochs),'samples':probe[:20]})+'\n')
        explained_var=explained_variance(b.values.flatten(),b.returns.flatten())
        self.logger.record('train/entropy_loss',np.mean(entropy_losses))
        self.logger.record('train/policy_gradient_loss',np.mean(pg_losses))
        self.logger.record('train/value_loss',np.mean(value_losses))
        self.logger.record('train/approx_kl',np.mean(approx_kl_divs))
        if kl_fulls:
            # approx_kl (the sampled action's (r - 1) - log r) is dominated by a few near-deterministic states whose
            # choice flips between updates (C32); these two measure the whole distributions.
            self.logger.record('train/kl_full',np.mean(kl_fulls))
            self.logger.record('train/ratio_outlier_share',np.mean(outlier_shares))
        self.logger.record('train/clip_fraction',np.mean(clip_fractions))
        self.logger.record('train/loss',loss)
        self.logger.record('train/explained_variance',explained_var)
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip_range)
        if clip_range_vf is not None:self.logger.record('train/clip_range_vf',clip_range_vf)
        self.logger.record('train/optimizer_steps',steps)
        if proximal is not None:
            lag=proximal-b.log_probs;weight=torch.exp(lag)  # pi_proximal/pi_behaviour
            self.logger.record('train/lag_kl',float(((weight-1)-lag).mean()))
            self.logger.record('train/lag_weight_truncated',float((weight>self.lag_weight_max).float().mean()))

    def _probe_samples(self,b,lr,log_prob,anchor,old,t,w,extra,epoch,batch,k=3):
        """--kl-probe: the k samples of a micro-batch with the largest log(pi_new / pi_proximal), with per-head
        log-probabilities, whether each taken action is allowed by its stored mask, and the frame's context."""
        nvec=[int(n) for n in self.action_space.nvec];offsets=np.cumsum([0]+nvec[:-1]).tolist()
        out=[]
        for i in torch.topk(lr,min(k,len(lr))).indices.tolist():
            ti,wi=int(t[i]),int(w[i]);action=b.actions[ti,wi].long().tolist();mask=b.action_masks[ti,wi]
            out.append(dict(log_ratio=float(lr[i]),epoch=epoch,batch=batch,t=ti,w=wi,action=action,
                            allowed=[bool(mask[o+a]) for o,a in zip(offsets,action)],
                            new=[round(float(x),4) for x in extra['head_log_prob'][i]],
                            prox=[round(float(x),4) for x in self._prox_heads[ti,wi]],
                            joint_new=float(log_prob[i]),joint_prox=float(anchor[i]),behaviour=float(old[i]),
                            advantage=float(b.advantages[ti,wi]),start=bool(b.episode_starts[ti,wi]),
                            entities=(float(b.frames['entity_mask'][ti+b.history-1,wi].sum())
                                      if 'entity_mask' in b.frames else None),
                            greedy_actor=wi<int(self.greedy_actors),
                            prox_mode=self._prox_mode[ti,wi].tolist(),new_mode=extra['mode'][i].tolist(),
                            **{k:(b.frames[k][ti+b.history-1,wi].tolist() if b.frames[k][ti+b.history-1,wi].dim() else
                                  float(b.frames[k][ti+b.history-1,wi]))
                               for k in ('aim_label','fire_distance','approach') if k in b.frames}))
        return out

    def _log_diagnostics(self,b,diag,heads):
        """Per-head entropy, auxiliary head, mode aiming/approach (first epoch) and hit advantage gap."""
        count=b.buffer_size*b.n_envs
        if diag is not None:
            diag=diag.tolist();k=len(heads)
            for i,name in enumerate(heads):self.logger.record(f'train/entropy_{name}',diag[i]/count)
            if len(diag)>k:
                ce,se,hit,aligned,aligned_hit=diag[k:k+5]
                self.logger.record('aux/aim_ce',ce/count);self.logger.record('aux/distance_mse',se/count)
                self.logger.record('aux/aim_accuracy',hit/count);self.logger.record('aux/aligned_share',aligned/count)
                if aligned:self.logger.record('aux/aim_accuracy_aligned',aligned_hit/aligned)
                if len(diag)>k+5:
                    mode_aim,far,closer=diag[k+5:k+8]
                    if aligned:self.logger.record('behavior/mode_aim_rate',mode_aim/aligned)
                    if far:self.logger.record('behavior/mode_approach_rate',closer/far)
                    self.logger.record('behavior/far_share',far/count)
        hit=b.hits>0
        if hit.any() and (~hit).any():
            adv=b.advantages;std=adv.std()+1e-8
            self.logger.record('train/hit_advantage_gap',float((adv[hit].mean()-adv[~hit].mean())/std))
            self.logger.record('train/hit_step_share',float(hit.float().mean()))
        if self.block_moves:   # C30: share of steps whose masks drop at least one move (the terrain blocks it)
            first=int(self.action_space.nvec[0])
            self.logger.record('behavior/move_blocked_share',float((~b.action_masks[...,:first]).any(-1).float().mean()))

    def collect_rollouts(self,env,callback,rollout_buffer,n_rollout_steps,use_masking=True):
        if not use_masking:raise ValueError('Isaac requires action masking')
        if n_rollout_steps!=rollout_buffer.buffer_size:raise ValueError('Collect the configured rollout length')
        sampling=self.sampling_policy
        sampling.set_training_mode(False)
        self._collecting=True
        versions=tuple(p._version for p in sampling.parameters())
        try:
            with torch.no_grad():
                if self._gpu_sampler is None:self._gpu_sampler=(self.sampler_class or FrameSampler)(self)
                s=self._gpu_sampler;b=rollout_buffer;s.begin()
                callback.on_rollout_start()
                dones=self._last_episode_starts
                for t in range(n_rollout_steps):
                    if versions!=tuple(p._version for p in sampling.parameters()):
                        raise RuntimeError('Policy weights changed during frozen collection')
                    old=b.frame_pos;new=old+1;pending={}
                    for c,ids in zip(env.chunks,s.workers):
                        slot=c.slots[s.steps%2]
                        with torch.cuda.stream(c.compute_stream):
                            actions,values,log_prob,masks=s.action(old,ids,self.sampling_deterministic)
                            starts=s.starts[ids]
                            slot.actions.copy_(env.transfer_actions(actions),non_blocking=True)
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
                            raw=decode_frame(slot.device,env.observation_space,env.frame_dtype)
                            b.lengths[new,ids]=(b.lengths[old,ids]+1).clamp_max(b.history)
                            s.encode(new,ids,raw)
                            if s.block_moves:s.move_block[ids]=metadata_array(slot.device,'move_block',env.frame_dtype)>0.5
                            reward=metadata(slot.device,'reward',env.frame_dtype).clone()
                            hits=metadata(slot.device,'hit',env.frame_dtype).clone() if 'hit' in env.frame_dtype.names else None
                            # combat-hitrate-fire: the part of this step's reward that belongs to the steps
                            # its tears were fired in; moved there before GAE (GpuHistoryRolloutBuffer).
                            credit=(metadata_array(slot.device,'credit',env.frame_dtype)
                                    if 'credit' in env.frame_dtype.names else None)
                            if len(terminal):
                                local=torch.as_tensor(terminal,device=self.device);done_ids=ids[local]
                                terminal_obs=b.window(new,done_ids)
                                for j,i in enumerate(terminal):ci[i]['terminal_observation']={k:v[j] for k,v in terminal_obs.items()}
                                timeout=[i for i in terminal if ci[i]['TimeLimit.truncated']]
                                if timeout:
                                    ti=torch.as_tensor(timeout,device=self.device)
                                    reward[ti]+=self.gamma*s.values(new,ids[ti])
                                reset=decode_frame(slot.reset_device[:len(terminal)],env.observation_space,env.frame_dtype)
                                s.encode(new,done_ids,reset);b.lengths[new,done_ids]=1
                                if s.block_moves:
                                    s.move_block[done_ids]=metadata_array(slot.reset_device[:len(terminal)],'move_block',
                                                                          env.frame_dtype)>0.5
                            b.add_chunk(t,ids,actions,values,log_prob,masks,reward,starts,hits)
                            if credit is not None:b.add_credit(t,ids,credit)
                            s.starts[ids]=metadata(slot.device,'done',env.frame_dtype).bool()
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
                if 'credit' in env.frame_dtype.names:
                    moved,kept=b.apply_credits()   # scaled reward per rollout
                    self.logger.record('train/credit_moved',float(moved))
                    self.logger.record('train/credit_kept',float(kept))
                b.compute_returns_and_advantage(last_values,dones)
                if versions!=tuple(p._version for p in sampling.parameters()):
                    raise RuntimeError('Policy weights changed during frozen collection')
                callback.on_rollout_end()
                return True
        finally:self._collecting=False
