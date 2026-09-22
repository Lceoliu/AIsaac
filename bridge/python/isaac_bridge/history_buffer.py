"""Lossless raw-frame storage for SB3 MaskablePPO finite-history observations.

One frame per transition, plus H-1 prefix frames per worker, instead of H
duplicated frames per transition. Minibatches reconstruct their own windows;
episode lengths prevent cross-episode leakage. No learned feature caching.
"""
import numpy as np
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples

STORAGE_DTYPES={'player_anim':np.uint8,'entity_anim':np.uint8,'active_kind':np.uint16,
                'entity_kind':np.uint16,'entity_mask':np.bool_,'terrain':np.bool_,
                'history_mask':np.bool_}


class HistoryRolloutBuffer(MaskableDictRolloutBuffer):
    def reset(self):
        self.history=self.obs_shape['history_mask'][0]
        self.mask_dims=int(sum(self.action_space.nvec))
        if not hasattr(self,'frames'):
            self.frames={k:np.empty((self.buffer_size+self.history-1,self.n_envs,*shape[1:]),
                dtype=STORAGE_DTYPES.get(k,self.observation_space[k].dtype)) for k,shape in self.obs_shape.items()}
            self.lengths=np.empty((self.buffer_size,self.n_envs),np.int32)
        self.actions=np.zeros((self.buffer_size,self.n_envs,self.action_dim),self.action_space.dtype)
        for name in ('rewards','returns','episode_starts','values','log_probs','advantages'):
            setattr(self,name,np.zeros((self.buffer_size,self.n_envs),np.float32))
        self.action_masks=np.ones((self.buffer_size,self.n_envs,self.mask_dims),np.float32)
        self.pos=0;self.full=False;self.generator_ready=False

    @property
    def observation_storage_bytes(self):
        return sum(a.nbytes for a in self.frames.values())+self.lengths.nbytes

    def add(self,obs,action,reward,episode_start,value,log_prob,action_masks=None):
        lengths=obs['history_mask'].sum(axis=1).astype(np.int32)
        workers=np.arange(self.n_envs)
        self.lengths[self.pos]=lengths
        for key,data in self.frames.items():
            if self.pos==0:
                # Align the initial prefix immediately before this rollout's first frame.
                data[:self.history-1]=0
                for e,length in enumerate(lengths):
                    data[self.history-length:self.history-1,e]=obs[key][e,:length-1]
            data[self.history-1+self.pos]=obs[key][workers,lengths-1]
        self.actions[self.pos]=action.reshape(self.n_envs,self.action_dim)
        self.rewards[self.pos]=reward;self.episode_starts[self.pos]=episode_start
        self.values[self.pos]=value.detach().cpu().numpy().flatten()
        self.log_probs[self.pos]=log_prob.detach().cpu().numpy().flatten()
        if action_masks is not None:self.action_masks[self.pos]=action_masks.reshape(self.n_envs,self.mask_dims)
        self.pos+=1;self.full=self.pos==self.buffer_size

    def get(self,batch_size=None):
        assert self.full
        indices=np.random.permutation(self.buffer_size*self.n_envs)
        size=batch_size or len(indices)
        for start in range(0,len(indices),size):yield self._get_samples(indices[start:start+size])

    def _get_samples(self,batch_inds,env=None):
        # SB3's env-major flatten order, without flattening/copying the frame store.
        worker,t=divmod(np.asarray(batch_inds),self.buffer_size)
        length=self.lengths[t,worker]
        positions=np.arange(self.history)[None,:]
        valid=positions<length[:,None]
        times=np.where(valid,self.history-1+t[:,None]-length[:,None]+1+positions,0)
        observations={}
        for k,frames in self.frames.items():
            values=frames[times,worker[:,None]].astype(self.observation_space[k].dtype,copy=False)
            values[~valid]=0
            observations[k]=self.to_torch(values)
        return MaskableDictRolloutBufferSamples(observations=observations,
            actions=self.to_torch(self.actions[t,worker]),old_values=self.to_torch(self.values[t,worker]),
            old_log_prob=self.to_torch(self.log_probs[t,worker]),advantages=self.to_torch(self.advantages[t,worker]),
            returns=self.to_torch(self.returns[t,worker]),action_masks=self.to_torch(self.action_masks[t,worker]))
