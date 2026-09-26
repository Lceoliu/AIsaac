"""GPU-resident raw-frame PPO storage. No learned features enter training batches."""
import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer,MaskableDictRolloutBufferSamples


class GpuHistoryRolloutBuffer(MaskableDictRolloutBuffer):
    def reset(self):
        if self.device.type!='cuda':raise ValueError('GPU rollout requires a CUDA device')
        if not hasattr(self,'frames'):
            self.history=self.obs_shape['history_mask'][0]
            self.frames={k:torch.zeros((self.buffer_size+self.history,self.n_envs,*s[1:]),
                dtype=torch.from_numpy(np.empty((),self.observation_space[k].dtype)).dtype,device=self.device)
                for k,s in self.obs_shape.items()}
            self.lengths=torch.zeros((self.buffer_size+self.history,self.n_envs),dtype=torch.long,device=self.device)
            self.frame_pos=self.history-1
            self.mask_dims=int(sum(self.action_space.nvec))
            self.actions=torch.empty((self.buffer_size,self.n_envs,self.action_dim),dtype=torch.long,device=self.device)
            for name in ('rewards','_returns','episode_starts','_values','log_probs','advantages'):
                setattr(self,name,torch.empty((self.buffer_size,self.n_envs),device=self.device))
            # Each step's hit reward when the frames carry one (combat-v5 diagnostics), else 0.
            self.hits=torch.zeros((self.buffer_size,self.n_envs),device=self.device)
            self.action_masks=torch.empty((self.buffer_size,self.n_envs,self.mask_dims),dtype=torch.bool,device=self.device)
        self.pos=0;self.full=False;self.generator_ready=False

    # SB3's explained_variance logger consumes NumPy. This is a once-per-update
    # diagnostic copy; the authoritative trajectory and minibatches stay on GPU.
    @property
    def values(self):return self._values.detach().cpu().numpy()
    @property
    def returns(self):return self._returns.detach().cpu().numpy()
    @property
    def observation_storage_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.frames.values())+self.lengths.numel()*8

    def retain_prefix(self):
        start=self.frame_pos-self.history+1
        for v in self.frames.values():v[:self.history].copy_(v[start:self.frame_pos+1].clone())
        self.lengths[:self.history].copy_(self.lengths[start:self.frame_pos+1].clone())
        self.frame_pos=self.history-1
        self.reset()

    def indices(self,position,workers):
        workers=torch.as_tensor(workers,dtype=torch.long,device=self.device)
        length=self.lengths[position,workers]
        j=torch.arange(self.history,device=self.device)[None,:]
        valid=j<length[:,None]
        positions=torch.as_tensor(position,device=self.device)
        if positions.ndim:positions=positions[:,None]
        times=torch.where(valid,positions-length[:,None]+1+j,0)
        return workers,times,valid

    def window(self,position,workers):
        workers,times,valid=self.indices(position,workers)
        result={}
        for k,frames in self.frames.items():
            value=frames[times,workers[:,None]]
            value.masked_fill_(~valid.reshape(*valid.shape,*([1]*(value.ndim-2))),0)
            result[k]=value
        return result

    def add_chunk(self,t,workers,actions,values,log_prob,masks,rewards,starts,hits=None):
        self.actions[t,workers]=actions
        if hits is not None:self.hits[t,workers]=hits
        self._values[t,workers]=values.flatten()
        self.log_probs[t,workers]=log_prob
        self.action_masks[t,workers]=masks
        self.rewards[t,workers]=rewards
        self.episode_starts[t,workers]=starts.float()

    def compute_returns_and_advantage(self,last_values,dones):
        last=torch.zeros(self.n_envs,device=self.device)
        dones=torch.as_tensor(dones,device=self.device)
        for step in reversed(range(self.buffer_size)):
            if step==self.buffer_size-1:
                nonterminal=1.-dones.float();next_values=last_values.flatten()
            else:
                nonterminal=1.-self.episode_starts[step+1];next_values=self._values[step+1]
            delta=self.rewards[step]+self.gamma*next_values*nonterminal-self._values[step]
            last=delta+self.gamma*self.gae_lambda*nonterminal*last
            self.advantages[step]=last
        self._returns.copy_(self.advantages+self._values)

    def get(self,batch_size=None):
        assert self.full
        indices=torch.randperm(self.buffer_size*self.n_envs,device=self.device)
        size=batch_size or len(indices)
        for start in range(0,len(indices),size):yield self._get_samples(indices[start:start+size])

    def _get_samples(self,batch_inds,env=None):
        indices=torch.as_tensor(batch_inds,device=self.device)
        worker=torch.div(indices,self.buffer_size,rounding_mode='floor');t=indices%self.buffer_size
        return MaskableDictRolloutBufferSamples(self.window(t+self.history-1,worker),
            self.actions[t,worker],self._values[t,worker],self.log_probs[t,worker],
            self.advantages[t,worker],self._returns[t,worker],self.action_masks[t,worker])
