"""SB3 VecEnv over a persistent Rust/Rayon batch. No game, rendering or IPC.

Uses the original engine's Transformer observation space, 15 Hz actions and
reward weights. Terminal windows are saved before autoreset; timeout bootstraps.
"""
import ctypes as C
import json
from pathlib import Path
import sys
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from .sim_obs import rust_visible_observation
from .transformer_obs import VisibleHistory,HISTORY,ENTITY_CAPACITY


class RustBatch:
    def __init__(self, n, seed=0, threads=1, library=None):
        if n<1 or threads<1:raise ValueError('n and threads must be positive')
        path=Path(library) if library else Path(__file__).resolve().parents[3]/'sim/target/release'/('isaac_sim.dll' if sys.platform=='win32' else 'libisaac_sim.so')
        self.lib=C.CDLL(str(path));self.n=n
        signatures={'new':([C.c_size_t,C.c_uint32,C.c_size_t],C.c_void_p),
                    'reset':([C.c_void_p,C.c_size_t,C.c_uint32],None),
                    'step':([C.c_void_p,C.POINTER(C.c_int32)],None),
                    'state':([C.c_void_p],C.c_char_p),'free':([C.c_void_p],None)}
        for name,(args,result) in signatures.items():
            f=getattr(self.lib,'isaac_batch_'+name);f.argtypes=args;f.restype=result
        self.handle=self.lib.isaac_batch_new(n,seed,threads)

    def states(self):return json.loads(self.lib.isaac_batch_state(self.handle))

    def reset(self,index,seed):
        if not 0<=index<self.n:raise IndexError(index)
        self.lib.isaac_batch_reset(self.handle,index,seed)

    def step(self,actions):
        a=np.ascontiguousarray(actions,dtype=np.int32)
        if a.shape!=(self.n,3) or np.any(a<0) or np.any(a[:,0]>=45) or np.any(a[:,1]>1) or np.any(a[:,2]!=0):
            raise ValueError('Expected [N,3]: joint 0..44, bomb 0..1, item 0 (no item equipped)')
        self.lib.isaac_batch_step(self.handle,a.ctypes.data_as(C.POINTER(C.c_int32)))

    def close(self):
        if self.handle:self.lib.isaac_batch_free(self.handle);self.handle=None


class SimVecEnv(VecEnv):
    def __init__(self,n=8,seed=1,threads=4,history=HISTORY,capacity=ENTITY_CAPACITY,library=None):
        self.batch=RustBatch(n,seed,threads,library)
        self.histories=[VisibleHistory(history,capacity) for _ in range(n)]
        self.base_seed=seed;self.episodes=np.zeros(n,dtype=np.int64)
        self.raw=[None]*n;self.actions=None;self.returns=np.zeros(n);self.lengths=np.zeros(n,dtype=np.int64)
        super().__init__(n,self.histories[0].space,spaces.MultiDiscrete([45,2,2]))

    def reset(self):
        for i,h in enumerate(self.histories):
            seed=self._seeds[i] if self._seeds[i] is not None else self.base_seed+i
            self.batch.reset(i,int(seed));h.clear()
        self._reset_seeds();self.episodes[:]=0;self.returns[:]=0;self.lengths[:]=0
        rows=self.batch.states();return self._encode(rows)

    def _encode(self,rows):
        obs=[]
        for i,r in enumerate(rows):
            self.raw[i]=rust_visible_observation(r['state'])
            obs.append(self.histories[i].append(self.raw[i]))
        return {k:np.stack([o[k] for o in obs]) for k in obs[0]}

    def action_masks(self):
        return np.asarray([[True]*45+[True,r['players'][0]['bombs']>0]+[True,False] for r in self.raw],bool)

    def step_async(self,actions):
        self.actions=np.asarray(actions).copy()
        for h,a in zip(self.histories,self.actions):h.previous_action[:]=*a,1
        self.batch.step(self.actions)

    def step_wait(self):
        rows=self.batch.states();observations=self._encode(rows)
        rewards=np.asarray([r['reward'] for r in rows],np.float32);dones=np.asarray([r['done'] for r in rows],bool)
        self.returns+=rewards;self.lengths+=1
        infos=[dict(outcome=r['outcome'],elapsed_frames=r['state']['frame'],layout=r['state']['layout'],**{'TimeLimit.truncated':r['truncated']}) for r in rows]
        for i in np.flatnonzero(dones):
            infos[i]['terminal_observation']={k:v[i].copy() for k,v in observations.items()}
            infos[i]['episode']={'r':float(self.returns[i]),'l':int(self.lengths[i])}
            self.returns[i]=0;self.lengths[i]=0;self.episodes[i]+=1
            self.batch.reset(int(i),int(self.base_seed+i+self.num_envs*self.episodes[i]))
            self.histories[i].clear()
        if dones.any():
            fresh=self.batch.states()
            for i in np.flatnonzero(dones):
                self.raw[i]=rust_visible_observation(fresh[i]['state'])
                encoded=self.histories[i].append(self.raw[i])
                for k in observations:observations[k][i]=encoded[k]
        return observations,rewards,dones,infos

    def close(self):self.batch.close()
    def get_attr(self,name,indices=None):
        ids=self._get_indices(indices)
        if name=='render_mode':return [None for _ in ids]
        return [getattr(self,name) for _ in ids]
    def set_attr(self,name,value,indices=None):raise NotImplementedError('Set sim parameters through constructor/reset')
    def env_method(self,method_name,*args,indices=None,**kwargs):
        if method_name=='action_masks':return [self.action_masks()[i] for i in self._get_indices(indices)]
        raise AttributeError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
