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
from .transformer_obs import VisibleHistory,HISTORY,ENTITY_CAPACITY

# Fixed C record, independently checked against Rust sizeof at load time.
FRAME_DTYPE=np.dtype([
    ('player','f4',(23,)),('player_anim','i4',(32,)),('active_kind','i4',(1,)),
    ('entities','f4',(256,31)),('entity_kind','i4',(256,3)),('entity_anim','i4',(256,32)),
    ('entity_mask','f4',(256,)),('terrain','f4',(7,9,15)),('previous_action','f4',(4,)),
    ('time','f4'),('history_mask','f4'),('reward','f4'),('done','i4'),('truncated','i4'),
    ('outcome','i4'),('elapsed','u4'),('layout','u4'),('count','u4')])


class RustBatch:
    def __init__(self, n, seed=0, threads=1, library=None):
        if n<1 or threads<1:raise ValueError('n and threads must be positive')
        path=Path(library) if library else Path(__file__).resolve().parents[3]/'sim/target/release'/('isaac_sim.dll' if sys.platform=='win32' else 'libisaac_sim.so')
        self.lib=C.CDLL(str(path));self.n=n
        signatures={'new':([C.c_size_t,C.c_uint32,C.c_size_t],C.c_void_p),
                    'reset':([C.c_void_p,C.c_size_t,C.c_uint32],None),
                    'step':([C.c_void_p,C.POINTER(C.c_int32)],None),
                    'step_one':([C.c_void_p,C.c_size_t,C.POINTER(C.c_int32)],None),
                    'state':([C.c_void_p],C.c_char_p),'free':([C.c_void_p],None),
                    'observe':([C.c_void_p,C.c_void_p],C.c_size_t)}
        for name,(args,result) in signatures.items():
            f=getattr(self.lib,'isaac_batch_'+name);f.argtypes=args;f.restype=result
        self.handle=self.lib.isaac_batch_new(n,seed,threads)
        self.lib.isaac_frame_size.restype=C.c_size_t
        if self.lib.isaac_frame_size()!=FRAME_DTYPE.itemsize:raise RuntimeError('Rebuild matching sim library: frame ABI mismatch')
        self.frames=np.empty(n,dtype=FRAME_DTYPE)

    def observe(self,out=None):
        target=self.frames if out is None else out
        overflow=self.lib.isaac_batch_observe(self.handle,target.ctypes.data)
        if overflow:raise ValueError(f'Visible entity overflow: {overflow} > 256; never truncate')
        return target

    def states(self):return json.loads(self.lib.isaac_batch_state(self.handle))

    def reset(self,index,seed):
        if not 0<=index<self.n:raise IndexError(index)
        self.lib.isaac_batch_reset(self.handle,index,seed)

    def step(self,actions):
        a=np.ascontiguousarray(actions,dtype=np.int32)
        if a.shape!=(self.n,3) or np.any(a<0) or np.any(a[:,0]>=45) or np.any(a[:,1]>1) or np.any(a[:,2]!=0):
            raise ValueError('Expected [N,3]: joint 0..44, bomb 0..1, item 0 (no item equipped)')
        self.lib.isaac_batch_step(self.handle,a.ctypes.data_as(C.POINTER(C.c_int32)))

    def step_one(self,index,action):
        if not 0<=index<self.n:raise IndexError(index)
        a=np.ascontiguousarray(action,dtype=np.int32)
        if a.shape!=(3,) or np.any(a<0) or a[0]>=45 or a[1]>1 or a[2]!=0:raise ValueError('Invalid action')
        self.lib.isaac_batch_step_one(self.handle,index,a.ctypes.data_as(C.POINTER(C.c_int32)))

    def close(self):
        if self.handle:self.lib.isaac_batch_free(self.handle);self.handle=None


class SimVecEnv(VecEnv):
    def __init__(self,n=8,seed=1,threads=4,history=HISTORY,capacity=ENTITY_CAPACITY,library=None):
        self.batch=RustBatch(n,seed,threads,library)
        if not 1<=capacity<=256:raise ValueError('Binary sim capacity must be 1..256')
        self.history=history;self.capacity=capacity
        space=VisibleHistory(history,capacity).space
        self.ring={k:np.zeros((n,*v.shape),v.dtype) for k,v in space.spaces.items()}
        self.cursor=np.zeros(n,np.int64);self.valid=np.zeros(n,np.int64)
        self.base_seed=seed;self.episodes=np.zeros(n,dtype=np.int64)
        self.actions=None;self.returns=np.zeros(n);self.lengths=np.zeros(n,dtype=np.int64)
        super().__init__(n,space,spaces.MultiDiscrete([45,2,2]))

    def reset(self):
        for i in range(self.num_envs):
            seed=self._seeds[i] if self._seeds[i] is not None else self.base_seed+i
            self.batch.reset(i,int(seed))
        # Keep autoreset seeds rooted in an explicit VecEnv.seed() override.
        if self._seeds[0] is not None:self.base_seed=int(self._seeds[0])
        for v in self.ring.values():v.fill(0)
        self.cursor[:]=0;self.valid[:]=0
        self._reset_seeds();self.episodes[:]=0;self.returns[:]=0;self.lengths[:]=0
        self._append(self.batch.observe());return self._windows()

    def _append(self,frames,indices=None):
        ids=np.arange(self.num_envs) if indices is None else np.asarray(indices)
        if np.any(frames['count'][ids]>self.capacity):raise ValueError('Visible entity overflow; increase capacity, never truncate')
        for k,v in self.ring.items():
            values=frames[k][ids]
            if k.startswith('entity') or k=='entities':values=values[:,:self.capacity]
            v[ids,self.cursor[ids]]=values
        self.cursor[ids]=(self.cursor[ids]+1)%self.history
        self.valid[ids]=np.minimum(self.valid[ids]+1,self.history)

    def _windows(self):
        start=np.where(self.valid==self.history,self.cursor,0)
        times=(start[:,None]+np.arange(self.history))%self.history
        return {k:v[np.arange(self.num_envs)[:,None],times] for k,v in self.ring.items()}

    def action_masks(self):
        masks=np.ones((self.num_envs,49),bool)
        masks[:,46]=self.batch.frames['player'][:,9]>0;masks[:,48]=False
        return masks

    def step_async(self,actions):
        self.actions=np.asarray(actions).copy()
        self.batch.step(self.actions)

    def step_wait(self):
        frames=self.batch.observe();self._append(frames)
        rewards=frames['reward'].copy();dones=frames['done'].astype(bool)
        self.returns+=rewards;self.lengths+=1
        outcomes=('running','death','win','time_limit')
        infos=[dict(outcome=outcomes[r['outcome']],elapsed_frames=int(r['elapsed']),layout=int(r['layout']),**{'TimeLimit.truncated':bool(r['truncated'])}) for r in frames]
        terminal=self._windows() if dones.any() else None
        for i in np.flatnonzero(dones):
            infos[i]['terminal_observation']={k:v[i].copy() for k,v in terminal.items()}
            infos[i]['episode']={'r':float(self.returns[i]),'l':int(self.lengths[i])}
            self.returns[i]=0;self.lengths[i]=0;self.episodes[i]+=1
            self.batch.reset(int(i),int(self.base_seed+i+self.num_envs*self.episodes[i]))
            for v in self.ring.values():v[i].fill(0)
            self.cursor[i]=0;self.valid[i]=0
        if dones.any():
            self._append(self.batch.observe(),np.flatnonzero(dones))
        return self._windows(),rewards,dones,infos

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
