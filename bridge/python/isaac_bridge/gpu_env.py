"""Chunked Rust sources for GpuMaskablePPO; CUDA transfers use owned pinned slots.

The VecEnv facade supplies SB3 spaces/seeds/reset. Its ordinary step interface is
deliberately unavailable: the GPU collector drives independent chunks instead.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from .sim_vec import RustBatch,FRAME_DTYPE
from .transformer_obs import VisibleHistory,HISTORY,ENTITY_CAPACITY

FULL_START=(6.0,1.0)  # player half-hearts, Boss HP fraction


def sample_start(seed,config):
    """Seed-deterministic training start, independent of the combat RNG.

    OpenAI Five randomised Roshan's HP so agents that had learned "never approach" still met
    kills; here some Boss starts are weakened and some player starts reduced. None -> full HP.
    Resume restores the starts actually applied, so a later config change never rewrites them.
    """
    if not config:return FULL_START
    rng=np.random.default_rng([int(seed),0x5EED])
    player=float(rng.integers(config['player_hp_min'],6)) if rng.random()<config['player_hp_prob'] else 6.0
    boss=float(rng.uniform(config['boss_hp_min'],1.0)) if rng.random()<config['boss_hp_prob'] else 1.0
    return player,boss


def validate_start_randomization(config):
    if not config:return None
    config={k:config[k] for k in ('boss_hp_prob','boss_hp_min','player_hp_prob','player_hp_min')}
    if not (0<=config['boss_hp_prob']<=1 and 0<config['boss_hp_min']<=1 and 0<=config['player_hp_prob']<=1
            and config['player_hp_min']==int(config['player_hp_min']) and 1<=config['player_hp_min']<=5):
        raise ValueError(f'Invalid start randomisation {config}')
    return config if config['boss_hp_prob'] or config['player_hp_prob'] else None


class TransferSlot:
    def __init__(self,n,device):
        words=FRAME_DTYPE.itemsize//4
        self.host=torch.empty((n,words),dtype=torch.int32,pin_memory=True)
        self.frames=self.host.numpy().view(FRAME_DTYPE).reshape(n)
        self.reset_host=torch.empty_like(self.host,pin_memory=True)
        self.reset_frames=self.reset_host.numpy().view(FRAME_DTYPE).reshape(n)
        self.device=torch.empty((n,words),dtype=torch.int32,device=device)
        self.reset_device=torch.empty_like(self.device)
        self.actions=torch.empty((n,3),dtype=torch.int32,pin_memory=True)
        self.action_ready=torch.cuda.Event()
        self.upload_done=torch.cuda.Event()
        self.consumed=torch.cuda.Event()
        self.used=False


class FrameChunk:
    def __init__(self,owner,start,stop,threads):
        self.owner=owner;self.start=start;self.stop=stop;self.n=stop-start
        self.batch=RustBatch(self.n,owner.base_seed+start,threads,reward_profile=owner.reward_profile)
        self.copy_stream=torch.cuda.Stream(device=owner.device)
        self.compute_stream=torch.cuda.Stream(device=owner.device)
        self.ready=torch.cuda.Event()
        self.slots=[TransferSlot(self.n,owner.device) for _ in range(2)]
        self.episodes=np.zeros(self.n,np.int64);self.returns=np.zeros(self.n);self.lengths=np.zeros(self.n,np.int64)
        self.seeds=[owner.base_seed+i for i in range(start,stop)]
        self.actions=[[] for _ in range(self.n)]
        self.starts=[FULL_START]*self.n

    def reset(self,seeds,starts=None):
        """starts=None samples from the owner's training config; resume passes the saved starts."""
        self.seeds=list(map(int,seeds));self.actions=[[] for _ in range(self.n)]
        self.starts=([tuple(map(float,s)) for s in starts] if starts is not None else
                     [sample_start(s,self.owner.start_randomization) for s in self.seeds])
        for i,(seed,start) in enumerate(zip(self.seeds,self.starts)):self.batch.reset(i,seed,*start)
        self.episodes.fill(0);self.returns.fill(0);self.lengths.fill(0)
        self.batch.observe(self.slots[0].frames)

    def advance(self,slot):
        # Only this worker owns its Rust world. Main thread never reads a pinned
        # action before D2H ends, nor overwrites a frame while DMA still reads it.
        slot.action_ready.synchronize()
        if slot.used:slot.upload_done.synchronize()
        self.batch.step(slot.actions.numpy())
        for trace,action in zip(self.actions,slot.actions.numpy().tolist()):trace.append(action)
        frames=self.batch.observe(slot.frames)
        if np.any(frames['count']>self.owner.capacity):raise ValueError('Visible entity overflow; never truncate')
        rewards=frames['reward'].copy();dones=frames['done'].astype(bool)
        self.returns+=rewards;self.lengths+=1
        outcomes=('running','death','win','time_limit')
        infos=[dict(outcome=outcomes[r['outcome']],elapsed_frames=int(r['elapsed']),layout=int(r['layout']),
                    **{'TimeLimit.truncated':bool(r['truncated'])}) for r in frames]
        for i,info in enumerate(infos):info['seed']=self.seeds[i]
        if self.owner.record_states:
            for info,state in zip(infos,self.batch.states()):info['replay_state']=state
        for i in np.flatnonzero(dones):
            infos[i]['episode']={'r':float(self.returns[i]),'l':int(self.lengths[i])}
            infos[i]['episode_start']={'player_hp':self.starts[i][0],'boss_hp_fraction':self.starts[i][1]}
            self.returns[i]=0;self.lengths[i]=0;self.episodes[i]+=1
            seed=self.owner.base_seed+self.start+i+self.owner.num_envs*self.episodes[i]
            if self.owner.training_seeds and seed>=2**31:raise ValueError('Training seed entered held-out namespace')
            self.seeds[i]=int(seed);self.actions[i]=[]
            self.starts[i]=sample_start(seed,self.owner.start_randomization)
            self.batch.reset(int(i),int(seed),*self.starts[i])
        if dones.any():
            self.batch.observe(slot.reset_frames)
            # Pack only reset rows in the pinned prefix; do not upload a whole
            # chunk again when just one world terminates.
            terminal=np.flatnonzero(dones)
            slot.reset_frames[:len(terminal)]=slot.reset_frames[terminal]
        return rewards,dones,infos

    def advance_and_upload(self,slot):
        result=self.advance(slot)
        count=int(result[1].sum())
        # The CPU worker queues DMA immediately after native export. Waiting for
        # the Python policy loop to retrieve its future serialized these copies
        # behind other chunks' policy kernels, despite using separate streams.
        with torch.cuda.stream(self.copy_stream):
            if slot.used:self.copy_stream.wait_event(slot.consumed)
            slot.device.copy_(slot.host,non_blocking=True)
            if count:slot.reset_device[:count].copy_(slot.reset_host[:count],non_blocking=True)
            slot.upload_done.record(self.copy_stream)
        slot.used=True
        return result


class GpuFrameVecEnv(VecEnv):
    def __init__(self,n=8,seed=1,threads=4,chunks=1,history=HISTORY,capacity=ENTITY_CAPACITY,device='cuda',reward_profile='legacy',
                 start_randomization=None):
        if not 1<=chunks<=min(n,threads):raise ValueError('chunks must be <= envs and total Rust threads')
        if not 1<=capacity<=256:raise ValueError('capacity must be 1..256')
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('GpuFrameVecEnv requires CUDA')
        self.base_seed=seed;self.capacity=capacity;self.history=history;self.generation=0
        self.reward_profile=reward_profile
        # Training-only; evaluation constructs this env without it and therefore starts at full HP.
        self.start_randomization=validate_start_randomization(start_randomization)
        self.record_states=False;self.training_seeds=False
        super().__init__(n,VisibleHistory(history,capacity,deadline=reward_profile=='combat-v1').space,spaces.MultiDiscrete([45,2,2]))
        boundaries=np.linspace(0,n,chunks+1,dtype=int)
        self.chunks=[FrameChunk(self,int(boundaries[i]),int(boundaries[i+1]),threads//chunks+int(i<threads%chunks)) for i in range(chunks)]
        self.executor=ThreadPoolExecutor(max_workers=chunks,thread_name_prefix='isaac-frame')
        self.closed=False

    def reset(self):
        torch.cuda.synchronize(self.device)
        if self._seeds[0] is not None:self.base_seed=int(self._seeds[0])
        for c in self.chunks:
            seeds=[self._seeds[i] if self._seeds[i] is not None else self.base_seed+i for i in range(c.start,c.stop)]
            c.reset(seeds)
        self._reset_seeds();self.generation+=1
        # One conventional SB3 reset observation. The custom collector releases
        # this immediately; no CPU history or full-window hot-path allocation.
        obs={k:np.zeros((self.num_envs,*s.shape),s.dtype) for k,s in self.observation_space.spaces.items()}
        for c in self.chunks:
            for k,v in obs.items():
                source=np.ones(c.n,np.float32) if k=='remaining_time' else c.slots[0].frames[k]
                if k.startswith('entity') or k=='entities':source=source[:,:self.capacity]
                v[c.start:c.stop,0]=source
        return obs

    def step_async(self,actions):raise NotImplementedError('Use GpuMaskablePPO chunk collector, or SimVecEnv for CPU stepping')
    def step_wait(self):raise NotImplementedError('Use GpuMaskablePPO chunk collector')
    def close(self):
        if self.closed:return
        self.executor.shutdown(wait=True);torch.cuda.synchronize(self.device)
        for c in self.chunks:c.batch.close()
        self.closed=True
    def get_attr(self,name,indices=None):
        return [None if name=='render_mode' else getattr(self,name) for _ in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):raise NotImplementedError('Configure source through constructor')
    def env_method(self,method_name,*args,indices=None,**kwargs):raise NotImplementedError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]


def decode_frame(words,space):
    result={}
    for k,s in space.spaces.items():
        if k=='remaining_time':continue
        dtype,offset=FRAME_DTYPE.fields[k]
        v=words[:,offset//4:(offset+dtype.itemsize)//4]
        if dtype.base==np.dtype('float32'):v=v.view(torch.float32)
        v=v.reshape(len(words),*dtype.shape)
        if k.startswith('entity') or k=='entities':v=v[:,:s.shape[1]]
        result[k]=v
    if 'remaining_time' in space.spaces:result['remaining_time']=(1-result['time']/120).clamp(0,1)
    return result


def metadata(words,name):
    dtype,offset=FRAME_DTYPE.fields[name]
    v=words[:,offset//4]
    return v.view(torch.float32) if dtype==np.dtype('float32') else v
