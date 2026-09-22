"""Touch and sample the production compact PPO storage; never train a model."""
import argparse,json,time
from pathlib import Path
import numpy as np
import psutil,torch
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.history_buffer import HistoryRolloutBuffer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--envs',type=int,default=128);p.add_argument('--steps',type=int,default=128)
p.add_argument('--out',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(4)
env=SimVecEnv(args.envs)
process=psutil.Process()
try:
    obs=env.reset()
    buffer=HistoryRolloutBuffer(args.steps,env.observation_space,env.action_space,device='cpu',n_envs=args.envs)
    actions=np.tile([1,0,0],(args.envs,1));values=torch.zeros(args.envs);starts=np.ones(args.envs,bool)
    peak_rss=process.memory_info().rss
    start=time.perf_counter()
    for _ in range(args.steps):
        masks=env.action_masks()
        new,rewards,dones,infos=env.step(actions)
        buffer.add(obs,actions,rewards,starts,values,values,action_masks=masks)
        obs=new;starts=dones
        peak_rss=max(peak_rss,process.memory_info().rss)
    seconds=time.perf_counter()-start
    buffer.compute_returns_and_advantage(values,starts)
    minibatch_start=time.perf_counter()
    batches=0
    for batch in buffer.get(8):
        assert batch.observations['entities'].shape[1:]==(64,256,31)
        batches+=1
    report=dict(mode='storage_and_sampling_only',envs=args.envs,steps=args.steps,history=64,entity_capacity=256,
        optimizer_steps=0,backward_calls=0,policy_calls=0,
        observation_storage_bytes=buffer.observation_storage_bytes,
        legacy_observation_bytes=sum(v.nbytes for v in obs.values())*args.steps,
        peak_rss_gib=peak_rss/2**30,collect_and_store_decisions_per_s=args.envs*args.steps/seconds,
        minibatch_size=8,minibatches=batches,reconstruction_seconds=time.perf_counter()-minibatch_start)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
finally:env.close()
