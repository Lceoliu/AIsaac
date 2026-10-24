"""Sampling + unchanged-policy inference benchmark. NEVER calls learn/backward/step.

Reports separately from PPO training throughput: optimizer/minibatch epochs are
not measured. Full production history=64/capacity=256/layers=4, no feature cache.
"""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import psutil,torch
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.transformer_obs import SCHEMA

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--envs',type=int,nargs='+',default=[1,4,8,16,32,64,128]);ap.add_argument('--steps',type=int,default=16)
ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
device='cuda' if torch.cuda.is_available() else 'cpu'
report=dict(mode='benchmark_only',optimizer_steps=0,backward_calls=0,schema=SCHEMA,
 hardware=dict(cpu=platform.processor(),logical_cpus=psutil.cpu_count(),ram_gib=psutil.virtual_memory().total/2**30,
 gpu=torch.cuda.get_device_name() if device=='cuda' else None),history=64,entity_capacity=256,layers=4,threads=args.threads,results=[])
process=psutil.Process()
policy=None
for n in args.envs:
 env=SimVecEnv(n,seed=812,threads=args.threads)
 try:
  obs=env.reset();actions=np.tile([1,0,0],(n,1))
  for _ in range(64):obs,_,_,_=env.step(actions)
  if policy is None:
   policy=MaskableMultiInputActorCriticPolicy(env.observation_space,env.action_space,lambda _:3e-4,
    features_extractor_class=CombatTransformer,features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
    net_arch=dict(pi=[256],vf=[256]),normalize_images=False).to(device)
   policy.set_training_mode(False)
   weights=[p.detach().clone() for p in policy.parameters()]
  observation_bytes=sum(v.nbytes for v in obs.values())//n
  start=time.perf_counter()
  for _ in range(args.steps):obs,_,_,_=env.step(actions)
  sampling_s=time.perf_counter()-start
  row=dict(envs=n,observation_bytes=observation_bytes,rollout_128_gib=observation_bytes*n*128/2**30,
           sample_decisions_per_s=n*args.steps/sampling_s,sampling_batch_ms=sampling_s/args.steps*1000,
           valid_history_mean=float(obs['history_mask'].sum(axis=1).mean()),rss_gib=process.memory_info().rss/2**30)
  if device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  try:
   with torch.inference_mode():
    for _ in range(2):
     policy(obs_as_tensor(obs,device),action_masks=env.action_masks())
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();cpu_start=sum(process.cpu_times()[:2]);completed=0;max_entities=0
    for _ in range(args.steps):
     acts,_,_=policy(obs_as_tensor(obs,device),action_masks=env.action_masks())
     obs,_,done,_=env.step(acts.cpu().numpy());completed+=int(done.sum())
     max_entities=max(max_entities,int(obs['entity_mask'].sum(axis=2).max()))
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
   row.update(end_to_end_decisions_per_s=n*args.steps/seconds,batch_latency_ms=seconds/args.steps*1000,
              benchmark_episodes=completed,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0,
              peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/2**30 if device=='cuda' else 0,
              cpu_core_equivalents=(sum(process.cpu_times()[:2])-cpu_start)/seconds,
              max_visible_entities=max_entities,rss_gib=process.memory_info().rss/2**30)
  except torch.cuda.OutOfMemoryError as error:
   row['inference_error']=str(error);torch.cuda.empty_cache()
  report['results'].append(row)
  assert all(torch.equal(a,b) for a,b in zip(weights,policy.parameters())),'benchmark changed model weights'
  report['weights_unchanged']=True
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2),encoding='utf8')
  print(json.dumps(row),flush=True)
 finally:env.close()
 del obs,env
