"""Prepare local Monstro PPO. Training requires the explicit --train flag."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.transformer_obs import SCHEMA,DEADLINE_SCHEMA,VisibleHistory
from isaac_bridge.combat_reward import REWARD_PROFILES
from isaac_bridge.history_buffer import HistoryRolloutBuffer,STORAGE_DTYPES

class CompleteEpisodes(BaseCallback):
    def __init__(self,target,path):super().__init__();self.target=target;self.completed=0;self.path=path
    def _on_step(self):
        with self.path.open('a',encoding='utf8') as f:
            for worker,(done,info) in enumerate(zip(self.locals['dones'],self.locals['infos'])):
                if done:
                    self.completed+=1
                    f.write(json.dumps(dict(episode=self.completed,worker=worker,outcome=info['outcome'],
                        layout=info['layout'],frames=info['elapsed_frames'],**info['episode']))+'\n')
        return self.completed<self.target

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',action='store_true');p.add_argument('--envs',type=int,default=8)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--n-steps',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--episodes',type=int,default=512)
    p.add_argument('--pipeline',choices=['gpu','legacy'],default='gpu')
    p.add_argument('--chunks',type=int,default=1)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint-every',type=int,default=10,help='completed PPO updates')
    p.add_argument('--eval-every',type=int,default=25,help='completed PPO updates')
    p.add_argument('--resume',type=Path,help='trusted checkpoint directory or checkpoints/latest.json; use a new --out')
    p.add_argument('--warm-start',type=Path,help='weights-only migration; optimizer, RNG and rooms start fresh')
    p.add_argument('--reward-profile',choices=REWARD_PROFILES,default='combat-v1')
    p.add_argument('--gamma',type=float,default=None)
    p.add_argument('--n-epochs',type=int,default=4)
    p.add_argument('--micro-batch',type=int,default=None,
                   help='enable frame-deduplicated segment minibatches; bounds activation memory only')
    p.add_argument('--segment-length',type=int,default=32,help='consecutive steps per worker segment (segment minibatches)')
    p.add_argument('--ent-coef',type=float,default=0.0)
    p.add_argument('--boss-hp-prob',type=float,default=0.0,help='share of training episodes whose Boss starts weakened')
    p.add_argument('--boss-hp-min',type=float,default=0.1,help='lowest Boss HP fraction of a weakened start')
    p.add_argument('--player-hp-prob',type=float,default=0.0,help='share of training episodes whose player starts hurt')
    p.add_argument('--player-hp-min',type=int,default=3,help='lowest player half-hearts of a hurt start (drawn from min..5)')
    p.add_argument('--out',type=Path,default=Path('runs/sim-training'));args=p.parse_args()
    from isaac_bridge.training_session import TrainingSession,HELD_OUT_SEEDS,resolve_checkpoint,resume_model,warm_start_model
    if args.resume and args.warm_start:p.error('--resume and --warm-start are mutually exclusive')
    # Optimisation and training-start distribution may change on resume; world/reward semantics may not.
    tunable=('episodes','checkpoint_every','eval_every','batch_size','n_epochs','micro_batch','segment_length','ent_coef',
             'boss_hp_prob','boss_hp_min','player_hp_prob','player_hp_min')
    if args.resume:
        args.resume=resolve_checkpoint(args.resume)
        previous=json.loads((args.resume/'state.json').read_text())['config']
        for key,value in dict(reward_profile='legacy',gamma=0.99,n_epochs=4,micro_batch=None,segment_length=32,ent_coef=0.0,
                              boss_hp_prob=0.0,boss_hp_min=0.1,player_hp_prob=0.0,player_hp_min=3).items():
            previous.setdefault(key,value)
        for key in ('envs','threads','chunks','n_steps','pipeline','seed','reward_profile','gamma')+tunable:
            flag='--'+key.replace('_','-')
            if not any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:]):setattr(args,key,previous[key])
            elif key not in tunable and getattr(args,key)!=previous[key]:
                p.error(f'Resume requires unchanged {flag}')
    if args.micro_batch is not None:
        if args.pipeline!='gpu':p.error('--micro-batch requires the GPU pipeline')
        if (args.micro_batch<=0 or args.segment_length<=0 or args.batch_size%args.micro_batch
                or args.micro_batch%args.segment_length or args.n_steps%args.segment_length):
            p.error('need segment-length | n-steps, segment-length | micro-batch and micro-batch | batch-size')
    if args.n_epochs<=0 or args.ent_coef<0:p.error('n-epochs must be positive and ent-coef non-negative')
    start_randomization=None
    if args.boss_hp_prob or args.player_hp_prob:
        if args.pipeline!='gpu':p.error('Start randomisation requires the GPU pipeline')
        from isaac_bridge.gpu_env import validate_start_randomization
        try:start_randomization=validate_start_randomization(dict(boss_hp_prob=args.boss_hp_prob,boss_hp_min=args.boss_hp_min,
                player_hp_prob=args.player_hp_prob,player_hp_min=args.player_hp_min))
        except ValueError as e:p.error(str(e))
    if args.gamma is None:args.gamma=0.999 if args.reward_profile=='combat-v1' else 0.99
    if not 0<args.gamma<1:p.error('gamma must be between 0 and 1')
    if args.warm_start:
        args.warm_start=resolve_checkpoint(args.warm_start)
        if args.pipeline!='gpu':p.error('Warm start requires GPU pipeline')
    if min(args.checkpoint_every,args.eval_every,args.episodes)<=0:p.error('intervals and episode target must be positive')
    if not 0<=args.seed<2**31 or args.seed+args.envs>=2**31:p.error('Training seeds must remain below 2**31')
    if args.resume and args.pipeline!='gpu':p.error('Resume requires GPU pipeline')
    space=VisibleHistory(deadline=args.reward_profile=='combat-v1').space
    bytes_per_obs=sum(int(np.prod(s.shape))*s.dtype.itemsize for s in space.spaces.values())
    frame_bytes=sum(int(np.prod(s.shape[1:]))*np.dtype(STORAGE_DTYPES.get(k,s.dtype)).itemsize
                    for k,s in space.spaces.items())
    config={**vars(args),'out':str(args.out.resolve()),'resume':str(args.resume) if args.resume else None,
            'warm_start':str(args.warm_start) if args.warm_start else None,
            'eval_seeds':HELD_OUT_SEEDS,'schema':DEADLINE_SCHEMA if args.reward_profile=='combat-v1' else SCHEMA,'history':64,'entity_capacity':256,
            'timeout_semantics':'termination' if args.reward_profile=='combat-v1' else 'truncation',
            'max_episode_seconds':120,'decisions_per_game_second':15,
            'minibatch':('segments: frame-deduplicated, advantages normalised per batch_size, gradient-accumulated micro-batches'
                         if args.micro_batch else 'SB3 per-window random samples'),
            'start_randomization':start_randomization,'evaluation_start':'full HP',
            'legacy_rollout_observation_gib':bytes_per_obs*args.envs*args.n_steps/2**30,
            'rollout_observation_gib':(frame_bytes*args.envs*(args.n_steps+63)+4*args.envs*args.n_steps)/2**30,
            'simulation_device':'cpu/rayon','policy_device':args.device,'rollout_buffer':'HistoryRolloutBuffer'}
    if args.pipeline=='gpu':
        raw_bytes=bytes_per_obs//64
        config.update(rollout_observation_gib=(raw_bytes+8)*args.envs*(args.n_steps+64)/2**30,
                      rollout_buffer='GpuHistoryRolloutBuffer',frame_cache_gib=(args.n_steps+64)*args.envs*256*4/2**30,
                      observation_storage_device=args.device,transfer='pinned double-buffer/chunk CUDA streams',
                      frozen_collection_weights=True)
    print(json.dumps(config,indent=2))
    if not args.train:
        print('PREPARED ONLY: no environment, model, rollout buffer or optimizer created.');return
    if args.envs*args.n_steps%args.batch_size:raise ValueError('batch-size must divide envs*n-steps')
    torch.set_num_threads(4)
    if args.pipeline=='gpu':
        from isaac_bridge.gpu_env import GpuFrameVecEnv
        from isaac_bridge.gpu_ppo import GpuMaskablePPO
        from isaac_bridge.gpu_buffer import GpuHistoryRolloutBuffer
        env=GpuFrameVecEnv(args.envs,args.seed,args.threads,args.chunks,device=args.device,reward_profile=args.reward_profile,
                           start_randomization=start_randomization)
        algorithm=GpuMaskablePPO;buffer_class=GpuHistoryRolloutBuffer
    else:
        env=SimVecEnv(args.envs,args.seed,args.threads,reward_profile=args.reward_profile)
        algorithm=MaskablePPO;buffer_class=HistoryRolloutBuffer
    args.out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    config['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    # Uncommitted deployments stay reproducible: keep the exact tracked-file diff next to the run.
    patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=repo)
    if patch:
        (args.out/'source.patch').write_bytes(patch)
        config['source_patch_sha256']=hashlib.sha256(patch).hexdigest()
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    try:
        if args.resume:
            model,state=resume_model(args.resume,env,args.device)
            if args.episodes<=state['completed_episodes']:raise ValueError('--episodes must exceed already completed episodes')
            model.batch_size=args.batch_size;model.n_epochs=args.n_epochs;model.ent_coef=args.ent_coef
        else:model=algorithm('MultiInputPolicy',env,n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=args.n_epochs,
            gamma=args.gamma,ent_coef=args.ent_coef,rollout_buffer_class=buffer_class,
            policy_kwargs=dict(features_extractor_class=CombatTransformer,
               features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
               net_arch=dict(pi=[256],vf=[256]),normalize_images=False),device=args.device,seed=args.seed,verbose=1)
        if args.pipeline=='gpu':model.micro_batch_size=args.micro_batch;model.segment_length=args.segment_length
        if args.warm_start:
            migration=warm_start_model(model,args.warm_start)
            (args.out/'migration.json').write_text(json.dumps(migration,indent=2),encoding='utf8')
        if args.pipeline=='gpu':
            env.training_seeds=True
            counter=TrainingSession(config,args.out,state['completed_episodes'] if args.resume else 0,
                                    state['updates'] if args.resume else 0)
            model.session=counter
            model.learn(total_timesteps=(args.episodes-counter.completed)*1800+args.envs,
                        callback=counter,reset_num_timesteps=not bool(args.resume))
            counter.finish(model)
        else:
            counter=CompleteEpisodes(args.episodes,args.out/'episodes.jsonl')
            model.learn(total_timesteps=args.episodes*1800+args.envs,callback=counter)
            model.save(args.out/'last')
            (args.out/'result.json').write_text(json.dumps(dict(completed_episodes=counter.completed,timesteps=model.num_timesteps)),encoding='utf8')
    finally:env.close()

if __name__=='__main__':main()
