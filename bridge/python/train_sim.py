"""Prepare local Monstro PPO. Training requires the explicit --train flag."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.transformer_obs import SCHEMA,VisibleHistory
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
    p.add_argument('--out',type=Path,default=Path('runs/sim-training'));args=p.parse_args()
    from isaac_bridge.training_session import TrainingSession,HELD_OUT_SEEDS,resolve_checkpoint,resume_model
    if args.resume:
        args.resume=resolve_checkpoint(args.resume)
        previous=json.loads((args.resume/'state.json').read_text())['config']
        for key in ('envs','threads','chunks','n_steps','batch_size','pipeline','seed','checkpoint_every','eval_every','episodes'):
            flag='--'+key.replace('_','-')
            if not any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:]):setattr(args,key,previous[key])
            elif key not in ('episodes','checkpoint_every','eval_every') and getattr(args,key)!=previous[key]:
                p.error(f'Resume requires unchanged {flag}')
    if min(args.checkpoint_every,args.eval_every,args.episodes)<=0:p.error('intervals and episode target must be positive')
    if not 0<=args.seed<2**31 or args.seed+args.envs>=2**31:p.error('Training seeds must remain below 2**31')
    if args.resume and args.pipeline!='gpu':p.error('Resume requires GPU pipeline')
    bytes_per_obs=sum(int(np.prod(s.shape))*s.dtype.itemsize for s in VisibleHistory().space.spaces.values())
    frame_bytes=sum(int(np.prod(s.shape[1:]))*np.dtype(STORAGE_DTYPES.get(k,s.dtype)).itemsize
                    for k,s in VisibleHistory().space.spaces.items())
    config={**vars(args),'out':str(args.out.resolve()),'resume':str(args.resume) if args.resume else None,
            'eval_seeds':HELD_OUT_SEEDS,'schema':SCHEMA,'history':64,'entity_capacity':256,
            'max_episode_seconds':120,'decisions_per_game_second':15,'n_epochs':4,
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
        env=GpuFrameVecEnv(args.envs,args.seed,args.threads,args.chunks,device=args.device)
        algorithm=GpuMaskablePPO;buffer_class=GpuHistoryRolloutBuffer
    else:
        env=SimVecEnv(args.envs,args.seed,args.threads)
        algorithm=MaskablePPO;buffer_class=HistoryRolloutBuffer
    args.out.mkdir(parents=True,exist_ok=False)
    config['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    try:
        if args.resume:
            model,state=resume_model(args.resume,env,args.device)
            if args.episodes<=state['completed_episodes']:raise ValueError('--episodes must exceed already completed episodes')
        else:model=algorithm('MultiInputPolicy',env,n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=4,
            rollout_buffer_class=buffer_class,
            policy_kwargs=dict(features_extractor_class=CombatTransformer,
               features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
               net_arch=dict(pi=[256],vf=[256]),normalize_images=False),device=args.device,seed=args.seed,verbose=1)
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
