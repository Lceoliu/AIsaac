"""Prepare local Monstro PPO. Training requires the explicit --train flag."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.transformer_obs import SCHEMA,VisibleHistory

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
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--episodes',type=int,default=512)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cuda')
    p.add_argument('--out',type=Path,default=Path('runs/sim-training'));args=p.parse_args()
    bytes_per_obs=sum(np.prod(s.shape)*s.dtype.itemsize for s in VisibleHistory().space.spaces.values())
    config={**vars(args),'out':str(args.out),'schema':SCHEMA,'history':64,'entity_capacity':256,
            'max_episode_seconds':120,'decisions_per_game_second':15,'n_epochs':4,
            'rollout_observation_gib':bytes_per_obs*args.envs*args.n_steps/2**30}
    print(json.dumps(config,indent=2))
    if not args.train:
        print('PREPARED ONLY: no environment, model, rollout buffer or optimizer created.');return
    if args.envs*args.n_steps%args.batch_size:raise ValueError('batch-size must divide envs*n-steps')
    torch.set_num_threads(4)
    env=SimVecEnv(args.envs,args.seed,args.threads)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    try:
        model=MaskablePPO('MultiInputPolicy',env,n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=4,
            policy_kwargs=dict(features_extractor_class=CombatTransformer,
               features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
               net_arch=dict(pi=[256],vf=[256]),normalize_images=False),device=args.device,seed=args.seed,verbose=1)
        counter=CompleteEpisodes(args.episodes,args.out/'episodes.jsonl')
        model.learn(total_timesteps=args.episodes*1800+args.envs,callback=counter)
        model.save(args.out/'last')
        (args.out/'result.json').write_text(json.dumps(dict(completed_episodes=counter.completed,timesteps=model.num_timesteps)),encoding='utf8')
    finally:env.close()

if __name__=='__main__':main()
