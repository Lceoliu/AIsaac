"""Fixed-seed deterministic evaluation with authoritative state/action replays."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .gpu_env import GpuFrameVecEnv
from .gpu_ppo import GpuMaskablePPO


def visible_snapshot(record):
    # Diagnostic AI targets/RNG are neither policy inputs nor visual cues.
    for boss in record['state']['bosses']:
        boss.pop('debug_state');boss.pop('debug_target')
    return record


def write_replay(path,metadata,rows):
    with gzip.open(path.with_suffix('.jsonl.gz'),'wt',encoding='utf8') as f:
        f.write(json.dumps(dict(metadata=metadata))+'\n')
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    template=Path(__file__).with_name('replay.html').read_text(encoding='utf8')
    payload=json.dumps(dict(metadata=metadata,rows=rows),separators=(',',':')).replace('</','<\\/')
    path.with_suffix('.html').write_text(template.replace('/*REPLAY_DATA*/',payload),encoding='utf8')


def evaluate_checkpoint(checkpoint,out,seeds,device='cuda'):
    checkpoint=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if len(set(seeds))!=len(seeds) or not seeds or any(not 2**31<=s<2**32 for s in seeds):
        raise ValueError('Evaluation seeds must be unique uint32 held-out seeds >= 2**31')
    saved=json.loads((checkpoint/'state.json').read_text());config=saved['config']
    env=GpuFrameVecEnv(len(seeds),seeds[0],min(4,len(seeds)),1,
                       history=config['history'],capacity=config['entity_capacity'],device=device)
    try:
        model=GpuMaskablePPO.load(checkpoint/'model.zip',env=env,device=device,
                                custom_objects={'n_envs':len(seeds)},force_reset=True)
        model.session=None;model.sampling_deterministic=True
        env.record_states=True
        # SB3's seeded reset is followed by explicit fixed seeds, never training seeds.
        _,callback=model._setup_learn(1801*len(seeds),BaseCallbackForEvaluation())
        env._seeds=list(seeds);env.reset();model._last_obs={}
        rows=[[dict(action=None,**visible_snapshot(r))] for r in env.chunks[0].batch.states()]
        finished=np.zeros(len(seeds),bool);results=[]
        class Recorder(BaseCallback):
            def _on_step(self):
                actions=self.locals['actions'].cpu().numpy().tolist()
                for i,info in enumerate(self.locals['infos']):
                    if finished[i]:continue
                    row=dict(action=actions[i],**visible_snapshot(info['replay_state']))
                    rows[i].append(row)
                    if self.locals['dones'][i]:
                        finished[i]=True
                        metadata=dict(format='isaac-state-replay-v1',seed=seeds[i],
                            checkpoint=str(checkpoint.resolve()),timesteps=saved['timesteps'],
                            updates=saved['updates'],deterministic=True,logic_fps=30,action_repeat=2,
                            schema=config['schema'],source_revision=config.get('source_revision'),
                            visualization='Recorded geometry, not original sprite animation')
                        write_replay(out/f'seed-{seeds[i]}',metadata,rows[i])
                        results.append(dict(seed=seeds[i],outcome=info['outcome'],layout=info['layout'],
                            frames=info['elapsed_frames'],**info['episode'],replay=f'seed-{seeds[i]}.jsonl.gz'))
                return not finished.all()
        callback=Recorder();callback.init_callback(model)
        while not finished.all():
            model.collect_rollouts(env,callback,model.rollout_buffer,model.n_steps)
        results.sort(key=lambda r:r['seed'])
        summary=dict(seeds=seeds,episodes=len(results),wins=sum(r['outcome']=='win' for r in results),
                     deaths=sum(r['outcome']=='death' for r in results),
                     timeouts=sum(r['outcome']=='time_limit' for r in results),
                     mean_return=float(np.mean([r['r'] for r in results])),results=results,
                     path=str(out.resolve()))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        links='\n'.join(f'<li><a href="seed-{r["seed"]}.html">Seed {r["seed"]}: {r["outcome"]}</a></li>' for r in results)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Held-out evaluation</title><h1>Held-out evaluation</h1><ul>'+links+'</ul>',encoding='utf8')
        return summary
    finally:env.close()


class BaseCallbackForEvaluation(BaseCallback):
    def _on_step(self):return True
