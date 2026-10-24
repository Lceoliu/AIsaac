"""Published checkpoints, exact room reconstruction, and held-out validation.

Checkpoints are trusted local PyTorch/SB3 files, not safe inputs from strangers.
Only the H-frame prefix is retained: PPO batches already consumed by an update
are never replayed into the optimizer on resume.
"""
import json
import gzip
import os
import random
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from .gpu_ppo import GpuMaskablePPO,FrameSampler
from .gpu_env import FULL_START

HELD_OUT_SEEDS=list(range(0x80000000,0x80000010))


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def resolve_checkpoint(path):
    path=Path(path).resolve()
    if path.name=='latest.json':path=path.parent/json.loads(path.read_text())['checkpoint']
    # state.json is the commit marker inside an atomically renamed directory.
    if path.name.startswith('.'):raise ValueError('Unpublished checkpoint')
    json.loads((path/'state.json').read_text())
    return path


class TrainingSession(BaseCallback):
    def __init__(self,config,out,completed=0,updates=0):
        super().__init__();self.config=config;self.out=Path(out)
        self.completed=completed;self.updates=updates;self.latest=None
        self.last_update_timesteps=0

    def _on_step(self):
        with (self.out/'episodes.jsonl').open('a',encoding='utf8') as f:
            for worker,(done,info) in enumerate(zip(self.locals['dones'],self.locals['infos'])):
                if done:
                    self.completed+=1
                    start={'start':info['episode_start']} if 'episode_start' in info else {}
                    f.write(json.dumps(dict(episode=self.completed,worker=worker,seed=info['seed'],
                        outcome=info['outcome'],layout=info['layout'],frames=info['elapsed_frames'],
                        **info['episode'],**start))+'\n')
        return self.completed<self.config['episodes']

    def after_update(self,model):
        self.updates+=1
        self.last_update_timesteps=model.num_timesteps
        evaluate=self.updates%self.config['eval_every']==0
        if evaluate or self.updates%self.config['checkpoint_every']==0:
            checkpoint=self.save(model,'update')
            if evaluate:self.evaluate(model,checkpoint)

    def save(self,model,phase):
        if model._collecting:raise RuntimeError('Cannot checkpoint an in-flight rollout')
        torch.cuda.synchronize(model.device)
        name=f'update-{self.updates:08d}-step-{model.num_timesteps:012d}-{phase}'
        parent=self.out/'checkpoints';parent.mkdir(exist_ok=True)
        final=parent/name;temp=parent/('.'+name)
        temp.mkdir()
        model.save(temp/'model.zip')
        b=model.rollout_buffer;s=model._gpu_sampler;start=b.frame_pos-b.history+1
        state=dict(format=1,config=self.config,completed_episodes=self.completed,updates=self.updates,
                   timesteps=model.num_timesteps,phase=phase,
                   partial_rollout_discarded=model.num_timesteps>self.last_update_timesteps,
                   policy_version=model.policy_version)
        chunks=[]
        for c in model.env.chunks:
            chunks.append(dict(seeds=c.seeds,starts=[list(s) for s in c.starts],actions=c.actions,episodes=c.episodes.copy(),
                               returns=c.returns.copy(),lengths=c.lengths.copy(),states=c.batch.states()))
        with gzip.open(temp/'continuation.pt.gz','wb',compresslevel=1) as f:
            torch.save(dict(rng=rng_state(),chunks=chunks,base_seed=model.env.base_seed,
                        frames={k:v[start:b.frame_pos+1].cpu() for k,v in b.frames.items()},
                        lengths=b.lengths[start:b.frame_pos+1].cpu(),starts=s.starts.cpu(),
                        sampler_steps=s.steps),f)
        (temp/'state.json').write_text(json.dumps(state,indent=2),encoding='utf8')
        os.replace(temp,final)
        pointer=parent/'.latest.json'
        pointer.write_text(json.dumps(dict(checkpoint=name)),encoding='utf8')
        os.replace(pointer,parent/'latest.json');self.latest=final
        print(json.dumps(dict(event='checkpoint',path=str(final),**{k:state[k] for k in ('updates','timesteps','completed_episodes','phase')})),flush=True)
        return final

    def evaluate(self,model,checkpoint):
        from .evaluation import evaluate_checkpoint
        rng=rng_state()
        try:
            result=evaluate_checkpoint(checkpoint,self.out/'evaluations'/checkpoint.name,
                                       self.config['eval_seeds'],model.device)
            print(json.dumps(dict(event='evaluation',**result)),flush=True)
        finally:restore_rng(rng)

    def finish(self,model):
        checkpoint=self.save(model,'final');self.evaluate(model,checkpoint)
        (self.out/'result.json').write_text(json.dumps(dict(completed_episodes=self.completed,
            timesteps=model.num_timesteps,updates=self.updates,checkpoint=str(checkpoint))),encoding='utf8')


def resume_model(checkpoint,env,device='cuda'):
    """Rebuild native RNG/Previous state from seed+actions; restore raw history."""
    checkpoint=resolve_checkpoint(checkpoint)
    state=json.loads((checkpoint/'state.json').read_text())
    if state['config'].get('reward_profile','legacy')!=env.reward_profile:
        raise ValueError('Reward changed: use --warm-start, not exact --resume')
    with gzip.open(checkpoint/'continuation.pt.gz','rb') as f:
        data=torch.load(f,map_location='cpu',weights_only=False)
    model=GpuMaskablePPO.load(checkpoint/'model.zip',env=env,device=device,force_reset=False)
    env.base_seed=data['base_seed'];env.generation+=1
    for c,saved in zip(env.chunks,data['chunks'],strict=True):
        # Rebuild in-progress rooms with the starts actually used; older checkpoints predate
        # start randomisation and were all full HP. New episodes follow the env's current config.
        c.reset(saved['seeds'],saved.get('starts') or [FULL_START]*len(saved['seeds']))
        for i,actions in enumerate(saved['actions']):
            for action in actions:c.batch.step_one(i,action)
        if c.batch.states()!=saved['states']:
            raise RuntimeError('Room reconstruction differs: use the matching simulator build/platform')
        c.actions=saved['actions'];c.episodes[:]=saved['episodes']
        c.returns[:]=saved['returns'];c.lengths[:]=saved['lengths']
        c.batch.observe(c.slots[0].frames)
    b=model.rollout_buffer;b.frame_pos=b.history-1
    for k,v in data['frames'].items():b.frames[k][:b.history].copy_(v)
    b.lengths[:b.history].copy_(data['lengths'])
    s=FrameSampler(model);s.generation=env.generation;s.steps=data['sampler_steps']
    s.starts.copy_(data['starts']);s.version=-1 # rebuild prefix under the restored weights
    model._gpu_sampler=s;model._last_obs={};model._last_episode_starts=data['starts'].numpy()
    for c in env.chunks:c.ready.record(torch.cuda.current_stream(model.device))
    torch.cuda.synchronize(model.device)
    restore_rng(data['rng'])
    return model,state


# Goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md): the modules a checkpoint from before it does not have
# (transformer_policy.GOAL_LINE_MODULES; C44 adds the full-room branch, ROOM_MODULES).
from .transformer_policy import GOAL_LINE_MODULES, ROOM_MODULES, module_name


def _goal_line_key(key):
    return module_name(key,GOAL_LINE_MODULES)


def _room_key(key):
    return module_name(key,ROOM_MODULES)


def warm_start_model(model,checkpoint):
    """Weights only. New countdown input starts at zero; optimizer/RNG stay fresh.

    Into a goal-line policy (CombatTransformer.goal_dim): every source key must exist with the same shape; the target's
    keys the source lacks must be goal-line modules (GOAL_LINE_MODULES), which keep their fresh values, and the layers that
    add the goal to the existing computation must still be zero afterwards, so the migrated policy computes the source's
    function on COMBAT frames. From a goal-line checkpoint (C43: C42's final) every key is there and loads strictly,
    trained goal layers included."""
    checkpoint=resolve_checkpoint(checkpoint)
    _,params,_=load_from_zip_file(checkpoint/'model.zip',device='cpu')
    source=params['policy'];target=model.policy.state_dict();expanded=[]
    unknown=[k for k in source if k not in target]
    if unknown:raise ValueError(f'Checkpoint keys the model does not have: {unknown[:5]}')
    for key,value in source.items():
        if value.shape!=target[key].shape:
            if not (key.endswith('player.0.weight') and value.ndim==2 and
                    target[key].shape==(value.shape[0],value.shape[1]+1)):
                raise ValueError(f'Unsupported migration shape: {key}')
            extra=torch.zeros((value.shape[0],1),dtype=value.dtype)
            source[key]=torch.cat([value,extra],dim=1);expanded.append(key)
    missing=[k for k in target if k not in source]
    bad=[k for k in missing if not _goal_line_key(k)]
    if bad:raise ValueError(f'Model keys missing from the checkpoint: {bad[:5]}')
    result=model.policy.load_state_dict(source,strict=not missing)
    if missing and (result.unexpected_keys or set(result.missing_keys)!=set(missing)):
        raise RuntimeError(f'Unexpected load result: {result}')
    fe=model.policy.features_extractor
    goal_source=any(_goal_line_key(k) for k in source)   # a goal-line checkpoint (C42, C43)
    if getattr(fe,'goal_dim',0) and goal_source:
        loaded=model.policy.state_dict()
        if any(not torch.equal(loaded[k].cpu(),source[k]) for k in source if _goal_line_key(k)):
            raise RuntimeError('Goal-line parameters differ from the checkpoint after loading')
        if missing and (any(not _room_key(k) for k in missing) or not fe.room_injection_zero()):
            raise RuntimeError('From a goal-line checkpoint only the full-room branch may be new, and it must start at zero')
    if getattr(fe,'goal_dim',0) and missing and not goal_source:
        nav=model.policy.action_net.nav[-1]
        if not fe.goal_injections_zero() or nav.weight.any() or nav.bias.any() or not fe.room_injection_zero():
            raise RuntimeError('Goal injections are not zero after the migration')
    if model.policy.optimizer.state:raise RuntimeError('Warm start requires a fresh optimizer')
    return dict(checkpoint=str(checkpoint),expanded_zero_columns=expanded,new_goal_line_keys=len(missing),
                optimizer_restored=False,episode_state_restored=False)


def _optimizer_names(policy,groups):
    """Parameter names in the flat order of an optimizer state's param_groups: one unnamed group is named_parameters()'s
    order (every checkpoint before C44); named groups ('shared', 'goal') are GeometryPolicy(lr_groups=True)'s."""
    named=list(policy.named_parameters())
    if len(groups)==1 and not groups[0].get('name'):return [n for n,_ in named]
    order=[]
    for g in groups:
        pick={'shared':lambda n:not _goal_line_key(n),'goal':_goal_line_key}.get(g.get('name'))
        if pick is None:raise RuntimeError(f"unknown optimizer group {g.get('name')!r}")
        order+=[n for n,_ in named if pick(n)]
    return order


def warm_start_optimizer(model,checkpoint,only=None):
    """The checkpoint's Adam state for the parameters it shares with the model, matched by name (a goal-line policy has
    more parameters than a checkpoint from before it, so positions differ; C44's optimizer has named groups); the other
    parameters keep theirs. only: 'shared' or 'goal' restricts it to that kind of parameter (C44: the shared ones from C39's
    final, whose frozen stage-1 successors have none; the goal line's from C43's). The parameter order of the checkpoint is
    that of its own policy, rebuilt from its saved class and arguments."""
    checkpoint=resolve_checkpoint(checkpoint)
    data,params,_=load_from_zip_file(checkpoint/'model.zip',device=model.device)
    saved=params['policy.optimizer']
    old=data['policy_class'](data['observation_space'],data['action_space'],lambda _:0.0,**data['policy_kwargs'])
    old_names=_optimizer_names(old,saved['param_groups'])
    flat=[i for g in saved['param_groups'] for i in g['params']]
    if len(old_names)!=len(flat):raise RuntimeError('Optimizer state does not match its policy')
    optimizer=model.policy.optimizer
    by_id={id(q):n for n,q in model.policy.named_parameters()}
    new_names=[by_id[id(q)] for g in optimizer.param_groups for q in g['params']]
    index={n:i for i,n in enumerate(new_names)}
    state=optimizer.state_dict()
    keep=(lambda n:True) if only is None else (lambda n:_goal_line_key(n)==(only=='goal'))
    restored=0
    for position,name in enumerate(old_names):
        if name not in index:raise ValueError(f'Checkpoint parameter {name} is not in the model')
        i=flat[position]
        if i in saved['state'] and keep(name):
            state['state'][index[name]]=saved['state'][i];restored+=1
    optimizer.load_state_dict(state)
    return dict(optimizer_restored=True,optimizer_state_entries=restored,
                optimizer_fresh_parameters=len(new_names)-len(old_names),optimizer_only=only)
