"""Bounded CUDA correctness tests, including one isolated optimizer update."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.gpu_env import GpuFrameVecEnv,metadata
from isaac_bridge.gpu_ppo import GpuMaskablePPO
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_policy import CombatTransformer


class Callback(BaseCallback):
    def _on_step(self):return True


def model_for(env,steps=8):
    return GpuMaskablePPO('MultiInputPolicy',env,n_steps=steps,batch_size=4,n_epochs=1,
        device='cuda',seed=91,policy_kwargs=dict(features_extractor_class=CombatTransformer,
        features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GpuInfraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_raw_windows_actions_rewards_match_cpu_and_gae(self):
        env=GpuFrameVecEnv(3,seed=91,threads=2,chunks=2,history=8,capacity=128)
        cpu=SimVecEnv(3,seed=91,threads=2,history=8,capacity=128)
        try:
            steps=128
            model=model_for(env,steps);_,cb=model._setup_learn(3*steps,Callback());expected=cpu.reset()
            saved=[];starts=[];all_rewards=[]
            class Oracle(Callback):
                def _on_step(inner):
                    nonlocal expected
                    b=model.rollout_buffer;t=b.pos-1
                    actual=b.window(b.history-1+t,torch.arange(3,device='cuda'))
                    for k in actual:np.testing.assert_array_equal(actual[k].cpu().numpy(),expected[k],err_msg=k)
                    saved.append({k:v.copy() for k,v in expected.items()})
                    starts.append(b.episode_starts[t].cpu().numpy().copy())
                    expected,reward,done,info=cpu.step(b.actions[t].cpu().numpy())
                    np.testing.assert_array_equal(inner.locals['rewards'],reward)
                    np.testing.assert_array_equal(inner.locals['dones'],done)
                    all_rewards.append(reward.copy())
                    return True
            cb=Oracle();cb.init_callback(model)
            weights=[p.detach().clone() for p in model.policy.parameters()]
            self.assertTrue(model.collect_rollouts(env,cb,model.rollout_buffer,steps))
            b=model.rollout_buffer;s=model._gpu_sampler
            with torch.no_grad():last=s.values(b.frame_pos,torch.arange(3,device='cuda')).cpu().numpy()
            values=b.values;gae=np.zeros(3,np.float32);advantages=np.zeros((steps,3),np.float32)
            for t in reversed(range(steps)):
                nt=1-model._last_episode_starts if t==steps-1 else 1-starts[t+1]
                nv=last if t==steps-1 else values[t+1]
                gae=all_rewards[t]+model.gamma*nv*nt-values[t]+model.gamma*model.gae_lambda*nt*gae
                advantages[t]=gae
            np.testing.assert_allclose(b.advantages.cpu(),advantages,atol=2e-6,rtol=2e-5)
            for indices in ([0,15,16,31,47],[3,19,39]):
                sample=b._get_samples(indices)
                for i,index in enumerate(indices):
                    worker,t=divmod(index,steps)
                    for k in saved[t]:np.testing.assert_array_equal(sample.observations[k][i].cpu(),saved[t][k][worker])
            self.assertTrue(all(torch.equal(a,p) for a,p in zip(weights,model.policy.parameters())))
            self.assertGreaterEqual(s.encoded_frames,3*(steps+1))
            self.assertEqual(s.upload_bytes,s.encoded_frames*72664)
            # Another rollout must retain previous history rather than zero it.
            saved.clear();starts.clear();all_rewards.clear()
            self.assertTrue(model.collect_rollouts(env,cb,b,steps));self.assertEqual(s.prefix_rebuilds,0)
        finally:env.close();cpu.close()

    def test_cached_distribution_training_invalidation_and_checkpoint(self):
        env=GpuFrameVecEnv(4,history=8,capacity=128,threads=2,chunks=2)
        try:
            m=model_for(env);_,cb=m._setup_learn(32,Callback());b=m.rollout_buffer
            m.collect_rollouts(env,cb,b,8);s=m._gpu_sampler;ids=torch.arange(4,device='cuda')
            with torch.no_grad():
                cached,cv,cl,masks=s.action(b.frame_pos,ids,True)
                raw=b.window(b.frame_pos,ids)
                actions,values,lp=m.policy(raw,deterministic=True,action_masks=masks)
                torch.testing.assert_close(cached,actions,rtol=0,atol=0)
                torch.testing.assert_close(cv,values.flatten(),rtol=2e-5,atol=2e-5)
                torch.testing.assert_close(cl,lp,rtol=2e-5,atol=2e-5)
            self.assertEqual(m._last_obs,{})
            weights=[p.detach().clone() for p in m.policy.parameters()]
            m.train()  # One tiny test update, not an experiment/training run.
            self.assertTrue(any(not torch.equal(a,p) for a,p in zip(weights,m.policy.parameters())))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in s.encoder.map_cnn.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in s.encoder.entity.parameters()))
            m.collect_rollouts(env,cb,b,8);self.assertEqual(s.prefix_rebuilds,1)
            with torch.no_grad():
                torch.testing.assert_close(s.latent(b.frame_pos,ids),s.encoder(b.window(b.frame_pos,ids)),rtol=2e-4,atol=2e-5)
            with tempfile.TemporaryDirectory(dir='runs') as tmp:
                path=Path(tmp)/'model';m.save(path)
                loaded=GpuMaskablePPO.load(path,env=env,device='cuda')
                _,lc=loaded._setup_learn(32,Callback())
                self.assertTrue(loaded.collect_rollouts(env,lc,loaded.rollout_buffer,8))
        finally:env.close()

    def test_independent_reset_and_timeout_bootstrap(self):
        env=GpuFrameVecEnv(2,history=8,capacity=128,threads=2,chunks=2)
        try:
            m=model_for(env,4);_,cb=m._setup_learn(8,Callback());c=env.chunks[0];advance=c.advance
            calls=0
            def forced(slot):
                nonlocal calls
                reward,done,infos=advance(slot);calls+=1
                if calls==2:
                    # Synthetic time-limit injection tests collector semantics,
                    # not Monstro gameplay or the native timeout implementation.
                    done[0]=True;slot.frames['done'][0]=1;slot.frames['truncated'][0]=1
                    infos[0].update({'TimeLimit.truncated':True,'episode':{'r':float(reward[0]),'l':2}})
                    c.batch.reset(0,123);c.batch.observe(slot.reset_frames)
                return reward,done,infos
            class Check(Callback):
                def _on_step(inner):
                    if calls==2:
                        b=m.rollout_buffer;info=inner.locals['infos'][0]
                        obs={k:v[None] for k,v in info['terminal_observation'].items()}
                        with torch.no_grad():value=m.policy.predict_values(obs).item()
                        inner.test.assertAlmostEqual(float(b.rewards[1,0]),float(inner.locals['rewards'][0])+m.gamma*value,places=4)
                        inner.test.assertEqual(int(b.lengths[b.frame_pos,0]),1)
                        inner.test.assertEqual(int(b.lengths[b.frame_pos,1]),3)
                    return True
            check=Check();check.test=self;check.init_callback(m)
            with patch.object(c,'advance',forced):m.collect_rollouts(env,check,m.rollout_buffer,4)
            self.assertEqual(float(m.rollout_buffer.episode_starts[2,0]),1)
            self.assertEqual(float(m.rollout_buffer.episode_starts[2,1]),0)
        finally:env.close()

    def test_single_frame_history_and_frozen_weight_guard(self):
        env=GpuFrameVecEnv(2,history=1,capacity=128,threads=2,chunks=2)
        try:
            m=model_for(env,4);_,cb=m._setup_learn(8,Callback());b=m.rollout_buffer
            m.collect_rollouts(env,cb,b,4)
            torch.testing.assert_close(b.episode_starts[0],torch.ones(2,device='cuda'))
            torch.testing.assert_close(b.episode_starts[1:],torch.zeros((3,2),device='cuda'))
            class Mutator(Callback):
                def _on_step(inner):
                    with torch.no_grad():next(m.policy.parameters()).add_(0.01)
                    return True
            bad=Mutator();bad.init_callback(m)
            with self.assertRaisesRegex(RuntimeError,'weights changed'):m.collect_rollouts(env,bad,b,4)
        finally:env.close()

    def test_gpu_default_memory_and_minibatch(self):
        import io,json,train_sim
        from contextlib import redirect_stdout
        output=io.StringIO()
        with patch('sys.argv',['train_sim.py','--envs','128','--n-steps','128']),redirect_stdout(output):train_sim.main()
        c=json.loads(output.getvalue().split('PREPARED ONLY:')[0])
        self.assertEqual(c['batch_size'],32);self.assertEqual(c['pipeline'],'gpu')
        self.assertAlmostEqual(c['rollout_observation_gib'],1.66278076171875)
        self.assertEqual(c['reward_profile'],'combat-v1');self.assertEqual(c['gamma'],0.999)

    def test_segment_minibatches_match_window_path(self):
        env=GpuFrameVecEnv(3,seed=17,threads=2,chunks=1,history=8,capacity=128)
        try:
            m=model_for(env,8);m.segment_length=4;_,cb=m._setup_learn(10**6,Callback())
            for _ in range(3):self.assertTrue(m.collect_rollouts(env,cb,m.rollout_buffer,8))
            b=m.rollout_buffer;H=b.history
            # Episode starts inside segments and inside the retained prefix (both paths read lengths).
            for worker,row in ((1,H-1+5),(2,H-1)):
                for k,r in enumerate(range(row,b.buffer_size+H)):b.lengths[r,worker]=min(k+1,H)
            workers=torch.tensor([0,1,2,0,1,2],device='cuda');t0=torch.tensor([0,4,0,4,0,4],device='cuda')
            m.policy.set_training_mode(True)
            v,lp,ent,t,w=m.evaluate_segments(workers,t0)
            d=b._get_samples(w*b.buffer_size+t)
            v2,lp2,ent2=m.policy.evaluate_actions(d.observations,d.actions,action_masks=d.action_masks)
            torch.testing.assert_close(v,v2.flatten(),rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(lp,lp2,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(ent,ent2,rtol=1e-4,atol=1e-5)
            grads=[]
            for loss in ((v*1.3+lp*0.7+ent).sum(),(v2.flatten()*1.3+lp2*0.7+ent2).sum()):
                m.policy.optimizer.zero_grad();loss.backward()
                grads.append({n:p.grad.clone() for n,p in m.policy.named_parameters() if p.grad is not None})
            self.assertEqual(grads[0].keys(),grads[1].keys())
            for n in grads[0]:torch.testing.assert_close(grads[0][n],grads[1][n],rtol=1e-3,atol=2e-5,msg=n)
        finally:env.close()

    def test_segment_training_updates_with_accumulation(self):
        env=GpuFrameVecEnv(4,seed=23,threads=2,chunks=1,history=8,capacity=128)
        try:
            m=model_for(env,8);m.segment_length=4;m.micro_batch_size=4;m.batch_size=16;m.n_epochs=2;m.ent_coef=0.01
            _,cb=m._setup_learn(10**6,Callback());self.assertTrue(m.collect_rollouts(env,cb,m.rollout_buffer,8))
            weights=[p.detach().clone() for p in m.policy.parameters()]
            m.train()
            self.assertTrue(any(not torch.equal(a,p) for a,p in zip(weights,m.policy.parameters())))
            logged=m.logger.name_to_value
            self.assertEqual(logged['train/optimizer_steps'],4)  # 8 segments / 4 per batch x 2 epochs
            self.assertTrue(0<=logged['train/clip_fraction']<=1 and np.isfinite(logged['train/approx_kl']))
            self.assertEqual(m.policy_version,1)
            m.micro_batch_size=3
            with self.assertRaisesRegex(ValueError,'segment_length'):m.train()
        finally:env.close()

    def test_training_start_randomisation_is_seeded_and_evaluation_full_hp(self):
        from isaac_bridge.gpu_env import sample_start,FULL_START
        from isaac_bridge.sim_vec import RustBatch
        cfg=dict(boss_hp_prob=0.5,boss_hp_min=0.1,player_hp_prob=0.25,player_hp_min=3)
        self.assertEqual(sample_start(123,None),FULL_START)
        starts=[sample_start(s,cfg) for s in range(4000)]
        self.assertEqual(starts,[sample_start(s,cfg) for s in range(4000)])
        players=np.array([p for p,_ in starts]);bosses=np.array([b for _,b in starts])
        self.assertTrue(set(players)<={3.0,4.0,5.0,6.0} and 0.2<(players<6).mean()<0.3)
        self.assertTrue(((bosses>=0.1)&(bosses<=1)).all() and 0.45<(bosses<1).mean()<0.55)
        batch=RustBatch(1,0,1,reward_profile='combat-v1')
        try:
            batch.reset(0,77,3.0,0.4);f=batch.observe()
            boss=int(np.flatnonzero(f['entity_kind'][0,:,0]==20)[0])
            self.assertAlmostEqual(float(f['player'][0,6])*6,3.0,places=5)
            self.assertAlmostEqual(float(f['entities'][0,boss,15])*250,100.0,places=3)
            with self.assertRaisesRegex(ValueError,'Start HP'):batch.reset(0,77,0.0,1.0)
        finally:batch.close()
        for randomised in (True,False):
            env=GpuFrameVecEnv(8,seed=40,threads=2,chunks=1,history=8,capacity=128,
                               start_randomization=cfg if randomised else None)
            try:
                env.reset();c=env.chunks[0];frames=c.slots[0].frames
                expected=[sample_start(s,cfg) if randomised else FULL_START for s in c.seeds]
                self.assertEqual(c.starts,expected)
                for i,(player,boss_frac) in enumerate(expected):
                    boss=int(np.flatnonzero(frames['entity_kind'][i,:,0]==20)[0])
                    self.assertAlmostEqual(float(frames['player'][i,6])*6,player,places=5)
                    self.assertAlmostEqual(float(frames['entities'][i,boss,15]),boss_frac,places=4)
                if randomised:self.assertNotEqual(expected,[FULL_START]*8)
                # An episode end reports the start it used, and the next seed gets its own start.
                slot=c.slots[0];slot.actions.zero_();slot.action_ready.record();torch.cuda.synchronize()
                used=c.starts[0];real_observe=c.batch.observe;calls=[0]
                def observe(out=None):
                    result=real_observe(out);calls[0]+=1
                    if calls[0]==1:result['done'][0]=1  # pretend worker 0 just finished
                    return result
                with patch.object(c.batch,'observe',side_effect=observe):_,dones,infos=c.advance(slot)
                self.assertTrue(dones[0])
                self.assertEqual(infos[0]['episode_start'],{'player_hp':used[0],'boss_hp_fraction':used[1]})
                self.assertEqual(c.seeds[0],40+8)
                self.assertEqual(c.starts[0],sample_start(48,cfg) if randomised else FULL_START)
            finally:env.close()

    def test_training_on_a_rollout_copy_matches_the_rollout(self):
        # copy_rollout must carry everything segment training reads (async_training trains on it).
        import copy
        env=GpuFrameVecEnv(3,seed=29,threads=2,chunks=1,history=8,capacity=128)
        try:
            m=model_for(env,8);m.segment_length=4;m.micro_batch_size=4;m.batch_size=12
            _,cb=m._setup_learn(10**6,Callback())
            for _ in range(2):self.assertTrue(m.collect_rollouts(env,cb,m.rollout_buffer,8))
            b=m.rollout_buffer
            snapshot=m.rollout_buffer_class(m.n_steps,m.observation_space,m.action_space,device=m.device,
                gamma=m.gamma,gae_lambda=m.gae_lambda,n_envs=m.n_envs)
            m.copy_rollout(b,snapshot)
            start={k:v.clone() for k,v in m.policy.state_dict().items()}
            optimizer=copy.deepcopy(m.policy.optimizer.state_dict())
            results=[]
            for buffer in (b,snapshot):
                m.policy.load_state_dict(start);m.policy.optimizer.load_state_dict(optimizer)
                torch.manual_seed(5)
                m._train_segments(buffer)
                results.append([p.detach().clone() for p in m.policy.parameters()])
            for x,y in zip(*results):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6)
            self.assertTrue(any(not torch.equal(x,start[n]) for x,(n,_) in zip(results[0],m.policy.named_parameters())))
        finally:env.close()

    def test_proximal_objective_without_lag_is_ppo(self):
        import copy
        env=GpuFrameVecEnv(3,seed=31,threads=2,chunks=1,history=8,capacity=128)
        try:
            m=model_for(env,8);m.segment_length=4;m.micro_batch_size=4;m.batch_size=12
            _,cb=m._setup_learn(10**6,Callback())
            for _ in range(2):self.assertTrue(m.collect_rollouts(env,cb,m.rollout_buffer,8))
            b=m.rollout_buffer
            # The collecting weights reproduce the recorded log-probabilities.
            torch.testing.assert_close(m.proximal_log_probs(b),b.log_probs,rtol=1e-4,atol=1e-5)
            start={k:v.clone() for k,v in m.policy.state_dict().items()}
            optimizer=copy.deepcopy(m.policy.optimizer.state_dict())
            results=[]
            for proximal in (None,b.log_probs.clone()):
                m.policy.load_state_dict(start);m.policy.optimizer.load_state_dict(optimizer)
                torch.manual_seed(5)
                m._train_segments(b,proximal)
                results.append([p.detach().clone() for p in m.policy.parameters()])
            for x,y in zip(*results):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
            self.assertEqual(m.logger.name_to_value['train/lag_weight_truncated'],0.0)
            self.assertAlmostEqual(m.logger.name_to_value['train/lag_kl'],0.0,places=6)
        finally:env.close()

    def test_async_learn_trains_while_the_actor_collects(self):
        env=GpuFrameVecEnv(2,history=8,capacity=128,threads=2,chunks=2)
        try:
            m=model_for(env,4);m.segment_length=4;m.micro_batch_size=4;m.async_training=True
            seen=[]
            class Session:
                def after_update(inner,model):
                    same=all(torch.equal(a,p) for a,p in zip(model.actor.parameters(),model.policy.parameters()))
                    seen.append((model.policy_version,model._collecting,same))
            m.session=Session()
            m.learn(total_timesteps=24)  # 3 rollouts of 2 envs x 4 steps, each one trained
            self.assertEqual(m.num_timesteps,24)
            self.assertEqual(m.policy_version,3)
            # Each update is joined at a rollout boundary, after the actor (update k-1) collected.
            self.assertEqual(seen,[(1,False,False),(2,False,False),(3,False,False)])
            self.assertIs(m._gpu_sampler.policy,m.actor)
            self.assertIsNone(m.actor.optimizer);self.assertTrue(m.policy.optimizer.state)
            self.assertFalse(any(p.requires_grad for p in m.actor.parameters()))
            self.assertEqual(m.logger.name_to_value.get('train/optimizer_steps'),2)
            self.assertTrue(np.isfinite(m.logger.name_to_value['train/lag_kl']))
            self.assertEqual(m.logger.name_to_value['train/lag_weight_truncated'],0.0)
            with tempfile.TemporaryDirectory() as d:
                m.save(Path(d)/'m.zip')
                loaded=GpuMaskablePPO.load(Path(d)/'m.zip',env=env,device='cuda')
                self.assertIsNone(loaded.actor);self.assertFalse(loaded.async_training)
                for a,p in zip(loaded.policy.parameters(),m.policy.parameters()):self.assertTrue(torch.equal(a,p))
        finally:env.close()

    def test_sb3_learn_two_bounded_updates(self):
        env=GpuFrameVecEnv(2,history=8,capacity=128,threads=2,chunks=2)
        try:
            m=model_for(env,4)
            m.learn(total_timesteps=16)
            self.assertEqual(m.num_timesteps,16)
            self.assertEqual(m.policy_version,2)
            self.assertEqual(m._gpu_sampler.prefix_rebuilds,1)
        finally:env.close()


if __name__=='__main__':unittest.main()
