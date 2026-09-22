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
        self.assertAlmostEqual(c['rollout_observation_gib'],1.662689208984375)

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
