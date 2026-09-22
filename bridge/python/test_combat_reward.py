"""Finite-deadline objective, unchanged physics, countdown and weights migration."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.combat_reward import combat_v1_reward
from isaac_bridge.sim_vec import RustBatch,SimVecEnv
from isaac_bridge.gpu_env import GpuFrameVecEnv,decode_frame
from isaac_bridge.gpu_ppo import GpuMaskablePPO
from isaac_bridge.training_session import TrainingSession,warm_start_model,resume_model
from isaac_bridge.transformer_policy import CombatTransformer


class Callback(BaseCallback):
    def _on_step(self):return True


class RewardTest(unittest.TestCase):
    def test_recorded_win_and_deadline_have_exact_terminal_contract(self):
        fixtures=json.loads((Path(__file__).with_name('fixtures')/'combat_reward_episodes.json').read_text())
        for case in fixtures.values():
            native=RustBatch(1,case['seed']);new=RustBatch(1,case['seed'],reward_profile='combat-v1')
            try:
                for action in case['actions']:native.step([action]);new.step([action])
                a=native.observe()[0];b=new.observe()[0];record=new.states()[0]
                self.assertTrue(b['done']);self.assertEqual(b['truncated'],0)
                self.assertEqual(record['outcome'],case['outcome']);self.assertEqual(int(b['elapsed']),case['frames'])
                expected=float(a['reward'])+(3-case['frames']/3600 if case['outcome']=='win' else -1)
                self.assertAlmostEqual(float(b['reward']),expected,places=5)
                self.assertEqual(float(b['reward']),record['reward']);self.assertFalse(record['truncated'])
            finally:native.close();new.close()

    def test_terminal_rewards_no_living_or_death_penalty(self):
        actual=combat_v1_reward([0,-1,.064,1,1,1,0,-1],[0,1,0,2,2,2,3,3],
                               [150,150,150,0,1800,3600,3600,3600])
        np.testing.assert_allclose(actual,[0,-1,.064,4,3.5,3,-1,-2],atol=1e-6)

    def test_reward_profile_preserves_native_physics(self):
        old=RustBatch(2,38);new=RustBatch(2,38,reward_profile='combat-v1')
        try:
            rng=np.random.default_rng(20)
            for t in range(300):
                actions=np.column_stack([rng.integers(0,45,2),[int(t==2)]*2,[0,0]])
                old.step(actions);new.step(actions)
                a=old.observe().copy();b=new.observe().copy()
                for k in a.dtype.names:
                    if k not in ('reward','truncated'):np.testing.assert_array_equal(a[k],b[k],err_msg=k)
                np.testing.assert_array_equal(b['reward'],combat_v1_reward(a['reward'],a['outcome'],a['elapsed']))
                self.assertFalse(b['truncated'].any())
                for i in np.flatnonzero(a['done']):old.reset(int(i),int(t+i));new.reset(int(i),int(t+i))
        finally:old.close();new.close()


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DeadlineGpuTest(unittest.TestCase):
    def test_deadline_collector_does_not_bootstrap(self):
        from unittest.mock import patch
        env=GpuFrameVecEnv(2,12,2,1,history=8,capacity=128,reward_profile='combat-v1')
        try:
            model=GpuMaskablePPO('MultiInputPolicy',env,n_steps=4,batch_size=4,n_epochs=1,gamma=.999,
                device='cuda',policy_kwargs=dict(features_extractor_class=CombatTransformer,
                features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))
            _,cb=model._setup_learn(8,Callback());c=env.chunks[0];advance=c.advance;calls=0
            def deadline(slot):
                nonlocal calls
                rewards,dones,infos=advance(slot);calls+=1
                if calls==2:
                    rewards[0]=-1;dones[0]=True;slot.frames['reward'][0]=-1;slot.frames['done'][0]=1
                    slot.frames['truncated'][0]=0;slot.frames['time'][0]=120
                    infos[0].update({'TimeLimit.truncated':False,'episode':{'r':-1.,'l':2}})
                    c.batch.reset(0,99);c.batch.observe(slot.reset_frames)
                return rewards,dones,infos
            class Check(Callback):
                def _on_step(inner):
                    if calls==2:
                        b=model.rollout_buffer;terminal=inner.locals['infos'][0]['terminal_observation']
                        self.assertEqual(float(b.rewards[1,0]),-1.)
                        last=int(terminal['history_mask'].sum().item())-1
                        self.assertEqual(float(terminal['remaining_time'][last]),0.)
                        self.assertEqual(float(b.window(b.frame_pos,[0])['remaining_time'][0,0]),1.)
                    return True
            cb=Check();cb.init_callback(model)
            with patch.object(c,'advance',deadline):model.collect_rollouts(env,cb,model.rollout_buffer,4)
        finally:env.close()

    def test_migration_countdown_gpu_cpu_and_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'runs') as tmp:
            out=Path(tmp);legacy=GpuFrameVecEnv(2,91,2,1,history=8,capacity=128)
            new=GpuFrameVecEnv(2,91,2,1,history=8,capacity=128,reward_profile='combat-v1')
            cpu=SimVecEnv(2,91,2,history=8,capacity=128,reward_profile='combat-v1')
            restored_env=None
            kwargs=dict(n_steps=8,batch_size=4,n_epochs=1,seed=91,device='cuda',
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                    features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))
            try:
                old=GpuMaskablePPO('MultiInputPolicy',legacy,**kwargs)
                cfg=dict(episodes=100,checkpoint_every=100,eval_every=100,reward_profile='legacy')
                baseline=out/'baseline';baseline.mkdir();session=TrainingSession(cfg,baseline)
                old.learn(16,callback=session);session.updates=1;checkpoint=session.save(old,'update')
                model=GpuMaskablePPO('MultiInputPolicy',new,gamma=.999,**kwargs)
                record=warm_start_model(model,checkpoint)
                self.assertTrue(record['expanded_zero_columns']);self.assertFalse(model.policy.optimizer.state)
                sample=legacy.reset();obs={k:torch.as_tensor(v,device='cuda') for k,v in sample.items()}
                obs_new={**obs,'remaining_time':torch.ones((2,8),device='cuda')}
                old.policy.eval();model.policy.eval()
                with torch.no_grad():
                    torch.testing.assert_close(old.policy.features_extractor(obs),model.policy.features_extractor(obs_new),atol=2e-6,rtol=2e-5)
                expected=cpu.reset()
                class Oracle(Callback):
                    def _on_step(inner):
                        nonlocal expected
                        b=model.rollout_buffer
                        actual=b.window(b.frame_pos,[0,1])
                        expected,_,_,_=cpu.step(inner.locals['actions'].cpu().numpy())
                        for k in actual:np.testing.assert_allclose(actual[k].cpu(),expected[k],atol=1e-7,rtol=1e-6,err_msg=k)
                        return True
                model.learn(16,callback=Oracle())
                self.assertEqual(model.gamma,.999);self.assertTrue(model.policy.optimizer.state)
                self.assertGreater(model.policy.features_extractor.player[0].weight[:,-1].abs().sum().item(),0)
                folder=out/'new';folder.mkdir();cfg['reward_profile']='combat-v1'
                s=TrainingSession(cfg,folder);s.updates=1;cp=s.save(model,'update')
                restored_env=GpuFrameVecEnv(2,91,2,1,history=8,capacity=128,reward_profile='combat-v1')
                restored,_=resume_model(cp,restored_env)
                self.assertEqual(restored.gamma,.999)
                with self.assertRaisesRegex(ValueError,'Reward changed'):resume_model(checkpoint,restored_env)
                _,cb=restored._setup_learn(16,Callback(),reset_num_timesteps=False)
                self.assertTrue(restored.collect_rollouts(restored_env,cb,restored.rollout_buffer,8))
            finally:
                legacy.close();new.close();cpu.close()
                if restored_env is not None:restored_env.close()


if __name__=='__main__':unittest.main()
