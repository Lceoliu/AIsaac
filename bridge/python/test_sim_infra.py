"""Binary/JSON oracle, ring ownership and lossless PPO buffer tests; no training."""
import unittest
from collections import deque
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.sim_obs import rust_visible_observation
from isaac_bridge.transformer_obs import VisibleHistory
from isaac_bridge.history_buffer import HistoryRolloutBuffer


class BinaryObservationTest(unittest.TestCase):
    def test_all_fields_match_json_reference_with_resets(self):
        env=SimVecEnv(4,seed=812,history=64)
        refs=[VisibleHistory() for _ in range(4)]
        rng=np.random.default_rng(71)
        try:
            obs=env.reset()
            before={k:v.copy() for k,v in obs.items()}
            original=obs
            for step in range(240):
                states=env.batch.states()
                for i,ref in enumerate(refs):
                    expected=ref.append(rust_visible_observation(states[i]['state']))
                    for key in expected:
                        np.testing.assert_allclose(obs[key][i],expected[key],rtol=1e-6,atol=2e-7,
                            err_msg=f'step={step} worker={i} field={key}')
                actions=np.column_stack([rng.integers(0,45,4),np.full(4,step==1),np.zeros(4)]).astype(np.int32)
                for ref,action in zip(refs,actions):ref.previous_action[:]=*action,1
                obs,_,done,_=env.step(actions)
                for i in np.flatnonzero(done):refs[i].clear()
            for k,v in before.items():np.testing.assert_array_equal(original[k],v)
        finally:env.close()

    def test_seed_override_and_overflow(self):
        env=SimVecEnv(2,seed=1,history=8,capacity=1)
        try:
            env.seed(999);env.reset();self.assertEqual(env.base_seed,999)
            with self.assertRaisesRegex(ValueError,'overflow'):
                for _ in range(50):env.step(np.tile([1,0,0],(2,1)))
        finally:env.close()


class CompactBufferTest(unittest.TestCase):
    def test_dryrun_128_memory_arithmetic_does_not_overflow_int32(self):
        import io,json
        from contextlib import redirect_stdout
        from unittest.mock import patch
        import train_sim
        output=io.StringIO()
        with patch('sys.argv',['train_sim.py','--envs','128','--n-steps','128']),redirect_stdout(output):
            train_sim.main()
        config=json.loads(output.getvalue().split('PREPARED ONLY:')[0])
        self.assertEqual(config['legacy_rollout_observation_gib'],70.93359375)
        self.assertAlmostEqual(config['rollout_observation_gib'],1046928896/2**30)

    def test_reconstructs_shuffled_windows_gae_and_rollout_prefix(self):
        h,n,t=8,3,17
        space=VisibleHistory(h,5).space
        action_space=spaces.MultiDiscrete([45,2,2])
        compact=HistoryRolloutBuffer(t,space,action_space,device='cpu',n_envs=n)
        dense=MaskableDictRolloutBuffer(t,space,action_space,device='cpu',n_envs=n)
        rng=np.random.default_rng(19);histories=[deque(maxlen=h) for _ in range(n)]
        torch.manual_seed(9)
        for rollout in range(3):
            compact.reset();dense.reset();saved=[]
            for step in range(t):
                obs={k:np.zeros((n,*s.shape),s.dtype) for k,s in space.spaces.items()}
                starts=np.zeros(n,bool)
                for e in range(n):
                    if (step+e*3)%11==0:histories[e].clear();starts[e]=True
                    frame={k:rng.integers(0,2,size=s.shape[1:]).astype(s.dtype) for k,s in space.spaces.items()}
                    frame['history_mask'][...]=1
                    frame['entities']=rng.normal(size=(5,31)).astype(np.float32)
                    frame['player_anim'][:]=255;frame['entity_kind'][:]=65535
                    histories[e].append(frame)
                    for j,f in enumerate(histories[e]):
                        for k in obs:obs[k][e,j]=f[k]
                saved.append(obs)
                actions=np.column_stack([np.full(n,step),np.arange(n),np.zeros(n)])
                kwargs=dict(obs=obs,action=actions,reward=rng.normal(size=n),episode_start=starts,
                    value=torch.randn(n),log_prob=torch.randn(n),action_masks=rng.integers(0,2,(n,49)))
                compact.add(**kwargs);dense.add(**kwargs)
            for buf in (compact,dense):buf.compute_returns_and_advantage(torch.ones(n),np.zeros(n,bool))
            np.testing.assert_array_equal(compact.advantages,dense.advantages)
            np.testing.assert_array_equal(compact.returns,dense.returns)
            for _ in range(3):
                for sample in compact.get(5):
                    for j,(step,e,_) in enumerate(sample.actions.int().tolist()):
                        for key in sample.observations:
                            np.testing.assert_array_equal(sample.observations[key][j],saved[step][key][e])
                        np.testing.assert_array_equal(sample.action_masks[j],dense.action_masks[step,e])
                        self.assertEqual(sample.old_values[j],dense.values[step,e])
            self.assertLess(compact.observation_storage_bytes,sum(v.nbytes for v in dense.observations.values())/4)

    def test_sb3_collects_without_optimizer_and_retains_terminal_windows(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from isaac_bridge.transformer_policy import CombatTransformer
        class Callback(BaseCallback):
            def _on_step(self):return True
        torch.set_num_threads(2)
        env=SimVecEnv(2,history=8,capacity=128)
        try:
            model=MaskablePPO('MultiInputPolicy',env,n_steps=16,batch_size=4,device='cpu',
                rollout_buffer_class=HistoryRolloutBuffer,
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                    features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))
            weights=[p.detach().clone() for p in model.policy.parameters()]
            _,callback=model._setup_learn(32,Callback())
            self.assertTrue(model.collect_rollouts(env,callback,model.rollout_buffer,16))
            self.assertTrue(model.rollout_buffer.full)
            for sample in model.rollout_buffer.get(4):
                with torch.no_grad():
                    values,log_prob,entropy=model.policy.evaluate_actions(sample.observations,sample.actions,sample.action_masks)
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(torch.isfinite(log_prob).all())
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(weights,model.policy.parameters())))
        finally:env.close()


if __name__=='__main__':unittest.main()
