"""Bounded checkpoint/resume and recorded held-out evaluation acceptance."""
import gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from isaac_bridge.gpu_env import GpuFrameVecEnv
from isaac_bridge.gpu_ppo import GpuMaskablePPO
from isaac_bridge.sim_vec import RustBatch
from isaac_bridge.transformer_policy import CombatTransformer
from isaac_bridge.training_session import TrainingSession,resume_model,rng_state,restore_rng,resolve_checkpoint
from isaac_bridge.evaluation import visible_snapshot


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TrainingSessionTest(unittest.TestCase):
    def test_partial_stop_reset_history_and_unpublished_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'runs') as tmp:
            out=Path(tmp);env=GpuFrameVecEnv(2,12,2,1,history=8,capacity=128);other=None
            try:
                model=GpuMaskablePPO('MultiInputPolicy',env,n_steps=8,batch_size=4,n_epochs=1,
                    seed=12,device='cuda',policy_kwargs=dict(features_extractor_class=CombatTransformer,
                    features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))
                session=TrainingSession(dict(episodes=1,checkpoint_every=10,eval_every=25),out)
                _,cb=model._setup_learn(4000,session)
                while model.collect_rollouts(env,cb,model.rollout_buffer,8):pass
                self.assertGreaterEqual(session.completed,1)
                checkpoint=session.save(model,'final')
                self.assertTrue(json.loads((checkpoint/'state.json').read_text())['partial_rollout_discarded'])
                with patch.object(model,'save',side_effect=OSError('injected interrupted write')):
                    with self.assertRaisesRegex(OSError,'interrupted'):session.save(model,'interrupted')
                self.assertEqual(resolve_checkpoint(out/'checkpoints/latest.json'),checkpoint)
                with self.assertRaisesRegex(ValueError,'Unpublished'):resolve_checkpoint(checkpoint.parent/('.'+checkpoint.name))
                other=GpuFrameVecEnv(2,12,2,1,history=8,capacity=128)
                resumed,state=resume_model(checkpoint,other)
                self.assertEqual(state['completed_episodes'],session.completed)
                self.assertEqual(env.chunks[0].actions,other.chunks[0].actions)
                np.testing.assert_array_equal(env.chunks[0].episodes,other.chunks[0].episodes)
                np.testing.assert_array_equal(env.chunks[0].returns,other.chunks[0].returns)
                before=model.rollout_buffer.window(model.rollout_buffer.frame_pos,[0,1])
                after=resumed.rollout_buffer.window(resumed.rollout_buffer.frame_pos,[0,1])
                for k in before:self.assertTrue(torch.equal(before[k],after[k]),k)
                self.assertTrue(torch.equal(model._gpu_sampler.starts,resumed._gpu_sampler.starts))
            finally:
                env.close()
                if other is not None:other.close()

    def test_periodic_resume_and_replay(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'runs') as tmp:
            root=Path(tmp);out=root/'first';out.mkdir()
            config=dict(episodes=1000,checkpoint_every=1,eval_every=1,eval_seeds=[2**31,2**31+1],
                        history=8,entity_capacity=128,schema='monstro-transformer-v3')
            env=GpuFrameVecEnv(2,91,2,2,history=8,capacity=128)
            env2=None
            try:
                model=GpuMaskablePPO('MultiInputPolicy',env,n_steps=8,batch_size=4,n_epochs=1,
                    seed=91,device='cuda',policy_kwargs=dict(features_extractor_class=CombatTransformer,
                        features_extractor_kwargs=dict(features_dim=64,layers=1,heads=4),normalize_images=False))
                session=TrainingSession(config,out);model.session=session
                model.learn(16,callback=session)
                checkpoint=resolve_checkpoint(out/'checkpoints/latest.json')
                self.assertEqual(session.updates,1)
                report=json.loads(next((out/'evaluations').glob('*/summary.json')).read_text())
                self.assertEqual(report['episodes'],2)
                for file in (out/'evaluations').glob('*/*.jsonl.gz'):
                    with gzip.open(file,'rt') as f:records=[json.loads(line) for line in f]
                    meta=records.pop(0)['metadata'];native=RustBatch(1,meta['seed'])
                    try:
                        self.assertEqual(records[0]['state'],visible_snapshot(native.states()[0])['state'])
                        for row in records[1:]:
                            native.step([row['action']])
                            self.assertEqual(visible_snapshot(native.states()[0]),{k:v for k,v in row.items() if k!='action'})
                        self.assertTrue(records[-1]['done'])
                        self.assertEqual(len(records)-1,report['results'][meta['seed']-2**31]['l'])
                    finally:native.close()
                    html=file.with_name(file.name.replace('.jsonl.gz','.html')).read_text(encoding='utf8')
                    self.assertNotIn('/*REPLAY_DATA*/',html);self.assertIn('requestAnimationFrame(tick)',html)
                saved_rng=rng_state()
                # Evaluation must leave all training RNG streams untouched.
                with gzip.open(checkpoint/'continuation.pt.gz','rb') as f:restore_rng(torch.load(f,weights_only=False)['rng'])
                expected=torch.rand(20,device='cuda');restore_rng(saved_rng)
                self.assertTrue(torch.equal(expected,torch.rand(20,device='cuda')))
                restore_rng(saved_rng)
                session.config.update(checkpoint_every=100,eval_every=100)
                _,cb=model._setup_learn(16,session,reset_num_timesteps=False)
                model.collect_rollouts(env,cb,model.rollout_buffer,8)
                actions=model.rollout_buffer.actions.clone();rewards=model.rollout_buffer.rewards.clone()
                states=[c.batch.states() for c in env.chunks]
                model.train();weights=[p.clone() for p in model.policy.parameters()]
                env2=GpuFrameVecEnv(2,91,2,2,history=8,capacity=128)
                resumed,state=resume_model(checkpoint,env2)
                resumed_out=root/'resumed';resumed_out.mkdir()
                counter=TrainingSession(config,resumed_out,state['completed_episodes'],state['updates'])
                resumed.session=counter
                _,cb=resumed._setup_learn(16,counter,reset_num_timesteps=False)
                resumed.collect_rollouts(env2,cb,resumed.rollout_buffer,8)
                self.assertTrue(torch.equal(actions,resumed.rollout_buffer.actions))
                self.assertTrue(torch.equal(rewards,resumed.rollout_buffer.rewards))
                self.assertEqual(states,[c.batch.states() for c in env2.chunks])
                resumed.train()
                for a,b in zip(weights,resumed.policy.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                self.assertEqual(resumed.num_timesteps,32);self.assertEqual(counter.updates,2)
                self.assertEqual(resumed.policy.optimizer.state_dict()['state'].keys(),model.policy.optimizer.state_dict()['state'].keys())
                print('VERIFIED periodic checkpoint + 2 complete held-out replays + identical resumed actions/rewards/worlds + optimizer continuation')
            finally:
                env.close()
                if env2 is not None:env2.close()


if __name__=='__main__':unittest.main()
