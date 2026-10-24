"""Bounded full-model collector benchmark; no training unless --update-probe.

The optional update probe runs only two minibatches of an isolated test model.
It never saves a trained checkpoint or starts an episode-budget training job.
"""
import argparse,gc,itertools,json,time
from pathlib import Path
from unittest.mock import patch
import psutil,torch
from stable_baselines3.common.callbacks import BaseCallback
from isaac_bridge.gpu_env import GpuFrameVecEnv
from isaac_bridge.gpu_ppo import GpuMaskablePPO
from isaac_bridge.transformer_policy import CombatTransformer


class Callback(BaseCallback):
    def _on_step(self):return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs',type=int,default=128);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--chunks',type=int,nargs='+',default=[1,2,4]);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--update-probe',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    report=dict(gpu=torch.cuda.get_device_name(),history=64,capacity=256,layers=4,
                minibatch=32,formal_training=False,results=[])
    process=psutil.Process()
    for chunks in a.chunks:
        env=GpuFrameVecEnv(a.envs,seed=812,threads=a.threads,chunks=chunks)
        try:
            m=GpuMaskablePPO('MultiInputPolicy',env,n_steps=a.steps,batch_size=32,n_epochs=1,device='cuda',seed=42,
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                    features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
                    net_arch=dict(pi=[256],vf=[256]),normalize_images=False))
            _,cb=m._setup_learn(a.envs*a.steps,Callback());b=m.rollout_buffer
            weights=[x.detach().clone() for x in m.policy.parameters()]
            m.collect_rollouts(env,cb,b,a.steps)  # Warm history and kernels.
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            s=m._gpu_sampler;before=(s.upload_bytes,s.encoded_frames)
            start=time.perf_counter();cpu_start=sum(process.cpu_times()[:2])
            m.collect_rollouts(env,cb,b,a.steps);torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            row=dict(envs=a.envs,chunks=chunks,steps=a.steps,seconds=seconds,
                decisions_per_s=a.envs*a.steps/seconds,batch_ms=seconds/a.steps*1000,
                cpu_core_equivalents=(sum(process.cpu_times()[:2])-cpu_start)/seconds,
                rss_gib=process.memory_info().rss/2**30,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_vram_gib=torch.cuda.max_memory_reserved()/2**30,
                raw_rollout_gib=b.observation_storage_bytes/2**30,
                upload_bytes=s.upload_bytes-before[0],encoded_frames=s.encoded_frames-before[1],
                valid_history_mean=b.lengths[b.frame_pos].float().mean().item(),optimizer_steps=0)
            assert all(torch.equal(x,y) for x,y in zip(weights,m.policy.parameters()))
            row['sampling_weights_unchanged']=True
            # Compare cached heads with a fresh raw-window forward for every worker.
            with torch.no_grad():
                ids=torch.arange(a.envs,device='cuda');raw=b.window(b.frame_pos,ids)
                ca,cv,cl,mask=s.action(b.frame_pos,ids,True)
                ra,rv,rl=m.policy(raw,deterministic=True,action_masks=mask)
                row['cache_value_max_abs']=(cv-rv.flatten()).abs().max().item()
                row['cache_logprob_max_abs']=(cl-rl).abs().max().item()
                row['cache_action_equal']=torch.equal(ca,ra)
                # Keep failed precision gates as evidence, not just successful timings.
                a.out.parent.mkdir(parents=True,exist_ok=True)
                a.out.write_text(json.dumps({**report,'pending_result':row},indent=2),encoding='utf8')
                torch.testing.assert_close(cv,rv.flatten(),atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(cl,rl,atol=2e-5,rtol=2e-5)
                assert row['cache_action_equal']
                del raw,ca,cv,cl,mask,ra,rv,rl
            if a.update_probe:
                original=b.get
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                with patch.object(b,'get',lambda size:itertools.islice(original(size),2)):m.train()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                row.update(optimizer_steps=2,update_probe_seconds=elapsed,
                    update_probe_samples_per_s=64/elapsed,update_peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
            report['results'].append(row)
            a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf8')
            print(json.dumps(row),flush=True)
        finally:env.close()
        if a.update_probe:del original
        del m,b,s,env,weights,cb;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
