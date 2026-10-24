"""Persistent process VJP matches the tested in-process pipeline across versions."""
import argparse,copy,gc,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch,SourceView
from isaac_bridge.source_pipeline import SourcePipeline


def run():
    p=argparse.ArgumentParser();p.add_argument('--learner',default='cpu');p.add_argument('--sources',default='cpu,cpu')
    p.add_argument('--queries',type=int,default=8);p.add_argument('--bf16',action='store_true')
    p.add_argument('--defer',action='store_true')
    p.add_argument('--shared',action='store_true')
    p.add_argument('--saved-gpu-gb',type=float,default=0)
    p.add_argument('--saved-cpu-gb',type=float,default=1)
    p.add_argument('--saved-8bit',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(95)
    model=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
        temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
        rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).to(a.learner).eval()
    models=[model,copy.deepcopy(model)]
    pipes=[SourcePipeline(m,a.sources.split(','),period=0,micro=8,update_rollouts=2,
                         amp=a.bf16,flash_source=a.bf16 and a.learner.startswith('cuda'),compile_source=a.bf16 and a.learner.startswith('cuda'),
                         shared_devices=a.sources.split(',') if a.shared else (),
                         saved_gpu_gb=a.saved_gpu_gb if i else 0,
                         saved_cpu_gb=a.saved_cpu_gb,
                         saved_8bit=a.saved_8bit,
                         processes=bool(i),defer_backward=a.defer) for i,m in enumerate(models)]
    for replicas in pipes[1].versions.values():
        assert len({p.untyped_storage().data_ptr() for p in replicas[0].parameters() if not p.is_meta})==1
    opts=[torch.optim.Adam(m.main_parameters(),lr=1e-4,eps=1e-5) for m in models]
    rng=np.random.default_rng(95);hs=[RawHistory.empty(),RawHistory.empty()];batches=[]
    for t,rows in enumerate(TS.make_seq(rng,nw=2,T=24 if a.defer else 20,changes={(0,4),(1,8)},resets=set(),exits=set())):
        if os.environ.get('ISAAC_SOURCE_COMPONENT_GRAPH_CACHE')=='1':
            if t==0:static={field:rows[field].copy() for field in ('grid','map','patch')}
            else:
                for field,value in static.items():rows[field]=value
        pd=M.synth_pending(rng,2,model.heads)
        if t in (7,15,19,23):batches.append([(hs[w],rows[w],pd[w]) for w in range(2) if w==0 or t==7])
        for w in range(2):hs[w]=hs[w].append(rows[w],pd[w],4)
    worst=0.;encoded=[]
    for requests in batches:
        req=[requests[i%len(requests)] for i in range(a.queries)]
        for pipe,opt,m in zip(pipes,opts,models):
            pipe.poll_finish(opt,1e9,wait=not a.defer)
            pipe.prefetch(req,write_ahead=True)
            bank=SourceBatch(m,req,cache=False,pipeline=pipe)
            pipe.retain_histories({h.uid for h,_,_ in req},overlap=a.defer)
            if a.defer:
                pipe.dispatch_finish()
                pipe.poll_finish(opt,1e9,wait=True)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                view=SourceView(bank,np.arange(a.queries));out=view.forward(np.arange(a.queries),need_vg=True)
                (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
                view.backward();pipe.fast_step(opt,1e9)
            pipe.finish_rollout(opt,1e9,overlap=True)
            del view,out,bank
        for (name,x),(_,y) in zip(models[0].named_parameters(),models[1].named_parameters()):
            worst=max(worst,float((x-y).detach().abs().max()))
            torch.testing.assert_close(x,y,rtol=3e-3 if a.bf16 else 1e-4,atol=3e-4 if a.bf16 else 2e-6,msg=name)
    for pipe,opt in zip(pipes,opts):pipe.finish(opt,1e9);encoded.append(pipe.encoded_rows)
    for (name,x),(_,y) in zip(models[0].named_parameters(),models[1].named_parameters()):
        worst=max(worst,float((x-y).detach().abs().max()))
        torch.testing.assert_close(x,y,rtol=3e-3 if a.bf16 else 1e-4,atol=3e-4 if a.bf16 else 2e-6,msg=name)
    assert encoded==([32,32] if a.defer else [28,28]),encoded
    assert [pipe.version for pipe in pipes]==[2,2]
    offloaded=sum(e.get('saved_on_cpu',False) for e in pipes[1].events)
    if a.saved_gpu_gb:assert offloaded>0, 'CPU spill branch was not exercised'
    peak_cpu=max((m.get('saved_cpu_bytes',0) for m in pipes[1].remote_memory),default=0)
    assert peak_cpu<=a.saved_cpu_gb*2**30
    limited=sum(e.get('spill_limited_bytes',0) for e in pipes[1].events)
    if a.saved_cpu_gb<.001:assert limited>0, 'bounded spill branch was not exercised'
    gc.collect()
    for pipe in pipes:pipe.close()
    print(json.dumps(dict(result='PASS',queries=a.queries,source_bf16=a.bf16,saved_8bit_block=a.saved_8bit,weight_max_abs=worst,
        encoded_rows=encoded,source_updates=2,early_retirement=True,partial_checkpoint=True,deferred_until_reader_ready=a.defer,
        offloaded_encode_calls=offloaded,retained_cpu_max_bytes=peak_cpu,spill_limited_bytes=limited)))


if __name__=='__main__':run()
