"""Serve new source rows inside an old VJP, without changing its update clock."""
import argparse,copy,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
import test_memory_smoke as M
import test_temporal_smoke as TS
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch,SourceView
from isaac_bridge.source_pipeline import SourcePipeline

def run():
    p=argparse.ArgumentParser();p.add_argument('--learner',default='cpu');p.add_argument('--sources',default='cpu,cpu')
    p.add_argument('--queries',type=int,default=64);p.add_argument('--bf16',action='store_true');a=p.parse_args()
    os.environ['ISAAC_SOURCE_PREEMPT_VJP']='1'
    torch.set_num_threads(2);torch.manual_seed(105)
    model=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
        temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
        rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).to(a.learner).eval()
    models=[model,copy.deepcopy(model)]
    pipes=[SourcePipeline(m,a.sources.split(','),period=0,micro=8,amp=a.bf16,
        compile_source=a.bf16,flash_source=a.bf16,processes=bool(i),defer_backward=not i)
        for i,m in enumerate(models)]
    rng=np.random.default_rng(105);hs=[RawHistory.empty() for _ in range(64)];requests=[]
    for t,rows in enumerate(TS.make_seq(rng,nw=64,T=12,changes={(w,8) for w in range(64)},resets=set(),exits=set())):
        pd=M.synth_pending(rng,64,model.heads)
        if t in (7,11):requests.append([(hs[w],rows[w],pd[w]) for w in range(64)])
        for w in range(64):hs[w]=hs[w].append(rows[w],pd[w],4)
    requests=[[rs[i%64] for i in range(a.queries)] for rs in requests]
    for i,(m,pipe) in enumerate(zip(models,pipes)):
        opt=torch.optim.Adam(m.main_parameters(),lr=1e-4,eps=1e-5)
        bank=SourceBatch(m,requests[0],cache=False,pipeline=pipe)
        view=SourceView(bank,np.arange(a.queries));out=view.forward(np.arange(a.queries),need_vg=True)
        (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
        view.backward();pipe.fast_step(opt,1e9);pipe.finish_rollout(opt,1e9,overlap=True)
        if i:
            assert all(remote.backward_started.wait(30) for remote in pipe.remotes)
        pipe.prefetch(requests[1][:1])
        bank=SourceBatch(m,requests[1],cache=False,pipeline=pipe)
        if i:bank_ready_during_vjp=any(not f.done() for f in pipe.pending[0])
        pipe.dispatch_finish();pipe.poll_finish(opt,1e9,wait=True)
        opt.zero_grad(set_to_none=True)
        view=SourceView(bank,np.arange(a.queries));out=view.forward(np.arange(a.queries),need_vg=True)
        (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
        view.backward();pipe.fast_step(opt,1e9);pipe.finish_rollout(opt,1e9,overlap=False)
        pipe.finish(opt,1e9)
        assert pipe.version==2 and pipe.encoded_rows==768,(pipe.version,pipe.encoded_rows)
    worst=0.
    for (name,x),(_,y) in zip(models[0].named_parameters(),models[1].named_parameters()):
        worst=max(worst,float((x-y).detach().abs().max()))
        torch.testing.assert_close(x,y,rtol=3e-3 if a.bf16 else 1e-4,atol=3e-4 if a.bf16 else 2e-6,msg=name)
    served=sum(e.get('serviced_forward_calls',0) for e in pipes[1].events)
    assert served>0, 'new source work never ran inside the old VJP'
    for pipe in pipes:pipe.close()
    assert bank_ready_during_vjp, 'Reader bank still waited for the complete old VJP'
    print(json.dumps(dict(result='PASS',queries=a.queries,weights_max_abs=worst,
                         serviced_forward_calls=served,bank_ready_during_vjp=bank_ready_during_vjp,
                         encoded_rows=768,source_updates=2,encoder_replay=0)))

if __name__=='__main__':run()
