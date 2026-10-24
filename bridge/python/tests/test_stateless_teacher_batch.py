"""Stateless teacher entries batch, keep identity until overwrite, and sum gradients."""
import argparse,copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch
from isaac_bridge.source_pipeline import SourcePipeline,source_parameter
p=argparse.ArgumentParser();p.add_argument('--learner',default='cpu');p.add_argument('--sources',default='cpu,cpu')
p.add_argument('--queries',type=int,default=64);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(17)
m=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).to(a.learner)
reference=copy.deepcopy(m);pipe=SourcePipeline(m,a.sources.split(','),period=0,micro=8192)
rng=np.random.default_rng(17);n=32
rows=TS.make_seq(rng,nw=n,T=1,changes=set(),resets=set(),exits=set())[0]
pending=M.synth_pending(rng,n,m.heads);hist=[RawHistory.empty() for _ in range(n)]
requests=[(hist[i%n],rows[i%n],pending[i%n]) for i in range(a.queries)]
stateless={h.uid for h in hist}
# New ring entries are queued before the historical VJP, not on their first
# randomly sampled Reader minibatch. Their prior and later reads share sources.
pipe.prefetch(requests,stateless=stateless)
with torch.inference_mode(False),torch.no_grad():
    prior=SourceBatch(m,requests,cache=False,pipeline=pipe,stateless=stateless)
with torch.inference_mode():prior.forward(np.arange(a.queries),need_vg=True)
assert pipe.encoded_rows==n
bank=SourceBatch(m,requests,cache=False,pipeline=pipe,stateless=stateless)
actual=bank.forward(np.arange(a.queries),need_vg=True)
expected=SourceBatch(reference,requests,cache=False).forward(np.arange(a.queries),need_vg=True)
torch.testing.assert_close(actual[0],expected[0],rtol=5e-4,atol=3e-5)
for out in (actual,expected): (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
bank.backward();opt=torch.optim.Adam(m.main_parameters(),lr=1e-4)
pipe.fast_step(opt,1e9);pipe.finish(opt,1e9)
error=0.
for name,param in m.named_parameters():
    target=dict(reference.named_parameters())[name]
    if source_parameter(name) and target.grad is not None:
        torch.testing.assert_close(param.grad,target.grad,rtol=2e-3,atol=5e-5,msg=name)
        error=max(error,float((param.grad-target.grad).abs().max()))
encodes=[e for e in pipe.events if e['kind']=='encode']
assert len(encodes)<=len(pipe.devices) and sum(e['rows'] for e in encodes)==n,encodes
assert pipe.encoded_rows==n
with torch.no_grad(): SourceBatch(m,requests,cache=False,pipeline=pipe,stateless=stateless)
assert pipe.encoded_rows==n,'same ring identities re-encoded after source update'
old=hist[0].uid;hist[0]=RawHistory.empty();rows[0]['player'][0]+=.2
replacement=[(hist[0],rows[0],pending[0])]
with torch.no_grad():SourceBatch(m,replacement,cache=False,pipeline=pipe,stateless={hist[0].uid})
assert pipe.encoded_rows==n+1
pipe.retain_histories({h.uid for h in hist})
assert (old,0) not in pipe.frames and (hist[0].uid,0) in pipe.frames
pipe.close()
print('PASS',n,'teacher identities in',len(encodes),'encoder batches; repeated queries=',a.queries,
      'source_grad_max_abs=',error,'; cross-cycle reuse; overwritten ring entry gets new identity')
