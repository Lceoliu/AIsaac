"""A delayed source packet cannot be changed by the next Reader interval."""
import copy,sys,threading,argparse
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch,SourceView
from isaac_bridge.source_pipeline import SourcePipeline,source_parameter
p=argparse.ArgumentParser();p.add_argument('--learner',default='cpu');p.add_argument('--sources',default='cpu,cpu')
p.add_argument('--queries',type=int,default=4);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(91)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg).to(a.learner).eval();ref=copy.deepcopy(model)
pipe=SourcePipeline(model,a.sources.split(','),period=2,micro=8)
opt=torch.optim.Adam(model.main_parameters(),lr=1e-4,eps=1e-5)
fast=torch.optim.Adam([v for n,v in ref.named_parameters() if not source_parameter(n)],lr=1e-4,eps=1e-5)
slow=torch.optim.Adam([v for n,v in ref.named_parameters() if source_parameter(n)],lr=1e-4,eps=1e-5)
rng=np.random.default_rng(91);hs=[RawHistory.empty(),RawHistory.empty()];req=[]
for t,rows in enumerate(TS.make_seq(rng,nw=2,T=16,changes={(0,4),(1,8)},resets=set(),exits=set())):
 pd=M.synth_pending(rng,2,model.heads)
 for w in range(2):
  if t in (11,15):req.append((hs[w],rows[w],pd[w]))
  hs[w]=hs[w].append(rows[w],pd[w],4)
req=[req[i%len(req)] for i in range(a.queries)]
with torch.no_grad():resident=SourceBatch(model,req,cache=False,pipeline=pipe)
early=pipe.chunks[0]['future'].result()['u'];early.retain_grad()
packets=[];event=threading.Event();backward=pipe._backward
def held(worker,packet):
 event.wait();return backward(worker,packet)
with patch.object(pipe,'_backward',side_effect=held):
 for step in range(3):
  model.zero_grad();ref.zero_grad()
  bank=SourceView(resident,np.arange(len(req)));out=bank.forward(np.arange(len(req)),need_vg=True)
  dense=SourceBatch(ref,req,cache=False);expected=dense.forward(np.arange(len(req)),need_vg=True)
  torch.testing.assert_close(out[0],expected[0],rtol=5e-4,atol=3e-5)
  (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward();bank.backward()
  (expected[0].square().mean()+expected[1].square().mean()+expected[-1].square().mean()).backward()
  packets.append({n:v.grad.clone() for n,v in ref.named_parameters() if source_parameter(n) and v.grad is not None})
  pipe.fast_step(opt,1e9);fast.step()
  if step==1:
   frozen=pipe.chunks[0]['future'].result()['grads']
   saved={k:v.clone() for k,v in frozen.items()}
   pipe.start_finish(opt,1e9)
   assert pipe.pending is not None and pipe.version==0 and pipe.steps==0
  if step==2:
   assert pipe.version==0 and pipe.steps==1
   assert pipe.chunks[0]['future'].result()['grads'] is not frozen
   for k,v in saved.items():torch.testing.assert_close(frozen[k],v,rtol=0,atol=0)
 event.set();pipe.poll_finish(opt,1e9,wait=True)
 assert pipe.version==1 and pipe.steps==1
 pipe.finish(opt,1e9)
for group in (packets[:2],packets[2:]):
 slow.zero_grad(set_to_none=True)
 for name,param in ref.named_parameters():
  if source_parameter(name) and name in group[0]:param.grad=sum(g[name] for g in group)/len(group)
 slow.step()
err=0.
for (name,value),(_,target) in zip(model.named_parameters(),ref.named_parameters()):
 err=max(err,float((value-target).detach().abs().max()))
 torch.testing.assert_close(value,target,rtol=2e-3,atol=5e-5,msg=name)
assert early.grad[0].abs().sum()>0
assert pipe.version==2 and pipe.fast_steps==3 and pipe.pending is None
assert pipe.encoded_rows==32
pipe.close()
print('PASS asynchronous source intervals match explicit delayed-Adam reference; frozen packets; no source replay;',
      'weight_max_abs=',err,'early_gradient=',float(early.grad[0].abs().sum()),'queries=',a.queries)
