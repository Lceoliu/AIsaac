"""Changing player data cannot force an unchanged terrain patch to be encoded again."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy,to_batch
torch.set_num_threads(2);torch.manual_seed(32)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg);reference=copy.deepcopy(model);reference.source_cache=False
rng=np.random.default_rng(32)
rows=TS.make_seq(rng,nw=16,T=1,changes=set(),resets=set(),exits=set())[0]
rows['patch']=rows['patch'][0]
pending=M.synth_pending(rng,16,model.heads)
seen=[]
hook=model.patch.register_forward_pre_hook(lambda module,args:seen.append(len(args[0])))
out=[]
for m in (model,reference):
    b=to_batch(rows,pending,'cpu');b['_raw_rows']=rows;b['_raw_pending']=pending
    y,_,u=m.encode_rebuild(b);out.append((y,u))
    (y.square().mean()+u.square().mean()).backward()
hook.remove()
assert seen==[1],seen
for a,b in zip(out[0],out[1]):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
error=0.
for (name,a),(_,b) in zip(model.named_parameters(),reference.named_parameters()):
    if a.grad is not None:
        error=max(error,float((a.grad-b.grad).abs().max()))
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-3,atol=5e-5,msg=name)
print('PASS 16 distinct players share one patch CNN evaluation; uncached forward/gradient parity; max_grad_abs=',error)
