"""Immutable BF16 values share storage; per-forward gradients still sum in FP32."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import torch
from isaac_bridge.source_pipeline import ImmutableSourceCast

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(129)
parameter=torch.nn.Parameter(torch.randn(64,64,device=a.device))
reference=torch.nn.Parameter(parameter.detach().clone())
cached=parameter.detach().to(torch.bfloat16)
shared=[];ordinary=[];pointers=set()
for i in range(128):
    x=torch.randn(7,64,device=a.device)
    with torch.autocast(torch.device(a.device).type,dtype=torch.bfloat16):
        alias=ImmutableSourceCast.apply(parameter,cached)
        pointers.add(alias.untyped_storage().data_ptr())
        shared.append(torch.nn.functional.linear(x,alias).float().square().mean())
        ordinary.append(torch.nn.functional.linear(x,reference).float().square().mean())
sum(shared).backward();sum(ordinary).backward()
assert len(pointers)==1
assert parameter.grad.dtype==torch.float32
torch.testing.assert_close(parameter.grad,reference.grad,rtol=0,atol=0)
print('PASS 128 forwards share one immutable BF16 weight storage; FP32 accumulated gradient exactly matches 128 independent autocasts')
