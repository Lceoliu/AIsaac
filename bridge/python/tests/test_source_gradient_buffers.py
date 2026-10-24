"""Old source versions sum into one gradient buffer without updating unused Adam slots."""
import argparse,sys
from concurrent.futures import Future
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_temporal_smoke as TS
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.source_pipeline import SourcePipeline,source_parameter

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(43)
model=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).to(a.device)
pipe=SourcePipeline(model,[a.device],period=0)
packets=[];expected=torch.zeros_like(model.reader.inp.weight);pointers=[]
for version in range(4):
    pipe.version=version
    param=pipe.replicas[0].reader.inp.weight
    param.register_hook(lambda grad,p=param:pointers.append(0 if p.grad is None else p.grad.data_ptr()))
    output=(param.square()*(version+1)).sum()
    expected.add_(2*(version+1)*param.detach())
    f=Future();f.set_result(dict(u=output))
    packets.append((dict(worker=0,future=f),dict(u=torch.ones_like(output))))
    with torch.no_grad():model.reader.inp.weight.add_(.01)
grads=pipe._backward(0,packets)
torch.testing.assert_close(grads['reader.inp.weight'],expected)
assert set(grads)=={'reader.inp.weight'}, 'unused source parameters must keep Adam grad=None'
assert len(pointers)==4 and all(pointers) and len(set(pointers))==1, pointers
again=pipe._backward(0,packets)
torch.testing.assert_close(again['reader.inp.weight'],expected)
assert all(param.grad is None for replicas in pipe.versions.values() for replica in replicas for param in replica.parameters())
unused=sum(p.numel()*p.element_size() for replicas in pipe.versions.values() for replica in replicas
           for n,p in replica.named_parameters() if not source_parameter(n) and not p.is_meta)
assert unused==0, f'source snapshots retain {unused} bytes of unused Reader/head parameters'
pipe.close()
print('PASS four immutable versions share one source gradient buffer; exact sum; unused Adam slots absent; repeated VJP reset; no unused Reader/head storage')
