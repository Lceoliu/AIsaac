"""New source weights must reuse compiled component code, not its old values."""
import os,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.tok_obs import GRID
from isaac_bridge.source_pipeline import SourcePipeline

os.environ['ISAAC_SOURCE_POINTWISE_RECOMPUTE']='1'
os.environ['ISAAC_SOURCE_POINTWISE_COMPONENTS']='1'
torch.set_num_threads(2);torch.manual_seed(98)
device='cuda:0' if '--cuda' in sys.argv else 'cpu'
m=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).to(device)
pipe=SourcePipeline(m,[device],compile_source=1,amp=True)
x=torch.randn(16,*GRID,device=device)
counts=[];errors=[];times=[]
for version in range(3):
    if version:
        with torch.no_grad():
            for p in m.room.parameters():p.add_(.001)
        pipe.version=version;pipe.versions[version]=pipe._replicas()
    start=time.perf_counter()
    shape_counts=[]
    for n in (1,16,2,3,31,64):
        x=torch.randn(n,*GRID,device=device)
        replica=pipe.versions[version][0]
        with pipe._cast_context(replica),torch.autocast(device.split(':')[0],dtype=torch.bfloat16):
            out=replica.room(x)
        with torch.autocast(device.split(':')[0],dtype=torch.bfloat16):expected=m.room(x)
        errors.append(float((out-expected).detach().abs().max()))
        for _ in range(3):out.square().mean().backward(retain_graph=True)
        shape_counts.append(torch._dynamo.utils.counters['stats']['unique_graphs'])
    print('SHAPE_GRAPHS',version,shape_counts,flush=True)
    counts.append(torch._dynamo.utils.counters['stats']['unique_graphs']);times.append(time.perf_counter()-start)
print(json.dumps(dict(device=device,compiled_graph_counts=counts,output_max_abs=errors,seconds=times)),flush=True)
assert max(errors)<1e-5,errors
assert counts[0]>0,counts
assert len(set(counts))==1,counts
pipe.close()
print('PASS new source versions reuse compiled component code and use new weights')
