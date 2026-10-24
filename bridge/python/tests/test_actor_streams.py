"""A capacity change must preserve writes queued by the other actor stream."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_actor import TensorActorMemory
torch.set_num_threads(2)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg).cuda().eval()
state=TensorActorMemory(model,4)
slow=torch.cuda.Stream()
# Populate allocator blocks first: an incidental cudaMalloc synchronization
# must not hide a missing stream dependency in the steady-state growth path.
spare=[torch.empty((state.slots+state.scratch,128,32),device='cuda'),
       torch.empty((state.slots+state.scratch,128,2,64),device='cuda')]
torch.cuda.synchronize()
del spare
with torch.cuda.stream(slow):
    torch.cuda._sleep(80_000_000)
    state.mem[0,0].fill_(7.)
    state.mkv[0,0].fill_(9.)
state.grow(128)
torch.cuda.synchronize()
torch.testing.assert_close(state.mem[0,0],torch.full_like(state.mem[0,0],7),rtol=0,atol=0)
torch.testing.assert_close(state.mkv[0,0],torch.full_like(state.mkv[0,0],9),rtol=0,atol=0)
print('PASS growth preserves pending writes from the other actor CUDA stream')
