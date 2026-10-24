"""The H-free path must not allocate or copy the legacy H/KV snapshot bank."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import test_temporal_smoke as TS
import torch
from isaac_bridge.tok_policy import TokPolicy,TemporalState
m=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=256,temporal_layers=4,temporal_heads=4,temporal_window=256,
    temporal_rooms=128,rebuild_memory=True,memory_block=64,memory_tokens=4)
# Meta tensors verify the real 960-slot layout without allocating gigabytes.
state=TemporalState(m,960,'meta',torch.float32)
print('legacy KV bytes allocated by rebuild snapshot layout:',state.kv.numel()*state.kv.element_size())
assert state.kv.numel()==0,'H-free state still allocates the obsolete temporal KV bank'
m.rebuild_memory=False
legacy=TemporalState(m,3,'cpu',torch.float32)
assert legacy.kv.shape==(4,2,4,4,384,64)
legacy.kv.normal_();expected=legacy.kv[:,:,0].clone();legacy.copy_slot_(0,1)
torch.testing.assert_close(legacy.kv[:,:,1],expected)
print('PASS rebuilt state has no legacy H/KV allocation; legacy state shape and slot copy unchanged')
