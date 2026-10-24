"""Snapshot banks retain source values, not redundant fast-Reader projections."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import test_temporal_smoke as TS
import torch
from unittest.mock import patch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_actor import TensorActorMemory
torch.set_num_threads(2);torch.manual_seed(72)
m=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
active=TensorActorMemory(m,2,scratch=0);active.length[:]=[7,11]
active.u.normal_();active.mem.normal_();active.tail.normal_()
bank=active.new(960);bank.copy(0,17,active)
assert bank.kv is None and bank.mkv is None,'snapshot bank unnecessarily allocates Reader K/V'
u=bank.u[17].clone();memory=bank.mem[17,:2].clone()
with torch.no_grad():
    m.reader.layers[0].kv.weight.add_(.02);m.invalidate_source_cache()
    untouched=active.kv[1].clone()
    active.copy(17,0,bank)
    with patch.object(m,'encode_rebuild',wraps=m.encode_rebuild) as enc:
        active.refresh([0]);assert enc.call_count==0
    times=6-(6-torch.arange(4))%4
    expected=torch.stack([l.kv(u+m.reader.position(times,u.dtype)) for l in m.reader.layers],1)
    torch.testing.assert_close(active.kv[0],expected)
    mt=(torch.arange(2)//m.reader.tokens+1)*m.reader.block-1
    expected=torch.stack([l.kv(memory+m.reader.position(mt,memory.dtype)) for l in m.reader.layers],1)
    torch.testing.assert_close(active.mkv[0,:2],expected)
    torch.testing.assert_close(active.kv[1],untouched,rtol=0,atol=0)
    assert active.generations[1]!=m.cache_generation
    active.refresh([1]);assert active.generations[1]==m.cache_generation
torch.testing.assert_close(active.u[0],u,rtol=0,atol=0)
torch.testing.assert_close(active.mem[0,:2],memory,rtol=0,atol=0)
print('PASS 960-slot snapshot bank has no Reader K/V; restored slots reproject current Reader; source bytes unchanged; replay=0')
