"""Fast Reader versions reproject immutable source values without re-encoding."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import test_temporal_smoke as TS
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_actor import TensorActorMemory
torch.set_num_threads(2);torch.manual_seed(61)
m=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True,memory_refresh_updates=8)
s=TensorActorMemory(m,1,scratch=0);s.length[0]=7
s.u.normal_();s.mem.normal_();before=s.u.clone();mem=s.mem.clone()
with torch.no_grad():
    m.reader.layers[0].kv.weight.add_(.02);m.invalidate_source_cache()
    assert m.source_generation==0 and m.cache_generation==1
    with patch.object(m,'encode_rebuild',wraps=m.encode_rebuild) as encoder:
        s.refresh();assert encoder.call_count==0
    times=6-(6-torch.arange(4))%4
    x=s.u[0]+m.reader.position(times,s.u.dtype)
    expected=torch.stack([layer.kv(x) for layer in m.reader.layers],1)
    torch.testing.assert_close(s.kv[0],expected)
    count=int(s.length[0])//m.reader.block*m.reader.tokens
    mt=(torch.arange(count)//m.reader.tokens+1)*m.reader.block-1
    expected_mem=torch.stack([layer.kv(s.mem[0,:count]+m.reader.position(mt,s.mem.dtype))
                              for layer in m.reader.layers],1)
    torch.testing.assert_close(s.mkv[0,:count],expected_mem)
torch.testing.assert_close(before,s.u,rtol=0,atol=0)
torch.testing.assert_close(mem,s.mem,rtol=0,atol=0)
print('PASS fast Reader version 1 reprojects K/V while source generation remains 0; source bytes unchanged; replay=0')
