"""An unpublished early branch record stays live until its search completes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
from types import SimpleNamespace
import numpy as np
from isaac_bridge.tok_sampler import TokSampler,ST
from isaac_bridge.tok_teacher2 import V2_RING,V2_META,LANE_OUT
from isaac_bridge.tok_obs import ROW
s=SimpleNamespace(cfg=SimpleNamespace(workers=2,memory_dim=0),
    v2_rows=np.zeros((2,V2_RING),ROW),v2_meta=np.zeros((2,V2_RING,V2_META)),
    v2_t=np.zeros((2,V2_RING,66),np.float32),v2_seen=np.zeros(2,np.int64),
    stats=np.zeros((2,len(ST))),lane_t=np.zeros((2,66),np.float32),
    lane_out=np.zeros((2,LANE_OUT)),rebuild_contexts={1:'early',2:'latest',3:'other-worker'},
    rebuild_context_epochs={1:(0,0),2:(0,0),3:(1,0)})
s.lane_t[0,0]=2
TokSampler.v2_records(s,with_t=True)
assert s.rebuild_contexts[1]=='early', 'unpublished selected prefix was retired during active search'
# The worker publishes an early step, then acknowledges the whole search.
s.v2_t[0,0,0]=1;s.stats[0,ST['v2_records']]=1;s.lane_out[0,5]=1
out=TokSampler.v2_records(s,with_t=True)
assert out[-1]==['early']
assert s.rebuild_contexts[3]=='other-worker'
# Once the completed search's ring and lane no longer reference its prefixes,
# reclaim them. A different worker's ongoing search must remain unaffected.
s.v2_t[:]=0;s.lane_t[:]=0
TokSampler.v2_records(s,with_t=True)
assert s.rebuild_contexts=={3:'other-worker'}
s.lane_out[1,5]=1
TokSampler.v2_records(s,with_t=True)
assert s.rebuild_contexts=={} and s.rebuild_context_epochs=={}
print('PASS unpublished teacher prefixes survive rollout handoff; completed overwritten searches retire per worker')
