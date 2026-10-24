"""Retiring one episode releases its saved graph while neighbours remain alive."""
import gc,sys,weakref
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch
from isaac_bridge.source_pipeline import SourcePipeline
torch.set_num_threads(2);torch.manual_seed(16)
model=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
                temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
                rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
pipe=SourcePipeline(model,['cpu'],micro=2048)
rng=np.random.default_rng(16);hs=[RawHistory.empty() for _ in range(4)]
for t,rows in enumerate(TS.make_seq(rng,nw=4,T=16,changes={(0,4),(1,8)},resets=set(),exits=set())):
    pending=M.synth_pending(rng,4,model.heads)
    if t==15:req=[(hs[w],rows[w],pending[w]) for w in range(4)]
    for w in range(4):hs[w]=hs[w].append(rows[w],pending[w],4)
bank=SourceBatch(model,req,cache=False,pipeline=pipe)
dead=hs[0].uid
cid=pipe.frames[(dead,0)][0]
wid=next(w for k,(w,_) in pipe.blocks.items() if k[0][0]==dead)
encoder=weakref.ref(pipe.chunks[cid]['future'].result()['u'])
writer=weakref.ref(pipe.writes[wid]['future'].result()['mem'])
del bank
pipe.retain_histories({h.uid for h in hs[1:]});gc.collect()
assert encoder() is None, 'retired encoder graph is pinned by unrelated live episodes'
assert writer() is None, 'retired Writer graph is pinned by unrelated live episodes'
assert len(pipe.frames)==48
pipe.close()
print('PASS retired episode encoder and Writer tensors released; three neighbouring episodes remain cached')
