"""Exact recent-frame membership, including duplicates and empty early rooms."""
import json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from isaac_bridge.rebuild_memory import recent_frame_union

rng=np.random.default_rng(102)
for seed in range(20):
    sizes=rng.integers(1,600,size=20)
    offsets=np.r_[0,np.cumsum(sizes)[:-1]]
    uid=rng.integers(0,len(sizes),size=1000)
    lengths=(rng.random(len(uid))*sizes[uid]).astype(np.int64)
    lengths[:20]=0
    query=offsets[uid]+lengths
    for window in (1,4,256):
        expected=np.unique(np.concatenate([np.arange(q-min(t,window),q) for q,t in zip(query,lengths)]))
        actual=recent_frame_union(query,lengths,window,int(sizes.sum()))
        np.testing.assert_array_equal(actual,expected)

sizes=np.full(64,3000,np.int64);offsets=np.r_[0,np.cumsum(sizes)[:-1]]
uid=rng.integers(0,64,size=32768);lengths=rng.integers(256,3000,size=len(uid))
query=offsets[uid]+lengths
seconds={};results={}
for name in ('expanded_unique','range_union'):
    elapsed=[]
    for _ in range(5):
        start=time.perf_counter()
        if name=='expanded_unique':
            result=np.unique(np.concatenate([np.arange(q-256,q) for q in query]))
        else:
            result=recent_frame_union(query,lengths,256,int(sizes.sum()))
        elapsed.append(time.perf_counter()-start)
    seconds[name]=float(np.median(elapsed));results[name]=result
np.testing.assert_array_equal(results['range_union'],results['expanded_unique'])
print(json.dumps(dict(result='PASS',random_cases=60,queries=32768,window=256,rows=int(sizes.sum()),
                      seconds=seconds,expanded_indices=len(query)*256)))
