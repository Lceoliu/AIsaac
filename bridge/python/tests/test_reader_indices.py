"""Pinned index ownership and non-blocking Reader submission."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from isaac_bridge.rebuild_memory import reader_indices

device=torch.device('cuda:0' if '--cuda' in sys.argv else 'cpu')
arrays=[np.arange(8192,dtype=np.int64),np.arange(16,dtype=np.int64).reshape(4,4),np.zeros(0,np.int64)]
expected=[a.copy() for a in arrays]
reader_indices(arrays,device)
if device.type=='cuda':
    torch.cuda.synchronize()
    torch.cuda._sleep(200000000)
    before=torch.cuda.Event();before.record()
start=time.perf_counter();out=reader_indices(arrays,device);seconds=time.perf_counter()-start
if device.type=='cuda':assert not before.query(), 'index submission waited for previous GPU work'
for a in arrays:a.fill(-1)
for a,b in zip(out,expected):np.testing.assert_array_equal(a.cpu().numpy(),b)
print(json.dumps(dict(result='PASS',device=str(device),host_submit_s=seconds,owned_copy=True)))
