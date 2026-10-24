"""Partial shared-IPC handoffs enqueue source/Writer work before a full rollout."""
import copy, pickle, sys, threading
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.source_pipeline import SourcePipeline
from isaac_bridge.rebuild_memory import RawHistory,SourceBatch,pack_histories,unpack_histories

torch.set_num_threads(2);torch.manual_seed(94)
model=TokPolicy(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
    temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
    rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True).eval()
ref=copy.deepcopy(model)
pipe=SourcePipeline(model,['cpu','cpu'],period=0,micro=8)
rng=np.random.default_rng(94)
seq=TS.make_seq(rng,nw=2,T=16,changes={(0,4),(1,8)},resets=set(),exits=set())
hist=[RawHistory.empty(),RawHistory.empty()];contexts=[[],[]];pds=[]
for rows in seq:
    pending=M.synth_pending(rng,2,model.heads);pds.append(pending)
    for w in range(2):
        contexts[w].append(hist[w]);hist[w]=hist[w].append(rows[w],pending[w],4)
rows=np.stack([r.view(np.uint8).reshape(2,-1) for r in seq],axis=1).view(seq[0].dtype).reshape(2,16)
pending=np.stack(pds,axis=1)
sent={};received={};gate=threading.Event();encode=pipe._encode
def held(*args):
    assert gate.wait(10), 'source dispatch blocked the producer'
    return encode(*args)
with patch.object(pipe,'_encode',side_effect=held):
    packed=pack_histories([r[:8] for r in contexts],previous=sent,shared=True)
    restored=unpack_histories(pickle.loads(pickle.dumps(packed)),4,previous=received,rows=rows,pending=pending,latest_only=True)
    req=[(h,rows[w,t],pending[w,t]) for h,w,t in restored]
    assert [(h.uid,h.length) for h,_,_ in req]==[(h.uid,7) for h in hist]
    pipe.prefetch(req,write_ahead=True)
    assert pipe.encoded_rows==16 and pipe.written_blocks==4
    assert not any(c['future'].done() for c in pipe.chunks)
    gate.set()
packed=pack_histories(contexts,previous=sent,shared=True)
restored=unpack_histories(pickle.loads(pickle.dumps(packed)),4,previous=received,rows=rows,pending=pending)
req=[(restored[w][-1],rows[w,15],pending[w,15]) for w in range(2)]
for c in pipe.chunks:c['future'].result()
before=(dict(pipe.lengths),dict(pipe.frames),dict(pipe.owners),pipe.encoded_rows)
with patch.object(pipe,'_memory_budgets',return_value=[1e-9,1e-9]):
    assert pipe.prefetch(req,write_ahead=True,speculative=True) is False
assert (pipe.lengths,pipe.frames,pipe.owners,pipe.encoded_rows)==before
assert pipe.prefetch_pressure
pipe.prefetch(req,write_ahead=True)
assert pipe.encoded_rows==32 and pipe.written_blocks==8
# A future rollout's speculative sources survive current-rollout retirement.
pipe.prefetch_live[1]={hist[1].uid}
pipe.retain_histories({hist[0].uid})
assert (hist[1].uid,0) in pipe.frames
bank=SourceBatch(model,req,cache=False,pipeline=pipe)
dense=SourceBatch(ref,[(contexts[w][-1],rows[w,15],pending[w,15]) for w in range(2)],micro=8,cache=False)
early=pipe.chunks[pipe.frames[(hist[0].uid,0)][0]]['future'].result()['u'];early.retain_grad()
for b in (bank,dense):
    out=b.forward([0,1],need_vg=True)
    (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward()
bank.backward()
pipe.retain_histories(set())
assert (hist[0].uid,0) in pipe.frames, 'in-flight Reader credit was retired before its source transfer'
pipe._flush_collected()
packets=[(c,c['future'].result()['grads']) for c in pipe.chunks+pipe.writes]
results=[pipe.workers[i].submit(pipe._backward,i,packets).result() for i in range(2)]
for name,p in ref.named_parameters():
    if p.grad is not None and any(name in g for g in results):
        actual=sum(g[name] for g in results if name in g)
        # Shared norm readout contributes directly on the Reader model.
        direct=dict(model.named_parameters())[name].grad
        if direct is not None:actual=actual+direct
        torch.testing.assert_close(actual,p.grad,rtol=2e-3,atol=3e-5,msg=name)
assert early.grad[0].abs().sum()>0
assert pipe.encoded_rows==32
pipe.close()
pipe.devices=[torch.device('cuda:0'),torch.device('cuda:1')]
pipe.shared_devices={pipe.devices[0]};pipe.actor_reserve_gb=8
with patch('torch.cuda.mem_get_info',return_value=(22*2**30,24*2**30)), patch('torch.cuda.memory_reserved',return_value=2**30):
    assert pipe._memory_budgets()==[14.,21.], 'placement consumed the Actor future graph-pool reserve'
print('PASS nonblocking partial IPC -> source + Writer prefetch; no duplicate encode; protected next-rollout lifetime; cross-room gradients match dense reference')
print('PASS shared GPU capacity reserves 8 GiB for Actor plus 2 GiB source workspace independently of current Actor usage')
print('PASS over-budget lookahead is deferred without advancing source identities; required hand-off encodes every missing row once')

# Closing an interval only waits for source graphs that already have credit.
# A speculative next-rollout forward has no Reader leaves or cotangents yet.
from concurrent.futures import Future
import time
probe=SourcePipeline(model,['cpu'],period=0)
unready=Future();probe.chunks=[dict(future=unready)];probe.steps=1
timer=threading.Timer(.5,lambda:unready.set_result(dict(grads={})))
timer.start();start=time.perf_counter()
probe.start_finish(None,1.)
elapsed=time.perf_counter()-start
timer.join();probe.close()
assert elapsed<.25, 'source commit waited for an unconsumed future-rollout forward'
print('PASS source packet closes without fencing speculative forward; dispatch_seconds=',elapsed)

# A returned IPC handle only means host dispatch finished, not CUDA VJP.
from unittest.mock import Mock
completion=Mock();completion.query.return_value=False
submitted=Future();submitted.set_result({'_ready':completion})
assert not SourcePipeline._backward_ready(submitted)
completion.query.return_value=True
assert SourcePipeline._backward_ready(submitted)
print('PASS source commit readiness follows CUDA completion, not RPC return')

# Retirement credit must not fence a ready Reader, but must arrive before Adam.
retired=SourcePipeline(model,['cpu'],period=0)
future=Future();retired.retired_pending=[future]
name=retired.names[0];gradient=torch.ones_like(dict(model.named_parameters())[name])
timer=threading.Timer(.5,lambda:future.set_result({name:gradient}))
timer.start();start=time.perf_counter();retired._collect_retired(wait=False)
assert time.perf_counter()-start<.25
assert retired.retired_pending and not retired.direct_grads
retired._collect_retired(wait=True);timer.join()
torch.testing.assert_close(retired.direct_grads[name],gradient,rtol=0,atol=0)
assert not retired.retired_pending
retired.close()
print('PASS retired credit is nonblocking for Reader and completely collected before source Adam')

# A failed retirement RPC must not skip shutdown of its healthy siblings.
cleanup=SourcePipeline(model,['cpu'],period=0)
cleanup.processes=True;cleanup.remotes=[Mock(),Mock()]
broken=Future();broken.set_exception(RuntimeError('observed source failure'))
cleanup.control_futures=[broken]
try:cleanup.close()
except RuntimeError as exc:assert str(exc)=='observed source failure'
else:raise AssertionError('source failure was swallowed')
for remote in cleanup.remotes:remote.close.assert_called_once()
print('PASS source failure is propagated and all sibling processes are closed')
