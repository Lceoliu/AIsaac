"""Across-rollout history IPC transmits new immutable suffixes only."""
import pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
from isaac_bridge.rebuild_memory import RawHistory,pack_histories,unpack_histories
import copy, torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import SourceBatch
torch.set_num_threads(2);torch.manual_seed(71)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg).eval();reference=copy.deepcopy(model)
rng=np.random.default_rng(19)
seq=TS.make_seq(rng,nw=3,T=24,changes={(0,4),(1,8)},resets=set(),exits=set())
hist=[RawHistory.empty() for _ in range(3)];contexts=[[],[],[]];pending_seq=[]
for t,rows in enumerate(seq):
    if t==10:hist[1]=RawHistory.empty()
    if t==18:hist[2]=hist[2].fork()
    pending=M.synth_pending(rng,3,model.heads)
    pending_seq.append(pending)
    for w in range(3):
        contexts[w].append(hist[w]);hist[w]=hist[w].append(rows[w],pending[w],4)
sent={};received={};sizes=[];counts=[]
for lo,hi in [(0,8),(7,16),(15,24),(15,24)]:
    batch=[row[lo:hi] for row in contexts]
    packed=pack_histories(batch,previous=sent)
    data=pickle.dumps(packed,protocol=5);sizes.append(len(data))
    counts.append(sum(sum(len(r) for r,p in chunks) for uid,start,end,chunks in packed[1]))
    restored=unpack_histories(pickle.loads(data),4,previous=received)
    assert {uid:h.length for uid,h in received.items()}==sent, 'receiver cache length must include the retained prefix'
    for expected_row,actual_row in zip(batch,restored):
        for expected,actual in zip(expected_row,actual_row):
            assert (actual.uid,actual.length)==(expected.uid,expected.length)
            if actual.length:
                for a,b in zip(actual.arrays(),expected.arrays()):
                    if a.tobytes()!=b.tobytes():
                        print('MISMATCH',lo,hi,actual.uid,actual.length,a.dtype,
                              [f for f in a.dtype.names if a[f].tobytes()!=b[f].tobytes()] if a.dtype.names else '')
                        raise AssertionError('IPC prefix bytes differ')
    assert set(sent)==set(received)=={h.uid for row in batch for h in row}
assert counts==[21,23,41,0],counts
print('PASS incremental IPC: resets, partial blocks, forks, repeated payload; transmitted rows=',counts,
      'pickle_bytes=',sizes,'; full-prefix bytes restored exactly')

# Each producer has an independent FIFO/cache, including when half-rollouts
# arrive interleaved. Returning to an older prefix must not lose its suffix.
senders=[{},{}]; receivers=[{},{}]
for aid,lo,hi in [(1,0,8),(0,0,5),(0,4,16),(1,7,24),(1,0,8),(0,15,24)]:
    batch=[contexts[aid][lo:hi]]
    packed=pickle.loads(pickle.dumps(pack_histories(batch,previous=senders[aid]),protocol=5))
    result=unpack_histories(packed,4,previous=receivers[aid])
    for a,b in zip(result[0],batch[0]):
        assert (a.uid,a.length)==(b.uid,b.length)
        if a.length:
            assert all(x.tobytes()==y.tobytes() for x,y in zip(a.arrays(),b.arrays()))
print('PASS two interleaved Actor streams, shorter-prefix reuse and independent caches')

shared_rows=np.stack([r.view(np.uint8).reshape(3,-1) for r in seq],axis=1).view(seq[0].dtype).reshape(3,24)
shared_pending=np.stack(pending_seq,axis=1)
send={};receive={};wire=[];raw_counts=[]
for lo,hi in [(0,8),(7,16),(15,24),(15,24)]:
    batch=[r[lo:hi] for r in contexts]
    packed=pack_histories(batch,previous=send,shared=True)
    data=pickle.dumps(packed,protocol=5);wire.append(len(data))
    raw_counts.append(sum(len(r) for uid,start,end,parts in packed[1] for r,p in parts if not isinstance(r,str)))
    restored=unpack_histories(pickle.loads(data),4,previous=receive,
        rows=shared_rows[:,lo:hi],pending=shared_pending[:,lo:hi])
    for expected_row,actual_row in zip(batch,restored):
        for expected,actual in zip(expected_row,actual_row):
            assert (actual.uid,actual.length)==(expected.uid,expected.length)
            if actual.length:assert all(a.tobytes()==b.tobytes() for a,b in zip(actual.arrays(),expected.arrays()))
assert raw_counts==[0,0,18,0],raw_counts
assert wire[0]<sizes[0]/100 and wire[1]<sizes[1]/100
# Receiver histories own their bytes before producer buffers are reused.
saved=restored[0][-1].arrays()[0].tobytes();shared_rows['player']=0
assert restored[0][-1].arrays()[0].tobytes()==saved
print('PASS shared rollout IPC; wire bytes=',wire,'fallback raw rows=',raw_counts,'; producer reuse cannot mutate restored history')

requests=[]; decoded=[]
for t in (15,23):
    pend=M.synth_pending(rng,1,model.heads)[0]
    requests.append((contexts[0][t],seq[t][0],pend))
    decoded.append((restored[0][t-15],seq[t][0],pend))
batches=[SourceBatch(m,r,micro=7,cache=False) for m,r in [(reference,requests),(model,decoded)]]
for b in batches:
    b.u.retain_grad()
    out=b.forward([0,1],need_vg=True)
    (out[0].square().sum()+out[1].square().sum()+out[-1].square().sum()).backward()
for (n,a),(_,b) in zip(reference.named_parameters(),model.named_parameters()):
    assert (a.grad is None)==(b.grad is None),n
    if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=n)
assert batches[1].u.grad[0].abs().sum()>0
print('PASS incremental IPC exact parameter-gradient parity; early-room source gradient',
      float(batches[1].u.grad[0].abs().sum()))

