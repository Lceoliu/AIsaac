"""Full graph versus deferred VJP, causality and cross-room source credit."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch

torch.set_num_threads(2)
torch.manual_seed(63)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg).eval()
rng=np.random.default_rng(63)
seq=TS.make_seq(rng,nw=1,T=16,changes={(0,4),(0,8),(0,12)},resets=set(),exits=set())
pending=[M.synth_pending(rng,1,model.heads)[0] for _ in seq]
h=RawHistory.empty(); requests=[]
for t,r in enumerate(seq):
    if t in (5,9,15): requests.append((h,r[0],pending[t]))
    h=h.append(r[0],pending[t],4)

reference=copy.deepcopy(model)
full=SourceBatch(reference,requests,micro=7,cache=False)
output=full.forward([0,1,2],need_vg=True)
loss=output[0].square().sum()+output[1].square().sum()+output[-1].square().sum()
loss.backward()
cached=SourceBatch(model,requests,micro=5,cache=True)
parts=[]
for ix in ([0],[1,2]):
    o=cached.forward(ix,need_vg=True);parts.append(o[0].detach())
    (o[0].square().sum()+o[1].square().sum()+o[-1].square().sum()).backward()
cached.backward()
err=float((torch.cat(parts)-output[0].detach()).abs().max())
assert err<1e-5,err
errors=[]
for (name,a),(name2,b) in zip(model.named_parameters(),reference.named_parameters()):
    assert name==name2
    assert (a.grad is None)==(b.grad is None),name
    if a.grad is not None:
        errors.append(float((a.grad-b.grad).abs().max()))
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5,msg=name)
assert cached.u.grad[0].abs().sum()>0,'early room source got no gradient'
assert model.reader.queries.grad.abs().sum()>0,'Writer got no gradient'
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('blocks.'))
print('PASS full graph / microbatch VJP logits max_abs',err,'parameter gradient max_abs',max(errors))
print('PASS cross-room early source gradient',float(cached.u.grad[0].abs().sum()),'writer gradient',float(model.reader.queries.grad.abs().sum()))

# Production direct mode must train all shared-source consumers without a second
# encoder forward. The final consumer also represents an unmerged teacher batch.
from unittest.mock import patch
direct_model = copy.deepcopy(reference); direct_model.zero_grad()
with patch.object(direct_model, 'encode_rebuild', wraps=direct_model.encode_rebuild) as encode:
    direct = SourceBatch(direct_model, requests, micro=7, cache=False)
    direct.u.retain_grad()
    calls = encode.call_count
    for i, ix in enumerate(([0], [1], [2])):
        out = direct.forward(ix, need_vg=True)
        (out[0].square().sum()+out[1].square().sum()+out[-1].square().sum()).backward(retain_graph=i < 2)
    direct.backward()
    assert encode.call_count == calls, 'direct training replayed the encoder'
assert direct.u.grad[0].abs().sum() > 0
for (name, a), (_, b) in zip(direct_model.named_parameters(), reference.named_parameters()):
    assert (a.grad is None) == (b.grad is None), name
    if a.grad is not None:
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-5, msg=name)
print('PASS direct source graph: three consumers, zero encoder replay, cross-room gradient and full-loss parity')

# A later request shares the same episode source bank. Changing its future must
# not change the earlier decision, even when whole blocks are encoded together.
changed=[]; h2=RawHistory.empty()
for t,r in enumerate(seq):
    rr=r[0].copy()
    if t>5: rr['player']+=.3
    if t in (5,9,15):changed.append((h2,rr,pending[t]))
    h2=h2.append(rr,pending[t],4)
with torch.no_grad():
    altered=SourceBatch(model,changed,micro=5,cache=False)
    a=cached.forward([0])[0]; b=altered.forward([0])[0]
torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
print('PASS future observations do not leak through completed-block memory')
print('PASS immutable history / branch isolation',h.length,h.fork().uid!=h.uid)
assert h.fork().uid!=h.uid

from isaac_bridge.tok_policy import TemporalState, to_batch
actor = copy.deepcopy(model).eval()
actor.source_cache = True
state = TemporalState(actor, 2, 'cpu', torch.float32)
for t, rows in enumerate(seq):
    pend = np.asarray([pending[t]])
    if t == 10:
        with torch.no_grad():
            actor.reader.inp.weight.add_(.03)
            actor.player[0].weight.add_(.002)
        actor.invalidate_source_cache()
    temporal = state.step_inputs(actor, rows, pend, np.array([0]))
    with torch.no_grad():
        a = actor(to_batch(rows,pend,'cpu'),temporal=temporal,need_vg=True)
        req = [(temporal['out']['raw_contexts'][0],rows[0],pending[t])]
        fresh = SourceBatch(actor,req,micro=5,cache=False)
        b = fresh.forward([0],need_vg=True)
    for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=2e-4,atol=1e-5)
print('PASS actor incremental / learner rebuild parity across rooms and parameter refresh')
state.copy_slot_(0,1)
assert state.rebuild.raw[0].uid != state.rebuild.raw[1].uid
state.reset_slot_(1)
assert state.rebuild.raw[0].length == 16 and state.rebuild.raw[1].length == 0
print('PASS actor branch fork and episode reset isolate histories')

lagged = copy.deepcopy(actor); lagged.memory_refresh_updates = 8; lagged.cache_generation = 0
lag_state = TemporalState(lagged, 1, 'cpu', torch.float32)
lag_state.rebuild.raw[0] = h
lag_state.rebuild.refresh(lagged, [0])
retained = lag_state.rebuild.recent[0]
for version in range(1, 8):
    with torch.no_grad(): lagged.reader.inp.weight.add_(.001)
    lagged.invalidate_source_cache()
    with patch.object(lagged, 'encode_rebuild', wraps=lagged.encode_rebuild) as encode:
        lag_state.rebuild.refresh(lagged, [0])
        assert encode.call_count == 0
    assert lag_state.rebuild.recent[0] is retained
lagged.invalidate_source_cache()
lag_state.rebuild.refresh(lagged, [0])
strict_state = TemporalState(lagged, 1, 'cpu', torch.float32)
strict_state.rebuild.raw[0] = h
strict_state.rebuild.refresh(lagged, [0])
torch.testing.assert_close(lag_state.rebuild.recent[0], strict_state.rebuild.recent[0])
assert lag_state.rebuild.recent[0] is not retained
print('PASS bounded refresh: versions 1..7 reuse history, version 8 rebuilds to the current-weight reference')

# Repeated inputs: use identical full components while retaining distinct frame
# identities. Attribute/charge/item/map changes must invalidate the right source.
dupe = seq[0].repeat(4)
pd = np.repeat(np.asarray([pending[0]]),4,axis=0)
base = copy.deepcopy(model).eval(); base.source_cache = False; base.zero_grad()
shared = copy.deepcopy(model).eval(); shared.source_cache = True; shared.zero_grad()
for m in (base,shared):
    b = to_batch(dupe,pd,'cpu',entities=64)
    b['_raw_rows'],b['_raw_pending'] = dupe,pd
    y,_,u = m.encode_rebuild(b)
    (y.square().sum()+u.square().sum()).backward()
for (name,a),(name2,b) in zip(base.named_parameters(),shared.named_parameters()):
    assert (a.grad is None)==(b.grad is None),name
    if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=4e-4,atol=3e-5,msg=name)
assert shared._components.calls['items'][0] == 3
for field in ('player','pcharge','phearts','inv','pitem','pinv','patch','grid','map','pchar'):
    changed = dupe.copy()
    changed[field][0].flat[0] += 1
    with torch.no_grad():
        b = to_batch(changed,pd,'cpu',entities=64)
        b['_raw_rows'],b['_raw_pending'],b['_actor_components'] = changed,pd,True
        actual = shared.encode_rebuild(b)
        b.pop('_raw_rows')
        expected = base.encode_rebuild(b)
        torch.testing.assert_close(actual[0],expected[0],rtol=2e-4,atol=1e-5,msg=field)
print('PASS changed component payloads invalidate; repeated consumers sum exact encoder gradients')
print('COMPONENT_REUSE',shared._components.calls)

import pickle
import train_tok
import argparse
parsers = []
with patch.object(argparse.ArgumentParser, 'parse_args', lambda parser, argv: parsers.append(parser) or sys.exit(0)):
    try:
        train_tok.main()
    except SystemExit as exc:
        assert exc.code == 0
assert parsers[0].get_default('micro') == 8192
assert parsers[0].get_default('source_replay') == 0
assert parsers[0].get_default('memory_refresh_updates') == 8
print('PASS training defaults: micro=8192, source_replay=0, memory_refresh_updates=8')
long=RawHistory.empty()
for t in range(320):
    row=seq[t%len(seq)][0].copy();row['first']=0;row['t']=t
    long=long.append(row,pending[t%len(pending)],4)
long=pickle.loads(pickle.dumps(long))
late=SourceBatch(model,[(long,row,pending[0])],micro=32)
model.zero_grad()
late.forward([0])[0].square().sum().backward();late.backward()
assert late.u.grad[0].abs().sum()>0
assert len(late.block_indices)==80 and len(late.rows)==321
print('PASS 320-decision history, 80 completed blocks, earliest-source gradient',float(late.u.grad[0].abs().sum()))
tcfg=dict(P=4,L=2,D=32,R=4,rows=16,device='cpu',kv='fp32',rebuild_memory=True)
old=train_tok.Rollout(1,4,n_heads=5,temporal=tcfg)
new=train_tok.Rollout(1,4,n_heads=5,temporal=tcfg)
old.open[0]=2;old.length[0]=3
old.raw_contexts[0][2]=long
old.t_assign(np.array([0]),np.array([2]))
new.carry(old)
assert new.raw_contexts[0][0].length==320 and new.raw_contexts[0][0].uid==long.uid
restored=pickle.loads(pickle.dumps(train_tok.rollout_scalars(new)))
assert restored['raw_contexts'][0][0].length==320
print('PASS rollout carry and learner IPC preserve complete raw history')

from types import SimpleNamespace
from test_lane_batch_smoke import FakeSampler
from isaac_bridge.tok_lanes import LaneService
from isaac_bridge.tok_sampler import TokSampler, ST
from isaac_bridge.tok_teacher2 import V2_RING,V2_META
fake=FakeSampler(2,32)
lanes=TemporalState(actor,2,'cpu',torch.float32)
service=LaneService(fake,actor,'cpu',5,tstate=lanes,bank=state,gtok=True,wait=0)
row=seq[-1][0].copy();row['first']=0
fake.request(0,row,pending[-1],1,tbegin=1)
assert service.tick()
ref=int(fake.lane_t[0,0]) | (int(fake.lane_t[0,1])<<16)
assert fake.rebuild_contexts[ref].length==16
assert fake.rebuild_contexts[ref].uid!=state.rebuild.raw[0].uid
fake.cfg=SimpleNamespace(workers=1,memory_dim=0)
fake.v2_rows=np.zeros((1,V2_RING),row.dtype);fake.v2_rows[0,0]=row
fake.v2_meta=np.zeros((1,V2_RING,V2_META),np.float64)
fake.v2_t=np.zeros((1,V2_RING,66),np.float32);fake.v2_t[0,0]=fake.lane_t[0]
fake.v2_seen=np.zeros(1,np.int64);fake.stats=np.zeros((1,len(ST)))
fake.stats[0,ST['v2_records']]=1
selected=TokSampler.v2_records(fake,with_t=True)
assert selected[-1][0].uid==fake.rebuild_contexts[ref].uid
with torch.no_grad():
    teacher=SourceBatch(actor,[(selected[-1][0],selected[0][0],pending[-1])],cache=False)
    expected=teacher.forward([0],need_vg=True)
np.testing.assert_allclose(fake.lane_out[0,0],expected[1].item(),rtol=1e-4,atol=1e-5)
print('PASS real CF lane service / selected teacher ring roundtrip carries raw branch prefix')


# Compact IPC preserves every historical prefix, including partial blocks and forks.
from isaac_bridge.rebuild_memory import pack_histories, unpack_histories
import pickle
contexts = [[None]]
history = RawHistory.empty()
for t, row in enumerate(seq):
    contexts[0].append(history)
    history = history.append(row[0], pending[t], 4)
contexts[0].append(history)
contexts.append([None] * len(contexts[0]))
branch = history.fork().append(seq[0][0], pending[0], 4)
contexts[1][-2:] = [history.fork(), branch]
restored = unpack_histories(pickle.loads(pickle.dumps(pack_histories(contexts), protocol=5)), 4)
for before, after in zip(sum(contexts, []), sum(restored, [])):
    if before is None:
        assert after is None
    else:
        assert (before.uid,before.length) == (after.uid,after.length)
        if before.length:
            for x,y in zip(before.arrays(),after.arrays()): np.testing.assert_array_equal(x,y)
            assert not (after.blocks + after.tail)[0][0].flags.writeable
print('PASS compact IPC preserves all partial-block prefixes and forks exactly')
