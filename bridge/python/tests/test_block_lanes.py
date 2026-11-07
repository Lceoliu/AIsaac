"""Real lane service/V2 ring adapter checks, without spawning the game engine."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch

from test_lane_batch_smoke import FakeSampler
from test_block_training import history_actor
from test_visible_input import synth_rows,synth_pending
from isaac_bridge.block_lanes import BlockLaneService,decode_reference
from isaac_bridge.block_training import Chunk,forward_chunks,cf_loss
from isaac_bridge.tok_sampler import TokSampler,ST
from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_teacher2 import V2_RING,V2_META


def ring_sampler(n=2):
    s=FakeSampler(n,D=16)
    s.cfg=SimpleNamespace(workers=n,memory_dim=0)
    s.v2_rows=np.zeros((n,V2_RING),ROW)
    s.v2_meta=np.zeros((n,V2_RING,V2_META),np.float64)
    s.v2_t=np.zeros((n,V2_RING,34),np.float32)
    s.v2_h=None
    s.stats=np.zeros((n,max(ST.values())+1))
    s.v2_seen=np.zeros(n,np.int64)
    return s


def test_lane_to_real_v2_ring_to_teacher_gradient():
    model,main,rng=history_actor()
    snap=main.snapshot(0)
    s=ring_sampler()
    service=BlockLaneService(s,model,"cpu",wait=0)
    assert service.capture_point(0,snap,model,model_version=0)
    rows=synth_rows(rng,1);rows["first"]=0
    pend=snap.pending.copy()
    s.request(0,rows[0],pend[0],seq=13,tbegin=1)
    assert service.tick()
    assert s.lane_out[0,4]==13 and s.lane_out[0,2]==0
    ident=decode_reference(s.lane_t[0])
    context=s.block_contexts.get(ident)
    assert context.memory.step==snap.memory.step and context.history.step==snap.history.step
    np.testing.assert_array_equal(context.pending,pend)
    # Worker publishes the accepted winner record using the existing shared ABI.
    s.v2_rows[0,0]=rows[0];s.v2_t[0,0]=s.lane_t[0]
    s.stats[0,ST["v2_records"]]=1
    service.release_point(0)
    rr,meta,tt,contexts=TokSampler.v2_records(s,with_t=True,with_block=True)
    assert len(rr)==1 and contexts[0] is context
    assert ident in s.block_contexts.values
    out,b,_,_=forward_chunks(model,[Chunk(rr,pend,contexts[0])])
    action=torch.as_tensor(s.lane_io[:1,5:10],dtype=torch.long)
    loss,_=cf_loss(model,out,b,action,torch.ones(1),torch.ones(1),1.,.2,.5)
    loss.backward()
    assert model.block_memory.writer.queries.grad.abs().sum()>1e-6
    assert main.memory[0].step==snap.memory.step
    # Retire only after publication fence and removal of ring/lane references.
    s.v2_t[:]=0;s.lane_t[:]=0
    s.block_contexts.retire(set())
    assert not s.block_contexts.values


def test_pinned_generations_and_no_unbounded_replication():
    model,main,rng=history_actor()
    s=ring_sampler()
    service=BlockLaneService(s,model,"cpu",wait=0)
    snap=main.snapshot(0)
    assert service.capture_point(0,snap,model,0)
    with torch.no_grad(): model.pi.bias[0]+=1.
    snap1=replace(snap,policy_version=1)
    assert service.capture_point(1,snap1,model,1)
    assert not service.capture_point(2,replace(snap,policy_version=2),model,2)
    rows=synth_rows(rng,2);rows["first"]=0
    s.request(0,rows[0],snap.pending[0],seq=5,tbegin=1)
    s.request(1,rows[1],snap.pending[0],seq=6,tbegin=2)
    service.tick()
    np.testing.assert_array_equal(s.lane_out[:,2],[0,1])
    np.testing.assert_array_equal(s.lane_out[:,4],[5,6])
    assert len(service.generations.slots)==2
    service.release_point(0)
    assert service.capture_point(2,replace(snap,policy_version=2),model,2)


def test_batch_order_does_not_change_branch_actions():
    model,main,rng=history_actor()
    snap=main.snapshot(0)
    rows=synth_rows(rng,2);rows["first"]=0
    answers=[]
    for order in ([0,1],[1,0]):
        s=ring_sampler()
        service=BlockLaneService(s,model,"cpu",wait=0)
        for i in range(2): assert service.capture_point(i,snap,model,0)
        for i in order: s.request(i,rows[i],snap.pending[0],seq=19+i,tbegin=i+1)
        service.tick()
        answers.append((s.lane_io[:,5:10].copy(),s.lane_out[:,:4].copy()))
    np.testing.assert_array_equal(answers[0][0],answers[1][0])
    np.testing.assert_allclose(answers[0][1],answers[1][1],atol=2e-6,rtol=2e-5)


def test_in_flight_action_snapshot_is_last_answer():
    model,main,rng=history_actor()
    row=synth_rows(rng,1);row["first"]=0
    pending=main.snapshot(0).pending
    out=main.step(row,pending,[0],ukeys=(np.array([42]),np.array([0])))
    np.testing.assert_array_equal(main.snapshot(0).pending,out[0].numpy())
    np.testing.assert_array_equal(main.last_contexts[0].pending,pending)


def test_completed_search_without_winner_releases_contexts():
    model,main,rng=history_actor()
    snap=main.snapshot(0)
    s=ring_sampler()
    service=BlockLaneService(s,model,"cpu",wait=0)
    for attempt in range(5):
        assert service.capture_point(0,snap,model,0)
        rows=synth_rows(rng,1);rows["first"]=0
        s.request(0,rows[0],snap.pending[0],seq=attempt+1,tbegin=1)
        service.tick()
        assert len(s.block_contexts.values)==1
        service.release_point(0)
        s.lane_t[:]=0  # worker completion fence: no accepted winner published
        rr,meta,contexts=TokSampler.v2_records(s,with_block=True)
        assert len(rr)==len(contexts)==0
        assert not s.block_contexts.values


def test_recapturing_old_generation_does_not_select_latest():
    model,main,rng=history_actor()
    s=ring_sampler()
    service=BlockLaneService(s,model,"cpu",wait=0)
    snap=main.snapshot(0)
    assert service.capture_point(0,snap,model,0)
    with torch.no_grad(): model.pi.bias[0]+=1.
    assert service.capture_point(1,replace(snap,policy_version=1),model,1)
    assert service.capture_point(2,snap,model,0)
    assert service.points[2][1:3]==service.points[0][1:3]
    assert service.generations.refs==[2,1]
    rows=synth_rows(rng,1);rows["first"]=0
    s.request(0,rows[0],snap.pending[0],seq=13,tbegin=3)
    service.tick()
    assert s.lane_out[0,2]==0

