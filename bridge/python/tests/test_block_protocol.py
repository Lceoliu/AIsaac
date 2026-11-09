"""Production control-plane races, source pins and bounded KV snapshots."""
from dataclasses import replace
import numpy as np
import torch
import pytest

from test_block_lanes import ring_sampler
from test_block_training import history_actor
from test_visible_input import synth_rows,synth_pending
from isaac_bridge.block_protocol import shapes,write_injury_jobs
from isaac_bridge.block_lanes import BlockLaneService,decode_reference
from isaac_bridge.tok_sampler import TokSamplerConfig,TokSampler,ST,TEACH_RING,TEACH_META


def production(steps=8):
    model,actor,rng=history_actor(steps=steps)
    s=ring_sampler()
    s.cfg=TokSamplerConfig(workers=2,teacher=True,cf=True,block_memory=True,teacher_queue=2,
                          teacher_depths=(0,1,2),teacher_death_depths=(3,),cf_distill_steps=1)
    s.block_arrays=[np.zeros(shape,np.int64) for shape in shapes(s.cfg)]
    s.block_arrays[2][:]=-1
    s.cf_snap=np.zeros(2,np.int64)
    s.teach_rows=np.zeros((2,TEACH_RING),s.v2_rows.dtype)
    s.teach_meta=np.zeros((2,TEACH_RING,TEACH_META),np.int32)
    s.teach_seen=np.zeros(2,np.int64)
    lanes=BlockLaneService(s,model,'cpu',wait=0,distill_steps=1,main_actor=actor)
    lanes.RECLAIM_S=0   # (B46: production reclaims at most every 0.1 s; here at every call)
    # (B47: these checks follow one round to its answers within step_main / tick: synchronous)
    lanes.lanes_async=lanes.main_async=lanes.lane_batch=False
    return model,actor,rng,s,lanes


def test_cancel_pending_capture_does_not_pin_generation():
    model,actor,rng,s,lanes=production()
    rows=synth_rows(rng,1);rows['first']=0
    s.cf_snap[0]=1;s.block_arrays[0][0,0]=3
    lanes.capture_main([0],rows,actor.pending[0])
    assert s.cf_snap[0]==0 and s.block_arrays[0][0,0]==0
    assert not lanes.points and lanes.generations.refs==[0,0]


def test_late_wake_cancel_slot_reuse_and_duplicate_answer():
    model,actor,rng,s,lanes=production()
    snap=actor.snapshot(0)
    assert lanes.capture_point(0,snap,model,0)
    row=synth_rows(rng,1);row['first']=0
    s.request(0,row[0],snap.pending[0],101,tbegin=1)
    s.lane_ready=[]  # pipe wake arrives after release acknowledgement
    s.block_arrays[0][0,0]=3
    lanes.service_control()
    assert s.block_arrays[0][0,0]==0
    assert lanes.capture_point(0,replace(snap,policy_version=1),model,1)
    s.arrive(0);lanes.tick()
    assert s.lane_out[0,4]==101 and s.lane_out[0,5]==-1
    s.arrive(0);lanes.tick()
    assert s.lane_out[0,5]==-1
    s.request(0,row[0],snap.pending[0],102,tbegin=1);lanes.tick()
    generation=lanes.points[0][1]
    step=lanes.actors[generation].memory[0].step
    result=s.lane_out[0].copy()
    s.arrive(0);lanes.tick()
    assert lanes.actors[generation].memory[0].step==step
    np.testing.assert_array_equal(s.lane_out[0],result)
    assert len(lanes.cancelled_requests)==1


def test_deferred_contexts_kept_after_first_eight_and_released():
    model,actor,rng,s,lanes=production()
    snap=actor.snapshot(0);assert lanes.capture_point(0,snap,model,0)
    kept=0
    for j in range(15):
        row=synth_rows(rng,1);row['first']=0
        s.request(0,row[0],snap.pending[0],j+10,tbegin=1 if j==0 else 0)
        lanes.tick()
        ref=decode_reference(s.lane_t[0]);assert ref>0
        if j==10:
            kept=ref;s.block_arrays[4][0,0]=ref
        lanes.service_control()
        assert len(lanes.contexts.values)<=2
    assert kept in lanes.contexts.values
    s.v2_rows[0,0]=row[0]
    s.v2_t[0,0,:2]=[kept&65535,kept>>16];s.v2_t[0,0,-1]=1
    s.stats[0,ST['v2_records']]=1
    s.block_arrays[4][:]=0
    lanes.service_control()
    got=TokSampler.v2_records(s,with_block=True)
    assert len(got[2])==1
    s.v2_t[:]=0;s.lane_t[:]=0;lanes.service_control()
    assert not lanes.contexts.values


def test_terminal_hurt_sources_survive_queue_active_and_ring_transfer():
    model,actor,rng,s,lanes=production(0)
    recorded={}
    for t in range(7):
        row=synth_rows(rng,1);row['first']=int(t==0);row['episode']=1;row['t']=t;row['hurt']=0
        pending=synth_pending(rng,1,model.heads)
        lanes.observe_rows([0],row)
        actor.step(row,pending,[0]);lanes.record_main([0],row,actor.last_contexts)
        recorded[t]=row[0].copy()
    terminal=np.array([recorded[6]],dtype=s.teach_rows.dtype)
    terminal['t']=7;terminal['hurt']=1;terminal['done']=2
    lanes.observe_rows([0],terminal)
    item=(None,None,None,7,123,1,0,True)
    write_injury_jobs(s.block_arrays,0,[item],[])
    first=synth_rows(rng,1);first['first']=1;first['episode']=2;first['t']=0;first['hurt']=0
    lanes.observe_rows([0],first);lanes.service_control()
    context=lanes.injury_context(0,1,4)
    write_injury_jobs(s.block_arrays,0,[],[item]);lanes.service_control()
    s.teach_rows[0,0]=recorded[4];s.block_arrays[5][0,0]=1;s.stats[0,ST['teach']]=1
    write_injury_jobs(s.block_arrays,0,[],[]);lanes.service_control()
    rows,meta,contexts=TokSampler.teacher_records(s,with_block=True)
    assert len(rows)==1 and contexts[0] is context
    # A torn/overwritten slot is never read as the older serial.
    s.teach_seen[0]=0;s.block_arrays[5][0,0]=TEACH_RING+1
    assert len(TokSampler.teacher_records(s,with_block=True)[0])==0
    lanes.service_control()
    assert (0,1,4) not in lanes.injury_sources


def test_compact_snapshots_share_kv_rows_and_replay_never_samples():
    model,actor,rng,s,lanes=production()
    before=actor.snapshot(0)
    assert all(k.numel()==0 for k,v in before.memory.kv)
    row=synth_rows(rng,1);row['first']=0
    random_state=torch.random.get_rng_state()
    actor.step(row,before.pending,[0],evaluate=True)
    torch.testing.assert_close(torch.random.get_rng_state(),random_state)
    after=actor.snapshot(0)
    assert before.kv_rows[-1] is after.kv_rows[-2]
    dense=after.to('cpu')
    for (a,b),(c,d) in zip(dense.memory.kv,actor.memory[0].kv):
        torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)


@pytest.mark.parametrize('matching', ['none','merge','replay'])
def test_main_reply_precedes_old_teacher_without_dropping_any_lane(matching):
    model,actor,rng,s,lanes=production()
    snap=actor.snapshot(0)
    assert lanes.capture_point(0,snap,model,0)
    old=lanes.actors[lanes.points[0][1]]
    actor.version=1
    if matching!='none':
        assert lanes.capture_point(1,replace(snap,policy_version=1),model,1)
    rows=synth_rows(rng,2);rows['first']=0
    s.request(0,rows[0],snap.pending[0],101,tbegin=1)
    if matching!='none':
        s.request(1,rows[1],snap.pending[0],102,tbegin=2)
        s.block_arrays[1][1]=int(matching=='replay')
    replies=[]
    def on_ready(host,contexts):
        # A ready main answer cannot wait behind an old-generation forward.
        assert old.memory[0].step==snap.memory.step
        assert actor.memory[0].step==snap.memory.step+1
        assert contexts[0].memory.step==snap.memory.step
        replies.append(host.copy())
    lanes.step_main(rows[:1],snap.pending,[0],on_main_ready=on_ready)
    assert len(replies)==1 and lanes.main_request is None
    assert s.lane_out[0,4]==101 and s.lane_out[0,2]==0
    assert old.memory[0].step==snap.memory.step+1
    if matching!='none':
        assert s.lane_out[1,4]==102 and s.lane_out[1,2]==1
        assert lanes.stat.get('merged_main_rows',0)==1


@pytest.mark.parametrize('device,amp',[('cpu',False),('cuda',False),('cuda',True)])
def test_same_generation_replay_and_search_use_one_forward_with_identical_answers(device,amp):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    outputs=[];counts=[]
    for combined in (False,True):
        model,actor,rng,s,lanes=production()
        model.to(device);actor.bf16=amp;lanes.device=device
        snap=actor.snapshot(0)
        assert lanes.capture_point(0,snap,model,0)
        assert lanes.capture_point(1,snap,model,0)
        branch=lanes.actors[lanes.points[0][1]]
        calls=[]
        original=branch.step
        def counted(*args,**kwargs):
            calls.append(1)
            return original(*args,**kwargs)
        branch.step=counted
        rows=synth_rows(rng,2);rows['first']=0
        s.block_arrays[1][1]=1
        before=torch.random.get_rng_state()
        before_cuda=torch.cuda.get_rng_state() if device=='cuda' else None
        for i in range(2):
            s.request(i,rows[i],snap.pending[0],201+i,tbegin=i+1)
            if not combined:lanes.tick()
        if combined:lanes.tick()
        assert torch.equal(before,torch.random.get_rng_state())
        if before_cuda is not None:assert torch.equal(before_cuda,torch.cuda.get_rng_state())
        np.testing.assert_array_equal(s.lane_io[1,5:10],snap.pending[0])
        assert s.lane_out[1,1]==0 and decode_reference(s.lane_t[1])==0
        outputs.append((s.lane_io[:,5:10].copy(),s.lane_out.copy(),branch.memory))
        counts.append(len(calls))
    assert counts==[2,1]
    np.testing.assert_array_equal(outputs[0][0],outputs[1][0])
    np.testing.assert_allclose(outputs[0][1],outputs[1][1],atol=3e-6,rtol=3e-5)
    for a,b in zip(outputs[0][2],outputs[1][2]):
        assert a.step==b.step
        for (ak,av),(bk,bv) in zip(a.kv,b.kv):
            torch.testing.assert_close(ak,bk,atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(av,bv,atol=3e-6,rtol=3e-5)


def test_throttled_reclamation_only_delays_release_and_never_reuses_ids():
    """B46: service_control's context / injury-source reclamation runs at most every RECLAIM_S
    seconds. Releases (status 3), cancellations and new contexts are handled at every call;
    context ids are never reused (monotonic), a context a ring still references survives every
    reclamation, and an unreferenced one is released at the next one that runs."""
    model,actor,rng,s,lanes=production()
    lanes.RECLAIM_S=1e9
    lanes._reclaimed_t=-1e9   # the first call reclaims, later ones within the window do not
    snap=actor.snapshot(0);assert lanes.capture_point(0,snap,model,0)
    ids=[]
    for j in range(40):
        row=synth_rows(rng,1);row['first']=0
        s.request(0,row[0],snap.pending[0],j+10,tbegin=1 if j==0 else 0)
        lanes.tick()
        ids.append(decode_reference(s.lane_t[0]))
        if j%7==3:   # a cancellation / release every few requests (handled at once, not throttled)
            s.block_arrays[0][0,0]=3;lanes.service_control()
            assert s.block_arrays[0][0,0]==0
            assert lanes.capture_point(0,snap,model,0)
    assert len(set(ids))==len(ids) and ids==sorted(ids)   # never reused
    kept=ids[-1];s.block_arrays[4][0,0]=kept
    assert len(lanes.contexts.values)>2                   # nothing released inside the window
    lanes._reclaimed_t=-1e9;lanes.service_control()      # the window over: released, the kept one stays
    assert kept in lanes.contexts.values and len(lanes.contexts.values)<=2


def _slot_lanes(lanes_async):
    """B47: the production lane service on a CUDA SlotBlockActor (step graphs, lanes in its arena)."""
    from test_block_memory import make_policy
    from test_block_slot_integration import frame
    from isaac_bridge.block_paged import SlotBlockActor
    _,_,_,s,_=production()
    torch.manual_seed(43)
    model=make_policy('cuda').eval()
    actor=SlotBlockActor(model,2,graph_entities=128,lane_slots=2)
    rng=np.random.default_rng(43)
    for t in range(6):
        actor.step(frame(rng,2,t,np.array([4,4]),int(t==0)),synth_pending(rng,2,model.heads),[0,1],
                   ukeys=(np.array([t,t]),np.array([0,1])))
    lanes=BlockLaneService(s,model,'cuda',wait=0,distill_steps=1,main_actor=actor)
    lanes.RECLAIM_S=0
    lanes.lanes_async=lanes.main_async=lanes.lane_batch=bool(lanes_async)
    return model,actor,rng,s,lanes,frame


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graphs')
def test_async_lane_rounds_and_main_answers_equal_sync(monkeypatch):
    """B47: the same request sequence (same lanes, sequence numbers = CRN keys, main rows, host
    draw order) through the lane service with every step answered at once (B46) and with lane
    rounds / main answers launched and answered at later passes gives bit-identical lane answers
    (actions, value, logp, version, value_global, echo) and main answers; an old-generation lane
    (lane-only step) and a current-generation lane (merged into the main step) both covered."""
    from isaac_bridge import block_paged
    monkeypatch.setattr(block_paged,'PRECAPTURE',False)   # (lane graphs captured on first use)
    results=[]
    for lanes_async in (False,True):
        model,actor,rng,s,lanes,frame=_slot_lanes(lanes_async)
        answers={};mains=[];deferred=0
        def on_reply(idx):
            for i in idx:
                answers[int(s.lane_io[i,10])]=(s.lane_io[i,5:10].copy(),s.lane_out[i,:5].copy())
        s.on_reply=on_reply
        assert lanes.capture_point(0,actor.snapshot(0),model,0)   # generation of version 0
        actor.version=1
        assert lanes.capture_point(1,actor.snapshot(1),model,1)   # generation of the current version
        torch.manual_seed(7)   # (the main rows' host uniforms: same draws in both runs)
        for r in range(8):
            lane=r%2
            s.request(lane,frame(rng,1,20+r,np.array([9]),0)[0],synth_pending(rng,1,model.heads)[0],
                      300+r,tbegin=lane+1 if r<2 else 0)
            main=frame(rng,1,6+r,np.array([4]),0)
            lanes.step_main(main,synth_pending(rng,1,model.heads),[r%2],
                            on_main_ready=lambda h,c:mains.append(h.copy()))
            deferred+=len(lanes.work)
        lanes.complete_all()
        assert not lanes.work and not actor.state.inflight
        assert sorted(answers)==list(range(300,308)) and len(mains)==8
        assert (deferred>0)==lanes_async
        assert lanes.stat.get('merged_main_rows',0)>0
        results.append((answers,mains))
    (a0,m0),(a1,m1)=results
    for k in a0:
        np.testing.assert_array_equal(a0[k][0],a1[k][0])
        np.testing.assert_array_equal(a0[k][1],a1[k][1])
    for x,y in zip(m0,m1):
        np.testing.assert_array_equal(x,y)
