"""Exercise real policy, source IPC and the existing PPO/CF/injury losses."""
import io
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch

from test_block_memory import make_policy
from test_visible_input import synth_rows,synth_pending
from isaac_bridge.block_training import (BlockActor,Chunk,forward_chunks,pack_snapshots,unpack_snapshots,
    ppo_loss,cf_loss,injury_loss,TeacherTrajectory)
from isaac_bridge.tok_policy import to_batch


def test_historical_spatial_checkpoint_matches_small_batches():
    torch.manual_seed(312)
    model=make_policy()
    reference=copy.deepcopy(model)
    actor=BlockActor(model,40)
    rng=np.random.default_rng(312)
    for t in range(8):
        rows=synth_rows(rng,40);rows['first']=int(t==0);rows['episode']=1
        actor.step(rows,synth_pending(rng,40,model.heads),range(40))
    rows=synth_rows(rng,40);rows['first']=0;rows['episode']=1
    pending=synth_pending(rng,40,model.heads)
    chunks=[Chunk(rows[i:i+1],pending[i:i+1],actor.snapshot(i)) for i in range(40)]
    # 40 x 2 x 2 historical observations crosses the 128-row checkpoint budget.
    out,_,_,_=forward_chunks(model,chunks,generator=torch.Generator().manual_seed(41),spatial_micro=128)
    loss=out[1].square().sum()/40
    loss.backward()
    gen=torch.Generator().manual_seed(41)
    expected=0.
    for begin in (0,20):
        y,_,_,_=forward_chunks(reference,chunks[begin:begin+20],generator=gen)
        part=y[1].square().sum()/40
        expected+=float(part.detach());part.backward()
    assert abs(float(loss.detach())-expected)<2e-6
    for name in ('structured_spatial.body.0.weight','block_memory.writer.queries','block_memory.memory_age'):
        got=dict(model.named_parameters())[name].grad
        ref=dict(reference.named_parameters())[name].grad
        assert got is not None and got.abs().sum()>0
        torch.testing.assert_close(got,ref,atol=3e-6,rtol=1e-4)


def history_actor(device="cpu",steps=8):
    torch.manual_seed(41)
    m=make_policy(device)
    actor=BlockActor(m,2)
    rng=np.random.default_rng(41)
    for t in range(steps):
        rows=synth_rows(rng,2);rows["first"]=int(t==0);rows["episode"]=4
        pending=synth_pending(rng,2,m.heads)
        actor.step(rows,pending,[0,1],ukeys=(np.array([t,t]),np.array([0,1])))
    return m,actor,rng


def test_source_lifetimes_ipc_and_fork():
    m,actor,rng=history_actor(steps=7)
    snap=actor.snapshot(0)
    assert snap.memory.step==snap.history.step==7
    assert len(snap.history.blocks)==3 and len(snap.history.tail)==1
    assert all(block.source is not None for block in snap.memory.blocks)
    payload=pack_snapshots([snap,snap])
    a,b=unpack_snapshots(payload,"cpu")
    assert a.memory.step==7 and len(a.history.tail)==1
    # Trusted payload round trip preserves arrays and historical identities.
    np.testing.assert_array_equal(a.history.blocks[-1].rows,snap.history.blocks[-1].rows)
    actor.restore(1,snap)
    before=snap.memory.partial.clone()
    row=synth_rows(rng,1);row["first"]=0;row["episode"]=4
    actor.step(row,np.zeros((1,5),np.int64),[1])
    assert actor.memory[0].step==7 and actor.memory[1].step==8
    torch.testing.assert_close(snap.memory.partial,before)
    assert actor.memory[1].blocks[-1].source is actor.history[1].blocks[-1]


@pytest.mark.parametrize("loss_kind",["ppo","cf","injury"])
def test_real_losses_reach_historical_spatial_and_writer(loss_kind):
    m,actor,rng=history_actor()
    snap=actor.snapshot(0)
    rows=synth_rows(rng,3);rows["first"]=0;rows["episode"]=4
    pending=synth_pending(rng,3,m.heads)
    chunk=Chunk(rows,pending,snap)
    encoded=[]
    def remember(module,args,out):
        out.retain_grad(); encoded.append(out)
    hook=m.block_memory.resampler.register_forward_hook(remember)
    outs,b,ctx,selected=forward_chunks(m,[chunk],generator=torch.Generator().manual_seed(11),version=7)
    hook.remove()
    actions=torch.zeros((3,5),dtype=torch.long)
    old,_=m.log_prob(outs[0].detach(),m.gate(b),actions)
    old_before=old.clone()
    if loss_kind=="ppo":
        loss,_=ppo_loss(m,outs,b,actions,old,torch.tensor([1.,-.4,.3]),torch.tensor([1.,0.,-1.]),
                        global_returns=torch.tensor([.5,-.2,.8]))
    elif loss_kind=="cf":
        loss,_=cf_loss(m,outs,b,actions,torch.tensor([1.,.4,1.5]),torch.tensor([1.,0.,-1.]),.1,.2,.5)
    else:
        meta=torch.zeros((3,16),dtype=torch.long);meta[:,3]=1;meta[:,14]=8
        loss=injury_loss(outs[0],outs[2],meta,.1,.2,.1)
    loss.backward()
    assert torch.isfinite(loss)
    # First resampler batch consists exclusively of chosen historical source rows.
    assert encoded[0].grad is not None and encoded[0].grad.abs().sum()>1e-6
    assert m.block_memory.writer.queries.grad.abs().sum()>1e-6
    assert m.structured_spatial.body[0].weight.grad.abs().sum()>1e-6
    torch.testing.assert_close(old,old_before)
    assert len(selected[0])==2


def test_optimizer_second_step_has_fresh_graph():
    m,actor,rng=history_actor()
    chunk=Chunk(synth_rows(rng,2),np.zeros((2,5),np.int64),actor.snapshot(0))
    chunk.rows["first"]=0
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for version in range(2):
        opt.zero_grad(set_to_none=True)
        outs,b,_,_=forward_chunks(m,[chunk],version=version)
        actions=torch.zeros((2,5),dtype=torch.long)
        loss,_=cf_loss(m,outs,b,actions,torch.ones(2),torch.ones(2),1.,.2,.5)
        loss.backward(); opt.step()


def test_generation_labels_and_no_recomputed_cache_writeback():
    m,actor,rng=history_actor(steps=7)
    snap=actor.snapshot(0)
    rows=synth_rows(rng,1);rows["first"]=0
    out,_,ctx,_=forward_chunks(m,[Chunk(rows,np.zeros((1,5),np.int64),snap)],version=7)
    state=ctx["states_out"][0]
    assert state.blocks[-1].writer_version==7
    assert state.blocks[-1].source_versions==(7,7)
    # Old canonical blocks remain original; replacements were only a learner view.
    assert state.blocks[-2].writer_version==snap.memory.blocks[-1].writer_version
    torch.testing.assert_close(state.blocks[-2].value,snap.memory.blocks[-1].value)


def test_teacher_taken_bootstrap_replays_recorded_observations_only():
    m,actor,rng=history_actor()
    snap=actor.snapshot(0)
    rows=synth_rows(rng,3);rows["first"]=0
    pending=synth_pending(rng,3,m.heads)
    traj=TeacherTrajectory(snap,rows,pending,np.zeros((3,5),np.int64))
    got=traj.bootstrap(m)
    with torch.no_grad():
        outputs,_,_,_=forward_chunks(m,[Chunk(rows,pending,snap)],train_history=False)
    assert abs(got-float(outputs[-1][-1]))<1e-6
    assert replace(traj,terminal=True).bootstrap(m)==0.
    assert actor.memory[0].step==8


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_policy_heads_fp32_and_training():
    m,actor,rng=history_actor("cuda")
    rows=synth_rows(rng,2);rows["first"]=0
    b=to_batch(rows,np.zeros((2,5),np.int64),"cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=m(b,need_vg=True,temporal={"states":[actor.memory[0],actor.memory[1]]})
    assert all(x.dtype==torch.float32 for x in out)
    outs,b,_,_=forward_chunks(m,[Chunk(rows,np.zeros((2,5),np.int64),actor.snapshot(0))])
    loss,_=cf_loss(m,outs,b,torch.zeros((2,5),dtype=torch.long,device="cuda"),torch.ones(2,device="cuda"),
                   torch.ones(2,device="cuda"),1.,.2,.5)
    loss.backward()
    assert torch.isfinite(loss)

