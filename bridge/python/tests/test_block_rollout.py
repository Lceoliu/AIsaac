"""Native rollout storage, carry, two-actor IPC and chunk batching."""
import numpy as np
import torch
from types import SimpleNamespace

from test_block_training import history_actor
from test_visible_input import synth_rows,synth_pending
from isaac_bridge.block_training import forward_bucketed,forward_chunks,move_snapshots
from train_tok import Rollout,store,rollout_scalars,rollout_view,merge_rollouts,gae


def _spawned_visible_learner(queue,pid,descriptors,scalars):
    from train_tok import attach_arrays
    from isaac_bridge.structured_input import from_rows
    torch.set_num_threads(1)
    model,_,_=history_actor(steps=0)
    keep=[]
    roll=rollout_view(attach_arrays(pid,descriptors,keep),scalars,None)
    used=np.concatenate([roll.rows[i,:n] for i,n in enumerate(roll.length)])
    b=from_rows(used,'cpu')
    assert b['v_inventory_mask'].sum()==len(used)*20
    _,_,mask=gae(roll,.99,.95)
    plan=roll.block.plan(roll,mask)
    with torch.no_grad(): result,_=forward_bucketed(model,plan['chunks'],train_history=False)
    roll.reward[0,0]=42.
    queue.put((result[1].numpy(),len(used),len(plan['chunks'])))


def test_spawned_memfd_visible_rows_and_history():
    import os,sys,pytest
    from train_tok import SharedArrays
    if not sys.platform.startswith('linux'):
        pytest.skip('Linux memfd /proc mapping')
    torch.set_num_threads(1)
    model,_,roll,_=collect()
    allocator=SharedArrays()
    for key,value in list(vars(roll).items()):
        if isinstance(value,np.ndarray):
            shared=allocator(value.shape,value.dtype);shared[:]=value;setattr(roll,key,shared)
    _,_,mask=gae(roll,.99,.95)
    plan=roll.block.plan(roll,mask)
    with torch.no_grad(): expected,_=forward_bucketed(model,plan['chunks'],train_history=False)
    ctx=torch.multiprocessing.get_context('spawn');queue=ctx.Queue()
    child=ctx.Process(target=_spawned_visible_learner,
                      args=(queue,os.getpid(),allocator.describe(roll),rollout_scalars(roll)))
    child.start()
    values,rows,chunks=queue.get(timeout=120)
    child.join(30)
    assert child.exitcode==0 and rows==22 and chunks==len(plan['chunks'])
    np.testing.assert_array_equal(values,expected[1].numpy())
    assert roll.reward[0,0]==42.


def collect(n=11):
    model,actor,rng=history_actor(steps=0)
    roll=Rollout(2,20,5,value_g=True,block_steps=4)
    values=[]
    for t in range(n):
        rows=synth_rows(rng,2);rows["first"]=int(t==0);rows["episode"]=1
        if t>=6:
            rows["episode"][1]=2
            rows["first"][1]=int(t==6)
        rows["done"]=0;rows["t"]=t
        pending=synth_pending(rng,2,model.heads)
        result=actor.step(rows,pending,[0,1])
        action,lp,v,vg=[x.cpu().numpy() for x in result]
        store(roll,np.array([0,1]),rows,pending,action,lp,v,0,vg=vg,block_context=actor.last_contexts)
        roll.reward[:,t]=.1
        if t==5: roll.terminal[1,t]=True
        values.append(v)
    return model,actor,roll,np.stack(values,1)


def test_native_rollout_plan_and_batched_recompute():
    model,actor,roll,values=collect()
    _,_,mask=gae(roll,.99,.95)
    plan=roll.block.plan(roll,mask)
    assert max(len(c.rows) for c in plan["chunks"])<=4
    assert len(roll.block.starts[0])==3
    assert len(roll.block.starts[1])==4
    assert len(roll.block.open)==2
    # No per-step KV array; only starts plus the currently open decision.
    assert roll.t_kv is None and roll.block.open[0][0]==10
    with torch.no_grad():
        output,ctx=forward_bucketed(model,plan["chunks"],train_history=False)
    torch.testing.assert_close(output[1],torch.from_numpy(roll.value.reshape(-1)[plan["pos"]]),atol=3e-6,rtol=3e-5)
    assert ctx["o_seq"].shape[0]==len(plan["pos"])
    assert all(not c.rows["first"][1:].any() for c in plan["chunks"])
    assert plan["weight"].sum()==mask.sum()


def test_open_decision_carry_has_pre_observation_state():
    model,actor,roll,_=collect()
    carry=Rollout(2,20,5,value_g=True,block_steps=4)
    carry.carry(roll)
    assert carry.length.tolist()==[1,1]
    for worker in range(2):
        old=roll.block.open[worker][1]
        new=carry.block.starts[worker][0]
        assert new is old
        assert new.memory.step==actor.memory[worker].step-1
        np.testing.assert_array_equal(carry.pending[worker,0],new.pending[0])
    mask=np.zeros((2,20),bool);mask[:,0]=True
    plan=carry.block.plan(carry,mask)
    with torch.no_grad(): out,_=forward_bucketed(model,plan["chunks"],train_history=False)
    torch.testing.assert_close(out[1],torch.from_numpy(carry.value[:,0]),atol=3e-6,rtol=3e-5)


def test_single_and_two_actor_native_ipc():
    model,actor,roll,_=collect()
    arrays={k:v for k,v in vars(roll).items() if isinstance(v,np.ndarray)}
    scalars=rollout_scalars(roll)
    one=rollout_view(arrays,scalars,None)
    assert one.block.starts[1][6].memory.step==0
    full={k:np.concatenate([v,v]) for k,v in arrays.items()}
    merged,nbytes=merge_rollouts(full,[(0,scalars,None),(2,scalars,None)],{},"cpu")
    assert nbytes==0 and merged.n==4
    assert sorted(merged.block.starts)==[0,1,2,3]
    _,_,mask=gae(merged,.99,.95)
    plan=merged.block.plan(merged,mask)
    with torch.no_grad(): result,_=forward_bucketed(model,plan["chunks"],train_history=False)
    torch.testing.assert_close(result[1],torch.from_numpy(merged.value.reshape(-1)[plan["pos"]]),atol=3e-6,rtol=3e-5)


def test_shared_bank_is_transferred_once_between_chunk_snapshots():
    model,actor,roll,_=collect()
    a=roll.block.starts[0][4]
    b=roll.block.starts[0][8]
    assert a.memory.blocks[-1] is b.memory.blocks[0]
    aa,bb=move_snapshots([a,b],"cuda" if torch.cuda.is_available() else "cpu")
    assert aa.memory.blocks[-1] is bb.memory.blocks[0]
    assert aa.memory.blocks[-1].source is bb.memory.blocks[0].source


def test_bucketed_loss_reaches_all_lengths_and_history():
    model,actor,roll,_=collect()
    _,_,mask=gae(roll,.99,.95)
    plan=roll.block.plan(roll,mask)
    output,ctx=forward_bucketed(model,plan["chunks"])
    weights=torch.from_numpy(plan["weight"])
    loss=(output[1].square()*weights).sum()/weights.sum()
    loss.backward()
    assert model.block_memory.writer.queries.grad.abs().sum()>1e-6
    assert torch.isfinite(loss)


def test_legacy_rollout_like_object_without_block_attribute():
    model,actor,roll,_=collect(1)
    legacy=SimpleNamespace(**{k:v for k,v in vars(roll).items() if k!='block'})
    assert 'block_payload' not in rollout_scalars(legacy)
    store(legacy,np.array([0]),roll.rows[0,:1],roll.pending[0,:1],roll.actions[0,:1],
          roll.logp[0,:1],roll.value[0,:1],0)
    assert legacy.length[0]==2


def test_rollout_view_consumes_serialized_history_payload():
    _,_,roll,_=collect(1)
    arrays={k:v for k,v in vars(roll).items() if isinstance(v,np.ndarray)}
    view=rollout_view(arrays,rollout_scalars(roll),None)
    assert 'block_payload' not in vars(view)
    assert view.block.starts
