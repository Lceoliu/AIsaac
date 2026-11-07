"""Loss-only training must not encode causal-invisible Writer tails."""
import copy
from dataclasses import asdict,replace
import numpy as np
import torch
import pytest
from test_block_memory import small
from test_block_training import history_actor
from test_visible_input import synth_rows,synth_pending
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.block_training import BlockActor,Chunk,forward_chunks,cf_loss,injury_loss,pack_snapshots,unpack_snapshots


def test_legacy_graph_rejects_visible_input_instead_of_silently_dropping_it():
    from isaac_bridge.tok_policy import GraphActor
    rows=synth_rows(np.random.default_rng(45),1)
    with pytest.raises(ValueError,match='requires BlockActor'):
        GraphActor.__new__(GraphActor).submit(rows,np.zeros((1,5),np.int64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='snapshot H2D/D2H requires CUDA')
def test_snapshot_metadata_bulk_copy_preserves_dtypes_values_and_cpu_ownership():
    from isaac_bridge.block_training import move_snapshots
    model,actor,rng=history_actor(steps=8)
    snapshots=[]
    for i in range(4):
        snapshots.append(actor.snapshot(0))
        rows=synth_rows(rng,1);rows['first']=0
        actor.step(rows,actor.pending[0],[0])
    gpu=move_snapshots(snapshots,'cuda',materialize=False)
    restored=move_snapshots(gpu,'cpu',materialize=False)
    for original,on_gpu,on_cpu in zip(snapshots,gpu,restored):
        for name in ('kv_steps','kv_versions','partial'):
            x=getattr(original.memory,name);y=getattr(on_gpu.memory,name);z=getattr(on_cpu.memory,name)
            assert y.device.type=='cuda' and x.dtype==y.dtype==z.dtype
            torch.testing.assert_close(x,y.cpu(),rtol=0,atol=0)
            torch.testing.assert_close(x,z,rtol=0,atol=0)
            assert z.untyped_storage().nbytes()==z.numel()*z.element_size()


def test_same_device_snapshot_metadata_does_not_pin_a_larger_batch():
    from isaac_bridge.block_training import move_snapshots
    _,actor,_=history_actor(steps=9)
    snap=actor.snapshot(0)
    fields={}
    for name in ('kv_steps','kv_versions','partial'):
        x=getattr(snap.memory,name)
        fields[name]=torch.cat((x.reshape(-1),x.reshape(-1)))[:x.numel()].reshape(x.shape)
    batched=replace(snap,memory=replace(snap.memory,**fields))
    restored=move_snapshots([batched],'cpu',materialize=False)[0]
    for name in fields:
        x=getattr(restored.memory,name)
        torch.testing.assert_close(x,fields[name],rtol=0,atol=0)
        assert x.untyped_storage().nbytes()==x.numel()*x.element_size()


@pytest.mark.parametrize('device,amp',[('cpu',False),('cuda',False),('cuda',True)])
def test_teacher_all_32_phases_loss_only_preserves_losses_and_all_gradients(device,amp):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(211)
    rng=np.random.default_rng(211)
    model=TokPolicy(width=32,layers=1,heads=4,items=True,item_tokens=True,global_token=True,
                    block_memory=True,block_config=asdict(small(block_steps=32,window=128,memory_blocks=8))).to(device)
    actor=BlockActor(model,1)
    chunks=[]
    for step in range(288):
        rows=synth_rows(rng,1);rows['first']=int(step==0);rows['episode']=3
        pending=synth_pending(rng,1,model.heads)
        if step>=256:chunks.append(Chunk(rows,pending,actor.snapshot(0)))
        actor.step(rows,pending,[0])
    # Cached tail latents deliberately predate the learner's current parameters.
    with torch.no_grad():model.structured_spatial.body[0].weight.add_(.007)
    reference=copy.deepcopy(model)
    results=[]
    for m,require_next_state in ((reference,True),(model,False)):
        encoded=[]
        hook=m.block_memory.resampler.register_forward_pre_hook(lambda module,args:encoded.append(len(args[0])))
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
            out,b,ctx,selected=forward_chunks(m,chunks,generator=torch.Generator().manual_seed(73),
                                             require_next_state=require_next_state)
        hook.remove()
        actions=torch.zeros((32,5),dtype=torch.long,device=device)
        cf,_=cf_loss(m,out,b,actions,torch.ones(32,device=device),torch.linspace(-1,1,32,device=device),.4,.2,.5)
        meta=torch.zeros((32,16),dtype=torch.long,device=device);meta[:,3]=1;meta[:,14]=8
        injury=injury_loss(out[0],out[2],meta,.4,.2,.1)
        (cf+injury).backward()
        assert all(len(x)==4 for x in selected)
        assert m.block_memory.writer.queries.grad.abs().sum()>0
        results.append((out,cf.detach(),injury.detach(),sum(encoded),ctx))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    for i in (1,2):torch.testing.assert_close(results[0][i],results[1][i],atol=3e-6,rtol=3e-5)
    assert results[0][3]-results[1][3]==sum(range(32))
    assert len(results[0][4]['states_out'])==32 and results[1][4]['states_out']==[]
    for (name,a),(other,b) in zip(reference.named_parameters(),model.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=1e-4,msg=name)


def test_loss_only_still_recomputes_tail_when_new_block_becomes_visible():
    model,actor,rng=history_actor(steps=7)
    reference=copy.deepcopy(model)
    rows=synth_rows(rng,2);rows['first']=0;rows['episode']=4
    chunks=[Chunk(rows,synth_pending(rng,2,model.heads),actor.snapshot(0))]
    results=[]
    for m,require_next_state in ((reference,True),(model,False)):
        encoded=[]
        hook=m.block_memory.resampler.register_forward_pre_hook(lambda module,args:encoded.append(len(args[0])))
        out,_,_,selected=forward_chunks(m,chunks,generator=torch.Generator().manual_seed(9),
                                        require_next_state=require_next_state)
        hook.remove();out[1].square().sum().backward()
        results.append((out,sum(encoded)))
    assert results[0][1]==results[1][1]
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
    for (name,a),(_,b) in zip(reference.named_parameters(),model.named_parameters()):
        assert (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=1e-4,msg=name)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_same_writer_inputs_across_ipc_share_recompute_not_loss_budget(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import isaac_bridge.block_training as training
    model,actor,rng=history_actor(device=device,steps=6)
    reference=copy.deepcopy(model)
    snap=actor.snapshot(0)
    a=unpack_snapshots(pack_snapshots([snap]),'cpu')[0]
    b=unpack_snapshots(pack_snapshots([snap]),'cpu')[0]
    assert a.history.blocks[0] is not b.history.blocks[0]
    # Fix the same two available window-external blocks, independent of draw order.
    monkeypatch.setattr(training,'historical_blocks',lambda s,g,config:(s.blocks[0],s.blocks[1]))
    rows=synth_rows(rng,2);rows['first']=0;rows['episode']=4
    pending=synth_pending(rng,2,model.heads)
    chunks=[Chunk(rows[i:i+1],pending[i:i+1],s) for i,s in enumerate((a,b))]
    results=[]
    for m,dedup in ((reference,False),(model,True)):
        out,batch,ctx,selected=forward_chunks(m,chunks,require_next_state=False,deduplicate_sources=dedup)
        loss,_=cf_loss(m,out,batch,torch.zeros((2,5),dtype=torch.long,device=device),torch.tensor([1.,.7],device=device),
                       torch.tensor([.5,-.3],device=device),.4,.2,.5)
        loss.backward();results.append((out,ctx))
        assert [len(s) for s in selected]==[2,2]
    assert results[0][1]['recompute_stats']['encoded_blocks']==4
    assert results[1][1]['recompute_stats']['encoded_blocks']==2
    for x,y in zip(results[0][0],results[1][0]):torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
    for (name,x),(_,y) in zip(reference.named_parameters(),model.named_parameters()):
        assert (x.grad is None)==(y.grad is None),name
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=1e-4,msg=name)
    source=a.history.blocks[0]
    other=source.pending.copy();other[0,0]+=1
    assert source.encoding_key!=replace(source,pending=other).encoding_key


def test_room_unpack_all_19bit_values_channel_first():
    from isaac_bridge.structured_input import _unpack_room_host,ROOM_BITS,ROOM_SCALES
    packed=np.arange(1<<19,dtype=np.uint32)
    x=np.empty((1,len(packed),4),dtype=np.float32)
    x[0,:,0]=packed;x[0,:,1:]=np.array([.125,-.75,3.5],np.float32)
    expected=np.empty((1,16,len(packed)),dtype=np.float32)
    shift=0
    for c,(bits,scale) in enumerate(zip(ROOM_BITS,ROOM_SCALES)):
        expected[0,c]=((packed>>shift)&((1<<bits)-1)).astype(np.float32)/np.float32(scale)
        shift+=bits
    expected[:,13:]=x[...,1:].transpose(0,2,1)
    actual=_unpack_room_host(x.reshape(1,-1))
    np.testing.assert_array_equal(actual,expected)
    assert actual.flags.c_contiguous
