"""Performance changes must preserve full inputs, ownership and frozen CRN."""
from dataclasses import replace
import copy
import os
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_visible_input import components, synth_rows
from test_block_training import history_actor
from test_block_lanes import ring_sampler
from isaac_bridge.structured_input import pack_frame,write_frame,from_rows,decode_frame,check_frames
from isaac_bridge.tok_obs import ROW
from isaac_bridge.block_training import move_snapshots,BlockActor,SpatialEncoderGraph,pack_snapshots,unpack_snapshots
from isaac_bridge.block_lanes import BlockLaneService,decode_reference


def test_same_gpu_requires_actual_mps_registration(monkeypatch):
    import subprocess,train_tok
    monkeypatch.delenv('CUDA_MPS_PIPE_DIRECTORY',raising=False)
    with pytest.raises(RuntimeError,match='private'):train_tok.require_mps_client()
    monkeypatch.setenv('CUDA_MPS_PIPE_DIRECTORY','private-test-pipe')
    replies=iter(['42\n','999999999\n'])
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=next(replies)))
    with pytest.raises(RuntimeError,match='not registered'):train_tok.require_mps_client()
    replies=iter(['42\n',str(os.getpid())+'\n'])
    assert train_tok.require_mps_client()==42


def test_compressed_snapshot_preserves_raw_history_and_uncompressed_compatibility():
    import lz4.frame
    model,actor,rng=history_actor(steps=8)
    original=[actor.snapshot(0),actor.snapshot(1)]
    payload=pack_snapshots(original)
    assert payload.startswith(b'\x04\x22\x4d\x18')
    raw=lz4.frame.decompress(payload)
    # History is already compressed in memory, before the outer IPC codec.
    logical_raw=sum(len(b.rows.tobytes()) for s in original for b in s.history.blocks)
    assert len(payload)<logical_raw/3
    for blob in (payload,raw):
        restored=unpack_snapshots(blob,'cpu')
        for a,b in zip(original,restored):
            assert a.history.episode==b.history.episode and a.history.step==b.history.step
            for aa,bb in zip(a.history.blocks,b.history.blocks):
                assert aa.rows.tobytes()==bb.rows.tobytes()
                assert aa.pending.tobytes()==bb.pending.tobytes()
            torch.testing.assert_close(a.memory.partial,b.memory.partial,rtol=0,atol=0)


def test_raw_history_is_resident_compressed_and_legacy_pickle_migrates():
    import pickle,gc,weakref
    from isaac_bridge.block_training import RawBlock
    model,actor,rng=history_actor(steps=9)
    snap=actor.snapshot(0)
    sources=[b.rows for b in snap.history.blocks]+[t[0] for t in snap.history.tail]
    expected=[s.tobytes() for s in sources]
    assert all(not isinstance(s,np.ndarray) for s in sources)
    # Keep the original transport-memory budget, not a compression ratio inflated
    # by 128 KiB of zero padding that the model never consumes.
    from isaac_bridge.tok_obs import ROW
    logical_input=sum(map(len,expected))
    assert sum(len(s.data) for s in sources)<logical_input
    assert logical_input<sum(len(s) for s in sources)*ROW.itemsize/4
    # Existing real saved jobs contain the former ndarray fields.
    old=object.__new__(RawBlock)
    old.__dict__.update(end_step=1,rows=np.asarray(sources[0]),
                        pending=snap.history.blocks[0].pending,versions=(0,0))
    migrated=pickle.loads(pickle.dumps(old,protocol=5))
    assert not isinstance(migrated.rows,np.ndarray)
    assert migrated.rows.tobytes()==expected[0]
    ring=[unpack_snapshots(pack_snapshots([snap]),'cpu')[0] for _ in range(32)]
    for item in ring:
        got=[b.rows for b in item.history.blocks]+[t[0] for t in item.history.tail]
        assert [s.tobytes() for s in got]==expected
        assert all(isinstance(s.data,bytes) for s in got)
    refs=[weakref.ref(item.history.blocks[0].rows) for item in ring]
    del item,got,ring
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_cpu_snapshot_does_not_pin_unselected_packet_storage():
    model,actor,rng=history_actor(steps=7)
    snap=actor.snapshot(0)
    count=len(snap.kv_rows)
    packet=torch.stack(list(snap.kv_rows)*16)
    bank=torch.stack([b.value for b in snap.memory.blocks]*16)
    blocks=tuple(replace(b,value=bank[i]) for i,b in enumerate(snap.memory.blocks))
    snap=replace(snap,kv_rows=packet.unbind()[:count],memory=replace(snap.memory,blocks=blocks))
    for _ in range(2):
        a,b=unpack_snapshots(pack_snapshots([snap,snap]),'cpu')
        for original,x,y in zip(snap.kv_rows,a.kv_rows,b.kv_rows):
            assert x is y
            torch.testing.assert_close(original,x,rtol=0,atol=0)
            assert x.untyped_storage().nbytes()==x.numel()*x.element_size()
        for original,x,y in zip(snap.memory.blocks,a.memory.blocks,b.memory.blocks):
            assert x is y
            torch.testing.assert_close(original.value,x.value,rtol=0,atol=0)
            assert x.value.untyped_storage().nbytes()==x.value.numel()*x.value.element_size()
        snap=a


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_host_decode_complete_ragged_and_dtypes(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rows=np.zeros(3,ROW)
    for i,args in enumerate(((1,0,0,0),(8,128,4416,32),(2,1,81,11))):
        c=components(*args)
        c[1][13:]=np.arange(3)[:,None,None]*.37
        write_frame(rows[i:i+1],pack_frame(*c))
        rows['visible_frame'][i,int(rows['visible_bytes'][i]):]=255
    pending=np.arange(15,dtype=np.int64).reshape(3,5)
    got=from_rows(rows,device,pending)
    args=[torch.from_numpy(rows[k].copy()).to(device) for k in ('visible_frame','visible_bytes','visible_schema')]
    # B42: device decode with host header counts (fixed buckets, no sync) == host decode; the
    # checked legacy device decode (exact buckets) == the same values on its narrower slots.
    ref=decode_frame(*args,counts=check_frames(rows))
    legacy=decode_frame(*args)
    for name,value in ref.items():
        assert got[name].dtype==value.dtype,name
        torch.testing.assert_close(got[name],value,rtol=1e-6,atol=1e-7)
        old=legacy[name]
        common=tuple(slice(0,min(x,y)) for x,y in zip(got[name].shape,old.shape))
        torch.testing.assert_close(got[name][common],old[common],rtol=1e-6,atol=1e-7)
        if name.endswith('_mask'):assert int(got[name].sum())==int(old.sum())
    assert got['v_entity'].shape[1]==128 and got['v_body'].shape[1]==8 and got['v_slots'].shape[1]==32
    assert got['pending'].dtype==torch.int64
    assert got['first'].dtype==torch.uint8
    assert got['v_inventory_mask'].sum().item()==4497


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gather_shared_snapshot_kv(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,actor,rng=history_actor(steps=7)
    one=actor.snapshot(0)
    snapshots=[one,replace(one,kv_rows=one.kv_rows[1:]),actor.snapshot(1),one]
    # The second ref list is deliberately shorter, with overlapping immutable rows.
    for materialize in (False,True):
        moved=move_snapshots(snapshots,device,materialize=materialize)
        assert moved[0].kv_rows[1] is moved[1].kv_rows[0]
        assert moved[0].kv_rows[0] is moved[3].kv_rows[0]
        if materialize:
            for original,got in zip(snapshots,moved):
                ref=torch.stack(original.kv_rows).permute(1,2,3,0,4,5).flatten(3,4).to(device)
                for i,(k,v) in enumerate(got.memory.kv):
                    torch.testing.assert_close(k,ref[i,0],rtol=0,atol=0)
                    torch.testing.assert_close(v,ref[i,1],rtol=0,atol=0)
        again=move_snapshots(moved,device,materialize=True)
        for got in again:assert got.memory.kv[0][0].shape[-2]==len(got.kv_rows)*4


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph parity')
def test_bucketed_spatial_graph_reuses_shapes_without_stale_padding():
    from test_visible_input import model
    m=model().cuda().eval()
    graph=SpatialEncoderGraph(m,4,128,False)
    with torch.inference_mode():
        for n,e,items in ((3,20,20),(4,28,28),(3,17,17),(4,18,18)):
            rows=np.zeros(n,ROW)
            for i in range(n):write_frame(rows[i:i+1],pack_frame(*components(entities=e,inventory=items,seed=i)))
            pending=np.zeros((n,5),np.int64)
            source,z=graph(rows,pending)
            ref=m.encode_block(from_rows(rows,'cuda',pending))
            torch.testing.assert_close(z,ref,rtol=2e-5,atol=3e-6)
            assert len(source['first'])==n
    assert graph.captures==1 and graph.replays==4


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA Reader graph parity')
@pytest.mark.parametrize('bf16',[False,True])
def test_reader_graph_steps_resets_and_weight_publication(bf16):
    from test_visible_input import model
    torch.manual_seed(781)
    m=model().cuda().eval();ref=copy.deepcopy(m)
    actor=BlockActor(m,4,graph_entities=128,bf16=int(bf16))
    eager=BlockActor(ref,4,bf16=bf16)
    pending=np.zeros((4,5),np.int64)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        for t in range(12):
            n=3 if t%3==1 else 4
            rows=synth_rows(n);rows['first']=int(t==0);rows['episode']=1
            if t==6:rows['first'][0]=1;rows['episode'][0]=2
            keys=(np.full(n,t),np.arange(n))
            if t==8:
                probe=from_rows(rows,'cuda',pending[:n])
                before=ref(probe,temporal={'states':eager.memory[:n],'reset_done':True})[0].clone()
                with torch.no_grad():
                    m.pi.bias[0]+=.2
                    m.block_memory.slot_embedding[0,0]+=.5
                    m.structured_spatial.body[0].weight[0,0]+=.5
                ref.load_state_dict(m.state_dict());actor.version=eager.version=1
                after=ref(probe,temporal={'states':eager.memory[:n],'reset_done':True})[0]
                assert (before-after).abs().max().item()>.01
            actor.step(rows,pending[:n],range(n),ukeys=keys)
            eager.step(rows,pending[:n],range(n),ukeys=keys)
            np.testing.assert_allclose(actor.host_output[:,5:],eager.host_output[:,5:],
                                       rtol=.02 if bf16 else 2e-5,atol=.01 if bf16 else 3e-6)
            np.testing.assert_array_equal(actor.host_output[:,:5],eager.host_output[:,:5])
            for i in range(n):
                a,b=actor.memory[i],eager.memory[i]
                assert a.step==b.step and a.partial_versions==b.partial_versions
                torch.testing.assert_close(a.kv_steps,b.kv_steps,rtol=0,atol=0)
                for av,bv in zip(a.kv,b.kv):
                    for aa,bb in zip(av,bv):torch.testing.assert_close(aa,bb,rtol=.03 if bf16 else 2e-5,atol=.02 if bf16 else 3e-6)
    assert actor.reader.replays==12


@pytest.mark.parametrize('replay,old_version',[(False,False),(True,False),(False,True)])
def test_main_teacher_merge_owner_crn_and_version(replay,old_version,monkeypatch):
    model,main,rng=history_actor(steps=7)
    snap=main.snapshot(0)
    rows=synth_rows(rng,3);rows['first']=0
    teacher_model=copy.deepcopy(model)
    if old_version:
        with torch.no_grad():teacher_model.pi.bias[0]+=4.
    ref_actor=BlockActor(teacher_model,2)
    for i in range(2):ref_actor.restore(i,snap)
    pend=np.repeat(snap.pending,2,axis=0)
    seq=np.array([17,19]);lanes=np.array([0,1])
    ref_actor.step(rows[1:],pend,lanes,ukeys=(seq,lanes),evaluate=replay)
    expected=ref_actor.host_output.copy()
    s=ring_sampler()
    service=BlockLaneService(s,model,'cpu',wait=0)
    service.main_actor=main
    s.block_arrays=[None,np.full(2,int(replay))]
    version=int(old_version)
    teacher_snap=replace(snap,policy_version=version)
    assert service.capture_point(0,teacher_snap,teacher_model,version)
    for i in range(2):s.request(i,rows[i+1],snap.pending[0],seq=int(seq[i]),tbegin=1)
    calls=[]
    original=model.forward
    def measured(b,*args,**kwargs):
        calls.append(len(b['v_room']))
        return original(b,*args,**kwargs)
    monkeypatch.setattr(model,'forward',measured)
    service.step_main(rows[:1],snap.pending,[0])
    teacher=service.actors[service.points[0][1]]
    np.testing.assert_array_equal(teacher.host_output[:,:5],expected[:,:5])
    np.testing.assert_allclose(teacher.host_output[:,5:],expected[:,5:],rtol=2e-5,atol=2e-6)
    assert main.memory[0].step==snap.memory.step+1
    assert teacher.memory[0].step==snap.memory.step+1
    assert main.last_contexts[0].memory.step==snap.memory.step
    assert len(main.last_contexts)==1 and len(teacher.last_contexts)==2
    assert main.host_output.shape[0]==1
    assert np.all(s.lane_out[:,2]==version)
    if not old_version:
        assert calls==[3]
        assert service.stat['merged_lane_rows']==2
        if not replay:
            for i in range(2):
                context=s.block_contexts.get(decode_reference(s.lane_t[i]))
                assert context.memory.step==snap.memory.step
        else:
            assert all(decode_reference(s.lane_t[i])==0 for i in range(2))
    else:
        assert calls==[1]
        assert service.stat.get('merged_lane_rows',0)==0
