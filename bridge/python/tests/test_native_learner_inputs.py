"""Native learner sources must not expand transport padding on the training path."""
import copy
from dataclasses import replace
import numpy as np
import torch
import pytest
from test_block_training import history_actor
from test_visible_input import synth_rows, synth_pending
from isaac_bridge.tok_obs import ROW
from isaac_bridge.block_training import PackedRows, Chunk, forward_chunks


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_packed_canvas_expansion_matches_all_new_input_fields(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from isaac_bridge.structured_input import from_rows,decode_frame_host,expand_canvases
    rng=np.random.default_rng(88)
    retained=[]
    for n in (1,3,17):
        rows=synth_rows(rng,n);pending=synth_pending(rng,n,(9,9,2,2,2))
        expected=from_rows(rows,'cpu',pending)
        if device=='cuda':actual=from_rows(rows,'cuda',pending)
        else:
            packed=decode_frame_host(rows,packed=True)
            actual={k:torch.from_numpy(v) for k,v in packed.items()}
            actual['v_room'],actual['v_local'],actual['v_map']=expand_canvases(
                actual['v_room'],actual['v_local'],actual['v_map'])
            actual.update(pending=expected['pending'],first=expected['first'])
        assert actual.keys()==expected.keys()
        for key in actual:torch.testing.assert_close(actual[key].cpu(),expected[key],atol=1e-7,rtol=1e-7,msg=key)
        retained.append((actual,{k:v.clone() for k,v in actual.items()}))
    for actual,saved in retained:
        for key in actual:torch.testing.assert_close(actual[key],saved[key],atol=0,rtol=0,msg=key)


def test_compact_history_preserves_input_not_legacy_transport_garbage():
    from isaac_bridge.block_training import source_rows,RawBlock
    from isaac_bridge.structured_input import from_rows
    rows=synth_rows(np.random.default_rng(73),4)
    used=np.zeros(ROW.itemsize,dtype=bool)
    for dtype,offset in ROW.fields.values():used[offset:offset+dtype.itemsize]=True
    rows.view(np.uint8).reshape(len(rows),ROW.itemsize)[:,~used]=173
    packed=PackedRows.pack(rows)
    assert packed.compact_array().dtype.itemsize < ROW.itemsize//2
    expected=from_rows(rows,'cpu',np.zeros((4,5),np.int64))
    for i,n in enumerate(rows['visible_bytes']):rows['visible_frame'][i,n:]=19
    rows['player']=173
    other=PackedRows.pack(rows)
    assert packed.data==other.data
    assert PackedRows.pack(source_rows([packed])).data==packed.data
    actual=from_rows(np.asarray(other),'cpu',np.zeros((4,5),np.int64))
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0,msg=key)
    assert RawBlock(3,packed,np.zeros((4,5),np.int64),(0,)*4).encoding_key==RawBlock(
        3,other,np.zeros((4,5),np.int64),(0,)*4).encoding_key


def test_historical_forward_never_expands_full_transport_rows(monkeypatch):
    model,actor,rng=history_actor(steps=7)
    rows=synth_rows(rng,2);rows['first']=0
    chunks=[Chunk(rows,synth_pending(rng,2,model.heads),actor.snapshot(0))]
    def forbidden(*args,**kwargs):
        raise AssertionError('FULL_TRANSPORT_HISTORY_EXPANSION')
    monkeypatch.setattr(PackedRows,'__array__',forbidden)
    out,_,ctx,chosen=forward_chunks(model,chunks,require_next_state=False)
    (out[0].square().mean()+out[1].square().mean()).backward()
    assert len(chosen[0])==2 and ctx['recompute_stats']['tail_rows']>0
    assert model.block_memory.writer.queries.grad.abs().sum()>0


def test_compact_sources_preserve_outputs_gradients_and_adamw(monkeypatch):
    import isaac_bridge.block_training as training
    model,actor,rng=history_actor(steps=7)
    reference=copy.deepcopy(model)
    rows=synth_rows(rng,2);rows['first']=0
    chunks=[Chunk(rows,synth_pending(rng,2,model.heads),actor.snapshot(0))]
    compact=training.source_rows
    def expanded_sources(sources):
        arrays=[]
        for source in sources:
            source=np.asarray(source)
            array=np.zeros(len(source),dtype=ROW)
            for name in source.dtype.names:
                if name=='visible_frame':array[name][:,:source[name].shape[1]]=source[name]
                else:array[name]=source[name]
            arrays.append(array)
        return np.concatenate(arrays)
    results=[]
    for m,legacy in ((reference,True),(model,False)):
        monkeypatch.setattr(training,'source_rows',
            expanded_sources if legacy else compact)
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1)
        out,_,_,_=forward_chunks(m,chunks,generator=torch.Generator().manual_seed(22),
                                 require_next_state=False,spatial_micro=3)
        (out[0].square().mean()+out[1].square().mean()).backward()
        grads={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()}
        opt.step();results.append((out,grads,opt.state_dict()))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for (n,a),(_,b) in zip(reference.named_parameters(),model.named_parameters()):
        ga,gb=results[0][1][n],results[1][1][n]
        assert (ga is None)==(gb is None),n
        if ga is not None:torch.testing.assert_close(ga,gb,rtol=0,atol=0,msg=n)
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=n)
    for key,a in results[0][2]['state'].items():
        for field,value in a.items():
            torch.testing.assert_close(value,results[1][2]['state'][key][field],rtol=0,atol=0)
