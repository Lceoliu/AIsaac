"""New input contract fixtures and executable acceptance; no legacy input conversion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
torch.set_num_threads(4)
from isaac_bridge.structured_input import (PF,EF,NF,SF,RC,SCHEMA,pack_frame,write_frame,
                                          from_rows,StructuredSpatial)
from isaac_bridge.tok_obs import ROW
from isaac_bridge.tok_policy import TokPolicy,to_batch
from test_memory_smoke import synth_pending


def components(bodies=1,entities=3,inventory=20,slots=5,seed=1):
    rng=np.random.default_rng(seed)
    p=rng.random((bodies,PF),dtype=np.float32);p[:,35]=np.arange(bodies);p[:,36]=0;p[:,37:]=1
    r=rng.integers(0,2,(RC,16,28)).astype(np.float32)
    l=rng.integers(0,2,(bodies,RC,9,9)).astype(np.float32)
    m=rng.integers(0,2,(8,13,13)).astype(np.float32)
    e=rng.random((entities,EF),dtype=np.float32);e[:,:3]=[10,1,0];e[:,-1]=1
    n=np.zeros((inventory,NF),np.float32);n[:,0]=np.arange(inventory)%bodies;n[:,1]=np.arange(inventory)+1;n[:,2:]=1
    s=np.zeros((slots,SF),np.float32);s[:,0]=np.arange(slots)%bodies;s[:,1]=1;s[:,2]=np.arange(slots);s[:,3:11]=1
    return [p,r,l,m,e,n,s]


def synth_rows(rng,n=None,seed=1):
    # Keep only old metadata for existing trainer tests; model inputs are independent.
    from test_memory_smoke import synth_rows as legacy_fixture
    if n is None:n,rng=rng,np.random.default_rng(seed)
    rows=legacy_fixture(rng,n)
    for i in range(n):write_frame(rows[i:i+1],pack_frame(*components(seed=seed+i)))
    return rows


def model():
    return TokPolicy(width=32,layers=1,heads=4,items=True,item_tokens=True,global_token=True,
                     block_memory=True,block_config=dict(latent_width=32,width=32,layers=2,
                     heads=4,writer_heads=4,block_steps=2,window=3,memory_tokens=2,memory_blocks=3))


def test_block_rejects_compatible_weight_migration():
    from isaac_bridge.tok_policy import load_compatible
    m=model()
    with pytest.raises(ValueError,match='visible-v2'):
        load_compatible(m,m.state_dict())


def test_eval_requires_explicit_input_schema():
    from eval_tok import build_eval_model
    m=model();cfg=dict(m.config);cfg.pop('input_schema')
    with pytest.raises(ValueError,match='schema'):
        build_eval_model({'config':cfg,'model':m.state_dict()},True,'cpu')
    exact,changed=build_eval_model({'config':m.config,'model':m.state_dict()},True,'cpu')
    assert changed==[]
    assert all(torch.equal(v,exact.state_dict()[k]) for k,v in m.state_dict().items())


def test_auto_resume_rejects_legacy_schema_before_runtime(monkeypatch,tmp_path,capsys):
    import train_tok
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{'config':{'block_config':{}}})
    with pytest.raises(SystemExit) as exc:
        train_tok.main(['--resume','old-block.pt','--groups-file','unused','--out',str(tmp_path)])
    assert exc.value.code==2
    assert 'exact visible-v2 checkpoint' in capsys.readouterr().err


def test_complete_inventory_owner_slots_and_entities():
    rows=np.zeros(2,ROW)
    for i,args in enumerate(((2,128,80,11),(1,1,2,3))):write_frame(rows[i:i+1],pack_frame(*components(*args)))
    b=from_rows(rows,'cpu')
    assert b['v_body_mask'].sum().item()==3
    assert b['v_entity_mask'].sum().item()==129
    assert b['v_inventory_mask'].sum().item()==82
    assert b['v_slots_mask'].sum().item()==14
    assert b['v_inventory'][0,79,1].item()==80


def test_reject_old_schema_and_transport_overflow():
    with pytest.raises(ValueError,match='schema'):from_rows(np.zeros(1,ROW),'cpu')
    with pytest.raises(ValueError,match='no truncation'):pack_frame(*components(inventory=10000))


def test_complete_target_catalog_multibody_budget():
    # Eight bodies, every AB+ collectible on each, all 128 entities: no N truncation.
    c=components(bodies=8,entities=128,inventory=8*552,slots=32)
    c[5][:,1]=np.arange(8*552)%552+1
    rows=np.zeros(1,ROW);write_frame(rows,pack_frame(*c))
    b=from_rows(rows,'cpu')
    assert b['v_inventory_mask'].sum().item()==8*552


def test_new_spatial_never_calls_old_and_reads_independent_tokens():
    torch.manual_seed(12);m=model();r=synth_rows(1);b=to_batch(r,np.zeros((1,5),np.int64),'cpu')
    def forbidden(*a,**kw):raise AssertionError('legacy embedding called')
    m._embed_old=forbidden
    z=m.encode_block(b)
    assert z.shape==(1,4,32)
    changed={k:v.clone() for k,v in b.items()}
    for k in ('player','grid','map','patch','inv','pinv','pitem','doors'):changed[k].fill_(0)
    torch.testing.assert_close(m.encode_block(changed),z,rtol=0,atol=0)
    assert torch.equal(m.gate(changed),m.gate(b))
    for name,index in [('v_body',(0,0,0)),('v_room',(0,0,2,5)),('v_local',(0,0,5)),
                       ('v_inventory',(0,19,2)),('v_slots',(0,4,6))]:
        c={k:v.clone() for k,v in b.items()};c[name][index]+=2
        assert not torch.equal(m.encode_block(c),z),name
    z.square().mean().backward()
    for prefix in ('body','room','local','inventory','slots'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.structured_spatial.named_parameters() if n.startswith(prefix))


def test_padding_unknown_and_semantic_permutation():
    m=model().eval();r=synth_rows(1);b=to_batch(r,np.zeros((1,5),np.int64),'cpu');z=m.encode_block(b)
    c={k:v.clone() for k,v in b.items()}
    # Padded raw values must not become keys or affect real tokens.
    for name in ('body','local','entity','inventory','slots'):
        c['v_'+name][~c['v_'+name+'_mask']]=0
    torch.testing.assert_close(m.encode_block(c),z)
    for name in ('inventory','slots'):
        c['v_'+name]=c['v_'+name].flip(1);c['v_'+name+'_mask']=c['v_'+name+'_mask'].flip(1)
    torch.testing.assert_close(m.encode_block(c),z,rtol=2e-5,atol=2e-6)
    b['v_slots'][:,:,8]=0;b['v_slots'][:,:,9]=0;b['v_slots'][:,:,10]=0
    z=m.encode_block(b);b['v_slots'][:,:,3]=123;b['v_slots'][:,:,5:8]=234
    torch.testing.assert_close(m.encode_block(b),z,rtol=0,atol=0)


def test_transport_tail_and_copied_history():
    r=synth_rows(1);saved=r.copy();size=int(r['visible_bytes'][0]);r['visible_frame'][:,size:]=255
    a=from_rows(r,'cpu');b=from_rows(saved,'cpu')
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    write_frame(r,pack_frame(*components(seed=5)))
    assert not np.array_equal(r['visible_frame'],saved['visible_frame'])


def test_new_math_path_and_no_legacy_parameters():
    m=model();b=to_batch(synth_rows(2),np.zeros((2,5),np.int64),'cpu')
    normal=m.encode_block(b);m.set_math_attention(True);fast=m.encode_block(b)
    torch.testing.assert_close(fast,normal,rtol=2e-5,atol=2e-6)
    assert not any(k.startswith(('player.','room.','itok.','blocks.','gtok.tok')) for k in m.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA collation parity')
def test_one_copy_collation_matches_device_decode():
    from isaac_bridge.structured_input import decode_frame
    r=synth_rows(4)
    a=from_rows(r,'cuda')
    b=decode_frame(torch.from_numpy(r['visible_frame'].copy()).cuda(),
                   torch.from_numpy(r['visible_bytes'].copy()).cuda(),
                   torch.from_numpy(r['visible_schema'].copy()).cuda())
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=1e-6,atol=1e-7)


def test_log_rejects_missing_schema_and_wrong_row_layout(tmp_path):
    import json
    from isaac_bridge.tok_stream import StreamLog,iter_records
    log=StreamLog(tmp_path,32,5,input_schema=SCHEMA);log.close()
    p=tmp_path/'stream/format.json';fmt=json.loads(p.read_text(encoding='utf-8'))
    fmt.pop('input_schema');p.write_text(json.dumps(fmt),encoding='utf-8')
    with pytest.raises(ValueError,match='schema'):list(iter_records(p.parent,'row'))
    fmt['input_schema']=SCHEMA;fmt['row_itemsize']=20800;p.write_text(json.dumps(fmt),encoding='utf-8')
    with pytest.raises(ValueError,match='layout'):list(iter_records(p.parent,'row'))
