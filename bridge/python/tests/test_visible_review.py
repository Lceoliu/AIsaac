"""Real observation packing for the reviewed input decisions, not legacy fixtures."""
from types import SimpleNamespace
import numpy as np
import torch
from isaac_bridge.structured_input import (SCHEMA,P_FIELDS,PF,STAGE_INDEX,CURSE_START,OVERFLOW_START,
    observation_frame,write_frame,from_rows,prioritize_entities,action_gate)
from isaac_bridge.abplus_lean import ENTITY
from isaac_bridge.tok_obs import ROW
from test_visible_input import model,components,pack_frame


def observation(entities):
    players=[dict.fromkeys(P_FIELDS,0) for _ in range(2)]
    for p in players:p.update(hearts=6,max_hearts=8,soul=2,black=1,bone=1,eternal=1,golden=1)
    players[1].update(x=20)
    return SimpleNamespace(players=players,entities=entities,doors=[dict(x=0,y=0,open=1,locked=0,target_type=1,seen=3)],
        grid=[(0,17,0,0,0,0,0)],room=[0,0,0,0,0,7,0,0,0,0],map=[],
        terrain=dict(width=1,height=1,cells=[(0,0,0,0,1,1,0,0,0,0)]),
        visible=dict(schema=SCHEMA,stage=7,curses=(1<<31)|8|64,capabilities=1,inventory=[],slots=[],
                     bodies=[dict(controller=i,controller_known=1,animation=0,animation_known=1) for i in range(2)]))


def test_stage_curse_bits_overflow_and_uncapped_doors_survive_real_packer():
    entities=np.zeros(170,dtype=ENTITY)
    entities['type'][:10]=5;entities['kind'][:10]=5
    entities['type'][10:90]=10;entities['kind'][10:90]=6
    entities['type'][90:110]=3
    entities['type'][110:160]=9;entities['kind'][110:160]=2
    entities['type'][160:]=1000
    entities['x']=np.arange(170)[::-1]
    row=np.zeros(1,dtype=ROW);write_frame(row,observation_frame(observation(entities)))
    b=from_rows(row,'cpu',np.zeros((1,5),np.int64))
    assert b['v_body'].shape[-1]==PF
    torch.testing.assert_close(b['v_body'][0,:2,STAGE_INDEX],torch.full((2,),7/16))
    for bit in range(32):assert torch.equal(b['v_body'][0,:2,CURSE_START+bit],torch.full((2,),float(bit in (3,6,31))))
    torch.testing.assert_close(b['v_body'][0,:2,OVERFLOW_START:]*128,torch.tensor([[0.,0.,0.,32.,10.]]).repeat(2,1))
    assert b['v_entity_mask'].sum()==128
    assert b['v_room'][0,6,0,0]==1 and b['v_room'][0,12,0,0]==1
    # Unknown curse continues masking health; adding curse inputs must not reveal it.
    assert b['v_body'][0,:2,5:12].count_nonzero()==0
    known=observation(entities);known.visible['curses']&=~8
    write_frame(row,observation_frame(known))
    unmasked=from_rows(row,'cpu',np.zeros((1,5),np.int64))
    assert unmasked['v_body'][0,:2,5:12].count_nonzero()==14


def test_priority_distance_and_stable_ties():
    entities=np.zeros(132,dtype=ENTITY);entities['type']=9;entities['kind']=2
    entities['variant']=np.arange(132)
    entities['x']=np.arange(132)
    entities['type'][-3:]=[3,10,1023];entities['kind'][-3:]=[0,6,0]
    entities['x'][0:2]=0
    kept,overflow=prioritize_entities(entities,[dict(x=0,y=0)])
    assert kept['variant'][:5].tolist()==[131,130,129,0,1]
    assert overflow.tolist()==[0,0,0,4,0]


def test_use_actions_have_no_existence_or_charge_masks():
    c=components();c[0][:,14]=0;c[-1][:]=0
    rows=np.zeros(1,dtype=ROW);write_frame(rows,pack_frame(*c))
    b=from_rows(rows,'cpu',np.zeros((1,5),np.int64))
    assert action_gate(b).tolist()==[[True,True,True]]
    m=model();logits=torch.zeros(1,sum(m.heads))
    lp,entropy=m.log_prob(logits,m.gate(b),torch.tensor([[0,0,1,1,1]]))
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()


def test_local_token_retains_left_right_layout():
    torch.manual_seed(100)
    m=model();local=m.structured_spatial.local
    assert not any(isinstance(layer,torch.nn.AdaptiveAvgPool2d) for layer in local.modules())
    a=torch.zeros(1,16,9,9);a[0,0,4,2]=1
    b=torch.flip(a,[-1])
    assert local(a).shape==(1,32)
    assert not torch.allclose(local(a),local(b))
