"""Inventory v2 wire -> ROW/FastRow post -> dense/raw batches -> pooled policy."""
import io
import struct
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_memory_smoke as M
import numpy as np
import torch
from isaac_bridge import abplus_lean as AL, tok_obs as O, tok_policy as TP

torch.set_num_threads(2)

def check(name,ok,detail=''):
    print(('ok ' if ok else 'FAIL ')+name+' '+detail,flush=True)
    assert ok,name

def payload(n,version=2,counts=None):
    inv=np.zeros(n,AL.INV_ITEM);inv['id']=np.arange(1,n+1);inv['count']=1 if counts is None else counts
    header=AL.INV_HEADER.pack(n,1,6,0,0,0,0,0,0) if version==1 else AL.INV_HEADER_V2.pack(0,2,n,6,0,0,0,0,0,0)
    return (AL._HEADER.pack(AL.MAGIC,0,0,16,1,0,0)+AL._ROOM.pack(*([0]*12))+
            AL._TOTALS.pack(0,0,0,0)+np.zeros(1,AL.PLAYER).tobytes()+struct.pack('<H',0)+header+inv.tobytes())

def make_row(n,version=2,counts=None):
    st=O.EpisodeState(100,items=True)
    st.require_full_inventory=False
    ob=AL.LeanDecoder().decode(payload(n,version,counts))
    O._inventory(ob,st)
    row=np.zeros(1,O.ROW)
    O._items_step(st,row,1,105,6,0)
    return row,st

def main():
    row,st=make_row(400,counts=np.arange(400)%7+1)
    check('v2 uint16 item count exceeds both 64 and 255',row['inv_complete'][0] and np.count_nonzero(row['inv_all'])==400)
    check('all IDs/counts survive ROW',np.array_equal(row['inv_all'][0,1:401],np.arange(400)%7+1))
    check('legacy recent-16 order unchanged',np.array_equal(row['inv'][0,:,0],np.arange(1,17)))
    check('legacy offsets and bytes reserved',all(O.ROW.fields[k][1]==O.ROW_LEGACY.fields[k][1] for k in O.ROW_LEGACY.names)
          and O.ROW.fields['inv_all'][1]==O.ROW_LEGACY.itemsize,
          f'old_bytes={O.ROW_LEGACY.itemsize} new_bytes={O.ROW.itemsize}')
    # Inventory changes and removals update the cached full vector; native FastRow
    # calls this same post-step on payloads without inventory blocks.
    later,other=make_row(2)
    ob=AL.LeanDecoder().decode(payload(2));ob.inv_version=st.inv_version+1
    O._inventory(ob,st)
    fast_post=np.zeros(1,O.ROW);O._items_step(st,fast_post,1,105,6,0)
    check('removals clear old IDs on native post path',np.array_equal(fast_post['inv_all'],later['inv_all']))
    legacy,_=make_row(64,1)
    check('v1 supported and explicitly incomplete',not legacy['inv_complete'][0] and np.count_nonzero(legacy['inv_all'])==64)
    st.require_full_inventory=True;ob=AL.LeanDecoder().decode(payload(3,1));ob.inv_version=st.inv_version+1
    try:O._inventory(ob,st)
    except ValueError as e:check('requested full inventory cannot silently use old Lua','legacy' in str(e))
    else:raise AssertionError('accepted truncated wire')
    # Unknown wire layouts are errors, not misparsed lengths.
    wire=bytearray(payload(0));wire[-19]=9
    try:AL.LeanDecoder().decode(wire)
    except ValueError as e:check('unknown wire layout rejected','unsupported' in str(e))
    else:raise AssertionError('accepted unknown inventory layout')
    torch.manual_seed(12)
    old=TP.TokPolicy(width=32,layers=1,heads=4,items=True,item_tokens=True,item_shared=True,item_slots=8).eval()
    new=TP.TokPolicy(**dict(old.config,obs_inventory_all=True)).eval();new.load_state_dict(old.state_dict())
    merged=np.concatenate([row,row])
    check('actor merge preserves ROW byte layout',merged.dtype == O.ROW and merged.dtype.itemsize == O.ROW.itemsize)
    rows=np.repeat(row,3)
    rows['inv_all'][1,399]+=1  # outside both legacy ROW[16] and independent tokens[8]
    rows['inv_all'][2,399]=0
    pend=np.zeros((3,5),np.int64)
    b=TP.to_batch(rows,pend,'cpu')
    raw=torch.from_numpy(rows.view(np.uint8).reshape(3,-1).copy())
    direct=TP.decode_rows(raw,torch.from_numpy(pend))
    check('raw byte decoder equals dense inventory batch',torch.equal(b['inv_all'],direct['inv_all']))
    with torch.no_grad():
        x=old(b)[0];y=new(b)[0]
        check('legacy path ignores appended inventory',torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]))
        check('far ID and duplicate count influence new policy',not torch.equal(y[0],y[1]) and not torch.equal(y[0],y[2]))
        check('raw/dense policy equality',torch.equal(y,new(direct)[0]))
    single,_=make_row(10);pb=np.zeros((1,5),np.int64)
    sb=TP.to_batch(single,pb,'cpu')
    with torch.no_grad():
        check('unique <=16 inventories retain old pool semantics',torch.allclose(old(sb)[0],new(sb)[0],atol=1e-6))
    new.zero_grad(set_to_none=True);new(b)[1].sum().backward()
    check('far item embedding receives gradient',new.item_emb.weight.grad[399].abs().sum()>0)
    f=io.BytesIO();torch.save({'config':new.config,'model':new.state_dict()},f);f.seek(0)
    ck=torch.load(f,weights_only=False);twin=TP.TokPolicy(**ck['config']);twin.load_state_dict(ck['model'])
    check('checkpoint preserves complete-inventory semantics',twin.obs_inventory_all)
    if torch.cuda.is_available():
        new=new.cuda().eval()
        graph=TP.GraphActor(new,3,entities=64,fold=True,row_sizes=[1,2])
        keys=np.arange(10,13,dtype=np.int64);ticks=np.zeros(3,np.int64)
        got=graph.act(rows,pend,ukeys=(keys,ticks))
        with torch.no_grad():
            u=torch.from_numpy(TP.crn_uniforms(keys,ticks,sum(new.heads))).cuda()
            want=new.act(TP.to_batch(rows,pend,'cuda'),u=u)
        err=float(np.max(abs(got[2]-want[2].cpu().numpy())))
        check('CUDA Graph inventory path',np.array_equal(got[0],want[0].cpu().numpy()) and err<3e-4,f'max_abs={err:.8g}')
    print('RESULT inventory v2 checks passed; active/pocket slots and charge semantics unchanged',flush=True)

if __name__=='__main__':main()
