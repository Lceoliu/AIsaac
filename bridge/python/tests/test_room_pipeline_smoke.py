"""Production Rollout/IPC/temporal_part/CF/eval integration; optional CUDA graphs."""
import copy
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
import train_tok as TT
from isaac_bridge import tok_policy as TP
from isaac_bridge.room_memory import RoomSourceBatch, encode_visits

torch.set_num_threads(2)
CFG = dict(width=32, layers=1, heads=4, **TS.BASE, **TS.FUSED,
           temporal=True, temporal_width=32, temporal_layers=2, temporal_heads=4,
           temporal_window=4, temporal_rooms=4, room_memory=True, obs_inventory_all=True)

def check(name, ok, detail=''):
    print(('ok ' if ok else 'FAIL ') + name + ' ' + detail, flush=True)
    assert ok, name

def collect(model, seq, pend, cut=7):
    n = len(seq[0]); dev = next(model.parameters()).device
    cfg = dict(P=3, L=2, D=32, R=4, kv='fp32', device=dev, raw_rooms=True)
    rolls = [TT.Rollout(n, 32, len(model.heads), value_g=True, temporal=cfg) for _ in range(2)]
    st = TP.TemporalState(model, n, kv_dtype=torch.float32)
    vals, logits = [], []
    roll = rolls[0]
    with torch.no_grad():
        for t, (r, p) in enumerate(zip(seq, pend)):
            if t == cut:
                rolls[1].carry(roll)
                roll = rolls[1]
            td = st.step_inputs(model, r, p, np.arange(n))
            out = model(TP.to_batch(r, p, dev), temporal=td, need_vg=True)
            vals.append(out[1].cpu()); logits.append(out[0].cpu())
            TT.store(roll, np.arange(n), r, p, np.zeros((n, len(model.heads)), np.int64),
                     np.zeros(n, np.float32), out[1].cpu().numpy(), 0,
                     vg=out[-1].cpu().numpy(), tinfo=TP.temporal_out(td['out']))
    return rolls, vals, logits, st

def plan(roll):
    mask = np.arange(roll.cap)[None] < roll.length[:, None]
    mp = TP.temporal_chunks(roll.length, mask, 4, None)
    return mp, TP.temporal_inputs(roll, mp['tch'])

def part(model, roll, mp, td, ids, table):
    flat = np.concatenate([mp['pos'][o:o + n] for o, n in mp['tch'][ids, :2]])
    tr = TT.temporal_rows(roll, flat, roll.cap, next(model.parameters()).device)
    tp = TP.temporal_part(td, ids, mp['tch'][ids, 1], tr['ep'], table)
    b = TP.to_batch(roll.rows.reshape(-1)[flat], roll.pending.reshape(-1, roll.nh)[flat], tr['ep'].device)
    return model(b, temporal=dict(mode='chunk', plan=tp, o_prev=tr['o_prev'], has=tr['has']), need_vg=True), flat

def main():
    torch.manual_seed(9)
    model = TP.TokPolicy(**CFG).eval()
    rng = np.random.default_rng(9)
    seq = TS.make_seq(rng, nw=2, T=18, changes={(0,3),(0,10),(0,13),(1,9),(1,15)}, resets={(0,8)}, exits={(1,6)})
    for row in seq:
        for w in range(len(row)):
            row['inv_all'][w,row['inv'][w,:,0]]=row['inv'][w,:,1]
        row['inv_all'][:,0]=0
        row['inv_all'][:,399]=2
        row['inv_complete']=1
    pend = [M.synth_pending(rng, 2, model.heads) for _ in seq]
    rolls, vals, logits, st = collect(model, seq, pend)
    for ri, roll in enumerate(rolls):
        mp, td = plan(roll)
        ids = np.arange(len(mp['tch']))
        with torch.no_grad():
            rb = RoomSourceBatch(model, td, ids, 4)
            table = model.tatt.room_table(rb.source, rb.td, track_grad=True)
            out, flat = part(model, roll, mp, rb.td, ids, table)
        times = flat % roll.cap + (0 if ri == 0 else 6)
        workers = flat // roll.cap
        target = torch.stack([logits[t][w] for w, t in zip(workers, times)])
        err = float((out[0] - target).abs().max())
        check(f'production actor/chunk parity rollout{ri}', err < 3e-5, f'max_abs={err:.8g}')
    roll = rolls[1]
    check('room crossing rollout boundary keeps entire visit',
          any(len(r[0]) == 3 and r[0]['t'][0] == 6 for r in roll.raw_rooms[1]))
    # IPC pickle transports immutable visits in scalars; full shared ndarray layout remains unchanged.
    restored = TT.rollout_view({k:v for k,v in vars(roll).items() if isinstance(v,np.ndarray)},
                               pickle.loads(pickle.dumps(TT.rollout_scalars(roll))), (roll.t_kv,roll.t_o))
    check('IPC raw visit transport', np.array_equal(restored.raw_rooms[1][0][0], roll.raw_rooms[1][0][0]))
    halves=[]; full={}
    for w in range(2):
        local=collect(model,[r[w:w+1] for r in seq],[p[w:w+1] for p in pend])[0][1]
        if w==0:
            full=TT.full_from_local(local,2,0,np.zeros)
        else:
            for k,v in vars(local).items():
                if isinstance(v,np.ndarray): full[k][w:w+1]=v
        halves.append((w,pickle.loads(pickle.dumps(TT.rollout_scalars(local))),(local.t_kv,local.t_o)))
    merged,_=TT.merge_rollouts(full,halves,{},'cpu')
    mm,mt=plan(merged)
    with torch.no_grad():
        rb=RoomSourceBatch(model,mt,np.arange(len(mm['tch'])),4)
        mo,mf=part(model,merged,mm,rb.td,np.arange(len(mm['tch'])),model.tatt.room_table(rb.source,rb.td,True))
    check('two-actor merged rollout parity', torch.allclose(mo[0],out[0],atol=3e-5))
    # Snapshot and branch append must not mutate parent raw state (including a partial room).
    child=TP.TemporalState(model,2)
    child.copy_slot_([0,1],[1,0],st)
    parent_len=[len(x) for x in st.raw_rooms.frames]
    snap=pickle.dumps(st.raw_rooms)
    r=seq[-1][0:1].copy(); r['first']=0
    child.step_inputs(model,r,pend[-1][0:1],[1])
    check('CF state copy and append isolation', pickle.dumps(st.raw_rooms)==snap and len(child.raw_rooms.frames[1])==parent_len[0]+1)
    child.reset_slot_([1]); check('CF reset clears raw visit',len(child.raw_rooms.frames[1])==0)
    from test_lane_batch_smoke import FakeSampler
    from isaac_bridge.tok_lanes import LaneService
    fake=FakeSampler(2,32)
    lanes=TP.TemporalState(model,2)
    service=LaneService(fake,model,'cpu',len(model.heads),tstate=lanes,bank=st,gtok=True)
    # The CF branch begins with the parent's complete raw open visit, then leaves it.
    for w in range(2):
        row=seq[-1][w].copy(); row['events']=2; row['first']=0
        fake.request(w,row,pend[-1][w],100+w,tbegin=w+1)
    service.submit(); service.finish()
    check('teacher/CF LaneService consumes raw snapshot',
          fake.replies==[[0,1]] and np.isfinite(fake.lane_out).all() and pickle.dumps(st.raw_rooms)==snap)
    # Exact whole-graph vs cached VJP, over two PPO micro-parts, with sources before the chunk.
    mp,td=plan(roll); ids=np.arange(len(mp['tch'])); groups=np.array_split(ids,2)
    model.zero_grad(set_to_none=True)
    direct=encode_visits(model,td['raw_visits'],4)
    sources=torch.cat((direct,direct.new_zeros(1,32)))
    dt=dict(td,o=sources,r_src=torch.arange(len(sources)),raw_room_memory=True)
    table=model.tatt.room_table(sources,dt,True)
    loss=0
    for g in groups:
        oo,_=part(model,roll,mp,dt,g,table); loss=loss+oo[0].square().sum()+oo[1].square().sum()
    loss.backward()
    expected={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    rb=RoomSourceBatch(model,td,ids,4)
    for g in groups:
        oo,_=part(model,roll,mp,rb.td,g,model.tatt.room_table(rb.source,rb.td,True))
        (oo[0].square().sum()+oo[1].square().sum()).backward()
    rb.backward()
    diff=max(float((p.grad-expected[k]).abs().max()) for k,p in model.named_parameters() if k in expected)
    check('gradient cache equals full graph',diff < 2e-4,f'max_abs={diff:.8g}')
    check('writer and Spatial receive production learner gradients',model.room_writer.query.grad.abs().sum()>0 and model.blocks[0].qkv.weight.grad.abs().sum()>0)
    before=rb.source.detach().clone()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); opt.step()
    newer=RoomSourceBatch(model,td,ids,4)
    check('next optimizer step recomputes current writer',not torch.equal(before,newer.source))
    import eval_tok
    ev=TP.TemporalState(model,2)
    for t in range(5):
        a=eval_tok.policy_actions(model,seq[t],pend[t],[0,1],torch.device('cpu'),t_state=ev)
    check('evaluator raw writer dispatch',a.shape==(2,len(model.heads)) and ev.nroom[0]==1)
    if torch.cuda.is_available():
        cuda_checks(model,seq,pend)
    else:
        print('SKIP CUDA graph parity (no visible GPU)',flush=True)
    print('RESULT production room pipeline checks passed',flush=True)

def cuda_checks(cpu_model,seq,pend):
    model=copy.deepcopy(cpu_model).cuda().eval()
    for fold in (False,True):
        st=TP.TemporalState(model,2,kv_dtype=torch.float32)
        ref=TP.TemporalState(model,2,kv_dtype=torch.float32)
        graph=TP.GraphActor(model,2,entities=64,fold=fold,temporal=st)
        errors=[]
        for t,(r,p) in enumerate(zip(seq,pend)):
            keys=np.array([101,102],np.int64); times=np.array([t,t],np.int64)
            got=graph.act(r,p,widx=np.arange(2),ukeys=(keys,times))
            with torch.no_grad():
                td=ref.step_inputs(model,r,p,[0,1])
                u=torch.from_numpy(TP.crn_uniforms(keys,times,sum(model.heads))).cuda()
                want=model.act(TP.to_batch(r,p,'cuda'),u=u,temporal=td)
            check(f'graph actions fold={fold} t={t}',np.array_equal(got[0],want[0].cpu().numpy()))
            errors.append(float(np.max(abs(got[2]-want[2].cpu().numpy()))))
            check('graph raw metadata',all((v is not None)==bool(c) for v,c in zip(graph.t_out['room_visits'],graph.t_out['create'])))
        check(f'CUDA graph value parity fold={fold}',max(errors)<3e-4,f'max_abs={max(errors):.8g}')
        del graph

if __name__=='__main__': main()
