"""Real Linux engine / block actor / CF / injury protocol smoke, without CUDA.

This is not a throughput or production-training acceptance run. Search horizons
and quotas are deliberately shortened/increased to exercise the protocol quickly.
The exported jobs can be replayed through train_tok's real learner on a healthy GPU.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from goexplore_abplus import default_bridge_lua,default_preload,load_spec
from isaac_bridge.tok_sampler import TokSampler,TokSamplerConfig
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.tok_priv import PRIV
from isaac_bridge.block_training import BlockActor,pack_snapshots
from isaac_bridge.block_lanes import BlockLaneService
from isaac_bridge.block_memory import warm_start_spatial
from isaac_bridge.tok_teacher2 import V2_META
from train_tok import Rollout,Episodes,store,FLOOR_REWARD,cpu_ticks
from abplus_bench_learner import save_job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups-file',required=True)
    p.add_argument('--init',default='')
    p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=480)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--port',type=int,default=38600)
    p.add_argument('--name',default='blockcpusmoke')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(4242)
    model=TokPolicy(width=192,layers=4,heads=4,items=True,item_tokens=True,global_token=True,
                    charge=True,ent_ext=True,pchar=True,block_memory=True).to(a.device).eval()
    if a.init:
        loaded=warm_start_spatial(model,torch.load(a.init,map_location='cpu',weights_only=False)['model'])
        print('SPATIAL_WARM_INIT',len(loaded),flush=True)
    opt=torch.optim.Adam(model.main_parameters(),lr=3e-4)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=model.config,
                    update=0,decisions=0,seconds=0),out/'initial.pt')
    spec=load_spec(argparse.Namespace(groups_file=a.groups_file,group='normal',tasks='',seconds=0.))
    stub=Path(default_preload()).parent/'stub_render_g.txt'
    cfg=TokSamplerConfig(workers=a.workers,specs=[spec],assign=[0]*a.workers,items=True,priv=True,
        mode='floor',start_hp=(6,6),floor_seconds=60,floor_stall_seconds=30,seed=4242,name=a.name,port=a.port,
        bridge_lua=default_bridge_lua(),preload=default_preload(),stub_list=str(stub) if stub.is_file() else '',
        teacher=True,teacher_thread=1,teacher_fork=1,teacher_pair=True,teacher_share=.4,
        teacher_death_depths=(2,4,8),cf=True,cf_points=3600,cf_random=1800,cf_branches=4,
        cf_seconds=10,cf_target_seconds=3,cf_vglobal=True,block_memory=True,temporal_dim=256)
    (out/'config.json').write_text(json.dumps(dict(model=model.config,sampler=asdict(cfg)),default=str,indent=2))
    rw=dict(room=0.,explore=0.,boss=0.,exit=0.,heal=0.,resource=0.)
    rw.update({k:float(v) for k,v in (x.split('=') for x in FLOOR_REWARD.split(','))})
    sampler=TokSampler(cfg)
    actor=BlockActor(model,a.workers)
    lanes=BlockLaneService(sampler,model,a.device,main_actor=actor)
    book=Episodes(a.workers,rw,True,items=True)
    roll=Rollout(a.workers,2048,5,value_g=True,close=True,priv=PRIV,block_steps=128)
    pending=np.zeros((a.workers,5),np.int64)
    counts=dict(main_rows=0,terminal_rows=0,hurt_rows=0,injury_records=0,cf_records=0,jobs=0)
    teachers=[];cfs=[]
    t0=last=time.perf_counter()
    try:
        while time.perf_counter()-t0<a.seconds and sampler.live:
            lanes.tick()
            idx,slots=sampler.poll_ready(idle=lanes.idle(.01))
            if len(idx):
                rows=sampler.rows[slots]
                lanes.observe_rows(idx,rows)
                counts['hurt_rows']+=int((rows['hurt']>0).sum())
                go=book.arrive(roll,rows,idx,pending)
                counts['terminal_rows']+=int((~go).sum())
                if not go.all():sampler.reply(idx[~go])
                if go.any():
                    wi=idx[go];sub=rows[go];under=pending[wi].copy()
                    lanes.capture_main(wi,sub,under)
                    actor.step(sub,under,wi)
                    h=actor.host_output;actions=h[:,:5].astype(np.int64)
                    lanes.record_main(wi,sub,actor.last_contexts)
                    sampler.actions[wi,:5]=actions
                    sampler.main_value[wi]=h[:,6];sampler.main_vg[wi]=h[:,7]
                    priv=sampler.priv[slots[go]].copy()
                    sampler.reply(wi)
                    store(roll,wi,sub,under,actions,h[:,5],h[:,6],0,vg=h[:,7],priv=priv,
                          block_context=actor.last_contexts)
                    pending[wi]=actions
                    counts['main_rows']+=len(wi)
            tr=sampler.teacher_records(with_block=True)
            vr=sampler.v2_records(with_block=True)
            if len(tr[0]):teachers.append(tr);counts['injury_records']+=len(tr[0])
            if len(vr[0]):cfs.append(vr);counts['cf_records']+=len(vr[0])
            if roll.usable()>=512:
                if counts['jobs']<4 or teachers or cfs:
                    records=[]
                    for batches,width in ((teachers,16),(cfs,V2_META)):
                        rr=np.concatenate([x[0] for x in batches]) if batches else roll.rows[0,:0]
                        mm=np.concatenate([x[1] for x in batches]) if batches else np.zeros((0,width),np.float32)
                        cc=[s for x in batches for s in x[2]]
                        records.append((rr,mm,pack_snapshots(cc)))
                    job=dict(roll=roll,teach=records[0],v2=records[1],finished=list(book.finished),
                        collect_s=time.perf_counter()-last,cycle_s=time.perf_counter()-last,t_start=t0,
                        cpu0=cpu_ticks(),wait_learner_s=0.,worker=sampler.totals(),
                        prof=dict.fromkeys(('latency','calls','rows','eager','act_busy','calls_busy',
                                           'poll','book','act','reply','store','polls'),0))
                    save_job(out/f'job{counts["jobs"]}.pkl',job,sampler.stats,model,opt,{})
                    counts['jobs']+=1
                    teachers.clear();cfs.clear()
                nxt=Rollout(a.workers,2048,5,value_g=True,close=True,priv=PRIV,block_steps=128)
                nxt.carry(roll);roll=nxt;book.finished=[]
                last=time.perf_counter()
                print('PROGRESS',json.dumps(counts),flush=True)
        summary=dict(counts=counts,seconds=time.perf_counter()-t0,totals=sampler.totals(),
                     lane_stat=lanes.stat,contexts_held=len(lanes.contexts.values),
                     injury_sources_held=len(lanes.injury_sources),device=a.device,training_updates=0)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('REAL_GAME_SUMMARY',json.dumps(summary),flush=True)
        if counts['main_rows']<512 or counts['jobs']==0:
            raise RuntimeError('real-game capture failed: insufficient live rows/jobs; worker exit is not success')
    finally:
        sampler.close()


if __name__=='__main__':main()
