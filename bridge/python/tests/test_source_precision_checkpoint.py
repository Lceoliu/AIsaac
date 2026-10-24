"""Numerical BF16 drift on a recorded production rollout and its exact weights."""
import argparse, gc, json, os, pickle, sys, time, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.rebuild_memory import SourceBatch, unpack_histories
from isaac_bridge.source_pipeline import SourcePipeline, source_parameter

p=argparse.ArgumentParser()
p.add_argument('--job',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--queries',type=int,default=8192)
p.add_argument('--compile-source',nargs='?',const=1,type=int,default=0,choices=(0,1,2))
p.add_argument('--flash-source',action='store_true')
p.add_argument('--compare-pointwise',action='store_true')
p.add_argument('--saved-8bit',type=int,default=0)
p.add_argument('--vjp-repeats',type=int,default=0)
p.add_argument('--learner',default='cuda:0');p.add_argument('--sources',default='cuda:1,cuda:2')
a=p.parse_args();torch.set_num_threads(2)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
job=torch.load(a.job,map_location='cpu',weights_only=False)
contexts=job['scal']['raw_contexts']
if isinstance(contexts,bytes):contexts=pickle.loads(contexts)
if isinstance(contexts,tuple):contexts=unpack_histories(contexts,ck['config']['memory_block'])
rows,pending=job['arrays']['rows'],job['arrays']['pending']
# Preserve complete prefixes; choose two different episodes, not an artificial
# shortened history. Their current records and all input bytes come from the job.
latest={}
for w,context in enumerate(contexts):
    for t,h in enumerate(context[:int(job['arrays']['length'][w])]):
        if h.uid not in latest or h.length>latest[h.uid][0].length:
            latest[h.uid]=(h,rows[w,t],pending[w,t])
chosen=sorted(latest.values(),key=lambda r:r[0].length,reverse=True)[:2]
req=[chosen[i%len(chosen)] for i in range(a.queries)]
prefixes=[]
for h,_,_ in chosen:
    r,_=h.arrays()
    room=r['map'][:,3].reshape(len(r),-1)
    prefixes.append(dict(length=h.length,room_changes=int(np.any(room[1:]!=room[:-1],axis=1).sum())))
del job,contexts,rows,pending,latest;gc.collect()
outputs={};grads={};times={};old={};memory={};packed={};vjp_seconds={};timing_jobs={}
variants=[('fp32',False),('bf16',True)]
if a.compare_pointwise:variants.append(('pointwise',True))
if a.saved_8bit:variants=[('bf16',True),('activation8',True)]
for label,amp in variants:
    if a.compare_pointwise:os.environ['ISAAC_SOURCE_POINTWISE_RECOMPUTE']='1' if label=='pointwise' else '0'
    allocated_before=sum(torch.cuda.memory_allocated(d) for d in a.sources.split(','))
    model=TokPolicy(**ck['config']).to(a.learner).eval();model.load_state_dict(ck['model'])
    pipe=SourcePipeline(model,a.sources.split(','),period=0,micro=8192,amp=amp,compile_source=amp and a.compile_source,flash_source=amp and a.flash_source,
                        saved_8bit=a.saved_8bit if label=='activation8' else 0)
    opt=torch.optim.Adam(model.main_parameters(),lr=1e-4,eps=1e-5)
    torch.cuda.synchronize();start=time.perf_counter()
    bank=SourceBatch(model,req,cache=False,pipeline=pipe)
    for device in a.sources.split(','):torch.cuda.synchronize(device)
    allocated=sum(torch.cuda.memory_allocated(d) for d in a.sources.split(','))
    memory[label]=dict(source_allocated_bytes=allocated-allocated_before,
                       source_allocated_before_bytes=allocated_before,source_allocated_total_bytes=allocated)
    packed[label]={key:sum(e.get(key,0) for e in pipe.events) for key in
                   ('activation_raw_bytes','activation_packed_bytes','activation_packs','activation_pack_hits')}
    early=pipe.chunks[0]['future'].result()['u'];early.retain_grad()
    out=bank.forward(np.arange(a.queries),need_vg=True)
    outputs[label]=torch.cat([out[0].flatten(),out[1].flatten(),out[-1].flatten()]).detach().cpu()
    (out[0].square().mean()+out[1].square().mean()+out[-1].square().mean()).backward();bank.backward()
    if a.vjp_repeats:
        # Benchmark the very same retained graph/cotangents, without any new
        # forward or optimizer step. The subsequent finish still applies credit
        # exactly once; diagnostic VJPs do not update master parameters.
        pipe._flush_collected()
        packets=[(c,c['future'].result()['grads']) for c in pipe.chunks+pipe.writes
                 if c is not None and c['future'].result()['grads']]
        timing_jobs[label]=(pipe,packets)
    pipe.fast_step(opt,1e9);pipe.finish(opt,1e9);torch.cuda.synchronize()
    times[label]=time.perf_counter()-start
    grads[label]={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()
                  if source_parameter(n) and v.grad is not None}
    old[label]=float(early.grad[0].abs().sum())
    assert old[label]>0
    assert all(v.dtype==torch.float32 for v in model.parameters())
    assert all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 for s in opt.state.values())
    if not a.vjp_repeats:pipe.close()
    del pipe,bank,early,out,opt,model;gc.collect();torch.cuda.empty_cache()
if a.vjp_repeats:
    # Both graphs are warm and resident before paired, alternating AB/BA
    # measurements. Sequential all-A then all-B was sensitive to clock drift.
    samples={label:[] for label in timing_jobs}
    for repeat in range(a.vjp_repeats+5):
        order=list(timing_jobs)
        if repeat%2:order.reverse()
        for label in order:
            pipe,packets=timing_jobs[label]
            for device in a.sources.split(','):torch.cuda.synchronize(device)
            tick=time.perf_counter()
            futures=[w.submit(pipe._backward,i,packets,True) for i,w in enumerate(pipe.workers)]
            results=[f.result() for f in futures]
            for device in a.sources.split(','):torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-tick
            if repeat>=5:samples[label].append(elapsed)
            del results,futures
    vjp_seconds={label:dict(samples=values,median=statistics.median(values)) for label,values in samples.items()}
    for pipe,_ in timing_jobs.values():pipe.close()
    timing_jobs.clear();del pipe,packets;gc.collect();torch.cuda.empty_cache()
actual_label,reference_label=('activation8','bf16') if a.saved_8bit else ('bf16','fp32')
ga=torch.cat([grads[actual_label][n].flatten() for n in grads[reference_label]]).double()
gb=torch.cat([grads[reference_label][n].flatten() for n in grads[reference_label]]).double()
d=outputs[actual_label]-outputs[reference_label]
metrics=dict(output_max_abs=float(d.abs().max()),output_relative_l2=float(d.norm()/outputs[reference_label].norm()),
             source_grad_relative_l2=float((ga-gb).norm()/gb.norm()),
             source_grad_cosine=float(torch.nn.functional.cosine_similarity(ga,gb,dim=0)))
error_by_parameter=sorted([(n,float((grads[actual_label][n].double()-v.double()).norm()),float(v.double().norm()))
                           for n,v in grads[reference_label].items()],key=lambda x:x[1],reverse=True)[:6]
print(json.dumps(dict(metrics=metrics,config=ck['config'],prefixes=prefixes,queries=a.queries,
                     old_room_gradients=old,cold_seconds=times,memory=memory,packed=packed,
                     saved_8bit_block=a.saved_8bit,error_by_parameter=error_by_parameter,
                     retained_vjp_seconds=vjp_seconds,encoder_replay=0),indent=2),flush=True)
assert metrics['output_relative_l2']<.03,metrics
assert metrics['source_grad_relative_l2']<.10,metrics
if a.saved_8bit:
    assert torch.equal(outputs['bf16'],outputs['activation8']),metrics
    assert metrics['source_grad_relative_l2']<.02,metrics
    assert metrics['source_grad_cosine']>.999,metrics
    assert memory['activation8']['source_allocated_bytes']<memory['bf16']['source_allocated_bytes'],memory
if a.compare_pointwise:
    ref=torch.cat([grads['bf16'][n].flatten() for n in grads['bf16']]).double()
    actual=torch.cat([grads['pointwise'][n].flatten() for n in grads['bf16']]).double()
    delta=outputs['pointwise']-outputs['bf16']
    extra=dict(output_max_abs=float(delta.abs().max()),output_relative_l2=float(delta.norm()/outputs['bf16'].norm()),
        source_grad_relative_l2=float((actual-ref).norm()/ref.norm()),
        source_grad_cosine=float(torch.nn.functional.cosine_similarity(actual,ref,dim=0)))
    print('POINTWISE_INCREMENTAL_DRIFT '+json.dumps(extra),flush=True)
    assert extra['output_relative_l2']<1e-4,extra
    assert extra['source_grad_relative_l2']<.01,extra
print('PASS production checkpoint source BF16 drift; FP32 Reader/master/Adam; complete recorded prefixes')
