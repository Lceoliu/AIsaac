"""Real CUDA autocast, deferred source VJP and actor refresh parity."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy, TemporalState, to_batch, GraphActor
from isaac_bridge.rebuild_memory import RawHistory, SourceBatch

assert torch.cuda.is_available(), 'this gate requires a real CUDA device'
torch.set_num_threads(1)
torch.manual_seed(63)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
rng=np.random.default_rng(63)
seq=TS.make_seq(rng,nw=1,T=16,changes={(0,4),(0,8),(0,12)},resets=set(),exits=set())
pending=[M.synth_pending(rng,1,(9,5,2,2,2))[0] for _ in seq]
h=RawHistory.empty();requests=[]
for t,row in enumerate(seq):
    if t in (5,9,15):requests.append((h,row[0],pending[t]))
    h=h.append(row[0],pending[t],4)
for bf16 in (False,True):
    m=TokPolicy(**cfg).cuda().eval();ref=copy.deepcopy(m)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
        full=SourceBatch(ref,requests,micro=8,cache=False)
        o=full.forward([0,1,2],need_vg=True)
        (o[0].float().square().sum()+o[1].square().sum()+o[-1].square().sum()).backward()
        batch=SourceBatch(m,requests,micro=5)
        assert all(kv.dtype == torch.float32 for kv in batch.kvs)
        logits=[]
        for ix in ([0],[1,2]):
            part=batch.forward(ix,need_vg=True);logits.append(part[0].detach())
            (part[0].float().square().sum()+part[1].square().sum()+part[-1].square().sum()).backward()
        batch.backward()
    error=float((torch.cat(logits).float()-o[0].float()).abs().max())
    differences=[]; squared_error=0.; squared_reference=0.
    for (name,p),(name2,q) in zip(m.named_parameters(),ref.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            differences.append(float((p.grad-q.grad).abs().max()))
            squared_error+=float((p.grad.float()-q.grad.float()).square().sum())
            squared_reference+=float(q.grad.float().square().sum())
            torch.testing.assert_close(p.grad,q.grad,rtol=.12 if bf16 else 5e-4,
                                       atol=.02 if bf16 else 6e-5,msg=name)
    assert batch.u.grad[0].abs().sum()>0
    relative_l2=(squared_error/squared_reference)**.5
    assert relative_l2 < (.02 if bf16 else 1e-5),relative_l2
    print('CUDA VJP PASS bf16',bf16,'logits max_abs',error,'gradient max_abs',max(differences),
          'gradient relative_L2',relative_l2,flush=True)

    direct_model = copy.deepcopy(ref); direct_model.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        direct = SourceBatch(direct_model, requests, micro=8, cache=False)
        direct.u.retain_grad()
        for i, ix in enumerate(([0], [1, 2])):
            part = direct.forward(ix, need_vg=True)
            (part[0].float().square().sum()+part[1].square().sum()+part[-1].square().sum()).backward(retain_graph=i == 0)
        direct.backward()
    error_sq = reference_sq = 0.
    for (name, a), (_, b) in zip(direct_model.named_parameters(), ref.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.isfinite(a.grad).all(), name
            error_sq += float((a.grad.float()-b.grad.float()).square().sum())
            reference_sq += float(b.grad.float().square().sum())
    relative = (error_sq/reference_sq)**.5
    assert relative < (.02 if bf16 else 1e-5), relative
    assert direct.u.grad[0].abs().sum() > 0
    print('CUDA direct shared graph PASS bf16', bf16, 'gradient relative_L2', relative, flush=True)

m=TokPolicy(**cfg).cuda().eval()
state=TemporalState(m,2,'cuda',torch.bfloat16)
actor=GraphActor(m,2,bf16=True,temporal=state)
for t,row in enumerate(seq):
    if t==10:
        with torch.no_grad():m.reader.inp.weight.add_(.03)
        m.invalidate_source_cache()
    pd=np.asarray([pending[t]])
    assert actor.submit(row,pd,widx=np.array([0]),ukeys=(np.array([t]),np.array([0])))
    actions,logp,value=actor.result()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        current=SourceBatch(m,[(actor.t_out['raw_contexts'][0],row[0],pending[t])],micro=5,cache=False)
        out=current.forward([0],need_vg=True)
        lp=m.log_prob(out[0],m.gate(to_batch(row,pd,'cuda')),torch.from_numpy(actions).cuda())[0]
    np.testing.assert_allclose(logp,lp.float().cpu().numpy(),rtol=.015,atol=.015)
    np.testing.assert_allclose(value,out[1].float().cpu().numpy(),rtol=.015,atol=.015)
print('CUDA async actor / current-weight learner PASS including generation refresh',flush=True)


# Many readers share the same source: preserve the complete gradient reduction.
from torch.nn import functional as F
indices = torch.randint(0, 64, (512, 257), device='cuda')
values = torch.randn(64, 64, device='cuda').bfloat16().float()
upstream = torch.randn(512, 257, 64, device='cuda').bfloat16()
a = values.clone().requires_grad_(); b = values.clone().requires_grad_()
a[indices].bfloat16().backward(upstream)
F.embedding(indices, b).bfloat16().backward(upstream)
relative = float((a.grad-b.grad).norm()/a.grad.norm())
assert relative < 1e-5, relative
print('CUDA shared-source FP32 gradient accumulation PASS relative_L2',relative)
