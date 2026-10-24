"""Repeated VJPs, saved bytes and GEMM/attention counts for GELU-only remat."""
import argparse, copy, json, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import test_memory_smoke
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch import config as aot_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from functorch.compile import make_boxed_func
from isaac_bridge.tok_policy import Block
from isaac_bridge.source_rematerialize import source_ff

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu')
p.add_argument('--inductor',action='store_true');p.add_argument('--components',action='store_true');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(141);aot_config.donated_buffer=False
cuda=a.device.startswith('cuda');width=192 if cuda else 32
class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([Block(width,4) for _ in range(4 if cuda else 2)])
        self.remat_components=False
        if a.components:
            self.mlp=torch.nn.Sequential(torch.nn.Linear(width,width),torch.nn.GELU(),torch.nn.Linear(width,width))
            self.cnn=torch.nn.Sequential(torch.nn.Conv1d(width,32,3,padding=1),torch.nn.GELU(),torch.nn.Conv1d(32,width,3,padding=1))
    def forward(self,x):
        if a.components:
            if self.remat_components:
                x=source_ff(self.mlp,x)+source_ff(self.cnn,x.transpose(1,2)).transpose(1,2)
            else:x=self.mlp(x)+self.cnn(x.transpose(1,2)).transpose(1,2)
        for block in self.blocks:x=block(x,None)
        return x

original=Stack().to(a.device);changed=copy.deepcopy(original)
changed.remat_components=True
for block in changed.blocks:block.source_pointwise_recompute=True
x=torch.randn(128 if cuda else 2,64 if cuda else 8,width,device=a.device,requires_grad=True)
seeds=[torch.randn_like(x) for _ in range(3)]
records=[];answers=[]
for model in [original,changed]:
    graphs={}
    def fw(gm,inputs):
        graphs['forward']=Counter(str(n.target) for n in gm.graph.nodes if n.op=='call_function')
        return make_boxed_func(gm.forward)
    def bw(gm,inputs):
        graphs['backward']=Counter(str(n.target) for n in gm.graph.nodes if n.op=='call_function')
        return make_boxed_func(gm.forward)
    backend='inductor' if a.inductor else aot_autograd(fw_compiler=fw,bw_compiler=bw,partition_fn=min_cut_rematerialization_partition)
    options={'emulate_precision_casts':True,'layout_optimization':False} if a.inductor and a.components else None
    compiled=torch.compile(model,backend=backend,fullgraph=True,dynamic=True,options=options)
    # Compile outside the measured forward/backward and saved-tensor hooks.
    with torch.autocast(torch.device(a.device).type,dtype=torch.bfloat16,enabled=cuda):y=compiled(x)
    y.backward(seeds[0]);del y
    model.zero_grad(set_to_none=True);x.grad=None
    saved={};excluded={t.untyped_storage().data_ptr() for t in [x,*model.parameters()]}
    def pack(t):
        storage=t.untyped_storage();ptr=storage.data_ptr()
        if ptr not in excluded:saved[ptr]=storage.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        with torch.autocast(torch.device(a.device).type,dtype=torch.bfloat16,enabled=cuda):y=compiled(x)
    gradients=[];counts=[];times=[]
    for seed in seeds:
        model.zero_grad(set_to_none=True);x.grad=None
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            y.backward(seed,retain_graph=True)
            if cuda:torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        counts.append({e.key:e.count for e in prof.key_averages()
                       if e.key.startswith('aten::') and (e.key in ('aten::mm','aten::addmm','aten::bmm') or 'attention' in e.key or 'convolution' in e.key)})
        gradients.append([x.grad.detach().clone(),*[p.grad.detach().clone() for p in model.parameters()]])
    answers.append((y.detach(),gradients))
    records.append(dict(saved_bytes=sum(saved.values()),graphs=graphs,backward_ops=counts,profiled_seconds=times))
    del y,compiled
torch.testing.assert_close(answers[0][0],answers[1][0],rtol=2e-3 if cuda else 1e-5,atol=2e-3 if cuda else 1e-6)
worst=0.;error2=norm2=0.
for ref,got in zip(answers[0][1],answers[1][1]):
    for expected,actual in zip(ref,got):
        delta=(expected-actual).float();worst=max(worst,float(delta.abs().max()))
        error2+=float(delta.square().sum());norm2+=float(expected.float().square().sum())
        if not a.components:
            torch.testing.assert_close(actual,expected,rtol=3e-3 if cuda else 1e-5,atol=.03 if cuda else 1e-5)
relative=(error2/norm2)**.5
print('POINTWISE_GRADIENT_DRIFT '+json.dumps(dict(max_abs=worst,relative_l2=relative)),flush=True)
assert relative<.01, 'compiled component BF16 gradient drift exceeds 1%'
assert records[1]['saved_bytes']<records[0]['saved_bytes'],records
assert records[0]['backward_ops']==records[1]['backward_ops'],records
if not a.inductor:
    for stage in ['forward','backward']:
        heavy=lambda r:{op:n for op,n in r['graphs'][stage].items() if any(s in op for s in ('.mm.','.addmm.','.bmm.','attention','convolution'))}
        assert heavy(records[0])==heavy(records[1]),records
    assert records[1]['graphs']['backward']['aten.gelu.default']>records[0]['graphs']['backward']['aten.gelu.default'],records
print(json.dumps(dict(result='PASS',device=a.device,backend='inductor' if a.inductor else 'aot',
    repeated_vjps=3,gradient_max_abs=worst,gradient_relative_l2=(error2/norm2)**.5,
    saved_reduction=1-records[1]['saved_bytes']/records[0]['saved_bytes'],records=records)))
