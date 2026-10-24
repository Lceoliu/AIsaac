"""Release duplicate exported values, retain their native repeated-VJP edges."""
import argparse,copy,gc,json,weakref
import torch

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(97)
model=torch.nn.Sequential(torch.nn.Linear(192,768),torch.nn.GELU(),torch.nn.Linear(768,192)).to(a.device)
x=torch.randn(8192,192,device=a.device)
records=[];gradients=[];exports=[]
for edge_only in (False,True):
    current=copy.deepcopy(model);calls=[]
    handle=current.register_forward_hook(lambda *args:calls.append(1))
    y=current(x)+x
    u=y.index_select(0,torch.arange(8,device=a.device)).sum(-1)
    exported=y.detach().clone();exports.append(exported)
    y_ref=weakref.ref(y);size=y.numel()*y.element_size()
    before=torch.cuda.memory_allocated(a.device) if x.is_cuda else 0
    output=torch.autograd.graph.get_gradient_edge(y) if edge_only else y
    del y;gc.collect()
    released=before-torch.cuda.memory_allocated(a.device) if x.is_cuda else 0
    if edge_only:
        assert y_ref() is None
        if x.is_cuda:assert released>=size,(released,size)
    accumulated=[]
    for i in range(3):
        current.zero_grad(set_to_none=True)
        torch.autograd.backward([output,u],[torch.full_like(exported,(i+1)*.001),torch.ones_like(u)],retain_graph=True)
        accumulated.append(torch.cat([v.grad.flatten() for v in current.parameters()]).clone())
    gradients.append(accumulated)
    assert len(calls)==1
    records.append(dict(edge_only=edge_only,released_bytes=released,output_bytes=size,forwards=len(calls)))
    handle.remove();del output,u,current;gc.collect()
assert torch.equal(exports[0],exports[1])
error=max(float((x-y).abs().max()) for x,y in zip(*gradients));assert error==0,error
print(json.dumps(dict(result='PASS',device=a.device,repeated_vjps=3,gradient_max_abs=error,records=records)))
