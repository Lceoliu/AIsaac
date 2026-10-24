"""Isolate PyTorch 2.7 compilation under saved-tensor hooks."""
import contextlib
import json
import torch

torch.set_num_threads(2)
for mode in ('none','detach','leaf'):
    torch._dynamo.reset()
    model=torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,128)).cuda()
    used=set()
    accumulated=set()
    for n,p in model.named_parameters():
        p.register_hook(lambda g,n=n:used.add(n))
        p.register_post_accumulate_grad_hook(lambda p,n=n:accumulated.add(n))
    x=torch.randn(512,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    fn=torch.compile(model,dynamic=True)
    def pack(t):return t if mode=='leaf' and t.is_leaf else t.detach()
    hooks=contextlib.nullcontext() if mode=='none' else torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t)
    with hooks,torch.autocast('cuda',dtype=torch.bfloat16):out=fn(x)
    out.float().square().mean().backward()
    print(json.dumps(dict(mode=mode,hooks=sorted(used),post_accumulate=sorted(accumulated),gradients={n:None if p.grad is None else float(p.grad.abs().sum())
          for n,p in model.named_parameters()})),flush=True)
    assert accumulated == {n for n,p in model.named_parameters() if p.grad is not None}
