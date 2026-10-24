"""CUDA storage/stride/lifetime and repeated-VJP gate for opt-in activation8."""
import json
import argparse
import sys
import weakref
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from isaac_bridge.saved_activation import SavedActivation8bit

torch.set_num_threads(2)
torch.manual_seed(83)
p=argparse.ArgumentParser();p.add_argument('--blocksize',type=int,default=64);a=p.parse_args()
model = torch.nn.Linear(128, 128, bias=False, device='cuda', dtype=torch.bfloat16)
store = SavedActivation8bit(model, a.blocksize)
x = torch.randn(512, 3, 4, 2, 16, device='cuda', dtype=torch.bfloat16)
original_storage = weakref.ref(x.untyped_storage())
with store:
    q = store.pack(x[:, 0])
    k = store.pack(x[:, 1])
    assert q[0] is k[0]
    weight = store.pack(model.weight)
    assert isinstance(weight, torch.Tensor) and weight.data_ptr() == model.weight.data_ptr()
    attention = torch.randn(512, 4, 32, device='cuda', dtype=torch.bfloat16)
    assert store.pack(attention).data_ptr() == attention.data_ptr()
    mask = torch.full((128, 128), float('-inf'), device='cuda')
    assert torch.equal(store.unpack(store.pack(mask)), mask)
    restored = store.unpack(k)
    assert restored.stride() == x[:, 1].stride()
    assert restored.storage_offset() == x[:, 1].storage_offset()
    error = float((restored.float()-x[:, 1].float()).norm()/x[:, 1].float().norm())
    assert error < .02, error
del x
assert original_storage() is None, 'packed record retained original activation storage'
assert store.packed_bytes < .54 * store.raw_bytes, store.stats()

source = torch.randn(512, 128, device='cuda', dtype=torch.bfloat16)
grads, outputs = [], []
for compressed in (False, True):
    inp = source.clone().requires_grad_()
    with SavedActivation8bit(model, a.blocksize) if compressed else torch.autograd.graph.saved_tensors_hooks(lambda t:t, lambda t:t):
        out = torch.nn.functional.gelu(model(inp))
        loss = out.float().square().mean()
    outputs.append(out.detach())
    grad = torch.autograd.grad(loss, (inp, model.weight), retain_graph=True)
    again = torch.autograd.grad(loss, (inp, model.weight), retain_graph=True)
    assert all(torch.equal(a,b) for a,b in zip(grad,again)), 'repeated VJP changed'
    grads.append(torch.cat([g.float().flatten() for g in grad]))
assert torch.equal(*outputs), 'saved hooks changed forward'
relative = float((grads[1]-grads[0]).norm()/grads[0].norm())
assert relative < .02, relative
print(json.dumps(dict(result='PASS',format='blockwise-linear-u8',blocksize=a.blocksize,storage=store.stats(),activation_relative_l2=error,
                      grad_relative_l2=relative,forward_equal=True,repeated_vjp_equal=True,
                      original_storage_released=True)), flush=True)
