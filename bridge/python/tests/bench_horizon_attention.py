"""Time grouped Reader attention: native SDPA autograd vs HorizonAttention.

Production-like shapes (groups x queries per group x bank entries, 4 heads of
64). Reports forward equality and median forward+backward milliseconds.
usage: python tests/bench_horizon_attention.py [--device cuda:0] [--groups 96] [--count 128] [--bank 1600]
"""
import argparse, json, time
import torch
from torch.nn import functional as F
from isaac_bridge.rebuild_memory import HorizonAttention

p = argparse.ArgumentParser()
p.add_argument('--device', default='cpu')
p.add_argument('--groups', type=int, default=96)
p.add_argument('--count', type=int, default=128)
p.add_argument('--bank', type=int, default=1600)
p.add_argument('--sources', type=int, default=60000)
p.add_argument('--repeats', type=int, default=10)
a = p.parse_args()
torch.manual_seed(5)
dev, heads, dh, D = torch.device(a.device), 4, 64, 256
src = torch.randn(a.sources, D, device=dev, requires_grad=True)
layer = torch.nn.Linear(D, 2 * D).to(dev)
banks = torch.randint(0, a.sources, (a.groups, a.bank), device=dev)
valid = torch.rand(a.groups, a.count, a.bank, device=dev) < .7
valid[:, :, 0] = True
grad_valid = valid & (torch.rand(a.groups, a.count, a.bank, device=dev) < .8)
q0 = torch.randn(a.groups, heads, a.count, dh, device=dev)


def native():
    q = q0.clone().requires_grad_()
    kv = layer(src)
    bank = F.embedding(banks, kv).reshape(a.groups, a.bank, 2, heads, dh)
    k, v = bank[:, :, 0].transpose(1, 2), bank[:, :, 1].transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None])
    out.square().sum().backward()
    return out


def horizon():
    q = q0.clone().requires_grad_()
    kv = layer(src.detach())
    out = HorizonAttention.apply(q, kv, src, layer.weight, banks, valid, grad_valid, heads)
    out.square().sum().backward()
    return out


def timed(fn):
    samples = []
    for i in range(a.repeats + 2):
        if dev.type == 'cuda': torch.cuda.synchronize(dev)
        start = time.perf_counter(); fn()
        if dev.type == 'cuda': torch.cuda.synchronize(dev)
        if i >= 2: samples.append(1000 * (time.perf_counter() - start))
    return sorted(samples)[len(samples) // 2]


x, y = native(), horizon()
result = dict(device=a.device, groups=a.groups, count=a.count, bank=a.bank, forward_equal=bool(torch.equal(x, y)))
if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
result['native_ms'] = timed(native)
if dev.type == 'cuda':
    result['native_peak_gib'] = torch.cuda.max_memory_allocated(dev) / 2**30; torch.cuda.reset_peak_memory_stats(dev)
result['horizon_ms'] = timed(horizon)
if dev.type == 'cuda': result['horizon_peak_gib'] = torch.cuda.max_memory_allocated(dev) / 2**30
print(json.dumps(result))
assert result['forward_equal']
