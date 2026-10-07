"""B10, learner side: how many rows per second the GPU trains for the stand-in network of abplus_bench_fork_sampler.py.

Random rows of the sampler's record (FRAME) on the GPU, one policy-gradient-shaped loss (two categorical heads and a
value head), Adam; forward + backward + step timed for --steps batches after a warm-up. Reported for strict FP32 (the
project's setting so far: TF32 off) and for TF32 + bfloat16 autocast. The numbers bound how many decisions per second a
learner can consume per pass over the data; they say nothing about learning.

usage: python abplus_bench_standin_train.py [--width 256 --layers 4 --heads 8 --batch 1024]
"""
import argparse
import json
import time

import torch

from abplus_bench_fork_sampler import make_model
from isaac_bridge.fork_sampler import ENTITY_CAP, ENTITY_F, GRID, PLAYER_F


def run(args, fast):
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast
    device = torch.device('cuda')
    model, params = make_model(args.width, args.layers, args.heads, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    b = args.batch
    g = torch.Generator(device=device).manual_seed(0)
    player = torch.randn((b, PLAYER_F), device=device, generator=g)
    entities = torch.randn((b, ENTITY_CAP, ENTITY_F), device=device, generator=g)
    count = torch.randint(1, args.entities + 1, (b,), device=device, generator=g).int()
    grid = torch.randint(0, 2, (b, *GRID), device=device, generator=g).float()
    move = torch.randint(0, 9, (b,), device=device, generator=g)
    shoot = torch.randint(0, 5, (b,), device=device, generator=g)
    target = torch.randn((b,), device=device, generator=g)

    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=fast):
            lm, ls, v = model(player, entities, count, grid)
            loss = torch.nn.functional.cross_entropy(lm.float(), move) + \
                torch.nn.functional.cross_entropy(ls.float(), shoot) + ((v.float()[:, 0] - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return dict(precision='TF32 + bfloat16 autocast' if fast else 'strict FP32', parameters=params,
                rows_per_s=b * args.steps / dt, ms_per_batch=1000 * dt / args.steps,
                peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--entities', type=int, default=ENTITY_CAP, help='entity tokens per row: 1..this many are real')
    p.add_argument('--steps', type=int, default=50)
    args = p.parse_args()
    out = dict(gpu=torch.cuda.get_device_name(0), batch=args.batch, width=args.width, layers=args.layers,
               tokens=ENTITY_CAP + 2, runs=[run(args, False), run(args, True)])
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
