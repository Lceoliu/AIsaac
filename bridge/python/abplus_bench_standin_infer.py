"""B10, server side: what one batched inference call of the stand-in network costs, by batch size and by part.

Random rows of the sampler's record (FRAME, as the server gets them: one fancy-indexed copy of the shared array) go
through the same steps as abplus_bench_fork_sampler.py's model policy: host-to-GPU copies, the forward pass, sampling,
the copy of the rows into the GPU ring buffer and the actions' way back to the host. Each part is timed with the GPU
synchronised after it, so the parts add up to more than the unsynchronised call, which is timed as well.

usage: python abplus_bench_standin_infer.py [--width 256 --layers 4 --heads 8 --entities 12]
"""
import argparse
import json
import time

import numpy as np
import torch

from abplus_bench_fork_sampler import make_model
from isaac_bridge.fork_sampler import ENTITY_CAP, ENTITY_F, FRAME, GRID, PLAYER_F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--entities', type=int, default=12, help='the largest entity count in a batch')
    p.add_argument('--rounds', type=int, default=200)
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--compile', default='', help="torch.compile mode for the network ('' = eager), e.g. reduce-overhead")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    device = torch.device('cuda')
    model, params = make_model(args.width, args.layers, args.heads, device)
    if args.compile:
        model = torch.compile(model, mode=args.compile)
    rng = np.random.default_rng(0)
    shared = np.zeros((64,), FRAME)
    shared['player'] = rng.normal(size=shared['player'].shape)
    shared['entities'] = rng.normal(size=shared['entities'].shape)
    shared['n_entities'] = rng.integers(1, args.entities + 1, size=64)
    store = dict(player=torch.zeros((32768, PLAYER_F), device=device),
                 entities=torch.zeros((32768, ENTITY_CAP, ENTITY_F), device=device),
                 grid=torch.zeros((32768, *GRID), dtype=torch.uint8, device=device))
    out = dict(parameters=params, gpu=torch.cuda.get_device_name(0), tf32=args.tf32, compile=args.compile, batches={})

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    with torch.inference_mode():
        for b in (4, 8, 16, 32, 64):
            idx = list(range(b))
            parts = dict(index=0.0, to_gpu=0.0, forward=0.0, sample=0.0, store=0.0, to_host=0.0)
            whole = 0.0
            for r in range(args.rounds + 20):
                t0 = sync()
                batch = shared[idx]
                t1 = time.perf_counter()
                player = torch.from_numpy(batch['player']).to(device)
                entities = torch.from_numpy(batch['entities']).to(device)
                grid = torch.from_numpy(batch['grid']).to(device)
                count = torch.from_numpy(batch['n_entities']).to(device)
                t2 = sync()
                used = min(ENTITY_CAP, max(8, -(-int(batch['n_entities'].max()) // 8) * 8))
                move, shoot, _ = model(player, entities[:, :used], count, grid.float())
                t3 = sync()
                act = torch.stack([torch.multinomial(torch.softmax(move, -1), 1)[:, 0],
                                   torch.multinomial(torch.softmax(shoot, -1), 1)[:, 0]], 1)
                t4 = sync()
                at = torch.arange(b, device=device)
                store['player'][at], store['entities'][at], store['grid'][at] = player, entities, grid
                t5 = sync()
                act.cpu().numpy()
                t6 = sync()
                # the same call without the synchronisation points
                batch = shared[idx]
                player = torch.from_numpy(batch['player']).to(device)
                entities = torch.from_numpy(batch['entities']).to(device)
                grid = torch.from_numpy(batch['grid']).to(device)
                count = torch.from_numpy(batch['n_entities']).to(device)
                move, shoot, _ = model(player, entities[:, :used], count, grid.float())
                act = torch.stack([torch.multinomial(torch.softmax(move, -1), 1)[:, 0],
                                   torch.multinomial(torch.softmax(shoot, -1), 1)[:, 0]], 1)
                store['player'][at], store['entities'][at], store['grid'][at] = player, entities, grid
                act.cpu().numpy()
                t7 = time.perf_counter()
                if r >= 20:
                    for k, v in zip(parts, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
                        parts[k] += v
                    whole += t7 - t6
            n = args.rounds
            out['batches'][b] = dict(ms_per_call=1000 * whole / n, rows_per_s=b * n / whole,
                                     parts_ms={k: round(1000 * v / n, 3) for k, v in parts.items()})
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
