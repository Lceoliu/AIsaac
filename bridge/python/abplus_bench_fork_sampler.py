"""B10: throughput of the fork sampler (isaac_bridge/fork_sampler.py) on one host.

  --policy random   every worker draws its own actions: what the game, the bridge and the per-decision Python cost;
  --policy uniform  the server answers every ready row with a random action: + the worker <-> server hand-off;
  --policy model    the server runs a stand-in network on the GPU on each batch of ready rows and stores the rows in a
                    GPU ring buffer, as a learner would: + inference. The network (entity, player and terrain tokens
                    through a small Transformer, --width/--layers) only stands in for the policy that is still to be
                    designed; its actions are sampled from its (untrained) output.
After --warm seconds (instances start, first states are built) the counters are read every --every seconds for
--seconds; the summary gives logic frames per second, the multiple of real time (30 frames = 1 game second), where a
worker's time went, the batch sizes and the memory of all game processes.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_fork_sampler.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --workers 32 --policy model --seconds 60 --out <dir>
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.fork_sampler import ACTION_F, ENTITY_CAP, ENTITY_F, GRID, PLAYER_F, ForkSampler, SamplerConfig


def game_memory_mib():
    """(processes, summed Pss MiB, summed Rss MiB) of every isaac.x64 process."""
    n = pss = rss = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            if open(f'/proc/{pid}/comm').read().strip() != 'isaac.x64':
                continue
            rows = dict(line.split(':', 1) for line in open(f'/proc/{pid}/smaps_rollup') if ':' in line)
            n += 1
            pss += int(rows['Pss'].split()[0])
            rss += int(rows['Rss'].split()[0])
        except (OSError, KeyError, ValueError):
            pass
    return n, pss / 1024, rss / 1024


def make_model(width, layers, heads, device):
    import torch
    from torch import nn

    class StandIn(nn.Module):
        def __init__(self):
            super().__init__()
            self.player = nn.Sequential(nn.Linear(PLAYER_F, width), nn.GELU(), nn.Linear(width, width))
            self.entity = nn.Sequential(nn.Linear(ENTITY_F, width), nn.GELU(), nn.Linear(width, width))
            self.grid = nn.Sequential(nn.Conv2d(GRID[0], 32, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                      nn.Linear(64 * ((GRID[1] + 3) // 4) * ((GRID[2] + 3) // 4), width))
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.move, self.shoot, self.value = nn.Linear(width, 9), nn.Linear(width, 5), nn.Linear(width, 1)

        def forward(self, player, entities, count, grid):
            tokens = torch.cat([self.player(player)[:, None], self.grid(grid)[:, None], self.entity(entities)], 1)
            pad = torch.arange(entities.shape[1], device=count.device)[None] >= count[:, None]
            pad = torch.cat([torch.zeros((len(count), 2), dtype=torch.bool, device=count.device), pad], 1)
            h = self.norm(self.encoder(tokens, src_key_padding_mask=pad)[:, 0])
            return self.move(h), self.shoot(h), self.value(h)

    model = StandIn().to(device).eval()
    return model, sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--policy', default='random', choices=('random', 'uniform', 'model'))
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--episodes-per-state', type=int, default=4)
    p.add_argument('--full-bridge', action='store_true', help='episode clones keep the full bridge (no lean mode)')
    p.add_argument('--stub-list', default='', help="ABP_STUB_LIST for the instances (default: the exact mode's own)")
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--store', type=int, default=32768, help='rows of the GPU ring buffer (model policy)')
    p.add_argument('--nice', type=int, default=10, help='niceness of the workers and games (the server stays at 0)')
    p.add_argument('--min-batch', type=int, default=0, help='rows the server waits for (0: a third of the workers)')
    p.add_argument('--max-wait-ms', type=float, default=1.0)
    p.add_argument('--servers', type=int, default=1,
                   help='inference threads, each serving its share of the workers (uniform and model policies)')
    p.add_argument('--lag', type=int, default=0, choices=(0, 1),
                   help='1: an action is applied one decision after its observation; the game does not wait for it')
    p.add_argument('--warm', type=float, default=90.0)
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--every', type=float, default=10.0)
    p.add_argument('--episode-seconds', type=float, default=0.0, help='episode time limit (0: the group\'s)')
    p.add_argument('--port', type=int, default=28100)
    p.add_argument('--name', default='fs')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    args.seconds_limit = args.episode_seconds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    ns = argparse.Namespace(groups_file=args.groups_file, group=args.group, tasks=args.tasks,
                            seconds=args.episode_seconds)
    spec = load_spec(ns)
    cfg = SamplerConfig(workers=args.workers, policy='random' if args.policy == 'random' else 'server',
                        frames_per_decision=args.frames_per_decision, episodes_per_state=args.episodes_per_state,
                        seed=args.seed, name=args.name, port=args.port, bridge_lua=default_bridge_lua(),
                        preload=default_preload(), lean=not args.full_bridge, stub_list=args.stub_list,
                        nice=args.nice, min_batch=args.min_batch, max_wait_ms=args.max_wait_ms, lag=args.lag)
    policy = on_batch = None
    info = dict(policy=args.policy, lean=not args.full_bridge, stub_list=args.stub_list, nice=args.nice,
                lag=args.lag, servers=args.servers)
    sizes = []
    if args.policy == 'uniform':
        rng = np.random.default_rng(args.seed)

        def policy(batch, idx):
            sizes.append(len(idx))
            a = np.zeros((len(idx), ACTION_F), np.int32)
            a[:, 0] = rng.integers(9, size=len(idx))
            a[:, 1] = rng.integers(5, size=len(idx))
            return a
    elif args.policy == 'model':
        import torch
        device = torch.device('cuda')
        model, params = make_model(args.width, args.layers, args.heads, device)
        info.update(parameters=params, width=args.width, layers=args.layers, heads=args.heads,
                    gpu=torch.cuda.get_device_name(0))
        store = dict(player=torch.zeros((args.store, PLAYER_F), device=device),
                     entities=torch.zeros((args.store, ENTITY_CAP, ENTITY_F), device=device),
                     grid=torch.zeros((args.store, *GRID), dtype=torch.uint8, device=device),
                     count=torch.zeros((args.store,), dtype=torch.int32, device=device),
                     action=torch.zeros((args.store, 2), dtype=torch.int64, device=device))
        cursor = [0]
        infer_s = [0.0]
        store_lock = threading.Lock()

        @torch.inference_mode()
        def policy(batch, idx):
            t = time.perf_counter()
            n = len(idx)
            sizes.append(n)
            player = torch.from_numpy(batch['player']).to(device)
            entities = torch.from_numpy(batch['entities']).to(device)
            grid = torch.from_numpy(batch['grid']).to(device)
            count = torch.from_numpy(batch['n_entities']).to(device)
            # only as many entity tokens as this batch needs (in steps of 8), not the record's 96
            used = min(ENTITY_CAP, max(8, -(-int(batch['n_entities'].max()) // 8) * 8))
            move, shoot, _ = model(player, entities[:, :used], count, grid.float())
            act = torch.stack([torch.multinomial(torch.softmax(move, -1), 1)[:, 0],
                               torch.multinomial(torch.softmax(shoot, -1), 1)[:, 0]], 1)
            with store_lock:   # the ring buffer's cursor is shared by the inference threads
                start = cursor[0]
                cursor[0] = (start + n) % args.store
            at = (start + torch.arange(n, device=device)) % args.store   # the learner's copy of the rows
            store['player'][at], store['entities'][at], store['grid'][at] = player, entities, grid
            store['count'][at], store['action'][at] = count, act
            a = np.zeros((n, ACTION_F), np.int32)
            a[:, :2] = act.cpu().numpy()
            infer_s[0] += time.perf_counter() - t
            return a
    sampler = ForkSampler(cfg, spec)
    samples = []
    try:
        stop = threading.Event()
        threads = []
        if policy is not None and args.servers > 1:
            # each thread answers its own share of the workers; the GIL is released inside the CUDA calls
            for k in range(args.servers):
                share = sampler.conns[k::args.servers]
                th = threading.Thread(target=sampler.serve, args=(policy, 1e9), kwargs=dict(conns=share, stop=stop),
                                      daemon=True)
                th.start()
                threads.append(th)

        def wait(seconds):
            if policy is None or threads:
                time.sleep(seconds)
                return 0, 0
            return sampler.serve(policy, seconds)

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.warm:   # until every worker has played, at most --warm seconds
            wait(2.0)
            if (sampler.stats[:, 2] > 0).all():
                break
        info['warm_s'] = round(time.perf_counter() - t0, 1)
        wait(5.0)
        sizes.clear()
        if args.policy == 'model':
            infer_s[0] = 0.0
        before, t_before = sampler.totals(), time.perf_counter()
        first = dict(before)
        t_first = t_before
        batches_total = rows_total = 0
        while time.perf_counter() - t_first < args.seconds:
            b, r = wait(args.every)
            batches_total += b
            rows_total += r
            now, t_now = sampler.totals(), time.perf_counter()
            dt = t_now - t_before
            sample = dict(t=round(t_now - t_first, 1), frames_per_s=(now['frames'] - before['frames']) / dt,
                          decisions_per_s=(now['decisions'] - before['decisions']) / dt,
                          episodes_per_s=(now['episodes'] - before['episodes']) / dt, errors=now['errors'],
                          load=os.getloadavg()[0])
            samples.append(sample)
            print(json.dumps(sample), flush=True)
            before, t_before = now, t_now
        last, t_last = sampler.totals(), time.perf_counter()
        d = {k: last[k] - first[k] for k in last}
        dt = t_last - t_first
        procs, pss, rss = game_memory_mib()
        decisions = max(d['decisions'], 1.0)
        summary = dict(info, workers=cfg.workers, frames_per_decision=cfg.frames_per_decision,
                       episodes_per_state=cfg.episodes_per_state, group=spec['name'], seconds=round(dt, 1),
                       frames_per_s=d['frames'] / dt, x_real_time=d['frames'] / dt / 30,
                       decisions_per_s=d['decisions'] / dt, episodes_per_s=d['episodes'] / dt,
                       game_hours_per_hour=d['frames'] / dt / 30,
                       decisions_per_episode=d['decisions'] / max(d['episodes'], 1.0),
                       outcomes=dict(win=d['wins'], death=d['deaths'], time_limit=d['timeouts']),
                       forks=d['forks'], states=d['states'], extra_episodes=d['extra_episodes'], errors=last['errors'],
                       worker_ms_per_decision=dict(step=1000 * d['step_s'] / decisions,
                                                   encode=1000 * d['encode_s'] / decisions,
                                                   wait_for_action=1000 * d['wait_s'] / decisions,
                                                   fork=1000 * d['fork_s'] / decisions,
                                                   wait_for_state=1000 * d['state_wait_s'] / decisions,
                                                   swap_state=1000 * d['swap_s'] / decisions,
                                                   close=1000 * d['close_s'] / decisions,
                                                   loop=1000 * d['loop_s'] / decisions),
                       worker_busy_share=(d['step_s'] + d['encode_s'] + d['fork_s']) / (dt * cfg.workers),
                       fork_ms=1000 * d['fork_s'] / max(d['forks'], 1.0),
                       game_processes=procs, game_pss_mib=round(pss), game_rss_mib=round(rss),
                       load_average=os.getloadavg()[0])
        if sizes:
            summary.update(batches_per_s=len(sizes) / dt, batch_mean=float(np.mean(sizes)),
                           batch_p50=float(np.median(sizes)), batch_max=int(np.max(sizes)))
        if args.policy == 'model':
            summary.update(server_infer_share=infer_s[0] / dt, infer_ms_per_batch=1000 * infer_s[0] / max(len(sizes), 1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        (out / 'samples.json').write_text(json.dumps(samples, indent=1))
        print('SUMMARY', json.dumps(summary))
        stop.set()
        for th in threads:
            th.join(5.0)
    finally:
        sampler.close()


if __name__ == '__main__':
    main()
