"""Per-step inference cost of the AB+ learner: the current eager path versus a static, sync-free
path replayed as a CUDA graph (research for speeding up collection).

One collection step for N environments = encode each environment's newest frame (entity encoder)
+ the 64-frame temporal Transformer + policy/value heads + masked sampling, then the actions to the
host. The eager path is FrameSampler's (gpu_ppo.py): encode_frames compacts entities with boolean
indexing, .item() and nonzero(), and SB3's masked distributions validate their arguments, so the
host waits for the device several times per step. The static path encodes all 256 entity slots
with an attention padding mask, keeps every shape fixed and samples with Gumbel-max, so a whole
step can be captured once and replayed.

Checks: static encoding and log-probabilities against the eager ones (max abs difference), and
the same greedy actions. Times: median per step over many steps after warm-up, plus a profiler
count of host-device synchronisations in the eager step.
usage: python abplus_bench_sampler.py <checkpoint dir> [envs] [steps]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaac_bridge.gpu_ppo import GpuMaskablePPO

CKPT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
H = 64
dev = torch.device('cuda')
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

model = GpuMaskablePPO.load(CKPT / 'model.zip', device=dev)
policy = model.policy
policy.set_training_mode(False)
enc = policy.features_extractor
space = policy.observation_space
INT_KEYS = {'player_anim', 'active_kind', 'entity_kind', 'entity_anim'}


def random_frames(n, t0):
    """One frame per environment, shaped (n, 1, ...) like FrameSampler.encode's input."""
    g = torch.Generator(device='cpu').manual_seed(int(t0))
    obs = {}
    for k, s in space.spaces.items():
        shape = (n, 1) + tuple(s.shape[1:])
        if k in INT_KEYS:
            v = torch.randint(0, 64, shape, generator=g, dtype=torch.int32)
        else:
            v = torch.randn(shape, generator=g) * 0.5
        obs[k] = v
    kinds = obs['entity_kind']
    kinds[..., 0] = torch.randint(1, 300, kinds[..., 0].shape, generator=g, dtype=torch.int32)
    count = torch.randint(3, 40, (n, 1), generator=g)
    obs['entity_mask'] = (torch.arange(256)[None, None, :] < count[..., None]).float()
    obs['terrain'] = (torch.rand(obs['terrain'].shape, generator=g) > 0.6).float()
    pa = obs['previous_action']
    pa[..., 0] = torch.randint(0, 45, pa[..., 0].shape, generator=g).float()
    pa[..., 1] = torch.randint(0, 2, pa[..., 1].shape, generator=g).float()
    pa[..., 2] = torch.randint(0, 2, pa[..., 2].shape, generator=g).float()
    pa[..., 3] = 1.0
    obs['time'] = torch.full((n, 1), float(t0) * 2 / 30)
    obs['history_mask'] = torch.ones((n, 1))
    obs['remaining_time'] = torch.rand((n, 1), generator=g)
    obs['player'][..., 9] = torch.randint(0, 2, (n, 1), generator=g).float()   # bombs held
    return {k: v.to(dev) for k, v in obs.items()}


# ---- the static path -------------------------------------------------------------------------------
def encode_static(o):
    """encode_frames for single, valid frames without data-dependent shapes: (n, features_dim)."""
    n = o['player'].shape[0]
    inputs = [o['player'], enc.animation_embedding(o['player_anim']), enc.active_item(o['active_kind'].long().squeeze(-1))]
    if enc.has_deadline:
        inputs.append(o['remaining_time'].unsqueeze(-1))
    player = enc.player(torch.cat(inputs, -1))
    kinds = o['entity_kind'].long()
    encoded = enc.entity(torch.cat([o['entities'], enc.entity_type(kinds[..., 0]), enc.variant(kinds[..., 1]),
                                    enc.subtype(kinds[..., 2]), enc.animation_embedding(o['entity_anim'])], -1))
    keys = torch.cat([enc.empty_entity.expand(n, 1, -1), encoded], 1)
    padding = torch.cat([torch.zeros((n, 1), dtype=torch.bool, device=dev), ~o['entity_mask'].bool()], 1)
    queries = enc.queries.unsqueeze(0) + enc.player_query(player).unsqueeze(1)
    summary, _ = enc.entity_attention(queries, keys, keys, key_padding_mask=padding, need_weights=False)
    terrain = enc.map_cnn(o['terrain'])
    old = o['previous_action']
    actions = enc.action(torch.cat([enc.joint_action(old[:, 0].long()), enc.bomb_action(old[:, 1].long()),
                                    enc.item_action(old[:, 2].long()), old[:, 3:4]], -1)) * old[:, 3:4]
    return enc.fusion(torch.cat([player, summary.flatten(1), terrain, actions], -1))


SPLITS = [45, 2, 2]


def act_static(latent, masks, noise):
    """Masked MultiCategorical sampling with Gumbel-max (same distribution as SB3's sampling)."""
    pi, vf = policy.mlp_extractor(latent)
    logits = policy.action_net(pi)
    logits = torch.where(masks, logits, torch.full_like(logits, -1e8))
    actions, logps = [], []
    for part, g in zip(torch.split(logits, SPLITS, -1), torch.split(noise, SPLITS, -1)):
        logp = F.log_softmax(part, -1)
        a = torch.argmax(part - torch.log(-torch.log(g)), -1)
        actions.append(a)
        logps.append(logp.gather(-1, a[:, None])[:, 0])
    return torch.stack(actions, -1), torch.stack(logps, -1).sum(-1), policy.value_net(vf).flatten()


# ---- state shared by both paths: a feature window per environment ----------------------------------
features = torch.zeros((H + 1, N, enc.features_dim), device=dev)
times_buf = torch.zeros((H + 1, N), device=dev)
lengths = torch.full((N,), H, dtype=torch.long, device=dev)
with torch.no_grad():
    for t in range(H):
        o = random_frames(N, t)
        features[t] = enc.encode_frames(o)[:, 0]
        times_buf[t] = o['time'][:, 0]
POS = H  # the newest frame's row


def window():
    ids = torch.arange(N, device=dev)
    j = torch.arange(H, device=dev)[None, :]
    valid = j < lengths[:, None]
    times = torch.where(valid, POS - lengths[:, None] + 1 + j, 0)
    fused = features[times, ids[:, None]].masked_fill(~valid[:, :, None], 0)
    elapsed = times_buf[times, ids[:, None]].masked_fill(~valid, 0)
    seq = enc.temporal_features(fused, elapsed, valid)
    return seq[torch.arange(N, device=dev), valid.long().sum(-1) - 1]


def masks_of(o):
    m = torch.ones((N, 49), dtype=torch.bool, device=dev)
    m[:, 46] = o['player'][:, 0, 9] > 0
    m[:, 48] = False
    return m


def eager_step(o):
    f = enc.encode_frames(o)[:, 0]
    features[POS] = f
    times_buf[POS] = o['time'][:, 0]
    latent = window()
    pi, vf = policy.mlp_extractor(latent)
    dist = policy._get_action_dist_from_latent(pi)
    dist.apply_masking(masks_of(o))
    actions = dist.get_actions(deterministic=False)
    logp = dist.log_prob(actions)
    value = policy.value_net(vf).flatten()
    return actions.cpu(), logp, value


static_in = {k: v[:, 0].clone() for k, v in random_frames(N, 1000).items()}
static_noise = torch.rand((N, 49), device=dev).clamp_(1e-10, 1 - 1e-7)


def static_step():
    features[POS] = encode_static(static_in)
    times_buf[POS] = static_in['time']
    m = torch.ones((N, 49), dtype=torch.bool, device=dev)
    m[:, 46] = static_in['player'][:, 9] > 0
    m[:, 48] = False
    static_noise.uniform_(0, 1).clamp_(1e-10, 1 - 1e-7)
    return act_static(window(), m, static_noise)


def timed(fn, steps):
    out = []
    for i in range(steps):
        t = time.perf_counter()
        fn(i)
        out.append(1000 * (time.perf_counter() - t))
    out = np.array(out[steps // 10:])
    return {'median_ms': round(float(np.median(out)), 3), 'p25_ms': round(float(np.percentile(out, 25)), 3),
            'p75_ms': round(float(np.percentile(out, 75)), 3)}


report = {'envs': N, 'steps': STEPS, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
with torch.no_grad():
    # 1. equivalence of the static encoding and log-probabilities
    o = random_frames(N, 2000)
    eager_f = enc.encode_frames(o)[:, 0]
    static_f = encode_static({k: v[:, 0] for k, v in o.items()})
    report['encode_max_abs_diff'] = float((eager_f - static_f).abs().max())
    latent = window()
    m = masks_of(o)
    pi, vf = policy.mlp_extractor(latent)
    dist = policy._get_action_dist_from_latent(pi)
    dist.apply_masking(m)
    a = dist.get_actions(deterministic=False)
    eager_logp = dist.log_prob(a)
    logits = torch.where(m, policy.action_net(pi), torch.full((N, 49), -1e8, device=dev))
    static_logp = sum(F.log_softmax(part, -1).gather(-1, a[:, i:i + 1])[:, 0]
                      for i, part in enumerate(torch.split(logits, SPLITS, -1)))
    report['logprob_max_abs_diff'] = float((eager_logp - static_logp).abs().max())
    greedy = dist.get_actions(deterministic=True)
    report['greedy_actions_equal'] = bool(torch.equal(greedy, torch.stack(
        [part.argmax(-1) for part in torch.split(logits, SPLITS, -1)], -1)))

    # 2. eager step time and its host-device synchronisations
    frames = [random_frames(N, 3000 + i) for i in range(8)]
    for i in range(20):
        eager_step(frames[i % 8])
    torch.cuda.synchronize()
    report['eager'] = timed(lambda i: eager_step(frames[i % 8]), STEPS)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for i in range(10):
            eager_step(frames[i % 8])
    ev = prof.key_averages()
    count = lambda names: int(sum(e.count for e in ev if e.key in names))
    report['eager_per_step'] = {
        'cuda_sync_calls': count({'cudaStreamSynchronize', 'cudaDeviceSynchronize'}) / 10,
        'cuda_memcpy_calls': count({'cudaMemcpyAsync'}) / 10,
        'kernel_launches': count({'cudaLaunchKernel'}) / 10,
        'gpu_kernel_ms': round(sum(getattr(e, 'device_time_total', getattr(e, 'cuda_time_total', 0))
                                   for e in ev if e.device_type.name == 'CUDA') / 1000 / 10, 3),
    }

    # 3. the static step, eager and captured
    for i in range(20):
        static_step()
    torch.cuda.synchronize()
    report['static_eager'] = timed(lambda i: static_step()[0].cpu(), STEPS)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            static_step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = static_step()

    def graph_step(i):
        fresh = frames[i % 8]
        for k, v in static_in.items():
            v.copy_(fresh[k][:, 0], non_blocking=True)
        graph.replay()
        return graph_out[0].cpu()
    for i in range(20):
        graph_step(i)
    torch.cuda.synchronize()
    report['static_graph'] = timed(graph_step, STEPS)
    # the graph's actions follow the same distribution; check its log-probs against eager
    graph_step(0)
    torch.cuda.synchronize()
    o = frames[0]
    features[POS] = enc.encode_frames(o)[:, 0]
    pi, vf = policy.mlp_extractor(window())
    dist = policy._get_action_dist_from_latent(pi)
    dist.apply_masking(masks_of(o))
    report['graph_logprob_vs_eager_max_abs_diff'] = float((dist.log_prob(graph_out[0]) - graph_out[1]).abs().max())
    report['graph_value_vs_eager_max_abs_diff'] = float((policy.value_net(vf).flatten() - graph_out[2]).abs().max())
print(json.dumps(report, indent=1))
