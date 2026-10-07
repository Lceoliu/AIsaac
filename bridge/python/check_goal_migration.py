"""Goal-conditioned line M0: the C39 checkpoint migrated into the goal-line policy (training_session.warm_start_model) against
the checkpoint itself on real COMBAT frames. The checkpoint plays greedy episodes of a tasks file (its own settings) on one
AB+ instance; the observation carries the goal fields (COMBAT); on every decision both policies are evaluated on the same
window, on each device (cpu, cuda: strict FP32, TF32 off as in training) and by both encoding paths (full window, the
per-frame static path of the CUDA-graph sampler); logits and values must be equal bit for bit.

usage: python check_goal_migration.py <checkpoint_dir> <tasks_file> <episodes> [port] [first_seed] [goal_checkpoint_dir]
With goal_checkpoint_dir, that goal-line checkpoint (e.g. after stage-1 training) is compared instead of a fresh migration.
Run on the Linux host only (one AB+ instance at nice 19).
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from abplus_eval import load_policy
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_tasks import TaskSampler
from isaac_bridge.abplus_worker import observation_options
from isaac_bridge.training_session import warm_start_model
from isaac_bridge.transformer_obs import factored_masks, factored_to_joint
from isaac_bridge.transformer_policy import GeometryPolicy

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def heads(p, obs):
    pi, vf = p.mlp_extractor(p.features_extractor(obs))
    return p.action_net(pi), p.value_net(vf).flatten()


def static_heads(p, obs):
    fe = p.features_extractor
    b, h = obs['history_mask'].shape
    frames = fe.encode_frame_static({k: v.flatten(0, 1) for k, v in obs.items()}).view(b, h, -1)
    latent = fe.temporal_features(frames, obs['time'], obs['history_mask'].bool())
    last = obs['history_mask'].long().sum(-1) - 1
    pi, vf = p.mlp_extractor(latent[torch.arange(b), last])
    return p.action_net(pi), p.value_net(vf).flatten()


def main():
    checkpoint, tasks_file, episodes = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 27880
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 2147495000
    config = json.loads((checkpoint / 'state.json').read_text())['config']
    options = observation_options(config['reward_profile'])
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    old, new = {}, {}
    for device in devices:
        old[device] = load_policy(checkpoint, device)
    proc = launch_abplus('goalcheck', port, 'exact')
    env = AbplusTransformerEnv(port=port, max_episode_frames=int(config['max_episode_seconds'] * 30), **options, goal=True,
                               deadline_s=float(config['max_episode_seconds']),
                               frames_per_decision=int(config.get('frames_per_decision', 2)),
                               history=config['history'], entity_capacity=config['entity_capacity'])
    env.bridge.binary_obs = True
    if config.get('lineage_mode') is not None:
        env.bridge.lineage_mode = int(config['lineage_mode'])
    spec = json.loads(Path(tasks_file).read_text(encoding='utf8'))
    env.bridge.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss'], None)
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _, _ = load_from_zip_file(checkpoint / 'model.zip', device='cpu', custom_objects={
        'lr_schedule': lambda _: 0.0, 'learning_rate': 0.0, 'clip_range': lambda _: 0.0, 'rollout_buffer_class': None})
    trained = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    for device in devices:
        if trained is not None:
            new[device] = load_policy(trained, device)
            migration = {'trained_checkpoint': str(trained)}
            if new[device].observation_space != env.observation_space:
                raise ValueError('the goal checkpoint was trained on another observation space')
            continue
        p = GeometryPolicy(env.observation_space, data['action_space'], lambda _: 0.0, **data['policy_kwargs'])
        migration = warm_start_model(SimpleNamespace(policy=p), checkpoint)
        new[device] = p.to(device).eval()
    print(json.dumps({'migration': migration, 'devices': devices}), flush=True)
    counts = {f'{d}/{path}': [0, 0, 0.0] for d in devices for path in ('full', 'static')}   # frames, mismatches, max diff
    t0 = time.monotonic()
    try:
        for e in range(episodes):
            seed = first + e
            obs, info = env.reset(options={'arena_seed': seed})
            steps = 0
            while True:
                mask = torch.as_tensor(factored_masks(env.action_masks()))[None]
                action = None
                for device in devices:
                    window = {k: torch.as_tensor(np.asarray(v), device=device)[None].float() for k, v in obs.items()}
                    old_window = {k: v for k, v in window.items() if k in old[device].observation_space.spaces}
                    with torch.no_grad():
                        for name, path in (('full', heads), ('static', static_heads)):
                            ol, ov = path(old[device], old_window)
                            nl, nv = path(new[device], window)
                            c = counts[f'{device}/{name}']
                            c[0] += 1
                            if not (torch.equal(ol, nl) and torch.equal(ov, nv)):
                                c[1] += 1
                                c[2] = max(c[2], float((ol - nl).abs().max()), float((ov - nv).abs().max()))
                            if device == 'cpu' and name == 'full':
                                logits = ol.masked_fill(~mask, -1e8)
                                action = torch.stack([t.argmax(-1) for t in torch.split(logits, [9, 5, 2, 2], -1)], -1)[0]
                obs, _, terminated, truncated, info = env.step(factored_to_joint(action.numpy()))
                steps += 1
                if terminated or truncated:
                    break
            print(json.dumps({'episode': e, 'seed': seed, 'outcome': info['outcome'], 'steps': steps,
                              'counts': counts}), flush=True)
    finally:
        try:
            env.close()
        finally:
            stop_abplus(proc, 'goalcheck')
    ok = all(c[1] == 0 for c in counts.values())
    print(json.dumps({'result': 'identical' if ok else 'MISMATCH', 'counts': counts,
                      'seconds': round(time.monotonic() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
