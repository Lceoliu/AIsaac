"""C44 start check (rl/docs/GOAL_CONDITIONED_DESIGN.md 2.2 item 5): a goal-line checkpoint (C43's final, observation goal-hp)
migrated into goal-hp2 (window-relative player position, the full-room branch at zero) computes the checkpoint's function
on real 1x1-room frames, every task, bit for bit.

The checkpoint plays whole option sequences of goal groups (abplus_eval_options: the chain, emptied rooms, the single-room
COMBAT) greedy on one AB+ instance. The environment's history is a tee: every raw observation is encoded by the
checkpoint's own VisibleHistory (goal-hp) and by a goal-hp2 one, with the same option boundaries, goals and room changes. Per
decision: the two frames agree on every key goal-hp has (goal-hp2 only adds room_bits), and the checkpoint on its window and
the migrated policy on the goal-hp2 window give equal logits and values, on the CPU and CUDA (TF32 off), by the full-window
path and the per-frame static path (the CUDA-graph sampler's).

usage: python check_c44_migration.py <goal-line checkpoint> <groups file> [seeds per group] [port] [first seed]
Linux host only (one AB+ instance)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import abplus_eval_options as E
from abplus_eval import CachedPolicy, load_policy
from isaac_bridge.abplus import launch_abplus, stop_abplus
from isaac_bridge.abplus_groups import load_groups
from isaac_bridge.abplus_options import OptionSequence
from isaac_bridge.abplus_worker import observation_options
from isaac_bridge.training_session import _room_key
from isaac_bridge.transformer_obs import ROOM_BITS, VisibleHistory
from isaac_bridge.transformer_policy import GeometryPolicy
from test_goal_model import heads, static_heads

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class Tee:
    """The environment's history, doubled: reads come from the primary (goal-hp, what the checkpoint acts on); the
    option machinery's writes and calls reach both; append keeps the secondary's window in .second."""
    MIRRORED = ('goal_state', 'deadline_s')

    def __init__(self, primary, secondary):
        object.__setattr__(self, 'primary', primary)
        object.__setattr__(self, 'secondary', secondary)
        object.__setattr__(self, 'second', None)

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def __setattr__(self, name, value):
        if name in self.MIRRORED:
            setattr(self.primary, name, value)
            setattr(self.secondary, name, value)
        else:
            object.__setattr__(self, name, value)

    def _both(name):
        def call(self, *args, **kwargs):
            getattr(self.secondary, name)(*args, **kwargs)
            return getattr(self.primary, name)(*args, **kwargs)
        return call

    clear = _both('clear')
    soft_clear = _both('soft_clear')
    room_clear = _both('room_clear')
    set_previous_action = _both('set_previous_action')
    encode = _both('encode')

    def append(self, obs):
        object.__setattr__(self, 'second', self.secondary.append(obs))
        return self.primary.append(obs)


def migrated(checkpoint, space):
    """The checkpoint's policy rebuilt on the goal-hp2 space (optimizer groups as C44) with the checkpoint's weights; only
    the full-room branch is new, and it starts at zero."""
    from stable_baselines3.common.save_util import load_from_zip_file
    data, params, _ = load_from_zip_file(Path(checkpoint) / 'model.zip', device='cpu')
    policy = GeometryPolicy(space, data['action_space'], lambda _: 1e-4, **data['policy_kwargs'], lr_groups=True).eval()
    result = policy.load_state_dict(params['policy'], strict=False)
    if result.unexpected_keys or not result.missing_keys or not all(_room_key(k) for k in result.missing_keys):
        raise RuntimeError(f'unexpected migration: {result}')
    if not policy.features_extractor.room_injection_zero():
        raise RuntimeError('room_fusion is not zero')
    return policy


class Greedy:
    """abplus_eval_options.act's arguments: the checkpoint plays every option, greedy."""
    scripted_goto, stochastic = False, False


def batch(window, device):
    return {k: torch.as_tensor(v[None]).to(device) for k, v in window.items()}


def main():
    ck, groups_file = Path(sys.argv[1]), Path(sys.argv[2])
    per_group = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 27896
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 2147496000
    config = json.loads((ck / 'state.json').read_text())['config']
    groups = {g['name']: g for g in load_groups(groups_file)}
    old = load_policy(ck, 'cpu')
    second = VisibleHistory(config['history'], config['entity_capacity'], **observation_options('goal-hp2'))
    new = migrated(ck, second.space)
    cuda = torch.cuda.is_available()
    pairs = {'cpu': (old, new)}
    if cuda:
        import copy
        pairs['cuda'] = (copy.deepcopy(old).to('cuda'), copy.deepcopy(new).to('cuda'))
    cached = CachedPolicy(old, config['history'], True)
    counts = {f'{d}/{p}': [0, 0] for d in pairs for p in ('full', 'static')}
    frames_checked = frame_mismatch = bits_bad = 0
    proc = launch_abplus('checkcf', port, 'exact')
    try:
        for name in ('chain', 'gotoempty', 'combat'):
            group = dict(groups[name])
            group.pop('goal_strata', None)
            mode = group.get('mode', 'combat')
            env = E.make_env(port, config, group, mode)
            env.history = Tee(env.history, VisibleHistory(config['history'], config['entity_capacity'],
                                                          **observation_options('goal-hp2')))
            try:
                for seed in range(first, first + per_group):
                    obs, info = env.reset(options={'arena_seed': seed, 'bombs': 1})
                    seq = OptionSequence(group, seed, info, None)
                    started = seq.start(env, first=True)
                    steps = 0
                    while started is not None:
                        cached.reset()
                        while True:
                            action = E.act(env, cached, seq, Greedy, True)
                            _, _, terminated, truncated, inf = env.step(action)
                            steps += 1
                            h = env.history
                            a, b = h.primary.frames[-1], h.secondary.frames[-1]
                            frames_checked += 1
                            frame_mismatch += any(not np.array_equal(a[k], b[k]) for k in a)
                            player = [(int(v) >> ROOM_BITS.index('player')) & 1 for v in b['room_bits'].flat]
                            bits_bad += sum(player) != 1
                            for device, (po, pn) in pairs.items():
                                wo, wn = batch(_window(h.primary), device), batch(_window(h.secondary), device)
                                with torch.no_grad():
                                    for path, fn in (('full', heads), ('static', static_heads)):
                                        lo, vo = fn(po, wo)
                                        ln, vn = fn(pn, wn)
                                        c = counts[f'{device}/{path}']
                                        c[0] += 1
                                        c[1] += int(not (torch.equal(lo, ln) and torch.equal(vo, vn)))
                            out = seq.outcome(env.raw_obs, inf['outcome'])
                            seq.step(env.raw_obs, out, 0)
                            if terminated or truncated:
                                break
                        if not seq.next(out, env.raw_obs['room']['room_idx']):
                            break
                        started = seq.start(env, room_changed=seq.option['task'] == 'goto_door')
                    print(json.dumps(dict(group=name, seed=seed, steps=steps, counts=counts, frames=frames_checked,
                                          frame_mismatch=frame_mismatch, room_bits_bad=bits_bad)), flush=True)
            finally:
                env.close()
    finally:
        stop_abplus(proc, 'checkcf')
    ok = frame_mismatch == 0 and bits_bad == 0 and all(c[1] == 0 for c in counts.values())
    print(json.dumps(dict(result='identical' if ok else 'DIFFERENT', counts=counts, frames=frames_checked,
                          frame_mismatch=frame_mismatch, room_bits_bad=bits_bad)))


def _window(history):
    """The history's current window as VisibleHistory.append builds it (right-padded)."""
    result = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in history.space.spaces.items()}
    for i, record in enumerate(history.frames):
        for key in result:
            result[key][i] = record[key]
    return result


if __name__ == '__main__':
    main()
