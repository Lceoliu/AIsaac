"""Bridge v1 transfer evaluation: a simulator-trained checkpoint played on AB+ v1.06.

Each worker process owns one AB+ instance (render-lite exact mode by default), one
AbplusTransformerEnv and its own copy of the policy. The policy runs incrementally with
GpuMaskablePPO FrameSampler semantics: each new frame is encoded once and the causal temporal
Transformer runs over the cached window (right-padded windows make the valid prefix sufficient).
--verify-steps compares that path against the full-window policy forward.

Seeds, arena and metrics follow the simulator audit (runs/ppo-audit/20260923-e1e2/e2_reeval.py),
so AB+ and simulator episodes pair seed by seed. Results append to <out>/results.jsonl; a rerun
skips finished seeds.

usage: python abplus_eval.py --checkpoint DIR --seeds seeds.json --out DIR [--instances 4]
                             [--limit N] [--device cuda|cpu] [--stochastic --sample-seed S]
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, sim_arena, stop_abplus
from isaac_bridge.abplus_reward import CombatV2
from isaac_bridge.abplus_tasks import TaskSampler

DEADLINE_PROFILES = ('combat-v1', 'combat-v2')  # reward profiles trained with the 120 s deadline input

BOSS_MAX_HP = 250.0  # Monstro; the simulator audit reports absolute boss HP
TYPE_MONSTRO = 20


def load_policy(checkpoint, device):
    # Only the policy is needed: skip the rollout buffer and schedules of the saved algorithm.
    data, params, _ = load_from_zip_file(Path(checkpoint) / 'model.zip', device=device, custom_objects={
        'lr_schedule': lambda _: 0.0, 'learning_rate': 0.0, 'clip_range': lambda _: 0.0,
        'rollout_buffer_class': None})
    policy = data['policy_class'](data['observation_space'], data['action_space'], lambda _: 0.0,
                                  **data['policy_kwargs'])
    policy.load_state_dict(params['policy'], strict=True)
    policy.to(device).set_training_mode(False)
    return policy


class CachedPolicy:
    """One environment's policy with per-frame feature caching (FrameSampler.latent/action)."""

    def __init__(self, policy, history, deterministic):
        self.policy, self.encoder = policy, policy.features_extractor
        self.deterministic, self.device = deterministic, policy.device
        self.features = collections.deque(maxlen=history)
        self.times = collections.deque(maxlen=history)

    def reset(self):
        self.features.clear()
        self.times.clear()

    @torch.inference_mode()
    def distribution(self, frame, mask):
        obs = {k: torch.as_tensor(np.asarray(v), device=self.device)[None, None].float() for k, v in frame.items()}
        self.features.append(self.encoder.encode_frames(obs)[0, 0])
        self.times.append(float(frame['time']))
        sequence = torch.stack(tuple(self.features))[None]
        elapsed = torch.tensor([list(self.times)], dtype=torch.float32, device=self.device)
        latent = self.encoder.temporal_features(sequence, elapsed, torch.ones_like(elapsed, dtype=torch.bool))[:, -1]
        pi, _ = self.policy.mlp_extractor(latent)
        dist = self.policy._get_action_dist_from_latent(pi)
        dist.apply_masking(torch.as_tensor(mask, device=self.device)[None])
        return dist

    @torch.inference_mode()
    def full_distribution(self, obs, mask):
        tensor, _ = self.policy.obs_to_tensor(obs)
        return self.policy.get_distribution(tensor, action_masks=np.asarray(mask)[None])


def logits(dist):
    return torch.cat([d.logits for d in dist.distributions], -1)


class Metrics:
    """e2_reeval.Acc on AB+ observations (hp in half hearts, boss HP from the visible boss bar)."""

    def __init__(self, raw):
        p = raw['players'][0]
        self.origin = raw['logic_frames']
        self.hp0 = self.hp = self.health(p)
        self.boss0 = self.boss = self.boss_hp(raw)
        self.bombs = p['bombs']
        self.n = self.shoot = self.idle = self.aim_n = self.aim_ok = 0
        self.dist_sum = self.dist_n = self.close = self.far = 0
        self.hurts = self.hits = self.bombs_used = self.proj_sum = 0
        self.first_hurt = None
        self.anomalies = []  # frames where the arena does not hold exactly one Monstro

    @staticmethod
    def health(p):
        return p['hearts'] + p.get('soul', 0)

    @staticmethod
    def bosses(raw):
        return [e for e in raw['entities'] if e['type'] == TYPE_MONSTRO and e.get('boss')]

    def boss_hp(self, raw):
        return sum(max(0.0, e.get('boss_hp', 0.0)) for e in self.bosses(raw)) * BOSS_MAX_HP

    def step(self, action, raw, outcome):
        p = raw['players'][0]
        self.n += 1
        joint = int(action[0])
        shoot = joint % 5
        self.shoot += shoot != 0
        self.idle += joint // 5 == 0
        bosses = self.bosses(raw)
        monstros = [e for e in raw['entities'] if e['type'] == TYPE_MONSTRO]
        if (len(monstros) != 1 or any(e.get('boss_hp', 0) > 1.0 for e in monstros)) and outcome != 'win'                 and len(self.anomalies) < 8:
            self.anomalies.append([raw['logic_frames'] - self.origin, [
                (e['id'], e['variant'], e['subtype'], e.get('boss'), e.get('boss_hp'), e.get('anim'), e.get('aframe'),
                 e.get('age'), e['pos']) for e in monstros]])
        if bosses:
            dx, dy = bosses[0]['pos'][0] - p['pos'][0], bosses[0]['pos'][1] - p['pos'][1]
            d = math.hypot(dx, dy)
            self.dist_sum += d
            self.dist_n += 1
            self.close += d < 120
            self.far += d > 260
            if shoot:
                self.aim_n += 1
                best = (1 if dy < 0 else 3) if abs(dy) > abs(dx) else (2 if dx > 0 else 4)
                self.aim_ok += shoot == best
        # The bridge blocks the lethal hit (AB+ cannot revive a dead player); count it as the
        # simulator does, where the final state has hp 0.
        hp = 0 if outcome == 'death' else self.health(p)
        if hp < self.hp:
            self.hurts += 1
            if self.first_hurt is None:
                self.first_hurt = (raw['logic_frames'] - self.origin) / 30
        self.hp = hp
        # A boss that is not visible keeps its last HP (only a win may remove it).
        bh = self.boss_hp(raw) if bosses or outcome == 'win' else self.boss
        if bh < self.boss:
            self.hits += 1
        self.boss = bh
        if p['bombs'] < self.bombs:
            self.bombs_used += self.bombs - p['bombs']
        self.bombs = p['bombs']
        self.proj_sum += sum(1 for e in raw['entities'] if e.get('projectile'))

    def summary(self, frames):
        minutes = max(frames, 1) / 30 / 60
        return dict(hurt_events=self.hurts, health_lost=self.hp0 - self.hp, boss_hp_final=self.boss,
                    damage_frac=(self.boss0 - self.boss) / self.boss0 if self.boss0 else None,
                    first_hurt_s=self.first_hurt, boss_hp_drop_steps=self.hits, hits_per_min=self.hits / minutes,
                    mean_dist=self.dist_sum / self.dist_n if self.dist_n else None,
                    close_frac=self.close / self.dist_n if self.dist_n else None,
                    far_frac=self.far / self.dist_n if self.dist_n else None,
                    shoot_frac=self.shoot / self.n, idle_move_frac=self.idle / self.n,
                    aim_axis_ok=self.aim_ok / self.aim_n if self.aim_n else None,
                    bombs_used=self.bombs_used, mean_projectiles=self.proj_sum / self.n,
                    **({'anomalies': self.anomalies} if self.anomalies else {}))


def run_episode(env, cached, seed, args, replay_path=None):
    arena = sim_arena(seed)
    if args.stochastic:
        torch.manual_seed(args.sample_seed + seed)
    obs, info = env.reset(options={'arena_seed': seed})
    env.bridge.last_reset = dict(info)
    task = env.bridge.last_reset.get('task', 'arena')
    reward_v2 = CombatV2()  # reported for every checkpoint, whatever reward it was trained with
    reward_v2.reset(env.raw_obs, task)
    frames = 0
    cached.reset()
    metrics = Metrics(env.raw_obs)
    writer = gzip.open(replay_path, 'wt', encoding='utf8') if replay_path else None
    if writer:
        writer.write(json.dumps({'metadata': {'seed': seed, 'arena': arena, 'format': 'abplus-raw-obs-v1'}}) + '\n')
        writer.write(json.dumps({'action': None, 'obs': env.raw_obs}, separators=(',', ':')) + '\n')
    verify, total_return, steps, t0 = [], 0.0, 0, time.monotonic()
    trajectory = hashlib.sha256()
    policy_s = 0.0
    try:
        while True:
            mask = env.action_masks()
            tp = time.perf_counter()
            dist = cached.distribution(env.history.frames[-1], mask)
            action = dist.get_actions(deterministic=not args.stochastic)[0].cpu().numpy()
            if steps < args.verify_steps:
                full = cached.full_distribution(obs, mask)
                verify.append(float((logits(dist) - logits(full)).abs().max()))
            policy_s += time.perf_counter() - tp
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            steps += 1
            reward_v2.step(env.raw_obs, info['outcome'], info['elapsed_frames'] - frames)
            frames = info['elapsed_frames']
            metrics.step(action, env.raw_obs, info['outcome'])
            raw = env.raw_obs
            trajectory.update(json.dumps([action.tolist(), raw['players'][0]['pos'], raw['players'][0]['hearts'],
                                          sorted((e['type'], e['pos'], str(e.get('anim')), e.get('aframe'))
                                                 for e in raw['entities'])]).encode())
            if writer:
                writer.write(json.dumps({'action': action.tolist(), 'obs': env.raw_obs}, separators=(',', ':')) + '\n')
            if terminated or truncated:
                break
    finally:
        if writer:
            writer.close()
    seconds = time.monotonic() - t0
    result = dict(seed=seed, task=task, outcome=info['outcome'],
                  layout=env.bridge.last_reset.get('room_variant', arena['variant']), frames=info['elapsed_frames'],
                  r=total_return, l=steps, **metrics.summary(info['elapsed_frames']),
                  entrance=env.bridge.last_reset.get('entrance', arena['entrance']), seconds=round(seconds, 2),
                  policy_ms=round(1000 * policy_s / max(1, steps), 3), trajectory=trajectory.hexdigest()[:16],
                  reward_v2=round(sum(reward_v2.totals.values()), 4),
                  reward_v2_components={k: round(v, 4) for k, v in reward_v2.totals.items()},
                  reward_v2_start=reward_v2.start)
    if verify:
        result['verify_max_logit_diff'] = max(verify)
    return result


def worker(index, args, config, tasks, results):
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.allow_tf32 = False  # GpuMaskablePPO evaluates in strict FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    name, port = f'{args.name}{index}', args.port + index
    policy = load_policy(args.checkpoint, args.device)
    cached = CachedPolicy(policy, config['history'], deterministic=not args.stochastic)
    state = {'proc': None, 'env': None}

    def start():
        state['proc'] = launch_abplus(name, port, args.mode)
        env = AbplusTransformerEnv(port=port, max_episode_frames=int(config['max_episode_seconds'] * 30),
                                   deadline=config.get('reward_profile') in DEADLINE_PROFILES,
                                   history=config['history'], entity_capacity=config['entity_capacity'])
        env.bridge.binary_obs = not args.json_obs
        env.bridge.tasks = TaskSampler.from_file(args.tasks_file) if args.tasks_file != 'none' else None
        if env.observation_space != policy.observation_space:
            raise ValueError(f'AB+ observation space {env.observation_space} differs from the checkpoint '
                             f'policy {policy.observation_space}')
        state['env'] = env

    def stop():
        try:
            if state['env'] is not None:
                state['env'].close()
        except Exception:
            pass
        finally:
            state['env'] = None
            if state['proc'] is not None:
                stop_abplus(state['proc'], name)
                state['proc'] = None

    try:
        start()
        while True:
            try:
                seed, replay = tasks.get_nowait()
            except queue.Empty:
                break
            for attempt in range(2):
                try:
                    result = run_episode(state['env'], cached, seed, args, replay)
                    break
                except Exception:
                    result = dict(seed=seed, outcome='error', layout=sim_arena(seed)['variant'],
                                  error=traceback.format_exc()[-3000:], attempt=attempt)
                    stop()
                    time.sleep(2)
                    start()
            result['worker'] = index
            results.put(result)
    finally:
        stop()
        results.put(None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seeds', required=True, help='e2 seeds.json ({"seeds":[{"seed":..}]}) or a JSON list')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--instances', type=int, default=4)
    p.add_argument('--mode', default='exact', choices=('exact', 'skip', 'render'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--verify-steps', type=int, default=0, help='compare cached vs full forward for N steps/episode')
    p.add_argument('--replays', type=int, default=0, help='write raw-observation replays for the first N seeds')
    p.add_argument('--repeat', type=int, default=1, help='episodes per seed (reproducibility check)')
    p.add_argument('--tasks-file', default='none', help="room mixture spec (abplus_tasks); 'none' = Monstro arena")
    p.add_argument('--json-obs', action='store_true', help='bridge v1 JSON observations instead of binary v2')
    p.add_argument('--port', type=int, default=27200)
    p.add_argument('--name', default='beval')
    args = p.parse_args()
    checkpoint, out = Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((checkpoint / 'state.json').read_text())
    config = saved['config']
    if args.seeds.startswith('range:'):
        start, count = map(int, args.seeds.split(':')[1:])
        seeds = list(range(start, start + count))
    else:
        raw = json.loads(Path(args.seeds).read_text())
        seeds = [s['seed'] if isinstance(s, dict) else int(s) for s in (raw['seeds'] if isinstance(raw, dict) else raw)]
    if args.limit:
        seeds = seeds[:args.limit]
    results_path = out / 'results.jsonl'
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if r['outcome'] != 'error':
                done.add(r['seed'])
    todo = [s for s in seeds if s not in done for _ in range(args.repeat)]
    meta = dict(checkpoint=str(checkpoint.resolve()), updates=saved['updates'], timesteps=saved['timesteps'],
                schema=config['schema'], reward_profile=config.get('reward_profile'), mode=args.mode,
                deterministic=not args.stochastic, sample_seed=args.sample_seed if args.stochastic else None,
                device=args.device, instances=args.instances, seeds=len(seeds), engine='abplus-1.06',
                tasks_file=args.tasks_file, observation_transport='json' if args.json_obs else 'binary v2',
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    (out / 'replays').mkdir(exist_ok=True)
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    for i, s in enumerate(todo):
        tasks.put((s, str(out / 'replays' / f'seed-{s}-{i}.jsonl.gz') if i < args.replays else None))
    n = min(args.instances, len(todo))
    print(f'{len(done)} done, {len(todo)} to run on {n} AB+ instances ({args.mode}, {args.device})', flush=True)
    procs = [ctx.Process(target=worker, args=(i, args, config, tasks, results), daemon=True) for i in range(n)]
    for proc in procs:
        proc.start()
    finished, t0, steps = 0, time.monotonic(), 0
    counts = collections.Counter()
    per_task = collections.defaultdict(lambda: {'outcomes': collections.Counter(), 'reward_v2': []})
    with results_path.open('a') as fh:
        while finished < n:
            try:
                r = results.get(timeout=60)
            except queue.Empty:
                if not any(proc.is_alive() for proc in procs):
                    print('all workers exited without reporting completion', flush=True)
                    break
                continue
            if r is None:
                finished += 1
                continue
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            counts[r['outcome']] += 1
            per_task[r.get('task', 'arena')]['outcomes'][r['outcome']] += 1
            if 'reward_v2' in r:
                per_task[r.get('task', 'arena')]['reward_v2'].append(r['reward_v2'])
            steps += r.get('l', 0)
            elapsed = time.monotonic() - t0
            print(f"{time.strftime('%H:%M:%S')} seed {r['seed']} L{r['layout']} {r['outcome']:<10} "
                  f"frames {r.get('frames', '-'):>5} dmg {r.get('damage_frac') or 0:.2f} | "
                  f"{sum(counts.values())}/{len(todo)} {dict(counts)} {steps / elapsed:.0f} decisions/s", flush=True)
    for proc in procs:
        proc.join()
    print(json.dumps({'outcomes': counts, 'seconds': round(time.monotonic() - t0, 1),
                      'decisions': steps, 'decisions_per_s': round(steps / max(1e-9, time.monotonic() - t0), 1),
                      'tasks': {k: {'outcomes': v['outcomes'],
                                    'reward_v2_mean': round(float(np.mean(v['reward_v2'])), 3) if v['reward_v2'] else None}
                                for k, v in sorted(per_task.items())}}))


if __name__ == '__main__':
    main()
