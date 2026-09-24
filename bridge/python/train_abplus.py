"""PPO on the original engine: GpuMaskablePPO over AbplusFrameVecEnv. Training requires --train.

One exclusive CUDA learner process batches every worker's newest frame into one policy call per
chunk; each worker process owns one environment slot with two AB+ instances (active + standby).
Task: a room-level combat mixture (abplus_tasks; default rl/bridge/abplus/catalog/mixture_basement1.json):
the simulator's Monstro arena, Basement I normal rooms with their own enemies and Basement I boss
rooms, chosen per episode from the seed; --tasks-file none trains the arena only. combat-v2 reward
(isaac_bridge/abplus_reward.py; --reward-profile combat-v1 for the simulator's reward), 120 s deadline,
optional start randomisation. Per rollout the log has each task's episodes and win rate and the mean
reward components; episodes.jsonl has the components of every episode. Progress is also logged in
game time: one decision is 2 logic frames, 30 logic frames are one game second, so
game/speed_x_realtime = decisions/s / 15.

Steam watch (isaac_bridge/steam_watch.py): every rollout checks that the Steam client runs (every
AB+ start needs it) and alerts when it stops, is still down every 10 min, or comes back, and when
workers report instance starts that failed: a JSON event in this log, abplus/steam_ok, a STEAM_DOWN
file in the run directory, a desktop notification on the host and $ABP_ALERT_CMD if set.
Evaluations are skipped while Steam is down; workers defer instance recycles.

Checkpoints hold weights, optimizer, counters and RNG. A resume starts fresh episodes (the
in-progress rooms are not rebuilt). Periodic evaluation runs abplus_eval.py as a separate CPU
process on a few extra instances, so the learner keeps the GPU to itself.
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from isaac_bridge.transformer_obs import DEADLINE_SCHEMA
from isaac_bridge.transformer_policy import CombatTransformer

GAME_FPS = 30
FRAMES_PER_DECISION = 2
HERE = Path(__file__).resolve().parent
SOURCES = ('train_abplus.py', 'abplus_eval.py', 'isaac_bridge/abplus.py', 'isaac_bridge/abplus_worker.py',
           'isaac_bridge/abplus_obs.py', 'isaac_bridge/abplus_tasks.py', 'isaac_bridge/abplus_reward.py',
           'isaac_bridge/abplus_vec.py', 'isaac_bridge/transformer_obs.py', 'isaac_bridge/transformer_policy.py',
           'isaac_bridge/gpu_ppo.py', 'isaac_bridge/gpu_buffer.py', 'isaac_bridge/gpu_env.py',
           'isaac_bridge/combat_reward.py', 'isaac_bridge/monstro_gym.py', 'isaac_bridge/env.py',
           'isaac_bridge/steam_watch.py')


def game_hours(decisions):
    return decisions * FRAMES_PER_DECISION / GAME_FPS / 3600


def source_hashes():
    result = {}
    for name in SOURCES:
        path = HERE / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    home = Path(os.environ.get('ABP_HOME', Path.home() / 'isaac-abplus'))
    for extra in (home / 'bridge' / 'abp_bridge.lua', home / 'tools' / 'libabp_turbo.so'):
        if extra.exists():
            result[str(extra)] = hashlib.sha256(extra.read_bytes()).hexdigest()[:16]
    return result


def session_class():
    from isaac_bridge.steam_watch import SteamWatch
    from isaac_bridge.training_session import TrainingSession

    class AbplusSession(TrainingSession):
        """TrainingSession with game-time/worker logging and out-of-process AB+ evaluation."""

        def __init__(self, config, out, completed=0, updates=0, start_timesteps=0):
            super().__init__(config, out, completed, updates)
            self.start_timesteps = start_timesteps
            self.last_time = time.monotonic()
            self.last_steps = None
            self.evaluation = None
            self.rollout_episodes = []
            self.steam = SteamWatch(out)

        def _on_rollout_end(self):
            now, steps = time.monotonic(), self.model.num_timesteps
            if self.last_steps is not None and now > self.last_time:
                rate = (steps - self.last_steps) / (now - self.last_time)
                self.logger.record('game/decisions_per_s', rate)
                self.logger.record('game/speed_x_realtime', rate * FRAMES_PER_DECISION / GAME_FPS)
            self.logger.record('game/hours_this_run', game_hours(steps - self.start_timesteps))
            self.logger.record('game/hours_total', game_hours(steps))
            diagnostics = self.model.env.diagnostics()
            for key, value in diagnostics.items():
                self.logger.record('abplus/' + key, value)
            self.logger.record('abplus/steam_ok', int(self.steam.poll()))
            self.steam.launch_failed(diagnostics.get('instance_start_failures', 0))
            episodes, self.rollout_episodes = self.rollout_episodes, []
            for task in sorted({e['task'] for e in episodes}):
                mine = [e for e in episodes if e['task'] == task]
                self.logger.record(f'task/{task}/episodes', len(mine))
                self.logger.record(f'task/{task}/win_rate', sum(e['outcome'] == 'win' for e in mine) / len(mine))
                self.logger.record(f'task/{task}/return', float(np.mean([e['r'] for e in mine])))
            parts = [e['reward_components'] for e in episodes if 'reward_components' in e]
            for key in (parts[0] if parts else ()):
                self.logger.record('reward/' + key, float(np.mean([c[key] for c in parts])))
            self.last_time, self.last_steps = now, steps

        def _on_step(self):
            # TrainingSession._on_step plus the task of each finished episode.
            with (self.out / 'episodes.jsonl').open('a', encoding='utf8') as f:
                for worker, (done, info) in enumerate(zip(self.locals['dones'], self.locals['infos'])):
                    if done:
                        self.completed += 1
                        start = {'start': info['episode_start']} if 'episode_start' in info else {}
                        parts = ({'reward_components': info['reward_components']}
                                 if 'reward_components' in info else {})
                        record = dict(episode=self.completed, worker=worker, seed=info['seed'],
                                      task=info.get('task'), outcome=info['outcome'], layout=info['layout'],
                                      frames=info['elapsed_frames'], **info['episode'], **start, **parts)
                        f.write(json.dumps(record) + '\n')
                        self.rollout_episodes.append(record)
            keep = self.completed < self.config['episodes']
            limit = self.config.get('game_hours')
            return keep and not (limit and game_hours(self.model.num_timesteps - self.start_timesteps) >= limit)

        def evaluate(self, model, checkpoint):
            count = self.config.get('eval_seeds_count', 0)
            if not count:
                return
            if not self.steam.poll():
                print(json.dumps(dict(event='evaluation_skipped', reason='steam client not running',
                                      checkpoint=checkpoint.name)), flush=True)
                return
            if self.evaluation is not None and self.evaluation.poll() is None:
                print(json.dumps(dict(event='evaluation_skipped', reason='previous evaluation still running',
                                      checkpoint=checkpoint.name)), flush=True)
                return
            out = self.out / 'evaluations' / checkpoint.name
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', str(HERE / 'abplus_eval.py'), '--checkpoint', str(checkpoint),
                   '--out', str(out), '--instances', str(self.config['eval_instances']), '--device', 'cpu',
                   '--name', 'ev', '--port', str(self.config['port'] - 200)]
            if self.config.get('tasks_spec') and self.config.get('eval_mode', 'mixture') == 'mixture':
                spec = self.out / 'tasks_spec.json'
                if not spec.exists():
                    spec.write_text(json.dumps(self.config['tasks_spec']), encoding='utf8')
                cmd += ['--tasks-file', str(spec), '--seeds', f"range:{self.config['eval_seed_start']}:{count}"]
            else:
                cmd += ['--seeds', self.config['eval_seeds_file'], '--limit', str(count)]
            log = (out / 'run.log').open('w', encoding='utf8')
            self.evaluation = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                               env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            log.close()
            print(json.dumps(dict(event='evaluation_started', checkpoint=checkpoint.name, out=str(out),
                                  pid=self.evaluation.pid)), flush=True)

        def finish(self, model):
            checkpoint = self.save(model, 'final')
            if self.evaluation is not None:
                self.evaluation.wait()  # the final checkpoint is always evaluated
            self.evaluate(model, checkpoint)
            (self.out / 'result.json').write_text(json.dumps(dict(
                completed_episodes=self.completed, timesteps=model.num_timesteps, updates=self.updates,
                game_hours_this_run=game_hours(model.num_timesteps - self.start_timesteps),
                checkpoint=str(checkpoint))), encoding='utf8')

    return AbplusSession


def resume(checkpoint, env, device):
    """Weights, optimizer, counters and RNG from a checkpoint; episodes restart fresh."""
    from isaac_bridge.gpu_ppo import GpuMaskablePPO
    from isaac_bridge.training_session import resolve_checkpoint, restore_rng
    checkpoint = resolve_checkpoint(checkpoint)
    state = json.loads((checkpoint / 'state.json').read_text())
    if state['config'].get('reward_profile') != env.reward_profile:
        raise ValueError(f"checkpoint reward {state['config'].get('reward_profile')} != {env.reward_profile}: "
                         'use --warm-start (weights only) to change the reward')
    with gzip.open(checkpoint / 'continuation.pt.gz', 'rb') as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    model = GpuMaskablePPO.load(checkpoint / 'model.zip', env=env, device=device, force_reset=True)
    restore_rng(data['rng'])
    return model, state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', action='store_true')
    p.add_argument('--envs', type=int, default=16, help='environment slots = worker processes (2 AB+ instances each)')
    p.add_argument('--chunks', type=int, default=1, help='policy calls per step; 1 batches every worker together')
    p.add_argument('--n-steps', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--n-epochs', type=int, default=2)
    p.add_argument('--micro-batch', type=int, default=256)
    p.add_argument('--segment-length', type=int, default=32)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--episodes', type=int, default=10 ** 9, help='stop after this many completed episodes')
    p.add_argument('--game-hours', type=float, default=None, help='stop after this much game time in this run')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mode', choices=('exact', 'skip', 'render'), default='exact')
    p.add_argument('--nice', type=int, default=0, help='niceness of workers and AB+ instances')
    p.add_argument('--name', default='tr', help='AB+ instance name prefix (lower case)')
    p.add_argument('--port', type=int, default=27400)
    p.add_argument('--boss-hp-prob', type=float, default=0.5)
    p.add_argument('--boss-hp-min', type=float, default=0.1)
    p.add_argument('--player-hp-prob', type=float, default=0.25)
    p.add_argument('--player-hp-min', type=int, default=3)
    p.add_argument('--checkpoint-every', type=int, default=10, help='completed PPO updates')
    p.add_argument('--eval-every', type=int, default=20, help='completed PPO updates')
    p.add_argument('--eval-seeds-file', default=str(Path.home() / 'isaac-abplus' / 'eval' / 'seeds-e2.json'))
    p.add_argument('--eval-seeds-count', type=int, default=64, help='0 disables periodic evaluation')
    p.add_argument('--eval-instances', type=int, default=2)
    p.add_argument('--eval-mode', choices=('mixture', 'arena'), default='mixture',
                   help='mixture: held-out seeds from --eval-seed-start through the task mixture; arena: E2 seeds file')
    p.add_argument('--eval-seed-start', type=int, default=2147490000,
                   help='held-out block (>= 2**31, after the E2 validation block, before the final test block)')
    p.add_argument('--warm-start', type=Path, help='weights only (e.g. a simulator checkpoint)')
    p.add_argument('--resume', type=Path, help='AB+ checkpoint directory or checkpoints/latest.json; use a new --out')
    p.add_argument('--tasks-file', default=str(HERE.parent / 'abplus' / 'catalog' / 'mixture_basement1.json'),
                   help='room mixture spec (weights, normal and boss room lists); none = Monstro arena only')
    p.add_argument('--task-weights', default=None, help='override, e.g. arena=0.2,normal=0.45,boss=0.35')
    p.add_argument('--json-obs', action='store_true', help='bridge v1 JSON observations instead of binary v2')
    p.add_argument('--recycle-episodes', type=int, default=200,
                   help='restart each AB+ process after this many episodes (the game leaks memory while it plays); 0 = never')
    p.add_argument('--reward-profile', choices=('combat-v2', 'combat-v1'), default='combat-v2',
                   help='combat-v2: AB+ mixture reward (abplus_reward.py); combat-v1: the simulator run reward')
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from isaac_bridge.abplus_reward import describe as describe_reward
    from isaac_bridge.gpu_env import validate_start_randomization
    from isaac_bridge.training_session import resolve_checkpoint, warm_start_model
    if args.resume and args.warm_start:
        p.error('--resume and --warm-start are mutually exclusive')
    if (args.micro_batch <= 0 or args.segment_length <= 0 or args.batch_size % args.micro_batch
            or args.micro_batch % args.segment_length or args.n_steps % args.segment_length):
        p.error('need segment-length | n-steps, segment-length | micro-batch and micro-batch | batch-size')
    if args.envs * args.n_steps % args.batch_size:
        p.error('batch-size must divide envs * n-steps')
    if not 0 <= args.seed < 2 ** 31 or args.seed + args.envs >= 2 ** 31:
        p.error('Training seeds must remain below 2**31 (held-out namespace)')
    start_randomization = validate_start_randomization(dict(
        boss_hp_prob=args.boss_hp_prob, boss_hp_min=args.boss_hp_min,
        player_hp_prob=args.player_hp_prob, player_hp_min=args.player_hp_min))
    tasks = None
    if args.tasks_file != 'none':
        tasks = json.loads(Path(args.tasks_file).read_text(encoding='utf8'))
        if args.task_weights:
            tasks['weights'] = {k: float(v) for k, v in (kv.split('=') for kv in args.task_weights.split(','))}
        tasks = {k: tasks[k] for k in ('weights', 'normal', 'boss')}
    if args.resume:
        args.resume = resolve_checkpoint(args.resume)
    if args.warm_start:
        args.warm_start = resolve_checkpoint(args.warm_start)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'out': str(args.out.resolve()), 'backend': 'abplus-1.06-linux',
              'task': ({'mixture': tasks['weights'], 'normal_rooms': len(tasks['normal']),
                        'boss_rooms': len(tasks['boss'])} if tasks else 'monstro-arena (sim seeds)'),
              'tasks_spec': tasks,
              'observation_transport': 'json (bridge v1)' if args.json_obs else 'binary (bridge v2, abp-0.2.1)',
              'reward_profile': args.reward_profile,
              'reward': (describe_reward() if args.reward_profile == 'combat-v2'
                         else 'combat-v1: legacy hurt/hit/damage/clear, win +2 + speed bonus, timeout -1'),
              'schema': DEADLINE_SCHEMA, 'history': 64, 'entity_capacity': 256,
              'timeout_semantics': 'termination', 'max_episode_seconds': 120, 'decisions_per_game_second': 15,
              'game_time': 'decisions * 2 logic frames / 30 frames per game second',
              'start_randomization': start_randomization, 'evaluation_start': 'full HP',
              'minibatch': 'segments: frame-deduplicated, advantages normalised per batch_size, gradient-accumulated',
              'instances_per_env': 2, 'resume_semantics': 'weights/optimizer/RNG; fresh episodes',
              'precision': 'strict FP32 (TF32 off)', 'eval_seeds': 'first eval_seeds_count of eval_seeds_file'}
    print(json.dumps(config, indent=2))
    if not args.train:
        print('PREPARED ONLY: no workers, AB+ instances, model or optimizer created.')
        return
    torch.set_num_threads(args.torch_threads)
    from isaac_bridge.abplus_vec import AbplusFrameVecEnv
    from isaac_bridge.gpu_buffer import GpuHistoryRolloutBuffer
    from isaac_bridge.gpu_ppo import GpuMaskablePPO
    args.out.mkdir(parents=True, exist_ok=False)
    config['sources_sha256'] = source_hashes()
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    env = AbplusFrameVecEnv(args.envs, args.seed, args.chunks, device=args.device, start_randomization=start_randomization,
                            mode=args.mode, name=args.name, port=args.port, nice=args.nice, tasks=tasks,
                            binary_obs=not args.json_obs, reward_profile=args.reward_profile,
                            recycle_episodes=args.recycle_episodes)
    try:
        completed = updates = 0
        if args.resume:
            model, state = resume(args.resume, env, args.device)
            completed, updates = state['completed_episodes'], state['updates']
            model.batch_size, model.n_epochs, model.ent_coef = args.batch_size, args.n_epochs, args.ent_coef
        else:
            model = GpuMaskablePPO(
                'MultiInputPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                gamma=args.gamma, ent_coef=args.ent_coef, learning_rate=args.learning_rate,
                rollout_buffer_class=GpuHistoryRolloutBuffer,
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                                   features_extractor_kwargs=dict(features_dim=256, layers=4, heads=8),
                                   net_arch=dict(pi=[256], vf=[256]), normalize_images=False),
                device=args.device, seed=args.seed, verbose=1)
        model.micro_batch_size, model.segment_length = args.micro_batch, args.segment_length
        if args.warm_start:
            migration = warm_start_model(model, args.warm_start)
            (args.out / 'migration.json').write_text(json.dumps(migration, indent=2), encoding='utf8')
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        env.training_seeds = True
        session = session_class()(config, args.out, completed, updates,
                                  start_timesteps=model.num_timesteps if args.resume else 0)
        model.session = session
        budget = (int(args.game_hours * 3600 * GAME_FPS / FRAMES_PER_DECISION) if args.game_hours
                  else (args.episodes - completed) * 1800) + args.envs
        model.learn(total_timesteps=budget, callback=session, reset_num_timesteps=not bool(args.resume))
        session.finish(model)
    finally:
        env.close()


if __name__ == '__main__':
    main()
