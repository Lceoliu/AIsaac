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

Collection sampler (--sampler): graph (default) replays each chunk's per-step inference as CUDA
graphs (isaac_bridge/graph_sampler.py, ~1.7 ms instead of ~10.8 ms per step for 16 environments)
and logs its self-check against the eager path as sampler/graph_*; eager is FrameSampler.

--async-train overlaps each PPO update with the next rollout (GpuMaskablePPO.async_training: an
actor copy collects while the learner trains on the previous rollout; one update of policy lag,
handled by the decoupled PPO objective, see isaac_bridge/gpu_ppo.py). The log adds
async/collect_s, async/update_s, async/join_wait_s (time the collector waited for the update),
train/lag_kl and train/lag_weight_truncated.

combat-v3 (default; user decisions 2026-09-25, abplus_reward.py): clearing the room is the goal
(timeout -20, death -25 plus the rest of the deadline's time cost, time priced by the enemies still
alive on a rising curve, -10 and -1/6 per second after any 20 s without a hit, rewards
per hit, per kill and per clear); its frames add the room state input (transformer_obs.COMBAT_FIELDS,
bridge abp-0.2.2). --start-bombs 0.5:3: the player starts with 0 bombs in half of the episodes and
1-3 otherwise. --room-sampling plr draws the rooms by Prioritized Level Replay (isaac_bridge/plr.py,
updated after every rollout from the GAE advantages; its state is saved with every checkpoint): the
room kinds keep the mixture's weights and PLR picks the room within each kind.
The log adds behavior/* (share of episodes with no hit by 60 s, first hit, share with 20 s or more
in one spot, cells visited, bomb use), win rates split by the start bombs, and plr/* (room shares and
effective number of rooms by kind, rooms seen, effective number of rooms). Each evaluation plays the
held-out seeds with the greedy policy and, with --eval-sampled, with the sampled one, records
--eval-replays episodes of each and builds replays.html (abplus_replay_view.py).

Checkpoints hold weights, optimizer, counters and RNG. A resume starts fresh episodes (the
in-progress rooms are not rebuilt). Periodic evaluation runs abplus_eval.py as a separate CPU
process on a few extra instances, so the learner keeps the GPU to itself.
"""
import argparse
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from isaac_bridge.transformer_obs import DEADLINE_SCHEMA
from isaac_bridge.transformer_policy import CombatTransformer, GeometryPolicy

GAME_FPS = 30
FRAMES_PER_DECISION = 2
HERE = Path(__file__).resolve().parent
SOURCES = ('train_abplus.py', 'abplus_eval.py', 'isaac_bridge/abplus.py', 'isaac_bridge/abplus_worker.py',
           'isaac_bridge/abplus_obs.py', 'isaac_bridge/abplus_tasks.py', 'isaac_bridge/abplus_reward.py',
           'isaac_bridge/abplus_vec.py', 'isaac_bridge/transformer_obs.py', 'isaac_bridge/transformer_policy.py',
           'isaac_bridge/gpu_ppo.py', 'isaac_bridge/gpu_buffer.py', 'isaac_bridge/gpu_env.py',
           'isaac_bridge/combat_reward.py', 'isaac_bridge/monstro_gym.py', 'isaac_bridge/env.py',
           'isaac_bridge/steam_watch.py', 'isaac_bridge/graph_sampler.py', 'isaac_bridge/plr.py',
           'isaac_bridge/abplus_geometry.py',
           'abplus_replay_view.py', 'isaac_bridge/abplus_replay.html')


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

        def __init__(self, config, out, completed=0, updates=0, start_timesteps=0, plr=None):
            super().__init__(config, out, completed, updates)
            self.start_timesteps = start_timesteps
            self.last_time = time.monotonic()
            self.last_steps = None
            self.evaluation = None
            self.rollout_episodes = []
            self.steam = SteamWatch(out)
            self.plr = plr              # plr.PrioritizedLevels when rooms are drawn by PLR
            self.level_ids = None       # room of every step of the current rollout (PLR scores)

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
                self.logger.record(f'task/{task}/death_rate', sum(e['outcome'] == 'death' for e in mine) / len(mine))
                self.logger.record(f'task/{task}/timeout_rate',
                                   sum(e['outcome'] == 'time_limit' for e in mine) / len(mine))
                for label, armed in (('0bombs', False), ('bombs', True)):
                    sub = [e for e in mine if 'bombs' in e.get('start', {}) and (e['start']['bombs'] > 0) == armed]
                    if sub:
                        self.logger.record(f'task/{task}/win_rate_{label}', sum(e['outcome'] == 'win' for e in sub) / len(sub))
            parts = [e['reward_components'] for e in episodes if 'reward_components' in e]
            for key in (parts[0] if parts else ()):
                self.logger.record('reward/' + key, float(np.mean([c[key] for c in parts])))
            stats = [e['stats'] for e in episodes if 'stats' in e]
            if stats:
                first = [st['first_hit_s'] for st in stats]
                self.logger.record('behavior/no_hit_by_60s', float(np.mean([f < 0 or f > 60 for f in first])))
                if any(f >= 0 for f in first):
                    self.logger.record('behavior/first_hit_s', float(np.mean([f for f in first if f >= 0])))
                self.logger.record('behavior/camp_20s', float(np.mean([st['longest_stationary_s'] >= 20 for st in stats])))
                self.logger.record('behavior/longest_no_hit_s', float(np.mean([st['longest_no_hit_s'] for st in stats])))
                self.logger.record('behavior/cells', float(np.mean([st['cells'] for st in stats])))
                armed = [e['stats'] for e in episodes if 'stats' in e and e.get('start', {}).get('bombs', 0) > 0]
                if armed:
                    self.logger.record('behavior/bomb_use', float(np.mean([st['bombs_used'] > 0 for st in armed])))
            if self.plr is not None and self.level_ids is not None:
                # PLR scores from this rollout's GAE (computed before on_rollout_end), then the new
                # room distribution for the workers' next episodes.
                b = self.model.rollout_buffer
                self.plr.update(b.advantages.cpu().numpy(), b.episode_starts.cpu().numpy(), self.level_ids)
                self.level_ids.fill(-1)
                probabilities = self.plr.probabilities()
                self.model.env.set_level_probabilities(probabilities)
                for key, value in self.plr.summary(probabilities).items():
                    self.logger.record('plr/' + key, value)
            sampler = self.model._gpu_sampler
            if getattr(sampler, 'diffs', None) is not None:
                for key, value in sampler.diffs.items():
                    self.logger.record('sampler/graph_' + key, value)
                sampler.diffs = dict.fromkeys(sampler.diffs, 0.0)
            self.last_time, self.last_steps = now, steps

        def _on_step(self):
            # TrainingSession._on_step plus the task of each finished episode (the file is opened
            # only on steps where an episode ended).
            dones = self.locals['dones']
            if self.plr is not None:
                if self.level_ids is None:
                    self.level_ids = np.full((self.model.n_steps, self.model.n_envs), -1, np.int64)
                self.level_ids[self.locals['t']] = [info.get('level', -1) for info in self.locals['infos']]
            if dones.any():
                with (self.out / 'episodes.jsonl').open('a', encoding='utf8') as f:
                    for worker, (done, info) in enumerate(zip(dones, self.locals['infos'])):
                        if not done:
                            continue
                        self.completed += 1
                        start = {'start': info['episode_start']} if 'episode_start' in info else {}
                        parts = ({'reward_components': info['reward_components']}
                                 if 'reward_components' in info else {})
                        if 'episode_stats' in info:
                            parts['stats'] = info['episode_stats']
                        record = dict(episode=self.completed, worker=worker, seed=info['seed'],
                                      task=info.get('task'), outcome=info['outcome'], layout=info['layout'],
                                      level=info.get('level', -1), frames=info['elapsed_frames'],
                                      truncated=bool(info.get('TimeLimit.truncated')), **info['episode'],
                                      **start, **parts)
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
            replays = int(self.config.get('eval_replays', 0))
            cmd = [sys.executable, '-u', str(HERE / 'abplus_eval.py'), '--checkpoint', str(checkpoint),
                   '--instances', str(self.config['eval_instances']), '--device', 'cpu',
                   '--name', 'ev', '--port', str(self.config['port'] - 200), '--replays', str(replays)]
            if self.config.get('tasks_spec') and self.config.get('eval_mode', 'mixture') == 'mixture':
                spec = self.out / 'tasks_spec.json'
                if not spec.exists():
                    spec.write_text(json.dumps(self.config['tasks_spec']), encoding='utf8')
                cmd += ['--tasks-file', str(spec), '--seeds', f"range:{self.config['eval_seed_start']}:{count}"]
            else:
                cmd += ['--seeds', self.config['eval_seeds_file'], '--limit', str(count)]
            # Greedy first (comparable across runs), then the sampled policy (as it trains), then one
            # replay page of both; the chain is one process for the "previous evaluation" check.
            runs = [cmd + ['--out', str(out)]]
            labels = [f'{out}=最优']
            if self.config.get('eval_sampled'):
                runs.append(cmd + ['--out', str(out / 'sampled'), '--stochastic', '--sample-seed', '0'])
                labels.append(f"{out / 'sampled'}=采样")
            if replays:
                runs.append([sys.executable, str(HERE / 'abplus_replay_view.py'), str(out / 'replays.html'), *labels,
                             '--title', f'AB+ 评估回放 · {self.out.name} · {checkpoint.name}'])
            log = (out / 'run.log').open('w', encoding='utf8')
            self.evaluation = subprocess.Popen(['bash', '-c', '; '.join(shlex.join(c) for c in runs)],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                               env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            log.close()
            print(json.dumps(dict(event='evaluation_started', checkpoint=checkpoint.name, out=str(out),
                                  pid=self.evaluation.pid)), flush=True)

        def save(self, model, phase):
            checkpoint = super().save(model, phase)
            if self.plr is not None:
                (checkpoint / 'plr.json').write_text(json.dumps(self.plr.state_dict()), encoding='utf8')
            return checkpoint

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
    p.add_argument('--ent-coef', type=float, default=0.003)
    p.add_argument('--ent-coef-heads', default=None,
                   help='per-head entropy weights, e.g. move=0.005,shoot=0.002 (others keep --ent-coef); factored heads only')
    p.add_argument('--gamma', type=float, default=0.9995)
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
    p.add_argument('--sampler', choices=('graph', 'eager'), default='graph',
                   help='graph: per-step inference replayed as CUDA graphs (graph_sampler.py); eager: FrameSampler')
    p.add_argument('--graph-check-every', type=int, default=1024,
                   help='graph sampler steps between self-checks against the eager path (0 = never)')
    p.add_argument('--async-train', action='store_true',
                   help='train on the previous rollout while collecting the next (one update of policy lag)')
    p.add_argument('--start-bombs', default='0.5:3',
                   help='ZERO_PROB:MAX: start with 0 bombs with this probability, else 1..MAX; none = the game\'s 1')
    p.add_argument('--room-sampling', choices=('plr', 'mixture'), default='plr',
                   help='plr: Prioritized Level Replay over the rooms of --tasks-file; mixture: its fixed weights by seed')
    p.add_argument('--plr-beta', type=float, default=1.0, help='rank temperature (P ~ (1/rank)^(1/beta))')
    p.add_argument('--plr-staleness', type=float, default=0.3)
    p.add_argument('--plr-floor', type=float, default=0.1, help='uniform share over the rooms of each kind')
    p.add_argument('--eval-sampled', action=argparse.BooleanOptionalAction, default=True,
                   help='also evaluate the sampled policy (the greedy one always runs)')
    p.add_argument('--eval-replays', type=int, default=8, help='replays recorded per evaluation mode')
    p.add_argument('--recycle-episodes', type=int, default=200,
                   help='restart each AB+ process after this many episodes (the game leaks memory while it plays); 0 = never')
    p.add_argument('--reward-profile', choices=('combat-v5', 'combat-v4', 'combat-v3', 'combat-v2', 'combat-v1'),
                   default='combat-v5',
                   help='combat-v5: stage one revised (roster-lineage hits, alignment potential, factored heads, '
                        'auxiliary geometry head); combat-v4: stage one, learn to clear (no penalties, clear bonus, timeout truncates); '
                        'combat-v3: clear-first with penalties; combat-v2: score-calibrated (abplus_reward.py); '
                        'combat-v1: the simulator run reward')
    p.add_argument('--align-coef', type=float, default=None, help='combat-v5 alignment potential weight (default 0.2)')
    p.add_argument('--hit-hp', type=float, default=None, help='combat-v5: HP per +1 of hit reward (default 5)')
    p.add_argument('--lineage-mode', type=int, choices=(0, 1, 2), default=None,
                   help='combat-v5 death successors in the lineage: 0 none, 1 all (default), 2 a single one')
    p.add_argument('--aux-coef', type=float, default=0.2, help='combat-v5 auxiliary geometry head loss weight')
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from isaac_bridge.abplus_reward import V5, describe as describe_reward, describe_v3, describe_v4, describe_v5
    from isaac_bridge.abplus_worker import observation_options
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
    start_bombs = None
    if args.start_bombs != 'none':
        zero, most = args.start_bombs.split(':')
        start_bombs = {'zero_prob': float(zero), 'max': int(most)}
        if not (0 <= start_bombs['zero_prob'] <= 1 and 1 <= start_bombs['max'] <= 99):
            p.error('--start-bombs ZERO_PROB:MAX with 0 <= ZERO_PROB <= 1 and 1 <= MAX')
    v5 = args.reward_profile == 'combat-v5'
    # combat-v5's potential must discount with the learner's gamma to stay potential-based.
    reward_options = (dict(hit_hp=args.hit_hp if args.hit_hp is not None else V5['hit_hp'],
                           align=args.align_coef if args.align_coef is not None else V5['align'], gamma=args.gamma)
                      if v5 else {})
    lineage_mode = (args.lineage_mode if args.lineage_mode is not None else V5['lineage_mode']) if v5 else None
    if not v5 and (args.align_coef is not None or args.hit_hp is not None or args.lineage_mode is not None or args.ent_coef_heads):
        p.error('--align-coef/--hit-hp/--lineage-mode/--ent-coef-heads are combat-v5 options')
    head_names = ('move', 'shoot', 'bomb', 'item')
    head_ent_coefs = None
    if args.ent_coef_heads:
        given = {k: float(v) for k, v in (kv.split('=') for kv in args.ent_coef_heads.split(','))}
        if set(given) - set(head_names):
            p.error(f'--ent-coef-heads names are {head_names}')
        head_ent_coefs = [given.get(k, args.ent_coef) for k in head_names]
    plr_levels = None
    if args.room_sampling == 'plr':
        if not tasks:
            p.error('--room-sampling plr needs a --tasks-file')
        from isaac_bridge.plr import mixture_levels
        plr_levels = mixture_levels(tasks)
    if args.resume:
        args.resume = resolve_checkpoint(args.resume)
    if args.warm_start:
        args.warm_start = resolve_checkpoint(args.warm_start)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'out': str(args.out.resolve()), 'backend': 'abplus-1.06-linux',
              'task': ({'mixture': tasks['weights'], 'normal_rooms': len(tasks['normal']),
                        'boss_rooms': len(tasks['boss'])} if tasks else 'monstro-arena (sim seeds)'),
              'tasks_spec': tasks,
              'observation_transport': 'json (bridge v1)' if args.json_obs else 'binary (bridge v2, abp-0.2.2)',
              'start_bombs': start_bombs,
              'room_sampling': (dict(method='plr', levels=len(plr_levels), beta=args.plr_beta,
                                     staleness=args.plr_staleness, floor=args.plr_floor,
                                     score='positive value loss (mean max(GAE, 0) per episode)',
                                     kinds='each kind keeps its mixture weight; PLR picks the room within the kind',
                                     kind_weights=tasks['weights'])
                                if plr_levels else 'mixture weights by seed'),
              'reward_profile': args.reward_profile, 'collection_sampler': args.sampler,
              'update_schedule': ('asynchronous: each update trains while the next rollout is collected by the '
                                  'weights it started from (one update of policy lag); decoupled PPO objective: '
                                  'ratio clipped against the update start, samples weighted by '
                                  'pi_start/pi_behaviour truncated at 2' if args.async_train else 'synchronous'),
              'reward': (describe_v5(**reward_options) if v5 else describe_v4() if args.reward_profile == 'combat-v4'
                         else describe_v3()
                         if args.reward_profile == 'combat-v3' else describe_reward()
                         if args.reward_profile == 'combat-v2'
                         else 'combat-v1: legacy hurt/hit/damage/clear, win +2 + speed bonus, timeout -1'),
              'observation': {**observation_options(args.reward_profile),
                              'combat_fields': list(observation_options(args.reward_profile)['combat_state'] or ())},
              'schema': DEADLINE_SCHEMA, 'history': 64, 'entity_capacity': 256,
              'timeout_semantics': 'truncation (bootstrap)' if args.reward_profile in ('combat-v4', 'combat-v5') else 'termination',
              'reward_options': reward_options, 'lineage_mode': lineage_mode,
              'model': (dict(policy='GeometryPolicy', heads=dict(zip(head_names, (9, 5, 2, 2))),
                             entropy=dict(zip(head_names, head_ent_coefs)) if head_ent_coefs else f'{args.ent_coef:g} on the summed entropy',
                             aux_head='5-way aim label + d_fire/40 regression from the actor features, loss weight '
                                      f'{args.aux_coef:g}',
                             critic_input='fire_distance = d_fire/40, value branch only')
                        if v5 else 'MultiInputPolicy, joint 45-way move x shoot head'),
              'max_episode_seconds': 120, 'decisions_per_game_second': 15,
              'game_time': 'decisions * 2 logic frames / 30 frames per game second',
              'start_randomization': start_randomization, 'evaluation_start': 'full HP',
              'minibatch': 'segments: frame-deduplicated, advantages normalised per batch_size, gradient-accumulated',
              'instances_per_env': 2, 'resume_semantics': 'weights/optimizer/RNG; fresh episodes',
              'precision': 'strict FP32 (TF32 off)', 'eval_seeds': 'first eval_seeds_count of eval_seeds_file',
              'evaluation_bombs': 'the training start-bomb draw, fixed per seed' if start_bombs else 'the game\'s 1'}
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
                            recycle_episodes=args.recycle_episodes, start_bombs=start_bombs, plr_levels=plr_levels,
                            reward_options=reward_options, lineage_mode=lineage_mode)
    try:
        completed = updates = 0
        if args.resume:
            model, state = resume(args.resume, env, args.device)
            completed, updates = state['completed_episodes'], state['updates']
            model.batch_size, model.n_epochs, model.ent_coef = args.batch_size, args.n_epochs, args.ent_coef
            if v5 != hasattr(model.policy, 'aux_outputs'):
                raise ValueError('--reward-profile combat-v5 and the checkpoint policy (GeometryPolicy) must go together')
        else:
            model = GpuMaskablePPO(
                GeometryPolicy if v5 else 'MultiInputPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma, ent_coef=args.ent_coef, learning_rate=args.learning_rate,
                rollout_buffer_class=GpuHistoryRolloutBuffer,
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                                   features_extractor_kwargs=dict(features_dim=256, layers=4, heads=8),
                                   net_arch=dict(pi=[256], vf=[256]), normalize_images=False),
                device=args.device, seed=args.seed, verbose=1)
        model.micro_batch_size, model.segment_length = args.micro_batch, args.segment_length
        model.aux_coef = args.aux_coef if v5 else 0.0
        model.head_ent_coefs = head_ent_coefs
        model.async_training = args.async_train
        if args.sampler == 'graph':
            from functools import partial
            from isaac_bridge.graph_sampler import GraphFrameSampler
            model.sampler_class = partial(GraphFrameSampler, check_every=args.graph_check_every)
        if args.warm_start:
            migration = warm_start_model(model, args.warm_start)
            (args.out / 'migration.json').write_text(json.dumps(migration, indent=2), encoding='utf8')
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        env.training_seeds = True
        plr = None
        if plr_levels:
            from isaac_bridge.plr import PrioritizedLevels
            saved = args.resume / 'plr.json' if args.resume else None
            if saved is not None and saved.exists():
                # The kind weights come from --tasks-file / --task-weights, not from the checkpoint.
                plr = PrioritizedLevels.from_state(json.loads(saved.read_text(encoding='utf8')), tasks['weights'])
                if plr.levels != [tuple(level) for level in plr_levels]:
                    raise ValueError('PLR rooms of the checkpoint differ from --tasks-file')
            else:
                plr = PrioritizedLevels(plr_levels, tasks['weights'], args.plr_beta, args.plr_staleness, args.plr_floor)
            env.set_level_probabilities(plr.probabilities())
        session = session_class()(config, args.out, completed, updates,
                                  start_timesteps=model.num_timesteps if args.resume else 0, plr=plr)
        model.session = session
        budget = (int(args.game_hours * 3600 * GAME_FPS / FRAMES_PER_DECISION) if args.game_hours
                  else (args.episodes - completed) * 1800) + args.envs
        model.learn(total_timesteps=budget, callback=session, reset_num_timesteps=not bool(args.resume))
        session.finish(model)
    finally:
        env.close()


if __name__ == '__main__':
    main()
