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

Duel (--duel-file, isaac_bridge/abplus_duel.py): each seed is a duel under the game rules (a death or the deadline ends
it; the first hit, the training outcome under --hurt-ends-episode, is recorded). --opponent self: the checkpoint plays
both sides, one episode per seed; scripted: against abplus_duel.ScriptedDuellist, or <checkpoint dir>: against that
policy, two episodes per seed (the checkpoint as the player, then as the NPC). Results carry policy_side, first_hit,
winner and per-side counters; replays carry both sides' actions.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import math
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, sim_arena, stop_abplus
from isaac_bridge.abplus_geometry import blocked_moves
from isaac_bridge.abplus_reward import REWARDS, CombatV2
from isaac_bridge.abplus_worker import EPISODE_STATS, EpisodeStats, observation_options, sample_bombs
from isaac_bridge.abplus_tasks import TaskSampler
from isaac_bridge.steam_watch import steam_running
from isaac_bridge.transformer_obs import factored_masks, factored_to_joint


BOSS_MAX_HP = 250.0  # Monstro; the simulator audit reports absolute boss HP
TYPE_MONSTRO = 20


def rss_mib(pid):
    """Resident memory of a process in MiB (0 when it cannot be read)."""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


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


class Comparison:
    """--compare (goal-conditioned M0, EXPERIMENTS.md A9): other checkpoints fed the acting policy's frames and masks;
    per action head KL(other || acting) and whether the two greedy actions agree, summed over the episode's decisions."""

    def __init__(self, label, cached):
        self.label, self.cached = label, cached
        self.reset()

    def reset(self):
        self.cached.reset()
        self.kl, self.agree, self.steps = None, None, 0
        self.kl_max = 0.0

    @torch.inference_mode()
    def step(self, frame, mask, dist):
        other = self.cached.distribution(frame, mask)
        kl, agree = [], []
        for mine, theirs in zip(dist.distributions, other.distributions):
            lp, lq = torch.log_softmax(mine.logits, -1), torch.log_softmax(theirs.logits, -1)
            kl.append(float((lq.exp() * (lq - lp)).sum()))
            agree.append(float(mine.logits.argmax(-1).item() == theirs.logits.argmax(-1).item()))
        if self.kl is None:
            self.kl, self.agree = [0.0] * len(kl), [0.0] * len(agree)
        self.kl = [a + b for a, b in zip(self.kl, kl)]
        self.agree = [a + b for a, b in zip(self.agree, agree)]
        self.kl_max = max(self.kl_max, sum(kl))
        self.steps += 1

    def summary(self):
        n = max(1, self.steps)
        return dict(steps=self.steps, kl=[round(v / n, 6) for v in self.kl or []], kl_sum=round(sum(self.kl or []) / n, 6),
                    kl_max=round(self.kl_max, 6), agree=[round(v / n, 4) for v in self.agree or []])


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


def run_episode(env, cached, seed, args, replay_path=None, config=None):
    arena = sim_arena(seed)
    config = config or {}
    if args.stochastic:
        torch.manual_seed(args.sample_seed + seed)
    # Runs trained with a random bomb start are evaluated with the same draw, fixed per seed, unless --bombs fixes the
    # start (C41: the game's 1 bomb, as the base stats and full HP).
    bombs = (int(args.bombs) if getattr(args, 'bombs', None) is not None
             else sample_bombs(seed, config['start_bombs']) if config.get('start_bombs') else None)
    obs, info = env.reset(options={'arena_seed': seed, **({'bombs': bombs} if bombs is not None else {})})
    env.bridge.last_reset = dict(info)
    task = env.bridge.last_reset.get('task', 'arena')
    reward_v2 = CombatV2()  # reported for every checkpoint, whatever reward it was trained with
    reward_v2.reset(env.raw_obs, task)
    # The checkpoint's own reward (combat-v3/v4) next to combat-v2, which every checkpoint reports.
    profile = config.get('reward_profile')
    own = (REWARDS[profile](**config.get('reward_options', {})) if profile in REWARDS and profile != 'combat-v2'
           else None)
    # combat-v5 / combat-hitrate policies have factored heads (move, shoot, bomb, item); the env takes
    # the joint layout.
    factored = len(cached.policy.action_space.nvec) == 4
    if own is not None:
        own.reset(env.raw_obs, task)
    stats = EpisodeStats(env.raw_obs)
    frames = 0
    cached.reset()
    compare = getattr(cached, 'compare', ())
    for c in compare:
        c.reset()
    metrics = Metrics(env.raw_obs)
    writer = gzip.open(replay_path, 'wt', encoding='utf8') if replay_path else None
    if writer:
        writer.write(json.dumps({'metadata': {'seed': seed, 'arena': arena, 'format': 'abplus-raw-obs-v1'}}) + '\n')
        writer.write(json.dumps({'action': None, 'obs': env.raw_obs}, separators=(',', ':')) + '\n')
    verify, total_return, steps, t0 = [], 0.0, 0, time.monotonic()
    trajectory = hashlib.sha256()
    policy_s = 0.0
    # C30: runs trained with --block-moves are evaluated with the same mask (--block-moves forces it).
    block = factored and (bool(config.get('block_moves')) or bool(getattr(args, 'block_moves', False)))
    try:
        while True:
            mask = factored_masks(env.action_masks()) if factored else env.action_masks()
            if block:
                mask[:9] &= ~np.asarray(blocked_moves(env.raw_obs), bool)
            tp = time.perf_counter()
            dist = cached.distribution(env.history.frames[-1], mask)
            for c in compare:
                c.step(env.history.frames[-1], mask, dist)
            action = dist.get_actions(deterministic=not args.stochastic)[0].cpu().numpy()
            if factored:
                action = factored_to_joint(action)
            if steps < args.verify_steps:
                full = cached.full_distribution(obs, mask)
                verify.append(float((logits(dist) - logits(full)).abs().max()))
            policy_s += time.perf_counter() - tp
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            steps += 1
            reward_v2.step(env.raw_obs, info['outcome'], info['elapsed_frames'] - frames)
            if own is not None:
                own.step(env.raw_obs, info['outcome'], info['elapsed_frames'] - frames)
            stats.step(env.raw_obs, info['elapsed_frames'] - frames)
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
                  reward_v2_start=reward_v2.start, start_bombs=env.bridge.start_bombs,
                  stats={k: round(float(v), 2) for k, v in zip(EPISODE_STATS, stats.array())})
    if own is not None:
        tag = 'reward_' + profile[len('combat-'):].replace('-', '_')
        result.update({tag: round(sum(own.totals.values()), 4),
                       tag + '_components': {k: round(v, 4) for k, v in own.totals.items()}})
    if verify:
        result['verify_max_logit_diff'] = max(verify)
    if compare:
        result['compare'] = {c.label: c.summary() for c in compare}
    return result


DUEL_SIDES = ('player', 'npc')


def duel_mask():
    """Factored masks of a duel side: no bombs, no active item (both views)."""
    return factored_masks(np.asarray([True] * 45 + [True, False] + [True, False], bool))


def run_duel_episode(env, controllers, seed, args, replay_path=None, config=None, labels=('policy', 'policy'),
                     policy_side='both'):
    """One duel under the game rules (a death or the deadline ends it; the first hit is recorded, the training outcome
    under --hurt-ends-episode). controllers: per side (player, NPC) ('policy', CachedPolicy) or ('script',
    ScriptedDuellist)."""
    config = config or {}
    if args.stochastic:
        torch.manual_seed(args.sample_seed + seed)
    frames, reset_info = env.reset(seed)
    for kind, c in controllers:
        c.reset()
    profile = config.get('reward_profile')
    rewards = [REWARDS[profile](**config.get('reward_options', {})) if profile in REWARDS else None for _ in DUEL_SIDES]
    for r, v in zip(rewards, env.views):
        if r is not None:
            r.reset(v, 'normal')
    stats = [EpisodeStats(v) for v in env.views]
    writer = gzip.open(replay_path, 'wt', encoding='utf8') if replay_path else None
    if writer:
        writer.write(json.dumps({'metadata': {'seed': seed, 'format': 'abplus-raw-obs-v1', 'duel': True, 'sides': list(labels),
                                              'policy_side': policy_side}}) + '\n')
        writer.write(json.dumps({'action': None, 'duel_action': None, 'obs': env.raw_obs}, separators=(',', ':')) + '\n')
    first_hit, first_hit_s, steps, frames_played = None, None, 0, 0
    trajectory = hashlib.sha256()
    # C30: runs trained with --block-moves are evaluated with the same mask, per side from its own view
    block = bool(config.get('block_moves')) or bool(getattr(args, 'block_moves', False))
    t0, policy_s = time.monotonic(), 0.0
    try:
        while True:
            actions = []
            tp = time.perf_counter()
            for side, (kind, c) in enumerate(controllers):
                if kind == 'policy':
                    mask = duel_mask()
                    if block:
                        mask[:9] &= ~np.asarray(blocked_moves(env.views[side]), bool)
                    dist = c.distribution(env.histories[side].frames[-1], mask)
                    a = dist.get_actions(deterministic=not args.stochastic)[0].cpu().numpy()
                    actions.append((int(a[0]), int(a[1])))
                else:
                    actions.append(c(env.views[side]))
            policy_s += time.perf_counter() - tp
            before = [v['combat']['player_damage_events'] for v in env.views]
            frames, outcomes, terminated, truncated, info = env.step([a[0] for a in actions], [a[1] for a in actions])
            steps += 1
            advanced = info['elapsed_frames'] - frames_played
            frames_played = info['elapsed_frames']
            for side in range(2):
                view = env.views[side]
                if rewards[side] is not None:
                    rewards[side].step(view, outcomes[side], advanced)
                stats[side].step(view, advanced)
            hurt = [v['combat']['player_damage_events'] > b for v, b in zip(env.views, before)]
            if first_hit is None and any(hurt):
                # the side whose shot landed first (both: the same step)
                first_hit = 'both' if all(hurt) else DUEL_SIDES[1 - hurt.index(True)]
                first_hit_s = frames_played / 30
            raw = env.raw_obs
            trajectory.update(json.dumps([actions, raw['players'][0]['pos'], raw['duel']['pos'], raw['duel']['hp'],
                                          raw['players'][0]['hearts']]).encode())
            if writer:
                writer.write(json.dumps({'action': [actions[0][0] * 5 + actions[0][1], 0, 0],
                                         'duel_action': [actions[1][0] * 5 + actions[1][1], 0, 0], 'obs': raw},
                                        separators=(',', ':')) + '\n')
            if terminated or truncated:
                break
    finally:
        if writer:
            writer.close()
    d = env.raw_obs['duel']
    sides = {}
    for side, key in enumerate(('player', 'npc_side')):
        c = d[key]
        sides[DUEL_SIDES[side]] = dict(
            controller=labels[side], outcome=outcomes[side], shots=c['shots'], hits=c['hits'], misses=c['misses'],
            hit_rate=round(c['hits'] / c['shots'], 4) if c['shots'] else None, hurt=c['hurt'],
            stats={k: round(float(v), 2) for k, v in zip(EPISODE_STATS, stats[side].array())},
            **({'reward': round(sum(rewards[side].totals.values()), 4),
                'reward_components': {k: round(v, 4) for k, v in rewards[side].totals.items()}} if rewards[side] else {}))
    winner = ('player' if outcomes == ['win', 'death'] else 'npc' if outcomes == ['death', 'win']
              else 'both_dead' if outcomes == ['death', 'death'] else 'none')
    # the checkpoint's view of the episode: its own outcome and first hit (self-play: the player side's)
    mine = 0 if policy_side in ('player', 'both') else 1
    result = dict(seed=seed, task='duel', policy_side=policy_side, sides_played=list(labels),
                  outcome=outcomes[mine], first_hit=first_hit, first_hit_s=first_hit_s,
                  first_hit_won=None if first_hit in (None, 'both') else first_hit == DUEL_SIDES[mine],
                  winner=winner, layout=reset_info.get('room_variant'), arm=reset_info.get('duel_arm'),
                  cells=reset_info.get('duel_cells'), frames=frames_played, l=steps, sides=sides,
                  seconds=round(time.monotonic() - t0, 2), policy_ms=round(1000 * policy_s / max(1, steps), 3),
                  trajectory=trajectory.hexdigest()[:16])
    if rewards[mine] is not None:
        tag = 'reward_' + profile[len('combat-'):].replace('-', '_')
        result.update({tag: sides[DUEL_SIDES[mine]]['reward'], tag + '_components': sides[DUEL_SIDES[mine]]['reward_components']})
    return result


def worker(index, args, config, tasks, results):
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.allow_tf32 = False  # GpuMaskablePPO evaluates in strict FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    name, port = f'{args.name}{index}', args.port + index
    policy = load_policy(args.checkpoint, args.device)
    cached = CachedPolicy(policy, config['history'], deterministic=not args.stochastic)
    cached.compare = [Comparison(Path(c).name, CachedPolicy(load_policy(c, args.device), config['history'], True))
                      for c in (args.compare.split(',') if args.compare else ())]
    for c in cached.compare:
        if c.cached.policy.observation_space != policy.observation_space:
            raise ValueError(f'--compare {c.label}: observation space differs from the checkpoint')
    state = {'proc': None, 'env': None}
    duel = json.loads(Path(args.duel_file).read_text(encoding='utf8')) if args.duel_file else None
    if duel:
        from isaac_bridge.abplus_duel import DuelEnv, ScriptedDuellist, duel_tasks
        # a second cache of the same weights for the other side, or the opponent checkpoint's policy
        other = None
        if args.opponent == 'self':
            other = ('policy', CachedPolicy(policy, config['history'], deterministic=not args.stochastic))
        elif args.opponent != 'scripted':
            other = ('policy', CachedPolicy(load_policy(args.opponent, args.device), config['history'],
                                            deterministic=not args.stochastic))

    def start_duel():
        state['proc'] = launch_abplus(name, port, args.mode)
        env = DuelEnv(port, int(config['max_episode_seconds'] * 30), duel['duel'], duel_tasks(duel), hurt_ends=False,
                      lineage_mode=int(config.get('lineage_mode') or 3), miss_cap=int(config.get('miss_cap', 20)),
                      binary_obs=not args.json_obs, **observation_options(config.get('reward_profile')),
                      history=config['history'], entity_capacity=config['entity_capacity'])
        if env.observation_space != policy.observation_space:
            raise ValueError('duel observation space differs from the checkpoint policy')
        state['env'] = env

    def start():
        if duel:
            return start_duel()
        state['proc'] = launch_abplus(name, port, args.mode)
        # C39: the checkpoint's frames per decision, and remaining_time over this evaluation's deadline.
        env = AbplusTransformerEnv(port=port, max_episode_frames=int(config['max_episode_seconds'] * 30),
                                   **observation_options(config.get('reward_profile')),
                                   deadline_s=float(config['max_episode_seconds']),
                                   frames_per_decision=int(config.get('frames_per_decision', 2)),
                                   history=config['history'], entity_capacity=config['entity_capacity'])
        env.bridge.binary_obs = not args.json_obs
        if config.get('lineage_mode') is not None:
            env.bridge.lineage_mode = int(config['lineage_mode'])
        # The hit-rate test trains and evaluates an invincible player (bridge abp-0.2.4); --mortal evaluates
        # such a checkpoint with damage on (the stage-2 baseline, C33).
        env.bridge.invincible = bool(config.get('invincible', False)) and not args.mortal
        # combat-hitrate-miss: the per-miss penalty's cap (bridge abp-0.2.6; 0 = none).
        env.bridge.miss_cap = int(config.get('miss_cap', 0))
        # The single-enemy aiming arena (C22): the run's tasks file carries the target; its arms (tier 6,
        # C28) choose the normal rooms, so the sampler takes them from the target in use.
        env.bridge.target = config.get('target')
        if args.tasks_file != 'none':
            spec = json.loads(Path(args.tasks_file).read_text(encoding='utf8'))
            env.bridge.tasks = TaskSampler(spec['weights'], spec['normal'], spec['boss'],
                                           (config.get('target') or {}).get('arms'))
        else:
            env.bridge.tasks = None
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
                seed, replay, side = tasks.get_nowait()
            except queue.Empty:
                break
            for attempt in range(2):
                try:
                    if duel:
                        me = ('policy', cached)
                        rival = other if other is not None else ('script', ScriptedDuellist())
                        rival_label = 'self' if args.opponent == 'self' else args.opponent
                        pair = (me, rival) if side in ('player', 'both') else (rival, me)
                        labels = ('checkpoint', rival_label) if side in ('player', 'both') else (rival_label, 'checkpoint')
                        result = run_duel_episode(state['env'], pair, seed, args, replay, config, labels, side)
                    else:
                        result = run_episode(state['env'], cached, seed, args, replay, config)
                    break
                except Exception:
                    result = dict(seed=seed, outcome='error', layout=0 if duel else sim_arena(seed)['variant'],
                                  error=traceback.format_exc()[-3000:], attempt=attempt, policy_side=side)
                    stop()
                    time.sleep(2)
                    start()
            result['worker'] = index
            # C39 t3r: an instance over the memory cap is restarted before its next episode.
            if args.recycle_rss_mib > 0 and state['proc'] is not None and \
                    rss_mib(state['proc'].pid) >= args.recycle_rss_mib:
                stop()
                time.sleep(2)
                start()
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
    p.add_argument('--bombs', type=int, default=None,
                   help='bombs at the start of every episode (default: the run start-bomb draw fixed per seed, or '
                        'the game default 1 without one; C41: 1)')
    p.add_argument('--recycle-rss-mib', type=float, default=0,
                   help='restart an instance between episodes once its resident memory reaches this many MiB '
                        '(C39 t3r); 0 = never')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--verify-steps', type=int, default=0, help='compare cached vs full forward for N steps/episode')
    p.add_argument('--compare', default='',
                   help='comma-separated checkpoint directories fed the same frames: per head KL(other || this) and greedy '
                        'agreement per episode (goal-conditioned M0 drift reference)')
    p.add_argument('--replays', type=int, default=0, help='write raw-observation replays for the first N seeds')
    p.add_argument('--repeat', type=int, default=1, help='episodes per seed (reproducibility check)')
    p.add_argument('--tasks-file', default='none', help="room mixture spec (abplus_tasks); 'none' = Monstro arena")
    p.add_argument('--target-from-tasks', action='store_true',
                   help="the tasks file's target arena replaces the checkpoint's (another tier's seeds, C27)")
    p.add_argument('--block-moves', action='store_true',
                   help='mask the moves the terrain stops dead even if the checkpoint was trained without (C30)')
    p.add_argument('--mortal', action='store_true',
                   help='the player takes damage even if the checkpoint was trained invincible (C33)')
    p.add_argument('--episode-seconds', type=float, default=0,
                   help="the episode deadline instead of the checkpoint's, also for a death's cost (another task, C35)")
    p.add_argument('--json-obs', action='store_true', help='bridge v1 JSON observations instead of binary v2')
    p.add_argument('--port', type=int, default=27200)
    p.add_argument('--software-gl', action='store_true',
                   help="the AB+ instances use Mesa's software OpenGL (no GPU memory, identical trajectories; B7)")
    p.add_argument('--name', default='beval')
    p.add_argument('--duel-file', default=None, help='duel spec (catalog/duel_rooms.json): duel evaluation')
    p.add_argument('--opponent', default='self',
                   help="duel: self (the checkpoint on both sides), scripted (abplus_duel.ScriptedDuellist) or an opponent "
                        "checkpoint directory")
    args = p.parse_args()
    try:
        # An evaluation is the first process the kernel kills when memory runs out (C39 t3 died of an OOM while one ran):
        # its workers and AB+ instances inherit this.
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('1000')
    except OSError:
        pass
    if args.device == 'cpu':
        # The workers (spawned) then open no CUDA context: ~270 MiB of the learner's GPU each otherwise (B7).
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    if args.software_gl:
        from isaac_bridge.abplus import SOFTWARE_GL_ENV
        os.environ.update(SOFTWARE_GL_ENV)
    if not steam_running():
        # Every AB+ start needs the Steam client; without it the game only launches steam.sh and exits.
        print(json.dumps({'event': 'evaluation_skipped', 'reason': 'steam client not running'}), flush=True)
        raise SystemExit(3)
    checkpoint, out = Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((checkpoint / 'state.json').read_text())
    config = saved['config']
    if args.target_from_tasks:
        if args.tasks_file == 'none':
            raise SystemExit('--target-from-tasks needs --tasks-file')
        config = dict(config, target=json.loads(Path(args.tasks_file).read_text()).get('target'))
    if args.episode_seconds:
        options = dict(config.get('reward_options', {}))
        if 'deadline_s' in options:   # combat-hitrate-hurt: a death costs the rest of the deadline
            options['deadline_s'] = args.episode_seconds
        config = dict(config, max_episode_seconds=args.episode_seconds, reward_options=options)
    if args.seeds.startswith('range:'):
        start, count = map(int, args.seeds.split(':')[1:])
        seeds = list(range(start, start + count))
    else:
        raw = json.loads(Path(args.seeds).read_text())
        seeds = [s['seed'] if isinstance(s, dict) else int(s) for s in (raw['seeds'] if isinstance(raw, dict) else raw)]
    if args.limit:
        seeds = seeds[:args.limit]
    results_path = out / 'results.jsonl'
    if args.duel_file:
        # duel: self-play one episode per seed; against another controller the checkpoint plays each side once
        sides = ('both',) if args.opponent == 'self' else ('player', 'npc')
        config = dict(config, max_episode_seconds=args.episode_seconds or config['max_episode_seconds'])
    else:
        sides = (None,)
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if r['outcome'] != 'error':
                done.add((r['seed'], r.get('policy_side')))
    todo = [(s, side) for s in seeds for side in sides if (s, side) not in done for _ in range(args.repeat)]
    meta = dict(checkpoint=str(checkpoint.resolve()), updates=saved['updates'], timesteps=saved['timesteps'],
                schema=config['schema'], reward_profile=config.get('reward_profile'), mode=args.mode,
                deterministic=not args.stochastic, sample_seed=args.sample_seed if args.stochastic else None,
                device=args.device, instances=args.instances, seeds=len(seeds), engine='abplus-1.06',
                tasks_file=args.tasks_file, observation_transport='json' if args.json_obs else 'binary v2',
                target=config.get('target'), target_from_tasks=args.target_from_tasks,
                block_moves=bool(config.get('block_moves')) or args.block_moves,
                invincible=bool(config.get('invincible', False)) and not args.mortal,
                episode_seconds=config['max_episode_seconds'], frames_per_decision=int(config.get('frames_per_decision', 2)),
                started=time.strftime('%Y-%m-%d %H:%M:%S'),
                duel_file=args.duel_file, opponent=args.opponent if args.duel_file else None,
                compare=[str(Path(c).resolve()) for c in args.compare.split(',')] if args.compare else None)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    (out / 'replays').mkdir(exist_ok=True)
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    for i, (s, side) in enumerate(todo):
        name = f'seed-{s}-{i}' + (f'-{side}' if side else '')
        tasks.put((s, str(out / 'replays' / f'{name}.jsonl.gz') if i < args.replays else None, side))
    n = min(args.instances, len(todo))
    print(f'{len(done)} done, {len(todo)} to run on {n} AB+ instances ({args.mode}, {args.device})', flush=True)
    procs = [ctx.Process(target=worker, args=(i, args, config, tasks, results), daemon=True) for i in range(n)]
    for proc in procs:
        proc.start()
    finished, t0, steps = 0, time.monotonic(), 0
    counts = collections.Counter()
    per_task = collections.defaultdict(lambda: {'outcomes': collections.Counter(), 'reward_v2': [], 'reward_own': [],
                                                'bombs_0': collections.Counter(), 'bombs_1+': collections.Counter()})
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
            for tag in ('reward_v3', 'reward_v4', 'reward_v5', 'reward_hitrate', 'reward_hitrate_walk',
                        'reward_hitrate_miss', 'reward_hitrate_fire', 'reward_hitrate_hurt', 'reward_hp'):
                if tag in r:
                    per_task[r.get('task', 'arena')]['reward_own'].append(r[tag])
            if r.get('start_bombs') is not None:
                split = 'bombs_0' if r['start_bombs'] == 0 else 'bombs_1+'
                per_task[r.get('task', 'arena')][split][r['outcome']] += 1
            steps += r.get('l', 0)
            elapsed = time.monotonic() - t0
            detail = (f"side {r.get('policy_side')} first hit {r.get('first_hit')} winner {r.get('winner')}"
                      if args.duel_file else f"dmg {r.get('damage_frac') or 0:.2f}")
            print(f"{time.strftime('%H:%M:%S')} seed {r['seed']} L{r['layout']} {r['outcome']:<10} "
                  f"frames {r.get('frames', '-'):>5} {detail} | "
                  f"{sum(counts.values())}/{len(todo)} {dict(counts)} {steps / elapsed:.0f} decisions/s", flush=True)
    for proc in procs:
        proc.join()
    mean = lambda values: round(float(np.mean(values)), 3) if values else None
    print(json.dumps({'outcomes': counts, 'seconds': round(time.monotonic() - t0, 1),
                      'decisions': steps, 'decisions_per_s': round(steps / max(1e-9, time.monotonic() - t0), 1),
                      'tasks': {k: {'outcomes': v['outcomes'], 'reward_v2_mean': mean(v['reward_v2']),
                                    **({'reward_own_mean': mean(v['reward_own'])} if v['reward_own'] else {}),
                                    **({'bombs_0': v['bombs_0'], 'bombs_1+': v['bombs_1+']}
                                       if v['bombs_0'] or v['bombs_1+'] else {})}
                                for k, v in sorted(per_task.items())}}))


if __name__ == '__main__':
    main()
