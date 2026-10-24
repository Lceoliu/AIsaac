"""PPO on the original engine: GpuMaskablePPO over AbplusFrameVecEnv. Training requires --train.

One exclusive CUDA learner process batches every worker's newest frame into one policy call per
chunk; each worker process owns one environment slot with two AB+ instances (active + standby).
Task: a room-level combat mixture (abplus_tasks; default rl/bridge/abplus/catalog/mixture_basement1.json):
the simulator's Monstro arena, Basement I normal rooms with their own enemies and Basement I boss
rooms, chosen per episode from the seed; --tasks-file none trains the arena only. combat-v2 reward
(isaac_bridge/abplus_reward.py; --reward-profile combat-v1 for the simulator's reward), 120 s deadline,
optional start randomisation. Per rollout the log has each task's episodes and win rate and the mean
reward components; episodes.jsonl has the components of every episode. Progress is also logged in
game time: one decision is --frames-per-decision logic frames (default 2), 30 logic frames are one game second,
so game/speed_x_realtime = decisions/s x frames per decision / 30.

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

--reward-profile combat-hitrate is the hit-rate test (user spec 2026-09-26): an invincible player
(--invincible), no bombs, a 180 s deadline (--episode-seconds) and only per-step hit, kill and time
terms, the time price moving with the running tear hit rate (abplus_reward.CombatHitRate). It keeps
combat-v5's observation, heads and auxiliary head. The log adds behavior/hit_rate and
behavior/shots_per_s. combat-hitrate-walk adds the walking-distance potential; combat-hitrate-miss
adds every enemy (lineage mode 3) and a miss penalty growing per miss in a row (--miss-cost) up to
the --miss-cap-th miss (bridge abp-0.2.6); combat-hitrate-fire moves each tear's hit, kill and miss
terms to the step it was fired in (bridge abp-0.2.7 combat.credits; GpuMaskablePPO applies the frame
'credit' before GAE).

Checkpoints hold weights, optimizer, counters and RNG. A resume starts fresh episodes (the
in-progress rooms are not rebuilt); it may change --envs and --n-steps (the rollout buffer is rebuilt), so
the number of environments can grow between stages of one training. Periodic evaluation runs
abplus_eval.py as a separate CPU process on a few extra instances, so the learner keeps the GPU to itself.

Parallel task groups (--groups-file, user plan 2026-09-28; isaac_bridge/abplus_groups.py): room and arena
families trained together, each with its share of the budget and its deadline; --budget splits the collected
steps (not the episode starts) between them, and the group changes only at an episode boundary. Rooms within
a group follow its seed mixture (--room-sampling mixture, no PLR). The log adds group/<name>/* (step share,
episodes, outcome rates, return, episode length) and each checkpoint is evaluated per group
(evaluations-<name>/, the group's rooms, target and deadline).

Model size (new models): --model-width, --model-layers, --model-heads, --entity-queries, --entity-width and
--head-width (transformer_policy.CombatTransformer; the defaults are the architecture of C1-C37).
Online distillation (--distill-from, gpu_ppo.GpuMaskablePPO): a frozen teacher policy; each sample adds the
masked four-head KL(teacher || student), next to the aux aim loss and the value loss, with the PPO term
weighted by --distill-rl-coef; --distill-teacher-acts lets the teacher collect the first updates.

Duel arena (--duel-file, user request 2026-09-28; isaac_bridge/abplus_duel.py, bridge abp-0.2.9): every AB+ game is the
player against the duel NPC, a second Isaac calibrated to the player; both are driven by the policy being trained
(self-play), each through its own first-person view, each paid by the same reward (the run's profile, one instance per
side). --envs counts policy slots, two per game (slot 2k the player, 2k + 1 the NPC); the rooms (terrains) are the duel
file's arms. Game time (game/hours_*, --game-hours) counts each game once: a step of n slots is n / 2 games x the frames
per decision. The log adds duel/<side>/* (outcome rates, return, hit rate, shots per second, first hit) and duel/*; each
checkpoint is evaluated by self-play (greedy, sampled) and against abplus_duel.ScriptedDuellist on either side.

C39 (user decisions 2026-09-28): --reward-profile combat-hp (abplus_reward.CombatHp: +1 per 3.5 HP any monster
loses, -0.5 per half heart, clear +100, deadline -100 and death -100, both terminal; the remaining time of the
group's deadline is observed), --frames-per-decision 4, and --groups-file with --room-sampling plr: every group
draws its (room, arm) levels by PLR (plr.group_levels), --plr-by-steps making PLR's shares shares of the steps
(each room's start probability divided by its measured episode length), while the groups keep their step budget.

C39 (user design 2026-09-29): --room-sampling buffer, a Room Buffer per group (isaac_bridge/room_buffer.py): a level is a
seed; fresh seeds take --buffer-fresh of each group's steps, the rest replays buffer seeds by the priority
4p(1-p) + lambda p clip(d/d0, 0, 1) + eta S + eps (p: clear EMA, d: half hearts lost in cleared episodes, S: staleness),
divided by their episode length. The log adds buffer/<group>/* and group/<name>/fresh_win_rate (the clear rate of fresh
seeds, the curriculum-free capability) and replay_win_rate; checkpoints carry room_buffer.json.
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

from isaac_bridge.abplus import BRIDGE_VERSION
from isaac_bridge.transformer_obs import DEADLINE_SCHEMA

# torch, SB3 and the policy are imported where they are used (resume, main): the worker processes are spawned, and a
# spawned process imports this module again as __mp_main__, so a top-level torch import cost each of them ~300 MiB
# (C39 smoke test on host 2: 32 workers x 360 MiB).

GAME_FPS = 30
FRAMES_PER_DECISION = 2   # --frames-per-decision (C39) sets it before anything is logged
HERE = Path(__file__).resolve().parent
SOURCES = ('train_abplus.py', 'abplus_eval.py', 'isaac_bridge/abplus.py', 'isaac_bridge/abplus_worker.py',
           'isaac_bridge/abplus_obs.py', 'isaac_bridge/abplus_tasks.py', 'isaac_bridge/abplus_reward.py',
           'isaac_bridge/abplus_vec.py', 'isaac_bridge/transformer_obs.py', 'isaac_bridge/transformer_policy.py',
           'isaac_bridge/gpu_ppo.py', 'isaac_bridge/gpu_buffer.py', 'isaac_bridge/gpu_env.py',
           'isaac_bridge/combat_reward.py', 'isaac_bridge/monstro_gym.py', 'isaac_bridge/env.py',
           'isaac_bridge/steam_watch.py', 'isaac_bridge/graph_sampler.py', 'isaac_bridge/plr.py',
           'isaac_bridge/abplus_geometry.py', 'isaac_bridge/hit_rate.py', 'isaac_bridge/abplus_groups.py',
           'isaac_bridge/abplus_duel.py', 'abplus_replay_view.py', 'isaac_bridge/abplus_replay.html')


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
    from isaac_bridge.abplus_worker import GOAL_PROFILES
    from isaac_bridge.steam_watch import SteamWatch
    from isaac_bridge.training_session import TrainingSession

    class AbplusSession(TrainingSession):
        """TrainingSession with game-time/worker logging and out-of-process AB+ evaluation."""

        def __init__(self, config, out, completed=0, updates=0, start_timesteps=0, plr=None, buffer=None, groups=None):
            super().__init__(config, out, completed, updates)
            self.buffer = buffer        # room_buffer.RoomBuffer (C39); groups: load_groups() for its family keys
            self.groups = groups
            self.start_timesteps = start_timesteps
            self.last_time = time.monotonic()
            self.last_steps = None
            self.evaluation = None
            self.rollout_episodes = []
            self.steam = SteamWatch(out)
            self.plr = plr              # plr.PrioritizedLevels when rooms are drawn by PLR
            self.level_ids = None       # room of every step of the current rollout (PLR scores)
            # Parallel task groups: names, and the steps each group collected in the current rollout.
            groups = (config.get('groups') or {}).get('groups') or []
            self.group_names = [g['name'] for g in groups]
            self.group_steps = np.zeros(len(groups), np.int64)
            self.fresh_steps = np.zeros(len(groups), np.int64)   # C39: steps of fresh (not replayed) seeds per group
            # Duel: two policy slots per game; game time counts each game once.
            self.agents = 2 if config.get('duel') else 1

        def _on_rollout_end(self):
            now, steps = time.monotonic(), self.model.num_timesteps
            if self.last_steps is not None and now > self.last_time:
                rate = (steps - self.last_steps) / (now - self.last_time)
                self.logger.record('game/decisions_per_s', rate)
                self.logger.record('game/speed_x_realtime', rate / self.agents * FRAMES_PER_DECISION / GAME_FPS)
            self.logger.record('game/hours_this_run', game_hours((steps - self.start_timesteps) / self.agents))
            self.logger.record('game/hours_total', game_hours(steps / self.agents))
            diagnostics = self.model.env.diagnostics()
            for key, value in diagnostics.items():
                self.logger.record('abplus/' + key, value)
            self.logger.record('abplus/steam_ok', int(self.steam.poll()))
            self.steam.launch_failed(diagnostics.get('instance_start_failures', 0))
            episodes, self.rollout_episodes = self.rollout_episodes, []
            if self.agents == 2:
                self._log_duel(episodes)
            for task in sorted({e['task'] for e in episodes} if self.agents == 1 else ()):
                mine = [e for e in episodes if e['task'] == task]
                self.logger.record(f'task/{task}/episodes', len(mine))
                self.logger.record(f'task/{task}/win_rate', sum(e['outcome'] == 'win' for e in mine) / len(mine))
                self.logger.record(f'task/{task}/return', float(np.mean([e['r'] for e in mine])))
                self.logger.record(f'task/{task}/death_rate', sum(e['outcome'] == 'death' for e in mine) / len(mine))
                self.logger.record(f'task/{task}/timeout_rate',
                                   sum(e['outcome'] == 'time_limit' for e in mine) / len(mine))
                if self.config.get('hurt_ends_episode'):   # C37: episodes ended by the first damage
                    self.logger.record(f'task/{task}/hurt_rate', sum(e['outcome'] == 'hurt' for e in mine) / len(mine))
                for label, armed in (('0bombs', False), ('bombs', True)):
                    sub = [e for e in mine if 'bombs' in e.get('start', {}) and (e['start']['bombs'] > 0) == armed]
                    if sub:
                        self.logger.record(f'task/{task}/win_rate_{label}', sum(e['outcome'] == 'win' for e in sub) / len(sub))
            if self.group_names:
                total = int(self.group_steps.sum())
                for i, name in enumerate(self.group_names):
                    self.logger.record(f'group/{name}/step_share', float(self.group_steps[i]) / total if total else 0.0)
                    if self.buffer is not None and self.group_steps[i]:
                        self.logger.record(f'group/{name}/fresh_step_share', float(self.fresh_steps[i] / self.group_steps[i]))
                    mine = [e for e in episodes if e.get('group') == name]
                    self.logger.record(f'group/{name}/episodes', len(mine))
                    if mine:
                        for outcome in ('win', 'hurt', 'death', 'time_limit'):
                            self.logger.record(f'group/{name}/{outcome}_rate',
                                               sum(e['outcome'] == outcome for e in mine) / len(mine))
                        self.logger.record(f'group/{name}/return', float(np.mean([e['r'] for e in mine])))
                        self.logger.record(f'group/{name}/episode_s', float(np.mean([e['frames'] for e in mine])) / GAME_FPS)
                        if self.buffer is not None:   # C39: fresh seeds are the curriculum-free sample
                            for label, replay in (('fresh', False), ('replay', True)):
                                sub = [e for e in mine if bool(e.get('replay')) == replay]
                                if sub:
                                    self.logger.record(f'group/{name}/{label}_win_rate',
                                                       sum(e['outcome'] == 'win' for e in sub) / len(sub))
                                    self.logger.record(f'group/{name}/{label}_episodes', len(sub))
                self.group_steps[:] = 0
                self.fresh_steps[:] = 0
            if self.buffer is not None:
                # C39: the finished episodes update their seeds, then the workers get the new table.
                from isaac_bridge.room_buffer import family_key
                options = self.config['reward_options']
                hurt = (options.get('combat') or options).get('hurt', 0.5)
                for e in episodes:
                    if 'group' not in e or 'reward_components' not in e:
                        continue
                    g = self.group_names.index(e['group'])
                    if 'option' in e:
                        # goal line (C44, user decision 2026-09-30): a seed's episode is its whole option sequence; success
                        # = every option of it succeeded (single-room COMBAT: the clear), d = the half hearts it lost
                        seq = e.get('sequence')
                        if seq is None:
                            continue   # the sequence goes on (or an engine failure ended it)
                        win, damage, steps = seq['ok'], float(seq['hurt']), seq['steps']
                    else:
                        win, steps = e['outcome'] == 'win', e['l']
                        damage = -float(e['reward_components'].get('hurt', 0.0)) / hurt
                    self.buffer.observe(g, e['seed'], win, damage, steps,
                                        family_key(self.groups[g], e['seed'], e['layout']), bool(e.get('replay')))
                self.model.env.set_room_buffer(*self.buffer.table())
                for key, value in self.buffer.summary().items():
                    self.logger.record('buffer/' + key, value)
            if any('option' in e for e in episodes):
                # goal line: COMBAT and GOTO options have their own reward components; per option kind and outcome rates
                for kind in ('combat', 'goto'):
                    mine = [e for e in episodes if 'option' in e and (e['option'] == 'combat') == (kind == 'combat')]
                    parts = [e['reward_components'] for e in mine if 'reward_components' in e]
                    for key in (parts[0] if parts else ()):
                        self.logger.record(f'reward/{kind}/{key}', float(np.mean([c[key] for c in parts])))
                    for outcome in ('win', 'goal', 'death', 'time_limit', 'exit', 'wrong_door', 'error'):
                        if mine:
                            self.logger.record(f'option/{kind}/{outcome}_rate',
                                               sum(e['outcome'] == outcome for e in mine) / len(mine))
                    self.logger.record(f'option/{kind}/episodes', len(mine))
                for source in ('single', 'chain'):
                    mine = [e for e in episodes if e.get('option') == 'combat' and e.get('source') == source]
                    if mine:
                        self.logger.record(f'option/combat_{source}/win_rate', sum(e['outcome'] == 'win' for e in mine) / len(mine))
                        self.logger.record(f'option/combat_{source}/episodes', len(mine))
                door = [e for e in episodes if e.get('option') == 'goto_door']
                if door:
                    self.logger.record('option/goto_door/goal_rate', sum(e['outcome'] == 'goal' for e in door) / len(door))
                    self.logger.record('option/goto_door/episodes', len(door))
            else:
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
                shots = sum(st.get('shots', 0) for st in stats)
                seconds = sum(e['frames'] for e in episodes if 'stats' in e) / GAME_FPS
                if shots:
                    self.logger.record('behavior/hit_rate', sum(st.get('tear_hits', 0) for st in stats) / shots)
                    self.logger.record('behavior/miss_share', sum(st.get('misses', 0) for st in stats) / shots)
                if seconds:
                    self.logger.record('behavior/shots_per_s', shots / seconds)
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

        def _log_duel(self, episodes):
            """Per side (player, npc): outcome rates, return, hit rate, shots per second, first hit; per game: length,
            decided share, both hurt in the same step."""
            for side in ('player', 'npc'):
                mine = [e for e in episodes if e.get('side') == side]
                self.logger.record(f'duel/{side}/episodes', len(mine))
                if not mine:
                    continue
                for outcome in ('win', 'hurt', 'death', 'time_limit'):
                    self.logger.record(f'duel/{side}/{outcome}_rate', sum(e['outcome'] == outcome for e in mine) / len(mine))
                self.logger.record(f'duel/{side}/return', float(np.mean([e['r'] for e in mine])))
                stats = [e['stats'] for e in mine if 'stats' in e]
                shots = sum(st.get('shots', 0) for st in stats)
                seconds = sum(e['frames'] for e in mine if 'stats' in e) / GAME_FPS
                if shots:
                    self.logger.record(f'duel/{side}/hit_rate', sum(st.get('tear_hits', 0) for st in stats) / shots)
                if seconds:
                    self.logger.record(f'duel/{side}/shots_per_s', shots / seconds)
                first = [st['first_hit_s'] for st in stats if st['first_hit_s'] >= 0]
                if first:
                    self.logger.record(f'duel/{side}/first_hit_s', float(np.mean(first)))
            games = [e for e in episodes if e.get('side') == 'player']
            if games:
                self.logger.record('duel/episode_s', float(np.mean([e['frames'] for e in games])) / GAME_FPS)
                self.logger.record('duel/decided_rate', float(np.mean([e['outcome'] != 'time_limit' for e in games])))
                # the NPC of the player's game is the next slot, with the same seed
                npc = {(e['seed'], e['worker']): e['outcome'] for e in episodes if e.get('side') == 'npc'}
                both = [e['outcome'] == 'hurt' and npc.get((e['seed'], e['worker'] + 1)) == 'hurt' for e in games]
                self.logger.record('duel/both_hurt_rate', float(np.mean(both)))

        def _on_step(self):
            # TrainingSession._on_step plus the task of each finished episode (the file is opened
            # only on steps where an episode ended).
            dones = self.locals['dones']
            if self.group_names:
                played = [info.get('group', -1) for info in self.locals['infos']]
                self.group_steps += np.bincount([g for g in played if g >= 0], minlength=len(self.group_names))
                if self.buffer is not None:
                    fresh = [info.get('group', -1) for info in self.locals['infos'] if not info.get('replay')]
                    self.fresh_steps += np.bincount([g for g in fresh if g >= 0], minlength=len(self.group_names))
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
                        group = ({'group': self.group_names[info['group']]}
                                 if self.group_names and info.get('group', -1) >= 0 else {})
                        if 'side' in info:   # duel
                            group['side'] = info['side']
                        replay = {'replay': bool(info['replay'])} if 'replay' in info and self.buffer is not None else {}
                        # C39 t3r: the resident memory of the process that played the episode, at its end.
                        memory = ({'rss_mib': round(info['rss_mib'], 1), 'instance': info['instance'],
                                   'instance_episodes': info['instance_episodes']} if 'rss_mib' in info else {})
                        # goal line: the option (GOAL_TASKS) and the sample source of this PPO episode
                        option = ({'option': info['option'], 'source': info['source']} if 'option' in info else {})
                        if 'sequence' in info:   # C44: the option sequence ended with this option
                            option['sequence'] = info['sequence']
                        record = dict(episode=self.completed, worker=worker, seed=info['seed'], **group, **replay, **memory,
                                      **option,
                                      task=info.get('task'), outcome=info['outcome'], layout=info['layout'],
                                      level=info.get('level', -1), frames=info['elapsed_frames'],
                                      truncated=bool(info.get('TimeLimit.truncated')), **info['episode'],
                                      **start, **parts)
                        f.write(json.dumps(record) + '\n')
                        self.rollout_episodes.append(record)
            keep = self.completed < self.config['episodes']
            limit = self.config.get('game_hours')
            played = game_hours((self.model.num_timesteps - self.start_timesteps) / self.agents)
            return keep and not (limit and played >= limit)

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
            replays = int(self.config.get('eval_replays', 0))
            cmd = [sys.executable, '-u', str(HERE / 'abplus_eval.py'), '--checkpoint', str(checkpoint),
                   '--instances', str(self.config['eval_instances']), '--device', self.config.get('eval_device', 'cpu'),
                   '--name', 'ev', '--port', str(self.config['port'] - 200), '--replays', str(replays),
                   '--recycle-rss-mib', str(self.config.get('recycle_rss_mib') or 0),
                   *(['--bombs', str(self.config['eval_bombs'])] if self.config.get('eval_bombs') is not None else [])]
            if self.group_names:
                self._evaluate_groups(checkpoint, cmd, count, replays)
                return
            if self.agents == 2:
                self._evaluate_duel(checkpoint, cmd, count, replays)
                return
            out = self.out / 'evaluations' / checkpoint.name
            out.mkdir(parents=True, exist_ok=True)
            if self.config.get('tasks_spec') and self.config.get('eval_mode', 'mixture') == 'mixture':
                spec = self.out / 'tasks_spec.json'
                if not spec.exists():
                    spec.write_text(json.dumps(self.config['tasks_spec']), encoding='utf8')
                cmd += ['--tasks-file', str(spec), '--seeds', f"range:{self.config['eval_seed_start']}:{count}"]
            else:
                cmd += ['--seeds', self.config['eval_seeds_file'], '--limit', str(count)]
            # Greedy first (comparable across runs), then the sampled policy (as it trains), then one
            # replay page of both; the chain is one process for the "previous evaluation" check. Device: _eval_env.
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
                                               env=self._eval_env())
            log.close()
            print(json.dumps(dict(event='evaluation_started', checkpoint=checkpoint.name, out=str(out),
                                  pid=self.evaluation.pid)), flush=True)

        def _evaluate_groups(self, checkpoint, cmd, count, replays):
            """Parallel task groups: every group's rooms (its eval_tasks when it has them), target and deadline on the
            held-out seeds, greedy and sampled, one replay page each (evaluations-<group>/<checkpoint>/); one chained
            process."""
            runs = []
            for g in self.config['groups']['groups']:
                out = self.out / f"evaluations-{g['name']}" / checkpoint.name
                out.mkdir(parents=True, exist_ok=True)
                if g.get('mode', 'combat') != 'combat' or self.config.get('reward_profile') in GOAL_PROFILES:
                    # goal line: whole option sequences (abplus_eval_options), greedy and sampled; letters-only names;
                    # a goal-line run's single-room COMBAT group too (C44: its observation has the goal fields)
                    opt = [sys.executable, '-u', str(HERE / 'abplus_eval_options.py'), '--checkpoint', str(checkpoint),
                           '--groups-file', self.config['groups_file'], '--group', g['name'],
                           '--seeds', f"range:{self.config['eval_seed_start']}:{count}",
                           '--instances', str(self.config['eval_instances']), '--name', 'evo',
                           '--port', str(self.config['port'] - 200),
                           '--recycle-rss-mib', str(self.config.get('recycle_rss_mib') or 400)]
                    runs.append(opt + ['--out', str(out)])
                    if self.config.get('eval_sampled'):
                        runs.append(opt + ['--out', str(out / 'sampled'), '--stochastic', '--sample-seed', '0'])
                    continue
                base = cmd + ['--tasks-file', g.get('eval_tasks') or g['tasks'],
                              '--seeds', f"range:{self.config['eval_seed_start']}:{count}",
                              '--episode-seconds', str(g['episode_seconds'])]
                if g.get('has_target'):
                    base.append('--target-from-tasks')
                runs.append(base + ['--out', str(out)])
                labels = [f'{out}=最优']
                if self.config.get('eval_sampled'):
                    runs.append(base + ['--out', str(out / 'sampled'), '--stochastic', '--sample-seed', '0'])
                    labels.append(f"{out / 'sampled'}=采样")
                if replays:
                    runs.append([sys.executable, str(HERE / 'abplus_replay_view.py'), str(out / 'replays.html'), *labels,
                                 '--title', f"AB+ 评估回放 · {self.out.name} · {g['name']} · {checkpoint.name}"])
            logs = self.out / 'eval-logs'
            logs.mkdir(exist_ok=True)
            log = (logs / f'{checkpoint.name}.log').open('w', encoding='utf8')
            self.evaluation = subprocess.Popen(['bash', '-c', '; '.join(shlex.join(c) for c in runs)],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                               env=self._eval_env())
            log.close()
            print(json.dumps(dict(event='evaluation_started', checkpoint=checkpoint.name, groups=self.group_names,
                                  pid=self.evaluation.pid)), flush=True)

        def _evaluate_duel(self, checkpoint, cmd, count, replays):
            """Duel: the held-out seeds by self-play (greedy, sampled), against the scripted duellist and against the
            warm-start policy (the checkpoint on either side, greedy), evaluations-<mode>/<checkpoint>/; then the transfer
            tasks (--duel-transfer: C37's normal-room and dodging-arena evaluations, greedy and sampled,
            evaluations-<name>/<checkpoint>/[sampled/]); one replay page each; one chained process."""
            runs = []
            seeds = f"range:{self.config['eval_seed_start']}:{count}"
            base = cmd + ['--duel-file', self.config['duel_file'], '--seeds', seeds]
            modes = [('selfplay', ['--opponent', 'self'], '自对弈 · 最优')]
            if self.config.get('eval_sampled'):
                modes.append(('selfplay-sampled', ['--opponent', 'self', '--stochastic', '--sample-seed', '0'], '自对弈 · 采样'))
            modes.append(('scripted', ['--opponent', 'scripted'], '对脚本 · 最优'))
            if self.config.get('warm_start'):   # a fixed reference: the policy the run started from
                modes.append(('vs-start', ['--opponent', self.config['warm_start']], '对起点 · 最优'))
            for name, extra, label in modes:
                out = self.out / f'evaluations-{name}' / checkpoint.name
                out.mkdir(parents=True, exist_ok=True)
                runs.append(base + extra + ['--out', str(out)])
                if replays:
                    runs.append([sys.executable, str(HERE / 'abplus_replay_view.py'), str(out / 'replays.html'),
                                 f'{out}={label}', '--title', f'AB+ 对战评估 · {self.out.name} · {label} · {checkpoint.name}'])
            for t in self.config.get('duel_transfer') or ():
                out = self.out / f"evaluations-{t['name']}" / checkpoint.name
                out.mkdir(parents=True, exist_ok=True)
                transfer = cmd + ['--tasks-file', t['tasks'], '--seeds', seeds, '--episode-seconds', str(t['seconds'])]
                if t.get('target'):
                    transfer.append('--target-from-tasks')
                runs.append(transfer + ['--out', str(out)])
                labels = [f'{out}=最优']
                if self.config.get('eval_sampled'):
                    runs.append(transfer + ['--out', str(out / 'sampled'), '--stochastic', '--sample-seed', '0'])
                    labels.append(f"{out / 'sampled'}=采样")
                if replays:
                    runs.append([sys.executable, str(HERE / 'abplus_replay_view.py'), str(out / 'replays.html'), *labels,
                                 '--title', f"AB+ 评估回放 · {self.out.name} · {t['label']} · {checkpoint.name}"])
            logs = self.out / 'eval-logs'
            logs.mkdir(exist_ok=True)
            log = (logs / f'{checkpoint.name}.log').open('w', encoding='utf8')
            self.evaluation = subprocess.Popen(['bash', '-c', '; '.join(shlex.join(c) for c in runs)],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                               env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
            log.close()
            print(json.dumps(dict(event='evaluation_started', checkpoint=checkpoint.name, duel=[m[0] for m in modes],
                                  transfer=[t['name'] for t in self.config.get('duel_transfer') or ()],
                                  pid=self.evaluation.pid)), flush=True)

        def _eval_env(self):
            """Environment of an evaluation chain. On the CPU (--eval-device cpu, the default) CUDA_VISIBLE_DEVICES=''
            keeps each worker from opening a CUDA context (~270 MiB of the learner's GPU each, B7); --eval-device cuda
            (C39) leaves the GPU visible: CPU inference ran ~90 decisions/s on 4 instances, slower than the checkpoints
            came."""
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            if self.config.get('eval_device', 'cpu') == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            return env

        def save(self, model, phase):
            checkpoint = super().save(model, phase)
            if self.plr is not None:
                (checkpoint / 'plr.json').write_text(json.dumps(self.plr.state_dict()), encoding='utf8')
            if self.buffer is not None:
                (checkpoint / 'room_buffer.json').write_text(json.dumps(self.buffer.state_dict()), encoding='utf8')
            return checkpoint

        def finish(self, model):
            checkpoint = self.save(model, 'final')
            if self.evaluation is not None:
                self.evaluation.wait()  # the final checkpoint is always evaluated
            self.evaluate(model, checkpoint)
            (self.out / 'result.json').write_text(json.dumps(dict(
                completed_episodes=self.completed, timesteps=model.num_timesteps, updates=self.updates,
                game_hours_this_run=game_hours((model.num_timesteps - self.start_timesteps) / self.agents),
                checkpoint=str(checkpoint))), encoding='utf8')

    return AbplusSession


def resume(checkpoint, env, device, n_steps=None):
    """Weights, optimizer, counters and RNG from a checkpoint; episodes restart fresh. The environment count
    may differ from the checkpoint's (SB3 takes env.num_envs) and n_steps replaces the saved rollout length;
    the rollout buffer is built for both."""
    import torch
    from isaac_bridge.gpu_ppo import GpuMaskablePPO
    from isaac_bridge.training_session import resolve_checkpoint, restore_rng
    checkpoint = resolve_checkpoint(checkpoint)
    state = json.loads((checkpoint / 'state.json').read_text())
    if state['config'].get('reward_profile') != env.reward_profile:
        raise ValueError(f"checkpoint reward {state['config'].get('reward_profile')} != {env.reward_profile}: "
                         'use --warm-start (weights only) to change the reward')
    with gzip.open(checkpoint / 'continuation.pt.gz', 'rb') as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    model = GpuMaskablePPO.load(checkpoint / 'model.zip', env=env, device=device, force_reset=True,
                                custom_objects={'n_steps': int(n_steps)} if n_steps else None)
    restore_rng(data['rng'])
    return model, state


def main():
    global FRAMES_PER_DECISION
    # A job started in the background (setsid nohup ... &) inherits SIGINT ignored, and Python then installs no
    # KeyboardInterrupt handler: kill -INT did nothing to the C39 t1 run. Restore it, so kill -INT stops a run cleanly
    # (env.close() stops the workers and their AB+ instances).
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)
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
    p.add_argument('--lr-schedule', choices=('constant', 'linear', 'cosine'), default='constant',
                   help='learning rate over this run\'s --game-hours: constant, or linear / cosine from --learning-rate '
                        'down to --lr-final (C36)')
    p.add_argument('--lr-final', type=float, default=None,
                   help='--lr-schedule linear / cosine: the learning rate at the end (default 0.1 x --learning-rate)')
    p.add_argument('--shared-learning-rate', type=float, default=None,
                   help='C44 (design 4.8): two optimizer groups; the shared (pre-goal-line) parameters peak at this rate '
                        '(--lr-schedule down to --shared-lr-final, after --shared-lr-warmup-updates linear updates from '
                        '--shared-lr-warmup-start); the goal-line parameters follow --learning-rate / --lr-final')
    p.add_argument('--shared-lr-final', type=float, default=None, help='default 0.1 x --shared-learning-rate')
    p.add_argument('--shared-lr-warmup-updates', type=int, default=0)
    p.add_argument('--shared-lr-warmup-start', type=float, default=None, help='default --shared-lr-final')
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
    p.add_argument('--eval-device', choices=('cpu', 'cuda'), default='cpu',
                   help='device of the periodic evaluations\' policies (cpu keeps the learner\'s GPU memory; C39: cuda)')
    p.add_argument('--eval-mode', choices=('mixture', 'arena'), default='mixture',
                   help='mixture: held-out seeds from --eval-seed-start through the task mixture; arena: E2 seeds file')
    p.add_argument('--eval-seed-start', type=int, default=2147490000,
                   help='held-out block (>= 2**31, after the E2 validation block, before the final test block)')
    p.add_argument('--warm-start', type=Path, help='weights only (e.g. a simulator checkpoint)')
    p.add_argument('--warm-start-optimizer', action='store_true',
                   help='with --warm-start: also the checkpoint\'s optimizer state (Adam moments); the learning rate follows '
                        'this run\'s schedule (C41 continuing C39)')
    p.add_argument('--warm-start-optimizer-shared', type=Path, default=None,
                   help='C44: the Adam state of the shared (pre-goal-line) parameters from this checkpoint (the final of C39: they '
                        'were frozen after it); --warm-start-optimizer then restores only the state of the goal-line parameters')
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
    p.add_argument('--greedy-actors', type=int, default=0,
                   help='environments (the first N) that sample at --greedy-temperature, near the greedy policy '
                        '(C29; needs --sampler graph and --async-train, whose decoupled weights correct for it)')
    p.add_argument('--greedy-temperature', type=float, default=0.25)
    p.add_argument('--kl-probe', action='store_true',
                   help='diagnostic: log the samples with the largest pi_new / pi_proximal per update to kl_probe.jsonl (C32)')
    p.add_argument('--block-moves', action='store_true',
                   help='mask the moves the terrain stops dead (abplus_geometry.blocked_moves) in training and in '
                        'this run\'s evaluations (C30)')
    p.add_argument('--goal-reward', type=float, default=None,
                   help='goal line: the GOTO reward for reaching the goal, unscaled (default 10 = +1 trained; C45 15)')
    p.add_argument('--damage-hp', type=float, default=None,
                   help='goal line: monster HP per unscaled reward point of COMBAT damage (default 3.5 = +0.1 trained per '
                        '3.5 HP; C45 1.75 = +0.2)')
    p.add_argument('--expert-coef', type=float, default=0.0,
                   help='goal-hp3 (C45): imitation of the scripted A* expert on the move head (gpu_ppo.expert_coef), '
                        'falling linearly to 0 at --expert-until of the run')
    p.add_argument('--expert-until', type=float, default=0.5)
    p.add_argument('--expert-only', action='store_true',
                   help='diagnosis (A15): the expert cross-entropy is the only loss (a supervised fit on the policy states)')
    p.add_argument('--grad-probe', action='store_true',
                   help='diagnosis (A15): log each loss term gradient norm per optimizer group and the clipping')
    p.add_argument('--max-grad-norm', type=float, default=None,
                   help='the gradient clipping norm (default: the checkpoint or SB3 value, 0.5)')
    p.add_argument('--hurt-cost', type=float, default=None,
                   help='combat-hp / combat-hp2: the cost per half heart (default 0.5 / 1.0)')
    p.add_argument('--stat-noise', default=None,
                   help='C41: KEY=A,...: each episode offsets the player\'s stat KEY by U(-A, A), drawn from its seed; '
                        'keys speed (MoveSpeed), damage, shot_speed, tears (shots per second), range (Repentance units '
                        'of 40 px); bridge abp-0.2.11 (evaluations keep the base stats)')
    p.add_argument('--start-bombs', default=None,
                   help='ZERO_PROB:MAX: start with 0 bombs with this probability, else 1..MAX; 0 = never any bomb; '
                        'none = the game\'s 1 (default 0.5:3, combat-hitrate 0)')
    p.add_argument('--buffer-capacity', type=int, default=2000, help='--room-sampling buffer: seeds kept per group (C39)')
    p.add_argument('--buffer-fresh', type=float, default=0.3, help='--room-sampling buffer: fresh seeds\' share of the steps')
    p.add_argument('--buffer-alpha', type=float, default=0.25, help='--room-sampling buffer: EMA weight of p, d and length')
    p.add_argument('--buffer-lambda', type=float, default=0.5, help='--room-sampling buffer: weight of p clip(d/d0, 0, 1)')
    p.add_argument('--buffer-d0', type=float, default=2.0, help='--room-sampling buffer: half hearts that normalise d')
    p.add_argument('--buffer-eta', type=float, default=0.1, help='--room-sampling buffer: staleness weight')
    p.add_argument('--buffer-eps', type=float, default=0.02, help='--room-sampling buffer: every seed\'s priority floor')
    p.add_argument('--room-sampling', choices=('plr', 'mixture', 'buffer'), default='plr',
                   help='plr: Prioritized Level Replay over the rooms of --tasks-file; mixture: its fixed weights by seed')
    p.add_argument('--plr-beta', type=float, default=1.0, help='rank temperature (P ~ (1/rank)^(1/beta))')
    p.add_argument('--plr-staleness', type=float, default=0.3)
    p.add_argument('--plr-floor', type=float, default=0.1, help='uniform share over the rooms of each kind')
    p.add_argument('--plr-by-steps', action='store_true',
                   help="PLR's room shares are shares of the collected steps: a room starts with its PLR probability "
                        'divided by its measured episode length (C39, plr.PrioritizedLevels by_steps)')
    p.add_argument('--frames-per-decision', type=int, default=2,
                   help='logic frames each action is held for (default 2: 15 decisions per game second; C39: 4)')
    p.add_argument('--eval-sampled', action=argparse.BooleanOptionalAction, default=True,
                   help='also evaluate the sampled policy (the greedy one always runs)')
    p.add_argument('--eval-replays', type=int, default=8, help='replays recorded per evaluation mode')
    p.add_argument('--eval-bombs', type=int, default=None,
                   help='bombs at the start of every evaluation episode (default: the training start-bomb draw fixed '
                        'per seed, or the game default 1 without one; C41: 1, with the base stats and full HP)')
    p.add_argument('--recycle-episodes', type=int, default=200,
                   help='restart each AB+ process after this many episodes (the game leaks memory while it plays); 0 = never')
    p.add_argument('--recycle-rss-mib', type=float, default=0,
                   help='also restart an AB+ process once its resident memory reaches this many MiB, checked at the end '
                        'of each of its episodes (C39 t3r); 0 = never')
    p.add_argument('--freeze-shared', action='store_true',
                   help='goal line stage 1: every parameter but the goal-line modules (goal encoder and injections, the '
                        'navigation residual, the GOTO critic) is frozen and the losses average over the GOTO samples')
    p.add_argument('--preserve-coef', type=float, default=0.0,
                   help='goal line: weight of KL(frozen copy of the migrated policy || policy) on the single-room samples '
                        '(user decision 2026-09-30: standard single-room combat keeps its PPO data and gets this KL)')
    p.add_argument('--preserve-target', type=float, default=None,
                   help='goal line: adapt --preserve-coef after each update towards this mean KL on the single-room samples '
                        '(x1.5 above 1.5 target, /1.5 below target / 1.5; EXPERIMENTS.md A10 measured C39 u800 vs u820 at '
                        '0.07-0.14 per decision)')
    p.add_argument('--reward-profile', choices=('combat-v5', 'combat-hitrate', 'combat-hitrate-walk', 'combat-hitrate-miss',
                                                'combat-hitrate-fire', 'combat-hitrate-hurt', 'combat-hp', 'combat-hp2',
                                                'combat-hp2-camera', 'goal-hp', 'goal-hp2', 'goal-hp3', 'combat-v4',
                                                'combat-v3', 'combat-v2', 'combat-v1'),
                   default='combat-v5',
                   help='goal-hp (goal-conditioned line): COMBAT options as combat-hp2, GOTO options by '
                        'abplus_reward.GotoReward, the observation with the goal fields (camera view), the group modes '
                        'combat / combat_goto / goto_empty / chain; '
                        'combat-hp (C39): +1 per 3.5 HP any monster loses, -0.5 per half heart, clear +100, deadline and '
                        'death -100 (terminal), the remaining time observed; combat-hp2 (C41): the same with -1 per half '
                        'heart on rooms of every shape (16x28 terrain canvas, positions in 1x1-room units); '
                        'combat-hp2-camera (C41 continuing C39): the same reward, the terrain the 15x9 window around the '
                        'player (the game camera), so combat-hp checkpoints warm-start; '
                        'combat-hitrate: the temporary hit-rate test (invincible, no bombs, 180 s, hit + kill + '
                        'hit-rate time price); combat-hitrate-walk: the same plus the walking-distance potential; '
                        'combat-hitrate-miss: that plus every enemy rewarded and a growing miss penalty; '
                        'combat-hitrate-fire: that with each tear credited to the step it was fired in; '
                        'combat-hitrate-hurt: combat-hitrate-miss without invincibility, health curve and a death '
                        'that costs the rest of the deadline (stage 2, C33); '
                        'combat-v5: stage one revised (roster-lineage hits, alignment potential, factored heads, '
                        'auxiliary geometry head); combat-v4: stage one, learn to clear (no penalties, clear bonus, timeout truncates); '
                        'combat-v3: clear-first with penalties; combat-v2: score-calibrated (abplus_reward.py); '
                        'combat-v1: the simulator run reward')
    p.add_argument('--align-coef', type=float, default=None,
                   help='combat-v5 / combat-hitrate-walk alignment potential weight per cell (default 0.2)')
    p.add_argument('--hit-hp', type=float, default=None,
                   help='combat-v5 / combat-hitrate: HP per +1 of hit reward (default 5)')
    p.add_argument('--time-cost-hit', type=float, default=None,
                   help='combat-hitrate: time price per game second at hit rate 1 (default 0.5)')
    p.add_argument('--time-cost-miss', type=float, default=None,
                   help='combat-hitrate: time price per game second at hit rate 0 (default 2)')
    p.add_argument('--episode-seconds', type=float, default=None,
                   help='episode deadline in game seconds (default 120, combat-hitrate 180)')
    p.add_argument('--invincible', action=argparse.BooleanOptionalAction, default=None,
                   help='the player takes no damage (bridge abp-0.2.4; default on for combat-hitrate only)')
    p.add_argument('--hurt-ends-episode', action='store_true',
                   help='the first damage ends the episode as a failure (outcome hurt, terminal); needs a mortal '
                        'player; evaluation keeps the game rules (C37)')
    p.add_argument('--hurt-rest-cost', action='store_true',
                   help='with --hurt-ends-episode (combat-hitrate-miss / -fire): that hurt also costs the rest of the '
                        'deadline at the no-hit time price, --time-cost-miss per second (reward component rest)')
    p.add_argument('--no-death-cost', action='store_true',
                   help='combat-hitrate-hurt: no death term; the lethal hit still ends the episode and costs the '
                        'health curve (C36)')
    p.add_argument('--miss-cost', type=float, default=None,
                   help='combat-hitrate-miss: penalty of the first miss in a row, the k-th costs k times it (default 0.01)')
    p.add_argument('--miss-cap', type=int, default=None,
                   help='combat-hitrate-miss: the k-th miss in a row costs min(k, cap) times --miss-cost '
                        '(bridge abp-0.2.6; default 20, 0 = no cap)')
    p.add_argument('--lineage-mode', type=int, choices=(0, 1, 2, 3), default=None,
                   help='combat-v5 death successors in the lineage: 0 none, 1 all (default), 2 a single one; '
                        '3 every doors-blocking NPC when first seen (combat-hitrate-miss default)')
    p.add_argument('--aux-coef', type=float, default=0.2, help='combat-v5 auxiliary geometry head loss weight')
    p.add_argument('--aux-distance-coef', type=float, default=1.0,
                   help='weight of the auxiliary d_fire regression next to the aim cross-entropy (C21: 0)')
    p.add_argument('--aux-balance', action='store_true',
                   help='aligned and unaligned samples weigh half each in the aux aim cross-entropy (C22)')
    p.add_argument('--groups-file', type=Path, default=None,
                   help='parallel task groups (isaac_bridge/abplus_groups.py): room and arena families trained together, '
                        'each with its budget share and deadline; replaces --tasks-file, needs --room-sampling mixture; '
                        'checkpoints are evaluated per group (evaluations-<group>/)')
    p.add_argument('--budget', choices=('steps', 'slots', 'episodes'), default='steps',
                   help='--groups-file: steps: every slot starts the group furthest below its share of the steps it '
                        'collected; slots: every slot plays one group; episodes: each episode drawn by share')
    p.add_argument('--model-width', type=int, default=256, help='Transformer and fusion width (new models)')
    p.add_argument('--model-layers', type=int, default=4, help='temporal Transformer layers (new models)')
    p.add_argument('--model-heads', type=int, default=8, help='temporal attention heads (new models)')
    p.add_argument('--entity-queries', type=int, default=4, help='entity summary queries (new models)')
    p.add_argument('--entity-width', type=int, default=128, help='entity token width (new models)')
    p.add_argument('--head-width', type=int, default=256, help='policy and value MLP width (new models)')
    p.add_argument('--distill-from', type=Path, default=None,
                   help='online distillation: the frozen policy of this checkpoint is the teacher')
    p.add_argument('--distill-coef', type=float, default=1.0, help='weight of the masked four-head KL(teacher || student)')
    p.add_argument('--distill-coef-end', type=float, default=None,
                   help='the KL weight at the end of this run\'s budget (linear from --distill-coef)')
    p.add_argument('--distill-temperature', type=float, default=1.0, help='softens both distributions (loss x T^2)')
    p.add_argument('--distill-rl-coef', type=float, default=0.0,
                   help='weight of the PPO policy-gradient term while distilling (0: pure distillation; >0: kickstarting)')
    p.add_argument('--distill-teacher-acts', type=int, default=0,
                   help='the teacher collects the rollouts until the model has made this many updates')
    p.add_argument('--duel-file', type=Path, default=None,
                   help='the duel arena (isaac_bridge/abplus_duel.py, bridge abp-0.2.9): a duel spec such as '
                        'catalog/duel_rooms.json; every game is the player against the duel NPC, both played by this '
                        'policy (self-play), same reward per side; --envs counts slots, two per game; replaces '
                        '--tasks-file; default deadline 30 s')
    p.add_argument('--duel-transfer', default='rooms,dodge',
                   help='--duel-file: tasks every checkpoint is also evaluated on, as C37 was (greedy and sampled, the held-out '
                        'seeds): rooms (the 202 positioning rooms, 180 s), dodge (the dodging arena, 30 s); none = no transfer')
    p.add_argument('--software-gl', action='store_true',
                   help='AB+ instances render through Mesa\'s software OpenGL instead of the GPU driver: no GPU memory '
                        'per instance (about 80 MiB each otherwise), identical trajectories (EXPERIMENTS.md B7)')
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.frames_per_decision <= 30:
        p.error('--frames-per-decision must be 1..30')
    FRAMES_PER_DECISION = args.frames_per_decision
    if args.software_gl:   # inherited by the workers, their AB+ instances and the evaluation processes
        from isaac_bridge.abplus import SOFTWARE_GL_ENV
        os.environ.update(SOFTWARE_GL_ENV)
    from isaac_bridge.abplus_reward import (HPR, HR, HRM, HRW, V5, describe as describe_reward, describe_hitrate,
                                            describe_hitrate_fire, describe_hitrate_hurt, describe_hitrate_miss,
                                            describe_hitrate_walk, describe_hp, describe_hp2, describe_goal, HPR2,
                                            describe_v3, describe_v4, describe_v5)
    from isaac_bridge.abplus_worker import GOAL_PROFILES, observation_options
    from isaac_bridge.gpu_env import validate_start_randomization
    from isaac_bridge.training_session import resolve_checkpoint, warm_start_model
    if args.resume and args.warm_start:
        p.error('--resume and --warm-start are mutually exclusive')
    if (args.goal_reward is not None or args.damage_hp is not None) and args.reward_profile not in GOAL_PROFILES:
        p.error('--goal-reward / --damage-hp are goal-line options')
    if args.expert_coef and args.reward_profile != 'goal-hp3':
        p.error('--expert-coef needs goal-hp3 (the frames carry the expert labels)')
    if args.expert_only and not args.expert_coef:
        p.error('--expert-only needs --expert-coef')
    if args.shared_learning_rate and args.reward_profile not in GOAL_PROFILES:
        p.error('--shared-learning-rate is a goal-line option (goal-hp, goal-hp2)')
    if args.warm_start_optimizer_shared and not (args.warm_start and args.shared_learning_rate):
        p.error('--warm-start-optimizer-shared needs --warm-start and --shared-learning-rate')
    if args.warm_start_optimizer and not args.warm_start:
        p.error('--warm-start-optimizer needs --warm-start')
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
    tasks = target = groups = duel = duel_transfer = None
    if args.duel_file:
        duel = json.loads(args.duel_file.read_text(encoding='utf8'))
        if 'duel' not in duel:
            p.error(f'{args.duel_file} has no duel spec')
        if args.groups_file:
            p.error('--duel-file and --groups-file are exclusive')
        if args.envs % 2:
            p.error('--duel-file: --envs counts policy slots, two per game: it must be even')
        if args.room_sampling != 'mixture':
            p.error('--duel-file draws its rooms by the arms\' weights: pass --room-sampling mixture')
        if args.episode_seconds is None:
            args.episode_seconds = 30.0
        # Transfer evaluations: the tasks and settings of C37's own evaluations, so the numbers compare directly.
        catalog = HERE.parent / 'abplus' / 'catalog'
        transfer_tasks = {'rooms': dict(name='rooms', label='普通房', tasks=str(catalog / 'mixture_positioning_rooms.json'),
                                        seconds=180.0, target=False),
                          'dodge': dict(name='dodge', label='躲子弹场', tasks=str(catalog / 'mixture_target_dodge.json'),
                                        seconds=30.0, target=True)}
        names = [] if args.duel_transfer in ('', 'none') else args.duel_transfer.split(',')
        if set(names) - set(transfer_tasks):
            p.error(f'--duel-transfer names are {sorted(transfer_tasks)} or none')
        duel_transfer = [transfer_tasks[n] for n in names]
    elif args.groups_file:
        from isaac_bridge.abplus_groups import describe_groups, load_groups
        groups = load_groups(args.groups_file)
        if args.episode_seconds is not None:
            p.error('--groups-file sets each group\'s deadline (episode_seconds)')
        if args.task_weights:
            p.error('--task-weights: set the group shares in the groups file')
    elif args.tasks_file != 'none':
        tasks = json.loads(Path(args.tasks_file).read_text(encoding='utf8'))
        if args.task_weights:
            tasks['weights'] = {k: float(v) for k, v in (kv.split('=') for kv in args.task_weights.split(','))}
        target = tasks.get('target')   # single-enemy aiming arena (C22): {type, variant, min_cells}
        tasks = {k: tasks[k] for k in ('weights', 'normal', 'boss')}
    v5 = args.reward_profile == 'combat-v5'
    firing = args.reward_profile == 'combat-hitrate-fire'
    hurting = args.reward_profile == 'combat-hitrate-hurt'   # stage 2 (C33): not invincible, hurt and death terms
    missing = args.reward_profile in ('combat-hitrate-miss', 'combat-hitrate-fire', 'combat-hitrate-hurt')
    walk = args.reward_profile in ('combat-hitrate-walk', 'combat-hitrate-miss', 'combat-hitrate-fire', 'combat-hitrate-hurt')
    hitrate = args.reward_profile in ('combat-hitrate', 'combat-hitrate-walk', 'combat-hitrate-miss', 'combat-hitrate-fire',
                                      'combat-hitrate-hurt')
    goal = args.reward_profile in GOAL_PROFILES   # goal-conditioned line (goal-hp, C44 goal-hp2): COMBAT as combat-hp2
    hp = args.reward_profile in ('combat-hp', 'combat-hp2', 'combat-hp2-camera') or goal   # C39, C41
    hp2 = args.reward_profile in ('combat-hp2', 'combat-hp2-camera') or goal
    if (args.freeze_shared or args.preserve_coef or args.preserve_target) and not goal:
        p.error('--freeze-shared / --preserve-coef / --preserve-target are goal-hp options')
    if groups and not goal and any(g.get('mode', 'combat') != 'combat' for g in groups):
        p.error('group modes other than combat need --reward-profile goal-hp')
    geometric = v5 or hitrate or hp   # combat-v5's observation, factored heads and auxiliary head
    if duel and not hitrate:
        p.error('--duel-file needs a combat-hitrate profile (the duel views carry its combat fields)')
    if duel and args.frames_per_decision != 2:
        p.error('--duel-file is calibrated at 2 frames per decision (A6; abplus_duel.DuelEnv holds actions 2 frames)')
    if args.start_bombs is None:
        args.start_bombs = '0' if hitrate else 'none' if hp else '0.5:3'
    start_bombs = None
    if args.start_bombs == '0':
        start_bombs = {'zero_prob': 1.0, 'max': 1}   # sample_bombs: always 0
    elif args.start_bombs != 'none':
        zero, most = args.start_bombs.split(':')
        start_bombs = {'zero_prob': float(zero), 'max': int(most)}
        if not (0 <= start_bombs['zero_prob'] <= 1 and 1 <= start_bombs['max'] <= 99):
            p.error('--start-bombs ZERO_PROB:MAX with 0 <= ZERO_PROB <= 1 and 1 <= MAX')
    episode_seconds = (args.episode_seconds if args.episode_seconds is not None else
                       HR['deadline_s'] if hitrate else 180.0 if hp else 120.0)
    if groups:   # the longest group deadline; every episode uses its own group's (abplus_worker)
        episode_seconds = max(g['seconds'] for g in groups)
    invincible = args.invincible if args.invincible is not None else (hitrate and not hurting and not duel)
    if duel and invincible:
        p.error('the duel has no invincible side')
    if not 1 <= episode_seconds <= 3600:
        p.error('--episode-seconds must be 1..3600')
    # combat-v5's potential must discount with the learner's gamma to stay potential-based.
    if v5:
        reward_options = dict(hit_hp=args.hit_hp if args.hit_hp is not None else V5['hit_hp'],
                              align=args.align_coef if args.align_coef is not None else V5['align'], gamma=args.gamma)
    elif hitrate:
        reward_options = dict(hit_hp=args.hit_hp if args.hit_hp is not None else HR['hit_hp'],
                              cost_hit=args.time_cost_hit if args.time_cost_hit is not None else HR['cost_hit'],
                              cost_miss=args.time_cost_miss if args.time_cost_miss is not None else HR['cost_miss'])
        if walk:   # the potential must discount with the learner's gamma to stay potential-based
            reward_options.update(align=args.align_coef if args.align_coef is not None else HRW['align'],
                                  gamma=args.gamma)
        if missing:
            reward_options.update(miss=args.miss_cost if args.miss_cost is not None else HRM['miss'])
        if args.hurt_rest_cost:   # the hurt that ends the episode costs the rest of this deadline
            reward_options.update(deadline_s=float(episode_seconds), hurt_rest=True)
        if hurting:   # a death costs the rest of this deadline (C33), or nothing more (--no-death-cost, C36)
            reward_options.update(deadline_s=float(episode_seconds))
            if args.no_death_cost:
                reward_options.update(death=False)
    elif goal:   # COMBAT options: combat-hp2 (all combat -1 per half heart, user decision 2026-09-30); GOTO: the
        # learner's gamma keeps the shaping potential-based
        reward_options = dict(combat=dict(hurt=args.hurt_cost if args.hurt_cost is not None else HPR2['hurt']),
                              goto=dict(gamma=args.gamma))
        # C45: the same price per half heart for GOTO, and the goal / damage rewards when given
        if args.hurt_cost is not None:
            reward_options['goto']['hurt'] = args.hurt_cost
        if args.goal_reward is not None:
            reward_options['goto']['goal'] = args.goal_reward
        if args.damage_hp is not None:
            reward_options['combat']['damage_hp'] = args.damage_hp
    elif hp:   # the cost per half heart is always recorded: the Room Buffer's d is the damage in half hearts
        reward_options = dict(hurt=args.hurt_cost if args.hurt_cost is not None else (HPR2 if hp2 else HPR)['hurt'])
    else:
        reward_options = {}
    if args.hurt_cost is not None and not hp:
        p.error('--hurt-cost is a combat-hp / combat-hp2 option')
    stat_noise = None
    if args.stat_noise:
        from isaac_bridge.abplus_worker import STAT_KEYS
        stat_noise = {}
        for part in args.stat_noise.split(','):
            key, _, value = part.partition('=')
            if key not in STAT_KEYS or not value:
                p.error(f'--stat-noise: unknown or empty {part!r} (keys {STAT_KEYS})')
            stat_noise[key] = float(value)
        # the base stats minus the half width stay above the bridge's floors (speed 0.1, damage 0.5, 0.5 shots per
        # second, a positive range; shot speed stops at the engine's 0.6)
        limits = dict(speed=0.9, damage=3.0, shot_speed=0.9, tears=2.2, range=6.0)
        bad = {k: a for k, a in stat_noise.items() if not 0 <= a < limits[k]}
        if bad:
            p.error(f'--stat-noise half widths out of range: {bad} (limits {limits})')
    lineage_mode = ((args.lineage_mode if args.lineage_mode is not None else
                     HRM['lineage_mode'] if missing else HPR['lineage_mode'] if hp else V5['lineage_mode'])
                    if geometric else None)
    if not (v5 or walk) and args.align_coef is not None:
        p.error('--align-coef is a combat-v5 / combat-hitrate-walk option')
    if not geometric and (args.hit_hp is not None or args.lineage_mode is not None or args.ent_coef_heads):
        p.error('--hit-hp/--lineage-mode/--ent-coef-heads are combat-v5 / combat-hitrate options')
    if not hitrate and (args.time_cost_hit is not None or args.time_cost_miss is not None):
        p.error('--time-cost-hit/--time-cost-miss are combat-hitrate options')
    if not missing and (args.miss_cost is not None or args.miss_cap is not None):
        p.error('--miss-cost/--miss-cap are combat-hitrate-miss / combat-hitrate-fire options')
    miss_cap = (args.miss_cap if args.miss_cap is not None else HRM['miss_cap']) if missing else 0
    if miss_cap < 0:
        p.error('--miss-cap must be >= 0')
    if args.no_death_cost and not hurting:
        p.error('--no-death-cost is a combat-hitrate-hurt option')
    if args.hurt_ends_episode and invincible:
        p.error('--hurt-ends-episode needs a mortal player (--no-invincible)')
    if args.hurt_rest_cost and not (args.hurt_ends_episode and args.reward_profile in ('combat-hitrate-miss',
                                                                                     'combat-hitrate-fire')):
        p.error('--hurt-rest-cost needs --hurt-ends-episode and combat-hitrate-miss / combat-hitrate-fire')
    architecture = dict(model_width=256, model_layers=4, model_heads=8, entity_queries=4, entity_width=128, head_width=256)
    changed = [k for k, v in architecture.items() if getattr(args, k) != v]
    if changed and args.resume:
        p.error(f'--resume keeps the checkpoint\'s architecture ({", ".join(changed)} given)')
    if changed and not geometric:
        p.error('the model size options are for combat-v5 / combat-hitrate models')
    if args.model_width % args.model_heads or args.entity_width % 4:
        p.error('--model-width must be a multiple of --model-heads and --entity-width of 4')
    if args.distill_from and not geometric:
        p.error('--distill-from needs a combat-v5 / combat-hitrate profile (GeometryPolicy, factored heads)')
    if not args.distill_from and (args.distill_rl_coef or args.distill_teacher_acts or args.distill_coef_end is not None):
        p.error('--distill-* options need --distill-from')
    if args.distill_temperature <= 0 or args.distill_coef < 0 or args.distill_rl_coef < 0:
        p.error('--distill-temperature must be > 0 and the distillation weights >= 0')
    if args.lr_schedule == 'constant':
        if args.lr_final is not None:
            p.error('--lr-final needs --lr-schedule linear or cosine')
        lr_final = args.learning_rate
    else:
        if not args.game_hours:
            p.error('--lr-schedule linear / cosine needs --game-hours (the budget it decays over)')
        lr_final = args.lr_final if args.lr_final is not None else 0.1 * args.learning_rate
        if not 0 < lr_final <= args.learning_rate:
            p.error('--lr-final must be in (0, --learning-rate]')
    head_names = ('move', 'shoot', 'bomb', 'item')
    head_ent_coefs = None
    if args.ent_coef_heads:
        given = {k: float(v) for k, v in (kv.split('=') for kv in args.ent_coef_heads.split(','))}
        if set(given) - set(head_names):
            p.error(f'--ent-coef-heads names are {head_names}')
        head_ent_coefs = [given.get(k, args.ent_coef) for k in head_names]
    if args.block_moves and not geometric:
        p.error('--block-moves needs a combat-v5 / combat-hitrate profile (their frames carry move_block)')
    if args.greedy_actors:
        if not (0 < args.greedy_actors <= args.envs and 0 < args.greedy_temperature <= 1):
            p.error('--greedy-actors must be 1..--envs and --greedy-temperature in (0, 1]')
        if args.sampler != 'graph' or not args.async_train:
            p.error('--greedy-actors needs --sampler graph and --async-train (the decoupled weights correct the tempered '
                    'behaviour)')
    plr_levels = None
    if target and target.get('arms') and args.room_sampling != 'mixture':
        p.error("the tasks file's target arms (tier 6) need --room-sampling mixture")
    plr_weights = None
    if args.room_sampling == 'plr':
        if groups:   # C39: every group draws its (room, arm) levels by PLR; the groups keep their step budget
            from isaac_bridge.plr import group_levels
            plr_levels, plr_weights = group_levels(groups)
        elif not tasks:
            p.error('--room-sampling plr needs a --tasks-file or a --groups-file')
        else:
            from isaac_bridge.plr import mixture_levels
            plr_levels, plr_weights = mixture_levels(tasks), tasks['weights']
    elif args.plr_by_steps:
        p.error('--plr-by-steps needs --room-sampling plr')
    buffer_options = None
    if args.room_sampling == 'buffer':   # C39
        if not groups or not hp:
            p.error('--room-sampling buffer needs --groups-file and --reward-profile combat-hp (its hurt term gives d)')
        from isaac_bridge.room_buffer import RoomBuffer
        buffer_options = dict(capacity=args.buffer_capacity, fresh_share=args.buffer_fresh, alpha=args.buffer_alpha,
                              lam=args.buffer_lambda, d0=args.buffer_d0, eta=args.buffer_eta, eps=args.buffer_eps)
        try:
            RoomBuffer([g['name'] for g in groups], **buffer_options)
        except ValueError as exc:
            p.error(str(exc))
    if args.resume:
        args.resume = resolve_checkpoint(args.resume)
    if args.warm_start:
        args.warm_start = resolve_checkpoint(args.warm_start)
    if args.distill_from:
        args.distill_from = resolve_checkpoint(args.distill_from)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'out': str(args.out.resolve()), 'backend': 'abplus-1.06-linux',
              'task': ({'mixture': tasks['weights'], 'normal_rooms': len(tasks['normal']),
                        'boss_rooms': len(tasks['boss'])} if tasks else
                       'duel (duel)' if duel else
                       'parallel task groups (groups)' if groups else 'monstro-arena (sim seeds)'),
              'duel': (dict(file=str(args.duel_file), npc=duel['duel'].get('npc'), hp=duel['duel'].get('hp'),
                            min_cells=duel['duel'].get('min_cells'),
                            arms=[dict(name=a.get('name'), weight=a.get('weight', 1.0), rooms=len(a['rooms']),
                                       obstacles=bool(a.get('obstacles'))) for a in duel['duel'].get('arms', ())],
                            slots='two per game: 2k the player, 2k + 1 the duel NPC; one policy acts for both (self-play)',
                            views='first-person per side (abplus_duel.duel_views)', reward='the profile, one instance per side',
                            games=args.envs // 2, starts='never on one row or column (same_line false)',
                            transfer=duel_transfer) if duel else None),
              'duel_transfer': duel_transfer,
              'tasks_spec': tasks,
              'groups': describe_groups(groups, args.budget) if groups else None,
              'groups_file': str(args.groups_file.resolve()) if groups else None,
              'observation_transport': 'json (bridge v1)' if args.json_obs else f'binary (bridge v2, {BRIDGE_VERSION})',
              'start_bombs': start_bombs,
              'room_sampling': (dict(method='plr', levels=len(plr_levels), beta=args.plr_beta,
                                     staleness=args.plr_staleness, floor=args.plr_floor,
                                     score='positive value loss (mean max(GAE, 0) per episode)',
                                     kinds=('the groups keep their step budget; PLR picks the (room, arm) within the '
                                            'group' if groups else
                                            'each kind keeps its mixture weight; PLR picks the room within the kind'),
                                     kind_weights=plr_weights,
                                     shares=('of the steps: start probability = PLR probability / the room\'s measured '
                                             'episode length (mean of its first 10 episodes, then an exponential average)'
                                             if args.plr_by_steps else 'of the episode starts'))
                                if plr_levels else
                                dict(method='room buffer (C39, isaac_bridge/room_buffer.py)', level='seed', **buffer_options,
                                     success='clear (death and deadline 0)',
                                     priority='4p(1-p) + lam p clip(d/d0, 0, 1) + eta S + eps; p = EMA of success, d = EMA of '
                                              'the half hearts lost in cleared episodes, S = min(1, group episodes since the '
                                              'last play / capacity)',
                                     replay_start='probability proportional to priority / L\', the seed\'s EMA episode length '
                                                  'shrunk towards its family\'s: (n L + 4 L_family) / (n + 4), n its plays',
                                     fresh='the slot\'s own next training seed; fresh or replay by each slot\'s step budget '
                                           'per group (GroupScheduler over fresh, replay with shares fresh_share, rest)',
                                     admission='after its episode; a full buffer replaces its lowest 4p(1-p) + lam p clip(d/d0) '
                                               'seed if the new one is higher',
                                     prior='a seed starts from its family\'s (group, room, arm) EMAs')
                                if buffer_options else 'mixture weights by seed'),
              'reward_profile': args.reward_profile, 'collection_sampler': args.sampler,
              'learning_rate_schedule': (f'{args.learning_rate:g} constant' if args.lr_schedule == 'constant' else
                                         f'{args.lr_schedule} from {args.learning_rate:g} to {lr_final:g} over this run\'s '
                                         f'{args.game_hours:g} game hours'),
              'expert_imitation': (dict(coef=args.expert_coef, until=args.expert_until,
                                        schedule=f'{args.expert_coef:g} falling linearly to 0 at {args.expert_until:g} of the run',
                                        labels='GOTO: the scripted navigator (abplus_nav.descent_move); COMBAT: the approach '
                                               'moves once stuck against the terrain for 3 s (transformer_obs.STUCK_FRAMES)')
                                   if args.expert_coef else None),
              'diagnosis': (dict(expert_only=args.expert_only, grad_probe=args.grad_probe, max_grad_norm=args.max_grad_norm)
                            if args.expert_only or args.grad_probe or args.max_grad_norm is not None else None),
              'learning_rate_groups': (dict(
                  goal=f'goal-line parameters: {args.lr_schedule} from {args.learning_rate:g} to {lr_final:g}',
                  shared=(f'shared parameters: {args.shared_lr_warmup_updates} updates linear from '
                          f'{args.shared_lr_warmup_start if args.shared_lr_warmup_start is not None else (args.shared_lr_final if args.shared_lr_final is not None else 0.1 * args.shared_learning_rate):g}, '
                          f'then {args.lr_schedule} from {args.shared_learning_rate:g} to '
                          f'{args.shared_lr_final if args.shared_lr_final is not None else 0.1 * args.shared_learning_rate:g}'))
                                       if args.shared_learning_rate else None),
              'update_schedule': ('asynchronous: each update trains while the next rollout is collected by the '
                                  'weights it started from (one update of policy lag); decoupled PPO objective: '
                                  'ratio clipped against the update start, samples weighted by '
                                  'pi_start/pi_behaviour truncated at 2' if args.async_train else 'synchronous'),
              'reward': (dict(describe_goal(**reward_options), profile=args.reward_profile) if goal else
                         ({**describe_hp2(**reward_options), 'profile': args.reward_profile} if hp2
                          else describe_hp(**reward_options)) if hp
                         else describe_v5(**reward_options) if v5
                         else describe_hitrate_hurt(**reward_options, miss_cap=miss_cap) if hurting
                         else describe_hitrate_fire(**reward_options, miss_cap=miss_cap) if firing
                         else describe_hitrate_miss(**reward_options, miss_cap=miss_cap) if missing
                         else describe_hitrate_walk(**reward_options) if walk
                         else describe_hitrate(**reward_options) if hitrate
                         else describe_v4() if args.reward_profile == 'combat-v4'
                         else describe_v3()
                         if args.reward_profile == 'combat-v3' else describe_reward()
                         if args.reward_profile == 'combat-v2'
                         else 'combat-v1: legacy hurt/hit/damage/clear, win +2 + speed bonus, timeout -1'),
              'observation': {**observation_options(args.reward_profile),
                              'combat_fields': list(observation_options(args.reward_profile)['combat_state'] or ())},
              'schema': DEADLINE_SCHEMA, 'history': 64, 'entity_capacity': 256,
              'timeout_semantics': ('truncation (bootstrap)'
                                    if args.reward_profile in ('combat-v4', 'combat-v5', 'combat-hitrate', 'combat-hitrate-walk',
                                                               'combat-hitrate-miss', 'combat-hitrate-fire',
                                                               'combat-hitrate-hurt')
                                    else 'termination'),
              'invincible': invincible, 'miss_cap': miss_cap, 'target': target,
              'game_opengl': 'Mesa software (llvmpipe)' if args.software_gl else 'the display\'s driver',
              'episode_end': ('the first damage ends the episode (outcome hurt, terminal, no bootstrap); evaluation '
                              'keeps the game rules' if args.hurt_ends_episode else 'the game rules'),
              'reward_options': reward_options, 'lineage_mode': lineage_mode,
              'model': (dict(policy='GeometryPolicy', heads=dict(zip(head_names, (9, 5, 2, 2))),
                             size=({k: getattr(args, k) for k in architecture} if not args.resume
                                   else 'the checkpoint\'s (policy_kwargs in model.zip)'),
                             entropy=dict(zip(head_names, head_ent_coefs)) if head_ent_coefs else f'{args.ent_coef:g} on the summed entropy',
                             aux_head='5-way aim label + d_fire/40 regression from the actor features, loss weight '
                                      f'{args.aux_coef:g} (regression x {args.aux_distance_coef:g}'
                                      f'{", aim classes balanced aligned / unaligned" if args.aux_balance else ""})',
                             critic_input=('fire_distance = d_walk/40 (walking distance), value branch only' if walk
                                           else 'fire_distance = d_fire/40, value branch only'))
                        if geometric else 'MultiInputPolicy, joint 45-way move x shoot head'),
              'distillation': (dict(teacher=str(args.distill_from), kl='sum over the four heads of KL(teacher || student) '
                                    'under the sample\'s action masks', coef=args.distill_coef,
                                    coef_end=args.distill_coef_end, temperature=args.distill_temperature,
                                    rl_coef=args.distill_rl_coef, teacher_collects_until_update=args.distill_teacher_acts)
                               if args.distill_from else None),
              'max_episode_seconds': episode_seconds, 'frames_per_decision': args.frames_per_decision,
              'decisions_per_game_second': GAME_FPS / args.frames_per_decision,
              'game_time': f'decisions * {args.frames_per_decision} logic frames / 30 frames per game second',
              'start_randomization': start_randomization, 'evaluation_start': 'full HP',
              'minibatch': 'segments: frame-deduplicated, advantages normalised per batch_size, gradient-accumulated',
              'instances_per_env': 2, 'resume_semantics': 'weights/optimizer/RNG; fresh episodes',
              'precision': 'strict FP32 (TF32 off)', 'eval_seeds': 'first eval_seeds_count of eval_seeds_file',
              'evaluation_bombs': (f'{args.eval_bombs} (fixed)' if args.eval_bombs is not None
                                   else 'the training start-bomb draw, fixed per seed' if start_bombs else 'the game\'s 1'),
              'evaluation_stats': 'the base stats (training --stat-noise is not applied)'}
    print(json.dumps(config, indent=2))
    if not args.train:
        print('PREPARED ONLY: no workers, AB+ instances, model or optimizer created.')
        return
    import torch
    from stable_baselines3.common.logger import configure
    from isaac_bridge.transformer_policy import CombatTransformer, GeometryPolicy
    torch.set_num_threads(args.torch_threads)
    from isaac_bridge.abplus_vec import AbplusFrameVecEnv
    from isaac_bridge.gpu_buffer import GpuHistoryRolloutBuffer
    from isaac_bridge.gpu_ppo import GpuMaskablePPO, LearningRateSchedule
    # C36: a schedule over this run's budget (SB3 progress_remaining); constant keeps the plain float.
    learning_rate = (args.learning_rate if args.lr_schedule == 'constant'
                     else LearningRateSchedule(args.lr_schedule, args.learning_rate, lr_final))
    args.out.mkdir(parents=True, exist_ok=False)
    config['sources_sha256'] = source_hashes()
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    env = AbplusFrameVecEnv(args.envs, args.seed, args.chunks, device=args.device, start_randomization=start_randomization,
                            mode=args.mode, name=args.name, port=args.port, nice=args.nice, tasks=tasks,
                            binary_obs=not args.json_obs, reward_profile=args.reward_profile,
                            recycle_episodes=args.recycle_episodes, recycle_rss_mib=args.recycle_rss_mib,
                            start_bombs=start_bombs, plr_levels=plr_levels,
                            reward_options=reward_options, lineage_mode=lineage_mode,
                            max_episode_frames=int(round(episode_seconds * GAME_FPS)), invincible=invincible,
                            miss_cap=miss_cap, target=target, hurt_ends=args.hurt_ends_episode,
                            groups=groups, budget=args.budget, duel=duel, frames_per_decision=args.frames_per_decision,
                            room_buffer=(dict(capacity=args.buffer_capacity, fresh_share=args.buffer_fresh)
                                         if buffer_options else None),
                            stat_noise=stat_noise)
    try:
        completed = updates = 0
        if args.resume:
            model, state = resume(args.resume, env, args.device, args.n_steps)
            completed, updates = state['completed_episodes'], state['updates']
            model.batch_size, model.n_epochs, model.ent_coef = args.batch_size, args.n_epochs, args.ent_coef
            model.aux_distance_coef, model.aux_balance = args.aux_distance_coef, args.aux_balance
            if args.lr_schedule != 'constant':
                model.learning_rate = model.lr_schedule = learning_rate
            if geometric != hasattr(model.policy, 'aux_outputs'):
                raise ValueError('combat-v5 / combat-hitrate and the checkpoint policy (GeometryPolicy) must go together')
        else:
            model = GpuMaskablePPO(
                GeometryPolicy if geometric else 'MultiInputPolicy', env, n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma, ent_coef=args.ent_coef, learning_rate=learning_rate,
                rollout_buffer_class=GpuHistoryRolloutBuffer,
                policy_kwargs=dict(features_extractor_class=CombatTransformer,
                                   features_extractor_kwargs=dict(features_dim=args.model_width, layers=args.model_layers,
                                                                  heads=args.model_heads,
                                                                  entity_queries=args.entity_queries,
                                                                  entity_dim=args.entity_width),
                                   net_arch=dict(pi=[args.head_width], vf=[args.head_width]), normalize_images=False,
                                   **(dict(lr_groups=True) if args.shared_learning_rate else {})),
                device=args.device, seed=args.seed, verbose=1)
        if args.shared_learning_rate:
            # C44 (design 4.8): the shared parameters warm up to their own peak, the goal-line ones follow --learning-rate
            from isaac_bridge.gpu_ppo import GroupLearningRate
            shared_final = args.shared_lr_final if args.shared_lr_final is not None else 0.1 * args.shared_learning_rate
            model.lr_groups = dict(
                shared=GroupLearningRate(args.lr_schedule, args.shared_learning_rate, shared_final,
                                         args.shared_lr_warmup_updates, args.shared_lr_warmup_start),
                goal=GroupLearningRate(args.lr_schedule, args.learning_rate, lr_final))
            if [g.get('name') for g in model.policy.optimizer.param_groups] != ['shared', 'goal']:
                raise RuntimeError('--shared-learning-rate needs the two optimizer groups of the policy (lr_groups)')
        model.expert_coef, model.expert_until = args.expert_coef, args.expert_until   # C45
        model.imitation_only, model.grad_probe = args.expert_only, args.grad_probe   # A15 diagnosis
        if args.max_grad_norm is not None:
            model.max_grad_norm = args.max_grad_norm
        model.micro_batch_size, model.segment_length = args.micro_batch, args.segment_length
        model.aux_coef = args.aux_coef if geometric else 0.0
        model.aux_distance_coef = args.aux_distance_coef
        model.aux_balance = args.aux_balance
        model.block_moves = args.block_moves
        model.greedy_actors = args.greedy_actors
        if args.kl_probe:
            model.kl_probe, model.kl_probe_path = True, str(args.out / 'kl_probe.jsonl')
        model.head_ent_coefs = head_ent_coefs
        model.async_training = args.async_train
        if args.sampler == 'graph':
            from functools import partial
            from isaac_bridge.graph_sampler import GraphFrameSampler
            model.sampler_class = partial(GraphFrameSampler, check_every=args.graph_check_every,
                                          greedy_actors=args.greedy_actors, greedy_temperature=args.greedy_temperature)
        if args.warm_start:
            migration = warm_start_model(model, args.warm_start)
            if args.warm_start_optimizer:
                # C41: the Adam moments of the checkpoint, matched by parameter name (a goal-line policy has more
                # parameters); each update sets this run's learning rate.
                from isaac_bridge.training_session import warm_start_optimizer
                migration.update(warm_start_optimizer(model, args.warm_start,
                                                      'goal' if args.warm_start_optimizer_shared else None))
            if args.warm_start_optimizer_shared:
                # C44: the shared parameters' moments from C39's final (frozen in stage 1, they have none since)
                from isaac_bridge.training_session import warm_start_optimizer
                migration['shared_optimizer'] = dict(checkpoint=str(args.warm_start_optimizer_shared),
                                                     **warm_start_optimizer(model, args.warm_start_optimizer_shared, 'shared'))
            (args.out / 'migration.json').write_text(json.dumps(migration, indent=2), encoding='utf8')
        if goal:
            # goal line (user decisions 2026-09-30): the item head by the active item's charge (no item is given); stage 1
            # freezes everything but the goal-line modules; the preservation KL's teacher is the policy as migrated (on
            # COMBAT frames exactly the checkpoint's function)
            import copy
            from isaac_bridge.training_session import _goal_line_key
            model.item_available = True
            if args.freeze_shared:
                frozen = 0
                for name, param in model.policy.named_parameters():
                    if not _goal_line_key(name):
                        param.requires_grad_(False)
                        frozen += param.numel()
                model.loss_tasks = 'goto'
                (args.out / 'freeze.json').write_text(json.dumps(dict(frozen_parameters=frozen, trainable=sum(
                    q.numel() for q in model.policy.parameters() if q.requires_grad)), indent=2), encoding='utf8')
            if args.preserve_coef > 0:
                teacher = copy.deepcopy(model.policy)
                teacher.set_training_mode(False)
                for param in teacher.parameters():
                    param.requires_grad_(False)
                model.preserve_teacher, model.preserve_coef = teacher, args.preserve_coef
                model.preserve_target = args.preserve_target
        if args.distill_from:
            from isaac_bridge.gpu_ppo import load_frozen_policy
            model.teacher = load_frozen_policy(args.distill_from / 'model.zip', model.device, env.observation_space,
                                               env.action_space)
            model.distill_coef, model.distill_coef_end = args.distill_coef, args.distill_coef_end
            model.distill_temperature, model.rl_coef = args.distill_temperature, args.distill_rl_coef
            model.teacher_acts_updates = args.distill_teacher_acts
        parameters = sum(p.numel() for p in model.policy.parameters())
        (args.out / 'model_size.json').write_text(json.dumps(dict(
            parameters=parameters, policy_kwargs={k: (v.__name__ if isinstance(v, type) else v)
                                                  for k, v in model.policy_kwargs.items()},
            teacher_parameters=sum(p.numel() for p in model.teacher.parameters()) if model.teacher is not None else None),
            indent=2, default=str), encoding='utf8')
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        env.training_seeds = True
        plr = None
        if plr_levels:
            from isaac_bridge.plr import PrioritizedLevels
            saved = args.resume / 'plr.json' if args.resume else None
            if saved is not None and saved.exists():
                # The kind weights come from --tasks-file / --task-weights / --groups-file, not from the checkpoint.
                plr = PrioritizedLevels.from_state(json.loads(saved.read_text(encoding='utf8')), plr_weights)
                if plr.levels != [tuple(level) for level in plr_levels]:
                    raise ValueError('PLR rooms of the checkpoint differ from --tasks-file / --groups-file')
                plr.by_steps = args.plr_by_steps
            else:
                plr = PrioritizedLevels(plr_levels, plr_weights, args.plr_beta, args.plr_staleness, args.plr_floor,
                                        by_steps=args.plr_by_steps)
            env.set_level_probabilities(plr.probabilities())
        buffer = None
        if buffer_options:
            from isaac_bridge.room_buffer import RoomBuffer
            saved = args.resume / 'room_buffer.json' if args.resume else None
            if saved is not None and saved.exists():
                buffer = RoomBuffer.from_state(json.loads(saved.read_text(encoding='utf8')), **buffer_options)
                if buffer.names != [g['name'] for g in groups]:
                    raise ValueError('the Room Buffer groups of the checkpoint differ from --groups-file')
            else:
                buffer = RoomBuffer([g['name'] for g in groups], **buffer_options)
            env.set_room_buffer(*buffer.table())
        session = session_class()(config, args.out, completed, updates,
                                  start_timesteps=model.num_timesteps if args.resume else 0, plr=plr, buffer=buffer,
                                  groups=groups)
        model.session = session
        agents = 2 if duel else 1   # duel: game hours count each game once, its two slots decide together
        budget = (int(args.game_hours * 3600 * GAME_FPS / FRAMES_PER_DECISION) * agents if args.game_hours
                  else (args.episodes - completed) * 1800) + args.envs
        model.learn(total_timesteps=budget, callback=session, reset_num_timesteps=not bool(args.resume))
        session.finish(model)
    finally:
        env.close()


if __name__ == '__main__':
    main()
