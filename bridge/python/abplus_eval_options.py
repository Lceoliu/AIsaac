"""Evaluation of the goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md, E-GOTO / E-CHAIN, M1 chain v0): whole option
sequences of one group (abplus_options: combat, combat_goto, goto_empty, chain) on held-out seeds, one results line per
seed with a record per option.

The checkpoint plays the COMBAT options (greedy, or --stochastic). The GOTO options are played by the checkpoint when it is a
goal-line policy (goal-hp), else, or with --scripted-goto, by the scripted navigator walking down the d_geo field
(abplus_nav.descent_move; the GOTO reference and the M1 chain v0). The environment is the checkpoint's own observation
(a C39 checkpoint sees no goal fields; the sequence still sets the bridge's goal check and the walking distance).
--control-b (chain groups): after a chain that entered B, B's room is loaded again by goto with the player's half hearts and
bombs at B's entry and the checkpoint plays that single COMBAT option (the E-CHAIN control: the same room without the
continuous run; its spawns are not seed-paired with the chain's B, EXPERIMENTS.md A9).

Per option: task, source, stratum (GOTO_POSITION), d0 (d_geo at the start, px), outcome, decisions, frames, walked px,
half hearts lost, the option's return under the checkpoint's rewards (combat-hp2 / GotoReward), the player's half hearts
and bombs at its start.

--replays N writes the first N seeds' sequences as raw-observation replays (replays/seed-<seed>.jsonl.gz, abplus_eval.py's
format abplus-raw-obs-v1 plus, per step, the option [index, task, target x, target y, goal radius]) for
abplus_replay_view.py.

usage: python abplus_eval_options.py --checkpoint DIR --groups-file FILE --group NAME --seeds range:START:COUNT --out DIR
       [--instances 4] [--scripted-goto] [--control-b] [--stochastic --sample-seed 0] [--port 27600] [--name ev]
       [--replays N]
"""
import argparse
import collections
import gzip
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from abplus_eval import CachedPolicy, load_policy, rss_mib
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_groups import load_groups, sample_group_hp
from isaac_bridge.abplus_geometry import MOVES, blocked_moves
from isaac_bridge.abplus_nav import descent_move, goal_field
from isaac_bridge.abplus_options import OptionSequence
from isaac_bridge.abplus_reward import REWARDS, GotoReward
from isaac_bridge.abplus_tasks import Task, TaskSampler
from isaac_bridge.abplus_worker import GOAL_PROFILES, observation_options, sample_bombs
from isaac_bridge.steam_watch import steam_running
from isaac_bridge.transformer_obs import factored_masks, factored_to_joint

SUCCESS = ('win', 'goal')


class FixedRoom:
    """TaskSampler stand-in: always one room (the control B)."""

    def __init__(self, variant, entrance):
        self.task = Task('normal', int(variant), int(entrance))

    def choose(self, seed, retry=0):
        return self.task


def health(obs):
    p = obs['players'][0]
    return p['hearts'] + p.get('soul', 0)


def rewards_of(config):
    profile = config['reward_profile']
    options = config.get('reward_options', {})
    gamma = float(config.get('gamma', 0.999))
    if profile in GOAL_PROFILES:
        return REWARDS[profile](**options.get('combat', {})), GotoReward(**{'gamma': gamma, **options.get('goto', {})})
    return REWARDS[profile](**options) if profile in REWARDS else None, GotoReward(gamma=gamma)


def make_env(port, config, group, mode):
    env = AbplusTransformerEnv(port=port, max_episode_frames=int(group['frames']),
                               **observation_options(config['reward_profile']), deadline_s=float(group['seconds']),
                               frames_per_decision=int(config.get('frames_per_decision', 2)),
                               history=config['history'], entity_capacity=config['entity_capacity'])
    env.bridge.binary_obs = True
    env.combat_multi_room = True   # a COMBAT option goes on when the player leaves its room (EXPERIMENTS.md A13)
    if config.get('lineage_mode') is not None:
        env.bridge.lineage_mode = int(config['lineage_mode'])
    env.bridge.tasks = TaskSampler(group['spec']['weights'], group['spec']['normal'], group['spec']['boss'], None)
    env.bridge.reset_mode = {'chain': 'chain', 'goto_empty': 'goto_room'}.get(mode)
    if mode == 'chain':
        env.bridge.chain_rooms = frozenset(int(v) for v in group['spec']['normal'])
    return env


def act(env, cached, seq, args, goal_policy):
    """The joint action of this decision: scripted navigation for a GOTO option (non-goal checkpoint or --scripted-goto),
    else the checkpoint's."""
    option = seq.option
    if option['task'] != 'combat' and (args.scripted_goto or not goal_policy):
        nav = option['state']['nav']
        px, py = env.raw_obs['players'][0]['pos']
        move = descent_move(px, py, nav) if nav is not None else 0
        return np.array([move * 5, 0, 0])
    label = env.history.frames[-1].get('expert_move') if env.history.frames else None
    if option['task'] == 'combat' and getattr(args, 'finisher', False):
        # the floor runner's scripted finisher (a room the policies leave uncleared, e.g. a lone Horf): walk along the
        # frame's approach moves (shorter walk to a firing position), shoot the aim label once aligned; the shield
        # still vets the move
        frame = env.history.frames[-1]
        approach = np.asarray(frame['approach'], np.float64)
        move = int(np.argmax(approach)) if approach.any() else 0
        shot = int(round(float(frame['aim_label'])))
        if getattr(args, 'shield', False):
            from isaac_bridge.abplus_shield import shield
            prefer = approach.copy()
            prefer[move] += 1.0
            move, _ = shield(getattr(env.history, 'before_previous', None), env.raw_obs, prefer, move)
        return np.array([move * 5 + shot, 0, 0])
    if option['task'] != 'combat' and args.expert_actions:
        # C45 check: GOTO by the label the training loss reads (the newest frame's expert_move), nothing else
        return np.array([int(np.argmax(label)) * 5 if label is not None and label.any() else 0, 0, 0])
    mask = factored_masks(env.action_masks())
    dist = cached.distribution(env.history.frames[-1], mask)
    action = dist.get_actions(deterministic=not args.stochastic)[0].cpu().numpy()
    if option['task'] == 'combat' and getattr(args, 'shield', False):
        # the floor runner's --shield (isaac_bridge.abplus_shield): a dangerous move gives way to the best safe one
        from isaac_bridge.abplus_shield import shield
        move, hit = shield(getattr(env.history, 'before_previous', None), env.raw_obs,
                           dist.distributions[0].logits[0].cpu().numpy(), int(action[0]))
        args.shield_stats['decisions'] += 1
        if hit:
            action = action.copy()
            action[0] = move
            args.shield_stats['interventions'] += 1
    joint = factored_to_joint(action)
    if getattr(args, 'no_bombs', False):   # the floor runner's --no-bombs: the policy's bomb press is dropped
        joint = joint.copy()
        joint[1] = 0
    if option['task'] == 'combat' and args.expert_unstick and label is not None and label.any():
        joint = joint.copy()   # C45 check: the stuck label's move replaces the policy's move, its shot stays
        joint[0] = int(np.argmax(label)) * 5 + int(joint[0]) % 5
    return joint


def check_field(nav):
    """C45 check of a GOTO_POSITION field: (cells, Bellman violations, descent failures). Bellman: the field equals an
    independent Bellman-Ford over the same grid (the step cost, plus HAZARD_COST into a hazard; fires blocked; the target
    0), on exactly the cells that reach the target. Descent: from every reachable cell centre the scripted move goes to a
    legal neighbour of strictly lower value, and the walk ends on the target."""
    from isaac_bridge.abplus_nav import HAZARD_COST, CELL as NAV_CELL
    origin, field, neighbours = nav['origin'], nav['field'], nav['neighbours']
    source = [c for c, d in field.items() if d == 0.0]
    if len(source) != 1:
        return len(field), len(field), len(field)
    blocked, hazards = nav['blocked'], nav['hazards']
    ref = {c: math.inf for c in neighbours if c not in blocked}
    ref[source[0]] = 0.0
    for _ in range(len(ref)):
        changed = False
        for c, out in neighbours.items():
            if c in blocked or ref.get(c, math.inf) == math.inf:
                continue
            for n, cost in out:
                if n in blocked:
                    continue
                nd = ref[c] + cost + (HAZARD_COST if n in hazards else 0.0)
                if nd < ref[n] - 1e-9:
                    ref[n] = nd
                    changed = True
        if not changed:
            break
    reach = {c for c, d in ref.items() if d < math.inf}
    bellman = sum(1 for c in reach if c not in field or abs(field[c] - ref[c]) > 1e-6) + len(set(field) - reach)
    failures = 0
    for c in field:
        cell, steps = c, 0
        while field[cell] > 0.0 and steps <= len(field):
            move = descent_move(origin[0] + NAV_CELL * cell[0], origin[1] + NAV_CELL * cell[1], nav)
            nxt = (cell[0] + MOVES[move][0], cell[1] + MOVES[move][1])
            legal = {n for n, _ in neighbours[cell]}
            if move == 0 or nxt not in legal or nxt not in field or field[nxt] >= field[cell]:
                failures += 1
                break
            cell, steps = nxt, steps + 1
        else:
            if field[cell] > 0.0:
                failures += 1
    return len(field), bellman, failures


def option_tag(seq):
    """A replay step's option: [index, task, target x, target y, goal radius] (no target: None, None)."""
    option, target = seq.option, seq.option.get('target')
    return [seq.index, option['task'], *(list(map(float, target)) if target is not None else [None, None]),
            float(seq.group.get('goal_radius', 20.0))]


def play_option(env, cached, seq, args, goal_policy, writer=None):
    """Play the current option to its end; its record. writer: the replay's file (a line per step)."""
    option = seq.option
    raw = env.raw_obs
    start = dict(hp=health(raw), bombs=raw['players'][0]['bombs'])
    rec = dict(task=option['task'], source=option['source'], stratum=option.get('stratum'),
               d0=round(option['d0'], 1) if option.get('d0') is not None else None, start=start,
               grid=[int(raw['terrain']['width']), int(raw['terrain']['height'])])
    prev = tuple(raw['players'][0]['pos'])
    walked, elapsed = 0.0, 0
    checks = collections.Counter()
    if args.check_expert and option['task'] == 'goto_position' and option['state'].get('nav') is not None:
        checks['field_cells'], checks['bellman_violations'], checks['descent_failures'] = check_field(option['state']['nav'])
    free_step = 5.4 * env.bridge.action_repeat   # px per decision at full speed
    while True:
        before = env.raw_obs
        label = env.history.frames[-1].get('expert_move') if env.history.frames else None
        if args.check_expert and label is not None:
            px, py = before['players'][0]['pos']
            if option['task'] != 'combat':
                nav = option['state'].get('nav')
                mine = int(np.argmax(label)) if label.any() else 0
                direct = descent_move(px, py, nav) if nav is not None else 0
                fresh = 0
                if nav is not None:
                    rebuilt = goal_field(before, nav['target'])
                    if nav.get('exit') is not None:
                        rebuilt['exit'] = nav['exit']
                    fresh = descent_move(px, py, rebuilt)
                checks['label_steps'] += 1
                checks['label_vs_direct_mismatch'] += mine != direct
                checks['label_vs_fresh_field_mismatch'] += mine != fresh
                checks['label_one_hot'] += float(label.sum()) == 1.0
            elif label.any():
                blocked = blocked_moves(before)
                checks['stuck_label_frames'] += 1
                checks['stuck_label_into_blocked'] += any(label[m] > 0 and blocked[m] for m in range(len(MOVES)))
        action = act(env, cached, seq, args, goal_policy)
        obs, _, terminated, truncated, info = env.step(action)
        raw = env.raw_obs
        if raw['room']['room_idx'] == before['room']['room_idx'] and int(action[0]) // 5:
            checks['move_steps'] += 1
            checks['stuck_steps'] += math.dist(before['players'][0]['pos'], raw['players'][0]['pos']) < 0.3 * free_step
        if writer is not None:
            writer.write(json.dumps({'action': np.asarray(action).tolist(), 'obs': raw, 'option': option_tag(seq)},
                                    separators=(',', ':')) + '\n')
        if option.get('distance_fn') is not None and not goal_policy:
            option['distance_fn'](raw)   # the history has no goal fields: keep d_geo current for the reward
        outcome = seq.outcome(raw, info['outcome'])
        seq.step(raw, outcome, int(info['elapsed_frames']) - elapsed)
        elapsed = int(info['elapsed_frames'])
        pos = tuple(raw['players'][0]['pos'])
        walked += math.dist(prev, pos)
        prev = pos
        if terminated or truncated:
            break
    totals = seq.totals()
    reward = seq.combat_reward if option['task'] == 'combat' else seq.goto_reward
    rec.update(outcome=outcome, decisions=option['decisions'], frames=elapsed, walked=round(walked, 1),
               hurt=max(0, start['hp'] - health(raw)),
               d_end=round(option['state']['distance'], 1) if option['state'].get('distance') is not None else None,
               ret=round(reward.scale * float(totals.sum()), 4) if totals is not None else None, **checks)
    return rec


def run_sequence(env, cached, group, mode, seed, args, config, goal_policy, writer=None):
    t0 = time.monotonic()
    # C45: the start the run trains with, fixed per seed (bombs: the run's start_bombs, 1 without; half hearts: the group's
    # start_hp, full without); every earlier run keeps 1 bomb and full HP
    start = dict(arena_seed=int(seed), bombs=sample_bombs(seed, config.get('start_bombs')))
    hp = sample_group_hp(seed, group.get('start_hp'))
    if hp is not None:
        start['start'] = (hp, 1.0)
    obs, info = env.reset(options=start)
    grid = [int(env.raw_obs['terrain']['width']), int(env.raw_obs['terrain']['height'])]
    seq = OptionSequence(dict(group), seed, info, rewards_of(config))
    options = []
    started = seq.start(env, first=True)
    if writer is not None:
        writer.write(json.dumps({'metadata': {'seed': int(seed), 'format': 'abplus-raw-obs-v1', 'goal': True,
                                              'group': group['name'], 'mode': mode, 'chain': info.get('chain')}}) + '\n')
        writer.write(json.dumps({'action': None, 'obs': env.raw_obs, 'option': option_tag(seq) if started else None},
                                separators=(',', ':')) + '\n')
    while started is not None:
        cached.reset()
        rec = play_option(env, cached, seq, args, goal_policy, writer)
        options.append(rec)
        if not seq.next(rec['outcome'], env.raw_obs['room']['room_idx']):
            break
        started = seq.start(env, room_changed=rec['task'] == 'goto_door')
        if started is None:
            options.append(dict(task=seq.plan[seq.index]['task'], outcome='no_goal'))
    result = dict(seed=int(seed), group=group['name'], mode=mode, room_variant=info.get('room_variant'), room_grid=grid,
                  start_bombs=start['bombs'], start_hp=hp,
                  chain=info.get('chain'), options=options, planned=len(seq.plan),
                  completed=len(options) == len(seq.plan) and all(o['outcome'] in SUCCESS for o in options),
                  seconds=round(time.monotonic() - t0, 2))
    if args.control_b and mode == 'chain' and len(options) == len(seq.plan):
        result['control_b'] = run_control_b(env, cached, group, seed, info['chain'], options[-1]['start'], args, config)
        env.bridge.reset_mode, env.bridge.tasks = 'chain', TaskSampler(group['spec']['weights'], group['spec']['normal'],
                                                                       group['spec']['boss'], None)
    return result


def run_control_b(env, cached, group, seed, chain, entry, args, config):
    """B's room by goto with the chain's B-entry half hearts and bombs, one COMBAT option."""
    slot_a, a, slot_b, b, variant_a, variant_b = chain
    env.bridge.reset_mode = None
    env.bridge.tasks = FixedRoom(variant_b, (slot_b + 2) % 4)
    obs, info = env.reset(options={'arena_seed': int(seed), 'start': (float(entry['hp']), 1.0), 'bombs': int(entry['bombs'])})
    seq = OptionSequence(dict(group, mode='combat'), seed, info, rewards_of(config))
    seq.start(env, first=True)
    cached.reset()
    rec = play_option(env, cached, seq, args, config['reward_profile'] in GOAL_PROFILES)
    return dict(rec, room_variant=info.get('room_variant'))


def worker(index, args, config, group, mode, tasks, results):
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    name, port = f'{args.name}{chr(ord("a") + index)}', args.port + index   # no digits in instance names
    policy = load_policy(args.checkpoint, args.device)
    cached = CachedPolicy(policy, config['history'], deterministic=not args.stochastic)
    goal_policy = config['reward_profile'] in GOAL_PROFILES
    state = {'proc': None, 'env': None}

    def start():
        state['proc'] = launch_abplus(name, port, args.mode)
        state['env'] = make_env(port, config, group, mode)
        if state['env'].observation_space != policy.observation_space:
            raise ValueError('the environment observation differs from the checkpoint policy')

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
                seed = tasks.get_nowait()
            except queue.Empty:
                break
            for attempt in range(2):
                try:
                    if args.stochastic:
                        torch.manual_seed(args.sample_seed + seed)
                    writer = None
                    if seed in args.replay_seeds:
                        (args.out / 'replays').mkdir(exist_ok=True)
                        writer = gzip.open(args.out / 'replays' / f'seed-{seed}.jsonl.gz', 'wt', encoding='utf8')
                    try:
                        result = run_sequence(state['env'], cached, group, mode, seed, args, config, goal_policy, writer)
                    finally:
                        if writer is not None:
                            writer.close()
                    break
                except Exception:
                    result = dict(seed=int(seed), group=group['name'], mode=mode, error=traceback.format_exc()[-3000:],
                                  attempt=attempt)
                    stop()
                    time.sleep(2)
                    start()
            result['worker'] = index
            if args.recycle_rss_mib > 0 and state['proc'] is not None and rss_mib(state['proc'].pid) >= args.recycle_rss_mib:
                stop()
                time.sleep(2)
                start()
            results.put(result)
    finally:
        stop()
        results.put(None)


def summarize(rows):
    """Per option kind: counts of outcomes; GOTO: success rate by stratum and SPL; chains: stage success."""
    out = {}
    options = [o for r in rows if 'options' in r for o in r['options']]
    for task in ('combat', 'goto_position', 'goto_door'):
        mine = [o for o in options if o['task'] == task]
        if not mine:
            continue
        entry = dict(n=len(mine), outcomes=dict(collections.Counter(o['outcome'] for o in mine)),
                     success=round(sum(o['outcome'] in SUCCESS for o in mine) / len(mine), 4))
        if task != 'combat':
            spl = [(o['outcome'] == 'goal') * (o['d0'] / max(o['walked'], o['d0'])) for o in mine
                   if o.get('d0') and o.get('walked') is not None]
            entry['spl'] = round(sum(spl) / len(spl), 4) if spl else None
            ok = [o for o in mine if o['outcome'] == 'goal' and o.get('d0')]
            entry['decisions_per_100px_median'] = round(statistics.median(100 * o['decisions'] / o['d0'] for o in ok), 3) if ok else None
            if task == 'goto_position':
                entry['by_stratum'] = {s: round(sum(o['outcome'] == 'goal' for o in m) / len(m), 4)
                                       for s in ('straight', 'detour', 'dead_end')
                                       for m in [[o for o in mine if o.get('stratum') == s]] if m}
            # C45: by the bombs held at the option's start (0 / 1+) and by the room's size (1x1 / larger)
            entry['by_bombs'] = {k: [sum(o['outcome'] == 'goal' for o in m), len(m)]
                                 for k, m in (('0', [o for o in mine if o['start']['bombs'] == 0]),
                                              ('1+', [o for o in mine if o['start']['bombs'] > 0])) if m}
            entry['by_size'] = {k: [sum(o['outcome'] == 'goal' for o in m), len(m)]
                                for k, m in (('1x1', [o for o in mine if o.get('grid') == [15, 9]]),
                                             ('big', [o for o in mine if o.get('grid') not in (None, [15, 9])])) if m}
        else:
            for source in ('single', 'chain'):
                m = [o for o in mine if o['source'] == source]
                if m:
                    entry[f'{source}_win'] = round(sum(o['outcome'] == 'win' for o in m) / len(m), 4)
            entry['hurt_mean'] = round(statistics.mean(o['hurt'] for o in mine), 3)
            entry['by_size'] = {k: [sum(o['outcome'] == 'win' for o in m), len(m)]
                                for k, m in (('1x1', [o for o in mine if o.get('grid') == [15, 9]]),
                                             ('big', [o for o in mine if o.get('grid') not in (None, [15, 9])])) if m}
        out[task] = entry
    seqs = [r for r in rows if 'options' in r]
    out['sequences'] = dict(n=len(seqs), errors=sum('error' in r for r in rows),
                            completed=round(sum(r['completed'] for r in seqs) / max(1, len(seqs)), 4))
    controls = [r for r in seqs if r.get('control_b')]
    if controls:
        chain_b = [r['options'][-1] for r in controls]
        ctrl = [r['control_b'] for r in controls]
        out['control_b'] = dict(n=len(controls),
                                chain_b_win=round(sum(o['outcome'] == 'win' for o in chain_b) / len(chain_b), 4),
                                control_win=round(sum(o['outcome'] == 'win' for o in ctrl) / len(ctrl), 4),
                                chain_b_hurt=round(statistics.mean(o['hurt'] for o in chain_b), 3),
                                control_hurt=round(statistics.mean(o['hurt'] for o in ctrl), 3),
                                chain_b_seconds_median=round(statistics.median(o['frames'] / 30 for o in chain_b if o['outcome'] == 'win'), 2)
                                if any(o['outcome'] == 'win' for o in chain_b) else None,
                                control_seconds_median=round(statistics.median(o['frames'] / 30 for o in ctrl if o['outcome'] == 'win'), 2)
                                if any(o['outcome'] == 'win' for o in ctrl) else None)
    return out


def parse_start_bombs(text):
    """none: the game's 1; 0: always 0; ZERO_PROB:MAX: 0 with probability ZERO_PROB else uniform 1..MAX (train_abplus)."""
    if text == 'none':
        return None
    if text == '0':
        return {'zero_prob': 1.0, 'max': 1}
    zero, most = text.split(':')
    return {'zero_prob': float(zero), 'max': int(most)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--groups-file', required=True, type=Path)
    p.add_argument('--group', required=True)
    p.add_argument('--seeds', required=True, help='range:START:COUNT')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--instances', type=int, default=4)
    p.add_argument('--mode', default='exact', choices=('exact', 'skip', 'render'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--scripted-goto', action='store_true', help='the GOTO options by the scripted d_geo navigator')
    p.add_argument('--control-b', action='store_true', help="chain groups: B's room again by goto (the E-CHAIN control)")
    p.add_argument('--replays', type=int, default=0, help='raw-observation replays of the first N seeds')
    p.add_argument('--training-strata', action='store_true',
                   help="keep the group's training GOTO_POSITION strata weights (goal_strata; default: the uniform draw)")
    p.add_argument('--rooms', default='eval', choices=('eval', 'train'),
                   help="eval: the group's eval_tasks when it has them (M2: held-out rooms); train: its training tasks")
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--expert-actions', action='store_true',
                   help='GOTO options act by the observation expert_move label (goal-hp3), the label the imitation loss reads')
    p.add_argument('--expert-unstick', action='store_true',
                   help='COMBAT options: while the stuck label fires, its move replaces the checkpoint move (goal-hp3)')
    p.add_argument('--check-expert', action='store_true',
                   help='per step: the label against descent_move on the option field and on a field rebuilt from scratch; '
                        'per GOTO_POSITION option: Bellman and descent certificate of the field')
    p.add_argument('--start-bombs', default=None,
                   help='override the checkpoint run start bombs: none (1), 0, or ZERO_PROB:MAX as train_abplus')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--recycle-rss-mib', type=float, default=400)
    p.add_argument('--port', type=int, default=27600)
    p.add_argument('--name', default='ev', help='instance name prefix (letters only: run_instance.sh reads digits)')
    args = p.parse_args()
    if any(c.isdigit() for c in args.name):
        p.error('--name must not contain digits (run_instance.sh derives a window position from them)')
    try:
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('1000')
    except OSError:
        pass
    if args.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    if not steam_running():
        print(json.dumps({'event': 'evaluation_skipped', 'reason': 'steam client not running'}), flush=True)
        raise SystemExit(3)
    saved = json.loads((args.checkpoint / 'state.json').read_text())
    config = saved['config']
    if args.start_bombs is not None:   # C45 closing evaluation: earlier runs under the C45 start, C45 under the C44 start
        config = dict(config, start_bombs=parse_start_bombs(args.start_bombs))
    groups = {g['name']: g for g in load_groups(args.groups_file)}
    group = groups[args.group]
    training_strata = group.get('goal_strata') if args.training_strata else group.pop('goal_strata', None)
    # C43: goal_strata are training-only weights; evaluations draw uniformly unless --training-strata
    rooms_file = group['tasks']
    if args.rooms == 'eval' and 'eval_spec' in group:
        group, rooms_file = dict(group, spec=group['eval_spec']), group['eval_tasks']
    mode = group.get('mode', 'combat')
    start, count = map(int, args.seeds.split(':')[1:])
    seeds = list(range(start, start + count))
    args.replay_seeds = set(seeds[:max(0, args.replays)])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'results.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(l)['seed'] for l in path.read_text().splitlines() if 'error' not in json.loads(l)}
    todo = [s for s in seeds if s not in done]
    meta = dict(checkpoint=str(args.checkpoint.resolve()), updates=saved['updates'], reward_profile=config['reward_profile'],
                groups_file=str(args.groups_file.resolve()), group=args.group, mode=mode, goals=group.get('goals'),
                rooms=args.rooms, rooms_file=rooms_file, room_count=len(group['spec']['normal']) + len(group['spec']['boss']),
                training_goal_strata=training_strata, strata_weighted=bool(args.training_strata and training_strata),
                scripted_goto=args.scripted_goto or config['reward_profile'] not in GOAL_PROFILES,
                control_b=args.control_b, deterministic=not args.stochastic, seeds=len(seeds),
                expert_actions=args.expert_actions, expert_unstick=args.expert_unstick, check_expert=args.check_expert,
                sample_seed=args.sample_seed, start_bombs=config.get('start_bombs'), frames_per_decision=int(config.get('frames_per_decision', 2)),
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=1))
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    for s in todo:
        tasks.put(s)
    n = min(args.instances, len(todo))
    print(f'{len(done)} done, {len(todo)} to run on {n} instances ({mode}, group {args.group})', flush=True)
    procs = [ctx.Process(target=worker, args=(i, args, config, group, mode, tasks, results), daemon=True) for i in range(n)]
    for proc in procs:
        proc.start()
    finished, rows = 0, [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    with open(path, 'a') as f:
        while finished < n:
            r = results.get()
            if r is None:
                finished += 1
                continue
            f.write(json.dumps(r) + '\n')
            f.flush()
            rows.append(r)
            opts = ' '.join(f"{o['task'][:6]}:{o['outcome']}" for o in r.get('options', []))
            print(time.strftime('%H:%M:%S'), r['seed'], 'ERROR' if 'error' in r else opts, flush=True)
    for proc in procs:
        proc.join(timeout=60)
    summary = summarize([r for r in rows if r.get('seed') in set(seeds)])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
