"""Option sequences of the goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md, user decisions 2026-09-30), shared by
the training worker (abplus_worker) and the evaluation (abplus_eval_options).

A group's mode (GROUP_MODES) turns one reset into a plan of options played in place on the same instance:
  combat       COMBAT (source single)                                       every earlier run's episode
  combat_goto  COMBAT (single), then `goals` GOTO_POSITION in the cleared room
  goto_empty   `goals` GOTO_POSITION in an emptied room (reset_mode goto_room)
  chain        a real floor (reset_mode chain): COMBAT in A (source chain), `goals` GOTO_POSITION, GOTO_DOOR to B,
               COMBAT in B (chain)
The next option starts when the last one won (COMBAT) or reached its goal (GOTO); any other outcome ends the sequence.
Starting an option sets the observation history (the sequence's first option and a new room: a fresh window; the same room:
soft_clear, the last observation kept for velocities), its goal (VisibleHistory.goal_state: task, target, the walking
distance d_geo as a callable of the observation, source), the bridge's per-frame goal check (GOTO_POSITION: goal_radius),
the option's frame budget (COMBAT: the group's deadline; GOTO: goto_seconds) and the rewards' baselines.

GOTO outcomes (outcome()): 'goal' (position: within the radius on a logic frame; door: the room changed through its door
into its room), 'exit' (a position task left the room), 'wrong_door', 'death', 'time_limit'.
"""
import math

import numpy as np

from .abplus_nav import door_target, goal_distance, goal_field, nav_key, sample_goal

GROUP_MODES = ('combat', 'combat_goto', 'goto_empty', 'chain')
GOTO_SECONDS, GOTO_RADIUS = 8.0, 20.0


def option_plan(group, info):
    """The options of the sequence a reset started, from the group spec (mode, goals) and the reset's info (chain)."""
    mode = group.get('mode', 'combat')
    goals = int(group.get('goals', 0))
    if mode == 'combat':
        return [dict(task='combat', source='single')]
    if mode == 'combat_goto':
        return [dict(task='combat', source='single')] + [dict(task='goto_position', source='goto')] * goals
    if mode == 'goto_empty':
        return [dict(task='goto_position', source='goto')] * max(1, goals)
    if mode == 'chain':
        slot_a, a, slot_b, b, variant_a, variant_b = info['chain']
        return ([dict(task='combat', source='chain', room=a, variant=variant_a)]
                + [dict(task='goto_position', source='goto')] * goals
                + [dict(task='goto_door', source='goto', slot=slot_b, room=b)]
                + [dict(task='combat', source='chain', room=b, variant=variant_b)])
    raise ValueError(f'unknown group mode {mode!r}')


class OptionSequence:
    """The options of one reset. rewards: (COMBAT reward, GotoReward) objects to reset and step, or None."""

    def __init__(self, group, seed, info, rewards=None):
        self.group, self.seed = group, int(seed)
        self.plan = option_plan(group, info)
        self.index = 0
        self.option = None
        self.combat_reward, self.goto_reward = rewards if rewards is not None else (None, None)

    def start(self, env, first=False, room_changed=False):
        """Start plan[index] in place on env's current observation; returns (its first frame, entity rows), or None when a
        GOTO option finds no goal (the sequence then ends)."""
        spec = self.plan[self.index]
        raw = env.raw_obs
        history = env.history
        g = self.group
        task = spec['task']
        target = nav = bridge_goal = None
        stratum = None
        if task == 'goto_position':
            if spec.get('target') is not None:   # the floor runner: a given point (a pickup, the trapdoor)
                target, stratum = tuple(float(v) for v in spec['target']), spec.get('stratum')
            else:
                rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 0x60A1, self.index])
                got = sample_goal(raw, rng, weights=g.get('goal_strata'))
                if got is None:
                    return None
                target, stratum, _ = got
            nav = goal_field(raw, target)
            bridge_goal = (target[0], target[1], float(g.get('goal_radius', GOTO_RADIUS)))
        elif task == 'goto_door':
            door = next((d for d in raw['doors'] if d['slot'] == spec['slot']), None)
            inside = door_target(raw, spec['slot'])
            if door is None or inside is None:
                return None
            target = tuple(door['pos'])
            nav = goal_field(raw, inside)
            nav['exit'] = target   # the scripted navigator walks on through the door from the inside cell
        state = dict(nav=nav, distance=None)

        def distance(obs):
            """d_geo of the player on the observation's map (the field is rebuilt when the terrain or the fires change);
            the last value when no nearby cell reaches the goal."""
            if state['nav'] is None:
                return None
            if nav_key(obs) != state['nav']['key']:
                exit_ = state['nav'].get('exit')
                state['nav'] = goal_field(obs, state['nav']['target'])
                if exit_ is not None:
                    state['nav']['exit'] = exit_
            px, py = obs['players'][0]['pos']
            d = goal_distance(px, py, state['nav'])
            if d is not None:
                state['distance'] = d
            return state['distance']

        if first or room_changed:
            history.clear()
        else:
            history.soft_clear()
        history.goal_state = dict(task=task, target=target, distance=distance if nav is not None else None,
                                  source=spec['source'], nav_state=state if nav is not None else None)
        env.bridge.goal = bridge_goal
        env.bridge.stop_clear = task == 'combat'   # a COMBAT option ends at the clear frame, before a door can be used
        if task == 'combat':
            seconds = float(g.get('seconds', env.max_episode_frames / 30))
            frames = int(g.get('frames', env.max_episode_frames))
        else:
            seconds = float(g.get('goto_seconds', GOTO_SECONDS))
            terrain = raw.get('terrain') or {}
            if 'goto_seconds_big' in g and (terrain.get('height', 9) > 9 or terrain.get('width', 15) > 15):
                seconds = float(g['goto_seconds_big'])   # C45: a room larger than 1x1 (grid beyond 15 x 9)
            frames = int(round(seconds * 30))
        history.deadline_s = seconds
        env.begin_option('combat' if task == 'combat' else 'goto', frames)
        frame = history.encode(raw)
        if nav is not None and state['distance'] is None:
            distance(raw)   # a history without goal fields (a checkpoint from before the goal line) never asks
        if task == 'combat':
            if self.combat_reward is not None:
                self.combat_reward.reset(raw, 'normal')
        elif self.goto_reward is not None:
            self.goto_reward.reset(raw, state['distance'])
        # distance_fn: a history without goal fields never evaluates the distance; its caller does (abplus_eval_options)
        self.option = dict(spec, state=state, distance_fn=distance if nav is not None else None, decisions=0, frames=0,
                           deadline_decisions=int(math.ceil(frames / max(1, env.bridge.action_repeat))),
                           room=raw['room']['room_idx'] if task != 'goto_door' else spec['room'], target=target,
                           stratum=stratum, d0=state['distance'])
        return frame, history.last_rows

    def outcome(self, raw, outcome):
        """An option's outcome from the env's: a room change is a door goal's success only through its door into its room;
        otherwise a wrong door (door task) or an exit (position task). COMBAT outcomes pass through."""
        option = self.option
        if option['task'] == 'combat' or outcome != 'room':
            return outcome
        nav = raw.get('nav') or {}
        if (option['task'] == 'goto_door' and int(nav.get('leave_door', -1)) == option['slot']
                and raw['room']['room_idx'] == option['room']):
            return 'goal'
        return 'wrong_door' if option['task'] == 'goto_door' else 'exit'

    def step(self, raw, outcome, frames):
        """Bookkeeping of one decision and its reward (scaled, components, hit) when rewards were given."""
        option = self.option
        option['decisions'] += 1
        option['frames'] += frames
        if self.combat_reward is None:
            return 0.0, {}, 0.0
        if option['task'] == 'combat':
            parts = self.combat_reward.step(raw, outcome, frames)
            return self.combat_reward.scale * float(sum(parts.values())), parts, float(parts.get('damage', 0.0))
        left = max(0, option['deadline_decisions'] - option['decisions'])
        parts = self.goto_reward.step(raw, outcome, option['state']['distance'], left)
        return self.goto_reward.scale * float(sum(parts.values())), parts, 0.0

    def totals(self):
        """The finished option's reward totals (COMBAT or GOTO components), or None without rewards."""
        if self.combat_reward is None:
            return None
        return (self.combat_reward if self.option['task'] == 'combat' else self.goto_reward).totals_array()

    def next(self, outcome, room=None):
        """True when the plan goes on after this outcome (index moved to the next option; call start next). room: the
        player's room now; a COMBAT won from outside its room (left during the clear countdown, EXPERIMENTS.md A13) ends
        the plan, whose next options belong to that room."""
        if self.option is not None and self.option['task'] == 'combat' and room is not None and room != self.option['room']:
            return False
        if outcome in ('win', 'goal') and self.index + 1 < len(self.plan):
            self.index += 1
            return True
        return False
