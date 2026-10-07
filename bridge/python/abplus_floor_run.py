"""Floor runner (floor-clear mandate, user 2026-10-01): play a whole Basement I floor on the real engine and report whether
the agent clears it: from the start room of a fresh run (bridge reset_mode 'floor': Isaac's own start, 3 red hearts, 1 bomb,
no items; floors with Labyrinth, Lost or Maze rerolled as the chain does), explore room by room, clear every room it enters,
beat the boss and drop through the trapdoor into the next floor (success = the level's stage changes).

Who does what:
  COMBAT        the checkpoint's policy (greedy, or --stochastic), one COMBAT option per try, the room's roster marked first
                (AbpRosterMark, as every training reset does); a try that times out is followed by another (--combat-tries).
  navigation    the scripted A* navigator (abplus_nav.descent_move; EXPERIMENTS.md A14): GOTO_DOOR to the next room, GOTO
                to pickups, to the treasure room's item and to the trapdoor.
  exploration   only what a player sees: the doors of the rooms visited, where they lead and the kind of room behind them
                (the door's look / the minimap's icon of an adjacent room; Lua Door.TargetRoomIndex and the target's
                RoomDescriptor type). Locked doors and rooms other than normal, treasure and boss are not entered. The boss
                room is entered as soon as one of its doors has been seen; until then the frontier room farthest from the
                start room (grid distance; the generator puts the boss at a far dead end) is explored next, ties to the
                shorter walk.
  pickups       after a clear: hearts while hurt (soul, black, eternal, gold and blended hearts always), bombs (not troll
                bombs); in the treasure room the item (--take-items); in the boss room nothing but the trapdoor.

One results line per seed: outcome (cleared / death / timeout / stuck / error), rooms entered, combats (room type, outcome,
half hearts before and after, seconds), the boss, items taken, game seconds. --replays N writes the first N seeds as
raw-observation replays (abplus_replay_view.py).

usage: python abplus_floor_run.py --checkpoint DIR --seeds range:START:COUNT --out DIR [--instances 4] [--stochastic]
       [--take-items/--no-take-items] [--replays N]
"""
import argparse
import collections
import gzip
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

from abplus_eval import CachedPolicy, load_policy
from abplus_eval_options import option_tag, play_option, rewards_of
from isaac_bridge.abplus import AbplusTransformerEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_geometry import CELL, walk_grid
from isaac_bridge.abplus_nav import move_code
from isaac_bridge.abplus_options import OptionSequence
from isaac_bridge.abplus_worker import observation_options

NORMAL, TREASURE, BOSS = 1, 4, 5
ENTERABLE = (NORMAL, TREASURE, BOSS)
HEARTS, BOMBS, PEDESTAL = 10, 40, 100
RED_HEARTS = (1, 2, 5, 9)       # full, half, double, scared: only while hurt
GOOD_BOMBS = (1, 2, 4)          # single, double pack, golden (3 and 5 are troll bombs)
# Items that replace the tears the policy learned to fire (bombs, a knife, a charged laser, explosive or bursting shots):
# left on their pedestal (C47 u80: two of the runs that took Mom's Knife / Monstro's Lung died right after)
D6 = 105   # Isaac's own active item: it drops on the pedestal when another active item is taken
AVOID_ITEMS = {52: "Dr. Fetus", 114: "Mom's Knife", 118: 'Brimstone', 149: 'Ipecac', 168: 'Epic Fetus', 229: "Monstro's Lung"}
GROUP = dict(name='floor', mode='combat', goto_seconds=10.0, goto_seconds_big=15.0, goal_radius=20.0)

DOORS_LUA = """
local room = Game():GetRoom(); local level = Game():GetLevel(); local out = {}
local here = level:GetCurrentRoomDesc().SafeGridIndex
out[1] = tostring(here)
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d and d:GetVariant() ~= 7 then
    local t = d.TargetRoomIndex; local desc = level:GetRoomByIdx(t)
    local ty, safe = -1, -999
    pcall(function() ty = desc.Data.Type; safe = desc.SafeGridIndex end)
    out[#out + 1] = table.concat({slot, t, safe, ty, d:IsLocked() and 1 or 0}, ",")
  end
end
return table.concat(out, ";")
"""
TRAPDOOR_LUA = """
local room = Game():GetRoom(); local out = {}
for i = 0, room:GetGridSize() - 1 do
  local g = room:GetGridEntity(i)
  if g ~= nil and g:GetType() == 17 then local p = room:GetGridPosition(i); out[#out + 1] = p.X .. "," .. p.Y end
end
return table.concat(out, ";")
"""
GRID_DUMP_LUA = """
local room = Game():GetRoom(); local level = Game():GetLevel(); local out = {"stage=" .. level:GetStage() .. " type=" .. room:GetType() ..
  " clear=" .. tostring(room:IsClear()) .. " center=" .. room:GetCenterPos().X .. "," .. room:GetCenterPos().Y}
for i = 0, room:GetGridSize() - 1 do
  local g = room:GetGridEntity(i)
  if g ~= nil and g:GetType() ~= 15 and g:GetType() ~= 16 and g:GetType() ~= 1 then out[#out + 1] = i .. ":" .. g:GetType() .. "/" .. g.State end
end
for _, e in ipairs(Isaac.GetRoomEntities()) do out[#out + 1] = "e" .. e.Type .. "." .. e.Variant .. "." .. e.SubType end
return table.concat(out, " ")
"""
HP_LUA = "local p = Isaac.GetPlayer(0); return p:GetHearts() .. ',' .. p:GetMaxHearts() .. ',' .. p:GetSoulHearts()"


class Args:
    """play_option / act switches: GOTO by the scripted navigator, COMBAT by the policy."""
    scripted_goto = True
    expert_actions = False
    expert_unstick = False
    check_expert = False

    def __init__(self, stochastic, no_bombs=False, shield=False):
        self.stochastic, self.no_bombs, self.shield = stochastic, no_bombs, shield
        self.finisher = False   # set per COMBAT try by Floor.fight
        self.shield_stats = collections.Counter()


def make_env(port, config):
    env = AbplusTransformerEnv(port=port, max_episode_frames=30 * 600, **observation_options(config['reward_profile']),
                               deadline_s=180.0, frames_per_decision=int(config.get('frames_per_decision', 4)),
                               history=config['history'], entity_capacity=config['entity_capacity'])
    env.bridge.binary_obs = True
    env.combat_multi_room = True
    env.bridge.reset_mode = 'floor'
    return env


class Floor:
    """One floor of one seed on one environment."""

    def __init__(self, env, cached, config, args, seed, writer, fallback=None, boss=None):
        self.env, self.cached, self.config, self.args, self.seed, self.writer = env, cached, config, args, seed, writer
        self.primary, self.fallback = cached, fallback   # fallback: another checkpoint for a room the first try left uncleared
        self.boss_policy = boss                          # boss: the checkpoint that fights in the boss room
        self.play_args = Args(args.stochastic, args.no_bombs, args.shield)
        self.seq = None
        self.adj = collections.defaultdict(dict)   # room -> {neighbour room: (door slot, TargetRoomIndex)}
        self.kind = {}                             # room -> RoomDescriptor type
        self.visited, self.cleared = [], set()
        self.events, self.combats, self.items = [], [], 0
        self.item_rooms = set()
        self.dead_doors = set()   # (room, next room) whose door the navigator could not reach twice
        self.start_room = None
        self.frames = 0

    # -- low level ---------------------------------------------------------------------------------------------------
    @property
    def raw(self):
        return self.env.raw_obs

    def lua(self, code):
        return self.env.bridge.lua(code)

    def here(self):
        parts = self.lua(DOORS_LUA).split(';')
        room = int(parts[0])
        for row in parts[1:]:
            slot, target, safe, kind, locked = (int(float(x)) for x in row.split(','))
            self.kind.setdefault(safe, kind)
            if not locked and safe >= 0:
                self.adj[room][safe] = (slot, target)
                self.adj[safe].setdefault(room, None)   # the way back is known once entered
        return room

    def hp(self):
        hearts, max_hearts, soul = (int(x) for x in self.lua(HP_LUA).split(','))
        return hearts, max_hearts, soul

    def raw_steps(self, n, move=0):
        """n decisions of `move` straight on the bridge (no option: idling while drops settle, a push)."""
        bridge = self.env.bridge
        bridge.goal, bridge.stop_clear = None, False
        for _ in range(n):
            obs, _, _, _, _ = bridge.step({'move': int(move)}, repeat=bridge.action_repeat)
            self.env.raw_obs = obs
            self.frames += bridge.action_repeat
            if self.writer is not None:
                self.writer.write(json.dumps({'action': [int(move) * 5, 0, 0], 'obs': obs, 'option': None},
                                             separators=(',', ':')) + '\n')
            if obs['players'][0]['dead'] or obs['room']['stage'] != 1:
                break

    def option(self, spec, room_changed, seconds=None):
        """Play one option (OptionSequence with a one-option plan) to its end; its record, or None without a goal."""
        if self.seq is None:
            self.seq = OptionSequence(dict(GROUP), self.seed, {}, rewards_of(self.config))
        # COMBAT: its own budget (frames), else OptionSequence falls back to env.max_episode_frames, which the last
        # option (a GOTO: 10 s) set
        self.seq.group = dict(GROUP, seconds=seconds, frames=int(seconds * 30)) if seconds else dict(GROUP)
        self.seq.plan, self.seq.index = [spec], 0
        if self.seq.start(self.env, room_changed=room_changed) is None:
            return None
        self.cached.reset()
        rec = play_option(self.env, self.cached, self.seq, self.play_args, True, self.writer)
        self.frames += rec['frames']
        return rec

    # -- room handling -----------------------------------------------------------------------------------------------
    def fight(self, room):
        """Clear the current room with the policy; the last outcome ('win', 'death', 'time_limit', 'left')."""
        kind = self.kind.get(room, NORMAL)
        outcome = None
        for attempt in range(self.args.combat_tries):
            # a try after a time-out: the fallback checkpoint (C47's careful policy leaves stationary shooters alone)
            self.cached = self.fallback if (attempt > 0 and self.fallback is not None) else self.primary
            if kind == BOSS and self.boss_policy is not None:
                self.cached = self.boss_policy
            # the last try in a normal room: the scripted finisher (--finisher)
            self.play_args.finisher = bool(self.args.finisher and kind != BOSS and attempt == self.args.combat_tries - 1
                                           and attempt > 0)
            self.lua(f"return tostring(AbpRosterMark({int(self.config.get('lineage_mode', 0))}))")
            before = self.hp()
            t0 = self.frames
            shield0 = dict(self.play_args.shield_stats)
            rec = self.option(dict(task='combat', source='single'), room_changed=attempt == 0,
                              seconds=self.args.boss_seconds if kind == BOSS else self.args.combat_seconds)
            outcome = rec['outcome'] if rec else 'no_option'
            after = self.hp() if not self.raw['players'][0]['dead'] else (0, before[1], 0)
            self.combats.append(dict(room=room, kind=kind, attempt=attempt, fallback=self.cached is self.fallback,
                                     finisher=self.play_args.finisher,
                                     outcome=outcome, hp_before=before[0] + before[2],
                                     hp_after=after[0] + after[2], seconds=round((self.frames - t0) / 30, 1),
                                     bosses=sorted({(e['type'], e['variant']) for e in self.raw['entities'] if e.get('boss')}),
                                     left=sorted({(e['type'], e['variant']) for e in self.raw['entities'] if e.get('enemy')})[:8],
                                     shield={k: v - shield0.get(k, 0) for k, v in self.play_args.shield_stats.items()},
                                     variant=self.raw['room'].get('variant')))
            if outcome != 'time_limit' or self.raw['players'][0]['dead']:
                break
        self.cached = self.primary
        self.play_args.finisher = False
        here = self.raw['room']['room_idx']
        if outcome == 'win' and here != room:
            outcome = 'left'
        return outcome

    def goto_point(self, target, room_changed=False):
        rec = self.option(dict(task='goto_position', source='goto', target=tuple(target)), room_changed=room_changed)
        return rec['outcome'] if rec else 'no_goal'

    def walkable_near(self, pos):
        """The centre of the walkable cell next to pos nearest to the player (a pedestal's own cell is blocked)."""
        grid = walk_grid(self.raw)
        if grid is None:
            return None
        origin, walkable, _ = grid
        col, row = round((pos[0] - origin[0]) / CELL), round((pos[1] - origin[1]) / CELL)
        px, py = self.raw['players'][0]['pos']
        best = None
        for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            cell = (col + dc, row + dr)
            if cell in walkable:
                xy = (origin[0] + CELL * cell[0], origin[1] + CELL * cell[1])
                d = math.dist(xy, (px, py))
                if best is None or d < best[0]:
                    best = (d, xy)
        return best[1] if best else None

    def cell_away(self, lo, hi):
        """The centre of a walkable cell lo..hi px from the player (the nearest to the middle of that ring), or None."""
        grid = walk_grid(self.raw)
        if grid is None:
            return None
        origin, walkable, _ = grid
        px, py = self.raw['players'][0]['pos']
        best = None
        for c in walkable:
            xy = (origin[0] + CELL * c[0], origin[1] + CELL * c[1])
            d = math.dist(xy, (px, py))
            if lo <= d <= hi and (best is None or abs(d - (lo + hi) / 2) < best[0]):
                best = (abs(d - (lo + hi) / 2), xy)
        return best[1] if best else None

    def collect(self, room):
        """After a clear: hearts while hurt (and the never-wasted kinds), bombs, the treasure room's item."""
        self.raw_steps(8)   # drops spawn and settle
        for _ in range(self.args.max_pickups):
            if self.raw['players'][0]['dead']:
                return
            hearts, max_hearts, _ = self.hp()
            px, py = self.raw['players'][0]['pos']
            wanted = []
            for e in self.raw['entities']:
                if e['type'] != 5:
                    continue
                v, s = e['variant'], e.get('subtype', 0)
                if v == HEARTS and (s not in RED_HEARTS or hearts < max_hearts):
                    wanted.append(e)
                elif v == BOMBS and s in GOOD_BOMBS:
                    wanted.append(e)
                elif (v == PEDESTAL and s > 0 and s not in AVOID_ITEMS and s != D6 and self.args.take_items
                      and self.kind.get(room) == TREASURE and room not in self.item_rooms):
                    wanted.append(e)
            if not wanted:
                return
            e = min(wanted, key=lambda x: math.dist(x['pos'], (px, py)))
            if e['variant'] == PEDESTAL:
                near = self.walkable_near(e['pos'])
                if near is None or self.goto_point(near) != 'goal':
                    return
                q = self.raw['players'][0]['pos']
                self.raw_steps(6, move_code(e['pos'][0] - q[0], e['pos'][1] - q[1]))
                self.raw_steps(20)   # the pick-up animation
                self.items += 1
                self.item_rooms.add(room)
                self.events.append(('item', room, e.get('subtype')))
            else:
                before = len(self.raw['entities'])
                if self.goto_point(e['pos']) != 'goal':
                    return
                self.raw_steps(2)
                self.events.append(('pickup', room, e['variant'], e.get('subtype'), before))

    def trapdoor(self):
        """The boss is dead: walk into the trapdoor (it appears a moment after the clear); True when the stage changed."""
        got = ''
        for attempt in range(60):
            got = self.lua(TRAPDOOR_LUA)
            if got:
                break
            if attempt == 10:   # the trapdoor does not appear while the player stands where it goes: step away
                away = self.cell_away(100.0, 160.0)
                if away is not None:
                    self.goto_point(away)
            self.raw_steps(2)
        if not got:
            self.events.append(('no_trapdoor', self.lua(GRID_DUMP_LUA)))
            return False
        x, y = (float(v) for v in got.split(';')[0].split(','))
        for tries in range(4):
            got_there = self.goto_point((x, y))
            if got_there == 'death' or self.raw['players'][0]['dead']:
                self.events.append(('trapdoor_try', tries, 'death', None, None, None))
                return False
            for _ in range(12):
                if self.raw['room']['stage'] != 1:
                    return True
                self.raw_steps(2)
            # a trapdoor that appeared next to the player stays shut until the player has walked away from it
            away = self.cell_away(180.0, 280.0) or self.cell_away(120.0, 200.0)
            went = self.goto_point(away) if away is not None else 'no_cell'
            self.events.append(('trapdoor_try', tries, got_there, away, went,
                                [round(v) for v in self.raw['players'][0]['pos']]))
            self.raw_steps(8)
        if self.raw['room']['stage'] == 1:
            self.events.append(('trapdoor_shut', got, self.lua(GRID_DUMP_LUA)))
        return self.raw['room']['stage'] != 1

    # -- exploration -------------------------------------------------------------------------------------------------
    def route(self, room, goal):
        """Rooms from room to goal through visited rooms (BFS), the goal last; None without one."""
        prev, frontier = {room: None}, collections.deque([room])
        while frontier:
            r = frontier.popleft()
            if r == goal:
                path = []
                while r is not None:
                    path.append(r)
                    r = prev[r]
                return path[::-1][1:]
            for n, door in self.adj[r].items():
                if door is None or n in prev or (n != goal and n not in self.visited) or (r, n) in self.dead_doors:
                    continue   # a door seen from r (r visited) into a visited room, or into the goal
                prev[n] = r
                frontier.append(n)
        return None

    def grid_distance(self, a, b):
        return abs(a % 13 - b % 13) + abs(a // 13 - b // 13)

    def next_goal(self, room):
        known = [r for r in self.kind if r not in self.visited and self.kind[r] in ENTERABLE
                 and any(r in self.adj[v] and self.adj[v][r] is not None and (v, r) not in self.dead_doors
                         for v in self.visited)]
        boss = [r for r in known if self.kind[r] == BOSS]
        if boss:
            return boss[0]
        treasure = [r for r in known if self.kind[r] == TREASURE]
        if self.args.take_items and treasure:
            r = treasure[0]
            path = self.route(room, r)
            if path is not None and len(path) <= self.args.treasure_detour:
                return r
        normal = [r for r in known if self.kind[r] == NORMAL]
        if not normal:
            return treasure[0] if treasure else None
        scored = []
        for r in normal:
            path = self.route(room, r)
            if path is not None:
                scored.append((-self.grid_distance(r, self.start_room), len(path), r))
        return min(scored)[2] if scored else None

    def walk(self, room, goal):
        """Through the doors of the route towards goal, one room at a time; the room reached (it may be uncleared)."""
        path = self.route(room, goal)
        if not path:
            return None
        nxt = path[0]
        slot, target = self.adj[room][nxt]
        for _ in range(2):   # a second try after a time-out (the door approach rarely misses)
            rec = self.option(dict(task='goto_door', source='goto', slot=slot, room=target), room_changed=False)
            if rec is None:
                return None
            self.events.append(('door', room, nxt, rec['outcome'], rec['frames']))
            if rec['outcome'] in ('goal', 'wrong_door'):   # through the door (a big room may report another index)
                return self.raw['room']['room_idx']
            if rec['outcome'] != 'time_limit' or self.raw['players'][0]['dead']:
                return None
        self.dead_doors.add((room, nxt))   # unreachable (e.g. rocks in the way): the explorer plans around it
        return room

    # -- the floor -----------------------------------------------------------------------------------------------------
    def run(self):
        obs, info = self.env.reset(options={'arena_seed': int(self.seed)})
        self.frames = 0
        if self.writer is not None:
            self.writer.write(json.dumps({'metadata': {'seed': int(self.seed), 'format': 'abplus-raw-obs-v1', 'goal': True,
                                                       'group': 'floor', 'mode': 'floor'}}) + '\n')
            self.writer.write(json.dumps({'action': None, 'obs': self.raw, 'option': None}, separators=(',', ':')) + '\n')
        floor = info['floor']
        result = dict(seed=int(self.seed), floor_rooms=len(floor), floor_curses=info.get('floor_curses'),
                      boss_variant=next((r['variant'] for r in floor.values() if r['type'] == BOSS), None))
        self.start_room = self.here()
        self.kind[self.start_room] = NORMAL
        self.visited.append(self.start_room)
        outcome = 'stuck'
        while self.frames < self.args.max_seconds * 30:
            if self.raw['players'][0]['dead']:
                outcome = 'death'
                break
            room = self.here()
            if room not in self.visited:
                self.visited.append(room)
            if not self.raw['room']['clear']:
                fought = self.fight(room)
                if fought == 'death' or self.raw['players'][0]['dead']:
                    outcome = 'death'
                    break
                if fought == 'left':
                    continue
                if fought != 'win':
                    outcome = 'combat_' + str(fought)
                    break
            if room not in self.cleared:
                self.cleared.add(room)
                if self.kind.get(room) == BOSS:
                    dropped = self.trapdoor()
                    dead = self.raw['players'][0]['dead'] or 'death' in {e[2] for e in self.events if e[0] == 'trapdoor_try'}
                    outcome = 'cleared' if dropped else ('death_after_boss' if dead else 'no_trapdoor')
                    break
                self.collect(room)
                if self.raw['players'][0]['dead']:
                    outcome = 'death'
                    break
            goal = self.next_goal(room)
            if goal is None:
                outcome = 'stuck'
                break
            if self.walk(room, goal) is None:
                outcome = 'walk_failed'
                break
        else:
            outcome = 'timeout'
        hearts, max_hearts, soul = self.hp() if not self.raw['players'][0]['dead'] else (0, 0, 0)
        try:
            owned = self.lua("local p = Isaac.GetPlayer(0); local out = {} "
                             "for id = 1, CollectibleType.NUM_COLLECTIBLES - 1 do "
                             "if p:HasCollectible(id) then out[#out + 1] = tostring(id) end end return table.concat(out, ',')")
            result['collectibles'] = [int(x) for x in owned.split(',') if x]
        except Exception:
            result['collectibles'] = None
        result.update(outcome=outcome, cleared=outcome == 'cleared', rooms=self.visited, rooms_entered=len(self.visited),
                      combats=self.combats, items=self.items, events=self.events[-60:], game_seconds=round(self.frames / 30, 1),
                      end_hp=hearts + soul, boss_seen=any(k == BOSS for k in self.kind.values()),
                      boss_reached=any(c['kind'] == BOSS for c in self.combats),
                      boss_won=any(c['kind'] == BOSS and c['outcome'] == 'win' for c in self.combats))
        return result


def worker(index, args, config, tasks, results):
    torch.set_num_threads(1)
    from isaac_bridge import abplus_shield
    if args.shield_horizon is not None:
        abplus_shield.HORIZON = int(args.shield_horizon)
    if args.shield_margin is not None:
        abplus_shield.MARGIN = float(args.shield_margin)
    abplus_shield.BOMBS, abplus_shield.CREEP = bool(args.shield_bombs), bool(args.shield_creep)
    name, port = f'{args.name}{chr(ord("a") + index)}', args.port + index
    policy = load_policy(args.checkpoint, args.device)
    cached = CachedPolicy(policy, config['history'], deterministic=not args.stochastic)
    fallback = None
    if args.fallback_checkpoint is not None:
        other = load_policy(args.fallback_checkpoint, args.device)
        if other.observation_space != policy.observation_space:
            raise ValueError('the fallback checkpoint sees another observation')
        fallback = CachedPolicy(other, config['history'], deterministic=not args.stochastic)
    boss = None
    if args.boss_checkpoint is not None:
        other = load_policy(args.boss_checkpoint, args.device)
        if other.observation_space != policy.observation_space:
            raise ValueError('the boss checkpoint sees another observation')
        boss = CachedPolicy(other, config['history'], deterministic=not args.stochastic)
    state = {'proc': None, 'env': None}

    def start():
        state['proc'] = launch_abplus(name, port, args.mode)
        state['env'] = make_env(port, config)
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
                writer = None
                try:
                    if args.stochastic:
                        torch.manual_seed(args.sample_seed + seed)
                    if seed in args.replay_seeds:
                        (args.out / 'replays').mkdir(exist_ok=True)
                        writer = gzip.open(args.out / 'replays' / f'seed-{seed}.jsonl.gz', 'wt')
                    t0 = time.monotonic()
                    r = Floor(state['env'], cached, config, args, seed, writer, fallback, boss).run()
                    r.update(seconds=round(time.monotonic() - t0, 1), worker=index)
                    results.put(r)
                    break
                except Exception as exc:
                    err = f'{type(exc).__name__}: {exc}'
                    if attempt == 1:
                        results.put(dict(seed=seed, error=err, trace=traceback.format_exc()[-1500:]))
                    stop()
                    start()
                finally:
                    if writer is not None:
                        writer.close()
    finally:
        stop()
        results.put(None)


def summarize(rows):
    ok = [r for r in rows if 'error' not in r]
    out = dict(n=len(ok), errors=len(rows) - len(ok), cleared=sum(r['cleared'] for r in ok),
               outcomes=dict(collections.Counter(r['outcome'] for r in ok)),
               boss_reached=sum(r['boss_reached'] for r in ok), boss_won=sum(r['boss_won'] for r in ok),
               rooms_entered_mean=round(float(np.mean([r['rooms_entered'] for r in ok])), 2) if ok else None,
               game_seconds_mean=round(float(np.mean([r['game_seconds'] for r in ok])), 1) if ok else None)
    normal = [c for r in ok for c in r['combats'] if c['kind'] != BOSS]
    boss = [c for r in ok for c in r['combats'] if c['kind'] == BOSS]
    out['normal_combats'] = dict(n=len(normal), win=sum(c['outcome'] == 'win' for c in normal),
                                 hurt_mean=round(float(np.mean([c['hp_before'] - c['hp_after'] for c in normal])), 2)
                                 if normal else None)
    out['boss_combats'] = dict(n=len(boss), win=sum(c['outcome'] == 'win' for c in boss),
                               hp_at_boss_mean=round(float(np.mean([c['hp_before'] for c in boss])), 2) if boss else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--seeds', required=True, help='range:START:COUNT')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--instances', type=int, default=4)
    p.add_argument('--mode', default='exact', choices=('exact', 'skip', 'render'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--take-items', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--no-bombs', action='store_true', help='the policy never places a bomb (its own bombs hurt it)')
    p.add_argument('--shield', action='store_true',
                   help='COMBAT: a move predicted to run into a harmful entity gives way to the best safe one (abplus_shield)')
    p.add_argument('--finisher', action='store_true',
                   help='normal rooms: the last COMBAT try is the scripted approach-and-shoot finisher')
    p.add_argument('--shield-horizon', type=int, default=None, help='logic frames the shield looks ahead (default 8)')
    p.add_argument('--shield-bombs', action='store_true', help='shield v2: live bombs as blasts when they go off')
    p.add_argument('--shield-creep', action='store_true', help='shield v2: enemy creep (effects 22, 23) as static threats')
    p.add_argument('--shield-margin', type=float, default=None, help='px beyond the two radii (default 4)')
    p.add_argument('--treasure-detour', type=int, default=2, help='rooms walked at most to fetch a seen treasure room')
    p.add_argument('--combat-seconds', type=float, default=180.0,
                   help='a COMBAT try (normal rooms); 180 s = the training episode, whose remaining time the policy sees')
    p.add_argument('--boss-seconds', type=float, default=180.0)
    p.add_argument('--combat-tries', type=int, default=3)
    p.add_argument('--boss-checkpoint', type=Path, default=None, help='fights in the boss room (same observation)')
    p.add_argument('--fallback-checkpoint', type=Path, default=None,
                   help='plays the tries after a COMBAT time-out (same observation as --checkpoint)')
    p.add_argument('--max-pickups', type=int, default=4)
    p.add_argument('--max-seconds', type=float, default=1800.0, help='game seconds per floor')
    p.add_argument('--replays', type=int, default=0)
    p.add_argument('--port', type=int, default=27800)
    p.add_argument('--name', default='flr')
    args = p.parse_args()
    if any(c.isdigit() for c in args.name):
        p.error('--name must not contain digits')
    try:
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('1000')
    except OSError:
        pass
    if args.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    config = json.loads((args.checkpoint / 'state.json').read_text())['config']
    start, count = map(int, args.seeds.split(':')[1:])
    seeds = list(range(start, start + count))
    args.replay_seeds = set(seeds[:max(0, args.replays)])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'results.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(l)['seed'] for l in path.read_text().splitlines() if 'error' not in json.loads(l)}
    todo = [s for s in seeds if s not in done]
    (args.out / 'meta.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()), deterministic=not args.stochastic, take_items=args.take_items,
        no_bombs=args.no_bombs, shield=args.shield, shield_bombs=args.shield_bombs, shield_creep=args.shield_creep,
        finisher=args.finisher, shield_horizon=args.shield_horizon, shield_margin=args.shield_margin,
        boss_checkpoint=str(args.boss_checkpoint) if args.boss_checkpoint else None,
        fallback_checkpoint=str(args.fallback_checkpoint) if args.fallback_checkpoint else None,
        treasure_detour=args.treasure_detour, combat_seconds=args.combat_seconds, boss_seconds=args.boss_seconds,
        combat_tries=args.combat_tries, max_seconds=args.max_seconds, seeds=len(seeds),
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    for s in todo:
        tasks.put(s)
    n = min(args.instances, len(todo))
    print(f'{len(done)} done, {len(todo)} to run on {n} instances', flush=True)
    procs = [ctx.Process(target=worker, args=(i, args, config, tasks, results), daemon=True) for i in range(n)]
    for proc in procs:
        proc.start()
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    finished = 0
    with open(path, 'a') as f:
        while finished < n:
            r = results.get()
            if r is None:
                finished += 1
                continue
            f.write(json.dumps(r) + '\n')
            f.flush()
            rows.append(r)
            if 'error' in r:
                print(time.strftime('%H:%M:%S'), r['seed'], 'ERROR', r['error'], flush=True)
            else:
                print(time.strftime('%H:%M:%S'), r['seed'], r['outcome'], 'rooms', r['rooms_entered'], 'combats',
                      [(c['kind'], c['outcome'], c['hp_before'], c['hp_after']) for c in r['combats']], 'items', r['items'],
                      f"{r['game_seconds']} s", flush=True)
    for proc in procs:
        proc.join(timeout=60)
    summary = summarize([r for r in rows if r.get('seed') in set(seeds)])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
