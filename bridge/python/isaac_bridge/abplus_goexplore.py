"""Go-Explore on AB+ rooms, phase 1 (exploration): per room an archive of cells; choose a cell, return to it, explore
from it (Ecoffet et al., Go-Explore; user request 2026-09-30, EXPERIMENTS.md A8). Driver: goexplore_abplus.py.

Return. reset(seed), then the cell's actions in one batched replay (bridge abp-0.2.12 play: the frames of the same step
calls, one observation at the end), then checked: every action must play without ending the episode, and the
hidden-state digest (DIGEST_LUA, as in A7: the player, every entity incl. NPC AI state, seeds, grid, grid paths and the
global MT) must equal the one taken when the cell was reached. After the abp_turbo fixes A7 found such replays
identical across instance histories in 127 of 128 rooms; a rare residual (a Mulligan, cause unknown) remains, so a
return can diverge. A failed return explores nothing and counts against the cell, which is retired after max_fails
failed returns in a row.

Cells (domain knowledge; the attributes and bins are our choice): the player's position in cell_px bins from the room's
top left, health (half hearts, as the bridge's health_units), bombs (capped at 2), the doors-blocking NPCs alive and
their summed HP in hp_buckets of the room's starting blocking HP. A won room is one cell per health. A cell keeps the
best trajectory that reached it by rank (damage taken, logic frames, -monster damage): the least damage taken (the
bridge's combat.player_damage, half hearts; health alone would hide damage made up by hearts picked up), then the
fewest frames, then the most monster damage.

Selection weight (count-based as in Go-Explore; the constants and the progress and health factors are our choice):
    (1/sqrt(chosen + 1) + 1/sqrt(chosen since the cell last led to a new or better cell + 1) + 0.5/sqrt(seen + 1))
    x (1 + progress_bonus x progress) x hurt_factor^(damage taken) x 0.5^(failed returns in a row)
with progress = 1 - blocking HP / starting blocking HP, clipped to 0..1. Won cells are kept but never chosen; deaths,
time-outs (and with hurt_ends any damage) end an exploration and are not archived.

Exploration from a cell: explore_steps decisions of a sticky random policy. With repeat_prob the previous move and shoot
are kept; otherwise a new move (uniform over the 9) and shoot (with aim_prob along the dominant axis towards the nearest
doors-blocking NPC, else uniform over the 5). A bomb with bomb_prob while the player has one; never the active item.

Action codes: joint + 45 bomb + 90 item (joint = 5 move + shoot, AbplusTransformerEnv's action), one byte each.
"""
import hashlib
import math
import os
import queue
import signal
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np

from .abplus import ABP_HOME, AbplusTransformerEnv, kill_abplus, launch_abplus, stop_abplus
from .abplus_tasks import TaskSampler

# The A7 digest (EXPERIMENTS.md A7) without its trace; diag adds the lines that differ between instance histories by
# design (global frame counter, entity list order and indices, abp_turbo counters). Returns the digest text.
DIGEST_LUA = r'''
function ABPGX_DIGEST(diag)
  local game = Game()
  local room = game:GetRoom()
  local level = game:GetLevel()
  local function n(x)
    if type(x) == 'number' then return string.format('%.17g', x) end
    return tostring(x)
  end
  local function v(p)
    if p == nil or p == 'x' then return tostring(p) end
    return n(p.X) .. ',' .. n(p.Y)
  end
  local function try(f)
    local ok, r = pcall(f)
    if ok then return r end
    return 'x'
  end
  local out = {}
  local function add(...) out[#out + 1] = table.concat({...}, '|') end
  add('G', n(game:GetFrameCount()), n(room:GetFrameCount()), tostring(room:IsClear()), n(room:GetAliveEnemiesCount()),
      n(try(function() return room:GetAwardSeed() end)), n(try(function() return room:GetDecorationSeed() end)),
      n(try(function() return room:GetSpawnSeed() end)),
      n(try(function() return level:GetCurrentRoomDesc().SpawnSeed end)),
      n(try(function() return game:GetSeeds():GetStartSeed() end)))
  add('M', tostring(os.getenv('ABP_MT')))
  for i = 0, game:GetNumPlayers() - 1 do
    local p = Isaac.GetPlayer(i)
    add('P', v(p.Position), v(p.Velocity), n(p:GetHearts()), n(p:GetSoulHearts()), n(p:GetMaxHearts()),
        n(p.FireDelay), n(p.MaxFireDelay), n(p:GetDamageCooldown()), n(p:GetTotalDamageTaken()),
        n(p:GetHeadDirection()), n(p:GetFireDirection()), n(p:GetMovementDirection()), n(p:GetSprite():GetFrame()),
        n(p.FrameCount), n(p.InitSeed), n(p.DropSeed), n(p.TearHeight), n(p.TearFallingSpeed), n(p.ShotSpeed),
        n(p.Damage), n(p.MoveSpeed), n(p:GetNumBombs()), n(p:GetNumCoins()), n(p:GetNumKeys()),
        n(p.EntityCollisionClass), n(p.GridCollisionClass), tostring(p.Visible), tostring(p.ControlsEnabled),
        n(try(function() return p:GetDropRNG():GetSeed() end)))
  end
  local ents, order, idx = {}, {}, {}
  for _, e in ipairs(Isaac.GetRoomEntities()) do
    order[#order + 1] = n(e.Type) .. '.' .. n(e.Variant) .. '.' .. n(e.InitSeed)
    idx[#idx + 1] = n(e.Index)
    if e.Type ~= 1 then
      local f = {'E', n(e.Type), n(e.Variant), n(e.SubType), n(e.InitSeed), n(e.DropSeed), v(e.Position),
        v(e.Velocity), n(e.HitPoints), n(e.MaxHitPoints), n(e.FrameCount), tostring(e.Visible),
        n(e.EntityCollisionClass), n(e.GridCollisionClass), n(e.CollisionDamage), n(e.Size), n(e.SpawnerType),
        n(e.SpawnerVariant), n(e:GetSprite():GetFrame()), tostring(e:IsDead()),
        n(try(function() return e:GetDropRNG():GetSeed() end))}
      local npc = e:ToNPC()
      if npc then
        f[#f + 1] = table.concat({n(npc.State), n(npc.StateFrame), n(npc.ProjectileCooldown),
          n(npc.ProjectileDelay), n(npc.I1), n(npc.I2), v(npc.V1), v(npc.V2),
          v(try(function() return npc.TargetPosition end)), n(npc:GetChampionColorIdx())}, ',')
      end
      local tear = e:ToTear()
      if tear then
        f[#f + 1] = table.concat({n(tear.Height), n(tear.FallingSpeed),
          n(try(function() return tear.FallingAcceleration end)), n(tear.Scale),
          n(try(function() return tear.TearFlags end))}, ',')
      end
      local pr = e:ToProjectile()
      if pr then
        f[#f + 1] = table.concat({n(pr.Height), n(pr.FallingSpeed), n(pr.FallingAccel), n(pr.Scale),
          n(try(function() return pr.ProjectileFlags end))}, ',')
      end
      local ef = e:ToEffect()
      if ef then
        f[#f + 1] = table.concat({n(try(function() return ef.Timeout end)), n(try(function() return ef.State end)),
          n(try(function() return ef.LifeSpan end))}, ',')
      end
      ents[#ents + 1] = table.concat(f, '|')
    end
  end
  table.sort(ents)
  for _, s in ipairs(ents) do out[#out + 1] = s end
  for i = 0, room:GetGridSize() - 1 do
    local g = room:GetGridEntity(i)
    if g then add('R', n(i), n(g:GetType()), n(g:GetVariant()), n(g.State), n(g.CollisionClass)) end
  end
  local paths = {}
  for i = 0, room:GetGridSize() - 1 do
    local c = try(function() return room:GetGridPath(i) end)
    if c ~= 0 then paths[#paths + 1] = n(i) .. ':' .. n(c) end
  end
  add('Q', table.concat(paths, ' '))
  if diag then
    add('D', 'turbo', tostring(os.getenv('ABP_TURBO_COUNTERS')))
    add('D', 'frames', n(Isaac.GetFrameCount()))
    add('D', 'order', table.concat(order, ' '))
    add('D', 'index', table.concat(idx, ' '))
  end
  return table.concat(out, '\n')
end
return 'ok'
'''

WIN = 'win'
TERMINAL = ('death', 'time_limit', 'hurt')   # end an exploration, never archived


@dataclass
class GxConfig:
    """Everything a worker and an archive need (picklable: goes to the spawned workers)."""
    # cells and selection
    cell_px: float = 80.0
    hp_buckets: int = 10
    progress_bonus: float = 2.0
    hurt_factor: float = 0.5
    max_fails: int = 3
    retry_mismatch: int = 1       # a return whose digest differs is repeated once in the same instance (0: no)
    # exploration
    explore_steps: int = 50
    repeat_prob: float = 0.9
    aim_prob: float = 0.5
    bomb_prob: float = 0.01
    hurt_ends: bool = False
    # episode
    frames_per_decision: int = 4
    start_hp: int = 6
    bombs: int = 1
    lineage_mode: int = 3
    # instances
    mode: str = 'exact'
    bridge_lua: str = ''
    preload: str = ''             # libabp_turbo.so for the instances ('' = $ABP_HOME/tools)
    al_stopped: bool = True       # ABP_AL_STOPPED: OpenAL sources read as stopped (A8: otherwise play follows real time)
    stub_list: str = ''           # ABP_STUB_LIST for the instances ('' = the mode's own list)
    nice: int = 10
    recycle_rss_mib: float = 600.0
    die_with_parent: bool = False # the instance gets SIGKILL when its launching process ends (abplus.launch_abplus)


def pack(joint, bomb=0, item=0):
    return int(joint) + 45 * int(bomb) + 90 * int(item)


def unpack(code):
    code = int(code)
    return code % 45, (code // 45) % 2, code // 90


def health_units(p):
    """Half hearts that absorb damage before death (abp_bridge.lua health_units)."""
    return int(round(p['hearts'] + p['soul'] + p['eternal'] + 2 * p['bone']))


def room_context(obs):
    """The room's start, which every cell of its archive is measured against."""
    return dict(hp0=float(obs['combat']['blocking_hp']), health0=health_units(obs['players'][0]))


def describe(obs, ctx, cfg):
    """(cell key, data) of an observation."""
    p, c = obs['players'][0], obs['combat']
    health = health_units(p)
    hp = max(0.0, float(c['blocking_hp']))
    hp0 = float(ctx['hp0'])
    progress = min(1.0, max(0.0, 1.0 - hp / hp0)) if hp0 > 0 else 1.0
    data = dict(health=health, bombs=int(p['bombs']), alive=int(c['blocking_count']), blocking_hp=round(hp, 3),
                progress=round(progress, 6), dealt=round(float(c.get('monster_damage', 0.0)), 6),
                hurt=round(float(c.get('player_damage', 0.0)), 6),
                pos=[round(p['pos'][0], 1), round(p['pos'][1], 1)])
    if obs['room']['clear'] and not p['dead']:
        return (WIN, health), data
    left, top = obs['room']['top_left']
    gx = int((p['pos'][0] - left) // cfg.cell_px)
    gy = int((p['pos'][1] - top) // cfg.cell_px)
    bucket = 0 if hp <= 0 or hp0 <= 0 else min(2 * cfg.hp_buckets, math.ceil(cfg.hp_buckets * hp / hp0 - 1e-9))
    return (gx, gy, health, min(int(p['bombs']), 2), int(c['blocking_count']), bucket), data


def aim(obs):
    """Shoot direction (1 up, 2 right, 3 down, 4 left) along the dominant axis towards the nearest doors-blocking NPC,
    None without one."""
    p = obs['players'][0]
    best = None
    for e in obs['entities']:
        if e.get('blocking'):
            dx, dy = e['pos'][0] - p['pos'][0], e['pos'][1] - p['pos'][1]
            d = dx * dx + dy * dy
            if best is None or d < best[0]:
                best = (d, dx, dy)
    if best is None:
        return None
    _, dx, dy = best
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


class Explorer:
    """The sticky random exploration policy (module docstring)."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.prev = None

    def act(self, obs):
        cfg, rng = self.cfg, self.rng
        if self.prev is not None and rng.random() < cfg.repeat_prob:
            move, shoot = self.prev
        else:
            move = int(rng.integers(9))
            shoot = aim(obs) if rng.random() < cfg.aim_prob else None
            if shoot is None:
                shoot = int(rng.integers(5))
        self.prev = (move, shoot)
        bomb = int(obs['players'][0]['bombs'] > 0 and rng.random() < cfg.bomb_prob)
        return 5 * move + shoot, bomb, 0


def rank(data, frames):
    """Order of trajectories reaching the same cell, lower is better: damage taken, logic frames, -monster damage."""
    return float(data['hurt']), int(frames), -float(data['dealt'])


def better(r, cell):
    """A trajectory of rank r reaching an archived cell replaces the cell's own."""
    return r < cell.rank


@dataclass
class Cell:
    key: tuple
    actions: bytes
    frames: int
    dealt: float
    hurt: float
    health: int
    progress: float
    outcome: str
    digest: str
    data: dict
    found: int                 # iteration of the first discovery
    improved_at: int = -1      # iteration of the last better trajectory
    chosen: int = 0
    since_new: int = 0         # times chosen since it last led to a new or better cell
    seen: int = 0              # explorations that passed through it
    fails: int = 0             # failed returns in a row
    fails_total: int = 0
    returns: int = 0
    improvements: int = 0

    @property
    def rank(self):
        return self.hurt, self.frames, -self.dealt

    def summary(self, with_actions=False):
        out = {k: v for k, v in asdict(self).items() if k not in ('actions', 'key')}
        out['key'] = list(self.key)
        out['steps'] = len(self.actions)
        if with_actions:
            out['actions'] = list(self.actions)
        return out


class Archive:
    """One room's cells (by key), their selection weights and the log of (key, rank) of every new or better
    trajectory, which the workers' local copies follow."""

    def __init__(self, seed, root, cfg):
        self.seed, self.cfg = int(seed), cfg
        self.root = root                                  # key, data, digest, hp0, health0, room
        self.ctx = dict(hp0=root['hp0'], health0=root['health0'])
        self.cells, self.index = [], {}
        self.weights = np.zeros(256)
        self.log = []
        self.dispatched = 0                               # exploration tasks sent
        self.stats = Counter()
        self._add(tuple(root['key']), b'', 0, root['data'], 'running', root['digest'], 0)

    def _add(self, key, actions, frames, data, outcome, digest, iteration):
        cell = Cell(key=key, actions=bytes(actions), frames=int(frames), dealt=float(data['dealt']),
                    hurt=float(data['hurt']), health=int(data['health']), progress=float(data['progress']), outcome=outcome, digest=digest,
                    data=dict(data), found=iteration)
        self.index[key] = len(self.cells)
        self.cells.append(cell)
        if len(self.cells) > len(self.weights):
            self.weights = np.concatenate([self.weights, np.zeros(len(self.weights))])
        self._refresh(len(self.cells) - 1)
        self.log.append((key, cell.rank))
        return cell

    def weight(self, c):
        cfg = self.cfg
        if c.outcome != 'running' or c.fails >= cfg.max_fails:
            return 0.0
        count = 1 / math.sqrt(c.chosen + 1) + 1 / math.sqrt(c.since_new + 1) + 0.5 / math.sqrt(c.seen + 1)
        return count * (1 + cfg.progress_bonus * c.progress) * cfg.hurt_factor ** c.hurt * 0.5 ** c.fails

    def _refresh(self, i):
        self.weights[i] = self.weight(self.cells[i])

    def selectable(self):
        return float(self.weights[:len(self.cells)].sum()) > 0

    def select(self, rng):
        """A cell by weight (its chosen counters count this choice), None when none is selectable."""
        w = self.weights[:len(self.cells)]
        total = float(w.sum())
        if total <= 0:
            return None
        i = min(int(np.searchsorted(np.cumsum(w), rng.random() * total, side='right')), len(self.cells) - 1)
        cell = self.cells[i]
        cell.chosen += 1
        cell.since_new += 1
        self._refresh(i)
        return cell

    def failed(self, key, error=False):
        """A return to the cell did not reproduce it (or its task failed with an error)."""
        i = self.index[key]
        cell = self.cells[i]
        cell.fails += 1
        cell.fails_total += 1
        self.stats['return_errors' if error else 'return_fails'] += 1
        if cell.fails >= self.cfg.max_fails:
            self.stats['retirements'] += 1
        self._refresh(i)

    def merge(self, key, prefix, result, iteration):
        """An exploration from `key` that started from the trajectory `prefix` (the one the task replayed: the cell may
        have a better one by now). Returns (new cells, better trajectories)."""
        new = improved = 0
        explore = result['explore']
        for cand in result['cands']:
            k = tuple(cand['key'])
            if cand['outcome'] in TERMINAL:
                continue
            actions = prefix + explore[:cand['n']]
            j = self.index.get(k)
            if j is None:
                self._add(k, actions, cand['frames'], cand['data'], cand['outcome'], cand['digest'], iteration)
                new += 1
            elif better(rank(cand['data'], cand['frames']), self.cells[j]):
                c = self.cells[j]
                c.actions, c.frames, c.dealt = bytes(actions), int(cand['frames']), float(cand['data']['dealt'])
                c.hurt = float(cand['data']['hurt'])
                c.digest, c.data, c.improved_at = cand['digest'], dict(cand['data']), iteration
                c.health, c.progress = int(cand['data']['health']), float(cand['data']['progress'])
                c.improvements += 1
                c.fails = 0   # a new trajectory: its returns start over
                self._refresh(j)
                self.log.append((k, c.rank))
                improved += 1
        for k, n in result['seen'].items():
            j = self.index.get(tuple(k))
            if j is not None:
                self.cells[j].seen += int(n)
                self._refresh(j)
        i = self.index.get(key)
        if i is not None:
            cell = self.cells[i]
            cell.returns += 1
            cell.fails = 0
            if new or improved:
                cell.since_new = 0
            self._refresh(i)
        self.stats['returns'] += 1
        self.stats['new'] += new
        self.stats['improved'] += improved
        for end, n in result['ends'].items():
            self.stats['end_' + end] += int(n)
        return new, improved

    def best(self):
        """The best state reached: a win first, then less damage taken, more health, more progress, fewer frames."""
        return max(self.cells, key=lambda c: (c.outcome == WIN, -c.hurt, c.health, c.progress, -c.frames))

    def wins(self):
        """The won cells, best first: least damage taken, most health, fewest frames."""
        return sorted((c for c in self.cells if c.outcome == WIN), key=lambda c: (c.hurt, -c.health, c.frames))

    def summary(self):
        running = [c for c in self.cells if c.outcome == 'running']
        best = self.best()
        out = {k: int(v) for k, v in self.stats.items()}
        out.update(cells=len(self.cells), running=len(running),
                   retired=sum(c.fails >= self.cfg.max_fails for c in running), wins=len(self.wins()),
                   best=dict(outcome=best.outcome, hurt=best.hurt, health=best.health, progress=best.progress,
                             frames=best.frames, steps=len(best.actions)),
                   dispatched=self.dispatched)
        return out


# ------------------------------------------------------------------------------------------------------------ workers

class ExploreEnv(AbplusTransformerEnv):
    """AbplusTransformerEnv that hands out the raw observation (exploration needs no policy input)."""

    def encode_observation(self, obs):
        return obs


def make_env(port, cfg, spec):
    """The environment of a task spec: {'spec': {weights, normal, boss}, 'target', 'seconds'} (abplus_groups style)."""
    frames = int(round(float(spec['seconds']) * 30))
    env = ExploreEnv(port=port, max_episode_frames=frames, deadline_s=float(spec['seconds']),
                     frames_per_decision=int(cfg.frames_per_decision))
    env.bridge.binary_obs = True
    env.bridge.lineage_mode = int(cfg.lineage_mode)
    env.bridge.invincible = False
    env.bridge.miss_cap = 0
    env.bridge.target = spec.get('target')
    s = spec['spec']
    env.bridge.tasks = TaskSampler(s['weights'], s['normal'], s['boss'], (spec.get('target') or {}).get('arms'))
    return env


class Instance:
    """One AB+ process and its environment for Go-Explore tasks."""

    def __init__(self, name, port, cfg, spec):
        self.name, self.port, self.cfg, self.spec = name, port, cfg, spec
        self.proc = self.env = None
        self.digest_ready = False
        self.launches = 0
        self.launch()

    def launch(self):
        extra = {}
        if self.cfg.preload:
            extra['ABP_PRELOAD'] = self.cfg.preload
        if self.cfg.al_stopped:
            extra['ABP_AL_STOPPED'] = '1'
        if self.cfg.stub_list:
            extra['ABP_STUB_LIST'] = self.cfg.stub_list
        # 2026-10-06 (the root's first-build hang): the bridge writes the port it listens on to this file, and binds a
        # free port when the configured one is taken (abp_bridge.lua try_bind); the client connects to the file's port
        # (AbplusTrainingEnv.connect). A stale file of the previous launch goes first.
        home = ABP_HOME / 'instances' / self.name.lower()
        self.port_file = home / 'bridge_port'
        self.stack_dir = home / 'stalls' / 'stacks'   # abp_turbo writes SIGUSR2 stack dumps here (tok_sampler)
        try:
            self.stack_dir.mkdir(parents=True, exist_ok=True)
            self.port_file.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        extra['ISAAC_RL_PORT_FILE'] = str(self.port_file)
        extra['ABP_STACK_DUMP_DIR'] = str(self.stack_dir)
        self.proc = launch_abplus(self.name, self.port, self.cfg.mode, bridge_lua=self.cfg.bridge_lua or None,
                                  nice=self.cfg.nice, extra_env=extra or None,
                                  die_with_parent=getattr(self.cfg, 'die_with_parent', False))
        self.env = make_env(self.port, self.cfg, self.spec)
        self.env.bridge.port_file = str(self.port_file)
        self.env.bridge.process_exit = self.proc.poll   # the connect loop gives up at once when the game has ended
        self.digest_ready = False
        self.launches += 1

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass
        finally:
            self.env = None
            if self.proc is not None:
                stop_abplus(self.proc, self.name)
                try:
                    self.proc.wait(timeout=30)
                except Exception:   # still there 30 s after SIGTERM (stopped or hung): SIGKILL (2026-10-05)
                    kill_abplus(self.proc, self.name)
                self.proc = None

    def kill(self):
        """End this instance now, whatever state it is in (2026-10-05; tok_sampler: a root instance hung during a
        reset that a helper thread runs). The client is aborted first (AbplusTrainingEnv.abort: a recv blocked in that
        thread returns, its connect loop can no longer reach this port), then abplus.kill_abplus (SIGKILL to the
        process group, clones included, and to every game process of this name). Returns the pids still alive
        afterwards (empty: all gone)."""
        env, self.env = self.env, None
        if env is not None:
            try:
                env.bridge.abort()
            except Exception:
                pass
        proc, self.proc = self.proc, None
        self.digest_ready = False
        return kill_abplus(proc, self.name)

    def retire(self):
        """End this instance's game process alone (2026-10-06, the tok workers' root recycling): the client is given up
        (abort) and closed, SIGKILL goes to the game's pid only (launch_abplus execs it, so it is the launcher's pid) and
        the launcher is reaped. Its fork clones (parked templates and room entries: same name and process group) are left
        running: they live as long as their own connections. launch() then starts a new process."""
        env, self.env = self.env, None
        if env is not None:
            try:
                env.bridge.abort()
                if env.bridge._sock is not None:
                    env.bridge._sock.close()
                    env.bridge._sock = None
            except Exception:
                pass
        proc, self.proc = self.proc, None
        self.digest_ready = False
        if proc is not None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def relaunch(self, hard=False):
        """A new process for this instance; hard: the old one killed (kill) instead of closed."""
        if hard:
            self.kill()
        else:
            self.close()
        time.sleep(1.0)
        self.launch()

    def rss_mib(self, field='VmRSS'):
        """The game process's resident memory (MiB) from /proc/<pid>/status: VmRSS, or RssAnon (its private heap and
        stacks, without the files and shared memory it maps, 2026-10-06)."""
        try:
            with open(f'/proc/{self.proc.pid}/status') as f:
                for line in f:
                    if line.startswith(field + ':'):
                        return int(line.split()[1]) / 1024
        except (AttributeError, OSError, ValueError):
            pass
        return 0.0

    def reset(self, seed):
        obs, info = self.env.reset(options={'arena_seed': int(seed), 'start': (int(self.cfg.start_hp), 1.0),
                                            'bombs': int(self.cfg.bombs)})
        if not self.digest_ready:   # the bridge connects on the first reset; the function lives in its Lua state
            if self.env.bridge.lua(DIGEST_LUA) != 'ok':
                raise RuntimeError('digest function not installed')
            counters = self.env.bridge.lua("return tostring(os.getenv('ABP_TURBO_COUNTERS'))") or ''
            if self.cfg.al_stopped and 'al_stopped=' not in counters:
                raise RuntimeError(f'al_stopped needs a libabp_turbo.so with ABP_AL_STOPPED (A8), got {counters!r}: '
                                   f'build analysis/scripts/abplus/abp_turbo.c and pass it as preload')
            self.digest_ready = True
        return obs, info

    def digest_text(self, diag=False):
        return self.env.bridge.lua(f'return ABPGX_DIGEST({"true" if diag else "false"})')

    def digest(self):
        return hashlib.blake2b(self.digest_text().encode('utf8'), digest_size=16).hexdigest()


def replay(inst, task, actions, out):
    """Return to the task's cell: reset, one batched replay, the digest. (failure fields or None, digest or None)."""
    env = inst.env
    t = time.perf_counter()
    inst.reset(task['seed'])
    out['reset_s'] += time.perf_counter() - t
    t = time.perf_counter()
    played, outcome = 0, 'running'
    if actions:
        _, played, outcome = env.play(actions)
    out['return_s'] += time.perf_counter() - t
    out['return_frames'] += env.elapsed_frames
    if played != len(actions) or outcome != task.get('outcome', 'running') or env.elapsed_frames != task['frames']:
        return dict(reason='ended', played=played, outcome=outcome, frames=env.elapsed_frames), None
    t = time.perf_counter()
    digest = inst.digest()
    out['digest_s'] += time.perf_counter() - t
    out['digests'] += 1
    return None, digest


def run_task(inst, task, known, cfg):
    """One task: 'root' (reset, the room's first cell, explore), 'explore' (return to a cell, check, explore) or
    'verify' (return, check). known: key -> rank of this room's archive as far as this worker knows.
    A return whose digest differs is repeated once in the same instance with retry_mismatch (out['retry']: whether the
    second one matched the cell, and whether it equalled the first: a divergence of this instance's history or not)."""
    env = inst.env
    out = dict(status='ok', explore=b'', cands=[], seen={}, ends={}, return_frames=0, return_s=0.0, reset_s=0.0,
               explore_frames=0, explore_s=0.0, digest_s=0.0, digests=0)
    if task['kind'] == 'root':
        t = time.perf_counter()
        obs, info = inst.reset(task['seed'])
        out['reset_s'] += time.perf_counter() - t
        ctx = room_context(obs)
        key, data = describe(obs, ctx, cfg)
        t = time.perf_counter()
        digest = inst.digest()
        out['digest_s'] += time.perf_counter() - t
        out['digests'] += 1
        room = obs['room']
        out['root'] = dict(key=key, data=data, digest=digest, **ctx,
                           room=dict(task=info.get('task'), variant=room.get('variant'), subtype=room.get('subtype'),
                                     name=room.get('name'), shape=room.get('shape'), retries=info.get('room_retries'),
                                     target_arm=info.get('target_arm')))
        if ctx['hp0'] <= 0:
            out['status'] = 'unusable'
            return out
        known.setdefault(key, rank(data, 0))
    else:
        ctx = task['ctx']
        actions = [unpack(c) for c in task['actions']]
        failure, digest = replay(inst, task, actions, out)
        if failure is None and digest != task['digest']:
            failure = dict(reason='digest')
            if cfg.retry_mismatch:
                again, second = replay(inst, task, actions, out)
                out['retry'] = dict(match=second == task['digest'], same=second == digest,
                                    ended=again is not None)
                if again is None and second == task['digest']:
                    failure = None
        if failure is not None:
            out.update(status='fail', **failure)
            return out
        key, data = describe(env.raw_obs, ctx, cfg)
        if key != tuple(task['key']):
            out.update(status='fail', reason='key', key=key)
            return out
        if task['kind'] == 'verify':
            return out
    # exploration
    rng = np.random.default_rng(task['rng'])
    explorer = Explorer(cfg, rng)
    explore, cands, seen, ends = bytearray(), [], Counter(), Counter()
    t1, f1 = time.perf_counter(), env.elapsed_frames
    for i in range(cfg.explore_steps):
        action = explorer.act(env.raw_obs)
        _, _, terminated, truncated, info = env.step(np.asarray(action))
        explore.append(pack(*action))
        obs = env.raw_obs
        outcome = info['outcome']
        if cfg.hurt_ends and outcome == 'running' and float(obs['combat']['player_damage']) > 0:
            outcome = 'hurt'
        if outcome in TERMINAL:
            ends[outcome] += 1
            break
        key, data = describe(obs, ctx, cfg)
        seen[key] += 1
        frames = env.elapsed_frames
        r = rank(data, frames)
        old = known.get(key)
        if old is None or r < old:
            known[key] = r
            t = time.perf_counter()
            digest = inst.digest()
            out['digest_s'] += time.perf_counter() - t
            out['digests'] += 1
            cands.append(dict(key=key, n=i + 1, frames=frames, outcome=outcome, digest=digest, data=data))
        if outcome != 'running':   # won
            ends[outcome] += 1
            break
    out.update(explore=bytes(explore), cands=cands, seen=dict(seen), ends=dict(ends),
               explore_frames=env.elapsed_frames - f1, explore_s=time.perf_counter() - t1)
    return out


def apply_delta(known, delta):
    for key, r in delta:
        old = known.get(key)
        if old is None or r < old:
            known[key] = r


def _exit_on_term(signum, frame):
    raise SystemExit(143)


def worker_main(index, name, port, cfg, spec, tasks, results):
    """Worker process: one instance; tasks from its own queue (None ends), results to the shared queue."""
    try:
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('500')
    except OSError:
        pass
    # A driver started in the background passes SIGINT on as ignored; SIGTERM ends the loop so the instance is closed.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _exit_on_term)
    inst = None
    known = {}   # seed -> key -> rank
    parent = os.getppid()
    try:
        inst = Instance(name, port, cfg, spec)
        results.put(dict(worker=index, kind='ready'))
        while True:
            try:
                task = tasks.get(timeout=30)
            except queue.Empty:
                if os.getppid() != parent:   # the driver is gone: close the instance
                    break
                continue
            if task is None:
                break
            for seed in task.get('drop', ()):
                known.pop(seed, None)
            room = known.setdefault(task['seed'], {})
            apply_delta(room, task.get('delta', ()))
            t = time.perf_counter()
            try:
                out = run_task(inst, task, room, cfg)
            except Exception:
                out = dict(status='error', error=traceback.format_exc()[-2000:])
                try:
                    inst.relaunch()
                except Exception:
                    out['relaunch_error'] = traceback.format_exc()[-1000:]
            out.update(worker=index, kind='result', id=task['id'], seed=task['seed'], task=task['kind'],
                       seconds=time.perf_counter() - t)
            results.put(out)
            if cfg.recycle_rss_mib > 0 and inst.rss_mib() > cfg.recycle_rss_mib:
                inst.relaunch()
                results.put(dict(worker=index, kind='recycled'))
    except Exception:
        results.put(dict(worker=index, kind='fatal', error=traceback.format_exc()[-3000:]))
    finally:
        if inst is not None:
            inst.close()
