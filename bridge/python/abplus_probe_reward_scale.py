"""Reward scale of the room mixture: per room, the NPCs at the start and those spawned during a fight,
their HP, the engine's enemy flags, and what combat-v1's terms add up to.

Plays every mixture room (plus the Monstro arena) once with a naive policy: shoot along the dominant
axis at the nearest vulnerable enemy, step toward alignment on the other axis, some random moves.
The player gets 12 heart containers so the fight runs until the clear or the time cap. This is a
measurement of room content, not of a trained policy.
usage: python abplus_probe_reward_scale.py <mixture.json> <out.jsonl> [shard] [shards] [seconds]
"""
import json, sys, time
import numpy as np
from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus
from isaac_bridge.abplus_tasks import Task

spec = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
shards = int(sys.argv[4]) if len(sys.argv) > 4 else 1
seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
REPEAT = 4
FIELDS = ('order', 'first', 'type', 'variant', 'subtype', 'max_hp', 'hp0', 'min_hp', 'healed', 'enemy',
          'vulnerable', 'active', 'boss', 'can_shut_doors', 'can_shut_doors_field', 'champion', 'dead', 'last')

PROBE_LUA = r"""
ABP_PROBE = {}
local function b(f) local ok, v = pcall(f); if not ok then return -1 end; if v then return 1 end; return 0 end
function ABP_PROBE.reset() ABP_PROBE.seen = {}; ABP_PROBE.order = 0 end
function ABP_PROBE.track(frame)
  for _, e in ipairs(Isaac.GetRoomEntities()) do
    local npc = e:ToNPC()
    if npc then
      local key = tostring(e.Index) .. ':' .. tostring(e.InitSeed)
      local r = ABP_PROBE.seen[key]
      if not r then
        ABP_PROBE.order = ABP_PROBE.order + 1
        r = { order = ABP_PROBE.order, first = frame, t = e.Type, v = e.Variant, s = e.SubType, max = e.MaxHitPoints,
              enemy = b(function() return e:IsEnemy() end), vuln = b(function() return e:IsVulnerableEnemy() end),
              active = b(function() return e:IsActiveEnemy(false) end), boss = b(function() return e:IsBoss() end),
              csd = b(function() return e:CanShutDoors() end), csdf = b(function() return npc.CanShutDoors end),
              champ = b(function() return npc:IsChampion() end),
              hp0 = e.HitPoints, minhp = e.HitPoints, heal = 0, last = e.HitPoints, dead = 0, lastframe = frame }
        ABP_PROBE.seen[key] = r
      end
      local hp = e.HitPoints
      if hp > r.last then r.heal = r.heal + (hp - r.last) end
      if hp < r.minhp then r.minhp = hp end
      r.last = hp; r.lastframe = frame
      if e:IsDead() then r.dead = 1 end
    end
  end
end
function ABP_PROBE.dump()
  local out = {}
  for _, r in pairs(ABP_PROBE.seen) do
    out[#out + 1] = string.format('%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%d,%d,%d', r.order, r.first,
      r.t, r.v, r.s, r.max, r.hp0, r.minhp, r.heal, r.enemy, r.vuln, r.active, r.boss, r.csd, r.csdf, r.champ, r.dead, r.lastframe)
  end
  return table.concat(out, ';')
end
return 'ok'
"""
START_LUA = """ABP_PROBE.reset(); ABP_PROBE.track(0)
local p = Isaac.GetPlayer(0); p:AddMaxHearts(18); p:AddHearts(24)
local r = Game():GetRoom()
return tostring(r:GetAliveEnemiesCount()) .. ',' .. tostring(r:GetAliveBossesCount())"""


class Skip(Exception):
    pass


class Fixed:
    """Task sampler that always returns one room; an unusable room skips to the next seed."""
    def __init__(self, task):
        self.task = task

    def choose(self, seed, retry=0):
        if retry:
            raise Skip(seed)
        return self.task


def naive(obs, rng, state):
    px, py = obs['players'][0]['pos']
    targets = [e for e in obs['entities'] if e.get('enemy') and e.get('vulnerable') and e['type'] != 33]
    shoot = move = 0
    if targets:
        t = min(targets, key=lambda e: (e['pos'][0] - px) ** 2 + (e['pos'][1] - py) ** 2)
        dx, dy = t['pos'][0] - px, t['pos'][1] - py
        if abs(dx) >= abs(dy):
            shoot = 3 if dx > 0 else 7
            move = (5 if dy > 0 else 1) if abs(dy) > 12 else 0
            if abs(dx) < 70:
                move = 7 if dx > 0 else 3
        else:
            shoot = 5 if dy > 0 else 1
            move = (3 if dx > 0 else 7) if abs(dx) > 12 else 0
            if abs(dy) < 70:
                move = 1 if dy > 0 else 5
        shoot = {1: 1, 3: 2, 5: 3, 7: 4}[shoot]
    if state['k'] > 0 or rng.random() < 0.15:
        if state['k'] <= 0:
            state['move'], state['k'] = int(rng.integers(9)), int(rng.integers(3, 8))
        state['k'] -= 1
        move = state['move']
    return {'move': move, 'shoot': shoot}


rooms = [('arena', 0)] + [('normal', v) for v in spec['normal']] + [('boss', v) for v in spec['boss']]
rooms = rooms[shard::shards]
name, port = f'rs{shard}', 27380 + shard
proc = launch_abplus(name, port, 'exact', nice=19)
bridge = AbplusTrainingEnv(port=port)
bridge.binary_obs = True
t0 = time.time()
try:
    bridge.connect()
    assert bridge.lua(PROBE_LUA) == 'ok'
    with open(out_path, 'w', encoding='utf8') as out:
        for i, (kind, variant) in enumerate(rooms):
            row = dict(kind=kind, variant=variant)
            for seed in range(5000 + 7 * i, 5000 + 7 * i + 3):
                bridge.tasks = Fixed(Task(kind, variant, seed % 4))
                try:
                    obs, info = bridge.reset_monstro(seed, 6, 1.0)
                except Skip:
                    continue
                row.update(seed=seed, entrance=info.get('entrance'))
                break
            else:
                row['skipped'] = True
                out.write(json.dumps(row) + '\n')
                continue
            alive = bridge.lua(START_LUA)
            row['alive_start'] = [int(x) for x in alive.split(',')]
            player = obs['players'][0]
            row['player'] = {k: player.get(k) for k in ('damage', 'fire_delay_max', 'shot_speed', 'range', 'speed')}
            c0 = dict(obs['combat'])
            rng = np.random.default_rng(seed)
            state = {'k': 0, 'move': 0}
            frames, outcome = 0, 'time_limit'
            while frames < seconds * 30:
                obs = bridge.step(naive(obs, rng, state), repeat=REPEAT)[0]
                frames += REPEAT
                bridge.lua(f'ABP_PROBE.track({frames})')
                if obs['players'][0]['dead']:
                    outcome = 'death'
                    break
                if obs['room']['clear']:
                    outcome = 'clear'
                    break
            c = obs['combat']
            row.update(outcome=outcome, frames=frames,
                       combat={k: c[k] - c0.get(k, 0) for k in ('player_damage_events', 'player_damage',
                                                                'enemy_damage_events', 'enemy_damage',
                                                                'enemy_damage_fraction')})
            dump = bridge.lua('return ABP_PROBE.dump()') or ''
            row['npcs'] = sorted((dict(zip(FIELDS, (float(x) if '.' in x else int(x) for x in rec.split(','))))
                                  for rec in dump.split(';') if rec), key=lambda r: r['order'])
            out.write(json.dumps(row) + '\n')
            out.flush()
            if i % 20 == 0:
                print(f'{i}/{len(rooms)} {kind} {variant} {outcome} {frames}f {time.time() - t0:.0f}s', flush=True)
finally:
    try:
        bridge.close()
    finally:
        stop_abplus(proc, name)
print('done', len(rooms), f'{time.time() - t0:.0f}s', flush=True)
