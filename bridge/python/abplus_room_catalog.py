"""Room catalog of the live AB+ engine for the room-level training mixture.

For every candidate id it runs `goto d.<id>` (normal rooms of the current stage, Basement I) or
`goto s.boss.<id>` and records what the engine actually loads: room type/variant/name/shape,
grid size, doors, the enemies the layout spawned (type/variant/subtype/boss/champion) and the
grid entities. A goto of an id that does not exist leaves the room unchanged, so the reset does
not wait for a room change; the room frame counter restarting marks a real load.
The room data comes from the engine itself (AB+ ships its own room files; the repository's XML
set is Repentance+).
usage: python abplus_room_catalog.py <out.json> [normal_max] [boss_min] [boss_max]
"""
import json
import sys
import time

from isaac_bridge.abplus import AbplusTrainingEnv, launch_abplus, stop_abplus

SETTLE = 8


def probe(env, command):
    # wait_room=False: a failed goto never fires MC_POST_NEW_ROOM, so do not wait for one.
    env._send({'cmd': 'reset', 'commands': [command], 'settle': SETTLE, 'wait_room': False})
    obs = env._expect_obs()
    return obs, obs['room']['frame'] <= SETTLE + 2


def describe(obs):
    room = obs['room']
    enemies = [dict(type=e['type'], variant=e['variant'], subtype=e['subtype'], boss=bool(e.get('boss')),
                    champion=e.get('champion', -1)) for e in obs['entities'] if e.get('enemy')]
    others = sorted({(e['type'], e['variant']) for e in obs['entities'] if not e.get('enemy')})
    grid = sorted({(g[1], g[2]) for g in obs['grid'] if g[1] not in (15, 16)})
    return dict(variant=room.get('variant'), subtype=room.get('subtype'), name=room.get('name'), type=room['type'],
                shape=room['shape'], gw=room['gw'], gh=room['gh'], doors=[d['slot'] for d in obs['doors']],
                enemies=enemies, other_entities=others, grid_kinds=grid, clear=room['clear'])


def main():
    out = sys.argv[1]
    normal_max = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    boss_min = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    boss_max = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    proc = launch_abplus('catalog', 27191, 'exact')
    env = AbplusTrainingEnv(port=27191)
    catalog = {'normal': [], 'boss': [], 'engine': 'abplus-1.06', 'stage': None}
    t0 = time.time()
    try:
        env.connect()
        env.reset(phases=[['restart 0']], settle=2)
        catalog['stage'] = env.lua("local l = Game():GetLevel(); return tostring(l:GetStage()) .. ' ' .. "
                                   "tostring(l:GetStageType())")
        for kind, command, ids in (('normal', 'goto d.{}', range(0, normal_max + 1)),
                                   ('boss', 'goto s.boss.{}', range(boss_min, boss_max + 1))):
            for i in ids:
                obs, loaded = probe(env, command.format(i))
                if loaded and obs['room'].get('variant') == i:
                    catalog[kind].append(describe(obs))
                if i % 250 == 0:
                    print(json.dumps({kind: i, 'found': len(catalog[kind]), 'seconds': round(time.time() - t0)}),
                          flush=True)
                    # Leave the debug room occasionally: a fresh run keeps the level state small.
                    env.reset(phases=[['restart 0']], settle=2)
    finally:
        with open(out, 'w', encoding='utf8') as f:
            json.dump(catalog, f, indent=1)
        try:
            env.close()
        finally:
            stop_abplus(proc, 'catalog')
    print(json.dumps({'normal': len(catalog['normal']), 'boss': len(catalog['boss']), 'seconds': round(time.time() - t0)}))


if __name__ == '__main__':
    main()
