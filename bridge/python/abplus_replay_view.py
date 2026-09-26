"""One self-contained HTML player for abplus_eval.py raw-observation replays.

  python abplus_replay_view.py OUT.html RUN_DIR=LABEL [RUN_DIR=LABEL ...] [--notes notes.json] [--title T]

RUN_DIR is an abplus_eval.py --out directory (replays/seed-*.jsonl.gz, results.jsonl, meta.json); the
first run groups the episode list (task and outcome). The page (isaac_bridge/abplus_replay.html)
draws the recorded engine state step by step (one step = 2 logic frames = 1/15 s): the room grid,
the player with its move and shot input, NPCs by name, tears, projectiles, bombs and creep, with the
blocking HP (the NPCs that keep the doors shut) and the player's hearts over time. Nothing is
re-simulated. Episodes of different runs with the same seed can be switched side by side.

notes.json maps "LABEL|SEED" or "SEED" to a short note shown with the episode.
"""
import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

NAMES = {10: 'Gaper', 11: 'Gusher/Pacer', 12: 'Horf', 13: 'Fly', 14: 'Pooter', 15: 'Clotty', 16: 'Mulligan',
         18: 'Attack Fly', 19: 'Larry Jr.', 20: 'Monstro', 21: 'Maggot', 22: 'Hive', 23: 'Charger', 24: 'Globin',
         25: 'Boom Fly', 26: 'Maw', 27: 'Host', 28: 'Chub', 29: 'Hopper', 30: 'Boil', 31: 'Spitty', 32: 'Brain',
         33: 'Fireplace', 34: 'Leaper', 35: 'Mr. Maw', 36: 'Gurdy', 38: 'Baby', 39: 'Vis', 40: 'Guts',
         41: 'Knight', 42: 'Stone Grimace', 44: 'Poky', 53: 'Dople', 55: 'Leech', 56: 'Lump', 58: 'Para-Bite',
         61: 'Sucker', 62: 'Pin', 63: 'Famine', 67: 'Duke of Flies', 68: 'Peep', 77: 'Embryo', 79: 'Gemini',
         80: 'Moter', 82: 'Headless Horseman', 83: 'Headless Horseman Head',
         85: 'Spider', 86: 'Keeper', 88: 'Walking Boil', 89: 'Buttlicker', 90: 'Hanger', 91: 'Swarmer',
         94: 'Big Spider', 96: 'Eternal Fly', 99: 'Gurdy Jr.', 100: 'Widow', 204: 'Mobile Host', 205: 'Nest',
         206: 'Baby Long Legs', 207: 'Crazy Long Legs', 208: 'Fatty', 209: 'Fat Sack', 210: 'Blubber', 211: 'Half Sack',
         212: "Death's Head", 214: 'Fly L2', 215: 'Spider L2', 216: 'Swinger', 217: 'Dip', 218: 'Wall Hugger',
         220: 'Squirt', 221: 'Cod Worm', 222: 'Ring of Flies', 223: 'Dinga', 226: 'Skinny', 227: 'Bony',
         229: 'Tumor', 231: 'Nerve Ending', 232: 'Skinball', 234: 'One Tooth', 237: 'Gurgling', 238: 'Splasher',
         239: 'Grub', 240: 'Wall Creep', 241: 'Rage Creep', 242: 'Blind Creep', 243: 'Conjoined Spitty',
         244: 'Round Worm', 246: 'Ragling', 247: 'Flesh Mobile Host', 248: 'Psychic Horf', 249: 'Full Fly',
         250: 'Ticking Spider', 252: 'Nulls', 255: 'Night Crawler', 256: 'Dart Fly', 257: 'Conjoined Fatty',
         258: 'Fat Bat', 259: 'Imp', 261: 'Dingle', 276: 'Roundy', 281: 'Swarm', 284: 'Cyclopia', 289: 'Ulcer',
         292: 'Movable TNT', 300: 'Mushroom', 302: 'Stoney', 303: 'Blister', 305: 'Ministro', 306: 'Portal',
         307: 'Tar Boy', 309: 'Gush',
         310: 'Leper', 401: 'Stain', 402: 'Brownie', 404: 'Little Horn', 405: 'Rag Man'}
# Variants with a name of their own in AB+ entities2.xml (the mixture's rooms spawn these).
VARIANT_NAMES = {(10, 0): 'Frowning Gaper', (10, 1): 'Gaper', (11, 0): 'Gusher', (11, 1): 'Pacer',
                 (14, 1): 'Super Pooter', (15, 1): 'Clot', (15, 2): 'I.Blob', (16, 1): 'Mulligoon',
                 (16, 2): 'Mulliboom', (29, 1): 'Trite', (33, 1): 'Red Fire Place', (79, 1): 'Steven',
                 (79, 10): 'Gemini Baby', (79, 11): 'Steven Baby', (79, 20): 'Umbilical Cord',
                 (206, 1): 'Small Baby Long Legs', (207, 1): 'Small Crazy Long Legs', (217, 1): 'Corn',
                 (217, 2): 'Brownie Corn', (237, 1): 'Gurgling (boss)', (237, 2): 'Turdling',
                 (246, 1): "Rag Man's Ragling", (261, 1): 'Dangle'}


def npc_name(etype, variant):
    return VARIANT_NAMES.get((etype, variant)) or NAMES.get(etype, f'type {etype}')

# Grid cell codes (GridEntityType; lower case = destroyed or open).
ROCKS = (2, 5, 6, 22)


def r1(v):
    return round(float(v), 1)


def grid_string(obs, gw, gh):
    types = {g[0]: g[1] for g in (obs.get('grid') or [])}
    chars = ['.'] * (gw * gh)
    for index, x, y, collision, inside, walkable, pit, hazard, solid, destructible in obs['terrain']['cells']:
        gt = types.get(index)
        if gt == 16:
            ch = 'D'
        elif not inside or gt == 15:
            ch = 'W'
        elif pit:
            ch = 'O'
        elif gt == 14:
            ch = 'P' if solid else 'p'
        elif gt == 3:
            ch = 'B'
        elif gt == 4:
            ch = 'T' if solid else 'r'
        elif gt == 12:
            ch = 'X' if solid else 'r'
        elif gt == 11:
            ch = 'L' if solid else '.'
        elif gt == 21:
            ch = 'M'
        elif gt in ROCKS:
            ch = 'R' if solid else 'r'
        elif gt in (8, 9):
            ch = 'S'
        elif gt == 10:
            ch = 'w'
        elif solid:
            ch = 'R'
        else:
            ch = '.'
        chars[index] = ch
    return ''.join(chars)


def entity(e):
    x, y, size = r1(e['pos'][0]), r1(e['pos'][1]), r1(e['size'])
    if 'enemy' in e:
        flags = int(e['enemy']) | int(e['vulnerable']) << 1 | int(e['boss']) << 2
        record = [1, x, y, size, e['type'], e['variant'], flags]
        if e['boss']:
            record += [round(e.get('boss_hp', 0.0), 3), e.get('anim', '')]
        return record
    if e.get('projectile'):
        return [3, x, y, size, r1(e.get('height', 0.0))]
    if e['type'] == 2:
        return [2, x, y, size, r1(e.get('height', 0.0))]
    if e.get('bomb'):
        return [4, x, y, size]
    if e.get('pickup'):
        return [5, x, y, size, e['variant']]
    if 'laser' in e:
        return [6, x, y, r1(e['laser']['end'][0]), r1(e['laser']['end'][1]), r1(e['laser']['width'])]
    if e['type'] == 1000 and e.get('cdmg', 0) > 0:
        return [7, x, y, size]
    return None


def park_start(positions, radius=48.0):
    """First step from which the player stays within radius of where it ends."""
    fx, fy = positions[-1]
    k = len(positions) - 1
    while k > 0 and math.dist(positions[k - 1], (fx, fy)) <= radius:
        k -= 1
    return k


def episode(path, result, meta, label, notes):
    rows = [json.loads(line) for line in gzip.open(path, 'rt', encoding='utf8')]
    seed = rows[0]['metadata']['seed']
    first = rows[1]['obs']
    room = first['room']
    gw, gh = room['gw'], room['gh']
    cell0 = first['terrain']['cells'][0]
    frames, grids, last_grid = [], [], None
    positions, cells, dists = [], set(), []
    moves = shots = 0
    last_hit, previous_hits = 0, None
    for k, row in enumerate(rows[1:]):
        obs, action = row['obs'], row['action']
        move, shot = divmod(int(action[0]), 5) if action else (0, 0)
        bomb = int(action[1]) if action else 0
        grid = grid_string(obs, gw, gh)
        if grid != last_grid:
            grids.append([k, grid])
            last_grid = grid
        p, c = obs['players'][0], obs['combat']
        hits = int(c['enemy_damage_events'])
        ents = [x for x in (entity(e) for e in obs['entities']) if x]
        frames.append([r1(p['pos'][0]), r1(p['pos'][1]), p['hearts'], p['soul'], move, shot, bomb,
                       round(c['blocking_hp'], 2), hits, int(c['player_damage_events']), ents])
        positions.append(tuple(p['pos']))
        cells.add((round((p['pos'][0] - cell0[1]) / 40), round((p['pos'][1] - cell0[2]) / 40)))
        if k:
            moves += move != 0
            shots += shot != 0
            enemies = [e for e in obs['entities'] if e.get('enemy')]
            if enemies:
                dists.append(min(math.dist(p['pos'], e['pos']) for e in enemies))
            if previous_hits is not None and hits > previous_hits:
                last_hit = k
        previous_hits = hits
    n = max(1, len(frames) - 1)
    dists.sort()
    parked = park_start(positions)
    start_npcs = Counter(npc_name(e['type'], e['variant']) for e in first['entities'] if e.get('enemy'))
    note = notes.get(f'{label}|{seed}') or notes.get(str(seed))
    # The checkpoint's own reward (abplus_eval reward_v3/v4) when it has one; combat-v2 otherwise.
    own = next((k for k in ('reward_v5', 'reward_v4', 'reward_v3') if k in result), 'reward_v2')
    info = dict(
        id=f'{label}|{seed}', label=label, seed=seed, task=result.get('task'), outcome=result.get('outcome'),
        layout=result.get('layout'), room=room.get('name'), shape=[gw, gh], cell0=[cell0[1], cell0[2]],
        max_hearts=first['players'][0]['max_hearts'], steps=len(frames) - 1,
        seconds=round(result.get('frames', 2 * n) / 30, 1), reward=result.get(own),
        reward_name='combat-' + own.split('_')[1], components=result.get(own + '_components'),
        updates=meta.get('updates'), start_bombs=result.get('start_bombs'),
        deterministic=meta.get('deterministic'), sample_seed=meta.get('sample_seed'),
        enemies=[f'{v}× {k}' if v > 1 else k for k, v in start_npcs.most_common()],
        doors=[[d['pos'][0], d['pos'][1], int(d['locked'])] for d in first['doors']],
        stats=dict(cells=len(cells), move=round(moves / n, 2), shoot=round(shots / n, 2),
                   dist_median=round(dists[len(dists) // 2]) if dists else None,
                   last_hit_s=round(last_hit * 2 / 30, 1), parked_from_s=round(parked * 2 / 30, 1),
                   hits=frames[-1][8], hurt=frames[-1][9],
                   blocking=[frames[0][7], frames[-1][7]]),
        note=note)
    return info, dict(grids=grids, frames=frames)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path)
    p.add_argument('runs', nargs='+', help='abplus_eval.py --out directory, optionally =LABEL')
    p.add_argument('--notes', type=Path)
    p.add_argument('--title', default='AB+ 回放')
    p.add_argument('--subtitle', default='')
    args = p.parse_args()
    notes = json.loads(args.notes.read_text(encoding='utf8')) if args.notes else {}
    infos, blocks = [], []
    for spec in args.runs:
        run, _, label = spec.partition('=')
        run = Path(run)
        label = label or run.name
        meta = json.loads((run / 'meta.json').read_text(encoding='utf8'))
        results = {}
        for line in (run / 'results.jsonl').read_text(encoding='utf8').splitlines():
            r = json.loads(line)
            results[r['seed']] = r
        for path in sorted((run / 'replays').glob('seed-*.jsonl.gz')):
            seed = int(path.name.split('-')[1])
            info, data = episode(path, results.get(seed, {}), meta, label, notes)
            infos.append(info)
            blocks.append(f'<script type="application/json" id="ep-{len(infos) - 1}">'
                          + json.dumps(data, separators=(',', ':')).replace('</', '<\\/') + '</script>')
            print(f"{label} {seed} {info['task']} {info['outcome']} steps {info['steps']}", flush=True)
    template = (Path(__file__).parent / 'isaac_bridge' / 'abplus_replay.html').read_text(encoding='utf8')
    dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    variants = {f'{t}.{v}': name for (t, v), name in VARIANT_NAMES.items()}
    page = (template.replace('/*EPISODES*/', dump(infos)).replace('/*NAMES*/', dump(NAMES))
            .replace('/*VARIANTS*/', dump(variants))
            .replace('/*TITLE*/', dump([args.title, args.subtitle])).replace('<!--DATA-->', '\n'.join(blocks)))
    args.out.write_text(page, encoding='utf8')
    print(f'{args.out} {len(infos)} episodes {args.out.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
