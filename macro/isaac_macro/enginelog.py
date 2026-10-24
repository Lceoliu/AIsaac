"""Parse the engine's own log (log.txt) into per-level generation records.

Afterbirth+ logs, for every Level::Init: the stage seed, curses, the LevelGenerator messages
("place_room: shape N" for non-1x1 rooms, "N rooms in L loops", "not enough dead ends ..."),
"Map Generated in K Loops" and the start room's "SpawnRNG seed". These are exact per-seed facts
that the offline generator must reproduce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CURSE_NAMES = {'Curse of Darkness!': 1, 'Curse of the Labyrinth!': 2, 'Curse of the Lost!': 4,
               'Curse of the Unknown': 8, 'Curse of Maze': 0x20, 'Curse of Blind': 0x40}


@dataclass
class LevelLog:
    stage: int
    stage_type: int
    seed: int
    curses: int = 0
    attempts: list = field(default_factory=list)   # per Generate(): dict(shapes, rooms, loops, new_dead_ends, verdict)
    map_loops: int | None = None
    start_spawn_seed: int | None = None
    start_room: str | None = None


def parse(path: str) -> list[LevelLog]:
    levels: list[LevelLog] = []
    cur: LevelLog | None = None
    attempt = None
    re_init = re.compile(r'Level::Init m_Stage (\d+), m_StageType (\d+) Seed (\d+)')
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        m = re_init.search(line)
        if m:
            cur = LevelLog(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            levels.append(cur)
            attempt = None
            continue
        if cur is None:
            continue
        msg = line.split(' - ', 1)[-1]
        if msg in CURSE_NAMES:
            cur.curses |= CURSE_NAMES[msg]
        elif msg == 'generate...':
            attempt = dict(shapes=[], rooms=None, loops=None, new_dead_ends=0, verdict=None)
            cur.attempts.append(attempt)
        elif attempt is not None and (m := re.fullmatch(r'place_room: shape (\d+)', msg)):
            attempt['shapes'].append(int(m.group(1)))
        elif attempt is not None and (m := re.fullmatch(r'(\d+) rooms in (\d+) loops', msg)):
            attempt['rooms'], attempt['loops'] = int(m.group(1)), int(m.group(2))
        elif attempt is not None and (m := re.fullmatch(r'\[LevelGenerator\] fail (\d+)/(\d+) in (\d+) loops', msg)):
            attempt['rooms'], attempt['loops'], attempt['fail'] = int(m.group(1)), int(m.group(3)), True
        elif attempt is not None and msg == 'not enough dead ends, trying to make new one':
            attempt['new_dead_ends'] += 1
        elif attempt is not None and (m := re.fullmatch(r'not enough dead ends \((\d+)/(\d+)\)', msg)):
            attempt['verdict'] = f'dead_ends {m.group(1)}/{m.group(2)}'
        elif attempt is not None and msg == 'generated level is not usable':
            attempt['verdict'] = 'unusable'
        elif attempt is not None and msg == 'placing rooms...':
            attempt['verdict'] = 'placing'
        elif (m := re.fullmatch(r'Map Generated in (\d+) Loops', msg)):
            cur.map_loops = int(m.group(1))
        elif (m := re.fullmatch(r'Room (\S+)\((.*)\)', msg)) and cur.start_room is None:
            cur.start_room = m.group(1) + ' ' + m.group(2)
        elif (m := re.fullmatch(r'SpawnRNG seed: (\d+)', msg)) and cur.start_spawn_seed is None:
            cur.start_spawn_seed = int(m.group(1))
    return levels
