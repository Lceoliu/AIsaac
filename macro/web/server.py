"""Local web front end for the floor generators: AB+ v1.06 (isaac_macro, engine-validated) and
Repentance+ v1.9.7.17 (isaac_macro.rep, a static port not yet checked against the game).

    python rl/macro/web/server.py [--port 8792] [--host 127.0.0.1]

Serves web/static and a small JSON API. It reads each game's afterbirthp.a (read only), named by
roomconfig.default_archive_path() and rep.roomconfig.default_archive_path(); nothing is uploaded
anywhere. Repentance+ entity names come from its own string table (official Simplified Chinese).

  GET /api/run?game=abplus|repplus&seed=DXNH%20NZLG&mode=debug|normal&last=8|11&route=sheol|cathedral
              &coins=0&keys=0&hearts=6&max_hearts=6&soul=0
  GET /api/layout?game=abplus|repplus&stage=<room file id>&type=<room type>&variant=<variant>
  GET /api/random
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import random
import re
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaac_macro.floor import HIDDEN_TYPES, Floor                             # noqa: E402
from isaac_macro.level import GameContext, Player                             # noqa: E402
from isaac_macro.levelgen import GRID, TRAVEL, index                          # noqa: E402
from isaac_macro.rng import Seeds, seed_to_string, string_to_seed            # noqa: E402
from isaac_macro.roomconfig import default_room_config                       # noqa: E402
from isaac_macro.run import iter_run                                          # noqa: E402
from isaac_macro.secret import LayoutEvidence, hidden_posterior              # noqa: E402

STATIC = Path(__file__).resolve().parent / 'static'

STAGE_NAMES = {  # (stage group, stage type) -> (Chinese, English); the official Repentance+ Chinese names
    (1, 0): ('地下室', 'Basement'), (1, 1): ('地窖', 'Cellar'), (1, 2): ('燃烧地下室', 'Burning Basement'),
    (1, 4): ('下水道', 'Downpour'), (1, 5): ('污水井', 'Dross'),
    (3, 0): ('洞穴', 'Caves'), (3, 1): ('墓穴', 'Catacombs'), (3, 2): ('淹水洞穴', 'Flooded Caves'),
    (3, 4): ('矿洞', 'Mines'), (3, 5): ('灰坑', 'Ashpit'),
    (5, 0): ('深牢', 'Depths'), (5, 1): ('坟场', 'Necropolis'), (5, 2): ('阴湿深牢', 'Dank Depths'),
    (5, 4): ('陵墓', 'Mausoleum'), (5, 5): ('炼狱', 'Gehenna'),
    (7, 0): ('子宫', 'Womb'), (7, 1): ('血宫', 'Utero'), (7, 2): ('结痂子宫', 'Scarred Womb'),
    (7, 4): ('尸宫', 'Corpse'), (9, 0): ('蓝色子宫', 'Blue Womb'),
    (10, 0): ('阴间', 'Sheol'), (10, 1): ('教堂', 'Cathedral'),
    (11, 0): ('暗室', 'Dark Room'), (11, 1): ('玩具箱', 'The Chest'), (12, 0): ('虚空', 'The Void'),
    (13, 0): ('家', 'Home'),
}
CURSES = [(1, '黑暗', 'Darkness'), (2, '迷宫', 'Labyrinth'), (4, '迷途', 'Lost'), (8, '未知', 'Unknown'),
          (0x10, '诅咒', 'Cursed'), (0x20, '混乱', 'Maze'), (0x40, '致盲', 'Blind'), (0x80, '大房间', 'Giant')]
ROOM_TYPES = {  # type -> (Chinese name, map label)
    1: ('普通', ''), 2: ('商店', '店'), 3: ('错误房', '错'), 4: ('宝箱房', '宝'), 5: ('头目房', '头'),
    6: ('小头目房', '小'), 7: ('隐藏房', '隐'), 8: ('超级隐藏房', '超'), 9: ('赌博房', '赌'),
    10: ('诅咒房', '咒'), 11: ('挑战房', '挑'), 12: ('图书馆', '书'), 13: ('献祭房', '祭'),
    14: ('恶魔房', '魔'), 15: ('天使房', '天'), 16: ('夹层', '夹'), 17: ('头目车轮战', '车'),
    18: ('干净的卧室', '卧'), 19: ('肮脏的卧室', '卧'), 20: ('宝库', '库'), 21: ('骰子房', '骰'),
    22: ('黑市', '黑'), 23: ('贪婪出口', '出'), 24: ('星象房', '星'), 25: ('传送房', '传'),
    26: ('传送出口', '传'), 27: ('隐藏出口', '出'), 28: ('蓝色房间', '蓝'), 29: ('究极隐藏房', '极'),
}
SHAPES = ['', '1×1', '横向贮藏室', '纵向贮藏室', '1×2', '竖向长走廊', '2×1', '横向长走廊', '2×2',
          'L 型（缺左上）', 'L 型（缺右上）', 'L 型（缺左下）', 'L 型（缺右下）']
SLOTS = ['左', '上', '右', '下', '左2', '上2', '右2', '下2']
GRID_NAMES = [  # (first type, last type, Chinese, category)
    (1000, 1000, '石头', 'rock'), (1001, 1001, '炸弹石', 'rock'), (1002, 1002, '罐子/蘑菇等', 'rock'),
    (1003, 1003, '标记石头', 'rock'), (1004, 1004, '超级标记石头', 'rock'), (1005, 1099, '石头类', 'rock'),
    (1300, 1300, '炸药桶', 'tnt'), (1490, 1501, '大便', 'poop'), (1900, 1900, '钢铁方块', 'block'),
    (1930, 1931, '地刺', 'spikes'), (1940, 1940, '蛛网', 'web'), (3000, 3000, '沟壑', 'pit'),
    (4000, 4000, '钥匙方块', 'block'), (4500, 4500, '按钮', 'plate'), (9000, 9000, '活板门', 'door'),
    (9100, 9100, '暗门', 'door'), (10000, 10000, '装饰物', 'deco'),
]
STRING_CATEGORY_ORDER = ('Entities', 'Minibosses', 'Stages', 'Curses')   # for keys defined twice


def load_strings(archives) -> dict[str, tuple[str, str]]:
    """resources/stringtable.sta (Repentance+ only): key -> (English, Simplified Chinese)."""
    try:
        xml = archives.read('resources/stringtable.sta').decode('utf-8', 'replace')
    except FileNotFoundError:
        return {}
    langs = dict((name, int(i)) for i, name in re.findall(r'<language [^>]*index="(\d+)" name="([^"]*)"', xml))
    en, zh = langs.get('English', 1) - 1, langs.get('Chinese (Simple)', 4) - 1   # index 0 is the key itself
    cats = {m.group(1): m.group(2) for m in re.finditer(r'<category name="([^"]*)">(.*?)</category>', xml, re.S)}
    order = [c for c in STRING_CATEGORY_ORDER if c in cats] + [c for c in cats if c not in STRING_CATEGORY_ORDER]
    out: dict[str, tuple[str, str]] = {}
    for cat in order:
        for m in re.finditer(r'<key name="([^"]*)">(.*?)</key>', cats[cat], re.S):
            strings = [htmllib.unescape(s or '') for s in re.findall(r'<string(?:\s*/>|>(.*?)</string>)', m.group(2), re.S)]
            if len(strings) > max(en, zh):
                out.setdefault(m.group(1), (strings[en], strings[zh]))
    return out


def load_entities(archives, strings: dict) -> dict[tuple, str]:
    xml = archives.read('resources/entities2.xml').decode('utf-8-sig', 'replace')
    out: dict[tuple, str] = {}
    for m in re.finditer(r'<entity\s([^>]*?)/?>', xml):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        try:
            key = (int(a['id']), int(a.get('variant', 0) or 0), int(a.get('subtype', 0) or 0))
        except (KeyError, ValueError):
            continue
        name = a.get('name', '')
        if name.startswith('#'):      # a string table key (Repentance+)
            en, zh = strings.get(name[1:], ('', ''))
            name = f'{zh}（{en}）' if zh and en and zh != en else (zh or en or name[1:].replace('_', ' ').title())
        out.setdefault(key, name)
    return out


class Game:
    """One game's generator and data, loaded on first use."""

    def __init__(self, key: str, title: str, version: str, validated: bool, hidden_types: tuple):
        self.key, self.title, self.version, self.validated = key, title, version, validated
        self.hidden_types = hidden_types
        self.lock = threading.Lock()   # RoomConfig keeps the per-floor room weights: one generation at a time
        self._load_lock = threading.Lock()
        self._rc = None
        self.entities: dict = {}
        self.evidence: dict = {}

    @property
    def rep(self) -> bool:
        return self.key == 'repplus'

    def room_config(self):
        if self._rc is None:
            with self._load_lock:
                if self._rc is None:
                    if self.rep:
                        from isaac_macro.rep.roomconfig import default_room_config as rep_room_config
                        rc = rep_room_config()
                        strings = load_strings(rc.archives)
                    else:
                        rc, strings = default_room_config(), {}
                    self.entities = load_entities(rc.archives, strings)
                    self._rc = rc
        return self._rc

    def layout_evidence(self, stage: int, stage_type: int, curses: int):
        if stage > 11 or stage == 9:
            return None
        key = (stage, stage_type, self.rep and bool(curses & 2))
        if key not in self.evidence:
            if self.rep:
                from isaac_macro.rep.secret import RepLayoutEvidence
                self.evidence[key] = RepLayoutEvidence(self.room_config(), stage, stage_type, labyrinth=key[2])
            else:
                self.evidence[key] = LayoutEvidence(self.room_config(), stage, stage_type)
        return self.evidence[key]

    def levels(self, seed: int, player: Player, last: int, debug: bool, cathedral: bool) -> list:
        rc = self.room_config()
        if self.rep:
            from isaac_macro.rep.level import RepGameContext
            from isaac_macro.rep.rng import Seeds as RepSeeds
            from isaac_macro.rep.run import iter_run as rep_iter_run
            return list(rep_iter_run(rc, RepSeeds(seed), RepGameContext(player=player), last_stage=last,
                                     debug_start=debug, cathedral=cathedral))
        return list(iter_run(rc, Seeds(seed), GameContext(player=player), last_stage=last, debug_start=debug,
                             cathedral=cathedral))

    def posteriors(self, floor: Floor, lv) -> tuple[dict, dict, dict]:
        vis = floor.visible(self.hidden_types)
        ev = self.layout_evidence(lv.stage, lv.stage_type, lv.curses)
        if self.rep:
            from isaac_macro.rep.secret import hidden_posterior as rep_hidden_posterior, strange_door_floor
            return rep_hidden_posterior(vis, ev, strange_door=strange_door_floor(lv.stage, lv.stage_type, lv.curses))
        ps, pss = hidden_posterior(vis, ev)
        return ps, pss, {}

    def entity_name(self, t: int, v: int, s: int) -> tuple[str, str]:
        if t == 0:
            return '（空）', 'none'
        if t >= 1000:
            for lo, hi, name, cat in GRID_NAMES:
                if lo <= t <= hi:
                    return name, cat
            return f'网格物体 {t}', 'grid'
        ents = self.entities
        name = ents.get((t, v, s)) or ents.get((t, v, 0)) or ents.get((t, 0, 0)) or f'实体 {t}.{v}.{s}'
        if t == 5:
            return name, 'pickup'
        if t in (6, 33, 292):
            return name, 'object'
        return name, 'enemy'


def _rep_hidden_types() -> tuple:
    from isaac_macro.rep.secret import HIDDEN_TYPES as REP_HIDDEN_TYPES
    return REP_HIDDEN_TYPES


GAMES = {
    'abplus': Game('abplus', '胎衣†', 'v1.06', True, HIDDEN_TYPES),
    'repplus': Game('repplus', '忏悔+', 'v1.9.7.17', False, _rep_hidden_types()),
}


def get_game(params: dict) -> Game:
    game = GAMES.get(params.get('game', 'abplus'))
    if game is None:
        raise ValueError('game 只能是 abplus 或 repplus')
    return game


def stage_label(stage: int, stage_type: int, curses: int) -> tuple[str, str]:
    group = stage if stage >= 9 else stage - (stage + 1) % 2
    zh, en = STAGE_NAMES.get((group, stage_type), (f'第 {stage} 层', f'Stage {stage}'))
    if stage <= 8:
        if curses & 2:
            return zh + ' XL', en + ' XL'
        roman = 'I' if stage % 2 else 'II'
        return f'{zh} {roman}', f'{en} {roman}'
    return zh, en


def parse_seed(text: str) -> int:
    text = (text or '').strip().upper()
    if not text:
        raise ValueError('请输入种子')
    if re.fullmatch(r'\d+', text):
        v = int(text)
        if not 0 < v < 2 ** 32:
            raise ValueError('数字种子要在 1 到 4294967295 之间')
        return v
    compact = re.sub(r'\s+', '', text)
    if len(compact) != 8:
        raise ValueError('种子格式应为 XXXX XXXX（8 个字符）或一个数字')
    v = string_to_seed(compact[:4] + ' ' + compact[4:])
    if not v:
        raise ValueError('种子无效：含有不在字母表里的字符，或校验位不对')
    return v


def room_json(game: Game, floor: Floor, desc, fr) -> dict:
    cfg = desc.config
    zh, label = ROOM_TYPES.get(cfg.type, (f'类型 {cfg.type}', '?'))
    start = desc.grid_index == floor.start and cfg.type == 1 and cfg.stage in (0, 16, 17)
    return dict(index=desc.list_index, x=fr.x, y=fr.y, shape=desc.shape, shape_name=SHAPES[desc.shape],
                cells=list(fr.cells), type=cfg.type, type_name='起始房间' if start else zh,
                label='起' if start else label, start=start, variant=cfg.variant, subtype=cfg.subtype,
                name=cfg.name, difficulty=cfg.difficulty, weight=float(cfg.initial_weight), file=cfg.stage,
                doors=fr.doors, door_names=[SLOTS[s] for s in range(8) if fr.doors >> s & 1],
                layout_doors=cfg.doors, depth=fr.depth, hidden=cfg.type in game.hidden_types,
                seeds=dict(decoration=desc.decoration_seed, spawn=desc.spawn_seed, award=desc.award_seed))


def door_segments(floor: Floor) -> list:
    """One entry per connection: the two cells on either side of the door (drawn as a bridge)."""
    out, seen = [], set()
    for r in floor.rooms:
        for slot in range(8):
            if not r.doors >> slot & 1:
                continue
            t = r.slot_target(slot)
            d = slot & 3
            src = index(t % GRID - TRAVEL[d][0], t // GRID - TRAVEL[d][1])
            key = tuple(sorted((src, t)))
            if key in seen:
                continue
            seen.add(key)
            other = floor.room_at(t)
            out.append(dict(a=src, b=t, rooms=[r.index, other.index if other else -1]))
    return out


def _probs(post: dict) -> dict:
    return {str(c): round(p, 4) for c, p in post.items() if p >= 0.0005}


def run_json(params: dict) -> dict:
    t0 = time.time()
    game = get_game(params)
    seed = parse_seed(params.get('seed', ''))
    mode = params.get('mode', 'debug')
    last = 11 if params.get('last', '11') == '11' else 8
    cathedral = params.get('route', 'sheol') == 'cathedral'

    def num(key, default, lo, hi):
        try:
            v = int(params.get(key, default))
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    max_hearts = num('max_hearts', 6, 0, 24)
    player = Player(hearts=num('hearts', 6, 0, max_hearts), max_hearts=max_hearts, soul_hearts=num('soul', 0, 0, 24),
                    keys=num('keys', 0, 0, 99), coins=num('coins', 0, 0, 99))
    with game.lock:
        levels = game.levels(seed, player, last, mode == 'debug', cathedral)
        floors = []
        for lv in levels:
            floor = Floor.from_level(lv)
            by_index = {r.index: r for r in floor.rooms}
            rooms = [room_json(game, floor, d, by_index[d.list_index]) for d in lv.rooms if d.list_index in by_index]
            ps, pss, pus = game.posteriors(floor, lv)
            zh, en = stage_label(lv.stage, lv.stage_type, lv.curses)
            floors.append(dict(
                stage=lv.stage, stage_type=lv.stage_type, name=zh, name_en=en, stage_seed=lv.stage_seed,
                curses=[zh_c for bit, zh_c, _ in CURSES if lv.curses & bit], curse_bits=lv.curses,
                attempts=lv.attempts, start=floor.start, rooms=rooms, doors=door_segments(floor),
                bosses=[dict(name=r['name'], variant=r['variant'], subtype=r['subtype']) for r in rooms if r['type'] == 5],
                secret_posterior=_probs(ps), super_secret_posterior=_probs(pss), ultra_secret_posterior=_probs(pus)))
    return dict(game=game.key, game_name=game.title, version=game.version, validated=game.validated,
                seed=dict(value=seed, text=seed_to_string(seed)), mode=mode, last=last,
                route='cathedral' if cathedral else 'sheol',
                player=dict(hearts=player.hearts, max_hearts=player.max_hearts, soul=player.soul_hearts,
                            keys=player.keys, coins=player.coins),
                floors=floors, ms=round((time.time() - t0) * 1000))


def layout_json(params: dict) -> dict:
    game = get_game(params)
    rc = game.room_config()
    sid, rtype, variant = int(params['stage']), int(params['type']), int(params['variant'])
    room = rc.get_room(sid, rtype, variant)
    if room is None:
        raise ValueError('没有这个布局')
    spawns = []
    for s in room.spawns:
        entries = []
        for e in s.entries:
            name, cat = game.entity_name(e.type, e.variant, e.subtype)
            entries.append(dict(type=e.type, variant=e.variant, subtype=e.subtype, weight=round(e.weight, 3),
                                name=name, kind=cat))
        spawns.append(dict(x=s.x, y=s.y, entries=entries))
    return dict(game=game.key, stage=sid, type=rtype, variant=variant, name=room.name, shape=room.shape,
                width=room.width, height=room.height, difficulty=room.difficulty,
                weight=float(room.initial_weight), doors=room.doors, door_list=[list(d) for d in room.door_list],
                file=Path(rc.paths[sid]).stem if sid in rc.paths else str(sid), spawns=spawns,
                missing=[list(c) for c in _missing_quadrant(room.shape)])


def _missing_quadrant(shape: int) -> list:
    """Tile rectangles (x, y, w, h) that are not part of an L-shaped room (13x7 quadrants)."""
    return {9: [(0, 0, 13, 7)], 10: [(13, 0, 13, 7)], 11: [(0, 7, 13, 7)], 12: [(13, 7, 13, 7)]}.get(shape, [])


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith('/api/'):
            return super().do_GET()
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == '/api/run':
                return self._json(200, run_json(params))
            if url.path == '/api/layout':
                return self._json(200, layout_json(params))
            if url.path == '/api/random':
                v = random.getrandbits(32) or 1
                return self._json(200, dict(value=v, text=seed_to_string(v)))
            return self._json(404, dict(error='未知接口'))
        except FileNotFoundError as exc:
            return self._json(500, dict(error=f'找不到游戏资源文件：{exc}'))
        except (ValueError, KeyError) as exc:
            return self._json(400, dict(error=str(exc)))
        except NotImplementedError as exc:
            return self._json(400, dict(error=f'暂不支持：{exc}'))
        except Exception as exc:  # pragma: no cover - surfaced to the page for debugging
            traceback.print_exc()
            return self._json(500, dict(error=f'{type(exc).__name__}: {exc}'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Isaac floor generator web UI (AB+ and Repentance+)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8792)
    args = ap.parse_args()
    for game in GAMES.values():
        try:
            game.room_config()
        except FileNotFoundError as exc:
            print(f'{game.title}: game files not found ({exc}); its requests will fail', flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Isaac floor generator UI: http://{args.host}:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
