"""查看器"真实美术"模式的素材支持：读解包出来的 Repentance 资源（只读），给浏览器端渲染器（static/viewer_art.js）提供
查表数据（实体 -> anm2、角色皮肤、道具 -> 服装、道具/饰品图标、backdrop 定义），并决定自包含 HTML 要内嵌哪些文件。

素材目录默认 <仓库外层>/experiments/run-recorder/assets/rep-extracted/resources（官方 ResourceExtractor 解包、按
base < afterbirth < afterbirth+ < repentance 叠加，见 experiments/run-recorder/phase6-art/REPORT.md），--assets DIR 指定。
路径一律用"小写、正斜杠、相对 resources 根"的形式（解包出来的文件名本来就是小写；anm2 里写的路径大小写不一，统一小写）。
"""
from __future__ import annotations

import base64
import json
import posixpath
import re
from pathlib import Path

_ATTR = re.compile(r'([A-Za-z_][\w\-]*)\s*=\s*"([^"]*)"')
_SHEET = re.compile(r'<Spritesheet\b[^>]*?\bPath="([^"]*)"', re.I)

# HUD 与通用素材（总是需要）
HUD_ASSETS = ['gfx/backdrop/controls.anm2', 'gfx/ui/ui_hearts.anm2', 'gfx/ui/hudpickups.anm2', 'gfx/ui/minimap1.anm2', 'gfx/ui/minimap_icons.anm2',
              'gfx/ui/ui_chargebar.anm2', 'gfx/ui/ui_cardfronts.anm2', 'gfx/ui/ui_cardspills.anm2',
              'gfx/001.000_player.anm2']
HUD_PNGS = ['gfx/ui/ui_bosshealthbar.png', 'gfx/shadow.png', 'gfx/grid/grid_pit_water.png', 'gfx/grid/door_00_shopdoor.png',
            'gfx/grid/door_00_sacrificeroomdoor.png', 'gfx/grid/door_00_diceroomdoor.png', 'gfx/grid/door_02b_chestroomdoor.png']
GRID_ANM2 = ['gfx/grid/grid_rock.anm2', 'gfx/grid/grid_pit.anm2', 'gfx/grid/grid_spikes.anm2', 'gfx/grid/grid_poop.anm2',
             'gfx/grid/grid_tnt.anm2', 'gfx/grid/grid_web.anm2', 'gfx/grid/grid_pressureplate.anm2', 'gfx/grid/grid_locks.anm2',
             'gfx/grid/door_11_trapdoor.anm2', 'gfx/grid/door_01_normaldoor.anm2', 'gfx/grid/door_08_holeinwall.anm2',
             'gfx/grid/grid_teleporter.anm2', 'gfx/grid/grid_redtnt.anm2',
             # v1 文件按目标房间类型估计门（viewer_art.js doorSprite 的表）
             'gfx/grid/door_02_treasureroomdoor.anm2', 'gfx/grid/door_10_bossroomdoor.anm2', 'gfx/grid/door_05_arcaderoomdoor.anm2',
             'gfx/grid/door_04_selfsacrificeroomdoor.anm2', 'gfx/grid/door_03_ambushroomdoor.anm2',
             'gfx/grid/door_07_devilroomdoor.anm2', 'gfx/grid/door_07_holyroomdoor.anm2']


def default_assets_dir():
    return Path(__file__).resolve().parents[4] / 'experiments' / 'run-recorder' / 'assets' / 'rep-extracted' / 'resources'


def norm(p):
    """anm2/xml 里的路径 -> 素材键（小写、正斜杠、去掉 ./ 与 ..）。"""
    p = (p or '').replace('\\', '/').strip().lower()
    p = posixpath.normpath(p) if p else p
    return p.lstrip('/')


def _tags(text, tag):
    out = []
    for m in re.finditer(r'<%s\b([^>]*?)/?>' % tag, text, re.S):
        out.append(dict(_ATTR.findall(m.group(1))))
    return out


class ArtIndex:
    def __init__(self, root=None):
        self.root = Path(root) if root else default_assets_dir()
        self.ok = (self.root / 'entities2.xml').is_file()
        self._tables = None
        self._sheets = {}

    # ---- 文件
    def path(self, key):
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            return None   # 不出根目录
        return p if p.is_file() else None

    def read_bytes(self, key):
        p = self.path(norm(key))
        return p.read_bytes() if p else None

    def _xml(self, name):
        p = self.root / name
        return p.read_text(encoding='utf-8', errors='replace') if p.is_file() else ''

    def sheets_of(self, anm2_key):
        """anm2 引用的 spritesheet 素材键。"""
        if anm2_key in self._sheets:
            return self._sheets[anm2_key]
        b = self.read_bytes(anm2_key)
        out = []
        if b:
            base = posixpath.dirname(anm2_key)
            for m in _SHEET.finditer(b.decode('utf-8', 'replace')):
                out.append(norm(posixpath.join(base, m.group(1))))
        self._sheets[anm2_key] = out
        return out

    # ---- 查表（送给浏览器）
    def tables(self):
        if self._tables is not None:
            return self._tables
        t = dict(entities={}, players={}, costumes_item={}, costumes_id={}, items_gfx={}, trinkets_gfx={}, backdrops={},
                 cards={}, root=str(self.root))
        if not self.ok:
            self._tables = t
            return t
        for a in _tags(self._xml('entities2.xml'), 'entity'):
            i, v, s, anm = a.get('id'), a.get('variant', '0'), a.get('subtype'), a.get('anm2path')
            if i is None or not anm:
                continue
            rec = [norm('gfx/' + anm), float(a.get('shadowSize', '0') or 0)]
            if s is not None:
                t['entities'].setdefault(f'{i}.{v}.{s}', rec)
            t['entities'].setdefault(f'{i}.{v}', rec)
            t['entities'].setdefault(i, rec)
        ptext = self._xml('players.xml')
        proot = (re.search(r'<players\b[^>]*\broot="([^"]*)"', ptext) or [None, 'gfx/characters/costumes/'])[1]
        for a in _tags(ptext, 'player'):
            if a.get('id') is None:
                continue
            t['players'][a['id']] = dict(skin=norm(proot + a['skin']) if a.get('skin') else None,
                                         costume=a.get('costume'), skin_color=a.get('skinColor'),
                                         name=a.get('name'), suffix=a.get('costumeSuffix'))
        ctext = self._xml('costumes2.xml')
        croot = (re.search(r'<costumes\b[^>]*\banm2root="([^"]*)"', ctext) or [None, 'gfx/characters/'])[1]
        for a in _tags(ctext, 'costume'):
            cid, anm, typ = a.get('id'), a.get('anm2path'), a.get('type', 'none')
            if cid is None or not anm:
                continue
            rec = [norm(croot + anm), int(a.get('priority', '0') or 0), typ]
            if typ in ('passive', 'active', 'familiar', 'trinket'):
                key = ('t' if typ == 'trinket' else '') + cid
                t['costumes_item'].setdefault(key, []).append(rec)
            else:
                t['costumes_id'].setdefault(cid, rec)
        itext = self._xml('items.xml')
        for tag in ('passive', 'active', 'familiar'):
            for a in _tags(itext, tag):
                if a.get('id') and a.get('gfx'):
                    t['items_gfx'][a['id']] = norm('gfx/items/collectibles/' + a['gfx'])
        for a in _tags(itext, 'trinket'):
            if a.get('id') and a.get('gfx'):
                t['trinkets_gfx'][a['id']] = norm('gfx/items/trinkets/' + a['gfx'])
        btext = self._xml('backdrops.xml')
        groot = (re.search(r'gridgfxroot="([^"]*)"', btext) or [None, 'gfx/grid/'])[1]
        broot = (re.search(r'<backdrops\b[^>]*\bgfxroot="([^"]*)"', btext) or [None, 'gfx/backdrop/'])[1]
        for a in _tags(btext, 'backdrop'):
            if a.get('id') is None:
                continue
            d = {k: v for k, v in a.items()}
            for k in ('gfx', 'lfloorgfx', 'nfloorgfx'):
                if d.get(k):
                    d[k] = norm(broot + d[k])
            for k in ('rocks', 'pit', 'waterpit', 'bridge', 'door', 'holeinwall', 'props'):
                if d.get(k):
                    d[k] = norm(groot + d[k])
            t['backdrops'][a['id']] = d
        self._tables = t
        return t

    # ---- 本局用到的素材（自包含 HTML 只内嵌这些）
    def used_assets(self, data):
        t = self.tables()
        anm2 = set(HUD_ASSETS) | set(GRID_ANM2)
        pngs = set(HUD_PNGS)
        strings = data.get('strings') or {}
        for s in strings.values():
            if isinstance(s, str) and s.lower().endswith('.anm2'):
                anm2.add(norm(s))
        ent_keys, items, trinkets, backdrops, ptypes = set(), set(), set(), set(), set()
        for fr in data['frames']:
            for row in (fr[7] or []):
                if isinstance(row, list) and len(row) > 3 and not isinstance(row[1], list):
                    ent_keys.add((row[1], row[2], row[3]))
                    if row[1] == 5 and row[2] == 100:
                        items.add(row[3])
                    if row[1] == 5 and row[2] == 350:
                        trinkets.add(row[3] & 0x7fff)
        for it in data['items']:
            items.update(i for i, _ in it['items'])
        for r in data['rooms']:
            if r.get('backdrop') is not None:
                backdrops.add(str(r['backdrop']))
        for p in (data['header'].get('meta') or {}).get('players') or []:
            ptypes.add(str(p.get('type')))
        ppl = data['header']['fields']['player']
        if 'ptype' in ppl:
            pi = ppl.index('ptype')
            for fr in data['frames']:
                if fr[3]:
                    for row in fr[6]:
                        if isinstance(row, list) and len(row) == len(ppl) and isinstance(row[pi], int):
                            ptypes.add(str(row[pi]))
            for nm in ('active0', 'active1', 'trinket0', 'trinket1'):
                ci = ppl.index(nm)
                for fr in data['frames']:
                    if fr[3]:
                        for row in fr[6]:
                            if isinstance(row, list) and len(row) == len(ppl) and isinstance(row[ci], int) and row[ci] > 0:
                                (trinkets.add(row[ci] & 0x7fff) if nm.startswith('trinket') else items.add(row[ci]))
        for (ty, v, s) in ent_keys:
            rec = t['entities'].get(f'{ty}.{v}.{s}') or t['entities'].get(f'{ty}.{v}') or t['entities'].get(str(ty))
            if rec:
                anm2.add(rec[0])
        for i in items:
            g = t['items_gfx'].get(str(i))
            if g:
                pngs.add(g)
            for rec in t['costumes_item'].get(str(i), []):
                anm2.add(rec[0])
        for i in trinkets:
            g = t['trinkets_gfx'].get(str(i))
            if g:
                pngs.add(g)
        for pt in ptypes:
            pd = t['players'].get(pt)
            if pd:
                if pd.get('skin'):
                    pngs.add(pd['skin'])
                if pd.get('costume') and pd['costume'] in t['costumes_id']:
                    anm2.add(t['costumes_id'][pd['costume']][0])
        for b in backdrops:
            d = t['backdrops'].get(b) or {}
            for k in ('gfx', 'lfloorgfx', 'nfloorgfx', 'rocks', 'pit', 'waterpit', 'bridge', 'door', 'holeinwall'):
                if d.get(k):
                    pngs.add(d[k])
            if d.get('props'):
                anm2.add(d['props'])
        anm2 = {a for a in anm2 if self.path(a)}
        for a in sorted(anm2):
            pngs.update(self.sheets_of(a))
        pngs = {p for p in pngs if self.path(p)}
        return sorted(anm2), sorted(pngs)

    def embed(self, keys):
        """{键: 文本（anm2/xml）或 data URI（png）}。"""
        out, total = {}, 0
        for k in keys:
            b = self.read_bytes(k)
            if b is None:
                continue
            total += len(b)
            if k.endswith('.png'):
                out[k] = 'data:image/png;base64,' + base64.b64encode(b).decode('ascii')
            else:
                out[k] = b.decode('utf-8', 'replace')
        return out, total


def find_footage(run_path, explicit=None):
    """实录画面层：<会话目录>/footage/<run>.mp4 + <run>.json（schema isaac-footage-v1）。run 名 = 文件名去掉 .jsonl(.zst)。
    返回 (mp4 Path, meta dict) 或 (None, None)。"""
    run_path = Path(run_path)
    stem = run_path.name.split('.jsonl')[0]
    cands = []
    if explicit:
        e = Path(explicit)
        cands.append((e, e.with_suffix('.json')))
    sess = run_path.parent.parent if run_path.parent.name == 'runs' else run_path.parent
    cands.append((sess / 'footage' / f'{stem}.mp4', sess / 'footage' / f'{stem}.json'))
    for mp4, js in cands:
        if mp4.is_file() and js.is_file():
            try:
                meta = json.loads(js.read_text(encoding='utf-8'))
            except ValueError:
                continue
            if meta.get('schema') == 'isaac-footage-v1':
                return mp4, meta
    return None, None
