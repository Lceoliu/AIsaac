"""Room layouts in the game's own art: backdrop sheets and sprites pre-rendered from its resources.

The layout view (web/static/js/roomart.js) draws a room the way the game does. The backdrop is
composed in the page from the stage's sheet, as Backdrop::pre_render_floor / pre_render_walls do it
(AB+ 0x1150A0 / 0x10A2C0). Everything else is a sprite: one frame of an animation (anm2) with its
layers, crops, pivots, scales and tints applied here, cut to its bounding box and saved with the
position of its origin, so the page only places it.

Both games use the Repentance+ art: graphics.a, afterbirth.a, afterbirthp.a and repentance.a from
the game's resources/packed, read only. Afterbirth+ rooms borrow it; the backdrop, grid and entity
ids they use are the same.

    python web/roomart.py [--out DIR]        default rl/runs/macro/art (the local server's cache)

writes art.json (the index), bd/*.png (backdrop sheets) and s/*.png (sprites, named by content).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import posixpath
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaac_macro.archive import ArchiveSet                                   # noqa: E402
from isaac_macro.rep.roomconfig import default_archive_path as rep_archive  # noqa: E402

VERSION = 2
REP_GFX = ('graphics.a', 'afterbirth.a', 'afterbirthp.a', 'repentance.a')
DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'runs' / 'macro' / 'art'

# Backdrops the layout view can ask for: the stages' (stages.xml) and the special rooms'
# (api.TYPE_BACKDROPS / SUPER_SECRET_BACKDROPS). Mega Satan, the error room, the planetarium and the
# Home floors are animations or not generated here.
BACKDROPS = (tuple(range(1, 18)) + (19, 20, 21, 22, 23, 24, 25, 27, 28, 30)
             + (31, 32, 33, 34, 45, 46, 47))
ROCK_ANIMS = ('normal', 'alt', 'alt2', 'bombrock', 'tinted', 'superspecial', 'spiked', 'foolsgold',
              'pillar', 'black')
POOPS = {   # stb poop kinds -> the sheets grid_poop.anm2 is drawn with
    'normal': ['grid_poop_1.png', 'grid_poop_2.png', 'grid_poop_3.png'],
    'red': ['grid_poop_red_1.png', 'grid_poop_red_2.png', 'grid_poop_red_3.png'],
    'white': ['grid_poop_white_1.png', 'grid_poop_white_2.png', 'grid_poop_white_3.png'],
    'black': ['grid_poop_black.png'], 'charming': ['grid_poop_charming.png'], 'corn': ['grid_poop_corn.png'],
    'gold': ['grid_poop_gold.png'], 'rainbow': ['grid_poop_rainbow.png'],
}
SINGLE = {  # sprite key -> (anm2 under gfx/grid, animation)
    'giantpoop': ('grid_poop_giant.anm2', 'State1'),
    'tnt': ('grid_tnt.anm2', 'Idle'),
    'web': ('grid_web.anm2', 'Idle'),
    'lock': ('grid_locks.anm2', 'Idle'),
    'plate': ('grid_pressureplate.anm2', 'Off'),
    'trapdoor': ('Door_11_TrapDoor.anm2', 'Opened'),
    'crawlspace': ('door_20_secrettrapdoor.anm2', 'Opened'),
    'teleporter': ('grid_teleporter.anm2', 'IdleOn'),
}
SPIKES = ('Spikes01', 'Spikes02', 'Spikes03', 'Spikes04', 'WombSpikes01', 'WombSpikes02', 'WombSpikes03',
          'WombSpikes04')
# The frame that stands for an entity: its idle or walking pose, else the file's default animation.
POSES = ('Idle', 'IdleDown', 'Idle Down', 'WalkDown', 'Walk Down', 'WalkVert', 'Move Down', 'Walk', 'Fly',
         'Float', 'FloatDown', 'Hop', 'Move', 'Shake', 'Stand')
OVERLAYS = ('Head', 'HeadDown', 'Head Down')   # drawn on top of the pose (gapers' heads)
TRANSIENT = re.compile(r'appear|death|die|spawn|dissapear|disappear|collect', re.I)


def gfx_archives() -> ArchiveSet:
    packed = Path(rep_archive()).parent
    return ArchiveSet([packed / name for name in REP_GFX])


def _attrs(text: str) -> dict:
    return dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', text))


def _num(frame: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(frame.get(key, default))
    except ValueError:
        return default


class Anm2:
    """One animation file: spritesheets, layers and each animation's key frames."""

    def __init__(self, text: str, path: str):
        text = re.sub(r'^\s*<\?xml[^>]*\?>', '', text.lstrip('﻿'))
        root = ET.fromstring(text)
        base = posixpath.dirname(path)
        self.sheets = {int(s.get('Id')): posixpath.normpath(posixpath.join(base, s.get('Path', '').replace('\\', '/')))
                       for s in root.iter('Spritesheet')}
        content = root.find('Content')
        layers = content.find('Layers') if content is not None else None
        self.layers = {int(l.get('Id')): int(l.get('SpritesheetId', 0)) for l in (layers if layers is not None else [])}
        anims = root.find('Animations')
        self.default = anims.get('DefaultAnimation') if anims is not None else None
        self.anims: dict[str, dict] = {}
        for an in (anims if anims is not None else []):
            ra = an.find('RootAnimation')
            la = an.find('LayerAnimations')
            self.anims[an.get('Name')] = dict(
                root=[dict(f.attrib) for f in ra.iter('Frame')] if ra is not None else [],
                layers=[(int(l.get('LayerId')), l.get('Visible', 'true') == 'true', [dict(f.attrib) for f in l.iter('Frame')])
                        for l in (la if la is not None else [])])

    def keyframes(self, anim: str) -> int:
        """Key frames of the animation's first layer (a rock's variants, a pit's 33 tiles)."""
        layers = self.anims[anim]['layers']
        return len(layers[0][2]) if layers else 0

    def key_time(self, anim: str, index: int) -> int:
        frames = self.anims[anim]['layers'][0][2]
        return int(sum(_num(f, 'Delay', 1) for f in frames[:index]))


def _frame_at(frames: list, t: int) -> dict | None:
    start = 0
    for f in frames:
        delay = max(1, int(_num(f, 'Delay', 1)))
        if t < start + delay:
            return f
        start += delay
    return None


def _matrix(f: dict) -> np.ndarray:
    """anm2 frame transform: position, rotation (degrees, clockwise on screen), scale, around the pivot."""
    sx, sy = _num(f, 'XScale', 100) / 100, _num(f, 'YScale', 100) / 100
    a = math.radians(_num(f, 'Rotation'))
    c, s = math.cos(a), math.sin(a)
    px, py = _num(f, 'XPivot'), _num(f, 'YPivot')
    x, y = _num(f, 'XPosition'), _num(f, 'YPosition')
    lin = np.array([[c * sx, -s * sy], [s * sx, c * sy]])
    out = np.eye(3)
    out[:2, :2] = lin
    out[:2, 2] = lin @ np.array([-px, -py]) + np.array([x, y])
    return out


def _tint(img: Image.Image, frames: list[dict]) -> Image.Image:
    mul = np.ones(4, dtype=np.float32)
    add = np.zeros(4, dtype=np.float32)
    for f in frames:
        mul *= np.array([_num(f, k, 255) for k in ('RedTint', 'GreenTint', 'BlueTint', 'AlphaTint')], dtype=np.float32) / 255
        add[:3] += np.array([_num(f, k, 0) for k in ('RedOffset', 'GreenOffset', 'BlueOffset')], dtype=np.float32) / 255
    if np.all(mul == 1) and np.all(add == 0):
        return img
    arr = np.asarray(img, dtype=np.float32) / 255
    arr = np.clip(arr * mul + add, 0, 1)
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGBA')


def _place(crop: Image.Image, m: np.ndarray) -> tuple[Image.Image, int, int]:
    """The crop drawn with transform m (crop pixels -> origin-relative screen pixels)."""
    w, h = crop.size
    lin = m[:2, :2]
    if np.allclose(np.abs(lin), np.eye(2)):          # plain or flipped: no resampling
        if lin[0, 0] < 0:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if lin[1, 1] < 0:
            crop = crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        xs = [m[0, 2], m[0, 2] + lin[0, 0] * w]
        ys = [m[1, 2], m[1, 2] + lin[1, 1] * h]
        return crop, int(round(min(xs))), int(round(min(ys)))
    corners = m @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], dtype=float)
    left, top = math.floor(corners[0].min()), math.floor(corners[1].min())
    right, bottom = math.ceil(corners[0].max()), math.ceil(corners[1].max())
    inv = np.linalg.inv(m)
    data = (inv[0, 0], inv[0, 1], inv[0, 0] * left + inv[0, 1] * top + inv[0, 2],
            inv[1, 0], inv[1, 1], inv[1, 0] * left + inv[1, 1] * top + inv[1, 2])
    out = crop.transform((max(1, right - left), max(1, bottom - top)), Image.Transform.AFFINE, data,
                         resample=Image.Resampling.NEAREST)
    return out, left, top


class Renderer:
    def __init__(self, archives: ArchiveSet):
        self.archives = archives
        self._anm2: dict[str, Anm2 | None] = {}
        self._sheets: dict[str, Image.Image | None] = {}

    def anm2(self, path: str) -> Anm2 | None:
        key = path.lower()
        if key not in self._anm2:
            try:
                self._anm2[key] = Anm2(self.archives.read(path).decode('utf-8-sig', 'replace'), path)
            except (FileNotFoundError, ET.ParseError, ValueError, AttributeError):
                self._anm2[key] = None
        return self._anm2[key]

    def sheet(self, path: str) -> Image.Image | None:
        key = path.lower()
        if key not in self._sheets:
            try:
                self._sheets[key] = Image.open(io.BytesIO(self.archives.read(path))).convert('RGBA')
            except (FileNotFoundError, OSError, ValueError):
                self._sheets[key] = None
        return self._sheets[key]

    def parts(self, anm: Anm2, anim: str, t: int, sheets: dict | None = None, only: set | None = None) -> list:
        a = anm.anims.get(anim)
        if a is None:
            return []
        root = _frame_at(a['root'], t) or {}
        if root.get('Visible', 'true') == 'false':
            return []
        mroot = _matrix(root)
        out = []
        for layer_id, visible, frames in a['layers']:
            if not visible or (only is not None and layer_id not in only):
                continue
            f = _frame_at(frames, t)
            if f is None or f.get('Visible', 'true') == 'false':
                continue
            sid = anm.layers.get(layer_id, 0)
            path = (sheets or {}).get(sid) or anm.sheets.get(sid)
            img = self.sheet(path) if path else None
            if img is None:
                continue
            x, y, w, h = (int(_num(f, k)) for k in ('XCrop', 'YCrop', 'Width', 'Height'))
            if w <= 0 or h <= 0:
                continue
            crop = _tint(img.crop((x, y, x + w, y + h)), [root, f])
            out.append(_place(crop, mroot @ _matrix(f)))
        return out

    def render(self, anm: Anm2, anims, t: int = 0, sheets: dict | None = None, only: set | None = None):
        """Layers of one or more animations at time t -> (image, origin x, origin y), or None."""
        parts = []
        for anim in ([anims] if isinstance(anims, str) else anims):
            parts += self.parts(anm, anim, t, sheets, only)
        if not parts:
            return None
        left = min(l for _, l, _ in parts)
        top = min(tp for _, _, tp in parts)
        right = max(l + im.width for im, l, _ in parts)
        bottom = max(tp + im.height for im, _, tp in parts)
        canvas = Image.new('RGBA', (right - left, bottom - top))
        for im, l, tp in parts:
            layer = Image.new('RGBA', canvas.size)
            layer.paste(im, (l - left, tp - top))
            canvas = Image.alpha_composite(canvas, layer)
        box = canvas.getbbox()
        if box is None:
            return None
        return canvas.crop(box), -left - box[0], -top - box[1]


class Sprites:
    """Sprite files named by content (identical frames share a file) and their index entries."""

    def __init__(self, out: Path):
        self.out = out
        (out / 's').mkdir(parents=True, exist_ok=True)
        self.index: dict[str, list] = {}
        self.groups: dict[str, int] = {}
        self._files: set[str] = set()

    def add(self, key: str, rendered) -> bool:
        if rendered is None:
            return False
        img, ox, oy = rendered
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        data = buf.getvalue()
        name = f's/{hashlib.sha1(data).hexdigest()[:12]}.png'
        if name not in self._files:
            (self.out / name).write_bytes(data)
            self._files.add(name)
        self.index[key] = [name, img.width, img.height, int(ox), int(oy)]
        return True

    def group(self, prefix: str, rendered_frames: list) -> None:
        """prefix/0, prefix/1, ...: frames the page picks from (a rock's shapes, a pit's tiles)."""
        n = 0
        for r in rendered_frames:
            if self.add(f'{prefix}/{n}', r):
                n += 1
        if n:
            self.groups[prefix] = n


def load_backdrops(archives: ArchiveSet) -> dict[int, dict]:
    xml = archives.read('resources/backdrops.xml').decode('utf-8-sig', 'replace')
    roots = _attrs(re.search(r'<backdrops\s([^>]*)>', xml).group(1))
    out = {}
    for m in re.finditer(r'<backdrop\s(.*?)/>', xml, re.S):
        a = _attrs(m.group(1))
        if 'id' in a:
            a['_gfxroot'], a['_gridroot'] = roots.get('gfxroot', 'gfx/backdrop/'), roots.get('gridgfxroot', 'gfx/grid/')
            out[int(a['id'])] = a
    return out


def load_entities(archives: ArchiveSet) -> dict[tuple, dict]:
    xml = archives.read('resources/entities2.xml').decode('utf-8-sig', 'replace')
    out = {}
    for m in re.finditer(r'<entity\s([^>]*?)/?>', xml):
        a = _attrs(m.group(1))
        try:
            key = (int(a['id']), int(a.get('variant', 0) or 0), int(a.get('subtype', 0) or 0))
        except (KeyError, ValueError):
            continue
        out.setdefault(key, a)
    return out


def room_entity_keys() -> set[tuple]:
    """(type, variant, subtype) of every entity the two games' room files spawn."""
    import api   # web/api.py, next to this file
    keys = set()
    for game in api.GAMES.values():
        rc = game.room_config()
        for sid in rc.paths:
            try:
                rooms = rc.rooms(sid)
            except (FileNotFoundError, NotImplementedError, ValueError):
                continue
            for room in rooms:
                for spawn in room.spawns:
                    for e in spawn.entries:
                        if 0 < e.type < 1000:
                            keys.add((e.type, e.variant, e.subtype))
    return keys


def load_item_gfx(archives: ArchiveSet) -> dict[int, str]:
    """Collectible id -> its sprite (items.xml)."""
    xml = archives.read('resources/items.xml').decode('utf-8-sig', 'replace')
    out = {}
    for m in re.finditer(r'<(?:passive|active|familiar)\s([^>]*?)/?>', xml):
        a = _attrs(m.group(1))
        if a.get('id', '').isdigit() and a.get('gfx'):
            out[int(a['id'])] = 'resources/gfx/items/collectibles/' + a['gfx']
    return out


def entity_sprite(r: Renderer, entities: dict, items: dict, key: tuple):
    t, v, s = key
    a = entities.get(key) or entities.get((t, v, 0)) or (entities.get((t, v, 1)) if t == 5 else None) \
        or (entities.get((t, 0, 0)) if t != 5 else None)
    if not a or not a.get('anm2path'):
        return None
    anm = r.anm2('resources/gfx/' + a['anm2path'].replace('\\', '/'))
    if anm is None or not anm.anims:
        return None
    if (t, v) == (5, 100) and 'Alternates' in anm.anims:     # the altar, and the item or "?" above it
        item = items.get(s, 'resources/gfx/items/collectibles/questionmark.png')
        return r.render(anm, ['Alternates', 'Idle'], 0, {1: item})
    if (t, v) == (5, 150):                                    # a shop slot: some item for sale, "?" here
        shop = r.anm2('resources/gfx/005.100_Collectible.anm2')
        return r.render(shop, 'ShopIdle', 0, {1: 'resources/gfx/items/collectibles/questionmark.png'}) if shop else None
    if t == 33 and 'Flickering' in anm.anims:               # a fire place with its flame
        return r.render(anm, 'Flickering', 0)
    pose = next((n for n in POSES if n in anm.anims), None)
    if pose is None:
        pose = anm.default if anm.default in anm.anims and not TRANSIENT.search(anm.default) else \
            next((n for n in anm.anims if not TRANSIENT.search(n)), anm.default or next(iter(anm.anims)))
    anims = [pose] + [n for n in OVERLAYS if n in anm.anims and n != pose][:1]
    for t0 in (0, 1, 2, 4, 8):
        out = r.render(anm, anims, t0)
        if out is not None:
            return out
    return None


def build(out: Path = DEFAULT_OUT, log=print) -> dict:
    t0 = time.time()
    out = Path(out)
    for old in ('bd', 's'):                     # sprite files are named by content: start clean
        shutil.rmtree(out / old, ignore_errors=True)
    (out / 'bd').mkdir(parents=True, exist_ok=True)
    archives = gfx_archives()
    r = Renderer(archives)
    sprites = Sprites(out)
    backdrops = load_backdrops(archives)

    bd_index = {}
    copied = {}

    def copy_png(path: str) -> str | None:
        stem = re.sub(r'[^a-z0-9_.-]+', '_', posixpath.basename(path).lower())
        if stem not in copied:
            try:
                data = archives.read(path)
            except FileNotFoundError:
                copied[stem] = None
            else:
                (out / 'bd' / stem).write_bytes(data)
                copied[stem] = f'bd/{stem}'
        return copied[stem]

    rocks, pits, doors = {'rocks_basement.png'}, {'grid_pit.png'}, {'door_01_normaldoor.png'}
    for bid in BACKDROPS:
        a = backdrops.get(bid)
        if not a or not a.get('gfx', '').lower().endswith('.png'):
            continue
        root = 'resources/' + a['_gfxroot']
        sheet = copy_png(root + a['gfx'])
        if sheet is None:
            continue
        entry = dict(sheet=sheet, walls=int(a.get('walls', 0) or 0), wallvariants=int(a.get('wallvariants', 1) or 1),
                     floors=int(a.get('floors', 1) or 1), floorvariants=int(a.get('floorvariants', 1) or 1))
        if a.get('lfloorgfx'):
            lfloor = copy_png(root + a['lfloorgfx'])
            if lfloor:
                entry['lfloor'] = lfloor
        for attr, bucket in (('rocks', rocks), ('pit', pits), ('door', doors)):
            if a.get(attr):
                entry[attr] = a[attr].lower()
                bucket.add(a[attr].lower())
        bd_index[str(bid)] = entry
    log(f'backdrops: {len(bd_index)} ({len([v for v in copied.values() if v])} sheets)')

    grid = 'resources/gfx/grid/'
    rock = r.anm2(grid + 'grid_rock.anm2')
    for sheet in sorted(rocks):
        if r.sheet(grid + sheet) is None:
            continue
        for anim in ROCK_ANIMS:
            if anim in rock.anims:
                sprites.group(f'rock/{sheet}/{anim}', [r.render(rock, anim, rock.key_time(anim, i), {0: grid + sheet})
                                                       for i in range(rock.keyframes(anim))])
    pit = r.anm2(grid + 'grid_pit.anm2')
    for sheet in sorted(pits):
        if r.sheet(grid + sheet) is not None:
            sprites.group(f'pit/{sheet}', [r.render(pit, 'pit', i, {0: grid + sheet}, only={0})
                                           for i in range(pit.keyframes('pit'))])
    poop = r.anm2(grid + 'grid_poop.anm2')
    for kind, sheets in POOPS.items():
        sprites.group(f'poop/{kind}', [r.render(poop, 'State1', 0, {0: grid + s}, only={0}) for s in sheets])
    spikes = r.anm2(grid + 'grid_spikes.anm2')
    for anim in SPIKES:
        sprites.add(f'spikes/{anim}', r.render(spikes, anim, 0))
    for key, (name, anim) in SINGLE.items():
        anm = r.anm2(grid + name)
        if anm is not None:
            sprites.add(key, r.render(anm, anim if anim in anm.anims else anm.default, 0))
    door = r.anm2(grid + 'Door_01_NormalDoor.anm2')
    for sheet in sorted(doors):
        if r.sheet(grid + sheet) is not None:
            sprites.add(f'door/{sheet}', r.render(door, 'Opened', 0, {0: grid + sheet}))
    log(f'grid sprites: {len(sprites.index)}')

    entities = load_entities(archives)
    items = load_item_gfx(archives)
    # fire places come from stb 1400 / 1410; statues (stb 5000 / 5001) are effects 1000.6 / 1000.9
    # (GridEntity_Statue::InitSubclass, AB+ 0x30C4E0)
    keys = room_entity_keys() | {(33, 0, 0), (33, 1, 0), (1000, 6, 0), (1000, 9, 0)}
    missing = []
    for key in sorted(keys):
        if not sprites.add('e/%d.%d.%d' % key, entity_sprite(r, entities, items, key)):
            missing.append(key)
    log(f'entities: {len(keys) - len(missing)} of {len(keys)} drawn; no sprite for {len(missing)}')

    index = dict(version=VERSION, tile=26, backdrops=bd_index, sprites=sprites.index, groups=sprites.groups,
                 missing=['%d.%d.%d' % k for k in missing])
    (out / 'art.json').write_text(json.dumps(index, separators=(',', ':')), encoding='utf-8')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    log(f'art: {len(sprites.index)} sprites in {len(sprites._files)} files, {size / 1e6:.1f} MB, '
        f'{time.time() - t0:.1f} s -> {out}')
    return index


def ensure(out: Path = DEFAULT_OUT, log=print) -> Path:
    """The art directory, built first if it is missing or from an older version of this script."""
    out = Path(out)
    try:
        if json.loads((out / 'art.json').read_text(encoding='utf-8')).get('version') == VERSION:
            return out
    except (OSError, ValueError):
        pass
    build(out, log)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    build(Path(ap.parse_args().out))


if __name__ == '__main__':
    main()
