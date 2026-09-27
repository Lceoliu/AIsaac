"""Build the static web site (GitHub Pages): the page runs the generators in the browser with Pyodide.

usage: python tools/build_site.py [--out DIR] [--pyodide URL] [--supabase FILE]

The output directory holds the page (web/static), config.js (backend = Pyodide), py/macro.zip (the
isaac_macro package and web/api.py), data/<game>.zip: the game files the generators read, taken
from each game's afterbirthp.a with their paths lower-cased (archive.FileSet reads them back), and
art/: the room layouts' backdrop sheets and sprites (web/roomart.py, Repentance+ art).

The scoreboard is shared when --supabase names a JSON file {"url": ..., "key": ...} with the Supabase
project's URL and anon public key (web/scoreboard.sql sets the project up); the default is
rl/runs/macro/supabase.json when it exists. Without it the page keeps scores in the browser.

The data zips and art/ are the games' own room files, string tables, sprites, backdrops and sounds.
They are copyrighted game assets: publishing the site redistributes them. The default output
directory (rl/runs/...) is outside git for that reason.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

MACRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MACRO))
sys.path.insert(0, str(MACRO / 'web'))

from isaac_macro.archive import _normalise                         # noqa: E402
from isaac_macro.rep.roomconfig import default_room_config as rep_room_config   # noqa: E402
from isaac_macro.roomconfig import default_room_config             # noqa: E402
import roomart                                                     # noqa: E402

PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'
SUPABASE = MACRO.parent / 'runs' / 'macro' / 'supabase.json'
COMMON = ['resources/stages.xml', 'resources/entities2.xml', 'resources/gfx/ui/minimap1.png',
          'resources/gfx/ui/minimap1.anm2']
EXTRA = {
    'abplus': [],
    'repplus': ['resources/bosspools.xml', 'resources/bossportraits.xml', 'resources/stringtable.sta',
                'resources/gfx/ui/minimap_icons.png', 'resources/gfx/ui/minimap_icons.anm2'],
}


def game_files(key: str) -> dict[str, bytes]:
    rc = rep_room_config() if key == 'repplus' else default_room_config()
    names = COMMON + EXTRA[key] + sorted(set(rc.paths.values()))
    out = {}
    for name in names:
        try:
            out[_normalise(name).decode('latin-1')] = rc.archives.read(name)
        except FileNotFoundError:
            print(f'  {key}: {name} is not in the archive, skipped')
    return out


def scoreboard_config(path) -> dict | None:
    """{url, key} for the shared scoreboard, or None (the page then keeps scores locally)."""
    if not path or not Path(path).is_file():
        return None
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if not cfg.get('url') or not cfg.get('key'):
        raise SystemExit(f'{path}: needs "url" and "key" (the anon public key)')
    return dict(url=cfg['url'].rstrip('/'), key=cfg['key'])


def config_js(backend: str, pyodide: str | None = None, supabase: dict | None = None) -> str:
    cfg = dict(backend=backend)
    if pyodide:
        cfg['pyodide'] = pyodide
    if supabase:
        cfg['supabase'] = supabase
    return f'window.MAPGEN_CONFIG = {json.dumps(cfg, ensure_ascii=False)};\n'


def write_zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            z.writestr(name, files[name])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(MACRO.parent / 'runs' / 'macro' / 'site'))
    ap.add_argument('--pyodide', default=PYODIDE, help='Pyodide distribution URL (ends with /full/)')
    ap.add_argument('--supabase', default=str(SUPABASE), help='JSON file with the scoreboard\'s url and anon key')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(MACRO / 'web' / 'static', out)
    supabase = scoreboard_config(args.supabase)
    (out / 'config.js').write_text(config_js('pyodide', args.pyodide, supabase), encoding='utf-8')
    print('scoreboard: ' + (f"shared ({supabase['url']})" if supabase else 'this browser only (no Supabase config)'))
    (out / '.nojekyll').write_text('', encoding='utf-8')

    code = {}
    for path in sorted((MACRO / 'isaac_macro').rglob('*.py')):
        code[path.relative_to(MACRO).as_posix()] = path.read_bytes()
    code['api.py'] = (MACRO / 'web' / 'api.py').read_bytes()
    write_zip(out / 'py' / 'macro.zip', code)
    print(f'py/macro.zip: {len(code)} files')

    for key in ('abplus', 'repplus'):
        files = game_files(key)
        write_zip(out / 'data' / f'{key}.zip', files)
        size = (out / 'data' / f'{key}.zip').stat().st_size
        print(f'data/{key}.zip: {len(files)} files, {size / 1e6:.1f} MB')
    roomart.build(out / 'art', log=lambda m: print(f'art/: {m}'))
    print(f'site: {out}')


if __name__ == '__main__':
    main()
