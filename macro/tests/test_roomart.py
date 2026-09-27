"""The room layouts' game art (web/roomart.py): which backdrop a room gets, and the sprites cut from
the Repentance+ animation files."""
import importlib.util
import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / 'web'


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, WEB / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def api():
    return _load('macro_web_api_art', 'api.py')


@pytest.fixture(scope='module')
def renderer(rep_rc):
    sys.path.insert(0, str(WEB))
    roomart = _load('macro_web_roomart', 'roomart.py')
    return roomart, roomart.Renderer(roomart.gfx_archives())


def test_backdrops_follow_room_type_and_stage(api, rc, rep_rc):
    """Room::LoadBackdropGraphics: special rooms have their own backdrop, the rest the stage's."""
    for game in ('abplus', 'repplus'):
        out = api.run_json(dict(game=game, seed='DXNH NZLG', mode='normal', last='8'))
        for f in out['floors']:
            sid = api.GAMES[game].stage_id(f['stage'], f['stage_type'])
            assert f['backdrop'] == (sid if game == 'abplus' else api.GAMES[game].stage_backdrop(sid))
            for r in f['rooms']:
                if r['type'] == 8 and r['variant'] in api.SUPER_SECRET_BACKDROPS:
                    assert r['backdrop'] == api.SUPER_SECRET_BACKDROPS[r['variant']]
                elif game == 'repplus' and r['type'] in api.REP_TYPE_BACKDROPS:
                    assert r['backdrop'] == api.REP_TYPE_BACKDROPS[r['type']]
                elif r['type'] in api.TYPE_BACKDROPS:
                    assert r['backdrop'] == api.TYPE_BACKDROPS[r['type']]
                else:
                    assert r['backdrop'] == f['backdrop']
    assert api.GAMES['repplus'].stage_backdrop(27) == 31      # Downpour (stages.xml)


def test_grid_sprites(renderer):
    roomart, r = renderer
    rock = r.anm2('resources/gfx/grid/grid_rock.anm2')
    assert rock.keyframes('normal') == 3
    img, ox, oy = r.render(rock, 'normal', 0, {0: 'resources/gfx/grid/rocks_basement.png'})
    assert img.size[0] <= 32 and img.size[1] <= 32 and 0 < ox <= 16 and 0 < oy <= 17   # frame at (-16, -17)
    pit = r.anm2('resources/gfx/grid/grid_pit.anm2')
    assert pit.keyframes('pit') == 33                                   # GridEntity_Pit::PostInit's frames
    img, ox, oy = r.render(pit, 'pit', 15, only={0})
    assert img.size == (26, 26) and (ox, oy) == (14, 14)


def test_entity_sprites(renderer):
    roomart, r = renderer
    entities = roomart.load_entities(r.archives)
    items = roomart.load_item_gfx(r.archives)
    gaper, _, head_y = roomart.entity_sprite(r, entities, items, (10, 1, 0))
    body, _, body_y = r.render(r.anm2('resources/gfx/010.001_Gaper.anm2'), 'WalkVert', 0)
    assert gaper.size[1] > body.size[1] and head_y > body_y             # the Head overlay is drawn on top
    pedestal, _, _ = roomart.entity_sprite(r, entities, items, (5, 100, 0))
    assert pedestal.size[1] > 32                                        # the altar and the "?" above it
    assert roomart.entity_sprite(r, entities, items, (5, 0, 0)) is None  # a random pickup has no sprite


def test_ui_assets(renderer, tmp_path):
    """The page's own bits: Isaac's thumbs up (the player's Happy animation), the Red Key (items.xml
    580), minimap icons, and the secret room jingle and thumbs-up sound when they are there."""
    roomart, r = renderer
    ui = roomart.build_ui(r, tmp_path, log=lambda m: None)
    happy = ui['happy']
    assert len(happy['delays']) == 7 and happy['delays'][happy['thumb']] == 12      # the thumb is held longest
    assert (tmp_path / happy['file']).stat().st_size > 0
    assert (ui['redkey']['w'], ui['redkey']['h']) == (27, 14)
    assert {'IconSecretRoom', 'IconSuperSecretRoom', 'IconUltraSecretRoom', 'IconBomb'} <= set(ui['icons'])
    assert set(ui['sounds']) == {'secret', 'thumbsup'}
    menu = ui['menu']                              # the home page is built from the game's own menus
    assert {'wall', 'pinned', 'board', 'strip', 'streak', 'cursor', 'fly', 'note'} <= set(menu)
    assert (menu['wall']['w'], menu['wall']['h']) == (480, 270) and (menu['board']['w'], menu['board']['h']) == (416, 240)

