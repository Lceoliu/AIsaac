"""Smoke test of the web page's JSON API (web/api.py) for both games, from the archives and from the
extracted files the static site ships."""
import importlib.util
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[1] / 'web' / 'api.py'


@pytest.fixture(scope='module')
def server():
    spec = importlib.util.spec_from_file_location('macro_web_api', API)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_run(server, game: str, hidden: set):
    out = server.run_json(dict(game=game, seed='DXNH NZLG', mode='debug', last='8', route='sheol'))
    assert out['game'] == game and len(out['floors']) >= 7
    for f in out['floors']:
        cells = [c for r in f['rooms'] for c in r['cells']]
        assert len(cells) == len(set(cells))
        assert {r['type'] for r in f['rooms'] if r['hidden']} <= hidden
        assert abs(sum(f['secret_posterior'].values()) - 1) < 0.02
        assert abs(sum(row[3] for row in f['joint']) - 1) < 1e-3          # the joint the page conditions
        visible = {c for r in f['rooms'] if not r['hidden'] for c in r['cells']}
        assert not visible & set(f['door_targets'])     # Red Key targets are empty on the player's map
    room = next(r for r in out['floors'][0]['rooms'] if r['type'] == 1 and not r['start'])
    lay = server.layout_json(dict(game=game, stage=room['file'], type=room['type'], variant=room['variant']))
    assert lay['width'] >= 13 and lay['spawns'] is not None
    return out


def test_run_abplus(server, rc):
    out = _check_run(server, 'abplus', {7, 8})
    assert out['validated'] and not any(f['ultra_secret_posterior'] for f in out['floors'])


def test_run_repplus(server, rep_rc):
    out = _check_run(server, 'repplus', {7, 8, 29})
    assert not out['validated']
    assert all(abs(sum(f['ultra_secret_posterior'].values()) - 1) < 0.02 for f in out['floors'])


def test_bad_game(server):
    with pytest.raises(ValueError):
        server.run_json(dict(game='rebirth', seed='DXNH NZLG'))


def test_sprites_are_the_games_minimap(server, rc, rep_rc):
    """The minimap sheets come out of the archives through the stored (ISAAC-scrambled) block path;
    Archive.read_entry checks each file against its checksum."""
    for game, icon_names in (('abplus', {'IconShop', 'IconBoss', 'IconSecretRoom'}),
                             ('repplus', {'IconShop', 'IconPlanetarium', 'IconUltraSecretRoom'})):
        sprites = server.GAMES[game].sprites()
        assert sprites['tiles']['url'].startswith('data:image/png;base64,')
        assert all(len(sprites['tiles']['frames'][k]) == 12 for k in ('RoomVisited', 'RoomUnvisited', 'RoomCurrent'))
        assert sprites['tiles']['frames']['RoomVisited'][0][2:4] == [9, 8]          # a 1x1 room is 9x8 pixels
        assert icon_names <= set(sprites['icons']['frames'])


def test_extracted_files_give_the_same_run(server, rc, rep_rc, tmp_path):
    """The static site reads tools/build_site.py's extracted files (archive.FileSet) instead of the
    archives; the runs must be identical."""
    spec = importlib.util.spec_from_file_location('build_site', API.parents[1] / 'tools' / 'build_site.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    params = dict(seed='DXNH NZLG', last='8', mode='normal')
    ref = {game: server.run_json(dict(params, game=game))['floors'] for game in ('abplus', 'repplus')}
    for game in ref:
        for name, data in build.game_files(game).items():
            path = tmp_path / game / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    server.configure({game: str(tmp_path / game) for game in ref})
    try:
        for game in ref:
            assert server.run_json(dict(params, game=game))['floors'] == ref[game]
    finally:
        server.configure({game: None for game in ref})


def test_challenge_runs_and_batch_layouts(server, rep_rc):
    """A challenge generates only its first floors; a floor's layouts come in one request."""
    out = server.run_json(dict(game='repplus', seed='1234', mode='normal', floors='5'))
    full = server.run_json(dict(game='repplus', seed='1234', mode='normal'))
    assert len(out['floors']) == 5 and len(full['floors']) > 5
    assert out['floors'] == full['floors'][:5]                   # the same floors, just fewer
    rooms = out['floors'][0]['rooms']
    keys = sorted({f"{r['file']}.{r['type']}.{r['variant']}" for r in rooms})
    lays = server.layouts_json(dict(game='repplus', keys=','.join(keys + ['0.1.999999'])))
    assert all(lays[k] and lays[k]['spawns'] is not None for k in keys)
    assert lays['0.1.999999'] is None                              # no such layout
    assert server.warmup_json(dict(game='repplus', stage='1'))['rooms'] > 0

