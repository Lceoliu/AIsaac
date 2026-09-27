"""Smoke test of the web UI's JSON API (web/server.py) for both games."""
import importlib.util
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / 'web' / 'server.py'


@pytest.fixture(scope='module')
def server():
    spec = importlib.util.spec_from_file_location('macro_web_server', SERVER)
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
