"""The page's own logic (web/static/js: the AI replay, the Red Key's targets, the points) on floors the
generators make, run with Node when it is installed."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

MACRO = Path(__file__).resolve().parents[1]


def test_page_logic_on_generated_floors(rc, rep_rc, tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    spec = importlib.util.spec_from_file_location('macro_web_api_js', MACRO / 'web' / 'api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    runs = [api.run_json(dict(game=game, seed=seed, mode='normal'))
            for game in ('abplus', 'repplus') for seed in ('DXNH NZLG', '98765', '424242')]
    path = tmp_path / 'runs.json'
    path.write_text(json.dumps(runs), encoding='utf-8')
    out = subprocess.run([node, str(MACRO / 'tests' / 'js' / 'check_page.mjs'), str(path)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr or out.stdout
    assert out.stdout.startswith('ok ')


def test_score_queue_drops_refused_scores():
    """A score the database refuses (an old day's challenge) must not block the scores behind it."""
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    out = subprocess.run([node, str(MACRO / 'tests' / 'js' / 'check_scoreboard.mjs')], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr or out.stdout
    assert out.stdout.startswith('ok ')
