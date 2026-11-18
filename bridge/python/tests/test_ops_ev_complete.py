"""bm1's evaluation completeness check (bridge/ops/bm1/evtools/ev_complete.py, 2026-11-18): an evaluation with a seed
missing, or whose job ended non-zero, reads "INCOMPLETE" and does not count as done; a complete one reads as before.
Made-up summary.json files, both eval_tok forms: with seeds_requested / played / missing (the v3 hardening) and
without (episodes per group + empty against --seeds first:count)."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ev_complete', HERE.parent.parent / 'ops' / 'bm1' / 'evtools' /
                                              'ev_complete.py')
EC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EC)


def write(d, summary, rc=None):
    d.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (d / 'summary.json').write_text(json.dumps(summary))
    if rc is not None:
        (d.parent / f'{d.name}.log.rc').write_text(f'{rc}\n')
    return d, (d.parent / f'{d.name}.log.rc')


def old_form(episodes, empty=0.0):
    return {'groups': {'run': {'episodes': episodes}}, 'empty': empty}


def test_complete_reads_as_before(tmp_path):
    d, rc = write(tmp_path / 'a', old_form(128), rc=0)
    line, ok = EC.status(d, '2147490000:128', rc)
    assert ok and line == 'summary yes (requested 128, played 128, missing 0, rc 0)'
    assert EC.main(['--dir', str(d), '--seeds', '2147490000:128', '--rc-file', str(rc), '--quiet']) == 0


def test_a_missing_seed_is_incomplete(tmp_path):
    d, rc = write(tmp_path / 'b', old_form(126, empty=1.0), rc=0)
    line, ok = EC.status(d, '2147490000:128', rc)
    assert not ok and line.startswith('summary yes, INCOMPLETE') and 'missing 1' in line
    assert EC.main(['--dir', str(d), '--seeds', '2147490000:128', '--quiet']) == 1


def test_eval_tok_own_counts_win(tmp_path):
    s = dict(old_form(128), seeds_requested=128, seeds_played=127, seeds_missing=1)
    d, _ = write(tmp_path / 'c', s)
    line, ok = EC.status(d, '2147490000:128')
    assert not ok and 'played 127, missing 1' in line and 'rc' not in line


def test_a_nonzero_job_is_incomplete_even_with_every_seed(tmp_path):
    d, rc = write(tmp_path / 'd', old_form(128), rc=1)
    line, ok = EC.status(d, '2147490000:128', rc)
    assert not ok and 'INCOMPLETE' in line and 'rc 1' in line


def test_no_summary(tmp_path):
    d, rc = write(tmp_path / 'e', None, rc=1)
    line, ok = EC.status(d, '2147490000:128', rc)
    assert not ok and line == 'summary NO (rc 1)'
