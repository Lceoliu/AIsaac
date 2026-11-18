"""Completeness uses the requested seed block and the exit status of this attempt."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'ev_complete', HERE.parent.parent / 'ops' / 'bm1' / 'evtools' / 'ev_complete.py')
EC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EC)
REQUEST = '2147490000:128'


def write(d, summary, rc=None):
    d.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (d / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    sidecar = d.with_name(d.name + '.log.rc')
    if rc is not None:
        sidecar.write_text(f'{rc}\n', encoding='utf-8')
    return d, sidecar


def old_form(episodes, empty=0.0, **extra):
    return dict({'seeds': REQUEST, 'groups': {'run': {'episodes': episodes}}, 'empty': empty}, **extra)


def test_complete_reads_as_before(tmp_path):
    d, rc = write(tmp_path / 'a', old_form(128), rc=0)
    line, ok = EC.status(d, REQUEST, rc)
    assert ok and line == 'summary yes (requested 128, played 128, missing 0, rc 0)'
    assert EC.main(['--dir', str(d), '--seeds', REQUEST, '--rc-file', str(rc), '--quiet']) == 0


def test_a_missing_seed_is_incomplete(tmp_path):
    d, rc = write(tmp_path / 'b', old_form(126, empty=1.0), rc=0)
    line, ok = EC.status(d, REQUEST, rc)
    assert not ok and 'INCOMPLETE' in line and 'missing 1' in line


def test_eval_tok_own_counts_are_checked(tmp_path):
    s = old_form(126, empty=1, seeds_requested=128, seeds_played=126, seeds_missing=1)
    d, rc = write(tmp_path / 'c', s, rc=0)
    line, ok = EC.status(d, REQUEST, rc)
    assert not ok and 'played 126, missing 1' in line
    (d / 'summary.json').write_text(json.dumps(dict(s, seeds_missing=-1)), encoding='utf-8')
    assert EC.status(d, REQUEST, rc)[1] is False


def test_a_nonzero_job_is_incomplete_even_with_every_seed(tmp_path):
    d, rc = write(tmp_path / 'd', old_form(128), rc=1)
    line, ok = EC.status(d, REQUEST, rc)
    assert not ok and 'INCOMPLETE' in line and 'rc 1' in line


def test_no_summary(tmp_path):
    d, rc = write(tmp_path / 'e', None, rc=1)
    assert EC.status(d, REQUEST, rc) == ('summary NO (rc 1)', False)


def test_historical_and_required_rc_are_distinct(tmp_path):
    d, rc = write(tmp_path / 'e', old_form(128))
    line, ok = EC.status(d, REQUEST, rc)
    assert ok is None and 'EXIT STATUS UNKNOWN' in line
    assert EC.main(['--dir', str(d), '--seeds', REQUEST, '--quiet']) == 2
    assert EC.status(d, REQUEST, rc, require_rc=True)[1] is False
    d.with_name(d.name + '.log.recorded').write_text(REQUEST, encoding='utf-8')
    assert EC.status(d, REQUEST, rc)[1] is False
    for bad in ('', 'oops', '--3', '999'):
        rc.write_text(bad, encoding='utf-8')
        assert EC.status(d, REQUEST, rc)[1] is False


def test_different_seed_block_and_contradictory_counts(tmp_path):
    for s in (
        old_form(128, seeds='2147490001:128'),
        old_form(32, seeds_requested=32, seeds_played=32, seeds_missing=0),
        old_form(128, seeds_requested=128, seeds_played=127, seeds_missing=0),
        old_form(128, seeds_requested=128, seeds_played=127, seeds_missing=1),
        old_form(129),
        old_form(128, empty=-1),
    ):
        d, rc = write(tmp_path / 'e', s, rc=0)
        assert EC.status(d, REQUEST, rc)[1] is False


def test_legacy_multi_group_cannot_pass_on_partial_or_nominal_sum(tmp_path):
    for groups in ({'normal': {'episodes': 128}, 'boss': {'episodes': 0}},
                   {'normal': {'episodes': 128}, 'boss': {'episodes': 128}}):
        d, rc = write(tmp_path / 'e', old_form(128, groups=groups), rc=0)
        assert EC.status(d, REQUEST, rc)[1] is False


def test_unpublished_attempt_cannot_use_a_stale_rc(tmp_path):
    d, rc = write(tmp_path / 'e', old_form(128), rc=0)
    d.with_name(d.name + '.log.request').write_text(REQUEST, encoding='utf-8')
    assert EC.status(d, REQUEST, rc)[1] is False
    d.with_name(d.name + '.log.recorded').write_text(REQUEST, encoding='utf-8')
    assert EC.status(d, REQUEST, rc)[1] is True
