"""Status of one requested seed block. Exit 0 complete, 1 incomplete, 2 historical rc unknown."""
import argparse
import json
import re
import sys
from pathlib import Path


def _seed_block(text):
    if not isinstance(text, str) or not re.fullmatch(r'[0-9]+:[0-9]+', text):
        raise ValueError('expected first:count')
    first, count = map(int, text.split(':'))
    if not count:
        raise ValueError('empty seed block')
    return first, count


def _count(value):
    # eval_tok writes empty as an integral float; don't round or accept negative counts.
    if type(value) not in (int, float) or value < 0 or not value == int(value):
        raise ValueError('invalid count')
    return int(value)


def status(d, seeds, rc_file=None, require_rc=False):
    """Return (line, True/False/None); None is historical unknown, never complete."""
    d = Path(d)
    claim = d.with_name(d.name + '.log.request')
    receipt = d.with_name(d.name + '.log.recorded')
    rc_path = Path(rc_file) if rc_file else d.with_name(d.name + '.log.rc')
    require_rc = require_rc or receipt.exists()
    try:
        rc_text = rc_path.read_text().strip()
    except FileNotFoundError:
        rc_text = None
    except OSError:
        rc_text = ''
    if rc_text is None:
        rc = 'missing' if require_rc else None
    elif re.fullmatch(r'[0-9]{1,3}', rc_text) and int(rc_text) <= 255:
        rc = int(rc_text)
    else:
        rc = 'invalid'
    if claim.exists() and not receipt.exists():
        return 'summary INCOMPLETE (current job exit status not published)', False
    tail = '' if rc is None else f'rc {rc}'
    p = d / 'summary.json'
    if not p.is_file():
        return f'summary NO{f" ({tail})" if tail else ""}', False
    try:
        requested = _seed_block(seeds)
        s = json.loads(p.read_text(encoding='utf-8'))
        if _seed_block(s['seeds']) != requested:
            return 'summary yes, INCOMPLETE (seed block differs from request)', False
        req = requested[1]
        empty = _count(s.get('empty', 0))
        groups = s.get('groups', {})
        if not isinstance(groups, dict):
            raise ValueError('invalid groups')
        fields = ('seeds_requested', 'seeds_played', 'seeds_missing')
        if any(k in s for k in fields):
            declared, played, missing = (_count(s[k]) for k in fields)
            if declared != req or played + empty + missing != req:
                raise ValueError('inconsistent seed counts')
            if groups and sum(_count(g['episodes']) for g in groups.values()) != played:
                raise ValueError('inconsistent group counts')
        else:
            if len(groups) != 1:
                raise ValueError('legacy summary must have exactly one group')
            played = _count(next(iter(groups.values()))['episodes']) + empty
            missing = req - played
            if missing < 0:
                raise ValueError('more outcomes than requested seeds')
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return f'summary yes, INCOMPLETE ({exc})', False
    parts = f'requested {req}, played {played}, missing {missing}'
    if rc is None:
        return f'summary yes, EXIT STATUS UNKNOWN ({parts}; historical, no rc)', None
    ok = missing == 0 and rc == 0
    return f'summary yes{"" if ok else ", INCOMPLETE"} ({parts}, {tail})', ok


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--dir', required=True)
    a.add_argument('--seeds', required=True)
    a.add_argument('--rc-file')
    a.add_argument('--require-rc', action='store_true')
    a.add_argument('--quiet', action='store_true')
    args = a.parse_args(argv)
    line, ok = status(args.dir, args.seeds, args.rc_file, args.require_rc)
    if not args.quiet:
        print(line)
    return 2 if ok is None else 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
