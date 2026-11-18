"""Is a held-out evaluation complete? (2026-11-18, the coordinator: an evaluation with seeds missing must not read as
done.) One status line for ev_c113.sh / ev_ab2.sh, and the exit code for the *_after.sh counters:

  summary yes (requested 128, played 128, missing 0)                     exit 0
  summary yes, INCOMPLETE (requested 128, played 126, missing 2)         exit 1
  summary yes, INCOMPLETE (requested 128, played 128, missing 0, rc 1)   exit 1   (the job's container ended non-zero)
  summary NO (rc 1)                                                      exit 1

requested / played / missing: eval_tok's own summary fields (seeds_requested / seeds_played / seeds_missing, written
since the v3 hardening) when present; else the requested count of --seeds first:count against the summary's episodes
(every group) plus its 'empty' count (start states with nothing to fight count as played, as eval_tok does). rc: the
job's container exit code, from job.sh's <log>.rc when --rc-file is given (absent file: not known, not shown).
  usage: python3 ev_complete.py --dir <evaluation dir> --seeds <first:count> [--rc-file <log>.rc] [--quiet]"""
import argparse
import json
import sys
from pathlib import Path


def status(d, seeds, rc_file=None):
    """(line, complete)"""
    rc = None
    if rc_file and Path(rc_file).is_file():
        txt = Path(rc_file).read_text().strip()
        rc = int(txt) if txt.lstrip('-').isdigit() else txt
    tail = '' if rc is None else f'rc {rc}'
    p = Path(d) / 'summary.json'
    if not p.is_file():
        return f'summary NO{f" ({tail})" if tail else ""}', False
    s = json.loads(p.read_text())
    if 'seeds_missing' in s:
        req, played, missing = int(s['seeds_requested']), int(s['seeds_played']), int(s['seeds_missing'])
    else:
        req = int(str(seeds).split(':')[1])
        eps = sum(int(g.get('episodes', 0)) for g in s.get('groups', {}).values())
        played = min(req, eps + int(round(float(s.get('empty', 0) or 0))))
        missing = req - played
    parts = f'requested {req}, played {played}, missing {missing}' + (f', {tail}' if tail else '')
    ok = missing <= 0 and rc in (None, 0)
    return f'summary yes{"" if ok else ", INCOMPLETE"} ({parts})', ok


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--dir', required=True)
    a.add_argument('--seeds', required=True)
    a.add_argument('--rc-file')
    a.add_argument('--quiet', action='store_true')
    args = a.parse_args(argv)
    line, ok = status(args.dir, args.seeds, args.rc_file)
    if not args.quiet:
        print(line)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
