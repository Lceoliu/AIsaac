"""(Codex round 24-4) The verdict of a FastRow check run (ISAAC_RL_FAST_ROW=2, or the canary: =1 with
ISAAC_RL_FAST_ROW_CHECK_WORKERS) from its own, fresh ISAAC_RL_FAST_ROW_CHECK_DIR: a table of every report, and PASS only
when
  - every expected worker (--workers: '0-15', '0', ...) left a final report (final: true -- a periodic one only does
    not count), once;
  - every child a worker forked (its fastrow-children-<pid>.txt: pid, role) left a final report of that role;
  - no report has a mismatch, no mismatch dump is there, every report compared something or took no fast record;
  - none of the run's logs (--log, any number: its stdout / stderr) shows a worker's fatal (exit 86 / FastRowFatal /
    a visible-v3 protocol error).
Also printed: the boundary hit counts summed (hit_*), the records compared, the slow records by hand-back code.

   usage: python tools/fastrow_verify.py <check dir> --workers 0-15 [--log run.stdout --log run.stderr]
   exit 0: PASS, 1: FAIL"""
import argparse
import glob
import json
import os
import re
import sys


def parse_set(spec):
    out = set()
    for part in filter(None, spec.split(',')):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dir')
    ap.add_argument('--workers', required=True, help="the workers that run check mode, e.g. '0-15' or '0'")
    ap.add_argument('--log', action='append', default=[])
    a = ap.parse_args(argv)
    bad = []
    reps = {}
    for f in sorted(glob.glob(os.path.join(a.dir, 'fastrow-[0-9]*.json'))):
        try:
            r = json.load(open(f))
            reps[int(r['pid'])] = r
        except Exception as exc:   # noqa: BLE001
            bad.append(f'unreadable report {os.path.basename(f)}: {exc}')
    want = parse_set(a.workers)
    workers = {}
    for pid, r in reps.items():
        if r.get('role') == 'worker':
            workers.setdefault(int(r.get('worker', -1)), []).append(r)
    for w in sorted(want):
        rs = workers.get(w, [])
        if not rs:
            bad.append(f'worker {w}: no report')
        elif len(rs) > 1:
            bad.append(f'worker {w}: {len(rs)} reports (pids {[x["pid"] for x in rs]})')
        elif not rs[0].get('final'):
            bad.append(f'worker {w}: only a periodic report (no final)')
    for w in sorted(set(workers) - want):
        bad.append(f'worker {w}: a report from a worker not expected to check')
    children = []
    for f in sorted(glob.glob(os.path.join(a.dir, 'fastrow-children-*.txt'))):
        ppid = int(re.search(r'children-(\d+)', f).group(1))
        for line in open(f):
            if line.strip():
                pid_, role_ = line.split()
                children.append((ppid, int(pid_), role_))
    for ppid, pid_, role_ in children:
        r = reps.get(pid_)
        if r is None:
            bad.append(f'child {pid_} ({role_} of {ppid}): no report')
        elif not r.get('final'):
            bad.append(f'child {pid_} ({role_} of {ppid}): only a periodic report (no final)')
        elif r.get('role') != role_:
            bad.append(f'child {pid_}: listed as {role_}, reported as {r.get("role")}')
    listed = {c[1] for c in children}
    for pid, r in reps.items():
        if r.get('role', '').endswith('child') and pid not in listed:
            bad.append(f'child {pid} ({r.get("role")}): a report but not listed by its parent')
    forked = sum(int(r.get('children_forked', 0)) for r in reps.values())
    if forked != len(children):
        bad.append(f'children forked {forked} != children listed {len(children)}')
    for pid, r in reps.items():
        if r.get('check_mismatch'):
            bad.append(f'{r.get("role")} {pid}: {r["check_mismatch"]} mismatch(es)')
        if r.get('fast', 0) and not r.get('check_records', 0):
            bad.append(f'{r.get("role")} {pid}: {r["fast"]} fast records, none compared')
    dumps = glob.glob(os.path.join(a.dir, 'fastrow-mismatch-*.npz'))
    if dumps:
        bad.append(f'{len(dumps)} mismatch dump(s)')
    pat = re.compile(r'FastRowFatal|exit 86|visible-v3 protocol error|FastRow check:')
    for lg in a.log:
        try:
            with open(lg, errors='replace') as f:
                hits = [l.strip()[:200] for l in f if pat.search(l)]
        except OSError as exc:
            bad.append(f'log {lg}: {exc}')
            continue
        if hits:
            bad.append(f'log {os.path.basename(lg)}: {len(hits)} fatal line(s), e.g. {hits[0]}')
    # the table
    print(f'{"role":<13} {"worker":>6} {"pid":>8} {"ppid":>8} {"final":>5} {"compared":>9} {"mismatch":>8} {"fast":>8} '
          f'{"slow":>7} {"forked":>6}')
    for pid, r in sorted(reps.items(), key=lambda kv: (kv[1].get('role', ''), kv[1].get('worker', -1), kv[0])):
        print(f'{r.get("role", "?"):<13} {r.get("worker", -1):>6} {pid:>8} {r.get("ppid", 0):>8} '
              f'{str(bool(r.get("final"))):>5} {r.get("check_records", 0):>9} {r.get("check_mismatch", 0):>8} '
              f'{r.get("fast", 0):>8} {r.get("slow", 0):>7} {r.get("children_forked", 0):>6}')
    tot = {}
    for r in reps.values():
        for k, v in r.items():
            if isinstance(v, (int, float)) and (k.startswith('hit_') or k.startswith('slow_') or
                                                k in ('check_records', 'check_mismatch', 'fast', 'slow')):
                tot[k] = tot.get(k, 0) + v
    print('totals:', json.dumps({k: tot[k] for k in sorted(tot) if tot[k]}, sort_keys=True))
    print('boundary hits (0 = not met in this run):', json.dumps({k: tot.get(k, 0) for k in sorted(tot)
                                                                   if k.startswith('hit_')}, sort_keys=True))
    print(f'reports {len(reps)}, workers expected {len(want)}, children listed {len(children)}')
    if bad:
        print('FAIL')
        for b in bad:
            print('  -', b)
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
