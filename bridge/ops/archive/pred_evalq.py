"""Relay of the four arms' checkpoints: 8-card host -> this PC -> host 2's evaluation queue (host2_pred_evalq.sh). The two hosts do not reach each other; nothing is stored
on the PC, the bytes go through a pipe. A file appears on host 2 under its final name only when complete.

  python -X utf8 pred_evalq.py <case,case,...> [--every S]     keep relaying (new update-*.pt of each arm and its resumes <case>-c<n>; the final checkpoint as final.pt when runs/<case>/COMPLETE exists)
  python -X utf8 pred_evalq.py <case,case,...> --once          one pass
  ... --no-final                                               no final.pt (a test: 512 seeds are about two hours of host 2)
  python -X utf8 pred_evalq.py --table <case,case,...>         the results so far, per case and checkpoint
"""
import argparse
import json
import re
import subprocess
import time

import pred_host as H

HOST2 = 'rhythmo@10.19.131.132'
Q = 'isaac-abplus/pred/evalq'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST2]


def host2(cmd, timeout=120):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)


def copy(sftp, remote, case, name):
    """The remote file -> host 2's evalq/<case>/<name>, through a pipe."""
    p = subprocess.Popen(SSH + [f'mkdir -p {Q}/{case} && cat > {Q}/{case}/.{name}.part && mv {Q}/{case}/.{name}.part {Q}/{case}/{name}'], stdin=subprocess.PIPE)
    size = 0
    with sftp.open(remote, 'rb') as f:
        f.prefetch()
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            p.stdin.write(b)
            size += len(b)
    p.stdin.close()
    rc = p.wait()
    print(time.strftime('%F %T'), case, name, f'{size / 2**30:.2f} GiB', 'rc', rc, flush=True)
    return rc == 0


def one_pass(cases, final=True):
    c = H.client()
    s = c.open_sftp()
    runs = s.listdir(f'{H.ROOT}/runs')
    for case in cases:
        have = set(host2(f'ls {Q}/{case} 2>/dev/null').stdout.split())
        # (a resume's run goes on the same arm: its checkpoints under the arm's name, update numbers go on)
        for run_ in sorted(r for r in runs if r == case or re.fullmatch(re.escape(case) + r'-c\d+', r)):
            d = f'{H.ROOT}/runs/{run_}/checkpoints'
            try:
                names = s.listdir(d)
            except FileNotFoundError:
                continue
            for name in sorted(n for n in names if n.startswith('update-') and n.endswith('.pt')):
                if name not in have and name + '.failed' not in have:
                    # (a checkpoint being written: only one that has stopped growing)
                    a = s.stat(f'{d}/{name}').st_size
                    time.sleep(5)
                    if s.stat(f'{d}/{name}').st_size == a and copy(s, f'{d}/{name}', case, name):
                        have.add(name)
        if final and 'final.pt' not in have and 'final.pt.failed' not in have:
            try:
                with s.open(f'{H.ROOT}/runs/{case}/COMPLETE', 'r') as f:
                    done = json.loads(f.read())
            except FileNotFoundError:
                continue
            copy(s, done['path'], case, 'final.pt')
    c.close()


TABLE = r'''
import glob, json, os, sys
for case in sys.argv[1].split(','):
    for f in sorted(glob.glob(os.path.expanduser('~/isaac-abplus/pred/evalq/' + case + '/*.pt.eval/SUMMARY.json'))):
        r = json.load(open(f))
        g = r['groups']['run']
        print(case, os.path.basename(os.path.dirname(f))[:-8], 'update', r.get('update'), 'episodes', g.get('episodes'), 'errors', r.get('errors'),
              'floors', round(g.get('floors_cleared', float('nan')), 3), 'death', round(g.get('death', float('nan')), 3), 'timeout', round(g.get('timeout', float('nan')), 3),
              'round_trips', round(g.get('round_trips', float('nan')), 2), 'room_changes', round(g.get('room_changes', float('nan')), 1), 'stages', g.get('stage_reached'))
'''


def main():
    a = argparse.ArgumentParser()
    a.add_argument('cases', nargs='?')
    a.add_argument('--once', action='store_true')
    a.add_argument('--every', type=int, default=600)
    a.add_argument('--no-final', action='store_true')
    a.add_argument('--table')
    args = a.parse_args()
    if args.table:
        r = subprocess.run(SSH + [f'python3 - {args.table}'], input=TABLE, capture_output=True, text=True, timeout=120)
        print(r.stdout + r.stderr)
        return
    cases = args.cases.split(',')
    while True:
        try:
            one_pass(cases, final=not args.no_final)
        except Exception as e:   # (a connection's hiccup: the next pass)
            print(time.strftime('%F %T'), 'pass failed:', repr(e), flush=True)
        if args.once:
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
