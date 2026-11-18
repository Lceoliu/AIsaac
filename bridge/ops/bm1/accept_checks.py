"""accept_g.sh's pass conditions (2026-11-18, Codex round 2: a check passes on the counts, not on a probe's exit code
or 'tested > 0' alone). Prints 1 (pass) or 0, exits 0 either way (accept_g.sh reads the digit).
  python3 accept_checks.py frame <summary.json>      every tested seed with equal payloads and equal digests
  python3 accept_checks.py r2 <clone-stats.txt>      native visible checks > 0 and 0 mismatches (clone_stats.py)"""
import json
import re
import sys


def frame_ok(summary):
    t = int(summary.get('tested') or 0)
    return t > 0 and summary.get('payload_equal') == t and summary.get('digests_equal') == t


def r2_ok(text):
    m = re.search(r'native visible checks (\d+) mismatches (\d+)', text)
    return bool(m) and int(m.group(1)) > 0 and int(m.group(2)) == 0


def main(argv):
    try:
        if argv[1] == 'frame':
            ok = frame_ok(json.load(open(argv[2])))
        elif argv[1] == 'r2':
            ok = r2_ok(open(argv[2]).read())
        else:
            ok = False
    except (OSError, ValueError, IndexError, KeyError):
        ok = False
    print(int(ok))


if __name__ == '__main__':
    main(sys.argv)
