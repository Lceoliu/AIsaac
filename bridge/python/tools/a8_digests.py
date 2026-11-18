"""A8 across two trees (the coordinator, 2026-11-18): abplus_probe_play.py (8 seeds, the gate's A8) run inside one tree
with every hidden-state digest it takes recorded in order (<out>/digests.jsonl: seed, call number, blake2b of the
digest text), so two trees' runs compare digest for digest. Run by the CPU runner:
  PROBE=<this file> cpurun.sh <tree> <name> --group normal --seeds range:2147500000:8 --steps 300
compare on the host: python3 a8_digests.py compare <out A> <out B>
"""
import hashlib
import json
import os
import sys


def run(argv):
    sys.path.insert(0, os.getcwd())   # (the tree's python dir)
    for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE'):
        os.environ.pop(k, None)
    out = argv[argv.index('--out') + 1]
    os.makedirs(out, exist_ok=True)
    import isaac_bridge.abplus_goexplore as G
    f = open(os.path.join(out, 'digests.jsonl'), 'w')
    orig = G.Instance.digest_text
    n = [0]

    def digest_text(self, *a, **kw):
        text = orig(self, *a, **kw)
        n[0] += 1
        f.write(json.dumps(dict(call=n[0], d=hashlib.blake2b(text.encode('utf8'), digest_size=16).hexdigest())) + '\n')
        f.flush()
        return text
    G.Instance.digest_text = digest_text
    import abplus_probe_play
    sys.argv = ['abplus_probe_play.py'] + argv
    try:
        abplus_probe_play.main()
    finally:
        f.close()


def compare(a, b):
    la = [json.loads(x) for x in open(os.path.join(a, 'digests.jsonl'))]
    lb = [json.loads(x) for x in open(os.path.join(b, 'digests.jsonl'))]
    eq = sum(x['d'] == y['d'] for x, y in zip(la, lb))
    sa = json.load(open(os.path.join(a, 'summary.json')))
    sb = json.load(open(os.path.join(b, 'summary.json')))
    ok = len(la) == len(lb) > 0 and eq == len(la)
    print(json.dumps(dict(digests=[len(la), len(lb)], equal=eq, ok=ok,
                          summary_a={k: sa.get(k) for k in ('seeds', 'ok')},
                          summary_b={k: sb.get(k) for k in ('seeds', 'ok')},
                          first_diff=next((i for i, (x, y) in enumerate(zip(la, lb)) if x['d'] != y['d']), None))))
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1:])
