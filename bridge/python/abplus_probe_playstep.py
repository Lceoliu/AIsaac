"""A8 follow-up: a trajectory whose end state differs between one play and step by step (abplus_probe_gxdiverge.py
variants). In one fresh instance: the digest after every step of a step-by-step run; then play(actions[:k]) from a reset
for the k found by bisection, the first k whose end state differs from the step run's after k steps. Prints the lines only
one side has at that k and the diagnostic lines, and checks that play(actions[:k-1]) still matched.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_playstep.py --run ../runs/<run> --seed S --key 5,0,4,0,6,3 \
      --groups-file ../abplus/catalog/scaling2_groups.json --group normal --out <dir>
"""
import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, load_spec
from isaac_bridge.abplus_goexplore import GxConfig, Instance, unpack


def split(text):
    lines = text.split('\n')
    return [l for l in lines if not l.startswith('D|')], {l.split('|')[1]: l for l in lines if l.startswith('D|')}


def order_test(args, cfg, spec, actions, out):
    """Two fresh instances: A plays the trajectory by play, then by step, alternating (--repeats each); B starts with
    step. Prints the end-state hash of every run with the player's entity Index: a mode effect makes the hashes follow
    the mode, a history effect the run order."""
    rows = []
    for tag, first in (('A', 'play'), ('B', 'step')):
        inst = Instance(f'{args.name}{tag.lower()}', args.port + (tag == 'B'), cfg, spec)
        try:
            modes = [first, 'step' if first == 'play' else 'play'] * args.repeats
            for i, mode in enumerate(modes):
                inst.reset(args.seed)
                if args.remove_effects:
                    # remove these effect variants (e.g. 68, the Wall Bug decoration) right after the reset
                    variants = '{' + args.remove_effects + '}'
                    inst.env.bridge.lua(f"local n = 0; for _, v in ipairs({variants}) do for _, e in "
                                        f"ipairs(Isaac.FindByType(1000, v, -1, false, false)) do e:Remove(); n = n + 1 "
                                        f"end end; return n")
                if mode == 'play':
                    inst.env.play(actions)
                else:
                    for a in actions:
                        if inst.env.finished:
                            break
                        if args.step_sleep:
                            time.sleep(args.step_sleep)
                        inst.env.step(np.asarray(a))
                text = inst.digest_text(diag=True)
                state, diag = split(text)
                h = hashlib.blake2b('\n'.join(state).encode('utf8'), digest_size=8).hexdigest()
                row = dict(instance=tag, run=i, mode=mode, state=h, frames=inst.env.elapsed_frames,
                           player_index=diag.get('index', '').split('|')[2].split(' ')[0] if 'index' in diag else None)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            inst.close()
    (out / f'order-{args.seed}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    by_mode = {m: sorted({r['state'] for r in rows if r['mode'] == m}) for m in ('play', 'step')}
    print(json.dumps(dict(states_by_mode=by_mode)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--key', required=True, help='comma-separated cell key')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tasks')
    src.add_argument('--groups-file')
    p.add_argument('--group', default='normal')
    p.add_argument('--seconds', type=float, default=None)
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--name', default='gxps')
    p.add_argument('--port', type=int, default=27585)
    p.add_argument('--out', required=True)
    p.add_argument('--order-test', action='store_true',
                   help='instead: fresh instances running play then step, and step then play (mode or history?)')
    p.add_argument('--repeats', type=int, default=3, help='with --order-test: runs of each mode per instance')
    p.add_argument('--remove-effects', default='', help='with --order-test: comma-separated effect variants removed '
                                                        'right after each reset (e.g. 68,21: the decorative bugs)')
    p.add_argument('--mode', default='exact', help='instance mode (exact, skip: the render path skipped)')
    p.add_argument('--step-sleep', type=float, default=0.0, help='with --order-test: seconds slept before each step')
    p.add_argument('--fixed-time', type=int, default=None, help='ABP_FIXED_TIME for the instances (time() fixed)')
    args = p.parse_args()
    if args.fixed_time is not None:
        os.environ['ABP_FIXED_TIME'] = str(args.fixed_time)   # launch_abplus passes the environment on
    key = [int(v) if v.lstrip('-').isdigit() else v for v in args.key.split(',')]
    actions = None
    with gzip.open(Path(args.run) / 'rooms' / f'{args.seed}.cells.json.gz', 'rt', encoding='utf8') as f:
        for line in f:
            c = json.loads(line)
            if c['key'] == key:
                actions = [unpack(b) for b in bytes.fromhex(c['actions'])]
                break
    if actions is None:
        raise SystemExit('cell not found')
    spec = load_spec(args)
    cfg = GxConfig(frames_per_decision=args.frames_per_decision, bridge_lua=default_bridge_lua(), mode=args.mode)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.order_test:
        order_test(args, cfg, spec, actions, out)
        return
    inst = Instance(args.name, args.port, cfg, spec)
    try:
        inst.reset(args.seed)
        steps = [inst.digest_text(diag=True)]
        for a in actions:
            inst.env.step(np.asarray(a))
            steps.append(inst.digest_text(diag=True))

        def played(k):
            inst.reset(args.seed)
            if k:
                inst.env.play(actions[:k])
            return inst.digest_text(diag=True)

        def same(k):
            return split(played(k))[0] == split(steps[k])[0]

        n = len(actions)
        if same(n):
            print(json.dumps(dict(result='play equals step at the end', steps=n)), flush=True)
            return
        lo, hi = 0, n   # same(lo) (a reset is a reset), not same(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if same(mid):
                lo = mid
            else:
                hi = mid
        dp, ds = played(hi), steps[hi]
        sp, dgp = split(dp)
        ss, dgs = split(ds)
        P, S = set(sp), set(ss)
        rec = dict(seed=args.seed, key=key, steps=n, first_k=hi, previous_same=same(hi - 1),
                   action=list(actions[hi - 1]), prev_action=list(actions[hi - 2]) if hi >= 2 else None,
                   only_play=[l[:500] for l in sp if l not in S][:8], only_step=[l[:500] for l in ss if l not in P][:8],
                   diag_play={k: v[:400] for k, v in dgp.items()}, diag_step={k: v[:400] for k, v in dgs.items()})
        (out / f'playstep-{args.seed}.json').write_text(json.dumps(dict(rec, play_digest=dp, step_digest=ds,
                                                                         step_before=steps[hi - 1])))
        print(json.dumps(rec, indent=1)[:6000], flush=True)
    finally:
        inst.close()


if __name__ == '__main__':
    main()
