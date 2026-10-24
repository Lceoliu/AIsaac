"""A8 follow-up: where do Go-Explore returns that failed their digest check diverge?

Takes the cells of a goexplore_abplus.py run whose returns hit a digest mismatch (events.jsonl digest_retry), and replays
each cell's trajectory step by step in two instances with different histories: X freshly started, Y after --warmup
random episodes in other rooms. After every step both take the full digest (abplus_goexplore.DIGEST_LUA with its
diagnostic lines). Reports per cell the first step where the state lines differ, the lines only one side has, and
whether the diagnostic lines (entity order and Index) differed before that; also which side matches the cell's stored
digest at the end.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_gxdiverge.py --run ../runs/<run> --groups-file ../abplus/catalog/scaling2_groups.json \
      --group normal --count 6 --out <dir>
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus_goexplore import GxConfig, Instance, unpack


def cells_with_failures(run, count, only_same=False):
    """[(seed, key, actions, stored digest, event counts)] of the cells with the most digest mismatches (only_same: those
    whose retries reproduced the mismatch at least once)."""
    tally = {}
    for line in open(Path(run) / 'events.jsonl'):
        e = json.loads(line)
        if e.get('event') == 'digest_retry':
            k = (e['seed'], tuple(e['key']))
            tally.setdefault(k, {'match': 0, 'same': 0, 'other': 0})
            tally[k]['match' if e['match'] else 'same' if e['same'] else 'other'] += 1
    picked = sorted(tally.items(), key=lambda kv: -(kv[1]['same'] + kv[1]['other'] + kv[1]['match']))
    if only_same:
        picked = [kv for kv in picked if kv[1]['same']]
    out = []
    for (seed, key), counts in picked:
        path = Path(run) / 'rooms' / f'{seed}.cells.json.gz'
        if not path.exists():
            continue
        with gzip.open(path, 'rt', encoding='utf8') as f:
            for line in f:
                c = json.loads(line)
                if tuple(c['key']) == key:
                    out.append((seed, key, [unpack(b) for b in bytes.fromhex(c['actions'])], c['digest'], counts))
                    break
        if len(out) >= count:
            break
    return out


def split(text):
    lines = text.split('\n')
    return [l for l in lines if not l.startswith('D|')], {l.split('|')[1]: l for l in lines if l.startswith('D|')}


def run_trajectory(inst, seed, actions):
    inst.reset(seed)
    digests = [inst.digest_text(diag=True)]
    for a in actions:
        _, _, terminated, truncated, _ = inst.env.step(np.asarray(a))
        digests.append(inst.digest_text(diag=True))
        if terminated or truncated:
            break
    return digests


def variants(inst, seed, actions):
    """Hashes of the end state reached four ways in one instance: step with a digest after every step, step with one
    digest at the end, one play, half play and half step (the return and the exploration paths of Go-Explore)."""
    h = lambda: hashlib.blake2b(inst.digest_text().encode('utf8'), digest_size=16).hexdigest()
    out = {}
    d = run_trajectory(inst, seed, actions)
    out['step_each'] = hashlib.blake2b('\n'.join(split(d[-1])[0]).encode('utf8'), digest_size=16).hexdigest()
    inst.reset(seed)
    for a in actions:
        inst.env.step(np.asarray(a))
    out['step_end'] = h()
    inst.reset(seed)
    inst.env.play(actions)
    out['play'] = h()
    inst.reset(seed)
    k = len(actions) // 2
    if k:
        inst.env.play(actions[:k])
    for a in actions[k:]:
        inst.env.step(np.asarray(a))
    out['play_step'] = h()
    return out


def warm_up_same_room(inst, run, seed, skip_key, count):
    """Y's history as in Go-Explore: returns (play) to `count` other cells of the same room."""
    done = 0
    with gzip.open(Path(run) / 'rooms' / f'{seed}.cells.json.gz', 'rt', encoding='utf8') as f:
        for line in f:
            c = json.loads(line)
            if tuple(c['key']) == skip_key or c['outcome'] != 'running' or not c['actions']:
                continue
            inst.reset(seed)
            inst.env.play([unpack(b) for b in bytes.fromhex(c['actions'])])
            done += 1
            if done >= count:
                break
    return done


def warm_up(inst, seeds, steps, rng):
    for seed in seeds:
        inst.reset(seed)
        for _ in range(steps):
            _, _, terminated, truncated, _ = inst.env.step(np.asarray([int(rng.integers(45)), 0, 0]))
            if terminated or truncated:
                break


def compare(dx, dy, stored):
    first, diag_first = None, None
    for t, (a, b) in enumerate(zip(dx, dy)):
        sa, da = split(a)
        sb, db = split(b)
        if diag_first is None and (da.get('order') != db.get('order') or da.get('index') != db.get('index')):
            diag_first = t
        if sa != sb:
            A, B = set(sa), set(sb)
            first = dict(step=t, only_x=[l[:400] for l in sa if l not in B][:4],
                         only_y=[l[:400] for l in sb if l not in A][:4],
                         index_x=da.get('index', '')[:300], index_y=db.get('index', '')[:300])
            break
    h = lambda text: hashlib.blake2b('\n'.join(split(text)[0]).encode('utf8'), digest_size=16).hexdigest()
    return dict(digests=(len(dx), len(dy)), first_state_diff=first, first_diag_diff=diag_first,
                x_matches_stored=h(dx[-1]) == stored, y_matches_stored=h(dy[-1]) == stored)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tasks')
    src.add_argument('--groups-file')
    p.add_argument('--group', default='normal')
    p.add_argument('--seconds', type=float, default=None)
    p.add_argument('--count', type=int, default=6)
    p.add_argument('--variants', action='store_true',
                   help='also reach each end state by step (digest each step / at the end), play, and play + step; '
                        'variants: {way: (matches the stored digest, matches step_each)}')
    p.add_argument('--only-same', action='store_true', help='only cells whose retries reproduced the mismatch')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--warmup-start', type=int, default=2147700000)
    p.add_argument('--warmup-same-room', type=int, default=0,
                   help='instead of random episodes: Y first returns to this many other cells of the same room')
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--name', default='gxdv')
    p.add_argument('--port', type=int, default=27580)
    p.add_argument('--out', required=True)
    p.add_argument('--no-al-stopped', action='store_true', help='OpenAL source states as the audio thread reports them (A8)')
    args = p.parse_args()
    spec = load_spec(args)
    cfg = GxConfig(frames_per_decision=args.frames_per_decision, bridge_lua=default_bridge_lua(),
                   preload=default_preload(), al_stopped=not args.no_al_stopped)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = cells_with_failures(args.run, args.count, args.only_same)
    print(json.dumps(dict(cells=[(s, list(k), len(a), c) for s, k, a, _, c in cells])), flush=True)
    x = Instance(args.name + 'x', args.port, cfg, spec)
    y = Instance(args.name + 'y', args.port + 1, cfg, spec)
    try:
        if not args.warmup_same_room:
            warm_up(y, range(args.warmup_start, args.warmup_start + args.warmup), args.warmup_steps,
                    np.random.default_rng(0))
        for seed, key, actions, stored, counts in cells:
            if args.warmup_same_room:
                warm_up_same_room(y, args.run, seed, key, args.warmup_same_room)
            dx = run_trajectory(x, seed, actions)
            dy = run_trajectory(y, seed, actions)
            rec = dict(seed=seed, key=list(key), steps=len(actions), mismatches=counts, **compare(dx, dy, stored))
            # and once more in X after the Y run: is X stable against itself?
            dx2 = run_trajectory(x, seed, actions)
            rec['x_again_same'] = [split(a)[0] for a in dx2] == [split(a)[0] for a in dx]
            if args.variants:
                v = variants(x, seed, actions)
                rec['variants'] = {k: (h == stored, h == v['step_each']) for k, h in v.items()}
            with open(out / 'diverge.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            with gzip.open(out / f'{seed}-{"_".join(map(str, key))}.digests.gz', 'wt', encoding='utf8') as f:
                f.write(json.dumps(dict(x=dx, y=dy, x2=dx2)))
            brief = {k: v for k, v in rec.items() if k != 'first_state_diff'}
            brief['first_step'] = (rec['first_state_diff'] or {}).get('step')
            print(json.dumps(brief), flush=True)
            if rec['first_state_diff']:
                print(json.dumps(rec['first_state_diff'])[:1500], flush=True)
    finally:
        x.close()
        y.close()


if __name__ == '__main__':
    main()
