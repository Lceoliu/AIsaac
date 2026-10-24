"""Replays of a goexplore_abplus.py run's best trajectories (EXPERIMENTS.md A8), for abplus_replay_view.py.

Each room's best cell (the verified win when there is one) is played again step by step on one instance with the run's
settings, recording the raw observation after the reset and after every step, and its final state is checked against the
stored digest once more. Writes an abplus_eval.py-style directory: meta.json, results.jsonl and
replays/seed-<seed>.jsonl.gz; then make the page with
  python abplus_replay_view.py <out>.html <out dir>=Go-Explore --title "Go-Explore 最好轨迹"

usage (from the bridge's python dir, PYTHONPATH=.):
  python goexplore_replays.py --run ../runs/<run> --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --out ../runs/<run>-replays
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus_goexplore import GxConfig, Instance, unpack


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--tasks')
    src.add_argument('--groups-file')
    p.add_argument('--group', default='normal')
    p.add_argument('--seconds', type=float, default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--al-stopped', action=argparse.BooleanOptionalAction, default=None,
                   help="override the run's ABP_AL_STOPPED (--no-al-stopped: replay as the training environment is)")
    p.add_argument('--name', default='gxrp')
    p.add_argument('--port', type=int, default=27535)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    (out / 'replays').mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'config.json').read_text().splitlines()[-1])
    gx = config['gx']
    cfg = GxConfig(**{**gx, 'bridge_lua': default_bridge_lua(), 'preload': default_preload()})
    if args.al_stopped is not None:
        cfg.al_stopped = args.al_stopped
    spec = load_spec(args)
    rooms = [json.loads(line) for line in open(run / 'results.jsonl')]
    rooms = [r for r in rooms if 'best_cell' in r]
    if args.limit:
        rooms = rooms[:args.limit]
    (out / 'meta.json').write_text(json.dumps(dict(
        checkpoint=f'Go-Explore {run.name}', updates=None, deterministic=True, sample_seed=None,
        frames_per_decision=cfg.frames_per_decision, episode_seconds=spec['seconds'], engine='abplus-1.06',
        tasks_file=spec['tasks'], al_stopped=cfg.al_stopped, source=str(run.resolve()),
        policy_note='Go-Explore 找到的最好轨迹，按记录的动作开环重放' + ('（音源状态固定）' if cfg.al_stopped else
                                                                '（音源状态未固定，同训练环境）')), indent=1))
    inst = Instance(args.name, args.port, cfg, spec)
    try:
        with open(out / 'results.jsonl', 'w') as results:
            for r in rooms:
                seed = int(r['seed'])
                detail = json.loads((run / 'rooms' / f'{seed}.json').read_text())
                best = detail['best']
                actions = [unpack(b) for b in bytes.fromhex(best['actions'])]
                inst.reset(seed)
                path = out / 'replays' / f'seed-{seed}.jsonl.gz'
                outcome, frames = 'running', 0
                with gzip.open(path, 'wt', encoding='utf8') as w:
                    w.write(json.dumps({'metadata': {'seed': seed, 'format': 'abplus-raw-obs-v1',
                                                     'goexplore_key': best['key']}}) + '\n')
                    w.write(json.dumps({'action': None, 'obs': inst.env.raw_obs}, separators=(',', ':')) + '\n')
                    for a in actions:
                        _, _, terminated, truncated, info = inst.env.step(np.asarray(a))
                        w.write(json.dumps({'action': list(a), 'obs': inst.env.raw_obs}, separators=(',', ':')) + '\n')
                        outcome, frames = info['outcome'], int(info['elapsed_frames'])
                        if terminated or truncated:
                            break
                same = inst.digest() == best['digest']
                room = r.get('room') or {}
                rec = dict(seed=seed, task=room.get('task', 'normal'), outcome=outcome, layout=room.get('variant'),
                           frames=frames, start_bombs=cfg.bombs, steps=len(actions), digest_matches=same,
                           best_verified=r.get('best_verified'), hurt=best.get('hurt'), health=best.get('health'))
                results.write(json.dumps(rec) + '\n')
                results.flush()
                print(json.dumps(rec), flush=True)
    finally:
        inst.close()


if __name__ == '__main__':
    main()
