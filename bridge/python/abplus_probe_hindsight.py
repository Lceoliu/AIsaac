"""C56 probe: after the player is hurt, was there a move that would have avoided it, and how early did it have to come?

The search teacher's question, asked with clones (A19). One instance; per seed a start state (template clone) and
episodes of a sticky random policy played from it. When a step hurts the player (step T-1, between the observations
T-1 and T), for k in --depths the episode is restored at observation d = T-1-k (a clone of the template with the
episode's reseed, the first d actions replayed with one `play`) and nine clones of that state each take the action
already under way for one step and then hold one of the nine moves, with the episode's own shots, until `margin`
decisions after the hurt. A move is safe if the clone's player lost no health. The earliest depth is tried last: the
answer of interest is the latest observation that still had a safe move.

Reported: how many hurts had a safe move at each depth, how many safe moves there were, whether the move the episode
actually took is among the dangerous ones (it must be: the clones share the episode's random numbers), whether the
restored state equals the episode's (player position and health, entity count), and the time one search takes.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_hindsight.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --episodes 3 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV, action_code
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.env import BridgeError


def play(clone, decoder, actions, repeat):
    """Apply the (move, shoot, bomb) actions back to back; the lean observation after the last one."""
    clone._send({"cmd": "play", "actions": [action_code(*a) for a in actions], "repeat": repeat, "stop_clear": False})
    return read_lean(clone, decoder)


def restore(template, reseed, actions, repeat):
    """A clone of the template in the state after `actions` of the episode with this reseed, and its observation."""
    clone = template.fork(reseed=reseed, lean=True, alarm=60)
    decoder = LeanDecoder()
    if actions:
        obs = play(clone, decoder, actions, repeat)
    else:
        clone._send({"cmd": "obs"})
        obs = read_lean(clone, decoder)
    return clone, obs


def probe_moves(state, obs, under_way, shots, repeat):
    """Half hearts lost by nine clones of `state` that take `under_way` for one step and then hold each move with the
    given shots (one per later step)."""
    lost = []
    for move in range(9):
        clone = state.fork(lean=True, alarm=60)
        try:
            end = play(clone, LeanDecoder(), [under_way] + [(move, s, 0) for s in shots], repeat)
            lost.append(float(end.damage_taken - obs.damage_taken) + (100.0 if end.dead else 0.0))
        finally:
            clone.close()
    return lost


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--episodes', type=int, default=3, help='episodes per start state')
    p.add_argument('--depths', default='2,4,8', help='decisions before the hurting step at which the episode is restored')
    p.add_argument('--margin', type=int, default=4, help='decisions after the hurting step that must stay unhurt too')
    p.add_argument('--max-hurts', type=int, default=3, help='hurts searched per episode')
    p.add_argument('--port', type=int, default=27890)
    p.add_argument('--name', default='fkhind')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    rep = cfg.frames_per_decision
    depths = sorted(int(d) for d in args.depths.split(','))
    inst = Instance(args.name, args.port, cfg, spec)
    rows = []
    try:
        for seed in parse_seeds(args.seeds):
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            rng = np.random.default_rng(seed)
            for ep in range(args.episodes):
                reseed = int(rng.integers(1, 2 ** 31 - 1))
                episode = template.fork(reseed=reseed, lean=True, alarm=300)
                decoder = LeanDecoder()
                episode._send({"cmd": "obs"})
                obs = read_lean(episode, decoder)
                if obs.clear:
                    episode.close()
                    break
                trace = [(float(obs.players[0]['x']), float(obs.players[0]['y']), float(obs.damage_taken),
                          len(obs.entities))]
                actions, hurts, move, shoot = [], [], 0, 0
                for t in range(400):
                    if rng.random() >= 0.9:
                        move, shoot = int(rng.integers(9)), int(rng.integers(5))
                    actions.append((move, shoot, 0))
                    obs = play(episode, decoder, actions[-1:], rep)
                    trace.append((float(obs.players[0]['x']), float(obs.players[0]['y']), float(obs.damage_taken),
                                  len(obs.entities)))
                    if trace[-1][2] > trace[-2][2] and not obs.dead:
                        hurts.append(len(actions))   # T: the observation that shows the damage
                    if obs.dead or obs.clear:
                        break
                episode.close()
                for T in hurts[:args.max_hurts]:
                    row = dict(seed=seed, episode=ep, T=T, depths={})
                    t0 = time.perf_counter()
                    for k in depths:
                        d = T - 1 - k
                        if d < 0:
                            continue
                        try:
                            state, at = restore(template, reseed, actions[:d], rep)
                        except (BridgeError, OSError, RuntimeError) as exc:
                            row['error'] = repr(exc)
                            break
                        same = (float(at.players[0]['x']), float(at.players[0]['y']), float(at.damage_taken),
                                len(at.entities)) == trace[d]
                        shots = [a[1] for a in actions[d + 1:T]] + [actions[T - 1][1]] * args.margin
                        try:
                            lost = probe_moves(state, at, actions[d], shots, rep)
                        finally:
                            state.close()
                        taken = actions[d + 1][0] if d + 1 < len(actions) else actions[d][0]
                        row['depths'][k] = dict(restored_equal=same, safe=[m for m in range(9) if lost[m] == 0],
                                                taken=taken, taken_lost=lost[taken])
                        if row['depths'][k]['safe']:
                            break
                    row['seconds'] = time.perf_counter() - t0
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            template.close()
    finally:
        inst.close()
        done = [r for r in rows if 'error' not in r]
        summary = dict(hurts=len(rows), errors=len(rows) - len(done), depths={})
        for k in depths:
            tried = [r['depths'][k] for r in done if k in r['depths']]
            summary['depths'][k] = dict(
                tried=len(tried), with_safe_move=sum(bool(x['safe']) for x in tried),
                safe_moves_mean=float(np.mean([len(x['safe']) for x in tried if x['safe']] or [0])),
                restored_equal=sum(x['restored_equal'] for x in tried))
        summary['avoidable'] = sum(any(x['safe'] for x in r['depths'].values()) for r in done)
        summary['seconds_per_search'] = float(np.mean([r['seconds'] for r in done] or [0]))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
