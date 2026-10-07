"""Hindsight teacher check and cost (2026-10-04): tok_sampler.hindsight with one fork + play + close per held move (the
way it was) against the bridge's fork_many, on recorded hurts of real episodes.

Per seed (room mode): reset, a template clone parked at the start (as tok_sampler's worker), then episodes: a lean clone
of the template reseeded as the worker reseeds, sticky random actions until the player dies, the room is clear or
--steps; every decision whose observation shows damage is a hurt. For up to --hurts hurts per episode both searches run
(order alternating): the results must be the same, depth by depth: the depth, the action under way, the move taken, the
lost[] vector (half hearts lost per held move, +100 for a death) and the restored observation's ROW record
(encode_row with a fresh EpisodeState, as the worker writes a teacher record). Reported: seconds per search of each
and the parts (fork of the restore clone, replay, the nine children, close), depths searched, decisions replayed.

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_teacher.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_sampler import TokSamplerConfig, hindsight


def record_episode(template, reseed, actions, fpd):
    """(applied actions, hurt decisions) of one episode played with step lines."""
    ep = template.fork(alarm=600, reseed=reseed, lean=True)
    dec = LeanDecoder()
    ep._send({"cmd": "obs"})
    o = read_lean(ep, dec)
    applied, hurts = [], []
    prev = o.damage_taken
    if not (o.dead or o.clear):
        for t, code in enumerate(actions):
            m, s, b, _ = split_code(code)
            applied.append((m, s, b))
            ep.step_line(m, s, b, 0, fpd)
            o = read_lean(ep, dec)
            if o.damage_taken - prev > 0:
                hurts.append(t + 1)
            prev = o.damage_taken
            if o.dead or o.clear:
                break
    ep.close()
    return applied, hurts


def summarize(found, hurt_at, limit):
    out = []
    for k, at, under_way, lost, taken in found:
        row = np.zeros(1, ROW)
        encode_row(at, EpisodeState(limit), row, hurt_at - 1 - k)
        out.append((k, tuple(int(v) for v in under_way), [float(v) for v in lost], int(taken), row.tobytes()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--hurts', type=int, default=3)
    p.add_argument('--port', type=int, default=34400)
    p.add_argument('--name', default='stpteach')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=stub if Path(stub).is_file() else '')
    cfg = TokSamplerConfig()
    fpd = cfg.frames_per_decision
    limit = 10 ** 6
    inst = Instance(args.name, args.port, gx, spec)
    rows = []
    tot = {'old': {}, 'new': {}}
    secs = {'old': 0.0, 'new': 0.0}
    searches, equal = 0, 0
    t_start = time.perf_counter()
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                for e in range(args.episodes):
                    reseed = int(rng.integers(1, 2 ** 31 - 1))
                    applied, hurts = record_episode(template, reseed, sticky_actions(rng, args.steps), fpd)
                    for j, hurt_at in enumerate(hurts[:args.hurts]):
                        res = {}
                        for name in (('old', 'new') if searches % 2 == 0 else ('new', 'old')):
                            timing = {}
                            t0 = time.perf_counter()
                            found = hindsight(template, reseed, applied, hurt_at, cfg, limit,
                                              fork_many=(name == 'new'), timing=timing)
                            secs[name] += time.perf_counter() - t0
                            for k, v in timing.items():
                                tot[name][k] = tot[name].get(k, 0) + v
                            res[name] = summarize(found, hurt_at, limit)
                            res[name + '_s'] = time.perf_counter() - t0
                        searches += 1
                        same = res['old'] == res['new']
                        equal += same
                        row = dict(seed=seed, episode=e, hurt_at=hurt_at, decisions=len(applied), same=same,
                                   depths=[r[0] for r in res['old']], lost_old=[r[2] for r in res['old']],
                                   lost_new=[r[2] for r in res['new']], old_s=round(res['old_s'], 4),
                                   new_s=round(res['new_s'], 4))
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            finally:
                template.close()
    finally:
        inst.close()
        summary = dict(group=args.group, searches=searches, equal=equal, load=os.getloadavg(),
                       seconds=round(time.perf_counter() - t_start, 1))
        for name in ('old', 'new'):
            n = max(searches, 1)
            summary[name] = dict(s_per_search=round(secs[name] / n, 4),
                                 **{k: round(v / n, 4) for k, v in tot[name].items()})
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
