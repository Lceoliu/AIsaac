"""No-active start probe (B39, 2026-10-18; train_tok --no-active-prob, eval_tok --no-active 1).

Per character (--chars, default 0 Isaac, 3 Judas, 5 Eve) and seed: the root's floor reset (bridge.character = id), the
template S, and lean forks of it with the same reseed R, as tok_floor makes an episode (src.fork(reseed=R, lean=True)):
  with     : the unchanged start; its first ROW (has_active = player[21], active_ready = player[22], pitem[0], pinv[0]
             charge share, pinv[1] max charge / 12, pinv[6] held count / 16, held-item tokens);
  without  : tok_floor.no_active_start (tok_pickup.NO_ACTIVE_LUA) before the first observation; its first ROW and the
             Lua answer; then --steps decisions of a fixed sticky random action sequence (moves, shots, the item key
             pressed now and then), every record's tok_floor.row_digest;
  restore  : the episode parked after its first record as tok_floor does for a changed start (episode.fork(tag='snap',
             alarm=0)) and forked again with reseed None (tok_teacher2.restore / tok_branch.restore of the starting
             room): its first ROW, and the same action sequence replayed: the digests must equal `without`'s;
  old      : the restore point before B39 (S forked with the episode's reseed R): its first ROW (the D6 back) and the
             same actions' digests (expected to differ from `without`'s);
  build    : no-active, then a pedestal start build (tok_pickup.spawn_lua) as tok_floor orders them: pedestals in the
             first ROW (ent_id 5 / 100) next to has_active 0.
Expected (A29 numbers with the D6: has_active 1, max charge 0.5, one held token): without / restore has_active 0,
pitem[0] 0, pinv[0..1] 0, no held-item token; restore digests == without digests.

usage (bridge python dir, PYTHONPATH=., ABP_BRIDGE_LUA = the sandbox's bridge):
  python abplus_probe_no_active.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147490000:2 --out <dir>
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_floor import CHARACTER_NAMES, no_active_start, row_digest
from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row
from isaac_bridge.tok_pickup import parse_spawned, spawn_lua
from isaac_bridge.tok_sampler import STATS, ST, lean_step


class Track:
    """One clone's records through encode_row (run mode, items), as a worker encodes them."""

    def __init__(self, clone):
        self.c, self.dec = clone, LeanDecoder()
        self.st = EpisodeState(10 ** 6, floor=True, run=True, items=True)
        self.row = np.zeros(1, ROW)
        self.t = 0
        clone._send({'cmd': 'obs'})
        self.o = read_lean(clone, self.dec)
        encode_row(self.o, self.st, self.row, 0)
        self.first = fields(self.row)
        self.digests = [row_digest(self.row)]

    def play(self, actions):
        for a in actions:
            self.o = lean_step(self.c, self.dec, a, 4)
            self.t += 1
            done = encode_row(self.o, self.st, self.row, self.t)
            self.digests.append(row_digest(self.row))
            if done or self.o.dead:
                break
        return self


def fields(row):
    r = row[0]
    n = int(r['n_ent'])
    ids = r['ent_id'][:n]
    return dict(has_active=float(r['player'][21]), active_ready=float(r['player'][22]), pitem0=int(r['pitem'][0]),
                charge_share=round(float(r['pinv'][0]), 4), max_charge=round(float(r['pinv'][1]), 4),
                held=round(float(r['pinv'][6]) * 16, 2), tokens=[[int(i), int(c)] for i, c in r['inv'] if i > 0],
                pchar=int(r['pchar']), pedestals=int(((ids[:, 0] == 5) & (ids[:, 1] == 100)).sum()))


def actions_of(rng, n):
    out, move, shoot = [], 0, 0
    for t in range(n):
        if t % 6 == 0:
            move, shoot = int(rng.integers(0, 9)), int(rng.integers(0, 5))
        item = 1 if rng.random() < 0.05 else 0
        out.append((move, shoot, 0, item, 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147490000:2')
    p.add_argument('--chars', default='0,3,5')
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--build', default='1,2', help='pedestal build of the build check')
    p.add_argument('--port', type=int, default=49800)
    p.add_argument('--name', default='hb39p')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    build = [int(v) for v in args.build.split(',') if v.strip()]
    inst = Instance(args.name, args.port, gx, spec)
    report = []
    try:
        for char in [int(c) for c in args.chars.split(',') if c.strip()]:
            for seed in range(first, first + count):
                t0 = time.perf_counter()
                b = inst.env.bridge
                b.reset_mode = 'floor'
                b.character = char
                inst.reset(seed)
                S = b.fork(tag='template', alarm=0)
                R = seed % (2 ** 31 - 1) + 1   # (eval_tok's reseed of a seed)
                acts = actions_of(np.random.default_rng(seed), args.steps)
                rec = dict(char=char, name=CHARACTER_NAMES.get(char), seed=seed, reseed=R)
                clones = []
                try:
                    W = S.fork(reseed=R, lean=True, alarm=600)
                    clones.append(W)
                    rec['with'] = Track(W).first
                    E = S.fork(reseed=R, lean=True, alarm=600)
                    clones.append(E)
                    mine = np.zeros(len(STATS))
                    rec['removed'] = no_active_start(E, mine)
                    rec['counters'] = dict(no_actives=float(mine[ST['no_actives']]),
                                           no_active_removed=float(mine[ST['no_active_removed']]))
                    te = Track(E)
                    rec['without'] = te.first
                    P = E.fork(tag='snap', alarm=0)   # (tok_floor: the starting room's restore point of a changed start)
                    clones.append(P)
                    te.play(acts)
                    Q = P.fork(reseed=None, lean=True, alarm=600)
                    clones.append(Q)
                    tq = Track(Q).play(acts)
                    rec['restore'] = tq.first
                    O = S.fork(reseed=R, lean=True, alarm=600)
                    clones.append(O)
                    to = Track(O).play(acts)
                    rec['old'] = to.first
                    rec['records'] = len(te.digests)
                    rec['restore_equal'] = te.digests == tq.digests
                    rec['old_equal'] = te.digests == to.digests
                    rec['old_first_diff'] = next((k for k, (x, y) in enumerate(zip(te.digests, to.digests)) if x != y),
                                                 None)
                    B = S.fork(reseed=R, lean=True, alarm=600)
                    clones.append(B)
                    no_active_start(B, mine)
                    spawned = parse_spawned(B.lua(spawn_lua(build)))
                    rec['build'] = dict(spawned=[i for i, _, _ in spawned], first=Track(B).first)
                except Exception as e:
                    rec['error'] = f'{type(e).__name__}: {e}'
                finally:
                    for c in clones:
                        try:
                            c.close()
                        except OSError:
                            pass
                    S.close()
                w, wo = rec.get('with', {}), rec.get('without', {})
                rec['ok'] = bool(
                    'error' not in rec and wo.get('has_active') == 0 and wo.get('active_ready') == 0
                    and wo.get('pitem0') == 0 and wo.get('max_charge') == 0 and wo.get('charge_share') == 0
                    and rec['restore'] == wo and rec['restore_equal']
                    and (w.get('has_active') == 0 or (not any(t[0] == w['pitem0'] for t in wo['tokens'])
                                                      and len(wo['tokens']) == len(w['tokens']) - 1))
                    and rec['build']['first']['has_active'] == 0
                    and rec['build']['first']['pedestals'] == len(rec['build']['spawned']) > 0)
                rec['seconds'] = round(time.perf_counter() - t0, 1)
                report.append(rec)
                print(json.dumps(rec), flush=True)
    finally:
        inst.kill()
    summary = dict(cases=len(report), ok=sum(r['ok'] for r in report),
                   restore_equal=sum(bool(r.get('restore_equal')) for r in report),
                   old_equal=sum(bool(r.get('old_equal')) for r in report),
                   errors=sum('error' in r for r in report))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'cases.json').write_text(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
