"""Teacher v2 probe (2026-10-10, rl/docs/SCALING_THESIS.md S5b, isaac_bridge/tok_teacher2.py): the plumbing of the
branch search without the actor.

Per seed: a floor start (the bridge's reset_mode 'floor', as tok_floor's worker), a template clone parked there, then
an episode of sticky random actions (always shooting) on a lean clone reseeded as the worker reseeds, until the player
dies or --max-decisions. The episode is recorded as the worker records it: each record (encode_row with the worker's
EpisodeState), its log (tok_teacher2.V2Log: reward fields, stall counter; the stand-in's value 0), the applied actions,
a room-entry snapshot parked at every room change (the worker's restore points: reseed None, offset = that record), and
its points: hurts, the death, and --random random records (as the worker's random points).
Per point (all deaths, up to --hurts hurts, the random ones):
  replay   a branch whose "policy" answers the episode's own next action (intent: none), from the restore of the first
           depth: its records must equal the episode's records (every policy input; the reward fields from the second
           record on), its actions under way the episode's, and its score the taken score (both with value 0).
  search   tok_teacher2.search_point with a stand-in policy (sticky random shooting, random moves where the intent does
           not set them, value 0): per depth the taken score, the candidates' scores, the best, the gain, the
           replicate's score and spread, improving; for an improving point the imitation records (records_of): count,
           weights, the first record's action under way = the episode's, a held move's actions = its move for h steps,
           the return-to-go recursion. Seconds per point (the stand-in costs nothing: wall ~ the game's work).

usage (bridge python dir, PYTHONPATH=.):
  python abplus_probe_teacher_v2.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:6 \
      --out <dir> [--port 44600] [--name tv] [--mode run]
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
from isaac_bridge.tok_branch import parse_reward
from isaac_bridge.tok_floor import Parked
from isaac_bridge.tok_obs import EV_EXIT, ROW, EpisodeState, encode_row, fast_row_function
from isaac_bridge.tok_sampler import TokSamplerConfig, lean_step, row_library
from isaac_bridge.tok_teacher2 import (KIND_FATAL, KIND_HURT, KIND_RANDOM, VI, Context, Intent, V2Log, V2Point, five,
                                       play_branch, records_of, restore, search_point, taken_score)

FLOOR_REWARD = 'damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002'
INPUTS = ('player', 'patch', 'grid', 'map', 'n_ent', 'n_doors', 't', 'bombs', 'stage', 'pcharge', 'pchar', 'inv',
          'pitem', 'pinv', 'done')
REWARD_FIELDS = ('hurt', 'damage', 'events')


def row_diff(a, b, first):
    """The fields in which records a and b (ROW scalars) differ, as the policy sees them (entity / door lists cut to
    their counts) and, unless first, the reward fields."""
    out = [k for k in INPUTS if not np.array_equal(a[k], b[k])]
    ne, nd = int(a['n_ent']), int(a['n_doors'])
    for k in ('ent', 'ent_id', 'ent_item'):
        if not np.array_equal(a[k][:ne], b[k][:ne]):
            out.append(k)
    if not np.array_equal(a['doors'][:nd], b['doors'][:nd]):
        out.append('doors')
    if not first:
        out += [k for k in REWARD_FIELDS if not np.array_equal(a[k], b[k])]
    return out


class EpisodePolicy:
    """Answers record t with the episode's own action applied at step t + 1 (its decision there); checks that the
    action under way the branch sends is the episode's applied[t]."""

    def __init__(self, applied):
        self.applied, self.t, self.under_bad = applied, 0, 0

    def request(self, row, under_way):
        self.t = int(row['t'][0])
        if self.t < len(self.applied) and tuple(under_way) != five(self.applied[self.t]):
            self.under_bad += 1

    def answer(self):
        t = self.t + 1
        return (five(self.applied[t]) if t < len(self.applied) else (0, 0, 0, 0, 0)), 0.0, 0.0


class StandIn:
    """The actor's stand-in: random moves, sticky random shooting (kept with probability 0.85), no bombs; value 0."""

    def __init__(self, seed):
        self.rng, self.shoot = np.random.default_rng(seed), 1
        self.calls = 0

    def request(self, row, under_way):
        self.calls += 1

    def answer(self):
        if self.rng.random() > 0.85:
            self.shoot = int(self.rng.integers(1, 5))
        return (int(self.rng.integers(9)), self.shoot, 0, 0, 0), 0.0, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:6')
    p.add_argument('--mode', default='floor', choices=('floor', 'run'))
    p.add_argument('--max-decisions', type=int, default=900)
    p.add_argument('--seconds', type=float, default=20.0, help='teacher_v2_seconds')
    p.add_argument('--hurts', type=int, default=2, help='hurt points per episode at most (deaths always)')
    p.add_argument('--random', type=int, default=1, help='random points per episode')
    p.add_argument('--death-depths', default='2,4,8,16,32')
    p.add_argument('--port', type=int, default=44600)
    p.add_argument('--name', default='tv')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    run = args.mode == 'run'
    cfg = TokSamplerConfig(workers=1, specs=[spec], assign=[0], frames_per_decision=4, mode=args.mode, teacher=True,
                           teacher_v2=True, teacher_v2_seconds=args.seconds, branch_reward=FLOOR_REWARD,
                           teacher_death_depths=tuple(int(v) for v in args.death_depths.split(',') if v.strip()),
                           preload=default_preload())
    limit = int(round((cfg.run_seconds if run else cfg.floor_seconds) * 30 / cfg.frames_per_decision))
    stall = int(round(cfg.floor_stall_seconds * 30 / cfg.frames_per_decision))
    fast_fn = fast_row_function(row_library(cfg))
    ctx = Context(cfg, limit, stall, run, fast_fn, 0, np.zeros(1, ROW), parse_reward(FLOOR_REWARD))
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(29)
    report = []
    try:
        for seed in range(first, first + count):
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            template = Parked(inst.env.bridge.fork(tag='template', alarm=0), seed=seed, visited=(), stage0=None,
                              uses=0)
            reseed = int(rng.integers(1, 2 ** 31 - 1))
            E = template.fork(lean=True, alarm=900, reseed=reseed)
            dec = LeanDecoder()
            E._send({'cmd': 'obs'})
            obs = read_lean(E, dec)
            st = EpisodeState(limit, floor=True, stall=stall, run=run, items=False)
            st.visited, st.stage0 = set(), None
            row = np.zeros(1, ROW)
            rows, applied, log, pts = [], [], V2Log(), []
            base_of, changed_at = [], []   # per record: (restore point, its reseed, its record), a room change there
            snaps = []                     # the room-entry restore points (released after the episode)
            base, base_reseed, base_t, room_now = template, reseed, 0, None
            t, move, shoot, done = 0, 0, 1, 0
            while True:
                done = encode_row(obs, st, row, t)
                row['episode'], row['seed'], row['group'], row['first'] = 1, seed, 0, t == 0
                rows.append(row[0].copy())
                log.add(row, done, st.progress_t)
                room_idx = int(obs.room[4])
                changed = t > 0 and (room_idx != room_now or bool(row['events'][0] & EV_EXIT))
                base_of.append((base, base_reseed, base_t))   # (before the rebase, as the worker's hurts)
                changed_at.append(changed)
                if row['hurt'][0] > 0 or done == 2:
                    pts.append((t, KIND_FATAL if done == 2 else KIND_HURT))
                if changed and not done:   # the worker's restore point: the episode as it enters the room, parked
                    snap = Parked(E.fork(tag='snap', alarm=0), seed=seed, visited=tuple(st.visited),
                                  stage0=st.stage0, uses=0, boss=False, stage=st.stage0)
                    snaps.append(snap)
                    base, base_reseed, base_t = snap, None, t
                room_now = room_idx
                if done or t >= args.max_decisions:
                    break
                log.value.append(0.0)
                if t % 6 == 0:
                    move, shoot = int(rng.integers(0, 9)), int(rng.integers(1, 5))
                action = (move, shoot, 0)
                applied.append(action)
                obs = lean_step(E, dec, action, 4)
                t += 1
            E.close()
            ep = dict(seed=seed, decisions=t, done=int(done), rooms=len(snaps) + 1,
                      hurts=sum(k == KIND_HURT for _, k in pts), points=[])
            picked = [q for q in pts if q[1] == KIND_FATAL]
            hurts_ = [q for q in pts if q[1] == KIND_HURT]
            picked += [hurts_[int(j)] for j in rng.permutation(len(hurts_))[:args.hurts]]
            # random points as the worker's: not at a room change, two or more records after their restore point
            cand = [j for j in range(t) if not changed_at[j] and not rows[j]['done'] and j > base_of[j][2] + 1]
            for j in rng.permutation(len(cand))[:args.random]:
                picked.append((cand[int(j)], KIND_RANDOM))
            for at_t, kind in picked:
                parked, reseed_, offset = base_of[at_t]
                pt = V2Point(base=parked, reseed=reseed_, applied=applied, log=log,
                             at=at_t + 1 if kind == KIND_RANDOM else at_t, offset=offset, kind=kind, seed=seed,
                             episode=1, visited=parked.info.get('visited', ()), stage0=parked.info.get('stage0'),
                             pid=len(ep['points']) + 1)
                rec = probe_point(pt, ctx, rows, applied, rng)
                rec.update(kind=kind, at=at_t, offset=offset)
                ep['points'].append(rec)
                print(json.dumps(rec), flush=True)
            for snap in snaps:
                snap.release()
            template.release()
            report.append(ep)
    finally:
        inst.kill()
    pts_all = [q for e in report for q in e['points']]
    searched = [q for q in pts_all if q.get('search')]
    summary = dict(
        episodes=len(report), deaths=sum(e['done'] == 2 for e in report), points=len(pts_all),
        replay_points=sum(1 for q in pts_all if q.get('replay')),
        replay_records=sum(q['replay']['records'] for q in pts_all if q.get('replay')),
        replay_mismatch_records=sum(q['replay']['mismatch'] for q in pts_all if q.get('replay')),
        replay_under_bad=sum(q['replay']['under_bad'] for q in pts_all if q.get('replay')),
        replay_score_max_abs_diff=max([abs(q['replay']['score'] - q['replay']['taken']) for q in pts_all
                                       if q.get('replay')] or [0.0]),
        searched=len(searched), improving=sum(bool(q['search']['improving']) for q in searched),
        mean_gain0=float(np.mean([q['search']['gain0'] for q in searched])) if searched else None,
        mean_spread=float(np.mean([q['search']['spread'] for q in searched if q['search']['spread'] is not None]))
        if any(q['search']['spread'] is not None for q in searched) else None,
        records=sum(q['search'].get('records', 0) for q in searched),
        record_checks_bad=sum(q['search'].get('checks_bad', 0) for q in searched),
        seconds_per_point=float(np.mean([q['search']['seconds'] for q in searched])) if searched else None,
        decisions_per_point=float(np.mean([q['search']['decisions'] for q in searched])) if searched else None,
        fast_row=fast_fn is not None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'episodes.json').write_text(json.dumps(report, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


def probe_point(pt, ctx, rows, applied, rng):
    """The replay check and the search at one point (see the module docstring)."""
    rec = dict()
    k0 = 0 if pt.kind == KIND_RANDOM else (ctx.cfg.teacher_death_depths[0] if pt.kind == KIND_FATAL and
                                           ctx.cfg.teacher_death_depths else ctx.cfg.teacher_depths[0])
    d = pt.at - 1 - k0
    if d < pt.offset + 1:
        rec['skip'] = f'd {d} before the restore point {pt.offset} + 1'
        return rec
    # replay: the episode's own actions from the restored state at d
    S, hp0 = restore(pt, d, ctx)
    try:
        pol, trace = EpisodePolicy(applied), []
        b = play_branch(S, ctx, pt, d, hp0, Intent('hold', 0, h=0), pol, trace=trace)
    finally:
        S.close()
    taken, steps = taken_score(pt.log, d, ctx.cap, ctx.gamma, ctx.rw, ctx.run)
    bad, first_bad = 0, None
    for j, (r, _) in enumerate(trace):
        if d + j >= len(rows):
            bad += 1
            continue
        diff = row_diff(r, rows[d + j], j == 0)
        if diff:
            bad += 1
            if first_bad is None:
                first_bad = dict(j=j, fields=diff)
    rec['replay'] = dict(d=d, records=len(trace), steps=b['n'], taken_steps=steps, mismatch=bad, first=first_bad,
                         under_bad=pol.under_bad, score=round(b['score'], 6), taken=round(taken, 6), end=b['end'])
    # the search with the stand-in policy
    t0, c0 = time.perf_counter(), time.process_time()
    s0 = dict(ctx.stats)
    depths = []

    def on_depth(k, d_, taken_, scores, S_, hp0_):
        depths.append(dict(k=k, d=d_, taken=round(taken_, 4), scores=[round(s, 4) for s in scores]))
    res = search_point(pt, ctx, StandIn(int(rng.integers(1 << 30))), rng, on_depth=on_depth)
    srec = dict(seconds=round(time.perf_counter() - t0, 2), cpu=round(time.process_time() - c0, 2),
                decisions=ctx.stats['decisions'] - s0['decisions'], branches=ctx.stats['branches'] - s0['branches'],
                depths=depths)
    if res is not None:
        b = res['best']
        srec.update(k=res['k'], taken=round(res['taken'], 4), best=round(b['score'], 4), gain=round(res['gain'], 4),
                    gain0=round(res['gain0'], 4), rep=None if res['rep'] is None else round(res['rep'], 4),
                    spread=None if res['spread'] is None else round(res['spread'], 4), improving=bool(res['improving']),
                    intent=b['intent'].code, hold=b['intent'].h, best_steps=b['n'], best_end=b['end'])
        if res['improving']:
            r_rows, meta = records_of(res, ctx, pt, 0, ctx.cfg.teacher_v2_cap)
            checks = []
            checks.append(len(r_rows) == b['n'])
            checks.append(tuple(int(v) for v in meta[0, VI['pend0']:VI['pend0'] + 5]) == five(applied[res['d']]))
            checks.append(abs(meta[0, VI['ret']] - b['score']) < 1e-9)
            checks.append(bool(np.all(meta[:, VI['weight']] == min(res['gain'], ctx.cfg.teacher_v2_cap))))
            g = meta[:, VI['ret']]
            rw = b['rews']
            checks.append(all(abs(g[j] - (rw[j] + ctx.gamma * (g[j + 1] if j + 1 < len(g) else
                                                                 (b['v_end'] if b['boot'] else 0.0)))) < 1e-9
                              for j in range(len(g))))
            if b['intent'].kind == 'hold':
                hh = min(b['intent'].h, len(r_rows))
                checks.append(bool(np.all(meta[:hh, VI['act0']] == b['intent'].move)))
            checks.append(bool(np.all(r_rows['t'] == res['d'] + np.arange(len(r_rows)))))
            srec.update(records=len(r_rows), checks_bad=sum(not c for c in checks), weight=float(meta[0, VI['weight']]))
    else:
        srec['none'] = True
    rec['search'] = srec
    return rec


if __name__ == '__main__':
    main()
