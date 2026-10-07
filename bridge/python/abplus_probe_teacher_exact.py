"""Teacher label equality under a speed-up (2026-10-05): the same recorded hurts searched by tok_sampler.hindsight
twice, once as the reference and once as the candidate, compared depth by depth.

Per seed: reset, a template clone parked at the start (as tok_sampler's worker), then episodes of sticky random
actions played on lean clones of the template reseeded as the worker reseeds; every decision whose observation shows
damage is a hurt. For up to --hurts hurts per episode both searches run (order alternating). Equal means, for every
depth searched: the depth, the action under way, the move taken, the lost[] vector (half hearts lost per held move,
+100 for a death) and the restored observation's ROW record (encode_row with a fresh EpisodeState, as the worker
writes a teacher record). records_equal compares only what the teacher record holds: lost[] as danger bits. --digest also restores the k = 2 state both ways and compares the hidden-state digests
(abplus_goexplore.DIGEST_LUA). --shared: the candidate searches each episode's picked hurts in one
tok_sampler.hindsight_shared call. --snap-at N: the candidate restores from a clone parked at decision N (offset N,
reseed None: the floor worker's room-entry restore points).

The two sides differ by:
  --ref-lua / --cand-lua   Lua run on the template before that side's search (e.g. "ABP_PU.play = true; return 'ok'")
  --cand-cfg               JSON of TokSamplerConfig fields for the candidate (e.g. '{"teacher_share": 0}')
  --cand-env               KEY=VALUE pairs set in this process's environment during the candidate's search

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_teacher_exact.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --out <dir> --cand-lua "ABP_PU.play = true; return 'ok'" \
      --ref-lua "ABP_PU.play = false; return 'ok'"
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import digest, sticky_actions
from abplus_probe_teacher import record_episode, summarize
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder
from isaac_bridge.tok_sampler import TokSamplerConfig, hindsight, hindsight_shared, known_hurt_move, play_actions


def restore_digest(template, reseed, applied, d, fpd):
    st = template.fork(reseed=reseed, lean=True, alarm=300)
    try:
        play_actions(st, LeanDecoder(), applied[:d], fpd)
        return digest(st)[0]
    finally:
        st.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--hurts', type=int, default=3)
    p.add_argument('--port', type=int, default=38600)
    p.add_argument('--name', default='tchex')
    p.add_argument('--stub-list', default='')
    p.add_argument('--ref-lua', default='')
    p.add_argument('--cand-lua', default='')
    p.add_argument('--cand-cfg', default='{}')
    p.add_argument('--cand-env', default='')
    p.add_argument('--digest', action='store_true')
    p.add_argument('--shared', action='store_true',
                   help='candidate: the picked hurts of each episode through hindsight_shared in one call')
    p.add_argument('--snap-at', type=int, default=0,
                   help='candidate: restore from a clone parked at this decision (offset, as tok_floor); only hurts '
                        'whose every depth lies after it')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  stub_list=args.stub_list)
    ref_cfg = TokSamplerConfig()
    cand_cfg = dataclasses.replace(ref_cfg, **json.loads(args.cand_cfg))
    cand_env = dict(kv.split('=', 1) for kv in args.cand_env.split() if '=' in kv)
    fpd = ref_cfg.frames_per_decision
    limit = 10 ** 6
    inst = Instance(args.name, args.port, gx, spec)
    rows = []
    secs = {'ref': 0.0, 'cand': 0.0}
    searches = equal = digests = digests_equal = records_equal = 0
    t_start = time.perf_counter()
    lua_answers = {}

    def side(name, template, reseed, applied, hurt_at, snap=None):
        code = args.ref_lua if name == 'ref' else args.cand_lua
        if code:
            lua_answers.setdefault(name, template.lua(code))
        saved = {k: os.environ.get(k) for k in cand_env}
        if name == 'cand':
            os.environ.update(cand_env)
        try:
            t0 = time.perf_counter()
            if name == 'cand' and snap is not None:   # restore from the mid-episode snapshot (as tok_floor)
                found = hindsight(snap, None, applied, hurt_at, cand_cfg, limit, args.snap_at)
            else:
                found = hindsight(template, reseed, applied, hurt_at, ref_cfg if name == 'ref' else cand_cfg, limit)
            secs[name] += time.perf_counter() - t0
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return summarize(found, hurt_at, limit)

    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            inst.reset(seed)
            template = inst.env.bridge.fork(tag='template', alarm=0)
            try:
                for e in range(args.episodes):
                    reseed = int(rng.integers(1, 2 ** 31 - 1))
                    if args.ref_lua:
                        template.lua(args.ref_lua)
                    applied, hurts = record_episode(template, reseed, sticky_actions(rng, args.steps), fpd)
                    snap = None
                    if args.snap_at > 0:
                        # the candidate restores from a clone parked at decision snap_at (tok_floor's room-entry
                        # snapshots, reseed None, offset snap_at); only hurts whose every depth lies after it
                        hurts = [h for h in hurts if h - 1 - max(ref_cfg.teacher_depths) >= args.snap_at + 1]
                        if hurts:
                            snap = template.fork(reseed=reseed, lean=True, alarm=900)
                            play_actions(snap, LeanDecoder(), applied[:args.snap_at], fpd)
                    shared = {}
                    if args.shared and hurts:   # the candidate: all picked hurts of the episode in one shared search
                        picks = hurts[:args.hurts]
                        if args.cand_lua:
                            lua_answers.setdefault('cand', template.lua(args.cand_lua))
                        t0 = time.perf_counter()
                        if snap is not None:
                            founds = hindsight_shared(snap, None, applied, picks, cand_cfg, limit, args.snap_at)
                        else:
                            founds = hindsight_shared(template, reseed, applied, picks, cand_cfg, limit)
                        secs['cand'] += time.perf_counter() - t0
                        shared = {h: summarize(f, h, limit) for h, f in zip(picks, founds)}
                        if args.ref_lua:
                            template.lua(args.ref_lua)
                    for hurt_at in hurts[:args.hurts]:
                        res = {}
                        for name in (('ref', 'cand') if searches % 2 == 0 else ('cand', 'ref')):
                            if name == 'cand' and args.shared:
                                res[name] = shared[hurt_at]
                            else:
                                res[name] = side(name, template, reseed, applied, hurt_at, snap)
                        if args.ref_lua:
                            template.lua(args.ref_lua)
                        searches += 1
                        same = res['ref'] == res['cand']
                        equal += same
                        # the teacher record only: depth, action under way, danger bits, move taken, ROW
                        rec = [[(r[0], r[1], [int(v > 0) for v in r[2]], r[3], r[4]) for r in res[s]]
                               for s in ('ref', 'cand')]
                        records_equal += rec[0] == rec[1]
                        row = dict(seed=seed, episode=e, hurt_at=hurt_at, decisions=len(applied), same=same,
                                   depths=[r[0] for r in res['ref']], lost_ref=[r[2] for r in res['ref']],
                                   lost_cand=[r[2] for r in res['cand']],
                                   rows_equal=[a[4] == b[4] for a, b in zip(res['ref'], res['cand'])],
                                   records_same=rec[0] == rec[1],
                                   known=known_hurt_move(applied, hurt_at - 3, hurt_at))
                        d2 = hurt_at - 1 - ref_cfg.teacher_depths[0]
                        if args.digest and d2 >= 1:
                            dg = {}
                            for name, code in (('ref', args.ref_lua), ('cand', args.cand_lua)):
                                if code:
                                    template.lua(code)
                                dg[name] = restore_digest(template, reseed, applied, d2, fpd)
                            if args.ref_lua:
                                template.lua(args.ref_lua)
                            digests += 1
                            digests_equal += dg['ref'] == dg['cand']
                            row['digest_equal'] = dg['ref'] == dg['cand']
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    if snap is not None:
                        snap.close()
            finally:
                template.close()
    finally:
        inst.close()
        summary = dict(group=args.group, seeds=args.seeds, searches=searches, equal=equal,
                       records_equal=records_equal, digests=digests,
                       digests_equal=digests_equal, ref_s=round(secs['ref'] / max(searches, 1), 4),
                       cand_s=round(secs['cand'] / max(searches, 1), 4), lua=lua_answers, cand_cfg=args.cand_cfg,
                       cand_env=args.cand_env, load=os.getloadavg(), seconds=round(time.perf_counter() - t_start, 1))
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
