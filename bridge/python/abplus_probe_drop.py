"""visible-v3 acceptance step 4 (DESIGN 6, package C, 2026-11-17): the drop action end to end on AB+, through the three
ways a client sends a step -- the step line's 7th number (the sampler's path, lean clones), the play command's code
(+ 360) and the JSON step's drop field -- from one template per case, so the three must end in the same state:

  forgotten   PlayerType 16: one drop press switches to the Soul (PlayerType and the sub-player as the game shows them)
  schoolbag   Schoolbag (534) + two active items: one press swaps the active item
  pockets     a card and a pill: one press cycles the pocket items (GetCard(0) / GetPill(0))
  trinket     a trinket: drop held for --hold decisions drops it (GetTrinket(0) 0, a trinket pickup in the room)
  (the items given, the press waits --wait idle decisions: no swap while the pickup animation runs; pockets: Polydactyly
  for a second pocket)
  release     one press, then decisions with drop 0: the swap happens once (no repeat while released)
  zero        drop 0 written out ("S r m s b i 0 0", play codes < 360, JSON drop 0) against the old forms (5 numbers,
              no field): the game digest after a fixed 40-decision sequence is equal (G6: drop 0 leaves trajectories)

Per case and path: the state before / after (JSON in <out>/drop.jsonl) and ABPGX_DIGEST of the end state; the summary
says per case whether the three paths agree and whether the expected change happened.

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_drop.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from abplus_probe_native_obs import read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV, action_code
from isaac_bridge.abplus_goexplore import GxConfig, Instance

STATE_LUA = r"""
local json = require("json")
local p = Isaac.GetPlayer(0)
local function try(f) local ok, v = pcall(f) if ok then return v end return "ERR:" .. tostring(v) end
local sub = try(function() local s = p:GetSubPlayer() return s and s:GetPlayerType() or false end)
local trinkets = 0
for _, e in ipairs(Isaac.GetRoomEntities()) do if e.Type == 5 and e.Variant == 350 then trinkets = trinkets + 1 end end
return json.encode({ ptype = p:GetPlayerType(), sub = sub, active = p:GetActiveItem(),
  secondary = try(function() return p.SecondaryActiveItem.Item end), card = p:GetCard(0), pill = p:GetPill(0),
  trinket = p:GetTrinket(0), trinkets_on_floor = trinkets, x = p.Position.X, y = p.Position.Y,
  frame = Game():GetFrameCount(), digest = ABPGX_DIGEST and ABPGX_DIGEST() or "no digest" })
"""
SETUP = {
    'forgotten': None,
    'schoolbag': "local p = Isaac.GetPlayer(0) p:AddCollectible(534, 0, false) p:AddCollectible(45, 0, false) "
                 "p:AddCollectible(63, 0, false) return 'ok'",
    'pockets': "local p = Isaac.GetPlayer(0) p:AddCollectible(454, 0, false) p:AddCard(1) p:AddPill(3) return 'ok'",
    'trinket': "local p = Isaac.GetPlayer(0) p:AddTrinket(1) return 'ok'",
    'release': "local p = Isaac.GetPlayer(0) p:AddCollectible(534, 0, false) p:AddCollectible(45, 0, false) "
               "p:AddCollectible(63, 0, false) return 'ok'",
}
CHAR = {'forgotten': 16}


def state(c):
    return json.loads(c.lua(STATE_LUA))


def run_line(c, seq, form):
    """A lean clone: each action (m, s, b, i, p, d) as a step line ('new': 7 numbers always, 'old': as before)."""
    for m, s, b, i, p, d in seq:
        if form == 'new':
            c._sock.sendall(b'S 4 %d %d %d %d %d %d\n' % (m, s, b, i, p, d))
        elif d:
            c._sock.sendall(b'S 4 %d %d %d %d %d %d\n' % (m, s, b, i, p, d))
        elif p:
            c._sock.sendall(b'S 4 %d %d %d %d %d\n' % (m, s, b, i, p))
        else:
            c._sock.sendall(b'S 4 %d %d %d %d\n' % (m, s, b, i))
        read_raw(c)


def run_play(c, seq):
    c._send({"cmd": "play", "actions": [action_code(*a) if a[5] else action_code(*a[:5]) for a in seq], "repeat": 4, "stop_clear": False})
    read_raw_after_ok(c)


def read_raw_after_ok(c):
    while True:
        line = c._read_line()
        if line.startswith(b'L '):
            c._read_exact(int(line.split(b' ')[1]))
            return
        msg = json.loads(line.decode('utf-8'))
        if msg.get('type') == 'error':
            raise RuntimeError(msg.get('msg'))


def run_json(c, seq, form):
    for m, s, b, i, p, d in seq:
        msg = {"cmd": "step", "repeat": 4, "move": m, "shoot": s, "bomb": b, "item": i, "pill": p}
        if d or form == 'new':
            msg["drop"] = d
        c._send(msg)
        read_raw_after_ok(c)


def case_seq(case, hold, wait):
    idle = (0, 0, 0, 0, 0, 0)
    press = (0, 0, 0, 0, 0, 1)
    if case == 'trinket':
        return [idle] * wait + [press] * hold + [idle] * 4
    if case == 'release':
        return [idle] * wait + [press] + [idle] * 10
    if case == 'zero':
        moves = [(m, s, 0, 0, 0, 0) for m, s in ((3, 0), (3, 2), (5, 0), (5, 3), (7, 1), (1, 4), (0, 0), (2, 2))]
        return (moves * 5)[:40]
    return [idle] * wait + [press] + [idle] * 6


def one(inst, seed, case, hold, wait, old_only, out_f):
    b = inst.env.bridge
    b.reset_mode, b.character = 'floor', CHAR.get(case, 0)
    inst.reset(seed)
    if SETUP.get(case):
        assert b.lua(SETUP[case]) == 'ok'
        b.step({}, repeat=4)
    S = b.fork(tag='template', alarm=0)
    res = {}
    try:
        forms = [('line', 'new'), ('line', 'old'), ('play', ''), ('json', 'new'), ('json', 'old')] if case == 'zero' \
            else [('line', 'new'), ('play', ''), ('json', 'new')]
        if old_only:   # (a bridge before the drop: only the forms it reads)
            forms = [f for f in forms if f[1] != 'new']
        seq = case_seq(case, hold, wait)
        for path, form in forms:
            c = S.fork(lean=True, alarm=600)
            try:
                c._send({"cmd": "obs"}); read_raw(c)
                before = state(c)
                if path == 'line':
                    run_line(c, seq, form)
                elif path == 'play':
                    run_play(c, seq)
                else:
                    run_json(c, seq, form)
                after = state(c)
            finally:
                c.close()
            key = f'{path}-{form}' if form else path
            res[key] = dict(before=before, after=after)
            out_f.write(json.dumps(dict(case=case, seed=seed, path=key, before=before, after=after)) + '\n')
            out_f.flush()
    finally:
        S.close()
    return res


def verdict(case, res):
    afters = {k: v['after'] for k, v in res.items()}
    keys = ('ptype', 'sub', 'active', 'secondary', 'card', 'pill', 'trinket', 'trinkets_on_floor', 'digest')
    same = len({json.dumps([a.get(k) for k in keys]) for a in afters.values()}) == 1
    b0 = next(iter(res.values()))['before']
    a0 = next(iter(afters.values()))
    expected = {
        'forgotten': a0['ptype'] != b0['ptype'],
        'schoolbag': a0['active'] != b0['active'],
        'pockets': (a0['card'], a0['pill']) != (b0['card'], b0['pill']),
        'trinket': a0['trinket'] == 0 and b0['trinket'] != 0,
        'release': a0['active'] != b0['active'],   # one swap (two would swap back)
        'zero': True,
    }[case]
    return dict(paths_agree=same, expected_change=expected, before=b0, after=a0)


def summary_ok(summary):
    """Every case ran, both input paths agree and the change is the expected one; no case at all is a failure."""
    return bool(summary) and all(v.get('paths_agree') and v.get('expected_change') for v in summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147800000:2')
    p.add_argument('--cases', default='forgotten,schoolbag,pockets,trinket,release,zero')
    p.add_argument('--hold', type=int, default=20)
    p.add_argument('--wait', type=int, default=20, help='idle decisions before the press (the pickup animation ends)')
    p.add_argument('--old-only', type=int, default=0, help='1: only the old forms (a tree before the drop)')
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--port', type=int, default=44490)
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
                  frames_per_decision=args.fpd, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    inst = Instance('drop', args.port, gx, spec)
    summary, t0 = [], time.perf_counter()
    with open(out / 'drop.jsonl', 'w') as f:
        try:
            for seed in range(first, first + count):
                for case in args.cases.split(','):
                    try:
                        res = one(inst, seed, case, args.hold, args.wait, args.old_only, f)
                        v = dict(case=case, seed=seed, **verdict(case, res))
                    except Exception as e:   # reported; the instance relaunched
                        v = dict(case=case, seed=seed, error=f'{type(e).__name__}: {e}')
                        inst.relaunch(hard=True)
                    summary.append(v)
                    print(json.dumps(v)[:600], flush=True)
        finally:
            inst.kill()
    ok = summary_ok(summary)
    (out / 'summary.json').write_text(json.dumps(dict(ok=ok, cases=summary, seconds=round(time.perf_counter() - t0, 1)),
                                                 indent=1))
    print('SUMMARY ok' if ok else 'SUMMARY FAIL', flush=True)
    if not ok:   # (2026-11-18, Codex round 2: a failing case fails the run)
        sys.exit(1)


if __name__ == '__main__':
    main()
