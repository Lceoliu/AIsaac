"""Character probe (2026-10-08, character randomisation, EXPERIMENTS.md A25): floor-mode starts as each AB+ PlayerType.

Per character (--chars, default every id 0 .. 17 of AB+'s resources/scripts/enums.lua PlayerType) and seed: the root's
floor reset with bridge.character = id (abplus.AbplusTrainingEnv._restart_floor), then a lean clone of the
start state (the inventory block on: ISAAC_RL_LEAN_ITEMS=1) whose first observation gives the player's lean fields
(hearts of every kind, coins / bombs / keys, damage, fire delay, shot speed, range, speed, luck, flight, active item and
charge, weapon-type bits, the PlayerType), the inventory block (held collectibles, trinkets, pocket pill / card) and
the ROW the policy gets (pchar, the player columns of the hearts); a Lua query adds what the lean record does not hold
(name, max charge of the active item). The floor's layout (grid index -> type / variant / shape / doors) is hashed per
seed: does the character change the generated floor? Then --steps decisions of sticky random moves and aimed shots: no
error, deaths, the PlayerType at the end; every record goes through both row encoders (abplus_probe_fast_row.Pair:
encode_row and abp_turbo's abp_row_encode, byte for byte) and the clone runs the native observation in check mode
(abp_bridge.lua native_obs 2: the Lua and the native player records compared at every step). --kill (default 8,
Lazarus): one more clone of these characters whose player is killed by Lua (Kill()) and stepped --kill-steps decisions
without input: the lean 'dead' flag over time and the PlayerType afterwards (Lazarus revives as Lazarus II; an episode
ends at the first dead record). Without a revival the game-over screen stops the logic frames and a step never returns.
The restarts as another character go through abp_turbo's ABP_PLAYERTYPE (abp_turbo.c start_debug_player_init): the
instances run a debug game, and `restart <id>` alone starts Isaac (seen in the first run of this probe).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_probe_characters.py --groups-file ../abplus/catalog/scaling2_groups.json --seeds 2147600000:2 \
      --out <dir>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from abplus_probe_fast_row import Pair, step
from abplus_probe_native_obs import aim, native_obs, read_raw
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_floor import CHARACTER_NAMES
from isaac_bridge.tok_obs import fast_row_function

PLAYER_KEYS = ('hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal', 'golden', 'lives', 'coins', 'bombs', 'keys',
               'damage', 'fire_delay_max', 'shot_speed', 'range', 'speed', 'luck', 'can_fly', 'active',
               'active_charge', 'active_ready', 'weapons', 'charge', 'ptype')
WEAPONS = ('tears', 'brimstone', 'technology', 'knife', 'fetus', 'epic_fetus', 'lung', 'ludovico', 'tech_x', 'bone')
INFO_LUA = ("local p = Isaac.GetPlayer(0) local name = '?' pcall(function() name = p:GetName() end) "
            "local mc = -1 pcall(function() local c = Isaac.GetItemConfig():GetCollectible(p:GetActiveItem()) "
            "if c then mc = c.MaxCharges end end) "
            "return tostring(p:GetPlayerType()) .. '|' .. name .. '|' .. tostring(mc) .. '|' .. "
            "tostring(p:GetMaxHearts()) .. '|' .. tostring(p:GetHearts())")
KILL_PLAYER_LUA = "Isaac.GetPlayer(0):Kill() return 'ok'"


def player_of(o):
    pl = o.players[0]
    out = {k: round(float(pl[k]), 4) for k in PLAYER_KEYS}
    out['weapon_names'] = [WEAPONS[w - 1] for w in range(1, 11) if int(pl['weapons']) >> w & 1]
    return out


def inventory_of(dec):
    if dec.inv_head is None:
        return None
    n, _, max_charge, tr0, tr1, pill, effect, card, curses = dec.inv_head
    return dict(collectibles=[[int(i), int(c)] for i, c in zip(dec.inv['id'], dec.inv['count'])],
                active_max_charge=int(max_charge), trinkets=[int(tr0), int(tr1)], pill_color=int(pill),
                pill_effect=int(effect), card=int(card), curses=int(curses))


def floor_hash(floor):
    text = json.dumps({str(k): floor[k] for k in sorted(floor)}, sort_keys=True)
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def one(inst, seed, char, args, fn, rng):
    b = inst.env.bridge
    b.reset_mode = 'floor'
    b.character = char
    t0 = time.perf_counter()
    _, info = inst.reset(seed)
    rec = dict(char=char, name=CHARACTER_NAMES.get(char), seed=seed, reset_s=round(time.perf_counter() - t0, 2),
               floor_hash=floor_hash(info['floor']), start_room=info['start_room'], curses=info['floor_curses'],
               room_attempts=info['room_attempts'])
    S = b.fork(tag='template', alarm=0)
    try:
        c = S.fork(lean=True, alarm=600)
        try:
            lua = c.lua(INFO_LUA).split('|')
            rec['lua'] = dict(ptype=int(lua[0]), name=lua[1], active_max_charge=int(lua[2]), max_hearts=int(lua[3]),
                              hearts=int(lua[4]))
            native_obs(c, 2)   # check mode: Lua and native fixed parts compared at every observation
            pair = Pair(fn, 10 ** 6, True, 450, run=True, items=True)
            c._send({"cmd": "obs"})
            o, _ = pair.record(read_raw(c))
            rec['player'] = player_of(o)
            rec['inventory'] = inventory_of(pair.dec_a)
            row = pair.last_row[0]
            rec['row'] = dict(pchar=int(row['pchar']), hearts_col=round(float(row['player'][4]), 4),
                              soul_col=round(float(row['player'][5]), 4), max_col=round(float(row['player'][6]), 4),
                              units_col=round(float(row['player'][7]), 4),
                              pitem=row['pitem'].tolist(), inv=[v for v in row['inv'].tolist() if v[0]])
            move, shoot, dead_at, err = 0, 0, None, None
            try:
                for t in range(args.steps):
                    if t % 6 == 0:
                        move = int(rng.integers(0, 9))
                    shoot = aim(o) if rng.random() < 0.8 else int(rng.integers(0, 5))
                    o, _ = pair.record(step(c, (move, shoot, 0, 0), args.fpd))
                    if o.dead:
                        dead_at = t
                        break
            except Exception as e:   # reported, not raised: the next character goes on
                err = f'{type(e).__name__}: {e}'
            nat = native_obs(c)
            rec['play'] = dict(records=pair.t, error=err, dead_at=dead_at, ptype_end=float(o.players[0]['ptype']),
                               hearts_end=float(o.players[0]['hearts']), soul_end=float(o.players[0]['soul']),
                               fast=pair.enc.fast, slow=pair.enc.slow, fast_row_mismatch=pair.mismatch,
                               native_available=nat.get('available'), native_checks=nat.get('checks'),
                               native_mismatches=nat.get('mismatches'), native_first=nat.get('first'))
        finally:
            c.close()
        if char in args.kill_chars:   # (a dead player without a revival ends the run: the game-over screen, no steps)
            k = S.fork(lean=True, alarm=600)
            try:
                k.lua(KILL_PLAYER_LUA)
                pair = Pair(fn, 10 ** 6, True, 450, run=True, items=True)
                k._send({"cmd": "obs"})
                o, _ = pair.record(read_raw(k))
                dead, types = [bool(o.dead)], [int(o.players[0]['ptype'])]
                for _ in range(args.kill_steps):
                    o, _ = pair.record(step(k, (0, 0, 0, 0), args.fpd))
                    dead.append(bool(o.dead))
                    types.append(int(o.players[0]['ptype']))
                rec['kill'] = dict(dead_records=int(sum(dead)), first_dead=dead.index(True) if any(dead) else None,
                                   last_dead=len(dead) - 1 - dead[::-1].index(True) if any(dead) else None,
                                   alive_at_end=not dead[-1], ptypes=sorted(set(types)), ptype_end=types[-1],
                                   hearts_end=float(o.players[0]['hearts']), lives_end=float(o.players[0]['lives']))
            except Exception as e:
                rec['kill'] = dict(error=f'{type(e).__name__}: {e}')
            finally:
                k.close()
    finally:
        S.close()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--seeds', default='2147600000:2')
    p.add_argument('--chars', default=','.join(str(k) for k in range(18)))
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--kill', default='8', help='characters whose player is killed in one more clone (Lazarus: 8)')
    p.add_argument('--kill-steps', type=int, default=60)
    p.add_argument('--port', type=int, default=44410)
    p.add_argument('--name', default='chrprobe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    args.kill_chars = {int(v) for v in args.kill.split(',') if v.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    fn = fast_row_function(default_preload())
    if fn is None:
        raise SystemExit('abp_row_encode (version 6) not available')
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=args.fpd, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    first, count = (int(v) for v in args.seeds.split(':'))
    chars = [int(v) for v in args.chars.split(',') if v.strip()]
    inst = Instance(args.name, args.port, gx, spec)
    rng = np.random.default_rng(23)
    report = []
    t0 = time.perf_counter()
    try:
        for seed in range(first, first + count):
            for char in chars:
                try:
                    rec = one(inst, seed, char, args, fn, rng)
                except Exception as e:   # a reset that fails: reported, the instance relaunched
                    rec = dict(char=char, seed=seed, error=f'{type(e).__name__}: {e}')
                    inst.relaunch(hard=True)
                report.append(rec)
                print(json.dumps(rec, default=str), flush=True)
        # the same seed as Isaac again at the end: the reset is not changed by the characters before it
        b = inst.env.bridge
        b.character = 0
        _, info = inst.reset(first)
        again = floor_hash(info['floor'])
    finally:
        inst.kill()
    ok = [r for r in report if 'error' not in r]
    summary = dict(characters=chars, seeds=count, resets=len(report), errors=len(report) - len(ok),
                   play_errors=sum(r['play']['error'] is not None for r in ok),
                   fast_row_mismatches=sum(r['play']['fast_row_mismatch'] is not None for r in ok),
                   native_checks=sum(r['play']['native_checks'] or 0 for r in ok),
                   native_mismatches=sum(r['play']['native_mismatches'] or 0 for r in ok),
                   ptype_ok=sum(r['player']['ptype'] == r['char'] and r['row']['pchar'] == r['char'] for r in ok),
                   floor_same_as_isaac={str(s): sorted({r['char'] for r in ok if r['seed'] == s and r['floor_hash'] == next(
                       (q['floor_hash'] for q in ok if q['seed'] == s and q['char'] == 0), None)}) for s in
                       range(first, first + count)},
                   isaac_again_same=again == next((q['floor_hash'] for q in ok if q['seed'] == first and q['char'] == 0),
                                                  None),
                   seconds=round(time.perf_counter() - t0, 1))
    (out / 'characters.json').write_text(json.dumps(report, indent=1, default=str))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
