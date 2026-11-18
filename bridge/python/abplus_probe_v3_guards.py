"""visible-v3 guards on AB+ (2026-11-18, Codex round 2, the must-fix items that only the game can show):
  mode_cmd   the bridge's visible_v3 command refuses a mode outside 0-3 (4, -1, 'x', 2.5, true) with an error and the
             mode kept; 0-3 are taken; the query (no mode) answers the current mode; visible_producer.check_bridge_mode
             passes on that answer for the same mode and fails for another (item 2 / item 3)
  connect    the root connected under this process's ISAAC_RL_VISIBLE_V3 (AbplusTrainingEnv.connect runs the check)
  lasers     two lasers spawned with one InitSeed: abp_native_v3 hands back with 'v3: two lasers with one InitSeed';
             two with different seeds: a native frame, equal to the reference encoder's frame of the raw block (item 6a)
  flag64     a lean payload of a mode 2 clone carries the native frame (flag 64) and visible_producer.needs_decode says
             so (the C ROW path never takes it)
  decorative a listed decorative effect without damage (1000.7) left out with entity collision class 0 and sent with
             class 4, in the raw block and the native frame alike (C11)
Exits 1 when a check fails (each check's result in <out>/summary.json).

usage (bridge python dir, PYTHONPATH=.): python abplus_probe_v3_guards.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('ISAAC_RL_VISIBLE_V3', '0')

from abplus_probe_native_obs import read_raw   # noqa: E402
from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.abplus import FORK_ENV, check_visible_v3   # noqa: E402
from isaac_bridge.env import BridgeError   # noqa: E402
from isaac_bridge.abplus_goexplore import GxConfig, Instance   # noqa: E402

LASERS = r"""
local p = Isaac.GetPlayer(0)
local seeds = { __A__, __B__ }
local out = {}
for k = 1, 2 do
  local e = Game():Spawn(7, 2, p.Position + Vector(40 * k, 0), Vector(0, 0), p, 0, seeds[k])
  local l = e:ToLaser()
  l.Timeout = 60
  l.Angle = 90 * k
  out[#out + 1] = tostring(e.InitSeed)
end
return table.concat(out, ',')
"""
NATIVE = ("local f, why = abp_native_v3(V3_MAP_BYTES(), V3_LASER_BYTES()) "
          "if not f then return 'nil:' .. tostring(why) end "
          "return (f:gsub('.', function(ch) return string.format('%02x', ch:byte()) end))")
SEEDS_NOW = "local s = {} for _, e in ipairs(Isaac.FindByType(7, -1, -1, false, false)) do s[#s + 1] = tostring(e.InitSeed) end return table.concat(s, ',')"


def query(c, mode=None):
    c._send({"cmd": "visible_v3"} if mode is None else {"cmd": "visible_v3", "mode": mode})
    try:
        return c._recv()
    except BridgeError as exc:   # (_recv raises on an error reply)
        return {"type": "error", "msg": str(exc)}


def check_mode_cmd(c):
    res = {}
    start = query(c, 1)
    res['set 1'] = start.get('type') == 'ok' and start.get('mode') == 1
    for bad in (4, -1, 'x', 2.5, True):
        r = query(c, bad)
        after = query(c)
        res[f'refuse {bad!r}'] = r.get('type') == 'error' and 'mode must be 0-3' in str(r.get('msg')) and \
            after.get('type') == 'ok' and after.get('mode') == 1
    for m in (0, 2, 3, 1):
        r = query(c, m)
        res[f'set {m}'] = r.get('type') == 'ok' and r.get('mode') == m and query(c).get('mode') == m
    try:   # (the connect's own check, on this clone: mode 1 now)
        res['check same mode'] = check_visible_v3(c, 1) == 1
    except VP.VisibleV3Fatal:
        res['check same mode'] = False
    return res


def check_refused(c):
    """The connect's check against another mode (this clone at mode 1, the client at 2): VisibleV3Fatal (not a
    BridgeError, which the samplers retry) and the connection closed (2026-11-18, Codex round 3). The clone is used up."""
    query(c, 1)
    try:
        check_visible_v3(c, 2)
        return dict(refused=False)
    except VP.VisibleV3Fatal as exc:
        return dict(refused='configured for mode 2' in str(exc), closed=c._sock is None)
    except BridgeError as exc:
        return dict(refused=False, bridge_error=str(exc)[:200])


def check_lasers(c, seed_a, seed_b):
    """On a mode 2 clone: spawn two lasers, step once (they take part in a frame), then the native call itself."""
    made = c.lua(LASERS.replace('__A__', str(seed_a)).replace('__B__', str(seed_b)))
    c._sock.sendall(b'S 1 0 0 0 0\n')
    payload = read_raw(c)
    seeds = c.lua(SEEDS_NOW)
    native = c.lua(NATIVE)
    out = dict(spawned=made, seeds_after_step=seeds, step_flag64=bool(payload[12] & 64))
    if native.startswith('nil:'):
        out['native'] = native
    else:
        frame = bytes.fromhex(native)
        raw = json.loads(c.lua("return visible_v3_raw()"))
        out['native'] = 'frame'
        out['equals_reference'] = frame == VP.observation_frame_v3(raw)
    return out


DECOR = r"""
local p = Isaac.GetPlayer(0)
local e = Isaac.Spawn(1000, 7, 0, p.Position + Vector(60, 0), Vector(0, 0), nil)
e.CollisionDamage = 0
e.EntityCollisionClass = __CLASS__
return tostring(e.EntityCollisionClass) .. ',' .. tostring(e.CollisionDamage)
"""


def check_decorative(c, cls):
    """C11 (2026-11-18): a listed decorative effect (1000.7 BLOOD_SPLAT, no damage) is left out with no entity collision
    and sent with one -- in the raw block and in the native frame alike (and the native frame equal to the reference)."""
    made = c.lua(DECOR.replace('__CLASS__', str(cls)))
    raw = json.loads(c.lua("return visible_v3_raw()"))
    in_raw = sum(1 for e in raw['entities'] if e['type'] == 1000 and e['variant'] == 7)
    native = c.lua(NATIVE)
    out = dict(spawned=made, in_raw=in_raw, native=native[:40] if native.startswith('nil:') else 'frame')
    if not native.startswith('nil:'):
        frame = bytes.fromhex(native)
        f = VP.unpack_frame_v3(frame)
        out['in_native'] = int(sum(1 for r in f['E'] if int(r[0]) == 1000 and int(r[1]) == 7))
        out['equals_reference'] = frame == VP.observation_frame_v3(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seed', type=int, default=2148200000)
    ap.add_argument('--port', type=int, default=44560)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    t0 = time.perf_counter()
    inst = Instance('v3guards', args.port, gx, spec)
    checks, detail = {}, {}
    try:
        b = inst.env.bridge
        b.reset_mode, b.character = 'floor', 0
        inst.reset(args.seed)
        checks['connect'] = True   # (the connect's check_bridge_mode passed: mode VP.VISIBLE_V3_MODE)
        detail['configured_mode'] = VP.VISIBLE_V3_MODE
        c = b.fork(tag='v3guards', lean=True, alarm=600)
        try:
            detail['mode_cmd'] = check_mode_cmd(c)
            checks['mode_cmd'] = all(detail['mode_cmd'].values())
            a = query(c, 2)
            detail['native_available'] = a.get('native_available')
            if not a.get('native_available'):
                checks['native available'] = False
            else:
                c._send({"cmd": "obs"})
                p = read_raw(c)
                checks['flag64'] = bool(p[12] & 64) and VP.needs_decode(p)
        finally:
            c.close()
        c = b.fork(tag='v3guards', lean=True, alarm=600)
        try:
            d = detail['check another mode refused'] = check_refused(c)
            checks['check another mode refused, connection closed'] = d.get('refused') is True and d.get('closed') is True
        finally:
            try:
                c.close()
            except Exception:
                pass
        if detail.get('native_available'):
            for name, (sa, sb) in (('two seeds', (11111, 22222)), ('one seed', (33333, 33333))):
                c = b.fork(tag='v3guards', lean=True, alarm=600)
                try:
                    query(c, 2)
                    detail[name] = check_lasers(c, sa, sb)
                finally:
                    c.close()
            for cls, want in ((0, 0), (4, 1)):
                c = b.fork(tag='v3guards', lean=True, alarm=600)
                try:
                    query(c, 2)
                    d = detail[f'decorative class {cls}'] = check_decorative(c, cls)
                    checks[f'decorative 1000.7 with collision class {cls}: {"sent" if want else "left out"}'] =                         d['in_raw'] == want and d.get('in_native') == want and d.get('equals_reference') is True
                finally:
                    c.close()
            d1, d2 = detail['two seeds'], detail['one seed']
            checks['lasers two seeds: a frame equal to the reference'] = d1['native'] == 'frame' and \
                d1.get('equals_reference') is True and d1['seeds_after_step'].count(',') == 1
            checks['lasers one seed: handed back'] = d2['native'] == 'nil:v3: two lasers with one InitSeed' and \
                len(set(d2['seeds_after_step'].split(','))) == 1 and d2['seeds_after_step'].count(',') == 1 and \
                not d2['step_flag64']
    except Exception as e:   # reported as a failure
        checks['ran'] = False
        detail['error'] = f'{type(e).__name__}: {e}'
    finally:
        inst.kill()
    ok = bool(checks) and all(checks.values())
    summary = dict(ok=ok, checks=checks, detail=detail, seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('SUMMARY', json.dumps(summary, default=str), flush=True)
    print('PASS' if ok else 'FAIL', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
