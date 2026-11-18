"""Level:GetRoomByIdx's domain on AB+ (2026-11-18, the hardening batch, Codex round 2's C14: the door's target index
goes to GetRoomByIdx with no check before the call). Three readings:
  disasm   objdump of Level::GetRoomByIdx (abp_turbo.c L_ROOM_BY_IDX, 0x733cd0) from the game binary: what it checks
  calls    on a throwaway lean clone, level:GetRoomByIdx(k) for k in [-20, 180] (pcall): which k give a descriptor and
           its GridIndex / SafeGridIndex / Data (a crash of the clone is reported, the next k on a new clone)
  doors    over floors of several seeds and stages, every door's TargetRoomIndex in every room entered: the values
           the producers actually pass
Writes <out>/room_idx.json; prints SUMMARY. usage (bridge python dir, PYTHONPATH=.; the CPU runner adds the args):
  python abplus_probe_room_idx.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir> [--seeds 2148300000:4]
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import ABP_HOME, FLOOR_LUA, FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance

CALL = r"""
local level = Game():GetLevel()
local out = {}
for k = __A__, __B__ do
  local ok, d = pcall(function() return level:GetRoomByIdx(k) end)
  if not ok then out[#out + 1] = k .. ':error'
  elseif d == nil then out[#out + 1] = k .. ':nil'
  else
    local ok2, s = pcall(function()
      return k .. ':' .. tostring(d.GridIndex) .. ',' .. tostring(d.SafeGridIndex) .. ',' .. tostring(d.Data ~= nil)
    end)
    out[#out + 1] = ok2 and s or (k .. ':fields-error')
  end
end
return table.concat(out, ';')
"""
DOORS = r"""
local room = Game():GetRoom()
local out = {}
for slot = 0, 7 do
  local d = room:GetDoor(slot)
  if d ~= nil then out[#out + 1] = tostring(d.TargetRoomIndex) end
end
return table.concat(out, ',')
"""
ENTER = "Game():ChangeRoom(%d) return 'ok'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='2148300000:4')
    ap.add_argument('--port', type=int, default=44580)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(disasm=None, calls={}, doors={}, errors=[])
    binary = Path(ABP_HOME) / 'runtime' / 'isaac.x64'
    try:
        r = subprocess.run(['objdump', '-d', '--no-show-raw-insn', '--start-address=0x733cd0',
                            '--stop-address=0x733e40', str(binary)], capture_output=True, text=True, timeout=120)
        res['disasm'] = r.stdout[-6000:] if r.returncode == 0 else f'objdump rc {r.returncode}: {r.stderr[:300]}'
    except Exception as exc:
        res['disasm'] = f'{type(exc).__name__}: {exc}'
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = str(Path(default_preload()).parent / 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if Path(stub).is_file() else '')
    inst = Instance('roomidx', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    t0 = time.perf_counter()
    try:
        b = inst.env.bridge
        b.reset_mode, b.character = 'floor', 0
        for n, seed in enumerate(range(first, first + count)):
            stage = 1 + n % 6
            try:
                inst.reset(seed)
                if stage > 1:
                    b.reset(phases=[[f'stage {stage}']], settle=2)
                rooms = [int(r.split(',')[0]) for r in b.lua(FLOOR_LUA).split(';')[2:]]
            except Exception as exc:
                res['errors'].append(f'seed {seed}: {type(exc).__name__}: {exc}')
                inst.relaunch(hard=True)
                continue
            # calls, in blocks on throwaway clones (a crash costs one clone)
            for a in range(-20, 181, 20):
                bnd = min(a + 19, 180)
                c = b.fork(tag='roomidx', lean=True, alarm=120)
                try:
                    text = c.lua(CALL.replace('__A__', str(a)).replace('__B__', str(bnd)))
                    for item in text.split(';'):
                        k, v = item.split(':', 1)
                        res['calls'].setdefault(k, set()).add(v if not v[0].isdigit() and v[0] != '-' else
                                                              ('self' if v.split(',')[0] == k else 'other') +
                                                              ',' + v.split(',')[2])
                except Exception as exc:
                    res['errors'].append(f'seed {seed} calls {a}..{bnd}: {type(exc).__name__}: {str(exc)[:200]}')
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
            for idx in rooms[:12]:
                c = b.fork(tag='roomidx', lean=True, alarm=120)
                try:
                    c.lua(ENTER % idx)
                    for t in c.lua(DOORS).split(','):
                        if t:
                            res['doors'][t] = res['doors'].get(t, 0) + 1
                except Exception as exc:
                    res['errors'].append(f'seed {seed} room {idx}: {type(exc).__name__}: {str(exc)[:200]}')
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
    finally:
        inst.kill()
    res['calls'] = {k: sorted(v) for k, v in sorted(res['calls'].items(), key=lambda kv: int(kv[0]))}
    door_values = sorted(int(t) for t in res['doors'])
    summary = dict(door_targets_min=door_values[0] if door_values else None,
                   door_targets_max=door_values[-1] if door_values else None,
                   door_targets_negative=sorted(v for v in door_values if v < 0),
                   calls_with_descriptor=[k for k, v in res['calls'].items() if any(x.startswith(('self', 'other')) for x in v)],
                   errors=len(res['errors']), seconds=round(time.perf_counter() - t0, 1))
    (out / 'room_idx.json').write_text(json.dumps(dict(res, summary=summary), indent=1, default=list))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
