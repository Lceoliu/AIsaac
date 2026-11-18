"""Cross-tree byte check (the coordinator, 2026-11-18: exp/visible-v3-c 1de5517 against mainline 93a1c4f before the
merge): with every visible-v3 / G switch at its default (ISAAC_RL_VISIBLE_V3 and ISAAC_RL_NATIVE_VISIBLE unset: the old
model's training setting, native step), the same seeds and the same action sequence must give, in both trees, the same
lean payload bytes at every step, the same ROW bytes from encode_row (tok_obs, the worker's path; FastRow's native row
encoder is off in both trees: fast_row_function returns None), and the same hidden-state digest at the end.

Two parts:
  record   run inside one tree (its bridge, its libabp_turbo.so, its isaac_bridge) by the CPU runner:
           PROBE=<this file> cpurun.sh <tree> <name> [--seeds A:N] [--steps N]
           per seed and lean_items off / on: a floor clone (floor mode, stage 1 + k mod 6), stepped --steps decisions
           with step lines (repeat 4) from a seeded action sequence: moves, shots, bombs, the active item (item 1) and
           the pocket (pill 1, the 6-number line; a pill and a card given at the start and again every 30 decisions,
           an active item with its charge at the start), a ChangeRoom into the floor's next room every 25 decisions.
           Writes <out>/payloads.bin + rows.bin (back to back) and index.jsonl (per step: offsets, lengths, sha256s).
  compare  python3 cross_tree_exact.py compare <out A> <out B> (every payload, every ROW, every digest; the first
           differences; numpy for the item / pill step counts), or by the CPU runner:
           PROBE=<this file> cpurun.sh <tree> <name> --compare <out A> <out B>
  analyze  PROBE=<this file> cpurun.sh <tree> <name> --analyze <out A> <out B>: part by part (see analyze()): two
           launches of the same tree differ in the payload bytes in two places that are not the tree's doing -- the
           header's logic_frames (a process-wide counter) and the terrain block's JSON text (Lua's string hash seed
           orders the keys); every other part byte for byte, those two as increments / parsed values
"""
import hashlib
import json
import os
import sys

ITEM_SETUP = ("local p = Isaac.GetPlayer(0) p:AddCollectible(45, 6, false) p:AddPill(3) p:AddCard(2) "
              "p:FullCharge() return 'ok'")
POCKET_AGAIN = "local p = Isaac.GetPlayer(0) p:AddPill(3) p:AddCard(2) p:FullCharge() return 'ok'"
ENTER_LUA = "Game():ChangeRoom(%d) return 'ok'"


def actions(seed, steps):
    import numpy as np
    rng = np.random.default_rng(seed)
    out, move, shoot = [], 0, 0
    for t in range(steps):
        if t % 6 == 0:
            move, shoot = int(rng.integers(0, 9)), int(rng.integers(0, 5))
        bomb = int(rng.random() < 0.03)
        item = int(t % 11 == 5)
        pill = int(t % 17 == 9)
        out.append((move, shoot, bomb, item, pill))
    return out


def record(argv):
    import argparse
    sys.path.insert(0, os.getcwd())   # (the tree's python dir: its isaac_bridge, its probes)
    for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE', 'ISAAC_RL_TRIGGER_ONCE'):   # (+ trigger, 2026-11-18)
        os.environ.pop(k, None)
    import numpy as np
    from abplus_probe_fork import digest
    from abplus_probe_native_obs import read_raw
    from goexplore_abplus import default_bridge_lua, default_preload, load_spec
    from isaac_bridge.abplus import FLOOR_LUA, FORK_ENV
    from isaac_bridge.abplus_goexplore import GxConfig, Instance
    from isaac_bridge.abplus_lean import LeanDecoder
    from isaac_bridge.tok_obs import ROW, EpisodeState, encode_row

    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--seeds', default='2148500000:8')
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--hop', type=int, default=25)
    ap.add_argument('--port', type=int, default=44600)
    ap.add_argument('--run', type=int, default=0, help='1: EpisodeState(run=True), as train_tok --mode run')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    stub = os.path.join(os.path.dirname(default_preload()), 'stub_render_h.txt')
    os.environ.update(FORK_ENV)
    for k, v in {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}.items():
        os.environ.setdefault(k, v)
    gx = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                  frames_per_decision=4, start_hp=6, bombs=1, stub_list=stub if os.path.isfile(stub) else '')
    inst = Instance('xtree', args.port, gx, spec)
    first, count = (int(v) for v in args.seeds.split(':'))
    fp, fr = open(os.path.join(out, 'payloads.bin'), 'wb'), open(os.path.join(out, 'rows.bin'), 'wb')
    fi = open(os.path.join(out, 'index.jsonl'), 'w')
    env = {k: os.environ.get(k) for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE', 'ISAAC_RL_TRIGGER_ONCE', 'ABP_FAST',
                                          'ISAAC_RL_LEAN_ITEMS')}
    fi.write(json.dumps(dict(env=env, row_itemsize=ROW.itemsize, args=vars(args))) + '\n')
    try:
        for k, seed in enumerate(range(first, first + count)):
            stage = 1 + k % 6
            b = inst.env.bridge
            b.reset_mode, b.character = 'floor', 0
            inst.reset(seed)
            b = inst.env.bridge
            if stage > 1:
                b.reset(phases=[[f'stage {stage}']], settle=2)
            rooms = [int(r.split(',')[0]) for r in b.lua(FLOOR_LUA).split(';')[2:]]
            acts = actions(seed, args.steps)
            for items in (False, True):
                c = b.fork(tag='xtree', lean=True, alarm=1800)
                try:
                    c._send({"cmd": "lean_items", "enabled": items})
                    c._recv()
                    setup = c.lua(ITEM_SETUP)
                    dec = LeanDecoder()
                    st = EpisodeState(limit=10 ** 6, floor=True, stall=120, items=items, run=bool(args.run))
                    row = np.zeros(1, ROW)
                    c._send({"cmd": "obs"})
                    payload = read_raw(c)
                    for t in range(args.steps + 1):
                        row[:] = 0
                        done = encode_row(dec.decode(payload), st, row, t)
                        rb = row.tobytes()
                        fi.write(json.dumps(dict(seed=seed, items=items, t=t, p_off=fp.tell(), p_len=len(payload),
                                                 r_off=fr.tell(), r_len=len(rb), done=int(done),
                                                 p_sha=hashlib.sha256(payload).hexdigest()[:16],
                                                 r_sha=hashlib.sha256(rb).hexdigest()[:16])) + '\n')
                        fp.write(payload)
                        fr.write(rb)
                        if t == args.steps:
                            break
                        if t % 30 == 29:
                            c.lua(POCKET_AGAIN)
                        if args.hop and t % args.hop == args.hop - 1 and rooms:
                            c.lua(ENTER_LUA % rooms[(t // args.hop) % len(rooms)])
                        m, s, bo, it, pi = acts[t]
                        if pi:
                            c._sock.sendall(b'S 4 %d %d %d %d %d\n' % (m, s, bo, it, pi))
                        else:
                            c._sock.sendall(b'S 4 %d %d %d %d\n' % (m, s, bo, it))
                        payload = read_raw(c)
                    fi.write(json.dumps(dict(seed=seed, items=items, digest=digest(c)[0], setup=setup)) + '\n')
                finally:
                    c.close()
            fi.flush()
    finally:
        fp.close()
        fr.close()
        fi.close()
        inst.kill()
    print('RECORDED', json.dumps(dict(seeds=count, steps=args.steps)), flush=True)


def compare(a, b):
    def load(d):
        lines = [json.loads(x) for x in open(os.path.join(d, 'index.jsonl'))]
        return lines[0], [x for x in lines[1:] if 't' in x], [x for x in lines[1:] if 'digest' in x]
    ha, sa, da = load(a)
    hb, sb, db = load(b)
    pa, pb = open(os.path.join(a, 'payloads.bin'), 'rb').read(), open(os.path.join(b, 'payloads.bin'), 'rb').read()
    ra, rb = open(os.path.join(a, 'rows.bin'), 'rb').read(), open(os.path.join(b, 'rows.bin'), 'rb').read()
    res = dict(steps=[len(sa), len(sb)], payload_equal=0, row_equal=0, digest_equal=0, digests=[len(da), len(db)],
               item_steps=0, pill_steps=0, first=[], env=[ha['env'], hb['env']],
               row_itemsize=[ha['row_itemsize'], hb['row_itemsize']])
    acts = {}
    for x, y in zip(sa, sb):
        key = (x['seed'], x['items'], x['t'])
        if key != (y['seed'], y['items'], y['t']):
            res['first'].append(dict(misaligned=[key, (y['seed'], y['items'], y['t'])]))
            break
        if x['seed'] not in acts:
            acts[x['seed']] = actions(x['seed'], ha['args']['steps'])
        if x['t'] >= 1:
            _, _, _, it, pi = acts[x['seed']][x['t'] - 1]
            res['item_steps'] += it
            res['pill_steps'] += pi
        p1, p2 = pa[x['p_off']:x['p_off'] + x['p_len']], pb[y['p_off']:y['p_off'] + y['p_len']]
        r1, r2 = ra[x['r_off']:x['r_off'] + x['r_len']], rb[y['r_off']:y['r_off'] + y['r_len']]
        if p1 == p2:
            res['payload_equal'] += 1
        elif len(res['first']) < 10:
            i = next((i for i, (u, v) in enumerate(zip(p1, p2)) if u != v), min(len(p1), len(p2)))
            res['first'].append(dict(what='payload', key=key, lens=[len(p1), len(p2)], index=i))
        if r1 == r2:
            res['row_equal'] += 1
        elif len(res['first']) < 10:
            i = next((i for i, (u, v) in enumerate(zip(r1, r2)) if u != v), min(len(r1), len(r2)))
            res['first'].append(dict(what='row', key=key, index=i))
    for x, y in zip(da, db):
        res['digest_equal'] += x['digest'] == y['digest'] and (x['seed'], x['items']) == (y['seed'], y['items'])
    n = len(sa)
    res['ok'] = (len(sa) == len(sb) and res['payload_equal'] == n and res['row_equal'] == n and len(da) == len(db)
                 and res['digest_equal'] == len(da) and n > 0)
    print(json.dumps(res, indent=1))
    return 0 if res['ok'] else 1


def sections(payload, sizes):
    """A mode-0 lean payload cut into its parts (abplus_lean.LeanDecoder's layout): {name: bytes}; the terrain block's two
    JSON texts apart (json.encode of Lua tables: the key order follows the process's string hash seed, so the text may
    differ between two launches of the same tree while the values are the same)."""
    import struct
    out = {'magic': payload[0:4], 'logic_frames': payload[4:8], 'header_rest': payload[8:16]}
    flags, n_players, n_doors, n_lasers = payload[12], payload[13], payload[14], payload[15]
    off = 16
    for name, size in (('room', 48), ('totals', 32), ('players', sizes['player'] * n_players),
                       ('doors', sizes['door'] * n_doors)):
        out[name] = payload[off:off + size]
        off += size
    n = struct.unpack_from('<H', payload, off)[0]
    out['entities'] = payload[off:off + 2 + sizes['entity'] * n]
    off += 2 + sizes['entity'] * n
    start = off
    for _ in range(n_lasers):
        k = struct.unpack_from('<q', payload, off + 57)[0]
        off += 65 + 16 * k
    out['lasers'] = payload[start:off]
    if flags & 2:
        version, n = struct.unpack_from('<II', payload, off)
        n2 = struct.unpack_from('<I', payload, off + 8 + n)[0]
        out['terrain_version'] = payload[off:off + 4]
        out['terrain_json'] = (payload[off + 8:off + 8 + n], payload[off + 12 + n:off + 12 + n + n2])
        off += 12 + n + n2
    if flags & 8:
        m = payload[off]
        out['map'] = payload[off:off + 1 + sizes['map'] * m]
        off += 1 + sizes['map'] * m
    if flags & 16:
        h = struct.unpack_from('<B', payload, off)[0]
        out['inventory'] = payload[off:off + 18 + sizes['inv'] * h]
        off += 18 + sizes['inv'] * h
    if flags & 32:   # the v2 visible block (Lua JSON)
        m = struct.unpack_from('<I', payload, off)[0]
        out['visible_json'] = payload[off + 4:off + 4 + m]
        off += 4 + m
    out['rest'] = payload[off:]
    return out


def json_paths(a, b, path='', out=None):
    """The paths (list indices as []) where two JSON values differ."""
    out = set() if out is None else out
    if type(a) != type(b):
        out.add(path)
    elif isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(path + '.' + k)
            else:
                json_paths(a[k], b[k], path + '.' + k, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.add(path + '#len')
        for x, y in zip(a, b):
            json_paths(x, y, path + '[]', out)
    elif a != b:
        out.add(path)
    return out


def analyze(a, b):
    """Part by part: every binary part byte for byte; the terrain JSON as text and as parsed values; logic_frames as its
    step-to-step increments within an episode (a process-wide counter: its start differs between launches)."""
    import struct
    sys.path.insert(0, os.getcwd())
    from isaac_bridge import abplus_lean as AL
    sizes = dict(player=AL.PLAYER.itemsize, door=AL.DOOR.itemsize, entity=AL.ENTITY.itemsize,
                 map=AL.MAP_ROOM.itemsize, inv=AL.INV_ITEM.itemsize)

    def load(d):
        lines = [json.loads(x) for x in open(os.path.join(d, 'index.jsonl'))]
        return [x for x in lines[1:] if 't' in x], open(os.path.join(d, 'payloads.bin'), 'rb').read()
    sa, pa = load(a)
    sb, pb = load(b)
    res = dict(steps=[len(sa), len(sb)], bytes_equal=0, parts={}, terrain_json_text_equal=0, terrain_json_value_equal=0,
               terrain_blocks=0, logic_frames_increment_equal=0, first=[], player_fields={}, visible_blocks=0,
               visible_text_equal=0, visible_paths={})
    prev = None
    for x, y in zip(sa, sb):
        p1, p2 = pa[x['p_off']:x['p_off'] + x['p_len']], pb[y['p_off']:y['p_off'] + y['p_len']]
        res['bytes_equal'] += p1 == p2
        s1, s2 = sections(p1, sizes), sections(p2, sizes)
        for k in set(s1) | set(s2):
            if k in ('logic_frames', 'terrain_json', 'visible_json'):
                continue
            if k == 'players' and s1.get(k) != s2.get(k) and len(s1.get(k) or b'') != len(s2.get(k) or b''):
                res['player_fields']['<count>'] = res['player_fields'].get('<count>', 0) + 1
            if k == 'players' and s1.get(k) != s2.get(k) and len(s1[k]) == len(s2[k]):
                import numpy as np
                u, v = np.frombuffer(s1[k], AL.PLAYER), np.frombuffer(s2[k], AL.PLAYER)
                for f in AL.PLAYER.names:
                    if u[f].tobytes() != v[f].tobytes():
                        res['player_fields'][f] = res['player_fields'].get(f, 0) + 1
            st = res['parts'].setdefault(k, [0, 0])
            st[0] += 1
            if s1.get(k) == s2.get(k):
                st[1] += 1
            elif len(res['first']) < 10:
                res['first'].append(dict(part=k, key=[x['seed'], x['items'], x['t']]))
        if 'terrain_json' in s1 or 'terrain_json' in s2:
            res['terrain_blocks'] += 1
            t1, t2 = s1.get('terrain_json'), s2.get('terrain_json')
            res['terrain_json_text_equal'] += t1 == t2
            ok = t1 is not None and t2 is not None and all(json.loads(u) == json.loads(v) for u, v in zip(t1, t2))
            res['terrain_json_value_equal'] += ok
            if not ok and len(res['first']) < 10:
                res['first'].append(dict(part='terrain_json values', key=[x['seed'], x['items'], x['t']]))
        if 'visible_json' in s1 or 'visible_json' in s2:
            res['visible_blocks'] += 1
            v1, v2 = s1.get('visible_json'), s2.get('visible_json')
            res['visible_text_equal'] += v1 == v2
            if v1 is not None and v2 is not None:
                for q in json_paths(json.loads(v1), json.loads(v2)):
                    res['visible_paths'][q] = res['visible_paths'].get(q, 0) + 1
            else:   # (a block on one side only)
                res['visible_paths']['<present on one side>'] = res['visible_paths'].get('<present on one side>', 0) + 1
        lf1, lf2 = struct.unpack('<I', s1['logic_frames'])[0], struct.unpack('<I', s2['logic_frames'])[0]
        key = (x['seed'], x['items'])
        if prev is not None and prev[0] == key:
            res['logic_frames_increment_equal'] += (lf1 - prev[1]) == (lf2 - prev[2])
        else:
            res['logic_frames_increment_equal'] += 1   # (an episode's first payload: nothing to compare)
        prev = (key, lf1, lf2)
    n = len(sa)
    # (2026-11-18, Codex round 3) ok: records on both sides (none is no pass); every part equal byte for byte but the
    # launch-dependent ones -- players only in the id field (p.Index), logic_frames as increments, the terrain and
    # visible JSON as values (no field of a visible block differing, none on one side only)
    players_ok = set(res['player_fields']) <= {'id'}
    res['ok'] = (n > 0 and len(sa) == len(sb) and players_ok and not res['visible_paths'] and
                 all(v[0] == v[1] == n or (k == 'players' and v[0] == n) or
                     (k not in ('magic', 'header_rest', 'room', 'totals', 'players', 'doors', 'entities', 'lasers',
                                'rest') and v[0] == v[1]) for k, v in res['parts'].items())
                 and res['terrain_json_value_equal'] == res['terrain_blocks'] and res['logic_frames_increment_equal'] == n)
    print(json.dumps(res, indent=1))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    if '--analyze' in sys.argv:   # (by the CPU runner: the tree's abplus_lean for the record sizes)
        i = sys.argv.index('--analyze')
        sys.exit(analyze(sys.argv[i + 1], sys.argv[i + 2]))
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    if '--compare' in sys.argv:   # (by the CPU runner, where numpy is: the item / pill step counts need the sequence)
        i = sys.argv.index('--compare')
        sys.exit(compare(sys.argv[i + 1], sys.argv[i + 2]))
    record(sys.argv[1:])
