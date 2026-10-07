"""Per-decision cost breakdown of a lean clone driven the way tok_sampler.worker_main drives it (2026-10-04).

Per seed: reset, `warm` decisions, then for each round and each variant (order alternating between rounds) a lean clone
of that same state plays the same sticky random actions until the player dies, the room is clear or --steps. Each
decision does what a sampler worker does: encode_row of the last observation into a ROW slot, the lag-1 hand-over with
a server (a separate echo process that answers every record at once, as TokSampler.poll_ready / reply do), then the
step command and the next observation (read + LeanDecoder.decode). perf_counter sections per decision:
  encode  encode_row and the record's bookkeeping fields
  pipe    the hand-over: the server's answer to the record before, then this record's message
  send    the step command (client side: building and writing it)
  wait    from the command written to the whole observation read (the game's frames and per-step work, both
          transports' wake-ups)
  decode  LeanDecoder.decode
plus the game process's CPU per decision (/proc/<pid>/schedstat), this process's CPU (time.process_time) and the echo
server's CPU. Variant `play`: the same actions in one `play` command (no per-step observation): the game's per-frame
cost. Variant names other than json / play are the optional fast paths (see --variants).

usage (from the bridge's python dir, PYTHONPATH=.):
  python abplus_bench_lean_breakdown.py --groups-file ../abplus/catalog/scaling2_groups.json --group normal \
      --seeds range:2147500000:8 --fpd 4,2 --variants json,play --out <dir>
"""
import argparse
import json
import multiprocessing as mp
import os
import select
import time
from pathlib import Path

import numpy as np

from abplus_probe_fork import split_code, sticky_actions
from goexplore_abplus import default_bridge_lua, default_preload, load_spec, parse_seeds
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.abplus_lean import LeanDecoder, read_lean
from isaac_bridge.tok_obs import ROW, EpisodeState, FastRow, encode_row, fast_row_function
from isaac_bridge import tok_sampler

SECTIONS = ('encode', 'pipe', 'send', 'wait', 'decode')


def cpu_ns(pid):
    try:
        return int(open(f'/proc/{pid}/schedstat').read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def echo_server(conn):
    """TokSampler's side of the hand-over, answering every record at once."""
    fd = conn.fileno()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while True:
        poller.poll()
        msg = os.read(fd, 5)
        if msg not in (tok_sampler._MSG0, tok_sampler._MSG1):
            return
        os.write(fd, tok_sampler._REPLY)


def set_mode(c, cmd, mode):
    c._send({"cmd": cmd} if mode is None else {"cmd": cmd, "mode": mode})
    msg = c._recv()
    if msg.get('type') != 'ok':
        raise RuntimeError(f'{cmd} failed: {msg}')
    return msg


def parse_variant(variant):
    """'<how><native step mode>[+opts]': how = json (the JSON step command), line (the step line), raw (the step line,
    no worker work); mode 0 / 1 = the bridge's native step off / on (absent: the bridge's default)."""
    head, _, opts = variant.partition('+')
    how = head.rstrip('0123456789')
    mode = head[len(how):]
    return how, (int(mode) if mode else None), set(opts.split('+')) - {''}


def run_steps(parent, variant, actions, fpd, conn, limit):
    how, mode, opts = parse_variant(variant)
    c = parent.fork(tag=f'bd-{variant}', lean=True, alarm=600)
    if mode is not None:
        set_mode(c, 'native_step', mode)
    dec = LeanDecoder()
    c._send({"cmd": "obs"})
    obs = read_lean(c, dec)
    st = EpisodeState(limit)
    rows = np.zeros(2, ROW)
    acc = dict.fromkeys(SECTIONS, 0.0)
    n, ents = 0, 0
    waiting = False
    pc = time.perf_counter
    cpu0, py0 = cpu_ns(c.pid), time.process_time()
    t_start = pc()
    raw = how == 'raw'
    line = how in ('line', 'raw')
    sock = c._sock
    enc = None
    if 'fast' in opts:   # tok_obs.FastRow: abp_row_encode on the raw payload
        enc = FastRow(fast_row_function(default_preload()), st, dec, 1, 2, 0)
        addrs = [rows[i:i + 1].ctypes.data for i in range(2)]
    item = obs
    for t, code in enumerate(actions):
        t0 = pc()
        if raw:   # no worker work: the game and the transport alone
            done = 0
            t1 = t2 = t0
        else:
            row = rows[t & 1:(t & 1) + 1]
            if enc is not None:
                done = enc.encode(item, row, addrs[t & 1], t)
            else:
                done = encode_row(obs, st, row, t)
                row['episode'], row['seed'], row['group'], row['first'] = 1, 2, 0, t == 0
            t1 = pc()
            if waiting:
                conn.recv_bytes()
            conn.send_bytes(b'1' if t & 1 else b'0')
            waiting = True
            t2 = pc()
        acc['encode'] += t1 - t0
        acc['pipe'] += t2 - t1
        if done or (enc is None and (obs.dead or obs.clear)):
            break
        move, shoot, bomb, item = split_code(code)
        if line:
            sock.sendall(b"S %d %d %d %d %d\n" % (fpd, move, shoot, bomb, 0))
        else:
            sock.sendall((json.dumps({"cmd": "step", "repeat": fpd, "move": move, "shoot": shoot, "bomb": bomb,
                                      "item": 0}, separators=(",", ":")) + "\n").encode())
        t3 = pc()
        hdr = c._read_line()
        payload = c._read_exact(int(hdr.split(b' ')[1]))
        t4 = pc()
        if enc is not None:
            item = payload
        else:
            obs = dec.decode(payload)
            ents += len(obs.entities)
        t5 = pc()
        acc['send'] += t3 - t2
        acc['wait'] += t4 - t3
        acc['decode'] += t5 - t4
        n += 1
    wall = pc() - t_start
    if waiting:
        conn.recv_bytes()
    game_cpu = (cpu_ns(c.pid) - cpu0) / 1e9
    py_cpu = time.process_time() - py0
    stats = set_mode(c, 'native_step', None) if mode is not None else {}
    c.close()
    prof = dict(kv.split('=') for kv in (stats.get('prof') or '').split() if '=' in kv)
    return dict(n=n, wall=wall, game_cpu=game_cpu, py_cpu=py_cpu, ents=ents, native_steps=stats.get('steps', 0),
                sp_n=int(prof.get('n', 0)), sp_build=int(prof.get('build_ns', 0)) / 1e9,
                sp_send=int(prof.get('send_ns', 0)) / 1e9, sp_recv=int(prof.get('recv_ns', 0)) / 1e9,
                sp_between=int(prof.get('between_ns', 0)) / 1e9, **acc)


def run_play(parent, actions, fpd):
    c = parent.fork(tag='bd-play', lean=True, alarm=600)
    dec = LeanDecoder()
    c._send({"cmd": "obs"})
    read_lean(c, dec)
    cpu0 = cpu_ns(c.pid)
    t0 = time.perf_counter()
    c._send({"cmd": "play", "actions": [int(a) for a in actions], "repeat": fpd, "stop_clear": True})
    read_lean(c, dec)
    wall = time.perf_counter() - t0
    game_cpu = (cpu_ns(c.pid) - cpu0) / 1e9
    c.close()
    return dict(n=len(actions), wall=wall, game_cpu=game_cpu, py_cpu=0.0, ents=0, native_steps=0, sp_n=0, sp_build=0.0,
                sp_send=0.0, sp_recv=0.0, sp_between=0.0, **dict.fromkeys(SECTIONS, 0.0))


def lean_profile(parent, n=200):
    c = parent.fork(tag='bd-prof', lean=True, alarm=600)
    c._send({"cmd": "obs"})
    read_lean(c, LeanDecoder())
    c._send({"cmd": "lean_profile", "n": n})
    msg = c._recv()
    c.close()
    return msg.get('ms')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', default='')
    p.add_argument('--group', default='normal')
    p.add_argument('--tasks', default='')
    p.add_argument('--seconds', type=float, default=0.0)
    p.add_argument('--seeds', default='range:2147500000:8')
    p.add_argument('--warm', type=int, default=10)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--fpd', default='4,2')
    p.add_argument('--variants', default='json,play')
    p.add_argument('--profile', action='store_true', help='also run the in-game lean_profile per seed')
    p.add_argument('--step-prof', action='store_true', help='ABP_STEP_PROF=1: thread CPU per part of a native step')
    p.add_argument('--port', type=int, default=34000)
    p.add_argument('--name', default='stpbd')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(args)
    os.environ.update(FORK_ENV)
    if args.step_prof:
        os.environ['ABP_STEP_PROF'] = '1'
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=default_preload(), al_stopped=True, nice=0,
                   stub_list=stub if Path(stub).is_file() else '')
    ctx = mp.get_context('spawn')
    ours, theirs = ctx.Pipe()
    server = ctx.Process(target=echo_server, args=(theirs,), daemon=True)
    server.start()
    theirs.close()
    inst = Instance(args.name, args.port, cfg, spec)
    fpds = [int(v) for v in args.fpd.split(',')]
    variants = args.variants.split(',')
    acc = {}
    rows, profiles = [], []
    load0 = os.getloadavg()
    server_cpu0 = cpu_ns(server.pid)
    try:
        for seed in parse_seeds(args.seeds):
            rng = np.random.default_rng(seed)
            warm, actions = sticky_actions(rng, args.warm), sticky_actions(rng, args.steps)
            inst.reset(seed)
            _, _, stop = inst.env.bridge.play(warm, repeat=cfg.frames_per_decision, stop_clear=True)
            if stop != 'done':
                continue
            row = dict(seed=seed)
            limit = 10 ** 6
            for r in range(args.rounds):
                for f in fpds:
                    order = variants if r % 2 == 0 else variants[::-1]
                    n_steps = None
                    for v in order:
                        if v == 'play':
                            continue
                        res = run_steps(inst.env.bridge, v, actions, f, ours, limit)
                        n_steps = res['n']
                        a = acc.setdefault((f, v), dict.fromkeys(res, 0.0))
                        for k, val in res.items():
                            a[k] += val
                        row[f'fpd{f}_{v}_ms'] = round(1000 * res['wall'] / max(res['n'], 1), 4)
                    if 'play' in variants and n_steps:
                        res = run_play(inst.env.bridge, actions[:n_steps], f)
                        a = acc.setdefault((f, 'play'), dict.fromkeys(res, 0.0))
                        for k, val in res.items():
                            a[k] += val
                        row[f'fpd{f}_play_ms'] = round(1000 * res['wall'] / max(res['n'], 1), 4)
            if args.profile:
                prof = lean_profile(inst.env.bridge)
                profiles.append(prof)
                row['profile'] = prof
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        inst.close()
        server_cpu = (cpu_ns(server.pid) - server_cpu0) / 1e9
        try:
            ours.send_bytes(b'x')
        except OSError:
            pass
        server.join(5)
        summary = dict(group=args.group, seeds=len(rows), load_start=load0, load_end=os.getloadavg(),
                       server_cpu_s=round(server_cpu, 3))
        for (f, v), a in acc.items():
            n = a['n']
            if not n:
                continue
            d = dict(decisions=int(n), ms_per_decision=round(1000 * a['wall'] / n, 4),
                     game_cpu_ms=round(1000 * a['game_cpu'] / n, 4), py_cpu_ms=round(1000 * a['py_cpu'] / n, 4),
                     frames_per_s=round(f * n / a['wall'], 1), x_real_time=round(f * n / a['wall'] / 30, 1),
                     entities_mean=round(a['ents'] / n, 2), native_steps=int(a['native_steps']))
            for k in SECTIONS:
                d[k + '_ms'] = round(1000 * a[k] / n, 4)
            if a['sp_n']:   # the game's own thread CPU per native step (ABP_STEP_PROF)
                for k in ('build', 'send', 'recv'):
                    d['game_' + k + '_ms'] = round(1000 * a['sp_' + k] / a['native_steps'], 4)
                d['game_between_ms'] = round(1000 * a['sp_between'] / a['sp_n'], 4)
            summary[f'fpd{f}_{v}'] = d
        if profiles:
            keys = sorted({k for pr in profiles if pr for k in pr})
            summary['lean_profile_ms'] = {k: round(float(np.mean([pr[k] for pr in profiles if pr and k in pr])), 5)
                                          for k in keys}
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))
        print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
