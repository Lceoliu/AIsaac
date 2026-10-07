"""The root instance's first-build hang (2026-10-06): launch / connect / floor reset over and over, as a tok floor
worker's root does, and catch every build that takes longer than --timeout s.

--mode loop: --workers threads, each its own instance (name <name><i>, port --port + i): launch, the first build
  (connect + floor reset of a seed), --resets more floor resets, kill; again until --minutes are over. A build past
  --timeout: tok_sampler.stall_evidence (log.txt, stacks, /proc, ss) and the instance killed. --hold-ephemeral N holds N
  extra TCP connections open meanwhile (the parked clones' connections of a whole-floor training take ephemeral local
  ports, about two per clone) so that bridge ports inside the ephemeral range can collide as in training.
--mode hold: the collision on purpose: for each of --trials ports, a connection is opened whose local port is that
  port, then an instance is launched on it; with --no-port-file as before (ISAAC_RL_PORT_FILE not set), else with the
  bridge's free-port fallback. Reports whether the first build finished, how long it took, the port it ended on, and
  what log.txt says.
Writes one JSON line per build to <out>.jsonl and a summary to stdout.
usage (in the bridge's python/, PYTHONPATH=.): python abplus_probe_launch_hang.py --mode loop --workers 16 --minutes 30
  --port 40100 --name memh --out ../runs/hang-loop
"""
import argparse
import json
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import ABP_HOME, FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_sampler import apply_instance_defaults, default_stub_list, stall_evidence


class Shim:
    """What stall_evidence reads of a StartBuilder."""

    def __init__(self, number, seed):
        self.number, self.seed, self.t0, self.retried = number, seed, time.monotonic(), None


def make_cfg(args):
    preload = default_preload()
    return GxConfig(bridge_lua=default_bridge_lua(), preload=preload, al_stopped=True, nice=args.nice,
                    frames_per_decision=4, start_hp=6, bombs=1, stub_list=default_stub_list(preload),
                    die_with_parent=True)


def build(inst, seed, timeout):
    """inst.reset(seed) in a thread with a time limit: (seconds, error text or None, finished)."""
    box = {}

    def run():
        t = time.monotonic()
        try:
            inst.env.bridge.reset_mode = 'floor'
            inst.reset(seed)
            box['err'] = None
        except Exception as exc:
            box['err'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        box['s'] = time.monotonic() - t
    th = threading.Thread(target=run, daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(timeout)
    if th.is_alive():
        return time.monotonic() - t0, 'timeout', False
    return box['s'], box['err'], True


def log_tail(name):
    try:
        text = (ABP_HOME / 'instances' / name.lower() / 'data' / 'binding of isaac afterbirth+' / 'log.txt').read_text(
            errors='replace')
        return [ln for ln in text.splitlines() if 'AbpRLBridge' in ln][-4:]
    except OSError:
        return []


def loop_worker(i, args, cfg, spec, out, lock, stop, counts):
    name, port = f'{args.name}{i}', args.port + i
    rng = random.Random(1000 + i)
    k = 0
    while not stop.is_set():
        k += 1
        inst = Instance(name, port, cfg, spec)
        if args.no_port_file:
            inst.env.bridge.port_file = None
        for j in range(1 + args.resets):
            seed = rng.randrange(1, 2 ** 31 - 1)
            shim = Shim(k * 100 + j, seed)
            s, err, finished = build(inst, seed, args.timeout)
            rec = dict(t=time.time(), worker=i, launch=k, build=j, seed=seed, seconds=round(s, 3), err=err,
                       port=port, bridge_port=getattr(inst.env.bridge, 'port', None) if inst.env else None)
            if not finished or err:
                rec['evidence'] = stall_evidence(inst, shim, 'timeout' if not finished else 'failed', i)
                rec['log'] = log_tail(name)
            with lock:
                out.write(json.dumps(rec) + '\n')
                out.flush()
                key = ('first' if j == 0 else 'later') + ('_ok' if finished and not err else '_bad')
                counts[key] = counts.get(key, 0) + 1
                if rec['bridge_port'] and rec['bridge_port'] != port:
                    counts['port_moves'] = counts.get('port_moves', 0) + 1
            if not finished or err or stop.is_set():
                break
        inst.kill()


def hold_connection(port):
    """A connection whose local port is `port` (as a parked clone's connection may get), plus its listener."""
    lst = socket.socket()
    lst.bind(('127.0.0.1', 0))
    lst.listen(4)
    c = socket.socket()
    c.bind(('127.0.0.1', port))
    c.connect(lst.getsockname())
    a, _ = lst.accept()
    return [c, a, lst]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('loop', 'hold'), default='loop')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--minutes', type=float, default=30.0)
    p.add_argument('--resets', type=int, default=2)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--trials', type=int, default=4)
    p.add_argument('--no-port-file', action='store_true')
    p.add_argument('--hold-ephemeral', type=int, default=0)
    p.add_argument('--port', type=int, default=40100)
    p.add_argument('--name', default='memh')
    p.add_argument('--nice', type=int, default=10)
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    os.environ['ABP_WATCH_PID'] = str(os.getpid())
    os.environ.pop('ISAAC_RL_PORT_FILE', None)
    cfg = make_cfg(args)
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    if args.no_port_file:   # launch without the variable: as before 2026-10-06
        import isaac_bridge.abplus_goexplore as gx
        real = gx.launch_abplus

        def launch(*a, extra_env=None, **kw):
            extra_env = {k: v for k, v in (extra_env or {}).items() if k != 'ISAAC_RL_PORT_FILE'}
            return real(*a, extra_env=extra_env, **kw)
        gx.launch_abplus = launch
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out + '.jsonl', 'a')
    held = []
    if args.hold_ephemeral:
        lst = socket.socket()
        lst.bind(('127.0.0.1', 0))
        lst.listen(1024)
        for _ in range(args.hold_ephemeral):
            c = socket.create_connection(lst.getsockname())
            a, _ = lst.accept()
            held += [c, a]
        held.append(lst)
    if args.mode == 'hold':
        res = []
        for k in range(args.trials):
            port = args.port + k
            hold = hold_connection(port)
            inst = Instance(f'{args.name}{k}', port, cfg, spec)
            if args.no_port_file:
                inst.env.bridge.port_file = None
            s, err, finished = build(inst, 1000 + k, args.timeout)
            rec = dict(mode='hold', no_port_file=args.no_port_file, port=port, seconds=round(s, 2), err=err,
                       finished=finished, bridge_port=getattr(inst.env.bridge, 'port', None) if inst.env else None,
                       log=log_tail(inst.name))
            pid = inst.proc.pid
            try:   # the game's CPU share over 2 s
                a = sum(int(v) for v in open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[11:13])
                time.sleep(2.0)
                b = sum(int(v) for v in open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[11:13])
                rec['cpu_share'] = round((b - a) / os.sysconf('SC_CLK_TCK') / 2.0, 2)
            except (OSError, ValueError):
                pass
            if not finished or err:
                rec['evidence'] = stall_evidence(inst, Shim(1, 1000 + k), 'timeout' if not finished else 'failed', k)
            inst.kill()
            for s_ in hold:
                s_.close()
            out.write(json.dumps(rec) + '\n')
            out.flush()
            print(json.dumps(rec), flush=True)
            res.append(rec)
        print('SUMMARY', json.dumps(dict(trials=len(res), finished=sum(r['finished'] and not r['err'] for r in res),
                                         moved=sum(bool(r['bridge_port']) and r['bridge_port'] != r['port']
                                                   for r in res))), flush=True)
        return
    lock, stop, counts = threading.Lock(), threading.Event(), {}
    threads = [threading.Thread(target=loop_worker, args=(i, args, cfg, spec, out, lock, stop, counts), daemon=True)
               for i in range(args.workers)]
    t0 = time.time()
    for th in threads:
        th.start()
        time.sleep(0.05)
    while time.time() - t0 < args.minutes * 60:
        time.sleep(10)
        print(time.strftime('%H:%M:%S'), json.dumps(counts), flush=True)
    stop.set()
    for th in threads:
        th.join(args.timeout + 30)
    print('SUMMARY', json.dumps(counts), flush=True)


if __name__ == '__main__':
    sys.exit(main())
