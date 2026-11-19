"""Close-race stress check (2026-11-18; Codex rounds 5 / 9: the check must show it hit the fault points and that the
worker came back, not only that no exit 86 came). Floor workers with the training's teacher config, short floors (their
episodes end within a round). Two runs, by --v3:
  --v3 0 (the default arm, ISAAC_RL_VISIBLE_V3 unset): rounds in turn
    play        random actions until the workers' records flow, a little more, then close()
    reset-kill  while a worker's start-state builder is inside Instance.reset (the real connect / hello / reset I/O --
                no hold added), that worker's own game process groups are killed (SIGKILL); the reset must end in an
                error and the same worker must give records again afterwards (it relaunched); then close()
    early       close() 0.5 to 3 s after the start
    and no mode query may be sent (mode 0: 05f1622's protocol without the query)
  --v3 1 (a v3 arm, mode 1: the Lua raw block): play / query-kill / early, where query-kill holds the real mode query
    after it was sent (0.3 s before its answer is read), kills that worker's games there, and needs the query to end
    in an error (a BridgeError / OSError: retried, not fatal) and the worker to give records again
The workers are traced by a sitecustomize.py this tool puts first on their PYTHONPATH (no production change):
Instance.reset and check_visible_v3 (start / end markers with the outcome; the query's hold in --v3 1; every visible_v3
query sent), and the floor worker's death-teaching line (a line tracer on floor_worker_main: each time the line
'fatal = death_teaching and done == 2 ...' runs -- the branch A1's smokes hit, by its own event).
Pass (coverage and outcome apart): >= 30 rounds; >= 8 play rounds with records; >= 8 kill rounds whose kill ended the
traced operation in an error and whose worker gave records afterwards; >= 8 early rounds; >= 8 death-teaching line
events; mode 0: no mode query sent; no worker exit code None, no exit 86, no VisibleV3Fatal, no fatal file, no
tracer error.

usage (bridge python dir, PYTHONPATH=.; the CPU runner adds --groups-file / --out):
  python tools/v3_close_stress.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir> [--rounds 36]
         [--v3 0|1] [--floor-seconds 5] [--teacher 1]
Writes <out>/rounds.jsonl and <out>/summary.json; prints SUMMARY; exit 0 on a pass, 1 otherwise.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE', 'ISAAC_RL_TRIGGER_ONCE'):
    os.environ.pop(k, None)

SITE = '''
import os, sys, time
_d = os.environ.get('V3_STRESS_TRACE')


def _mark(name, text):
    with open(os.path.join(_d, name), 'a') as f:
        f.write(text)


if _d:
    try:
        _n = [0]
        import isaac_bridge.abplus_goexplore as _gx
        _reset = _gx.Instance.reset

        def _traced_reset(self, *a, **kw):
            _n[0] += 1
            tag = 'r-%d-%d' % (os.getpid(), _n[0])
            _mark(tag + '.start', repr(time.time()))
            out = 'ok'
            try:
                return _reset(self, *a, **kw)
            except BaseException as e:
                out = type(e).__name__
                raise
            finally:
                _mark(tag + '.end', '%r %s' % (time.time(), out))
        _gx.Instance.reset = _traced_reset
        import isaac_bridge.abplus as _ab
        _check = _ab.check_visible_v3

        def _traced_check(client, mode=None):
            _n[0] += 1
            tag = 'q-%d-%d' % (os.getpid(), _n[0])
            send, recv = client._send, client._recv

            def _send(msg):
                if isinstance(msg, dict) and msg.get('cmd') == 'visible_v3':
                    _mark('sent-%d' % os.getpid(), 'x')
                return send(msg)

            def _recv():   # (the query sent: held here, before its answer is read)
                _mark(tag + '.start', repr(time.time()))
                time.sleep(0.3)
                out = 'ok'
                try:
                    return recv()
                except BaseException as e:
                    out = type(e).__name__
                    raise
                finally:
                    _mark(tag + '.end', '%r %s' % (time.time(), out))
            client._send, client._recv = _send, _recv
            try:
                return _check(client, mode)
            finally:
                client._send, client._recv = send, recv
        _ab.check_visible_v3 = _traced_check
        import inspect
        import isaac_bridge.tok_floor as _tf
        _src, _first = inspect.getsourcelines(_tf.floor_worker_main)
        _line = [_first + i for i, l in enumerate(_src) if 'death_teaching and done == 2' in l][0]
        _code = _tf.floor_worker_main.__code__

        def _local(frame, event, arg):
            if event == 'line' and frame.f_lineno == _line:
                _mark('death-%d' % os.getpid(), 'x')
            return _local

        def _global(frame, event, arg):
            return _local if frame.f_code is _code else None
        sys.settrace(_global)
    except Exception as e:
        _mark('site-error-%d' % os.getpid(), repr(e))
'''

import numpy as np   # noqa: E402

from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig   # noqa: E402


def children_groups(pids):
    """The process groups of the given processes' children (each game runs in its own session: its group)."""
    groups = set()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / 'stat').read_text().rsplit(')', 1)[1].split()
            ppid, pgrp = int(fields[1]), int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if ppid in pids and pgrp == int(d.name):
            groups.add(pgrp)
    return groups


def kill_groups(groups):
    n = 0
    for g in groups:
        try:
            os.killpg(g, signal.SIGKILL)
            n += 1
        except OSError:
            pass
    return n


def traced(trace, prefix):
    """The traced operations: {tag: (pid, start, end or None, outcome or None)}."""
    out = {}
    for p in trace.glob(f'{prefix}-*.start'):
        tag = p.name[:-6]
        try:
            start = float(p.read_text())
        except (OSError, ValueError):
            continue
        end, outcome = None, None
        e = trace / f'{tag}.end'
        if e.is_file():
            try:
                t, outcome = e.read_text().split(' ', 1)
                end = float(t)
            except (OSError, ValueError):
                pass
        out[tag] = (int(tag.split('-')[1]), start, end, outcome)
    return out


def one_round(k, kind, args, spec, out):
    trace = out / f'trace-{k}'
    trace.mkdir(parents=True, exist_ok=True)
    os.environ['V3_STRESS_TRACE'] = str(trace)
    os.environ['ISAAC_RL_FATAL_DIR'] = str(out / f'fatal-{k}')
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    teach = dict(teacher=True, teacher_share=0.5, teacher_death_depths=(2, 4, 8), teacher_thread=True,
                 teacher_fork=True) if args.teacher else {}
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4,
                           episodes_per_state=2, mode='floor', floor_seconds=args.floor_seconds, name=f'cs{k % 4}',
                           port=args.port + 40 * (k % 4), seed=1000 + 97 * k, bridge_lua=default_bridge_lua(),
                           preload=default_preload(), stub_list=stub if Path(stub).is_file() else '', nice=0, **teach)
    rng = np.random.default_rng(k)
    t0 = time.perf_counter()
    sampler = TokSampler(cfg)
    pid_index = {p.pid: i for i, p in enumerate(sampler.procs)}
    records, during = 0, None
    last = {}   # worker index -> the wall time of its last record

    def play(seconds):
        nonlocal records, during
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and sampler.live and during is None:
            try:
                idx, _ = sampler.poll(want=1, max_wait=0.0005, idle=0.02)
            except VP.VisibleV3Fatal as exc:
                during = str(exc)[:400]
                return
            now = time.time()
            for i in idx:   # (move, shoot; the rest 0)
                sampler.actions[i] = 0
                sampler.actions[i, 0], sampler.actions[i, 1] = rng.integers(0, 9), rng.integers(0, 5)
                last[i] = now
            records += len(idx)
            sampler.reply(idx)
    kill = None
    if kind == 'play':
        while records == 0 and time.perf_counter() - t0 < 90 and during is None and sampler.live:
            play(0.5)
        play(2 + 4 * rng.random())
    elif kind in ('reset-kill', 'query-kill'):
        prefix = 'r' if kind == 'reset-kill' else 'q'
        while time.perf_counter() - t0 < 60 and during is None and sampler.live:
            open_ = [q for q in traced(trace, prefix).values() if q[2] is None]
            if open_:
                pid = open_[0][0]
                kill = dict(pid=pid, index=pid_index.get(pid), t=time.time(),
                            killed=kill_groups(children_groups({pid})))
                break
            play(0.005)
        if kill is not None:   # the same worker giving records again (at most 60 s)
            t_k = time.perf_counter()
            while time.perf_counter() - t_k < 60 and during is None and sampler.live:
                play(0.5)
                if last.get(kill['index'], 0) > kill['t'] + 0.5:
                    break
            play(1)
    else:
        play(0.5 + 2.5 * rng.random())
    t_close = time.perf_counter()
    raised = None
    try:
        sampler.close(timeout=60.0)
    except VP.VisibleV3Fatal as exc:
        raised = str(exc)[:400]
    codes = [p.exitcode for p in sampler.procs]
    hit = None
    if kill is not None:
        ops = traced(trace, 'r' if kind == 'reset-kill' else 'q')
        broken = [dict(tag=t, outcome=o[3]) for t, o in ops.items() if o[0] == kill['pid'] and o[1] <= kill['t']
                  and (o[2] is None or o[2] >= kill['t'])]
        hit = dict(kill, interrupted=broken, failed=any(b['outcome'] not in (None, 'ok') for b in broken),
                   recovered=last.get(kill['index'], 0) > kill['t'] + 0.5)
    fatal_dir = out / f'fatal-{k}'
    return dict(round=k, kind=kind, records=records, hit=hit,
                queries_sent=len(list(trace.glob('sent-*'))),
                death_events=sum(len(p.read_text()) for p in trace.glob('death-*')),
                site_errors=[p.name for p in trace.glob('site-error-*')],
                exit_codes=codes, fatal_during=during, fatal_on_close=raised,
                fatal_files=sorted(p.name for p in fatal_dir.rglob('fatal-worker-*.txt')) if fatal_dir.is_dir() else [],
                close_s=round(time.perf_counter() - t_close, 2), seconds=round(time.perf_counter() - t0, 1))


def verdict(rows, v3=0, minimum=8, rounds=30):
    """The pass, coverage and outcome apart (Codex rounds 5 / 9): a kill counts only when it ended the traced
    operation in an error and its worker gave records afterwards."""
    done = [r for r in rows if 'error' not in r]
    codes = {}
    for r in done:
        for c in r['exit_codes']:
            codes[str(c)] = codes.get(str(c), 0) + 1
    kill_kind = 'query-kill' if v3 else 'reset-kill'
    cover = dict(play_with_records=sum(r['kind'] == 'play' and r['records'] > 0 for r in done),
                 kills_that_broke_and_recovered=sum(r['kind'] == kill_kind and bool(r['hit']) and
                                                    r['hit']['killed'] > 0 and r['hit']['failed'] and
                                                    r['hit']['recovered'] for r in done),
                 early=sum(r['kind'] == 'early' for r in done),
                 death_teaching_events=sum(r.get('death_events', 0) for r in done))
    outcome = dict(exit_86=codes.get(str(VP.V3_FATAL_EXIT), 0), exit_none=codes.get('None', 0),
                   fatal_during=sum(bool(r['fatal_during']) for r in done),
                   fatal_on_close=sum(bool(r['fatal_on_close']) for r in done),
                   fatal_files=sum(len(r['fatal_files']) for r in done),
                   site_errors=sum(len(r['site_errors']) for r in done),
                   mode_queries_sent_in_mode_0=0 if v3 else sum(r.get('queries_sent', 0) for r in done))
    covered = len(done) >= rounds and all(v >= minimum for v in cover.values())
    clean = all(v == 0 for v in outcome.values())
    outcomes = {}
    for r in done:
        for q in (r['hit'] or {}).get('interrupted', []):
            outcomes[str(q['outcome'])] = outcomes.get(str(q['outcome']), 0) + 1
    return dict(rounds=len(rows), ran=len(done), errors=len(rows) - len(done), v3=v3, exit_codes=codes,
                coverage=cover, interrupted_outcomes=outcomes, outcome=outcome, covered=covered, clean=clean,
                ok=covered and clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=36)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--port', type=int, default=46100)
    ap.add_argument('--floor-seconds', type=float, default=5.0, help='game seconds per floor episode (short: they end)')
    ap.add_argument('--teacher', type=int, default=1, help='the training teacher config (death teaching, thread, fork)')
    ap.add_argument('--v3', type=int, default=0, choices=(0, 1), help='0: the default arm; 1: a v3 arm (mode 1)')
    args = ap.parse_args()
    if args.v3:
        os.environ['ISAAC_RL_VISIBLE_V3'] = '1'   # (the workers and their games: spawned after this)
        VP.VISIBLE_V3_MODE = 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    site_dir = out / 'trace_site'
    site_dir.mkdir(exist_ok=True)
    (site_dir / 'sitecustomize.py').write_text(SITE)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(site_dir), os.getcwd()] +
                                               [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    kinds = ('play', 'query-kill' if args.v3 else 'reset-kill', 'early')
    rows, t0 = [], time.perf_counter()
    with open(out / 'rounds.jsonl', 'w') as f:
        for k in range(args.rounds):
            try:
                r = one_round(k, kinds[k % 3], args, spec, out)
            except Exception as exc:   # (a round that could not run: reported, counted, the next)
                r = dict(round=k, kind=kinds[k % 3], error=f'{type(exc).__name__}: {str(exc)[:300]}')
            rows.append(r)
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps(r)[:600], flush=True)
    summary = dict(verdict(rows, args.v3), workers=args.workers, seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('PASS' if summary['ok'] else 'FAIL', flush=True)
    sys.exit(0 if summary['ok'] else 1)


if __name__ == '__main__':
    main()
