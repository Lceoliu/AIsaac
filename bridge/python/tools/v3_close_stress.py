"""Close-race stress check (2026-11-18, the coordinator, after A1's mode-0 smoke on c80c9f2 ended with a worker exit 86;
Codex round 5: the check must show it covered the race, not only that no 86 came). Mode 0 (every visible / trigger
switch at its default), floor workers. Rounds of three kinds, in turn:
  play         random actions until the workers' records flow, a little more, then close()
  query-kill   as soon as a worker starts the bridge mode query (abplus.check_visible_v3, the builder's first build
               of a launch), that worker's own game process groups are killed (SIGKILL; only the groups of the
               worker's child processes -- each game runs in its own session) while the query is held open; then the
               round goes on a few seconds (the worker recovers) and closes
  early        close() 0.5 to 3 s after the start, the builders launching / connecting
The mode query is traced in the workers (spawned processes) by a sitecustomize.py this tool puts first on their
PYTHONPATH: it wraps isaac_bridge.abplus.check_visible_v3 to write a start marker, hold 0.3 s (the window a kill can
land in), run the check, and write an end marker with the outcome (ok / the exception's class). No production code
is changed for it.
Per round: the exit code of every worker, VisibleV3Fatal during the round or on close, the fatal files, the records,
the traced queries and which of them a kill interrupted (started before the kill, ended after it), the games killed.
Pass (coverage and outcome apart): >= 30 rounds; >= 8 play rounds with records; >= 8 query-kill rounds whose kill
interrupted a traced query and killed >= 1 game process of the round; >= 8 early rounds; no worker exit code None
(a worker not ended), no exit 86, no VisibleV3Fatal, no fatal file.

usage (bridge python dir, PYTHONPATH=.; the CPU runner adds --groups-file / --out):
  python tools/v3_close_stress.py --groups-file ../abplus/catalog/scaling2_groups.json --out <dir> [--rounds 36]
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
for k in ('ISAAC_RL_VISIBLE_V3', 'ISAAC_RL_NATIVE_VISIBLE', 'ISAAC_RL_TRIGGER_ONCE'):   # (the old arm: mode 0)
    os.environ.pop(k, None)

import numpy as np   # noqa: E402

from goexplore_abplus import default_bridge_lua, default_preload, load_spec   # noqa: E402
from isaac_bridge import visible_producer as VP   # noqa: E402
from isaac_bridge.tok_sampler import TokSampler, TokSamplerConfig   # noqa: E402

SITE = '''
import os, time
_d = os.environ.get('V3_STRESS_TRACE')
if _d:
    try:
        import isaac_bridge.abplus as _ab
        _orig = _ab.check_visible_v3

        def _traced(client, mode=None, _n=[0]):
            _n[0] += 1
            tag = '%d-%d' % (os.getpid(), _n[0])
            with open(os.path.join(_d, 'q-%s.start' % tag), 'w') as f:
                f.write(repr(time.time()))
            time.sleep(0.3)
            out = 'ok'
            try:
                return _orig(client, mode)
            except BaseException as e:
                out = type(e).__name__
                raise
            finally:
                with open(os.path.join(_d, 'q-%s.end' % tag), 'w') as f:
                    f.write('%r %s' % (time.time(), out))
        _ab.check_visible_v3 = _traced
    except Exception as e:
        with open(os.path.join(_d, 'site-error-%d' % os.getpid()), 'w') as f:
            f.write(repr(e))
'''


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


def queries(trace):
    """The traced mode queries: {tag: (pid, start, end or None, outcome or None)}."""
    out = {}
    for p in trace.glob('q-*.start'):
        tag = p.name[2:-6]
        try:
            start = float(p.read_text())
        except (OSError, ValueError):
            continue
        end, outcome = None, None
        e = trace / f'q-{tag}.end'
        if e.is_file():
            try:
                t, outcome = e.read_text().split(' ', 1)
                end = float(t)
            except (OSError, ValueError):
                pass
        out[tag] = (int(tag.split('-')[0]), start, end, outcome)
    return out


def one_round(k, kind, args, spec, out, site_dir):
    trace = out / f'trace-{k}'
    trace.mkdir(parents=True, exist_ok=True)
    os.environ['V3_STRESS_TRACE'] = str(trace)
    os.environ['ISAAC_RL_FATAL_DIR'] = str(out / f'fatal-{k}')
    stub = str(Path(default_preload()).parent / 'stub_render_g.txt')
    cfg = TokSamplerConfig(workers=args.workers, specs=[spec], assign=[0] * args.workers, frames_per_decision=4,
                           episodes_per_state=2, mode='floor', floor_seconds=60.0, name=f'cs{k % 4}',
                           port=args.port + 40 * (k % 4), seed=1000 + 97 * k, bridge_lua=default_bridge_lua(),
                           preload=default_preload(), stub_list=stub if Path(stub).is_file() else '', nice=0)
    rng = np.random.default_rng(k)
    t0 = time.perf_counter()
    sampler = TokSampler(cfg)
    workers = {p.pid for p in sampler.procs}
    records, during, killed, kill_t, kill_pid = 0, None, None, None, None
    early = 0.5 + 2.5 * rng.random()
    t_records = None

    def play(seconds):
        nonlocal records, during, t_records
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and sampler.live and during is None:
            try:
                idx, _ = sampler.poll(want=1, max_wait=0.0005, idle=0.02)
            except VP.VisibleV3Fatal as exc:
                during = str(exc)[:400]
                return
            for i in idx:   # (move, shoot; the rest 0)
                sampler.actions[i] = 0
                sampler.actions[i, 0], sampler.actions[i, 1] = rng.integers(0, 9), rng.integers(0, 5)
            if idx and t_records is None:
                t_records = time.perf_counter()
            records += len(idx)
            sampler.reply(idx)
    if kind == 'play':   # until records flow (at most 90 s), then 2 to 6 s more
        while records == 0 and time.perf_counter() - t0 < 90 and during is None and sampler.live:
            play(0.5)
        play(2 + 4 * rng.random())
    elif kind == 'query-kill':   # the first query held open: its worker's games killed
        while time.perf_counter() - t0 < 60 and during is None and sampler.live:
            open_q = [q for q in queries(trace).values() if q[2] is None]
            if open_q:
                kill_pid = open_q[0][0]
                kill_t = time.time()
                killed = kill_groups(children_groups({kill_pid} & workers))
                break
            play(0.01)
        play(4)
    else:
        play(early)
    t_close = time.perf_counter()
    raised = None
    try:
        sampler.close(timeout=60.0)
    except VP.VisibleV3Fatal as exc:
        raised = str(exc)[:400]
    codes = [p.exitcode for p in sampler.procs]
    qs = queries(trace)
    interrupted = [dict(tag=t, outcome=q[3]) for t, q in qs.items()
                   if kill_t is not None and q[0] == kill_pid and q[1] <= kill_t and (q[2] is None or q[2] >= kill_t)]
    fatal_dir = out / f'fatal-{k}'
    files = sorted(p.name for p in fatal_dir.rglob('fatal-worker-*.txt')) if fatal_dir.is_dir() else []
    return dict(round=k, kind=kind, records=records, t_records=None if t_records is None else round(t_records - t0, 1),
                queries=len(qs), queries_open_at_end=sum(q[2] is None for q in qs.values()),
                query_outcomes=sorted({q[3] for q in qs.values() if q[3]}), killed=killed, interrupted=interrupted,
                site_errors=[p.name for p in trace.glob('site-error-*')],
                exit_codes=codes, fatal_during=during, fatal_on_close=raised, fatal_files=files,
                close_s=round(time.perf_counter() - t_close, 2), seconds=round(time.perf_counter() - t0, 1))


def verdict(rows, minimum=8, rounds=30):
    """The pass, coverage and outcome apart (Codex round 5)."""
    done = [r for r in rows if 'error' not in r]
    codes = {}
    for r in done:
        for c in r['exit_codes']:
            codes[str(c)] = codes.get(str(c), 0) + 1
    cover = dict(
        play_with_records=sum(r['kind'] == 'play' and r['records'] > 0 for r in done),
        query_killed=sum(r['kind'] == 'query-kill' and bool(r['interrupted']) and (r['killed'] or 0) > 0 for r in done),
        early=sum(r['kind'] == 'early' for r in done))
    outcome = dict(exit_86=codes.get(str(VP.V3_FATAL_EXIT), 0), exit_none=codes.get('None', 0),
                   fatal_during=sum(bool(r['fatal_during']) for r in done),
                   fatal_on_close=sum(bool(r['fatal_on_close']) for r in done),
                   fatal_files=sum(len(r['fatal_files']) for r in done),
                   site_errors=sum(len(r['site_errors']) for r in done))
    covered = len(done) >= rounds and all(v >= minimum for v in cover.values())
    clean = all(v == 0 for v in outcome.values())
    interrupted_outcomes = {}
    for r in done:
        for q in r['interrupted']:
            interrupted_outcomes[str(q['outcome'])] = interrupted_outcomes.get(str(q['outcome']), 0) + 1
    return dict(rounds=len(rows), ran=len(done), errors=len(rows) - len(done), exit_codes=codes, coverage=cover,
                interrupted_query_outcomes=interrupted_outcomes, outcome=outcome, covered=covered, clean=clean,
                ok=covered and clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups-file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=36)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--port', type=int, default=46100)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    site_dir = out / 'trace_site'
    site_dir.mkdir(exist_ok=True)
    (site_dir / 'sitecustomize.py').write_text(SITE)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(site_dir), os.getcwd()] +
                                               [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    kinds = ('play', 'query-kill', 'early')
    rows, t0 = [], time.perf_counter()
    with open(out / 'rounds.jsonl', 'w') as f:
        for k in range(args.rounds):
            try:
                r = one_round(k, kinds[k % 3], args, spec, out, site_dir)
            except Exception as exc:   # (a round that could not run: reported, counted, the next)
                r = dict(round=k, kind=kinds[k % 3], error=f'{type(exc).__name__}: {str(exc)[:300]}')
            rows.append(r)
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps(r)[:600], flush=True)
    summary = dict(verdict(rows), workers=args.workers, seconds=round(time.perf_counter() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('PASS' if summary['ok'] else 'FAIL', flush=True)
    sys.exit(0 if summary['ok'] else 1)


if __name__ == '__main__':
    main()
