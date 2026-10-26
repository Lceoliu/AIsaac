"""Everything that needs the 8x3090 host, in order, once its SSH works again.

  python -X utf8 temp/memory-chain-v3/scripts/run_remote_all.py [steps...]

steps (default all, in this order):
  status   host snapshot; abort unless physical GPUs 1-5 have no compute process
  deploy   committed HEAD of candidate/bridge/python -> cpu-wide/memory-v3-code-<short>
  gates    CUDA gates on GPUs 3,4,5 (horizon test, attention timing, v2 CUDA tests)
  h0       H=0 short run (15 game hours, ~13 updates): v2 behaviour on the host
  h2048    long run, H=2048 (60 game hours, ~53 updates; first 10 discarded)
  h1024    long run, H=1024 (control)
  h4096    long run, H=4096 (optional control; not in the default list)
  final    host snapshot; assert no memory-v3 container is left
Each run: v2 long-59 command (64 workers, micro 8192, 5 GPUs, source BF16,
compile/flash/MPS/process sources, overlap rollout, update every 2 rollouts),
72 GiB container memory, guard as v2. Results: temp/memory-chain-v3/results/.
Exit code non-zero at the first failing step.
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable
BENCH = [PY, '-X', 'utf8', str(HERE / 'bench_v3.py')]
RESULTS = HERE / 'results'
FREE = [1, 2, 3, 4, 5]          # GPUs 0, 6, 7 belong to others
import os
TAG = 'v3-' + (os.environ.get('V3_REF') or subprocess.check_output(['git', '-C', str(HERE / 'candidate'), 'rev-parse', '--short', 'HEAD'], text=True).strip())
FIVE = ('3,2,1,4,5', None)                       # v2 long-59 device layout
FOUR = ('3,2,4,5', 'cuda:0,cuda:2,cuda:3')       # GPU1 taken: learner cuda:1, actors cuda:0,2, sources 0,2,3
SPILL0 = [' --source-saved-cpu-gb 0.75 ', ' --source-saved-cpu-gb 0 ']
TRIMS = []   # a) SPILL0 dropped: every tp run with it hit CUDA OOM in source backward (updates 5-37); b) stays (QUIET)
U1 = [' --source-update-rollouts 2 ', ' --source-update-rollouts 1 ']
QUIET = {'ISAAC_SOURCE_CPROFILE': '', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}
RUNS = dict(h0=(0, 15.0, FIVE), h2048=(2048, None, FIVE), h1024=(1024, None, FIVE), h4096=(4096, None, FIVE),
            ab4h2048=(2048, None, FOUR), ab4h0=(0, None, FOUR), ab4h1024=(1024, None, FOUR),
            ab4h2048acct=(2048, None, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1'}),
            # host-memory ablations, ~10 updates each (12 game hours), H=2048, accounting on
            ablA=(2048, 12.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1'}, [[' --teacher --teacher-queue', ' --teacher-queue'], [' --cf 1 ', ' --cf 0 ']]),
            ablB=(2048, 12.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1'}, [[' --cf 1 ', ' --cf 0 '], [' --teacher --teacher-queue', ' --teacher --teach-buffer 2000 --teacher-v2-buffer 1000 --teacher-queue']]),
            ablC=(2048, 12.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1'}, []),
            # per-process smaps audit (game roles from ISAAC_RL_MEM_LOG dumps); A/C to 20+ updates; old architecture
            audA=(2048, 25.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10'},
                  [[' --teacher --teacher-queue', ' --teacher-queue'], [' --cf 1 ', ' --cf 0 ']]),
            audC=(2048, 25.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10'}, []),
            audOld=(None, 12.0, ('3,2,4', None), {'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10'}, []),
            # record retention (approved policy), C configuration, 30 game hours (~27 updates)
            retC=(2048, 30.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10'}, []),
            # live-object census + allocator A/B (jemalloc 5.3 from the host, python processes only), ~21 updates each
            acctC=(2048, 24.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10'}, []),
            relC=(2048, 28.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_MEMORY_CENSUS': '0', 'ISAAC_RL_MEM_LOG': 'MEMLOG',
                                     'ISAAC_RL_MEM_EVERY': '10'}, []),
            # semantics-neutral host-memory trims, relC measurement, ~12 updates: a) no pinned source spill; a+b) + no inductor pool
            trimA=(2048, 14.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_MEMORY_CENSUS': '0', 'ISAAC_RL_MEM_LOG': 'MEMLOG',
                                      'ISAAC_RL_MEM_EVERY': '10'}, [SPILL0]),
            trimAB=(2048, 14.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_MEMORY_CENSUS': '0', 'ISAAC_RL_MEM_LOG': 'MEMLOG',
                                       'ISAAC_RL_MEM_EVERY': '10', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}, [SPILL0]),
            # throughput: every measurement off (the template's learner cProfile too), 62 game hours (~52 updates)
            tp4h2048=(2048, 62.0, FOUR, QUIET, TRIMS), tp4h0=(0, 62.0, FOUR, QUIET, TRIMS),
            tp5h2048=(2048, 62.0, FIVE, QUIET, TRIMS), tp5h0=(0, 62.0, FIVE, QUIET, TRIMS),
            # source-card memory at the VJP peak (ISAAC_SOURCE_MEMORY_BREAKDOWN), H=2048, 4 cards, until it stops
            memB4=(2048, 40.0, FOUR, dict(QUIET, ISAAC_SOURCE_MEMORY_BREAKDOWN='1'), TRIMS),
            # saved-tensor census (encodes 30-33 of each worker) + card breakdown; the same at H=1024 (graph frames)
            memC4=(2048, 40.0, FOUR, dict(QUIET, ISAAC_SOURCE_MEMORY_BREAKDOWN='1', ISAAC_SOURCE_SAVED_PROFILE='30'), TRIMS),
            memC4h1024=(1024, 40.0, FOUR, dict(QUIET, ISAAC_SOURCE_MEMORY_BREAKDOWN='1'), TRIMS),
            # user's B decision: source VJP / Adam every rollout; 5 cards, H=2048, everything else as tp5h2048
            tp5u1=(2048, 62.0, FIVE, QUIET, TRIMS + [[' --source-update-rollouts 2 ', ' --source-update-rollouts 1 ']]),
            # the user's next long run: H=1024, everything else as tp5u1, with the timeline
            tp5u1h1024=(1024, 62.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1'), TRIMS + [U1]),
            # table 5: table 4's configuration with the VJP dispatched at the interval's close (scheduling only)
            tp5u1h1024d=(1024, 62.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1'),
                         TRIMS + [U1, [' --source-flash 1 ', ' --source-flash 1 --source-dispatch-at-close 1 ']]),
            # table 6: table 5 + opt-in layer checkpointing (the user's decision, 2026-10-25); table 7: + retention fix
            t6=(1024, 62.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1'),
                TRIMS + [U1, [' --source-flash 1 ', ' --source-flash 1 --source-dispatch-at-close 1 --source-layer-checkpoint 1 ']]),
            t7=(1024, 62.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1'),   # (= t6's configuration; run with the retention fix's code)
                TRIMS + [U1, [' --source-flash 1 ', ' --source-flash 1 --source-dispatch-at-close 1 --source-layer-checkpoint 1 ']]),
            # t7 with the drop's learner-side context (ISAAC_LINEAGE_SOURCES: who reported each ancestor in flight)
            t7d=(1024, 62.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1', ISAAC_LINEAGE_SOURCES='1'),
                 TRIMS + [U1, [' --source-flash 1 ', ' --source-flash 1 --source-dispatch-at-close 1 --source-layer-checkpoint 1 ']]),
            # timeline of the source-every-rollout configuration; the same with opt-in layer checkpointing
            tlU1=(2048, 25.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1'), TRIMS + [U1]),
            ckU1=(2048, 25.0, FIVE, dict(QUIET, ISAAC_TIMELINE='1', ISAAC_SOURCE_MEMORY_BREAKDOWN='1', ISAAC_SOURCE_SAVED_PROFILE='30'),
                  TRIMS + [U1, [' --source-flash 1 ', ' --source-flash 1 --source-layer-checkpoint 1 ']]),
            # actor profile (ISAAC_ACTOR_TORCH_PROFILE, actor_profile.py): new architecture as tp5h2048, and the old
            # architecture's 843x command with its stock code + the same profiler (cpu-wide/memory-v3-code-oldprof-4)
            profNew=(2048, 22.0, FIVE, dict(QUIET, ISAAC_ACTOR_TORCH_PROFILE='RUNDIR/actorprof'), TRIMS),
            profOld=(None, 22.0, ('3,2,4', None), dict(ISAAC_ACTOR_TORCH_PROFILE='RUNDIR/actorprof', _oldcode='oldprof-4'), []),
            # the same sections by perf_counter only (no torch.profiler), 1000 owner calls
            timNew=(2048, 22.0, FIVE, dict(QUIET, ISAAC_ACTOR_TORCH_PROFILE='RUNDIR/actorprof', ISAAC_ACTOR_TORCH_PROFILE_MODE='timers',
                                           ISAAC_ACTOR_TORCH_PROFILE_COUNT='1000'), TRIMS),
            timOld=(None, 22.0, ('3,2,4', None), dict(ISAAC_ACTOR_TORCH_PROFILE='RUNDIR/actorprof', ISAAC_ACTOR_TORCH_PROFILE_MODE='timers',
                                                      ISAAC_ACTOR_TORCH_PROFILE_COUNT='1000', _oldcode='oldprof-6'), []),
            acctCje=(2048, 24.0, FOUR, {'ISAAC_MEMORY_ACCOUNT': '1', 'ISAAC_RL_MEM_LOG': 'MEMLOG', 'ISAAC_RL_MEM_EVERY': '10',
                                        'LD_PRELOAD': '/bench/v3/lib/libjemalloc.so.2', 'ISAAC_GAME_ENV_STRIP': 'LD_PRELOAD'}, []))
PYV = '/bench/home/isaac-abplus/.venv/bin/python -u '


def bench(*args, log=None, check=True):
    p = subprocess.run(BENCH + [str(a) for a in args], capture_output=True, text=True, encoding='utf-8', errors='replace')
    text = p.stdout + p.stderr
    if log: (RESULTS / log).write_text(text + f'\nEXIT {p.returncode}\n', encoding='utf-8')
    print(text[-3000:], f'\n[{" ".join(map(str, args[:2]))}] EXIT {p.returncode}', flush=True)
    if check and p.returncode: sys.exit(f'step failed: {args}')
    return p.returncode, text


def status(log):
    _, text = bench('status', log=log)
    uuids = {}
    for line in text.split('APPS')[0].splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2 and parts[0].isdigit(): uuids[int(parts[0])] = parts[1]
    apps = text.split('APPS')[1].split('CONTAINERS')[0]
    busy = [i for i in FREE if uuids.get(i) and uuids[i] in apps]
    return busy, text


def summarize(case):
    d = RESULTS / case
    result = json.loads((d / 'result.json').read_text())
    spec = result['spec']
    peak = {}
    for line in (d / 'telemetry.jsonl').read_text().splitlines():
        for row in json.loads(line)['gpus'].strip().splitlines():
            index, used = [x.strip() for x in row.split(',')[:2]]
            if int(index) in spec['gpu_indices']: peak[index] = max(peak.get(index, 0), float(used))
    guard = [json.loads(l) for l in (d / 'guard.jsonl').read_text().splitlines()] if (d / 'guard.jsonl').exists() else []
    host = max((g.get('memory_gib', 0) for g in guard), default=None)
    stops = [g['stop_reason'] for g in guard if g.get('stop_reason')]
    out = dict(case=case, horizon=spec['horizon'], exit_status=result['exit_status'], updates=result.get('updates'),
               steady_updates=result.get('steady_updates'), steady_x_real_time=result.get('steady_x_real_time'),
               medians=result.get('medians'), peak_gpu_mib=peak, peak_container_memory_gib=host, guard_stops=stops,
               completed=result['exit_status'] == 0 and (result.get('steady_updates') or 0) >= 40)
    (d / 'summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return out


def long_run(step):
    horizon, hours, (indices, sources), *more = RUNS[step]
    case = f'{TAG}-{step}-{time.strftime("%m%d%H%M")}'
    extra = dict(more[0]) if more else {}
    oldcode = extra.pop('_oldcode', None)
    env = [f'{k}={v}'.replace('MEMLOG', f'/bench/runs/memory-v3-{case}/memlog').replace('RUNDIR', f'/bench/runs/memory-v3-{case}')
           for k, v in extra.items()]
    if len(more) > 1 and more[1]: env.append('edits=' + json.dumps(more[1]))
    # sl_dev (another user's container, ~33 GB page cache) lost its cache to the
    # first A/B attempt. Start only while its block IO is quiet (<= 10 MB/s
    # read+write over 60 s); otherwise re-check every 5 minutes for 30 minutes.
    checks = []
    for attempt in range(7):
        _, text = bench('io', 'sl_dev', 60)
        io = json.loads(next(l for l in text.splitlines() if l.startswith('{')))
        io['active'] = io['read_mb_s'] + io['write_mb_s'] > 10
        checks.append(io)
        if not io['active'] or attempt == 6: break
        time.sleep(240)
    (RESULTS / f'{case}-sldev-io.json').write_text(json.dumps(checks, indent=2))
    busy, _ = status(f'{case}-status-before.log')
    wanted = [int(i) for i in indices.split(',')]
    if set(busy) & set(wanted): sys.exit(f'GPUs {busy} are in use by someone else; not launching {case}')
    bench('launch', case, ('oldarch:' + oldcode if oldcode else 'oldarch') if horizon is None else TAG, horizon if horizon is not None else 0,
          '-' if hours is None else hours, indices, sources or '-', *env,
          log=f'{case}-launch-cmd.log')
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        time.sleep(120)
        code, text = bench('collect', case, check=False)
        guard = RESULTS / case / 'guard.log'
        if code == 0 and guard.exists() and 'RUNNER_EXIT' in guard.read_text(errors='replace'): break
        # (the runner can die after writing its result, e.g. docker stop on a container already gone: no RUNNER_EXIT)
        result = RESULTS / case / 'result.json'
        if code == 0 and result.exists() and 'exit_status' in result.read_text(errors='replace'): break
    else:
        bench('stop', case, 'run_remote_all deadline (3 h) reached', check=False)
        bench('collect', case, check=False)
    return summarize(case)


def main():
    steps = sys.argv[1:] or ['status', 'deploy', 'gates', 'h0', 'h2048', 'h1024', 'final']
    RESULTS.mkdir(exist_ok=True)
    summaries = []
    for step in steps:
        print(f'===== {step} ({time.ctime()})', flush=True)
        if step == 'status':
            busy, _ = status('remote-status-start.log')
            if busy: sys.exit(f'GPUs {busy} busy (others\' work); wait and rerun')
        elif step == 'deploy':
            bench('deploy', TAG, os.environ.get('V3_REF') or 'HEAD', log=f'deploy-{TAG}.log')   # (the tagged commit, not HEAD)
        elif step == 'gates':
            gates = {
                'horizon-cuda': PYV + 'tests/test_source_horizon.py --learner cuda:0 --sources cuda:1,cuda:2',
                'horizon-bound': PYV + 'tests/test_source_horizon_bound.py && ' + PYV + 'tests/test_source_horizon_bound.py --process',
                'attention-timing': PYV + 'tests/bench_horizon_attention.py --device cuda:0 --groups 96 --count 128 --bank 1600 --sources 60000',
                'actor-cuda': ' && '.join(PYV + t for t in [
                    'tests/test_actor_d2h.py', 'tests/test_actor_streams.py',
                    # (--separate is v2's negative control: it must fail, see v2 actorpool-gate-separate-77)
                    'tests/test_actor_graph_pool.py']),
                # test_rebuild_memory_cuda fails identically on v2 64ae451 (rebuild_actor BF16 index_put,
                # gate-v2-64ae451-rebuild-memory-cuda.log): recorded as pre-existing, not a v3 gate.
                'v2-cuda-suite2': ' && '.join(PYV + t for t in [
                    'tests/test_rebuild_actor_cuda.py --compile',
                    'tests/test_source_pipeline.py --resident --joined --sparse --rollout-clock --overlap-rollout --learner cuda:0 --sources cuda:1,cuda:2 --queries 8192 --length 128',
                    'tests/test_source_process.py --learner cuda:0 --sources cuda:1,cuda:2 --queries 8192 --bf16 --defer --shared',
                    'tests/test_source_precision.py --learner cuda:0 --sources cuda:1,cuda:2 --queries 8192']),
            }
            for name, shell in gates.items():
                done = RESULTS / f'gate-{TAG}-{name}.exit'
                if done.exists() and done.read_text().strip() == '0':
                    print('gate already passed:', name, flush=True); continue
                bench('gate', f'{TAG}-{name}', TAG, '3,4,5', 'nice -n 10 timeout 1800 bash -c ' + json.dumps(shell),
                      log=f'gate-{TAG}-{name}-dispatch.log')
        elif step == 'actorgates':   # the actor's CUDA tests on TAG (graph/eager logits, state writes, profiler twin)
            gates = [('rebuild-actor', 'tests/test_rebuild_actor_cuda.py --compile'),
                     ('actor-state', 'tests/test_rebuild_actor_state_cuda.py --compile'),
                     ('actorprof', 'tests/test_actor_profile_cuda.py --compile --kinds rebuild'),
                     ('actor-crn', 'tests/test_actor_crn_cuda.py'),
                     ('layer-ckpt', 'tests/test_source_layer_checkpoint_cuda.py --frames 512')]
            only = os.environ.get('V3_GATES')   # (comma names; default: all but actor-crn unless V3_CRN_GATE)
            gates = [g for g in gates if (g[0] in only.split(',')) if only] if only else                 [g for g in gates if g[0] != 'actor-crn' or os.environ.get('V3_CRN_GATE')]
            for name, test in gates:
                bench('gate', f'{TAG}-{name}', TAG, '3,4,5', 'nice -n 10 timeout 1800 bash -c ' + json.dumps(PYV + test),
                      log=f'gate-{TAG}-{name}-dispatch.log')
        elif step == 'profgates':   # the profiler's CUDA twin test on both code trees
            for name, tag, shell in [(f'{TAG}-actorprof', TAG, PYV + 'tests/test_actor_profile_cuda.py --compile'),
                                     ('oldprof-4-actorprof', 'oldprof-4', PYV + 'tests/test_actor_profile_cuda.py --compile --kinds graph')]:
                bench('gate', name, tag, '3,4,5', 'nice -n 10 timeout 1800 bash -c ' + json.dumps(shell), log=f'gate-{name}-dispatch.log')
        elif step in RUNS:
            summaries.append(long_run(step))
            (RESULTS / f'{TAG}-runs-summary.json').write_text(json.dumps(summaries, indent=2))
        elif step == 'final':
            _, text = status('remote-status-final.log')
            left = [l for l in text.split('CONTAINERS')[1].splitlines() if 'memory-v3' in l]
            if left: sys.exit(f'own containers still running: {left}')
            print('no memory-v3 container left', flush=True)
        else:
            sys.exit(f'unknown step {step}')


if __name__ == '__main__':
    main()
