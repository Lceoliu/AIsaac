"""Opt-in actor profiler (a measurement aid; off unless ISAAC_ACTOR_TORCH_PROFILE=<dir> is set).

One window per process (the profiler is process-wide), taken by the main actor loop's instance (`role` 'main')
whose call is the process's AT-th main call:
  its next COUNT calls     torch.profiler (CPU + CUDA) over the calls as they run: the host sections of
                           submit()/result() as record_function ranges 'h.<name>', the replayed graphs' kernels;
                           per call: host seconds in submit, the wait in result, the latency, rows, buckets
  next EAGER calls         after each call's replay, the same network once more uncompiled and outside the graphs
                           under a second profiler ('p.core_eager'), its parts as ranges 'm.<name>' (forward hooks on
                           TokPolicy's top-level modules; the Reader by its parts, the attention between them is
                           functional and stays in the remainder): device time per part; the outputs are discarded,
                           state writes repeat the replay's values (or go to the scratch slot)
Results: <dir>/actor-<pid>-<id>.json (summary), .txt (key_averages tables), .trace.json.gz (chrome trace, window 1).
ISAAC_ACTOR_TORCH_PROFILE_AT (default 9000: main calls of the process), _COUNT (200), _EAGER (50).
The other actors of the process keep running during the window: their kernels are in the window's kernel totals
(kernel_ms_all_streams, divided by all main calls of the process in the window), not in the owner's h.* sections
(device time by launching CPU range: the owner's own calls only).
"""
import contextlib, gzip, json, os, shutil, time
from pathlib import Path

import numpy as np
import torch

DIR = os.environ.get('ISAAC_ACTOR_TORCH_PROFILE')
AT = int(os.environ.get('ISAAC_ACTOR_TORCH_PROFILE_AT', '9000'))
COUNT = int(os.environ.get('ISAAC_ACTOR_TORCH_PROFILE_COUNT', '200'))
EAGER = int(os.environ.get('ISAAC_ACTOR_TORCH_PROFILE_EAGER', '50'))
# 'timers': no torch.profiler (its CPU-op tracing inflates host time), the same sections by perf_counter only,
# no eager pass; <dir>/actor-<kind>-<pid>-<id>.timers.json
TIMERS = os.environ.get('ISAAC_ACTOR_TORCH_PROFILE_MODE') == 'timers'
NULL = contextlib.nullcontext()
PROCESS = dict(owner=None, main=0, window_main0=0, others={})   # (this process's calls by role, the window's owner)


def _device_us(e):
    for name in ('device_time_total', 'cuda_time_total'):
        v = getattr(e, name, None)
        if v is not None: return float(v)
    return 0.0


def _dynamo_counters():
    try:
        from torch._dynamo.utils import counters
        return {f'{k}/{k2}': int(v2) for k, v in counters.items() for k2, v2 in v.items()}
    except Exception:
        return {}


def _table(prof, sort, rows):
    for key in (sort, sort.replace('device', 'cuda'), 'cpu_time_total'):
        try: return prof.key_averages().table(sort_by=key, row_limit=rows)
        except Exception: continue
    return ''


def kernel_class(name):
    n = name.lower()
    if 'gemm' in n or 'cutlass' in n or 'sm80_xmma' in n or 'ampere_' in n or 'cublas' in n: return 'gemm'
    if 'fmha' in n or 'flash' in n or 'attention' in n or 'softmax' in n: return 'attention/softmax'
    if 'conv' in n or 'implicit' in n or 'winograd' in n or 'cudnn' in n: return 'conv'
    if n.startswith('triton'): return 'triton fused'
    if 'index' in n or 'gather' in n or 'scatter' in n: return 'index/gather/scatter'
    if 'memcpy' in n or 'memset' in n or 'copy' in n: return 'copy/memset'
    if 'reduce' in n or 'norm' in n: return 'reduce/norm'
    return 'other'


class ActorProfile:
    """Per-instance window; every method is a no-op outside it (and without the variable)."""

    def __init__(self, actor, kind):
        self.actor, self.kind, self.calls, self.on = actor, kind, 0, bool(DIR)
        self.prof = self.eprof = None
        self.phase = None   # 'window' (COUNT own calls), 'eager' (EAGER own calls), 'done'
        self.rows = []
        self.cur = self._open = None

    @property
    def role(self):
        return getattr(self.actor, 'prof_role', 'other')

    # ---- window 1: the calls as they run
    def begin(self, n):
        """At submit's start. Returns True inside window 1."""
        if not self.on: return False
        if self.role != 'main':
            if PROCESS['owner'] is not None and PROCESS['owner'].phase == 'window':
                PROCESS['others'][self.role] = PROCESS['others'].get(self.role, 0) + 1
            return False
        PROCESS['main'] += 1
        self.calls += 1
        if PROCESS['owner'] is None and PROCESS['main'] >= AT:
            PROCESS['owner'], PROCESS['window_main0'], self.phase = self, PROCESS['main'] - 1, 'window'
            self.counters0, self.t0 = _dynamo_counters(), time.perf_counter()
            self.extra0 = dict(graphs=self._graph_count())
            if not TIMERS:
                self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA])
                self.prof.__enter__()
        self.cur = dict(call=self.calls, n=int(n), t_submit=time.perf_counter()) if self.phase == 'window' else None
        return self.cur is not None

    def note(self, **kw):
        if self.cur is not None: self.cur.update(kw)

    def mark(self, name):
        """Close the open host section and open 'h.<name>' (None: close only)."""
        if self.cur is None: return
        if TIMERS:
            now = time.perf_counter()
            if self._open is not None:
                k = 't.' + self._open[0]; self.cur[k] = self.cur.get(k, 0.0) + now - self._open[1]
            self._open = (name, now) if name is not None else None
            return
        if self._open is not None: self._open.__exit__(None, None, None); self._open = None
        if name is not None:
            self._open = torch.profiler.record_function('h.' + name); self._open.__enter__()

    def submitted(self):
        if self.cur is not None:
            self.mark(None)
            self.cur['host_submit_s'] = time.perf_counter() - self.cur['t_submit']

    def finished(self, wait_s):
        """At result's end."""
        if self.cur is None: return
        cur, self.cur = self.cur, None
        cur['wait_s'] = wait_s
        cur['latency_s'] = time.perf_counter() - cur.pop('t_submit')
        self.rows.append(cur)
        if len(self.rows) == COUNT and TIMERS:
            self.phase = 'done'
            wall = time.perf_counter() - self.t0
            keys = sorted({k for r in self.rows for k in r if k.startswith('t.')})
            mean = lambda k: 1000 * float(np.mean([r.get(k, 0.0) for r in self.rows]))
            out = dict(kind=self.kind, pid=os.getpid(), calls=COUNT, window_wall_s=wall,
                       process_main_calls_in_window=PROCESS['main'] - PROCESS['window_main0'],
                       latency_ms=dict(mean=mean('latency_s'), median=1000 * float(np.median([r['latency_s'] for r in self.rows]))),
                       host_submit_ms=mean('host_submit_s'), wait_ms=mean('wait_s'),
                       sections_ms={k[2:]: mean(k) for k in keys},
                       rows_mean=float(np.mean([r['n'] for r in self.rows])),
                       notes={k: float(np.mean([r[k] for r in self.rows if k in r])) for k in ('len_max', 'len_mean', 'capacity', 'closed', 'component_rows', 'stale')
                              if any(k in r for r in self.rows)})
            Path(str(self._base()) + '.timers.json').write_text(json.dumps(out, indent=1))
            print('ACTOR_TIMERS', json.dumps({k: out[k] for k in ('kind', 'pid', 'latency_ms', 'host_submit_ms', 'wait_ms')}), flush=True)
            return
        if len(self.rows) == COUNT:
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)
            self.phase = 'eager'
            self.wall = time.perf_counter() - self.t0
            self.process_main = PROCESS['main'] - PROCESS['window_main0']
            self.process_others = dict(PROCESS['others'])
            self.counters1 = _dynamo_counters()
            self.extra1 = dict(graphs=self._graph_count())
            self._dump_window()

    # ---- window 2: the eager attribution pass
    def eager_wanted(self):
        return self.phase == 'eager'

    def eager(self, fn):
        """fn(): the uncompiled network on this call's inputs (after the replay)."""
        if self.eprof is None:
            self.eprof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA])
            self.eprof.__enter__()
            self.eager_calls = 0
        model = self.actor.model
        hooks, stack = [], []
        # TokPolicy's top-level modules; the Reader (whose forward the actor never calls: its parts run
        # from the actor's core) by its parts: input projection, layers, Writer, c_t fusion / read-outs
        named = [(n, c) for n, c in model.named_children() if n != 'reader']
        if getattr(model, 'reader', None) is not None:
            for n, c in model.reader.named_children():
                named.append(('reader_' + ('inp' if n == 'inp' else 'writer' if n.startswith('writer') else
                                           'layers' if n in ('layers', 'norm') else 'ct'), c))
        for name, child in named:
            def pre(mod, inp, name=name):
                rf = torch.profiler.record_function('m.' + name); rf.__enter__(); stack.append(rf)
            def post(mod, inp, out):
                stack.pop().__exit__(None, None, None)
            hooks += [child.register_forward_pre_hook(pre), child.register_forward_hook(post)]
        try:
            with torch.inference_mode(), torch.profiler.record_function('p.core_eager'):
                fn()
        finally:
            for h in hooks: h.remove()
        self.eager_calls += 1
        if self.eager_calls == EAGER:
            torch.cuda.synchronize()
            self.eprof.__exit__(None, None, None)
            self.phase = 'done'
            self._dump_eager()

    # ---- output
    def _graph_count(self):
        a = self.actor
        return len(getattr(a, 'graphs', ())) + len(getattr(a, 'component_graphs', ()))

    def _base(self):
        d = Path(DIR); d.mkdir(parents=True, exist_ok=True)
        return d / f'actor-{self.kind}-{os.getpid()}-{id(self.actor) % 100000}'

    def _summary_of(self, prof, calls):
        labels, kernels = {}, {}
        for e in prof.key_averages():
            if e.key.startswith(('h.', 'p.', 'm.')):
                labels[e.key] = dict(count=int(e.count), cpu_ms_per_call=e.cpu_time_total / 1000 / calls,
                                     device_ms_per_call=_device_us(e) / 1000 / calls)
        total_kernel = 0.0
        events = prof.events()
        device = [(e.name, e.time_range.elapsed_us()) for e in events
                  if getattr(e, 'device_type', None) == torch.autograd.DeviceType.CUDA]
        if not device:   # (older profilers: kernels hang off their launching op)
            device = [(k.name, k.duration) for e in events for k in getattr(e, 'kernels', ())]
        for name, us in device:
            k = kernel_class(name); kernels[k] = kernels.get(k, 0.0) + us
            total_kernel += us
        return labels, {k: v / 1000 / calls for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])}, total_kernel / 1000 / calls

    def _dump_window(self):
        base = self._base(); calls = len(self.rows)
        labels, kernels, kernel_ms = self._summary_of(self.prof, calls)
        med = lambda k: float(np.median([r[k] for r in self.rows if k in r])) if any(k in r for r in self.rows) else None
        dyn = {k: self.counters1.get(k, 0) - self.counters0.get(k, 0) for k in self.counters1 if self.counters1.get(k, 0) != self.counters0.get(k, 0)}
        summary = dict(kind=self.kind, pid=os.getpid(), first_call=AT, calls=calls, window_wall_s=self.wall,
                       process_main_calls_in_window=self.process_main, process_other_calls_in_window=self.process_others,
                       per_call=dict(latency_ms=1000 * med('latency_s'), host_submit_ms=1000 * med('host_submit_s'),
                                     wait_ms=1000 * med('wait_s'),
                                     owner_device_ms=sum(v['device_ms_per_call'] for k, v in labels.items() if k.startswith('h.')),
                                     kernel_ms_all_streams=kernel_ms * calls / max(1, self.process_main)),
                       latency_ms_quartiles=[1000 * float(x) for x in np.percentile([r['latency_s'] for r in self.rows], [25, 50, 75])],
                       host_sections_ms_per_call=labels, kernel_classes_ms_per_call=kernels,
                       rows=dict(n=np.bincount([r['n'] for r in self.rows]).tolist(),
                                 r=sorted({r.get('r') for r in self.rows if r.get('r')}),
                                 e=np.unique([r['e'] for r in self.rows if 'e' in r], return_counts=True)[1].tolist(),
                                 e_values=np.unique([r['e'] for r in self.rows if 'e' in r]).tolist()),
                       call_notes={k: float(np.mean([r[k] for r in self.rows if k in r])) for k in
                                   sorted({k for r in self.rows for k in r} - {'call', 'n', 'r', 'e', 'latency_s', 'host_submit_s', 'wait_s'})
                                   if all(isinstance(r.get(k, 0), (int, float, bool, np.integer, np.floating)) for r in self.rows)},
                       graphs_before=self.extra0['graphs'], graphs_after=self.extra1['graphs'], dynamo_counter_changes=dyn,
                       raw=self.rows)
        base.with_suffix('.json').write_text(json.dumps(summary, indent=1, default=float))
        with open(base.with_suffix('.txt'), 'w') as f:
            f.write(_table(self.prof, 'self_device_time_total', 60))
        tmp = str(base) + '.trace.json'
        self.prof.export_chrome_trace(tmp)
        with open(tmp, 'rb') as a, gzip.open(tmp + '.gz', 'wb') as b: shutil.copyfileobj(a, b)
        os.remove(tmp)
        print('ACTOR_TORCH_PROFILE', json.dumps({k: summary[k] for k in ('kind', 'pid', 'per_call', 'graphs_before', 'graphs_after')}), flush=True)

    def _dump_eager(self):
        base = self._base()
        labels, kernels, kernel_ms = self._summary_of(self.eprof, self.eager_calls)
        out = dict(kind=self.kind, eager_calls=self.eager_calls, parts_ms_per_call=labels, kernel_classes_ms_per_call=kernels,
                   kernel_ms_per_call=kernel_ms)
        Path(str(base) + '.eager.json').write_text(json.dumps(out, indent=1))
        with open(str(base) + '.eager.txt', 'w') as f:
            f.write(_table(self.eprof, 'device_time_total', 80))
        print('ACTOR_TORCH_PROFILE_EAGER', json.dumps(dict(kind=self.kind, pid=os.getpid(), kernel_ms_per_call=kernel_ms)), flush=True)
