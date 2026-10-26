"""Host wall time per h.* section from the actor profiles' chrome traces (ms per owner call, mean).

usage: python -X utf8 actor_trace_sections.py <run dir> ...
The trace's user_annotation spans (record_function ranges) carry the host duration of each
section; key_averages' cpu_time_total misses ranges whose ops are only launches.
"""
import gzip, json, sys
from pathlib import Path

for d in sys.argv[1:]:
    for f in sorted(Path(d, 'actorprof').glob('*.trace.json.gz')):
        ev = json.load(gzip.open(f))['traceEvents']
        span, gpu_launch = {}, {}
        calls = 0
        for e in ev:
            if e.get('ph') != 'X': continue
            n = e.get('name', '')
            if e.get('cat') in ('user_annotation', 'cpu_op') and n.startswith('h.'):
                span[n] = span.get(n, 0.0) + e['dur'] / 1000
                if n in ('h.reset_prefixes', 'h.stage_pack'): calls += 0
        s = json.loads(Path(str(f).replace('.trace.json.gz', '.json')).read_text())
        calls = s['calls']
        tot = sum(span.values()) / calls
        print(f'{f.parent.parent.name} {f.name[:-14]}: host section total {tot:.2f} ms/call; ' +
              ', '.join(f'{k[2:]} {v / calls:.3f}' for k, v in sorted(span.items(), key=lambda kv: -kv[1])))
