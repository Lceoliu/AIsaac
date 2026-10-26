"""Actor profiles side by side (ISAAC_ACTOR_TORCH_PROFILE runs), markdown.

usage: python -X utf8 actor_profile_table.py label=<run dir> ...
One column per profiled actor process (<run>/actorprof/actor-<kind>-<pid>-*.json and .eager.json).
Window rows: per owner call, medians for latency / host / wait, means for the h.* sections (CPU ms of the
range, device ms of the kernels launched inside it). Eager rows: device ms per call by module group (forward
hooks, inclusive: a group whose module calls another top-level module counts it too) and the remainder of
p.core_eager outside every hooked module (functional ops: decode, gathers, the Reader's attention, sampling).
"""
import json, sys
from pathlib import Path

GROUPS = [('Spatial CNN (patch, room, map, player)', ('m.patch', 'm.room', 'm.map', 'm.player')),
          ('Spatial tokens (embeddings, entity, door, blocks, items, global)',
           ('m.type_emb', 'm.variant_emb', 'm.subtype_emb', 'm.item_emb', 'm.trinket_emb', 'm.pill_emb', 'm.effect_emb',
            'm.card_emb', 'm.char_emb', 'm.entity', 'm.door', 'm.blocks', 'm.norm', 'm.itok', 'm.gtok')),
          ('Temporal attention (old: tatt)', ('m.tatt',)),
          ('Reader input / layers / Writer', ('m.reader_inp', 'm.reader_layers', 'm.reader_writer')),
          ('c_t fusion and heads', ('m.reader_ct', 'm.pi', 'm.value', 'm.danger', 'm.choice'))]

cols = []
for spec in sys.argv[1:]:
    label, d = spec.split('=', 1)
    for f in sorted(Path(d, 'actorprof').glob('actor-*-*[0-9].json')):
        s = json.loads(f.read_text())
        e_path = Path(str(f)[:-5] + '.eager.json')
        e = json.loads(e_path.read_text()) if e_path.exists() else None
        cols.append((f'{label} {s["kind"]} pid {s["pid"]}', s, e))

rows, keys = {}, []
def put(name, i, value):
    if name not in rows: rows[name] = [''] * len(cols); keys.append(name)
    rows[name][i] = value

for i, (_, s, e) in enumerate(cols):
    pc = s['per_call']
    put('owner calls in window / all main calls of the process', i, f"{s['calls']} / {s.get('process_main_calls_in_window', '')}")
    put('other calls in window (lane, ...)', i, json.dumps(s.get('process_other_calls_in_window', {})))
    put('**latency per call (ms, median)**', i, f"{pc['latency_ms']:.2f}")
    q = s['latency_ms_quartiles']; put('latency quartiles', i, f'{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}')
    put('**host time in submit (ms, median)**', i, f"{pc['host_submit_ms']:.2f}")
    put('wait in result (ms, median)', i, f"{pc['wait_ms']:.3f}")
    put('**owner GPU time (ms/call, sum of h.* device)**', i, f"{pc.get('owner_device_ms', float('nan')):.2f}")
    put('all kernels of the process per main call (ms)', i, f"{pc['kernel_ms_all_streams']:.2f}")
    for k, v in sorted(s['host_sections_ms_per_call'].items()):
        put(f'{k} CPU / GPU ms', i, f"{v['cpu_ms_per_call']:.3f} / {v['device_ms_per_call']:.3f}")
    rn = s['rows']['n']; tot = sum(rn)
    put('rows per call (mean)', i, f"{sum(n * c for n, c in enumerate(rn)) / max(1, tot):.1f}")
    put('row buckets r', i, str(s['rows']['r']))
    put('entity buckets e (calls)', i, ', '.join(f'{v}:{c}' for v, c in zip(s['rows']['e_values'], s['rows']['e'])))
    put('graphs before / after window', i, f"{s['graphs_before']} / {s['graphs_after']}")
    put('dynamo counter changes', i, json.dumps(s['dynamo_counter_changes']) if s['dynamo_counter_changes'] else 'none')
    for k, v in sorted(s.get('call_notes', {}).items()):
        put(f'note: {k} (mean)', i, f'{v:.2f}')
    for k, v in s['kernel_classes_ms_per_call'].items():
        put(f'kernels {k} (ms per owner call, all streams)', i, f'{v:.3f}')
    if e:
        parts = e['parts_ms_per_call']
        total = parts.get('p.core_eager', {}).get('device_ms_per_call', 0.0)
        put('**eager network, GPU ms per call (uncompiled, no graph)**', i, f'{total:.2f}')
        hooked = 0.0
        for g, names in GROUPS:
            v = sum(parts[n]['device_ms_per_call'] for n in names if n in parts)
            hooked += v
            put(f'eager: {g}', i, f'{v:.2f} ({100 * v / total:.0f}%)' if total else '')
        put('eager: remainder (functional ops)', i, f'{total - hooked:.2f} ({100 * (total - hooked) / total:.0f}%)' if total else '')

print('| | ' + ' | '.join(c[0] for c in cols) + ' |')
print('|---|' + '---|' * len(cols))
for k in keys: print(f'| {k} | ' + ' | '.join(rows[k]) + ' |')
