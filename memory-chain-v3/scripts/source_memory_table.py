"""Source-card memory by kind per update (ISAAC_SOURCE_MEMORY_BREAKDOWN runs), markdown.

usage: python -X utf8 source_memory_table.py <run dir> [--every 2] [--device cuda:3]
Columns (GiB unless noted), from SOURCE_TIMELINE source_memory[].worker.breakdown:
  graph frames / B per frame : encode calls still graphs (outputs + saved activations, measured
                               around the call) and their bytes per frame
  writer graphs              : Writer calls still graphs
  exports graph / detached   : the compact IPC copies the learner reads, by owner state; detached
                               split by tensor (y = Spatial outputs, u = Reader inputs, ...)
  versions                   : parameter snapshots (count, bytes)
  cache                      : reserved - allocated
  VJP before / peak          : allocated before the largest VJP since the last retire, and its peak
"""
import json, sys

G = 2 ** 30
d = sys.argv[1]
every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 1
only = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else None
rows = []
for line in open(f'{d}/train.log', errors='replace'):
    if not line.startswith('SOURCE_TIMELINE '): continue
    t = json.loads(line.split(' ', 1)[1])
    for m in t.get('source_memory', []):
        b = m['worker'].get('breakdown')
        if b and (only is None or m['device'] == only): rows.append((t['update'], m['device'], b))
if not rows: sys.exit('no breakdown lines')
first = rows[0][0]
names = sorted({k for _, _, b in rows for k in b if k.startswith('export_detached_')})
print('| update | device | alloc | graph frames (k) | graph GiB | B/frame (KiB) | writer graphs | exports graph | '
      + ' | '.join(n.replace('export_detached_', 'det ') for n in names) + ' | det rows (k) | versions | cache | VJP before | VJP peak | peak-before |')
print('|' + '---|' * (14 + len(names)))
for i, (u, dev, b) in enumerate(rows):
    if (u - first) % every and i != len(rows) - 1: continue
    eg = sum(v for k, v in b.items() if k.startswith('export_graph_'))
    per = b['graph_chunk_bytes'] / max(1, b['graph_chunk_rows']) / 1024
    cells = [u - first + 1, dev, f"{b.get('allocated', 0) / G:.1f}", f"{b['graph_chunk_rows'] / 1000:.0f}",
             f"{b['graph_chunk_bytes'] / G:.1f}", f'{per:.0f}', f"{b['graph_write_bytes'] / G:.2f}", f'{eg / G:.2f}']
    cells += [f"{b.get(n, 0) / G:.2f}" for n in names]
    cells += [f"{b['detached_chunk_rows'] / 1000:.0f}", f"{b['versions']} / {b['version_bytes'] / G:.2f}",
              f"{b.get('cache_bytes', 0) / G:.1f}", f"{b.get('vjp_before', 0) / G:.1f}", f"{b.get('vjp_peak', 0) / G:.1f}",
              f"{(b.get('vjp_peak', 0) - b.get('vjp_before', 0)) / G:.1f}"]
    print('| ' + ' | '.join(str(c) for c in cells) + ' |')
