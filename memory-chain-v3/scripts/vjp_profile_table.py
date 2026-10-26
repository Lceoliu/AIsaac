"""SOURCE_VJP_PROFILE lines (ISAAC_SOURCE_VJP_PROFILE runs) as markdown: one column per (run, source card).

usage: python -X utf8 vjp_profile_table.py label=<run dir> ...
"""
import json, sys

cols = []
for spec in [a for a in sys.argv[1:] if '=' in a]:
    label, d = spec.split('=', 1)
    for line in open(f'{d}/train.log', errors='replace'):
        if line.startswith('SOURCE_VJP_PROFILE '):
            p = json.JSONDecoder().raw_decode(line.split(' ', 1)[1])[0]
            cols.append((f"{label} {p['device']}", p))
rows = [('VJP wall s', lambda p: f"{p['wall_s']:.2f}"), ('frames (encode chunks)', lambda p: p['frames']),
        ('packets / encode chunks / writes', lambda p: f"{p['packets']} / {p['encode_chunks']} / {p['writes']}"),
        ('backward calls (episode groups)', lambda p: p['backward_calls']),
        ('frames per chunk min / median / mean / max', lambda p: '{min} / {median:.0f} / {mean:.0f} / {max}'.format(**p['frames_per_chunk'])),
        ('frames per backward call', lambda p: f"{p['frames_per_backward_call']:.0f}"),
        ('kernels', lambda p: p['kernels']), ('kernels per 1k frames', lambda p: f"{p['kernels_per_1k_frames']:.0f}"),
        ('kernel us mean / median / p90', lambda p: f"{p['kernel_us_mean']:.1f} / {p['kernel_us_median']:.1f} / {p['kernel_us_p90']:.1f}"),
        ('kernel s total', lambda p: f"{p['kernel_s_total']:.2f}"),
        ('copies (n / s)', lambda p: f"{p['copies']} / {p['copy_s_total']:.2f}"),
        ('GPU window s / busy s', lambda p: f"{p['gpu_window_s']:.2f} / {p['gpu_busy_s']:.2f}"),
        ('GPU idle share of the window', lambda p: f"{p['gpu_idle_share']:.1%}" if p['gpu_idle_share'] is not None else ''),
        ('autograd nodes (per 1k frames)', lambda p: f"{p['autograd_nodes']} ({p['autograd_nodes_per_1k_frames']:.0f})")]
print('| | ' + ' | '.join(c for c, _ in cols) + ' |'); print('|---|' + '---|' * len(cols))
for name, f in rows: print(f'| {name} | ' + ' | '.join(str(f(p)) for _, p in cols) + ' |')
for c, p in cols:
    print(f'\n**{c}: top kernels (count, s)**\n')
    for k in p['top_kernels']: print(f"- {k['count']} × {k['s']:.3f} s  `{k['name'][:100]}`")
    print('\nautograd node kinds: ' + ', '.join(f'{k} {n}' for k, n in p['top_nodes']))
