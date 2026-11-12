"""accept_g.sh: the fast clones' ABP_NATIVE_VIS counts (the frame probe's --status-lua answers, rows.json) summed."""
import json
import sys

tot = {}
for r in json.load(open(sys.argv[1])):
    st = (r.get('status') or {}).get('fast') or ''
    for part in st.split()[1:]:
        k, _, v = part.partition('=')
        try:
            tot[k] = tot.get(k, 0) + int(v)
        except ValueError:
            pass
print(' '.join(f'{k}={v}' for k, v in tot.items()))
