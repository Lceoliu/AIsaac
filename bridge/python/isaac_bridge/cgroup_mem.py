"""The memory of this process's own cgroup (inside a container: the container's), for the training run's guard
(train_tok's progress columns ctr_mem_gb / ctr_mem_limit_gb / ctr_mem_frac / ctr_oom_kills; shardk_in.sh's end line).

used = usage minus the inactive file cache (docker stats' figure: the cache the kernel reclaims before it OOM-kills);
limit = the cgroup's limit (None: none, or unknown); oom_kills = the kernel's OOM kills in this cgroup so far.
cgroup v2 (memory.current / memory.max / memory.events) or v1 (memory/memory.usage_in_bytes / limit_in_bytes /
oom_control); every value None where it cannot be read. Standard library only."""
import os

_NO_LIMIT = 1 << 62   # (v1 reports 'no limit' as a huge page-rounded number)


def _read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def _kv(text):
    out = {}
    for ln in (text or '').splitlines():
        k, _, v = ln.partition(' ')
        if v.strip().lstrip('-').isdigit():
            out[k] = int(v)
    return out


def cgroup_mem(root='/sys/fs/cgroup'):
    """(used bytes, limit bytes, oom kills): see the module's doc."""
    if os.path.exists(os.path.join(root, 'memory.current')):   # v2
        cur = _read(os.path.join(root, 'memory.current'))
        st = _kv(_read(os.path.join(root, 'memory.stat')))
        mx = (_read(os.path.join(root, 'memory.max')) or '').strip()
        ev = _kv(_read(os.path.join(root, 'memory.events')))
        used = int(cur) - st.get('inactive_file', 0) if cur and cur.strip().isdigit() else None
        limit = int(mx) if mx.isdigit() else None
        return used, limit, ev.get('oom_kill')
    d = os.path.join(root, 'memory')   # v1
    cur = (_read(os.path.join(d, 'memory.usage_in_bytes')) or '').strip()
    lim = (_read(os.path.join(d, 'memory.limit_in_bytes')) or '').strip()
    st = _kv(_read(os.path.join(d, 'memory.stat')))
    oc = _kv(_read(os.path.join(d, 'memory.oom_control')))
    used = int(cur) - st.get('total_inactive_file', st.get('inactive_file', 0)) if cur.isdigit() else None
    limit = int(lim) if lim.isdigit() and int(lim) < _NO_LIMIT else None
    return used, limit, oc.get('oom_kill')


if __name__ == '__main__':   # (shardk_in.sh: one line, 'used_gb limit_gb oom_kills', '-' where unknown)
    u, l_, k = cgroup_mem()
    print(' '.join('-' if x is None else f'{x / 2 ** 30:.2f}' for x in (u, l_)), '-' if k is None else k)
