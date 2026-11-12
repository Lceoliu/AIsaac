"""isaac_bridge.cgroup_mem on fake cgroup v1 / v2 trees (the guard's memory warning and its cgroup-OOM cause)."""
from isaac_bridge.cgroup_mem import cgroup_mem

G = 2 ** 30


def _w(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)


def test_v2(tmp_path):
    _w(tmp_path / 'memory.current', f'{100 * G}\n')
    _w(tmp_path / 'memory.stat', f'anon {80 * G}\nfile {20 * G}\ninactive_file {15 * G}\n')
    _w(tmp_path / 'memory.max', f'{128 * G}\n')
    _w(tmp_path / 'memory.events', 'low 0\nhigh 0\nmax 7\noom 1\noom_kill 1\n')
    assert cgroup_mem(str(tmp_path)) == (85 * G, 128 * G, 1)
    _w(tmp_path / 'memory.max', 'max\n')
    assert cgroup_mem(str(tmp_path))[1] is None


def test_v1(tmp_path):
    d = tmp_path / 'memory'
    _w(d / 'memory.usage_in_bytes', f'{90 * G}\n')
    _w(d / 'memory.limit_in_bytes', f'{128 * G}\n')
    _w(d / 'memory.stat', f'cache {12 * G}\ntotal_inactive_file {10 * G}\n')
    _w(d / 'memory.oom_control', 'oom_kill_disable 0\nunder_oom 0\noom_kill 2\n')
    assert cgroup_mem(str(tmp_path)) == (80 * G, 128 * G, 2)
    _w(d / 'memory.limit_in_bytes', '9223372036854771712\n')
    assert cgroup_mem(str(tmp_path))[1] is None


def test_unreadable(tmp_path):
    assert cgroup_mem(str(tmp_path / 'none')) == (None, None, None)
