"""B56: ISAAC_FX_CACHE switches Inductor's FX graph cache and the AOTAutograd cache (default: on since B56;
ISAAC_FX_CACHE=0 turns both off)."""
import os
import subprocess
import sys

CODE = ("import torch._inductor.config as i, torch._functorch.config as f\n"
        "from isaac_bridge import block_compile as b\n"
        "b.configure()\n"
        "c = b.cache_counters()\n"
        "assert set(c) == {'fxgraph_hit', 'fxgraph_miss', 'fxgraph_bypass', 'aot_hit', 'aot_miss',\n"
        "                  'aot_guard_miss', 'aot_bypass'} and all(v == 0 for v in c.values()), c\n"
        "print(b.FX_CACHE, i.fx_graph_cache, f.enable_autograd_cache if b.FX_CACHE else '-')\n")


def _run(value):
    env = dict(os.environ)
    env.pop('ISAAC_FX_CACHE', None)
    if value is not None:
        env['ISAAC_FX_CACHE'] = value
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = here + os.pathsep + env.get('PYTHONPATH', '')
    out = subprocess.run([sys.executable, '-c', CODE], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def test_fx_cache_default_on():
    assert _run(None) == ['True', 'True', 'True']
    assert _run('1') == ['True', 'True', 'True']


def test_fx_cache_off():
    assert _run('0') == ['False', 'False', '-']
