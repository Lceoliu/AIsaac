"""B56: ISAAC_FX_CACHE switches Inductor's FX graph cache and the AOTAutograd cache (default: off, as before)."""
import os
import subprocess
import sys

CODE = ("import torch._inductor.config as i, torch._functorch.config as f\n"
        "from isaac_bridge import block_compile as b\n"
        "b.configure()\n"
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


def test_fx_cache_default_off():
    assert _run(None) == ['False', 'False', '-']
    assert _run('0') == ['False', 'False', '-']


def test_fx_cache_on():
    assert _run('1') == ['True', 'True', 'True']
