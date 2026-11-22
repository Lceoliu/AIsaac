"""(2026-11-20, the native ROW encoder v9) Run the FastRow tests against the libabp_turbo.so a CPU runner just built
($ABP_HOME/tools/libabp_turbo.so, from this tree's abp_turbo.c): tests/test_fast_row_v9.py (the C path against
encode_row) and the FastRow parts of test_visible_v3_worker.py; then the older smoke tests' FastRow parts.

   usage (bridge/python): python tools/v8_selftest.py [pytest args ...]     exit: pytest's"""
import os
import subprocess
import sys

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    py = os.path.dirname(here)
    so = os.environ.get('ISAAC_RL_TEST_TURBO_SO') or os.path.join(os.environ.get('ABP_HOME', ''), 'tools',
                                                                   'libabp_turbo.so')
    if not os.path.isfile(so):
        sys.exit(f'v8_selftest: no library at {so}')
    os.environ['ISAAC_RL_TEST_TURBO_SO'] = so
    os.environ['ISAAC_RL_FAST_ROW'] = '1'
    os.chdir(py)
    sys.path.insert(0, py)
    import pytest
    rc = pytest.main(['-q', '-x', 'tests/test_fast_row_v9.py', 'tests/test_fast_row_v10.py', 'tests/test_visible_v3_worker.py',
                      'tests/test_ring_compact.py', '-p', 'no:cacheprovider']
                     + sys.argv[1:])
    for t in ('tests/test_stall_progress_smoke.py', 'tests/test_obs_v2_smoke.py'):
        r = subprocess.run([sys.executable, t], env=dict(os.environ)).returncode
        print(f'{t}: rc {r}', flush=True)
        rc = rc or r
    sys.exit(rc)
