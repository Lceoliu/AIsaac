"""Run a Python script under a probe launcher that appends its own arguments (cpurun.sh's --groups-file / --out): the
arguments after '--' are the script and its own; everything before is ignored.

   usage: python tools/argv_wrap.py [launcher arguments ...] -- <script.py> [script arguments ...]"""
import runpy
import sys

if __name__ == '__main__':
    a = sys.argv[1:]
    if '--' not in a:
        sys.exit('argv_wrap: no -- <script>')
    rest = a[a.index('--') + 1:]
    if rest[0] == '-m':   # (a module: -- -m pytest tests/x.py; the working directory on the path, as python -m does)
        import os
        sys.path.insert(0, os.getcwd())
        sys.argv = rest[1:]
        runpy.run_module(rest[1], run_name='__main__', alter_sys=True)
    else:
        sys.argv = rest
        runpy.run_path(rest[0], run_name='__main__')
