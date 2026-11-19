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
    sys.argv = rest
    runpy.run_path(rest[0], run_name='__main__')
