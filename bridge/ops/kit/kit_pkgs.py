"""kit: the effective Python package set and an import test, read from stdin inside a container:
    docker run -i ... <python> - < kit_pkgs.py
Prints host-path-free lines (diffable between the old setup and the kit image):
    PY <version>                          the interpreter
    PKG <name>==<version> <origin>        every distribution, first one on sys.path wins (as import does);
                                          origin = venv | pydeps | system | other
    SHADOWED <name>==<version> <origin>   a later copy hidden by the first (e.g. venv numpy under pydeps numpy)
    IMPORT <module> ok <version> | IMPORT <module> FAIL <error>
    TORCH <version> cuda <cuda> cudnn <cudnn>
rc 0 = every import ok."""
import importlib
import importlib.metadata as md
import re
import sys

print('PY', sys.version.split()[0])


def origin(path):
    p = str(path)
    if 'pydeps' in p:
        return 'pydeps'
    if '.venv' in p or '/venv/' in p:
        return 'venv'
    if p.startswith('/usr/lib/python3'):
        return 'system'
    return 'other'


seen = {}
for d in md.distributions():   # in sys.path order
    name = re.sub(r'[-_.]+', '-', (d.metadata['Name'] or '?')).lower()
    o = origin(getattr(d, '_path', ''))
    if name in seen:
        print('SHADOWED', f'{name}=={d.version}', o)
        continue
    seen[name] = (d.version, o)
for name in sorted(seen):
    print('PKG', f'{name}=={seen[name][0]}', seen[name][1])

ok = True
for mod in ('torch', 'numpy', 'zstandard', 'lz4', 'lz4.frame', 'pytest', 'gymnasium', 'psutil', 'cloudpickle'):
    try:
        m = importlib.import_module(mod)
        print('IMPORT', mod, 'ok', getattr(m, '__version__', '-'))
    except Exception as e:   # noqa: BLE001
        ok = False
        print('IMPORT', mod, 'FAIL', f'{type(e).__name__}: {e}'[:200])
try:
    import torch
    print('TORCH', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
except Exception as e:   # noqa: BLE001
    ok = False
    print('TORCH FAIL', e)
sys.exit(0 if ok else 1)
