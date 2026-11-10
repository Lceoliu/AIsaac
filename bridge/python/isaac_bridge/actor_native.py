"""B52: the actor's C++ scheduling hot path (bridge/engine/abp_actor.cpp + abp_visible.c), built on first use.

load() returns the extension module (PageStore, ActorDriver, run_graph) or None: ISAAC_BLOCK_CPP_ACTOR=0, no CUDA
torch, Windows, no g++, or a failed build (the reason is kept in REASON and printed once). The .so is built with g++
against the running torch (its own pybind11 headers, CUDA headers from the venv's nvidia/*/include packages so the
CUDA version matches torch's) into a per-user cache keyed by the sources' hash and torch's version; the build is
atomic (temporary file, rename), so concurrent processes are safe. Everything that uses it must fall back to the
Python path when load() is None (train_tok, PagedStore, ActorStepGraph) and tests skip.
"""
import glob
import hashlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / 'engine'
SOURCES = (ENGINE / 'abp_actor.cpp', ENGINE / 'abp_visible.c', ENGINE / 'abp_core.h')   # (B58c: the core header in the hash)
ENABLED = os.environ.get('ISAAC_BLOCK_CPP_ACTOR', '1') != '0'
REASON = None
_MOD = []


def _torch():
    import torch
    return torch


def build(verbose=False):
    """Path of the compiled extension for this source and torch, None (REASON set) when it cannot be built."""
    global REASON
    if os.name == 'nt':
        REASON = 'windows'; return None
    try:
        torch = _torch()
    except ImportError as e:
        REASON = f'no torch: {e}'; return None
    if not torch.cuda.is_available():
        REASON = 'torch without CUDA'; return None
    try:
        code = b''.join(Path(s).read_bytes() for s in SOURCES)
    except OSError as e:
        REASON = f'sources: {e}'; return None
    tag = hashlib.sha256(code + torch.__version__.encode() + sys.version.encode()).hexdigest()[:16]
    cache = Path(os.environ.get('ISAAC_ACTOR_CACHE', Path.home() / '.cache' / 'isaac-actor'))
    name = f'abp_actor_{tag}'
    path = cache / f'{name}.so'
    if path.is_file():
        return str(path)
    torch_dir = Path(torch.__file__).parent
    nv = glob.glob(str(torch_dir.parent / 'nvidia' / '*' / 'include'))
    inc = [torch_dir / 'include', torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include', torch_dir / 'include' / 'TH',
           torch_dir / 'include' / 'THC', sysconfig.get_paths()['include']] + nv
    # (the pip CUDA headers lack crt/host_defines.h: the system CUDA's include comes last, for that only)
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if os.path.isdir(os.path.join(cuda_home, 'include')):
        inc.append(os.path.join(cuda_home, 'include'))
    cmd = ['g++', '-O2', '-shared', '-fPIC', '-fvisibility=hidden',
           f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}', f'-DTORCH_EXTENSION_NAME={name}',
           '-DTORCH_API_INCLUDE_EXTENSION_H', '-DUSE_CUDA'] + [f'-I{p}' for p in inc]
    cmd += ['-x', 'c', '-ffp-contract=off', str(SOURCES[1]), '-x', 'c++', '-std=c++17', str(SOURCES[0])]
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix='.so', dir=str(cache))
        os.close(fd)
        cmd += ['-o', tmp, f'-L{torch_dir / "lib"}', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda',
                '-ltorch_python', f'-Wl,-rpath,{torch_dir / "lib"}']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode:
            REASON = 'build failed: ' + r.stderr[-3000:]
            if verbose:
                print(REASON, file=sys.stderr)
            os.unlink(tmp)
            return None
        os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError) as e:
        REASON = f'build error: {e}'
        try:
            os.unlink(tmp)
        except (OSError, UnboundLocalError):
            pass
        return None
    return str(path)


def load(verbose=None):
    """The extension module, or None when unavailable (REASON says why). Cached per process."""
    global REASON
    if _MOD:
        return _MOD[0]
    if verbose is None:
        verbose = os.environ.get('ISAAC_BLOCK_TIMING', '0') == '1'
    mod = None
    if not ENABLED:
        REASON = 'ISAAC_BLOCK_CPP_ACTOR=0'
    else:
        path = build(verbose)
        if path is not None:
            try:
                _torch()   # (libtorch_python's symbols must be loaded first)
                name = Path(path).stem
                loader = importlib.machinery.ExtensionFileLoader(name, path)
                spec = importlib.util.spec_from_file_location(name, path, loader=loader)
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
                from .structured_input import VERSION
                if mod.visible_version() != VERSION:
                    REASON = 'abp_visible schema version mismatch'; mod = None
            except Exception as e:   # noqa: BLE001 (any load failure: the Python path)
                REASON = f'load failed: {e}'; mod = None
    if mod is None and verbose:
        print(f'B52 cpp actor unavailable: {REASON}', file=sys.stderr, flush=True)
    _MOD.append(mod)
    return mod
