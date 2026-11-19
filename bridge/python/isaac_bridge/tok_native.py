"""P package (2026-11-19): the thin TokPolicy driver's host side in C++ (bridge/engine/abp_tok_core.h), built on first use.

load() returns the extension module (check_batch, fold_pack, crn) or None (REASON says why: Windows, no g++, no torch's
pybind11 headers, a failed build). Like actor_native: g++ against this Python and torch's pybind11 headers (no torch
library linked: the core is torch-free), into a per-user cache keyed by the sources' hash, built atomically.
v3_consts() / fold_layout() give the C++ side every number it uses, from schema_v3 / v3_input / ROW / FoldLayout:
nothing of the schema is written in the C++ sources.
"""
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
SOURCES = (ENGINE / 'abp_tok_py.cpp', ENGINE / 'abp_tok_core.h')
REASON = None
_MOD = []


def build(verbose=False):
    """Path of the compiled module for these sources and this Python, None (REASON set) when it cannot be built."""
    global REASON
    if os.name == 'nt':
        REASON = 'windows'
        return None
    try:
        import torch
    except ImportError as e:
        REASON = f'no torch (its pybind11 headers): {e}'
        return None
    try:
        code = b''.join(Path(s).read_bytes() for s in SOURCES)
    except OSError as e:
        REASON = f'sources: {e}'
        return None
    tag = hashlib.sha256(code + sys.version.encode() + torch.__version__.encode()).hexdigest()[:16]
    cache = Path(os.environ.get('ISAAC_ACTOR_CACHE', Path.home() / '.cache' / 'isaac-actor'))
    name = f'abp_tok_{tag}'
    path = cache / f'{name}.so'
    if path.is_file():
        return str(path)
    inc = [Path(torch.__file__).parent / 'include', sysconfig.get_paths()['include'], ENGINE]
    cmd = ['g++', '-O2', '-shared', '-fPIC', '-fvisibility=hidden', '-std=c++17', '-ffp-contract=off',
           f'-DTOK_EXTENSION_NAME={name}'] + [f'-I{p}' for p in inc] + [str(SOURCES[0])]
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix='.so', dir=str(cache))
        os.close(fd)
        r = subprocess.run(cmd + ['-o', tmp], capture_output=True, text=True, timeout=600)
        if r.returncode:
            try:   # (the whole compiler output beside the cache: the first error is what matters)
                (cache / f'{name}.err').write_text(r.stderr)
            except OSError:
                pass
            REASON = f'build failed (all of it: {cache / (name + ".err")}): ' + r.stderr[:2000] + ' ... ' + r.stderr[-1000:]
            if verbose:
                print(REASON, file=sys.stderr)
            os.unlink(tmp)
            return None
        os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError) as e:
        REASON = f'build: {e}'
        return None
    return str(path)


def load(verbose=False):
    """The module, or None (REASON)."""
    if _MOD:
        return _MOD[0]
    path = build(verbose)
    if path is None:
        return None
    name = Path(path).stem
    spec = importlib.util.spec_from_loader(name, importlib.machinery.ExtensionFileLoader(name, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD.append(mod)
    return mod


def v3_consts():
    """Every number abp_tok_core's check uses (schema_v3, v3_input)."""
    from . import schema_v3 as V3
    from . import v3_input as VI
    return dict(header=V3.HEADER, h_p=V3.H_P, fixed=V3.FIXED, body=V3.BODY, ef=V3.EF, nf=V3.NF, sf=V3.SF,
                frame_bytes=V3.FRAME_BYTES, magic=float(V3.MAGIC), version=float(V3.VERSION),
                version_word=float(V3.VERSION_WORD), main_role=float(VI._MAIN), cap_body=V3.CAP_BODY,
                cap_entity=V3.CAP_ENTITY, n_max=VI.N_MAX_V3, s_max=VI.S_MAX_V3, p_max=VI.P_MAX,
                p0_role=VI.P0_WORDS[0], p0_known=VI.P0_WORDS[1], e_buckets=list(VI.E_BUCKETS))


def fold_layout(lay):
    """A tok_policy.FoldLayout as abp_tok_core's Fold (with the ROW field offsets it reads)."""
    from .tok_obs import ROW
    from .tok_policy import ROW_BYTES
    off_e, per_e, end_e = lay.ent_span
    return dict(n=lay.n, nh=lay.nh, H=lay.H, cw=lay.cw, dest_at=lay.dest_at, row_bytes=ROW_BYTES, mem=lay.mem,
                t_on=lay.t_on, crn=lay.crn, sizes=list(lay.sizes), rsizes=list(lay.rsizes), off_e=off_e, per_e=per_e,
                end_e=end_e, off_n_ent=ROW.fields['n_ent'][1],
                off_crn=ROW.fields['crn'][1] if 'crn' in ROW.dtype.names else -1, off_t=ROW.fields['t'][1])
