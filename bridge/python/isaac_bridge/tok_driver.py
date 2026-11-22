"""P package phase 1 (2026-11-19): the thin TokPolicy driver (bridge/engine/abp_tok_driver.cpp + abp_tok_core.h).

Milestone A: Middle(graph_actor) -- one call's middle in C++ on a GraphActor(fold=True)'s own captured graphs and
buffers (registered, never captured in C++): the v3 check, the fold staging with the frame-prefix rule, the cf_snap
bank copies, three H2D copies, the replay, the D2H copy, the step's K / V, s and o into the output ring (set_ring),
on the GraphActor's stream. The same-batch gate compares it with GraphActor.submit / result (tools/p_replay.py, the
'cpp' backend). load() builds the extension like actor_native (g++ against the running torch, a per-user cache keyed
by the sources and torch's version, atomic); None and REASON when it cannot.
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

import numpy as np

ENGINE = Path(__file__).resolve().parents[2] / 'engine'
SOURCES = (ENGINE / 'abp_tok_driver.cpp', ENGINE / 'abp_tok_core.h', ENGINE / 'abp_tok_thread.h')
REASON = None
_MOD = []


def build(verbose=False, need_cuda=True):
    """need_cuda False: build without a GPU (a compile check in a CPU container)."""
    global REASON
    if os.name == 'nt':
        REASON = 'windows'
        return None
    try:
        import torch
    except ImportError as e:
        REASON = f'no torch: {e}'
        return None
    if need_cuda and not torch.cuda.is_available():
        REASON = 'torch without CUDA'
        return None
    code = b''.join(Path(s).read_bytes() for s in SOURCES)
    tag = hashlib.sha256(code + torch.__version__.encode() + sys.version.encode()).hexdigest()[:16]
    cache = Path(os.environ.get('ISAAC_ACTOR_CACHE', Path.home() / '.cache' / 'isaac-actor'))
    name = f'abp_tokdrv_{tag}'
    path = cache / f'{name}.so'
    if path.is_file():
        return str(path)
    torch_dir = Path(torch.__file__).parent
    nv = glob.glob(str(torch_dir.parent / 'nvidia' / '*' / 'include'))
    inc = [torch_dir / 'include', torch_dir / 'include' / 'torch' / 'csrc' / 'api' / 'include',
           sysconfig.get_paths()['include'], ENGINE] + nv
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if os.path.isdir(os.path.join(cuda_home, 'include')):
        inc.append(os.path.join(cuda_home, 'include'))
    cmd = ['g++', '-O2', '-shared', '-fPIC', '-fvisibility=hidden', '-std=c++17', '-ffp-contract=off',
           f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}', f'-DTORCH_EXTENSION_NAME={name}',
           '-DTORCH_API_INCLUDE_EXTENSION_H', '-DUSE_CUDA'] + [f'-I{p}' for p in inc] + [str(SOURCES[0])]
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix='.so', dir=str(cache))
        os.close(fd)
        cmd += ['-o', tmp, f'-L{torch_dir / "lib"}', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda',
                '-ltorch_python', f'-Wl,-rpath,{torch_dir / "lib"}']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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


def load(verbose=False, need_cuda=True):
    """need_cuda False: also without a GPU (the thread's tests in fake mode)."""
    if _MOD:
        return _MOD[0]
    import torch  # noqa: F401  (torch's libraries before the extension's)
    path = build(verbose, need_cuda=need_cuda)
    if path is None:
        return None
    name = Path(path).stem
    spec = importlib.util.spec_from_loader(name, importlib.machinery.ExtensionFileLoader(name, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD.append(mod)
    return mod


def row_fields():
    from .tok_obs import ROW
    return {k: int(ROW.fields[k][1]) for k in ('visible_schema', 'visible_bytes', 'visible_frame', 'first')}


class Middle:
    """One GraphActor(fold=True, v3, temporal)'s calls through abp_tok_driver's TokMiddle (milestone A)."""

    def __init__(self, ga, ring=None):
        import torch
        from . import tok_native
        mod = load(verbose=True)
        if mod is None:
            raise RuntimeError(f'tok_driver: {REASON}')
        if not (ga.fold and ga.v3 and ga.t_on) or ga.mem or ga.n_extra:
            raise ValueError('tok_driver.Middle: phase 1 is GraphActor(fold=True), v3, temporal, no GRU, no choice')
        self.ga, self.nh = ga, ga.nh
        cols = ga.out_host.shape[1]
        D = ga.model.tatt.D
        self.m = mod.TokMiddle(tok_native.fold_layout(ga.fold_lay), tok_native.v3_consts(), row_fields(), cols,
                               ga.t_L2D, D)
        self.m.set_buffers(ga.raw, ga.ctrl, ga.v3_counts, ga.stage, ga.ctrl_stage, ga.v3_counts_stage, ga.out_host)
        s = ga.stream if ga.stream is not None else torch.cuda.current_stream()
        self.m.set_stream(int(s.stream_id), int(s.device_index))
        for i, g in enumerate(ga.graphs):
            tkv, o = ga.t_outs[i]
            self.m.register_graph(i, g, ga.outs[i], tkv, o)
        if ring is not None:
            self.m.set_ring(ring['kv'], ring['s'], ring['o'], int(ring.get('obase', 0)))

    def set_cf(self, ts, bank):
        self.m.set_cf([getattr(ts, k) for k in ('kv', 'steps', 'nroom', 'prev_o', 'prev_stage', 'prev_map')],
                      [getattr(bank, k) for k in ('kv', 'steps', 'nroom', 'prev_o', 'prev_stage', 'prev_map')])

    def call(self, rows, pending, widx, ring_rows=None, ukeys=None, cf=None):
        """As GraphActor.submit + result: (actions, logp, value, vg or None, t_out dict) -- or raises ValueError with
        the v3 reject code where the check refuses the call."""
        k = len(rows)
        rb = np.ascontiguousarray(rows).view(np.uint8).reshape(k, -1)
        code, rec, out, eb, graph = self.m.call(rb, np.ascontiguousarray(pending, np.int64),
                                                np.ascontiguousarray(widx, np.int64),
                                                None if ring_rows is None else np.ascontiguousarray(ring_rows, np.int64),
                                                None if ukeys is None else ukeys[0], None if ukeys is None else ukeys[1],
                                                None if cf is None else cf[0], None if cf is None else cf[1])
        if code:
            raise ValueError(f'tok_driver: v3 check refused the call (code {code}, record {rec})')
        nh, ga = self.nh, self.ga
        at = nh + 2
        vg = out[:, at] if ga.vg_on else None
        tc = out[:, at + int(ga.vg_on):]
        t = dict(create=tc[:, 0] > 0.5, nroom=np.rint(tc[:, 1]).astype(np.int64), steps=np.rint(tc[:, 2]).astype(np.int64),
                 stage_prev=np.rint(tc[:, 3]).astype(np.int64))
        return out[:, :nh].astype(np.int64), out[:, nh], out[:, nh + 1], vg, t
