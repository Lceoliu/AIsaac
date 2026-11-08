"""B45: the visible-v2 frame of a lean observation in C (bridge/engine/abp_visible.c), for the sampler workers.

frame(o) returns structured_input.observation_frame(o)'s bytes (byte for byte: abplus_probe_visible_frame.py and
tests/test_visible_native.py compare them) at a fraction of the cost: the reference's nearest-cell search is a Python
min() over the room's ~450 cells for every grid record, door and body (~9 ms per decision on the workers, B45).
The library is built from the repository's source with gcc on first use into a per-user cache keyed by the source's
hash (atomic rename: concurrent workers are safe) and loaded with ctypes; without gcc (or ISAAC_VISIBLE_NATIVE=0)
frame() is the reference. A negative return of the C side (a value the reference rejects) calls the reference, which
raises the same error.
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'engine' / 'abp_visible.c'
ENABLED = os.environ.get('ISAAC_VISIBLE_NATIVE', '1') != '0'
_LIB = {}


def build(source=SOURCE):
    """Path of the compiled library for this source (built if needed), None when it cannot be built."""
    try:
        code = Path(source).read_bytes()
    except OSError:
        return None
    tag = hashlib.sha256(code).hexdigest()[:16]
    cache = Path(os.environ.get('ISAAC_VISIBLE_CACHE', Path.home() / '.cache' / 'isaac-visible'))
    path = cache / f'libabp_visible-{tag}.so'
    if path.is_file():
        return str(path)
    if os.name == 'nt':
        return None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix='.so', dir=str(cache))
        os.close(fd)
        subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-ffp-contract=off', '-o', tmp, str(source), '-lm'],
                       check=True, capture_output=True, timeout=120)
        os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError):
        try:
            os.unlink(tmp)
        except (OSError, UnboundLocalError):
            pass
        return None
    return str(path)


class Library:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        lib.abp_visible_version.restype = ctypes.c_int
        from .structured_input import VERSION
        if lib.abp_visible_version() != VERSION:
            raise OSError('abp_visible schema version mismatch')
        fn = lib.abp_visible_encode
        P, I, L, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double
        fn.argtypes = [P, I, D, D, L, P, I, P, I, P, I, P, I, P, I, I, I, P, P, I, P, I, L, L, D, P, L]
        fn.restype = L
        self.fn, self.path = fn, path

    def encoder(self):
        return Encoder(self)


def load(path=None):
    """The library (path or the built repository source), None when unavailable."""
    if path is None:
        if not ENABLED:
            return None
        path = build()
        if path is None:
            return None
    if path not in _LIB:
        try:
            _LIB[path] = Library(path)
        except OSError:
            _LIB[path] = None
    return _LIB[path]


_EMPTY = np.zeros(1, np.float64)


class Encoder:
    """Per-process frame encoder: the terrain's cell / grid arrays are cached per terrain block (the decoder replaces
    its terrain dict and grid list when the block's content changes)."""

    def __init__(self, lib):
        from .structured_input import FRAME_BYTES
        self.lib = lib
        self.out = np.zeros(FRAME_BYTES // 4, np.float32)
        self.key = None
        self.keep = None
        self.cells = self.grid = None
        self.decoder = None
        self.native = self.reference = 0

    def _terrain(self, o):
        key = (id(o.terrain), id(o.grid))
        if key != self.key:
            self.keep = (o.terrain, o.grid)   # (ids stay valid while referenced)
            cells = o.terrain['cells']
            self.cells = np.ascontiguousarray(np.asarray(cells, np.float64).reshape(len(cells), 10)) if cells else \
                np.zeros((0, 10), np.float64)
            grid = [g[:7] for g in (o.grid or ())]
            self.grid = np.ascontiguousarray(np.asarray(grid, np.float64).reshape(len(grid), 7)) if grid else \
                np.zeros((0, 7), np.float64)
            self.key = key

    def frame(self, o):
        from .structured_input import observation_frame, SCHEMA
        ext = o.visible
        if ext is None or ext['schema'] != SCHEMA or len(ext['bodies']) != len(o.players) or o.terrain is None or \
                o.terrain['width'] > 28 or o.terrain['height'] > 16:
            return observation_frame(o)   # (raises the reference's error)
        self._terrain(o)
        bodies = np.asarray([(m['controller'], m['controller_known'], m['animation'], m['animation_known'])
                             for m in ext['bodies']], np.float64).reshape(-1, 4)
        inv = np.ascontiguousarray(np.asarray(ext['inventory'], np.float32).reshape(-1, 4))
        slots = np.ascontiguousarray(np.asarray(ext['slots'], np.float32).reshape(-1, 13))
        pl, dr, en, mp = o.players, o.doors, o.entities, o.map
        n = self.lib.fn(pl.ctypes.data, len(pl), float(o.room[8]), float(o.room[9]), int(o.room[4]),
                        dr.ctypes.data, len(dr), en.ctypes.data, len(en), mp.ctypes.data, len(mp),
                        self.cells.ctypes.data, len(self.cells), self.grid.ctypes.data, len(self.grid),
                        int(o.terrain['width']), int(o.terrain['height']), bodies.ctypes.data, inv.ctypes.data,
                        len(inv), slots.ctypes.data, len(slots), int(ext['curses']), int(ext['stage']),
                        float(ext['capabilities']), self.out.ctypes.data, self.out.nbytes)
        if n < 0:
            self.reference += 1
            return observation_frame(o)
        self.native += 1
        return self.out.view(np.uint8)[:n].tobytes()

    def encode(self, payload):
        """frame() of a raw payload through this encoder's own LeanDecoder (probes and tests); None without a
        visible block."""
        from .abplus_lean import LeanDecoder
        if self.decoder is None:
            self.decoder = LeanDecoder()
        o = self.decoder.decode(payload)
        return self.frame(o) if o.visible is not None else None


_DEFAULT = []


def frame(o):
    """observation_frame(o)'s bytes: the native encoder of this process when available, else the reference."""
    if not _DEFAULT:
        lib = load()
        _DEFAULT.append(lib.encoder() if lib is not None else None)
    enc = _DEFAULT[0]
    if enc is None:
        from .structured_input import observation_frame
        return observation_frame(o)
    return enc.frame(o)
