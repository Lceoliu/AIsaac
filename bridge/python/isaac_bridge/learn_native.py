"""B60: the block learner's host packing in C (bridge/engine/abp_learn.c), outside the GIL.

The learner's host preparation (block_learner._prepare_packed) spent most of its time in numpy gathers and Python
loops that hold the GIL, competing with the learner thread and the rollout consumers. The byte-moving parts run here
in C, called through ctypes.CDLL (the GIL is released for the whole call):

* decode(parts)            structured_input.decode_frames_compact(parts, packed=True), byte for byte;
* concat(pieces, out, ...) block_learner._fill_merged's concatenation into the pinned staging slab;
* pages(store, which, ids, out) (B62 N3) PagedStore._gather's page rows, the Python or the native (abp_actor.cpp) store.
  Page tensors are allocated once and never moved or freed, and the rows a learner part reads belong to live
  snapshots (cpp design 9.1b), so a table of page addresses is reused until the store has more pages; it is rebuilt
  (through the store's own interface, the native store's mutex held per page lookup) only then.

ISAAC_BLOCK_NATIVE_PACK=0 (or no gcc) keeps the Python reference everywhere. The library is built from the
repository's source on first use into a per-user cache keyed by the source's hash (as visible_native).
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'engine' / 'abp_learn.c'
ENABLED = os.environ.get('ISAAC_BLOCK_NATIVE_PACK', '1') != '0'
VERSION = 6004
_LIB = {}
_LOCK = threading.Lock()
P, L = ctypes.c_void_p, ctypes.c_long


def build(source=SOURCE):
    """Path of the compiled library for this source (built if needed), None when it cannot be built."""
    try:
        code = Path(source).read_bytes()
    except OSError:
        return None
    tag = hashlib.sha256(code).hexdigest()[:16]
    cache = Path(os.environ.get('ISAAC_LEARN_NATIVE_CACHE', Path.home() / '.cache' / 'isaac-learn-native'))
    suffix = '.dll' if os.name == 'nt' else '.so'
    path = cache / f'libabp_learn-{tag}{suffix}'
    if path.is_file():
        return str(path)
    tmp = None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(cache))
        os.close(fd)
        subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', tmp, str(source), '-lm'],
                       check=True, capture_output=True, timeout=120)
        os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError):
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return None
    return str(path)


# ctypes signatures of abp_learn.c's exported functions (argument types, result type); tests check every argument
# list against the C source's parameter lists (test_b60_native_pack: no compiler needed)
SIGNATURES = {
    'abp_learn_scan': ([L, P, P, P, P] + [L]*8 + [P], L),
    'abp_learn_gather': ([L, P, P, L, L, L, L, P] + [P]*7, None),
    'abp_learn_concat': ([L, P, P, P, L], None),
    'abp_learn_pages': ([L, P, L, L, P, L, P], L),
    'abp_learn_reader': ([L, P, L, L, L, L, L, L] + [P]*19, L),
}


class Library:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        lib.abp_learn_version.restype = ctypes.c_int
        if lib.abp_learn_version() != VERSION:
            raise OSError('abp_learn version mismatch')
        fns = {}
        for name, (args, res) in SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
            fns[name] = fn
        self.scan, self.gather, self.concat_ = fns['abp_learn_scan'], fns['abp_learn_gather'], fns['abp_learn_concat']
        self.pages_ = fns['abp_learn_pages']
        self.reader_ = fns['abp_learn_reader']
        self.path = path


def load():
    """The loaded library, or None (disabled, no compiler, or a failed build / load)."""
    if not ENABLED:
        return None
    with _LOCK:
        if 'lib' not in _LIB:
            path = build()
            try:
                _LIB['lib'] = Library(path) if path else None
            except OSError:
                _LIB['lib'] = None
        return _LIB['lib']


def _addr(a):
    return a.__array_interface__['data'][0]


def decode(parts):
    """structured_input.decode_frames_compact(parts, packed=True) in C (same dicts, same bytes); the reference when
    the library is unavailable or any row fails the header check (the reference then raises its own error)."""
    from . import structured_input as si
    lib = load()
    if lib is None or not parts:
        return si.decode_frames_compact(parts, packed=True)
    heads, datas, schemas, sizes, avails, keep, lens = [], [], [], [], [], [], []
    for part in parts:
        if isinstance(part, tuple):
            rows, buf = part
            buf = np.ascontiguousarray(np.asarray(buf, np.uint8))
            off = np.asarray(rows['visible_off'], np.int64)
            if len(off) and (off % 4).any():   # (the reference reads such a frame's header at off, its floats at
                #                                 off // 4 * 4: the whole batch to the reference)
                return si.decode_frames_compact(parts, packed=True)
            a = np.int64(_addr(buf)) if len(buf) else np.int64(0)
            heads.append(a+off)
            datas.append(a+off//4*4)
            avails.append(len(buf)//4*4-off//4*4)
            keep.append(buf)
        else:
            rows = part
            vf = rows['visible_frame']
            n = len(rows)
            if vf.ndim != 2 or (n and vf.strides[1] != 1):
                return si.decode_frames_compact(parts, packed=True)
            a = np.int64(_addr(vf)) if n else np.int64(0)
            at = a+np.arange(n, dtype=np.int64)*np.int64(vf.strides[0] if n else 0)
            heads.append(at)
            datas.append(at)
            avails.append(np.full(n, vf.shape[1], np.int64))
            keep.append(vf)
        schemas.append(np.asarray(rows['visible_schema'], np.int64))
        sizes.append(np.asarray(rows['visible_bytes'], np.int64))
        lens.append(len(rows))
    head, data, schema, size, avail = (np.ascontiguousarray(np.concatenate(x), np.int64)
                                       for x in (heads, datas, schemas, sizes, avails))
    m = len(head)
    counts = np.empty((m, 5), np.int64)
    widths = np.asarray([si.RAGGED[k] for k in si.RAGGED], np.int64)
    bad = lib.scan(m, _addr(head), _addr(schema), _addr(size), _addr(avail), si.VERSION, si.FIXED,
                   si.PF, si.RAGGED['local'], si.EF, si.NF, si.SF, si.FRAME_BYTES, _addr(counts)) if m else -1
    if bad >= 0:
        return si.decode_frames_compact(parts, packed=True)
    room_n, map_n = 1792, si.FIXED-si.HEADER-1792
    room = np.empty((m, room_n), np.float32)
    minimap = np.empty((m, map_n), np.float32)
    lists = {}
    for k, name in enumerate(si.RAGGED):
        lists[name] = np.empty((int(counts[:, k].sum()), int(widths[k])), np.float32)
    outs = [lists[name] for name in si.RAGGED]
    if m:
        lib.gather(m, _addr(data), _addr(counts), si.HEADER, room_n, map_n, si.FIXED, _addr(widths),
                   _addr(room), _addr(minimap), *[_addr(o) for o in outs])
    del keep
    row_at = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    cums = {name: np.concatenate(([0], np.cumsum(counts[:, k]))) for k, name in enumerate(si.RAGGED)}
    res = []
    for i in range(len(parts)):
        lo, hi = row_at[i], row_at[i+1]
        d = {'v_room': room[lo:hi], 'v_map': minimap[lo:hi], 'counts': counts[lo:hi]}
        for name in si.RAGGED:
            cum = cums[name]
            d['c_'+name] = lists[name][cum[lo]:cum[hi]]
        res.append(d)
    return res


def concat(pieces, out, zero_rows=0):
    """out (a C-contiguous numpy array) <- the pieces (arrays of out's dtype and row shape) one after another, then
    zero_rows zero rows. A piece of another dtype or layout is assigned by numpy (the reference's conversion)."""
    lib = load()
    ok = lib is not None and out.flags.c_contiguous and all(x.ndim == out.ndim and x.shape[1:] == out.shape[1:]
                                                            for x in pieces)
    if ok:   # (another dtype or a strided piece: numpy's conversion / copy first, as the reference's assignment)
        pieces = [x if x.dtype == out.dtype and x.flags.c_contiguous else np.ascontiguousarray(x, out.dtype)
                  for x in pieces]
    else:
        at = 0
        for x in pieces:
            out[at:at+len(x)] = x
            at += len(x)
        if zero_rows:
            out[at:] = 0
        return
    n = len(pieces)
    src = np.fromiter((_addr(x) if x.size else 0 for x in pieces), np.uint64, n)
    nb = np.fromiter((x.nbytes for x in pieces), np.int64, n)
    row = out.itemsize*int(np.prod(out.shape[1:], dtype=np.int64))
    if int(nb.sum())+zero_rows*row > out.nbytes:
        raise ValueError('concat: pieces larger than the destination')
    lib.concat_(n, _addr(src), _addr(nb), _addr(out), zero_rows*row)


_PAGE_TABLES = {}   # (id(store), which) -> (store ref, n pages, address table, row bytes, dtype)


def pages(store, which, ids, out):
    """store.kv / store.z / store.blocks (which: 'kv' | 'z' | 'blocks') of ids into out (a contiguous CPU tensor of the
    page dtype) in C; False when the library or the layout does not allow it (the caller runs the reference)."""
    import weakref
    lib = load()
    if lib is None or not hasattr(lib, 'pages_') or not out.is_contiguous():
        return False
    page_list = getattr(store, {'kv': 'kv_pages', 'z': 'z_pages', 'blocks': 'b_pages'}[which])
    dtype = store.kv_dtype if which == 'kv' else out.dtype
    if out.dtype != dtype:
        return False
    ids = np.ascontiguousarray(ids, np.int64)
    if not len(ids):
        return True
    n = len(page_list)
    key = (id(store), which)
    t = _PAGE_TABLES.get(key)
    if t is None or t[0]() is not store or t[1] != n:
        tensors = [page_list[p] for p in range(n)]
        row = out.element_size()*int(np.prod(out.shape[1:], dtype=np.int64))
        for pg in tensors:
            if (not pg.is_contiguous() or pg.dtype != dtype or pg.numel()*pg.element_size() != store.page*row):
                return False
        t = (weakref.ref(store), n, np.fromiter((pg.data_ptr() for pg in tensors), np.uint64, n), row, dtype)
        _PAGE_TABLES[key] = t
    _, n, table, row, _ = t
    if row != out.element_size()*int(np.prod(out.shape[1:], dtype=np.int64)):
        return False
    bad = lib.pages_(len(ids), _addr(ids), int(store.page), n, _addr(table), row, out.data_ptr())
    if bad >= 0:
        raise IndexError(f'page row {int(ids[bad])} outside the store')
    return True


def reader_bucket(ids, T, Bb, max_m, Pd, lat, mt, n_main, n_ukv, n_old_total, lens, row_start, at_k, nk, kinv,
                  kv_steps, old_start, n_old, old_ends, new_start, complete, new_ends):
    """(B62 N4) One Reader bucket's tables in C (block_learner._prepare_packed's loop): (zidx, real, kvidx, kp, ends,
    dst, src) as the reference builds them, or None when the library is unavailable (the caller runs the reference)."""
    lib = load()
    if lib is None or not hasattr(lib, 'reader_'):
        return None
    c = (lambda a: np.ascontiguousarray(a, np.int64))
    ids, lens, row_start, at_k, nk, kinv, kv_steps, old_start, n_old, old_ends, new_start, complete, new_ends = (
        c(x) for x in (ids, lens, row_start, at_k, nk, kinv, kv_steps, old_start, n_old, old_ends, new_start,
                       complete, new_ends))
    zidx = np.full((Bb, T), n_main, np.int64)
    real = np.empty(int(lens[ids].sum()), np.int64)
    kvidx = np.full((Bb, max(Pd, 0)), n_ukv, np.int64)
    kp = np.full((Bb, Pd*lat), -1, np.int64)
    ends = np.full((Bb, max_m), -1, np.int64)
    n_tok = int((n_old[ids]+complete[ids]).sum())*mt
    dst, src = np.empty(n_tok, np.int64), np.empty(n_tok, np.int64)
    got = lib.reader_(len(ids), _addr(ids), int(T), int(max_m), int(Pd), int(lat), int(mt), int(n_old_total),
                      *[_addr(x) for x in (lens, row_start, at_k, nk, kinv, kv_steps, old_start, n_old, old_ends,
                                            new_start, complete, new_ends, zidx, real, kvidx, kp, ends, dst, src)])
    if got != n_tok:
        raise RuntimeError(f'abp_learn_reader wrote {got} bank entries, {n_tok} expected')
    return zidx, real, kvidx, kp, ends, dst, src
