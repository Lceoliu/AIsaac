"""read-only on its input, stdlib: a subset of a dumped rollout (npz of .npy members) for local analysis: rows of the
first K workers with the 'visible_frame' field dropped (every other field's bytes unchanged, offsets packed), every other
member of shape (W, ...) cut to its first K workers, scalars / other shapes copied whole.
  usage: npzsub.py <in.npz> <out.npz> <K>"""
import ast
import sys
import zipfile


def head(h):
    m = h.read(10)
    ver = m[6]
    if ver == 1:
        hl = int.from_bytes(m[8:10], 'little'); raw = h.read(hl)
    else:
        hl = int.from_bytes(m[8:10] + h.read(2), 'little'); raw = h.read(hl)
    return ast.literal_eval(raw.decode('latin1'))


def npy_header(d):
    s = repr(d).encode('latin1')
    pad = 64 - (10 + len(s) + 1) % 64
    s = s + b' ' * pad + b'\n'
    return b'\x93NUMPY\x01\x00' + len(s).to_bytes(2, 'little') + s


def isize(t, shape=()):
    if isinstance(t, list):
        n = sum(isize(*f[1:]) if len(f) > 2 else isize(f[1]) for f in t)
    else:
        n = int(t[2:])
    for s in shape:
        n *= s
    return n


src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
zi = zipfile.ZipFile(src)
zo = zipfile.ZipFile(dst, 'w', zipfile.ZIP_STORED, allowZip64=True)
for info in zi.infolist():
    with zi.open(info) as h:
        d = head(h)
        shape = d['shape']
        descr = d['descr']
        if info.filename == 'rows.npy':
            fields = [(f, isize(f[1], f[2] if len(f) > 2 else ())) for f in descr]
            rs = sum(n for _, n in fields)
            keep = [i for i, (f, _) in enumerate(fields) if f[0] != 'visible_frame']
            offs = []
            o = 0
            for f, n in fields:
                offs.append(o); o += n
            nd = dict(d, descr=[fields[i][0] for i in keep], shape=(K,) + tuple(shape[1:]))
            T = shape[1]
            with zo.open('rows.npy', 'w', force_zip64=True) as w:
                w.write(npy_header(nd))
                for r in range(K * T):
                    b = h.read(rs)
                    w.write(b''.join(b[offs[i]:offs[i] + fields[i][1]] for i in keep))
            print('rows', shape, '->', nd['shape'], 'record', rs, '->', sum(fields[i][1] for i in keep))
        else:
            item = isize(descr) if isinstance(descr, list) else int(descr[2:])
            if len(shape) >= 1 and shape[0] > K and info.filename != 'update.npy':
                per = item
                for s in shape[1:]:
                    per *= s
                nd = dict(d, shape=(K,) + tuple(shape[1:]))
                data = h.read(per * K)
            else:
                nd = d
                data = h.read()
            with zo.open(info.filename, 'w', force_zip64=True) as w:
                w.write(npy_header(nd)); w.write(data)
            print(info.filename, shape, '->', nd['shape'])
zo.close()
