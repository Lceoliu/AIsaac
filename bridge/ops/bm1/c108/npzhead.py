"""read-only: list the arrays in .npz files (stdlib only): name, dtype descr, shape"""
import sys, zipfile, ast
for f in sys.argv[1:]:
    z = zipfile.ZipFile(f)
    print('==', f)
    for i in z.infolist():
        with z.open(i) as h:
            m = h.read(10); hl = int.from_bytes(m[8:10], 'little') if m[6] == 1 else None
            if hl is None:
                hl = int.from_bytes(m[8:10] + h.read(2), 'little')
            d = ast.literal_eval(h.read(hl).decode('latin1'))
            print(' ', i.filename, d['shape'], str(d['descr'])[:600], i.file_size)
