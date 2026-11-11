"""kit: write the image build context as a tar stream to stdout (consumed by `docker build -`).

    python3 kit_ctx.py <Dockerfile> <venv dir> <pydeps dir> <old venv path> [<old venv path> ...] > ctx.tar

The context holds Dockerfile, venv/ and pydeps/. Every entry is owned by 0:0 with its mode and mtime kept. Path repair
happens here, in the stream, so the image needs no RUN step: in venv/bin and pydeps/bin, a text file whose first line
is a "#!" line naming python gets "#!/opt/aisaac/venv/bin/python"; in venv/bin's activate scripts every old venv path
becomes /opt/aisaac/venv. Every repaired file is listed on stderr ("fixed <path>"). Symlinks are kept as symlinks
(venv/bin/python3 -> /usr/bin/python3, the image's Python 3.10.12)."""
import io
import os
import sys
import tarfile

NEW = '/opt/aisaac/venv'


def fix(rel, data, old):
    parts = rel.split('/')
    if len(parts) != 3 or parts[1] != 'bin' or b'\0' in data[:4096]:
        return data
    out = data
    if out.startswith(b'#!'):
        nl = out.find(b'\n')
        first = out[:nl if nl >= 0 else len(out)]
        if b'python' in first:
            out = b'#!' + NEW.encode() + b'/bin/python' + out[len(first):]
    if parts[2].startswith('activate') or parts[2] == 'Activate.ps1':
        for o in old:
            out = out.replace(o.encode(), NEW.encode())
    return out


def add_tree(tf, src, arc, old):
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, src)
        names = [None] + sorted(filenames) + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
        for name in names:
            path = dirpath if name is None else os.path.join(dirpath, name)
            rel = arc if (name is None and rel_dir == '.') else \
                os.path.normpath(os.path.join(arc, rel_dir, name or '')).replace(os.sep, '/')
            st = os.lstat(path)
            ti = tf.gettarinfo(path, arcname=rel)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = 'root'
            if ti.isreg():
                with open(path, 'rb') as f:
                    if st.st_size < (1 << 20):
                        data = f.read()
                        new = fix(rel, data, old)
                        if new != data:
                            print('fixed', rel, file=sys.stderr)
                        ti.size = len(new)
                        tf.addfile(ti, io.BytesIO(new))
                    else:
                        tf.addfile(ti, f)
            else:
                tf.addfile(ti)
        # symlinked directories were added as links above; do not descend into them
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]


def main():
    dockerfile, venv, pydeps = sys.argv[1:4]
    old = sys.argv[4:]
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.PAX_FORMAT) as tf:
        data = open(dockerfile, 'rb').read()
        ti = tarfile.TarInfo('Dockerfile')
        ti.size, ti.mode, ti.mtime = len(data), 0o644, int(os.stat(dockerfile).st_mtime)
        tf.addfile(ti, io.BytesIO(data))
        add_tree(tf, venv, 'venv', old)
        add_tree(tf, pydeps, 'pydeps', old)


if __name__ == '__main__':
    main()
