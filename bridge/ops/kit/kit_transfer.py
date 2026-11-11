"""kit_transfer.py: runs on the PC. Relays kit files between two remote hosts through the PC (the hosts have no link
of their own); nothing is written to the PC's disk.

    python -u kit_transfer.py <src host> <dst host> <version> [--only meta|data|image] [--seg-mib 512]
    python -u kit_transfer.py ops <host> <commit>       deploy bridge/ops/kit + bm1/common.sh + bm1/cuda_smoke.py of
                                                        <commit> (git archive, read-only on the PC's repo) into
                                                        <ROOT>/kit/ops/<commit>/ and point <ROOT>/kit/ops/current at it
    python -u kit_transfer.py file <src host> <dst host> <name>   one file of <ROOT>/kit/

A file goes from <src ROOT>/kit/<name> to <dst ROOT>/kit/<name>:
  * the target keeps <name>.part; a new run resumes from its size after the prefix is verified (sha256 of the first
    N bytes computed on both hosts); a prefix that differs is moved to <dst ROOT>/kit/trash/ and the copy restarts
  * segments of --seg-mib MiB, each one exec pair (src: tail -c +K | head -c n, dst: cat >> .part), size checked
  * stall watchdog: 60 s without a byte moved in either direction = abort, rc 2 (the .part stays for the resume)
  * SSH keepalive every 15 s; one progress line every 5 s (flushed)
  * at the end: .part renamed, sha256 computed on BOTH hosts and compared (and with the manifest when known), the
    target gets <name>.sha256; a file already complete on the target with the same sha256 is skipped
Hosts: HOSTS below. A third host = one more entry. Credentials: the 8x3090 only through the coordinator's helper
(imported by path; its password never passes through this file), gpu57 through this PC's key. One authentication
attempt per connection, never a retry.
--die-after-mib N (tests): hard exit (os._exit) after N MiB in this run, like a killed relay."""
import argparse
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time

import paramiko

HELPER_3090 = os.environ.get(
    'KIT_HELPER_3090',
    r'C:/Users/eolc/AppData/Local/Temp/claude/c3d6d4df380a14da/6b19044b-b69c-45e7-9789-735c71597c82/scratchpad/remote3090.py')
HOSTS = {
    'dgene-8x3090': dict(kind='helper', helper=HELPER_3090, root='/home/runaiwusheng/aisaac-bench-20261019/cpu-wide',
                         pin='12-23,36-47'),
    'gpu57': dict(kind='key', host='10.19.136.57', port=15654, user='root', root='/root/AIsaac', pin='8-15,24-31'),
}
STALL_S = 60
WINDOW = 64 << 20
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Host:
    def __init__(self, name):
        self.name = name
        self.cfg = HOSTS[name]
        self.root = self.cfg['root']
        self.client = None

    def connect(self):
        c = self.cfg
        try:
            if c['kind'] == 'helper':
                spec = importlib.util.spec_from_file_location('kit_remote_helper', c['helper'])
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.client = mod.connect()
            else:
                cl = paramiko.SSHClient()
                cl.load_system_host_keys()
                cl.set_missing_host_key_policy(paramiko.RejectPolicy())
                cl.connect(c['host'], port=c['port'], username=c['user'],
                           key_filename=os.path.expanduser('~/.ssh/id_rsa'), look_for_keys=False, allow_agent=False,
                           timeout=15, banner_timeout=20, auth_timeout=20)
                self.client = cl
        except paramiko.AuthenticationException:
            say(f'{self.name}: AUTH_FAILED (one attempt, not retrying)')
            sys.exit(3)
        except (OSError, paramiko.SSHException) as e:
            say(f'{self.name}: CONNECT_FAILED {type(e).__name__} {str(e)[:200]}')
            sys.exit(2)
        tr = self.client.get_transport()
        tr.set_keepalive(15)
        tr.default_window_size = WINDOW
        return self

    def chan(self, cmd):
        ch = self.client.get_transport().open_session(window_size=WINDOW)
        ch.exec_command(cmd)
        return ch

    def run(self, cmd, timeout=3600):
        ch = self.chan(cmd)
        ch.settimeout(timeout)
        out = b''
        while True:
            d = ch.recv(1 << 16)
            if not d:
                break
            out += d
        err = b''
        while ch.recv_stderr_ready():
            err += ch.recv_stderr(1 << 16)
        rc = ch.recv_exit_status()
        ch.close()
        return rc, out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace')

    def must(self, cmd, timeout=3600):
        rc, out, err = self.run(cmd, timeout)
        if rc != 0:
            say(f'{self.name}: command failed rc {rc}: {cmd[:160]} :: {err.strip()[-300:]}')
            sys.exit(1)
        return out

    def pin(self, cmd):
        p = self.cfg.get('pin')
        return f'nice -n 10 ionice -c2 -n7 taskset -c {p} {cmd}' if p else f'nice -n 10 {cmd}'

    def size(self, path):
        rc, out, _ = self.run(f'stat -c %s {shlex.quote(path)} 2>/dev/null')
        return int(out.strip()) if rc == 0 and out.strip() else None

    def sha(self, path, nbytes=None):
        q = shlex.quote(path)
        cmd = (f'head -c {nbytes} {q} | ' + self.pin('sha256sum')) if nbytes is not None else self.pin(f'sha256sum {q}')
        return self.must(cmd).split()[0]


def par(*fns):
    """run the functions in threads, return their results in order"""
    res = [None] * len(fns)

    def w(i, f):
        res[i] = f()
    ts = [threading.Thread(target=w, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


class Relay:
    def __init__(self, args):
        self.die = args.die_after_mib << 20 if args.die_after_mib else None
        self.moved = 0          # bytes this run

    def segment(self, src, dst, spath, part, off, n, total, t_start, base):
        """copy bytes [off, off+n) of spath to the end of part; returns when done, exits rc 2 on a stall"""
        sc = src.chan(f'tail -c +{off + 1} {shlex.quote(spath)} | head -c {n}')
        dc = dst.chan(f'cat >> {shlex.quote(part)}')
        sc.settimeout(5)
        dc.settimeout(STALL_S)
        last = time.time()
        last_print = 0
        got = 0
        while True:
            try:
                d = sc.recv(1 << 20)
            except socket.timeout:
                if time.time() - last > STALL_S:
                    say(f'STALL: no byte from {src.name} for {STALL_S} s at offset {off + got}: abort (the .part stays)')
                    sys.exit(2)
                continue
            if not d:
                break
            try:
                dc.sendall(d)
            except socket.timeout:
                say(f'STALL: {dst.name} accepted no byte for {STALL_S} s at offset {off + got}: abort (the .part stays)')
                sys.exit(2)
            got += len(d)
            self.moved += len(d)
            last = time.time()
            if last - last_print >= 5:
                last_print = last
                done = off + got
                rate = (done - base) / max(last - t_start, 1e-3) / 1e6
                eta = (total - done) / max(rate * 1e6, 1)
                say(f'  {done / 1e9:7.3f} / {total / 1e9:.3f} GB ({100 * done / total:5.1f} %) {rate:6.1f} MB/s ETA {eta:5.0f} s')
            if self.die is not None and self.moved >= self.die:
                say(f'--die-after-mib reached ({self.moved} bytes this run): hard exit like a killed relay')
                os._exit(9)
        dc.shutdown_write()
        if not dc.status_event.wait(STALL_S):
            say(f'STALL: {dst.name} writer did not finish within {STALL_S} s')
            sys.exit(2)
        rc = dc.recv_exit_status()
        sc.close()
        dc.close()
        if got != n or rc != 0:
            say(f'segment at {off}: {got} of {n} bytes, writer rc {rc}: abort (the .part stays)')
            sys.exit(1)

    def file(self, src, dst, name, want_sha=None, seg=512 << 20):
        spath = f'{src.root}/kit/{name}'
        dpath = f'{dst.root}/kit/{name}'
        part = dpath + '.part'
        dst.must(f'mkdir -p {shlex.quote(dst.root)}/kit')
        total = src.size(spath)
        if total is None:
            say(f'{src.name}: {spath} missing')
            sys.exit(1)
        have = dst.size(dpath)
        if have == total:
            a, b = par(lambda: src.sha(spath), lambda: dst.sha(dpath))
            if a == b and (want_sha is None or a == want_sha):
                say(f'{name}: already on {dst.name}, sha256 {a} on both: skipped')
                return dict(name=name, bytes=total, sha256=a, seconds=0, skipped=True)
            say(f'{name}: on {dst.name} with a different sha256 ({b} vs {a}): moving it to trash')
            dst.must(f'd={shlex.quote(dst.root)}/kit/trash/$(date -u +%m%d-%H%M%S); mkdir -p $d; mv {shlex.quote(dpath)} $d/')
        off = dst.size(part) or 0
        if off > total:
            off = -1
        if off > 0:
            say(f'{name}: {dst.name} has {off} bytes of {total} in .part: verifying the prefix on both hosts')
            a, b = par(lambda: src.sha(spath, off), lambda: dst.sha(part, off))
            if a != b:
                off = -1
            else:
                say(f'{name}: prefix sha256 {a} equal on both hosts: resuming at {off}')
        if off < 0:
            say(f'{name}: .part does not match: moving it to trash, restarting at 0')
            dst.must(f'd={shlex.quote(dst.root)}/kit/trash/$(date -u +%m%d-%H%M%S); mkdir -p $d; mv {shlex.quote(part)} $d/')
            off = 0
        say(f'{name}: {src.name} -> {dst.name}, {total} bytes, from {off}')
        t0 = time.time()
        base = off
        while off < total:
            n = min(seg, total - off)
            self.segment(src, dst, spath, part, off, n, total, t0, base)
            off += n
            now = dst.size(part)
            if now != off:
                say(f'{name}: .part is {now} bytes after the segment, expected {off}: abort')
                sys.exit(1)
        dt = time.time() - t0
        say(f'{name}: {total - base} bytes relayed in {dt:.1f} s ({(total - base) / max(dt, 1e-3) / 1e6:.1f} MB/s); '
            f'sha256 on both hosts ...')
        a, b = par(lambda: src.sha(spath), lambda: dst.sha(part))
        ok = a == b and (want_sha is None or a == want_sha)
        say(f'{name}: sha256 {src.name} {a} | {dst.name} {b}' + (f' | manifest {want_sha}' if want_sha else '')
            + (' EQUAL' if ok else ' DIFFERENT'))
        if not ok:
            say(f'{name}: sha256 mismatch: the .part stays on {dst.name} for inspection; abort')
            sys.exit(1)
        dst.must(f'mv {shlex.quote(part)} {shlex.quote(dpath)} && echo {a}"  "{shlex.quote(name)} > {shlex.quote(dpath)}.sha256')
        return dict(name=name, bytes=total, relayed=total - base, sha256=a, seconds=round(dt, 1), skipped=False)


def cmd_version(args):
    src, dst = Host(args.src).connect(), Host(args.dst).connect()
    r = Relay(args)
    mname = f'{args.version}.manifest.json'          # version = kit-YYYYMMDD.N
    m = json.loads(src.must(f'cat {shlex.quote(src.root)}/kit/{mname}'))
    meta = [mname, m['data']['file_list'], f'{m["version"]}.packages.txt']
    plan = []
    if args.only in (None, 'meta'):
        plan += [(x, None) for x in meta]
    if args.only in (None, 'data'):
        plan.append((m['data']['file'], m['data']['sha256']))
    if args.only in (None, 'image'):
        plan.append((m['image']['archive']['file'], m['image']['archive']['sha256']))
    out = [r.file(src, dst, n, s, args.seg_mib << 20) for n, s in plan]
    say('SUMMARY ' + json.dumps(out))


def cmd_file(args):
    src, dst = Host(args.src).connect(), Host(args.dst).connect()
    say('SUMMARY ' + json.dumps([Relay(args).file(src, dst, args.name, None, args.seg_mib << 20)]))


def cmd_ops(args):
    h = Host(args.host).connect()
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--verify', args.commit + '^{commit}'],
                            capture_output=True, text=True, check=True).stdout.strip()
    tar = subprocess.run(['git', '-C', REPO, 'archive', '--format=tar', commit, 'bridge/ops/kit',
                          'bridge/ops/bm1/common.sh', 'bridge/ops/bm1/cuda_smoke.py'], capture_output=True, check=True).stdout
    d = f'{h.root}/kit/ops/{commit[:12]}'
    ch = h.chan(f'set -e; mkdir -p {d}; tar -xf - -C {d}; ln -sfn {commit[:12]} {h.root}/kit/ops/current; '
                f'echo {commit} > {d}/COMMIT; ls {d}/bridge/ops/kit | wc -l')
    ch.sendall(tar)
    ch.shutdown_write()
    out = ch.makefile('rb').read().decode()
    err = ch.makefile_stderr('rb').read().decode()
    rc = ch.recv_exit_status()
    say(f'ops {commit[:12]} -> {h.name}:{d} rc {rc}, {len(tar)} tar bytes, {out.strip()} files in kit/ {err.strip()[-200:]}')
    sys.exit(rc)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) > 1 and sys.argv[1] in ('ops', 'file'):
        p = argparse.ArgumentParser()
        p.add_argument('cmd')
        if sys.argv[1] == 'ops':
            p.add_argument('host', choices=HOSTS)
            p.add_argument('commit')
        else:
            p.add_argument('src', choices=HOSTS)
            p.add_argument('dst', choices=HOSTS)
            p.add_argument('name')
            p.add_argument('--seg-mib', type=int, default=512)
            p.add_argument('--die-after-mib', type=int, default=0)
        a = p.parse_args()
        return cmd_ops(a) if a.cmd == 'ops' else cmd_file(a)
    p = argparse.ArgumentParser()
    p.add_argument('src', choices=HOSTS)
    p.add_argument('dst', choices=HOSTS)
    p.add_argument('version')
    p.add_argument('--only', choices=('meta', 'data', 'image'))
    p.add_argument('--seg-mib', type=int, default=512)
    p.add_argument('--die-after-mib', type=int, default=0)
    cmd_version(p.parse_args())


if __name__ == '__main__':
    main()
