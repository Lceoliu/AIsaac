"""构建整局录制会话用的最小 DLL（rr_mt.dll）与挂起式注入器（rr_launch.exe），x86，zig 0.16 工具链。

rr_mt.dll = rr_mt.cpp + rr_turbo.cpp（可选加速模式，只有 RR_MT_TURBO=1 才安装；phase6 观战 RR_MT_WATCH=1 / 导出 RR_MT_CAPTURE=1
同在 rr_turbo.cpp，见其文件头）。

  python rl/bridge/engine/j746_rec/build.py [--out DIR]

默认输出到隔离副本 experiments/run-recorder/game-rec-01/rec/（构建产物不进 git）：
  rr_mt.dll、rr_launch.exe、obj/（MinHook 目标文件）、build.json（每一步的命令、退出码、stderr、产物 sha256）。
与 experiments/run-recorder/mt-probe/build.py、rl/turbo/build.ps1 同一套工具链与编译参数；MinHook 从
third_party/minhook 源码编译（与 rl/turbo/build.ps1 相同的 4 个文件）。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISAAC = HERE.parents[3]
FORTUNE = ISAAC.parent
ZIG = FORTUNE / '.tools/zig-x86_64-windows-0.16.0/zig.exe'
MH = ISAAC / 'third_party/minhook'
DEFAULT_OUT = ISAAC / 'experiments/run-recorder/game-rec-01/rec'
CXX = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-nullability-completeness', '-target', 'x86-windows-gnu',
       '-mstackrealign', '-I', str(HERE)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, steps):
    t = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    steps.append(dict(command=[str(c) for c in cmd], exit=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-4000:],
                      seconds=round(time.time() - t, 2)))
    if p.returncode != 0:
        raise RuntimeError(f'构建失败：{cmd[-1]}\n{p.stderr[-2000:]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    obj = out / 'obj'
    obj.mkdir(parents=True, exist_ok=True)
    if not ZIG.exists():
        raise SystemExit(f'找不到 zig：{ZIG}')
    os.environ.setdefault('ZIG_GLOBAL_CACHE_DIR', str(FORTUNE / '.tooling/zig-global-cache'))
    os.environ['ZIG_LOCAL_CACHE_DIR'] = str(out / 'zig-local-cache')
    steps = []
    rec = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), zig=str(ZIG), out=str(out), sources={}, steps=steps, outputs={})
    for f in ['rr_mt.cpp', 'rr_turbo.cpp', 'rr_turbo.hpp', 'virtual_clock.hpp', 'rr_launch.cpp', 'turbo_shared.hpp']:
        rec['sources'][f] = sha(HERE / f)
    try:
        mh_objs = []
        for rel in ['buffer.c', 'hook.c', 'trampoline.c', 'hde/hde32.c']:
            o = obj / (rel.replace('/', '_')[:-2] + '.o')
            run([ZIG, 'cc', '-O2', '-target', 'x86-windows-gnu', '-I', MH / 'include', '-I', MH / 'src', '-c', MH / 'src' / rel,
                 '-o', o], steps)
            mh_objs.append(o)
        run([ZIG] + CXX + ['-shared', '-I', MH / 'include', HERE / 'rr_mt.cpp', HERE / 'rr_turbo.cpp'] + mh_objs + ['-o', out / 'rr_mt.dll'], steps)
        run([ZIG] + CXX + ['-municode', HERE / 'rr_launch.cpp', '-o', out / 'rr_launch.exe'], steps)
        for f in ['rr_mt.dll', 'rr_launch.exe']:
            rec['outputs'][f] = dict(bytes=(out / f).stat().st_size, sha256=sha(out / f))
        rec['ok'] = True
    except RuntimeError as e:
        rec['ok'] = False
        rec['error'] = str(e)
    (out / 'build.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(ok=rec['ok'], outputs=rec['outputs'], error=rec.get('error')), ensure_ascii=False, indent=1))
    return 0 if rec['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
