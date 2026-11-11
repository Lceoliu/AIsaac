"""bm1 diagnosis: where the native (non-Python) threads a test leaves behind are born.

  cd bridge/python && python ../ops/bm1/diag_threads.py <pytest node id> [...]

Runs the tests in this process (pytest.main, conftest included) while a watcher thread polls /proc/self/task every
0.5 ms: when a thread id appears that is no Python thread, it records its name and every Python thread's current
stack (sys._current_frames): a native thread is created synchronously by whichever thread calls into the library, so
the caller's stack at that moment names the call. Prints, per new native thread, the stacks' innermost frames."""
import os
import sys
import threading
import time
import traceback


def tasks():
    out = {}
    for t in os.listdir('/proc/self/task'):
        try:
            with open(f'/proc/self/task/{t}/comm') as f:
                out[int(t)] = f.read().strip()
        except OSError:
            pass
    return out


def main():
    import pytest
    seen = set(tasks())
    births = []
    stop = threading.Event()

    def watch():
        me = threading.get_native_id()
        while not stop.is_set():
            cur = tasks()
            new = [t for t in cur if t not in seen]
            if new:
                py = {th.native_id for th in threading.enumerate()}
                frames = sys._current_frames()
                names = {th.ident: th.name for th in threading.enumerate()}
                stacks = {names.get(k, k): traceback.extract_stack(f)[-8:] for k, f in frames.items()
                          if names.get(k) != 'diag-watch'}
                for t in new:
                    seen.add(t)
                    births.append((time.time(), t, cur[t], t in py or t == me, stacks))
            time.sleep(0.0005)
    w = threading.Thread(target=watch, name='diag-watch', daemon=True)
    w.start()
    rc = pytest.main(['-q', '-p', 'no:cacheprovider'] + sys.argv[1:])
    time.sleep(0.5)
    stop.set()
    w.join()
    alive = tasks()
    py = {th.native_id for th in threading.enumerate()}
    print(f'\n=== pytest rc {rc}; native threads born during the run: '
          f'{sum(1 for b in births if not b[3])}, alive now (not Python): '
          f'{sorted((t, n) for t, n in alive.items() if t not in py)}')
    for when, t, name, is_py, stacks in births:
        if is_py or (t in py):
            continue
        print(f'\n--- native thread {t} "{name}" born, alive now: {t in alive}; Python stacks at that moment:')
        for th, st in stacks.items():
            print(f'  [{th}]')
            for fr in st[-6:]:
                print(f'     {os.path.basename(fr.filename)}:{fr.lineno} {fr.name}: {(fr.line or "")[:90]}')


if __name__ == '__main__':
    main()
