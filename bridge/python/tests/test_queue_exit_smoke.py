"""The end of a run with a second actor process (train_tok --actor-processes 2): a half the learner never read, larger than the pipe, must not hold
the actor's exit. Plain script, no GPU:
  cd rl/bridge/python && python -X utf8 tests/test_queue_exit_smoke.py
Checks:
  (a) the mechanism: a process that put more than a pipe holds into a Queue nobody reads does not end by itself (its finalizer joins the feeder thread);
  (b) with cancel_join_thread it ends at once;
  (c) a message that fits the pipe never held it (why most runs ended cleanly);
  (d) train_tok's second actor cancels the join in the block every way out passes through.
"""
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def child(q, size, cancel):
    q.put(b'x' * size)
    time.sleep(0.5)   # (the feeder thread is at the pipe)
    if cancel:
        q.cancel_join_thread()


def ends_within(size, cancel, seconds):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q, size, cancel))
    p.start()
    t0 = time.perf_counter()
    p.join(seconds)
    ended = not p.is_alive()
    took = time.perf_counter() - t0
    if not ended:
        p.terminate()
        p.join(10)
    return ended, took


def main():
    big, small = 1 << 20, 1 << 10
    ended, took = ends_within(big, False, 8.0)
    check('(a) 1 MiB nobody reads, no cancel: the process does not end', not ended, f'{took:.1f} s')
    ended, took = ends_within(big, True, 30.0)
    check('(b) 1 MiB nobody reads, cancel_join_thread: it ends', ended, f'{took:.1f} s')
    ended, took = ends_within(small, False, 30.0)
    check('(c) 1 KiB nobody reads, no cancel: it ends', ended, f'{took:.1f} s')
    src = (Path(__file__).resolve().parent.parent / 'train_tok.py').read_text(encoding='utf-8')
    m = re.search(r"    finally:\s+if actor is not None:[^\n]*\n\s+actor\['last'\]\.set\(\)\n(?:\s+#[^\n]*\n)*\s+jobs\.cancel_join_thread\(\)", src)
    check('(d) the second actor cancels the join in its finally', m is not None)
    print('ALL OK' if not FAIL else f'{len(FAIL)} FAILED')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
