"""B62: which CUDA calls invalidate a global-mode graph capture (measurement; the control groups of the capture gate
and of graph_capture's gc.disable).

  cd bridge/python && python ../ops/b62_capture_probe.py [case ...]     (all cases by default; CUDA)

Every case runs in its own subprocess (a broken capture poisons the process: the later captures and CUDA random
draws fail with it). The main thread captures a small graph on a side stream (capture_error_mode='global'); the case
acts while the capture is under way:
  side thread (another thread, as the learner prefetch worker beside an actor capture):
    none            nothing (the control of the controls)
    query           Event.query() of an event recorded and finished before the capture
    sync            Event.synchronize() of that event
    pin_alloc       a new pinned host buffer (torch.empty(pin_memory=True), a size not cached: cudaHostAlloc)
    pin_reuse       a pinned buffer of a cached size (the host allocator's free list: its event processing)
    pin_free_used   the last reference to a pinned buffer that served a non_blocking copy dropped (the host
                    allocator records an event on the copy's stream)
    pin_free_plain  the last reference to a pinned buffer never copied from dropped
    replay          a graph captured before, replayed on its own stream (as the C++ d2 lane thread's fill_and_run)
    h2d             a non_blocking pinned -> device copy on another stream
    stream_sync     torch.cuda.Stream.synchronize() of another (idle) stream
  dev1_<case> (two visible GPUs): the same side call, its event / stream / graph / buffers on cuda:1 and the side
  thread's current device cuda:1, while the capture runs on cuda:0 (one process driving two GPUs)
  capturing thread, gc.collect() inside the capture (graph_capture's gc.disable bypassed), the garbage being
  an object in a reference cycle, dropped before the capture, that holds:
    gc_graph        a captured torch.cuda.CUDAGraph (its private pool)
    gc_pinned       a pinned buffer that served a non_blocking copy
    gc_actor        a SlotBlockActor with its precaptured graphs (block_paged.precapture_actor), as an actor dropped
                    by the training loop
  and the normal groups through block_training.graph_capture (gc off inside; the collection after the capture):
    gc_graph_gated, gc_actor_gated
Prints one JSON line per case (errors with CUDA's raw text): the capture's outcome (its end and a replay checked against eager values), the side
call's outcome, whether the side call ran while the capture was under way, and whether a fresh capture works after."""
import json
import os
import subprocess
import sys
import threading

CASES = ('none', 'query', 'sync', 'pin_alloc', 'pin_reuse', 'pin_free_used', 'pin_free_plain', 'replay', 'h2d',
         'stream_sync',
         'dev1_query', 'dev1_sync', 'dev1_stream_sync', 'dev1_pin_alloc', 'dev1_replay', 'dev1_h2d',
         'gc_graph', 'gc_pinned', 'gc_actor', 'gc_graph_gated', 'gc_actor_gated')


def _err(e):
    """The exception's type and CUDA's raw text (first line of each message line kept, up to 600 characters)."""
    return (type(e).__name__+': '+' | '.join(l.strip() for l in str(e).splitlines() if l.strip()))[:600]


def child(case):
    import gc
    import torch
    sys.path.insert(0, os.getcwd())
    res = dict(case=case)
    dev = 'cuda'
    dev1 = case.startswith('dev1_')   # (the side thread's material and calls on cuda:1, the capture on cuda:0)
    base = case[5:] if dev1 else case
    sdev = 1 if dev1 else 0
    x = torch.ones(1 << 16, device=dev)
    torch.cuda.set_device(sdev)
    copy_stream = torch.cuda.Stream()
    # side-thread material, made before the capture
    ev = torch.cuda.Event()
    ev.record()
    torch.cuda.synchronize()
    held = {}
    if base == 'pin_reuse':
        t_ = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        del t_   # (cached by the host allocator: the next allocation of this size reuses it)
    if base in ('pin_free_used', 'gc_pinned'):
        p = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        with torch.cuda.stream(copy_stream):
            torch.empty(1 << 20, dtype=torch.uint8, device=dev).copy_(p, non_blocking=True)
        torch.cuda.synchronize()
        held['p'] = p
        del p
    if base == 'pin_free_plain':
        held['p'] = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
    other = torch.cuda.Stream()
    if base == 'replay':
        gr, yr = torch.cuda.CUDAGraph(), torch.zeros(1 << 16, device=dev)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr, stream=other):
            yr.add_(1)
        torch.cuda.synchronize()
        held['g'] = gr
    if base == 'h2d':
        held['src'] = torch.ones(1 << 20, dtype=torch.uint8, pin_memory=True)
        held['dst'] = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
    torch.cuda.set_device(0)
    # cyclic garbage holding CUDA resources
    if case.startswith('gc_'):
        class Holder:
            pass
        h = Holder()
        h.me = h
        if case.startswith('gc_graph'):
            g0, s0 = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            y0 = torch.zeros(1 << 16, device=dev)
            torch.cuda.synchronize()
            with torch.cuda.graph(g0, stream=s0):
                y0.add_(1)
            h.graph, h.out = g0, y0
            del g0, y0
        elif base == 'gc_pinned':
            h.pinned = held.pop('p')
        else:   # gc_actor*
            sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
            from test_block_memory import make_policy
            from isaac_bridge.block_paged import SlotBlockActor, precapture_actor
            torch.manual_seed(5)
            m = make_policy(dev).eval()
            a = SlotBlockActor(m, 1, graph_entities=128, bf16=True, lane_slots=0)
            res['actor_graphs'], _ = precapture_actor(a, merged=False)
            h.actor = a
            del a, m
        gc.collect()   # (everything older collected now: only h's cycle is left for the collection inside)
        gc.disable()
        del h
    go, done, side = threading.Event(), threading.Event(), {}

    def worker():
        go.wait()
        try:
            torch.cuda.set_device(sdev)
            if base == 'query':
                side['value'] = bool(ev.query())
            elif base == 'sync':
                ev.synchronize()
            elif base == 'pin_alloc':
                side['n'] = torch.empty(16 << 20, dtype=torch.uint8, pin_memory=True).numel()
            elif base == 'pin_reuse':
                side['n'] = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True).numel()
            elif base in ('pin_free_used', 'pin_free_plain'):
                held.pop('p')
            elif base == 'replay':
                with torch.cuda.stream(other):
                    held['g'].replay()
            elif base == 'h2d':
                with torch.cuda.stream(other):
                    held['dst'].copy_(held['src'], non_blocking=True)
            elif base == 'stream_sync':
                other.synchronize()
            side['outcome'] = 'ok'
        except BaseException as e:
            side['outcome'] = 'error: '+_err(e)
        done.set()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    out = None
    try:
        if case.endswith('_gated'):
            from isaac_bridge import block_training
            block_training.SideCuda.ENABLED = False   # (only the gc part under test)
            gc.enable()
            ctx = block_training.graph_capture(graph, stream)
        else:
            torch.cuda.synchronize()
            ctx = None
            graph_cm = torch.cuda.stream(stream)
            graph_cm.__enter__()
            graph.capture_begin(capture_error_mode=os.environ.get('PROBE_CAPTURE_MODE', 'global'))   # (thread_local: the alternative to a gate)
        try:
            if ctx is not None:
                ctx.__enter__()
            y = x*2
            if case.startswith('gc_'):
                if case.endswith('_gated'):
                    junk = [[i] for i in range(5000)]   # (allocations that would trigger an automatic collection)
                    del junk
                    res['gc_enabled_inside'] = gc.isenabled()
                else:
                    res['collected'] = gc.collect()
                done.set()
            else:
                go.set()
            res['side_during_capture'] = done.wait(2.0)
            for _ in range(32):
                y = y+1
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
            else:
                graph.capture_end()
                graph_cm.__exit__(None, None, None)
        graph.replay()
        torch.cuda.synchronize()
        res['capture'] = 'ok' if bool((y == 34).all()) else 'wrong values'
    except BaseException as e:
        res['capture'] = 'error: '+_err(e)
    go.set()
    t.join(10)
    if case.endswith('_gated'):
        try:
            res['collected_after'] = gc.collect()
            torch.cuda.synchronize()
        except BaseException as e:
            res['collect_after'] = 'error: '+_err(e)
    res['side'] = side.get('outcome', 'n/a' if case.startswith('gc_') else 'not run')
    try:   # a fresh capture after (the cascade: does the process still capture?)
        g2, s2 = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        z = torch.zeros(4, device=dev)
        torch.cuda.synchronize()
        with torch.cuda.graph(g2, stream=s2):
            z.add_(1)
        g2.replay()
        torch.cuda.synchronize()
        res['fresh_capture_after'] = 'ok'
    except BaseException as e:
        res['fresh_capture_after'] = 'error: '+_err(e)
    print('RESULT', json.dumps(res), flush=True)


def main():
    cases = [c for c in sys.argv[1:] if not c.startswith('-')] or list(CASES)
    for c in cases:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', c], capture_output=True, text=True,
                           timeout=600)
        line = next((l for l in r.stdout.splitlines() if l.startswith('RESULT ')), None)
        if line:
            print(line[7:], flush=True)
        else:
            print(json.dumps(dict(case=c, crashed=r.returncode, tail=(r.stdout+r.stderr)[-400:])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--child':
        child(sys.argv[2])
    else:
        main()
